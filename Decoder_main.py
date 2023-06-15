import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
import math
import os
import random
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from models import decoder2
from torchvision.models import vgg16
import torch.optim as optim

class MyDataset(Dataset):
    def __init__(self, image_dir, label_dirs, num_images_per_sample=32):
        self.image_dir = image_dir
        self.label_dirs = label_dirs
        self.num_images_per_sample = num_images_per_sample
        self.sample_ids = []
        for label_dir in label_dirs:
            subfolder_ids = os.listdir(label_dir)
            self.sample_ids.extend([(subfolder_id, label_dir) for subfolder_id in subfolder_ids])
        random.shuffle(self.sample_ids)
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        subfolder_id, label_dir = self.sample_ids[idx]
        subfolder_id = subfolder_id.replace("label", "").split(".")[0]
        
        image_name = 'frame_{:03d}_{:03d}.jpg'.format(int(subfolder_id), 16)
        image_path = os.path.join(self.image_dir, image_name)
        im = cv2.imread(image_path)
        im_resized = cv2.resize(im, (112, 112), interpolation=cv2.INTER_LINEAR)
        im_tensor = torch.from_numpy(im_resized)
        fmri_name = 'label{}.npy'.format(int(subfolder_id))
        fmri_path = os.path.join(label_dir, fmri_name)
        fmri = np.load(fmri_path)
        return fmri, im_tensor
        
        return image_batch, label
    

# Perceptual Similarity Loss (Lim)
def perceptual_similarity_loss(x, x_hat, vgg_model):
    x = x.view(1, 3, 112, 112)
    x = x.cuda().float()  # Convert input tensor to float
    x_hat = x_hat.view(1, 3, 112, 112)
    x_hat = x_hat.cuda().float()  # Convert reconstructed tensor to float
    x_features = vgg_model(x)
    x_hat_features = vgg_model(x_hat)
    lim_loss = nn.MSELoss()(x_features, x_hat_features)
    return lim_loss


def total_variation_loss(x_hat):
    diff_h = torch.abs(x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1])
    diff_v = torch.abs(x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :])
    tv_loss = torch.mean(diff_h) + torch.mean(diff_v)
    tv_loss.retain_grad()
    return tv_loss

def spatial_group_lasso_regularization(model):
    lasso_loss = 0
    for name, param in model.named_parameters():
        if 'fc_output' in name:  # Assuming the first FC layer is named 'fc1'
            lasso_loss += torch.norm(param, p=2)
    return lasso_loss

# Supervised Decoder Loss
def supervised_decoder_loss(x, x_hat, vgg_model, beta, gamma, delta, model):
    lim_loss = perceptual_similarity_loss(x, x_hat, vgg_model)
    tv_loss = total_variation_loss(x_hat)
    lasso_loss = spatial_group_lasso_regularization(model)

    loss = beta * lim_loss + delta * (tv_loss + lasso_loss)
    return loss

def get_learning_rate(epoch, initial_lr=1e-3, lr_factor=5, lr_step_size=25):
    return initial_lr / math.pow(lr_factor, epoch // lr_step_size)

def main():
    print(torch.__version__)

    print(torch.cuda.is_available())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = decoder2.Decoder2()
    model.cuda()

    image_dir = '/media/miplab-nas2/Data3/andre/stimuli_final'
    label_dirs = [
        'labels2/1',
        'labels2/2',
        'labels2/4',
        'labels2/5',
        'labels2/9'
    ]

    train_dataset = MyDataset(image_dir, label_dirs)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Example VGG model
    vgg_model = vgg16(pretrained=True)  # Select a subset of VGG layers
    vgg_model = vgg_model.cuda()
    vgg_model.eval()

    # Weight coefficients
    beta = 0.5
    gamma = 0.3
    delta = 0.2

    optimizer = optim.Adam(model.parameters(), lr=get_learning_rate(0))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_learning_rate)

    num_epochs = 100
    print_interval = 10
    losses_tot = []
    min_loss = 10000000000000 #max

    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    for epoch in range(num_epochs):
        for i, (fmri, image) in enumerate(train_dataloader):
            optimizer.zero_grad()
            image = image.to(device).float()
            fmri = fmri.to(device).float()
            print(fmri.shape, image.shape)
            x_hat = model(fmri)
            loss = supervised_decoder_loss(image, x_hat, vgg_model, beta, gamma, delta)
            losses_tot.append(loss.mean().item())
            if (loss.mean().item()) < min_loss:
                min_loss = loss.mean().item()
                print('saving model with loss =', min_loss)
                torch.save(model.state_dict(), 'saved_model/decoder2.pth')
            loss.mean().backward()
            optimizer.step()

            # Print training statistics
            if i % print_interval == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.mean().item():.4f}")

        # Update learning rate at the end of each epoch
        scheduler.step()


if __name__ == '__main__':
    main()