import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
import math
import os
from models import encoder
import random
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from models import encoder2
import torch.optim as optim
from losses import losses

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
        image_batch = []
        for i in range(1, self.num_images_per_sample + 1):
            image_name = 'frame_{:03d}_{:03d}.jpg'.format(int(subfolder_id), i)
            image_path = os.path.join(self.image_dir, image_name)
            im = cv2.imread(image_path)
            im = im[90:620,170:1120,:] #center the frame to minimize the downscale issue
            im_resized = cv2.resize(im, (112, 112), interpolation=cv2.INTER_LINEAR)
            im_tensor = torch.from_numpy(im_resized)
            image_batch.append(im_tensor)
        image_batch = torch.stack(image_batch)
        label_name = 'label{}.npy'.format(int(subfolder_id))
        label_path = os.path.join(label_dir, label_name)
        label = np.load(label_path)
        
        return image_batch, label
    
def get_learning_rate(epoch, initial_lr=1e-4, lr_factor=0.2, lr_step_size=3):
    return initial_lr * math.pow(lr_factor, math.floor(epoch / lr_step_size))

def main():
    print(torch.__version__)
    torch.cuda.is_available()
    #to get the current working directory
    directory = os.getcwd()

    print(directory)

    #load model 2
    model2 = encoder2.resnet101()
    model2 = model2.cuda()

    path = '/home/aandre/fmri_project/pretrained/MARS_HMDB51_16f.pth'
    encoder.load_weights(model2, path)

    image_dir = '/media/miplab-nas2/Data3/andre/stimuli_final'
    label_dirs = [
        '/home/aandre/fmri_project/labels/1',
        '/home/aandre/fmri_project/labels/2'
    ]

    dataset = MyDataset(image_dir, label_dirs)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device=",device)

    optimizer = optim.Adam(model2.parameters(), lr=get_learning_rate(0))
    alpha = 0.5
    #optimizer = optim.Adam(model1.parameters(), lr=0.001)
    num_epochs = 10
    print_interval = 10
    losses_tot = []
    min_loss = 10000000000000 #max

    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    for epoch in range(num_epochs):
        for i, (images, fmri) in enumerate(iter(train_dataloader)):
            optimizer.zero_grad()
            images = images.to(device).float()
            images = images.view(-1, 3, 32, 112, 112)
            fmri = fmri.to(device).float()
            r_hat = model2(images)
            loss = losses.fmri_loss(fmri, r_hat, alpha)
            losses_tot.append(loss.mean().item())
            if (loss.mean().item()) < min_loss:
                min_loss = loss.mean().item()
                print('saving model with loss =', min_loss)
                torch.save(model2.state_dict(), '/home/aandre/fmri_project/saved_model/encoder_sub1_2.pth')
            loss.mean().backward()
            optimizer.step()
            
            # Print training statistics
            if i % print_interval == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.mean().item():.4f}")

                # Adjust the learning rate
            current_lr = get_learning_rate(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        #scheduler.step()

    np.save('/home/aandre/fmri_project/saved_model/encoder_losses_sub1_2.npy', np.array(losses_tot))

if __name__ == '__main__':
    main()