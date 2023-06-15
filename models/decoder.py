import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # FC Output
        self.fc_output = nn.Linear(5014, 14 * 14 * 48)

        # Conv. 5x5 + ReLU + 2x US + BN
        self.conv1 = nn.Conv2d(48, 48, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn1 = nn.BatchNorm2d(48)

        # Conv. 5x5 + ReLU + 2x US + BN
        self.conv2 = nn.Conv2d(48, 48, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn2 = nn.BatchNorm2d(48)

        # Conv. 5x5 + ReLU + 2x US + BN
        self.conv3 = nn.Conv2d(48, 48, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn3 = nn.BatchNorm2d(48)

        # Conv. 5x5 + sigmoid + 2x US + BN
        self.conv4 = nn.Conv2d(48, 3, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.upsample4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn4 = nn.BatchNorm2d(3)

    def forward(self, x):
        #input is 5012x1
        x = self.fc_output(x)
        x = x.view(-1, 48, 14, 14)  # Corrected reshaping

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.upsample1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.upsample2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.upsample3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.sigmoid(x)
        x = self.bn4(x)

        return x


