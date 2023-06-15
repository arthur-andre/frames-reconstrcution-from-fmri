import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                 input_channels=3,
                 output_layers=[]):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            input_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        
        #layer to output on forward pass
        self.output_layers = output_layers
        
        self.conv2 = nn.Conv3d(256, 2048, kernel_size=(16, 1, 1), stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(2048)
        self.conv3 = nn.Conv2d(2048, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(25088, 13459)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = x2.view(-1, 2048, 28, 28)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = x3.unsqueeze(dim=0)
        x3= x3.view(-1, (128*14*14))

        x4 = self.flatten(x3)
        x4 = self.dropout(x4)

        x5 = self.fc2(x4)
        
        return x5

        
def resnet101():
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], num_classes=51,
            sample_size=112,
            sample_duration=16,
            input_channels=3,
            output_layers='avgpool')
    return model

#load model


def load_weights(model, check_path):
    checkpoint = torch.load(check_path)
    test = {}
    for previous_name in checkpoint['state_dict'].keys():
        previous = previous_name
        new_name = previous_name.replace("module.", "")
        test[new_name] = checkpoint['state_dict'][previous]
    model.load_state_dict(test, strict = False)