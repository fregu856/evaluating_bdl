# code-checked
# server-checked

import torch.nn as nn
from torch.nn import functional as F
import torch
import os
import sys
import numpy as np
from torch.autograd import Variable
import functools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from resnet_block import conv3x3, Bottleneck
from aspp import ASPP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../inplace_abn'))
from bn import InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        print ("model_mcdropout.py")

        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # NOTE! (ceil_mode=True will do that x (batch_size, 128, h/4, w/4) e.g. has shape (batch_size, 128, 33, 33) instead of (batch_size, 128, 32, 32) if h == w == 256)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1))

        self.aspp = ASPP()

        self.cls = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                                       BatchNorm2d(planes*block.expansion, affine=True))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes*block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        # (x has shape: (batch_size, 3, h, w))

        x = self.relu1(self.bn1(self.conv1(x))) # (shape: (batch_size, 64, h/2, w/2))
        x = self.relu2(self.bn2(self.conv2(x))) # (shape: (batch_size, 64, h/2, w/2))
        x = self.relu3(self.bn3(self.conv3(x))) # (shape: (batch_size, 128, h/2, w/2))
        x = self.maxpool(x) # (shape: (batch_size, 128, h/4, w/4))
        x = self.layer1(x) # (shape: (batch_size, 256, h/4, w/4))
        x = F.dropout(x, p=0.5, training=True) # (shape: (batch_size, 256, h/4, w/4))
        x = self.layer2(x) # (shape: (batch_size, 512, h/8, w/8))
        x = F.dropout(x, p=0.5, training=True) # (shape: (batch_size, 512, h/8, w/8))
        x = self.layer3(x) # (shape: (batch_size, 1024, h/8, w/8))
        x = F.dropout(x, p=0.5, training=True) # (shape: (batch_size, 1024, h/8, w/8))
        x = self.layer4(x) # (shape: (batch_size, 2048, h/8, w/8))
        x = F.dropout(x, p=0.5, training=True) # (shape: (batch_size, 2048, h/8, w/8))
        x = self.aspp(x) # (shape: (batch_size, 512, h/8, h/8))
        x = self.cls(x) # (shape: (batch_size, num_classes, h/8, w/8))

        return x

def get_model(num_classes=19):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model
