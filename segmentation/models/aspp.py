# code-checked
# server-checked

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../inplace_abn'))
from bn import InPlaceABNSync

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_img = nn.Sequential(nn.Conv2d(4*512, 512, kernel_size=1, padding=0, dilation=1, bias=False),
                                      InPlaceABNSync(512))

        self.conv1 = nn.Sequential(nn.Conv2d(4*512, 512, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(512))
        self.conv2 = nn.Sequential(nn.Conv2d(4*512, 512, kernel_size=3, padding=12, dilation=12, bias=False),
                                   InPlaceABNSync(512))
        self.conv3 = nn.Sequential(nn.Conv2d(4*512, 512, kernel_size=3, padding=24, dilation=24, bias=False),
                                   InPlaceABNSync(512))
        self.conv4 = nn.Sequential(nn.Conv2d(4*512, 512, kernel_size=3, padding=36, dilation=36, bias=False),
                                   InPlaceABNSync(512))

        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(5*512, 512, kernel_size=1, padding=0, dilation=1, bias=False),
                                             InPlaceABNSync(512),
                                             nn.Dropout2d(0.1))

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 4*512, h/8, w/8))

        feature_map_h = feature_map.size()[2] # (== h/8)
        feature_map_w = feature_map.size()[3] # (== w/8)

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 4*512, 1, 1))
        out_img = self.conv_img(out_img) # (shape: (batch_size, 512, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear", align_corners=True) # (shape: (batch_size, 512, h/8, w/8))

        out_1x1 = self.conv1(feature_map) # (shape: (batch_size, 512, h/8, w/8))
        out_3x3_1 = self.conv2(feature_map) # (shape: (batch_size, 512, h/8, w/8))
        out_3x3_2 = self.conv3(feature_map) # (shape: (batch_size, 512, h/8, w/8))
        out_3x3_3 = self.conv4(feature_map) # (shape: (batch_size, 512, h/8, w/8))

        out = torch.cat([out_img, out_1x1, out_3x3_1, out_3x3_2, out_3x3_3], 1) # (shape: (batch_size, 5*512, h/8, w/8))
        out = self.conv_bn_dropout(out) # (shape: (batch_size, 512, h/8, w/8))

        return out
