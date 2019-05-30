# code-checked

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights:
    for m in layers.modules():
        init_weights(m)

    return layers

def convt_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights:
    for m in layers.modules():
        init_weights(m)

    return layers

class DepthCompletionNet(nn.Module):
    def __init__(self, pretrained=False):
        super(DepthCompletionNet, self).__init__()
        print ("model.py")

        self.layers = 34
        self.pretrained = pretrained

        self.conv1_d = conv_bn_relu(1, 32, kernel_size=3, stride=1, padding=1)

        self.conv1_img = conv_bn_relu(1, 32, kernel_size=3, stride=1, padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(self.layers)](pretrained=self.pretrained)
        if not self.pretrained:
            pretrained_model.apply(init_weights)
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model # (clear memory)

        if self.layers <= 34:
            num_channels = 512
        elif self.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

        self.convt5 = convt_bn_relu(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256+128), out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128+64), out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.convtf_mean = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)
        self.convtf_var = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)

    def forward(self, img, sparse):
        # (img has shape: (batch_size, h, w)) (grayscale)
        # (sparse has shape: (batch_size, h, w))

        img = torch.unsqueeze(img, 1) # (shape: (batch_size, 1, h, w)))
        sparse = torch.unsqueeze(sparse, 1) # (shape: (batch_size, 1, h, w)))

        conv1_d = self.conv1_d(sparse) # (shape: (batch_size, 32, h, w)))
        conv1_img = self.conv1_img(img) # (shape: (batch_size, 32, h, w)))

        conv1 = torch.cat((conv1_d, conv1_img), 1) # (shape: (batch_size, 64, h, w)))

        conv2 = self.conv2(conv1) # (shape: (batch_size, 64, h, w)))
        conv3 = self.conv3(conv2) # (shape: (batch_size, 128, h/2, w/2)))
        conv4 = self.conv4(conv3) # (shape: (batch_size, 256, h/4, w/4)))
        conv5 = self.conv5(conv4) # (shape: (batch_size, 512, h/8, w/8)))
        conv6 = self.conv6(conv5) # (shape: (batch_size, 512, h/16, w/16)))

        convt5 = self.convt5(conv6) # (shape: (batch_size, 256, h/8, w/8)))
        y = torch.cat((convt5, conv5), 1) # (shape: (batch_size, 256+512, h/8, w/8)))

        convt4 = self.convt4(y) # (shape: (batch_size, 128, h/4, w/4)))
        y = torch.cat((convt4, conv4), 1) # (shape: (batch_size, 128+256, h/4, w/4)))

        convt3 = self.convt3(y) # (shape: (batch_size, 64, h/2, w/2)))
        y = torch.cat((convt3, conv3), 1) # (shape: (batch_size, 64+128, h/2, w/2)))

        convt2 = self.convt2(y) # (shape: (batch_size, 64, h, w)))
        y = torch.cat((convt2, conv2), 1) # (shape: (batch_size, 64+64, h, w)))

        convt1 = self.convt1(y) # (shape: (batch_size, 64, h, w)))
        y = torch.cat((convt1,conv1), 1) # (shape: (batch_size, 64+64, h, w)))

        mean = self.convtf_mean(y) # (shape: (batch_size, 1, h, w))

        log_var = self.convtf_var(y) # (shape: (batch_size, 1, h, w))

        mean = 100*mean

        return (mean, log_var)
