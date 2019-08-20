# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/10 0010, matt '

import torch
from torchvision import models
from torch.nn import functional as F
from torch import nn
import numpy as np


class FCN8s(nn.Module):
    def __init__(self, pretrained_net, num_class=2, verbose=False):
        super().__init__()
        self.num_class = num_class
        self.verbose = verbose

        self.stage1 = nn.Sequential(*list(pretrained_net.children()))[:-4]
        self.stage2 = nn.Sequential(*list(pretrained_net.children()))[-4]
        self.stage3 = nn.Sequential(*list(pretrained_net.children()))[-3]

        self.relu = nn.ReLU(True)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, num_class, kernel_size=1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        x3 = self.stage1(x)
        if self.verbose:
            print("x3 size: ", x3.size())
        x4 = self.stage2(x3)
        if self.verbose:
            print("x4 size: ", x4.size())
        x5 = self.stage3(x4)
        if self.verbose:
            print("x5 size: ", x5.size())

        score = self.relu(self.deconv1(x5))
        if self.verbose:
            print("deconv1 size: ", score.size())
        score = F.interpolate(score, x4.size()[2:])
        if self.verbose:
            print("interpolate_size", score.size())
        score = self.bn1(score+x4)
        if self.verbose:
            print("x5+x4 size: ", score.size())
        score = self.relu(self.deconv2(score))
        if self.verbose:
            print("deconv2 size: ", score.size())
        score = F.interpolate(score, x3.size()[2:])
        if self.verbose:
            print("interpolate_size", score.size())
        score = self.bn2(score+x3)
        if self.verbose:
            print("x4+x3 size: ", score.size())
        score = self.bn3(self.relu(self.deconv3(score)))
        if self.verbose:
            print("deconv3 size: ", score.size())
        score = self.bn4(self.relu(self.deconv4(score)))
        if self.verbose:
            print("deconv4 size: ", score.size())
        score = self.bn5(self.relu(self.deconv5(score)))
        if self.verbose:
            print("deconv5 size: ", score.size())

        score = self.classifier(score)
        if self.verbose:
            print("classifier size: ", score.size())
        return self.softmax(score)


if __name__ == "__main__":

    pretrained_net = models.resnet34()
    net = FCN8s(pretrained_net, 2, True)
    x = torch.randn(1, 3, 255, 324)
    y = net(x)

