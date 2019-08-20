# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/14 0014, matt '

import torch
import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, num_classes=2, verbose=False):
        super(SegNet, self).__init__()
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # second group

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # third group

        self.encoder_4 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # fourth group

        self.encoder_5 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )  # fourth group
        self.unpool_1 = nn.MaxUnpool2d(2, stride=2)  # get masks
        self.unpool_2 = nn.MaxUnpool2d(2, stride=2)
        self.unpool_3 = nn.MaxUnpool2d(2, stride=2)
        self.unpool_4 = nn.MaxUnpool2d(2, stride=2)
        self.unpool_5 = nn.MaxUnpool2d(2, stride=2)

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64)
        )  # first group

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64)
        )  # second group

        self.decoder_3 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64)
        )  # third group
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(64, 64, 7, padding=3),
            nn.BatchNorm2d(64)
        )

        self.decoder_5 = nn.Sequential(
            nn.Conv2d(64, 3, 7, padding=3),
            nn.BatchNorm2d(3)
        )

        self.classifier = nn.Conv2d(3, num_classes, 1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.verbose = verbose

    def forward(self, x):
        size_1 = x.size()
        x, indices_1 = self.encoder_1(x)
        if self.verbose:
            print(size_1, x.size())

        size_2 = x.size()
        x, indices_2 = self.encoder_2(x)
        if self.verbose:
            print(size_2, x.size())

        size_3 = x.size()
        x, indices_3 = self.encoder_3(x)
        if self.verbose:
            print(size_3, x.size())

        size_4 = x.size()
        x, indices_4 = self.encoder_4(x)
        if self.verbose:
            print(size_4, x.size())

        size_5 = x.size()
        x, indices_5 = self.encoder_5(x)
        if self.verbose:
            print(size_5, x.size())

        x = self.unpool_1(x, indices_5, output_size=size_5)
        if self.verbose:
            print(x.size())
        x = self.decoder_1(x)
        if self.verbose:
            print(x.size())

        x = self.unpool_2(x, indices_4, output_size=size_4)
        if self.verbose:
            print(x.size())
        x = self.decoder_2(x)
        if self.verbose:
            print(x.size())

        x = self.unpool_3(x, indices_3, output_size=size_3)
        if self.verbose:
            print(x.size())
        x = self.decoder_3(x)
        if self.verbose:
            print(x.size())

        x = self.unpool_4(x, indices_2, output_size=size_2)
        if self.verbose:
            print(x.size())
        x = self.decoder_4(x)
        if self.verbose:
            print(x.size())

        x = self.unpool_5(x, indices_1, output_size=size_1)
        if self.verbose:
            print(x.size())
        x = self.decoder_5(x)
        if self.verbose:
            print(x.size())

        x = self.classifier(x)
        if self.verbose:
            print(x.size())
        return self.softmax(x)


if __name__ == "__main__":
    x = torch.randn((1, 3, 320, 320))
    net = SegNet(2, True)
    y = net(x)
