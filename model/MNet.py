# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/15 0015, matt '


import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, verbose=False):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes//map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
            BasicConv(inter_planes*2, 2*inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2*inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=(7, 1), stride=stride, padding=(3, 0)),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=(7, 1), stride=stride, padding=(3, 0)),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=7, dilation=7, relu=False)
        )
        self.ConvLinear = BasicConv(8*inter_planes, out_planes, kernel_size=1, stride=1, relu=True)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.verbose = verbose

    def forward(self, x):
        if self.verbose:
            print("raw size: ", x.size())
        x0 = self.branch0(x)
        if self.verbose:
            print("branch0 size: ", x0.size())
        x1 = self.branch1(x)
        if self.verbose:
            print("branch1 size: ", x1.size())
        x2 = self.branch2(x)
        if self.verbose:
            print("branch2 size: ", x2.size())
        x3 = self.branch3(x)
        if self.verbose:
            print("branch3 size: ", x3.size())

        out = torch.cat((x0, x1, x2, x3), 1)
        if self.verbose:
            print("cat size: ", out.size())
        out = self.ConvLinear(out)
        if self.verbose:
            print("ConvLinear size: ", out.size())
        short = self.shortcut(x)
        if self.verbose:
            print("short size: ", short.size())
        out = out*self.scale + short
        if self.verbose:
            print("short size: ", out.size())
        return out


class BasicRFB_A(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, map_reduce=8, verbose=False):
        super(BasicRFB_A, self).__init__()

        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(inter_planes * 2, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(7, 1), stride=stride, padding=(3, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(7, 1), stride=stride, padding=(3, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=7, dilation=7, relu=False)
        )
        self.ConvLinear = BasicConv(8 * inter_planes, out_planes, kernel_size=1, stride=1, relu=True)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.verbose = verbose

        self.fc1 = nn.Linear(in_features=out_planes, out_features=out_planes//16)
        self.fc2 = nn.Linear(in_features=out_planes//16, out_features=out_planes)
        self.sigmoid = nn.Sigmoid()
        self.globalAvgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(True)

    def forward(self, x):

        if self.verbose:
            print("raw size: ", x.size())
        x0 = self.branch0(x)
        if self.verbose:
            print("branch0 size: ", x0.size())
        x1 = self.branch1(x)
        if self.verbose:
            print("branch1 size: ", x1.size())
        x2 = self.branch2(x)
        if self.verbose:
            print("branch2 size: ", x2.size())
        x3 = self.branch3(x)
        if self.verbose:
            print("branch3 size: ", x3.size())

        out = torch.cat((x0, x1, x2, x3), 1)
        if self.verbose:
            print("cat size: ", out.size())
        out = self.ConvLinear(out)
        if self.verbose:
            print("ConvLinear size: ", out.size())

        origin_out = out
        out = self.globalAvgpool(out)
        if self.verbose:
            print("global size: ", out.size())
        out = out.view(out.size(0), -1)
        if self.verbose:
            print("view size: ", out.size())
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out*origin_out

        short = self.shortcut(x)
        if self.verbose:
            print("short size: ", short.size())
        out = out + short
        if self.verbose:
            print("short size: ", out.size())
        return out


class MNet(nn.Module):
    def __init__(self, num_classes=2, verbose=False):
        super(MNet, self).__init__()
        pretrained_net = models.resnet50(pretrained=True)
        self.encoder_1 = nn.Sequential(
            BasicConv(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            pretrained_net.layer1,
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )
        self.encoder_2 = nn.Sequential(
            BasicRFB(256, 512, 1),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )

        self.encoder_3 = nn.Sequential(
            BasicRFB(512, 1024, 1),
            nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
        )

        self.unpool = nn.MaxUnpool2d(2, stride=2)

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512)
        )  # first group

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256)
        )  # second group

        self.decoder_3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )  # third group

        self.classifier = nn.Conv2d(64, num_classes, 1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.verbose = verbose

    def forward(self, x):
        size_1 = x.size()
        x1, indices_1 = self.encoder_1(x)
        if self.verbose:
            print(size_1, x1.size())

        size_2 = x1.size()
        x2, indices_2 = self.encoder_2(x1)
        if self.verbose:
            print(size_2, x2.size())

        size_3 = x2.size()
        x3, indices_3 = self.encoder_3(x2)
        if self.verbose:
            print(size_3, x3.size())

        x = self.unpool(x3, indices_3, output_size=size_3)
        if self.verbose:
            print(x.size())
        x = self.decoder_1(x)
        x = x + x2
        if self.verbose:
            print(x.size())

        x = self.unpool(x, indices_2, output_size=size_2)
        if self.verbose:
            print(x.size())
        x = self.decoder_2(x)
        x = x+x1
        if self.verbose:
            print(x.size())

        x = self.unpool(x, indices_1, output_size=size_1)
        if self.verbose:
            print(x.size())
        x = self.decoder_3(x)

        if self.verbose:
            print(x.size())

        x = self.classifier(x)
        return self.softmax(x)


if __name__ == "__main__":
    x = torch.randn((1, 3, 320, 320))
    net = MNet(verbose=True)
    print(net(x).size())


