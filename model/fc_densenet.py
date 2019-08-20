# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/20 0020, matt '


import torch
import torch.nn as nn


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module("bn", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module("drop", nn.Dropout2d(0.2))

    def forward(self, input):
        return super().forward(input)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels+i*growth_rate, growth_rate) for i in range(n_layers)
        ])

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module("bn", nn.BatchNorm2d(num_features=in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True))
        self.add_module("drop", nn.Dropout2d(0.2))
        self.add_module("maxpool", nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                            stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module("bottleneck", DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True
        ))

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w-max_width)//2
    xy2 = (h-max_height)//2
    return layer[:, :, xy2:(xy2+max_height), xy1:(xy1+max_width)]


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5), up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=2, verbose=False):
        super().__init__()
        self.down_blocks =down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        self.firstConv = nn.Conv2d(in_channels=in_channels, out_channels=out_chans_first_conv, kernel_size=3,
                                   stride=1, padding=1, bias=True)

        cur_channels_count = out_chans_first_conv

        self.denseBlockDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlockDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i])
            )
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        self.bottleneck = Bottleneck(cur_channels_count, growth_rate, bottleneck_layers)
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlockUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlockUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i], upsample=True
            ))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels
        ))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]
        self.denseBlockUp.append(DenseBlock(cur_channels_count, growth_rate, up_blocks[-1], upsample=False))

        cur_channels_count += growth_rate * up_blocks[-1]
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.verbose = verbose

    def forward(self, x):
        out = self.firstConv(x)
        if self.verbose:
            print("firstconv size", out.size())

        skip_connection = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlockDown[i](out)
            if self.verbose:
                print("denseBlockDown size", out.size())
            skip_connection.append(out)
            out = self.transDownBlocks[i](out)
            if self.verbose:
                print("transDownBlocks size", out.size())

        out = self.bottleneck(out)
        if self.verbose:
            print("bottleneck size", out.size())

        for i in range(len(self.up_blocks)):
            skip = skip_connection.pop()
            out = self.transUpBlocks[i](out, skip)
            if self.verbose:
                print("transUpBlocks size", out.size())
            out = self.denseBlockUp[i](out)
            if self.verbose:
                print("denseBlockUp size", out.size())

        out = self.finalConv(out)
        if self.verbose:
            print("finalConv size", out.size())
        return self.softmax(out)


def FCDenseNet57(n_classes=2, verbose=False):
    return FCDenseNet(in_channels=3, down_blocks=(4, 4, 4, 4, 4), up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
                      growth_rate=12, out_chans_first_conv=48, n_classes=n_classes, verbose=verbose)


def FCDenseNet67(n_classes=2, verbose=False):
    return FCDenseNet(in_channels=3, down_blocks=(5, 5, 5, 5, 5), up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                      growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, verbose=verbose)


def FCDenseNet103(n_classes=2, verbose=False):
    return FCDenseNet(in_channels=3, down_blocks=(4, 5, 7, 10, 12), up_blocks=(12, 10, 7, 5, 4), bottleneck_layers=15,
                      growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, verbose=verbose)



if __name__ == "__main__":
    x = torch.randn((1, 3, 240, 240))
    net = FCDenseNet57(verbose=True)
    print(net)
    print(net(x).size())
