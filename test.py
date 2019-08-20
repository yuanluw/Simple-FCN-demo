# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/10 0010, matt '


import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

import os
from PIL import Image

from dataset import *
from model import *
from utils import *

cur_path = os.path.abspath(os.path.dirname(__file__))


def get_image(name):
    im = Image.open(os.path.join(cur_path, "dataset", "bag_data", name)).convert('RGB')
    mask = Image.open(os.path.join(cur_path, "dataset", "bag_data_msk", name)).convert('L')
    im, mask = test_transform(im, mask)
    im = im.unsqueeze(0)

    return im, mask


def inference(arg):
    viz = Display_board(env_name="fcn test")

    if arg.net == "SegNet":
        net = SegNet()
    elif arg.net == "FCN8s":
        resnet = models.resnet34(pretrained=True)
        net = FCN8s(resnet)
    elif arg.net == "MNet":
        net = MNet()

    if arg.mul_gpu:
        net = nn.DataParallel(net)
    net.load_state_dict(torch.load(os.path.join(cur_path, "pre_train", str(arg.net + "_.pkl"))))
    net = net.cuda()
    net = net.eval()

    im, mask = get_image(name=arg.test_img_name)
    mask = mask.numpy()
    viz.show_image(mask)
    plt.imshow(mask)
    im = Variable(im.cuda())
    output = net(im)

    v, i = torch.max(output, 1)
    i = i[0].cpu().data.numpy()
    viz.show_image(i)


def run(arg):
    inference(arg)



