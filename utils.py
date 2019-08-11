# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/10 0010, matt '


from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch.nn import functional as F


def count_time(prev_time, cur_time):
    h, reminder = divmod((cur_time-prev_time).seconds, 3600)
    m, s = divmod(reminder, 60)
    time_str = "time %02d:%02d:%02d" %(h, m, s)
    return time_str


def accuracy(output, target):
    v, i = torch.max(output, 1)
    return torch.sum(target.long() == i).float()/target.numel()


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input (n, c, h, w) , target(n, h, w)
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


class Display_board:
    def __init__(self, port=8097, viz=None, env_name=None):
        if viz is None:
            self.viz = Visdom(port=port, env=env_name)
        else:
            self.viz = viz

    def add_Line_windows(self, name, X=0, Y=0):

        w = self.viz.line(X=np.array([X]), Y=np.array([Y]), opts=dict(title=name))
        return w

    def update_line(self, w, X, Y):
        self.viz.line(X=np.array([X]), Y=np.array([Y]), win=w, update="append")

    def show_image(self, image):
        plt.imshow(image)
        self.viz.matplot(plt)

