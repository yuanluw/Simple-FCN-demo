# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/10 0010, matt '


from model.fcn import FCN8s
from model.segnet import SegNet
from model.mnet import MNet
from model.fc_densenet import *

__all__ = ["FCN8s", "SegNet", "MNet", "FCDenseNet57", "FCDenseNet67", "FCDenseNet103"]

