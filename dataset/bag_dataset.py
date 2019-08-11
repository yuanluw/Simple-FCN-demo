# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/9 0009, matt '


import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tfs

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))


def read_image(val_ratio=0.3, train=True):
    data_dir = os.path.join(cur_path, "bag_data")
    file_name = os.listdir(data_dir)
    img_path = [os.path.join(cur_path, "bag_data", file_name[i]) for i in range(len(file_name))]
    mask_path = [os.path.join(cur_path, "bag_data_msk", file_name[i]) for i in range(len(file_name))]
    count = int(len(file_name)*val_ratio)

    if train:
        return img_path[count:], mask_path[count:]
    else:
        return img_path[:count], mask_path[:count]


def train_transform(img, mask):
    im_aug = tfs.Compose([
        tfs.Resize((320, 320)),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = im_aug(img)
    mask = tfs.Resize((320, 320))(mask)
    mask = np.array(mask, dtype=np.int64)
    mask[mask > 0] = -1
    mask += 1
    mask = torch.from_numpy(mask)
    return img, mask


def test_transform(img, mask):
    im_aug = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = im_aug(img)
    mask = np.array(mask, dtype=np.int64)
    mask[mask > 0] = -1
    mask += 1
    mask = torch.from_numpy(mask)
    return img, mask


class Bag_dataset(Dataset):
    def __init__(self, train=True, transforms=None, val_ratio=0.3):
        self.transforms = transforms
        self.data_path, self.mask_path = read_image(val_ratio, train)
        print("read " + str(len(self.data_path)) + " images")

    def __getitem__(self, item):
        img_path = self.data_path[item]
        mask_path = self.mask_path[item]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.data_path)


def get_dataset(arg, train=True):

    dataset = Bag_dataset(train, train_transform, arg.val_ratio)
    data_loader = DataLoader(dataset, arg.train_batch_size if train else arg.test_batch_size, shuffle=True,
                             num_workers=arg.num_worker)

    return data_loader


if __name__ == "__main__":
    print(read_image())
