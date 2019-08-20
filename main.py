# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/10 0010, matt '


import argparse


def get_augments():
    parser = argparse.ArgumentParser(description="pytorch bag fcn")

    parser.add_argument("--action", type=str, default="train", choices=("train", "test", "dataset"))
    parser.add_argument("--pre_train", type=int, default=0)
    parser.add_argument("--mul_gpu", type=int, default=1, help="use multiple gpu(default: 1")
    parser.add_argument("--net", type=str, default="SegNet", choices=("FCN8s", "SegNet", "MNet", "FCDenseNet57",
                                                                      "FCDenseNet67", "FCDenseNet103"))

    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate(default: 0.1)")
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_worker", type=int, default=4, help="number of data loading workers (default: 4)")
    parser.add_argument("--train_batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--test_batch_size", type=int, default=1, help="test batch size")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule")
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--test_img_name", type=str, default="102.jpg")

    parser.add_argument("--use_visdom", type=int, default=1)

    return parser.parse_args()


def main():
    arg = get_augments()
    if arg.action == "dataset":
        from dataset import get_dataset
        train_data = get_dataset(arg, train=True)
        for im, mask in train_data:
            print(mask.size())
        val_data = get_dataset(arg, train=False)
        for im, mask in val_data:
            print(im.size())
    elif arg.action == "train":
        from train import run
        run(arg)
    elif arg.action == "test":
        from test import run
        run(arg)


if __name__ == "__main__":
    main()
