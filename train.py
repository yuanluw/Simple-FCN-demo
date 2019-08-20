# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2019/8/10 0010, matt '


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.autograd import Variable

import os
from datetime import datetime

from model import *
from utils import *
from dataset import *

cur_path = os.path.abspath(os.path.dirname(__file__))


def train(net, train_data, val_data, optimizer, criterion, arg):

    net = net.cuda()
    best_acc = 0
    best_state_dict = 0
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=arg.gamma, )
    if arg.use_visdom:
        viz = Display_board(env_name="fcn train")
        train_acc_win = viz.add_Line_windows(name="train_acc")
        train_loss_win = viz.add_Line_windows(name="train_loss")
        val_acc_win = viz.add_Line_windows(name="val_acc")
        val_loss_win = viz.add_Line_windows(name="val_loss")
        train_y_axis = 0
        val_y_axis = 0
    print("start training: ", datetime.now())

    for epoch in range(arg.epochs):

        # train stage
        train_loss = 0.0
        train_acc = 0.0
        net = net.train()
        i = 0
        if arg.use_visdom is not True:
            prev_time = datetime.now()
        for im, mask in train_data:
            i += 1  # train number
            im = Variable(im.cuda())
            mask = Variable(mask.cuda())
            output = net(im)
            loss = criterion(output, mask)
            optimizer.zero_grad()
            loss.backward()
            # for item in net.named_parameters():
            #        h = item[1].register_hook(lambda grad: print(grad))
            optimizer.step()

            cur_loss = loss.item()
            cur_acc = accuracy(output, mask)

            train_loss += cur_loss
            train_acc += cur_acc

            # visualize curve
            if arg.use_visdom:
                train_y_axis += 1
                viz.update_line(w=train_acc_win, Y=cur_acc.data.cpu(), X=train_y_axis)
                viz.update_line(w=train_loss_win, Y=cur_loss, X=train_y_axis)
            else:
                now_time = datetime.now()
                time_str = count_time(prev_time, now_time)
                print("train: current (%d/%d) batch loss is %f acc is %f time is %s" % (i, len(train_data), cur_loss,
                                                                                        cur_acc, time_str))
                prev_time = now_time

        print("train: the (%d/%d) epochs acc: %f loss: %f, cur time: %s" %(epoch, arg.epochs, train_acc/len(
            train_data), train_loss/len(train_data), str(datetime.now())))

        # val stage
        if val_data is not None:
            val_loss = 0.0
            val_acc = 0.0
            val_f1_score, val_mcc = 0, 0
            net = net.eval()
            j = 0
            if arg.use_visdom is not True:
                prev_time = datetime.now()
            for im, mask in val_data:
                j += 1
                with torch.no_grad():
                    im = Variable(im.cuda())
                    mask = Variable(mask.cuda())
                output = net(im)
                loss = criterion(output, mask)

                cur_acc = accuracy(output, mask)
                cur_loss = loss.item()
                cur_f1_score, cur_mcc = get_F1_and_MCC(output.data.cpu(), mask.data.cpu())
                val_f1_score += cur_f1_score
                val_mcc += cur_mcc
                val_acc += cur_acc
                val_loss += cur_loss

                # visualize curve
                if arg.use_visdom:
                    val_y_axis += 1
                    viz.update_line(w=val_acc_win, Y=cur_acc.data.cpu(), X=val_y_axis)
                    viz.update_line(w=val_loss_win, Y=cur_loss, X=val_y_axis)
                else:
                    now_time = datetime.now()
                    time_str = count_time(prev_time, now_time)
                    print(
                        "val: current (%d/%d) batch loss is %f acc is %f time is %s" % (j, len(val_data), cur_loss,
                                                                                          cur_acc, time_str))
                    prev_time = now_time

            print("val: the (%d/%d) epochs acc: %f loss: %f f1_score: %f mcc: %f, cur time: %s" % (epoch, arg.epochs,
                val_acc / len(val_data), val_loss / len(val_data), val_f1_score/len(val_data), val_mcc/len(val_data),
                                                                                                   str(datetime.now())))
            if best_acc < val_acc / len(val_data):
                best_acc = val_acc / len(val_data)
                best_state_dict = net.state_dict()

        scheduler.step()
    print("end time: ", datetime.now())
    if os.path.exists(os.path.join(cur_path, "pre_train", str(arg.net + "_.pkl"))):
        os.remove(os.path.join(cur_path, "pre_train", str(arg.net + "_.pkl")))
    torch.save(best_state_dict, os.path.join(cur_path, "pre_train", str(arg.net + "_.pkl")))


def run(arg):
    print("lr %f, epoch_num %d, decay_rate %f pre_train %d gamma %f" % (arg.lr, arg.epochs, arg.decay,
                                                                        arg.pre_train, arg.gamma))

    train_data = get_dataset(arg, train=True)
    val_data = get_dataset(arg, train=False)

    if arg.net == "SegNet":
        net = SegNet()
    elif arg.net == "FCN8s":
        resnet = models.resnet34(pretrained=True)
        net = FCN8s(resnet)
    elif arg.net == "MNet":
        net = MNet()
    elif arg.net == "FCDenseNet57":
        net = FCDenseNet57()
    elif arg.net == "FCDenseNet103":
        net = FCDenseNet103()

    if arg.mul_gpu:
        net = nn.DataParallel(net)

    if arg.pre_train:
        net.load_state_dict(torch.load(os.path.join(cur_path, "pre_train", str(arg.net + "_.pkl"))))

    optimizer = optim.Adam(net.parameters(), lr=arg.lr, weight_decay=arg.decay)
    criterion = nn.NLLLoss().cuda()

    train(net, train_data, val_data, optimizer, criterion, arg)



















