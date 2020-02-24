from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from psmnet.utils import logger
from psmnet.dataloader import VKittiLoader as VKitti
from psmnet.models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=160,
                    help='maximum disparity')  # 65516
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='../vkitti',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='psmnet/trained/pretrained_sceneflow.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./psmnet/trained/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.002, metavar='S',
                    help='learning rate(default: 0.001)')
parser.add_argument('--lr_scale', type=int, default=100, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--split_file', default='Kitti/object/train.txt',
                    help='save model')
parser.add_argument('--btrain', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--start_epoch', type=int, default=1)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.isdir(args.savemodel):
    os.makedirs(args.savemodel)
print(os.path.join(args.savemodel, 'training.log'))
log = logger.setup_logger(os.path.join(args.savemodel, 'training.log'))

all_left_img, all_right_img, all_left_disp = VKitti.dataloader(
    args.datapath, "psmnet/vkitti/train.csv")

val_left_img, val_right_img, val_left_disp = VKitti.dataloader(
    args.datapath, "psmnet/vkitti/val.csv")

TrainImgLoader = torch.utils.data.DataLoader(
    VKitti.ImageLoader(all_left_img, all_right_img, all_left_disp, True),
    batch_size=args.btrain, shuffle=True, num_workers=args.num_workers, drop_last=False)

ValImgLoader = torch.utils.data.DataLoader(
    VKitti.ImageLoader(val_left_img, val_right_img, val_left_disp, False),
    batch_size=args.btrain, shuffle=False, num_workers=args.num_workers, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    log.info('load model ' + args.loadmodel)
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

def train(imgL, imgR, disp_L):
    model.train()

    if args.cuda:
        imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_L > 0)
    mask.detach_()
    print(mask.size())
    # ----

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = (0.5 * F.smooth_l1_loss(output1[mask], disp_L[mask],
                                      size_average=True)
               + 0.7 * F.smooth_l1_loss(output2[mask], disp_L[mask],
                                        size_average=True)
               + F.smooth_l1_loss(output3[mask], disp_L[mask],
                                  size_average=True))
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_L[mask],
                                size_average=True)

    loss.backward()
    optimizer.step()

    return loss.item()


def test(imgL, imgR, disp_true):
    model.eval()
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    pred_disp = output3.data.cpu()

    # computing 3-px error#
    true_disp = disp_true
    index = np.argwhere(true_disp > 0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[
            index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
            disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
        index[0][:], index[1][:], index[2][:]] * 0.05)
    torch.cuda.empty_cache()

    return 1 - (float(torch.sum(correct)) / float(len(index[0])))


def adjust_learning_rate(optimizer, epoch):
    if epoch <= args.lr_scale:
        lr = args.lr
    else:
        lr = args.lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    max_acc = 0
    max_epo = 0
    start_full_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)

        # training
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(
                TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            if batch_idx % 5 == 0:
                print('Iter %d training loss = %.3f , time = %.2f' % (
                    batch_idx, loss, time.time() - start_time))
            total_train_loss += loss

        print('epoch %d total training loss = %.3f' % (
        epoch, total_train_loss / len(TrainImgLoader)))

        # validation
        total_val_loss = 0
        for batch_idx, (imgL, imgR, disp_L) in enumerate(ValImgLoader):
            val_start_time = time.time()
            loss = test(imgL, imgR, disp_L)
            total_val_loss += loss
            if batch_idx == 50:
                break
                
        print('epoch %d total validation loss = %.3f' % (
        epoch, total_val_loss / len(ValImgLoader)))

        # SAVE
        if not os.path.isdir(args.savemodel):
            os.makedirs(args.savemodel)
        savefilename = args.savemodel + '/finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)


if __name__ == '__main__':
    main()
