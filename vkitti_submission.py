from __future__ import print_function

import argparse
import os
import time

import numpy as np
import skimage
import skimage.io
import skimage.transform
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

from psmnet.models import *
from psmnet.utils import preprocess
from psmnet.dataloader import VKittiLoader as DA

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datapath', default='../vkitti/',
                    help='datapath')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save_path', type=str, default='saved_figures',
                    metavar='S',
                    help='path to save the predict')
parser.add_argument('--save_figure', action='store_true',
                    help='if true, save the numpy file, not the png file')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_left_img, test_right_img, test_disp = DA.dataloader(args.datapath,
                                              "psmnet/vkitti/val.csv")

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model)
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))


def test(imgL, imgR):
    model.eval()

    if args.cuda:
        imgL = torch.from_numpy(imgL).cuda()
        imgR = torch.from_numpy(imgR).cuda()

    with torch.no_grad():
        output = model(imgL, imgR)
    output = torch.squeeze(output)
    pred_disp = output.data.cpu().numpy()

    return pred_disp


def main():
    processed = preprocess.get_transform(augment=False)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    for inx in range(5): #  len(test_left_img)):

        imgL_o = skimage.io.imread(test_left_img[inx]).astype('float32')
        imgR_o = skimage.io.imread(test_right_img[inx]).astype('float32')

        # GT
        disparity_GT = skimage.io.imread(test_disp[inx]) / 100.0
        # print(np.max(disparity_GT), np.min(disparity_GT), np.mean(disparity_GT))
        disparity = (1.5 * 0.54 * 725.0087) / (disparity_GT + 1e-6)
        if args.save_figure:
            skimage.io.imsave(
                args.save_path + '/' + test_disp[inx].split('/')[-1],
                (disparity).astype('uint16'))

        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        # pad to (384, 1248)
        height = 384
        width  = 1248

        top_pad  = height - imgL.shape[2]
        left_pad = width - imgL.shape[3]
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)),
                          mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)),
                          mode='constant', constant_values=0)

        start_time = time.time()
        pred_disp = test(imgL, imgR)  # normalize disparity to ground truth
        print('time = %.2f' % (time.time() - start_time))

        top_pad  = height - imgL_o.shape[0]
        left_pad = width - imgL_o.shape[1]
        img = pred_disp[top_pad:, :-left_pad]
        print(test_left_img[inx].split('/')[-1])
        if args.save_figure:
            skimage.io.imsave(args.save_path + '/' + \
                test_left_img[inx].split('/')[-1].split('.')[0] + '.png', # save as png
                (img * 256 / 1.5).astype('uint16')) # divide by 1.5 here to get actual depth
        else:
            np.save(
                args.save_path + '/' + test_left_img[inx].split('/')[-1][:-4],
                img)


if __name__ == '__main__':
    main()
