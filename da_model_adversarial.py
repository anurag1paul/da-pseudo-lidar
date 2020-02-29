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
from torch import autograd

from psmnet.models.mmd_loss import MMD_loss

from psmnet.utils import logger


from psmnet.dataloader import KITTILoader3D as kitti_ls
from psmnet.dataloader import KITTILoader_dataset3d as kitti_loader

from psmnet.dataloader import VKittiLoader as VKitti


from psmnet.models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=160,
                    help='maximum disparity')  # 65516
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/data/datasets',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--loadcritic', default=None,
                    help='load critic model')
parser.add_argument('--savemodel', default='./psmnet/trained_da_adv/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='S',
                    help='learning rate(default: 0.001)')
parser.add_argument('--lr_dis', type=float, default=0.0001, metavar='S',
                    help='learning rate(default: 0.001)')
parser.add_argument('--lr_scale', type=int, default=7, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--split_file', default='Kitti/object/train.txt',
                    help='save model')
parser.add_argument('--btrain', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--start_epoch', type=int, default=1)

parser.add_argument('--k-critic', type=int, default=1)
parser.add_argument('--iter_size', type=int, default=4)

parser.add_argument("--lambda_adv_target", type=float, default=0.001,
                    help="lambda_adv for adversarial training.")


# --loadmodel psmnet/trained_da_wdgrl/finetune_4.tar --loadcritic psmnet/trained_da_wdgrl/finetune_critic4.tar 

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

if not os.path.isdir(args.savemodel):
    os.makedirs(args.savemodel)
print(os.path.join(args.savemodel, 'training.log'))
log = logger.setup_logger(os.path.join(args.savemodel, 'training.log'))

all_left_img, all_right_img, all_left_disp = kitti_ls.dataloader(
    args.datapath+'/kitti/training/', "psmnet/kitti/train.txt")

val_left_img, val_right_img, val_left_disp = kitti_ls.dataloader(
    args.datapath+'/kitti/training/', "psmnet/kitti/val.txt")


all_left_img_v, all_right_img_v, all_left_disp_v = VKitti.dataloader(
    args.datapath+'/virtual_kitti/', "psmnet/vkitti/train.csv")

val_left_img_v, val_right_img_v, val_left_disp_v = VKitti.dataloader(
    args.datapath+'/virtual_kitti/', "psmnet/vkitti/val.csv")


half_batch_size = args.btrain // 2



# source dataset - vkitti
TrainImgLoader_vkitti = torch.utils.data.DataLoader(
    VKitti.ImageLoader(all_left_img_v, all_right_img_v, all_left_disp_v, True),
    batch_size=half_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

ValImgLoader_vkitti = torch.utils.data.DataLoader(
    VKitti.ImageLoader(val_left_img_v, val_right_img_v, val_left_disp_v, False),
    batch_size=half_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)



# target dataset - kitti
TrainImgLoader_kitti = torch.utils.data.DataLoader(
    kitti_loader.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=half_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

ValImgLoader_kitti = torch.utils.data.DataLoader(
    kitti_loader.myImageFloder(val_left_img, val_right_img, val_left_disp, False),
    batch_size=half_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)



# define critic model
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)



model = None
critic = None


# labels for adversarial training
source_label = 0
target_label = 1

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
elif args.model == 'basic_mmd':
    model = basic_mmd(args.maxdisp)
elif args.model == 'basic_adv':

    print('\n\nUsing basic adversarial model\n\n')

    model = basic_adv(args.maxdisp)
    critic = fcdiscriminator(args.maxdisp)
else:
    print('no model')


critic_criterion = torch.nn.MSELoss().cuda()


if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

    critic = nn.DataParallel(critic)
    critic.cuda()

if args.loadmodel is not None:
    log.info('load model ' + args.loadmodel)
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

    log.info('load critic ' + args.loadcritic)
    state_dict = torch.load(args.loadcritic)
    critic.load_state_dict(state_dict['state_dict'])



print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_dis,  betas=(0.9, 0.99))


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad



def train( source_data, target_data ):

    model.train()
    critic.train()


    # print( [x.shape for x in [imgL_s, imgR_s, disp_L_s, imgL_t, imgR_t] ] )


    total_loss = 0
    model_loss = 0
    critic_loss = 0
    target_adv_loss = 0

    optimizer_critic.zero_grad()
    optimizer.zero_grad()


    # accumulate gradients over iter_size number of iterations
    for sub_i in range(args.iter_size):


        # get data
        imgL_s, imgR_s, disp_L_s = source_data[sub_i]
        imgL_t, imgR_t = target_data[sub_i]

        # ---------
        mask = (disp_L_s > 0)
        mask.detach_()
        # ----


        #######################################################
        # train G
        #######################################################

        # don't accumulate grads in D
        set_requires_grad(critic, requires_grad=False)
        

        # train with source

        output_s, source_features = model(imgL_s, imgR_s)
        output_s = torch.squeeze(output_s, 1)
        clf_loss = F.smooth_l1_loss(output_s[mask], disp_L_s[mask], size_average=True)

        loss = clf_loss

        # proper normalization
        loss = loss / args.iter_size
        loss.backward()
        model_loss += loss.item()



        # train with target

        _, target_features = model(imgL_t, imgR_t)


        D_out1 = critic(target_features)

        loss_adv_target = critic_criterion(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())

        loss = args.lambda_adv_target * loss_adv_target
        loss = loss / args.iter_size
        loss.backward()
        target_adv_loss += loss.item()
        # print('loss_adv_target:',loss_adv_target, target_adv_loss)





        #######################################################
        # train D
        #######################################################


        # bring back requires_grad
        set_requires_grad(critic, requires_grad=True)


        # train with source
        source_features = source_features.detach()

        D_out1 = critic(source_features)

        loss_D1 = critic_criterion(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())

        loss_D1 = loss_D1 / args.iter_size / 2

        loss_D1.backward()

        critic_loss += loss_D1.item()



        # train with target
        target_features = target_features.detach()

        D_out1 = critic(target_features)

        loss_D1 = critic_criterion(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).cuda())

        loss_D1 = loss_D1 / args.iter_size / 2

        loss_D1.backward()

        critic_loss += loss_D1.item()


    # update weights
    optimizer.step()
    optimizer_critic.step()



    return model_loss, critic_loss, target_adv_loss





def test(imgL, imgR, disp_true):
    model.eval()
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3,_ = model(imgL, imgR)

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


def adjust_learning_rate(optimizer, epoch, org_lr):
    lr = org_lr
    if epoch <= args.lr_scale:
        lr = org_lr
    else:
        lr = org_lr / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def loop_iterable(iterable):
    while True:
        yield from iterable


def main():
    max_acc = 0
    max_epo = 0
    start_full_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch, args.lr)
        adjust_learning_rate(optimizer_critic, epoch, args.lr_dis)


        source_loader = loop_iterable(TrainImgLoader_vkitti)
        target_loader= loop_iterable(TrainImgLoader_kitti)

        epoch_num_batches = len( TrainImgLoader_vkitti ) // args.iter_size
        epoch_num_val_batches = int(len( ValImgLoader_vkitti ) // 5)


        # training
        for batch_idx in range(epoch_num_batches):


            source_data = []
            target_data = []

            for ind in range(args.iter_size):

                imgL_s, imgR_s, disp_L_s = next(source_loader)
                imgL_t, imgR_t, _ = next(target_loader)

                if args.cuda:
                    imgL_s, imgR_s, disp_L_s = imgL_s.cuda(), imgR_s.cuda(), disp_L_s.cuda()
                    imgL_t, imgR_t = imgL_t.cuda(), imgR_t.cuda()


                source_data.append( [imgL_s, imgR_s, disp_L_s] )
                target_data.append( [imgL_t, imgR_t] )


            start_time = time.time()

            model_loss, critic_loss, target_adv_loss = train( source_data, target_data )


            if batch_idx % 2 == 0:
                # print('Iter %d/%d model_loss = %.4f , critic_loss = %.4f , target_adv_loss = %.4f, batchtime = %.2f' % (
                #     batch_idx, epoch_num_batches, model_loss, critic_loss, target_adv_loss, (time.time() - start_time)/args.iter_size))

                log.info('Iter %d/%d model_loss = %.4f , critic_loss = %.4f , target_adv_loss = %.4f, batchtime = %.2f' % (
                    batch_idx, epoch_num_batches, model_loss, critic_loss, target_adv_loss, (time.time() - start_time)/args.iter_size))

            total_train_loss += model_loss + critic_loss

        # print('epoch %d total training loss = %.4f' % (
        # epoch, total_train_loss / epoch_num_batches))

        log.info('epoch %d total training loss = %.4f' % (
        epoch, total_train_loss / epoch_num_batches))


        # validation
        total_val_loss = 0
        val_zero_start_time = time.time()

        for batch_idx, (imgL, imgR, disp_L) in enumerate(ValImgLoader_vkitti):

            if batch_idx > epoch_num_val_batches:
                break
            val_start_time = time.time()
            loss = test(imgL, imgR, disp_L)
            total_val_loss += loss

            if (batch_idx % 2 == 0 ):
                log.info('Val_iter %d/%d loss = %.4f , batchtime = %.2f' % (
                    batch_idx, epoch_num_val_batches, loss, time.time() - val_start_time))




        # print('epoch %d total validation loss = %.4f , time = %.2f' % (
        # epoch, total_val_loss / epoch_num_val_batches, time.time() - val_zero_start_time) 
        #      )

        log.info('epoch %d total validation loss = %.4f , time = %.2f' % (
        epoch, total_val_loss / epoch_num_val_batches, time.time() - val_zero_start_time) 
             )

        # SAVE
        if not os.path.isdir(args.savemodel):
            os.makedirs(args.savemodel)

        # save model
        savefilename = args.savemodel + '/finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / epoch_num_batches,
        }, savefilename)

        # save critic
        savefilename = args.savemodel + '/finetune_critic' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': critic.state_dict(),
            'train_loss': total_train_loss / epoch_num_batches,
        }, savefilename)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
