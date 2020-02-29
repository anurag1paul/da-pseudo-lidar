########################################################
# NOTE : NOT ACTUALLY MMD, IT USES WDGRL METHOD
########################################################








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
parser.add_argument('--savemodel', default='./psmnet/trained_da_wdgrl/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='S',
                    help='learning rate(default: 0.001)')
parser.add_argument('--lr_scale', type=int, default=100, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--split_file', default='Kitti/object/train.txt',
                    help='save model')
parser.add_argument('--btrain', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--start_epoch', type=int, default=1)

parser.add_argument('--k-critic', type=int, default=5)
parser.add_argument('--wd-clf', type=float, default=1)
parser.add_argument('--gamma', type=float, default=10)

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
    args.datapath+'/virtual_kitti/', "train")

val_left_img_v, val_right_img_v, val_left_disp_v = VKitti.dataloader(
    args.datapath+'/virtual_kitti/', "val")



half_batch_size = args.btrain // 2



# source dataset - vkitti
TrainImgLoader_vkitti = torch.utils.data.DataLoader(
    VKitti.ImageLoader(all_left_img_v, all_right_img_v, all_left_disp_v, True),
    batch_size=half_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

ValImgLoader_vkitti = torch.utils.data.DataLoader(
    VKitti.ImageLoader(val_left_img_v, val_right_img_v, val_left_disp_v, False),
    batch_size=half_batch_size*2, shuffle=False, num_workers=args.num_workers, drop_last=False)



# target dataset - kitti
TrainImgLoader_kitti = torch.utils.data.DataLoader(
    kitti_loader.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=half_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

ValImgLoader_kitti = torch.utils.data.DataLoader(
    kitti_loader.myImageFloder(val_left_img, val_right_img, val_left_disp, False),
    batch_size=half_batch_size*2, shuffle=False, num_workers=args.num_workers, drop_last=False)



# define critic model
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


critic = nn.Sequential(

        nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
        
        nn.Conv3d(32, 32, kernel_size=3, padding=1,stride=1, bias=False),
        nn.BatchNorm3d(32),
        nn.ReLU(inplace=True),

        nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),

        nn.Conv3d(32, 1, kernel_size=3, padding=1,stride=1, bias=False),
        nn.BatchNorm3d(1),
        nn.ReLU(inplace=True),

        Flatten(),

        nn.Linear( 12*16*32 , 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )



if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
elif args.model == 'basic_mmd':
    model = basic_mmd(args.maxdisp)
else:
    print('no model')

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

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr,  betas=(0.9, 0.999))


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def gradient_penalty_lin(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    # print(h_s.size())
    alpha = torch.rand(h_s.size(0), 1, 1, 1, 1)
    if args.cuda:
        alpha = alpha.cuda()

    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty



def gradient_penalty(critic, h_s, h_t):
    # print "h_s: ", h_s.size(), h_t.size()
    
    use_cuda = args.cuda
    BATCH_SIZE = h_s.size(0)

    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, h_s.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 32, 48, 64, 128)
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * h_s + ((1 - alpha) * h_t)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = critic(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() #* LAMBDA
    return gradient_penalty





def train(imgL_s, imgR_s, disp_L_s, imgL_t, imgR_t ):

    model.train()

    if args.cuda:
        imgL_s, imgR_s, disp_L_s = imgL_s.cuda(), imgR_s.cuda(), disp_L_s.cuda()
        imgL_t, imgR_t = imgL_t.cuda(), imgR_t.cuda()

    # ---------
    mask = (disp_L_s > 0)
    mask.detach_()
    # ----

    # print( [x.shape for x in [imgL_s, imgR_s, disp_L_s, imgL_t, imgR_t] ] )


    total_loss = 0
    avg_gp = 0


    # Train critic
    set_requires_grad(model, requires_grad=False)
    set_requires_grad(critic, requires_grad=True)

    with torch.no_grad():
        _, h_s = model(imgL_s, imgR_s)
        _, h_t = model(imgL_t, imgR_t)


    for _ in range(args.k_critic):
        gp = gradient_penalty(critic, h_s, h_t)

        # TODO : gradient penalty expects flattened outputs. Will it work with conv features?

        critic_s = critic(h_s)
        critic_t = critic(h_t)
        wasserstein_distance = critic_s.mean() - critic_t.mean()

        critic_cost = -wasserstein_distance + args.gamma*gp

        critic_optim.zero_grad()
        critic_cost.backward()
        critic_optim.step()

        total_loss += critic_cost.item()
        avg_gp +=  (args.gamma*gp).item()

    critic_loss = total_loss /  args.k_critic
    avg_gp = avg_gp /  args.k_critic




    # Train model
    set_requires_grad(model, requires_grad=True)
    set_requires_grad(critic, requires_grad=False)
    
    output_s, source_features = model(imgL_s, imgR_s)
    _, target_features = model(imgL_t, imgR_t)

    output_s = torch.squeeze(output_s, 1)
    clf_loss = F.smooth_l1_loss(output_s[mask], disp_L_s[mask], size_average=True)


    wasserstein_distance = critic(source_features).mean() - critic(target_features).mean()

    loss = clf_loss + args.wd_clf * wasserstein_distance
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_loss = loss.item()


    return model_loss, critic_loss, avg_gp



    """
    optimizer.zero_grad()

    if args.model == 'basic':
        output_s, feat_mmd1_s = model(imgL_s, imgR_s)
        _, feat_mmd1_t = model(imgL_t, imgR_t)
        output_s = torch.squeeze(output_s, 1)
        loss = F.smooth_l1_loss(output_s[mask], disp_L_s[mask],
                                size_average=True)

    loss.backward()
    optimizer.step()

    return loss.item()
    """


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


def adjust_learning_rate(optimizer, epoch):
    if epoch <= args.lr_scale:
        lr = args.lr
    else:
        lr = args.lr / 10
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
        adjust_learning_rate(optimizer, epoch)

        source_loader = loop_iterable(TrainImgLoader_vkitti)
        target_loader= loop_iterable(TrainImgLoader_kitti)

        epoch_num_batches = len( TrainImgLoader_vkitti )
        epoch_num_val_batches = int(len( ValImgLoader_vkitti ) // 10)

        # training
        for batch_idx in range(epoch_num_batches):

            imgL_crop_s, imgR_crop_s, disp_crop_s_L = next(source_loader)
            imgL_crop_t, imgR_crop_t, _ = next(target_loader)


            start_time = time.time()

            model_loss, critic_loss, gp_loss = train(imgL_crop_s, imgR_crop_s, disp_crop_s_L,
                        imgL_crop_t, imgR_crop_t
                        )


            if batch_idx % 5 == 0:
                print('Iter %d model loss = %.3f , critic loss = %.3f , gp loss = %.3f, time = %.2f' % (
                    batch_idx, model_loss, critic_loss,gp_loss, time.time() - start_time))

                log.info('Iter %d model loss = %.3f , critic loss = %.3f , gp loss = %.3f, time = %.2f' % (
                    batch_idx, model_loss, critic_loss,gp_loss, time.time() - start_time))

            total_train_loss += model_loss + critic_loss

        print('epoch %d total training loss = %.3f' % (
        epoch, total_train_loss / epoch_num_batches))

        log.info('epoch %d total training loss = %.3f' % (
        epoch, total_train_loss / epoch_num_batches))


        # validation
        total_val_loss = 0
        for batch_idx, (imgL, imgR, disp_L) in enumerate(ValImgLoader_vkitti):

            if batch_idx > epoch_num_val_batches:
                break
            val_start_time = time.time()
            loss = test(imgL, imgR, disp_L)
            total_val_loss += loss
        print('epoch %d total validation loss = %.3f , time = %.2f' % (
        epoch, total_val_loss / epoch_num_val_batches, time.time() - val_start_time) 
             )

        log.info('epoch %d total validation loss = %.3f , time = %.2f' % (
        epoch, total_val_loss / epoch_num_val_batches, time.time() - val_start_time) 
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
