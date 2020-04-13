#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: Xu Kaixin
@License: Apache Licence
@Time: 2020.03.24 : 上午 9:14
@File Name: train_net.py
@Software: PyCharm
-----------------
"""

import os.path as osp
import random
import torch
import torch.nn as nn
import numpy as np
from dataloader import KittiGraph, KittiStandard
from point_net import Net
from utils import save_pose_predictions, AverageMeter, dDataLoaderX, gDataLoaderX, PadCollate, l2reg, ComposeAdapt, pose_error, val7_to_matrix
import tqdm
import re
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import GridSampling, RandomTranslate, NormalizeScale
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default="/content/drive/My Drive/dataset")
parser.add_argument("--n_fold", default=5, type=int)
parser.add_argument("-b", "--batch_size", default=1, type=int)
parser.add_argument("-e", "--epoch", default=30, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("-wd", "--weight_decay", default=5e-5, type=float)
parser.add_argument("-gs", "--grid_size", help="gridsampling size", type=float, default=1.)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-g", "--gpu_first", default=True, type=bool)
parser.add_argument("--model_pth", type=str)
parser.add_argument("--weights_dir", type=str, default="weights")
parser.add_argument("--num_workers", default=1, type=int)
parser.add_argument("-ld", "--lr_decay", default=10, type=int)
parser.add_argument("--log", default="log", type=str)
parser.add_argument("--dof", default=7, type=int)
parser.add_argument("--save_strategy", default="trainloss", type=str)
parser.add_argument("--reg_lambda", default=10e-4, type=float)
parser.add_argument("-rt", "--random_trans", default=1e-3, type=float)
args = parser.parse_args()
args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg]) for arg in args_dict]))
print('=' * 30)

N_FOLD = 5
BATCH_SIZE = args.batch_size
EPOCH = args.epoch
LR = args.lr
LR_DECAY = args.lr_decay
GRID_SAMPLE_SIZE = [args.grid_size] * 3
LOAD_GRAPH = False
DOF = args.dof
WEIGHT_DECAY = args.weight_decay
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
Kitti = KittiGraph if LOAD_GRAPH else KittiStandard
L2_LAMBDA = args.reg_lambda

transform_dict = OrderedDict()
transform_dict[GridSampling(GRID_SAMPLE_SIZE)] = ["train", "test"]
transform_dict[NormalizeScale()] = ["train", "test"]
transform_dict[RandomTranslate(args.random_trans)] = ["train"]
transform = ComposeAdapt(transform_dict)
#transform = GridSampling(GRID_SAMPLE_SIZE)


def adjust_lr(optimizer, epoch):
    lr = LR * (0.1 ** (epoch / LR_DECAY))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, epoch, train_loader, optimizer, criterion_x, criterion_rot):
    model.train()
    temp_loss = AverageMeter()
    temp_mse_loss = AverageMeter()
    temp_x_loss = AverageMeter()
    temp_rot_loss = AverageMeter()
    trans_error_meter = AverageMeter()
    rot_error_meter = AverageMeter()

    with tqdm.tqdm(len(train_loader)) as pbar:
        for data in train_loader:
            data = data.to(device) if not isinstance(data, (tuple, list)) else tuple([d.to(device) for d in data])
            optimizer.zero_grad()
            # loss = F.nll_loss(model(data), data.y)
            gt_poses = data.y.reshape(train_loader.batch_size, -1).float() if LOAD_GRAPH else data[-1].float()
            pred_pose = model(data)
            if len(gt_poses) == 2:
                print(data[0].shape)
                print(pred_pose.shape)
            x_loss = criterion_x(pred_pose[:, :3], gt_poses[:, :3])
            try:
                rot_loss = criterion_rot(pred_pose[:, 3:] / torch.norm(pred_pose[:, 3:], dim=1), gt_poses[:, 3:])

                mse_loss = torch.exp(-model.sx) * x_loss + torch.exp(-model.sq) * rot_loss
                loss = L2_LAMBDA * l2reg(model) + mse_loss + model.sx + model.sq
                loss.backward()
                optimizer.step()

                pose_error_avg = np.mean(np.array([list(pose_error(val7_to_matrix(gt.squeeze().cpu().detach().numpy()), val7_to_matrix(est.squeeze().cpu().detach().numpy())))
                                                   for gt, est in zip(data[-1], pred_pose)]), 0)
                temp_loss.update(loss.item())
                temp_mse_loss.update(mse_loss.item())
                temp_rot_loss.update(rot_loss.item())
                temp_x_loss.update(x_loss.item())
                pbar.set_postfix(OrderedDict(loss=loss.item(),
                                             mse_loss=mse_loss.item(),
                                             rot_loss=rot_loss.item(),
                                             x_loss=x_loss.item(),
                                             sx=float(model.sx.detach().cpu()),
                                             sq=float(model.sq.detach().cpu()),
                                             pose_error_avg=list(pose_error_avg)))
                trans_error_meter.update(pose_error_avg[0])
                rot_error_meter.update(pose_error_avg[1])
            except RuntimeError as e:
                print(gt_poses.shape, e)
            pbar.update(1)
    print(f"Translation Error: {trans_error_meter.avg:.4f}\tRot Error: {rot_error_meter.avg}")
    return temp_loss.avg, temp_mse_loss.avg, temp_x_loss.avg, temp_rot_loss.avg


@torch.no_grad()
def val(model, loader, criterion_x, criterion_rot):
    model.eval()
    temp_loss = AverageMeter()
    temp_x_loss = AverageMeter()
    temp_rot_loss = AverageMeter()
    trans_error_meter = AverageMeter()
    rot_error_meter = AverageMeter()

    pred_poses = []
    with tqdm.tqdm(len(loader)) as pbar:
        for data in loader:
            data = data.to(device) if not isinstance(data, (tuple, list)) else tuple([d.to(device) for d in data])
            # loss = F.nll_loss(model(data), data.y)
            gt_poses = data.y.reshape(loader.batch_size, -1).float() if LOAD_GRAPH else data[-1].float()
            pred_pose = model(data)
            x_loss = criterion_x(pred_pose[:, :3], gt_poses[:, :3])
            rot_loss = criterion_rot(pred_pose[:, 3:] / torch.norm(pred_pose[:, 3:], dim=1), gt_poses[:, 3:])
            loss = torch.exp(-model.sx) * x_loss + torch.exp(-model.sq) * rot_loss
            temp_loss.update(loss.item())
            temp_rot_loss.update(rot_loss.item())
            temp_x_loss.update(x_loss.item())
            pbar.update(1)
            pred_poses += [pred_pose.cpu().numpy()]

            pose_error_avg = np.mean(np.array([list(pose_error(val7_to_matrix(gt.squeeze().cpu().detach().numpy()), val7_to_matrix(est.squeeze().cpu().detach().numpy()))) for gt, est in zip(data[-1], pred_pose)]), 0)
            pbar.set_postfix(OrderedDict(mse_loss=loss.item(),
                                         rot_loss=rot_loss.item(),
                                         x_loss=x_loss.item(),
                                         sx=float(model.sx.detach().cpu()),
                                         sq=float(model.sq.detach().cpu()),
                                         pose_error_avg=list(pose_error_avg)))
            trans_error_meter.update(pose_error_avg[0])
            rot_error_meter.update(pose_error_avg[1])
    print(f"Translation Error: {trans_error_meter.avg:.4f}\tRot Error: {rot_error_meter.avg}")
    return temp_loss.avg, pred_poses, temp_x_loss.avg, temp_rot_loss.avg


if __name__ == '__main__':
    random.seed(args.seed)
    sequence = [f"{idx:02d}" for idx in range(11)]
    train_sequence = sequence[:7]
    val_sequence = sequence[7:]
    print(sequence)

    for i in [1,2,3,4]:
        writer = SummaryWriter(args.log, comment=f"{i}_fold/{args.weights_dir}")
        trainloss_hist = []
        valloss_hist = []
        valxloss_hist = []

        val_seqences = val_sequence # sequence[i * len(sequence) // N_FOLD :(i+1) * len(sequence) // N_FOLD]
        train_seqences = train_sequence # list(set(sequence) - set(val_seqences))

        print("training on seq:", train_seqences)
        print("validation on seq:", val_seqences)

        valsets = [Kitti(seq, root=args.data_dir) for seq in val_seqences]

        def save_best(model, epoch, savedir="checkpoint", strategy="valloss"):
            if min(eval(strategy + "_hist")) == eval(strategy + "_hist")[-1]:
                save_path = osp.join(savedir, f"ckpt_epoch={epoch}_{i}thfold_{strategy}={eval(strategy + '_hist')[-1]:.3f}.pth")
                torch.save(model.state_dict(), save_path)
                print("weights saved to", save_path)

        valloaders = [gDataLoaderX(valset, batch_size=1, shuffle=False, follow_batch=['x_s', "x_t"]) if LOAD_GRAPH else \
                          dDataLoaderX(valset, batch_size=1, shuffle=False, collate_fn=PadCollate(dim=0)) for valset in valsets]

        device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu_first else 'cpu')

        model = Net(graph_input=LOAD_GRAPH, act="LeakyReLU", transform=transform, dropout=args.dropout, dof=DOF).to(device)

        start_epoch = 0
        if osp.exists(args.model_pth):
            saved_state_dict = torch.load(args.model_pth)
            #saved_state_dict["sx"] = torch.nn.Parameter(torch.tensor(-10.0))
            model.load_state_dict(saved_state_dict)
            print("loaded weights from", args.model_pth)
            start_epoch = int(re.findall(r"epoch=(\d+?)_", args.model_pth)[0])
            print("start from epoch", start_epoch)
            eval(re.findall(r"fold_([a-zA-Z0-9]+?)=", args.model_pth)[0] + "_hist").append(float(re.findall(r"fold_[a-zA-Z0-9]+?=([-+]?[0-9]{1,}[.][0-9]*)", args.model_pth)[0]))
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        annealing = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, cooldown=5,
                                                               min_lr=1e-5, eps=1e-5, verbose=True)
        criterion_x = nn.MSELoss().to(device)
        criterion_rot = nn.MSELoss().to(device)

        for epoch in range(start_epoch, EPOCH):
            tem_train_loss = AverageMeter()
            tem_train_mse_loss = AverageMeter()
            tem_train_x_loss = AverageMeter()
            tem_train_rot_loss = AverageMeter()
            tem_val_rot_loss = AverageMeter()
            tem_val_x_loss = AverageMeter()
            tem_val_loss = AverageMeter()

            # train on all training sequences
            for seq in train_seqences:
                trainset = Kitti(seq, root=args.data_dir)
                trainloader = gDataLoaderX(trainset, batch_size=BATCH_SIZE, shuffle=False, follow_batch=['x_s', "x_t"]) if LOAD_GRAPH else \
                    dDataLoaderX(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=PadCollate(dim=0))

                epoch_loss, mse_epoch_loss, x_epoch_loss, rot_epoch_loss = train(model, epoch, trainloader, optimizer,
                                                                                 criterion_x, criterion_rot)
                tem_train_loss.update(epoch_loss)
                tem_train_mse_loss.update(mse_epoch_loss)
                tem_train_x_loss.update(x_epoch_loss)
                tem_train_rot_loss.update(rot_epoch_loss)
                print(f"Epoch: {epoch:03d}\tSeq: {seq}\tTTL: {epoch_loss:.3e}\tMSE: {mse_epoch_loss:.3e}\t"
                      f"X: {x_epoch_loss:.3e}\tRot: {rot_epoch_loss:.3e}\tsx: "
                      f"{float(model.sx.detach().cpu().numpy()):.3e}\tsq: {float(model.sq.detach().cpu().numpy()):.3e}")
                torch.cuda.empty_cache()
            trainloss_hist.append(tem_train_loss.avg)
            writer.add_scalar(f"train/loss", tem_train_loss.avg, global_step=epoch)
            writer.add_scalar(f"train/mse_loss", tem_train_mse_loss.avg, global_step=epoch)
            writer.add_scalar(f"train/x_loss", tem_train_x_loss.avg, global_step=epoch)
            writer.add_scalar(f"train/rot_loss", tem_train_rot_loss.avg, global_step=epoch)
            writer.add_scalar(f"sx", float(model.sx.detach().cpu()), global_step=epoch)
            writer.add_scalar(f"sq", float(model.sq.detach().cpu()), global_step=epoch)
            writer.flush()

            # validation on all validation sequences
            for valloader in valloaders:
                val_loss, pred_poses, x_epoch_loss, rot_epoch_loss = val(model, valloader, criterion_x, criterion_rot)
                tem_val_loss.update(val_loss)
                tem_val_x_loss.update(x_epoch_loss)
                tem_val_rot_loss.update(rot_epoch_loss)
                print(f'Epoch: {epoch:03d}\tSeq: {valloader.dataset.sequence}\tVal MSE: {val_loss:.3e}\t'
                      f'X: {x_epoch_loss:.3e}\tRot: {rot_epoch_loss:.3e}')
                save_pose_predictions(valloader.dataset.dataset.poses[0], pred_poses,
                                      osp.join(args.weights_dir, f"{epoch}_{valloader.dataset.sequence}.txt"))

            annealing.step(tem_val_loss.avg)
            valloss_hist.append(-tem_val_loss.avg)
            valxloss_hist.append(tem_val_x_loss.avg)
            writer.add_scalar(f"val/mse_loss", tem_val_loss.avg, global_step=epoch)
            writer.add_scalar(f"val/x_loss", tem_val_x_loss.avg, global_step=epoch)
            writer.add_scalar(f"val/rot_loss", tem_val_rot_loss.avg, global_step=epoch)
            writer.flush()
            save_best(model, epoch, savedir=args.weights_dir, strategy=args.save_strategy)
            adjust_lr(optimizer, epoch)
            print("-" * 50)
        writer.close()
