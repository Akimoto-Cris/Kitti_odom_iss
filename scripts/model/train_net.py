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

import os
import os.path as osp
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as dDataLoader
from prefetch_generator import BackgroundGenerator
from dataloader import KittiGraph, DataLoaderX as gDataLoaderX, PadCollate, KittiStandard
from point_net import Net
import tqdm
import re
from torch_geometric.data import DataLoader as gDataLoader, Data as gData
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import GridSampling, RandomTranslate, NormalizeScale, Compose
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default="/content/drive/My Drive/dataset")
parser.add_argument("--n_fold", default=5, type=int)
parser.add_argument("-b", "--batch_size", default=1, type=int)
parser.add_argument("-e", "--epoch", default=30, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--grid_size", help="gridsampling size", type=float, default=1.)
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-g", "--gpu_first", default=True, type=bool)
parser.add_argument("--model_pth", type=str)
parser.add_argument("--num_workers", default=1, type=int)
parser.add_argument("--log", default="log", type=str)
args = parser.parse_args()

args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                 for arg in args_dict]))
print('=' * 30)

N_FOLD = args.n_fold
BATCH_SIZE = args.batch_size
EPOCH = args.epoch
LR = args.lr
GRID_SAMPLE_SIZE = [args.grid_size] * 3
LOAD_GRAPH = False


class dDataLoaderX(dDataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


Kitti = KittiGraph if LOAD_GRAPH else KittiStandard
transform = Compose([NormalizeScale(),
                     RandomTranslate(0.001),
                     GridSampling(GRID_SAMPLE_SIZE)])
# transform = GridSampling(GRID_SAMPLE_SIZE)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num = 0.
        self.cnt = 0
        self.sum = 0.
        self.avg = 0.
        self.max = -10000000

    def update(self, num, c=1):
        self.num = num
        self.sum += num
        self.cnt += c
        self.avg = self.sum / self.cnt
        self.min = min(self.max, num)


def train(model, epoch, train_loader, optimizer, criterion):
    model.train()
    temp_loss = AverageMeter()

    with tqdm.tqdm(len(train_loader)) as pbar:
        for data in train_loader:
            data = data.to(device) if not isinstance(data, tuple) else tuple([d.to(device) for d in data])
            optimizer.zero_grad()
            # loss = F.nll_loss(model(data), data.y)
            gt_poses = data.y.reshape(train_loader.batch_size, -1).float() if LOAD_GRAPH else data[-1].float()
            loss = criterion(model(data), gt_poses)
            loss.backward()
            optimizer.step()
            temp_loss.update(loss.item())
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
    return temp_loss.avg


@torch.no_grad()
def val(model, loader, criterion):
    model.eval()
    temp_loss = AverageMeter()
    with tqdm.tqdm(total=len(loader)) as pbar:
        for data in loader:
            data = data.to(device) if not isinstance(data, tuple) else tuple([d.to(device) for d in data])
            # loss = F.nll_loss(model(data), data.y)
            gt_poses = data.y.reshape(loader.batch_size, -1).float() if LOAD_GRAPH else data[-1].float()
            loss = criterion(model(data), gt_poses)
            temp_loss.update(loss.item())
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
    return temp_loss.avg


if __name__ == '__main__':
    random.seed(args.seed)
    sequence = [f"{idx:02d}" for idx in range(11)]
    random.shuffle(sequence)
    print(sequence)

    writer = SummaryWriter(args.log)

    for i in range(N_FOLD):
        trainloss_meter = AverageMeter()
        valloss_meter = AverageMeter()

        train_seqences = sequence[(i+1) * len(sequence) // N_FOLD:]
        val_seqences = sequence[:(i+1) * len(sequence) // N_FOLD]
        valsets = [Kitti(seq, root=args.data_dir) for seq in val_seqences]

        def save_best(model, epoch, savedir="checkpoint", strategy="valloss"):
            if eval(strategy + "_meter").min >= eval(strategy + "_meter").num:
                save_path = osp.join(savedir, f"ckpt_epoch={epoch}_{i}thfold_{strategy}={eval(strategy + '_meter').num:.3f}.pth")
                torch.save(model.state_dict(), save_path)
                print("weights saved to", save_path)

        valloaders = [gDataLoaderX(valset, batch_size=BATCH_SIZE, shuffle=False, follow_batch=['x_s', "x_t"]) if LOAD_GRAPH else \
                      dDataLoaderX(valset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=PadCollate(dim=0)) for valset in valsets]

        device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu_first else 'cpu')

        model = Net(graph_input=LOAD_GRAPH, transform=transform).to(device)
        start_epoch = 0
        if args.model_pth is not None:
            model.load_state_dict(torch.load(args.model_pth))
            print("loaded weights from", args.model_pth)
            start_epoch = int(re.findall(r"epoch=(\d+?)_", args.model_pth)[0])
            print("start from epoch", start_epoch)
            eval(re.findall(r"fold_([a-zA-Z0-9]+?)=", args.model_pth)[0] + "_meter").max = float(re.findall(r"fold_[a-zA-Z0-9]+?=([0-9]{1,}[.][0-9]*)", args.model_pth)[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss().to(device)

        for epoch in range(start_epoch, EPOCH):
            tem_train_loss = AverageMeter()
            tem_val_loss = AverageMeter()
            # train on all training sequences
            for seq in train_seqences:
                trainset = Kitti(seq, root=args.data_dir)
                trainloader = gDataLoaderX(trainset, batch_size=BATCH_SIZE, shuffle=False, follow_batch=['x_s', "x_t"]) if LOAD_GRAPH else \
                              dDataLoaderX(trainset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=PadCollate(dim=0))

                epoch_loss = train(model, epoch, trainloader, optimizer, criterion)
                tem_train_loss.update(epoch_loss)
                print(f"Epoch: {epoch:03d}\t\tSeq: {seq}\t\tTrain: {epoch_loss:.4f}")
            trainloss_meter.update(tem_train_loss.avg)
            writer.add_scalar(f"{i}_fold/train_loss", tem_train_loss.avg, global_step=epoch)
            writer.flush()

            # validation on all validation sequences
            for valloader in valloaders:
                val_loss = val(model, valloader, criterion)
                tem_val_loss.update(val_loss)
                print('Epoch: {:03d}\t\tSeq: {}\t\tVal: {:.4f}'.format(epoch, valloader.dataset.sequence, val_loss))
            valloss_meter.update(tem_val_loss.avg)
            writer.add_scalar(f"{i}_fold/val_loss", tem_val_loss.avg, global_step=epoch)
            writer.flush()
            save_best(model, epoch, strategy="valloss")