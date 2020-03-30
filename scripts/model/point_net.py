#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: Xu Kaixin
@License: Apache Licence
@Time: 2020.03.24 : 上午 12:08
@File Name: point_net.py
@Software: PyCharm
-----------------
"""

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN, Dropout, LeakyReLU
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.data import Batch


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class PointNetXX(torch.nn.Module):
    def __init__(self, act="Sigmoid", dropout=0.5):
        super(PointNetXX, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([1 + 3, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 512]))

        self.lin1 = Lin(512, 256)
        self.lin2 = Lin(256, 64)

        self.act = eval(act)()
        self.dropout = Dropout(dropout)

    def forward(self, x, pos, batch):
        sa0_out = x, pos, batch
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out

        x = self.lin1(x)
        x = self.dropout(self.act(x))
        x = self.act(self.lin2(x))
        return x


class Net(torch.nn.Module):
    def __init__(self, dof=6, act="Sigmoid", dropout=0.5, graph_input=False, transform=None):
        super(Net, self).__init__()
        self.graph_input = graph_input
        self.transform = transform
        self.pointnet = PointNetXX(act=act, dropout=dropout)
        self.fcs = Seq(
            eval(act)(), Lin(64 * 2, 64), Dropout(dropout),
            eval(act)(), Lin(64, dof)
        )
        self.sx = torch.nn.Parameter(torch.tensor(-2.5), requires_grad=True)
        self.sq = torch.nn.Parameter(torch.tensor(-2.5), requires_grad=True)

        #for m in self.modules():
        #    if isinstance(m, Lin):
        #        torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        if self.graph_input:
            s_encode = self.pointnet(input.x_s, input.pos_s, input.x_s_batch.long())
            t_encode = self.pointnet(input.x_t, input.pos_t, input.x_t_batch.long())
        else:
            graph_s, graph_t = x_pos_batch_to_pair_biggraph_pair(*input[:4])
            graph_s = self.transform(graph_s) if self.transform else graph_s
            graph_t = self.transform(graph_t) if self.transform else graph_t

            s_encode = self.pointnet(graph_s.x, graph_s.pos, graph_s.batch)
            t_encode = self.pointnet(graph_t.x, graph_t.pos, graph_t.batch)

        fusion = torch.cat([s_encode, t_encode], dim=1)
        return self.fcs(fusion)


def x_pos_batch_to_pair_biggraph_pair(cloud_s_all, cloud_t_all, lss, lst):
    x_pos_s_all, x_pos_t_all, batch_s, batch_t = [], [], [], []
    for i, (ls, lt, cloud_s, cloud_t) in enumerate(zip(lss, lst, cloud_s_all, cloud_t_all)):
        x_pos_s_all += [cloud_s[:ls, :]]
        x_pos_t_all += [cloud_t[:lt, :]]
        batch_s += [torch.ones(ls, ).long().unsqueeze(1).to(lss.device) * i]
        batch_t += [torch.ones(lt, ).long().unsqueeze(1).to(lst.device) * i]

    x_pos_s_all = torch.cat(x_pos_s_all, dim=0)
    x_pos_t_all = torch.cat(x_pos_t_all, dim=0)
    batch_s = torch.cat(batch_s, dim=0).squeeze()
    batch_t = torch.cat(batch_t, dim=0).squeeze()

    graph_s = Batch(x=x_pos_s_all[:, 2:3], pos=x_pos_s_all[:, :3], batch=batch_s)
    graph_t = Batch(x=x_pos_t_all[:, 2:3], pos=x_pos_t_all[:, :3], batch=batch_t)
    return graph_s, graph_t
