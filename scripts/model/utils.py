#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: Xu Kaixin
@License: Apache Licence
@Time: 2020.03.28 : 上午 10:25
@File Name: utils.py
@Software: PyCharm
-----------------
"""
from itertools import chain
from scipy.spatial.transform import Rotation
from prefetch_generator import BackgroundGenerator
import numpy as np
import torch
from torch.utils.data import DataLoader as dDataLoader
from torch_geometric.data import DataLoader as gDataLoader, Data as gData


def save_pose_predictions(init_pose, pred_poses: list, save_path):
    absolute_pose = init_pose
    with open(save_path, "w") as f:
        for i, batch_pose in enumerate(pred_poses):
            for pose in batch_pose:
                rot = Rotation.from_quat(pose[3:]).as_dcm()
                absolute_rot = np.dot(np.linalg.inv(absolute_pose[:3, :3]), rot)
                absolute_trans = pose[:3] + absolute_pose[:3, -1]
                absolute_pose = np.hstack([absolute_rot, absolute_trans.reshape(-1, 1)])
                pose_dcm_str = map(lambda x: f"{x:4e}", list(absolute_pose.reshape(-1)))
                f.write(" ".join(tuple(pose_dcm_str)) + "\n")
        #print("poses writed to", save_path)


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


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    if not isinstance(vec, torch.Tensor):
        vec = torch.from_numpy(vec)
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, tensor, length1, length2, label) or (tensor, tensor, label)
                    length of both first 2 tensors will be inferred if not provided

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        assert len(batch[0]) in [3, 5]

        if len(batch[0]) == 3:
            # find longest sequence
            max_len_s = max(map(lambda x: x[0].shape[self.dim], batch))
            max_len_t = max(map(lambda x: x[1].shape[self.dim], batch))
            # pad according to max_len
            batch = list(map(lambda x:
                             (pad_tensor(x[0], pad=max_len_s, dim=self.dim),
                              pad_tensor(x[1], pad=max_len_t, dim=self.dim),
                              x[0].shape[self.dim], x[1].shape[self.dim], x[2]), batch))
        elif len(batch[0]) == 5:
            # find longest sequence
            max_len_s = max(map(lambda x: x[2], batch))
            max_len_t = max(map(lambda x: x[3], batch))
            # pad according to max_len
            batch = list(map(lambda x_s, x_t, ls, lt, y:
                             (pad_tensor(x_s, pad=max_len_s, dim=self.dim),
                              pad_tensor(x_t, pad=max_len_t, dim=self.dim),
                              ls, lt, y), batch))
        # stack all
        xs_s = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        xs_t = torch.stack(list(map(lambda x: x[1], batch)), dim=0)
        length1s = torch.LongTensor(list(map(lambda x: x[2], batch)))
        length2s = torch.LongTensor(list(map(lambda x: x[3], batch)))
        ys = torch.FloatTensor(list(map(lambda x: x[4], batch)))

        return xs_s, xs_t, length1s, length2s, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


class gDataLoaderX(gDataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class dDataLoaderX(dDataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class PairData(gData):
    """ Referred from https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html#pairs-of-graphs"""
    def __init__(self, s_data, t_data, y):
        super(PairData, self).__init__()
        self.edge_index_s = s_data.edge_index
        self.x_s = s_data.x
        self.pos_s = s_data.pos
        self.edge_index_t = t_data.edge_index
        self.x_t = t_data.x
        self.pos_t = t_data.pos
        self.y = y

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super(PairData, self).__inc__(key, value)


def l2reg(model):
    l2_reg = 0.
    for param in model.parameters():
        l2_reg += torch.norm(param)
    return l2_reg


class ComposeAdapt:
    def __init__(self, transforms: dict):
        self.transforms = transforms
        assert set(list(chain(*transforms.values()))) - {"train", "test"} == set()

    def __call__(self, data, mode):
        assert mode in ["train", "test"]
        for t, modes in self.transforms.items():
            if mode in modes:
                data = t(data)
        return data

    def __repr__(self):
        args = ['    {}: {},'.format(k, t) for k, t in self.transforms.items()]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))


def pose_error(gt_mat, est_mat):
    if gt_mat.shape[0] == 3:
        t = gt_mat
        gt_mat = np.eye(4)
        gt_mat[:3, :] = t
    if est_mat.shape[0] == 3:
        t = est_mat
        est_mat = np.eye(4)
        est_mat[:3, :] = t

    pe = np.dot(np.linalg.inv(est_mat), gt_mat)
    return translation_error(pe), rotation_error(pe)


def rotation_error(pe):
    a = pe[0, 0]
    b = pe[1, 1]
    c = pe[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1), -1))


def translation_error(pe):
    dx = pe[0, 3]
    dy = pe[1, 3]
    dz = pe[2, 3]
    return np.sqrt(dx**2 + dy**2 + dz**2)


def val7_to_matrix(pose_vect):
    translation = pose_vect[:3]
    quat = pose_vect[3:]
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat(quat).as_matrix()
    mat[:3, -1] = translation
    return mat
