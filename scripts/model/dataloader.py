#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: Xu Kaixin
@License: Apache Licence
@Time: 2020.03.24 : 上午 12:10
@File Name: dataloader.py
@Software: PyCharm
-----------------
"""
import pykitti
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset as dDataset
from prefetch_generator import BackgroundGenerator
from torch_geometric.data import Dataset as gDataset, Data as gData, DataLoader as gDataLoader


class KittiStandard(dDataset):
    def __init__(self, sequence, root='E:\\share_folder\\dataset'):
        super().__init__()
        frames = None  # range(0, 20, 5)
        self.dataset = pykitti.odometry(root, sequence, frames=frames)
        self.sequence = sequence

    def __len__(self):
        return len(self.dataset.poses) - 1

    def __getitem__(self, idx):
        """
        Actually return the (`idx`+1) th point cloud
        and the pose between `idx` th and (`idx`+1) th point cloud
        """
        try:
            target_cloud = self.dataset.get_velo(idx)
        except IndexError as e:
            print(e, "Happends on", idx)
        target_pose = self.dataset.poses[idx]

        source_cloud = self.dataset.get_velo(idx + 1)
        source_pose = self.dataset.poses[idx + 1]

        pose = np.dot(np.linalg.inv(target_pose), source_pose)
        rotation = list(Rotation.from_dcm(pose[:3, :3]).as_euler("xyz", degrees=False))
        translation = list(pose[:, -1])
        pose_vect = translation[:-1] + rotation

        return target_cloud, source_cloud, np.array(pose_vect)


class DataLoaderX(gDataLoader):
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


class KittiGraph(gDataset):
    def __init__(self, sequence, root='E:\\share_folder\\dataset', transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.dataset = pykitti.odometry(root, sequence, frames=None)

    def len(self):
        return len(self.dataset.poses)

    def __getitem__(self, idx):
        target_cloud = self.dataset.get_velo(idx)
        target_pose = self.dataset.poses[idx]

        source_cloud = self.dataset.get_velo(idx + 1)
        source_pose = self.dataset.poses[idx + 1]

        pose = np.dot(np.linalg.inv(target_pose), source_pose)
        rotation = list(Rotation.from_dcm(pose[:3, :3]).as_euler("xyz", degrees=False))
        translation = list(pose[:, -1])
        pose_vect = translation[:-1] + rotation

        s_data = gData(x=torch.from_numpy(source_cloud[:, 2:3]), pos=torch.from_numpy(source_cloud[:, :3]))
        t_data = gData(x=torch.from_numpy(target_cloud[:, 2:3]), pos=torch.from_numpy(target_cloud[:, :3]))

        s_data = s_data if self.transform is None else self.transform(s_data)
        t_data = t_data if self.transform is None else self.transform(t_data)

        return PairData(s_data, t_data, torch.from_numpy(np.array(pose_vect).T))


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
        ys = torch.LongTensor(list(map(lambda x: x[4], batch)))
        return xs_s, xs_t, length1s, length2s, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
