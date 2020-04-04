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
from torch_geometric.data import Dataset as gDataset, Data as gData
from utils import PairData


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

        delta_rotation = np.dot(source_pose[:3, :3], target_pose[:3, :3].T)
        rotation = Rotation.from_matrix(delta_rotation).as_quat()
        translation = source_pose[:, -1] - target_pose[:, -1]
        pose_vect = np.hstack([translation[:3], rotation])
        return target_cloud, source_cloud, pose_vect


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
