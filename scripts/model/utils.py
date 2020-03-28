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
from scipy.spatial.transform import Rotation
import numpy as np


def save_pose_predictions(pred_poses: list, save_path):

    with open(save_path, "w") as f:
        for batch_pose in pred_poses:
            for pose in batch_pose:
                rot = Rotation.from_euler("xyz", pose[3:]).as_dcm()
                pose_dcm = np.concatenate([rot, pose[:3].reshape(-1, 1)], axis=1).reshape(-1)
                pose_dcm_str = map(lambda x: str(x), list(pose_dcm))
                f.write(" ".join(tuple(pose_dcm_str)) + "\n")

        print("poses writed to", save_path)


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
