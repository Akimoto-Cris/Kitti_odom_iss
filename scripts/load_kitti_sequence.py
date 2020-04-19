#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: Xu Kaixin
@License: Apache Licence
@Time: 2020.03.19 : 上午 12:15
@File Name: dataloader.py
@Software: PyCharm
-----------------
"""

import pykitti
import rospy
from model.point_net import Net
from model.utils import ComposeAdapt, AverageMeter, pose_error, val7_to_matrix, save_pose_predictions
from collections import OrderedDict
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from kitti_localization.msg import CloudAndPose
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
import argparse
from torch_geometric.transforms import GridSampling, RandomTranslate, NormalizeScale, Compose
from geometry_msgs.msg import TransformStamped
import numpy as np
import torch
import os.path as osp
import time
import tf2_ros


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path")
parser.add_argument("-gs", "--grid_size", help="gridsampling size", type=float, default=3)
parser.add_argument("--data_dir", type=str, default='/home/kartmann/share_folder/dataset')
parser.add_argument("-s", "--sequence", type=str, default='00')
parser.add_argument("-r", "--rate", default=1, type=float)
parser.add_argument("--start", default=0, type=int)
args = parser.parse_args()
args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg]) for arg in args_dict]))
print('=' * 30)

sleep_rate = args.rate
queue_size = 10
LOAD_GRAPH = False


def inverse_pose(pose_mat):
    pose_mat[:3, :3] = Rotation.from_matrix(pose_mat[:3, :3]).inv().as_matrix()
    pose_mat[:3, -1] = -pose_mat[:3, -1]
    return pose_mat


def delta_poses(mat_1, mat_2):
    ret = np.eye(4)[:3, :]
    ret[:3, :3] = np.dot(mat_2[:3, :3].T, mat_1[:3, :3])
    ret[:3, -1] = mat_1[:3, -1] - mat_2[:3, -1]
    return ret


def kitti2rvizaxis(mat, delete_z=False):
    mat[:3, -1] = mat[[2, 0, 1], -1]
    mat[[1, 2], :3] = mat[[2, 1], :3]
    mat[:3, [1, 2]] = mat[:3, [2, 1]]
    mat[1, -1] = -mat[1, -1]
    if delete_z:
        mat[2, -1] = 0
    return mat


def trq2mat(tr, quat):
    ret = np.eye(4)[:3, :]
    ret[:3, :3] = Rotation.from_quat(quat).as_matrix()
    ret[:3, -1] = tr
    return ret


def mat2trq(mat):
    quat = Rotation.from_matrix(mat[:3, :3]).as_quat()
    tr = mat[:3, -1]
    return tr, quat


def add_poses(mat_1, mat_2):    # mat_1 is added by mat_2
    new = np.eye(4)[:3, :]
    new[:3, :3] = np.dot(mat_2[:3, :3], mat_1[:3, :3])
    new[:3, -1] = mat_1[:3, -1] + mat_2[:3, -1]
    return new


def transform_cloud(pointcloud: np.array, car2map: np.array):
    if pointcloud.shape[1] == 3:
        pointcloud = np.hstack([pointcloud, np.ones(pointcloud.shape[0], 1)])
    elif pointcloud.shape[1] == 4:
        pointcloud[:, -1] = 1
    if car2map.shape[0] == 3:
        temp = np.eye(4)
        temp[:3, :] = car2map
        car2map = temp
    return car2map.dot(pointcloud.T).T


class CloudPublishNode:
    def __init__(self, seq, node_name, cloud_topic_name, tf_topic_name, dataset, global_tf_name="map", child_tf_name="car"):
        rospy.init_node(node_name)
        self.cloud_pub = rospy.Publisher(cloud_topic_name, PointCloud2, queue_size=queue_size)
        self.transform_broadcaster = tf2_ros.TransformBroadcaster()
        self.est_tf_pub = rospy.Publisher(tf_topic_name, TransformStamped, queue_size=queue_size)    # for visualization
        self.gt_tf_pub = rospy.Publisher("gt_pose", TransformStamped, queue_size=queue_size)         # for visualization
        self.cap_pub = rospy.Publisher("CAP", CloudAndPose, queue_size=queue_size)
        self.rate = rospy.Rate(sleep_rate)
        self.header = Header(frame_id=global_tf_name)
        self.child_tf_name = child_tf_name      # base name before appending prefix
        self.dataset = dataset
        self.seq = seq

        transform_dict = OrderedDict()
        transform_dict[GridSampling([args.grid_size] * 3)] = ["train", "test"]
        transform_dict[NormalizeScale()] = ["train", "test"]
        transform = ComposeAdapt(transform_dict)
        self.model = Net(graph_input=LOAD_GRAPH, act="LeakyReLU", transform=transform, dof=7)
        if args.model_path is not None and osp.exists(args.model_path):
            self.model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
            print("loaded weights from", args.model_path)
        self.model.eval()

        self.absolute_gt_pose = np.eye(4)[:3, :]
        self.absolute_est_pose = np.eye(4)[:3, :]
        self.infer_time_meter = AverageMeter()
        self.tr_error_meter = AverageMeter()
        self.rot_error_meter = AverageMeter()

        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                       PointField('y', 4, PointField.FLOAT32, 1),
                       PointField('z', 8, PointField.FLOAT32, 1),
                       PointField('intensity', 12, PointField.FLOAT32, 1)]
        self.pose_list = []

    def estimate_pose(self, target_cloud, source_cloud):
        source_cloud = torch.from_numpy(source_cloud)
        target_cloud = torch.from_numpy(target_cloud)

        begin = time.time()
        pose = self.model((source_cloud.unsqueeze(0), target_cloud.unsqueeze(0),
                          torch.tensor(len(source_cloud)).unsqueeze(0), torch.tensor(len(target_cloud)).unsqueeze(0)))

        self.infer_time_meter.update(time.time() - begin)
        pose = pose.detach().numpy()
        self.pose_list.append(pose)
        return pose[0, :3], pose[0, 3:]

    def tq2tf_msg(self, translation, quaternion, header, typ="gt"):
        assert typ in ["gt", "est"]
        t = TransformStamped()
        t.header = header
        t.child_frame_id = "{}_{}".format(typ, self.child_tf_name)
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
        return t

    def mat2tf_msg(self, transform_mat, header, typ):
        translation = transform_mat[:3, -1]
        quat = Rotation.from_matrix(transform_mat[:3, :3]).as_quat()
        return self.tq2tf_msg(translation, quat, header, typ)

    def serve(self, idx):
        self.header.seq = idx
        self.header.stamp = rospy.Time.from_sec(self.dataset.timestamps[idx].total_seconds())

        current_cloud = self.dataset.get_velo(idx)
        if idx == 0:
            # guess 0 pose at first time frame
            tr, quat = np.zeros((3,)), np.array([0., 0., 0., 1.])
        else:
            # estimate coarse pose relative to the previous frame with model
            prev_cloud = self.dataset.get_velo(idx - 1)
            tr, quat = self.estimate_pose(prev_cloud, current_cloud)

        gt_pose = self.dataset.poses[idx]

        est_mat = trq2mat(tr, quat)
        delta_gt_pose = delta_poses(gt_pose.copy(), self.absolute_gt_pose.copy())
        self.absolute_gt_pose = gt_pose
        trans_error, rot_error = pose_error(delta_gt_pose, est_mat.copy())
        self.tr_error_meter.update(trans_error)
        self.rot_error_meter.update(rot_error)

        # correct the axis system of the estimated pose
        c_est_mat = kitti2rvizaxis(est_mat.copy())
        c_tr, c_quat = mat2trq(c_est_mat)
        cap_msg = CloudAndPose()
        cap_msg.seq = idx

        cap_msg.point_cloud2 = point_cloud2.create_cloud(self.header, self.fields, [point for point in current_cloud])
        cap_msg.init_guess = self.tq2tf_msg(*mat2trq(delta_gt_pose), self.header, "est")

        self.absolute_est_pose = add_poses(self.absolute_est_pose, c_est_mat)

        est_mat_temp = self.absolute_est_pose.copy()

        est_tf = self.mat2tf_msg(est_mat_temp, self.header, "est")
        gt_tf = self.mat2tf_msg(kitti2rvizaxis(gt_pose.copy()), self.header, "gt")
        self.est_tf_pub.publish(est_tf)
        self.gt_tf_pub.publish(gt_tf)
        self.transform_broadcaster.sendTransform(gt_tf)
        self.transform_broadcaster.sendTransform(est_tf)
        self.cloud_pub.publish(point_cloud2.create_cloud(Header(frame_id="gt_car"), self.fields, [point for point in current_cloud]))
        self.cap_pub.publish(cap_msg)

        print("[{}] inference spent: {:.2f} ms\t\t| Trans : {}\t\t| GT Trans: {}\t\t| Trans error: {:.4f}\t\t| "
              "Rot error: {:.4f}".format(idx, self.infer_time_meter.avg, list(c_tr), list(delta_gt_pose[:3, -1].reshape(3,)), trans_error, rot_error))
        self.rate.sleep()

    def __call__(self):
        for idx in range(args.start, len(self.dataset.poses)):
            if rospy.is_shutdown():
                break
            self.serve(idx)
        print("Avg Tr Error: {:.3e}\tAvg Rot Error: {:.3e}".format(self.tr_error_meter.avg, self.rot_error_meter.avg))
        save_pose_predictions(np.eye(4)[:3, :], self.pose_list, f"{self.seq}.txt")


if __name__ == '__main__':
    CloudPublishNode(args.sequence, "cloudPublisher", "point_cloud2", "est_pose", pykitti.odometry(args.data_dir, args.sequence), "map", "car")()
    rospy.spin()
