#!/usr/bin/env python
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
from model.utils import ComposeAdapt, AverageMeter, pose_error, val7_to_matrix
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


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path")
parser.add_argument("-gs", "--grid_size", help="gridsampling size", type=float, default=3)
parser.add_argument("--data_dir", type=str, default='/home/kartmann/share_folder/dataset')
parser.add_argument("-s", "--sequence", type=str, default='00')
parser.add_argument("-r", "--rate", default=1, type=float)
parser.add_argument("--dropout", type=float, default=0.5)
args = parser.parse_args()
args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg]) for arg in args_dict]))
print('=' * 30)

sleep_rate = args.rate
queue_size = 10
LOAD_GRAPH = False


class CloudPublishNode:
    def __init__(self, node_name, cloud_topic_name, tf_topic_name, dataset, global_tf_name="map", child_tf_name="car"):
        rospy.init_node(node_name)
        self.cloud_pub = rospy.Publisher(cloud_topic_name, PointCloud2, queue_size=queue_size)
        #self.transform_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_pub = rospy.Publisher(tf_topic_name, TransformStamped, queue_size=queue_size)
        self.gt_tf_pub = rospy.Publisher("gt_" + tf_topic_name, TransformStamped, queue_size=queue_size)
        self.cap_pub = rospy.Publisher("CAP", CloudAndPose, queue_size=queue_size)
        self.rate = rospy.Rate(sleep_rate)
        self.header = Header()
        self.header.frame_id = global_tf_name
        self.child_tf_name = child_tf_name
        self.dataset = dataset

        transform_dict = OrderedDict()
        transform_dict[GridSampling([args.grid_size] * 3)] = ["train", "test"]
        transform_dict[NormalizeScale()] = ["train", "test"]
        transform = ComposeAdapt(transform_dict)
        self.model = Net(graph_input=LOAD_GRAPH, act="LeakyReLU", transform=transform, dropout=args.dropout, dof=7)
        if args.model_path is not None and osp.exists(args.model_path):
            self.model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
            print("loaded weights from", args.model_path)
        self.model.eval()

        self.absolute_gt_pose = np.eye(4)
        self.absolute_est_pose = np.eye(4)
        self.infer_time_meter = AverageMeter()

        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                       PointField('y', 4, PointField.FLOAT32, 1),
                       PointField('z', 8, PointField.FLOAT32, 1),
                       PointField('intensity', 12, PointField.FLOAT32, 1)]

    def estimate_pose(self, target_cloud, source_cloud):
        source_cloud = torch.from_numpy(source_cloud)
        target_cloud = torch.from_numpy(target_cloud)

        begin = time.time()
        pose = self.model((source_cloud.unsqueeze(0), target_cloud.unsqueeze(0),
                          torch.tensor(len(source_cloud)).unsqueeze(0), torch.tensor(len(target_cloud)).unsqueeze(0)))

        self.infer_time_meter.update((time.time() - begin) / 1000)
        pose = pose.detach().numpy()
        return pose[0, :3], pose[0, 3:]

    def update_absolute_pose(pose, which):
        assert which in ["gt", "est"]
        eval(f"self.absolute_{which}_pose")[:3, :3] = np.dot(np.linalg.inv(eval(f"self.absolute_{which}_pose")[:3, :3]), pose[:3, :3])
        eval(f"self.absolute_{which}_pose")[:3, -1] += pose[:3, -1]

    def tq2tf_msg(self, translation, quaternion, header):
        t = TransformStamped()
        t.header = header
        t.child_frame_id = self.child_tf_name
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
        #self.tf_pub.publish(t)
        return t

    def mat2tf_msg(self, transform_mat, header):
        translation = transform_mat[:3, -1]
        quat = Rotation.from_matrix(transform_mat[:3, :3]).as_quat()
        return self.tq2tf_msg(translation, quat, header)

    def serve(self, idx):
        current_cloud = self.dataset.get_velo(idx)
        if idx == 0:
            # guess 0 pose at first time frame
            tr, quat = np.zeros((3,)), np.array([0., 0., 0., 1.])
        else:
            # estimate coarse pose relative to the previous frame with model
            prev_cloud = self.dataset.get_velo(idx - 1)
            tr, quat = self.estimate_pose(prev_cloud, current_cloud)
        gt_pose = self.dataset.poses[idx]
        est_mat = val7_to_matrix(np.concatenate([tr, quat]))
        trans_error, rot_error = pose_error(gt_pose, self.absolute_est_pose)

        self.header.seq = idx
        self.header.stamp = rospy.Time.from_sec(self.dataset.timestamps[idx].total_seconds())
        cap_msg = CloudAndPose()
        cap_msg.seq = idx
        cap_msg.point_cloud2 = point_cloud2.create_cloud(self.header, self.fields, [point for point in current_cloud])
        cap_msg.init_guess = self.tq2tf_msg(tr, quat, self.header)

        gt_tf = self.mat2tf_msg(gt_pose, self.header)

        self.cloud_pub.publish(cap_msg.point_cloud2)
        self.tf_pub.publish(cap_msg.init_guess)
        self.gt_tf_pub.publish(gt_tf)
        self.cap_pub.publish(cap_msg)
        self.rate.sleep()
        return trans_error, rot_error

    def __call__(self):
        for idx in range(len(self.dataset.poses)):
            if rospy.is_shutdown():
                break
            trans_error, rot_error = self.serve(idx)
            print(f"[{idx}] inference spent: {self.infer_time_meter.avg:.2f} s\t|"
                  f"\tTrans error: {trans_error:.4f}\t|\tRot error: {rot_error:.4f}")
        rospy.spin()


if __name__ == '__main__':
    CloudPublishNode("cloudPublisher", "point_cloud2", "init_guess",
                     pykitti.odometry(args.data_dir, args.sequence), "map", "car")()
