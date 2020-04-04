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
from model.utils import ComposeAdapt
from collections import OrderedDict
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import argparse
from torch_geometric.transforms import GridSampling, RandomTranslate, NormalizeScale, Compose
from geometry_msgs.msg import TransformStamped
import numpy as np
import torch
import os.path as osp


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path")
parser.add_argument("-gs", "--grid_size", help="gridsampling size", type=float, default=3)
parser.add_argument("--data_dir", type=str, default='/home/kartmann/share_folder/dataset')
parser.add_argument("-s", "--sequence", type=str, default='00')
parser.add_argument("--dropout", type=float, default=0.5)
args = parser.parse_args()
args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg]) for arg in args_dict]))
print('=' * 30)

sleep_rate = 1.
queue_size = 2
LOAD_GRAPH = False


class CloudPublishNode:
    def __init__(self, node_name, cloud_topic_name, tf_topic_name, dataset, global_tf_name="map", child_tf_name="car"):
        rospy.init_node(node_name)
        self.cloud_pub = rospy.Publisher(cloud_topic_name, PointCloud2, queue_size=queue_size)
        #self.transform_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_pub = rospy.Publisher(tf_topic_name, TransformStamped, queue_size=queue_size)
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

        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                       PointField('y', 4, PointField.FLOAT32, 1),
                       PointField('z', 8, PointField.FLOAT32, 1),
                       PointField('intensity', 12, PointField.FLOAT32, 1)]

    def estimate_pose(self, target_cloud, source_cloud):
        source_cloud = torch.from_numpy(source_cloud)
        target_cloud = torch.from_numpy(target_cloud)

        pose = self.model((source_cloud.unsqueeze(0), target_cloud.unsqueeze(0),
                          torch.tensor(len(source_cloud)).unsqueeze(0), torch.tensor(len(target_cloud)).unsqueeze(0)))
        pose = pose.detach().numpy()
        return pose[0, :3], pose[0, 3:]

    def publish_tfs(self, translation, quaternion, header):
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
        #self.transform_broadcaster.sendTransform(t)
        self.tf_pub.publish(t)

    def serve(self, idx):
        current_cloud = self.dataset.get_velo(idx)
        if idx == 0:
            # guess 0 pose at first time frame
            tr, quat = np.zeros((3,)), np.array([0., 0., 0., 1.])
        else:
            # estimate coarse pose relative to the previous frame with model
            prev_cloud = self.dataset.get_velo(idx - 1)
            tr, quat = self.estimate_pose(prev_cloud, current_cloud)

        self.header.seq = idx
        self.header.stamp = rospy.Time.from_sec(self.dataset.timestamps[idx].total_seconds())
        pc2 = point_cloud2.create_cloud(self.header, self.fields, [point for point in current_cloud])
        self.publish_tfs(tr, quat, self.header)
        self.cloud_pub.publish(pc2)
        rospy.logdebug(pc2)
        self.rate.sleep()

    def __call__(self):
        for idx in range(len(self.dataset.poses)):
            if rospy.is_shutdown():
                break
            self.serve(idx)
        rospy.spin()


if __name__ == '__main__':
    CloudPublishNode("cloudPublisher", "point_cloud2", "init_guess", pykitti.odometry(args.data_dir, args.sequence), "map", "car")()
