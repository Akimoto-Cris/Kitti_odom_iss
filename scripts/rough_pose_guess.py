#!/usr/bin/python3

from model.point_net import Net
import torch
import os.path as osp
import rospy
import argparse
from model.dataloader import PairData

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path", default=osp.join(osp.abspath(__file__), "model/checkpoint/"))


if __name__ == '__main__':
    rospy.init_node("rough_pose_guess")
    
