#!/usr/bin/env python

import rospy
from kitti_localization.msg import CloudAndPose


def callback(data):
    rospy.loginfo(data.init_guess)


if __name__ == '__main__':
    rospy.init_node('cap_msg_tester', anonymous=True)
    rospy.Subscriber("/CAP", CloudAndPose, callback)

    rospy.spin()
