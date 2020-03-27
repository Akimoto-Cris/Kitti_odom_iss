#!/usr/bin/env python

import pykitti
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


basedir = '/home/kartmann/share_folder/dataset'
sequence = '00'
frames = None   # range(0, 20, 5)
fields = [PointField('x',           0,  PointField.FLOAT32, 1),
          PointField('y',           4,  PointField.FLOAT32, 1),
          PointField('z',           8,  PointField.FLOAT32, 1),
          PointField('intensity',   12, PointField.FLOAT32, 1)]
sleep_rate = 1.
queue_size = 2

if __name__ == '__main__':
    rospy.init_node("cloudPublisher")
    pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=queue_size)
    rate = rospy.Rate(sleep_rate)
    header = Header()
    header.frame_id = "map"

    dataset = pykitti.odometry(basedir, sequence, frames=frames)

    for idx, gt_pose in enumerate(dataset.poses):
        if rospy.is_shutdown():
            break
        cur_cloud = dataset.get_velo(idx)
        header.seq = idx
        header.stamp = dataset.timestamps[idx]
        pc2 = point_cloud2.create_cloud(header, fields, [point for point in cur_cloud])
        pub.publish(pc2)
        rospy.logdebug(pc2)
        rate.sleep()

    rospy.spin()
