#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3, TransformStamped
from std_msgs.msg import Header, ColorRGBA
import numpy as np



def show_text_in_rviz(marker_publisher, i, pose):
    marker = Marker(type=Marker.LINE_STRIP,
                    id=i,
                    lifetime=rospy.Duration(15),
                    pose=pose,
                    scale=Vector3(0.06, 0.06, 0.06),
                    header=Header(frame_id='map'),
                    color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
                    text=text)
    marker_publisher.publish(marker)


class NavPath:
    def __init__(self):
        self.transforms = []

    def callback(self, msg):
        rospy.loginfo(msg)
        pose = Pose(Point(msg.transform.translation.x,
                          msg.transform.translation.y,
                          msg.transform.translation.z),
                    Quaternion(msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w))
        translation = np.array([msg.transform.translation.x,
                                msg.transform.translation.y,
                                msg.transform.translation.z])
        quat = np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
        transform = np.eye(4)
        transform[:3, :3] =
        if not self.transforms:
            self.transforms.append(pose)
        else:
            np.dot(np.linalg.inv(self._path))


if __name__ == '__main__':
    rospy.init_node('my_node')
    gt_pose_subscriber = rospy.Subscriber("/gt_pose", TransformStamped, )
    markerarray_publisher = rospy.Publisher('gt_markers', MarkerArray, queue_size=10)

    rospy.sleep(0.5)

    rospy.spin()
