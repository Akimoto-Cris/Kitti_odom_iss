#!/usr/bin/python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Vector3, TransformStamped
from std_msgs.msg import Header, ColorRGBA
import numpy as np
from scipy.spatial.transform import Rotation


class NavPath:
    def __init__(self, marker_publisher, frame_id, color4f=(0.0, 1.0, 0.0, 0.8), buffer_size=10, typ="gt"):
        self.translations = None
        self.header = Header(frame_id=frame_id)
        self.marker = Marker(type=Marker.LINE_STRIP, color=ColorRGBA(1, 1, 1, 1),
                             lifetime=rospy.Duration(1), header=self.header, frame_locked=True)
        self.marker.scale.x = 0.05

        self.line_color = ColorRGBA(*color4f)
        self.buffer_size = buffer_size
        self.cnt = 0
        assert 1 <= buffer_size
        self.type = typ

        self.marker_publisher = marker_publisher

    def update_marker(self):
        self.marker.points.append(Point(*list(self.translations[-1, :])))
        self.marker.colors.append(self.line_color)
        self.marker.id = self.cnt
        self.cnt += 1

    def publish_marker(self):
        self.marker_publisher.publish(self.marker)
        # rospy.loginfo(self.marker_array)

    def callback(self, msg):
        rospy.loginfo(msg)
        translation = np.array([msg.transform.translation.x,
                                msg.transform.translation.y,
                                msg.transform.translation.z]).reshape(1, -1)
        self.translations = -translation if self.translations is None else np.concatenate([self.translations, -translation])
        if self.translations.shape[0] > self.buffer_size:
            self.translations = self.translations[1:, :]

        self.update_marker()
        self.publish_marker()


if __name__ == '__main__':
    rospy.init_node('viz_markers_node')
    gt_mkr_pub = rospy.Publisher('gt_markers', Marker, queue_size=10)
    gt_traj = NavPath(gt_mkr_pub, "/gt_car", (0.0, 1.0, 0.0, 0.8))
    gt_pose_subscriber = rospy.Subscriber("/gt_pose", TransformStamped, gt_traj.callback)

    est_mkr_pub = rospy.Publisher('est_markers', Marker, queue_size=10)
    est_traj = NavPath(est_mkr_pub, "/est_car", (1.0, 0.0, 0.0, 0.8))
    est_pose_subscriber = rospy.Subscriber("/est_pose", TransformStamped, est_traj.callback)

    ndt_mkr_pub = rospy.Publisher('ndt_markers', Marker, queue_size=10)
    ndt_traj = NavPath(ndt_mkr_pub, "/ndt_car", (0.0, 0.0, 1.0, 0.8))
    ndt_pose_subscriber = rospy.Subscriber("/ndt_pose", TransformStamped, ndt_traj.callback)
    rospy.spin()
