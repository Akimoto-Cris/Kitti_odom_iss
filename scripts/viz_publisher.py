#!/usr/bin/python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3, TransformStamped
from std_msgs.msg import Header, ColorRGBA
import numpy as np
from scipy.spatial.transform import Rotation


class NavPath:
    def __init__(self, markerarray_publisher, frame_id, color4f=(0.0, 1.0, 0.0, 0.8), length=10):
        self.translations = None
        self.marker_array = MarkerArray()
        self.header = Header(frame_id=frame_id)
        self.line_color = ColorRGBA(*color4f)
        self.length = length
        self.cnt = 0
        assert 1 <= length

        self.markerarray_publisher = markerarray_publisher

    def update_markerarray(self):
        self.marker_array = MarkerArray()
        rospy.loginfo(self.translations.shape)
        for i, trans in enumerate(self.translations[:-1]):
            start_point = Point(*list(self.translations[i, :]))
            end_point = Point(*list(self.translations[i+1, :]))
            marker = Marker(type=Marker.LINE_STRIP, scale=Vector3(0.05, 0.05, 0.05), id=self.cnt,
                            lifetime=rospy.Duration(1), header=self.header)
            marker.points.append(start_point)
            marker.points.append(end_point)
            marker.colors.append(self.line_color)
            marker.colors.append(self.line_color)
            self.marker_array.markers.append(marker)
            self.cnt += 1

    def publish_markerarray(self):
        self.markerarray_publisher.publish(self.marker_array)

    def callback(self, msg):
        rospy.loginfo(msg)
        translation = np.array([msg.transform.translation.x,
                                msg.transform.translation.y,
                                msg.transform.translation.z]).reshape(1, -1)
        if self.translations is None:
            self.translations = translation
        else:
            # move the whole trajectory to keep the latest incoming marker at initial position/world center
            delta = translation - self.translations[-1, :]
            self.translations -= delta
            self.translations = np.concatenate([self.translations, delta])

            if self.translations.shape[0] > self.length:
                self.translations = self.translations[1:, :]
            if len(self.quats) > self.length:
                self.quats = self.quats[1:]

        self.update_markerarray()
        self.publish_markerarray()


if __name__ == '__main__':
    rospy.init_node('viz_markers_node')
    gt_mkrarray_pub = rospy.Publisher('gt_markers', MarkerArray, queue_size=10)
    gt_traj = NavPath(gt_mkrarray_pub, "map", (0.0, 1.0, 0.0, 0.8))
    gt_pose_subscriber = rospy.Subscriber("/delta_gt_pose", TransformStamped, gt_traj.callback)
    est_mkrarray_pub = rospy.Publisher('est_markers', MarkerArray, queue_size=10)
    est_traj = NavPath(est_mkrarray_pub, "map", (1.0, 0.0, 0.0, 0.8))
    est_pose_subscriber = rospy.Subscriber("/delta_est_pose", TransformStamped, gt_traj.callback)

    rospy.spin()
