#!/usr/bin/python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Vector3, TransformStamped, Quaternion
from std_msgs.msg import Header, ColorRGBA
import numpy as np
from scipy.spatial.transform import Rotation


class NavPath:
    def __init__(self, marker_publisher, frame_id, color4f=(0.0, 1.0, 0.0, 0.8), buffer_size=10, gt_navPath=None, clip_gt_dist=1):
        self.translations = None
        self.quats = None
        self.header = Header(frame_id=frame_id)
        self.marker = Marker()
        self.line_color = ColorRGBA(*color4f)
        self.buffer_size = buffer_size
        self.cnt = 0
        assert 1 <= buffer_size

        self.marker_publisher = marker_publisher
        self.gt_navPath = gt_navPath
        self.clip_gt_dist = clip_gt_dist

    def translation_to_gt(self, seq):
        if "gt" in self.header.frame_id:
            return np.zeros((3, ))
        raw_gt_dist = self.translations[seq] - self.gt_navPath.translations[seq]
        # clip the distance to gt if it's too much, just for visualization
        if 0 < self.clip_gt_dist < np.sqrt(np.sum(np.square(raw_gt_dist))):
            return raw_gt_dist * self.clip_gt_dist / np.sqrt(np.sum(np.square(raw_gt_dist)))
        return raw_gt_dist

    def update_marker(self):
        marker = Marker(type=Marker.LINE_STRIP, color=ColorRGBA(1, 1, 1, 1), lifetime=rospy.Duration(1),
                        header=self.header, frame_locked=False)
        marker.scale.x = 0.02
        marker.pose.orientation = Quaternion(*list(self.quats[-1]))
        for translation in self.translations:
            marker.points.append(Point(*list(- self.translations[-1] + translation - self.translation_to_gt(len(self.translations) - 1))))
            marker.colors.append(self.line_color)
            if len(self.translations) < 2:
                marker.points.append(Point(*list(- self.translations[-1] + translation - self.translation_to_gt(len(self.translations) - 1))))
                marker.colors.append(self.line_color)
                marker.id = self.cnt
                self.cnt += 1
            marker.id = self.cnt
            self.cnt += 1
        self.marker = marker

    def publish_marker(self):
        self.marker_publisher.publish(self.marker)
        if "gt" in self.header.frame_id:
            rospy.loginfo(self.translations)

    def callback(self, msg):
        translation = np.array([msg.transform.translation.x,
                                msg.transform.translation.y,
                                msg.transform.translation.z]).reshape(1, -1)
        quat = np.array([msg.transform.rotation.x,
                         msg.transform.rotation.y,
                         msg.transform.rotation.z,
                         msg.transform.rotation.w]).reshape(1, -1)
        self.translations = translation if self.translations is None else np.concatenate([self.translations, translation])
        self.quats = quat if self.quats is None else np.concatenate([self.quats, quat])

        self.update_marker()
        self.publish_marker()


if __name__ == '__main__':
    rospy.init_node('viz_markers_node')
    gt_mkr_pub = rospy.Publisher('gt_markers', Marker, queue_size=10)
    gt_traj = NavPath(gt_mkr_pub, "/gt_car", (0.0, 1.0, 0.0, 0.8))
    gt_pose_subscriber = rospy.Subscriber("/gt_pose", TransformStamped, gt_traj.callback)

    est_mkr_pub = rospy.Publisher('est_markers', Marker, queue_size=10)
    est_traj = NavPath(est_mkr_pub, "/est_car", (1.0, 0.0, 0.0, 0.8), gt_navPath=gt_traj)
    est_pose_subscriber = rospy.Subscriber("/est_pose", TransformStamped, est_traj.callback)

    ndt_mkr_pub = rospy.Publisher('ndt_markers', Marker, queue_size=10)
    ndt_traj = NavPath(ndt_mkr_pub, "/ndt_car", (0.0, 0.0, 1.0, 0.8), gt_navPath=gt_traj)
    ndt_pose_subscriber = rospy.Subscriber("/ndt_pose", TransformStamped, ndt_traj.callback)
    rospy.spin()
