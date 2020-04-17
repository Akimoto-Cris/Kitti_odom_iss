#!/usr/bin/python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Vector3, TransformStamped, Quaternion
from std_msgs.msg import Header, ColorRGBA
import numpy as np
from scipy.spatial.transform import Rotation


class NavPath:
    def __init__(self, marker_publisher, frame_id, color4f=(0.0, 1.0, 0.0, 0.8), buffer_size=50, gt_navPath=None, clip_gt_dist=1):
        self.transforms = []
        self.quat = None
        self.header = Header(frame_id="map")
        self.marker = Marker()
        self.line_color = ColorRGBA(*color4f)
        self.buffer_size = buffer_size
        self.cnt = 0
        assert 1 <= buffer_size

        self.marker_publisher = marker_publisher
        self.gt_navPath = gt_navPath
        self.clip_gt_dist = clip_gt_dist
        self.newest_pose = None
        self.frame_id = frame_id

    def delta_poses(self, C_nk, C_n0):
        ret = np.zeros(C_nk.shape)
        ret[:3, :3] = np.dot(C_nk[:3, :3], C_n0[:3, :3].T)
        ret[:3, -1] = C_nk[:3, -1] - C_n0[:3, -1]
        return ret

    def update_marker(self):
        marker = Marker(type=Marker.LINE_STRIP, lifetime=rospy.Duration(1), header=self.header,
                        frame_locked=True)
        marker.scale.x = 0.02
        #marker.pose.orientation = Quaternion(*list(self.quat))

        for transform in self.transforms:
            point = Point(*list(transform[:3, -1].squeeze()))
            marker.points.append(point)
            marker.colors.append(self.line_color)

            if len(self.transforms) < 2:
                marker.points.append(point)
                marker.colors.append(self.line_color)
                marker.id = self.cnt
                self.cnt += 1
            marker.id = self.cnt
            self.cnt += 1
        self.marker = marker

    def publish_marker(self):
        self.marker_publisher.publish(self.marker)

    def callback(self, msg):
        pose_mat = np.eye(4)[:3, :]
        pose_mat[:3, -1] = np.array([msg.transform.translation.x,
                                     msg.transform.translation.y,
                                     msg.transform.translation.z])
        self.quat = np.array([msg.transform.rotation.x,
                              msg.transform.rotation.y,
                              msg.transform.rotation.z,
                              msg.transform.rotation.w])
        pose_mat[:3, :3] = Rotation.from_quat(self.quat).as_matrix()
        self.newest_pose = pose_mat

        self.transforms.append(pose_mat)
        if len(self.transforms) > self.buffer_size:
            self.transforms = self.transforms[1:]

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
