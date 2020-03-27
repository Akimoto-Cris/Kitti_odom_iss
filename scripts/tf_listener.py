#!/usr/bin/env python

import rospy
import tf

# deprecated

if __name__ == '__main__':
    rospy.init_node("ndt_pose_listener")
    listener = tf.TransformListener()
    rate = rospy.Rate(1.)

    while not rospy.is_shutdown():
        if listener.frameExists("map") and listener.frameExists("car"):
            t = listener.getLatestCommonTime("car", "map")
            trans, rot = listener.lookupTransform("car", "map", t)
            print("At {}, get translation: {}, quat: {}".format(t, trans, rot))
        rate.sleep()

    rospy.spin()
