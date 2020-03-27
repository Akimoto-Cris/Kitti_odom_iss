#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <iostream>


int main(int argc, char** argv) {
  ros::init ( argc, argv, "tf_listener" );
  ros::NodeHandle nh;
  tf::TransformListener listener;
  ros::Rate (1.);

  while ( nh.ok () ) {
    tf::StampedTransform transform;
    try {
      ros::Time now = ros::Time::now ();
      listener.waitForTransform ( "car", "map",  now, ros::Duration( 3.0 ) );
      listener.lookupTransform ("car", "map", now, transform );

      std::cout << transform.frame_id_ << " transform callback:\ntranslation:\n"
                << *transform.getOrigin() << "\nrotation:\n" << *transform.getRotation() << std::endl;
    }
    catch ( tf::TransformException ex ) {
      ROS_ERROR ( "%s", ex.what () );
      ros::Duration ( 1. ).sleep ();
    }

  }
  return 0;
}
