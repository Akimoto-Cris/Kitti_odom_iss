/**
@Author: Xu Kaixin
@License: Apache Licence
**/
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <std_msgs/Header.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <geometry_msgs/TransformStamped.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <climits>
#include <bits/stdc++.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

using KdTreeReciprocal = pcl::search::KdTree<pcl::PointXYZ>;
using KdTreeReciprocalPtr = typename KdTreeReciprocal::Ptr;
using namespace std;

PointCloud::Ptr source_cloud_ptr ( new PointCloud );
PointCloud::Ptr target_cloud_ptr ( new PointCloud );
PointCloud::Ptr cloud_ptr ( new PointCloud );
KdTreeReciprocalPtr target_tree ( new KdTreeReciprocal );
map<uint, Eigen::Matrix4f> tf_queue;
uint seq = 0;
uint TF_QUEUE_SIZE = 10;

static tf::TransformBroadcaster br;
//tf::TransformListener tf_listener;


Eigen::Matrix4f localize (const Eigen::Matrix4f& init_guess)
{
  PointCloud::Ptr filtered_cloud_ptr ( new PointCloud );
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize ( 1, 1, 1 );
  approximate_voxel_filter.setInputCloud ( target_cloud_ptr );
  approximate_voxel_filter.filter ( *filtered_cloud_ptr );

  cout << "Filtered cloud contains " << filtered_cloud_ptr->size () << " data points" << endl;

  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  target_tree->setInputCloud ( target_cloud_ptr );

  ndt.setTransformationEpsilon ( 0.01 );
  ndt.setStepSize ( 0.1 );
  ndt.setResolution ( 1.0 );
  ndt.setMaximumIterations ( 35 );
  ndt.setInputSource ( filtered_cloud_ptr );
  ndt.setInputTarget ( source_cloud_ptr );
  ndt.setSearchMethodTarget ( target_tree );

  /*tf::Matrix3x3 tf3d;
  tf::Quaternion q ( transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w );
  tf3d.setRotation ( q );

  Eigen::Matrix4f init_guess;
  init_guess << tf3d[0][0], tf3d[0][1], tf3d[0][2], transform.translation.x,
                tf3d[1][0], tf3d[1][1], tf3d[1][2], transform.translation.y,
                tf3d[2][0], tf3d[2][1], tf3d[2][2], transform.translation.z,
                0.,         0.,         0.,         1.;

  init_guess << 1.,   0.0005,   -0.002,   -0.05,
                -0.0005,  1.,   -0.001,   -0.03,
                0.002, 0.001,       1.,   0.085,
                0.,       0.,       0.,      1.;*/

  PointCloud::Ptr output_cloud_ptr ( new PointCloud );
  ndt.align ( *output_cloud_ptr, init_guess );

  cout << "NDT has converged:" << ndt.hasConverged ()
       << " in " << ndt.getFinalNumIteration () << " iters \n"
       << " score: " << ndt.getFitnessScore () << endl;

  Eigen::Matrix4f transormMatrix = ndt.getFinalTransformation ();
  pcl::transformPointCloud ( *target_cloud_ptr, *output_cloud_ptr, transormMatrix );

  cout << "Transorm Matrix:\n" << transormMatrix << "\n==================" << endl;
  return transormMatrix;
}


void save_transform_to_file (const char* filename, const Eigen::Matrix4f& Tmf)
{
  FILE* fp;
  if ( ( fp = fopen ( filename, "a+" ) ) == NULL ) printf("cannot open file!\n");
  fprintf( fp, "%f %f %f %f %f %f %f %f %f %f %f %f",
                &Tmf(0, 0), &Tmf(0, 1), &Tmf(0, 2), &Tmf(0, 3),
                &Tmf(1, 0), &Tmf(1, 1), &Tmf(1, 2), &Tmf(1, 3),
                &Tmf(2, 0), &Tmf(2, 1), &Tmf(2, 2), &Tmf(2, 3) );
  fclose ( fp );
}


void broadcast_transform (Eigen::Matrix4f transformMatrix )
{
  Eigen::Matrix4d Tmd = transformMatrix.cast<double> ();
  tf::Matrix3x3 tf3d;
  tf3d.setValue ( Tmd(0, 0), Tmd(0, 1), Tmd(0, 2),
                  Tmd(1, 0), Tmd(1, 1), Tmd(1, 2),
                  Tmd(2, 0), Tmd(2, 1), Tmd(2, 2) );
  tf::Quaternion q;
  tf3d.getRotation ( q );

  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = ros::Time::now ();
  transformStamped.header.frame_id = "map";
  transformStamped.header.seq = seq;
  transformStamped.child_frame_id = "car";
  transformStamped.transform.translation.x = Tmd(0, 3);
  transformStamped.transform.translation.y = Tmd(1, 3);
  transformStamped.transform.translation.z = Tmd(2, 3);
  transformStamped.transform.rotation.x = q.x ();
  transformStamped.transform.rotation.y = q.y ();
  transformStamped.transform.rotation.z = q.z ();
  transformStamped.transform.rotation.w = q.w ();
  br.sendTransform ( transformStamped );
}


void pcl_callback (const sensor_msgs::PointCloud2ConstPtr input)
{
  seq = input->header.seq;
  cloud_ptr.reset ( new PointCloud );    // still need to reset meh?
  pcl::fromROSMsg ( *input, *cloud_ptr );

  cout << "Received Pointcloud of size " << cloud_ptr->size () << endl;

  if ( seq == 0 )
  {
    pcl::copyPointCloud ( *cloud_ptr, *source_cloud_ptr );
  }
  else
  {
    target_cloud_ptr = cloud_ptr;
    Eigen::Matrix4f transformMatrix = localize ( tf_queue[seq] );
    broadcast_transform ( transformMatrix );
    source_cloud_ptr = target_cloud_ptr;
  }
  // grasp the pose guess corresponding to this grame of scan
}


Eigen::Matrix4f tf2d_to_matrix4f (const geometry_msgs::TransformStamped transform)
{
  /*Eigen::Translation3f tl_btol(transform.getOrigin().getX(), transform.getOrigin().getY(), transform.getOrigin().getZ());
  double roll, pitch, yaw;
  tf::Matrix3x3(transform.getRotation()).getEulerYPR(yaw, pitch, roll);
  Eigen::AngleAxisf rot_x_btol(roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf rot_y_btol(pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_z_btol(yaw, Eigen::Vector3f::UnitZ());
  return (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix().inverse();*/
  Eigen::Affine3d tf3d = tf2::transformToEigen ( transform );
  Eigen::Affine3f tf3f = tf3d.cast<float> ();
  Eigen::Matrix4f tf4f = tf3f.matrix ();
  return tf4f;
}

void pose_guess_callback ( const geometry_msgs::TransformStamped stamped_transform )
{
  cout << "Received TF Guess:\n" << stamped_transform << endl;

  if ( tf_queue.size () >= TF_QUEUE_SIZE ) { tf_queue.erase ( tf_queue.begin ()->first ); }
  tf::StampedTransform temp_tf;
  tf::transformStampedMsgToTF (stamped_transform, temp_tf);
  tf_queue[stamped_transform.header.seq] = tf2d_to_matrix4f ( stamped_transform );
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "sub_pcl");
  ros::NodeHandle nh;
  ros::Subscriber subPoseGuess = nh.subscribe<geometry_msgs::TransformStamped> ("init_guess", 1, &pose_guess_callback);
  ros::Subscriber subPCL = nh.subscribe<sensor_msgs::PointCloud2> ("point_cloud2", 1, &pcl_callback);

  ros::spin();
}
