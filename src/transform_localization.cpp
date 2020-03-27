#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <std_msgs/Header.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

PointCloud::Ptr source_cloud_ptr ( new PointCloud );
PointCloud::Ptr target_cloud_ptr ( new PointCloud );
PointCloud::Ptr cloud_ptr ( new PointCloud );
uint seq = 0;

ros::Publisher pubTrans;


Eigen::Matrix4f localize ()
{
  PointCloud::Ptr filtered_cloud_ptr ( new PointCloud );
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize ( 1, 1, 1 );
  approximate_voxel_filter.setInputCloud ( target_cloud_ptr );
  approximate_voxel_filter.filter ( *filtered_cloud_ptr );

  std::cout << "Filtered cloud contains " << filtered_cloud_ptr->size ()
            << " data points" << std::endl;

  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setTransformationEpsilon ( 0.01 );
  ndt.setStepSize ( 0.1 );
  ndt.setResolution ( 1.0 );
  ndt.setMaximumIterations ( 35 );
  ndt.setInputSource ( filtered_cloud_ptr );
  ndt.setInputTarget ( source_cloud_ptr );

  Eigen::Matrix4f init_guess;
  init_guess << 1.,   0.0005,   -0.002,   -0.05,
                -0.0005,  1.,   -0.001,   -0.03,
                0.002, 0.001,       1.,   0.085,
                0.,       0.,       0.,      1.;

  PointCloud::Ptr output_cloud_ptr ( new PointCloud );
  ndt.align ( *output_cloud_ptr, init_guess );

  std::cout << "NDT has converged:" << ndt.hasConverged ()
            << " in " << ndt.getFinalNumIteration () << " iters \n"
            << " score: " << ndt.getFitnessScore () << std::endl;

  Eigen::Matrix4f transormMatrix = ndt.getFinalTransformation ();
  pcl::transformPointCloud ( *target_cloud_ptr, *output_cloud_ptr, transormMatrix );

  std::cout << "Transorm Matrix:\n" << transormMatrix << "\n=================="
            << std::endl;
  return transormMatrix;
}


void save_transform_to_file (string filename, Eigen::Matrix4f Tmf)
{
  FILE* fp;
  if ( ( fp = fopen ( filename, "a+") ) == NULL ) printf("cannot open file!\n");
  fprintf( fp, "%f %f %f %f %f %f %f %f %f %f %f %f",
                &Tmf(0, 0), &Tmf(0, 1), &Tmf(0, 2), &Tmf(0, 3),
                &Tmf(1, 0), &Tmf(1, 1), &Tmf(1, 2), &Tmf(1, 3),
                &Tmf(2, 0), &Tmf(2, 1), &Tmf(2, 2), &Tmf(2, 3) )
  fclose ( fp );
}

void broadcast_transform (Eigen::Matrix4f transformMatrix )
{
  static tf::TransformBroadcaster br;

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


void callback (const sensor_msgs::PointCloud2ConstPtr& input)
{
  seq = input->header->seq;
  cloud_ptr.reset ( new PointCloud );    // still need to reset meh?
  pcl::fromROSMsg ( *input, *cloud_ptr );

  std::cout << "Received Pointcloud of size " << cloud_ptr->size () << std::endl;

  if ( input->header->seq == 0 )
  {
    pcl::copyPointCloud ( *cloud_ptr, *source_cloud_ptr );
  }
  else
  {
    target_cloud_ptr = cloud_ptr;
    Eigen::Matrix4f transformMatrix = localize ();
    broadcast_transform ( transformMatrix );
    source_cloud_ptr = target_cloud_ptr;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sub_pcl");
  ros::NodeHandle nh;
  ros::Subscriber subPCL = nh.subscribe<sensor_msgs::PointCloud2>("point_cloud2", 1, &callback);

  ros::spin();
}
