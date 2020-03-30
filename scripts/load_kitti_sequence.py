#!/usr/bin/env python

from model.dataloader import KittiStandard
from model.utils import dDataLoaderX as DataLoader, PadCollate
import rospy
from model.point_net import Net
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import argparse
from torch_geometric.transforms import GridSampling, RandomTranslate, NormalizeScale, Compose


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path")
parser.add_argument("-gs", "--grid_size", help="gridsampling size", type=float, default=1.)
parser.add_argument("--dropout", type=float, default=0.5)
args = parser.parse_args()
args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg]) for arg in args_dict]))
print('=' * 30)

basedir = '/home/kartmann/share_folder/dataset'
sequence = '00'
frames = None   # range(0, 20, 5)
fields = [PointField('x',           0,  PointField.FLOAT32, 1),
          PointField('y',           4,  PointField.FLOAT32, 1),
          PointField('z',           8,  PointField.FLOAT32, 1),
          PointField('intensity',   12, PointField.FLOAT32, 1)]
sleep_rate = 1.
queue_size = 2

LOAD_GRAPH = False
GRID_SAMPLE_SIZE = [args.grid_size] * 3
transform = Compose([#RandomTranslate(0.001),
                     GridSampling(GRID_SAMPLE_SIZE),
                     NormalizeScale()])

model = Net(graph_input=LOAD_GRAPH, act="LeakyReLU", transform=transform, dropout=args.dropout)
model.eval()


def estimate_pose(source_cloud, target_cloud):



if __name__ == '__main__':
    rospy.init_node("cloudPublisher")
    cloud_pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=queue_size)
    pose_pub = rospy.Publisher("init_guess", PointCloud2, queue_size=queue_size)
    rate = rospy.Rate(sleep_rate)
    header = Header()
    header.frame_id = "map"

    dataset = KittiStandard(sequence, root=basedir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PadCollate(dim=0))

    for idx, (target_cloud, source_cloud, pose_vect) in enumerate(dataloader):
        if rospy.is_shutdown():
            break
        header.seq = idx
        header.stamp = dataset.timestamps[idx]
        pc2 = point_cloud2.create_cloud(header, fields, [point for point in cur_cloud])
        cloud_pub.publish(pc2)
        rospy.logdebug(pc2)
        rate.sleep()

    rospy.spin()
