# Prerequisites

1. 我的环境：Ubuntu 18.04 VM Ware 虚拟机
2. 按照 ros 官网指南搭建好 `~/catkin_ws` 下的工作空间
3. `cd ~/catkin_ws/src/ && mkdir kitti_localization && cd kitti_localization`
4. 将本 git repo 全部内容放在当前路径中
5. 下载 kitti 数据集： http://www.cvlibs.net/datasets/kitti/eval_odometry.php。只需要下载三个文件：

    - (velodyne laser data, 80 GB)
    - (calibration files, 1 MB)
    - ground truth poses (4 MB)
6. 下载的三个压缩包在同一目录下解压，解压后的目录结构应该是：
```
dataset
├── poses
│   ├── 00.txt
│   ├── 01.txt
│   └── ...
└── sequences
    ├── 00
    │   ├── velodyne
    │   ├── calib.txt
    │   └── times.txt
    ├── 01
    │   ├── velodyne
    │   ├── calib.txt
    │   └── times.txt
    ├── ...
    ...
```

# 编译

1. `cd ~/catkin_ws`
2. `catkin_make`
3. `source ./devel/setup.bash`

# 运行

1. `roscore`
2. 在另一 terminal 中：`rosrun kitti_localization kitti_localization_localizer`
3. 在另一 terminal 中：`rosrun kitti_localization load_kitti_sequence.py`

# 可能遇到的问题

## 运行 python 脚本时找不到 rospkg

Solution: 参考 https://blog.csdn.net/qq_36501182/article/details/79971570

## 运行 python 脚本时 No module named 'tf2_ros'

Solution: 请向我要 tf 源码，在 `catkin_ws` 下重新编译 （一般不会遇到这个问题）
