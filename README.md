# 数据集下载
---
下载 kitti 数据集： http://www.cvlibs.net/datasets/kitti/eval_odometry.php 。只需要下载三个文件：
- (velodyne laser data, 80 GB)
- (calibration files, 1 MB)
- ground truth poses (4 MB)

下载的三个压缩包在同一目录下解压，解压后的目录结构应该是：
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

# point_net 训练
---

## Anaconda 环境
```
conda create -n torch python=3.6 numpy
conda activate torch
conda install -c pytorch pytorch
conda install tensorflow==1.14.0 tensorflow-gpu==1.14.0 tensorboard==1.14.0 scipy
python -m pip install prefetch_generator tqdm pykitti

pip install torch-scatter==latest+${CUDA} torch-sparse==latest+${CUDA} torch-cluster==latest+${CUDA} torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```

## 训练
1. 在训练之前，请务修改 `.\scripts\model\train_net.py` 中 `sequence`, `train_seqences` 和 `val_seqences`，设置为所需分别用于训练和验证的 sequence。
2. 把 `.\scripts\model\train_net.cmd` 脚本中的 `datadir` 设为自己的路径
3. 双击 `.\scripts\model\train_net.cmd` 开始训练
4. （optional）另外开一个命令行，`conda activate torch && tensorboard --logdir=${project_root_dir}\scripts\model\log --host localhost --port=8080`，浏览器中打开 http://localhost:8080 进入 tensorboard 调试

__NOTE__: ${CUDA} 为 cpu, cu92, cu100 或 cu101，视 cuda 安装版本而定。可用 `nvcc -V` 查看 cuda 版本。

# ROS part
---
## Prerequisites

1. 我的环境：Ubuntu 18.04 VM Ware 虚拟机
2. 按照 ros 官网指南搭建好 `~/catkin_ws` 下的工作空间
3. `cd ~/catkin_ws/src/ && mkdir kitti_localization && cd kitti_localization`
4. 将本 git repo 全部内容放在当前路径中

## 编译

1. `cd ~/catkin_ws`
2. `catkin_make`
3. `source ./devel/setup.bash`

## 运行

1. `roscore`
2. 在另一 terminal 中：`rosrun kitti_localization kitti_localization_localizer`
3. 在另一 terminal 中：`rosrun kitti_localization load_kitti_sequence.py`

## 可能遇到的问题

### 运行 python 脚本时找不到 rospkg

Solution: 参考 https://blog.csdn.net/qq_36501182/article/details/79971570

## 运行 python 脚本时 No module named 'tf2_ros'

Solution: 请向我要 tf 源码，在 `catkin_ws` 下重新编译 （一般不会遇到这个问题）
