import sys
if sys.prefix == '/home/mostafa/miniconda3/envs/kitti':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/mostafa/Data/Git Repos/perception_testing_task/ros2_ws/src/detection_package/install/detection_package'
