#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate
from pathlib import Path
import numpy as np
import sys


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = Path(root_path)
        self.ext = ext
        if self.root_path.is_dir():
            data_file_list = glob.glob(str(self.root_path / ('*' + self.ext)))
        else:
            data_file_list = [str(self.root_path)]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    

class LiDARDetectionNode(Node):
    def __init__(self, lidar_path):
        super().__init__('lidar_detection_node')
        self.publisher = self.create_publisher(String, 'lidar_detection_topic', 10)
        self.model, self.dataset = self.load_second_model('../cfgs/kitti_models/second_ros.yaml', '../models/second_7862.pth', lidar_path)
        self.process_lidar(lidar_path)

    def load_second_model(self, cfg_file, checkpoint_path, data_path, ext='.bin'):
        cfg_from_yaml_file(cfg_file, cfg)
        logger = common_utils.create_logger()
        dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(data_path), ext=ext, logger=logger
        )
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=checkpoint_path, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        return model, dataset

    def process_lidar(self, lidar_path):
        self.get_logger().info(f"Processing LiDAR data from: {lidar_path}")
        try:
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            input_dict = {'points': points, 'frame_id': 'map'}
            data_dict = self.dataset.prepare_data(data_dict=input_dict)
            detections = self.detect_objects_second(data_dict)
            self.get_logger().info(f"SECOND detections: {str(detections)}")
            self.publisher.publish(String(data=str(detections)))
        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {e}")

    def detect_objects_second(self, data_dict):
        with torch.no_grad():
            data_dict = self.dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
        return pred_dicts

def main(args=None):
    rclpy.init(args=args)
    lidar_path = sys.argv[1] if len(sys.argv) > 1 else '../data/Kitti_00/velodyne/0000/processed/000000.bin'
    node = LiDARDetectionNode(lidar_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()