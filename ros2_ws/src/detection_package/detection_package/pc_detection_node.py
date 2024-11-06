import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import numpy as np
import json
import sensor_msgs_py.point_cloud2 as pc2

from .demo_dataset import DemoDataset

class PointcloudDetectionNode(Node):
    def __init__(self, cfg_file='../cfgs/kitti_models/second_ros.yaml', checkpoint_path='../models/second_7862.pth'):
        super().__init__('pc_detection_node')
        self.publisher = self.create_publisher(String, 'pc_detection_topic', 10)
        
        # Load model
        self.model, self.dataset = self.load_second_model(cfg_file, checkpoint_path)
        
        # Create a subscriber for the pointclouds topic
        self.subscription = self.create_subscription(
            PointCloud2,
            'pointclouds_topic',
            self.pc_callback,
            10)
        self.subscription  # prevent unused variable warning

    def load_second_model(self, cfg_file, checkpoint_path):
        cfg_from_yaml_file(cfg_file, cfg)
        logger = common_utils.create_logger()
        dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=None, logger=logger
        )
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=checkpoint_path, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        return model, dataset

    def pc_callback(self, msg):
        self.get_logger().info('Received point cloud message')
        try:
            # Convert PointCloud2 message to numpy array
            points_list = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                points_list.append([point[0], point[1], point[2], point[3]])
            points = np.array(points_list, dtype=np.float32)

            input_dict = {'points': points, 'frame_id': 'map'}
            data_dict = self.dataset.prepare_data(data_dict=input_dict)
            detections = self.detect_objects_second(data_dict)
            self.get_logger().info(f"SECOND detections: {str(detections)}")

            # Check if detections are in expected format
            if len(detections) > 0 and 'pred_boxes' in detections[0] and 'pred_labels' in detections[0] and 'pred_scores' in detections[0]:
                detection_list = []
                for i in range(len(detections[0]['pred_boxes'])):
                    if detections[0]['pred_scores'][i].item() > 0.2:
                        detection = {
                            'boxes': detections[0]['pred_boxes'][i].tolist(),
                            'labels': detections[0]['pred_labels'][i].item(),
                            'scores': detections[0]['pred_scores'][i].item()
                        }
                        detection_list.append(detection)
                
                detection_list_str = json.dumps(detection_list)
                self.get_logger().info(f"Publishing detection message: {detection_list_str}")
                self.publisher.publish(String(data=detection_list_str))
                self.get_logger().info("Published detection message successfully")
            else:
                self.get_logger().error("Unexpected detection format")
        except Exception as e:
            self.get_logger().error(f"Error processing pc data: {e}")

    def detect_objects_second(self, data_dict):
        with torch.no_grad():
            data_dict = self.dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
        return pred_dicts

def main(args=None):
    rclpy.init(args=args)
    node = PointcloudDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()