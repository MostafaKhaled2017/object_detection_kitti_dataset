#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import torch
import torchvision.ops as ops
import json

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        self.image_subscription = self.create_subscription(String, 'image_detection_topic', self.image_callback, 10)
        self.lidar_subscription = self.create_subscription(String, 'lidar_detection_topic', self.lidar_callback, 10)
        self.publisher = self.create_publisher(String, 'fused_detection_topic', 10)
        self.rgb_detections = None
        self.lidar_detections = None
        self.calib_file = '../data/Kitti_00/calib/0000.txt'

    def image_callback(self, msg):
        self.rgb_detections = msg.data
        self.get_logger().info(f"RGBD Detections Recieved: {self.rgb_detections}, type : {type(self.rgb_detections)}")
        if self.lidar_detections is not None:
            self.fuse_and_publish()

    def lidar_callback(self, msg):
        self.lidar_detections = msg.data
        self.get_logger().info(f"Lidar Detections Recieved: {self.lidar_detections}, type : {type(self.lidar_detections)}")
        if self.rgb_detections is not None:
            self.fuse_and_publish()

    def fuse_and_publish(self):
        if self.rgb_detections is not None and self.lidar_detections is not None:
            fused_detections = self.fuse_detections(self.rgb_detections, self.lidar_detections, self.calib_file)
            fused_detections_str = json.dumps(fused_detections)
            self.get_logger().info(f"Fused detections: {fused_detections_str}")
            self.publisher.publish(String(data=fused_detections_str))
            # Reset detections after publishing
            self.rgb_detections = None
            self.lidar_detections = None

    def fuse_detections(rgb_detections, lidar_detections, calib_file, device='cuda:0', weight_rgb=0.5, weight_lidar=0.5, iou_threshold=0.5):

        # Parsing the detections from string type to list type
        rgb_detections =  eval(rgb_detections)
        lidar_detections = eval(lidar_detections)
    
        print(f"[INFO] RGB Detections: {rgb_detections}, type : {type(rgb_detections)}")
        print(f"[INFO] Lidar Detections: {lidar_detections}, type : {type(lidar_detections)}")
        
        fused_detections = []

        for rgb_det, lidar_det_list in zip(rgb_detections, lidar_detections):
            # Move RGB detections to the specified device
            rgb_boxes = rgb_det['boxes'].to(device)
            rgb_labels = rgb_det['labels'].to(device)
            rgb_scores = rgb_det['scores'].to(device)
            
            # Handle the nested list in lidar_detections
            for lidar_det in lidar_det_list:
                # Project LiDAR boxes to 2D
                lidar_boxes_3d = lidar_det['pred_boxes'].cpu().numpy()
                lidar_boxes = project_lidar_to_2d(lidar_boxes_3d, calib_file).to(device)
                lidar_labels = lidar_det['pred_labels'].to(device)
                lidar_scores = lidar_det['pred_scores'].to(device)

                # Combine the RGB and LiDAR detections
                combined_boxes = torch.cat((rgb_boxes, lidar_boxes), dim=0)
                combined_labels = torch.cat((rgb_labels, lidar_labels), dim=0)
                combined_scores = torch.cat((rgb_scores * weight_rgb, lidar_scores * weight_lidar), dim=0)

                # Apply Non-Maximum Suppression (NMS) to merge overlapping boxes
                keep_indices = ops.nms(combined_boxes, combined_scores, iou_threshold)

                fused_boxes = combined_boxes[keep_indices]
                fused_labels = combined_labels[keep_indices]
                fused_scores = combined_scores[keep_indices]

                fused_det = {
                    'boxes': fused_boxes,
                    'labels': fused_labels,
                    'scores': fused_scores
                }
                fused_detections.append(fused_det)

        return fused_detections


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()