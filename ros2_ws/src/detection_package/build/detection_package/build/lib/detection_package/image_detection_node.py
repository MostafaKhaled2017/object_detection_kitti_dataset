#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
import os
import sys
import numpy as np
from PIL import Image as PILImage

class ImageDetectionNode(Node):
    def __init__(self, image_path):
        super().__init__('image_detection_node')
        self.publisher = self.create_publisher(String, 'image_detection_topic', 10)
        self.bridge = CvBridge()
        
        try:
            self.model = self.load_faster_rcnn_model('../models/faster_rcnn_model.pth')
        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")

        # Initial dummy publish to create the topic
        self.publisher.publish(String(data=""))

        # Process the image
        self.process_image(image_path)

    def load_faster_rcnn_model(self, checkpoint_path):
        self.get_logger().info("Loading Faster RCNN Model")
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=4)
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = {k.replace("0.0.", "0.").replace("1.0.", "1.").replace("2.0.", "2.").replace("3.0.", "3."): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        self.get_logger().info("Loaded Faster RCNN Model")
        return model

    def process_image(self, image_path):
        self.get_logger().info("Processing image")
        try:
            cv_image = PILImage.open(image_path).convert('RGB')
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(cv_image)
            detections = self.detect_objects_faster_rcnn(image_tensor.unsqueeze(0))
            self.get_logger().info(f"Faster R-CNN detections: {str(detections)}")
            self.publisher.publish(String(data=str(detections)))
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def detect_objects_faster_rcnn(self, images):
        with torch.no_grad():
            detections = self.model(images)
        return detections

def main(args=None):
    rclpy.init(args=args)
    image_path = sys.argv[1] if len(sys.argv) > 1 else '../data/Kitti_00/image_02/0000/000000.png'
    node = ImageDetectionNode(image_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()