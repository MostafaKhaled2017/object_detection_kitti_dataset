import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
import numpy as np
import json

class ImageDetectionNode(Node):
    def __init__(self):
        super().__init__('image_detection_node')
        self.get_logger().info("Initializing Image Detection Node")

        self.publisher = self.create_publisher(String, 'image_detection_topic', 10)
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            'images_topic',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

        try:
            self.model = self.load_faster_rcnn_model('../models/faster_rcnn_model.pth')
        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")

    def load_faster_rcnn_model(self, checkpoint_path):
        self.get_logger().info("Loading Faster RCNN Model")
        model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=4)
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = {k.replace("0.0.", "0.").replace("1.0.", "1.").replace("2.0.", "2.").replace("3.0.", "3."): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        self.get_logger().info("Loaded Faster RCNN Model")
        return model

    def image_callback(self, msg):
        self.get_logger().info("Received image")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.get_logger().info(f"Image shape: {cv_image.shape}")
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(cv_image)
            detections = self.detect_objects_faster_rcnn(image_tensor.unsqueeze(0))
            self.get_logger().info(f"Faster R-CNN detections: {str(detections)}")

            # Check if detections are in expected format
            if len(detections) > 0 and 'boxes' in detections[0] and 'labels' in detections[0] and 'scores' in detections[0]:
                detection_list = []
                for i in range(len(detections[0]['boxes'])):
                    if detections[0]['scores'][i].item() > 0.01:
                        detection = {
                            'boxes': detections[0]['boxes'][i].tolist(),
                            'labels': detections[0]['labels'][i].item(),
                            'scores': detections[0]['scores'][i].item()
                        }
                        detection_list.append(detection)

                detection_list_str = json.dumps(detection_list)
                self.get_logger().info(f"Publishing detection message: {detection_list_str}")
                self.publisher.publish(String(data=detection_list_str))
                self.get_logger().info("Published detection message successfully")
            else:
                self.get_logger().error("Unexpected detection format")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def detect_objects_faster_rcnn(self, images):
        with torch.no_grad():
            detections = self.model(images)
        return detections

def main(args=None):
    rclpy.init(args=args)
    node = ImageDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()