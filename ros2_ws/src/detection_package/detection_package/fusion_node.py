import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import torch
import torchvision.ops as ops
from .utils import *
import json

class FusionNode(Node):
    def __init__(self, calib_file='../data/Kitti_00/calib/0000.txt'):
        super().__init__('fusion_node')
        self.get_logger().info("Fusion Node has been started.")
        self.image_subscription = self.create_subscription(String, 'image_detection_topic', self.image_callback, 10)
        self.pc_subscription = self.create_subscription(String, 'pc_detection_topic', self.pc_callback, 10)
        self.publisher = self.create_publisher(String, 'fused_detection_topic', 10)
        self.rgb_detections = None
        self.pc_detections = None
        self.rgb_received = False
        self.pc_received = False
        self.calib_file = calib_file
        
        self.get_logger().info("Subscribed to image_detection_topic and pc_detection_topic.")
        
    def image_callback(self, msg):
        self.get_logger().info("Image callback triggered.")
        self.rgb_detections = self.parse_detections(msg.data)
        self.rgb_received = True
        self.get_logger().info(f"RGB Detections Parsed: {self.rgb_detections}")
        if self.rgb_received and self.pc_received:
            self.fuse_and_publish()

    def pc_callback(self, msg):
        self.get_logger().info("pc callback triggered.")
        self.pc_detections = self.parse_detections(msg.data)
        self.pc_received = True
        self.get_logger().info(f"pc Detections Parsed: {self.pc_detections}")
        if self.rgb_received and self.pc_received:
            self.fuse_and_publish()

    def parse_detections(self, msg_data):
        detections = json.loads(msg_data)
        return detections

    def fuse_and_publish(self):
        self.get_logger().info("Fuse and Publish function has been called.")
        fused_detections = self.fuse_detections(self.rgb_detections, self.pc_detections, self.calib_file)
        detection_list_str = json.dumps(fused_detections)
        self.publisher.publish(String(data=detection_list_str))
        # Reset detections and received flags after publishing
        self.rgb_detections = None
        self.pc_detections = None
        self.rgb_received = False
        self.pc_received = False

    def fuse_detections(self, rgb_detections, pc_detections, calib_file, device='cuda:0', weight_rgb=0.5, weight_pc=0.5, iou_threshold=0.5):
        fused_detections = []

        rgb_boxes = torch.tensor([det['boxes'] for det in rgb_detections]).to(device) if rgb_detections else torch.empty((0, 4)).to(device)
        rgb_labels = torch.tensor([det['labels'] for det in rgb_detections]).to(device) if rgb_detections else torch.empty((0,), dtype=torch.long).to(device)
        rgb_scores = torch.tensor([det['scores'] for det in rgb_detections]).to(device) if rgb_detections else torch.empty((0,), dtype=torch.float).to(device)

        pc_boxes_3d = torch.tensor([det['boxes'] for det in pc_detections]).cpu().numpy() if pc_detections else torch.empty((0, 7)).cpu().numpy()

        if pc_boxes_3d.size > 0 and isinstance(pc_boxes_3d[0], np.ndarray) and isinstance(pc_boxes_3d[0][0], np.ndarray):
            pc_boxes_3d = pc_boxes_3d[0]

        pc_boxes = project_lidar_to_2d(pc_boxes_3d, calib_file).to(device) if pc_detections else torch.empty((0, 4)).to(device)
        pc_labels = torch.tensor([det['labels'] for det in pc_detections]).to(device) if pc_detections else torch.empty((0,), dtype=torch.long).to(device)
        pc_scores = torch.tensor([det['scores'] for det in pc_detections]).to(device) if pc_detections else torch.empty((0,), dtype=torch.float).to(device)

        combined_boxes = torch.cat((rgb_boxes, pc_boxes), dim=0)
        combined_labels = torch.cat((rgb_labels, pc_labels), dim=0)
        combined_scores = torch.cat((rgb_scores * weight_rgb, pc_scores * weight_pc), dim=0)

        if combined_boxes.shape[0] > 0:
            keep_indices = ops.nms(combined_boxes, combined_scores, iou_threshold)
            fused_boxes = combined_boxes[keep_indices]
            fused_labels = combined_labels[keep_indices]
            fused_scores = combined_scores[keep_indices]

            fused_det = {
                'boxes': fused_boxes.tolist(),
                'labels': fused_labels.tolist(),
                'scores': fused_scores.tolist()
            }
            fused_detections.append(fused_det)

        self.get_logger().info(f"Fused Detections: {fused_detections}")

        return fused_detections

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()