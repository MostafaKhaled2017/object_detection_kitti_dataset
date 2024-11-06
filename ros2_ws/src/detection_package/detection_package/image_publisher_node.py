import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv2 import imread
import os

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher = self.create_publisher(Image, 'images_topic', 10)
        self.bridge = CvBridge()
        self.image_path = '../data/Kitti_00/image_02/0000/000000.png'
        self.publish_image()

    def publish_image(self):
        if not os.path.exists(self.image_path):
            self.get_logger().error(f"Image path does not exist: {self.image_path}")
            return
        cv_image = imread(self.image_path)
        if cv_image is None:
            self.get_logger().error(f"Failed to read image from: {self.image_path}")
            return
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.publisher.publish(ros_image)
        self.get_logger().info('Published image to "images_topic"')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()