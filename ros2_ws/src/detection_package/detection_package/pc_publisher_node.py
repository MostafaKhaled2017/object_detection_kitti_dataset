import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('pointcloud_publisher')
        self.publisher = self.create_publisher(PointCloud2, 'pointclouds_topic', 10)
        self.publish_pointcloud()

    def publish_pointcloud(self):
        points = np.fromfile('../data/Kitti_00/velodyne/0000/processed/000000.bin', dtype=np.float32).reshape(-1, 4)
        header = self.create_header()
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pointcloud_msg = pc2.create_cloud(header, fields, points)
        self.publisher.publish(pointcloud_msg)
        self.get_logger().info('Published point cloud to "pointclouds_topic"')

    def create_header(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        return header

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPublisher()
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()