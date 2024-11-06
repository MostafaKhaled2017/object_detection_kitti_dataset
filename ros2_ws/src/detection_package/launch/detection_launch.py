from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='detection_package',
            executable='fusion_node',
            name='fusion_node',
            output='screen'
        ),
        Node(
            package='detection_package',
            executable='image_detection_node',
            name='image_detection_node',
            output='screen'
        ),
        Node(
            package='detection_package',
            executable='pc_detection_node',
            name='pc_detection_node',
            output='screen'
        ),
        TimerAction(
            period=20.0,
            actions=[
                Node(
                    package='detection_package',
                    executable='image_publisher_node',
                    name='image_publisher_node',
                    output='screen'
                ),
                Node(
                    package='detection_package',
                    executable='pc_publisher_node',
                    name='pc_publisher_node',
                    output='screen'
                ),
            ]
        )
    ])