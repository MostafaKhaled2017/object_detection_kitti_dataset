from setuptools import setup
from glob import glob
import os

package_name = 'detection_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mostafa',
    maintainer_email='m.kira@innopolis.university',
    description='ROS2 package for image and pointcloud detection',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_detection_node = detection_package.image_detection_node:main',
            'pc_detection_node = detection_package.pc_detection_node:main',
            'fusion_node = detection_package.fusion_node:main',
            'image_publisher_node = detection_package.image_publisher_node:main',
            'pc_publisher_node = detection_package.pc_publisher_node:main',
        ],
    },
)
