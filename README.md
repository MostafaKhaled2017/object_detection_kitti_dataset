# KITTI Dataset Processing and Object Detection

This project contains scripts for processing the KITTI dataset, training object detection model, and performing sensor fusion. The object detection model is trained using Faster R-CNN, and the sensor fusion is performed using the detections from the images and the corresponding point clouds. The camera calibration script is used to obtain the intrinsic and extrinsic parameters of the camera.

All the codes were tested in the following environment:
- Linux (Ubuntu 22.04)
- Python 3.10
- CUDA 10.1

## Installation

1. Install the project environment:
   ```bash
   conda env create -f environment.yml
   conda activate kitti
   ```

2. Install the the OpenPCDet submodule:
   ```bash
   pip install -e submodules/OpenPCDet
   ```

## Usage

### 1. Point Cloud Preprocessing

This script preprocesses the point clouds from the KITTI dataset.

**Command:**
```bash
python scripts/point_cloud_preprocessing.py --directory_path /path/to/kitti/point_clouds
```

- `--directory_path`: Path to the raw point cloud data from the KITTI dataset.

### 2. Training Faster R-CNN

This script trains an object detection model on the images from the KITTI dataset using Faster R-CNN.

**Command:**
```bash
python scripts/train_faster_rcnn.py --images_directory /path/to/kitti/images --annotations_file /path/to/annotations --num_epochs num_epochs
```

- `--images_directory`: Path to the images from the KITTI dataset.
- `--annotations_file`: Path to the annotation files for the images.
- `--num_epochs`: Number of epochs to train the model.

### 3. Fusion Algorithm

This script performs fusion of detections from images and their corresponding point clouds.

**Command:**
```bash
python scripts/fusion_algorithm.py --image_path /path/to/kitti/images --point_cloud_path /path/to/point_clouds
```

- `--image_path`: Path to the images from the KITTI dataset.
- `--point_cloud_path`: Path to the preprocessed point clouds.

### 4. Camera Calibration

This script performs calibration to obtain intrinsic and extrinsic parameters of the camera using the checkerboard calibration method.

**Command:**
```bash
python scripts/camera_calibration.py --checkerboard_size checkerboard_size --square_size square_size
```

- `--checkerboard_size`: Size of the checkerboard (e.g., 8x6).
- `--square_size`: Size of the squares on the checkerboard.

## Testing
Pytest framework was used for wiriting unit test for the different code modules. To run the tests, you can simply use the following command:
```bash
pytest tests/
```

## Ros 2 Package
The fusion algorithm is wrapped in a ROS2 package. The package contains the following nodes:
- `image_detection_node`: Read the images published in the images topic and detects objects in the images.
- `pc_detection_node`: Read the point clouds published in the point clouds topic and detects objects in the point clouds.
- `fusion_node`: Reads the detections published by the image_detection_node and the pc_detection_node and performs sensor fusion.
- `image_publisher_node`: Publishes a sample image to the images topic.
- `pc_publisher_node`: Publishes a sample point cloud to the point clouds topic.

### Running the ROS2 Package
To run the nodes of the ROS2 package, follow the steps below:

1. Go to the `ros2_ws` directory:
   ```bash
   cd ros2_ws
   ```
2. Build the ROS package:
   ```bash
   colcon build
   ```
3. Source the ROS package:
   ```bash
    source install/setup.bash
    ```
4. Run the launch file:
   ```bash
   ros2 launch src/detection_package/launch/detection_launch.py
   ```