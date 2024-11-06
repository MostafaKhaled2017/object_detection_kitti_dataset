from pathlib import Path
import numpy as np
import cv2
import torch

import glob

# Load images
def load_images(image_path):
    image_files = []
    if Path(image_path).is_dir():
        image_files = sorted(glob.glob(str(Path(image_path) / '*.png')))
    else:
        image_files = [image_path]
    
    images = []
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            raise FileNotFoundError(f"Image file {image_file} not found")
        images.append(image)
    
    return images

# Visualize the detections
def visualize_detections(images, detections, window_name='Detections'):
    for image, det in zip(images, detections):
        boxes = det['boxes']
        labels = det['labels']
        scores = det['scores']
        
        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy().astype(int)
            label = labels[i].item()
            score = scores[i].item()
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(image, f'{label}:{score:.2f}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
        cv2.imshow(window_name, image)
        key = cv2.waitKey(0)
        if key == ord('q'): 
            break
    cv2.destroyAllWindows()

# Load calibration data from file.
def load_calibration(calib_file):
    """
    Load calibration data from file.
    """
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    calib = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()])
        else:
            key, value = line.split(' ', 1)
            calib[key] = np.array([float(x) for x in value.split()])
    
    return calib

# Transform points from LiDAR coordinates to Camera coordinates.
def transform_lidar_to_camera(points, Tr):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points_camera = np.dot(Tr, points.T).T
    return points_camera[:, :3]

# Project points from Camera coordinates to Image plane.
def project_to_image(points, P):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points_2d = np.dot(P, points.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    return points_2d

# Projects 3D LiDAR bounding boxes to 2D bounding boxes.
def project_lidar_to_2d(lidar_boxes, calib_file='data/calib.txt'):
    # Load calibration parameters
    calib = load_calibration(calib_file)
    P2 = calib['P2'].reshape(3, 4)
    Tr_velo_to_cam = calib['Tr_velo_cam'].reshape(3, 4)

    boxes_2d = []
    for box in lidar_boxes:
        # Extract the 3D box corners
        corners = np.array([
            [box[0] - box[3] / 2, box[1] - box[4] / 2, box[2] - box[5] / 2],
            [box[0] - box[3] / 2, box[1] - box[4] / 2, box[2] + box[5] / 2],
            [box[0] - box[3] / 2, box[1] + box[4] / 2, box[2] - box[5] / 2],
            [box[0] - box[3] / 2, box[1] + box[4] / 2, box[2] + box[5] / 2],
            [box[0] + box[3] / 2, box[1] - box[4] / 2, box[2] - box[5] / 2],
            [box[0] + box[3] / 2, box[1] - box[4] / 2, box[2] + box[5] / 2],
            [box[0] + box[3] / 2, box[1] + box[4] / 2, box[2] - box[5] / 2],
            [box[0] + box[3] / 2, box[1] + box[4] / 2, box[2] + box[5] / 2],
        ])
        
        # Transform corners to camera coordinates
        corners_cam = transform_lidar_to_camera(corners, Tr_velo_to_cam)
        
        # Project corners to 2D image plane
        corners_2d = project_to_image(corners_cam, P2)
        
        # Get the min and max coordinates for 2D bounding box
        x_min, y_min = np.min(corners_2d, axis=0)
        x_max, y_max = np.max(corners_2d, axis=0)
        
        boxes_2d.append([x_min, y_min, x_max, y_max])
    
    return torch.tensor(boxes_2d, dtype=torch.float32)