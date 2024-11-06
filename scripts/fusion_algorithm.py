import sys
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.ops as ops
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import numpy as np
import cv2
import yaml
from pathlib import Path
import glob
import argparse

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import *
from utils.demo_dataset import DemoDataset

def load_faster_rcnn_model(checkpoint_path):
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=4)
    state_dict = torch.load(checkpoint_path)
    
    # Filter out the unexpected keys
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("0.0.", "0.").replace("1.0.", "1.").replace("2.0.", "2.").replace("3.0.", "3.")
        new_state_dict[new_key] = state_dict[key]
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def load_second_model(cfg_file, checkpoint_path, data_path, ext='.bin'):
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger()
    point_cloud_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_path), ext=ext, logger=logger
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=point_cloud_dataset)
    model.load_params_from_file(filename=checkpoint_path, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    return model, point_cloud_dataset

def detect_objects_faster_rcnn(model, images, confidence_threshold=0.01):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    transformed_images = [transform(image) for image in images]
    with torch.no_grad():
        detections = model(transformed_images)
    
    filtered_detections = []
    for det in detections:
        high_confidence_indices = det['scores'] > confidence_threshold
        filtered_det = {
            'boxes': det['boxes'][high_confidence_indices],
            'labels': det['labels'][high_confidence_indices],
            'scores': det['scores'][high_confidence_indices]
        }
        filtered_detections.append(filtered_det)
    
    return filtered_detections

def detect_objects_second(model, dataset, confidence_threshold=0.2):
    with torch.no_grad():
        detections = []
        for idx, data_dict in enumerate(dataset):
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            filtered_pred_dicts = []
            for pred in pred_dicts:
                high_confidence_indices = pred['pred_scores'] > confidence_threshold
                filtered_pred = {
                    'pred_boxes': pred['pred_boxes'][high_confidence_indices],
                    'pred_labels': pred['pred_labels'][high_confidence_indices],
                    'pred_scores': pred['pred_scores'][high_confidence_indices]
                }
                filtered_pred_dicts.append(filtered_pred)
            detections.append(filtered_pred_dicts)
        return detections

def fuse_detections(rgb_detections, lidar_detections, calib_file, device='cuda:0', weight_rgb=0.5, weight_lidar=0.5, iou_threshold=0.5):
    print(f"RGB Detections: {rgb_detections}")
    print(f"Point cloud Detections: {lidar_detections}")

    fused_detections = []
    for rgb_det, lidar_det_list in zip(rgb_detections, lidar_detections):
        rgb_boxes = rgb_det['boxes'].to(device)
        rgb_labels = rgb_det['labels'].to(device)
        rgb_scores = rgb_det['scores'].to(device)
        for lidar_det in lidar_det_list:
            lidar_boxes_3d = lidar_det['pred_boxes'].cpu().numpy()
            lidar_boxes = project_lidar_to_2d(lidar_boxes_3d, calib_file).to(device)
            lidar_labels = lidar_det['pred_labels'].to(device)
            lidar_scores = lidar_det['pred_scores'].to(device)
            combined_boxes = torch.cat((rgb_boxes, lidar_boxes), dim=0)
            combined_labels = torch.cat((rgb_labels, lidar_labels), dim=0)
            combined_scores = torch.cat((rgb_scores * weight_rgb, lidar_scores * weight_lidar), dim=0)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Late Fusion for RGB and LiDAR detections.")
    parser.add_argument('--image_path', type=str, default='data/Kitti_00/image_02/0000/000000.png', help='Path to the image file.')
    parser.add_argument('--point_cloud_path', type=str, default='data/Kitti_00/velodyne/0000/processed/000000.bin', help='Path to the point cloud file.')
    args = parser.parse_args()

    faster_rcnn_checkpoint_path = 'models/faster_rcnn_model.pth'
    second_cfg_file = 'cfgs/kitti_models/second.yaml'
    second_checkpoint_path = 'models/second_7862.pth'
    calib_file = 'data/Kitti_00/calib/0000.txt'

    images = load_images(args.image_path)
    faster_rcnn_model = load_faster_rcnn_model(faster_rcnn_checkpoint_path)
    second_model, point_cloud_dataset = load_second_model(second_cfg_file, second_checkpoint_path, args.point_cloud_path)

    rgb_detections = detect_objects_faster_rcnn(faster_rcnn_model, images)
    lidar_detections = detect_objects_second(second_model, point_cloud_dataset)

    fused_detections = fuse_detections(rgb_detections, lidar_detections, calib_file)
    visualize_detections(images, fused_detections)