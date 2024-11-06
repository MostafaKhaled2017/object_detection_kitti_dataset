import pytest
import torch
import numpy as np
from pathlib import Path
import os
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.fusion_algorithm import (
    load_faster_rcnn_model,
    load_second_model,
    detect_objects_faster_rcnn,
    detect_objects_second,
    fuse_detections
)

@pytest.fixture
def faster_rcnn_model():
    checkpoint_path = 'models/faster_rcnn_model.pth'
    return load_faster_rcnn_model(checkpoint_path)

@pytest.fixture
def second_model_and_dataset():
    cfg_file = 'cfgs/kitti_models/second.yaml'
    checkpoint_path = 'models/second_7862.pth'
    data_path = 'data/Kitti_00/velodyne/0000/processed/'
    return load_second_model(cfg_file, checkpoint_path, data_path)

def test_load_faster_rcnn_model(faster_rcnn_model):
    assert faster_rcnn_model is not None
    assert isinstance(faster_rcnn_model, torch.nn.Module)

def test_load_second_model(second_model_and_dataset):
    model, dataset = second_model_and_dataset
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    assert len(dataset) > 0

def test_detect_objects_faster_rcnn(faster_rcnn_model):
    images = [np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)]
    detections = detect_objects_faster_rcnn(faster_rcnn_model, images)
    assert detections is not None
    assert len(detections) > 0

def test_detect_objects_second(second_model_and_dataset):
    model, dataset = second_model_and_dataset
    detections = detect_objects_second(model, dataset)
    assert detections is not None
    assert len(detections) > 0

def test_fuse_detections(faster_rcnn_model, second_model_and_dataset):
    images = [np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)]
    rgb_detections = detect_objects_faster_rcnn(faster_rcnn_model, images)
    model, dataset = second_model_and_dataset
    lidar_detections = detect_objects_second(model, dataset)
    calib_file = 'data/Kitti_00/calib/0000.txt'
    fused_detections = fuse_detections(rgb_detections, lidar_detections, calib_file)
    assert fused_detections is not None
    assert len(fused_detections) > 0

if __name__ == "__main__":
    pytest.main()