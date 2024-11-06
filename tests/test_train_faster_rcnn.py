import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.train_faster_rcnn import save_model


@pytest.fixture
def model():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = 4  # 3 classes (Van, Cyclist, Pedestrian) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def test_model_setup(model):
    assert isinstance(model, torch.nn.Module)
    assert model.roi_heads.box_predictor.cls_score.out_features == 4

def test_save_model(model):
    save_model(model, 'test_model.pth')
    assert os.path.exists('test_model.pth')
    os.remove('test_model.pth')