import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import sys
from torch.utils.data.dataloader import default_collate
import argparse

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import KittiDataset, get_transform

def collate_fn(batch):
    imgs, targets = zip(*batch)  # Unpack a list of tuples into separate tuples of images and targets
    imgs = list(img for img in imgs if img is not None)
    targets = [{k: torch.tensor(v) for k, v in t.items()} for t in targets if t is not None]
    return imgs, targets

def save_model(model, filepath):
    """Saves only the model (without optimizer or epoch information)."""
    torch.save(model.state_dict(), filepath)

def evaluate(model, data_loader, device):
    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)

            total_loss += sum(loss for loss in loss_dict.values())

    return total_loss / len(data_loader)

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs):
    print(f'Training Epoch [{epoch+1}/{num_epochs}]')

    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

def main():
    parser = argparse.ArgumentParser(description="Train an object detection model on KITTI dataset.")
    parser.add_argument('--images_directory', type=str, default='data/Kitti_00/image_02/0000', help='Path to the directory containing image files.')
    parser.add_argument('--annotations_file', type=str, default='data/Kitti_00/labels_02/0000.txt', help='Path to the annotations file.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training.')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = KittiDataset(args.images_directory, args.annotations_file, get_transform())

    # Split dataset into training and evaluation
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    eval_data_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Load the model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = 4  # 3 classes (Van, Cyclist, Pedestrian) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    for epoch in range(args.num_epochs):
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, args.num_epochs)
        eval_loss = evaluate(model, eval_data_loader, device)
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Evaluation Loss: {eval_loss}')

    # Save the final model
    save_model(model, 'models/pytorch_detection_model.pth')

if __name__ == '__main__':
    main()