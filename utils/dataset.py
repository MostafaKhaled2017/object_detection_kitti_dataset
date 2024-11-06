import os
from PIL import Image
import torch
from torchvision import transforms

# Custom dataset class for the Kitti dataset
class KittiDataset(torch.utils.data.Dataset):
    LABEL_MAP = {
        'DontCare': 0,
        'Van': 1,
        'Cyclist': 2,
        'Pedestrian': 3
    }

    def __init__(self, root, annotations_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(root)))
        self.labels = {}
        with open(annotations_file) as f:
            for line in f:
                elements = line.split()
                frame_number = int(elements[0])
                obj_type = elements[2]
                if obj_type in self.LABEL_MAP:
                    if frame_number not in self.labels:
                        self.labels[frame_number] = []
                    bbox = [float(elements[6]), float(elements[7]), float(elements[8]), float(elements[9])]
                    self.labels[frame_number].append({'bbox': bbox, 'label': self.LABEL_MAP[obj_type]})

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        frame_number = int(self.imgs[idx].split('.')[0])
        boxes = []
        labels = []

        if frame_number in self.labels:
            for obj in self.labels[frame_number]:
                boxes.append(obj['bbox'])
                labels.append(obj['label'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # Convert the PIL Image to a tensor.
    ])
