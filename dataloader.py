import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class Yolo8Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, img_ext=".jpg"):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_ext = img_ext

        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.endswith(img_ext) and os.path.isfile(os.path.join(label_dir, f.replace(img_ext, ".txt")))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(self.img_ext, ".txt"))

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        boxes = []
        labels = []

        try:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts)
                    # Convert YOLO format (normalized) to absolute [xmin, ymin, xmax, ymax]
                    xmin = (x_center - width / 2) * w
                    ymin = (y_center - height / 2) * h
                    xmax = (x_center + width / 2) * w
                    ymax = (y_center + height / 2) * h
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(class_id))
        except FileNotFoundError:
            print(f"Warning: Label file not found for {img_name}")
            boxes = []
            labels = []

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transform:
            image = self.transform(image)

        return image, target
