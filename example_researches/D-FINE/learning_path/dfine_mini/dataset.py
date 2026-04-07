"""COCO Dataset loader for training."""
import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class COCODataset(Dataset):
    """Load COCO dataset with YOLO-format labels."""
    
    def __init__(self, data_dir: str, split: str = "train2017"):
        """
        Args:
            data_dir: Path to extracted dataset directory containing coco_minitrain_25k/
            split: "train2017" or "val2017"
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Construct paths
        self.dataset_root = self.data_dir / "coco_minitrain_25k"
        self.images_dir = self.dataset_root / "images" / split
        self.labels_dir = self.dataset_root / "labels" / split
        self.list_file = self.dataset_root / f"{split}.txt"
        
        # Read list of images
        if not self.list_file.exists():
            raise FileNotFoundError(f"List file not found: {self.list_file}")
        
        with open(self.list_file) as f:
            # Paths are relative like "./images/train2017/000000312352.jpg"
            # Extract just the image ID
            lines = f.read().strip().split('\n')
            self.image_ids = [Path(line).stem for line in lines]
        
        print(f"Loaded {len(self.image_ids)} images from {split}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx: int):
        """Return (image, boxes, labels) where:
        - image: (3, H, W) tensor
        - boxes: (N, 4) in CXCYWH normalized format
        - labels: (N,) class IDs
        """
        img_id = self.image_ids[idx]
        
        # Load image
        img_path = self.images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            # Try PNG
            img_path = self.images_dir / f"{img_id}.png"
        
        image = Image.open(img_path).convert("RGB")
        
        # Resize to 256x256 to match model input
        image = image.resize((256, 256), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Load labels
        label_path = self.labels_dir / f"{img_id}.txt"
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append([cx, cy, w, h])
                        labels.append(class_id)
        
        if len(boxes) == 0:
            # Empty image - create dummy entry
            boxes = torch.zeros(0, 4)
            labels = torch.zeros(0, dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
        return image, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    """Collate batch of variable-sized annotations."""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images (they're all 256x256 so this works)
    images = torch.stack(images)
    
    return images, targets
