DATA_PATH = "./paths.json"
MAX_SIZE = 50000
IMG_PATH = 0
LABEL = 1
import json
import numpy as np
import torch
from torch import Tensor
from PIL import Image
# TODO: Change sizes to 224x224
class SingleLetterDataset:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.dataset = self.load_dataset()
        self.class_to_index = self.innit_class_name_to_index()
    def load_dataset(self):
        """Load the dataset from the JSON file."""
        with open(self.data_path, "r") as f:
            all_data = json.load(f)['paths']
            if len(all_data) > MAX_SIZE:
                indices = np.random.choice(len(all_data), MAX_SIZE, replace=False)
                data = [all_data[i] for i in indices]
            else:
                data = all_data
        return data
    def get_class_names(self):
        labels = [item[LABEL] for item in self.dataset]
        return list(set(labels))
    def innit_class_name_to_index(self):
        class_names = self.get_class_names()
        return {name: index for index, name in enumerate(class_names)}
class SingleLetterDataLoader:
    def __init__(self, dataset, class_to_index, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.class_to_index = class_to_index
        self.batch_size = batch_size
        self.shuffle = shuffle
    def resnet_normalize(self, imgs: Tensor) -> Tensor:
        """Normalize the image tensor using ResNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        return (imgs - mean) / std
    def get_class_name_to_index(self):
        """Create a mapping from class names to indices."""
        class_names = self.dataset.get_class_names()
        return {name: index for index, name in enumerate(class_names)}
    def label_to_index(self, label):
        if not hasattr(self, 'class_to_index'):
            self.class_to_index = self.get_class_name_to_index()
        return self.class_to_index.get(label, -1)
    def __iter__(self):
        data = self.dataset
        if self.shuffle:
            np.random.shuffle(data)
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            imgs = [np.array(Image.open(item[IMG_PATH]).convert("RGB"))/255.0 for item in batch_data]
            # Ensure images are identical in shape
            if len(imgs) == 0:
                continue
            img_shape = imgs[0].shape
            for img in imgs:
                if img.shape != img_shape:
                    raise ValueError(f"Image shape mismatch: expected {img_shape}, got {img.shape}")
            labels = [self.class_to_index[item[LABEL]] for item in batch_data]
            # reshape images to (batch_size, channels, height, width)
            imgs = np.array([np.transpose(img, (2, 0, 1)) for img in imgs])
            # Convert images to tensor
            imgs = torch.tensor(imgs, dtype=torch.float32)
            # Normalize images
            imgs = self.resnet_normalize(imgs)
            # Ensure labels are in tensor format
            labels = torch.tensor(labels, dtype=torch.long)
            yield imgs, labels
    
    
