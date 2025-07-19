DATA_PATH = "/Windows/training_data/paths.json"
MAX_SIZE = 50000
INPUT_IMG_PATH = 0
OUTPUT_IMG_PATH = 1
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
class SingleLetterDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, device="cpu"):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
    def resnet_normalize(self, imgs: Tensor) -> Tensor:
        """Normalize the image tensor using ResNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        return (imgs - mean) / std
    def __iter__(self):
        data = self.dataset
        if self.shuffle:
            np.random.shuffle(data)
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            # MAKE SURE YOU REMOVE RESIZING
            input_images = [np.array(Image.open(item[INPUT_IMG_PATH]).convert("RGB").resize((128, 128)))/255.0 for item in batch_data]
            output_images = [np.array(Image.open(item[OUTPUT_IMG_PATH]).convert("RGB").resize((32, 32)))/255.0 for item in batch_data]
            # Ensure images are identical in shape
            if len(input_images) == 0:
                continue
            input_img_shape = input_images[0].shape
            for img in input_images:
                if img.shape != input_img_shape:
                    raise ValueError(f"Image shape mismatch: expected {input_img_shape}, got {img.shape}")
            output_img_shape = output_images[0].shape
            for img in output_images:
                if img.shape != output_img_shape:
                    raise ValueError(f"Image shape mismatch: expected {output_img_shape}, got {img.shape}")
            # Convert images to numpy arrays and transpose to (C, H, W)
            input_images = np.array([np.transpose(img, (2, 0, 1)) for img in input_images])
            output_images = np.array([np.transpose(img, (2, 0, 1)) for img in output_images])
            # Convert images to tensor
            input_images = torch.tensor(input_images, dtype=torch.float32)
            output_images = torch.tensor(output_images, dtype=torch.float32)
            input_images = input_images.to(self.device)
            output_images = output_images.to(self.device)
            # Normalize images
            #input_images = self.resnet_normalize(input_images)
            #output_images = self.resnet_normalize(output_images)
            yield input_images, output_images
