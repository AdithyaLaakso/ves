DATA_PATH = "./paths.json"
IMG_PATH = 0
LABEL = 1
import json
import numpy as np
import torch
from torch import Tensor
from PIL import Image
# TODO: Normalize images with resnet logic
# TODO: Change sizes to 224x224
class SingleLetterDataLoader:
    def __init__(self, data_path=DATA_PATH, batch_size=32, shuffle=True):
        self.data_path = data_path
        self.class_to_index = self.innit_class_name_to_index()
        self.data_size = self.get_data_size()
        self.batch_size = batch_size
        self.shuffle = shuffle
    def get_data_size(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)['paths']
        return len(data)
    def get_class_names(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)['paths'][:1000]
        labels = [item[LABEL] for item in data]
        return list(set(labels))
    def innit_class_name_to_index(self):
        class_names = self.get_class_names()
        return {name: index for index, name in enumerate(class_names)}
    def label_to_index(self, label):
        if not hasattr(self, 'class_to_index'):
            self.class_to_index = self.get_class_name_to_index()
        return self.class_to_index.get(label, -1)
    def __iter__(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)['paths'][:1000]
        if self.shuffle:
            np.random.shuffle(data)
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            imgs = [np.array(Image.open(item[IMG_PATH]).convert("RGB")) for item in batch_data]
            # Ensure images are identical in shape
            if len(imgs) == 0:
                continue
            img_shape = imgs[0].shape
            for img in imgs:
                if img.shape != img_shape:
                    raise ValueError(f"Image shape mismatch: expected {img_shape}, got {img.shape}")
            labels = [self.class_to_index[item[LABEL]] for item in batch_data]
            # reshape images to (batch_size, channels, height, width)
            imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]
            yield torch.tensor(imgs, dtype=torch.float32), torch.tensor(labels, dtype = torch.long)
