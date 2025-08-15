import json
import random
from pathlib import Path
from typing import List, Tuple
from functools import partial

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import settings

DATA_PATH = settings.data_path
MAX_SIZE = settings.max_size

class SegData(Dataset):
    """Segmentation dataset for single letters, grayscale input only."""

    def __init__(self, level=0, data_path=DATA_PATH, input_size=(128, 128), output_size=(32, 32)):
        self.data_path = Path(data_path)
        self.input_size = input_size
        self.output_size = output_size
        self.dataset = self._load_dataset(level)

    def _load_dataset(self, level=0) -> List[List]:
        with open(self.data_path, "r") as f:
            all_data = json.load(f)["paths"]

        # Filter by level
        if isinstance(level, list):
            filtered = [item for item in all_data if int(item[3]) in level]
        else:
            filtered = [item for item in all_data if int(item[3]) == level]

        # Limit dataset size
        if len(filtered) > MAX_SIZE:
            idx = np.random.choice(len(filtered), MAX_SIZE, replace=False)
            filtered = [filtered[i] for i in idx]

        return filtered

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, int]:
        item = self.dataset[idx]

        # Load input image
        input_img = Image.open(item[1]).convert("L")  # grayscale
        input_img = input_img.resize(self.input_size, Image.Resampling.BILINEAR)
        input_array = np.array(input_img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(input_array).unsqueeze(0)  # (1, H, W)

        # Load mask
        output_img = Image.open(item[0]).convert("L")
        output_img = output_img.resize(self.output_size, Image.Resampling.BILINEAR)
        mask_array = np.array(output_img, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy((mask_array > mask_array.mean()).astype(np.float32)).unsqueeze(0)

        # Data augmentation: random rotation and horizontal flip
        if random.random() < 0.5:
            input_tensor = torch.flip(input_tensor, dims=[2])
            mask_tensor = torch.flip(mask_tensor, dims=[2])
        rot_steps = random.choice([0, 1, 2, 3])
        if rot_steps > 0:
            input_tensor = torch.rot90(input_tensor, k=rot_steps, dims=[1, 2])
            mask_tensor = torch.rot90(mask_tensor, k=rot_steps, dims=[1, 2])

        return input_tensor, mask_tensor

def collate_fn(batch, device="cuda"):
    inputs, masks = zip(*batch)
    inputs = torch.stack(inputs).to(device)
    masks = torch.stack(masks).to(device)
    return inputs, masks

def create_loader(dataset, batch_size=32, shuffle=True, device=settings.device, num_workers=settings.num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, device=device),
        pin_memory=False
    )

    return loader
