import json
import random
from pathlib import Path
from typing import List, Tuple, Union
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

    def __init__(self, level: Union[int, List[int]]=0, data_path=DATA_PATH, input_size=(128, 128), output_size=(32, 32)):
        self.data_path = Path(data_path)
        self.input_size = input_size
        self.output_size = output_size
        self.dataset = self._load_dataset(level)

    def _load_dataset(self, level=0) -> List[List]:
        print(level)
        with open(self.data_path, "r") as f:
            all_data = json.load(f)["paths"]

        # Limit dataset size
        if MAX_SIZE is not None:
            if len(all_data) > MAX_SIZE:
                idx = np.random.choice(len(all_data), MAX_SIZE, replace=False)
                filtered = [all_data[i] for i in idx]
            else:
                filtered = all_data
        else:
            filtered = all_data

        # Filter by level
        if settings.track_levels:
            if isinstance(level, list):
                filtered = [item for item in all_data if int(item[3]) in level]
            else:
                filtered = [item for item in all_data if int(item[3]) == level]
        else:
            filtered = [item for item in all_data]

        filtered = [item for item in filtered if item[2] in settings.letters]

        # Filter out missing files
        filtered = [item for item in filtered if Path(settings.add_to_path + item[0]).exists() and Path(settings.add_to_path + item[1]).exists()]

        return filtered

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        item = self.dataset[idx]

        # Load input image
        input_img = Image.open(settings.add_to_path + item[0]).convert("L")  # grayscale
        input_img = input_img.resize(self.input_size, Image.Resampling.BILINEAR)
        input_array = np.array(input_img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(input_array).unsqueeze(0)  # (1, H, W)

        # Load mask
        output_img = Image.open(settings.add_to_path + item[1]).convert("L")
        output_img = output_img.resize(self.output_size, Image.Resampling.BILINEAR)
        mask_array = np.array(output_img, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy((mask_array > mask_array.mean()).astype(np.float32)).unsqueeze(0)

        # Data augmentation: random rotation
        rot_steps = random.choice([0, 1, 2, 3])
        if rot_steps > 0:
            input_tensor = torch.rot90(input_tensor, k=rot_steps, dims=[1, 2])
            mask_tensor = torch.rot90(mask_tensor, k=rot_steps, dims=[1, 2])

        if random.random() > 0.5:
            input_tensor = 1 - input_tensor

        flip_type = random.choice([0, 1, 2, 3, 4])
        if flip_type == 1:
            input_tensor = torch.flip(input_tensor, dims=(0,))
            mask_tensor = torch.flip(mask_tensor, dims=(0,))
        elif flip_type == 2:
            input_tensor = torch.flip(input_tensor, dims=(1,))
            mask_tensor = torch.flip(mask_tensor, dims=(1,))
        elif flip_type == 3:
            input_tensor = torch.flip(input_tensor, dims=[0, 1])
            mask_tensor = torch.flip(mask_tensor, dims=[0, 1])
        elif flip_type == 4:
            input_tensor = torch.flip(input_tensor, dims=[1, 0])
            mask_tensor = torch.flip(mask_tensor, dims=[1, 0])

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
        pin_memory=False,
        persistent_workers=True
    )

    return loader
