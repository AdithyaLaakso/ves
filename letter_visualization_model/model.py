# Model that takes a 256x256 image and predicts a letter out of 24 options, plus a "no letter" option using pytorch.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class SingleLetterModel(nn.Module):
    def __init__(self):
        super(SingleLetterModel, self).__init__()
        # Encoder
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64

            # Output image of shape (C, 32, 32)
            nn.Conv2d(128, 3, kernel_size=1),  # Output 3 channels (RGB)
            nn.Sigmoid()  # If you want output in [0,1] range
        )
    def forward(self, x):
        return self.model(x)
