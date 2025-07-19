# Model that takes a 256x256 image and predicts a letter out of 24 options, plus a "no letter" option using pytorch.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class SingleLetterModel(nn.Module):
    def __init__(self):
        super(SingleLetterModel, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),             # Final output, channels=3
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
