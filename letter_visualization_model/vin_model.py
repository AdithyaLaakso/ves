import torch
import settings
from nystrom_attention import NystromAttention

import torch.nn as nn
class PatchEmbedConv(nn.Module):
    def __init__(self, in_channels=1, embed_size=800, patch_size=4):
        super().__init__()
        # Use deeper conv stack for richer features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_size, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_size),
            nn.ReLU(inplace=True),
        )
        self.patch_size = patch_size

    def forward(self, x):
        x = self.conv_layers(x)
        # Flatten spatial dimensions to patches
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_size)
        return x

class HybridVisionNystromformer(nn.Module):
    def __init__(
        self,
        num_landmarks=64,
    ):
        self.mode = settings.mode
        self.output_size = getattr(settings, "output_size", settings.image_size)
        self.in_channels = settings.in_channels
        self.out_channels = settings.out_channels
        self.output_size = settings.output_size
        self.image_size = settings.image_size
        self.embed_size = settings.embed_size
        self.num_blocks = settings.num_blocks
        self.num_heads = settings.num_heads
        self.dropout = settings.dropout
        self.patch_size = settings.patch_size
        self.patch_sizes = getattr(settings, "patch_sizes", [settings.patch_size])  # e.g., [8,16,32]
        super().__init__()
        self.patch_embed = PatchEmbedConv(
            in_channels=self.in_channels,
            embed_size=self.embed_size,
            patch_size=self.patch_size,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.nystromformer = NystromAttention(
            dim=self.embed_size,
            dim_head=self.num_heads**2,
            heads=self.num_heads,
            num_landmarks=num_landmarks,
            dropout=self.dropout,
        )

        self.norm = nn.LayerNorm(self.embed_size)
        self.head = nn.Linear(self.embed_size, (self.output_size * self.output_size) // (self.num_patches))
        self.classifier = nn.Linear(self.embed_size * self.num_patches, settings.num_letters)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.patch_embed(x)  # (B, num_patches, dim)
        x = self.nystromformer(x) # (B, num_patches, dim)
        x = self.norm(x) # (B, num_patches, dim)
        if self.mode == settings.RECONSTRUCTING:
            x = self.head(x) # (B, num_patches, out_channels * output_size * output_size)
            x = self.activation(x)
            x = x.view(-1, self.out_channels, self.output_size, self.output_size) # (B, out_channels, H, W)
        if self.mode == settings.CLASSIFYING:
            x = x.flatten(1)
            x = self.classifier(x)
            x = torch.softmax(x, dim=1)
        return x

def build_model():
    model = HybridVisionNystromformer(
        num_landmarks=64,
    )
    return model
