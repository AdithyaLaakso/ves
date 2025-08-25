import torch
import settings
from nystrom_attention import NystromAttention

import torch.nn as nn
class PatchEmbedConv(nn.Module):
    def __init__(self, in_channels=1, embed_size=800, patch_size=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, embed_size, kernel_size=patch_size, stride=patch_size)
        self.bn3 = nn.BatchNorm2d(embed_size)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        return x

class HybridVisionNystromformer(nn.Module):
    def __init__(
        self,
        num_landmarks=64,
        ff_dropout=0.1,
    ):
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
        num_patches = (self.image_size // self.patch_size) ** 2

        self.nystromformer = NystromAttention(
            dim=self.embed_size,
            dim_head=self.num_heads**2,
            heads=self.num_heads,
            num_landmarks=num_landmarks,
            dropout=self.dropout,
        )

        self.norm = nn.LayerNorm(self.embed_size)
        self.head = nn.Linear(self.embed_size, self.out_channels * self.output_size * self.output_size)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.patch_embed(x)  # (B, dim, H', W')
        x = self.nystromformer(x)
        x = self.norm(x)
        x = self.head(x)
        x = self.activation(x)
        x = x.view(-1, self.out_channels, self.output_size, self.output_size)
        return x

def build_model():
    model = HybridVisionNystromformer(
        num_landmarks=64,
        ff_dropout=0.1,
    )
    return model