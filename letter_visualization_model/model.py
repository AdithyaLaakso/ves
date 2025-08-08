import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

class DoubleConv(nn.Module):
    """Double convolution block used in U-Net"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv - FIXED VERSION"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # FIXED: The conv layer should handle concatenated channels correctly
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the upsampled tensor from deeper layer
        # x2 is the skip connection from encoder

        # Upsample x1
        x1 = self.up(x1)

        # Handle size differences due to padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection - this creates the channel dimension mismatch!
        # x2 (skip) + x1 (upsampled) = concatenated tensor
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UNetSegmentation(nn.Module):
    """FIXED U-Net for segmentation: 128x128x8 -> 32x32x1"""
    def __init__(self, n_channels=8, n_classes=1, bilinear=True):
        super(UNetSegmentation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)          # 128x128x8 -> 128x128x64
        self.down1 = Down(64, 128)                     # 128x128x64 -> 64x64x128
        self.down2 = Down(128, 256)                    # 64x64x128 -> 32x32x256
        self.down3 = Down(256, 512)                    # 32x32x256 -> 16x16x512

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)         # 16x16x512 -> 8x8x512

        # Decoder (upsampling path) - FIXED CHANNEL CALCULATIONS
        # When concatenating skip connections, channels double!
        # up1: 512 (upsampled) + 512 (skip) = 1024 channels input
        self.up1 = Up(1024, 512 // factor, bilinear)  # 8x8x512 + skip -> 16x16x256

        # up2: 256 (upsampled) + 256 (skip) = 512 channels input
        self.up2 = Up(512, 256 // factor, bilinear)   # 16x16x256 + skip -> 32x32x128

        # Final segmentation head - only operates at 32x32, no more upsampling
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1),  # Final 1x1 conv for segmentation
            nn.Sigmoid()  # Binary segmentation activation
        )

    def forward(self, x):
        # Encoder path with skip connections
        x1 = self.inc(x)      # 128x128x64
        x2 = self.down1(x1)   # 64x64x128
        x3 = self.down2(x2)   # 32x32x256
        x4 = self.down3(x3)   # 16x16x512
        x5 = self.down4(x4)   # 8x8x512
        # Decoder path with skip connections
        # up1: x5 (8x8x512) upsampled and concatenated with x4 (16x16x512) = 16x16x1024 -> 16x16x256
        x = self.up1(x5, x4)
        # up2: x (16x16x256) upsampled and concatenated with x3 (32x32x256) = 32x32x512 -> 32x32x128
        x = self.up2(x, x3)
        # Final segmentation - no more skip connections
        x = self.final_conv(x)      # 32x32x128 -> 32x32x1
        return x

class FixedUNetSegmentation(nn.Module):
    """Alternative fixed U-Net with explicit channel handling"""
    def __init__(self, n_channels=8, n_classes=1):
        super(FixedUNetSegmentation, self).__init__()

        # Encoder
        self.enc1 = DoubleConv(n_channels, 64)     # 128x128
        self.pool1 = nn.MaxPool2d(2)               # 64x64

        self.enc2 = DoubleConv(64, 128)            # 64x64
        self.pool2 = nn.MaxPool2d(2)               # 32x32

        self.enc3 = DoubleConv(128, 256)           # 32x32
        self.pool3 = nn.MaxPool2d(2)               # 16x16

        self.enc4 = DoubleConv(256, 512)           # 16x16
        self.pool4 = nn.MaxPool2d(2)               # 8x8

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)    # 8x8

        # Decoder - only go up to 32x32
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # 8x8 -> 16x16
        self.dec1 = DoubleConv(1024, 512)  # 512 (up) + 512 (skip) = 1024

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)   # 16x16 -> 32x32
        self.dec2 = DoubleConv(512, 256)   # 256 (up) + 256 (skip) = 512

        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # 128x128x64
        p1 = self.pool1(e1)        # 64x64x64

        e2 = self.enc2(p1)         # 64x64x128
        p2 = self.pool2(e2)        # 32x32x128

        e3 = self.enc3(p2)         # 32x32x256
        p3 = self.pool3(e3)        # 16x16x256

        e4 = self.enc4(p3)         # 16x16x512
        p4 = self.pool4(e4)        # 8x8x512

        # Bottleneck
        b = self.bottleneck(p4)    # 8x8x1024

        # Decoder with skip connections
        u1 = self.up1(b)           # 8x8x1024 -> 16x16x512

        # Concatenate skip connection
        u1 = torch.cat([u1, e4], dim=1)  # 16x16x512 + 16x16x512 = 16x16x1024
        u1 = self.dec1(u1)         # 16x16x1024 -> 16x16x512

        u2 = self.up2(u1)          # 16x16x512 -> 32x32x256

        # Concatenate skip connection
        u2 = torch.cat([u2, e3], dim=1)  # 32x32x256 + 32x32x256 = 32x32x512
        u2 = self.dec2(u2)         # 32x32x512 -> 32x32x256

        # Final output
        output = self.final(u2)    # 32x32x256 -> 32x32x1

        return output


class SimpleSegmentationCNN(nn.Module):
    """Simpler CNN for segmentation: 128x128x8 -> 32x32x1 - WORKING VERSION"""
    def __init__(self, input_channels=8, output_channels=1):
        super(SimpleSegmentationCNN, self).__init__()

        self.features = nn.Sequential(
            # First block: 128x128x8 -> 64x64x32
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64

            # Second block: 64x64x32 -> 32x32x64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            # Third block: 32x32x64 -> 32x32x128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, 1),  # 1x1 conv for final prediction
            nn.Sigmoid()  # Binary segmentation
        )

    def forward(self, x):
        features = self.features(x)
        segmentation = self.segmentation_head(features)
        return segmentation


# Utility function for proper weight initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# FIXED Drop-in replacement for your original ReconstructionModel
class ReconstructionModel(nn.Module):
    """FIXED model for segmentation: 128x128x8 -> 32x32x1"""
    def __init__(self):
        super(ReconstructionModel, self).__init__()

        self.input_channels = 8
        self.output_classes = 1

        print(f"Initializing ReconstructionModel with {self.input_channels} input channels -> {self.output_classes} output channels")

        # Choose architecture - USING WORKING SIMPLE CNN FOR NOW
        # Option 1: Simple CNN (guaranteed to work)
        # self.model = SimpleSegmentationCNN(input_channels, output_channels)

        # Option 2: Fixed U-Net (use this after testing simple CNN works)
        self.model = FixedUNetSegmentation(n_channels=self.input_channels, n_classes=self.output_classes)

        # Option 3: Original U-Net with debug prints (for troubleshooting)
        # self.model = UNetSegmentation(n_channels=input_channels, n_classes=output_channels)

        # Initialize weights properly
        self.model.apply(init_weights)
        print("Model initialized successfully!")

    def forward(self, x, i=None):
        """Forward pass - matches your original interface"""

        # Debug input shape
        # print(f"ReconstructionModel received input with shape: {x.shape}")
        # print(f"Expected input channels: {self.input_channels}")

        # Handle input channel mismatch
        if x.shape[1] != self.input_channels:
            print(f"WARNING: Input has {x.shape[1]} channels, but model expects {self.input_channels}")

            if x.shape[1] == 3 and self.input_channels == 8:
                print("Converting 3-channel input to 8-channel by padding with zeros")
                padding = torch.zeros(x.shape[0], 5, x.shape[2], x.shape[3],
                                    device=x.device, dtype=x.dtype)
                self.output_channels = 1
                x = torch.cat([x, padding], dim=1)
            elif x.shape[1] == 1 and self.input_channels == 8:
                print("Converting 1-channel input to 8-channel by replication")
                x = x.repeat(1, 8, 1, 1)
            elif x.shape[1] > self.input_channels:
                print(f"Truncating {x.shape[1]} channels to {self.input_channels}")
                x = x[:, :self.input_channels]
            else:
                raise ValueError(f"Cannot handle input with {x.shape[1]} channels for model expecting {self.input_channels}")

        # Check input spatial dimensions
        if x.shape[2] != 128 or x.shape[3] != 128:
            print(f"WARNING: Input spatial size is {x.shape[2]}x{x.shape[3]}, but model expects 128x128")
            print("Resizing input to 128x128")
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        # Debug code
        if i is not None and i != -1:
            os.makedirs("debug_images", exist_ok=True)
            self.save_tensor_as_image(x[:, :1], f"debug_images/input_image{i}.png", "Input")
            output = self.model(x)
            self.save_tensor_as_image(output, f"debug_images/output_segmentation{i}.png", "Segmentation")
            return output
        else:
            return self.model(x)

    def save_tensor_as_image(self, tensor, filepath, description=""):
        """Save a tensor as an image file"""
        try:
            if len(tensor.shape) == 4:
                img_tensor = tensor[0]
            else:
                img_tensor = tensor

            img_tensor = img_tensor.detach().cpu()
            img_tensor = torch.clamp(img_tensor, 0, 1)

            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)
                img = transforms.ToPILImage()(img_tensor)
            else:
                img_tensor = img_tensor[0:1]
                img = transforms.ToPILImage()(img_tensor)

            img.save(filepath)
            print(f"{description} image saved to: {filepath}")
            print(f"{description} tensor shape: {tensor.shape}, min: {tensor.min():.4f}, max: {tensor.max():.4f}")

        except Exception as e:
            print(f"Error saving {description} image: {e}")
            print(f"Tensor shape: {tensor.shape}, min: {tensor.min()}, max: {tensor.max()}")

    def getFCParams(self):
        """Get parameters of final layers (for compatibility)"""
        final_params = []
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in ['final', 'segmentation', 'head', 'outc']):
                final_params.append(param)
        return final_params if final_params else list(self.model.parameters())

    def getPretrainedParams(self):
        """Get parameters of earlier layers (for compatibility)"""
        pretrained_params = []
        final_param_ids = {id(p) for p in self.getFCParams()}
        for param in self.model.parameters():
            if id(param) not in final_param_ids:
                pretrained_params.append(param)
        return pretrained_params


# Test the fixed model
if __name__ == "__main__":
    print("Testing FIXED ReconstructionModel...")

    # Test with correct 8-channel input
    print("\n=== Test: 8-channel input ===")
    model = ReconstructionModel(input_channels=8, output_channels=1)
    x = torch.randn(2, 8, 128, 128)

    try:
        with torch.no_grad():
            output = model(x)
        print(f"✓ SUCCESS! Input: {x.shape} -> Output: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\nIf this test passes, your model architecture is now fixed!")
    print("Next steps:")
    print("1. Replace your model.py with this fixed version")
    print("2. Test with your actual dataloader")
    print("3. If simple CNN works, you can try the FixedUNetSegmentation")
