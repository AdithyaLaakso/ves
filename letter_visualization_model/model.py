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
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size differences due to padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetReconstruction(nn.Module):
    """U-Net style encoder-decoder for image reconstruction"""
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(UNetReconstruction, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder (upsampling path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, 1)

        # Optional: Use tanh activation for outputs in [-1, 1] range
        # or sigmoid for [0, 1] range
        self.output_activation = nn.Sigmoid()  # Change to nn.Tanh() if needed

    def forward(self, x):
        # Encoder path with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output
        x = self.outc(x)
        x = self.output_activation(x)

        return x


class SimpleEncoderDecoder(nn.Module):
    """Simpler encoder-decoder without skip connections"""
    def __init__(self, input_channels=3, output_channels=3):
        super(SimpleEncoderDecoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),  # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 512 x 2 x 2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Input: 512 x 2 x 2
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, output_channels, 4, stride=2, padding=1),  # 3 x 32 x 32
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ResidualAutoencoder(nn.Module):
    """Autoencoder with residual connections for better training"""
    def __init__(self, input_channels=3, output_channels=3):
        super(ResidualAutoencoder, self).__init__()

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Encoder with residual blocks
        self.encoder = nn.Sequential(
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),  # Downsample
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),  # Downsample
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            ResidualBlock(256),
        )

        # Decoder with residual blocks
        self.decoder = nn.Sequential(
            ResidualBlock(256),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, output_channels, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.init_conv(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.final_conv(decoded)
        return output


# Utility function for proper weight initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# Drop-in replacement for your original ReconstructionModel
class ReconstructionModel(nn.Module):
    """Drop-in replacement that matches your original interface"""
    def __init__(self, pretrained_model=None, input_channels=3, output_channels=3):
        super(ReconstructionModel, self).__init__()

        # Ignore the pretrained_model parameter for now and use a proper architecture
        # You can choose which architecture to use here:

        # Option 1: Simple encoder-decoder (recommended to start)
#         self.model = SimpleEncoderDecoder(input_channels, output_channels)

        # Option 2: U-Net (better results but more complex)
        self.model = UNetReconstruction(n_channels=input_channels, n_classes=output_channels)

        # Option 3: Residual autoencoder (good for identity mapping)
        # self.model = ResidualAutoencoder(input_channels, output_channels)

        # Initialize weights properly
        self.model.apply(init_weights)

    def forward(self, x, i=None):
        """Forward pass - matches your original interface"""

        # Debug code (only runs if i != -1, matching your original logic)
        if i is not None and i != -1:
            # Create output directory if it doesn't exist
            os.makedirs("debug_images", exist_ok=True)

            # Save input image
            self.save_tensor_as_image(x, f"debug_images/input_image{i}.png", "Input")

            # Forward pass
            output = self.model(x)

            # Save output image
            self.save_tensor_as_image(output, f"debug_images/output_image{i}.png", "Output")

            return output
        else:
            # Normal forward pass without debugging
            return self.model(x)

    def save_tensor_as_image(self, tensor, filepath, description=""):
        """Save a tensor as an image file"""
        try:
            # Handle batch dimension - take first image if batch size > 1
            if len(tensor.shape) == 4:
                img_tensor = tensor[0]  # Take first image from batch
            else:
                img_tensor = tensor

            # Move to CPU and detach from computation graph
            img_tensor = img_tensor.detach().cpu()

            # Clamp to [0, 1] range (important for Sigmoid outputs)
            img_tensor = torch.clamp(img_tensor, 0, 1)

            # Convert to PIL Image
            if img_tensor.shape[0] == 1:  # Grayscale
                img_tensor = img_tensor.squeeze(0)
                img = transforms.ToPILImage()(img_tensor)
            elif img_tensor.shape[0] == 3:  # RGB
                img = transforms.ToPILImage()(img_tensor)
            else:
                # Handle other channel numbers by converting to grayscale
                img_tensor = torch.mean(img_tensor, dim=0)
                img = transforms.ToPILImage()(img_tensor.unsqueeze(0))

            # Save the image
            img.save(filepath)
            print(f"{description} image saved to: {filepath}")
            print(f"{description} tensor shape: {tensor.shape}, min: {tensor.min():.4f}, max: {tensor.max():.4f}")

        except Exception as e:
            print(f"Error saving {description} image: {e}")
            print(f"Tensor shape: {tensor.shape}, min: {tensor.min()}, max: {tensor.max()}")

    def getFCParams(self):
        """Get parameters of final layers (for compatibility)"""
        # Return parameters of the final convolutional layers
        final_params = []
        for name, param in self.model.named_parameters():
            if 'final' in name or 'outc' in name or 'decoder' in name:
                final_params.append(param)
        return final_params if final_params else list(self.model.parameters())

    def getPretrainedParams(self):
        """Get parameters of earlier layers (for compatibility)"""
        # Return parameters of encoder layers
        pretrained_params = []
        for name, param in self.model.named_parameters():
            if 'encoder' in name or 'inc' in name or 'down' in name:
                pretrained_params.append(param)
        return pretrained_params if pretrained_params else []


# Example usage and training tips:
if __name__ == "__main__":
    # This should now work as a drop-in replacement
    model = ReconstructionModel(pretrained_model=None)

    # Test with dummy data
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 RGB 32x32 images
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test debug mode
    output_debug = model(x, i=0)  # This should save debug images

    print("\nTraining recommendations:")
    print("- Optimizer: Adam with lr=1e-3")
    print("- Loss: MSELoss() or L1Loss()")
    print("- Batch size: 16-64 depending on GPU memory")
    print("- Learning rate schedule: StepLR with step_size=10, gamma=0.5")
    print("- Input normalization: [0, 1] range to match Sigmoid output")
