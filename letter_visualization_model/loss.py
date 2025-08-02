import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def create_gaussian_kernel(window_size, sigma):
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.outer(g)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) Loss

    Args:
        window_size (int): Size of the sliding window. Default: 11
        size_average (bool): If True, returns the average loss. Default: True
        data_range (float): The dynamic range of the images. Default: 1.0
        K (tuple): Constants in the SSIM formula. Default: (0.01, 0.03)
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'
    """

    def __init__(self, window_size=11, size_average=True, data_range=1.0,
                 K=(0.01, 0.03), reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.reduction = reduction

        # Create Gaussian window
        sigma = 1.5
        self.window = create_gaussian_kernel(window_size, sigma)

    def forward(self, img1, img2):
        """
        Calculate SSIM loss between two images.

        Args:
            img1, img2: Input images with shape [N, C, H, W]

        Returns:
            SSIM loss (1 - SSIM index)
        """
        # Move window to the same device as input
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        # Get image dimensions
        N, C, H, W = img1.shape

        # Create window for each channel
        window = self.window.expand(C, 1, self.window_size, self.window_size)

        # Constants
        C1 = (self.K[0] * self.data_range) ** 2
        C2 = (self.K[1] * self.data_range) ** 2

        # Compute local means
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=C)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=C) - mu1_mu2

        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.reduction == 'mean':
            ssim_value = ssim_map.mean()
        elif self.reduction == 'sum':
            ssim_value = ssim_map.sum()
        else:  # 'none'
            ssim_value = ssim_map.mean(dim=[1, 2, 3])  # Average over spatial and channel dims

        # Return loss (1 - SSIM)
        return 1 - ssim_value


class MS_SSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity Index Measure (MS-SSIM) Loss

    Args:
        data_range (float): The dynamic range of the images. Default: 1.0
        size_average (bool): If True, returns the average loss. Default: True
        weights (list): Weights for different scales. Default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        K (tuple): Constants in the SSIM formula. Default: (0.01, 0.03)
    """

    def __init__(self, data_range=1.0, size_average=True,
                 weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                 K=(0.01, 0.03)):
        super(MS_SSIMLoss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.weights = torch.tensor(weights)
        self.K = K

        # Create Gaussian window
        window_size = 11
        sigma = 1.5
        self.window = create_gaussian_kernel(window_size, sigma)

    def _ssim(self, img1, img2, window_size=11):
        """Calculate SSIM for a single scale."""
        # Move window to the same device as input
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        N, C, H, W = img1.shape
        window = self.window.expand(C, 1, window_size, window_size)

        C1 = (self.K[0] * self.data_range) ** 2
        C2 = (self.K[1] * self.data_range) ** 2

        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, img1, img2):
        """Calculate MS-SSIM loss."""
        if self.weights.device != img1.device:
            self.weights = self.weights.to(img1.device)

        levels = len(self.weights)
        mssim = []

        for i in range(levels):
            ssim_val = self._ssim(img1, img2)
            mssim.append(ssim_val)

            if i < levels - 1:
                # Downsample for next level
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

        mssim = torch.stack(mssim)
        ms_ssim = torch.prod(mssim ** self.weights)

        return 1 - ms_ssim

class CombinedLoss(nn.Module):
    """
    Combined SSIM + L1/L2 Loss for better training stability

    Args:
        ssim_weight (float): Weight for SSIM loss. Default: 0.8
        l1_weight (float): Weight for L1 loss. Default: 0.2
        use_l1 (bool): If True, use L1 loss; otherwise use L2 (MSE). Default: True
    """

    def __init__(self, ssim_weight=0.8, l1_weight=0.2, use_l1=False):
        super(CombinedLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.ssim_loss = SSIMLoss()

        if use_l1:
            self.pixel_loss = nn.L1Loss()
        else:
            self.pixel_loss = nn.MSELoss()

    def forward(self, pred, target):
        """Calculate combined loss."""
        ssim_loss = self.ssim_loss(pred, target)
        pixel_loss = self.pixel_loss(pred, target)

        total_loss = self.ssim_weight * ssim_loss + self.l1_weight * pixel_loss
        total_loss = torch.sqrt(total_loss)
#         total_loss *= 100

        return total_loss


# Alternative: Using pytorch-msssim library (if you want to install it)
# pip install pytorch-msssim
"""
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class SSIMLoss_External(nn.Module):
    def __init__(self, data_range=1.0, size_average=True, channel=3):
        super(SSIMLoss_External, self).__init__()
        self.ssim_module = SSIM(data_range=data_range, size_average=size_average, channel=channel)

    def forward(self, img1, img2):
        return 1 - self.ssim_module(img1, img2)
"""


# Example usage and testing
if __name__ == "__main__":
    # Test the loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy images
    img1 = torch.randn(4, 3, 32, 32).to(device)  # Batch of 4 RGB 32x32 images
    img2 = torch.randn(4, 3, 32, 32).to(device)

    # Test SSIM Loss
    ssim_loss = SSIMLoss().to(device)
    loss_value = ssim_loss(img1, img2)
    print(f"SSIM Loss: {loss_value.item():.4f}")

    # Test MS-SSIM Loss
    ms_ssim_loss = MS_SSIMLoss().to(device)
    ms_loss_value = ms_ssim_loss(img1, img2)
    print(f"MS-SSIM Loss: {ms_loss_value.item():.4f}")

    # Test Combined Loss
    combined_loss = CombinedLoss().to(device)
    combined_value = combined_loss(img1, img2)
    print(f"Combined Loss: {combined_value.item():.4f}")

    # Test with identical images (should be close to 0)
    identical_loss = ssim_loss(img1, img1)
    print(f"SSIM Loss (identical images): {identical_loss.item():.4f}")

    print("\nRecommendations:")
    print("1. For image reconstruction: Use CombinedLoss (SSIM + L1)")
    print("2. For denoising tasks: Use SSIMLoss alone")
    print("3. For high-quality image generation: Use MS_SSIMLoss")
    print("4. Ensure your images are normalized to [0, 1] range")
    print("5. Start with learning rate 1e-4 when using SSIM loss")
