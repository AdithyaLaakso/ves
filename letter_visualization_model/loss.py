import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import classif
import constants
import torchvision.models as models

def rgb_biased_mse_loss(predictions, targets, black_threshold=0.5, black_penalty_factor=6.0):
    # Ensure tensors are on the same device
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()

    # Standard MSE loss per pixel per channel
    mse = (predictions - targets) ** 2

    # Calculate luminance/brightness for each pixel using standard RGB weights
    # Assumes RGB format - adjust channel dimension based on your data format
    if predictions.dim() == 4:  # (B, C, H, W) format
        if predictions.shape[1] == 3:  # channels first
            pred_luminance = 0.299 * predictions[:, 0] + 0.587 * predictions[:, 1] + 0.114 * predictions[:, 2]
            target_luminance = 0.299 * targets[:, 0] + 0.587 * targets[:, 1] + 0.114 * targets[:, 2]
        else:  # (B, H, W, C) format
            pred_luminance = 0.299 * predictions[:, :, :, 0] + 0.587 * predictions[:, :, :, 1] + 0.114 * predictions[:, :, :, 2]
            target_luminance = 0.299 * targets[:, :, :, 0] + 0.587 * targets[:, :, :, 1] + 0.114 * targets[:, :, :, 2]
    else:  # 3D tensor (H, W, C)
        pred_luminance = 0.299 * predictions[:, :, 0] + 0.587 * predictions[:, :, 1] + 0.114 * predictions[:, :, 2]
        target_luminance = 0.299 * targets[:, :, 0] + 0.587 * targets[:, :, 1] + 0.114 * targets[:, :, 2]

    # Identify pixels where we predicted black but target is not black
    predicted_black = pred_luminance < black_threshold
    target_not_black = target_luminance >= black_threshold
    wrong_black_predictions = predicted_black & target_not_black

    # Create penalty mask - expand to match MSE dimensions
    if predictions.dim() == 4 and predictions.shape[1] == 3:  # (B, C, H, W)
        penalty_mask = wrong_black_predictions.unsqueeze(1).expand_as(mse)
    elif predictions.dim() == 4:  # (B, H, W, C)
        penalty_mask = wrong_black_predictions.unsqueeze(-1).expand_as(mse)
    else:  # 3D tensor
        penalty_mask = wrong_black_predictions.unsqueeze(-1).expand_as(mse)

    # Apply penalty
    penalty_weights = torch.where(penalty_mask,
                                 torch.full_like(mse, black_penalty_factor),
                                 torch.ones_like(mse))

    biased_mse = mse * penalty_weights
    return biased_mse.mean()

def rgb_biased_mse_loss_v2(predictions, targets, black_threshold=0.2, black_penalty_factor=2.0,
                          use_max_channel=False):
    # Standard MSE
    mse = (predictions - targets) ** 2

    if use_max_channel:
        # Use maximum channel value to determine darkness
        pred_brightness = torch.max(predictions, dim=-1 if predictions.dim() == 3 else 1)[0]
        target_brightness = torch.max(targets, dim=-1 if targets.dim() == 3 else 1)[0]
    else:
        # Use luminance (same as v1)
        if predictions.dim() == 4 and predictions.shape[1] == 3:  # (B, C, H, W)
            pred_brightness = 0.299 * predictions[:, 0] + 0.587 * predictions[:, 1] + 0.114 * predictions[:, 2]
            target_brightness = 0.299 * targets[:, 0] + 0.587 * targets[:, 1] + 0.114 * targets[:, 2]
        else:  # (B, H, W, C) or (H, W, C)
            pred_brightness = 0.299 * predictions[..., 0] + 0.587 * predictions[..., 1] + 0.114 * predictions[..., 2]
            target_brightness = 0.299 * targets[..., 0] + 0.587 * targets[..., 1] + 0.114 * targets[..., 2]

    # Find wrong black predictions
    predicted_black = pred_brightness < black_threshold
    target_not_black = target_brightness >= black_threshold
    wrong_black_mask = predicted_black & target_not_black

    # Expand mask to match MSE shape
    if predictions.dim() == 4 and predictions.shape[1] == 3:  # (B, C, H, W)
        penalty_mask = wrong_black_mask.unsqueeze(1).expand_as(mse)
    else:
        penalty_mask = wrong_black_mask.unsqueeze(-1).expand_as(mse)

    penalty_weights = torch.where(penalty_mask, black_penalty_factor, 1.0)
    return (mse * penalty_weights).mean()

# Custom loss class for easy integration with training loops
class RGBBiasedMSELoss(nn.Module):
    def __init__(self, black_threshold=0.2, black_penalty_factor=2.0, use_max_channel=False):
        super().__init__()
        self.black_threshold = black_threshold
        self.black_penalty_factor = black_penalty_factor
        self.use_max_channel = use_max_channel

    def forward(self, predictions, targets):
        return rgb_biased_mse_loss_v2(predictions, targets,
                                     self.black_threshold,
                                     self.black_penalty_factor,
                                     self.use_max_channel)

# Example usage and testing
if __name__ == "__main__":
    # Create sample RGB data
    batch_size, height, width, channels = 2, 32, 32, 3

    # Random predictions and targets (values between 0 and 1)
    predictions = torch.rand(batch_size, height, width, channels)
    targets = torch.rand(batch_size, height, width, channels)

    # Make some predictions artificially dark (black) where targets are bright
    predictions[0, 10:15, 10:15, :] = 0.1  # Dark prediction
    targets[0, 10:15, 10:15, :] = 0.8      # Bright target (wrong black prediction)

    # Standard MSE
    standard_mse = nn.MSELoss()(predictions, targets)

    # Biased MSE with different thresholds
    biased_loss_02 = rgb_biased_mse_loss(predictions, targets, black_threshold=0.2, black_penalty_factor=3.0)
    biased_loss_05 = rgb_biased_mse_loss(predictions, targets, black_threshold=0.5, black_penalty_factor=3.0)

    print(f"Standard MSE: {standard_mse:.6f}")
    print(f"Biased MSE (threshold=0.2): {biased_loss_02:.6f}")
    print(f"Biased MSE (threshold=0.5): {biased_loss_05:.6f}")

    # Using the loss class
    loss_fn = RGBBiasedMSELoss(black_threshold=0.3, black_penalty_factor=2.5)
    class_loss = loss_fn(predictions, targets)
    print(f"Class-based loss: {class_loss:.6f}")

    # Test different data formats
    print("\nTesting different tensor formats:")

    # (B, C, H, W) format
    pred_bchw = predictions.permute(0, 3, 1, 2)  # Convert to channels first
    target_bchw = targets.permute(0, 3, 1, 2)
    loss_bchw = rgb_biased_mse_loss(pred_bchw, target_bchw, black_threshold=0.2)
    print(f"BCHW format loss: {loss_bchw:.6f}")

    # Single image (H, W, C)
    single_pred = predictions[0]
    single_target = targets[0]
    single_loss = rgb_biased_mse_loss(single_pred, single_target, black_threshold=0.2)
    print(f"Single image loss: {single_loss:.6f}")
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

def compute_classification_accuracy(denoised_images, true_labels, classifier, label_to_idx=constants.greek_letters):
    with torch.no_grad():
        predictions = classifier(denoised_images)
        predicted_labels = torch.argmax(predictions, dim=1)

        # Handle string labels
        if isinstance(true_labels, list) and len(true_labels) > 0 and isinstance(true_labels[0], np.ndarray):
            if label_to_idx is None:
                raise ValueError("label_to_idx mapping required for string labels")
            # Convert string labels to indices
            numeric_labels = [label_to_idx[str(label)] for label in true_labels]
            true_labels = torch.tensor(numeric_labels, device=denoised_images.device, dtype=torch.long)
        else:
            # Handle numeric labels
            if not isinstance(true_labels, torch.Tensor):
                true_labels = torch.tensor(true_labels, device=denoised_images.device)
            if true_labels.dim() > 1:
                true_labels = true_labels.squeeze()
            true_labels = true_labels.long()

        # Compute accuracy
        correct = (predicted_labels == true_labels).float()
        accuracy = correct.mean()
#         print(f"p {predicted_labels}, t {true_labels} c {correct} a {accuracy}")
    return accuracy
tmp = classif.SingleLetterModel(num_classes=25)
tmp.load_state_dict(torch.load("class.pth", weights_only=False))
tmp.to("cuda:0")
tmp.eval()
def compute_classification_improvement(original_image, denoised_images, true_labels, classifier=None):
    if classifier == None:
        classifier = tmp

    before = compute_classification_accuracy(original_image, true_labels, classifier)
    after = compute_classification_accuracy(denoised_images, true_labels, classifier)
    accuracy_loss = (before - after)
    return accuracy_loss

class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight=1.0, l1_weight=4.0, acc_weight=0.00, acc_rel_delta=0.000, use_l1=False):
        super(CombinedLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.acc_weight = acc_weight
        self.acc_rel_delta = acc_rel_delta
        self.ssim_loss = SSIMLoss()

        self.device = torch.device("cuda:0")

        # Load your pre-trained Greek letter classifier
        self.classifier = classif.SingleLetterModel(num_classes=25)
        self.classifier.load_state_dict(torch.load("class.pth", weights_only=False))
        self.classifier.to(self.device)
        self.classifier.eval()

        if use_l1:
            self.pixel_loss = nn.L1Loss()
        else:
            self.pixel_loss = nn.MSELoss()

    def forward(self, pred, target, inputs, label, epoch):
        """Calculate combined loss."""
        ssim_loss = self.ssim_loss(pred, target)
#         pixel_loss = self.pixel_loss(pred,target)
        pixel_loss = rgb_biased_mse_loss(pred,target, black_threshold=0.3)
        accuracy_loss = compute_classification_improvement(inputs, pred, label, classifier=self.classifier)
#         print(f"before: {before}, after: {after}, delta: {accuracy_loss}")

#         print(f"ssim_loss: {ssim_loss}, pixel_loss: {pixel_loss}, accuracy_loss: {accuracy_loss}")

        total_loss = self.ssim_weight * ssim_loss + \
                     self.l1_weight * pixel_loss + \
                    (self.acc_rel_delta * epoch + self.acc_weight) * accuracy_loss
#         total_loss = torch.sqrt(total_loss)
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
