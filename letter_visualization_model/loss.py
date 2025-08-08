from scipy.ndimage import distance_transform_edt as distance
from skimage.morphology import dilation, disk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import classif
import constants
import torchvision.models as models

def binary_biased_mse_loss(predictions, targets, background_threshold=0.5, background_penalty_factor=3.0):
    """
    Biased MSE loss for binary segmentation that penalizes false negative predictions
    (predicting background when target is foreground).

    Args:
        predictions: Model predictions (0-1 range after sigmoid)
        targets: Ground truth binary masks (0-1 range)
        background_threshold: Threshold below which predictions are considered background
        background_penalty_factor: Penalty multiplier for false negatives
    """
    # Ensure tensors are on the same device
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()

    # Standard MSE loss per pixel
    mse = (predictions - targets) ** 2

    # Identify false negative pixels (predicted background, actual foreground)
    predicted_background = predictions < background_threshold
    target_foreground = targets >= background_threshold
    false_negative_mask = predicted_background & target_foreground

    # Apply penalty to false negatives
    penalty_weights = torch.where(false_negative_mask,
                                 torch.full_like(mse, background_penalty_factor),
                                 torch.ones_like(mse))

    biased_mse = mse * penalty_weights
    return biased_mse.mean()


def dice_loss(predictions, targets, smooth=1e-6):
    """
    Dice loss for binary segmentation.

    Args:
        predictions: Model predictions (0-1 range after sigmoid)
        targets: Ground truth binary masks (0-1 range)
        smooth: Smoothing factor to avoid division by zero
    """
    # Flatten tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Calculate intersection and union
    intersection = (predictions * targets).sum()
    dice_coeff = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)

    return 1 - dice_coeff


def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss for binary segmentation to handle class imbalance.

    Args:
        predictions: Model predictions (0-1 range after sigmoid)
        targets: Ground truth binary masks (0-1 range)
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
    """
    # Calculate binary cross entropy
    bce = F.binary_cross_entropy(predictions, targets, reduction='none')

    # Calculate p_t
    p_t = predictions * targets + (1 - predictions) * (1 - targets)

    # Calculate alpha_t
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    # Calculate focal loss
    focal_loss = alpha_t * (1 - p_t) ** gamma * bce

    return focal_loss.mean()


def create_gaussian_kernel(window_size, sigma):
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.outer(g)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) Loss for single-channel images
    (adapted for binary segmentation masks)
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
        Calculate SSIM loss between two single-channel images.

        Args:
            img1, img2: Input images with shape [N, 1, H, W]

        Returns:
            SSIM loss (1 - SSIM index)
        """
        # Move window to the same device as input
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        # Get image dimensions
        N, C, H, W = img1.shape

        # Create window for single channel
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


def compute_classification_accuracy_from_segmentation(input_volumes, segmentation_masks, true_labels, classifier,
                                                    label_to_idx=constants.greek_letters):
    """
    Compute classification accuracy by applying segmentation masks to input volumes
    and then classifying the masked result.

    Args:
        input_volumes: Original 8-channel input volumes [N, 8, 128, 128]
        segmentation_masks: Predicted binary masks [N, 1, 32, 32]
        true_labels: Ground truth classification labels
        classifier: Pre-trained classification model
    """
    with torch.no_grad():
        # Upsample segmentation masks to match input volume size
        upsampled_masks = F.interpolate(segmentation_masks, size=(128, 128),
                                      mode='bilinear', align_corners=False)

        # Apply mask to input volumes (broadcast across all 8 channels)
        masked_volumes = input_volumes * upsampled_masks

        # Convert 8-channel volume to format expected by classifier
        # This depends on your classifier's expected input format
        # Option 1: Take first 3 channels and treat as RGB
        if input_volumes.shape[1] >= 3:
            classifier_input = masked_volumes[:, :3]  # Take first 3 channels
        else:
            # Option 2: Convert single/multi-channel to 3-channel by replication
            classifier_input = masked_volumes[:, 0:1].repeat(1, 3, 1, 1)

        # Resize to classifier's expected input size (assuming 32x32 or similar)
        classifier_input = F.interpolate(classifier_input, size=(32, 32),
                                       mode='bilinear', align_corners=False)

        # Get predictions from classifier
        predictions = classifier(classifier_input)
        predicted_labels = torch.argmax(predictions, dim=1)

        # Handle string labels
        if isinstance(true_labels, list) and len(true_labels) > 0 and isinstance(true_labels[0], np.ndarray):
            if label_to_idx is None:
                raise ValueError("label_to_idx mapping required for string labels")
            # Convert string labels to indices
            numeric_labels = [label_to_idx[str(label)] for label in true_labels]
            true_labels = torch.tensor(numeric_labels, device=input_volumes.device, dtype=torch.long)
        else:
            # Handle numeric labels
            if not isinstance(true_labels, torch.Tensor):
                true_labels = torch.tensor(true_labels, device=input_volumes.device)
            if true_labels.dim() > 1:
                true_labels = true_labels.squeeze()
            true_labels = true_labels.long()

        # Compute accuracy
        correct = (predicted_labels == true_labels).float()
        accuracy = correct.mean()

    return accuracy


def compute_classification_improvement_segmentation(input_volumes, predicted_masks, ground_truth_masks,
                                                   true_labels, classifier=None):
    """
    Compute classification improvement by comparing accuracy with predicted vs ground truth masks.

    Args:
        input_volumes: Original 8-channel input volumes [N, 8, 128, 128]
        predicted_masks: Predicted segmentation masks [N, 1, 32, 32]
        ground_truth_masks: Ground truth segmentation masks [N, 1, 32, 32]
        true_labels: Ground truth classification labels
        classifier: Pre-trained classification model
    """
    if classifier is None:
        # Load default classifier
        tmp = classif.SingleLetterModel(num_classes=25)
        tmp.load_state_dict(torch.load("class.pth", weights_only=False))
        tmp.to(input_volumes.device)
        tmp.eval()
        classifier = tmp

    # Compute accuracy with ground truth masks
    accuracy_with_gt = compute_classification_accuracy_from_segmentation(
        input_volumes, ground_truth_masks, true_labels, classifier)

    # Compute accuracy with predicted masks
    accuracy_with_pred = compute_classification_accuracy_from_segmentation(
        input_volumes, predicted_masks, true_labels, classifier)

    # Return accuracy loss (negative improvement)
    accuracy_loss = accuracy_with_gt - accuracy_with_pred
    return accuracy_loss


class SegmentationCombinedLoss(nn.Module):
    """
    Combined loss for segmentation that includes:
    1. Segmentation quality (Dice/MSE/Focal)
    2. SSIM for structural similarity
    3. Classification accuracy preservation
    """

    def __init__(self, dice_weight=2.0, mse_weight=0.0, ssim_weight=0.0,
                 acc_weight=0.0, acc_rel_delta=0.00, segmentation_loss_type='dice'):
        super(SegmentationCombinedLoss, self).__init__()

        self.dice_weight = dice_weight
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.acc_weight = acc_weight
        self.acc_rel_delta = acc_rel_delta
        self.segmentation_loss_type = segmentation_loss_type

        # Initialize loss components
        self.ssim_loss = SSIMLoss()

        # Choose segmentation loss type
        if segmentation_loss_type == 'dice':
            self.seg_loss_fn = dice_loss
        elif segmentation_loss_type == 'focal':
            self.seg_loss_fn = focal_loss
        elif segmentation_loss_type == 'biased_mse':
            self.seg_loss_fn = binary_biased_mse_loss
        elif segmentation_loss_type == 'BCELoss':
            self.seg_loss_fn = BCELOSS
        else:
            self.seg_loss_fn = nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load pre-trained classifier
        self.classifier = None
        try:
            self.classifier = classif.SingleLetterModel(num_classes=25)
            self.classifier.load_state_dict(torch.load("class.pth", weights_only=False))
            self.classifier.to(self.device)
            self.classifier.eval()
        except:
            print("Warning: Could not load classifier. Classification loss will be disabled.")

    def forward(self, pred_masks, target_masks, input_volumes, labels, epoch=0):
        """
        Calculate combined segmentation loss.

        Args:
            pred_masks: Predicted segmentation masks [N, 1, 32, 32]
            target_masks: Ground truth segmentation masks [N, 1, 32, 32]
            input_volumes: Original input volumes [N, 8, 128, 128]
            labels: Classification labels
            epoch: Current training epoch
        """
        total_loss = 0.0
        loss_components = {}

        # 1. Segmentation quality loss
        if self.segmentation_loss_type == 'dice':
            seg_loss = self.seg_loss_fn(pred_masks, target_masks)
            total_loss += self.dice_weight * seg_loss
            loss_components['dice_loss'] = seg_loss.item()
        elif self.segmentation_loss_type in ['biased_mse', 'focal']:
            seg_loss = self.seg_loss_fn(pred_masks, target_masks)
            total_loss += self.mse_weight * seg_loss
            loss_components['seg_loss'] = seg_loss.item()
        else:
            mse_loss = F.mse_loss(pred_masks, target_masks)
            total_loss += self.mse_weight * mse_loss
            loss_components['mse_loss'] = mse_loss.item()

        # 2. SSIM loss for structural similarity
        ssim_loss = self.ssim_loss(pred_masks, target_masks)
        total_loss += self.ssim_weight * ssim_loss
        loss_components['ssim_loss'] = ssim_loss.item()

        # 3. Classification accuracy preservation loss
        if self.classifier is not None and self.acc_weight > 0:
            try:
                accuracy_loss = compute_classification_improvement_segmentation(
                    input_volumes, pred_masks, target_masks, labels, self.classifier)

                # Progressive weight increase with epoch
                current_acc_weight = self.acc_weight + (self.acc_rel_delta * epoch)
                total_loss += current_acc_weight * accuracy_loss
                loss_components['accuracy_loss'] = accuracy_loss.item()
                loss_components['acc_weight'] = current_acc_weight
            except Exception as e:
                print(f"Warning: Could not compute classification loss: {e}")
                loss_components['accuracy_loss'] = 0.0

        loss_components['total_loss'] = total_loss.item()

        # Store loss components for logging
        self.last_loss_components = loss_components

        return total_loss

class BinarySegmentationLoss(nn.Module):
    """
    Binary segmentation loss combining Dice loss and Boundary loss.
    """

    def __init__(self, dice_weight=1.0, boundary_weight=1.0):
        super(BinarySegmentationLoss, self).__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, pred_masks, target_masks):
        """
        Calculate combined Dice + Boundary segmentation loss.

        Args:
            pred_masks: Predicted segmentation masks [N, 1, H, W]
            target_masks: Ground truth segmentation masks [N, 1, H, W]
        """
        dice = dice_loss(pred_masks, target_masks)
        boundary = boundary_loss(pred_masks, target_masks)

        total_loss = self.dice_weight * dice + self.boundary_weight * boundary
        return total_loss


def dice_loss(pred, target, epsilon=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()


def boundary_loss(pred, target, epsilon=1e-6):
    """
    Computes boundary loss based on distance transform of target mask edges.
    """

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    loss = 0.0

    for i in range(pred_np.shape[0]):
        gt_mask = target_np[i, 0]
        pred_mask = pred_np[i, 0]

        gt_boundary = mask_to_boundary(gt_mask)
        pred_boundary = mask_to_boundary(pred_mask)

        # Distance transform of ground truth boundary
        dt = distance(1 - gt_boundary)

        pred_boundary = torch.tensor(pred_boundary, device=pred.device, dtype=pred.dtype)
        dt = torch.tensor(dt, device=pred.device, dtype=pred.dtype)

        boundary_loss_sample = (pred_boundary * dt).mean()
        loss += boundary_loss_sample

    return loss / pred.shape[0]


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary map using morphological dilation.

    Args:
        mask: [H, W] numpy array
    Returns:
        boundary: [H, W] binary boundary map
    """

    mask = mask.astype(np.uint8)
    h, w = mask.shape
    img_diag = (h ** 2 + w ** 2) ** 0.5
    dilation_radius = max(1, int(round(dilation_ratio * img_diag)))
    struct = disk(dilation_radius)

    dilated = dilation(mask, struct)
    eroded = dilation(1 - mask, struct)
    boundary = dilated & eroded

    return boundary.astype(np.uint8)

# Usage example
if __name__ == "__main__":
    # Test the segmentation loss functions
    batch_size = 4

    # Create dummy data
    input_volumes = torch.rand(batch_size, 8, 128, 128)  # 8-channel input
    pred_masks = torch.sigmoid(torch.rand(batch_size, 1, 32, 32))  # Predicted masks
    target_masks = torch.round(torch.rand(batch_size, 1, 32, 32))  # Binary target masks
    labels = torch.randint(0, 25, (batch_size,))  # Classification labels

    # Test different loss functions
    print("Testing segmentation loss functions:")

    # Test Dice loss
    dice_loss_val = dice_loss(pred_masks, target_masks)
    print(f"Dice Loss: {dice_loss_val:.4f}")

    # Test Focal loss
    focal_loss_val = focal_loss(pred_masks, target_masks)
    print(f"Focal Loss: {focal_loss_val:.4f}")

    # Test biased MSE loss
    biased_mse_val = binary_biased_mse_loss(pred_masks, target_masks)
    print(f"Biased MSE Loss: {biased_mse_val:.4f}")

    # Test SSIM loss
    ssim_loss_fn = SSIMLoss()
    ssim_loss_val = ssim_loss_fn(pred_masks, target_masks)
    print(f"SSIM Loss: {ssim_loss_val:.4f}")

    # Test combined loss
    combined_loss_fn = SegmentationCombinedLoss(
        dice_weight=2.0, mse_weight=0.0, ssim_weight=0.5,
        acc_weight=0.0, segmentation_loss_type='dice'  # Disable classification loss for testing
    )

    combined_loss_val = combined_loss_fn(pred_masks, target_masks, input_volumes, labels, epoch=0)
    print(f"Combined Loss: {combined_loss_val:.4f}")
    print(f"Loss components: {combined_loss_fn.last_loss_components}")

    print("\nTraining recommendations:")
    print("- Use SegmentationCombinedLoss with dice_weight=2.0, ssim_weight=0.5")
    print("- Start with acc_weight=0.0, gradually increase if classification feedback is needed")
    print("- Monitor individual loss components for balanced training")
    print("- Consider focal loss for highly imbalanced segmentation tasks")
