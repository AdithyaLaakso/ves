from scipy.ndimage import distance_transform_edt
from skimage.morphology import dilation, disk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import constants
import torchvision.models as models
from functools import lru_cache
import settings

class BinarySegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_weight = settings.loss_settings.dice_weight
        self.boundary_weight = settings.loss_settings.boundary_weight
        self.focal_weight = settings.loss_settings.focal_weight
        self.mse_weight = settings.loss_settings.mse_weight
        self.focal_alpha = settings.loss_settings.focal_alpha
        self.focal_gamma = settings.loss_settings.focal_gamma
        self.mse_loss = nn.MSELoss()
        self.counter = 0
        self.print_every_batches = settings.print_every_batches

        self.running_b_loss = 0
        self.running_dice_loss = 0

        self.runner = 1

    @torch.compile
    def forward(self, pred_masks, target_masks):
        target_masks = target_masks.float()
        pred_probs = torch.sigmoid(pred_masks)

        # Compute all losses regardless of weight
        dice_val = dice_loss(pred_probs, target_masks)
        boundary_val = boundary_loss(pred_probs, target_masks)
        #focal_val = focal_loss(pred_probs, target_masks, self.focal_alpha, self.focal_gamma)
        #mse_val = self.mse_loss(pred_probs, target_masks)

        # Weighted sum
        loss = (
                self.dice_weight * dice_val +
                self.boundary_weight * boundary_val #+
                #self.focal_weight * focal_val +
                #self.mse_weight * mse_val
        )

        # print(boundary_val.item())

        self.running_dice_loss += self.dice_weight * dice_val.item()
        self.running_b_loss += (self.boundary_weight * boundary_val.item())

        # Optional: logging counter increment
        self.counter += 1
        if self.counter >= self.print_every_batches:
            print(
                f"({self.runner}): "
                f"d: {self.running_dice_loss / self.print_every_batches:.4f}, "
                f"b: {self.boundary_weight / self.print_every_batches:.4f}, "
                #f"f: {self.focal_weight * focal_val:.4f}, "
                #f"m: {self.mse_weight * mse_val:.4f}"
            )
            self.running_b_loss = 0
            self.running_dice_loss = 0
            self.counter = 0
            self.runner += 1

        return loss

@torch.compile
def focal_loss(pred, target, alpha=0.25, gamma=2.0, epsilon=1e-6):
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    pt = pred * target + (1 - pred) * (1 - target)  # pt = p if target=1 else 1-p
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = -alpha_t * (1 - pt) ** gamma * torch.log(pt)
    return loss.mean()

@torch.compile
def dice_loss(pred, target, epsilon=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()

@torch.compile
def euclidean_distance_transform_torch(mask: torch.Tensor) -> torch.Tensor:
    N, _, H, W = mask.shape
    device = mask.device

    # Create coordinate grids
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    coords = torch.stack((yy, xx), dim=0).float()  # (2, H, W)

    dist_maps = []
    for b in range(N):
        fg = mask[b, 0] > 0.5  # (H, W)

        if fg.sum() == 0:
            # No foreground pixels - return max possible distance
            max_dist = torch.sqrt(torch.tensor((H-1)**2 + (W-1)**2, device=device, dtype=torch.float32))
            dist_maps.append(torch.full((1, H, W), max_dist, device=device))
            continue

        # Get coordinates of foreground pixels
        fg_y, fg_x = torch.where(fg)  # Get actual coordinates
        fg_coords = torch.stack([fg_y, fg_x], dim=1).float()  # (n_fg, 2)

        # Create all pixel coordinates
        all_coords = coords.permute(1, 2, 0).reshape(-1, 2)  # (H*W, 2)

        # Compute distances from each pixel to all foreground pixels
        # Using broadcasting: (H*W, 1, 2) - (1, n_fg, 2) = (H*W, n_fg, 2)
        diff = all_coords.unsqueeze(1) - fg_coords.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1)  # (H*W, n_fg)

        # Get minimum distance for each pixel
        min_dist_sq = dist_sq.min(dim=1).values  # (H*W,)
        dist_map = torch.sqrt(min_dist_sq).reshape(1, H, W)

        dist_maps.append(dist_map)

    return torch.stack(dist_maps, dim=0)

@torch.compile
def compute_signed_distance_map_gpu(gt: torch.Tensor) -> torch.Tensor:
    gt_bool = gt > 0.5

    # Distance from outside pixels to boundary (positive outside object)
    dist_outside = euclidean_distance_transform_torch(gt_bool.float().unsqueeze(1) if gt_bool.dim() == 3 else gt_bool)

    # Distance from inside pixels to boundary (will be negative inside object)
    dist_inside = euclidean_distance_transform_torch((~gt_bool).float().unsqueeze(1) if gt_bool.dim() == 3 else (~gt_bool).float())

    # Create signed distance map
    phi_G = dist_outside.clone()
    if gt_bool.dim() == 3:
        phi_G[gt_bool.unsqueeze(1)] = -dist_inside[gt_bool.unsqueeze(1)]
    else:
        phi_G[gt_bool] = -dist_inside[gt_bool]

    return phi_G

@torch.compile
def boundary_loss(s_theta: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    phi_G = compute_signed_distance_map_gpu(g)
    return (phi_G * s_theta).mean()

# Debug version to test
def debug_distance_transform():
    # Create a simple test case
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a simple 5x5 mask with a central square
    gt = torch.zeros(1, 5, 5, device=device)
    gt[0, 1:4, 1:4] = 1.0  # 3x3 square in center

    print("Ground truth:")
    print(gt[0])

    phi_G = compute_signed_distance_map_gpu(gt)
    print("\nSigned distance map:")
    print(phi_G[0, 0])

    return phi_G
