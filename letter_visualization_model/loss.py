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
from torch.utils.tensorboard import SummaryWriter

class BinarySegmentationLoss(nn.Module):
    def __init__(self, log_dir="logs/"):
        super().__init__()
        self.dice_weight = settings.loss_settings.dice_weight
        self.boundary_weight = settings.loss_settings.boundary_weight
        self.focal_weight = settings.loss_settings.focal_weight
        self.mse_weight = settings.loss_settings.mse_weight
        self.focal_alpha = settings.loss_settings.focal_alpha
        self.focal_gamma = settings.loss_settings.focal_gamma
        self.mse_loss = nn.MSELoss()

        # optional TensorBoard writer
        #self.writer = SummaryWriter(log_dir) if log_dir is not None else None
        self.writer = None

        # tracking moved outside forward (not compiled)
        self._running_stats = {
            "dice": 0.0,
            "boundary": 0.0,
            "focal": 0.0,
        }
        self._counter = 0
        self._runner = 1
        self.print_every_batches = settings.print_every_batches

    @torch.compile
    def forward(self, pred_masks, target_masks):
        target_masks = target_masks.float()
        pred_probs = torch.sigmoid(pred_masks)

        # Compute all losses
        dice_val = dice_loss(pred_probs, target_masks)
        boundary_val = boundary_loss(pred_probs, target_masks)
        focal_val = focal_loss(pred_probs, target_masks, self.focal_alpha, self.focal_gamma)
        # mse_val = self.mse_loss(pred_probs, target_masks)

        # Weighted sum
        loss = (
            self.dice_weight * dice_val +
            self.boundary_weight * boundary_val +
            self.focal_weight * focal_val
            # + self.mse_weight * mse_val
        )
        return loss, (dice_val.detach(), boundary_val.detach(), focal_val.detach())

    def update_running_stats(self, dice_val, boundary_val, focal_val, global_step=None):
        """Call this OUTSIDE forward() with detached values"""
        self._running_stats["dice"] += self.dice_weight * dice_val
        self._running_stats["boundary"] += self.boundary_weight * boundary_val
        self._running_stats["focal"] += self.focal_weight * focal_val

        self._counter += 1
        if self._counter >= self.print_every_batches:
            mean_d = self._running_stats["dice"] / self.print_every_batches
            mean_b = self._running_stats["boundary"] / self.print_every_batches
            mean_f = self._running_stats["focal"] / self.print_every_batches

            if self.writer is not None and global_step is not None:
                self.writer.add_scalar("Loss/Dice", mean_d, global_step)
                self.writer.add_scalar("Loss/Boundary", mean_b, global_step)
                self.writer.add_scalar("Loss/Focal", mean_f, global_step)

            else:
                # fallback to console print if no writer
                print(
                    f"({self._runner}): "
                    f"d: {mean_d:.4f}, "
                    f"b: {mean_b:.4f}, "
                    f"f: {mean_f:.4f}"
                )

            # reset stats
            self._running_stats = {"dice": 0.0, "boundary": 0.0, "focal": 0.0}
            self._counter = 0
            self._runner += 1

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
    dist_outside = euclidean_distance_transform_torch(gt_bool.float())
    dist_inside = euclidean_distance_transform_torch((~gt_bool).float())
    phi_G = torch.where(gt_bool, -dist_inside, dist_outside)

    return phi_G

@torch.compile
def boundary_loss(s_theta: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    phi_G = compute_signed_distance_map_gpu(g)
    return (phi_G * s_theta).mean()
