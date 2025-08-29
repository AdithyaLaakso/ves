import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.utils.tensorboard import SummaryWriter

import settings

epsilon = 1e-6

class MetaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.BSL = BinarySegmentationLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.meta_div_weight = settings.meta_div_weight

        self.meta_f_weight = settings.meta_f_weight
        self.meta_b_weight = settings.meta_b_weight
        self.meta_m_weight = settings.meta_m_weight
        self.meta_d_weight = settings.meta_d_weight
        self.meta_c_weight = settings.meta_c_weight

        self.writer = SummaryWriter(settings.log_dir)

        # tracking moved outside forward (not compiled)
        self._running_stats = {
            "dice": 0.0,
            "boundary": 0.0,
            "focal": 0.0,
            "mse": 0.0
        }
        if settings.mode == settings.MULTITASK:
            self._running_stats["classification"] = 0.0
        self._counter = 0
        self._runner = 1
        self.print_every_batches = settings.print_every_batches
        self.global_step = 1

    @torch.compile
    def forward(self, pred, target, epoch=0):
        after, (a_d, a_b, a_f, a_m) = self.BSL(pred[0], target[0])
        a_c = 0
        if settings.mode == settings.MULTITASK or settings.mode == settings.CLASSIFICATION:
            cw = settings.loss_settings.class_weight + (epoch * settings.loss_settings.class_weight_delta)
            class_loss = self.cross_entropy(pred[1], target[1])
            a_c = class_loss * cw

        self.update_running_stats(a_d, a_b, a_f, a_m, a_c)
        self.global_step = self.global_step + 1

        return after + a_c

    def update_running_stats(self, dice_val, boundary_val, focal_val, mse_val, classification_val=0):
        self._running_stats["dice"] += dice_val
        self._running_stats["boundary"] += boundary_val
        self._running_stats["focal"] += focal_val
        self._running_stats["mse"] += mse_val
        if settings.mode == settings.MULTITASK:
            self._running_stats["classification"] += classification_val

        mean_d = self._running_stats["dice"]
        mean_b = self._running_stats["boundary"]
        mean_f = self._running_stats["focal"]
        mean_m = self._running_stats["mse"]
        mean_total = mean_d + mean_b + mean_m + mean_f

        mean_c = 0
        if settings.mode == settings.MULTITASK or settings.mode==settings.CLASSIFICATION:
            mean_c = self._running_stats["classification"]
            mean_total += mean_c

        # if self.writer is not None and global_step is not None:
        self.writer.add_scalar("Loss/Dice", mean_d / self.global_step, self.global_step)
        self.writer.add_scalar("Loss/Boundary", mean_b / self.global_step, self.global_step)
        self.writer.add_scalar("Loss/Focal", mean_f / self.global_step, self.global_step)
        self.writer.add_scalar("Loss/MSE", mean_m / self.global_step, self.global_step)

        if settings.mode == settings.MULTITASK:
            self.writer.add_scalar("Loss/Classification(raw)", mean_c / self.global_step, self.global_step)

        self.writer.add_scalar("Loss/total", mean_total / self.global_step, self.global_step)

class BinarySegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_weight = settings.loss_settings.dice_weight
        self.boundary_weight = settings.loss_settings.boundary_weight
        self.focal_weight = settings.loss_settings.focal_weight
        self.mse_weight = settings.loss_settings.mse_weight
        self.focal_alpha = settings.loss_settings.focal_alpha
        self.focal_gamma = settings.loss_settings.focal_gamma
        self.mse_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(settings.device)
        self.d = nn.BCELoss()


    @torch.compile
    def forward(self, pred_masks, target_masks) -> tuple[float, tuple[float, float, float, float]]:
        target_masks = target_masks.float()
        pred_probs = torch.sigmoid(pred_masks)

        # Compute all losses
        dice_val = dice_loss(pred_probs, target_masks, self.d) * self.dice_weight
        boundary_val = boundary_loss(pred_probs, target_masks) * self.boundary_weight
        focal_val = focal_loss(pred_probs, target_masks, self.focal_alpha, self.focal_gamma) * self.focal_weight
        mse_val = self.mse_loss(pred_probs, target_masks) * self.mse_weight

        # dice_val = dice_val * dice_val
        # boundary_val = boundary_val * boundary_val * boundary_val
        # focal_val = focal_val
        # mse_val = mse_val

        # Weighted sum
        loss = (
            dice_val +
            boundary_val +
            focal_val +
            mse_val
        )

        return loss, (dice_val, boundary_val, focal_val, mse_val)


@torch.compile
def focal_loss(pred, target, alpha=0.25, gamma=2.0, epsilon=1e-6):
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    pt = pred * target + (1 - pred) * (1 - target)  # pt = p if target=1 else 1-p
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = -alpha_t * (1 - pt) ** gamma * torch.log(pt)
    return loss.mean()

@torch.compile
def dice_loss(pred, target, loss):
    # pred = torch.nn.functional.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
    # pred = pred.contiguous()
    # target = target.contiguous()
    # intersection = (pred * target).sum(dim=(2, 3))
    # union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    # dice = (2. * intersection + epsilon) / (union + epsilon)
    # return 1 - dice.mean()
    return loss(pred, target)

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
def boundary_loss(s_theta, g):
    phi_G = compute_signed_distance_map_gpu(g)

    # Reduce penalty for distant pixels
    weight = torch.exp(-torch.abs(phi_G) / 10.0)  # Exponential decay
    loss = torch.mean(weight * torch.abs(phi_G * s_theta))
    return loss
