import torch
from collections import namedtuple

torch.set_float32_matmul_precision('high')

LossSettings = namedtuple('LossSettings', [
    'dice_weight', 'mse_weight', 'boundary_weight',
    'focal_weight', 'focal_alpha', 'focal_gamma'
])

loss_settings = LossSettings(
    dice_weight=0.0,
    mse_weight=0.0,
    boundary_weight=0.0,
    focal_weight=2.0,
    focal_alpha=0.5,
    focal_gamma=2.0
)

SegmentationHyperparams = namedtuple('SegmentationHyperparams', [
    'num_epochs', 'batch_size', 'learning_rate',
    'train_percent', 'optimizer_class', 'bias_factor'
])

segmentation_hyperparams = SegmentationHyperparams(
    num_epochs=1,
    batch_size=16,
    learning_rate=1e-4,
    train_percent=0.999,
    optimizer_class=torch.optim.Adam,  # Note: Capital A in Adam
    bias_factor=20.0
)

data_path = "/home/Adithya/Documents/noise_source_prog/paths.json"
