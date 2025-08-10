import torch
from collections import namedtuple

max_size = 50000

torch.set_float32_matmul_precision('medium')
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = False

LossSettings = namedtuple('LossSettings', [
    'dice_weight', 'mse_weight', 'boundary_weight',
    'focal_weight', 'focal_alpha', 'focal_gamma'
])

loss_settings = LossSettings(
    dice_weight=0.5,
    mse_weight=1.0,
    boundary_weight=0.0,
    focal_weight=2.0,
    focal_alpha=0.25,
    focal_gamma=2.0
)

SegmentationHyperparams = namedtuple('SegmentationHyperparams', [
    'num_epochs', 'batch_size', 'learning_rate',
    'train_percent', 'optimizer_class'
])

segmentation_hyperparams = SegmentationHyperparams(
    num_epochs=1,
    batch_size=16,
    learning_rate=8e-5,
    train_percent=0.99,
    optimizer_class=torch.optim.Adam,  # Note: Capital A in Adam
)

learning_rate_gamma=.99

data_path = "/home/Adithya/Documents/noise_source_prog/paths.json"

levels = [[1,2,3,4]]
display_levels = [1,2,3,4]
#levels += [4 for i in range(1,5)]
# levels += [2 for i in range(1,2)]
# levels += [i for i in range(1, 10) if i % 2 == 0]
print(f"training levels: {levels}")

image_size=128
patch_size=8
in_channels=1
out_channels=1
embed_size=768
num_blocks=12
num_heads=8
dropout=0.2
output_size=32
