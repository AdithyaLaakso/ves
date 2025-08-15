import torch
from collections import namedtuple
import torch._dynamo as dynamo
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

max_size = 50000

torch.set_float32_matmul_precision('medium')
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = False
#dynamo.config.recompile_limit = 1024 # or a higher value
#dynamo.config.accumulated_recompile_limit = 1024 # or a higher value
torch.cuda.empty_cache()
#dynamo.disable()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = GradScaler()

LossSettings = namedtuple('LossSettings', [
    'dice_weight', 'mse_weight', 'boundary_weight',
    'focal_weight', 'focal_alpha', 'focal_gamma'
])

loss_settings = LossSettings(
    dice_weight=10.0,
    mse_weight=0.0,
    boundary_weight=0.3,
    focal_weight=0.0,
    focal_alpha=0.25,
    focal_gamma=2.0
)

SegmentationHyperparams = namedtuple('SegmentationHyperparams', [
    'num_epochs', 'batch_size', 'learning_rate',
    'train_percent', 'optimizer_class'
])

segmentation_hyperparams = SegmentationHyperparams(
    num_epochs=1,
    batch_size=10,
    learning_rate=7e-5,
    train_percent=0.95,
    optimizer_class=torch.optim.AdamW,
)

learning_rate_gamma=1.0

num_workers=1

data_path = "/home/Adithya/Documents/noise_source_prog/paths.json"

levels = [[1,2,3,4]]
display_levels = [1,2,3,4]
print(f"training levels: {levels}")

image_size=128
patch_size=4
in_channels=1
out_channels=1
embed_size=768
num_blocks=12
num_heads=8
dropout=0.2
output_size=32

print_every_batches = 200

save_every_epoch = True

load_from = None
save_to = "/home/Adithya/Documents/ves/letter_visualization_model/new.pth"
display_from = save_to
