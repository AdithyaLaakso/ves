import torch
from collections import namedtuple
import torch._dynamo as dynamo
from torch.amp import  GradScaler
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

torch.backends.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"

dynamo.config.recompile_limit = 8
dynamo.config.accumulated_recompile_limit = 8
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.capture_scalar_outputs = True

torch.cuda.empty_cache()
#dynamo.disable()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_size = 10000000000

LossSettings = namedtuple('LossSettings', [
    'dice_weight', 'mse_weight', 'boundary_weight',
    'focal_weight', 'focal_alpha', 'focal_gamma'
])

loss_settings = LossSettings(
    dice_weight=1.0,
    mse_weight=0.0,
    boundary_weight=5.0,
    focal_weight=10.0,
    focal_alpha=0.2,
    focal_gamma=2.0
)

SegmentationHyperparams = namedtuple('SegmentationHyperparams', [
    'num_epochs', 'batch_size', 'learning_rate',
    'train_percent', 'optimizer_class'
])

segmentation_hyperparams = SegmentationHyperparams(
    num_epochs=3,
    batch_size=200,
    learning_rate=5e-2,
    train_percent=0.80,
    optimizer_class=torch.optim.AdamW,
)

learning_rate_gamma=0.75

num_workers=1

data_path = "/home/Adithya/Documents/noise_source_prog/paths.json"

# levels = [[1]]
# levels = [ [arr for arr in range(i,i+5)] for i in range(0, 25) if i%2==0] + [30]
levels = [5, 15, 30]
#display_levels = [1,2,3,4]
# display_levels = [i for i in range(4,8)]
display_levels = levels
# display_levels = [i for i in range(5, 31)]

image_size=128
patch_size=8
in_channels=1
out_channels=1
embed_size=968
num_blocks=12
num_heads=11
dropout=0.2
output_size=32

print_every_batches = 100

save_every_epoch = True

save_to = "/home/Adithya/Documents/ves/letter_visualization_model/new.pth"
display_from = save_to
load_from = None

save_to_dir = "/home/Adithya/Documents/ves/letter_visualization_model/checkpoints"
