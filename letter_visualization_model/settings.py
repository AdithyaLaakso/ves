import faulthandler
import torch
import torch._dynamo as dynamo
import torch.multiprocessing as mp

import constants

from collections import namedtuple
import os
import glob

mp.set_start_method('spawn', force=True)

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True

# torch.backends.fp32_precision = "tf32"
# torch.backends.cuda.matmul.fp32_precision = "tf32"
# torch.backends.cudnn.fp32_precision = "tf32"
# torch.backends.cudnn.conv.fp32_precision = "tf32"
# torch.backends.cudnn.rnn.fp32_precision = "tf32"

dynamo.config.recompile_limit = 8
dynamo.config.accumulated_recompile_limit = 8
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = False
torch._dynamo.config.disable = True
torch._dynamo.config.verbose = True
torch.cuda.empty_cache()
faulthandler.enable()

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_size = None

track_levels = True

LossSettings = namedtuple('LossSettings', [
    'dice_weight', 'mse_weight', 'boundary_weight',
    'focal_weight', 'focal_alpha', 'focal_gamma', "class_weight", "class_weight_delta"
])

SegmentationHyperparams = namedtuple('SegmentationHyperparams', [
    'num_epochs', 'batch_size', 'learning_rate',
    'train_percent', 'optimizer_class'
])

segmentation_hyperparams = SegmentationHyperparams(
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    train_percent=1.00,
    optimizer_class=torch.optim.AdamW,
)

# track_levels = True
learning_rate_gamma=0.00

num_workers=0

data_path = "/home/Adithya/Documents/noise_source_prog/paths.json"
add_to_path = ""
# data_path = "/home/Adithya/Documents/synthetic_ct_images/paths.json"
# add_to_path = "/home/Adithya/Documents/"

levels = [10]

display_levels = [5]
# display_levels = [i for i in range ()]

image_size=128
patch_sizes=(16, 8) # coarse, fine
#patch_size=4
in_channels=1
out_channels=1
embed_size=300
num_blocks=30
num_heads=10
dropout=0.2
output_size=32
use_gradient=True
num_classes=24

print_every_batches = 1

save_every_epoch = True
save_to = "/home/Adithya/Documents/ves/letter_visualization_model/new.pth"
# load_from = "/home/Adithya/Documents/ves/letter_visualization_model/start.pth"
load_from = None

# display_from = save_to
display_from = save_to

save_to_dir = "/home/Adithya/Documents/ves/letter_visualization_model/checkpoints"

# log_dir = None
stamp_files = glob.glob("*.stamp")
log_dir = "./logs/" + os.path.splitext(stamp_files[0])[0] if len(stamp_files) == 1 else None
print (f"logging to: {log_dir}")

meta_div_weight = 0.0
meta_f_weight = 1.0
meta_b_weight = 2.0
meta_d_weight = 3.0
meta_c_weight = 1.0
meta_m_weight = 0.0
meta_s = 1.0

loss_settings = LossSettings(
    dice_weight=0.0,
    mse_weight=3.0,
    boundary_weight=1.0,
    focal_weight=0.0,
    class_weight=1.00,
    class_weight_delta=0.00000,
    focal_alpha=0.2,
    focal_gamma=2.0
)

print(loss_settings)
print(segmentation_hyperparams)

letters = constants.greek_letters.keys()
letter_to_idx = constants.greek_letters

RECONSTRUCTION = 0
CLASSIFICATION = 1
MULTITASK = 2
mode = MULTITASK
