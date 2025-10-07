import faulthandler
import torch
import torch._dynamo as dynamo
import torch.multiprocessing as mp

import constants

from collections import namedtuple
import sys
import os
import glob
import logging

logging.getLogger("torch._dynamo").setLevel(logging.DEBUG)

mp.set_start_method('spawn', force=True)

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True

# torch.backends.fp32_precision = "tf32"
# torch.backends.cuda.matmul.fp32_precision = "tf32"
# torch.backends.cudnn.fp32_precision = "tf32"
# torch.backends.cudnn.conv.fp32_precision = "tf32"
# torch.backends.cudnn.rnn.fp32_precision = "tf32"
#
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

dynamo.config.recompile_limit = 100
dynamo.config.accumulated_recompile_limit = 100
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = False
torch._dynamo.config.disable = True
torch._dynamo.config.verbose = True
torch.cuda.empty_cache()
faulthandler.enable()

base_path = "/home/Adithya/Documents/ves/letter_visualization_model/"

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
    num_epochs=20,
    batch_size=256,
    learning_rate=7e-3,
    train_percent=0.80,
    optimizer_class=torch.optim.AdamW,
)


learning_rate_gamma=1.1

num_workers=0
persistent_workers=False

# data_path = "/home/Adithya/Documents/noise_source_prog/paths.json"
# add_to_path = ""
data_path = "../data/paths.json"
add_to_path = "../data/"
# add_to_path = "../data/"

track_levels = True
levels = [[i for i in range(0,31)]]
# levels = [0]

# display_levels = levels[0]
display_levels = levels

image_size=128
patch_sizes=(8, 32) # coarse, fine
#patch_size=4
in_channels=1
out_channels=1
embed_size=300
num_blocks=15
num_heads=10
dropout=0.2
input_size=128
output_size=32
use_gradient=True
num_classes=24

print_every_batches = 1

save_every_epoch = True
save_to = "./new.pth"
load_from = "./start.pth"
display_from = save_to

stamp_files = glob.glob("*.stamp")
stamp_path = os.path.splitext(stamp_files[0])[0] if len(stamp_files) == 1 else None
if not stamp_path:
    print("bruh no stamp_path")
    sys.exit()

stamp_path = stamp_path[0]
log_dir = "./logs/" + stamp_path
print (f"logging to: {log_dir}")
save_to_dir = "./checkpoints/" + stamp_path

meta_div_weight = 0.0
meta_f_weight = 1.0
meta_b_weight = 2.0
meta_d_weight = 3.0
meta_c_weight = 1.0
meta_m_weight = 0.0
meta_s = 1.0

freeze_class = False
freeze_recon = False

loss_settings = LossSettings(
    dice_weight=0.0,
    mse_weight=1.0,
    boundary_weight=0.1,
    focal_weight=10.0,
    class_weight=2.0,
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
