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

torch.backends.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

dynamo.config.recompile_limit = 8
dynamo.config.accumulated_recompile_limit = 8
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.capture_scalar_outputs = True

torch._dynamo.config.verbose = True

os.environ["TORCH_LOGS"] = "+dynamo"

os.environ["TORCHDYNAMO_VERBOSE"] = "1"

torch.cuda.empty_cache()
#dynamo.disable()

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_size = None

track_levels = True

LossSettings = namedtuple('LossSettings', [
    'dice_weight', 'mse_weight', 'boundary_weight',
    'focal_weight', 'focal_alpha', 'focal_gamma'
])

SegmentationHyperparams = namedtuple('SegmentationHyperparams', [
    'num_epochs', 'batch_size', 'learning_rate',
    'train_percent', 'optimizer_class'
])

segmentation_hyperparams = SegmentationHyperparams(
    num_epochs=10,
    batch_size=8,
    learning_rate=8e-4,
    train_percent=0.80,
    optimizer_class=torch.optim.AdamW,
)

# track_levels = True

learning_rate_gamma=1.0

num_workers=1

data_path = "/home/Adithya/Documents/noise_source_prog/paths.json"
add_to_path = "/home/Adithya/Documents/"
<<<<<<< HEAD
data_path = "/home/Adithya/Documents/synthetic_ct_images/paths.json" #"C:\\Users\\randt\\OneDrive\\Documents\\Vesuvius\\ves\\synthetic_ct_images\\paths.json" 
=======
# data_path = "/home/Adithya/Documents/synthetic_ct_images/paths.json"
>>>>>>> fd80bc7b1 (commiting before adding logs)
# add_to_path = "/home/Adithya/Documents/"
add_to_path = ""

# levels = [[1]]
# levels = [ [arr for arr in range(i,i+5)] for i in range(0, 25) if i%2==0] + [30]
levels = [21]
# levels = [[j for j in range(i // 2, i+1)] for i in range (0, 17)]
# levels = [0]

#display_levels = [1,2,3,4]
# display_levels = [i for i in range(4,8)]
# display_levels = [i for i in range (0, 17)]
display_levels = [20, 21]
# display_levels = [i for i in range(5, 31)]

image_size=128
patch_sizes=(4, 8)
#patch_size=4
in_channels=1
out_channels=1
embed_size=800
num_blocks=8
num_heads=8
dropout=0.2
output_size=32
use_gradient=True

print_every_batches = 1
save_every_print = True

save_every_epoch = True
save_to = "/home/Adithya/Documents/ves/letter_visualization_model/new.pth"
# display_from = save_to
display_from = "/home/Adithya/Documents/ves/letter_visualization_model/checkpoints/21-5.pth"

load_from = save_to

save_to_dir = "/home/Adithya/Documents/ves/letter_visualization_model/checkpoints"

# log_dir = None
stamp_files = glob.glob("*.stamp")
log_dir = "./logs/" + os.path.splitext(stamp_files[0])[0] if len(stamp_files) == 1 else None
print (f"logging to: {log_dir}")

meta_div_weight = 0.0
meta_f_weight = 1.0
meta_b_weight = 2.0
meta_d_weight = 3.0
meta_m_weight = 0.0
meta_s = 1.0

# meta_op_scale = meta_add_weight + meta_div_weight
# meta_add_weight /= meta_op_scale
# meta_div_weight /= meta_op_scale

# meta_multiloss_scale = meta_f_weight + meta_b_weight + meta_d_weight + meta_m_weight
# meta_f_weight /= meta_multiloss_scale
# meta_b_weight /= meta_multiloss_scale
# meta_m_weight /= meta_multiloss_scale
# meta_d_weight /= meta_multiloss_scale

loss_settings = LossSettings(
    dice_weight=1.0,
    mse_weight=1.0,
    boundary_weight=1.0,
    focal_weight=1.0,
    focal_alpha=0.2,
    focal_gamma=2.0
)

print(loss_settings)
print(segmentation_hyperparams)

letters = constants.greek_letters.keys()
# letters = ["ALPHA"]
