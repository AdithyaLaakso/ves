from collections import namedtuple
from pathlib import Path
import faulthandler
import json
import logging
import os
import sys

import torch
import torch._dynamo as dynamo
import torch.multiprocessing as mp

import constants

# logging.getLogger("torch._dynamo").setLevel(logging.DEBUG)

mp.set_start_method("spawn", force=True)

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True
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

ROOT = Path(__file__).resolve().parents[1]
base_path = str(ROOT)

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _select_device() -> torch.device:
    requested = os.getenv("VES_DEVICE")
    if requested:
        return torch.device(requested)
    if _env_flag("VES_FORCE_CPU"):
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


smoke_test = _env_flag("VES_SMOKE_TEST")
device: torch.device = _select_device()
max_size = _env_int("VES_MAX_SIZE", 64 if smoke_test else 0) or None

if smoke_test:
    torch.set_num_threads(_env_int("VES_TORCH_THREADS", 2))
    torch.set_num_interop_threads(_env_int("VES_TORCH_INTEROP_THREADS", 1))

LossSettings = namedtuple(
    "LossSettings",
    [
        "dice_weight",
        "mse_weight",
        "boundary_weight",
        "focal_weight",
        "focal_alpha",
        "focal_gamma",
        "class_weight",
        "class_weight_delta",
    ],
)

SegmentationHyperparams = namedtuple(
    "SegmentationHyperparams",
    [
        "num_epochs",
        "batch_size",
        "learning_rate",
        "train_percent",
        "optimizer_class",
    ],
)

segmentation_hyperparams = SegmentationHyperparams(
    num_epochs=_env_int("VES_NUM_EPOCHS", 1 if smoke_test else 5),
    batch_size=_env_int("VES_BATCH_SIZE", 4 if smoke_test else 64),
    learning_rate=1e-4,
    train_percent=0.80,
    optimizer_class=torch.optim.AdamW,
)

learning_rate_gamma = 0.975

num_workers = _env_int("VES_NUM_WORKERS", 0)
persistent_workers = False

data_path = str(ROOT / "data" / "alpub_v2_manifest.json")
add_to_path = str(ROOT)
data_format = "auto"

track_levels = False
levels = [0]
display_levels = levels[0] if levels else [0]

split_by_document = True
sampler_strategy = "document_inv_sqrt"

image_size = 96
patch_sizes = (8, 24)  # coarse, fine
in_channels = 1
out_channels = 1
embed_size = 300
num_blocks = 15
num_heads = 15
dropout = 0.2
input_size = 96
output_size = 24
use_gradient = not smoke_test

letters = tuple(constants.greek_letters.keys())
letter_to_idx = constants.greek_letters
num_classes = len(letter_to_idx)

print_every_batches = 1

save_every_epoch = True
save_to = "./new.pth"
load_from = None
display_from = save_to

log_dir = "./logs/"
print(f"logging to: {log_dir}")
save_to_dir = "./checkpoints/"

meta_div_weight = 0.0
meta_f_weight = 1.0
meta_b_weight = 2.0
meta_d_weight = 3.0
meta_c_weight = 1.0
meta_m_weight = 0.0
meta_s = 1.0

freeze_class = False
freeze_recon = False
freeze_shared = False

loss_settings = LossSettings(
    dice_weight=0.0,
    mse_weight=1.0,
    boundary_weight=0.000,
    focal_weight=10.0,
    class_weight=2.0,
    class_weight_delta=0.00000,
    focal_alpha=0.2,
    focal_gamma=2.0,
)

print(loss_settings)
print(segmentation_hyperparams)


def load_manifest_metadata() -> dict:
    manifest_path = Path(data_path)
    if not manifest_path.exists():
        return {}

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    if "records" not in data:
        return {}

    return data


manifest_metadata = load_manifest_metadata()

RECONSTRUCTION = 0
CLASSIFICATION = 1
MULTITASK = 2
mode = MULTITASK
