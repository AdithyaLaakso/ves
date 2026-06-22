import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


VERSION = 1
LATEST_NAME = "recovery-latest.pt"
SNAPSHOT_RE = re.compile(r"recovery-epoch(?P<epoch>\d+)-batch(?P<batch>\d+)\.pt$")


def latest_checkpoint_path(checkpoint_dir: Path) -> Path:
    return Path(checkpoint_dir) / LATEST_NAME


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _numbered_checkpoint_path(checkpoint_dir: Path, epoch: int, next_batch: int) -> Path:
    return Path(checkpoint_dir) / f"recovery-epoch{epoch}-batch{next_batch}.pt"


def _atomic_save(payload: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _prune_snapshots(checkpoint_dir: Path, keep_snapshots: int) -> None:
    if keep_snapshots < 1:
        for path in Path(checkpoint_dir).glob("recovery-epoch*-batch*.pt"):
            path.unlink(missing_ok=True)
        return

    def sort_key(path: Path):
        match = SNAPSHOT_RE.match(path.name)
        if not match:
            return (-1, -1, path.name)
        return (int(match.group("epoch")), int(match.group("batch")), path.name)

    snapshots = sorted(Path(checkpoint_dir).glob("recovery-epoch*-batch*.pt"), key=sort_key)
    for path in snapshots[:-keep_snapshots]:
        path.unlink(missing_ok=True)


def build_recovery_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    next_batch: int,
    global_step: int,
    train_loss_total: float = 0.0,
    train_loss_samples: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "train_loss_total": train_loss_total,
        "train_loss_samples": train_loss_samples,
        "rng_state": capture_rng_state(),
        "metadata": metadata or {},
    }


def save_recovery_checkpoint(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    next_batch: int,
    global_step: int,
    train_loss_total: float = 0.0,
    train_loss_samples: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    keep_snapshots: int = 0,
    write_numbered_snapshot: bool = False,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    payload = build_recovery_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        next_batch=next_batch,
        global_step=global_step,
        train_loss_total=train_loss_total,
        train_loss_samples=train_loss_samples,
        metadata=metadata,
    )

    latest_path = latest_checkpoint_path(checkpoint_dir)
    _atomic_save(payload, latest_path)

    if write_numbered_snapshot:
        numbered_path = _numbered_checkpoint_path(checkpoint_dir, epoch, next_batch)
        _atomic_save(payload, numbered_path)
        _prune_snapshots(checkpoint_dir, keep_snapshots)

    return latest_path


def load_recovery_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    map_location: Optional[torch.device] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint
