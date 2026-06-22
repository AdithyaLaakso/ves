import os
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler

import dataset
import settings
import training_recovery
from loss import MetaLoss
from model import build_model
from torch.amp.grad_scaler import GradScaler

import faulthandler

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

device = settings.device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(inputs, targets, device):
    non_blocking = device.type == "cuda"
    inputs = inputs.to(device, non_blocking=non_blocking)

    if isinstance(targets, tuple):
        targets = tuple(
            target.to(device, non_blocking=non_blocking)
            if torch.is_tensor(target)
            else target
            for target in targets
        )
    elif torch.is_tensor(targets):
        targets = targets.to(device, non_blocking=non_blocking)

    return inputs, targets


def signal_handler(sig, frame):
    print(f"Caught {sig} {frame}")
    print("Cleaning up...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    scheduler,
    *,
    epoch=0,
    start_batch=0,
    global_step=0,
    train_loss_total=0.0,
    train_loss_samples=0,
):
    last_checkpoint_time = time.monotonic()

    model.train()

    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx < start_batch:
            continue

        inputs, targets = move_batch_to_device(inputs, targets, device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, epoch=epoch)

        batch_size = inputs.size(0)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1

        train_loss_total += loss.item() * batch_size
        train_loss_samples += batch_size

        should_save_for_batch = (
            settings.step_checkpoint_every_batches > 0
            and global_step % settings.step_checkpoint_every_batches == 0
        )
        elapsed_minutes = (time.monotonic() - last_checkpoint_time) / 60.0
        should_save_for_time = (
            settings.step_checkpoint_every_minutes > 0
            and elapsed_minutes >= settings.step_checkpoint_every_minutes
        )
        if should_save_for_batch or should_save_for_time:
            recovery_path = training_recovery.save_recovery_checkpoint(
                checkpoint_dir=settings.save_to_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                next_batch=batch_idx + 1,
                global_step=global_step,
                train_loss_total=train_loss_total,
                train_loss_samples=train_loss_samples,
                metadata={
                    "run_dir": str(settings.run_dir) if settings.run_dir else None,
                    "level": None,
                    "batch_size": settings.segmentation_hyperparams.batch_size,
                    "max_size": settings.max_size,
                    "size_profile": settings.size_profile["profile"],
                },
                keep_snapshots=settings.keep_step_checkpoints,
                write_numbered_snapshot=settings.keep_step_checkpoints > 0,
            )
            print(
                "Saved recovery checkpoint "
                f"epoch={epoch} next_batch={batch_idx + 1} "
                f"global_step={global_step} -> {recovery_path}"
            )
            last_checkpoint_time = time.monotonic()

    return train_loss_total / max(train_loss_samples, 1), global_step


def evaluate_epoch(model, loader, criterion, epoch=0):
    total_loss = 0.0
    samples = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = move_batch_to_device(inputs, targets, device)
            outputs = model(inputs)
            loss = criterion(outputs, targets, epoch=epoch)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            samples += batch_size

    return total_loss / max(samples, 1)


def split_indices(data):
    n_total = len(data)
    if n_total == 0:
        raise ValueError("Dataset is empty!")

    if not settings.split_by_document:
        generator = torch.Generator().manual_seed(settings.seed)
        shuffled = torch.randperm(n_total, generator=generator)
        n_train = int(settings.segmentation_hyperparams.train_percent * n_total)
        train_idx = shuffled[:n_train].tolist()
        test_idx = shuffled[n_train:].tolist()
        return train_idx, test_idx

    groups = list(data.grouped_indices().values())
    rng = random.Random(42)
    rng.shuffle(groups)

    train_target = settings.segmentation_hyperparams.train_percent * n_total
    train_idx = []
    test_idx = []

    for group in groups:
        target = train_idx if len(train_idx) < train_target else test_idx
        target.extend(group)

    if not test_idx:
        split_at = max(1, len(train_idx) // 5)
        test_idx = train_idx[-split_at:]
        train_idx = train_idx[:-split_at]

    return train_idx, test_idx


def build_sampler(data, indices):
    if not settings.sampler_strategy:
        return None

    sample_weights = data.sample_weights(settings.sampler_strategy)
    if sample_weights is None:
        return None

    subset_weights = torch.as_tensor([sample_weights[i] for i in indices], dtype=torch.double)
    generator = torch.Generator().manual_seed(settings.seed)
    return WeightedRandomSampler(
        weights=subset_weights,
        num_samples=len(indices),
        replacement=True,
        generator=generator,
    )


def train_model():
    seed_everything(settings.seed)
    model = build_model()
    model.to(device)

    optimizer = settings.segmentation_hyperparams.optimizer_class(
        model.parameters(),
        lr=settings.segmentation_hyperparams.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=settings.learning_rate_gamma
    )
    scaler = GradScaler(device)
    resume_state = None
    resume_epoch = 0
    resume_batch = 0
    global_step = 0
    resume_loss_total = 0.0
    resume_loss_samples = 0

    compiled_train_epoch = train_epoch
    compiled_evaluate_epoch = evaluate_epoch
    criterion = MetaLoss()

    if settings.mode == settings.CLASSIFICATION:
        criterion = torch.nn.CrossEntropyLoss()

    if settings.resume_training_state:
        resume_state = training_recovery.load_recovery_checkpoint(
            settings.resume_training_state,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        resume_epoch = int(resume_state.get("epoch", 0))
        resume_batch = int(resume_state.get("next_batch", 0))
        global_step = int(resume_state.get("global_step", 0))
        resume_loss_total = float(resume_state.get("train_loss_total", 0.0))
        resume_loss_samples = int(resume_state.get("train_loss_samples", 0))
        print(
            "Resumed training state "
            f"from {settings.resume_training_state} "
            f"at epoch={resume_epoch} next_batch={resume_batch} "
            f"global_step={global_step}"
        )

    print(f"training levels: {settings.levels}")
    schedule_step = 1
    for level in settings.levels:
        print(f"Training level: {level}")

        data = dataset.SegData(level=level)
        n_total = len(data)
        print(f"training with {n_total} items")

        train_idx, test_idx = split_indices(data)
        train_sampler = build_sampler(data, train_idx)

        train_loader = dataset.create_loader(
            Subset(data, train_idx),
            batch_size=settings.segmentation_hyperparams.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )

        test_loader = dataset.create_loader(
            Subset(data, test_idx),
            batch_size=settings.segmentation_hyperparams.batch_size,
            shuffle=False,
        )

        for epoch in range(resume_epoch, settings.segmentation_hyperparams.num_epochs):
            print(f"On step {schedule_step}")
            schedule_step += 1
            if torch.cuda.is_available():
                print(torch.cuda.memory_allocated() / 1e9, "GB allocated")

            start_batch = resume_batch if epoch == resume_epoch else 0
            train_loss_total = resume_loss_total if epoch == resume_epoch else 0.0
            train_loss_samples = resume_loss_samples if epoch == resume_epoch else 0
            train_loss, global_step = compiled_train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                scheduler,
                epoch=epoch,
                start_batch=start_batch,
                global_step=global_step,
                train_loss_total=train_loss_total,
                train_loss_samples=train_loss_samples,
            )
            resume_batch = 0
            resume_loss_total = 0.0
            resume_loss_samples = 0
            eval_loss = compiled_evaluate_epoch(
                model,
                test_loader,
                criterion,
                epoch=epoch,
            )

            print(
                f"Epoch {epoch + 1}/{settings.segmentation_hyperparams.num_epochs} | "
                f"train_loss={train_loss:.4f} val_loss={eval_loss:.4f}"
            )

            scheduler.step()

            if settings.save_every_epoch:
                os.makedirs(settings.save_to_dir, exist_ok=True)
                if isinstance(level, list):
                    path = f"{settings.save_to_dir}/{level[0]}-{level[-1]}-{epoch + 1}.pth"
                else:
                    path = f"{settings.save_to_dir}/{level}-{epoch + 1}.pth"

                torch.save(model.state_dict(), path)
                print(f"Saved model for level {level} -> {path}")

        optimizer = settings.segmentation_hyperparams.optimizer_class(
            model.parameters(),
            lr=settings.segmentation_hyperparams.learning_rate,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


if __name__ == "__main__":
    print(f"training on {device}")

    model = train_model()
    path = settings.save_to
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")
