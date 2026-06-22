import os
import random
import signal
import sys

import torch
from torch.utils.data import Subset, WeightedRandomSampler

import dataset
import settings
from loss import MetaLoss
from model import build_model
from torch.amp.grad_scaler import GradScaler

import faulthandler

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

device = settings.device


def signal_handler(sig, frame):
    print(f"Caught {sig} {frame}")
    print("Cleaning up...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def train_epoch(model, loader, optimizer, criterion, scaler, epoch=0):
    total_loss = 0.0
    samples = 0

    model.train()

    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, epoch=epoch)

        batch_size = inputs.size(0)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size
        samples += batch_size

    return total_loss / max(samples, 1)


def evaluate_epoch(model, loader, criterion, epoch=0):
    total_loss = 0.0
    samples = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
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
        shuffled = torch.randperm(n_total)
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
    return WeightedRandomSampler(
        weights=subset_weights,
        num_samples=len(indices),
        replacement=True,
    )


def train_model():
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

    compiled_train_epoch = train_epoch
    compiled_evaluate_epoch = evaluate_epoch
    criterion = MetaLoss()

    if settings.mode == settings.CLASSIFICATION:
        criterion = torch.nn.CrossEntropyLoss()

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

        for epoch in range(settings.segmentation_hyperparams.num_epochs):
            print(f"On step {schedule_step}")
            schedule_step += 1
            if torch.cuda.is_available():
                print(torch.cuda.memory_allocated() / 1e9, "GB allocated")

            train_loss = compiled_train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                epoch=epoch,
            )
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
