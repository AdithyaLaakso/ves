import os
import signal
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.checkpoint import checkpoint_sequential

import settings
import dataset
from model import VisionTransformerForSegmentation, create_memory_efficient_vit
from loss import BinarySegmentationLoss
from torch.amp import autocast, GradScaler


#don't cook my vram and require a reboot if I SIGINT
def signal_handler(sig, frame):
    print('Cleaning up...')
    torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def train_epoch(model, loader, optimizer, criterion, scaler=None):
    total_loss = torch.zeros(1, device=device)
    n_batches = 0

    model.train()

    step = 0
    for inputs, targets in loader:
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss, (d, b, f) = criterion(outputs, targets)

        criterion.update_running_stats(d,b,f,step)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach()
        n_batches += 1
        step += 1

    return total_loss / max(n_batches, 1)

def evaluate_epoch(model, loader, criterion):
    total_loss = torch.zeros(1, device=device)
    n_batches = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            torch.cuda.empty_cache()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss, _ = criterion(outputs, targets)

            total_loss += loss
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model():
    model = create_memory_efficient_vit(use_fp16=True)

    model.to(device)

    optimizer = settings.segmentation_hyperparams.optimizer_class(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings.learning_rate_gamma)

    scaler = GradScaler(device)

    compiled_train_epoch = torch.compile(train_epoch)
    compiled_eval_epoch = torch.compile(evaluate_epoch)

    criterion = BinarySegmentationLoss()

    print(f"training levels: {settings.levels}")
    for level in settings.levels:
        if level is None:
            continue

        print(f"Training level: {level}")

        data = dataset.SegData(level=level)

        n_total = len(data)

        if n_total == 0:
            raise ValueError("Dataset is empty!")

        n_train = int(settings.segmentation_hyperparams.train_percent * n_total)

        shuffled = torch.randperm(n_total)
        train_idx = shuffled[:n_train].tolist()
        test_idx = shuffled[n_train:].tolist()

        train_loader = dataset.create_loader(
            Subset(data, train_idx),
            batch_size=settings.segmentation_hyperparams.batch_size,
            shuffle=True,
        )

        test_loader = dataset.create_loader(
            Subset(data, test_idx),
            batch_size=settings.segmentation_hyperparams.batch_size,
            shuffle=False,
        )


        for epoch in range(settings.segmentation_hyperparams.num_epochs):
            test_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler=scaler
            )

            train_loss = evaluate_epoch(model, test_loader, criterion)

            print(f"Epoch {epoch+1}/{settings.segmentation_hyperparams.num_epochs} | "
                  f"Train Loss: {train_loss/len(train_loader)} | "
                  f"Test Loss: {test_loss/len(test_loader)}")

            scheduler.step()

            # Save per-level model
            if settings.save_every_epoch:
                os.makedirs(settings.save_to_dir, exist_ok=True)
                if isinstance(level, list):
                    path = f"{settings.save_to_dir}/{level[0]}-{level[-1]}-{epoch+1}.pth"
                else:
                    path = f"{settings.save_to_dir}/{level}-{epoch+1}.pth"

                torch.save(model.state_dict(), path)
                print(f"Saved model for level {level} -> {path}")

    return model

if __name__ == "__main__":
    device = settings.device
    print(f"training on {device}")

    model = train_model()
    path = settings.save_to
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")
