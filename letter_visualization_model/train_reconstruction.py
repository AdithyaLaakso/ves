import os
import signal
import sys

import torch
from torch.utils.data import Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

import settings
import dataset
#from model import create_enhanced_memory_efficient_vit as create_memory_efficient_vit
from model import build_model
from loss import MetaLoss
from torch.amp.grad_scaler import GradScaler

#don't cook my vram and require a reboot if I SIGINT

def signal_handler(sig, frame):
    print('Cleaning up...')
    torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def train_epoch(model, loader, optimizer, criterion, scaler, epoch=0):
    total_loss = torch.zeros(1, device=device)
    n_batches = 0

    model.train()

    for inputs, targets in loader:
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        outputs = model(inputs)
        # loss = criterion(inputs, outputs, targets)
        loss = criterion(outputs, targets, epoch=epoch)

        # print(loss)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss
        n_batches += 1
        # if step % settings.print_every_batches == 0:
        #     path = f"{settings.save_to_dir}/{step // settings.print_every_batches}.pth"
        #     torch.save(model.state_dict(), path)

    return total_loss / max(n_batches, 1)

def evaluate_epoch(model, loader, criterion, epoch=0):
    total_loss = torch.zeros(1, device=device)
    n_batches = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            torch.cuda.empty_cache()
            outputs = model(inputs)
            loss = criterion(outputs, targets, epoch=epoch)

            total_loss += loss
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model():
    model = build_model()

    model.to(device)

    optimizer = settings.segmentation_hyperparams.optimizer_class(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings.learning_rate_gamma)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=1,        # first cycle length in epochs
    #     T_mult=2,      # each cycle doubles in length
    #     eta_min=5e-5   # minimum LR
    # )
    #
    scaler = GradScaler(device)

    compiled_train_epoch = torch.compile(train_epoch)
    compiled_evaluate_epoch = torch.compile(evaluate_epoch)

    # compiled_train_epoch = train_epoch
    # compiled_evaluate_epoch = evaluate_epoch
    criterion = MetaLoss()

    if settings.mode == settings.CLASSIFICATION:
        criterion = torch.nn.CrossEntropyLoss()

    print(f"training levels: {settings.levels}")
    schedule_step = 1
    for level in settings.levels:
        if level is None:
            continue

        print(f"Training level: {level}")

        data = dataset.SegData(level=level)

        n_total = len(data)
        print(f"training with {n_total} items")

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
            print(f"On step {schedule_step}")
            schedule_step += 1
            train_loss = compiled_train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                epoch=epoch
            )

            test_loss = compiled_evaluate_epoch(
                model,
                test_loader,
                criterion,
                epoch=epoch
            )

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
