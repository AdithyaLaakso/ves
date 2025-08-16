import os
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.checkpoint import checkpoint_sequential

import settings
import dataset
from model import VisionTransformerForSegmentation, create_memory_efficient_vit
from loss import BinarySegmentationLoss
from torch.cuda.amp import autocast, GradScaler

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"training on {device}")

@torch.compile
def train_epoch(model, loader, optimizer, criterion):
    total_loss = torch.zeros(1, device=device)
    n_batches = 0

    model.train()

    for inputs, targets in loader:
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # loss.backward()
        # optimizer.step()
        # scaler = torch.cuda.amp.GradScaler()
        settings.scaler.scale(loss).backward()
        settings.scaler.step(optimizer)
        settings.scaler.update()

        total_loss += loss.detach()
        n_batches += 1

    return total_loss / max(n_batches, 1)

@torch.compile
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
            loss = criterion(outputs, targets)

            total_loss += loss
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model():
    # Create or load model
    if getattr(settings, "load_from", None):
        print(f"Loading model from {settings.load_from}")
        model = VisionTransformerForSegmentation()
        model.load_state_dict(torch.load(settings.load_from, map_location=device))
    else:
        print("Creating new model")
        model = create_memory_efficient_vit()

    model.to(device)

    optimizer = settings.segmentation_hyperparams.optimizer_class(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings.learning_rate_gamma)

    scaler = GradScaler("cuda")

    compiled_train_epoch = torch.compile(train_epoch)
    compiled_eval_epoch = torch.compile(evaluate_epoch)

    for level in settings.levels:
        print(f"Training level: {level}")

        data = dataset.SegData(level=level)

        n_total = len(data)
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

        criterion = BinarySegmentationLoss()

        for epoch in range(settings.segmentation_hyperparams.num_epochs):
            test_loss = train_epoch(model, train_loader, optimizer, criterion)

            train_loss = evaluate_epoch(model, test_loader, criterion)

            print(f"Epoch {epoch+1}/{settings.segmentation_hyperparams.num_epochs} | "
                  f"Train Loss: {train_loss/len(train_loader)} | "
                  f"Test Loss: {test_loss/len(test_loader)}")

            scheduler.step()

        # Save per-level model
        if settings.save_every_epoch:
            os.makedirs("trained_segmentation_models", exist_ok=True)
            path = f"trained_segmentation_models/model_level{level}.pth"
            torch.save(model.state_dict(), path)
            print(f"Saved model for level {level} -> {path}")
    return model

if __name__ == "__main__":
    model = train_model()
    path = settings.save_to
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")
