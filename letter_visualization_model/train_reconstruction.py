import os
import torch
import torch.nn.functional as F

import settings
from dataset import SingleLetterSegmentationDataLoader, SingleLetterSegmentationDataset  # Updated imports
from model import VisionTransformerForSegmentation
from loss import BinarySegmentationLoss

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"training on {device}")

@torch.compile
def train_model(batch_size, learning_rate, num_epochs, train_percent, optimizer_class, bias_factor=3.0, pretrained_model=None):
    """
    Train the segmentation model with the fixed data pipeline.
    """

    # Initialize model for segmentation: 1 channel in, 1 channel out, 32x32 output
    model = VisionTransformerForSegmentation()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings.learning_rate_gamma)

    # Use segmentation-specific loss
    criterion = BinarySegmentationLoss()

    for level in settings.levels:
        print(f"Training level: {level}")

        # Load dataset for this level using the new segmentation dataset
        train_test_data = SingleLetterSegmentationDataset(level=level)
        dataset = train_test_data.dataset

        # Split dataset into train and test sets
        train_size = int(train_percent * len(dataset))
        indices = torch.randperm(len(dataset))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Create train and test datasets
        train_dataset = SingleLetterSegmentationDataset(level=level)
        train_dataset.dataset = [dataset[i] for i in train_indices]

        test_dataset = SingleLetterSegmentationDataset(level=level)
        test_dataset.dataset = [dataset[i] for i in test_indices]

        train_loader = SingleLetterSegmentationDataLoader(
                train_dataset,
                shuffle=True,
                batch_size=batch_size,
                device=device,
                create_synthetic_channels=False  # CHANGED: Disable 8-channel creation
        )

        test_loader = SingleLetterSegmentationDataLoader(
                test_dataset,
                shuffle=False,
                batch_size=batch_size,
                device=device,
                create_synthetic_channels=False  # CHANGED: Disable 8-channel creation
        )

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            model.train()
            running_loss = 0.0
            running_components = {
                'dice_loss': 0.0,
                'ssim_loss': 0.0,
                'accuracy_loss': 0.0,
                'total_loss': 0.0
            }
            batch_count = 0

            for inputs, targets, labels in train_loader:
                try:
                    optimizer.zero_grad()

                    # CHANGED: Ensure inputs are single channel 128x128
                    # Convert RGB to grayscale if needed
                    if inputs.shape[1] == 3:  # RGB input
                        inputs = torch.mean(inputs, dim=1, keepdim=True)  # Convert to grayscale

                    # Ensure correct input format: [batch, 1, 128, 128]
                    if inputs.shape[1] != 1:
                        raise ValueError(f"Expected single channel input, got {inputs.shape[1]} channels")

                    # Forward pass - inputs are now 1-channel 128x128
                    # targets are 1-channel 32x32 binary masks
                    outputs = model(inputs)

                    # Compute loss
                    loss = criterion.forward(outputs, targets)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Track losses
                    running_loss += loss.item()
                    batch_count += 1

                except Exception as e:
                    print(f"Error in training batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Calculate epoch averages
            if batch_count > 0:
                epoch_loss = running_loss / batch_count
                for key in running_components:
                    running_components[key] /= batch_count

                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
            else:
                print(f"No valid batches in epoch {epoch+1}")
                continue

            # Evaluation on test set
            model.eval()
            test_loss = 0.0

            test_batch_count = 0
            with torch.no_grad():
                for inputs, targets, labels in test_loader:
                    try:
                        # CHANGED: Apply same input processing for test data
                        if inputs.shape[1] == 3:  # RGB input
                            inputs = torch.mean(inputs, dim=1, keepdim=True)  # Convert to grayscale

                        if inputs.shape[1] != 1:
                            raise ValueError(f"Expected single channel input, got {inputs.shape[1]} channels")

                        outputs = model(inputs)
                        loss = criterion.forward(outputs, targets)
                        test_loss += loss.item()
                        test_batch_count += 1

                    except Exception as e:
                        print(f"Error in test batch: {e}")
                        continue

            if test_batch_count > 0:
                avg_test_loss = test_loss / test_batch_count
                print(f"Test Loss: {avg_test_loss:.4f}")
            else:
                print("No valid test batches")
                avg_test_loss = float('inf')

            scheduler.step()

    # Save the trained model
    optimizer_name = optimizer_class.__name__ if hasattr(optimizer_class, "__name__") else str(optimizer_class).split(".")[-1].split("'")[0]
    os.makedirs("trained_segmentation_models", exist_ok=True)
    model_path = f"trained_segmentation_models/trained_segmentation_model_{optimizer_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    return epoch_loss if 'epoch_loss' in locals() else 0.0, avg_test_loss if 'avg_test_loss' in locals() else 0.0

@torch.compile
def test_data_pipeline():
    """Test the data pipeline before training."""
    print("Testing data pipeline...")

    try:
        # Test dataset creation
        dataset = SingleLetterSegmentationDataset(level=0)
        print(f"✓ Dataset loaded: {len(dataset.dataset)} items")

        # Test dataloader - CHANGED: No synthetic channels
        dataloader = SingleLetterSegmentationDataLoader(
            dataset,
            device=device,
            create_synthetic_channels=False  # CHANGED: Disable synthetic channels
        )

        # Test one batch
        for inputs, targets, labels in dataloader:
            print(f"✓ Raw input shape: {inputs.shape}")
            print(f"✓ Target shape: {targets.shape} (expected: [batch_size, 1, 32, 32])")
            print(f"✓ Labels count: {len(labels)}")
            print(f"✓ Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            print(f"✓ Target range: [{targets.min():.3f}, {targets.max():.3f}]")

            # CHANGED: Convert to single channel if needed
            if inputs.shape[1] == 3:  # RGB input
                inputs = torch.mean(inputs, dim=1, keepdim=True)  # Convert to grayscale
                print(f"✓ Converted to grayscale: {inputs.shape} (expected: [batch_size, 1, 128, 128])")

            # Verify single channel
            if inputs.shape[1] != 1:
                raise ValueError(f"Expected single channel input after conversion, got {inputs.shape[1]} channels")

            # Test model forward pass - CHANGED: Use correct parameters
            model = VisionTransformerForSegmentation()
            model.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                print(f"✓ Model output shape: {outputs.shape} (expected: [batch_size, 1, 32, 32])")
                print(f"✓ Model output range: [{outputs.min():.3f}, {outputs.max():.3f}]")

            break

        print("✓ Data pipeline test passed!")
        return True

    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    params = settings.segmentation_hyperparams._asdict()  # Convert to dict first
    train_loss, test_loss = train_model(**params)
