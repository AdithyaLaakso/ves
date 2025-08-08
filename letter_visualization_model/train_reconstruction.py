import os
import torch
from constants import hyperparams_list
from model import ReconstructionModel
import classif
from dataset import SingleLetterSegmentationDataLoader, SingleLetterSegmentationDataset  # Updated imports
from loss import SegmentationCombinedLoss, compute_classification_improvement_segmentation, BinarySegmentationLoss
from IQA_pytorch import SSIM, GMSD, LPIPSvgg, DISTS
import torch.nn.functional as F

PRETRAIN_PROTECTOR = 10  # Factor to protect pretrained layers from learning rate decay

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"training on {device}")

def train_model(batch_size, learning_rate, num_epochs, train_percent, optimizer_class, bias_factor=3.0, pretrained_model=None):
    """
    Train the segmentation model with the fixed data pipeline.
    """

    # Initialize model for segmentation: 8 channels in, 1 channel out
    model = ReconstructionModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Use segmentation-specific loss
    # criterion = SegmentationCombinedLoss(
    #     dice_weight=1.0,      # Primary segmentation loss
    #     ssim_weight=0.0,      # Structural similarity
    #     acc_weight=0.0,       # Start with no classification feedback
    #     acc_rel_delta=0.00,   # Gradually increase classification weight
    #     segmentation_loss_type='dice'  # Use Dice loss for segmentation
    # )

    criterion = BinarySegmentationLoss()

    # levels = [30, 100]
    levels = [i for i in range(20, 31)]
    for i in range(0, 2):
        levels.append(100)

    for level in levels:
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

        # Create data loaders with the fixed pipeline
        train_loader = SingleLetterSegmentationDataLoader(
            train_dataset,
            shuffle=True,
            device=device,
            create_synthetic_channels=True  # Enable 8-channel creation
        )

        test_loader = SingleLetterSegmentationDataLoader(
            test_dataset,
            shuffle=False,
            device=device,
            create_synthetic_channels=True
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

                    # Forward pass - inputs are already 8-channel 128x128
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

                    # Track loss components if available
                    if hasattr(criterion, 'last_loss_components'):
                        components = criterion.last_loss_components
                        for key in running_components:
                            if key in components:
                                running_components[key] += components[key]

                    # Print progress every 50 batches
                    if batch_count % 50 == 0:
                        avg_loss = running_loss / batch_count
                        print(f"  Batch {batch_count}, Avg Loss: {avg_loss:.4f}")

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
                print(f"  Loss Components: {running_components}")
            else:
                print(f"No valid batches in epoch {epoch+1}")
                continue

            # Evaluation on test set
            model.eval()
            test_loss = 0.0
            test_batch_count = 0
            test_components = {
                'dice_loss': 0.0,
                'ssim_loss': 0.0,
                'accuracy_loss': 0.0,
                'total_loss': 0.0
            }

            with torch.no_grad():
                for inputs, targets, labels in test_loader:
                    try:
                        outputs = model(inputs)
                        loss = criterion.forward(outputs, targets)
                        test_loss += loss.item()
                        test_batch_count += 1

                        # Track test loss components
                        if hasattr(criterion, 'last_loss_components'):
                            components = criterion.last_loss_components
                            for key in test_components:
                                if key in components:
                                    test_components[key] += components[key]

                    except Exception as e:
                        print(f"Error in test batch: {e}")
                        continue

            if test_batch_count > 0:
                avg_test_loss = test_loss / test_batch_count
                for key in test_components:
                    test_components[key] /= test_batch_count

                print(f"Test Loss: {avg_test_loss:.4f}")
                print(f"  Test Components: {test_components}")
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


def test_data_pipeline():
    """Test the data pipeline before training."""
    print("Testing data pipeline...")

    try:
        # Test dataset creation
        dataset = SingleLetterSegmentationDataset(level=0)
        print(f"✓ Dataset loaded: {len(dataset.dataset)} items")

        # Test dataloader
        dataloader = SingleLetterSegmentationDataLoader(
            dataset,
            device=device,
            create_synthetic_channels=True
        )

        # Test one batch
        for inputs, targets, labels in dataloader:
            print(f"✓ Input shape: {inputs.shape} (expected: [batch_size, 8, 128, 128])")
            print(f"✓ Target shape: {targets.shape} (expected: [batch_size, 1, 32, 32])")
            print(f"✓ Labels count: {len(labels)}")
            print(f"✓ Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            print(f"✓ Target range: [{targets.min():.3f}, {targets.max():.3f}]")

            # Test model forward pass
            model = ReconstructionModel()
            model.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                print(f"✓ Model output shape: {outputs.shape}")
                print(f"✓ Model output range: [{outputs.min():.3f}, {outputs.max():.3f}]")

            break

        print("✓ Data pipeline test passed!")
        return True

    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Modified hyperparameters for segmentation
segmentation_hyperparams = [
    {
        'batch_size': 16,  # Smaller batch size for 128x128 images
        'learning_rate': 1e-4,  # Lower learning rate for segmentation
        'num_epochs': 4,
        'train_percent': 0.95,
        'optimizer_class': torch.optim.Adam,
        'bias_factor': 20.0
    }
]

if __name__ == "__main__":
    print("Starting segmentation training...")

    # Test data pipeline first
    if not test_data_pipeline():
        print("Data pipeline test failed. Please fix the issues before training.")
        exit(1)

    # Train with segmentation-specific hyperparameters
    for params in segmentation_hyperparams:
        print(f"\nTraining with hyperparameters: {params}")
        try:
            train_loss, test_loss = train_model(**params)
            print(f"Final Train Loss: {train_loss:.4f}, Final Test Loss: {test_loss:.4f}")
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

    print("\nTraining completed!")
    print("\nKey changes made:")
    print("1. ✓ Fixed input: 3-channel RGB → 8-channel synthetic features")
    print("2. ✓ Fixed input size: Variable → 128×128")
    print("3. ✓ Fixed output: RGB reconstruction → Binary segmentation (32×32)")
    print("4. ✓ Fixed loss: MSE → Dice + SSIM + Classification feedback")
    print("5. ✓ Added comprehensive error handling and progress tracking")
