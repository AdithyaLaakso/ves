import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from model import SingleLetterModel
from dataset import SingleLetterDataset, SingleLetterDataLoader
from class_model import SingleLetterModel

# Hyperparameters
batch_size = 16
learning_rate = 1e-3
num_epochs = 10
train_percent = 0.8

# Loss weighting parameters
alpha = 1.0  # Weight for reconstruction loss (how similar denoised image is to clean image)
beta = 0.25   # Weight for classification loss (how well classifier performs on denoised image)
beta_delta = 0.1

# Dataset
train_test_data = SingleLetterDataset()
dataset = train_test_data.dataset

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"training on {device}")

# create criterion that compares output and target images with bias against white pixels
class BiasedMSELoss(torch.nn.Module):
    def __init__(self, bias_factor=5.0):
        super().__init__()
        self.bias_factor = bias_factor

    def forward(self, output, target):
        mse = (output - target) ** 2
        penalty_by_pixel = output * self.bias_factor + 1
        mse *= penalty_by_pixel
        reg = torch.mean(output)
        # make var penalty based on the variance of each output channel
        var_penalty = torch.var(output, dim=(2, 3)).mean()
        return mse.mean() - var_penalty * 5

# Models
# Create a denoising model that outputs cleaned images at 256x256 resolution
class GreekLetterDenoiser(torch.nn.Module):
    def __init__(self, input_channels=3, target_size=256):
        super(GreekLetterDenoiser, self).__init__()
        self.target_size = target_size

        # Encoder (downsampling path)
        self.encoder = torch.nn.Sequential(
            # First block - maintain spatial dimensions
            torch.nn.Conv2d(input_channels, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),

            # Second block
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )

        # Decoder (upsampling path)
        self.decoder = torch.nn.Sequential(
            # First decoder block
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),

            # Final layer to output clean image
            torch.nn.Conv2d(64, input_channels, 3, padding=1),
            torch.nn.Sigmoid()  # Assuming images are normalized to [0,1]
        )

    def forward(self, x):
        # Store original size for potential resizing
        original_size = x.shape[-2:]

        # If input is not target_size, resize it
        if original_size[0] != self.target_size or original_size[1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)

        # Pass through encoder
        encoded = self.encoder(x)
        # Pass through decoder to get denoised image
        denoised = self.decoder(encoded)

        return denoised

# Initialize the denoising model for 256x256 images
model = GreekLetterDenoiser(input_channels=3, target_size=256)  # Adjust channels based on your images
model.to(device)

# Load your pre-trained Greek letter classifier
classifier = SingleLetterModel(num_classes=25)  # 24 Greek letters + 1 "no letter"
classifier.load_state_dict(torch.load("class.pth"))  # Uncomment and provide path
classifier.to(device)
classifier.eval()  # Set to eval mode - we don't want to train the classifier

# Loss functions and optimizer
reconstruction_criterion = BiasedMSELoss()
classification_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Split dataset into train and test sets
train_size = int(train_percent * len(dataset))
test_size = len(dataset) - train_size
indices = torch.randperm(len(dataset))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_dataset = [dataset[i] for i in train_indices]
test_dataset = [dataset[i] for i in test_indices]

train_loader = SingleLetterDataLoader(train_dataset, batch_size=batch_size, shuffle=True, device=device)
test_loader = SingleLetterDataLoader(test_dataset, batch_size=batch_size, shuffle=False, device=device)

def compute_classification_accuracy(denoised_images, true_labels, classifier, label_to_idx=None):
    with torch.no_grad():
        predictions = classifier(denoised_images)
        predicted_labels = torch.argmax(predictions, dim=1)

        # Handle string labels
        if isinstance(true_labels, list) and len(true_labels) > 0 and isinstance(true_labels[0], np.ndarray):
            if label_to_idx is None:
                raise ValueError("label_to_idx mapping required for string labels")
            # Convert string labels to indices
            numeric_labels = [label_to_idx[str(label)] for label in true_labels]
            true_labels = torch.tensor(numeric_labels, device=denoised_images.device, dtype=torch.long)
        else:
            # Handle numeric labels
            if not isinstance(true_labels, torch.Tensor):
                true_labels = torch.tensor(true_labels, device=denoised_images.device)
            if true_labels.dim() > 1:
                true_labels = true_labels.squeeze()
            true_labels = true_labels.long()

        # Compute accuracy
        correct = (predicted_labels == true_labels).float()
        accuracy = correct.mean()
    return accuracy

def compute_combined_loss(denoised_images, target_images, true_labels, classifier, alpha, beta):
    """
    Compute weighted combination of reconstruction loss and classification accuracy
    """

    # Extract string values from numpy arrays and get unique labels
    unique_labels_in_batch = list(set(str(label) for label in true_labels))
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels_in_batch))}
    denoised_images_resized = F.interpolate(denoised_images, size=(32, 32), mode='nearest')
    reconstruction_loss = reconstruction_criterion(denoised_images_resized, target_images)

    # Classification accuracy (converted to loss: 1 - accuracy)
    accuracy = compute_classification_accuracy(denoised_images, true_labels, classifier, label_to_idx=label_to_idx)
    classification_loss = 1.0 - accuracy  # Convert accuracy to loss

    # Combined loss
    total_loss = alpha * reconstruction_loss + beta * classification_loss

    return total_loss, reconstruction_loss, accuracy

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}, alpha/beta = {alpha}/{beta}")
    model.train()

    running_total_loss = 0.0
    running_recon_loss = 0.0
    running_accuracy = 0.0
    num_batches = 0

    for batch_data in train_loader:

        # Handle different possible dataset formats
        if len(batch_data) == 2:
            # Most likely: (images, labels) where labels are class indices
            data_item, label_item = batch_data

            if hasattr(label_item, 'shape') and len(label_item.shape) > 1:
                # If labels look like images, then we have (noisy_images, clean_images)
                noisy_inputs = data_item
                clean_targets = label_item
                letter_labels = None
            else:
                # If labels look like class indices, then we have (images, class_labels)
                noisy_inputs = data_item
                clean_targets = data_item  # Use same image as target for now
                letter_labels = label_item

        elif len(batch_data) == 3:
            # Ideal case: (noisy_images, clean_images, labels)
            noisy_inputs, clean_targets, letter_labels = batch_data
        else:
            raise ValueError(f"Unexpected batch data format: {len(batch_data)} elements")

        optimizer.zero_grad()

        # Forward pass: noisy images -> denoised images
        denoised_outputs = model(noisy_inputs)

        total_loss, recon_loss, accuracy = compute_combined_loss(
                denoised_outputs, clean_targets, letter_labels, classifier, alpha, beta
        )

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_total_loss += total_loss.item() * noisy_inputs.size(0)
        running_recon_loss += recon_loss.item() * noisy_inputs.size(0)
        running_accuracy += accuracy.item() * noisy_inputs.size(0)
        num_batches += noisy_inputs.size(0)

    # Calculate epoch metrics
    epoch_total_loss = running_total_loss / num_batches
    epoch_recon_loss = running_recon_loss / num_batches
    epoch_accuracy = running_accuracy / num_batches

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Total Loss: {epoch_total_loss:.4f}")
    print(f"  Reconstruction Loss: {epoch_recon_loss:.4f}")
    print(f"  Classification Accuracy: {epoch_accuracy:.4f}")

    # Evaluation on test set
    model.eval()
    test_total_loss = 0.0
    test_recon_loss = 0.0
    test_accuracy = 0.0
    test_batches = 0

    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                noisy_inputs, clean_targets, letter_labels = batch_data
            elif len(batch_data) == 2:
                noisy_inputs, clean_targets = batch_data
                letter_labels = None
            else:
                raise ValueError(f"Unexpected batch data format: {len(batch_data)} elements")

            denoised_outputs = model(noisy_inputs)

            if letter_labels is not None:
                total_loss, recon_loss, accuracy = compute_combined_loss(
                        denoised_outputs, clean_targets, letter_labels, classifier, alpha, beta
                        )
            else:
                total_loss = reconstruction_criterion(denoised_outputs, clean_targets)
                recon_loss = total_loss
                accuracy = torch.tensor(0.0)

            test_total_loss += total_loss.item() * noisy_inputs.size(0)
            test_recon_loss += recon_loss.item() * noisy_inputs.size(0)
            test_accuracy += accuracy.item() * noisy_inputs.size(0)
            test_batches += noisy_inputs.size(0)

    avg_test_total_loss = test_total_loss / test_batches
    avg_test_recon_loss = test_recon_loss / test_batches
    avg_test_accuracy = test_accuracy / test_batches

    print(f"  Test Total Loss: {avg_test_total_loss:.4f}")
    print(f"  Test Reconstruction Loss: {avg_test_recon_loss:.4f}")
    print(f"  Test Classification Accuracy: {avg_test_accuracy:.4f}")
    print("-" * 50)

    beta += beta_delta

# Save the trained denoising model
torch.save(model.state_dict(), "denoise.pth")

print("Training completed!")
print(f"Final weights - Alpha (reconstruction): {alpha}, Beta (classification): {beta}")
print("The denoising model has been trained to:")
print("1. Reconstruct clean images from noisy inputs (reconstruction loss)")
print("2. Preserve features important for Greek letter classification (classification accuracy)")
