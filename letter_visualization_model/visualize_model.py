import json
import random
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import settings

from model import VisionTransformerForSegmentation

# from constants import hyperparams_list  # Optional, not used here

limit = 10  # Number of samples to visualize

# Load paths
with open(settings.data_path, 'r') as f:
    paths_dict = json.load(f)

# Filter level 30 examples
level_30 = [i for i in paths_dict['paths'] if int(i[3]) in settings.display_levels]
#level_30 = [i for i in paths_dict['paths'] if int(i[3]) == 30]
# level_30 = data
paths = random.sample(level_30, limit)

# Initialize model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VisionTransformerForSegmentation()
model = model.to(device)

# Load trained weights
checkpoint_path = "trained_segmentation_models/trained_segmentation_model_Adam.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Preprocessing: Resize to 128x128 and ensure tensor format
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Resize outputs for easier viewing
resize_for_display = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)

# Loop over samples
shown_letters = set()

for noisy_path, clean_path, letter, level in paths:
    if letter in shown_letters:
        continue
    shown_letters.add(letter)

    # Load and preprocess images
    noisy_img = Image.open(noisy_path).convert("RGB")
    clean_img = Image.open(clean_path).convert("RGB")

    # Prepare input tensor
    input_tensor = preprocess(noisy_img).unsqueeze(0).to(device)  # Shape: [1, 3, 128, 128]

    # Pad to 8 channels if needed
    if input_tensor.shape[1] == 3:
        pad = torch.zeros((1, 5, 128, 128), dtype=input_tensor.dtype, device=input_tensor.device)
        input_tensor = torch.cat([input_tensor, pad], dim=1)

    # Model inference
    with torch.no_grad():
        output = model(input_tensor)

    output = output.squeeze(0).squeeze(0).cpu()  # [1, 1, 32, 32] -> [32, 32]
    # output = (output > 0.5).int()
    output_img = transforms.ToPILImage()(output)
    output_img = resize_for_display(output_img)

    # Visualization
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(noisy_img)
    axs[0, 0].set_title(f'Noisy Input\nLetter: {letter}')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(output_img)
    axs[0, 1].set_title('Model Output (Color)')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(clean_img)
    axs[0, 2].set_title('Clean Target')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(noisy_img.convert("L"), cmap='gray')
    axs[1, 0].set_title('Noisy (Grayscale)')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(output_img.convert("L"), cmap='gray')
    axs[1, 1].set_title('Model Output (Grayscale)')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(clean_img.convert("L"), cmap='gray')
    axs[1, 2].set_title('Clean Target (Grayscale)')
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # Debugging info
    print(f"Letter: {letter}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print("-" * 50)
