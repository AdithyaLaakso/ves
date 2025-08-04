import json
from PIL import Image
import torch
from torchvision import transforms
from model import ReconstructionModel as SingleLetterModel
import matplotlib.pyplot as plt
from constants import hyperparams_list
import random

limit = 10  # Limit the number of examples to visualize

with open('training_data/paths.json', 'r') as f:
    paths_dict = json.load(f)

for hyperparams in hyperparams_list[0:1]:
    optimizer_name = hyperparams['optimizer_class'].__name__
    past_letter = ""
    level_30 = [i for i in paths_dict['paths'] if int(i[3]) == 100]
    paths = random.sample(level_30, 10)

    # Initialize and load model ONCE outside the loop
    model = SingleLetterModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the trained model
    state_dict = torch.load(
        f"trained_image_reconstruction_models/trained_image_reconstruction_model_{optimizer_name}.pth",
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Fixed preprocessing - matches your model's expected input size
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),  # Match your model's expected input size
        transforms.ToTensor(),        # Already converts [0,255] to [0,1] - DON'T divide by 255 again!
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Optional: normalize to [-1,1] if using Tanh
    ])

    # For displaying outputs, we need to resize back to a reasonable size
    resize_for_display = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)

    for clean_path, noisy_path, letter, level in paths:
        # Only show one example per letter
        if past_letter == letter:
            continue
        past_letter = letter

        # Load images
        noisy_img = Image.open(noisy_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        # Prepare the image for the model
        input_tensor = preprocess(noisy_img)
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)

        # Move output back to CPU and convert to PIL
        output_cpu = output.squeeze(0).cpu()

        # Convert back to PIL Image and resize for display
        denoised_img = transforms.ToPILImage()(output_cpu)
        denoised_img = resize_for_display(denoised_img)

        # Optional: Convert to grayscale for better visualization
        denoised_img_gray = denoised_img.convert("L")

        # Create visualization
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        # Top row: Original images
        axs[0, 0].imshow(noisy_img)
        axs[0, 0].set_title(f'Noisy Input\n{letter}')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(denoised_img)
        axs[0, 1].set_title(f'Model Output (Color)\n{optimizer_name}')
        axs[0, 1].axis('off')

        axs[0, 2].imshow(clean_img)
        axs[0, 2].set_title('Clean Target')
        axs[0, 2].axis('off')

        # Bottom row: Grayscale versions for better comparison
        axs[1, 0].imshow(noisy_img.convert("L"), cmap='gray')
        axs[1, 0].set_title('Noisy (Grayscale)')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(denoised_img_gray, cmap='gray')
        axs[1, 1].set_title('Model Output (Grayscale)')
        axs[1, 1].axis('off')

        axs[1, 2].imshow(clean_img.convert("L"), cmap='gray')
        axs[1, 2].set_title('Clean Target (Grayscale)')
        axs[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

        # Debug information
        print(f"Letter: {letter}")
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
        print(f"Output tensor shape: {output.shape}")
        print(f"Output tensor range: [{output.min():.4f}, {output.max():.4f}]")
        print("-" * 50)
