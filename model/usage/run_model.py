import sys
import torch
from os import path as Path
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append('..')

import settings

from model import build_model

model_path = '../model.pth'

def run_model(input_folder_str):
    # Initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) > 1:
        load_from = Path.abspath(sys.argv[1])
    else:
        load_from = settings.display_from

    model = build_model(load_from=model_path).to(device)

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((settings.input_size, settings.input_size)),
        transforms.ToTensor(),
    ])

    # Resize outputs for easier viewing
    resize_for_display = transforms.Resize(
        (settings.input_size, settings.input_size),
        interpolation=transforms.InterpolationMode.NEAREST,
    )

    try:
        input_img = Image.open(input_folder_str).convert("RGB")
    except FileNotFoundError:
        print(f"Skipping: Image not found at {input_folder_str}")
        sys.exit(0)

    # Prepare input tensor
    input_tensor = preprocess(input_img).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        output, _ = model(input_tensor)


        output = output.squeeze(0).squeeze(0).cpu()
        output_img = transforms.ToPILImage()(output)
        output_img = resize_for_display(output_img)

        # Visualization
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        axs[0, 0].imshow(input_img)
        axs[0, 0].set_title('Original Input (Color)')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(output_img)
        axs[0, 1].set_title('Model Output (Color)')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(input_img.convert("L"), cmap='gray')
        axs[1, 0].set_title('Original Input (Grayscale)')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(output_img.convert("L"), cmap='gray')
        axs[1, 1].set_title('Model Output (Grayscale)')
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.show()


target_path = sys.argv[1]
run_model(target_path)
