import sys
import json
import random
import torch
from os import path as Path
from pathlib import Path as PPath
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/ece/projects/ves/ves/model')

import settings

from model import build_model 

def run_model(input_folder_str, limit=15):
    input_folder = PPath(input_folder_str)
    
    json_name = 'paths.json'
    json_path = input_folder / json_name
    
    try:
        with open(json_path, 'r') as f:
            paths_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    
    all_paths = paths_dict['paths']
    
    # Initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) > 1:
        load_from = Path.abspath(sys.argv[1]) 
    else:
        load_from = settings.display_from

    model = build_model(load_from=load_from).to(device)

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Resize outputs for easier viewing
    resize_for_display = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)

    # Loop over selected samples
    for path in all_paths:
        full_input_path = PPath(path) 
        
        try:
            input_img = Image.open(full_input_path).convert("RGB")
        except FileNotFoundError:
             print(f"Skipping: Image not found at {full_input_path}")
             continue
        
        # Prepare input tensor
        input_tensor = preprocess(input_img).unsqueeze(0).to(device)  

        # Pad to 8 channels if needed
        # Note: input_tensor.shape[1] is 1 after preprocess (Grayscale(1))
        # if input_tensor.shape[1] == 3:
        #     pad = torch.zeros((1, 5, 128, 128), dtype=input_tensor.dtype, device=input_tensor.device)
        #     input_tensor = torch.cat([input_tensor, pad], dim=1)

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

run_model('image_cropping/assets/sample_1_cropped')