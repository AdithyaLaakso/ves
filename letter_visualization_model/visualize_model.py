import json
from PIL import Image
import torch
from torchvision import transforms
from model import SingleLetterModel
import matplotlib.pyplot as plt
from constants import hyperparams_list

with open('/Windows/training_data/progressive_test/level0/paths.json', 'r') as f:
    paths_dict = json.load(f)
for hyperparams in hyperparams_list:
    optimizer_name = hyperparams['optimizer_class'].__name__
    past_letter = ""
    for noisy_path, clean_path, letter in paths_dict['paths']:
        #only show one example per letter
        if past_letter == letter:
            continue
        past_letter = letter

        noisy_img = Image.open(noisy_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        # Load the trained model
        model = SingleLetterModel()  # Initialize your model
        state_dict = torch.load(f"trained_image_reconstruction_models/trained_image_reconstruction_model_{optimizer_name}.pth", map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(state_dict)
        model.eval()

        # Prepare the image for the model
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 255),
        ])

        input_tensor = preprocess(noisy_img)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(input_tensor)

        denoised_img = transforms.ToPILImage()(output.squeeze(0))

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(noisy_img)
        axs[0].set_title('Noisy')
        axs[0].axis('off')

        axs[1].imshow(denoised_img)
        axs[1].set_title(f'Denoised {optimizer_name}')
        axs[1].axis('off')

        axs[2].imshow(clean_img)
        axs[2].set_title('Clean')
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()
