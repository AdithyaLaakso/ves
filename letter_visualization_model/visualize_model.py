import json
from PIL import Image
import torch
from torchvision import transforms
from model import SingleLetterModel
import matplotlib.pyplot as plt

with open('/Windows/training_data/paths.json', 'r') as f:
    paths_dict = json.load(f)
for noisy_path, clean_path, _ in paths_dict['paths'][:10]:
    noisy_img = Image.open(noisy_path).convert("RGB")
    clean_img = Image.open(clean_path).convert("RGB")

    # Load the trained model
    model = SingleLetterModel()  # Initialize your model
    state_dict = torch.load('denoise.pth', map_location=torch.device('cpu'))
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
    axs[1].set_title('Denoised')
    axs[1].axis('off')

    axs[2].imshow(clean_img)
    axs[2].set_title('Clean')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
print(paths_dict)
