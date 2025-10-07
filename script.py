import os
from PIL import Image, ImageEnhance
import numpy as np

# Adjust brightness and contrast values to fit PIL's method
BRIGHTNESS_ADJ = 75
CONTRAST_ADJ = 127

def desaturate_luminance(image):
    # Convert to grayscale using luminance
    return image.convert("L")

def adjust_brightness_contrast(image, brightness_adj, contrast_adj):
    # Convert image to numpy array for manual brightness and contrast adjustment
    arr = np.array(image).astype(np.int16)

    # Apply brightness
    arr += brightness_adj

    # Apply contrast
    factor = (259 * (contrast_adj + 255)) / (255 * (259 - contrast_adj))
    arr = factor * (arr - 128) + 128

    # Clip values to valid range
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)

def process_image(path, output_path):
    try:
        with Image.open(path) as img:
            img = desaturate_luminance(img)
            img = adjust_brightness_contrast(img, BRIGHTNESS_ADJ, CONTRAST_ADJ)
            img.save(output_path, format='BMP')
            print(f"Processed: {path}")
    except Exception as e:
        print(f"Failed to process {path}: {e}")

def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, os.path.splitext(file)[0] + '.bmp')
                process_image(input_path, output_path)

# Set your input and output directories
input_directory = "ALPUB_v2/"
output_directory = "output"

process_directory(input_directory, output_directory)
