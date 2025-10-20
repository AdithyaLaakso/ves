import os
import json
from pathlib import Path
from PIL import Image
import kagglehub
import numpy as np
from noise import pnoise2
import random

# Configuration
DATASET_HANDLE = "miswindall/al-pub-v2"  # Replace with actual Kaggle dataset
OUTPUT_DIR = "./data/"
PATHS_JSON = "paths.json"
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

def download_dataset(dataset_handle):
    """Download dataset from Kaggle using kagglehub."""
    print(f"Downloading dataset: {dataset_handle}")
    path = kagglehub.dataset_download(dataset_handle)
    print(f"Dataset downloaded to: {path}")
    return path

def find_all_images(root_dir):
    """Recursively find all image files in directory."""
    root_path = Path(root_dir)
    images = []

    for file_path in root_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            images.append(file_path)

    print(f"Found {len(images)} images")
    return images

def generate_perlin_noise(width=128, height=128, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Generate Perlin noise texture.

    Args:
        width, height: Dimensions of the noise texture
        scale: Zoom level of the noise
        octaves: Number of levels of detail
        persistence: How much each octave contributes
        lacunarity: Frequency multiplier between octaves
        seed: Random seed for reproducibility
    """
    if seed is None:
        seed = random.randint(0, 1000000)

    noise_array = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            noise_value = pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed
            )
            # Normalize from [-1, 1] to [0, 255]
            noise_array[y][x] = int((noise_value + 1) * 127.5)

    return noise_array.astype(np.uint8)

def apply_noise(img, level):
    """
    Apply Perlin noise overlay to image.

    Args:
        img: PIL Image
        level: Noise level (0-30)
               level=0: no noise
               level=30: 90% noise transparency (0.9 alpha)
    """
    # Convert to RGB and resize
    img = img.convert('RGB').resize((128, 128))

    # Generate Perlin noise
    noise_array = generate_perlin_noise(128, 128, seed=random.randint(0, 1000000))

    # Convert noise to RGB image
    noise_img = Image.fromarray(noise_array, mode='L').convert('RGB')

    # Calculate transparency: level/30 gives us 0.0 to 1.0
    # At level=30, we want 90% noise (0.9 alpha for noise)
    alpha = (level / 30) * 0.9

    # Blend images: result = img * (1-alpha) + noise * alpha
    blended = Image.blend(img, noise_img, alpha)

    return blended

def extract_letter_from_filename(filename):
    """
    Extract the letter (all caps) from filename.
    Example: Z_POxy.v0015.n1778.a.01_134856_111_Alpha_15-18.jpg -> returns 'Z'
    """
    # Get the first character before the underscore
    letter = filename.split('_')[0]
    return letter.upper() if letter else "UNKNOWN"

def process_images(image_paths, output_dir, noise_level=15):
    """Process all images and save to output directory.

    Args:
        image_paths: List of image paths to process
        output_dir: Directory to save processed images
        noise_level: Noise level to apply (0-30)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_paths = []
    failed_images = []

    for idx, img_path in enumerate(image_paths):
        try:
            # Open image
            with Image.open(img_path) as img:
                # Save original copy
                original_filename = f"original_{idx:06d}{img_path.suffix}"
                original_file_path = output_path / original_filename

                # Resize and convert original before saving
                original_resized = img.convert('RGB').resize((128, 128))
                original_resized.save(original_file_path)

                # Apply noise
                modified_img = apply_noise(img, noise_level)

                # Generate noisy output filename
                noisy_filename = f"noisy_{idx:06d}{img_path.suffix}"
                noisy_file_path = output_path / noisy_filename

                # Save modified image
                modified_img.save(noisy_file_path)

                # Extract letter from original filename
                letter = extract_letter_from_filename(img_path.name)

                # Store in the requested format: [noisy, original, LETTER, level]
                processed_paths.append([
                    str(noisy_file_path),
                    str(original_file_path),
                    letter,
                    noise_level
                ])

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_paths)} images")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            failed_images.append(str(img_path))

    print(f"Successfully processed {len(processed_paths)} images")
    if failed_images:
        print(f"Failed to process {len(failed_images)} images")

    return processed_paths, failed_images

def save_paths_json(data, output_dir, filename=PATHS_JSON):
    """Save paths and metadata to JSON file."""
    output_path = Path(output_dir)
    json_path = output_path / filename

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved paths to: {json_path}")

def main():
    """Main execution function."""
    for noise_level in range(0,31):
        # Step 1: Download dataset
        dataset_path = download_dataset(DATASET_HANDLE)

        # Step 2: Find all images
        image_paths = find_all_images(dataset_path)

        if not image_paths:
            print("No images found in dataset!")
            return

        # Step 3: Process images (set noise level here)
        processed_data, failed = process_images(image_paths, OUTPUT_DIR, noise_level)

        # Step 4: Save paths.json in the format: [noisy, original, LETTER, level]
        save_paths_json(processed_data, OUTPUT_DIR)

        print("\n=== Processing Complete ===")
        print(f"Total images found: {len(image_paths)}")
        print(f"Successfully processed: {len(processed_data)}")
        print(f"Failed: {len(failed)}")
        print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
