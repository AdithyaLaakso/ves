from pathlib import Path
from PIL import Image
import math

def crop(grid_size, file_name):
    base_name = Path(file_name).stem
    output_folder = Path(f"image_cropping/assets/{base_name}_cropped")

    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(file_name) as img:
            width, height = img.size

            grid_x = math.ceil(width / grid_size)
            grid_y = math.ceil(height / grid_size)

            for x in range(grid_x):
                for y in range(grid_y):
                    left = x * grid_size
                    upper = y * grid_size
                    right = min(left + grid_size, width)
                    lower = min(upper + grid_size, height)

                    patch_name = f"{base_name}_{x}x{y}.png"
                    output_path = output_folder / patch_name
                    patch = img.crop((left, upper, right, lower))
                    patch.save(output_path)

    except FileNotFoundError:
        print(f"Error: Image file not found at {file_name}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def delete(file_name):
    base_name = Path(file_name).stem
    output_folder = Path(f"image_cropping/assets/{base_name}_cropped")

    if output_folder.exists():
        for item in output_folder.iterdir():
            item.unlink()

delete("image_cropping/assets/sample_1.png")
crop(100, "image_cropping/assets/sample_1.png")
