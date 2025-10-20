from pathlib import Path
from PIL import Image
import math
import json

Image.MAX_IMAGE_PIXELS = 1000000000000000

def crop(grid_size, file_name):
    paths_list = []
    base_name = Path(file_name).stem
    output_folder = Path(f"{base_name}_cropped/")

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

                    patch_name = f"{base_name}_{x}x{y}.bmp"
                    output_path = output_folder / patch_name

                    if(left + grid_size > width or upper + grid_size > height):
                        temp_img = img.crop((left,upper, right, lower))
                        patch = Image.new('RGB', (grid_size, grid_size), (0,0,0))
                        patch.paste(temp_img, (0, 0))

                    else:
                        patch = img.crop((left, upper, right, lower))

                    patch = patch.convert("L")
                    patch.save(output_path, 'BMP')

                    paths_list.append(str(output_path))

        json_data = {"paths": paths_list}
        json_file_path = output_folder / "paths.json"

        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)

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

        json_file = output_folder / "paths.json"
        if json_file.exists():
             json_file.unlink()

DATA_PATH = "./20231012184421.tif"
delete(DATA_PATH)
crop(10, DATA_PATH)
