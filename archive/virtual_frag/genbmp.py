import numpy as np
from PIL import Image
import random
import os
from pathlib import Path
from typing import Iterator, Union, Set, Optional
import re

def add_noise_to_bmp(input_path, output_path, output_width, output_height,
                     noise_type='gaussian', noise_intensity=0.1,
                     salt_pepper_ratio=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Load the 1-bit BMP image
    try:
        img = Image.open(input_path)
        # Convert to grayscale array (0-255)
        img_array = np.array(img.convert('L'), dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

    # Normalize to 0-1 range
    img_array = img_array / 255.0

    # Apply noise based on type
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_intensity, img_array.shape)
        noisy_img = img_array + noise

    elif noise_type == 'salt_pepper':
        noisy_img = img_array.copy()
        # Salt noise (white pixels)
        salt_mask = np.random.random(img_array.shape) < (noise_intensity * salt_pepper_ratio)
        noisy_img[salt_mask] = 1.0
        # Pepper noise (black pixels)
        pepper_mask = np.random.random(img_array.shape) < (noise_intensity * (1 - salt_pepper_ratio))
        noisy_img[pepper_mask] = 0.0

    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_intensity, noise_intensity, img_array.shape)
        noisy_img = img_array + noise

    elif noise_type == 'speckle':
        noise = np.random.normal(0, noise_intensity, img_array.shape)
        noisy_img = img_array + img_array * noise

    else:
        raise ValueError("noise_type must be 'gaussian', 'salt_pepper', 'uniform', or 'speckle'")

    # Clip values to valid range
    noisy_img = np.clip(noisy_img, 0, 1)

    # Convert back to 0-255 range
    noisy_img = (noisy_img * 255).astype(np.uint8)

    # Create PIL Image and resize
    result_img = Image.fromarray(noisy_img, mode='L')
    result_img = result_img.resize((output_width, output_height), Image.Resampling.LANCZOS)

    # Save the image
    result_img.save(output_path)

    return result_img

def generate_noise_image(output_path, output_width, output_height,
                         base_width=45, base_height=49,
                         noise_type='gaussian', noise_intensity=0.5,
                         base_pattern='random', salt_pepper_ratio=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Generate base pattern
    if base_pattern == 'random':
        base_img = np.random.random((base_height, base_width))
    elif base_pattern == 'black':
        base_img = np.zeros((base_height, base_width))
    elif base_pattern == 'white':
        base_img = np.ones((base_height, base_width))
    elif base_pattern == 'checkerboard':
        base_img = np.zeros((base_height, base_width))
        for i in range(base_height):
            for j in range(base_width):
                if (i + j) % 2 == 0:
                    base_img[i, j] = 1.0
    elif base_pattern == 'gradient':
        base_img = np.linspace(0, 1, base_width)
        base_img = np.tile(base_img, (base_height, 1))
    else:
        raise ValueError("base_pattern must be 'random', 'black', 'white', 'checkerboard', or 'gradient'")

    # Apply noise based on type
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_intensity, base_img.shape)
        noisy_img = base_img + noise

    elif noise_type == 'salt_pepper':
        noisy_img = base_img.copy()
        # Salt noise (white pixels)
        salt_mask = np.random.random(base_img.shape) < (noise_intensity * salt_pepper_ratio)
        noisy_img[salt_mask] = 1.0
        # Pepper noise (black pixels)
        pepper_mask = np.random.random(base_img.shape) < (noise_intensity * (1 - salt_pepper_ratio))
        noisy_img[pepper_mask] = 0.0

    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_intensity, noise_intensity, base_img.shape)
        noisy_img = base_img + noise

    elif noise_type == 'speckle':
        noise = np.random.normal(0, noise_intensity, base_img.shape)
        noisy_img = base_img + base_img * noise

    else:
        raise ValueError("noise_type must be 'gaussian', 'salt_pepper', 'uniform', or 'speckle'")

    # Clip values to valid range
    noisy_img = np.clip(noisy_img, 0, 1)

    # Convert to 0-255 range
    noisy_img = (noisy_img * 255).astype(np.uint8)

    # Create PIL Image and resize
    result_img = Image.fromarray(noisy_img, mode='L')
    result_img = result_img.resize((output_width, output_height), Image.Resampling.LANCZOS)

    # Save the image
    result_img.save(output_path)


def find_images_recursive(directory: Union[str, Path],
                          extensions: Optional[Set[str]] = None,
                          case_sensitive: bool = False) -> Iterator[Path]:
    """
    Recursively find all image files in a directory and return as an iterator.

    Parameters:
        - directory: Path to the directory to search
    - extensions: Set of file extensions to look for (default: common image formats)
    - case_sensitive: Whether to match extensions case-sensitively

    Yields:
        - Path objects for each image file found
    """
    if extensions is None:
        extensions = {
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
                '.webp', '.svg', '.ico', '.psd', '.raw', '.cr2', '.nef',
                '.orf', '.sr2', '.dng', '.arw', '.rw2', '.rwl', '.srw'
                }

    # Convert to Path object
    directory = Path(directory)

    # Check if directory exists
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist")

    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a directory")

    # Normalize extensions for comparison
    if not case_sensitive:
        extensions = {ext.lower() for ext in extensions}

    # Walk through directory recursively
    for root, dirs, files in os.walk(directory):
        root_path = Path(root)

        for file in files:
            file_path = root_path / file

            # Get file extension
            file_ext = file_path.suffix
            if not case_sensitive:
                file_ext = file_ext.lower()

            # Check if it's an image file
            if file_ext in extensions:
                yield file_path

def find_images_with_pathlib(directory: Union[str, Path],
                             extensions: Optional[Set[str]] = None,
                             case_sensitive: bool = False) -> Iterator[Path]:
    """
    Alternative implementation using pathlib's rglob for finding images.

    Parameters:
        - directory: Path to the directory to search
    - extensions: Set of file extensions to look for (default: common image formats)
    - case_sensitive: Whether to match extensions case-sensitively

    Yields:
        - Path objects for each image file found
    """
    if extensions is None:
        extensions = {
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
                '.webp', '.svg', '.ico', '.psd', '.raw', '.cr2', '.nef',
                '.orf', '.sr2', '.dng', '.arw', '.rw2', '.rwl', '.srw'
                }

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist")

    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a directory")

    # Normalize extensions
    if not case_sensitive:
        extensions = {ext.lower() for ext in extensions}

    # Use rglob to find all files recursively
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            file_ext = file_path.suffix
            if not case_sensitive:
                file_ext = file_ext.lower()

            if file_ext in extensions:
                yield file_path

def find_images_with_filter(directory: Union[str, Path],
                            custom_filter: Optional[callable] = None,
                            extensions: Optional[Set[str]] = None,
                            case_sensitive: bool = False) -> Iterator[Path]:
    """
    Find images with additional custom filtering capability.

    Parameters:
        - directory: Path to the directory to search
    - custom_filter: Optional function that takes a Path and returns bool
    - extensions: Set of file extensions to look for
    - case_sensitive: Whether to match extensions case-sensitively

    Yields:
        - Path objects for each image file that passes all filters
    """
    if extensions is None:
        extensions = {
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
                '.webp', '.svg', '.ico', '.psd', '.raw', '.cr2', '.nef',
                '.orf', '.sr2', '.dng', '.arw', '.rw2', '.rwl', '.srw'
                }

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist")

    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a directory")

    if not case_sensitive:
        extensions = {ext.lower() for ext in extensions}

    for file_path in directory.rglob('*'):
        if file_path.is_file():
            file_ext = file_path.suffix
            if not case_sensitive:
                file_ext = file_ext.lower()

            # Check extension
            if file_ext in extensions:
                # Apply custom filter if provided
                if custom_filter is None or custom_filter(file_path):
                    yield file_path

class ImageIterator:
    """
    A class-based iterator for more advanced image directory traversal.
    """
    def __init__(self, directory: Union[str, Path],
                 extensions: Optional[Set[str]] = None,
                 case_sensitive: bool = False,
                 follow_symlinks: bool = False):
        self.directory = Path(directory)
        self.extensions = extensions or {
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
                '.webp', '.svg', '.ico', '.psd', '.raw', '.cr2', '.nef',
                '.orf', '.sr2', '.dng', '.arw', '.rw2', '.rwl', '.srw'
                }
        self.case_sensitive = case_sensitive
        self.follow_symlinks = follow_symlinks

        if not self.case_sensitive:
            self.extensions = {ext.lower() for ext in self.extensions}

        if not self.directory.exists():
            raise FileNotFoundError(f"Directory '{self.directory}' does not exist")

        if not self.directory.is_dir():
            raise NotADirectoryError(f"'{self.directory}' is not a directory")

    def __iter__(self):
        return self._generate_images()

    def _generate_images(self):
        for root, dirs, files in os.walk(self.directory, followlinks=self.follow_symlinks):
            root_path = Path(root)

            for file in files:
                file_path = root_path / file

                # Skip symlinks if not following them
                if not self.follow_symlinks and file_path.is_symlink():
                    continue

                file_ext = file_path.suffix
                if not self.case_sensitive:
                    file_ext = file_ext.lower()

                if file_ext in self.extensions:
                    yield file_path

    def count(self) -> int:
        """Count total number of images without consuming the iterator."""
        return sum(1 for _ in self._generate_images())

    def to_list(self) -> list:
        """Convert iterator to list (consumes the iterator)."""
        return list(self._generate_images())

# Example usage and utility functions
def print_image_stats(directory: Union[str, Path]):
    """Print statistics about images found in directory."""
    iterator = ImageIterator(directory)

    # Count by extension
    ext_counts = {}
    total = 0

    for img_path in iterator:
        ext = img_path.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        total += 1

    print(f"Total images found: {total}")
    print("By extension:")
    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext}: {count}")

# Example usage:
if __name__ == "__main__":
    # Example 1: Add noise to existing BMP
    """
    add_noise_to_bmp(
            input_path="input_45x49.bmp",
            output_path="noisy_output.png",
            output_width=224,
            output_height=224,
            noise_type='gaussian',
            noise_intensity=0.2,
            seed=42
            )

    # Example 2: Generate noise image from scratch
    generate_noise_image(
            output_path="generated_noise.png",
            output_width=512,
            output_height=512,
            noise_type='salt_pepper',
            noise_intensity=0.1,
            base_pattern='checkerboard',
            seed=42
            )
    """
    # Basic usage
    directory = "../hand_writing_dataset/"

    file_formats = {'.bmp'}
    images = find_images_recursive(directory, extensions=file_formats)
    images = list(images)
    random.shuffle(images)

    i = 0
    for image in images:
        path = Path(image)
        dir_name = path.parent.name
        dir_pattern = r'^LETT_([A-Z]+)_([A-Z]+)\.([A-Z]+)$'
        match = re.match(dir_pattern, dir_name)
        case_type, suffix_type, letter = match.groups()

        new_path = f"./training_images/{i}_{case_type}_{suffix_type}_{letter}.bmp"
        add_noise_to_bmp(
            input_path=image,
            output_path=new_path,
            output_width=128,
            output_height=128,
            noise_type='salt_pepper',
            noise_intensity=i * 0.00001,
            seed=42
        )

        i += 1

        if i % 5 == 0:
            new_path = f"./training_images/{i}_NOISE.bmp"
            generate_noise_image(
                    output_path=new_path,
                    output_width=128,
                    output_height=128,
                    noise_type='salt_pepper',
                    noise_intensity=i * 0.0001,
                    base_pattern='checkerboard',
                    seed=42
            )
            i += 1
