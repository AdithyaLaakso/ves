from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random

def read_file_num(directory_path: str, num: int):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist")
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a directory")
    files = [f for f in os.listdir(directory_path)
             if os.path.isfile(os.path.join(directory_path, f))]
    selected_file = files[num]
    file_path = os.path.join(directory_path, selected_file)
    # Read the file with UTF-8 encoding to handle Greek letters and other Unicode
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as file:
            content = file.read()
        return content

def read_random_file(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist")
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a directory")
    # Get all files in the directory (excluding subdirectories)
    files = [f for f in os.listdir(directory_path)
             if os.path.isfile(os.path.join(directory_path, f))]
    if not files:
        raise ValueError(f"No files found in directory '{directory_path}'")
    # Select a random file
    random_file = random.choice(files)
    file_path = os.path.join(directory_path, random_file)
    # Read the file with UTF-8 encoding to handle Greek letters and other Unicode
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as file:
            content = file.read()
        return content

def read_random_file(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist")
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a directory")
    # Get all files in the directory (excluding subdirectories)
    files = [f for f in os.listdir(directory_path)
             if os.path.isfile(os.path.join(directory_path, f))]
    if not files:
        raise ValueError(f"No files found in directory '{directory_path}'")
    # Select a random file
    random_file = random.choice(files)
    file_path = os.path.join(directory_path, random_file)
    # Read the file with UTF-8 encoding to handle Greek letters and other Unicode
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as file:
            content = file.read()
        return content

def spiral_coords(h, w, delta=1, direction=1):
    if h <= 0 or w <= 0:
        return
    
    cx, cy = h // 2, w // 2
    yield cx, cy
    
    # Track visited coordinates to avoid duplicates
    visited = {(cx, cy)}
    
    # Direction vectors: right, down, left, up
    directions = [(0, delta), (delta, 0), (0, -delta), (-delta, 0)]
    direction_idx = 0
    
    steps = 1
    
    while len(visited) < h * w:
        for _ in range(2):  # Each step size is used twice (except the first)
            dx, dy = directions[direction_idx]
            
            for _ in range(steps):
                cx += dx
                cy += dy
                
                if 0 <= cx < h and 0 <= cy < w and (cx, cy) not in visited:
                    yield cx, cy
                    visited.add((cx, cy))
                    
                    # Stop if we've visited all cells
                    if len(visited) == h * w:
                        return
            
            # Change direction (turn left in spiral)
            direction_idx = (direction_idx + direction) % 4
        steps += 1


def get_greek_font(font_size):
    """
    Try to find a font that supports Greek characters.
    Falls back to default if none found.
    """
    # Common fonts that support Greek on different systems
    greek_fonts = [
        # Windows
        "arial.ttf", "times.ttf", "calibri.ttf",
        # macOS  
        "Arial.ttc", "Times New Roman.ttc", "Helvetica.ttc",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS alternative path
    ]
    
    for font_path in greek_fonts:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            continue
    
    # Try system font loading (works on some systems)
    try:
        return ImageFont.truetype("arial", font_size)
    except (OSError, IOError):
        pass
    
    try:
        return ImageFont.truetype("DejaVuSans", font_size)
    except (OSError, IOError):
        pass
    
    # Last resort - default font (may not support Greek well)
    print("Warning: Using default font - Greek characters may not display correctly. It is suggested you install a font that supports greek letters.")
    return ImageFont.load_default()

import unicodedata

def remove_greek_accents(text):
    # Normalize the string to NFKD form (Compatibility Decomposition)
    # This separates base characters from combining diacritical marks.
    nfkd_form = unicodedata.normalize('NFKD', text)

    # Filter out combining characters (which include accent marks)
    # unicodedata.combining(c) returns True if 'c' is a combining character.
    stripped_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return stripped_text
