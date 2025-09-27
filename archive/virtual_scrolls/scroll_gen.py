from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import math

from etc import get_greek_font, spiral_coords

def text_to_text_image(text, 
                           width=15, 
                           font_size=20, 
                           font=None, 
                           padding=0, 
                           filename="text.bmp"):
    if font is None:
        font = get_greek_font(font_size)

    lines = [text[i:i+width] for i in range(0, len(text), width)]

    img_width = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines) + (2 * padding)
    img_height = (font_size + 5) * len(lines) + (2 * padding)

    image = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    for i, line in enumerate(lines):
        draw.text((padding, padding + i * (font_size + 5)), line, font=font, fill=(0, 0, 0))

    if filename:
        image.save(filename)

    return image

def text_image_to_scroll(text_image, width=15):
    text_image_array = np.array(text_image)

    text_image_width, text_image_height, colors = text_image_array.shape
    print(f"Image width: {text_image_width}, image height: {text_image_height}, colors: {colors}")

    scroll_z_dims = math.ceil(math.sqrt(text_image_height))
    scroll_shape = (scroll_z_dims, scroll_z_dims, text_image_width)
    scroll = np.zeros(scroll_shape, dtype="u1")  # use zeros directly

    # Folder to save each slice
    output_folder = "bmp_slices"
    os.makedirs(output_folder, exist_ok=True)

    # Fold text image into scroll and save each layer
    layer = 0
    for layer in range(text_image_width - 1):
        if layer == text_image_width:
            break

        spiral = spiral_coords(scroll.shape[0], scroll.shape[1], direction=-1)
        for pixel in text_image_array[layer, :]:
            try:
                x, y = next(spiral)
            except StopIteration:
                print("Ended spiral")
                break
            scroll[x, y, layer] = 0 if tuple(pixel) != (255, 255, 255) else 255

        # Save slice
        slice_img = Image.fromarray(scroll[:, :, layer])
        slice_img.save(os.path.join(output_folder, f"slice_{layer:03}.bmp"))
    
    return scroll

def reverse_scroll_to_text_image(scroll):
    scroll_height, scroll_width, num_layers = scroll.shape
    
    # The original image height is determined by the spiral capacity
    # which should match the original text_image_height
    spiral_capacity = scroll_height * scroll_width
    
    # Initialize the reconstructed image array
    # Original shape was (text_image_width, text_image_height, 3)
    # where text_image_width = num_layers and text_image_height = spiral_capacity
    reconstructed_array = np.zeros((num_layers, spiral_capacity, 3), dtype=np.uint8)
    
    # Process each layer
    for layer in range(num_layers):
        spiral = spiral_coords(scroll_height, scroll_width)
        pixel_idx = 0
        
        # Extract pixels following the same spiral pattern
        for x, y in spiral:
            if pixel_idx >= spiral_capacity:
                break
                
            # Convert back to RGB: 0 -> black (0,0,0), 255 -> white (255,255,255)
            if scroll[x, y, layer] == 255:
                reconstructed_array[layer, pixel_idx] = [255, 255, 255]  # White
            else:
                reconstructed_array[layer, pixel_idx] = [0, 0, 0]  # Black
            
            pixel_idx += 1
    
    # Convert back to PIL Image
    # Need to transpose to get the original orientation
    # Original was text_image_array[layer, :] so we need (height, width, channels)
    #final_image = reconstructed_array.transpose(1, 0, 2)
    final_image = reconstructed_array
    
    return Image.fromarray(final_image)

text = """Σοφία ἐστι μέγιστον ἀγαθὸν τοῖς ἀνθρώποις"""

text_image = text_to_text_image(text, width=len(text))
scroll = text_image_to_scroll(text_image, width=len(text))
reverse_image = reverse_scroll_to_text_image(scroll)
reverse_image.save(os.path.join('./', f"reversed.bmp"))
