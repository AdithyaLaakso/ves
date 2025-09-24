import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
import math
import os
import json
import kagglehub
import shutil

# Download latest version
path = kagglehub.dataset_download("vrushalipatel/handwritten-greek-characters-from-gcdb")
number_of_images_per_letter = 10
level = 1
print(f"Dataset downloaded to {path}")
def generate_irregular_blobs(size, smooth_sigma=8, min_area=0, max_area= 1500000):
    """Generate large, sparse, overlapping blobs from random noise."""
    h, w = size
    mask_clean = np.ones((h, w), dtype=np.uint8) * 255  # Start with a clean mask
    max_attempts = 10
    attempts = 0
    while mask_clean.mean() > 210 and attempts < max_attempts:
        # 1. Start with smooth random noise (higher sigma for larger blobs)
        noise = np.random.rand(h * 10, w * 10)
        noise = gaussian_filter(noise, sigma=smooth_sigma)

        # 2. Normalize
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        # 3. Lower threshold for fewer, larger blobs
        threshold = 0.4  # Lowered threshold for more blobs
        blobs = (noise > threshold).astype(np.uint8) * 255

        # 4. Morphological distortions for irregularity and overlap
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        for _ in range(np.random.randint(4, 8)):
            if np.random.rand() > 0.5:
                blobs = cv2.dilate(blobs, kernel, iterations=1)
            else:
                blobs = cv2.erode(blobs, kernel, iterations=1)

        # 5. Remove small fragments, keep only large blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
        mask_clean = np.zeros_like(blobs)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area // 2 and stats[i, cv2.CC_STAT_AREA] <= max_area * 2:
                mask_clean[labels == i] = 255
        attempts += 1

    if mask_clean.mean() > 210:
        # If still no blobs, just threshold the noise and return
        noise = np.random.rand(h, w)
        noise = gaussian_filter(noise, sigma=smooth_sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        blobs = (noise > 0.2).astype(np.uint8) * 255
        mask_clean = blobs

    return mask_clean[:h, :w]  # Crop back to original size
def rotate_letter(letter_mask):
    """Rotate the letter mask by a small random angle to simulate non-perfect alignment."""
    angle = np.random.uniform(-30, 30)  # random rotation
    h, w = letter_mask.shape
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(letter_mask, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def generate_papyrus_fiber_texture(size, fiber_density=0.3, fiber_strength=1.5):
    """Generate a synthetic papyrus fiber texture using noise + directional filtering, with random rotation."""
    h, w = size
    h += 1
    w += 1
    noise = np.random.rand(h, w)
    # Apply strong directional blur to create fibers
    fibers = fiber_density * cv2.GaussianBlur(noise, (1, int(fiber_strength * 10)), 0)
    # Randomly choose rotation angle: either between -30 and 30, or between 150 and 210 degrees
    angle1 = np.random.uniform(-30, 30)
    angle2 = angle1 + 180
    # Rotate the fiber image
    center = (w // 2, h // 2)
    # Choose one of the two angles randomly
    # angle1: x fibers, angle2: y fibers
    rot_mat_x = cv2.getRotationMatrix2D(center, angle1, 1.0)
    fibers_x = cv2.warpAffine(fibers, rot_mat_x, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rot_mat_y = cv2.getRotationMatrix2D(center, angle2, 1.0)
    fibers_y = cv2.warpAffine(fibers, rot_mat_y, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # Combine both layers for crisscross effect
    fibers_combined = np.maximum(fibers_x, fibers_y)
    # Crop back to original size if we added pixels
    fibers_combined = fibers_combined[:size[0], :size[1]]
    fibers_combined = (fibers_combined - fibers_combined.min()) / (fibers_combined.max() - fibers_combined.min())
    fibers_combined = (fibers_combined * 255).astype(np.uint8)
    return fibers_combined, angle1, angle2

def add_ct_fragment_noise(gt_mask, fiber_density = .3, fiber_strength=0.4, blob_strength=0.3, gauss_sigma=1.0, output_size=(128, 128)):
    """Add noise resembling CT papyrus fragment with fibers + extraction blobs."""
    gt_mask_rotated = rotate_letter(gt_mask)

    # Step 1: Base papyrus fibers
    fibers, angle1, angle2 = generate_papyrus_fiber_texture(output_size, fiber_density, fiber_strength)

    # Step 1b: Assign each fiber a random value (0-1), but keep variance within a fiber small
    h, w = output_size
    fiber_map = np.zeros((h, w), dtype=np.float32)
    num_fibers = int(fiber_density * w)
    for i in range(num_fibers):
        x_start = int(i * w / num_fibers)
        x_end = int((i + 1) * w / num_fibers)
        fiber_value = np.random.uniform(0, 1)
        fiber_noise = np.random.normal(fiber_value, 0.05, (h, x_end - x_start))
        fiber_noise = np.clip(fiber_noise, 0, 1)
        fiber_map[:, x_start:x_end] = fiber_noise
    # Rotate fiber_map to match the fiber rotation applied above
    center = (w // 2, h // 2)
    
    # Add curvature and sharpness to individual fibers
    rot_mat_1 = cv2.getRotationMatrix2D(center, angle1, 1.0)
    rot_mat_2 = cv2.getRotationMatrix2D(center, angle2, 1.0)

    fiber_map_1 = cv2.warpAffine(fiber_map, rot_mat_1, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fiber_map_2 = cv2.warpAffine(fiber_map, rot_mat_2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Add sinusoidal curvature to fibers
    y_indices = np.arange(h).reshape(-1, 1)
    x_indices = np.arange(w).reshape(1, -1)
    curvature_strength = np.random.uniform(0.0005, 0.002)
    # Apply per-row shift for fiber_map_1
    fiber_map_1_curved = np.zeros_like(fiber_map_1)
    shifts_1 = (np.sin(y_indices.flatten() * curvature_strength) * 10).astype(int)
    for idx, shift in enumerate(shifts_1):
        fiber_map_1_curved[idx] = np.roll(fiber_map_1[idx], shift)
    # Apply per-column shift for fiber_map_2
    fiber_map_2_curved = np.zeros_like(fiber_map_2)
    shifts_2 = (np.sin(x_indices.flatten() * curvature_strength) * 10).astype(int)
    for idx, shift in enumerate(shifts_2):
        fiber_map_2_curved[:, idx] = np.roll(fiber_map_2[:, idx], shift)

    # Enhance sharpness of fibers (stronger sharpening)
    fiber_map_1_sharp = cv2.GaussianBlur(fiber_map_1_curved, (0, 0), sigmaX=0.3)
    fiber_map_2_sharp = cv2.GaussianBlur(fiber_map_2_curved, (0, 0), sigmaX=0.3)
    fiber_map_1_sharp = cv2.addWeighted(fiber_map_1_curved, 3.0, fiber_map_1_sharp, -2.0, 0)
    fiber_map_2_sharp = cv2.addWeighted(fiber_map_2_curved, 3.0, fiber_map_2_sharp, -2.0, 0)

    fiber_map = (fiber_map_1_sharp[:output_size[0], :output_size[1]] + fiber_map_2_sharp[:output_size[0], :output_size[1]]) / 2
    # Step 3: Combine
    base_intensity = np.random.normal(127, 5, output_size)
    # Add fiber_map as a modulation to fiber strength
    noisy = base_intensity + fiber_strength * (fibers - 127) * fiber_map + blob_strength
    
    # Step 4: Embed ink with subtle contrast
    ink_intensity = -15  # ink slightly darker than background
    noisy[gt_mask_rotated > 0] += ink_intensity
    
    #blobs = generate_irregular_blobs(output_size, smooth_sigma=2, min_area=50)

    # Step 5: Add final Gaussian noise to simulate CT grain
    noisy += np.random.normal(0, gauss_sigma * 5, output_size)# - blobs
    
    # Normalize to 0–255
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    

    
    return noisy

def smooth_and_add_contrast(image, sigma=1.0, contrast_factor=1.2):
    """Smooth the image and add contrast."""
    smoothed = gaussian_filter(image, sigma=sigma)
    smoothed = np.clip(smoothed * contrast_factor, 0, 255).astype(np.uint8)
    return smoothed
# --- Step 1: Render or load Greek text mask ---
def load_or_render_text_mask(text_image_path, size=(128, 128)):
    # Add checks fro the existence of the file
    assert os.path.exists(text_image_path), f"Text image path {text_image_path} does not exist."
    assert text_image_path.endswith('.bmp'), "Only BMP images are supported for text masks."
    assert os.access(text_image_path, os.R_OK), f"File {text_image_path} is not readable or lacks permissions."
    mask = cv2.imread(text_image_path, cv2.IMREAD_GRAYSCALE)
    assert mask is not None, f"cv2.imread failed to load image from {text_image_path}."
    mask = cv2.resize(mask, size)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin

# --- Step 2: Warp text mask to scroll geometry ---
def cylindrical_warp(image, curvature=0.003):
    h, w = image.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    # shift X coords sinusoidally to simulate curvature
    x_shifted = x_coords + np.sin(y_coords * curvature) * (w * 0.05)
    coords = np.array([y_coords, x_shifted])
    warped = map_coordinates(image, coords, order=1, mode='reflect')
    return warped.astype(np.uint8)

# --- Step 3: Embed into papyrus ---
def embed_into_papyrus(mask, mu_p=100, sigma_p=5, delta_mu=1, sigma_i=5):
    h, w = mask.shape
    papyrus_bg = np.random.normal(mu_p, sigma_p, (h, w))
    ink_layer = np.random.normal(mu_p + delta_mu, sigma_i, (h, w))
    combined = np.where(mask > 0, ink_layer, papyrus_bg)
    return combined

# --- Step 4: Simulate CT resolution loss ---
def simulate_ct_blur(image, sigma_xy=1):
    # Z blur here = more in one axis to mimic slice thickness
    blurred = gaussian_filter(image, sigma=(sigma_xy, sigma_xy))
    return blurred

# --- Step 5: Add CT noise ---
def add_ct_noise(image, gauss_sigma=1, poisson_scale=1.0, ring_strength=0.01):
    noisy = image + np.random.normal(0, gauss_sigma, image.shape)
    noisy = np.random.poisson(noisy * poisson_scale) / poisson_scale
    
    # Simulate faint ring artifact
    h, w = image.shape
    y, x = np.indices((h, w))
    center = (h//2, w//2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    rings = np.sin(r / 3) * ring_strength * np.max(image)
    noisy += rings
    
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# --- Step 6: Full synthetic generator ---
def generate_synthetic_ct(text_image_path, params, output_size=(128, 128)):
    mask = load_or_render_text_mask(text_image_path, output_size)
    noisy_ct = add_ct_fragment_noise(mask, fiber_density=params['fiber_density'], fiber_strength=params['fiber_strength'], blob_strength=params['blob_strength'], gauss_sigma=params['gauss_sigma'], output_size = output_size)
    #warped_mask = cylindrical_warp(mask, params['curvature'])
    #base_img = embed_into_papyrus(warped_mask, params['mu_p'], params['sigma_p'], params['delta_mu'], params['sigma_i'])
    #blurred = simulate_ct_blur(base_img, params['sigma_xy'])
    #noisy_ct = add_ct_noise(blurred, params['gauss_sigma'], params['poisson_scale'], params['ring_strength'])
    return noisy_ct, mask

def get_img_path_list(kaggle_path):
    img_path_list = []
    for root, dirs, files in os.walk(kaggle_path):
        for file in files:
            if file.endswith('.bmp'):
                # Get greek letter name from directory- part of string after period
                folder_name = os.path.basename(root)
                letter_name = folder_name.split('.')[-1].upper()
                img_path_list.append([letter_name, os.path.join(root, file)])
    return img_path_list

def get_unigue_greek_letters(img_paths):
    unique_letters = set()
    for letter, _ in img_paths:
        unique_letters.add(letter)
    return sorted(unique_letters)

img_paths = get_img_path_list(path)
unique_greek_letters = get_unigue_greek_letters(img_paths)

# List of possible curvature values for cylindrical warping (simulates scroll curvature)
# Higher values = more curvature, which can make the letter harder to read due to distortion.
curvatures = np.array([0.001, 0.0012, 0.0014]) * level

# Mean intensity values for papyrus background (controls overall brightness)
# Lower values = darker background, which can make the letter harder to distinguish if ink is also dark.
mu_p_values =  np.array([30]) * level

# Standard deviation for papyrus background intensity (controls background noise)
# Higher values = more background noise, making the letter harder to read.
sigma_p_values =  np.array([1])* level/5

# Difference in mean intensity between ink and papyrus (controls ink contrast)
# Higher values = more contrast (easier to read); lower values = less contrast (harder to read).
delta_mu_values =  np.array([2]) * 5/level

# Standard deviation for ink intensity (controls ink noise)
# Higher values = noisier ink, which can make the letter harder to read.
sigma_i_values =  np.array([1, 2]) * level/5

# Standard deviation for CT blur in x/y (simulates CT resolution loss)
# Higher values = more blur, making the letter harder to read.
sigma_xy_values =  np.array([0.2]) * level/5

# Standard deviation for Gaussian noise added to CT image (controls graininess)
# Higher values = more grain/noise, making the letter harder to read.
gauss_sigma_values =  np.array([.0001]) * level**4

# Scaling factor for Poisson noise (simulates photon noise in CT)
# Values further from 1.0 = more noise, making the letter harder to read.
poisson_scale_values =  1 - np.array([.01**(1/level), -.01**(1/level)])

# Strength of ring artifact simulation (CT-specific artifact)
# Higher values = stronger ring artifacts, making the letter harder to read.
ring_strength_values =  np.array([0.001, 0.002])

# Density of papyrus fibers (controls how many fibers are generated)
# Higher values = more fibers, which can obscure the letter and make it harder to read.
fiber_density_values =  np.array([0.05])

# Strength of papyrus fibers (controls fiber visibility/contrast)
# Higher values = more visible fibers, which can make the letter harder to read.
fiber_strength_values =  np.array([0.1]) # Must be odd numbers/10

# Strength of blob artifacts (controls visibility of extraction blobs)
# Higher values = more prominent blobs, making the letter harder to read.
blob_strength_values =  np.array([0.5, .65, .8, 1])
i = 0

paths = []
output_dir = f"synthetic_ct_images/"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
for letter in unique_greek_letters:
    letter_img_paths = [path[1] for path in img_paths if path[0] == letter]
    for _ in range(number_of_images_per_letter):
        i += 1
        img_path = np.random.choice(letter_img_paths)
        assert os.path.exists(img_path)
        curvature = np.random.choice(curvatures)
        mu_p = np.random.choice(mu_p_values)
        sigma_p = np.random.choice(sigma_p_values)
        delta_mu = np.random.choice(delta_mu_values)
        sigma_i = np.random.choice(sigma_i_values)
        sigma_xy = np.random.choice(sigma_xy_values)
        gauss_sigma = np.random.choice(gauss_sigma_values)
        poisson_scale = np.random.choice(poisson_scale_values)
        ring_strength = np.random.choice(ring_strength_values)
        fiber_density = np.random.choice(fiber_density_values)
        fiber_strength = np.random.choice(fiber_strength_values)
        blob_strength = np.random.choice(blob_strength_values)
        params = {
            'curvature': curvature,
            'mu_p': mu_p,
            'sigma_p': sigma_p,
            'delta_mu': delta_mu,
            'sigma_i': sigma_i,
            'sigma_xy': sigma_xy,
            'gauss_sigma': gauss_sigma,
            'poisson_scale': poisson_scale,
            'ring_strength': ring_strength,
            'fiber_density': fiber_density,
            'fiber_strength': fiber_strength,
            'blob_strength': blob_strength
        }
        noisy, gt_mask = generate_synthetic_ct(img_path, params)
        os.makedirs(output_dir, exist_ok=True)
        #output_filename = f"{os.path.basename(img_path).split('.')[0]}_curvature_{curvature}_mu_p_{mu_p}_sigma_p_{sigma_p}_delta_mu_{delta_mu}_sigma_i_{sigma_i}_sigma_xy_{sigma_xy}_gauss_sigma_{gauss_sigma}_poisson_scale_{poisson_scale}_ring_strength_{ring_strength}.png"
        output_filename = f"{os.path.basename(img_path).split('.')[0]}_{letter}_fiber_strength_{fiber_strength}_fiber_density_{fiber_density}_blob_strength_{blob_strength}_gauss_sigma_{gauss_sigma}_{i}.bmp"
        cv2.imwrite(os.path.join(output_dir, output_filename), noisy)
        cv2.imwrite(os.path.join(output_dir, f"gt_mask_{output_filename}"), gt_mask)
        paths.append([os.path.join(output_dir, output_filename), os.path.join(output_dir, f"gt_mask_{output_filename}"), letter])

path_dict = {"paths": paths}

json.dump(path_dict, open(os.path.join(output_dir, "paths.json"), "w"), indent=4)

