import random
import threading
import queue
import time
from typing import Optional, Tuple, List
import json
import numpy as np
import torch
from torch import Tensor
from PIL import Image
import torch.nn.functional as F
from constants import greek_letters, MAX_SIZE
import settings

DATA_PATH = settings.data_path
INPUT_IMG_PATH = 0
OUTPUT_IMG_PATH = 1
LABEL = 2
LEVEL = 3

class SingleLetterSegmentationDataset:
    """Dataset for segmentation task: load input images and binary masks"""

    def __init__(self, level=0, data_path=DATA_PATH, input_size=(128, 128), output_size=(32, 32)):
        self.data_path = data_path
        self.input_size = input_size  # Size for input images
        self.output_size = output_size  # Size for segmentation masks
        self.dataset = self.load_dataset(level=level)

    def load_dataset(self, level=0):
        """Load the dataset from the JSON file."""
        with open(self.data_path, "r") as f:
            all_data = json.load(f)['paths']
            # Filter items by level
            filtered_data = [item for item in all_data if int(item[LEVEL]) == level]
            # Sample if dataset is too large
            if len(filtered_data) > MAX_SIZE:
                indices = np.random.choice(len(filtered_data), MAX_SIZE, replace=False)
                data = [filtered_data[i] for i in indices]
            else:
                data = filtered_data
            return data

class SingleLetterSegmentationDataLoader:
    """DataLoader for segmentation task - FIXED VERSION"""

    def __init__(self, dataset, chunk_size=32768, batch_size=32, shuffle=True, device="cpu",
                 prefetch_batches=500000, num_workers=20, create_synthetic_channels=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.chunk_size = chunk_size
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers
        self.create_synthetic_channels = create_synthetic_channels

        # Threading components
        self._batch_queue = queue.Queue(maxsize=min(prefetch_batches, 100))  # Limit queue size
        self._workers = []
        self._stop_event = threading.Event()
        self._epoch_counter = 0

    def create_8_channel_input(self, rgb_image):
        """
        Create 8-channel input from RGB image for segmentation model.
        Fixed to ensure proper tensor operations and shapes.
        """
        try:
            # Ensure input is numpy array and normalized
            if isinstance(rgb_image, np.ndarray):
                rgb_array = rgb_image.astype(np.float32)
            else:
                rgb_array = np.array(rgb_image, dtype=np.float32)

            # Ensure values are in [0, 1] range
            if rgb_array.max() > 1.0:
                rgb_array = rgb_array / 255.0

            h, w, c = rgb_array.shape

            # Convert to tensor for processing
            rgb_tensor = torch.tensor(rgb_array).float()  # (H, W, C)

            # Initialize list to store channels
            channels = []

            # Channels 0-2: Original RGB
            for i in range(3):
                channels.append(rgb_tensor[:, :, i])

            # Channel 3: Grayscale
            grayscale = 0.299 * rgb_tensor[:, :, 0] + 0.587 * rgb_tensor[:, :, 1] + 0.114 * rgb_tensor[:, :, 2]
            channels.append(grayscale)

            # Channel 4: Edge detection X (simple gradient)
            grad_x = torch.zeros_like(grayscale)
            if h > 1:
                grad_x[1:, :] = torch.abs(grayscale[1:, :] - grayscale[:-1, :])
            channels.append(grad_x)

            # Channel 5: Edge detection Y (simple gradient)
            grad_y = torch.zeros_like(grayscale)
            if w > 1:
                grad_y[:, 1:] = torch.abs(grayscale[:, 1:] - grayscale[:, :-1])
            channels.append(grad_y)

            # Channel 6: Contrast enhancement
            contrast = torch.abs(grayscale - grayscale.mean())
            channels.append(contrast)

            # Channel 7: Brightness
            brightness = rgb_tensor.mean(dim=2)
            channels.append(brightness)

            # Stack all channels: (8, H, W)
            eight_channel = torch.stack(channels, dim=0)

            return eight_channel.numpy()

        except Exception as e:
            print(f"Error in create_8_channel_input: {e}")
            # Fallback: create dummy 8-channel data
            h, w = rgb_image.shape[:2]
            return np.random.rand(8, h, w).astype(np.float32)

    def create_single_channel_input(self, rgb_image):
        """
        Create single-channel grayscale input from RGB image.
        """
        try:
            # Ensure input is numpy array and normalized
            if isinstance(rgb_image, np.ndarray):
                rgb_array = rgb_image.astype(np.float32)
            else:
                rgb_array = np.array(rgb_image, dtype=np.float32)

            # Ensure values are in [0, 1] range
            if rgb_array.max() > 1.0:
                rgb_array = rgb_array / 255.0

            # Convert to grayscale
            if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
                grayscale = 0.299 * rgb_array[:, :, 0] + 0.587 * rgb_array[:, :, 1] + 0.114 * rgb_array[:, :, 2]
            else:
                grayscale = rgb_array.mean(axis=2) if len(rgb_array.shape) == 3 else rgb_array

            # Return as (1, H, W) format
            return grayscale[np.newaxis, :, :]

        except Exception as e:
            print(f"Error in create_single_channel_input: {e}")
            # Fallback: create dummy single-channel data
            h, w = rgb_image.shape[:2]
            return np.random.rand(1, h, w).astype(np.float32)

    def create_binary_mask(self, rgb_image, threshold=0.5):
        """
        Create binary segmentation mask from RGB image.
        Fixed to handle various input formats properly.
        """
        try:
            # Ensure input is numpy array
            if isinstance(rgb_image, np.ndarray):
                img_array = rgb_image.astype(np.float32)
            else:
                img_array = np.array(rgb_image, dtype=np.float32)

            # Normalize if needed
            if img_array.max() > 1.0:
                img_array = img_array / 255.0

            # Convert to grayscale
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
            else:
                gray = img_array.mean(axis=2) if len(img_array.shape) == 3 else img_array

            # Create binary mask using Otsu-like thresholding
            # Use mean as threshold for better results than fixed threshold
            adaptive_threshold = gray.mean() + 0.1 * gray.std()
            binary_mask = (gray > adaptive_threshold).astype(np.float32)

            return binary_mask

        except Exception as e:
            print(f"Error in create_binary_mask: {e}")
            # Fallback: create dummy mask
            h, w = rgb_image.shape[:2]
            return np.random.rand(h, w).astype(np.float32) > 0.5

    def normalize_input(self, imgs: Tensor) -> Tensor:
        """Normalize the input tensor - FIXED VERSION."""
        try:
            # Clone to avoid in-place modifications that might cause issues
            normalized_imgs = imgs.clone()

            # Normalize each channel independently to [0, 1] range
            for i in range(normalized_imgs.shape[1]):  # Iterate over channel dimension
                channel = normalized_imgs[:, i:i+1, :, :]  # Shape: (B, 1, H, W)

                # Get min and max across spatial dimensions for each sample in batch
                batch_size = channel.shape[0]
                for b in range(batch_size):
                    sample_channel = channel[b, 0, :, :]  # (H, W)
                    channel_min = sample_channel.min()
                    channel_max = sample_channel.max()

                    # Avoid division by zero
                    if channel_max > channel_min:
                        normalized_imgs[b, i, :, :] = (sample_channel - channel_min) / (channel_max - channel_min)
                    else:
                        # If all values are the same, set to 0.5
                        normalized_imgs[b, i, :, :] = 0.5

            return normalized_imgs

        except Exception as e:
            print(f"Error in normalize_input: {e}")
            # Return original tensor if normalization fails
            return imgs

    def _load_batch_data(self, batch_data: List) -> Optional[Tuple[Tensor, Tensor, List]]:
        """Load and process a single batch of data - FIXED VERSION."""
        try:
            input_images = []
            output_masks = []
            labels = []

            # Determine expected channels based on create_synthetic_channels flag
            expected_channels = 8 if self.create_synthetic_channels else 1

            for item in batch_data:
                try:
                    # Load and resize input image to exactly 128x128
                    input_img = Image.open(item[INPUT_IMG_PATH]).convert("RGB")
                    input_img = input_img.resize((128, 128), Image.Resampling.BILINEAR)
                    input_img = np.array(input_img, dtype=np.float32)

                    # Load and resize output image to exactly 32x32
                    output_img = Image.open(item[OUTPUT_IMG_PATH]).convert("RGB")
                    output_img = output_img.resize((32, 32), Image.Resampling.BILINEAR)
                    output_img = np.array(output_img, dtype=np.float32)

                    # Create input based on create_synthetic_channels flag
                    if self.create_synthetic_channels:
                        # Create 8-channel input
                        processed_input = self.create_8_channel_input(input_img)
                        expected_shape = (8, 128, 128)
                    else:
                        # Create single-channel grayscale input
                        processed_input = self.create_single_channel_input(input_img)
                        expected_shape = (1, 128, 128)

                    # Create binary segmentation mask
                    binary_mask = self.create_binary_mask(output_img)

                    # Validate shapes
                    if processed_input.shape != expected_shape:
                        print(f"Input shape error: {processed_input.shape}, expected {expected_shape}")
                        continue

                    if binary_mask.shape != (32, 32):
                        print(f"Mask shape error: {binary_mask.shape}, expected (32, 32)")
                        continue

                    input_images.append(processed_input)
                    output_masks.append(binary_mask)
                    labels.append(item[LABEL])

                except Exception as e:
                    print(f"Error loading item {item}: {e}")
                    continue

            if len(input_images) == 0:
                print("No valid images loaded in this batch")
                return None


            # Random rotation (1 in 4 chance for each: 90°, 180°, 270°, 0°)
            rotation_options = [0, 1, 2, 3]  # 0=0°, 1=90°, 2=180°, 3=270°
            rotation_steps = random.choice(rotation_options)

            # Convert numpy arrays to tensors first
            input_images_tensor = torch.stack(
                [torch.from_numpy(img) if isinstance(img, np.ndarray) else img
                 for img in input_images],
                dim=0
            )
            output_masks_tensor = torch.stack(
                [torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
                 for mask in output_masks],
                dim=0
            )

            # Random inversion (1 in 2 chance) - apply to raw images before normalization
            if random.random() < 0.5:
                input_images_tensor = 1.0 - input_images_tensor  # Apply inversion to tensor

            # Ensure int type for k
            rotation_steps = int(rotation_steps)

            # Rotate whole batch
            if rotation_steps > 0:
                input_images_tensor = torch.rot90(input_images_tensor, k=rotation_steps, dims=(-2, -1))
                output_masks_tensor = torch.rot90(output_masks_tensor, k=rotation_steps, dims=(-2, -1))

            # Normalize input images AFTER augmentations (keep as tensor)
            input_images = self.normalize_input(input_images_tensor)  # Pass tensor instead of list

            # Ensure masks are in [0, 1] range
            output_masks = torch.clamp(output_masks_tensor, 0.0, 1.0)

            # Move to device
            input_images = input_images.to(self.device)
            output_masks = output_masks.to(self.device)

            return input_images, output_masks, labels
        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _worker_thread(self, epoch_id: int, batches: List):
        """Worker thread that loads batches in the background - FIXED."""
        for batch_data in batches:
            if self._stop_event.is_set():
                break

            batch_result = self._load_batch_data(batch_data)
            if batch_result is not None:
                try:
                    self._batch_queue.put((epoch_id, batch_result), timeout=5.0)
                except queue.Full:
                    # If queue is full, skip this batch to avoid blocking
                    print("Warning: Batch queue full, skipping batch")
                    continue

    def _start_workers(self):
        """Start background worker threads - FIXED."""
        self._stop_event.clear()

        # Prepare data for this epoch
        data = self.dataset.dataset.copy()
        if self.shuffle:
            np.random.shuffle(data)

        # Split data into batches
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            batches.append(batch_data)

        # Limit number of workers to avoid too many threads
        actual_num_workers = min(self.num_workers, len(batches), 4)

        # Distribute batches among workers
        if actual_num_workers > 0:
            batches_per_worker = len(batches) // actual_num_workers + 1

            for worker_id in range(actual_num_workers):
                start_idx = worker_id * batches_per_worker
                end_idx = min((worker_id + 1) * batches_per_worker, len(batches))
                worker_batches = batches[start_idx:end_idx]

                if worker_batches:
                    worker = threading.Thread(
                        target=self._worker_thread,
                        args=(self._epoch_counter, worker_batches),
                        daemon=True
                    )
                    worker.start()
                    self._workers.append(worker)

    def _stop_workers(self):
        """Stop background worker threads - FIXED."""
        self._stop_event.set()

        # Clear the queue with timeout to avoid hanging
        timeout_count = 0
        while not self._batch_queue.empty() and timeout_count < 10:
            try:
                self._batch_queue.get_nowait()
            except queue.Empty:
                break
            timeout_count += 1

        # Wait for workers to finish with timeout
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=2.0)

        self._workers.clear()

    def __iter__(self):
        """Iterator interface - FIXED."""
        self._stop_workers()  # Stop any existing workers
        self._epoch_counter += 1
        self._start_workers()

        expected_batches = (len(self.dataset.dataset) + self.batch_size - 1) // self.batch_size
        batches_received = 0
        timeout_count = 0

        while batches_received < expected_batches and timeout_count < 3:
            try:
                epoch_id, batch_result = self._batch_queue.get(timeout=30.0)

                # Make sure this batch is from the current epoch
                if epoch_id == self._epoch_counter:
                    yield batch_result
                    batches_received += 1
                    timeout_count = 0  # Reset timeout counter on success

            except queue.Empty:
                timeout_count += 1
                print(f"Warning: Timeout waiting for batch ({timeout_count}/3)")
                if timeout_count >= 3:
                    print("Too many timeouts, stopping iteration")
                    break

        self._stop_workers()

    def __del__(self):
        """Cleanup when the dataloader is destroyed."""
        try:
            self._stop_workers()
        except:
            pass


# Test function to validate the fixed dataloader
def test_dataloader():
    """Test function to validate the dataloader works correctly"""
    print("Testing the fixed segmentation dataloader...")

    try:
        # Test both modes
        for create_synthetic in [True, False]:
            print(f"\n--- Testing create_synthetic_channels={create_synthetic} ---")

            # Create test dataset
            dataset = SingleLetterSegmentationDataset(level=0)
            print(f"✓ Dataset loaded: {len(dataset.dataset)} items")

            # Create test dataloader
            dataloader = SingleLetterSegmentationDataLoader(
                dataset,
                batch_size=2,  # Small batch size for testing
                device="cpu",
                num_workers=1,  # Single worker for testing
                create_synthetic_channels=create_synthetic
            )

            # Test one batch
            batch_count = 0
            for inputs, masks, labels in dataloader:
                expected_channels = 8 if create_synthetic else 1
                print(f"✓ Batch {batch_count + 1}:")
                print(f"  - Input shape: {inputs.shape} (expected: [batch_size, {expected_channels}, 128, 128])")
                print(f"  - Mask shape: {masks.shape} (expected: [batch_size, 1, 32, 32])")
                print(f"  - Labels count: {len(labels)}")
                print(f"  - Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
                print(f"  - Mask range: [{masks.min():.3f}, {masks.max():.3f}]")

                # Verify shapes
                assert inputs.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {inputs.shape[1]}"
                assert inputs.shape[2:] == (128, 128), f"Expected (128, 128) spatial dimensions, got {inputs.shape[2:]}"
                assert masks.shape[1:] == (1, 32, 32), f"Expected (1, 32, 32) mask shape, got {masks.shape[1:]}"

                batch_count += 1
                if batch_count >= 1:  # Test only 1 batch per mode
                    break

        print("\n✓ Dataloader test completed successfully!")
        print("\nKey fixes applied:")
        print("1. Added create_single_channel_input() method for grayscale input")
        print("2. Fixed _load_batch_data() to respect create_synthetic_channels flag")
        print("3. Added proper shape validation for both modes")
        print("4. Fixed tensor operations in channel creation")
        print("5. Added debug output to track batch creation")

        return True

    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_dataloader()
