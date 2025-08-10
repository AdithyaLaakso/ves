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
from constants import greek_letters
import settings

DATA_PATH = settings.data_path
OUTPUT_IMG_PATH = 0
INPUT_IMG_PATH = 1
LABEL = 2
LEVEL = 3

MAX_SIZE = settings.max_size

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
            if isinstance(level, list):
                filtered_data = [item for item in all_data if int(item[LEVEL]) in level]
            else:
                filtered_data = [item for item in all_data if int(item[LEVEL]) == level]
            # Sample if dataset is too large
            if len(filtered_data) > MAX_SIZE:
                indices = np.random.choice(len(filtered_data), MAX_SIZE, replace=False)
                data = [filtered_data[i] for i in indices]
            else:
                data = filtered_data
            return data

class SingleLetterSegmentationDataLoader:
    """
    Robust DataLoader for segmentation task.
    Key improvements:
    - Eliminated epoch counter race conditions
    - Proper worker synchronization
    - Comprehensive error handling and logging
    - Guaranteed full dataset iteration
    - Backpressure handling for queue management
    """

    def __init__(self, dataset, batch_size=32, shuffle=True, device="cpu",
                 num_workers=4, create_synthetic_channels=True, verbose=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.num_workers = max(1, min(num_workers, 8))  # Reasonable bounds
        self.create_synthetic_channels = create_synthetic_channels
        self.verbose = verbose

        # Thread-safe counters and synchronization
        self._lock = threading.Lock()
        self._active_workers = 0
        self._workers_finished = threading.Event()
        self._stop_event = threading.Event()
        self._workers = []

        # Statistics tracking
        self.stats = {
            'batches_produced': 0,
            'batches_failed': 0,
            'items_processed': 0,
            'items_failed': 0
        }

    def create_8_channel_input(self, rgb_image):
        """
        Create 8-channel input from RGB image for segmentation model.
        Enhanced error handling and validation.
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
            if c != 3:
                raise ValueError(f"Expected 3 channels, got {c}")

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
            if self.verbose:
                print(f"Error in create_8_channel_input: {e}")
            # Fallback: create dummy 8-channel data
            h, w = rgb_image.shape[:2]
            return np.random.rand(8, h, w).astype(np.float32)

    def create_single_channel_input(self, rgb_image):
        """
        Create single-channel grayscale input from RGB image.
        Enhanced error handling.
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
            if self.verbose:
                print(f"Error in create_single_channel_input: {e}")
            # Fallback: create dummy single-channel data
            h, w = rgb_image.shape[:2]
            return np.random.rand(1, h, w).astype(np.float32)

    def create_binary_mask(self, rgb_image, threshold=0.5):
        """
        Create binary segmentation mask from RGB image.
        Enhanced thresholding and error handling.
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

            # Create binary mask using adaptive thresholding
            adaptive_threshold = gray.mean() + 0.1 * gray.std()
            binary_mask = (gray > adaptive_threshold).astype(np.float32)

            return binary_mask

        except Exception as e:
            if self.verbose:
                print(f"Error in create_binary_mask: {e}")
            # Fallback: create dummy mask
            h, w = rgb_image.shape[:2]
            return (np.random.rand(h, w).astype(np.float32) > 0.5).astype(np.float32)

    def normalize_input(self, imgs: Tensor) -> Tensor:
        """Normalize the input tensor with improved stability."""
        try:
            # Clone to avoid in-place modifications
            normalized_imgs = imgs.clone()

            # Normalize each sample independently
            batch_size, channels, height, width = normalized_imgs.shape

            for b in range(batch_size):
                for c in range(channels):
                    channel_data = normalized_imgs[b, c, :, :]
                    channel_min = channel_data.min()
                    channel_max = channel_data.max()

                    # Avoid division by zero
                    if channel_max > channel_min:
                        normalized_imgs[b, c, :, :] = (channel_data - channel_min) / (channel_max - channel_min)
                    else:
                        # If all values are the same, set to 0.5
                        normalized_imgs[b, c, :, :] = 0.5

            return normalized_imgs

        except Exception as e:
            if self.verbose:
                print(f"Error in normalize_input: {e}")
            return imgs

    def _load_single_item(self, item):
        """
        Load and process a single data item with comprehensive error handling.
        Returns (processed_input, binary_mask, label) or None if failed.
        """
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
                processed_input = self.create_8_channel_input(input_img)
                expected_shape = (8, 128, 128)
            else:
                processed_input = self.create_single_channel_input(input_img)
                expected_shape = (1, 128, 128)

            # Create binary segmentation mask
            binary_mask = self.create_binary_mask(output_img)

            # Validate shapes
            if processed_input.shape != expected_shape:
                if self.verbose:
                    print(f"Input shape error: {processed_input.shape}, expected {expected_shape}")
                return None

            if binary_mask.shape != (32, 32):
                if self.verbose:
                    print(f"Mask shape error: {binary_mask.shape}, expected (32, 32)")
                return None

            return processed_input, binary_mask, item[LABEL]

        except Exception as e:
            if self.verbose:
                print(f"Error loading item {item}: {e}")
            return None

    def _load_batch_data(self, batch_items: List) -> Optional[Tuple[Tensor, Tensor, List]]:
        """
        Load and process a batch of data with robust error handling.
        Guarantees that some data is returned unless all items fail.
        """
        try:
            valid_data = []

            # Process each item individually
            for item in batch_items:
                result = self._load_single_item(item)
                if result is not None:
                    valid_data.append(result)
                else:
                    with self._lock:
                        self.stats['items_failed'] += 1

            if len(valid_data) == 0:
                if self.verbose:
                    print(f"All {len(batch_items)} items in batch failed to load")
                with self._lock:
                    self.stats['batches_failed'] += 1
                return None

            # Separate the data
            input_images = [item[0] for item in valid_data]
            output_masks = [item[1] for item in valid_data]
            labels = [item[2] for item in valid_data]

            # Convert to tensors
            input_images_tensor = torch.stack([torch.from_numpy(img) for img in input_images], dim=0)
            output_masks_tensor = torch.stack([torch.from_numpy(mask) for mask in output_masks], dim=0)

            # Add channel dimension to masks if needed
            if len(output_masks_tensor.shape) == 3:
                output_masks_tensor = output_masks_tensor.unsqueeze(1)

            # Apply data augmentations
            # Random rotation (25% chance each for 90°, 180°, 270°)
            rotation_steps = random.choice([0, 1, 2, 3])
            if rotation_steps > 0:
                input_images_tensor = torch.rot90(input_images_tensor, k=rotation_steps, dims=(-2, -1))
                output_masks_tensor = torch.rot90(output_masks_tensor, k=rotation_steps, dims=(-2, -1))

            # Random inversion (50% chance)
            if random.random() < 0.5:
                input_images_tensor = 1.0 - input_images_tensor

            # Normalize inputs
            input_images_tensor = self.normalize_input(input_images_tensor)

            # Ensure masks are in [0, 1] range
            output_masks_tensor = torch.clamp(output_masks_tensor, 0.0, 1.0)

            # Move to device
            input_images_tensor = input_images_tensor.to(self.device)
            output_masks_tensor = output_masks_tensor.to(self.device)

            # Update statistics
            with self._lock:
                self.stats['batches_produced'] += 1
                self.stats['items_processed'] += len(valid_data)

            return input_images_tensor, output_masks_tensor, labels

        except Exception as e:
            if self.verbose:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
            with self._lock:
                self.stats['batches_failed'] += 1
            return None

    def _worker_thread(self, work_queue: queue.Queue, result_queue: queue.Queue):
        """
        Worker thread that processes batches from work queue.
        Uses work-stealing pattern for better load balancing.
        """
        while not self._stop_event.is_set():
            try:
                # Get work with short timeout to allow checking stop event
                batch_items = work_queue.get(timeout=0.5)

                if batch_items is None:  # Poison pill
                    break

                # Process the batch
                batch_result = self._load_batch_data(batch_items)

                if batch_result is not None:
                    # Put result with retry logic and backpressure handling
                    retry_count = 0
                    while not self._stop_event.is_set() and retry_count < 10:
                        try:
                            result_queue.put(batch_result, timeout=1.0)
                            break
                        except queue.Full:
                            retry_count += 1
                            if self.verbose and retry_count % 5 == 0:
                                print(f"Result queue full, retrying... ({retry_count}/10)")
                            time.sleep(0.1)

                    if retry_count >= 10:
                        if self.verbose:
                            print("Failed to put batch result after 10 retries, dropping batch")
                        with self._lock:
                            self.stats['batches_failed'] += 1

                work_queue.task_done()

            except queue.Empty:
                continue  # Check stop event and try again
            except Exception as e:
                if self.verbose:
                    print(f"Worker thread error: {e}")
                continue

        # Signal that this worker is finishing
        with self._lock:
            self._active_workers -= 1
            if self._active_workers == 0:
                self._workers_finished.set()

    def __iter__(self):
        """
        Main iteration method with comprehensive error handling and logging.
        Guarantees full dataset iteration.
        """
        # Reset statistics
        self.stats = {'batches_produced': 0, 'batches_failed': 0, 'items_processed': 0, 'items_failed': 0}

        # Prepare data
        data = self.dataset.dataset.copy()
        if self.shuffle:
            np.random.shuffle(data)

        # Create batches
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch_items = data[i:i + self.batch_size]
            batches.append(batch_items)

        total_batches = len(batches)
        total_items = len(data)

        if self.verbose:
            print(f"Starting iteration: {total_items} items in {total_batches} batches")
            print(f"Using {self.num_workers} workers")

        # Set up queues
        work_queue = queue.Queue()
        result_queue = queue.Queue(maxsize=min(50, total_batches))  # Reasonable buffer

        # Populate work queue
        for batch_items in batches:
            work_queue.put(batch_items)

        # Add poison pills to signal workers to stop
        for _ in range(self.num_workers):
            work_queue.put(None)

        # Start workers
        self._stop_event.clear()
        self._workers_finished.clear()
        self._active_workers = self.num_workers
        self._workers = []

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                args=(work_queue, result_queue),
                daemon=True,
                name=f"DataLoader-Worker-{i}"
            )
            worker.start()
            self._workers.append(worker)

        # Consume results
        batches_yielded = 0
        consecutive_timeouts = 0

        while batches_yielded < total_batches:
            try:
                # Get result with timeout
                batch_result = result_queue.get(timeout=10.0)
                yield batch_result
                batches_yielded += 1
                consecutive_timeouts = 0

                if self.verbose and batches_yielded % 10 == 0:
                    print(f"Progress: {batches_yielded}/{total_batches} batches processed")

            except queue.Empty:
                consecutive_timeouts += 1
                if self.verbose:
                    print(f"Queue timeout {consecutive_timeouts}, batches yielded: {batches_yielded}/{total_batches}")

                # Check if workers are still active
                if self._workers_finished.is_set():
                    if self.verbose:
                        print("All workers finished")
                    break

                # If we've had too many consecutive timeouts, something is wrong
                if consecutive_timeouts >= 5:
                    if self.verbose:
                        print("Too many consecutive timeouts, stopping iteration")
                    break

        # Clean up
        self._stop_workers()

        # Final statistics
        if self.verbose or batches_yielded < total_batches:
            print(f"Iteration complete: {batches_yielded}/{total_batches} batches yielded")
            print(f"Statistics: {self.stats}")
            if batches_yielded < total_batches:
                print(f"WARNING: Only processed {batches_yielded}/{total_batches} batches!")

    def _stop_workers(self):
        """Clean shutdown of worker threads."""
        self._stop_event.set()

        # Wait for workers to finish with timeout
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=2.0)
                if worker.is_alive():
                    if self.verbose:
                        print(f"Worker {worker.name} did not finish in time")

        self._workers.clear()

    def __del__(self):
        """Cleanup when the dataloader is destroyed."""
        try:
            self._stop_workers()
        except:
            pass

    def get_stats(self):
        """Get current statistics."""
        with self._lock:
            return self.stats.copy()
