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

DATA_PATH = "training_data/paths.json"
INPUT_IMG_PATH = 0
OUTPUT_IMG_PATH = 1
LABEL = 2
LEVEL = 3

class SingleLetterReconstructionDataset:
    def __init__(self, level=0, data_path=DATA_PATH):
        self.data_path = data_path
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

class SingleLetterReconstructionDataLoader:
    def __init__(self, dataset, chunk_size=32768, batch_size=32, shuffle=True, device="cpu",
                 prefetch_batches=500000, num_workers=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.chunk_size = chunk_size
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers

        # Threading components
        self._batch_queue = queue.Queue(maxsize=prefetch_batches)
        self._workers = []
        self._stop_event = threading.Event()
        self._epoch_counter = 0

    def resnet_normalize(self, imgs: Tensor) -> Tensor:
        """Normalize the image tensor using ResNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        imgs = (imgs - mean) / std
        return F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)

    def _load_batch_data(self, batch_data: List) -> Optional[Tuple[Tensor, Tensor, List]]:
        """Load and process a single batch of data."""
        try:
            # Load images from disk
            input_images = []
            output_images = []
            labels = []

            for item in batch_data:
                try:
                    input_img = np.array(Image.open(f"{item[INPUT_IMG_PATH]}").convert("RGB")) / 255.0
                    output_img = np.array(Image.open(f"{item[OUTPUT_IMG_PATH]}").convert("RGB").resize((32, 32))) / 255.0

                    input_images.append(input_img)
                    output_images.append(output_img)
                    labels.append(np.array(item[LABEL]))
                except Exception as e:
                    print(f"Error loading item {item}: {e}")
                    continue

            if len(input_images) == 0:
                return None

            # Validate shapes
            input_img_shape = input_images[0].shape
            for img in input_images:
                if img.shape != input_img_shape:
                    raise ValueError(f"Image shape mismatch: expected {input_img_shape}, got {img.shape}")

            output_img_shape = output_images[0].shape
            for img in output_images:
                if img.shape != output_img_shape:
                    raise ValueError(f"Image shape mismatch: expected {output_img_shape}, got {img.shape}")

            # Convert to tensors
            input_images = np.array([np.transpose(img, (2, 0, 1)) for img in input_images])
            output_images = np.array([np.transpose(img, (2, 0, 1)) for img in output_images])

            input_images = torch.tensor(input_images, dtype=torch.float32)
            output_images = torch.tensor(output_images, dtype=torch.float32)

            input_images = input_images.to(self.device)
            output_images = output_images.to(self.device)

            # Normalize input images
            input_images = self.resnet_normalize(input_images)

            return input_images, output_images, labels

        except Exception as e:
            print(f"Error processing batch: {e}")
            return None

    def _worker_thread(self, epoch_id: int, batches: List):
        """Worker thread that loads batches in the background."""
        for batch_data in batches:
            if self._stop_event.is_set():
                break

            batch_result = self._load_batch_data(batch_data)
            if batch_result is not None:
                try:
                    self._batch_queue.put((epoch_id, batch_result), timeout=1.0)
                except queue.Full:
                    # If queue is full, we might be going too fast
                    if not self._stop_event.is_set():
                        time.sleep(0.1)

    def _start_workers(self):
        """Start background worker threads."""
        self._stop_event.clear()

        # Prepare data for this epoch
        data = self.dataset.copy()
        if self.shuffle:
            np.random.shuffle(data)

        # Split data into batches
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            batches.append(batch_data)

        # Distribute batches among workers
        batches_per_worker = len(batches) // self.num_workers + 1

        for worker_id in range(self.num_workers):
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
        """Stop background worker threads."""
        self._stop_event.set()

        # Clear the queue
        while not self._batch_queue.empty():
            try:
                self._batch_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=1.0)

        self._workers.clear()

    def __iter__(self):
        self._stop_workers()  # Stop any existing workers
        self._epoch_counter += 1
        self._start_workers()

        expected_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        batches_received = 0

        while batches_received < expected_batches:
            try:
                epoch_id, batch_result = self._batch_queue.get(timeout=10.0)

                # Make sure this batch is from the current epoch
                if epoch_id == self._epoch_counter:
                    yield batch_result
                    batches_received += 1

            except queue.Empty:
                print("Warning: Timeout waiting for batch, stopping iteration")
                break

        self._stop_workers()

    def __del__(self):
        """Cleanup when the dataloader is destroyed."""
        self._stop_workers()

class SingleLetterClassificationDataset:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.dataset = self.load_dataset()

    def load_dataset(self, level=0):
        """Load the dataset from the JSON file."""
        with open(self.data_path, "r") as f:
            all_data = json.load(f)['paths']
            if len(all_data) > MAX_SIZE:
                indices = np.random.choice(len(all_data), MAX_SIZE, replace=False)
                data = [all_data[i] for i in indices]
            else:
                data = all_data
        return data

class SingleLetterClassificationDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, device="cpu",
                 prefetch_batches=3, num_workers=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.prefetch_batches = prefetch_batches
        self.num_workers = num_workers

        # Threading components
        self._batch_queue = queue.Queue(maxsize=prefetch_batches)
        self._workers = []
        self._stop_event = threading.Event()
        self._epoch_counter = 0

    def resnet_normalize(self, imgs: Tensor) -> Tensor:
        """Normalize the image tensor using ResNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        imgs = (imgs - mean) / std
        return imgs

    def _load_batch_data(self, batch_data: List) -> Optional[Tuple[Tensor, Tensor]]:
        """Load and process a single batch of data."""
        try:
            # Load images from disk
            input_images = []
            labels = []

            for item in batch_data:
                try:
                    input_img = np.array(Image.open(f"{item[INPUT_IMG_PATH]}").convert("RGB")) / 255.0
                    label = greek_letters[item[LABEL]]

                    input_images.append(input_img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading item {item}: {e}")
                    continue

            if len(input_images) == 0:
                return None

            # Validate shapes
            input_img_shape = input_images[0].shape
            for img in input_images:
                if img.shape != input_img_shape:
                    raise ValueError(f"Image shape mismatch: expected {input_img_shape}, got {img.shape}")

            # Convert to tensors
            input_images = np.array([np.transpose(img, (2, 0, 1)) for img in input_images])

            input_images = torch.tensor(input_images, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            input_images = input_images.to(self.device)
            labels = labels.to(self.device)

            # Normalize input images
            input_images = self.resnet_normalize(input_images)

            return input_images, labels

        except Exception as e:
            print(f"Error processing batch: {e}")
            return None

    def _worker_thread(self, epoch_id: int, batches: List):
        """Worker thread that loads batches in the background."""
        for batch_data in batches:
            if self._stop_event.is_set():
                break

            batch_result = self._load_batch_data(batch_data)
            if batch_result is not None:
                try:
                    self._batch_queue.put((epoch_id, batch_result), timeout=1.0)
                except queue.Full:
                    # If queue is full, we might be going too fast
                    if not self._stop_event.is_set():
                        time.sleep(0.1)

    def _start_workers(self):
        """Start background worker threads."""
        self._stop_event.clear()

        # Prepare data for this epoch
        data = self.dataset.copy()
        if self.shuffle:
            np.random.shuffle(data)

        # Split data into batches
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            batches.append(batch_data)

        # Distribute batches among workers
        batches_per_worker = len(batches) // self.num_workers + 1

        for worker_id in range(self.num_workers):
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
        """Stop background worker threads."""
        self._stop_event.set()

        # Clear the queue
        while not self._batch_queue.empty():
            try:
                self._batch_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=1.0)

        self._workers.clear()

    def __iter__(self):
        self._stop_workers()  # Stop any existing workers
        self._epoch_counter += 1
        self._start_workers()

        expected_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        batches_received = 0

        while batches_received < expected_batches:
            try:
                epoch_id, batch_result = self._batch_queue.get(timeout=10.0)

                # Make sure this batch is from the current epoch
                if epoch_id == self._epoch_counter:
                    yield batch_result
                    batches_received += 1

            except queue.Empty:
                print("Warning: Timeout waiting for batch, stopping iteration")
                break

        self._stop_workers()

    def __del__(self):
        """Cleanup when the dataloader is destroyed."""
        self._stop_workers()
