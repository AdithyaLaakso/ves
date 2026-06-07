import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import constants
import settings

DATA_PATH = settings.data_path
MAX_SIZE = settings.max_size


class DatasetPreflightError(RuntimeError):
    pass


def preflight_manifest(
    manifest_path: Path,
    root: Optional[Path] = None,
    min_loadable: int = 1,
    fail_on_missing: bool = True,
    max_missing_examples: int = 5,
) -> Dict:
    manifest_path = Path(manifest_path)
    root = Path(root) if root is not None else Path(settings.add_to_path)

    with manifest_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = raw_data.get("records", [])
    label_counts = defaultdict(int)
    missing_examples = []
    loadable_records = 0

    for item in records:
        raw_label = item.get("label")
        if raw_label is not None:
            label_counts[constants.canonicalize_label(raw_label)] += 1

        rel_path = item.get("path")
        if rel_path and (root / rel_path).exists():
            loadable_records += 1
        elif len(missing_examples) < max_missing_examples:
            missing_examples.append(rel_path)

    report = {
        "manifest_path": str(manifest_path),
        "total_records": len(records),
        "loadable_records": loadable_records,
        "missing_records": len(records) - loadable_records,
        "missing_examples": missing_examples,
        "class_count": len(label_counts),
    }

    if loadable_records < min_loadable:
        raise DatasetPreflightError(
            f"Manifest preflight failed: {loadable_records} loadable records "
            f"below required minimum {min_loadable}."
        )
    if fail_on_missing and report["missing_records"]:
        raise DatasetPreflightError(
            f"Manifest preflight failed: {report['missing_records']} missing records. "
            f"Examples: {missing_examples}"
        )

    return report


class SegData(Dataset):
    """Dataset that supports both legacy path pairs and ALPUB manifest records."""

    def __init__(
        self,
        level: Union[int, List[int]] = 0,
        data_path=DATA_PATH,
        input_size=None,
        output_size=None,
    ):
        self.data_path = Path(data_path)
        self.input_size = input_size or (settings.input_size, settings.input_size)
        self.output_size = output_size or (settings.output_size, settings.output_size)
        self.dataset, self.metadata = self._load_dataset(level)

    def _load_dataset(self, level=0) -> Tuple[List[Dict], Dict]:
        with self.data_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if settings.data_format == "manifest" or (
            settings.data_format == "auto" and "records" in raw_data
        ):
            report = preflight_manifest(
                self.data_path,
                root=Path(settings.add_to_path),
                min_loadable=settings.min_dataset_size,
                fail_on_missing=False,
            )
            print(
                "manifest preflight:",
                {
                    "total_records": report["total_records"],
                    "loadable_records": report["loadable_records"],
                    "missing_records": report["missing_records"],
                    "class_count": report["class_count"],
                },
            )
            records = self._load_manifest_records(raw_data)
            metadata = {
                "type": "manifest",
                "class_counts": raw_data.get("class_counts", {}),
                "class_document_counts": raw_data.get("class_document_counts", {}),
            }
            return self._limit_dataset(records), metadata

        records = self._load_legacy_records(raw_data, level)
        metadata = {"type": "legacy"}
        return self._limit_dataset(records), metadata

    def _limit_dataset(self, records: List[Dict]) -> List[Dict]:
        if MAX_SIZE is None or len(records) <= MAX_SIZE:
            return records

        rng = np.random.default_rng(settings.seed)
        idx = rng.choice(len(records), MAX_SIZE, replace=False)
        return [records[i] for i in idx]

    def _load_manifest_records(self, raw_data: Dict) -> List[Dict]:
        records = []
        root = Path(settings.add_to_path)
        allowed_labels = set(settings.letters)

        for item in raw_data["records"]:
            label = constants.canonicalize_label(item["label"])
            if label not in allowed_labels:
                continue

            image_path = root / item["path"]
            if not image_path.exists():
                continue

            records.append(
                {
                    "source": "manifest",
                    "input_path": image_path,
                    "target_path": image_path,
                    "label": label,
                    "document_id": item.get("document_id", item["filename"]),
                    "level": 0,
                }
            )

        return records

    def _load_legacy_records(self, raw_data: Dict, level=0) -> List[Dict]:
        all_data = raw_data["paths"]
        if settings.track_levels:
            if isinstance(level, list):
                filtered = [item for item in all_data if int(item[3]) in level]
            else:
                filtered = [item for item in all_data if int(item[3]) == level]
        else:
            filtered = list(all_data)

        records = []
        root = self.data_path.parent
        allowed_labels = set(settings.letters)

        for item in filtered:
            label = constants.canonicalize_label(item[2])
            if label not in allowed_labels:
                continue

            input_path = (root / item[0]).resolve()
            target_path = (root / item[1]).resolve()
            if not input_path.exists() or not target_path.exists():
                continue

            records.append(
                {
                    "source": "legacy",
                    "input_path": input_path,
                    "target_path": target_path,
                    "label": label,
                    "document_id": item[0],
                    "level": int(item[3]),
                }
            )

        return records

    def grouped_indices(self) -> Dict[str, List[int]]:
        groups = defaultdict(list)
        for idx, item in enumerate(self.dataset):
            groups[item["document_id"]].append(idx)
        return groups

    def sample_weights(self, scheme: Optional[str] = None) -> Optional[List[float]]:
        if not scheme or self.metadata.get("type") != "manifest":
            return None

        if scheme == "document_inv_sqrt":
            counts = self.metadata.get("class_document_counts", {})
            return [
                1.0 / np.sqrt(max(counts.get(item["label"], 1), 1))
                for item in self.dataset
            ]

        raise ValueError(f"Unknown sampler strategy: {scheme}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        if settings.mode == settings.CLASSIFICATION:
            return self._get_item_classifying(idx)

        item = self.dataset[idx]
        label = settings.letter_to_idx[item["label"]]

        if item["source"] == "manifest":
            input_tensor, target_tensor = self._get_manifest_pair(item)
        else:
            input_tensor, target_tensor = self._get_legacy_pair(item)

        if settings.mode == settings.RECONSTRUCTION:
            return input_tensor, target_tensor
        if settings.mode == settings.MULTITASK:
            return input_tensor, (target_tensor, label)

        raise ValueError(f"Unknown mode: {settings.mode}")

    def _load_grayscale(self, path: Path) -> Image.Image:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        return image.convert("L")

    def _image_to_tensor(self, image: Image.Image, size: Tuple[int, int]) -> Tensor:
        resized = image.resize(size, Image.Resampling.BILINEAR)
        array = np.array(resized, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def _degrade_manifest_image(self, image: Image.Image) -> Image.Image:
        degraded = image.resize(self.input_size, Image.Resampling.BILINEAR)
        degraded = degraded.rotate(
            random.uniform(-6.0, 6.0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=255,
        )
        degraded = ImageEnhance.Contrast(degraded).enhance(random.uniform(0.8, 1.35))
        degraded = ImageEnhance.Brightness(degraded).enhance(random.uniform(0.9, 1.1))

        if random.random() < 0.4:
            degraded = degraded.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))

        array = np.array(degraded, dtype=np.float32) / 255.0
        noise = np.random.normal(0.0, random.uniform(0.01, 0.05), size=array.shape)
        array = np.clip(array + noise, 0.0, 1.0)
        return Image.fromarray((array * 255.0).astype(np.uint8), mode="L")

    def _get_manifest_pair(self, item: Dict) -> Tuple[Tensor, Tensor]:
        clean = self._load_grayscale(item["input_path"])
        input_image = self._degrade_manifest_image(clean)
        input_tensor = self._image_to_tensor(input_image, self.input_size)
        target_tensor = self._image_to_tensor(clean, self.output_size)
        return input_tensor, target_tensor

    def _get_legacy_pair(self, item: Dict) -> Tuple[Tensor, Tensor]:
        input_tensor = self._image_to_tensor(self._load_grayscale(item["input_path"]), self.input_size)
        target_tensor = self._image_to_tensor(self._load_grayscale(item["target_path"]), self.output_size)
        return input_tensor, target_tensor

    def _get_item_classifying(self, idx) -> Tuple[Tensor, int]:
        item = self.dataset[idx]
        label = settings.letter_to_idx[item["label"]]

        if item["source"] == "manifest":
            clean = self._load_grayscale(item["input_path"])
            input_tensor = self._image_to_tensor(self._degrade_manifest_image(clean), self.input_size)
        else:
            input_tensor = self._image_to_tensor(self._load_grayscale(item["input_path"]), self.input_size)

        return input_tensor, label


def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs)

    if isinstance(labels[0], int):
        labels = torch.tensor(labels, dtype=torch.long)
        return inputs, labels

    if isinstance(labels[0], tuple):
        masks, class_labels = zip(*labels)
        masks = torch.stack(masks)
        class_labels = torch.tensor(class_labels, dtype=torch.long)
        return inputs, (masks, class_labels)

    raise TypeError(f"Unexpected label type: {type(labels[0])}")


def create_loader(
    dataset,
    batch_size=32,
    shuffle=True,
    sampler=None,
    num_workers=settings.num_workers,
):
    pin_memory = settings.device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=settings.persistent_workers,
    )

    return loader
