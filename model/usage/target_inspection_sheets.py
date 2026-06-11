import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import settings
from dataset import SegData


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensor_to_image(tensor: torch.Tensor, size: Optional[Sequence[int]] = None) -> Image.Image:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale tensor, got shape {tuple(tensor.shape)}")

    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.numpy() * 255.0).astype(np.uint8)
    image = Image.fromarray(array, mode="L")
    if size is not None and tuple(image.size) != tuple(size):
        image = image.resize(tuple(size), Image.Resampling.NEAREST)
    return image


def _otsu_threshold(image: Image.Image) -> int:
    array = np.array(image.convert("L"), dtype=np.uint8)
    hist = np.bincount(array.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128

    sum_total = np.dot(np.arange(256), hist)
    weight_background = 0.0
    sum_background = 0.0
    best_threshold = 128
    best_variance = -1.0

    for threshold in range(256):
        weight_background += hist[threshold]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * hist[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (
            mean_background - mean_foreground
        ) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return best_threshold


def _threshold_image(image: Image.Image) -> Image.Image:
    image = image.convert("L")
    threshold = _otsu_threshold(image)
    return image.point(lambda value: 0 if value <= threshold else 255).convert("L")


def _load_source_image(dataset: SegData, index: int) -> Image.Image:
    path = Path(dataset.dataset[index]["input_path"])
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("L")


def _caption(metadata: dict) -> str:
    label = metadata.get("label") or "unknown"
    document_id = metadata.get("document_id") or "unknown-doc"
    return f"#{metadata['index']} {label} {document_id}"


def _draw_sheet(samples: Sequence[dict], sheet_path: Path) -> None:
    cell_size = 150
    label_height = 36
    header_height = 28
    margin = 10
    columns = ["Degraded input", "Source crop", "Clean target", "Thresholded target"]

    width = margin * 2 + len(columns) * cell_size
    height = margin * 2 + header_height + len(samples) * (cell_size + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for col, title in enumerate(columns):
        x = margin + col * cell_size
        draw.text((x + 6, margin + 5), title, fill="black")

    for row, sample in enumerate(samples):
        y = margin + header_height + row * (cell_size + label_height)
        draw.text((margin + 6, y + 5), _caption(sample["metadata"])[:92], fill="black")

        image_y = y + label_height
        for col, key in enumerate(("input", "source", "target", "threshold")):
            x = margin + col * cell_size
            image = sample[key].resize((cell_size, cell_size), Image.Resampling.NEAREST)
            sheet.paste(image.convert("RGB"), (x, image_y))
            draw.rectangle((x, image_y, x + cell_size - 1, image_y + cell_size - 1), outline="black")

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)


def create_target_inspection_sheets(
    *,
    dataset: SegData,
    indices_by_label: dict,
    output_dir: Path,
    seed: int,
    samples_per_sheet: int,
) -> list[Path]:
    if samples_per_sheet <= 0:
        raise ValueError("samples_per_sheet must be positive")

    _seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths = []
    metadata = {
        "seed": seed,
        "size_profile": settings.size_profile["profile"],
        "indices_by_label": indices_by_label,
        "labels": {},
        "sheets": [],
    }

    for label, indices in indices_by_label.items():
        samples = []
        metadata["labels"][label] = []
        for index in indices:
            input_tensor, target = dataset[int(index)]
            target_tensor = target[0] if isinstance(target, tuple) else target
            display_size = (input_tensor.shape[-1], input_tensor.shape[-2])
            source_image = _load_source_image(dataset, int(index))
            target_image = _tensor_to_image(target_tensor, display_size)
            item = dataset.dataset[int(index)]
            sample_metadata = {
                "index": int(index),
                "label": item.get("label"),
                "document_id": item.get("document_id"),
                "input_path": str(item.get("input_path")) if item.get("input_path") else None,
            }

            samples.append(
                {
                    "metadata": sample_metadata,
                    "input": _tensor_to_image(input_tensor, display_size),
                    "source": source_image.resize(display_size, Image.Resampling.NEAREST),
                    "target": target_image,
                    "threshold": _threshold_image(target_image),
                }
            )
            metadata["labels"][label].append(sample_metadata)

        safe_label = label.lower()
        for start in range(0, len(samples), samples_per_sheet):
            sheet_number = start // samples_per_sheet + 1
            sheet_path = output_dir / safe_label / f"{safe_label}_targets-{sheet_number:03d}.png"
            _draw_sheet(samples[start : start + samples_per_sheet], sheet_path)
            sheet_paths.append(sheet_path)
            metadata["sheets"].append(str(sheet_path))

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return sheet_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate target-only inspection sheets for fixed VES dataset indices."
    )
    parser.add_argument(
        "--indices-json",
        type=Path,
        default=MODEL_DIR
        / "runs"
        / "comparisons"
        / "20260610_visual_progress"
        / "class_specific_indices.json",
        help="JSON object mapping label names to explicit dataset indices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR / "runs" / "comparisons" / "20260610_target_inspection",
        help="Directory for target inspection PNG sheets and metadata.",
    )
    parser.add_argument("--seed", type=int, default=20260610, help="Seed for deterministic degradation.")
    parser.add_argument("--samples-per-sheet", type=int, default=12, help="Rows per PNG contact sheet.")
    parser.add_argument("--level", type=int, default=0, help="Dataset level.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    indices_by_label = json.loads(args.indices_json.read_text(encoding="utf-8"))
    dataset = SegData(level=args.level)
    sheet_paths = create_target_inspection_sheets(
        dataset=dataset,
        indices_by_label=indices_by_label,
        output_dir=args.output_dir,
        seed=args.seed,
        samples_per_sheet=args.samples_per_sheet,
    )

    print(f"wrote metadata: {args.output_dir / 'metadata.json'}")
    for sheet_path in sheet_paths:
        print(f"wrote sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
