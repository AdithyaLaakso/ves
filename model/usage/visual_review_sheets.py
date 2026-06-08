import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import settings
from dataset import SegData
from model import build_model


@dataclass
class VisualReviewResult:
    sheet_paths: List[Path]
    metadata_path: Path


def select_indices(
    dataset_size: int,
    count: int,
    seed: int,
    indices: Optional[Sequence[int]] = None,
) -> List[int]:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")

    if indices is not None:
        selected = [int(idx) for idx in indices]
        invalid = [idx for idx in selected if idx < 0 or idx >= dataset_size]
        if invalid:
            raise ValueError(f"indices out of range for dataset of size {dataset_size}: {invalid}")
        return selected

    sample_count = min(count, dataset_size)
    rng = np.random.default_rng(seed)
    return [int(idx) for idx in rng.choice(dataset_size, sample_count, replace=False)]


def default_output_dir(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "visual_review"
    return checkpoint_path.parent / "visual_review"


def _parse_indices(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


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


def model_output_to_image(
    logits: torch.Tensor,
    size: Optional[Sequence[int]] = None,
) -> Image.Image:
    return _tensor_to_image(torch.sigmoid(logits), size)


def _extract_model_output(raw_output):
    if isinstance(raw_output, tuple):
        return raw_output[0]
    return raw_output


def _item_metadata(dataset, index: int) -> dict:
    items = getattr(dataset, "dataset", [])
    if index >= len(items):
        return {"index": index}

    item = items[index]
    return {
        "index": index,
        "label": item.get("label"),
        "document_id": item.get("document_id"),
        "input_path": str(item.get("input_path")) if item.get("input_path") is not None else None,
    }


def _draw_sheet(samples: Sequence[dict], sheet_path: Path) -> None:
    cell_size = 160
    label_height = 34
    header_height = 26
    margin = 10
    columns = ["Degraded input", "Model output", "Clean target"]

    width = margin * 2 + len(columns) * cell_size
    height = margin * 2 + header_height + len(samples) * (cell_size + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for col, title in enumerate(columns):
        x = margin + col * cell_size
        draw.text((x + 6, margin + 4), title, fill="black")

    for row, sample in enumerate(samples):
        y = margin + header_height + row * (cell_size + label_height)
        label = sample["metadata"].get("label") or "unknown"
        document_id = sample["metadata"].get("document_id") or "unknown-doc"
        caption = f"#{sample['metadata']['index']} {label} {document_id}"
        draw.text((margin + 6, y + 4), caption[:78], fill="black")

        image_y = y + label_height
        for col, key in enumerate(("input", "output", "target")):
            x = margin + col * cell_size
            image = sample[key].resize((cell_size, cell_size), Image.Resampling.NEAREST)
            sheet.paste(image.convert("RGB"), (x, image_y))
            draw.rectangle((x, image_y, x + cell_size - 1, image_y + cell_size - 1), outline="black")

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)


def create_visual_review(
    dataset,
    model: torch.nn.Module,
    checkpoint_path: Path,
    output_dir: Path,
    count: int,
    seed: int,
    indices: Optional[Sequence[int]] = None,
    samples_per_sheet: int = 12,
    device: Optional[torch.device] = None,
    size_profile: Optional[str] = None,
) -> VisualReviewResult:
    if samples_per_sheet <= 0:
        raise ValueError("samples_per_sheet must be positive")

    device = device or settings.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = select_indices(len(dataset), count, seed, indices)

    _seed_everything(seed)
    model = model.to(device)
    model.eval()

    samples = []
    metadata_samples = []
    with torch.no_grad():
        for index in selected_indices:
            input_tensor, target = dataset[index]
            if isinstance(target, tuple):
                target_tensor = target[0]
            else:
                target_tensor = target

            model_input = input_tensor.unsqueeze(0).to(device)
            output_tensor = _extract_model_output(model(model_input)).squeeze(0)
            display_size = (input_tensor.shape[-1], input_tensor.shape[-2])

            sample_metadata = _item_metadata(dataset, index)
            samples.append(
                {
                    "metadata": sample_metadata,
                    "input": _tensor_to_image(input_tensor),
                    "output": model_output_to_image(output_tensor, display_size),
                    "target": _tensor_to_image(target_tensor, display_size),
                }
            )
            metadata_samples.append(sample_metadata)

    sheet_paths = []
    for start in range(0, len(samples), samples_per_sheet):
        sheet_number = len(sheet_paths) + 1
        sheet_path = output_dir / f"sheet-{sheet_number:03d}.png"
        _draw_sheet(samples[start : start + samples_per_sheet], sheet_path)
        sheet_paths.append(sheet_path)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "count": len(selected_indices),
        "indices": selected_indices,
        "size_profile": size_profile or settings.size_profile["profile"],
        "samples": metadata_samples,
        "sheets": [str(path) for path in sheet_paths],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return VisualReviewResult(sheet_paths=sheet_paths, metadata_path=metadata_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate headless visual review contact sheets for a saved VES checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to a model checkpoint, such as runs/.../new.pth")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG sheets and metadata")
    parser.add_argument("--count", type=int, default=24, help="Number of samples to render")
    parser.add_argument("--seed", type=int, default=settings.seed, help="Seed for sample selection and degradation")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated explicit dataset indices, used instead of random sampling",
    )
    parser.add_argument("--samples-per-sheet", type=int, default=12, help="Rows per PNG contact sheet")
    parser.add_argument("--level", type=int, default=0, help="Dataset level for legacy datasets")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    checkpoint = args.checkpoint
    output_dir = args.output_dir or default_output_dir(checkpoint)
    device = torch.device(args.device) if args.device else settings.device

    dataset = SegData(level=args.level)
    model = build_model(load_from=str(checkpoint), device=device)
    result = create_visual_review(
        dataset=dataset,
        model=model,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        count=args.count,
        seed=args.seed,
        indices=_parse_indices(args.indices),
        samples_per_sheet=args.samples_per_sheet,
        device=device,
        size_profile=settings.size_profile["profile"],
    )

    print(f"wrote metadata: {result.metadata_path}")
    for sheet_path in result.sheet_paths:
        print(f"wrote sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
