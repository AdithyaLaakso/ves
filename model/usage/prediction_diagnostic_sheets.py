import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import settings
from dataset import SegData
from model import VisionTransformerForSegmentationMultiScale
from usage.evaluate_visual_regression import _extract_model_output, _image_metrics
from usage.target_inspection_sheets import _threshold_image


DEFAULT_CHECKPOINTS = [
    ("old_subset", MODEL_DIR / "runs" / "20260607T221147Z-ce45afd60" / "new.pth"),
    ("full_epoch1", MODEL_DIR / "runs" / "20260609T034800Z-508dd5292" / "new.pth"),
    ("full_epoch2", MODEL_DIR / "runs" / "20260610T015145Z-07415d46e" / "new.pth"),
]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_labels(raw: Optional[str]) -> Optional[set[str]]:
    if raw is None or not raw.strip():
        return None
    return {part.strip().upper() for part in raw.split(",") if part.strip()}


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


def _normalised_tensor_to_image(
    tensor: torch.Tensor,
    size: Optional[Sequence[int]] = None,
) -> Image.Image:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(tensor.shape)}")

    tensor_min = tensor.min()
    tensor_max = tensor.max()
    if torch.isclose(tensor_min, tensor_max):
        normalised = torch.zeros_like(tensor)
    else:
        normalised = (tensor - tensor_min) / (tensor_max - tensor_min)
    return _tensor_to_image(normalised, size)


def _fixed_threshold_tensor_to_image(
    tensor: torch.Tensor,
    size: Optional[Sequence[int]] = None,
) -> Image.Image:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    thresholded = torch.where(tensor < 0.5, torch.zeros_like(tensor), torch.ones_like(tensor))
    return _tensor_to_image(thresholded, size)


def _abs_error_image(
    pred: torch.Tensor,
    target: torch.Tensor,
    size: Optional[Sequence[int]] = None,
) -> Image.Image:
    pred = pred.detach().cpu().float().squeeze()
    target = target.detach().cpu().float().squeeze()
    if pred.shape != target.shape:
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(0),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
    return _tensor_to_image(torch.abs(pred - target), size)


def _load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = VisionTransformerForSegmentationMultiScale(
        use_gradient_checkpointing=settings.use_gradient,
        num_classes=settings.num_classes,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_stats(logits: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    logits = logits.detach().cpu().float()
    pred = pred.detach().cpu().float()
    return {
        "logit_min": float(logits.min().item()),
        "logit_max": float(logits.max().item()),
        "logit_mean": float(logits.mean().item()),
        "logit_std": float(logits.std(unbiased=False).item()),
        "sigmoid_min": float(pred.min().item()),
        "sigmoid_max": float(pred.max().item()),
        "sigmoid_mean": float(pred.mean().item()),
        "sigmoid_std": float(pred.std(unbiased=False).item()),
    }


def _caption(sample: dict) -> str:
    label = sample["metadata"].get("label") or "unknown"
    document_id = sample["metadata"].get("document_id") or "unknown-doc"
    return f"#{sample['metadata']['index']} {label} {document_id}"


def _draw_sheet(
    samples: Sequence[dict],
    checkpoints: Sequence[str],
    sheet_path: Path,
) -> None:
    cell_size = 92
    label_height = 34
    header_height = 36
    margin = 8

    columns = ["Input", "Target", "Target ink"]
    for checkpoint_name in checkpoints:
        columns.extend(
            [
                f"{checkpoint_name} logits",
                f"{checkpoint_name} sigmoid",
                f"{checkpoint_name} ink",
                f"{checkpoint_name} error",
            ]
        )

    width = margin * 2 + len(columns) * cell_size
    height = margin * 2 + header_height + len(samples) * (cell_size + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for col, title in enumerate(columns):
        x = margin + col * cell_size
        draw.text((x + 4, margin + 4), title[:15], fill="black")

    for row, sample in enumerate(samples):
        y = margin + header_height + row * (cell_size + label_height)
        draw.text((margin + 4, y + 4), _caption(sample)[:110], fill="black")

        image_y = y + label_height
        images = [sample["input"], sample["target"], sample["target_threshold"]]
        for checkpoint_name in checkpoints:
            images.extend(sample["checkpoints"][checkpoint_name][key] for key in ("logits", "sigmoid", "ink", "error"))

        for col, image in enumerate(images):
            x = margin + col * cell_size
            resized = image.resize((cell_size, cell_size), Image.Resampling.NEAREST)
            sheet.paste(resized.convert("RGB"), (x, image_y))
            draw.rectangle((x, image_y, x + cell_size - 1, image_y + cell_size - 1), outline="black")

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)


def _load_indices(indices_json: Path, labels: Optional[set[str]], samples_per_label: int) -> Dict[str, List[int]]:
    raw = json.loads(indices_json.read_text(encoding="utf-8"))
    selected = {}
    for label, indices in raw.items():
        upper_label = str(label).upper()
        if labels is not None and upper_label not in labels:
            continue
        selected[upper_label] = [int(index) for index in indices[:samples_per_label]]
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed-sample diagnostic sheets comparing logits, sigmoid outputs, thresholded ink, and errors."
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
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        default=None,
        help="Checkpoint to render. Repeatable. Defaults to old_subset, full_epoch1, full_epoch2.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated labels to render, e.g. OMICRON,EPSILON. Defaults to all labels in indices JSON.",
    )
    parser.add_argument("--samples-per-label", type=int, default=6, help="Rows per label to render.")
    parser.add_argument("--samples-per-sheet", type=int, default=6, help="Rows per PNG contact sheet.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR / "runs" / "comparisons" / "20260610_prediction_diagnostics",
        help="Directory for diagnostic PNG sheets, metadata, and CSV stats.",
    )
    parser.add_argument("--seed", type=int, default=20260610, help="Seed for deterministic degradation.")
    parser.add_argument("--level", type=int, default=0, help="Dataset level.")
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.samples_per_label <= 0:
        raise ValueError("--samples-per-label must be positive")
    if args.samples_per_sheet <= 0:
        raise ValueError("--samples-per-sheet must be positive")

    labels = _parse_labels(args.labels)
    indices_by_label = _load_indices(args.indices_json, labels, args.samples_per_label)
    checkpoints = args.checkpoint or [(name, str(path)) for name, path in DEFAULT_CHECKPOINTS]
    checkpoint_names = [name for name, _ in checkpoints]
    device = torch.device(args.device)

    _seed_everything(args.seed)
    dataset = SegData(level=args.level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for checkpoint_name, checkpoint_raw_path in checkpoints:
        checkpoint_path = Path(checkpoint_raw_path)
        print(f"loading {checkpoint_name}: {checkpoint_path}", flush=True)
        models[checkpoint_name] = _load_model(checkpoint_path, device)

    sheet_paths = []
    stats_rows: List[Dict[str, object]] = []
    metadata = {
        "seed": args.seed,
        "size_profile": settings.size_profile["profile"],
        "indices_json": str(args.indices_json),
        "labels": sorted(indices_by_label),
        "indices_by_label": indices_by_label,
        "checkpoints": [{"name": name, "path": path} for name, path in checkpoints],
        "sheets": [],
    }

    with torch.no_grad():
        for label, indices in indices_by_label.items():
            samples = []
            for index in indices:
                input_tensor, target = dataset[int(index)]
                target_tensor = target[0] if isinstance(target, tuple) else target
                display_size = (input_tensor.shape[-1], input_tensor.shape[-2])
                target_image = _tensor_to_image(target_tensor, display_size)
                item = dataset.dataset[int(index)]
                sample = {
                    "metadata": {
                        "index": int(index),
                        "label": item.get("label"),
                        "document_id": item.get("document_id"),
                        "input_path": str(item.get("input_path")) if item.get("input_path") else None,
                    },
                    "input": _tensor_to_image(input_tensor, display_size),
                    "target": target_image,
                    "target_threshold": _threshold_image(target_image),
                    "checkpoints": {},
                }

                for checkpoint_name, model in models.items():
                    model_input = input_tensor.unsqueeze(0).to(device)
                    logits = _extract_model_output(model(model_input)).squeeze(0)
                    pred = torch.sigmoid(logits)
                    metrics = _image_metrics(pred, target_tensor)
                    sample["checkpoints"][checkpoint_name] = {
                        "logits": _normalised_tensor_to_image(logits, display_size),
                        "sigmoid": _tensor_to_image(pred, display_size),
                        "ink": _fixed_threshold_tensor_to_image(pred, display_size),
                        "error": _abs_error_image(pred, target_tensor, display_size),
                    }
                    stats_rows.append(
                        {
                            "checkpoint": checkpoint_name,
                            "label": label,
                            "index": int(index),
                            "document_id": item.get("document_id"),
                            "input_path": str(item.get("input_path")) if item.get("input_path") else None,
                            **_checkpoint_stats(logits, pred),
                            **metrics,
                        }
                    )
                samples.append(sample)

            safe_label = label.lower()
            for start in range(0, len(samples), args.samples_per_sheet):
                sheet_number = start // args.samples_per_sheet + 1
                sheet_path = args.output_dir / safe_label / f"{safe_label}_diagnostic-{sheet_number:03d}.png"
                _draw_sheet(samples[start : start + args.samples_per_sheet], checkpoint_names, sheet_path)
                sheet_paths.append(sheet_path)
                metadata["sheets"].append(str(sheet_path))

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "per_sample_diagnostics.csv", stats_rows)

    print(f"wrote metadata: {metadata_path}")
    print(f"wrote stats: {args.output_dir / 'per_sample_diagnostics.csv'}")
    for sheet_path in sheet_paths:
        print(f"wrote sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
