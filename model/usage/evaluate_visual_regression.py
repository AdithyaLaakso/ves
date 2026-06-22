import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import settings
from dataset import SegData
from model import VisionTransformerForSegmentationMultiScale


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def _extract_model_output(raw_output):
    if isinstance(raw_output, tuple):
        return raw_output[0]
    return raw_output


def _ssim_like(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Compact global SSIM-style score for ranking visual drift. This is not a
    # replacement for image-local SSIM, but it catches luminance/contrast changes
    # without adding another dependency.
    pred = pred.float().flatten()
    target = target.float().flatten()
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = pred.mean()
    mu_y = target.mean()
    var_x = pred.var(unbiased=False)
    var_y = target.var(unbiased=False)
    cov_xy = ((pred - mu_x) * (target - mu_y)).mean()
    score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(score.item())


def _image_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().cpu().float().squeeze()
    target = target.detach().cpu().float().squeeze()
    if pred.shape != target.shape:
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(0),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = torch.nn.functional.mse_loss(pred, target)
    mae = torch.nn.functional.l1_loss(pred, target)

    pred_ink = pred < 0.5
    target_ink = target < 0.5
    intersection = torch.logical_and(pred_ink, target_ink).sum().float()
    union = torch.logical_or(pred_ink, target_ink).sum().float()
    pred_ink_count = pred_ink.sum().float()
    target_ink_count = target_ink.sum().float()
    false_ink = torch.logical_and(pred_ink, ~target_ink).sum().float()
    missed_ink = torch.logical_and(~pred_ink, target_ink).sum().float()

    h, w = target.shape[-2:]
    y0, y1 = h // 4, h - h // 4
    x0, x1 = w // 4, w - w // 4
    interior_pred_ink = pred_ink[y0:y1, x0:x1].float().mean()
    interior_target_ink = target_ink[y0:y1, x0:x1].float().mean()

    eps = torch.tensor(1e-6)
    return {
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "ssim_like": _ssim_like(pred, target),
        "ink_iou": float((intersection / (union + eps)).item()),
        "ink_precision": float((intersection / (pred_ink_count + eps)).item()),
        "ink_recall": float((intersection / (target_ink_count + eps)).item()),
        "pred_ink_fraction": float(pred_ink.float().mean().item()),
        "target_ink_fraction": float(target_ink.float().mean().item()),
        "ink_fraction_abs_diff": float(
            torch.abs(pred_ink.float().mean() - target_ink.float().mean()).item()
        ),
        "false_ink_fraction": float((false_ink / pred_ink.numel()).item()),
        "missed_ink_fraction": float((missed_ink / pred_ink.numel()).item()),
        "interior_pred_ink_fraction": float(interior_pred_ink.item()),
        "interior_target_ink_fraction": float(interior_target_ink.item()),
        "interior_ink_abs_diff": float(
            torch.abs(interior_pred_ink - interior_target_ink).item()
        ),
    }


def _mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [key for key in rows[0] if isinstance(rows[0][key], float)]
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def _delta_direction(metric: str) -> int:
    lower_is_better = {
        "mse",
        "mae",
        "ink_fraction_abs_diff",
        "false_ink_fraction",
        "missed_ink_fraction",
        "interior_ink_abs_diff",
    }
    higher_is_better = {"ssim_like", "ink_iou", "ink_precision", "ink_recall"}
    if metric in lower_is_better:
        return -1
    if metric in higher_is_better:
        return 1
    return 0


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: List[Dict[str, object]]) -> None:
    metrics = [
        "mse",
        "mae",
        "ssim_like",
        "ink_iou",
        "ink_precision",
        "ink_recall",
        "false_ink_fraction",
        "missed_ink_fraction",
        "interior_ink_abs_diff",
    ]
    lines = [
        "# Visual Regression Metrics",
        "",
        "Lower is better for MSE/MAE/error fractions. Higher is better for SSIM-like and ink overlap metrics.",
        "",
        "| Label | Metric | Full epoch 1 | Full epoch 2 | Delta | Direction |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        for metric in metrics:
            before = float(row[f"full_epoch1_{metric}"])
            after = float(row[f"full_epoch2_{metric}"])
            delta = after - before
            direction = _delta_direction(metric)
            if direction == 0:
                verdict = "n/a"
            elif (delta * direction) > 0:
                verdict = "improved"
            elif (delta * direction) < 0:
                verdict = "worse"
            else:
                verdict = "same"
            lines.append(
                f"| {row['label']} | `{metric}` | {before:.6f} | {after:.6f} | {delta:+.6f} | {verdict} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate class-specific visual regression metrics between two checkpoints."
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
        help="Checkpoint pair to evaluate. Repeatable. Defaults to full_epoch1 and full_epoch2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR / "runs" / "comparisons" / "20260610_visual_regression_metrics",
        help="Directory for JSON, CSV, and Markdown outputs.",
    )
    parser.add_argument("--seed", type=int, default=20260610, help="Seed for deterministic degradation.")
    parser.add_argument("--level", type=int, default=0, help="Dataset level.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to evaluate on.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device)
    checkpoints = args.checkpoint or [
        ("full_epoch1", str(MODEL_DIR / "runs" / "20260609T034800Z-508dd5292" / "new.pth")),
        ("full_epoch2", str(MODEL_DIR / "runs" / "20260610T015145Z-07415d46e" / "new.pth")),
    ]
    indices_by_label = json.loads(args.indices_json.read_text(encoding="utf-8"))

    _seed_everything(args.seed)
    dataset = SegData(level=args.level)

    cached_samples = {}
    for label, indices in indices_by_label.items():
        cached_samples[label] = []
        for index in indices:
            input_tensor, target = dataset[int(index)]
            target_tensor = target[0] if isinstance(target, tuple) else target
            metadata = getattr(dataset, "dataset", [])[int(index)]
            cached_samples[label].append(
                {
                    "index": int(index),
                    "input": input_tensor.detach().clone(),
                    "target": target_tensor.detach().clone(),
                    "document_id": metadata.get("document_id"),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_rows: List[Dict[str, object]] = []
    by_checkpoint_label: Dict[str, Dict[str, Dict[str, float]]] = {}

    for checkpoint_name, checkpoint_raw_path in checkpoints:
        checkpoint_path = Path(checkpoint_raw_path)
        print(f"loading {checkpoint_name}: {checkpoint_path}", flush=True)
        model = _load_model(checkpoint_path, device)
        by_checkpoint_label[checkpoint_name] = {}

        with torch.no_grad():
            for label, samples in cached_samples.items():
                metric_rows = []
                for sample in samples:
                    model_input = sample["input"].unsqueeze(0).to(device)
                    raw_output = model(model_input)
                    output_tensor = _extract_model_output(raw_output).squeeze(0)
                    pred = torch.sigmoid(output_tensor)
                    metrics = _image_metrics(pred, sample["target"])
                    metric_rows.append(metrics)
                    per_sample_rows.append(
                        {
                            "checkpoint": checkpoint_name,
                            "label": label,
                            "index": sample["index"],
                            "document_id": sample["document_id"],
                            **metrics,
                        }
                    )
                by_checkpoint_label[checkpoint_name][label] = _mean_metrics(metric_rows)

    summary_rows = []
    checkpoint_names = [name for name, _ in checkpoints]
    if len(checkpoint_names) == 2:
        first, second = checkpoint_names
        for label in indices_by_label:
            row: Dict[str, object] = {"label": label, "sample_count": len(indices_by_label[label])}
            for checkpoint_name in checkpoint_names:
                for key, value in by_checkpoint_label[checkpoint_name][label].items():
                    row[f"{checkpoint_name}_{key}"] = value
            summary_rows.append(row)
    else:
        for checkpoint_name in checkpoint_names:
            for label in indices_by_label:
                row = {"checkpoint": checkpoint_name, "label": label, "sample_count": len(indices_by_label[label])}
                row.update(by_checkpoint_label[checkpoint_name][label])
                summary_rows.append(row)

    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "indices_json": str(args.indices_json),
                "checkpoints": [{"name": name, "path": path} for name, path in checkpoints],
                "summary": summary_rows,
                "per_checkpoint_label": by_checkpoint_label,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "per_sample_metrics.csv", per_sample_rows)
    _write_csv(args.output_dir / "summary_metrics.csv", summary_rows)
    if len(checkpoint_names) == 2:
        _write_markdown(args.output_dir / "summary_metrics.md", summary_rows)

    print(f"wrote {args.output_dir / 'summary.json'}")
    print(f"wrote {args.output_dir / 'summary_metrics.csv'}")
    print(f"wrote {args.output_dir / 'per_sample_metrics.csv'}")
    if len(checkpoint_names) == 2:
        print(f"wrote {args.output_dir / 'summary_metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
