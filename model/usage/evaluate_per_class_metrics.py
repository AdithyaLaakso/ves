import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

if hasattr(torch.backends, "nnpack"):
    torch.backends.nnpack.enabled = False

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import settings
from dataset import SegData
from model import VisionTransformerForSegmentationMultiScale
from usage.evaluate_visual_regression import _extract_model_output, _image_metrics


DEFAULT_CHECKPOINTS = [
    ("old_subset", MODEL_DIR / "runs" / "20260607T221147Z-ce45afd60" / "new.pth"),
    ("full_epoch1", MODEL_DIR / "runs" / "20260609T034800Z-508dd5292" / "new.pth"),
    ("full_epoch2", MODEL_DIR / "runs" / "20260610T015145Z-07415d46e" / "new.pth"),
]


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


def _parse_labels(raw: Optional[str]) -> Optional[set[str]]:
    if raw is None or not raw.strip():
        return None
    return {part.strip().upper() for part in raw.split(",") if part.strip()}


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [key for key in rows[0] if isinstance(rows[0][key], float)]
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def _sample_indices_by_label(
    dataset: SegData,
    *,
    labels: Optional[set[str]],
    samples_per_class: int,
    seed: int,
) -> Dict[str, List[int]]:
    grouped = defaultdict(list)
    for index, item in enumerate(dataset.dataset):
        label = str(item.get("label", "")).upper()
        if labels is not None and label not in labels:
            continue
        grouped[label].append(index)

    rng = np.random.default_rng(seed)
    selected = {}
    for label in sorted(grouped):
        indices = grouped[label]
        if len(indices) > samples_per_class:
            picked = rng.choice(indices, samples_per_class, replace=False)
            selected[label] = sorted(int(index) for index in picked)
        else:
            selected[label] = [int(index) for index in indices]
    return selected


def _cache_samples(dataset: SegData, indices_by_label: Dict[str, List[int]]) -> Dict[str, list]:
    cached = {}
    for label, indices in indices_by_label.items():
        cached[label] = []
        for index in indices:
            input_tensor, target = dataset[index]
            target_tensor = target[0] if isinstance(target, tuple) else target
            item = dataset.dataset[index]
            cached[label].append(
                {
                    "index": int(index),
                    "input": input_tensor.detach().clone(),
                    "target": target_tensor.detach().clone(),
                    "document_id": item.get("document_id"),
                    "input_path": str(item.get("input_path")) if item.get("input_path") else None,
                }
            )
    return cached


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint metrics on deterministic per-class VES samples."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        default=None,
        help="Checkpoint to evaluate. Repeatable. Defaults to old_subset, full_epoch1, full_epoch2.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated labels to evaluate, e.g. OMICRON,EPSILON. Defaults to all classes.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=24,
        help="Maximum deterministic samples per class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR / "runs" / "comparisons" / "20260610_per_class_metrics",
        help="Directory for summary and per-sample metrics.",
    )
    parser.add_argument("--seed", type=int, default=20260610, help="Sampling and degradation seed.")
    parser.add_argument("--level", type=int, default=0, help="Dataset level.")
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")

    checkpoints = args.checkpoint or [
        (name, str(path)) for name, path in DEFAULT_CHECKPOINTS
    ]
    labels = _parse_labels(args.labels)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(args.seed)
    dataset = SegData(level=args.level)
    indices_by_label = _sample_indices_by_label(
        dataset,
        labels=labels,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    cached_samples = _cache_samples(dataset, indices_by_label)

    per_sample_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for checkpoint_name, checkpoint_raw_path in checkpoints:
        checkpoint_path = Path(checkpoint_raw_path)
        print(f"loading {checkpoint_name}: {checkpoint_path}", flush=True)
        model = _load_model(checkpoint_path, device)

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
                            "input_path": sample["input_path"],
                            **metrics,
                        }
                    )

                summary_rows.append(
                    {
                        "checkpoint": checkpoint_name,
                        "label": label,
                        "sample_count": len(samples),
                        **_mean_metrics(metric_rows),
                    }
                )

    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "samples_per_class": args.samples_per_class,
                "labels": sorted(indices_by_label),
                "indices_by_label": indices_by_label,
                "checkpoints": [{"name": name, "path": path} for name, path in checkpoints],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "summary_metrics.csv", summary_rows)
    _write_csv(args.output_dir / "per_sample_metrics.csv", per_sample_rows)

    print(f"wrote {args.output_dir / 'selection.json'}")
    print(f"wrote {args.output_dir / 'summary_metrics.csv'}")
    print(f"wrote {args.output_dir / 'per_sample_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
