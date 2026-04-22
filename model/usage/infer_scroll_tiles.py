#!/usr/bin/env python3
"""Run model inference across a tiled scroll manifest and save outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import settings  # noqa: E402
from model import build_model  # noqa: E402


def portable_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_manifest_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Path to tile manifest JSON")
    parser.add_argument(
        "--weights",
        default=None,
        help="Model weights path; default: settings.display_from",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output tiles and stitched preview; default: alongside manifest in inference_<profile>",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save a stitched preview image assembled from tile outputs",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_input_tensor(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        tensor = transforms.ToTensor()(image)
    return tensor.unsqueeze(0).to(settings.device)


def ensure_output_dir(manifest_path: Path, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir).expanduser().resolve()
    return manifest_path.parent / f"inference_{settings.size_profile['profile']}"


def save_tile_output(output_tensor: torch.Tensor, record: dict, output_dir: Path, tile_size: int) -> dict:
    resized = transforms.ToPILImage()(output_tensor)
    resized = resized.resize((tile_size, tile_size), Image.Resampling.BILINEAR)

    output_tiles_dir = output_dir / "tiles"
    output_tiles_dir.mkdir(parents=True, exist_ok=True)
    output_name = record["filename"].rsplit(".", 1)[0] + "_pred.png"
    output_path = output_tiles_dir / output_name
    resized.save(output_path)

    saved_record = dict(record)
    saved_record["output_path"] = portable_path(output_path)
    return saved_record


def build_preview(manifest: dict, output_records: list[dict], output_dir: Path) -> Path:
    preview = Image.new(
        "L",
        (manifest["image_width"], manifest["image_height"]),
        color=255,
    )

    for record in output_records:
        tile_path = resolve_manifest_path(record["output_path"])
        with Image.open(tile_path) as tile:
            tile = tile.convert("L")
            crop = tile.crop((0, 0, record["source_width"], record["source_height"]))
            preview.paste(crop, (record["left"], record["top"]))

    preview_path = output_dir / "stitched_preview.png"
    preview.save(preview_path)
    return preview_path


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    output_dir = ensure_output_dir(manifest_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights).expanduser().resolve() if args.weights else None
    try:
        model = build_model(load_from=str(weights_path) if weights_path else settings.display_from)
    except RuntimeError as exc:
        raise SystemExit(
            "Failed to load checkpoint for the active size profile. "
            "A 96x96 checkpoint cannot be used directly with the 64x64 model variant; "
            "train or export weights for the selected VES_SIZE_PROFILE first.\n"
            f"Original error: {exc}"
        ) from exc
    model.to(settings.device)
    model.eval()

    output_records = []
    for record in manifest["records"]:
        input_path = resolve_manifest_path(record["path"])
        input_tensor = load_input_tensor(input_path)
        with torch.no_grad():
            output, _ = model(input_tensor)
        output = output.squeeze(0).squeeze(0).cpu()
        output_records.append(
            save_tile_output(
                output_tensor=output,
                record=record,
                output_dir=output_dir,
                tile_size=manifest["tile_size"],
            )
        )

    results = {
        "source_manifest": manifest_path.as_posix(),
        "document_id": manifest["document_id"],
        "tile_size": manifest["tile_size"],
        "model_profile": settings.size_profile["profile"],
        "weights": str(weights_path) if weights_path else settings.display_from,
        "num_tiles": len(output_records),
        "records": output_records,
    }

    if args.save_preview:
        preview_path = build_preview(manifest, output_records, output_dir)
        results["stitched_preview"] = portable_path(preview_path)

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results: {results_path}")
    if "stitched_preview" in results:
        print(f"Wrote preview: {output_dir / 'stitched_preview.png'}")


if __name__ == "__main__":
    main()
