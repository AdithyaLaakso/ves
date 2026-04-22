#!/usr/bin/env python3
"""Tile a large scroll image and write an inference manifest.

This is intended for real-scroll or other large-image inference workflows where
the model should operate on fixed-size windows instead of resizing an entire
image down to the model input size.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]


def portable_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to the large source image")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Square tile size in pixels; default: 64",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between tiles in pixels; default: tile size (non-overlapping)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for tiles and manifest; default: data/scroll_tiles/<image-stem>_<tile-size>",
    )
    parser.add_argument(
        "--pad-color",
        type=int,
        default=255,
        help="Padding fill value for edge tiles in grayscale; default: 255",
    )
    return parser.parse_args()


def iter_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def crop_tile(image: Image.Image, left: int, top: int, tile_size: int, pad_color: int) -> tuple[Image.Image, int, int]:
    right = min(left + tile_size, image.width)
    bottom = min(top + tile_size, image.height)
    width = right - left
    height = bottom - top

    tile = image.crop((left, top, right, bottom))
    if width == tile_size and height == tile_size:
        return tile, width, height

    padded = Image.new("L", (tile_size, tile_size), color=pad_color)
    padded.paste(tile, (0, 0))
    return padded, width, height


def build_manifest(
    image_path: Path,
    output_dir: Path,
    tile_size: int,
    stride: int,
    pad_color: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        width, height = image.size

        x_positions = iter_positions(width, tile_size, stride)
        y_positions = iter_positions(height, tile_size, stride)

        document_id = image_path.stem
        records = []
        tile_count = 0

        for y_index, top in enumerate(y_positions):
            for x_index, left in enumerate(x_positions):
                tile, source_width, source_height = crop_tile(
                    image, left, top, tile_size, pad_color
                )
                tile_name = f"{document_id}_x{x_index:04d}_y{y_index:04d}.png"
                tile_path = tiles_dir / tile_name
                tile.save(tile_path)
                records.append(
                    {
                        "path": portable_path(tile_path),
                        "filename": tile_name,
                        "document_id": document_id,
                        "tile_size": tile_size,
                        "stride": stride,
                        "x_index": x_index,
                        "y_index": y_index,
                        "left": left,
                        "top": top,
                        "source_width": source_width,
                        "source_height": source_height,
                    }
                )
                tile_count += 1

    manifest = {
        "source_image": image_path.resolve().as_posix(),
        "document_id": image_path.stem,
        "tile_size": tile_size,
        "stride": stride,
        "pad_color": pad_color,
        "image_width": width,
        "image_height": height,
        "grid_width": len(x_positions),
        "grid_height": len(y_positions),
        "num_tiles": tile_count,
        "records": records,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Missing source image: {image_path}")

    tile_size = args.tile_size
    stride = args.stride or tile_size
    default_output_dir = ROOT / "data" / "scroll_tiles" / f"{image_path.stem}_{tile_size}"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir

    manifest_path = build_manifest(
        image_path=image_path,
        output_dir=output_dir,
        tile_size=tile_size,
        stride=stride,
        pad_color=args.pad_color,
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
