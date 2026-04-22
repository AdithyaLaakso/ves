# Real Scroll Tile Workflow

## Purpose

This note describes the current path for applying the model to a large real scroll image without resizing the entire image down to the model input size.

## Why This Is Needed

The current single-image inference script resizes one whole input image to the configured model input size. That is acceptable for a small sample image, but it is not appropriate for a large real scroll image. A large source image instead needs to be divided into fixed-size windows, processed tile by tile, and then saved in a structured way.

## What `document_id` Means Here

For real-scroll tiling, `document_id` is metadata that identifies all tiles as belonging to the same source image or source document. It is useful for:

- keeping tiles grouped by source
- building manifests
- supporting future evaluation or split logic

It is not required by the model itself at inference time. The model operates on the tile image. The `document_id` is carried alongside the tile records so the surrounding pipeline knows which source image those tiles came from.

## New Scripts

The repository now includes:

- `model/usage/build_scroll_tile_manifest.py`
- `model/usage/infer_scroll_tiles.py`

## Tiling Step

The tiling script cuts a large source image into fixed-size grayscale tiles and writes a manifest describing:

- the source image
- the shared `document_id`
- tile coordinates
- source widths and heights for edge tiles
- paths to the saved tile images

Example:

```bash
python model/usage/build_scroll_tile_manifest.py path/to/scroll_image.tif --tile-size 64
```

This creates a directory under `data/scroll_tiles/` containing:

- `tiles/`
- `manifest.json`

## Inference Step

The inference script reads that manifest, runs the model on each tile, saves the predicted output tile, and optionally creates a stitched preview image.

Example:

```bash
cd model
VES_SIZE_PROFILE=64 python usage/infer_scroll_tiles.py ../data/scroll_tiles/scroll_image_64/manifest.json --save-preview
```

## Checkpoint Compatibility

The size profile is part of the model architecture. A checkpoint trained under the `96 x 96` profile cannot be used directly with the `64 x 64` profile, because the patch embeddings, positional-bias shapes, and decoder dimensions differ. In practical terms, the tiling and manifest workflow can be prepared now, but meaningful `64 x 64` inference still requires weights trained for that `64 x 64` configuration.

## Why This Matters for the Paper

This workflow makes the transition to `64 x 64` inference more concrete. It shows that the project is not merely changing a configuration constant, but moving toward an actual submission-relevant evaluation path for large real images. It also clarifies that `document_id` remains useful for organizing real-scroll tiles even though inference itself is performed on each tile independently.
