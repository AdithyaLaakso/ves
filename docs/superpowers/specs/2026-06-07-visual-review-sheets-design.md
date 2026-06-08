# Visual Review Sheets Design

## Goal

Create a repeatable, headless visual inspection workflow for comparing degraded ALPUB inputs, model outputs, and clean targets from saved checkpoints.

## Design

Add a new script under `model/usage/` rather than changing the stale interactive `model/visualize_model.py`. The script will use the current `dataset.SegData` path so generated review samples match the manifest-based training pipeline with on-the-fly degradation.

The script will accept a checkpoint path, choose a deterministic sample set from the loaded dataset, run inference in evaluation mode, and save one or more PNG contact sheets. Each row will show:

1. degraded input
2. model output
3. clean target

The output directory will default to a `visual_review/` folder beside the checkpoint when possible, with command-line override support. A metadata JSON file will record the checkpoint, seed, selected indices, labels, document IDs, and active size profile.

## Constraints

- Must work over SSH on EC2 without a display server.
- Must support the current manifest dataset.
- Must be deterministic when a seed is provided.
- Must be useful for comparing multiple checkpoints against the same sample indices.
- Must keep the old interactive visualizer untouched for now.

## Testing

Unit tests will cover deterministic sample selection, output-directory inference, and contact-sheet file creation with a lightweight fake model. Full GPU inference remains a manual EC2 workflow because repository tests should run quickly on CPU.
