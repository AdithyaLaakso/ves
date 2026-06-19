# Grayscale-First Diagnostic Metrics Design

## Context

The 2026-06-19 focal-weight probes showed that fixed-threshold ink metrics can
reward visually poor outputs. The epoch-2 checkpoint scored high on ink IoU and
ink recall while producing nearly black sigmoid outputs on concern-class
diagnostics. The target-threshold audit showed why: for manifest records, the
supervised target is the clean grayscale ALPUB crop, not a binary mask, and the
current `target < 0.5` rule marks most concern-class crops as ink.

## Goal

Improve diagnostic reporting without changing training behavior. The project
should keep grayscale reconstruction quality primary, preserve existing metric
columns for continuity, and add better secondary ink-style metrics that are less
dependent on a global `0.5` cutoff.

## Non-Goals

- Do not change model training, loss functions, target tensors, or dataset
  loading in this change.
- Do not remove existing metric names yet; older CSV readers and notes may still
  refer to them.
- Do not launch new AWS training as part of this change.

## Design

Update `model/usage/evaluate_visual_regression.py` so `_image_metrics()` returns
three groups of metrics:

1. Existing grayscale metrics:
   - `mse`
   - `mae`
   - `ssim_like`

2. Fixed-threshold compatibility metrics:
   - Keep existing fields such as `ink_iou`, `ink_precision`, `ink_recall`,
     `pred_ink_fraction`, and `target_ink_fraction`.
   - Add explicit aliases prefixed with `fixed_0_5_`, such as
     `fixed_0_5_ink_iou` and `fixed_0_5_target_ink_fraction`, so future reports
     can distinguish compatibility metrics from preferred diagnostics.

3. New adaptive and soft ink diagnostics:
   - Add per-image Otsu threshold metrics:
     - `otsu_pred_threshold`
     - `otsu_target_threshold`
     - `otsu_ink_iou`
     - `otsu_ink_precision`
     - `otsu_ink_recall`
     - `otsu_pred_ink_fraction`
     - `otsu_target_ink_fraction`
   - Add continuous darkness-as-ink metrics:
     - `soft_ink_iou`
     - `soft_ink_dice`
     - `soft_ink_abs_diff`

The Otsu thresholds should be calculated independently for prediction and
target. They are diagnostics, not labels. The soft metrics should treat darkness
as ink strength with `ink_strength = 1 - grayscale`, avoiding a hard threshold.

## Testing

Add unit tests for `_image_metrics()` using small tensors:

- Existing fixed `0.5` metric behavior remains present and mirrored by the new
  `fixed_0_5_*` aliases.
- Otsu target ink fraction is lower than fixed `0.5` target ink fraction for a
  low-contrast target where most pixels are below `0.5`.
- Soft ink metrics rank a closer grayscale prediction better than a near-black
  prediction against the same grayscale target.

Run the focused test file and at least the existing CLI-help smoke tests that
cover diagnostic scripts.

## Expected Outcome

Per-class and prediction diagnostic CSVs will become wider but more informative.
Future model comparisons can continue to report legacy fixed-threshold metrics,
while journal interpretation and model selection can rely primarily on
grayscale MSE/SSIM-like metrics plus Otsu and soft-mask secondary diagnostics.
