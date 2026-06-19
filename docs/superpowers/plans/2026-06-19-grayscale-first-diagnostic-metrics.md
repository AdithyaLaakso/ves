# Grayscale-First Diagnostic Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add grayscale-first diagnostic metrics while preserving existing fixed-threshold ink metric columns.

**Architecture:** Keep the metric API centralized in `model/usage/evaluate_visual_regression.py` by extending `_image_metrics()`. Downstream scripts already import this function, so per-class and prediction diagnostics inherit the new columns without training changes.

**Tech Stack:** Python, PyTorch tensors, NumPy-compatible tensor operations, `unittest`, existing diagnostic CLIs.

---

### Task 1: Add Failing Metric Tests

**Files:**
- Modify: `tests/test_gpu_readiness.py`

- [ ] **Step 1: Add tests for fixed aliases, Otsu metrics, and soft metrics**

Add these methods before `test_visual_review_cli_help_runs_from_repo_root`:

```python
    def test_image_metrics_reports_fixed_threshold_aliases(self):
        from usage.evaluate_visual_regression import _image_metrics

        pred = torch.tensor([[0.25, 0.75], [0.25, 0.75]])
        target = torch.tensor([[0.25, 0.25], [0.75, 0.75]])

        metrics = _image_metrics(pred, target)

        self.assertEqual(metrics["fixed_0_5_ink_iou"], metrics["ink_iou"])
        self.assertEqual(metrics["fixed_0_5_ink_precision"], metrics["ink_precision"])
        self.assertEqual(metrics["fixed_0_5_ink_recall"], metrics["ink_recall"])
        self.assertEqual(metrics["fixed_0_5_pred_ink_fraction"], metrics["pred_ink_fraction"])
        self.assertEqual(metrics["fixed_0_5_target_ink_fraction"], metrics["target_ink_fraction"])

    def test_image_metrics_otsu_threshold_avoids_global_half_overmarking(self):
        from usage.evaluate_visual_regression import _image_metrics

        pred = torch.tensor(
            [
                [0.18, 0.20, 0.22, 0.24],
                [0.20, 0.22, 0.24, 0.26],
                [0.62, 0.66, 0.70, 0.74],
                [0.68, 0.72, 0.76, 0.80],
            ]
        )
        target = pred.clone()

        metrics = _image_metrics(pred, target)

        self.assertGreater(metrics["fixed_0_5_target_ink_fraction"], 0.45)
        self.assertLess(metrics["otsu_target_ink_fraction"], metrics["fixed_0_5_target_ink_fraction"])
        self.assertGreaterEqual(metrics["otsu_target_threshold"], 0.0)
        self.assertLessEqual(metrics["otsu_target_threshold"], 1.0)

    def test_image_metrics_soft_ink_prefers_closer_grayscale_prediction(self):
        from usage.evaluate_visual_regression import _image_metrics

        target = torch.tensor([[0.15, 0.25], [0.70, 0.85]])
        close_pred = torch.tensor([[0.18, 0.28], [0.68, 0.82]])
        black_pred = torch.zeros((2, 2))

        close_metrics = _image_metrics(close_pred, target)
        black_metrics = _image_metrics(black_pred, target)

        self.assertGreater(close_metrics["soft_ink_iou"], black_metrics["soft_ink_iou"])
        self.assertGreater(close_metrics["soft_ink_dice"], black_metrics["soft_ink_dice"])
        self.assertLess(close_metrics["soft_ink_abs_diff"], black_metrics["soft_ink_abs_diff"])
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_gpu_readiness.GPUReadinessTests.test_image_metrics_reports_fixed_threshold_aliases tests.test_gpu_readiness.GPUReadinessTests.test_image_metrics_otsu_threshold_avoids_global_half_overmarking tests.test_gpu_readiness.GPUReadinessTests.test_image_metrics_soft_ink_prefers_closer_grayscale_prediction -v
```

Expected: fail with missing keys such as `fixed_0_5_ink_iou`.

### Task 2: Implement Metrics

**Files:**
- Modify: `model/usage/evaluate_visual_regression.py`

- [ ] **Step 1: Add helper functions above `_image_metrics()`**

```python
def _otsu_threshold_tensor(image: torch.Tensor) -> float:
    values = image.detach().cpu().float().clamp(0.0, 1.0).flatten()
    if values.numel() == 0:
        return 0.5
    bins = torch.clamp((values * 255.0).round().long(), 0, 255)
    hist = torch.bincount(bins, minlength=256).float()
    total = hist.sum()
    if total <= 0:
        return 0.5

    bin_values = torch.arange(256, dtype=torch.float32)
    sum_total = (bin_values * hist).sum()
    weight_background = torch.tensor(0.0)
    sum_background = torch.tensor(0.0)
    max_between = torch.tensor(-1.0)
    best_threshold = 128

    for threshold in range(256):
        count = hist[threshold]
        weight_background = weight_background + count
        if weight_background <= 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground <= 0:
            break
        sum_background = sum_background + bin_values[threshold] * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between > max_between:
            max_between = between
            best_threshold = threshold

    return best_threshold / 255.0


def _binary_ink_metrics(pred_ink: torch.Tensor, target_ink: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    intersection = torch.logical_and(pred_ink, target_ink).sum().float()
    union = torch.logical_or(pred_ink, target_ink).sum().float()
    pred_ink_count = pred_ink.sum().float()
    target_ink_count = target_ink.sum().float()
    false_ink = torch.logical_and(pred_ink, ~target_ink).sum().float()
    missed_ink = torch.logical_and(~pred_ink, target_ink).sum().float()

    h, w = target_ink.shape[-2:]
    y0, y1 = h // 4, h - h // 4
    x0, x1 = w // 4, w - w // 4
    interior_pred_ink = pred_ink[y0:y1, x0:x1].float().mean()
    interior_target_ink = target_ink[y0:y1, x0:x1].float().mean()

    eps = torch.tensor(1e-6)
    return {
        f"{prefix}ink_iou": float((intersection / (union + eps)).item()),
        f"{prefix}ink_precision": float((intersection / (pred_ink_count + eps)).item()),
        f"{prefix}ink_recall": float((intersection / (target_ink_count + eps)).item()),
        f"{prefix}pred_ink_fraction": float(pred_ink.float().mean().item()),
        f"{prefix}target_ink_fraction": float(target_ink.float().mean().item()),
        f"{prefix}ink_fraction_abs_diff": float(
            torch.abs(pred_ink.float().mean() - target_ink.float().mean()).item()
        ),
        f"{prefix}false_ink_fraction": float((false_ink / pred_ink.numel()).item()),
        f"{prefix}missed_ink_fraction": float((missed_ink / pred_ink.numel()).item()),
        f"{prefix}interior_pred_ink_fraction": float(interior_pred_ink.item()),
        f"{prefix}interior_target_ink_fraction": float(interior_target_ink.item()),
        f"{prefix}interior_ink_abs_diff": float(
            torch.abs(interior_pred_ink - interior_target_ink).item()
        ),
    }
```

- [ ] **Step 2: Extend `_image_metrics()`**

Replace the fixed-threshold block with calls to `_binary_ink_metrics()`, add `fixed_0_5_` aliases, Otsu metrics, and soft metrics.

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run the same command from Task 1 Step 2.

Expected: all three tests pass.

### Task 3: Verify Diagnostic CLIs

**Files:**
- Modify: none unless tests fail.

- [ ] **Step 1: Run existing diagnostic CLI smoke tests**

```bash
python3 -m unittest tests.test_gpu_readiness.GPUReadinessTests.test_per_class_metrics_cli_help_runs_from_repo_root tests.test_gpu_readiness.GPUReadinessTests.test_prediction_diagnostics_cli_help_runs_from_repo_root -v
```

Expected: both tests pass.

- [ ] **Step 2: Run full GPU readiness test file if practical**

```bash
python3 -m unittest tests.test_gpu_readiness -v
```

Expected: all tests pass. If environment constraints prevent full execution, record the narrower commands that passed.

- [ ] **Step 3: Commit**

```bash
git add model/usage/evaluate_visual_regression.py tests/test_gpu_readiness.py
git commit -m "Add grayscale-first diagnostic metrics"
```
