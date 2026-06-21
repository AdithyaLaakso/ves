# Hybrid Auxiliary Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in hybrid target path that keeps the current grayscale reconstruction target and adds an auxiliary ink-aware target, with a clean rollback back to grayscale-only training.

**Architecture:** The dataset will emit a paired target only when hybrid mode is enabled. The model will expose a second head in hybrid mode, the training loop will sum a primary grayscale loss and an auxiliary loss, and the diagnostic scripts will show both outputs while preserving the existing grayscale-first baseline path. Default behavior stays unchanged so rollback is a flag flip, not a code revert.

**Tech Stack:** Python, PyTorch, pytest, PIL, existing VES training and diagnostic scripts.

---

### Task 1: Add the hybrid target contract

**Files:**
- Modify `model/settings.py`
- Modify `model/dataset.py`
- Modify `tests/test_gpu_readiness.py`

- [ ] **Step 1: Write the failing test**

Add a test that turns on hybrid mode and expects the dataset to return a primary grayscale target plus an auxiliary ink target. Also add a regression test that confirms the default path still returns the current single target when hybrid mode is off.

```python
def test_hybrid_target_returns_primary_and_auxiliary_targets(self):
    import importlib
    import dataset
    import settings

    with patch.dict("os.environ", {"VES_HYBRID_TARGET": "1"}, clear=False):
        importlib.reload(settings)
        seg_data = dataset.SegData(level=0)
        inputs, targets = seg_data[0]

    primary, auxiliary = targets
    self.assertEqual(primary.shape, auxiliary.shape)
    self.assertEqual(primary.dtype, torch.float32)
    self.assertEqual(auxiliary.dtype, torch.float32)
    self.assertGreaterEqual(float(auxiliary.min()), 0.0)
    self.assertLessEqual(float(auxiliary.max()), 1.0)

def test_default_target_remains_grayscale_only(self):
    import dataset

    seg_data = dataset.SegData(level=0)
    _, target = seg_data[0]

    self.assertTrue(torch.is_tensor(target))
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_gpu_readiness.py -k hybrid_target -v
```

Expected: fail because `VES_HYBRID_TARGET` is not yet implemented and the dataset still returns the current single target shape.

- [ ] **Step 3: Write the minimal implementation**

Add two env-backed settings in `model/settings.py`:

```python
hybrid_target_enabled = _env_flag("VES_HYBRID_TARGET", False)
aux_target_weight = _env_float("VES_AUX_TARGET_WEIGHT", 0.25)
```

In `model/dataset.py`, add a helper that derives the auxiliary target from the normalized clean crop:

```python
def _build_auxiliary_target(self, clean: Image.Image) -> Tensor:
    array = np.array(clean, dtype=np.float32) / 255.0
    auxiliary = 1.0 - array
    return torch.from_numpy(auxiliary).unsqueeze(0)
```

Then make `_get_manifest_pair()` return either:

```python
return input_tensor, target_tensor
```

or, when hybrid mode is enabled:

```python
return input_tensor, (target_tensor, auxiliary_tensor)
```

Update `collate_fn()` so it can stack a tuple of two tensors for hybrid targets without breaking the existing multitask `(mask, class_label)` path.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_gpu_readiness.py -k hybrid_target -v
```

Expected: pass, and the default grayscale-only case should still behave exactly as before.

- [ ] **Step 5: Commit**

```bash
git add model/settings.py model/dataset.py tests/test_gpu_readiness.py
git commit -m "feat: add hybrid auxiliary target contract"
```

### Task 2: Add the hybrid model and loss path

**Files:**
- Modify `model/model.py`
- Modify `model/loss.py`
- Modify `model/train_reconstruction.py`
- Modify `tests/test_gpu_readiness.py`

- [ ] **Step 1: Write the failing test**

Add a test that enables hybrid mode, builds the model, and expects the forward pass to return two heads. Add a small loss test that confirms the training wrapper can combine primary and auxiliary losses with the configured auxiliary weight.

```python
def test_hybrid_model_returns_two_outputs(self):
    import importlib
    import settings
    import model as ves_model

    with patch.dict("os.environ", {"VES_HYBRID_TARGET": "1"}, clear=False):
        importlib.reload(settings)
        importlib.reload(ves_model)
        net = ves_model.build_model()
        outputs = net(torch.zeros(1, 1, settings.input_size, settings.input_size))

    self.assertIsInstance(outputs, tuple)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].shape, outputs[1].shape)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_gpu_readiness.py -k hybrid_model -v
```

Expected: fail because the model still returns a single output and the training loss has no auxiliary branch.

- [ ] **Step 3: Write the minimal implementation**

In `model/model.py`, add a second decoder head that is only constructed when `settings.hybrid_target_enabled` is true. Keep the default single-head return path unchanged. In hybrid mode, `forward()` should return a tuple:

```python
primary_logits, auxiliary_logits
```

In `model/loss.py`, add a `HybridTargetLoss` wrapper that computes:

```python
primary_loss = BinarySegmentationLoss()(primary_pred, primary_target)
aux_loss = BinarySegmentationLoss()(aux_pred, aux_target)
total_loss = primary_loss + settings.aux_target_weight * aux_loss
```

In `model/train_reconstruction.py`, choose the loss object based on the hybrid flag and keep the current `move_batch_to_device()` logic. The training loop should unpack `(primary_target, auxiliary_target)` only when hybrid mode is active.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_gpu_readiness.py -k hybrid_model -v
```

Expected: pass, with default single-output behavior unchanged when hybrid mode is off.

- [ ] **Step 5: Commit**

```bash
git add model/model.py model/loss.py model/train_reconstruction.py tests/test_gpu_readiness.py
git commit -m "feat: add hybrid auxiliary model and loss path"
```

### Task 3: Extend diagnostics and keep rollback clean

**Files:**
- Modify `model/usage/evaluate_visual_regression.py`
- Modify `model/usage/prediction_diagnostic_sheets.py`
- Modify `model/usage/visual_review_sheets.py`
- Modify `tests/test_gpu_readiness.py`

- [ ] **Step 1: Write the failing test**

Add a small evaluation test that feeds a tuple of model outputs into the diagnostic helpers and expects the primary grayscale metrics plus auxiliary metrics to be present.

```python
def test_hybrid_diagnostics_include_auxiliary_metrics(self):
    from usage.evaluate_visual_regression import _hybrid_image_metrics

    target = torch.zeros(1, 1, 8, 8)
    pred = torch.zeros(1, 1, 8, 8)
    aux_target = torch.ones(1, 1, 8, 8)
    aux_pred = torch.ones(1, 1, 8, 8)
    metrics = _hybrid_image_metrics(pred, target, aux_pred, aux_target)

    self.assertIn("primary_mse", metrics)
    self.assertIn("primary_ssim_like", metrics)
    self.assertIn("aux_mse", metrics)
    self.assertIn("aux_ssim_like", metrics)
```

Add a second test around the output extraction helper so hybrid mode can still be visualized without changing the grayscale baseline path.

```python
def test_hybrid_output_extraction_returns_primary_and_auxiliary(self):
    from usage.visual_review_sheets import _extract_model_outputs

    primary = torch.zeros(1, 1, 8, 8)
    auxiliary = torch.ones(1, 1, 8, 8)
    extracted_primary, extracted_auxiliary = _extract_model_outputs((primary, auxiliary))

    self.assertTrue(torch.equal(extracted_primary, primary))
    self.assertTrue(torch.equal(extracted_auxiliary, auxiliary))
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_gpu_readiness.py -k diagnostics -v
```

Expected: fail until the diagnostics scripts understand hybrid outputs and expose auxiliary columns.

- [ ] **Step 3: Write the minimal implementation**

In `model/usage/evaluate_visual_regression.py`, teach the evaluation path to accept either a single output or a `(primary, auxiliary)` tuple and emit auxiliary metrics alongside the existing grayscale metrics.

In `model/usage/prediction_diagnostic_sheets.py` and `model/usage/visual_review_sheets.py`, keep the default single-output render unchanged, but in hybrid mode render and label both heads so the new run can be inspected side by side.

Keep the rollback path simple:

```bash
unset VES_HYBRID_TARGET
unset VES_AUX_TARGET_WEIGHT
```

With those unset, the code should behave exactly like the current grayscale-only baseline.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_gpu_readiness.py -k diagnostics -v
```

Expected: pass, with the existing grayscale-first diagnostics still intact and the new auxiliary diagnostics available only when hybrid mode is enabled.

- [ ] **Step 5: Commit**

```bash
git add model/usage/evaluate_visual_regression.py model/usage/prediction_diagnostic_sheets.py model/usage/visual_review_sheets.py tests/test_gpu_readiness.py
git commit -m "feat: add hybrid target diagnostics"
```

### Task 4: Verification and experiment handoff

**Files:**
- Modify `docs/research_journal.md`
- Modify `docs/experiment_ledger.csv`

- [ ] **Step 1: Run the focused test set**

Run:

```bash
python3 -m pytest tests/test_gpu_readiness.py
```

Expected: all existing readiness tests plus the new hybrid-target tests pass.

- [ ] **Step 2: Check the hybrid run contract locally**

Run one dry evaluation command against a saved checkpoint with `VES_HYBRID_TARGET=1` to confirm the scripts still produce readable output and auxiliary metrics.

- [ ] **Step 3: Record the experiment metadata**

Once the first AWS hybrid run is launched, add a research journal entry and ledger row that capture:

```text
hybrid target enabled
auxiliary weight
checkpoint source
evaluation summary
rollback status
```

- [ ] **Step 4: Commit**

```bash
git add docs/research_journal.md docs/experiment_ledger.csv
git commit -m "docs: record hybrid target experiment"
```

## Coverage Check

- Spec goal: covered by Tasks 1-3.
- Default grayscale behavior preserved: covered by Task 1 and Task 3 rollback.
- Auxiliary target derived from same clean crop: covered by Task 1.
- Model and training path distinguish hybrid from non-hybrid: covered by Task 2.
- Evaluation comparable to baseline: covered by Task 3.
- Rollback path simple and explicit: covered by Tasks 1 and 3.

## Self-Review

- No placeholders remain.
- The plan keeps the default path unchanged.
- The file boundaries follow existing module responsibilities.
- The hybrid target, model, loss, and diagnostics changes are separated so each task can land and be verified independently.
