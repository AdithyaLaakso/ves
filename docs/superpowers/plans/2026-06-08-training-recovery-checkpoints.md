# Training Recovery Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resumable intra-epoch training checkpoints for long VES GPU training runs.

**Architecture:** A small `model/training_recovery.py` module owns recovery checkpoint serialization, atomic writes, retention, and RNG capture. `model/settings.py` exposes environment controls, and `model/train_reconstruction.py` calls the recovery module at batch boundaries and restores training state when requested.

**Tech Stack:** Python, PyTorch, unittest, existing VES training scripts.

---

### Task 1: Recovery Checkpoint Module

**Files:**
- Create: `model/training_recovery.py`
- Modify: `tests/test_gpu_readiness.py`

- [ ] Write failing tests for saving and loading a recovery checkpoint with model, optimizer, scheduler, scaler, progress, and RNG state.
- [ ] Run `python -m unittest tests.test_gpu_readiness.TrainingRecoveryTests -v` and confirm import or assertion failure.
- [ ] Implement `capture_rng_state`, `restore_rng_state`, `save_recovery_checkpoint`, `load_recovery_checkpoint`, `latest_checkpoint_path`, and retention cleanup.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Settings

**Files:**
- Modify: `model/settings.py`
- Modify: `tests/test_gpu_readiness.py`

- [ ] Add tests proving the recovery settings read from environment variables.
- [ ] Add `VES_STEP_CHECKPOINT_EVERY_BATCHES`, `VES_STEP_CHECKPOINT_EVERY_MINUTES`, `VES_KEEP_STEP_CHECKPOINTS`, and `VES_RESUME_TRAINING_STATE`.
- [ ] Re-run the settings tests.

### Task 3: Training Loop Integration

**Files:**
- Modify: `model/train_reconstruction.py`

- [ ] Change `train_epoch` to accept `epoch`, `start_batch`, `global_step`, scheduler/scaler/optimizer state, and recovery interval configuration.
- [ ] Save recovery checkpoints during training when the batch or time interval is reached.
- [ ] On `VES_RESUME_TRAINING_STATE`, restore model, optimizer, scheduler, scaler, RNG state, epoch, batch, and global step.
- [ ] Preserve existing epoch checkpoint and `new.pth` behavior.

### Task 4: Documentation and Verification

**Files:**
- Modify: `README.md`

- [ ] Document how to run full-dataset training with `tmux`, tee logs, recovery checkpoint intervals, and `VES_RESUME_TRAINING_STATE`.
- [ ] Run `python -m unittest tests.test_gpu_readiness -v`.
- [ ] Run a minimal CPU smoke command if local dependencies permit.
