# Training Recovery Checkpoints Design

## Goal

Make long GPU training runs recoverable when SSH drops, EC2 is force-stopped, or the process exits before an epoch completes.

## Current Problem

`model/train_reconstruction.py` saves checkpoints only after a full epoch. On the full ALPUB_v2 dataset, one epoch can run for hours. If EC2 is stopped before the epoch boundary, all progress since the previous epoch checkpoint is lost.

## Design

Add step-level recovery checkpoints that are separate from the existing epoch checkpoints. Existing model-weight resumes through `VES_RESUME_FROM` continue to work. A new training-state resume path restores more state with `VES_RESUME_TRAINING_STATE`.

Recovery checkpoints will include:

- model state
- optimizer state
- scheduler state
- AMP scaler state
- epoch index
- next batch index
- global step
- Python, NumPy, and PyTorch RNG state
- basic metadata such as run directory and checkpoint version

The training loop will write `checkpoints/recovery-latest.pt` atomically by first saving to a temporary file and then replacing the latest path. Optional retained snapshots use names such as `recovery-epoch0-batch1500.pt`.

## Configuration

Add environment-backed settings:

- `VES_STEP_CHECKPOINT_EVERY_BATCHES`: save every N training batches; `0` disables batch-count checkpoints.
- `VES_STEP_CHECKPOINT_EVERY_MINUTES`: save after N elapsed minutes; `0` disables time-based checkpoints.
- `VES_KEEP_STEP_CHECKPOINTS`: number of retained numbered recovery snapshots; `0` keeps only `recovery-latest.pt`.
- `VES_RESUME_TRAINING_STATE`: path to a full recovery checkpoint.

Defaults should be conservative: recovery checkpointing is enabled for long runs with a batch interval, but smoke tests avoid extra files unless explicitly configured.

## Resume Semantics

When `VES_RESUME_TRAINING_STATE` is provided, training restores the full state and skips completed batches in the current epoch. This is practical recovery, not bit-for-bit replay. Weighted sampler replay may not be exact, but preserving model, optimizer, scheduler, scaler, and RNG state is enough to avoid losing hours of useful training.

When `VES_RESUME_FROM` is provided, behavior remains weight-only resume from the existing checkpoint format.

## Operational Guidance

Long EC2 jobs should still run under `tmux` and tee stdout/stderr to a file in the run directory. Recovery checkpoints reduce lost work, but they do not replace external logs.

## Testing

Unit tests should cover recovery checkpoint save/load behavior without requiring the dataset or GPU. A CPU smoke test can verify that training settings expose the new environment variables.
