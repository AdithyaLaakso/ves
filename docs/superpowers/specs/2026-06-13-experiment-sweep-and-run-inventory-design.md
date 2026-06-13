# Experiment Sweep and Run Inventory Design

## Purpose

The project needs a repeatable way to run short focal-weight probes above the baseline value of `1` without manually discussing and launching each test. The immediate experiment is a bounded focal-weight sweep that avoids the known blob-heavy behavior seen around `10`.

The project also needs organizational cleanup for previous outputs. Cleanup means cataloging, naming, and locating existing runs and review artifacts. It does not mean changing the underlying data, checkpoints, logs, metrics, or visual outputs.

## Scope

This design covers two connected pieces:

1. An experiment wrapper that runs a fixed set of short training probes and produces comparable review artifacts.
2. A run inventory that indexes existing and future run outputs before any physical rename or folder movement.

The first implementation should not add progressive in-run weight schedules. Constant focal-weight probes should establish the useful range first. A later experiment can test ramps near the best non-blob boundary.

## Experiment Wrapper

The wrapper should drive the existing `model/setup.zsh` training entry point rather than create a parallel training path. It should set environment variables, run one probe per configured focal weight, then optionally generate existing visual and metric review outputs.

The first sweep should use constant `VES_FOCAL_WEIGHT` values:

```text
1.25, 1.5, 2.0, 2.5, 3.0, 4.0
```

The wrapper should freeze all other controls across the sweep:

```text
VES_SEED
VES_MAX_SIZE
VES_NUM_EPOCHS
VES_BATCH_SIZE
VES_SIZE_PROFILE
VES_RESUME_FROM
VES_MSE_WEIGHT
VES_CLASS_WEIGHT
VES_FOCAL_ALPHA
VES_FOCAL_GAMMA
```

Each run should write to a child directory under a single experiment directory:

```text
model/runs/experiments/focal-weight-sweep-YYYYMMDD/
  experiment.json
  runs/
    focal_1_25/
    focal_1_50/
    focal_2_00/
    focal_2_50/
    focal_3_00/
    focal_4_00/
  reviews/
  metrics/
```

The wrapper should stop after the configured set of probes. Human review remains required before launching a second sweep.

## Review Artifacts

After each successful training run, the wrapper should optionally run existing review scripts with fixed sample selection. At minimum, visual review sheets should reuse the same seed or explicit sample indices across all probes.

The wrapper should record paths to generated review artifacts in `experiment.json` and in the run inventory. If review generation fails after training succeeds, the run should remain indexed with a failed review status instead of hiding the completed checkpoint.

## Run Inventory

Before any renaming, the project should add:

```text
model/runs/run_inventory.json
model/runs/RUN_INDEX.md
```

The JSON inventory is the machine-readable source of truth. The Markdown index is the human-readable table for quick review.

Each inventory entry should include:

```json
{
  "id": "20260610-local-focal10-256x2",
  "category": "local-probe",
  "path": "model/runs/20260610-local-focal10-256x2",
  "checkpoint": "model/runs/20260610-local-focal10-256x2/new.pth",
  "review_paths": [],
  "metric_paths": [],
  "known_variables": {
    "VES_FOCAL_WEIGHT": 10,
    "VES_MAX_SIZE": 256,
    "VES_NUM_EPOCHS": 2
  },
  "status": "indexed",
  "notes": "Known blob-heavy focal-weight failure case."
}
```

For existing runs, unknown values should be marked as `null` or omitted rather than guessed. For new wrapper-created runs, values should come directly from the environment/configuration used to launch the run.

## Rename Phase

The rename phase should happen only after the initial inventory has been generated and reviewed.

The migration process should:

1. Identify ambiguous or misleading folder names from the inventory.
2. Propose new names without changing files.
3. Move only approved folders.
4. Update `run_inventory.json` with `previous_paths`.
5. Preserve all checkpoints, logs, visual sheets, and metric files unchanged.

Existing notes and scripts may refer to old paths, so the inventory must retain old-to-new path mappings after any rename.

## Error Handling

The wrapper should fail fast if required inputs are missing, including the baseline checkpoint, manifest, or Python environment with Torch. If a probe fails, the wrapper should record the failed status and stop by default. A later option can allow continuing after failure, but the first implementation should favor simple, auditable behavior.

Review generation failures should not erase successful training outputs. They should be recorded separately from training failures.

## Testing

Tests should cover configuration expansion without launching real training. A dry-run mode should print or return the planned commands and inventory entries. Unit tests should verify:

1. Focal weights become stable directory names such as `focal_1_25`.
2. Shared controls remain identical across generated probe configs.
3. Inventory entries preserve paths and known variables.
4. Existing runs with unknown settings are indexed without fabricated values.
5. Failed training and failed review states are represented distinctly.

The implementation can be verified manually with a dry run before any GPU training is launched.
