# VES Research Journal

This journal is the source of truth for research context, experiment rationale,
decisions, results, and open questions. It is written chronologically so new
contributors can understand why the project changed direction and how current
choices were reached.

Structured experiment summaries are mirrored in `docs/experiment_ledger.csv`,
but the journal is authoritative when interpretation or context matters.

## 2026-06-16

### Focus

Establish a durable research record for the current VES fork before continuing
more model experiments or paper revisions.

### Context

The original project team is largely inactive, and the original organizer has
not been meaningfully participating in the current work. I do not have formal
machine-learning expertise, but I have stayed involved because I bring Ancient
Greek context that the remaining project otherwise lacks.

The current work is based on a fork of Hugo's original project. The original
training direction used Modern Greek ballpoint handwriting samples. This fork is
testing a different assumption: Ancient Greek uncial letterforms are a better
proxy task for the Vesuvius/Greek-letter problem than modern handwriting.

Because this diverges from the original author's assumptions, the fork needs a
clear record of what changed, why it changed, what evidence supports each
change, and which questions remain unresolved.

### Changes

- Added a design for a research journal and experiment ledger in
  `docs/superpowers/specs/2026-06-14-research-journal-and-experiment-ledger-design.md`.
- Adopted a journal-plus-ledger model:
  - `docs/research_journal.md` is the narrative source of truth.
  - `docs/experiment_ledger.csv` is a sortable summary of meaningful
    experiments.
  - `docs/research_log_guide.md` explains how future contributors should update
    both files.
- Decided to leave generated artifacts under `model/runs/` and
  `model/downloads/` out of git for now while reviewing which outputs, if any,
  should become curated paper or onboarding artifacts.

### Experiments

- `20260607-resumed-32768`: resumed the interrupted 32768-sample AWS run from
  an epoch-4 checkpoint. Validation loss improved to `3.5407`, confirming that
  checkpoint resume could continue the interrupted training path.
- `20260609-full-epoch1`: completed one full ALPUB_v2 manifest epoch on AWS.
  Validation loss improved to `2.6946`.
- `20260610-full-epoch2`: continued from the full epoch-1 checkpoint.
  Validation loss improved to `2.4500`.
- `20260610-visual-regression-review`: visual comparisons showed that aggregate
  validation-loss improvement did not cleanly translate into better human
  visual quality for all classes. Omicron and epsilon became concern classes.
- `20260613-focal-sweep-wrapper`: added a dry-run-capable focal-weight sweep
  wrapper and run inventory tooling.
- `20260613-controlled-focal-dryrun`: ran a controlled dry run for focal
  weights `1.25` and `2.5`, producing a valid manifest and planned inventory
  without launching training.

### Decisions

- Keep the Ancient Greek uncial proxy-task direction, but document the rationale
  explicitly because it is a meaningful departure from the original Modern Greek
  handwriting assumption.
- Pause additional paid full-dataset training until the visual-regression
  concern is better understood.
- Use controlled sweeps rather than ad hoc weight changes. Each run should
  record its purpose, configuration, artifact paths, result, interpretation, and
  next step.
- Treat validation loss as useful but insufficient. Visual review,
  class-specific behavior, and target/loss formulation need to be tracked
  alongside scalar metrics.
- Keep large generated outputs out of git unless a small representative artifact
  is intentionally selected for paper, onboarding, or review.

### Results

The current branch has a working CUDA training path, recovery checkpoints,
visual review tooling, full-dataset AWS evidence, a focal-weight sweep wrapper,
and a dry-run inventory workflow. The project is not at a final model result.
It is at a more disciplined research phase where experiment purpose, evidence,
and interpretation need to be logged consistently.

### Open Questions

- Does the current grayscale reconstruction target align with the desired
  letter-segmentation behavior?
- Are omicron, epsilon, and lunate sigma weak because of data ambiguity, output
  resolution, objective mismatch, class weighting, or some combination?
- Which loss-weight sweep values should be tested first, and what visual or
  metric evidence will count as improvement?
- Which generated artifacts should be curated for the paper or contributor
  onboarding, and which should remain local-only?
- Can a new contributor with agentic-engineering experience help automate run
  logging, handoff documents, or experiment setup without taking over research
  direction?

## 2026-06-19

### Focus

Run a bounded AWS focal-weight probe from the full-dataset epoch-2 checkpoint
and decide whether increasing focal pressure above the current candidate range
is worth further testing.

### Context

The focal-weight sweep wrapper and run inventory were already in place. The
first AWS dry run confirmed that six planned probes could be generated without
launching training. A from-scratch `focal_1_25` smoke run with `max_size=256`
only produced flat gray outputs, which confirmed the wrapper path but was not a
meaningful model-quality test.

The next probe therefore resumed from the full-dataset epoch-2 checkpoint:
`runs/20260610T015145Z-07415d46e/new.pth`.

Existing local notes show that explicit focal-weight `1` data does exist from
earlier bounded comparisons:

- `20260610-local-focal1-128`
- `20260610-local-focal1-256x2`
- AWS/local folders named `20260610-focal1-full` and `20260611-focal1-full`

However, the epoch-2 checkpoint used as the resume baseline is not recorded in
the current ledger as focal weight `1`. The code default for unspecified
`VES_FOCAL_WEIGHT` is `10.0`, so future notes should not call that checkpoint a
focal-1 baseline unless its original environment is verified from logs.

### Changes

- Started the stopped AWS `g6.xlarge` instance `ves-gpu-smoke-01`.
- Pulled the current `uncial-pilot-prep` branch on AWS.
- Used variable-based shell commands for long local/AWS paths to reduce copy and
  paste errors.
- Copied the completed AWS experiment folders into `model/downloads/` for local
  inspection, while keeping generated artifacts out of git.

### Experiments

- `20260619-focal-1-25-resume-smoke-256`: resumed from epoch 2 with
  `VES_FOCAL_WEIGHT=1.25`, `max_size=256`, `num_epochs=1`, `batch_size=4`.
  The run completed and wrote a checkpoint. Visual review was active rather
  than flat gray, but looked nearly identical to the epoch-2 baseline on the
  fixed 24-sample review set.
- `20260619-focal-1-25-2048`: resumed from epoch 2 with
  `VES_FOCAL_WEIGHT=1.25`, `max_size=2048`, `num_epochs=1`, `batch_size=4`.
  The run completed and generated fixed-seed review sheets.
- `20260619-focal-1-00-2048`: resumed from the same epoch-2 checkpoint with
  `VES_FOCAL_WEIGHT=1.0`, `max_size=2048`, `num_epochs=1`, `batch_size=4`.
  This run was accidentally written into the existing
  `runs/experiments/focal-weight-1-25-2048-20260619` experiment directory as
  `runs/focal_1_00/`, beside the earlier `runs/focal_1_25/` output. The copied
  local folder therefore contains both review sets, but its `experiment.json`
  now describes the later focal-1.0 probe.
- `20260619-focal-0-25-2048`: resumed from the same epoch-2 checkpoint with
  `VES_FOCAL_WEIGHT=0.25`, `max_size=2048`, `num_epochs=1`, `batch_size=4`.
  The run completed in its own experiment directory and generated fixed-seed
  review sheets.

### Decisions

- Do not run the full upward focal-weight sweep yet. The `1.25` value is already
  visually aggressive when given enough samples to move the model.
- Treat the previous focal-1 evidence as relevant, but keep it distinct from
  the epoch-2 baseline until the exact baseline focal setting is verified.
- Next focal-weight tests should move downward or compare against explicit
  `VES_FOCAL_WEIGHT=1`, rather than increasing to `1.5`, `2.0`, `2.5`, `3.0`,
  or `4.0`.

### Results

The 2048-sample resumed `focal_1_25` probe visibly worsened the model output.
Compared with the epoch-2 baseline sheets on the same fixed review indices, the
model-output column became more saturated and blob-like, with large black
regions, thick coarse shapes, and less letter-like structure. Omicron and
epsilon concerns were not improved.

The result supports the interpretation that increasing focal pressure above the
current candidate range pushes the model toward overconfident dark/blob outputs.
The pipeline worked, but the tested setting is not promising.

The explicit 2048-sample resumed `focal_1_00` probe did not visibly improve the
same fixed review sheets. Its outputs were nearly indistinguishable from the
`focal_1_25` sheets at visual-review scale: both showed saturated black regions,
coarse blocky shapes, and no clear rescue of omicron or epsilon. A pixel-level
comparison of the rendered PNGs showed small but non-zero differences, so the
files are not byte-identical, but the qualitative result is effectively the
same.

The `focal_0_25` probe also preserved the same qualitative failure mode. The
fixed review sheets still showed saturated dark blobs and coarse shapes rather
than smoother letter-like reconstructions. Pixel-level comparison against
`focal_1_00` and `focal_1_25` showed somewhat larger differences than the
`1.00` versus `1.25` comparison, but not enough to change the visual
interpretation. This suggests that late one-epoch focal-weight changes in the
`0.25` to `1.25` range are not the main lever for the current visual problem.

### Open Questions

- Was the full-dataset epoch-2 checkpoint trained with default focal `10.0`, or
  was an explicit focal value used in the original AWS environment?
- Should the next AWS work shift to per-class diagnostics, class-weight changes,
  target construction, or output-size experiments before more focal-weight
  runs?
- Should we compute per-class visual metrics before spending more GPU time on
  focal-weight changes?
