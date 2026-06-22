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
