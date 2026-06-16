# Research Journal and Experiment Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a lightweight research journal and experiment ledger workflow, with an initial backfill that explains the current project direction and experiment evidence.

**Architecture:** Add three documentation files under `docs/`: a running Markdown journal as the source of truth, a CSV experiment ledger for sortable run summaries, and a short contributor guide. The initial backfill should summarize the current research state and point to existing notes/artifacts without committing generated model outputs.

**Tech Stack:** Markdown, CSV, Python standard-library CSV parser for validation, existing docs under `docs/Notes/` and `docs/superpowers/specs/`.

---

## File Structure

- Create `docs/research_journal.md`
  - Running chronological source-of-truth journal.
  - First entry: `2026-06-16`, summarizing context, changes, experiments, decisions, results, and open questions.

- Create `docs/experiment_ledger.csv`
  - Structured experiment index with one header row and initial rows for the key AWS, diagnostic, and dry-run experiments.
  - CSV should parse cleanly with Python's `csv.DictReader`.

- Create `docs/research_log_guide.md`
  - Brief instructions for future updates.
  - Includes the journal template, ledger column guidance, and artifact policy.

- Do not modify or stage:
  - `model/runs/`
  - `model/downloads/`
  - `.codex`
  - existing uncommitted note drafts, unless explicitly requested.

---

### Task 1: Create the Research Journal Backfill

**Files:**
- Create: `docs/research_journal.md`

- [ ] **Step 1: Create the journal with the approved source-of-truth structure**

Create `docs/research_journal.md` with this content:

```markdown
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
```

- [ ] **Step 2: Verify the journal contains the required sections**

Run:

```bash
rg -n "^## 2026-06-16|^### Focus|^### Context|^### Changes|^### Experiments|^### Decisions|^### Results|^### Open Questions" docs/research_journal.md
```

Expected: output includes all eight headings.

- [ ] **Step 3: Commit Task 1**

Run:

```bash
git add docs/research_journal.md
git commit -m "docs: add research journal backfill"
```

---

### Task 2: Create the Experiment Ledger

**Files:**
- Create: `docs/experiment_ledger.csv`

- [ ] **Step 1: Create the CSV ledger with initial experiment rows**

Create `docs/experiment_ledger.csv` with this content:

```csv
experiment_id,date,kind,purpose,git_ref,dataset,max_size,epochs,batch_size,size_profile,key_variables,checkpoint_from,output_path,review_path,metrics,result,interpretation,next_step,journal_ref
20260607-resumed-32768,2026-06-07,training,Resume interrupted 32768-sample AWS run from epoch-4 checkpoint,uncial-pilot-prep,ALPUB_v2,32768,1,8,96,VES_SEED=42; VES_RESUME_FROM=runs/20260607T195744Z-6e1be9724/checkpoints/0-4.pth,runs/20260607T195744Z-6e1be9724/checkpoints/0-4.pth,model/runs/20260607T221147Z-ce45afd60,new.pth visual review pending,train_loss=4.0512; val_loss=3.5407,Completed resumed epoch and saved new.pth,Checkpoint resume worked for continuing interrupted training,Use as old subset comparison once copied locally,docs/research_journal.md#2026-06-16
20260609-full-epoch1,2026-06-09,training,Run first full ALPUB_v2 manifest epoch on AWS,uncial-pilot-prep,ALPUB_v2,205797,1,8,96,VES_SEED=42; VES_MAX_SIZE=0; VES_SIZE_PROFILE=96,recovery checkpoint from full-dataset run,model/runs/20260609T034800Z-508dd5292,model/runs/comparisons/20260610_visual_progress,train_loss=2.9289; val_loss=2.6946,Completed full-dataset epoch and saved new.pth,Full-manifest training improved aggregate validation loss,Compare visual outputs against subset and later epochs,docs/research_journal.md#2026-06-16
20260610-full-epoch2,2026-06-10,training,Continue full ALPUB_v2 training from epoch-1 checkpoint,uncial-pilot-prep,ALPUB_v2,205797,1,8,96,VES_SEED=42; VES_MAX_SIZE=0; VES_SIZE_PROFILE=96,model/runs/20260609T034800Z-508dd5292/new.pth,model/runs/20260610T015145Z-07415d46e,model/runs/comparisons/20260610_visual_progress,train_loss=2.5580; val_loss=2.4500,Completed second full-dataset epoch and saved new.pth,Validation loss improved but visual quality did not clearly improve for all classes,Pause paid full-dataset training and inspect weak classes,docs/research_journal.md#2026-06-16
20260610-visual-regression-review,2026-06-10,visual-review,Compare subset and full-dataset checkpoints on fixed visual samples and concern classes,uncial-pilot-prep,ALPUB_v2,n/a,n/a,n/a,96,concern_classes=omicron;epsilon;lunate_sigma;chi,model/runs/20260609T034800Z-508dd5292/new.pth; model/runs/20260610T015145Z-07415d46e/new.pth,model/runs/comparisons/20260610_visual_progress,model/runs/comparisons/20260610_visual_progress,visual review showed mixed quality; omicron and epsilon concern cases,Generated class-specific visual comparisons,Aggregate validation loss is not enough to judge letter quality,Compute per-class diagnostics and revisit target/loss formulation,docs/research_journal.md#2026-06-16
20260613-focal-sweep-wrapper,2026-06-13,dry-run,Add dry-run-capable focal-weight sweep wrapper and inventory planning,b32a14753,ALPUB_v2,256 planned,1 planned,4,96,VES_FOCAL_WEIGHT sweep=1.25|1.5|2.0|2.5|3.0|4.0,n/a,model/runs/experiments/focal-weight-sweep-YYYYMMDD,model/runs/experiments/focal-weight-sweep-YYYYMMDD/reviews,unit tests: 41 tests passed,Implemented sweep planning and inventory without launching training,Controlled sweep setup is ready for review before execution,Run controlled dry run and inspect manifest,docs/research_journal.md#2026-06-16
20260613-controlled-focal-dryrun,2026-06-13,dry-run,Verify focal sweep wrapper with two planned focal weights,b32a14753,ALPUB_v2,128,1,4,96,VES_FOCAL_WEIGHT=1.25;2.5,n/a,/tmp/ves-controlled-focal-dry-run,/tmp/ves-controlled-focal-dry-run/reviews,manifest probes=focal_1_25;focal_2_50; inventory statuses=planned,Dry run wrote experiment.json run_inventory.json and RUN_INDEX.md,Dry-run path works and does not launch training,Review plan before any execute-mode training,docs/research_journal.md#2026-06-16
```

- [ ] **Step 2: Verify the CSV parses and exposes expected IDs**

Run:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path("docs/experiment_ledger.csv")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
expected = {
    "20260607-resumed-32768",
    "20260609-full-epoch1",
    "20260610-full-epoch2",
    "20260610-visual-regression-review",
    "20260613-focal-sweep-wrapper",
    "20260613-controlled-focal-dryrun",
}
actual = {row["experiment_id"] for row in rows}
missing = expected - actual
if missing:
    raise SystemExit(f"missing IDs: {sorted(missing)}")
if len(rows) != 6:
    raise SystemExit(f"expected 6 rows, found {len(rows)}")
print("ledger ok")
PY
```

Expected: `ledger ok`.

- [ ] **Step 3: Commit Task 2**

Run:

```bash
git add docs/experiment_ledger.csv
git commit -m "docs: add experiment ledger"
```

---

### Task 3: Create the Research Log Guide

**Files:**
- Create: `docs/research_log_guide.md`

- [ ] **Step 1: Create contributor instructions**

Create `docs/research_log_guide.md` with this content:

```markdown
# Research Log Guide

Use this guide when adding to the VES research journal or experiment ledger.

## Source of Truth

`docs/research_journal.md` is authoritative. It should explain the story of the
work: why a change was made, what evidence existed, what happened, and what
remains uncertain.

`docs/experiment_ledger.csv` is an index. It summarizes experiments so they can
be sorted, filtered, and reviewed quickly. It should not replace the journal.

## When To Add A Journal Entry

Add or extend a dated journal entry when work changes the research state:

- a training run, dry run, diagnostic, or visual comparison happens;
- a meaningful model, data, loss, or evaluation assumption changes;
- a decision is made about the paper, collaborators, compute, or artifact
  policy;
- an experiment fails or produces ambiguous evidence;
- a new contributor needs context that was previously only in chat history.

## Journal Template

```markdown
## YYYY-MM-DD

### Focus
What we were trying to accomplish.

### Context
Relevant prior state, assumptions, constraints, or collaborator status.

### Changes
Code, documentation, data, experiment setup, or workflow changes.

### Experiments
Short summaries that reference ledger IDs when applicable.

### Decisions
What was decided and why.

### Results
Observed outcomes, including failures, ambiguous findings, or negative results.

### Open Questions
Unresolved technical, research, collaboration, or data questions.
```

## When To Add A Ledger Row

Add one row to `docs/experiment_ledger.csv` for each meaningful experiment,
sweep, diagnostic run, visual comparison, or dry-run planning step. Do not add
rows for routine edits unless the edit changes experiment interpretation or
research process.

The current columns are:

```text
experiment_id,date,kind,purpose,git_ref,dataset,max_size,epochs,batch_size,size_profile,key_variables,checkpoint_from,output_path,review_path,metrics,result,interpretation,next_step,journal_ref
```

Use `n/a` when a field does not apply. Keep long interpretation in the journal;
the ledger should remain compact.

## Experiment ID Convention

Use stable IDs that start with a date:

```text
YYYYMMDD-short-description
```

Examples:

- `20260613-controlled-focal-dryrun`
- `20260610-visual-regression-review`
- `20260609-full-epoch1`

## Artifact Policy

Generated artifacts under `model/runs/` and `model/downloads/` should stay out
of git by default. Record their paths in the journal and ledger. Commit only
small representative artifacts when they are intentionally selected for the
paper, onboarding, or review.

## Rationale Policy

Major changes from Hugo's original work need explicit rationale. This is
especially important for:

- moving from Modern Greek handwriting samples to Ancient Greek uncials;
- changing loss weights or target representations;
- adding new review or diagnostic tools;
- changing training, checkpoint, or resume behavior;
- changing how results are interpreted in the paper.

Use respectful wording. The goal is to explain the evidence trail, not assign
blame.

## Validation

After editing the ledger, verify it parses:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("docs/experiment_ledger.csv").open(newline="", encoding="utf-8")))
print(f"{len(rows)} ledger rows")
PY
```
```

- [ ] **Step 2: Verify the guide references all required files**

Run:

```bash
rg -n "docs/research_journal.md|docs/experiment_ledger.csv|model/runs|model/downloads|Hugo" docs/research_log_guide.md
```

Expected: output includes references to the journal, ledger, generated artifact
paths, and Hugo rationale policy.

- [ ] **Step 3: Commit Task 3**

Run:

```bash
git add docs/research_log_guide.md
git commit -m "docs: add research log guide"
```

---

### Task 4: Run Final Documentation Verification

**Files:**
- Verify: `docs/research_journal.md`
- Verify: `docs/experiment_ledger.csv`
- Verify: `docs/research_log_guide.md`

- [ ] **Step 1: Verify all expected files exist**

Run:

```bash
test -f docs/research_journal.md
test -f docs/experiment_ledger.csv
test -f docs/research_log_guide.md
```

Expected: command exits with status `0`.

- [ ] **Step 2: Verify journal and guide have required headings**

Run:

```bash
rg -n "^# VES Research Journal|^## 2026-06-16|^# Research Log Guide|^## Source of Truth|^## Artifact Policy" docs/research_journal.md docs/research_log_guide.md
```

Expected: output includes all listed headings.

- [ ] **Step 3: Verify ledger row count**

Run:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

rows = list(csv.DictReader(Path("docs/experiment_ledger.csv").open(newline="", encoding="utf-8")))
assert len(rows) == 6, len(rows)
print("ledger rows:", len(rows))
PY
```

Expected: `ledger rows: 6`.

- [ ] **Step 4: Confirm generated artifacts remain untracked and unstaged**

Run:

```bash
git status --short model/runs model/downloads .codex
```

Expected: output still shows those paths only as untracked, not staged.

- [ ] **Step 5: Confirm final branch status**

Run:

```bash
git status --short --branch
```

Expected: the branch may still show pre-existing uncommitted docs and generated
artifacts, but only the three new research log files should be committed by this
plan.

---

## Self-Review Notes

- Spec coverage:
  - `docs/research_journal.md` implements the Markdown source of truth.
  - `docs/experiment_ledger.csv` implements the sortable structured ledger.
  - `docs/research_log_guide.md` implements contributor instructions.
  - The initial backfill covers inactive-team context, fork context, Ancient
    Greek uncial rationale, AWS evidence, visual-regression concern, focal
    sweep wrapper, controlled dry run, and artifact policy.
- Generated outputs remain out of git.
- The plan intentionally avoids editing existing note drafts or paper drafts.
