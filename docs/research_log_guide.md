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
