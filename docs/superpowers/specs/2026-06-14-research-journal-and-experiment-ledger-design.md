# Research Journal and Experiment Ledger Design

## Purpose

VES needs a durable record of research activity, experiment intent, results,
and rationale for changes from the original project. The record should help
future contributors understand why the project moved from the original Modern
Greek handwriting assumption toward Ancient Greek uncial proxy training, what
experiments were run, and which results justified later choices.

The system should be easy to maintain during normal work. The Markdown journal
is the source of truth. A structured ledger mirrors experiment summaries for
sorting, filtering, and onboarding.

## Files

- `docs/research_journal.md`
  - Running chronological journal.
  - Append entries newest-last so the project story can be read forward.
  - Source of truth for narrative context, decisions, interpretation, and open
    questions.

- `docs/experiment_ledger.csv`
  - Structured experiment summary table.
  - One row per meaningful experiment, sweep, diagnostic run, or comparison.
  - Summaries should point to journal entries and artifact paths instead of
    replacing the journal.

- `docs/research_log_guide.md`
  - Contributor-facing instructions for how to update the journal and ledger.
  - Explains what counts as a meaningful experiment and how much detail to
    include.

## Journal Entry Template

Each journal entry should be dated and compact. It should explain what happened
and why, not just list commands.

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

## Ledger Columns

The initial ledger should use CSV so it can be edited by hand and opened in
ordinary spreadsheet tools.

```text
experiment_id,date,kind,purpose,git_ref,dataset,max_size,epochs,batch_size,size_profile,key_variables,checkpoint_from,output_path,review_path,metrics,result,interpretation,next_step,journal_ref
```

Column guidance:

- `experiment_id`: stable short identifier, such as `20260614-focal-dryrun`.
- `kind`: training, dry-run, visual-review, diagnostic, comparison, cleanup, or
  documentation.
- `purpose`: why the experiment was run.
- `git_ref`: commit or branch state used for the experiment.
- `key_variables`: compact semicolon-separated variables that matter for the
  comparison, such as `VES_FOCAL_WEIGHT=1.25; VES_CLASS_WEIGHT=2.0`.
- `metrics`: loss values or diagnostic summaries when available.
- `result`: direct observed outcome.
- `interpretation`: what the result means, with uncertainty if needed.
- `journal_ref`: date or anchor in `docs/research_journal.md`.

## Policy

- The Markdown journal is authoritative.
- The CSV ledger is an index extracted from journal entries.
- Generated artifacts under `model/runs/` and `model/downloads/` stay out of
  git unless a small representative artifact is intentionally selected.
- Each meaningful experiment should answer:
  - Why did we run this?
  - What changed?
  - What happened?
  - What did we learn?
  - What should happen next?
- Each major change from Hugo's original work should include a rationale,
  especially changes to data assumptions, loss weights, target representation,
  model behavior, evaluation methods, or collaboration process.
- Negative and ambiguous results should be logged. They are part of the
  research record.

## Initial Backfill Scope

The first implementation should create the files and backfill a concise entry
covering the current state rather than reconstructing every old detail. It
should include:

- the inactive-team context;
- the fork and Codex-assisted rewrite context;
- the move from Modern Greek handwriting samples to Ancient Greek uncials;
- AWS GPU validation and full-dataset training summaries;
- the visual-regression concern;
- the new focal-weight sweep wrapper and dry-run result;
- the decision to pause generated artifacts from git while reviewing docs and
  paper updates.

Older details can be backfilled gradually from existing notes, commits, and
artifacts as they become relevant.

## Success Criteria

- A new contributor can read the journal and understand the current research
  direction without the original chat history.
- A future collaborator can inspect the ledger and identify which experiments
  support or weaken a decision.
- The rationale for diverging from the original project is explicit,
  evidence-based, and respectful.
- The process remains lightweight enough to update during normal development
  and experiment work.
