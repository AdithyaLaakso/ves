# Visual Review Sheets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless PNG contact-sheet generator for visual inspection of saved VES checkpoints.

**Architecture:** Add a focused script in `model/usage/visual_review_sheets.py` with small helper functions for deterministic sampling, output-directory selection, tensor conversion, and sheet rendering. Tests use temporary tensors and a fake model so they do not need CUDA or ALPUB image files.

**Tech Stack:** Python, PyTorch, PIL, current `dataset.SegData`, current `model.build_model`.

---

### Task 1: Core Helper Tests

**Files:**
- Modify: `tests/test_gpu_readiness.py`
- Create: `model/usage/visual_review_sheets.py`

- [ ] Add tests for deterministic index selection and default output directory inference.
- [ ] Run the targeted tests and confirm they fail because the module does not exist.
- [ ] Implement helper functions in `model/usage/visual_review_sheets.py`.
- [ ] Re-run targeted tests and confirm they pass.

### Task 2: Contact Sheet Generation

**Files:**
- Modify: `tests/test_gpu_readiness.py`
- Modify: `model/usage/visual_review_sheets.py`

- [ ] Add a test that creates a contact sheet and metadata JSON from fake tensors and a fake model.
- [ ] Run the targeted test and confirm it fails because rendering is missing.
- [ ] Implement batch-free inference and contact-sheet rendering.
- [ ] Re-run targeted tests and confirm they pass.

### Task 3: CLI and Documentation

**Files:**
- Modify: `model/usage/visual_review_sheets.py`
- Modify: `README.md`

- [ ] Add CLI arguments for checkpoint, output directory, sample count, seed, indices, and columns-per-sheet.
- [ ] Document the command for reviewing the latest EC2 checkpoint.
- [ ] Run `python -m unittest tests.test_gpu_readiness`.
