# Experiment Sweep and Run Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dry-run-capable experiment wrapper for focal-weight sweeps and a run inventory that indexes old and new run outputs before any rename.

**Architecture:** Add a focused `model/experiment_runner.py` library for configuration expansion, inventory records, and command planning. Add a thin CLI at `model/usage/run_experiment_sweep.py` that calls the library and executes or dry-runs the planned commands. Extend `tests/test_gpu_readiness.py` with unit tests that validate directory naming, frozen controls, inventory behavior, and failure states without launching training.

**Tech Stack:** Python standard library, existing `model/setup.zsh`, existing review CLIs under `model/usage/`, `unittest`, subprocess command planning.

---

## File Structure

- Create `model/experiment_runner.py`
  - Owns dataclasses for sweep config, probe plans, command results, and inventory entries.
  - Converts focal weights into stable names such as `focal_1_25`.
  - Builds environment dictionaries for each probe while keeping shared controls frozen.
  - Writes `experiment.json`, `run_inventory.json`, and `RUN_INDEX.md`.
  - Executes commands only when called by the CLI with dry-run disabled.

- Create `model/usage/run_experiment_sweep.py`
  - Parses CLI options for experiment directory, focal weights, fixed controls, review generation, and dry-run mode.
  - Uses `experiment_runner` instead of duplicating planning logic.
  - Defaults to dry-run behavior unless `--execute` is passed.

- Modify `tests/test_gpu_readiness.py`
  - Add `ExperimentRunnerTests`.
  - Test planning and inventory logic without invoking training.

- Update `README.md`
  - Add a short usage section for dry-run planning and executing the first focal-weight sweep.

---

### Task 1: Add Experiment Planning Primitives

**Files:**
- Create: `model/experiment_runner.py`
- Modify: `tests/test_gpu_readiness.py`

- [ ] **Step 1: Write failing tests for focal names and frozen controls**

Append this test class near the other configuration tests in `tests/test_gpu_readiness.py`:

```python
class ExperimentRunnerTests(unittest.TestCase):
    def test_focal_weight_names_are_stable(self):
        import experiment_runner

        self.assertEqual(experiment_runner.focal_weight_name(1.25), "focal_1_25")
        self.assertEqual(experiment_runner.focal_weight_name(2.0), "focal_2_00")
        self.assertEqual(experiment_runner.focal_weight_name(4), "focal_4_00")

    def test_build_probe_plans_freezes_shared_controls(self):
        import experiment_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = experiment_runner.SweepConfig(
                experiment_dir=root / "focal-weight-sweep-20260613",
                focal_weights=(1.25, 2.5),
                seed=42,
                max_size=256,
                num_epochs=1,
                batch_size=4,
                size_profile="96",
                resume_from=Path("runs/baseline/new.pth"),
                mse_weight=1.0,
                class_weight=2.0,
                focal_alpha=0.2,
                focal_gamma=2.0,
                python_bin="python3",
                review_count=24,
                review_seed=99,
                review_indices=None,
                samples_per_sheet=12,
            )

            plans = experiment_runner.build_probe_plans(config)

            self.assertEqual([plan.name for plan in plans], ["focal_1_25", "focal_2_50"])
            self.assertEqual(plans[0].run_dir, root / "focal-weight-sweep-20260613" / "runs" / "focal_1_25")
            self.assertEqual(plans[1].run_dir, root / "focal-weight-sweep-20260613" / "runs" / "focal_2_50")
            for plan in plans:
                self.assertEqual(plan.env["VES_SEED"], "42")
                self.assertEqual(plan.env["VES_MAX_SIZE"], "256")
                self.assertEqual(plan.env["VES_NUM_EPOCHS"], "1")
                self.assertEqual(plan.env["VES_BATCH_SIZE"], "4")
                self.assertEqual(plan.env["VES_SIZE_PROFILE"], "96")
                self.assertEqual(plan.env["VES_RESUME_FROM"], "runs/baseline/new.pth")
                self.assertEqual(plan.env["VES_MSE_WEIGHT"], "1.0")
                self.assertEqual(plan.env["VES_CLASS_WEIGHT"], "2.0")
                self.assertEqual(plan.env["VES_FOCAL_ALPHA"], "0.2")
                self.assertEqual(plan.env["VES_FOCAL_GAMMA"], "2.0")
            self.assertEqual(plans[0].env["VES_FOCAL_WEIGHT"], "1.25")
            self.assertEqual(plans[1].env["VES_FOCAL_WEIGHT"], "2.5")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_gpu_readiness.ExperimentRunnerTests -v
```

Expected: fails with `ModuleNotFoundError: No module named 'experiment_runner'`.

- [ ] **Step 3: Create `model/experiment_runner.py` with planning code**

Create `model/experiment_runner.py`:

```python
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parent


def focal_weight_name(value: float) -> str:
    return f"focal_{float(value):.2f}".replace(".", "_")


def _path_string(path: Optional[Path]) -> Optional[str]:
    return None if path is None else str(path)


@dataclass(frozen=True)
class SweepConfig:
    experiment_dir: Path
    focal_weights: Tuple[float, ...]
    seed: int
    max_size: int
    num_epochs: int
    batch_size: int
    size_profile: str
    resume_from: Optional[Path]
    mse_weight: float
    class_weight: float
    focal_alpha: float
    focal_gamma: float
    python_bin: str
    review_count: int
    review_seed: int
    review_indices: Optional[str]
    samples_per_sheet: int


@dataclass(frozen=True)
class ProbePlan:
    name: str
    focal_weight: float
    run_dir: Path
    checkpoint_path: Path
    review_dir: Path
    env: Dict[str, str]
    train_command: List[str]
    review_command: List[str]


@dataclass
class CommandResult:
    command: List[str]
    returncode: int
    status: str


@dataclass
class InventoryEntry:
    id: str
    category: str
    path: str
    checkpoint: Optional[str]
    review_paths: List[str] = field(default_factory=list)
    metric_paths: List[str] = field(default_factory=list)
    known_variables: Dict[str, object] = field(default_factory=dict)
    status: str = "indexed"
    notes: str = ""
    previous_paths: List[str] = field(default_factory=list)


def default_experiment_dir(date: Optional[datetime] = None) -> Path:
    date = date or datetime.now(timezone.utc)
    return MODEL_DIR / "runs" / "experiments" / f"focal-weight-sweep-{date:%Y%m%d}"


def build_probe_plans(config: SweepConfig) -> List[ProbePlan]:
    plans: List[ProbePlan] = []
    for focal_weight in config.focal_weights:
        name = focal_weight_name(focal_weight)
        run_dir = config.experiment_dir / "runs" / name
        checkpoint_path = run_dir / "new.pth"
        review_dir = config.experiment_dir / "reviews" / name
        env = {
            "VES_RUN_DIR": str(run_dir),
            "VES_SEED": str(config.seed),
            "VES_MAX_SIZE": str(config.max_size),
            "VES_NUM_EPOCHS": str(config.num_epochs),
            "VES_BATCH_SIZE": str(config.batch_size),
            "VES_SIZE_PROFILE": config.size_profile,
            "VES_MSE_WEIGHT": str(config.mse_weight),
            "VES_FOCAL_WEIGHT": str(focal_weight),
            "VES_CLASS_WEIGHT": str(config.class_weight),
            "VES_FOCAL_ALPHA": str(config.focal_alpha),
            "VES_FOCAL_GAMMA": str(config.focal_gamma),
            "PYTHON_BIN": config.python_bin,
        }
        if config.resume_from is not None:
            env["VES_RESUME_FROM"] = str(config.resume_from)

        review_command = [
            config.python_bin,
            "usage/visual_review_sheets.py",
            str(checkpoint_path),
            "--output-dir",
            str(review_dir),
            "--count",
            str(config.review_count),
            "--seed",
            str(config.review_seed),
            "--samples-per-sheet",
            str(config.samples_per_sheet),
        ]
        if config.review_indices:
            review_command.extend(["--indices", config.review_indices])

        plans.append(
            ProbePlan(
                name=name,
                focal_weight=float(focal_weight),
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                review_dir=review_dir,
                env=env,
                train_command=["./setup.zsh"],
                review_command=review_command,
            )
        )
    return plans


def probe_inventory_entry(plan: ProbePlan, status: str, notes: str = "") -> InventoryEntry:
    return InventoryEntry(
        id=plan.name,
        category="experiment-probe",
        path=str(plan.run_dir),
        checkpoint=str(plan.checkpoint_path),
        review_paths=[str(plan.review_dir)],
        known_variables={
            "VES_FOCAL_WEIGHT": plan.focal_weight,
            "VES_MAX_SIZE": int(plan.env["VES_MAX_SIZE"]),
            "VES_NUM_EPOCHS": int(plan.env["VES_NUM_EPOCHS"]),
            "VES_BATCH_SIZE": int(plan.env["VES_BATCH_SIZE"]),
            "VES_SIZE_PROFILE": plan.env["VES_SIZE_PROFILE"],
            "VES_SEED": int(plan.env["VES_SEED"]),
            "VES_MSE_WEIGHT": float(plan.env["VES_MSE_WEIGHT"]),
            "VES_CLASS_WEIGHT": float(plan.env["VES_CLASS_WEIGHT"]),
            "VES_FOCAL_ALPHA": float(plan.env["VES_FOCAL_ALPHA"]),
            "VES_FOCAL_GAMMA": float(plan.env["VES_FOCAL_GAMMA"]),
            "VES_RESUME_FROM": plan.env.get("VES_RESUME_FROM"),
        },
        status=status,
        notes=notes,
    )


def write_experiment_manifest(config: SweepConfig, plans: Sequence[ProbePlan], path: Path) -> None:
    payload = {
        "experiment_dir": str(config.experiment_dir),
        "focal_weights": list(config.focal_weights),
        "fixed_controls": {
            "VES_SEED": config.seed,
            "VES_MAX_SIZE": config.max_size,
            "VES_NUM_EPOCHS": config.num_epochs,
            "VES_BATCH_SIZE": config.batch_size,
            "VES_SIZE_PROFILE": config.size_profile,
            "VES_RESUME_FROM": _path_string(config.resume_from),
            "VES_MSE_WEIGHT": config.mse_weight,
            "VES_CLASS_WEIGHT": config.class_weight,
            "VES_FOCAL_ALPHA": config.focal_alpha,
            "VES_FOCAL_GAMMA": config.focal_gamma,
        },
        "review": {
            "count": config.review_count,
            "seed": config.review_seed,
            "indices": config.review_indices,
            "samples_per_sheet": config.samples_per_sheet,
        },
        "probes": [
            {
                "name": plan.name,
                "run_dir": str(plan.run_dir),
                "checkpoint": str(plan.checkpoint_path),
                "review_dir": str(plan.review_dir),
                "env": plan.env,
                "train_command": plan.train_command,
                "review_command": plan.review_command,
            }
            for plan in plans
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_command(command: Sequence[str], *, cwd: Path, env: Dict[str, str]) -> CommandResult:
    merged_env = os.environ.copy()
    merged_env.update(env)
    completed = subprocess.run(list(command), cwd=cwd, env=merged_env, text=True)
    status = "succeeded" if completed.returncode == 0 else "failed"
    return CommandResult(command=list(command), returncode=completed.returncode, status=status)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m unittest tests.test_gpu_readiness.ExperimentRunnerTests -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add model/experiment_runner.py tests/test_gpu_readiness.py
git commit -m "feat: add experiment sweep planning"
```

---

### Task 2: Add Inventory Discovery and Index Writing

**Files:**
- Modify: `model/experiment_runner.py`
- Modify: `tests/test_gpu_readiness.py`

- [ ] **Step 1: Write failing tests for inventory behavior**

Add these methods to `ExperimentRunnerTests` in `tests/test_gpu_readiness.py`:

```python
    def test_existing_runs_are_indexed_without_fabricated_values(self):
        import experiment_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            run_dir = runs_dir / "20260610-local-focal10-256x2"
            run_dir.mkdir(parents=True)
            (run_dir / "new.pth").write_bytes(b"checkpoint")
            (run_dir / "07415d46e.stamp").write_text("", encoding="utf-8")
            (runs_dir / "logs").mkdir()
            (runs_dir / "comparisons").mkdir()

            entries = experiment_runner.discover_existing_runs(runs_dir)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].id, "20260610-local-focal10-256x2")
            self.assertEqual(entries[0].category, "local-probe")
            self.assertEqual(entries[0].checkpoint, str(run_dir / "new.pth"))
            self.assertEqual(entries[0].known_variables, {})
            self.assertEqual(entries[0].status, "indexed")

    def test_inventory_json_and_markdown_are_written(self):
        import experiment_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir) / "runs"
            runs_dir.mkdir()
            entries = [
                experiment_runner.InventoryEntry(
                    id="focal_2_50",
                    category="experiment-probe",
                    path="model/runs/experiments/example/runs/focal_2_50",
                    checkpoint="model/runs/experiments/example/runs/focal_2_50/new.pth",
                    review_paths=["model/runs/experiments/example/reviews/focal_2_50"],
                    known_variables={"VES_FOCAL_WEIGHT": 2.5},
                    status="review_failed",
                    notes="training succeeded; visual review failed",
                )
            ]

            experiment_runner.write_inventory(runs_dir, entries)

            inventory = json.loads((runs_dir / "run_inventory.json").read_text(encoding="utf-8"))
            markdown = (runs_dir / "RUN_INDEX.md").read_text(encoding="utf-8")
            self.assertEqual(inventory["runs"][0]["id"], "focal_2_50")
            self.assertEqual(inventory["runs"][0]["status"], "review_failed")
            self.assertIn("| focal_2_50 | experiment-probe | review_failed |", markdown)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_gpu_readiness.ExperimentRunnerTests -v
```

Expected: fails because `discover_existing_runs` and `write_inventory` are not defined.

- [ ] **Step 3: Implement discovery and index writing**

Add this code to `model/experiment_runner.py`:

```python
def _infer_category(path: Path) -> str:
    name = path.name
    if name.startswith("20") and "T" in name:
        return "full-training"
    if "focal" in name or "loss" in name:
        return "local-probe"
    if name.startswith("visual_review"):
        return "visual-review"
    return "run"


def discover_existing_runs(runs_dir: Path) -> List[InventoryEntry]:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []

    ignored = {"comparisons", "logs", "experiments", "__pycache__"}
    entries: List[InventoryEntry] = []
    for child in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if child.name in ignored:
            continue
        checkpoint = child / "new.pth"
        visual_review = child / "visual_review"
        review_paths = [str(visual_review)] if visual_review.exists() else []
        entries.append(
            InventoryEntry(
                id=child.name,
                category=_infer_category(child),
                path=str(child),
                checkpoint=str(checkpoint) if checkpoint.exists() else None,
                review_paths=review_paths,
                metric_paths=[],
                known_variables={},
                status="indexed",
                notes="Indexed from existing folder; variables not inferred from folder name.",
            )
        )
    return entries


def _entry_to_dict(entry: InventoryEntry) -> Dict[str, object]:
    return asdict(entry)


def write_inventory(runs_dir: Path, entries: Sequence[InventoryEntry]) -> None:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": [_entry_to_dict(entry) for entry in entries],
    }
    (runs_dir / "run_inventory.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Run Index",
        "",
        "This file indexes run outputs without changing checkpoint, log, metric, or review contents.",
        "",
        "| ID | Category | Status | Checkpoint | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        checkpoint = entry.checkpoint or ""
        notes = entry.notes.replace("|", "\\|")
        lines.append(
            f"| {entry.id} | {entry.category} | {entry.status} | {checkpoint} | {notes} |"
        )
    (runs_dir / "RUN_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m unittest tests.test_gpu_readiness.ExperimentRunnerTests -v
```

Expected: all `ExperimentRunnerTests` pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add model/experiment_runner.py tests/test_gpu_readiness.py
git commit -m "feat: index experiment runs"
```

---

### Task 3: Add CLI Wrapper With Dry-Run and Execution Modes

**Files:**
- Create: `model/usage/run_experiment_sweep.py`
- Modify: `model/experiment_runner.py`
- Modify: `tests/test_gpu_readiness.py`

- [ ] **Step 1: Write failing CLI and status tests**

Add these methods to `ExperimentRunnerTests` in `tests/test_gpu_readiness.py`:

```python
    def test_probe_inventory_entry_distinguishes_failure_states(self):
        import experiment_runner

        config = experiment_runner.SweepConfig(
            experiment_dir=Path("model/runs/experiments/example"),
            focal_weights=(2.5,),
            seed=42,
            max_size=256,
            num_epochs=1,
            batch_size=4,
            size_profile="96",
            resume_from=None,
            mse_weight=1.0,
            class_weight=2.0,
            focal_alpha=0.2,
            focal_gamma=2.0,
            python_bin="python3",
            review_count=24,
            review_seed=42,
            review_indices=None,
            samples_per_sheet=12,
        )
        plan = experiment_runner.build_probe_plans(config)[0]

        training_failed = experiment_runner.probe_inventory_entry(plan, "training_failed", "setup.zsh failed")
        review_failed = experiment_runner.probe_inventory_entry(plan, "review_failed", "visual review failed")

        self.assertEqual(training_failed.status, "training_failed")
        self.assertEqual(review_failed.status, "review_failed")
        self.assertEqual(training_failed.known_variables["VES_FOCAL_WEIGHT"], 2.5)

    def test_experiment_sweep_cli_help_runs_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "model/usage/run_experiment_sweep.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("Run or dry-run focal-weight experiment sweeps", result.stdout)

    def test_experiment_sweep_cli_dry_run_writes_manifest_and_inventory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "runs" / "experiments" / "focal-weight-sweep"
            result = subprocess.run(
                [
                    sys.executable,
                    "model/usage/run_experiment_sweep.py",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--focal-weights",
                    "1.25,2.5",
                    "--max-size",
                    "128",
                    "--num-epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--no-review",
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("DRY RUN", result.stdout)
            self.assertTrue((experiment_dir / "experiment.json").exists())
            self.assertTrue((Path(tmpdir) / "runs" / "run_inventory.json").exists())
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_gpu_readiness.ExperimentRunnerTests -v
```

Expected: fails because `model/usage/run_experiment_sweep.py` does not exist.

- [ ] **Step 3: Add sweep orchestration helper**

Add these functions to `model/experiment_runner.py`:

```python
def inventory_dir_for_experiment(experiment_dir: Path) -> Path:
    experiment_dir = Path(experiment_dir)
    if experiment_dir.parent.name == "experiments":
        return experiment_dir.parent.parent
    return MODEL_DIR / "runs"


def run_sweep(
    config: SweepConfig,
    *,
    execute: bool,
    run_review: bool,
    index_existing: bool,
) -> List[InventoryEntry]:
    plans = build_probe_plans(config)
    inventory_dir = inventory_dir_for_experiment(config.experiment_dir)
    config.experiment_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_manifest(config, plans, config.experiment_dir / "experiment.json")

    entries: List[InventoryEntry] = []
    if index_existing:
        entries.extend(discover_existing_runs(inventory_dir))

    for plan in plans:
        if not execute:
            entries.append(probe_inventory_entry(plan, "planned", "Dry run only; training not launched."))
            continue

        plan.run_dir.mkdir(parents=True, exist_ok=True)
        train_result = run_command(plan.train_command, cwd=MODEL_DIR, env=plan.env)
        if train_result.returncode != 0:
            entries.append(probe_inventory_entry(plan, "training_failed", "Training command failed."))
            write_inventory(inventory_dir, entries)
            break

        if run_review:
            review_result = run_command(plan.review_command, cwd=MODEL_DIR, env=plan.env)
            if review_result.returncode != 0:
                entries.append(probe_inventory_entry(plan, "review_failed", "Training succeeded; review command failed."))
                write_inventory(inventory_dir, entries)
                continue

        entries.append(probe_inventory_entry(plan, "completed", "Training completed."))

    write_inventory(inventory_dir, entries)
    return entries
```

- [ ] **Step 4: Create `model/usage/run_experiment_sweep.py`**

Create `model/usage/run_experiment_sweep.py`:

```python
import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import experiment_runner


def _parse_focal_weights(raw: str) -> Tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one focal weight is required")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or dry-run focal-weight experiment sweeps."
    )
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--focal-weights", type=_parse_focal_weights, default=_parse_focal_weights("1.25,1.5,2.0,2.5,3.0,4.0"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size-profile", type=str, default="96")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--class-weight", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.2)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--review-count", type=int, default=24)
    parser.add_argument("--review-seed", type=int, default=42)
    parser.add_argument("--review-indices", type=str, default=None)
    parser.add_argument("--samples-per-sheet", type=int, default=12)
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing-index", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    execute = bool(args.execute and not args.dry_run)
    experiment_dir = args.experiment_dir or experiment_runner.default_experiment_dir()
    config = experiment_runner.SweepConfig(
        experiment_dir=experiment_dir,
        focal_weights=args.focal_weights,
        seed=args.seed,
        max_size=args.max_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        size_profile=args.size_profile,
        resume_from=args.resume_from,
        mse_weight=args.mse_weight,
        class_weight=args.class_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        python_bin=args.python_bin,
        review_count=args.review_count,
        review_seed=args.review_seed,
        review_indices=args.review_indices,
        samples_per_sheet=args.samples_per_sheet,
    )

    entries = experiment_runner.run_sweep(
        config,
        execute=execute,
        run_review=not args.no_review,
        index_existing=not args.skip_existing_index,
    )
    mode = "EXECUTE" if execute else "DRY RUN"
    print(f"{mode}: wrote {experiment_dir / 'experiment.json'}")
    print(f"{mode}: indexed {len(entries)} run entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
python -m unittest tests.test_gpu_readiness.ExperimentRunnerTests -v
```

Expected: all `ExperimentRunnerTests` pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add model/experiment_runner.py model/usage/run_experiment_sweep.py tests/test_gpu_readiness.py
git commit -m "feat: add focal sweep wrapper"
```

---

### Task 4: Document Usage and Run Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with dry-run and execute examples**

Add this exact section after the existing loss/run controls in `README.md`:

````markdown
To plan a short focal-weight sweep without launching training:

```bash
python model/usage/run_experiment_sweep.py \
  --dry-run \
  --focal-weights 1.25,1.5,2.0,2.5,3.0,4.0 \
  --max-size 256 \
  --num-epochs 1 \
  --batch-size 4 \
  --size-profile 96 \
  --no-review
```

This writes `experiment.json`, `model/runs/run_inventory.json`, and `model/runs/RUN_INDEX.md`. Dry-run mode does not launch training.

To execute the same sweep after reviewing the plan:

```bash
python model/usage/run_experiment_sweep.py \
  --execute \
  --focal-weights 1.25,1.5,2.0,2.5,3.0,4.0 \
  --max-size 256 \
  --num-epochs 1 \
  --batch-size 4 \
  --size-profile 96
```

Each probe writes under `model/runs/experiments/focal-weight-sweep-YYYYMMDD/runs/`. Review the generated visual sheets and inventory before launching a second sweep or renaming old folders.
````

- [ ] **Step 2: Run focused tests**

Run:

```bash
python -m unittest tests.test_gpu_readiness.ExperimentRunnerTests -v
```

Expected: all `ExperimentRunnerTests` pass.

- [ ] **Step 3: Run broader existing test file**

Run:

```bash
python -m unittest tests.test_gpu_readiness -v
```

Expected: all tests pass. If runtime is high on CPU, capture the first failing or slow test and report it before changing implementation.

- [ ] **Step 4: Run a real dry-run command**

Run:

```bash
python model/usage/run_experiment_sweep.py \
  --dry-run \
  --experiment-dir /tmp/ves-focal-sweep-dry-run \
  --focal-weights 1.25,2.5 \
  --max-size 128 \
  --num-epochs 1 \
  --batch-size 4 \
  --no-review
```

Expected output includes:

```text
DRY RUN: wrote /tmp/ves-focal-sweep-dry-run/experiment.json
DRY RUN: indexed
```

Inspect:

```bash
python -m json.tool /tmp/ves-focal-sweep-dry-run/experiment.json
```

Expected: two probes named `focal_1_25` and `focal_2_50`, with identical controls except `VES_FOCAL_WEIGHT` and `VES_RUN_DIR`.

- [ ] **Step 5: Commit documentation and verification-ready state**

```bash
git add README.md
git commit -m "docs: document focal sweep runner"
```

---

## Self-Review Notes

- Spec coverage:
  - Constant focal-weight sweep is implemented by `SweepConfig`, `build_probe_plans`, and the CLI defaults.
  - Fixed controls are represented directly in `SweepConfig` and tested in `test_build_probe_plans_freezes_shared_controls`.
  - Review artifact planning is represented by `review_command`, `review_dir`, and review status handling.
  - Index-first cleanup is implemented by `discover_existing_runs` and `write_inventory`.
  - Rename behavior remains out of implementation scope except for preserving `previous_paths` in the inventory entry model.
  - Failure distinction is implemented by `training_failed` and `review_failed` statuses.

- No real training is required for tests.
- The first real command users should run is dry-run mode.
