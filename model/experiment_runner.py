import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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
    date = date.astimezone(timezone.utc)
    return MODEL_DIR / "runs" / "experiments" / f"focal-weight-sweep-{date:%Y%m%dT%H%M%SZ}"


def build_probe_plans(config: SweepConfig) -> List[ProbePlan]:
    plans: List[ProbePlan] = []
    seen_names = set()
    for focal_weight in config.focal_weights:
        name = focal_weight_name(focal_weight)
        if name in seen_names:
            raise ValueError(f"Duplicate focal weight plan name generated: {name}")
        seen_names.add(name)
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

        existing_outputs = [
            path
            for path in (plan.run_dir, plan.checkpoint_path, plan.review_dir)
            if path.exists()
        ]
        if existing_outputs:
            entries.append(
                probe_inventory_entry(
                    plan,
                    "output_exists",
                    f"Refusing to overwrite existing output: {existing_outputs[0]}",
                )
            )
            write_inventory(inventory_dir, entries)
            break

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


def run_command(command: Sequence[str], *, cwd: Path, env: Dict[str, str]) -> CommandResult:
    merged_env = os.environ.copy()
    merged_env.update(env)
    completed = subprocess.run(list(command), cwd=cwd, env=merged_env, text=True)
    status = "succeeded" if completed.returncode == 0 else "failed"
    return CommandResult(command=list(command), returncode=completed.returncode, status=status)
