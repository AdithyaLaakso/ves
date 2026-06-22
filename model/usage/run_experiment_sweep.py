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
