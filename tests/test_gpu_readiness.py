import json
import sys
import subprocess
import tempfile
import unittest
import random
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


class DataTransferTests(unittest.TestCase):
    def test_collate_fn_keeps_tensors_on_cpu_by_default(self):
        import dataset

        batch = [
            (torch.zeros(1, 4, 4), (torch.ones(1, 2, 2), 3)),
            (torch.ones(1, 4, 4), (torch.zeros(1, 2, 2), 5)),
        ]

        inputs, (masks, class_labels) = dataset.collate_fn(batch)

        self.assertEqual(inputs.device.type, "cpu")
        self.assertEqual(masks.device.type, "cpu")
        self.assertEqual(class_labels.device.type, "cpu")

    def test_move_batch_to_device_moves_nested_targets(self):
        import train_reconstruction

        inputs = torch.zeros(2, 1, 4, 4)
        targets = (torch.ones(2, 1, 2, 2), torch.tensor([3, 5]))

        moved_inputs, (moved_masks, moved_labels) = train_reconstruction.move_batch_to_device(
            inputs,
            targets,
            torch.device("cpu"),
        )

        self.assertEqual(moved_inputs.device.type, "cpu")
        self.assertEqual(moved_masks.device.type, "cpu")
        self.assertEqual(moved_labels.device.type, "cpu")


class ManifestPreflightTests(unittest.TestCase):
    def test_manifest_preflight_reports_missing_paths(self):
        import dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            existing = root / "ALPUB_v2" / "images" / "alpha.png"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"image")
            manifest = root / "manifest.json"
            manifest.write_text(
                """
                {
                  "records": [
                    {"path": "ALPUB_v2/images/alpha.png", "label": "alpha"},
                    {"path": "ALPUB_v2/images/beta.png", "label": "beta"}
                  ]
                }
                """,
                encoding="utf-8",
            )

            report = dataset.preflight_manifest(manifest, root=root, fail_on_missing=False)

            self.assertEqual(report["total_records"], 2)
            self.assertEqual(report["loadable_records"], 1)
            self.assertEqual(report["missing_records"], 1)
            self.assertEqual(report["missing_examples"], ["ALPUB_v2/images/beta.png"])

    def test_manifest_preflight_fails_when_loadable_below_minimum(self):
        import dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.json"
            manifest.write_text('{"records": []}', encoding="utf-8")

            with self.assertRaises(dataset.DatasetPreflightError):
                dataset.preflight_manifest(manifest, root=root, min_loadable=1)

    def test_dataset_limit_uses_seeded_selection(self):
        import dataset
        import settings

        original_max_size = dataset.MAX_SIZE
        original_seed = settings.seed
        try:
            dataset.MAX_SIZE = 3
            settings.seed = 123
            seg_data = object.__new__(dataset.SegData)
            records = [{"id": idx} for idx in range(10)]

            first = seg_data._limit_dataset(records)
            second = seg_data._limit_dataset(records)

            self.assertEqual(first, second)
        finally:
            dataset.MAX_SIZE = original_max_size
            settings.seed = original_seed


class RunConfigurationTests(unittest.TestCase):
    def test_settings_uses_run_dir_for_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code = """
import settings
print(settings.log_dir)
print(settings.save_to_dir)
print(settings.save_to)
"""
            env = {
                "PYTHONPATH": str(MODEL_DIR),
                "VES_RUN_DIR": tmpdir,
                "VES_SMOKE_TEST": "1",
                "VES_FORCE_CPU": "1",
            }
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env={**env},
                text=True,
                capture_output=True,
                check=True,
            )

            lines = result.stdout.strip().splitlines()
            self.assertIn(str(Path(tmpdir) / "logs"), lines)
            self.assertIn(str(Path(tmpdir) / "checkpoints"), lines)
            self.assertIn(str(Path(tmpdir) / "new.pth"), lines)

    def test_settings_allows_tf32_by_default_and_can_disable_it(self):
        code = """
import torch
import settings
print(settings.allow_tf32)
print(torch.backends.cudnn.allow_tf32)
"""
        base_env = {
            "PYTHONPATH": str(MODEL_DIR),
            "VES_SMOKE_TEST": "1",
            "VES_FORCE_CPU": "1",
        }
        default_result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={**base_env},
            text=True,
            capture_output=True,
            check=True,
        )
        disabled_result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={**base_env, "VES_ALLOW_TF32": "0"},
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("True\nTrue", default_result.stdout)
        self.assertIn("False\nFalse", disabled_result.stdout)

    def test_settings_reads_resume_checkpoint_from_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "0-4.pth"
            code = """
import settings
print(settings.load_from)
"""
            env = {
                "PYTHONPATH": str(MODEL_DIR),
                "VES_SMOKE_TEST": "1",
                "VES_FORCE_CPU": "1",
                "VES_RESUME_FROM": str(checkpoint),
            }
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env={**env},
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn(str(checkpoint), result.stdout.strip().splitlines())

    def test_settings_reads_training_recovery_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "recovery-latest.pt"
            code = """
import settings
print(settings.resume_training_state)
print(settings.step_checkpoint_every_batches)
print(settings.step_checkpoint_every_minutes)
print(settings.keep_step_checkpoints)
"""
            env = {
                "PYTHONPATH": str(MODEL_DIR),
                "VES_SMOKE_TEST": "1",
                "VES_FORCE_CPU": "1",
                "VES_RESUME_TRAINING_STATE": str(checkpoint),
                "VES_STEP_CHECKPOINT_EVERY_BATCHES": "7",
                "VES_STEP_CHECKPOINT_EVERY_MINUTES": "11",
                "VES_KEEP_STEP_CHECKPOINTS": "2",
            }
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env={**env},
                text=True,
                capture_output=True,
                check=True,
            )

            lines = result.stdout.strip().splitlines()
            self.assertIn(str(checkpoint), lines)
            self.assertIn("7", lines)
            self.assertIn("11", lines)
            self.assertIn("2", lines)

    def test_settings_reads_loss_weights_from_environment(self):
        code = """
import settings
print(settings.loss_settings.mse_weight)
print(settings.loss_settings.focal_weight)
print(settings.loss_settings.focal_alpha)
print(settings.loss_settings.class_weight)
"""
        env = {
            "PYTHONPATH": str(MODEL_DIR),
            "VES_SMOKE_TEST": "1",
            "VES_FORCE_CPU": "1",
            "VES_MSE_WEIGHT": "1.5",
            "VES_FOCAL_WEIGHT": "0",
            "VES_FOCAL_ALPHA": "0.75",
            "VES_CLASS_WEIGHT": "0.25",
        }
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={**env},
            text=True,
            capture_output=True,
            check=True,
        )

        lines = result.stdout.strip().splitlines()
        self.assertIn("1.5", lines)
        self.assertIn("0.0", lines)
        self.assertIn("0.75", lines)
        self.assertIn("0.25", lines)

    def test_setup_script_makes_cuda_debug_opt_in(self):
        setup_text = (MODEL_DIR / "setup.zsh").read_text(encoding="utf-8")

        self.assertIn("VES_DEBUG_CUDA", setup_text)
        self.assertNotIn("export CUDA_LAUNCH_BLOCKING=1", setup_text)


class ExperimentRunnerTests(unittest.TestCase):
    def _sweep_config(self, experiment_dir: Path, focal_weights=(1.25,)):
        import experiment_runner

        return experiment_runner.SweepConfig(
            experiment_dir=experiment_dir,
            focal_weights=focal_weights,
            seed=42,
            max_size=128,
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
            self.assertEqual(
                plans[0].run_dir,
                root / "focal-weight-sweep-20260613" / "runs" / "focal_1_25",
            )
            self.assertEqual(
                plans[1].run_dir,
                root / "focal-weight-sweep-20260613" / "runs" / "focal_2_50",
            )
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

    def test_build_probe_plans_rejects_duplicate_generated_names(self):
        import experiment_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            config = experiment_runner.SweepConfig(
                experiment_dir=Path(tmpdir) / "focal-weight-sweep-20260613",
                focal_weights=(1.235, 1.236),
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
                review_seed=99,
                review_indices=None,
                samples_per_sheet=12,
            )

            with self.assertRaisesRegex(ValueError, "focal_1_24"):
                experiment_runner.build_probe_plans(config)

    def test_default_experiment_dir_includes_utc_seconds(self):
        import experiment_runner

        path = experiment_runner.default_experiment_dir(
            datetime(2026, 6, 13, 14, 5, 9, tzinfo=timezone.utc)
        )

        self.assertEqual(
            path,
            MODEL_DIR / "runs" / "experiments" / "focal-weight-sweep-20260613T140509Z",
        )

    def test_run_sweep_dry_run_does_not_launch_training(self):
        import experiment_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._sweep_config(Path(tmpdir) / "runs" / "experiments" / "sweep")

            with patch("experiment_runner.run_command", side_effect=AssertionError("run_command called")) as run_command:
                entries = experiment_runner.run_sweep(
                    config,
                    execute=False,
                    run_review=False,
                    index_existing=False,
                )

            run_command.assert_not_called()
            self.assertEqual([entry.status for entry in entries], ["planned"])

    def test_custom_experiment_dir_keeps_inventory_local(self):
        import experiment_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "sweep"
            config = self._sweep_config(experiment_dir)

            self.assertEqual(
                experiment_runner.inventory_dir_for_experiment(experiment_dir),
                experiment_dir,
            )
            experiment_runner.run_sweep(
                config,
                execute=False,
                run_review=False,
                index_existing=False,
            )

            self.assertTrue((experiment_dir / "experiment.json").exists())
            self.assertTrue((experiment_dir / "run_inventory.json").exists())
            self.assertTrue((experiment_dir / "RUN_INDEX.md").exists())

    def test_run_sweep_execute_stops_when_planned_output_exists(self):
        import experiment_runner

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "runs" / "experiments" / "sweep"
            config = self._sweep_config(experiment_dir)
            existing_run_dir = experiment_dir / "runs" / "focal_1_25"
            existing_run_dir.mkdir(parents=True)

            with patch("experiment_runner.run_command", side_effect=AssertionError("run_command called")) as run_command:
                entries = experiment_runner.run_sweep(
                    config,
                    execute=True,
                    run_review=False,
                    index_existing=False,
                )

            run_command.assert_not_called()
            self.assertEqual([entry.status for entry in entries], ["output_exists"])
            self.assertTrue((Path(tmpdir) / "runs" / "run_inventory.json").exists())

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
        self.assertEqual(review_failed.known_variables["VES_FOCAL_WEIGHT"], 2.5)

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

    def test_experiment_sweep_cli_rejects_execute_with_dry_run(self):
        result = subprocess.run(
            [
                sys.executable,
                "model/usage/run_experiment_sweep.py",
                "--execute",
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not allowed with argument", result.stderr)

    def test_experiment_sweep_cli_returns_nonzero_for_current_sweep_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "runs" / "experiments" / "focal-weight-sweep"
            existing_run_dir = experiment_dir / "runs" / "focal_1_25"
            existing_run_dir.mkdir(parents=True)

            result = subprocess.run(
                [
                    sys.executable,
                    "model/usage/run_experiment_sweep.py",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--focal-weights",
                    "1.25",
                    "--no-review",
                    "--execute",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output_exists", (Path(tmpdir) / "runs" / "run_inventory.json").read_text(encoding="utf-8"))


class VisualReviewTests(unittest.TestCase):
    def test_visual_review_selects_deterministic_indices(self):
        from usage import visual_review_sheets

        first = visual_review_sheets.select_indices(dataset_size=20, count=6, seed=42)
        second = visual_review_sheets.select_indices(dataset_size=20, count=6, seed=42)
        different = visual_review_sheets.select_indices(dataset_size=20, count=6, seed=43)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len(first), 6)
        self.assertEqual(len(set(first)), 6)

    def test_visual_review_defaults_to_run_visual_review_directory(self):
        from usage import visual_review_sheets

        run_checkpoint = Path("runs/example/new.pth")
        epoch_checkpoint = Path("runs/example/checkpoints/0-4.pth")

        self.assertEqual(
            visual_review_sheets.default_output_dir(run_checkpoint),
            Path("runs/example/visual_review"),
        )
        self.assertEqual(
            visual_review_sheets.default_output_dir(epoch_checkpoint),
            Path("runs/example/visual_review"),
        )

    def test_visual_review_writes_sheet_and_metadata(self):
        from usage import visual_review_sheets

        class FakeDataset:
            dataset = [
                {"label": "alpha", "document_id": "doc-a"},
                {"label": "beta", "document_id": "doc-b"},
            ]

            def __len__(self):
                return 2

            def __getitem__(self, idx):
                base = torch.full((1, 8, 8), idx / 2.0)
                target = torch.ones(1, 4, 4) * (1.0 - idx / 3.0)
                return base, (target, idx)

        class FakeModel(torch.nn.Module):
            def forward(self, inputs):
                outputs = torch.full((inputs.shape[0], 1, 4, 4), -2.0)
                return outputs, torch.zeros(inputs.shape[0], 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "visual_review"

            result = visual_review_sheets.create_visual_review(
                dataset=FakeDataset(),
                model=FakeModel(),
                checkpoint_path=Path("runs/example/new.pth"),
                output_dir=output_dir,
                count=2,
                seed=7,
                indices=[0, 1],
                samples_per_sheet=2,
                device=torch.device("cpu"),
            )

            self.assertEqual(len(result.sheet_paths), 1)
            self.assertTrue(result.sheet_paths[0].exists())
            self.assertTrue(result.metadata_path.exists())

            metadata = result.metadata_path.read_text(encoding="utf-8")
            self.assertIn('"checkpoint": "runs/example/new.pth"', metadata)
            self.assertIn('"indices": [', metadata)
            self.assertIn('"label": "alpha"', metadata)

    def test_visual_review_applies_sigmoid_to_model_logits(self):
        from usage import visual_review_sheets

        logits = torch.tensor([[[[-2.0, 0.0], [2.0, 4.0]]]])

        image = visual_review_sheets.model_output_to_image(logits)

        pixels = list(image.getdata())
        self.assertLess(pixels[0], pixels[1])
        self.assertLess(pixels[1], pixels[2])
        self.assertLess(pixels[2], pixels[3])
        self.assertGreater(pixels[1], 120)
        self.assertLess(pixels[1], 135)

    def test_visual_review_cli_help_runs_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "model/usage/visual_review_sheets.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("Generate headless visual review contact sheets", result.stdout)

    def test_target_inspection_cli_help_runs_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "model/usage/target_inspection_sheets.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("Generate target-only inspection sheets", result.stdout)

    def test_per_class_metrics_cli_help_runs_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "model/usage/evaluate_per_class_metrics.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("Evaluate checkpoint metrics on deterministic per-class", result.stdout)

    def test_prediction_diagnostics_cli_help_runs_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "model/usage/prediction_diagnostic_sheets.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("Generate fixed-sample diagnostic sheets", result.stdout)


class TrainingRecoveryTests(unittest.TestCase):
    def test_training_order_resume_slices_directly_from_saved_batch(self):
        import train_reconstruction

        order = list(range(20))

        resumed = train_reconstruction.slice_train_order_for_resume(
            order,
            batch_size=4,
            next_batch=3,
        )

        self.assertEqual(resumed, list(range(12, 20)))

    def test_training_order_uses_deterministic_weighted_sampling(self):
        import settings
        import train_reconstruction

        class FakeData:
            def sample_weights(self, scheme):
                self.scheme = scheme
                return [1.0, 2.0, 3.0, 4.0]

        original_strategy = settings.sampler_strategy
        original_seed = settings.seed
        try:
            settings.sampler_strategy = "document_inv_sqrt"
            settings.seed = 99
            first = train_reconstruction.build_train_order(FakeData(), [0, 1, 2, 3], epoch=0)
            second = train_reconstruction.build_train_order(FakeData(), [0, 1, 2, 3], epoch=0)
            next_epoch = train_reconstruction.build_train_order(FakeData(), [0, 1, 2, 3], epoch=1)
        finally:
            settings.sampler_strategy = original_strategy
            settings.seed = original_seed

        self.assertEqual(first, second)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(idx in {0, 1, 2, 3} for idx in first))

    def test_recovery_checkpoint_round_trips_training_state(self):
        import training_recovery

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            scaler = torch.amp.GradScaler("cpu")

            recovery_path = training_recovery.save_recovery_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=3,
                next_batch=41,
                global_step=1234,
                metadata={"run_dir": "runs/example"},
            )

            loaded = training_recovery.load_recovery_checkpoint(
                recovery_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                map_location=torch.device("cpu"),
            )

            self.assertEqual(recovery_path, checkpoint_dir / "recovery-latest.pt")
            self.assertEqual(loaded["epoch"], 3)
            self.assertEqual(loaded["next_batch"], 41)
            self.assertEqual(loaded["global_step"], 1234)
            self.assertEqual(loaded["metadata"]["run_dir"], "runs/example")
            self.assertEqual(loaded["version"], 1)

    def test_recovery_checkpoint_allows_lazy_model_keys(self):
        import training_recovery

        class LazyModel(torch.nn.Module):
            def __init__(self, include_lazy=False):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                if include_lazy:
                    self.lazy_bias = torch.nn.Parameter(torch.zeros(1))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            saved_model = LazyModel(include_lazy=True)
            saved_optimizer = torch.optim.AdamW(saved_model.parameters(), lr=0.01)
            saved_scheduler = torch.optim.lr_scheduler.ExponentialLR(saved_optimizer, gamma=0.9)
            saved_scaler = torch.amp.GradScaler("cpu")

            recovery_path = training_recovery.save_recovery_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=saved_model,
                optimizer=saved_optimizer,
                scheduler=saved_scheduler,
                scaler=saved_scaler,
                epoch=0,
                next_batch=1,
                global_step=1,
            )

            loaded_model = LazyModel(include_lazy=False)
            loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=0.01)
            loaded_scheduler = torch.optim.lr_scheduler.ExponentialLR(loaded_optimizer, gamma=0.9)
            loaded_scaler = torch.amp.GradScaler("cpu")

            loaded = training_recovery.load_recovery_checkpoint(
                recovery_path,
                model=loaded_model,
                optimizer=loaded_optimizer,
                scheduler=loaded_scheduler,
                scaler=loaded_scaler,
                map_location=torch.device("cpu"),
            )

            self.assertEqual(loaded["next_batch"], 1)

    def test_recovery_checkpoint_restores_rng_state(self):
        import training_recovery

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            scaler = torch.amp.GradScaler("cpu")

            random.seed(12)
            np.random.seed(12)
            torch.manual_seed(12)

            training_recovery.save_recovery_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=0,
                next_batch=1,
                global_step=1,
            )

            expected_python = random.random()
            expected_numpy = np.random.random()
            expected_torch = torch.rand(1).item()

            random.seed(99)
            np.random.seed(99)
            torch.manual_seed(99)

            training_recovery.load_recovery_checkpoint(
                checkpoint_dir / "recovery-latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                map_location=torch.device("cpu"),
            )

            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.random(), expected_numpy)
            self.assertEqual(torch.rand(1).item(), expected_torch)

    def test_recovery_rng_restore_accepts_tensor_state(self):
        import training_recovery

        random.seed(34)
        np.random.seed(34)
        torch.manual_seed(34)
        state = training_recovery.capture_rng_state()
        state["torch"] = state["torch"].clone()

        expected_python = random.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(1).item()

        random.seed(56)
        np.random.seed(56)
        torch.manual_seed(56)

        training_recovery.restore_rng_state(state)

        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        self.assertEqual(torch.rand(1).item(), expected_torch)

    def test_recovery_checkpoint_retention_keeps_latest_snapshots(self):
        import training_recovery

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            scaler = torch.amp.GradScaler("cpu")

            for batch in range(1, 5):
                training_recovery.save_recovery_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=0,
                    next_batch=batch,
                    global_step=batch,
                    keep_snapshots=2,
                    write_numbered_snapshot=True,
                )

            snapshots = sorted(checkpoint_dir.glob("recovery-epoch*-batch*.pt"))

            self.assertEqual(
                [path.name for path in snapshots],
                ["recovery-epoch0-batch3.pt", "recovery-epoch0-batch4.pt"],
            )
            self.assertTrue((checkpoint_dir / "recovery-latest.pt").exists())

    def test_interrupted_recovery_matches_uninterrupted_tiny_training(self):
        import settings
        import train_reconstruction
        import training_recovery

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.dropout = torch.nn.Dropout(p=0.35)

            def forward(self, inputs):
                return self.linear(self.dropout(inputs))

        class TinyLoss(torch.nn.Module):
            def forward(self, outputs, targets, epoch=0):
                return torch.nn.functional.mse_loss(outputs, targets)

        def make_batches():
            inputs = torch.arange(24, dtype=torch.float32).view(6, 4) / 10.0
            targets = torch.flip(inputs, dims=[1]) * 0.5
            return [(inputs[idx : idx + 1], targets[idx : idx + 1]) for idx in range(6)]

        def make_training_stack():
            model = TinyModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            scaler = torch.amp.GradScaler("cpu")
            return model, optimizer, scheduler, scaler

        with tempfile.TemporaryDirectory() as tmpdir:
            original_device = train_reconstruction.device
            original_checkpoint_batches = settings.step_checkpoint_every_batches
            original_checkpoint_minutes = settings.step_checkpoint_every_minutes
            original_keep_checkpoints = settings.keep_step_checkpoints
            original_save_to_dir = settings.save_to_dir

            try:
                train_reconstruction.device = torch.device("cpu")
                settings.step_checkpoint_every_batches = 2
                settings.step_checkpoint_every_minutes = 0
                settings.keep_step_checkpoints = 1
                settings.save_to_dir = str(Path(tmpdir) / "checkpoints")

                torch.manual_seed(20260610)
                initial_model, _, _, _ = make_training_stack()
                initial_state = {
                    key: value.detach().clone()
                    for key, value in initial_model.state_dict().items()
                }

                torch.manual_seed(20260611)
                uninterrupted_model, uninterrupted_optimizer, uninterrupted_scheduler, uninterrupted_scaler = (
                    make_training_stack()
                )
                uninterrupted_model.load_state_dict(initial_state)
                train_reconstruction.train_epoch(
                    uninterrupted_model,
                    make_batches(),
                    uninterrupted_optimizer,
                    TinyLoss(),
                    uninterrupted_scaler,
                    uninterrupted_scheduler,
                )

                torch.manual_seed(20260611)
                interrupted_model, interrupted_optimizer, interrupted_scheduler, interrupted_scaler = (
                    make_training_stack()
                )
                interrupted_model.load_state_dict(initial_state)
                train_reconstruction.train_epoch(
                    interrupted_model,
                    make_batches()[:2],
                    interrupted_optimizer,
                    TinyLoss(),
                    interrupted_scaler,
                    interrupted_scheduler,
                )

                resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler = (
                    make_training_stack()
                )
                loaded = training_recovery.load_recovery_checkpoint(
                    Path(settings.save_to_dir) / "recovery-latest.pt",
                    model=resumed_model,
                    optimizer=resumed_optimizer,
                    scheduler=resumed_scheduler,
                    scaler=resumed_scaler,
                    map_location=torch.device("cpu"),
                )

                self.assertEqual(loaded["next_batch"], 2)
                train_reconstruction.train_epoch(
                    resumed_model,
                    make_batches()[2:],
                    resumed_optimizer,
                    TinyLoss(),
                    resumed_scaler,
                    resumed_scheduler,
                    batch_number_offset=loaded["next_batch"],
                    global_step=loaded["global_step"],
                    train_loss_total=loaded["train_loss_total"],
                    train_loss_samples=loaded["train_loss_samples"],
                )

                for name, uninterrupted_value in uninterrupted_model.state_dict().items():
                    torch.testing.assert_close(
                        resumed_model.state_dict()[name],
                        uninterrupted_value,
                        msg=f"Mismatch after recovery resume for {name}",
                    )
            finally:
                train_reconstruction.device = original_device
                settings.step_checkpoint_every_batches = original_checkpoint_batches
                settings.step_checkpoint_every_minutes = original_checkpoint_minutes
                settings.keep_step_checkpoints = original_keep_checkpoints
                settings.save_to_dir = original_save_to_dir


if __name__ == "__main__":
    unittest.main()
