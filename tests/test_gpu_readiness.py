import sys
import subprocess
import tempfile
import unittest
import random
from pathlib import Path

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

    def test_setup_script_makes_cuda_debug_opt_in(self):
        setup_text = (MODEL_DIR / "setup.zsh").read_text(encoding="utf-8")

        self.assertIn("VES_DEBUG_CUDA", setup_text)
        self.assertNotIn("export CUDA_LAUNCH_BLOCKING=1", setup_text)


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


class TrainingRecoveryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
