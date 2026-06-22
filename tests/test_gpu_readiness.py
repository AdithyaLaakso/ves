import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

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
                outputs = torch.nn.functional.interpolate(
                    inputs,
                    size=(4, 4),
                    mode="bilinear",
                    align_corners=False,
                )
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

    def test_visual_review_cli_help_runs_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "model/usage/visual_review_sheets.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("Generate headless visual review contact sheets", result.stdout)


if __name__ == "__main__":
    unittest.main()
