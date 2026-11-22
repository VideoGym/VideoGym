"""Zero-Torch unit tests for training presets, adapter inference config, and VBVR root resolution."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from evaluation.cross_benchmark.vbvr_pro.benchmark.catalog import resolve_gt_root
from inference.backends.trained_adapter.__main__ import _config as adapter_config
from training.configuration.schema import load_config


PRESETS = Path(__file__).resolve().parents[2] / "training" / "configuration" / "presets"


class TrainingConfigAndCliTests(unittest.TestCase):
    def test_all_training_presets_load_and_validate(self):
        preset_files = sorted(PRESETS.glob("*.toml"))
        self.assertEqual(
            [p.name for p in preset_files],
            ["ltx23.toml", "wan_i2v_a14b.toml", "wan_ti2v.toml"],
        )
        for preset in preset_files:
            with self.subTest(preset=preset.name):
                cfg = load_config(preset)
                self.assertEqual(cfg.stage, "sft")

    def test_trained_adapter_config_disables_fsdp_and_honors_tuning(self):
        args = SimpleNamespace(
            config=PRESETS / "ltx23.toml",
            model=None,
            model_path=None,
            checkpoint="/tmp/ckpt",
            tuning="lora",
            local_files_only=True,
            device="cuda:0",
        )
        cfg = adapter_config(args)
        self.assertEqual(cfg.sharding, "none")
        self.assertEqual(cfg.tuning, "lora")
        self.assertTrue(cfg.local_files_only)

    def test_vbvr_resolve_gt_root_accepts_parent_or_nested_video_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "VBVR-Pro-Bench-Video"
            (nested / "In-Domain_50").mkdir(parents=True)
            self.assertEqual(resolve_gt_root(root), nested.resolve())
            self.assertEqual(resolve_gt_root(nested), nested.resolve())


if __name__ == "__main__":
    unittest.main()
