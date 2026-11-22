"""Control expectations follow the public construction or marking contract."""

from pathlib import Path
import tempfile
import unittest

from evaluation.execution.datasets.generation import _generate
from evaluation.execution.scoring.qualification import qualify


class QualificationTests(unittest.TestCase):
    def test_single_rotation_arc_requires_visible_growth_through_occlusion(self):
        with tempfile.TemporaryDirectory() as directory:
            # A candidate disc hides the middle of this rotation arc.
            for task, seed in (("point_rotation", 4338489847459378337), ("inverse_rotation", 413)):
                with self.subTest(task=task):
                    task_dir = Path(directory) / task
                    sample = _generate(task, {}, seed, task_dir)
                    result = qualify(sample, controls=["reference", "terminal_cut"], base_dir=task_dir)
                    self.assertTrue(result["passed"], result)
                    self.assertTrue(result["controls"]["reference"]["accepted"])
                    self.assertFalse(result["controls"]["terminal_cut"]["accepted"])

    def test_direct_midpoint_mark_is_valid_but_cut_construction_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            for task, accepted in (("midpoint", True), ("angle_bisector", False)):
                with self.subTest(task=task):
                    task_dir = Path(directory) / task
                    sample = _generate(task, {}, 413, task_dir)
                    result = qualify(sample, controls=["terminal_cut", "unchanged_input"], base_dir=task_dir)
                    self.assertTrue(result["passed"], result)
                    self.assertEqual(result["controls"]["terminal_cut"]["accepted"], accepted)
                    self.assertFalse(result["controls"]["unchanged_input"]["accepted"])


if __name__ == "__main__":
    unittest.main()
