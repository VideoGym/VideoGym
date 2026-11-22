"""Do not demand an intermediate image the source renderer cannot expose."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.geometry.core.verifier import _single_endpoint_growth
from evaluation.shared.scenes.catalog import module
from evaluation.shared.scenes.media import frames


class CompositionVisibilityTests(unittest.TestCase):
    def test_short_exposed_arc_passes_but_full_solution_cut_fails(self):
        source = json.loads((Path(__file__).parent / "fixtures/short_arc_source.json").read_text())
        reference = frames(source)
        scene = module(source["key"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.webm"
            self.assertTrue(encode_rgb_frames_to_webm(reference, path, fps=16))
            self.assertEqual(scene.grade(source, path)["whole_video_process_correct"], 1)
            cut = [reference[0]] * 5 + [reference[-1]] * (len(reference) - 5)
            self.assertTrue(encode_rgb_frames_to_webm(cut, path, fps=16))
            self.assertEqual(scene.grade(source, path)["whole_video_process_correct"], 0)

    def test_partial_growth_remains_required_without_visibility_calibration(self):
        pop = [np.zeros(12, dtype=bool), np.ones(12, dtype=bool)]
        self.assertFalse(_single_endpoint_growth(pop)[0])
        self.assertTrue(_single_endpoint_growth(pop, require_partial=False)[0])
        self.assertFalse(_single_endpoint_growth([pop[0], pop[0]], require_partial=False)[0])


if __name__ == "__main__":
    unittest.main()
