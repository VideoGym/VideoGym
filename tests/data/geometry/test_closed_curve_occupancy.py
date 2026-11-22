"""Circular construction must grow spatially, independent of RGB edge flicker."""
import unittest
import numpy as np
from data.geometry.core.verifier import _closed_curve_occupancy_growth


class ClosedCurveOccupancyTests(unittest.TestCase):
    def setUp(self):
        self.parameters = np.linspace(0, 1, 192)
        self.progress = np.arange(192)[None, :] < np.linspace(0, 192, 10)[:, None]

    def test_single_sweep_may_start_anywhere_and_run_either_direction(self):
        for direction in (1, -1):
            for offset in (0, 36, 108):
                with self.subTest(direction=direction, offset=offset):
                    hits = np.roll(self.progress[:, ::direction], offset, axis=1)
                    self.assertTrue(_closed_curve_occupancy_growth(hits, self.parameters)[0])

    def test_whole_circle_fade_and_disconnected_sweeps_fail(self):
        fade = np.tile(np.arange(10)[:, None] >= 5, (1, 192))
        self.assertFalse(_closed_curve_occupancy_growth(fade, self.parameters)[0])
        two_pens = self.progress | np.roll(self.progress, 96, axis=1)
        self.assertFalse(_closed_curve_occupancy_growth(two_pens, self.parameters)[0])

    def test_erasure_and_missing_arc_fail(self):
        erased = self.progress.copy()
        erased[6:8, :48] = False
        self.assertFalse(_closed_curve_occupancy_growth(erased, self.parameters)[0])
        missing = self.progress.copy()
        missing[:, 168:] = False
        self.assertFalse(_closed_curve_occupancy_growth(missing, self.parameters)[0])
