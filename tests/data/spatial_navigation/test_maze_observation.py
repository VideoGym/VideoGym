"""Maze observations tolerate rendering variation without inventing a cursor."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from data.common.pipeline.path_tasks import _draw_pawn_marker, draw_path_line
from data.spatial_navigation.core import path_verifier as verifier
from data.common.verifiers.region_gates import STATIC_REGION_FLOOR, static_region_errors


class MazeObservationTests(unittest.TestCase):
    @staticmethod
    def red_mask(pixels):
        values = pixels.astype(np.int16)
        return (values[:, :, 0] >= 150) & (
            values[:, :, 0] - np.maximum(values[:, :, 1], values[:, :, 2]) >= 80
        )

    def cursor_result(self, corruption):
        progress = [0.0] * 5 + [i / 16 for i in range(1, 17)] + [1.0] * 8
        frames = []
        for index, fraction in enumerate(progress):
            image = Image.new("RGB", (128, 128), "white")
            tip_x = 16 + 96 * fraction
            missing = (corruption == "single_erasure" and index == 13) or (
                corruption == "long_erasure" and 10 <= index <= 20
            )
            draw_path_line(image, [(16.0, 64.0), (tip_x, 64.0)],
                           (220, 30, 30), 10, joint_style="square", tip_marker=not missing)
            draw = ImageDraw.Draw(image)
            if corruption == "deformed" and index == 13:
                draw.ellipse((tip_x - 10, 54, tip_x + 10, 74), fill="white")
                draw.ellipse((tip_x - 8, 61, tip_x + 8, 67), fill=(220, 30, 30))
            if corruption == "duplicate" and index == 16:
                _draw_pawn_marker(draw, (35.0, 64.0), 10, (220, 30, 30))
            if corruption == "deformed_duplicate" and index == 16:
                draw.ellipse((25, 54, 45, 74), fill="white")
                draw.ellipse((27, 62, 43, 66), fill=(220, 30, 30))
            ghost = corruption in {"local_small_ghost", "far_small_ghost", "local_equal_ghost"} and index == 16
            ghost |= corruption == "persistent_small_ghost" and 15 <= index <= 17
            if ghost:
                gx = 35 if corruption == "far_small_ghost" else tip_x - 23
                if corruption == "local_equal_ghost":
                    _draw_pawn_marker(draw, (gx, 64.0), 10, (220, 30, 30))
                else:
                    draw.ellipse((gx - 8, 56, gx + 8, 72), fill="white")
                    draw.ellipse((gx - 4, 60, gx + 4, 68), fill=(220, 30, 30))
            frames.append(self.red_mask(np.asarray(image)))
        with patch.object(verifier, "_maze_route_subpaths", return_value=(
            [[(16.0, 64.0), (112.0, 64.0)]], 10, "square",
        )):
            return verifier._verify_maze_tip_advance(
                SimpleNamespace(task_name="maze_square"), {},
                SimpleNamespace(_red_mask=self.red_mask), [1, 2],
                frames, frames, progress,
            )

    def test_visible_deformed_cursor_is_recovered(self):
        self.assertTrue(self.cursor_result("reference")[1])
        metrics, passed = self.cursor_result("deformed")
        self.assertTrue(passed)
        self.assertEqual(metrics["video_tip_recovered_shape_frames"], [13])

    def test_erased_or_duplicated_cursor_is_rejected(self):
        for corruption in ("single_erasure", "long_erasure", "duplicate", "deformed_duplicate",
                           "far_small_ghost", "local_equal_ghost", "persistent_small_ghost"):
            with self.subTest(corruption=corruption):
                self.assertFalse(self.cursor_result(corruption)[1])

    def test_one_small_local_ghost_retains_the_same_cursor(self):
        metrics, passed = self.cursor_result("local_small_ghost")
        self.assertTrue(passed)
        self.assertEqual(metrics["video_tip_recovered_isolated_artifact_frames"], [16])

    def test_dry_aqueduct_lining_is_not_flow_after_codec_shift(self):
        colours = np.array([[
            (48, 132, 210), (40, 150, 225), (24, 174, 236), (29, 168, 229),
        ]], dtype=np.uint8)
        self.assertEqual(verifier._aqueduct_flow_mask(colours).tolist(), [[False, False, True, True]])

    def test_one_pixel_edge_shift_passes_but_wall_removal_and_foreign_patch_fail(self):
        source = np.full((64, 64, 3), 240, dtype=np.uint8)
        source[:, 16:24] = (28, 34, 46)
        shifted = np.roll(source, 1, axis=1)
        removed = np.full_like(source, 240)
        foreign = source.copy()
        foreign[8:32, 36:60] = (255, 0, 255)
        static = np.ones((64, 64), dtype=bool)
        self.assertLessEqual(
            static_region_errors(shifted, source, static, raster_tolerance=1)[0],
            STATIC_REGION_FLOOR,
        )
        for candidate in (removed, foreign):
            with self.subTest(corruption="removed" if candidate is removed else "foreign"):
                self.assertGreater(
                    static_region_errors(candidate, source, static, raster_tolerance=1)[0],
                    STATIC_REGION_FLOOR,
                )


if __name__ == "__main__":
    unittest.main()
