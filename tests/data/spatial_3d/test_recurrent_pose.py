"""A repeated pose in a legal program is not a reversal of the program."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from data.spatial_3d.core.source_replay import replay_spatial_3d
from data.spatial_3d.core.verifier import _verify_motion_video


class RecurrentPoseTests(unittest.TestCase):
    def test_source_program_passes_but_reversal_and_omission_fail(self):
        root = Path(__file__).parent / "fixtures/recurrent_pose"
        fixture = json.loads((root / "source.json").read_text())
        replay = replay_spatial_3d(SimpleNamespace(
            task_name="polycube_rotation", semantic_input=fixture["source"],
        ))
        motion_frames = 16 * len(fixture["source"]["rotations"]) + 15
        initial = np.asarray(replay.render_progress(0.0).convert("RGB"), dtype=np.uint8)
        terminal = np.asarray(replay.render_progress(1.0).convert("RGB"), dtype=np.uint8)
        frames = [initial.copy() for _ in range(5)]
        for index in range(1, motion_frames + 5):
            frames.append(
                np.asarray(
                    replay.render_progress(index / (motion_frames + 4)).convert("RGB"),
                    dtype=np.uint8,
                )
            )
        frames.extend(terminal.copy() for _ in range(5))
        self.assertEqual(_verify_motion_video(frames, fixture["fps"], replay)["whole_video_process_correct"], 1)
        cut = frames[:45] + frames[60:] + [frames[-1]] * 15
        for corrupted in (list(reversed(frames)), cut):
            self.assertEqual(_verify_motion_video(corrupted, fixture["fps"], replay)["whole_video_process_correct"], 0)


if __name__ == "__main__":
    unittest.main()
