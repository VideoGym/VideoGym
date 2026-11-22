"""Sokoban color variation must not change the observed discrete program."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from data.common.schemas import CanonicalAnswer, CanonicalSample
from data.dynamics_game.core.sokoban import (
    _compress_states,
    _render_terminal,
    _scene_evidence,
    _token_mask,
    _track,
    _verify_video,
)


class SokobanObservationTests(unittest.TestCase):
    def setUp(self):
        grid = [[9 if row in (0, 6) or col in (0, 6) else 0 for col in range(7)] for row in range(7)]
        grid[3][1], grid[3][2], grid[3][4] = 3, 2, 4
        self.source = {"initial_grid": grid}
        self.states = [((3, 1), (3, 2)), ((3, 2), (3, 3)), ((3, 3), (3, 4))]
        self.frames = [self.render(state) for state in self.states]

    def render(self, state):
        return _render_terminal(self.source, {"final_player": state[0], "final_box": state[1], "target": (3, 4)})

    def test_complete_shifted_color_tokens_retain_the_same_cells(self):
        frame = self.frames[1].copy()
        frame[np.linalg.norm(frame.astype(float) - (255, 214, 10), axis=2) < 12] = (251, 217, 80)
        frame[np.linalg.norm(frame.astype(float) - (160, 113, 74), axis=2) < 12] = (183, 137, 69)
        player, box, detected, offsets = _track([frame], self.source)
        self.assertEqual((player[0], box[0]), self.states[1])
        self.assertTrue(detected[0])
        self.assertLess(offsets[0], 0.03)
        error, shapes = _scene_evidence(frame, self.frames[1], self.states[1], cell=64, margin=32)
        self.assertLess(error, 12 / 255)
        self.assertTrue(shapes)

    def test_cargo_platforms_and_bridges_are_not_tokens(self):
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase.new_scenes import _cargo

        with patch("data.dynamics_game.sokoban.generator.SokobanPlanGenerator._render_board", _cargo):
            frames = [self.render(state) for state in self.states]
        player, box, detected, offsets = _track(frames, self.source)
        self.assertEqual(list(zip(player, box)), self.states)
        self.assertTrue(all(detected))
        self.assertLess(max(offsets), 0.03)
        for frame, state in zip(frames, self.states):
            error, shapes = _scene_evidence(frame, frame, state, cell=64, margin=32)
            self.assertEqual(error, 0)
            self.assertTrue(shapes)

    def test_shortest_discrete_video_accepts_color_changes(self):
        frames = [frame.copy() for frame in self.frames for _ in range(4)]
        frames.append(frames[-1].copy())
        frames[5][np.linalg.norm(frames[5].astype(float) - (255, 214, 10), axis=2) < 12] = (251, 217, 80)
        frames[6][np.linalg.norm(frames[6].astype(float) - (160, 113, 74), axis=2) < 12] = (183, 137, 69)
        with tempfile.TemporaryDirectory() as directory:
            from PIL import Image
            image = Path(directory) / "source.png"
            Image.fromarray(self.frames[0]).save(image)
            sample = CanonicalSample(
                id="sokoban-test", task_name="sokoban", task_family="dynamics_game",
                prompt="Solve Sokoban.",
                image=str(image),
                ground_truth=CanonicalAnswer(kind="path", path_positions=[[4, 2], [4, 3], [4, 4]]),
                semantic_input=self.source,
            )
            with patch("data.dynamics_game.core.sokoban._video_frames", return_value=(frames, 16)),\
                    patch("data.dynamics_game.core.sokoban.validate_video_contract", return_value={}):
                metrics, passed = _verify_video(sample, Path(directory) / "candidate.webm", self.source)
            self.assertTrue(passed, metrics)

    def test_missing_duplicate_and_displaced_tokens_do_not_hide_in_scene_mask(self):
        expected = self.frames[1]
        for kind in ("missing", "duplicate", "displaced"):
            with self.subTest(kind=kind):
                frame = expected.copy()
                if kind == "missing":
                    frame[_token_mask(frame, "player") > 0] = (245, 229, 204)
                elif kind == "duplicate":
                    cv2.circle(frame, (320, 160), 21, (255, 214, 10), -1)
                else:
                    frame = self.render(((2, 2), (3, 3)))
                error, shapes = _scene_evidence(frame, expected, self.states[1], cell=64, margin=32)
                self.assertFalse(error <= 12 / 255 and shapes)

    def test_illegal_crate_changes_are_kept_for_rule_replay(self):
        player = [(3, 1), (3, 1), (3, 2)]
        box = [(3, 2), (3, 3), (3, 3)]
        self.assertEqual(_compress_states(player, box), list(zip(player, box)))

    def test_between_cell_motion_is_not_a_discrete_state(self):
        frame = self.frames[0].copy()
        frame[_token_mask(frame, "player") > 0] = (245, 229, 204)
        cv2.circle(frame, (160, 256), 21, (255, 214, 10), -1)
        _player, _box, detected, offsets = _track([frame], self.source)
        self.assertTrue(detected[0])
        self.assertGreater(offsets[0], 0.25)


if __name__ == "__main__":
    unittest.main()
