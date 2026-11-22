"""Decode overlapping public stroke hypotheses without fitting to the answer."""

import unittest
from unittest.mock import patch

import numpy as np

from evaluation.cross_task.engine import grade, reference_states, render, solve


SOURCES = [
    {"goal": "taut_obstacle_path", "start": [48, 199], "goal_point": [464, 335],
     "obstacles": [[135, 118, 195, 322], [290, 244, 350, 440]]},
    {"goal": "taut_obstacle_path", "start": [48, 168], "goal_point": [464, 349],
     "obstacles": [[135, 110, 195, 317], [290, 250, 350, 440]]},
    {"goal": "bichromatic_matching",
     "blue": [[200, 352], [308, 367], [182, 411], [123, 369], [183, 212]],
     "red": [[75, 159], [181, 149], [357, 292], [267, 298], [414, 442]]},
    {"goal": "disk_union_boundary",
     "circles": [[196, 236, 93], [319, 246, 100], [250, 346, 90]]},
]


class StrokeObservationTests(unittest.TestCase):
    def test_complete_but_wrong_path_and_matching_are_rejected(self):
        from evaluation.cross_task.families.geometry import path_graph

        source = SOURCES[0]
        _, edges = path_graph(source)
        route = [0, 5, 4, 9, 8, 1]
        selected = {edges.index(tuple(sorted(pair))) for pair in zip(route, route[1:])}
        wrong_path = tuple(1 if i in selected else -1 for i in range(len(edges)))
        wrong_matching = tuple(1 if i in (0, 6, 12, 18, 24) else -1 for i in range(25))
        for source, answer in ((source, wrong_path), (SOURCES[2], wrong_matching)):
            states = reference_states(source, answer=answer)
            frames = [np.asarray(render(source, state)) for state in states]
            self.assertEqual(grade(source, frames)[:2], (False, False))

    def test_valid_strokes_and_visible_missing_segments(self):
        for source in SOURCES:
            with self.subTest(source=source):
                states = reference_states(source)
                frames = [np.asarray(render(source, state)) for state in states]
                self.assertEqual(grade(source, frames)[:2], (True, True))
                missing = list(states[-1])
                missing[next(i for i, value in enumerate(missing) if value != -1)] = -1
                frame = np.asarray(render(source, missing))
                self.assertFalse(grade(source, [frames[0], frame])[1])

    def test_answer_changes_cannot_change_pixel_decoding(self):
        source = SOURCES[1]
        states = reference_states(source)
        frames = [np.asarray(render(source, state)) for state in states]
        expected = tuple(solve(source)[0])
        with patch("evaluation.cross_task.engine.solve", return_value=[(-1,) * len(expected)]):
            process, final, details = grade(source, frames)
        self.assertFalse(process)
        self.assertFalse(final)
        self.assertEqual(details["decoded_answer"], expected)


if __name__ == "__main__":
    unittest.main()
