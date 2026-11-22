import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.verification.rearrangement.puzzles import _stack


def test_stack_handoff_and_final_state_are_observed_independently():
    palette = [(235, 75, 50), (50, 160, 235), (70, 190, 95)]
    parameters = {
        "layout": {"block_width": 96, "block_height": 64, "ground_y": 824, "stack_spacing": 128},
        "num_stacks": 3,
        "block_colors": [0, 1, 2],
        "block_palette_rgb": palette,
        "initial_state": [[0], [1, 2], []],
        "target_state": [[0], [], [1, 2]],
    }

    def board(positions, duplicate=False):
        frame = np.full((1024, 1024, 3), 240, np.uint8)
        for x in (128, 256, 384, 640, 768, 896):
            cv2.line(frame, (x - 50, 829), (x + 50, 829), (120, 120, 120), 2)
        for color, (x, y) in enumerate(positions):
            cv2.rectangle(frame, (x - 48, y - 32), (x + 47, y + 31), palette[color][::-1], -1)
        if duplicate:
            cv2.rectangle(frame, (208, 696), (303, 759), palette[2][::-1], -1)
        return frame

    first = board([(128, 792), (256, 792), (256, 728)])
    in_flight = [(128, 792), (256, 792), (192, 650)]
    frames = [
        first,
        board(in_flight),
        board([(128, 792), (315, 784), (134, 722)]),
        board([(128, 792), (384, 792), (128, 728)]),
        board([(128, 792), (384, 792), (265, 650)]),
        board([(128, 792), (384, 792), (384, 728)]),
    ]
    correct = _stack(frames, first, parameters)
    assert correct["passed"], correct
    frames[1] = board(in_flight, duplicate=True)
    duplicated = _stack(frames, first, parameters)
    assert not duplicated["passed"]
    assert not duplicated["checks"]["every_original_block_exists_exactly_once"]
    assert duplicated["checks"]["final_stack_order_matches_goal"]
