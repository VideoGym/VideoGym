import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.verification.navigation.grid import verify


def make(path, obstacles=()):
    n, cell = 5, 40
    base = np.full((n * cell, n * cell, 3), 255, np.uint8)
    for i in range(n + 1):
        cv2.line(base, (i * cell, 0), (i * cell, n * cell), (180, 180, 180), 1)
        cv2.line(base, (0, i * cell), (n * cell, i * cell), (180, 180, 180), 1)
    for r, c in obstacles:
        cv2.rectangle(base, (c * cell + 5, r * cell + 5), ((c + 1) * cell - 5, (r + 1) * cell - 5), (0, 0, 0), -1)
    frames = []
    for r, c in path:
        f = base.copy()
        cv2.circle(f, (c * cell + 20, r * cell + 20), 10, (0, 165, 255), -1)
        frames.append(f)
    metadata = {
        "parameters": {},
        "semantic_ground_truth": {"start": [0, 4], "end": [4, 0], "obstacles": [[c, r] for r, c in obstacles]},
        "generic_declarative_render": {
            "canvas": {"width": 200, "height": 200},
            "first_frame": {
                "layers": [
                    {"type": "grid_lines", "rows": 5, "cols": 5, "cell_size_px": 40},
                    {
                        "type": "filled_shape",
                        "kind": "circle",
                        "cx_px": 20,
                        "cy_px": 180,
                        "radius_px": 10,
                        "fill_rgb": [255, 165, 0],
                    },
                ]
            },
        },
    }
    return frames, metadata


def evaluate(path, obstacles=(), transform=None):
    frames, meta = make(path, obstacles)
    first, final = frames[0].copy(), frames[-1].copy()
    if transform:
        frames = transform(frames)
    return verify("G-15_grid_avoid_obstacles_data-generator", frames, first, final, meta, [])


ROUTE = [(4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


def test_shortest_path_accepts_alternate_route():
    assert evaluate(ROUTE)["passed"]
    alternate = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (3, 4), (2, 4), (1, 4), (0, 4)]
    assert evaluate(alternate)["passed"]


def test_unobserved_turns_diagonal_motion_and_frozen_fail():
    assert not evaluate([ROUTE[0], ROUTE[-1]])["passed"]
    assert not evaluate([(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)])["passed"]
    assert not evaluate([ROUTE[0]] * 8)["passed"]


def test_wall_collision_and_detour_fail():
    assert not evaluate(ROUTE, [(2, 0)])["passed"]
    assert not evaluate([*ROUTE[:3], ROUTE[1], *ROUTE[2:]])["passed"]


def test_duplicate_agent_and_mid_video_corruption_fail():
    def duplicate(frames):
        for frame in frames:
            cv2.circle(frame, (180, 180), 10, (0, 165, 255), -1)
        return frames

    assert not evaluate(ROUTE, transform=duplicate)["passed"]

    def corrupt(frames):
        frames[3] = np.zeros_like(frames[3])
        return frames

    assert not evaluate(ROUTE, transform=corrupt)["passed"]


def test_drawn_maze_accepts_red_trail_and_checks_walls():
    cell = 40
    state = np.ones((5, 5), dtype=int)
    state[2, 1] = 0
    first = np.full((200, 200, 3), 255, dtype=np.uint8)
    first[80:120, 40:80] = 30
    cv2.circle(first, (20, 180), 6, (50, 200, 50), -1)
    cv2.circle(first, (180, 20), 6, (50, 50, 200), -1)
    metadata = {
        "parameters": {
            "start": list(ROUTE[0]),
            "end": list(ROUTE[-1]),
            "objects": [{"symbol": "path", "color": [255, 200, 0]}],
        },
        "semantic_ground_truth": {"grid": state.tolist()},
        "generic_declarative_render": {
            "canvas": {"width": 200, "height": 200},
            "first_frame": {
                "layers": [{"type": "cell_grid", "rows": 5, "cols": 5, "cell_size_px": cell, "state": state.tolist()}],
            },
        },
    }
    frames = [first.copy()]
    for (row, col), (next_row, next_col) in zip(ROUTE, ROUTE[1:]):
        frame = frames[-1].copy()
        cv2.line(frame, (40 * col + 20, 40 * row + 20), (40 * next_col + 20, 40 * next_row + 20), (40, 40, 220), 4)
        frames.append(frame)

    assert verify("O-39_maze_data-generator", frames, first, first, metadata, [])["passed"]

    cv2.line(frames[-1], (20, 100), (60, 100), (40, 40, 220), 4)
    assert not verify("O-39_maze_data-generator", frames, first, first, metadata, [])["checks"]["zero_wall_crossings"]
