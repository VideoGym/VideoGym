"""Independent source replay and complete-video verification for next-state tasks."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile
from data.common.verifiers.keyframes import keyframe_indices, visual_delta, ordered_state_observations

from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.transitions import complete_footprint_projection as _complete_footprint_projection
from data.common.verifiers.region_gates import (
    STATIC_REGION_FLOOR,
    static_region_errors,
    terminal_region_gates,
)
from data.next_state_prediction.core.catalog import EXPANSION_TASK_NAMES
from data.next_state_prediction.core.rendering import (
    gui_control_bboxes,
    render_gui_action_press,
    render_state,
)
from data.next_state_prediction.core.semantics import answer_payload, replay_case


_SHARED_TASKS = frozenset(EXPANSION_TASK_NAMES)
_LEGACY_TASKS = frozenset({
    "arc1d_compact_nonzero",
    "binary_matrix",
    "connected_components",
    "grid_bfs",
    "grid_component",
    "grid_shift",
    "rotten_oranges",
    "traffic_light_countdown",
})
_VIDEO_NAMES = (
    "solution.webm", "generated.webm", "candidate.webm", "output.webm",
    "prediction.webm", "solution.mp4", "generated.mp4", "candidate.mp4",
    "output.mp4", "prediction.mp4",
)
_GUI_STABLE_PANEL_ERROR = 0.040
_GUI_MIN_EDGE_RECALL = 0.68
_GUI_MIN_PRESS_PIXELS = 40
# The contact dot spans only ~10 target pixels; at the 4x supersampled render
# its LANCZOS anti-aliasing widens the per-pixel progress quantiles at faint
# opacities (measured 0.56 on reference presses).  0.60 keeps rejecting
# directional or radial dot reveals, whose spread approaches 1.0.
_GUI_DOT_MAX_FOOTPRINT_PROGRESS_SPREAD = 0.60
_LOCAL_FADE_MAX_FOOTPRINT_PROGRESS_SPREAD = 0.24
_LOCAL_FADE_MIN_FOOTPRINT_SUPPORT = 0.72
_LOCAL_FADE_MAX_LINEAR_FIT_ERROR = 0.20

@dataclass(frozen=True)
class _Replay:
    answer: Any
    stable_images: tuple[Image.Image, ...]
    source_variant: str


def _source_contract(sample: CanonicalSample) -> Mapping[str, Any]:
    semantic = sample.semantic_input
    if isinstance(semantic, Mapping):
        for key in ("semantic_state", "source_contract", "structured_input", "structured_answer"):
            structured = semantic.get(key)
            if isinstance(structured, Mapping):
                return structured
        if semantic:
            return semantic
    return {}


def _components4(grid: Sequence[Sequence[int]]) -> list[list[tuple[int, int]]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for row in range(rows):
        for col in range(cols):
            if int(grid[row][col]) != 1 or (row, col) in seen:
                continue
            queue = deque([(row, col)])
            seen.add((row, col))
            component: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                component.append((rr, cc))
                for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < rows and 0 <= nc < cols
                        and int(grid[nr][nc]) == 1
                        and (nr, nc) not in seen
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            components.append(component)
    return components


def _distance_grid(
    grid: Sequence[Sequence[Any]],
    *,
    source_value: Any,
    wall_value: Any = None,
) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    distances = [[-1 for _ in range(cols)] for _ in range(rows)]
    queue: deque[tuple[int, int]] = deque()
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == source_value:
                distances[row][col] = 0
                queue.append((row, col))
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if wall_value is not None and grid[nr][nc] == wall_value:
                continue
            if distances[nr][nc] >= 0:
                continue
            distances[nr][nc] = distances[row][col] + 1
            queue.append((nr, nc))
    return distances


def _arc_states(initial: Sequence[int]) -> list[list[int]]:
    states = [[int(value) for value in initial]]
    while True:
        old = states[-1]
        movers = [index for index, value in enumerate(old) if value != 0 and index > 0 and old[index - 1] == 0]
        if not movers:
            return states
        new = list(old)
        for index in movers:
            new[index - 1] = old[index]
            new[index] = 0
        states.append(new)


def _rotten_replay(initial: Sequence[Sequence[int]]) -> tuple[int, list[list[list[int]]]]:
    grid = [[int(value) for value in row] for row in initial]
    states = [deepcopy(grid)]
    while True:
        changes: list[tuple[int, int]] = []
        for row in range(len(grid)):
            for col in range(len(grid[row])):
                if grid[row][col] != 1:
                    continue
                if any(
                    0 <= row + dr < len(grid)
                    and 0 <= col + dc < len(grid[0])
                    and grid[row + dr][col + dc] == 2
                    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0))
                ):
                    changes.append((row, col))
        if not changes:
            break
        for row, col in changes:
            grid[row][col] = 2
        states.append(deepcopy(grid))
    remaining = any(value == 1 for row in grid for value in row)
    return (-1 if remaining else len(states) - 1), states


def _traffic_replay(
    structured: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[list[dict[str, Any]]]]:
    order = tuple(str(item) for item in structured.get("direction_order") or ("N", "S", "E", "W"))
    current = [deepcopy(dict(item)) for item in structured["initial_states"]]
    timeline = [deepcopy(current)]
    next_state = {"green": "yellow", "yellow": "red", "red": "green"}
    for _ in range(int(structured["simulate_seconds"])):
        successor: list[dict[str, Any]] = []
        for item in current:
            state = str(item["state"])
            countdown = int(item["countdown"])
            durations = {str(key): int(value) for key, value in dict(item["durations"]).items()}
            if countdown > 1:
                next_colour, next_count = state, countdown - 1
            else:
                next_colour = next_state[state]
                next_count = durations[next_colour]
            successor.append({**deepcopy(item), "state": next_colour, "visible_state": next_colour, "countdown": next_count})
        current = successor
        timeline.append(deepcopy(current))
    by_direction = {str(item["direction"]): item for item in current}
    answer = {
        direction: {
            "countdown": int(by_direction[direction]["countdown"]),
            "state": str(by_direction[direction]["state"]),
        }
        for direction in order
    }
    return answer, timeline


def _replay_shared(sample: CanonicalSample, structured: Mapping[str, Any]) -> _Replay:
    params = structured.get("params")
    initial = structured.get("initial_state")
    if not isinstance(params, Mapping) or not isinstance(initial, Mapping):
        raise ValueError("shared next-state source lacks params and initial_state")
    trace = replay_case(sample.task_name, params, initial)
    images = tuple(render_state(sample.task_name, state, params) for state in trace)
    return _Replay(answer_payload(sample.task_name, trace[-1], params), images, "shared24")


def _replay_legacy(sample: CanonicalSample, structured: Mapping[str, Any]) -> _Replay:
    task = sample.task_name
    if task == "arc1d_compact_nonzero":
        from data.next_state_prediction.arc1d_compact_nonzero.generator import Arc1DCompactNonzeroPattern
        states = _arc_states(structured["input_row"])
        renderer = Arc1DCompactNonzeroPattern(Path(tempfile.gettempdir()), seed=0, video=False)
        images = tuple(renderer._render_state(state, round_index=index, target_rounds=len(states) - 1) for index, state in enumerate(states))
        return _Replay(states[-1], images, "legacy8")

    if task == "binary_matrix":
        from data.next_state_prediction.binary_matrix.generator import BinaryMatrixPattern, _binary_distance_states
        initial = [[int(value) for value in row] for row in structured["input_matrix"]]
        distances = _distance_grid(initial, source_value=0)
        visual_states = _binary_distance_states(initial, distances)
        renderer = BinaryMatrixPattern(Path(tempfile.gettempdir()), seed=0, video=False)
        images = tuple(renderer._render_state(initial, distances, phase=phase, layer=layer) for phase, layer, _ in visual_states)
        return _Replay(distances, images, "legacy8")

    if task == "connected_components":
        from data.next_state_prediction.connected_components.generator import ConnectedComponentsPattern, _component_states
        grid = [[int(value) for value in row] for row in structured["input_grid"]]
        components = _components4(grid)
        states = _component_states(grid, components)
        renderer = ConnectedComponentsPattern(Path(tempfile.gettempdir()), seed=0, video=False)
        images = tuple(renderer._render_state(state, state_index=index, component_count=len(components)) for index, state in enumerate(states))
        return _Replay(len(components), images, "legacy8")

    if task == "grid_bfs":
        from data.next_state_prediction.grid_bfs.generator import GridBFSPattern, _bfs_visual_states
        grid = [[str(value) for value in row] for row in structured["input_grid"]]
        distances = _distance_grid(grid, source_value="1", wall_value="X")
        visual_states = _bfs_visual_states(grid, distances)
        renderer = GridBFSPattern(Path(tempfile.gettempdir()), seed=0, video=False)
        images = tuple(renderer._render_state(grid, distances, phase=phase, layer=layer) for phase, layer, _ in visual_states)
        return _Replay(distances, images, "legacy8")

    if task == "grid_component":
        from data.next_state_prediction.grid_component.generator import GridComponentPattern
        grid = [[int(value) for value in row] for row in structured["input_grid"]]
        components = _components4(grid)
        sizes = [len(component) for component in components]
        best_index = max(range(len(components)), key=lambda index: sizes[index])
        renderer = GridComponentPattern(Path(tempfile.gettempdir()), seed=0, video=False)
        images = [
            renderer._render_state(grid, components, phase="input"),
            renderer._render_state(grid, components, phase="final", component_index=best_index),
        ]
        return _Replay(sizes[best_index], tuple(images), "legacy8")

    if task == "grid_shift":
        from data.next_state_prediction.grid_shift.generator import GridShiftPattern
        base = [[value if value is None else int(value) for value in row] for row in structured["base_matrix"]]
        dr, dc = (int(value) for value in structured["shift"])
        rows, cols = len(base), len(base[0])
        target: list[list[Optional[int]]] = [[None for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            for col in range(cols):
                value = base[row][col]
                nr, nc = row + dr, col + dc
                if value is not None and 0 <= nr < rows and 0 <= nc < cols:
                    target[nr][nc] = value
        answer = [[0 if value is None else int(value) for value in row] for row in target]
        render_task = {**dict(structured), "base_matrix": base, "target_matrix": target, "answer_matrix": answer}
        renderer = GridShiftPattern(Path(tempfile.gettempdir()), seed=0, generate_videos=False)
        return _Replay(answer, (renderer._render_puzzle_image(render_task), renderer._render_solution_image(render_task)), "legacy8")

    if task == "rotten_oranges":
        from data.next_state_prediction.rotten_oranges.generator import RottenOrangesPattern
        answer, states = _rotten_replay(structured["input_grid"])
        renderer = RottenOrangesPattern(Path(tempfile.gettempdir()), seed=0, video=False)
        images = tuple(renderer._render_state(state, status="") for state in states)
        return _Replay(answer, images, "legacy8")

    if task == "traffic_light_countdown":
        from data.next_state_prediction.traffic_light_countdown.generator import TrafficLightCountdownPattern
        answer, timeline = _traffic_replay(structured)
        render_task = {**dict(structured), "timeline": timeline}
        renderer = TrafficLightCountdownPattern(Path(tempfile.gettempdir()), seed=0, video=False)
        images = tuple(renderer._render_scene(render_task, index) for index in range(len(timeline)))
        return _Replay(answer, images, "legacy8")

    raise ValueError(f"unsupported legacy next-state task: {task}")


def _independent_replay(sample: CanonicalSample) -> _Replay:
    structured = _source_contract(sample)
    if sample.task_name in _SHARED_TASKS:
        return _replay_shared(sample, structured)
    if sample.task_name in _LEGACY_TASKS:
        return _replay_legacy(sample, structured)
    raise ValueError(f"unsupported next-state task: {sample.task_name}")


def resolve_candidate_video(pred_root: Path, sample: CanonicalSample) -> Optional[Path]:
    directories = []
    if sample.split:
        directories.append(pred_root / sample.split / sample.id)
    directories.append(pred_root / sample.id)
    candidates: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        candidates.extend(directory / name for name in _VIDEO_NAMES)
        for suffix in ("*.webm", "*.mp4", "*.mov", "*.mkv"):
            candidates.extend(sorted(directory.glob(suffix)))
    candidates.extend(pred_root / f"{sample.id}{suffix}" for suffix in (".webm", ".mp4", ".mov", ".mkv"))
    return next((path for path in candidates if path.is_file()), None)


def _rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2)
    return array[:, :, :3]


def _resize_rgb(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if array.shape[:2] == (height, width):
        return array
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)


def _comparison_mask(initial: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    changed = np.max(np.abs(initial.astype(np.int16) - terminal.astype(np.int16)), axis=2) > 10
    if np.any(changed):
        changed = cv2.dilate(changed.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool)
    return changed


def _terminal_image_metrics(candidate: np.ndarray, initial: np.ndarray, terminal: np.ndarray) -> dict[str, Any]:
    candidate = _resize_rgb(candidate, terminal.shape[:2])
    diff_terminal = visual_delta(candidate, terminal)
    full_score = max(0.0, 1.0 - float(np.mean(diff_terminal)) / 255.0)
    changed = _comparison_mask(initial, terminal)
    if np.any(changed):
        changed_score = max(0.0, 1.0 - float(np.mean(diff_terminal[changed])) / 255.0)
    else:
        changed_score = full_score
    unchanged = ~changed
    if np.any(unchanged):
        preservation_error = visual_delta(candidate, initial)
        preservation_score = max(0.0, 1.0 - float(np.mean(preservation_error[unchanged])) / 255.0)
    else:
        preservation_score = full_score
    # Full-frame similarity alone is unsafe for sparse successor updates: an
    # unchanged input can be more than 99% identical to the terminal render.
    # The changed-region gate therefore carries the same strict threshold as
    # the preservation gate and prevents input-as-answer reward hacking.
    # All three remain means over a mask, so one erased cell, glyph or block is
    # still paid for by every surviving one; the per-region / per-member gates
    # hold each of them to its own evidence.
    region_metrics, region_pass = terminal_region_gates(
        candidate, initial, terminal, [initial], prefix="terminal_image_"
    )
    correct = (
        full_score >= 0.94
        and changed_score >= 0.94
        and preservation_score >= 0.94
        and region_pass
    )
    return {
        "terminal_image_full_score": full_score,
        "terminal_image_changed_region_score": changed_score,
        "terminal_image_preservation_score": preservation_score,
        **region_metrics,
        "terminal_image_correct": float(correct),
    }


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise ValueError("candidate video cannot be opened")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError("candidate video contains no readable frames")
    return frames, fps


def _dynamic_mask(stable: Sequence[np.ndarray]) -> np.ndarray:
    base = stable[0].astype(np.int16)
    mask = np.zeros(base.shape[:2], dtype=bool)
    for image in stable[1:]:
        mask |= np.max(np.abs(base - image.astype(np.int16)), axis=2) > 10
    if np.any(mask):
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    else:
        mask[:] = True
    return mask


def _stable_label(
    frame: np.ndarray,
    stable: Sequence[np.ndarray],
    mask: np.ndarray,
    *,
    dynamic_threshold: float = 0.055,
    full_threshold: float = 0.045,
) -> tuple[int, float, float]:
    resized = _resize_rgb(frame, stable[0].shape[:2]).astype(np.float32)
    errors = []
    full_errors = []
    for expected in stable:
        diff = visual_delta(resized, expected)
        errors.append(float(np.mean(diff[mask])) / 255.0)
        full_errors.append(float(np.mean(diff)) / 255.0)
    best = int(np.argmin(errors))
    if errors[best] <= dynamic_threshold and full_errors[best] <= full_threshold:
        return best, errors[best], full_errors[best]
    return -1, errors[best], full_errors[best]


def _runs(labels: Sequence[int]) -> list[tuple[int, int, int]]:
    if not labels:
        return []
    result: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            result.append((labels[start], start, index - 1))
            start = index
    return result


def _gui_press_mask(frame: np.ndarray) -> np.ndarray:
    """Locate the renderer's cyan contact ring with codec/style tolerance."""

    rgb = _rgb_array(frame).astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (
        (red < 100)
        & (green >= 105)
        & (green <= 220)
        & (blue >= 105)
        & (blue <= 220)
        & (np.abs(green - blue) <= 42)
        & (green - red >= 48)
        & (blue - red >= 48)
    )


def _gui_edge_recall(
    candidate: np.ndarray,
    expected: np.ndarray,
    panel_box: tuple[int, int, int, int],
) -> float:
    """Measure whether the complete old/new page remains structurally present."""

    left, top, right, bottom = panel_box
    candidate_crop = cv2.cvtColor(candidate[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
    expected_crop = cv2.cvtColor(expected[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
    candidate_edges = cv2.Canny(candidate_crop, 35, 105)
    expected_edges = cv2.Canny(expected_crop, 35, 105) > 0
    if not np.any(expected_edges):
        return 1.0
    candidate_nearby = cv2.dilate(
        (candidate_edges > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1,
    ).astype(bool)
    return float(np.mean(candidate_nearby[expected_edges]))


def _gui_navigation_process_metrics(
    frames: Sequence[np.ndarray],
    structured: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify source-action presses and atomic hierarchical page commits.

    No authored trace or saved reference video is read.  The complete state
    sequence and the visible target control for every CLICK/BACK are rebuilt
    from the source ``params`` and ``initial_state``.  Non-press frames must be
    complete stable pages in that exact sequence; press frames must preserve the
    old page and mark only the source action's control.  An empty-card skeleton,
    full-panel dissolve, extra press, skip, repeat, or direct hard cut therefore
    cannot be counted as process evidence.
    """

    try:
        params = structured.get("params")
        initial = structured.get("initial_state")
        if not isinstance(params, Mapping) or not isinstance(initial, Mapping):
            raise ValueError("GUI source contract lacks params and initial_state")
        trace = replay_case("gui_hierarchical_navigation", params, initial)
        if len(trace) < 2:
            raise ValueError("GUI source replay contains no actions")

        stable = [
            _rgb_array(render_state("gui_hierarchical_navigation", state, params))
            for state in trace
        ]
        shape = stable[0].shape[:2]
        arrays = [_resize_rgb(frame, shape) for frame in frames]
        raw_panel = gui_control_bboxes(trace[0], params)["PANEL"]
        scale_x = shape[1] / 512.0
        scale_y = shape[0] / 512.0
        panel_box = (
            max(0, int(np.floor(raw_panel[0] * scale_x))),
            max(0, int(np.floor(raw_panel[1] * scale_y))),
            min(shape[1], int(np.ceil(raw_panel[2] * scale_x))),
            min(shape[0], int(np.ceil(raw_panel[3] * scale_y))),
        )
        left, top, right, bottom = panel_box
        panel_mask = np.zeros(shape, dtype=bool)
        panel_mask[top:bottom, left:right] = True

        press_masks = [_gui_press_mask(frame) for frame in arrays]
        press_counts = [int(np.count_nonzero(mask & panel_mask)) for mask in press_masks]
        is_press = [count >= _GUI_MIN_PRESS_PIXELS for count in press_counts]
        press_runs = [
            (start, end)
            for label, start, end in _runs([int(value) for value in is_press])
            if label == 1
        ]

        stable_labels: list[int] = []
        for frame in arrays:
            label, _panel_error, _full_error = _stable_label(
                frame,
                stable,
                panel_mask,
                dynamic_threshold=_GUI_STABLE_PANEL_ERROR,
                full_threshold=_GUI_STABLE_PANEL_ERROR,
            )
            stable_labels.append(label)

        expected_transitions = len(trace) - 1
        press_targets_correct = len(press_runs) == expected_transitions
        press_pages_preserved = len(press_runs) == expected_transitions
        press_frame_counts: list[int] = []
        press_target_ratios: list[float] = []
        press_edge_recalls: list[float] = []
        press_progresses: list[list[float]] = []
        press_footprint_spreads: list[list[float]] = []
        press_footprint_supports: list[list[float]] = []
        press_linear_fit_errors: list[list[float]] = []
        press_dot_progresses: list[list[float]] = []
        press_dot_footprint_spreads: list[list[float]] = []
        press_dot_footprint_supports: list[list[float]] = []
        press_dot_linear_fit_errors: list[list[float]] = []
        expected_action_keys: list[str] = []
        press_complete_footprints = len(press_runs) == expected_transitions

        for index, (old, new) in enumerate(zip(trace, trace[1:])):
            action = new.get("last", {}).get("action", {})
            action_key = "BACK" if action.get("kind") == "BACK" else f"CLICK:{action.get('label')}"
            expected_action_keys.append(action_key)
            if index >= len(press_runs):
                press_targets_correct = False
                press_pages_preserved = False
                continue
            run_start, run_end = press_runs[index]
            press_frame_counts.append(run_end - run_start + 1)
            controls = gui_control_bboxes(old, params)
            target = controls.get(action_key)
            if target is None:
                press_targets_correct = False
                press_pages_preserved = False
                continue
            target_box = (
                max(0, int(np.floor(target[0] * scale_x))),
                max(0, int(np.floor(target[1] * scale_y))),
                min(shape[1], int(np.ceil(target[2] * scale_x))),
                min(shape[0], int(np.ceil(target[3] * scale_y))),
            )
            target_mask = np.zeros(shape, dtype=bool)
            target_mask[target_box[1]:target_box[3], target_box[0]:target_box[2]] = True
            full_press = _rgb_array(render_gui_action_press(
                old, new, params, progress=1.0,
            ))
            full_press = _resize_rgb(full_press, shape)
            # Mirror the renderer's source-derived contact-dot geometry.  It is
            # checked separately because a complete ring could otherwise hide
            # omission or late hard appearance of the much smaller dot.
            dot_x = (float(target[2]) - 11.0) * scale_x
            dot_y = (float(target[1]) + 11.0) * scale_y
            dot_box = (
                max(0, int(np.floor(dot_x - 7.0 * scale_x))),
                max(0, int(np.floor(dot_y - 7.0 * scale_y))),
                min(shape[1], int(np.ceil(dot_x + 7.0 * scale_x))),
                min(shape[0], int(np.ceil(dot_y + 7.0 * scale_y))),
            )
            run_ratios: list[float] = []
            run_recalls: list[float] = []
            run_observations: list[dict[str, float]] = []
            run_dot_observations: list[dict[str, float]] = []
            for frame_index in range(run_start, run_end + 1):
                mask = press_masks[frame_index] & panel_mask
                total = int(np.count_nonzero(mask))
                target_count = int(np.count_nonzero(mask & target_mask))
                run_ratios.append(target_count / max(1, total))
                run_recalls.append(_gui_edge_recall(arrays[frame_index], stable[index], panel_box))
                run_observations.append(_complete_footprint_projection(
                    arrays[frame_index], stable[index], full_press, box=target_box,
                ))
                run_dot_observations.append(_complete_footprint_projection(
                    arrays[frame_index], stable[index], full_press, box=dot_box,
                ))
            press_target_ratios.append(min(run_ratios, default=0.0))
            press_edge_recalls.append(min(run_recalls, default=0.0))
            progress = [item["progress"] for item in run_observations]
            dot_progress = [item["progress"] for item in run_dot_observations]
            press_progresses.append(progress)
            press_dot_progresses.append(dot_progress)
            press_footprint_spreads.append([
                item["progress_spread"] for item in run_observations
            ])
            press_footprint_supports.append([
                item["footprint_support"] for item in run_observations
            ])
            press_linear_fit_errors.append([
                item["linear_fit_error"] for item in run_observations
            ])
            press_dot_footprint_spreads.append([
                item["progress_spread"] for item in run_dot_observations
            ])
            press_dot_footprint_supports.append([
                item["footprint_support"] for item in run_dot_observations
            ])
            press_dot_linear_fit_errors.append([
                item["linear_fit_error"] for item in run_dot_observations
            ])
            partial_indices = [
                item_index
                for item_index, item in enumerate(run_observations)
                if 0.08 <= item["progress"] <= 0.94
            ]
            partial_levels = {
                round(run_observations[item_index]["progress"], 1)
                for item_index in partial_indices
            }
            monotonic = all(
                current + 0.08 >= previous
                for previous, current in zip(progress, progress[1:])
            ) and all(
                current + 0.10 >= previous
                for previous, current in zip(dot_progress, dot_progress[1:])
            )
            endpoints = bool(
                progress
                and dot_progress
                and progress[-1] >= 0.90
                and dot_progress[-1] >= 0.82
            )
            complete_partial = bool(
                len(partial_levels) >= 2
                and all(
                    run_observations[item_index]["progress_spread"]
                    <= _LOCAL_FADE_MAX_FOOTPRINT_PROGRESS_SPREAD
                    and run_observations[item_index]["footprint_support"]
                    >= _LOCAL_FADE_MIN_FOOTPRINT_SUPPORT
                    and run_observations[item_index]["linear_fit_error"]
                    <= _LOCAL_FADE_MAX_LINEAR_FIT_ERROR
                    and run_observations[item_index]["bounded_pixel_fraction"] >= 0.88
                    and run_dot_observations[item_index]["progress_spread"]
                    <= _GUI_DOT_MAX_FOOTPRINT_PROGRESS_SPREAD
                    and run_dot_observations[item_index]["footprint_support"]
                    >= _LOCAL_FADE_MIN_FOOTPRINT_SUPPORT
                    and run_dot_observations[item_index]["linear_fit_error"]
                    <= _LOCAL_FADE_MAX_LINEAR_FIT_ERROR
                    # The ~10px dot's 4x-supersampled anti-aliasing leaves a
                    # slightly larger share of its measured pixels outside the
                    # bounded progress band at faint opacities (measured 0.83
                    # on reference presses); 0.80 still rejects hard or
                    # off-footprint dot appearances.
                    and run_dot_observations[item_index]["bounded_pixel_fraction"] >= 0.80
                    for item_index in partial_indices
                )
            )
            press_complete_footprints = bool(
                press_complete_footprints
                and monotonic
                and endpoints
                and complete_partial
            )
            press_targets_correct = bool(
                press_targets_correct
                and run_end - run_start + 1 >= 2
                and min(run_ratios, default=0.0) >= 0.90
            )
            press_pages_preserved = bool(
                press_pages_preserved
                and min(run_recalls, default=0.0) >= _GUI_MIN_EDGE_RECALL
            )

        # Stable segments must be exactly S0, S1, ..., Sn around the n press
        # runs.  This rejects unknown skeleton/crossfade frames as well as any
        # skip, repeat, regression, or final-only/static substitution.
        atomic_sequence = len(press_runs) == expected_transitions
        stable_segment_lengths: list[int] = []
        unclassified_nonpress = 0
        wrong_stable_frames = 0
        if len(press_runs) == expected_transitions:
            for state_index in range(len(trace)):
                segment_start = 0 if state_index == 0 else press_runs[state_index - 1][1] + 1
                segment_end = len(arrays) - 1 if state_index == expected_transitions else press_runs[state_index][0] - 1
                segment_indices = list(range(segment_start, segment_end + 1))
                stable_segment_lengths.append(len(segment_indices))
                for frame_index in segment_indices:
                    # BACK can revisit a visually identical page. Match the
                    # expected page at this action position, not the first
                    # occurrence returned by a global nearest-state classifier.
                    label, _, _ = _stable_label(
                        arrays[frame_index], [stable[state_index]], panel_mask,
                        dynamic_threshold=_GUI_STABLE_PANEL_ERROR,
                        full_threshold=_GUI_STABLE_PANEL_ERROR,
                    )
                    if label < 0:
                        if stable_labels[frame_index] < 0:
                            unclassified_nonpress += 1
                        else:
                            wrong_stable_frames += 1
                    if (
                        label != 0
                        or _gui_edge_recall(arrays[frame_index], stable[state_index], panel_box)
                        < _GUI_MIN_EDGE_RECALL
                    ):
                        atomic_sequence = False
                if len(segment_indices) < 2:
                    atomic_sequence = False
        else:
            unclassified_nonpress = sum(
                not pressed and label < 0
                for pressed, label in zip(is_press, stable_labels)
            )
            wrong_stable_frames = sum(
                not pressed and label >= 0
                for pressed, label in zip(is_press, stable_labels)
            )

        process_correct = bool(
            press_targets_correct
            and press_pages_preserved
            and press_complete_footprints
            and atomic_sequence
        )
        return {
            "video_gui_source_replay_valid": 1.0,
            "video_gui_expected_action_keys": expected_action_keys,
            "video_gui_detected_press_runs": [list(run) for run in press_runs],
            "video_gui_press_frame_counts": press_frame_counts,
            "video_gui_press_target_ratios": press_target_ratios,
            "video_gui_press_targets_correct": float(press_targets_correct),
            "video_gui_press_page_edge_recalls": press_edge_recalls,
            "video_gui_complete_page_preserved_during_press": float(press_pages_preserved),
            "video_gui_press_complete_footprint_fade": float(press_complete_footprints),
            "video_gui_press_progresses": press_progresses,
            "video_gui_press_footprint_progress_spreads": press_footprint_spreads,
            "video_gui_press_footprint_supports": press_footprint_supports,
            "video_gui_press_linear_fit_errors": press_linear_fit_errors,
            "video_gui_press_dot_progresses": press_dot_progresses,
            "video_gui_press_dot_footprint_progress_spreads": press_dot_footprint_spreads,
            "video_gui_press_dot_footprint_supports": press_dot_footprint_supports,
            "video_gui_press_dot_linear_fit_errors": press_dot_linear_fit_errors,
            "video_local_fade_max_footprint_progress_spread": _LOCAL_FADE_MAX_FOOTPRINT_PROGRESS_SPREAD,
            "video_gui_dot_max_footprint_progress_spread": _GUI_DOT_MAX_FOOTPRINT_PROGRESS_SPREAD,
            "video_local_fade_min_footprint_support": _LOCAL_FADE_MIN_FOOTPRINT_SUPPORT,
            "video_local_fade_max_linear_fit_error": _LOCAL_FADE_MAX_LINEAR_FIT_ERROR,
            "video_gui_stable_segment_lengths": stable_segment_lengths,
            "video_gui_unclassified_nonpress_frames": int(unclassified_nonpress),
            "video_gui_wrong_or_regressed_stable_frames": int(wrong_stable_frames),
            "video_gui_atomic_commit_sequence_correct": float(atomic_sequence),
            "video_gui_action_process_correct": float(process_correct),
        }
    except Exception as exc:
        return {
            "video_gui_source_replay_valid": 0.0,
            "video_gui_process_error": str(exc),
            "video_gui_press_targets_correct": 0.0,
            "video_gui_complete_page_preserved_during_press": 0.0,
            "video_gui_press_complete_footprint_fade": 0.0,
            "video_gui_atomic_commit_sequence_correct": 0.0,
            "video_gui_action_process_correct": 0.0,
        }


def _discrete_sequence_metrics(
    frames: Sequence[np.ndarray],
    stable: Sequence[np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    """Read every frame as a complete discrete state, preserving free hold lengths."""
    shape = stable[0].shape[:2]
    arrays = [_resize_rgb(_rgb_array(frame), shape) for frame in frames]
    labels = [_stable_label(frame, stable, mask)[0] for frame in arrays]
    # Read flat interiors independently of the whole-region average. A color
    # halfway between two states must not be rounded to one of those states.
    core_masks = []
    for expected in stable:
        high = cv2.dilate(expected, np.ones((5, 5), np.uint8))
        low = cv2.erode(expected, np.ones((5, 5), np.uint8))
        core_masks.append(mask & (np.max(high.astype(float) - low, axis=2) <= 8))
    incomplete = []
    for index, (frame, label) in enumerate(zip(arrays, labels)):
        if label < 0:
            incomplete.append(index)
            continue
        core = core_masks[label]
        if core.any():
            changed = np.max(np.abs(frame.astype(float) - stable[label]), axis=2) > 16
            if np.count_nonzero(changed & core) > max(16, .02 * np.count_nonzero(core)):
                labels[index] = -1
                incomplete.append(index)
    compact, correct = ordered_state_observations(labels, len(stable))
    correct = bool(correct and not incomplete)
    runs = _runs(labels)
    return {
        "video_discrete_run_labels": compact,
        "video_discrete_hold_lengths": [end - start + 1 for label, start, end in runs if label >= 0],
        "video_discrete_ordered_exact": float(compact == list(range(len(stable)))),
        "video_discrete_unclassified_frames": sum(label < 0 for label in labels),
        "video_discrete_incomplete_frame_indices": incomplete,
        "video_discrete_sequence_correct": float(correct),
        "video_keyframe_indices": keyframe_indices(arrays),
    }


def _sandpile_numeric_metrics(frames, stable, structured):
    """Read each pile count; sparse wrong digits cannot average into a state."""
    from data.common.verifiers.numeric_glyphs import numeric_glyph_descriptor, read_numeric_glyph
    from data.next_state_prediction.core.rendering import stable_grid_layout

    params, initial = structured["params"], structured["initial_state"]
    trace = replay_case("sandpile_avalanche", params, initial)
    rows, cols = len(initial["grid"]), len(initial["grid"][0])
    left, top, width, height = stable_grid_layout("sandpile_avalanche", rows, cols)

    def crop(frame, row, col):
        x, y = left + (col + .5) * width, top + (row + .5) * height
        radius = min(24, int(min(width, height) * .32))
        return np.asarray(frame)[int(y)-radius:int(y)+radius+1, int(x)-radius:int(x)+radius+1]

    maximum = max(value for state in trace for row in state["grid"] for value in row)
    vocabulary = {}
    for value in range(max(9, maximum) + 1):
        state = {**initial, "grid": [[value] * cols for _ in range(rows)]}
        image = _rgb_array(render_state("sandpile_avalanche", state, params))
        descriptor = numeric_glyph_descriptor(crop(image, 0, 0))
        if descriptor is not None:
            vocabulary[value] = descriptor
    mask = _dynamic_mask(stable)
    labels, mismatches = [], []
    terminal = [[None] * cols for _ in range(rows)]
    for index, frame in enumerate(frames):
        label = _stable_label(frame, stable, mask)[0]
        expected = trace[label]["grid"] if label >= 0 else None
        valid = expected is not None
        for row in range(rows):
            for col in range(cols):
                observed, _similarity, _margin = read_numeric_glyph(crop(frame, row, col), vocabulary)
                if index == len(frames) - 1:
                    terminal[row][col] = observed
                if expected is not None and observed != expected[row][col]:
                    valid = False
                    if observed is not None:
                        mismatches.append({"frame": index, "row": row, "col": col,
                                           "observed": observed, "expected": expected[row][col]})
        labels.append(label if valid else -1)
    compact, ordered = ordered_state_observations(labels, len(trace))
    return {
        "video_sandpile_numeric_state_labels": compact,
        "video_sandpile_numeric_mismatches": mismatches[:30],
        "video_sandpile_terminal_grid": terminal,
        "video_sandpile_numeric_process_pass": bool(ordered and not mismatches),
        "video_sandpile_numeric_terminal_pass": terminal == trace[-1]["grid"],
    }


def _distance_wave_numeric_metrics(
    frames: Sequence[np.ndarray],
    stable: Sequence[np.ndarray],
    structured: Mapping[str, Any],
    *,
    task_name: str,
) -> dict[str, Any]:
    """Read committed distances independently of the surrounding cell colours."""
    from data.common.verifiers.numeric_glyphs import numeric_glyph_descriptor, read_numeric_glyph

    if task_name == "binary_matrix":
        from data.next_state_prediction.binary_matrix.generator import BinaryMatrixPattern, _binary_distance_states
        grid = [[int(value) for value in row] for row in structured["input_matrix"]]
        distances = _distance_grid(grid, source_value=0)
        states = _binary_distance_states(grid, distances)
        renderer = BinaryMatrixPattern(Path(tempfile.gettempdir()), seed=0, video=False)
        left, top, cell, _width = renderer._grid_geometry(len(grid))
    else:
        from data.next_state_prediction.grid_bfs.generator import GridBFSPattern, _bfs_visual_states
        grid = [[str(value) for value in row] for row in structured["input_grid"]]
        distances = _distance_grid(grid, source_value="1", wall_value="X")
        states = _bfs_visual_states(grid, distances)
        renderer = GridBFSPattern(Path(tempfile.gettempdir()), seed=0, video=False)
        left, top, cell, _width, _height = renderer._grid_geometry(len(grid), len(grid[0]))
    rows, cols = len(grid), len(grid[0])

    def crop(frame: np.ndarray, row: int, col: int) -> np.ndarray:
        cx, cy = (renderer._cell_center(rows,cols,row,col) if task_name=="grid_bfs" else (left+(col+.5)*cell,top+(row+.5)*cell))
        # The centered 18px value has no badge or neighbouring label in this box.
        radius = min(16, int(cell * .34))
        return frame[int(cy) - radius:int(cy) + radius + 1, int(cx) - radius:int(cx) + radius + 1]

    vocabulary = {}
    maximum = max(value for row in distances for value in row)
    blank = [["0"] * cols for _ in range(rows)]
    for value in range(-1, max(8, maximum) + 1):
        image = _rgb_array(renderer._render_state(blank, [[value] * cols for _ in range(rows)], phase="final"))
        descriptor = numeric_glyph_descriptor(crop(image, 0, 0), light=True)
        if descriptor is not None:
            vocabulary[value] = descriptor
    mask = _dynamic_mask(stable)
    labels = []
    mismatches = []
    terminal_observed = [[None] * cols for _ in range(rows)]
    terminal_ok = False
    for frame_index, frame in enumerate(frames):
        label = _stable_label(frame, stable, mask)[0]
        if label < 0:
            labels.append(-1)
            continue
        phase, _layer, categories = states[label]
        valid = True
        for row in range(rows):
            for col in range(cols):
                category = categories[row][col]
                required = category in {"frontier", "settled", "unreachable", "wall_final"}
                if not required:
                    continue
                value, similarity, margin = read_numeric_glyph(
                    crop(frame, row, col), vocabulary,
                    light=task_name == "binary_matrix" or category != "frontier",
                )
                if frame_index == len(frames) - 1:
                    terminal_observed[row][col] = value
                if value != distances[row][col]:
                    valid = False
                    if value is not None:
                        mismatches.append({"frame": frame_index, "row": row, "col": col,
                                           "observed": value, "expected": distances[row][col]})
        labels.append(label if valid else -1)
        if frame_index == len(frames) - 1:
            terminal_ok = bool(valid and phase == "final")
    compact, process_ok = ordered_state_observations(labels, len(stable))
    # A confidently read wrong integer in a committed wave is evidence of an
    # illegal state, even when the same hold contains correct frames around it.
    # Unreadable transition glyphs retain the ordinary ambiguity allowance.
    process_ok = bool(process_ok and not mismatches)
    return {
        f"video_{task_name}_numeric_state_labels": compact,
        f"video_{task_name}_numeric_mismatches": mismatches[:30],
        f"video_{task_name}_terminal_distances": terminal_observed,
        f"video_{task_name}_numeric_process_pass": bool(process_ok),
        f"video_{task_name}_numeric_terminal_pass": terminal_ok,
    }


def _verify_candidate_video(
    path: Optional[Path],
    replay: _Replay,
    *,
    task_name: str,
    structured: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if path is None:
        return {
            "whole_video_loaded": False,
            "whole_video_error": "candidate video is missing",
            "whole_video_process_correct": 0.0,
        }
    try:
        frames, fps = _read_video(path)
        stable = [_rgb_array(image) for image in replay.stable_images]
        shape = stable[0].shape[:2]
        stable = [_resize_rgb(image, shape) for image in stable]
        mask = _dynamic_mask(stable)

        terminal_metrics = _terminal_image_metrics(frames[-1], stable[0], stable[-1])
        static_mask = ~mask
        # A shared metric must not veto a legal process
        # it cannot observe.  The GUI press cue is a source-rendered contact
        # ring that appears in no stable page, so it lands outside the
        # stable-state dynamic mask; the control panel is owned end to end by
        # _gui_navigation_process_metrics (ring geometry, target, progress), so
        # the shared per-region gate hands that box back to it.  The frame-wide
        # static mean keeps its original coverage.
        region_static_mask = static_mask
        if task_name == "gui_hierarchical_navigation":
            params = (structured or {}).get("params") or {}
            initial = (structured or {}).get("initial_state")
            if initial is not None:
                left, top, right, bottom = gui_control_bboxes(initial, params)["PANEL"]
                scale_x, scale_y = shape[1] / 512.0, shape[0] / 512.0
                region_static_mask = static_mask.copy()
                region_static_mask[
                    max(0, int(np.floor(top * scale_y))):min(shape[0], int(np.ceil(bottom * scale_y))),
                    max(0, int(np.floor(left * scale_x))):min(shape[1], int(np.ceil(right * scale_x))),
                ] = False

        max_static_region_error = 0.0
        if np.any(static_mask):
            static_errors = []
            initial_float = stable[0].astype(np.float32)
            for frame_index in keyframe_indices(frames):
                resized = _resize_rgb(frames[frame_index], shape).astype(np.float32)
                static_errors.append(float(np.mean(visual_delta(resized, initial_float)[static_mask])) / 255.0)
                # A mean over the whole static mask dilutes a painted patch to
                # nothing; the same statistic per region does not.
                max_static_region_error = max(
                    max_static_region_error,
                    static_region_errors(resized, initial_float, region_static_mask)[0],
                )
            static_preservation_score = max(0.0, 1.0 - max(static_errors))
        else:
            static_preservation_score = 1.0
        static_preserved = (
            static_preservation_score >= 0.94
            and max_static_region_error <= STATIC_REGION_FLOOR
        )
        duration = len(frames) / fps if fps > 0 else None
        dimensions_valid = all(frame.shape[:2] == (512, 512) for frame in frames)
        fps_valid = abs(fps - 16.0) <= 0.2
        frame_count_valid = len(frames) >= 5 and (len(frames) - 1) % 4 == 0
        media_valid = bool(
            dimensions_valid
            and fps_valid
            and frame_count_valid
            and duration is not None
            and duration > 0
        )

        task_metrics: dict[str, Any] = {}
        if task_name == "gui_hierarchical_navigation":
            task_metrics = _gui_navigation_process_metrics(frames, structured or {})
            process_correct = bool(task_metrics["video_gui_action_process_correct"])
            failure_reason = (
                None
                if process_correct
                else "GUI transition lacks the exact source-action presses and complete atomic page commits"
            )
        else:
            task_metrics = _discrete_sequence_metrics(frames, stable, mask)
            process_correct = bool(task_metrics["video_discrete_sequence_correct"])
            failure_reason = (
                None
                if process_correct
                else "key states are missing, unreadable, or out of order"
            )

        if task_name in {"grid_bfs", "binary_matrix"}:
            numeric_metrics = _distance_wave_numeric_metrics(
                frames, stable, structured or {}, task_name=task_name,
            )
            task_metrics.update(numeric_metrics)
            numeric_process_ok = numeric_metrics[f"video_{task_name}_numeric_process_pass"]
            process_correct = bool(process_correct and numeric_process_ok)
            terminal_metrics["terminal_image_correct"] = float(
                bool(terminal_metrics["terminal_image_correct"] and numeric_metrics[f"video_{task_name}_numeric_terminal_pass"])
            )
            if not numeric_process_ok:
                failure_reason = "printed distances are missing, incorrect, or inconsistent with the ordered waves"

        if task_name == "protocol_handshake_tick":
            from data.next_state_prediction.core.protocol_observation import protocol_clock_metrics
            clock_metrics = protocol_clock_metrics(frames, structured or {})
            task_metrics.update(clock_metrics)
            process_correct &= clock_metrics["video_protocol_clock_process_pass"]
            terminal_metrics["terminal_image_correct"] *= clock_metrics["video_protocol_clock_terminal_pass"]
            if not clock_metrics["video_protocol_clock_process_pass"]:
                failure_reason = "TIMEOUT/TICK fields are incorrect or not committed with the ordered protocol states"

        if task_name == "sandpile_avalanche":
            numeric_metrics = _sandpile_numeric_metrics(frames, stable, structured or {})
            task_metrics.update(numeric_metrics)
            process_correct &= numeric_metrics["video_sandpile_numeric_process_pass"]
            terminal_metrics["terminal_image_correct"] *= numeric_metrics["video_sandpile_numeric_terminal_pass"]
            if not numeric_metrics["video_sandpile_numeric_process_pass"]:
                failure_reason = "pile counts are missing, incorrect, or inconsistent with the synchronous toppling states"

        correct = bool(
            process_correct
            and static_preserved
            and media_valid
            and terminal_metrics["terminal_image_correct"] == 1.0
        )
        return {
            "whole_video_loaded": True,
            "whole_video_error": None,
            "video_frame_count": len(frames),
            "video_fps": fps,
            "video_duration_seconds": duration,
            "video_dimensions_valid": float(dimensions_valid),
            "video_fps_valid": float(fps_valid),
            "video_frame_count_valid": float(frame_count_valid),
            "video_media_valid": float(media_valid),
            "whole_video_failure_reason": failure_reason,
            "video_static_preservation_score": static_preservation_score,
            "video_static_region_max_error": max_static_region_error,
            "video_static_region_bound": STATIC_REGION_FLOOR,
            "video_static_preserved": float(static_preserved),
            "video_terminal_state_correct": terminal_metrics["terminal_image_correct"],
            "whole_video_process_correct": float(correct),
            **task_metrics,
        }
    except Exception as exc:
        return {
            "whole_video_loaded": False,
            "whole_video_error": str(exc),
            "whole_video_process_correct": 0.0,
        }


def verify_next_state(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: str = "ti2v",
) -> list[EvalRecord]:
    """Verify complete candidate videos against independent source replay."""
    selected = str(modality).strip().lower()
    if selected != "ti2v":
        raise ValueError(f"unsupported next-state modality: {modality!r}")

    records: list[EvalRecord] = []
    for sample in samples:
        video_path = resolve_candidate_video(pred_root, sample)
        try:
            replay = _independent_replay(sample)
            replay_error = None
            video_metrics = _verify_candidate_video(
                video_path, replay, task_name=sample.task_name,
                structured=_source_contract(sample),
            )
        except Exception as exc:
            replay = _Replay(None, tuple(), "invalid")
            replay_error = str(exc)
            video_metrics = {
                "whole_video_loaded": False,
                "whole_video_error": replay_error,
                "whole_video_process_correct": 0.0,
            }
        video_metrics["video_replay_variant"] = replay.source_variant
        correct = bool(video_path is not None and video_metrics["whole_video_process_correct"])
        error = replay_error
        if error is None and video_path is None:
            error = "TI2V candidate video is missing"
        records.append(EvalRecord(
            sample_id=sample.id,
            task_family=sample.task_family,
            task_name=sample.task_name,
            split=sample.split,
            input_asset=sample.primary_input_image,
            prediction_asset=video_path.as_posix() if video_path is not None else None,
            offline_metrics={
                **video_metrics,
                "selected_modality": selected,
                "source_replay_variant": replay.source_variant,
                "source_replay_error": replay_error,
                "binary_correct": int(correct),
            },
            offline_pass=correct,
            error=error,
        ))
    return records


__all__ = ["verify_next_state"]
