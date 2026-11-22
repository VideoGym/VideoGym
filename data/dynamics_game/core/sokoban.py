"""Source-only Sokoban verification for the Dynamics Game family."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.video_encoding import validate_video_contract
from data.dynamics_game.sokoban.generator import SokobanPlanGenerator


Cell = Tuple[int, int]
State = Tuple[Cell, Cell]
_DIR_TO_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


def _record_for(sample: CanonicalSample) -> EvalRecord:
    return EvalRecord(
        sample_id=sample.id,
        task_name=sample.task_name,
        input_asset=sample.primary_input_image,
        task_family=sample.task_family,
        split=sample.split,
    )


def _structured(sample: CanonicalSample) -> Dict[str, Any]:
    raw = sample.semantic_input if isinstance(sample.semantic_input, dict) else {}
    value = raw.get("semantic_state") or raw.get("structured_answer") or raw
    return value if isinstance(value, dict) else {}


def _grid(value: Any) -> Optional[List[List[int]]]:
    if not isinstance(value, list) or not value:
        return None
    try:
        result = [[int(cell) for cell in row] for row in value]
    except (TypeError, ValueError):
        return None
    width = len(result[0]) if result else 0
    if width <= 0 or any(len(row) != width for row in result):
        return None
    return result


def _find_single(grid: Sequence[Sequence[int]], values: Sequence[int]) -> Optional[Cell]:
    matches = [
        (row, col)
        for row, cells in enumerate(grid)
        for col, value in enumerate(cells)
        if int(value) in values
    ]
    return matches[0] if len(matches) == 1 else None


def _source_state(structured: Dict[str, Any]) -> tuple[List[List[int]], Cell, Cell, Cell, set[Cell]]:
    initial = _grid(structured.get("initial_grid"))
    if initial is None:
        raise ValueError("Sokoban source is missing a rectangular initial_grid")
    player = _find_single(initial, (3,))
    box = _find_single(initial, (2, 5))
    target = _find_single(initial, (4,))
    if player is None or box is None or target is None:
        raise ValueError("Sokoban source must contain one player, box, and target")
    walls = {
        (row, col)
        for row, cells in enumerate(initial)
        for col, value in enumerate(cells)
        if int(value) == 9
    }
    return initial, player, box, target, walls


def _advance(
    state: State,
    move: str,
    *,
    rows: int,
    cols: int,
    walls: set[Cell],
) -> Optional[State]:
    delta = _DIR_TO_DELTA.get(str(move).lower())
    if delta is None:
        return None
    player, box = state
    next_player = (player[0] + delta[0], player[1] + delta[1])
    if not (0 <= next_player[0] < rows and 0 <= next_player[1] < cols) or next_player in walls:
        return None
    next_box = box
    if next_player == box:
        next_box = (box[0] + delta[0], box[1] + delta[1])
        if (
            not (0 <= next_box[0] < rows and 0 <= next_box[1] < cols)
            or next_box in walls
        ):
            return None
    return next_player, next_box


def _shortest_solution(
    initial: State,
    target: Cell,
    *,
    rows: int,
    cols: int,
    walls: set[Cell],
) -> Optional[int]:
    queue = deque([(initial, 0)])
    seen = {initial}
    while queue:
        state, distance = queue.popleft()
        if state[1] == target:
            return distance
        for move in _DIR_TO_DELTA:
            next_state = _advance(state, move, rows=rows, cols=cols, walls=walls)
            if next_state is not None and next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, distance + 1))
    return None


def _renderer(height: int, width: int) -> SokobanPlanGenerator:
    generator = SokobanPlanGenerator.__new__(SokobanPlanGenerator)
    generator.height = height
    generator.width = width
    return generator


def _render_terminal(structured: Dict[str, Any], plan: Dict[str, Any]) -> np.ndarray:
    initial_grid = _grid(structured.get("initial_grid")) or []
    board = [["#" if int(value) == 9 else "." for value in row] for row in initial_grid]
    generator = _renderer(len(initial_grid), len(initial_grid[0]))
    image = generator._render_board(
        board,
        tuple(plan["final_player"]),
        tuple(plan["final_box"]),
        tuple(plan["target"]),
    )
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _image_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 1.0
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) / 255.0)


def _video_frames(path: Path) -> tuple[List[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, fps


def _token_mask(frame: np.ndarray, token: str) -> np.ndarray:
    # Generated tokens retain their source hue while their saturation and
    # brightness vary. An RGB ball loses an otherwise complete yellow disk
    # or crate when just one channel shifts. The source palette separates
    # both bodies from the pale floor, green target, and slate walls in HSV.
    # Cargo docks have brown fixed bridges (S=79) and platform outlines
    # (S=95); the crate fill has S=137. Keep the fixed palette below the
    # saturation boundary for both tokens, including during color shifts.
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    if token == "player":
        return cv2.inRange(hsv, (18, 100, 145), (40, 255, 255))
    return cv2.inRange(hsv, (5, 100, 35), (22, 255, 235))


def _token_centers(
    frame: np.ndarray,
    token: str,
    min_area: int = 18,
) -> List[tuple[float, float, int]]:
    mask = _token_mask(frame, token)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [
        (float(centroids[index][0]), float(centroids[index][1]), int(stats[index, cv2.CC_STAT_AREA]))
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area
        and stats[index, cv2.CC_STAT_AREA] >= 0.45 * stats[index, cv2.CC_STAT_WIDTH] * stats[index, cv2.CC_STAT_HEIGHT]
    ]


def _scene_evidence(
    frame: np.ndarray, expected: np.ndarray, state: State, *, cell: float, margin: float,
) -> tuple[float, bool]:
    """Check fixed scenery and each token's source-derived footprint separately."""
    from data.common.verifiers.region_gates import static_region_errors

    static = np.ones(frame.shape[:2], dtype=bool)
    tokens_preserved = True
    tolerance = np.ones((5, 5), dtype=np.uint8)
    for (token, inset), (row, col) in zip((("player", 20), ("box", 14)), state):
        cx, cy = margin + (col + 0.5) * cell, margin + (row + 0.5) * cell
        half = (cell - inset) / 2 + 1
        body = np.zeros(frame.shape[:2], dtype=np.uint8)
        y0, y1 = max(0, int(cy - half)), min(frame.shape[0], int(cy + half) + 1)
        x0, x1 = max(0, int(cx - half)), min(frame.shape[1], int(cx + half) + 1)
        body[y0:y1, x0:x1] = _token_mask(expected, token)[y0:y1, x0:x1] > 0
        region = cv2.dilate(body, tolerance).astype(bool)
        observed = (_token_mask(frame, token) > 0) & region
        expected_area = int(body.sum())
        covered = cv2.dilate(observed.astype(np.uint8), tolerance).astype(bool)
        coverage = float((covered & (body > 0)).sum()) / max(1, expected_area)
        area_ratio = float(observed.sum()) / max(1, expected_area)
        tokens_preserved &= bool(expected_area and coverage >= 0.75 and 0.65 <= area_ratio <= 1.35)
        static &= ~region
    error = static_region_errors(frame.astype(np.float32), expected.astype(np.float32), static)[0]
    return error, tokens_preserved


def _cell_for_center(
    center: tuple[float, float],
    *,
    cell: float,
    margin: float,
    size: int,
) -> Optional[Cell]:
    col = int(round((center[0] - margin) / cell - 0.5))
    row = int(round((center[1] - margin) / cell - 0.5))
    return (row, col) if 0 <= row < size and 0 <= col < size else None


def _center_offset(center: tuple[float, float], *, cell: float, margin: float) -> float:
    """Distance (in cell units) from the nearest cell center."""

    col = (center[0] - margin) / cell - 0.5
    row = (center[1] - margin) / cell - 0.5
    return max(abs(col - round(col)), abs(row - round(row)))


def _track(
    frames: Sequence[np.ndarray],
    structured: Dict[str, Any],
) -> tuple[List[Optional[Cell]], List[Optional[Cell]], List[bool], List[float]]:
    initial, player, box, _target, _walls = _source_state(structured)
    size = max(len(initial), len(initial[0]))
    generator = _renderer(len(initial), len(initial[0]))
    cell = float(generator.cell_size)
    margin = float(generator.margin)
    starts = {
        "player": (margin + (player[1] + 0.5) * cell, margin + (player[0] + 0.5) * cell),
        "box": (margin + (box[1] + 0.5) * cell, margin + (box[0] + 0.5) * cell),
    }
    tracks: Dict[str, List[Optional[Cell]]] = {"player": [], "box": []}
    detections: List[bool] = []
    offsets: List[float] = []
    centers: Dict[str, Optional[tuple[float, float]]] = dict(starts)
    for frame in frames:
        frame_ok = True
        frame_offset = 0.0
        for name in ("player", "box"):
            # Use the source renderer's cell scale, not the training-grid pixel area.
            body_extent = cell - (20 if name == "player" else 14)
            candidates = _token_centers(frame, name, min_area=max(12, int(0.35 * body_extent**2)))
            previous = centers[name]
            if not candidates:
                selected = None
            else:
                selected = min(
                    candidates,
                    key=lambda item: (
                        (item[0] - previous[0]) ** 2 + (item[1] - previous[1]) ** 2
                        if previous is not None
                        else -item[2],
                        -item[2],
                    ),
                )
            if selected is None:
                tracks[name].append(None)
                frame_ok = False
                continue
            centers[name] = (selected[0], selected[1])
            tracks[name].append(
                _cell_for_center((selected[0], selected[1]), cell=cell, margin=margin, size=size)
            )
            frame_offset = max(
                frame_offset,
                _center_offset((selected[0], selected[1]), cell=cell, margin=margin),
            )
            frame_ok = frame_ok and tracks[name][-1] is not None
        detections.append(frame_ok)
        offsets.append(frame_offset)
    return tracks["player"], tracks["box"], detections, offsets


def _compress_states(
    player_track: Sequence[Optional[Cell]],
    box_track: Sequence[Optional[Cell]],
) -> List[State]:
    states: List[State] = []
    for player, box in zip(player_track, box_track):
        if player is None or box is None:
            continue
        state = (player, box)
        if states and states[-1] == state:
            continue
        # Preserve every observed complete state. Dropping an illegal crate
        # displacement or overlap here can turn an invalid program into a
        # legal one before the rule replay sees it.
        states.append(state)
    return states


def _verify_video(
    sample: CanonicalSample,
    path: Path,
    structured: Dict[str, Any],
) -> tuple[Dict[str, Any], bool]:
    frames, fps = _video_frames(path)
    try:
        contract = validate_video_contract(path)
        media_valid = True
    except Exception as exc:
        contract = {"error": str(exc), "fps": fps, "frame_count": len(frames)}
        media_valid = False
    if not frames:
        return {"video_media_valid": False, "video_process_reason": "no decodable frames"}, False

    source = _read_rgb(Path(sample.primary_input_image).expanduser().resolve())
    adjacent = [_image_distance(previous, current) for previous, current in zip(frames, frames[1:])]
    initial_distance = _image_distance(frames[0], source)
    final_hold = max(2, min(6, len(frames)))
    final_hold_distance = max(
        (_image_distance(frames[-1], frame) for frame in frames[-final_hold:]),
        default=1.0,
    )
    temporal = bool(
        initial_distance <= 0.035
        and sum(value > 0.00015 for value in adjacent) >= 2
        and final_hold_distance <= 0.012
        and max(adjacent, default=1.0) <= 0.24
    )
    player_track, box_track, detections, center_offsets = _track(frames, structured)
    observed_states = _compress_states(player_track, box_track)
    initial_grid, player, box, target, walls = _source_state(structured)
    rows, cols = len(initial_grid), len(initial_grid[0])
    replay_ok = bool(observed_states and observed_states[0] == (player, box))
    state = (player, box)
    for observed in observed_states[1:]:
        if not replay_ok:
            break
        if observed != _advance(state, _move_between(state[0], observed[0]), rows=rows, cols=cols, walls=walls):
            replay_ok = False
            break
        state = observed
    shortest = _shortest_solution((player, box), target, rows=rows, cols=cols, walls=walls)
    path_length = max(0, len(observed_states) - 1)
    # Discrete transition model: every frame must show both
    # tokens centered on exact cells; interpolated gliding between cells is
    # rejected.
    discrete_centering = bool(
        center_offsets and all(offset <= 0.25 for offset in center_offsets)
    )
    semantic = bool(
        replay_ok
        and state[1] == target
        and shortest is not None
        and path_length == shortest
        and all(detections)
        and discrete_centering
    )
    # Read the final answer independently of where a failed process replay
    # stopped. An unchanged crate is not a solved terminal arrangement.
    terminal_player = player_track[-1]
    terminal_box = box_track[-1]
    terminal_goal = bool(
        terminal_player is not None
        and terminal_player not in walls
        and terminal_player != target
        and terminal_box == target
        and detections[-1]
    )
    final_distance = None
    if terminal_goal:
        expected_final = _render_terminal(
            structured,
            {"final_player": terminal_player, "final_box": target, "target": target},
        )
        final_distance = _image_distance(frames[-1], expected_final)
    final_state = bool(terminal_goal and final_distance <= 0.095)
    from data.common.verifiers.region_gates import STATIC_REGION_FLOOR
    rendered = {}
    scene_error = 0.0
    token_shapes_preserved = True
    generator = _renderer(rows, cols)
    for frame, p, b in zip(frames,player_track,box_track):
        if p is None or b is None:
            scene_error = 1.0
            token_shapes_preserved = False
            break
        key = (p,b)
        if key not in rendered:
            rendered[key] = _render_terminal(structured,{"final_player":p,"final_box":b,"target":target})
        static_error, token_shapes = _scene_evidence(
            frame, rendered[key], key, cell=generator.cell_size, margin=generator.margin,
        )
        scene_error = max(scene_error, static_error)
        token_shapes_preserved &= token_shapes
    scene_preserved = scene_error <= STATIC_REGION_FLOOR
    passed = bool(media_valid and temporal and semantic and scene_preserved and token_shapes_preserved and final_state)
    metrics = {
        "video_semantic_verifier": "sokoban_source_replayed_whole_video_v2",
        "video_verifier_used_reference_media": False,
        "video_media_valid": media_valid,
        "video_contract": contract,
        "video_fps": fps,
        "video_frame_count": len(frames),
        "video_starts_at_source_input": initial_distance <= 0.035,
        "video_initial_source_distance": initial_distance,
        "video_final_hold_distance": final_hold_distance,
        "video_static_region_max_error": scene_error,
        "video_scene_preserved": scene_preserved,
        "video_token_shapes_preserved": bool(token_shapes_preserved),
        "video_continuous_process_pass": temporal,
        "video_all_token_detections": all(detections),
        "video_discrete_centering_pass": discrete_centering,
        "video_max_center_offset": max(center_offsets) if center_offsets else None,
        "video_observed_state_count": len(observed_states),
        "video_recomputed_shortest_length": shortest,
        "video_observed_path_length": path_length,
        "video_source_replay_pass": semantic,
        "video_source_terminal_distance": final_distance,
        "video_terminal_box_on_target": terminal_goal,
        "video_final_state_pass": final_state,
        "video_whole_process_pass": passed,
    }
    return metrics, passed


def _move_between(previous: Cell, current: Cell) -> str:
    delta = (current[0] - previous[0], current[1] - previous[1])
    for move, expected in _DIR_TO_DELTA.items():
        if delta == expected:
            return move
    return ""


def verify_sokoban_sample(
    sample: CanonicalSample,
    pred_root: Path,
    *,
    modality: str,
) -> EvalRecord:
    """Verify one Sokoban sample's declared final-delivery artifact."""

    record = _record_for(sample)
    modality = str(modality or "").lower()
    try:
        structured = _structured(sample)
        video_path = resolve_candidate_video(pred_root, sample.id, sample.split)
        if modality == "ti2v":
            if video_path is None:
                raise FileNotFoundError("Missing TI2V output video")
            record.prediction_asset = video_path.as_posix()
            video_metrics, passed = _verify_video(sample, video_path, structured)
            record.offline_metrics.update(video_metrics)
        else:
            raise ValueError(f"Unsupported Sokoban modality: {modality!r}")
        record.offline_pass = bool(passed)
        record.offline_metrics.update({
            "verifier_profile": "dynamics_game_sokoban_source_replay_v2",
            "verifier_source_domain": "semantic_input_and_candidate_only",
            "requested_modality": modality,
            "binary_correct": bool(passed),
            "failure_reason": None if passed else "Sokoban source replay or formal artifact check failed",
        })
    except Exception as exc:
        record.offline_pass = False
        record.error = str(exc)
        record.offline_metrics.update({
            "verifier_profile": "dynamics_game_sokoban_source_replay_v2",
            "verifier_source_domain": "semantic_input_and_candidate_only",
            "requested_modality": modality,
            "binary_correct": False,
            "failure_reason": str(exc),
        })
    return record


__all__ = ["verify_sokoban_sample"]
