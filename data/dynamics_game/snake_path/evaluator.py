"""Task-local source-replay verifier for the multi-food snake task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.region_gates import (
    REGION_ERROR_CEILING,
    region_error_map,
)
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.verifiers.routes import VerifierRoute
from data.common.video_encoding import validate_video_contract
from data.dynamics_game.snake_path.generator import (
    BODY_FILL,
    FOOD_FILL,
    HEAD_FILL,
    board_geometry,
    render_snake_board,
)


Cell = Tuple[int, int]
VERIFIER_PROFILE = "dynamics_game_snake_path_source_replay_v1"
TASK_NAME = "snake_path"
_DIR_TO_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
_DELTA_TO_DIR = {delta: move for move, delta in _DIR_TO_DELTA.items()}


@dataclass(frozen=True)
class SourceState:
    rows: int
    cols: int
    initial_snake: Tuple[Cell, ...]
    foods: FrozenSet[Cell]


@dataclass(frozen=True)
class ReplayState:
    snake: Tuple[Cell, ...]
    foods: FrozenSet[Cell]


@dataclass(frozen=True)
class DecodedState:
    head: Cell
    body: FrozenSet[Cell]
    foods: FrozenSet[Cell]


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
    if isinstance(value, dict) and isinstance(value.get("semantic_state"), dict):
        value = value["semantic_state"]
    return value if isinstance(value, dict) else {}


def _grid(value: Any) -> Optional[List[List[int]]]:
    if not isinstance(value, list) or not value:
        return None
    try:
        grid = [[int(cell) for cell in row] for row in value]
    except (TypeError, ValueError):
        return None
    width = len(grid[0]) if grid else 0
    if width <= 0 or any(len(row) != width for row in grid):
        return None
    return grid


def _cells(value: Any) -> Optional[List[Cell]]:
    if not isinstance(value, list):
        return None
    result: List[Cell] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            result.append((int(item[0]), int(item[1])))
        except (TypeError, ValueError):
            return None
    return result


def _source_state(structured: Dict[str, Any]) -> SourceState:
    initial_grid = _grid(structured.get("initial_grid"))
    initial_snake = _cells(structured.get("initial_snake"))
    foods = _cells(structured.get("foods"))
    if initial_grid is None or not initial_snake or foods is None:
        raise ValueError("snake_path source requires initial_grid, ordered initial_snake, and foods")
    rows = len(initial_grid)
    cols = len(initial_grid[0])
    if rows < 1 or cols < 1:
        raise ValueError("snake_path source grid is empty")
    if len(set(initial_snake)) != len(initial_snake):
        raise ValueError("snake_path source snake contains duplicate cells")
    if len(set(foods)) != len(foods) or len(foods) < 2:
        raise ValueError("snake_path source must contain at least two distinct food pellets")
    if set(initial_snake) & set(foods):
        raise ValueError("snake_path source food overlaps the initial snake")
    all_cells = set(initial_snake) | set(foods)
    if any(not (0 <= row < rows and 0 <= col < cols) for row, col in all_cells):
        raise ValueError("snake_path source contains an out-of-bounds cell")
    if any(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1
        for first, second in zip(initial_snake, initial_snake[1:])
    ):
        raise ValueError("snake_path source snake order is not contiguous")

    head_cells = {
        (row, col)
        for row, values in enumerate(initial_grid)
        for col, value in enumerate(values)
        if value == 3
    }
    body_cells = {
        (row, col)
        for row, values in enumerate(initial_grid)
        for col, value in enumerate(values)
        if value == 2
    }
    food_cells = {
        (row, col)
        for row, values in enumerate(initial_grid)
        for col, value in enumerate(values)
        if value == 1
    }
    if (
        head_cells != {initial_snake[0]}
        or body_cells != set(initial_snake[1:])
        or food_cells != set(foods)
        or any(value not in {0, 1, 2, 3} for row in initial_grid for value in row)
    ):
        raise ValueError("snake_path initial_grid disagrees with the visible ordered source objects")
    return SourceState(
        rows=rows,
        cols=cols,
        initial_snake=tuple(initial_snake),
        foods=frozenset(foods),
    )


def _advance(
    state: ReplayState,
    move: str,
    *,
    rows: int,
    cols: int,
) -> Optional[ReplayState]:
    delta = _DIR_TO_DELTA.get(str(move).lower())
    if delta is None:
        return None
    new_head = (state.snake[0][0] + delta[0], state.snake[0][1] + delta[1])
    grow = new_head in state.foods
    occupied = set(state.snake if grow else state.snake[:-1])
    if (
        not (0 <= new_head[0] < rows and 0 <= new_head[1] < cols)
        or new_head in occupied
    ):
        return None
    if grow:
        return ReplayState(
            snake=(new_head,) + state.snake,
            foods=frozenset(cell for cell in state.foods if cell != new_head),
        )
    return ReplayState(snake=(new_head,) + state.snake[:-1], foods=state.foods)


def _moves_from_path(source: SourceState, path: Sequence[Cell]) -> Optional[List[str]]:
    """Derive internal moves from one strict full head-position path."""

    if not path or tuple(path[0]) != source.initial_snake[0]:
        return None
    moves: List[str] = []
    for previous, current in zip(path, path[1:]):
        move = _DELTA_TO_DIR.get(
            (int(current[0]) - int(previous[0]), int(current[1]) - int(previous[1]))
        )
        if move is None:
            return None
        moves.append(move)
    return moves


def _replay_moves(
    source: SourceState,
    moves: Sequence[str],
) -> Tuple[Dict[str, Any], bool, List[ReplayState]]:
    state = ReplayState(source.initial_snake, source.foods)
    trace = [state]
    legal = True
    stopped_after_completion = True
    for move in moves:
        if not state.foods:
            legal = False
            stopped_after_completion = False
            break
        next_state = _advance(state, move, rows=source.rows, cols=source.cols)
        if next_state is None:
            legal = False
            break
        state = next_state
        trace.append(state)
    consumed = len(source.foods) - len(state.foods)
    solved = bool(
        legal
        and stopped_after_completion
        and not state.foods
        and len(state.snake) == len(source.initial_snake) + len(source.foods)
    )
    return {
        "text_parse_score": 1.0,
        "text_replay_legal": legal,
        "text_stopped_when_all_food_was_eaten": stopped_after_completion,
        "text_initial_food_count": len(source.foods),
        "text_consumed_food_count": consumed,
        "text_remaining_food_count": len(state.foods),
        "text_initial_snake_length": len(source.initial_snake),
        "text_final_snake_length": len(state.snake),
        "text_predicted_move_count": len(moves),
        "text_optimality_required": False,
        "text_replay_check_pass": solved,
        "text_answer_score": float(solved),
    }, solved, trace


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _colour_fraction(patch: np.ndarray, colour: Tuple[int, int, int], tolerance: float = 68.0) -> float:
    if patch.size == 0:
        return 0.0
    delta = patch.astype(np.float32) - np.asarray(colour, dtype=np.float32)
    return float(np.mean(np.linalg.norm(delta, axis=2) <= tolerance))


def _decode_frame(rgb: np.ndarray, source: SourceState) -> Optional[DecodedState]:
    if rgb.shape != (512, 512, 3):
        return None
    cell, margin_x, margin_y = board_geometry(source.cols, source.rows)
    heads: List[Cell] = []
    body: set[Cell] = set()
    foods: set[Cell] = set()
    inset = max(5, int(cell * 0.18))
    for row in range(source.rows):
        for col in range(source.cols):
            y0 = margin_y + row * cell + inset
            y1 = margin_y + (row + 1) * cell - inset
            x0 = margin_x + col * cell + inset
            x1 = margin_x + (col + 1) * cell - inset
            patch = rgb[y0:y1, x0:x1]
            scores = {
                "head": _colour_fraction(patch, HEAD_FILL),
                "body": _colour_fraction(patch, BODY_FILL),
                "food": _colour_fraction(patch, FOOD_FILL),
            }
            label = max(scores, key=scores.get)
            if scores[label] < 0.12:
                continue
            if label == "head":
                heads.append((row, col))
            elif label == "body":
                body.add((row, col))
            else:
                foods.add((row, col))
    if len(heads) != 1 or heads[0] in body or heads[0] in foods or body & foods:
        return None
    return DecodedState(head=heads[0], body=frozenset(body), foods=frozenset(foods))


def _decoded_matches(decoded: DecodedState, state: ReplayState) -> bool:
    return bool(
        decoded.head == state.snake[0]
        and decoded.body == frozenset(state.snake[1:])
        and decoded.foods == state.foods
    )


def _image_distance(candidate: np.ndarray, expected: np.ndarray) -> float:
    if candidate.shape != expected.shape:
        return 1.0
    return float(
        np.mean(np.abs(candidate.astype(np.float32) - expected.astype(np.float32))) / 255.0
    )


def _video_frames(path: Path) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    media = validate_video_contract(path)
    capture = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, media


def _verify_video(
    candidate_path: Path,
    source: SourceState,
) -> Tuple[Dict[str, Any], bool]:
    frames, media = _video_frames(candidate_path)
    decoded_frames = [_decode_frame(frame, source) for frame in frames]
    media_valid = bool(frames and all(state is not None for state in decoded_frames))
    if not media_valid:
        return {
            "video_media_valid": False,
            "video_frame_count": len(frames),
            "video_all_frames_decoded": False,
            "video_whole_process_pass": False,
            "video_final_state_pass": False,
        }, False

    distinct: List[DecodedState] = []
    hold_counts: List[int] = []
    frame_state_indices: List[int] = []
    for decoded in decoded_frames:
        assert decoded is not None
        if not distinct or decoded != distinct[-1]:
            distinct.append(decoded)
            hold_counts.append(1)
        else:
            hold_counts[-1] += 1
        frame_state_indices.append(len(distinct) - 1)

    replay = ReplayState(source.initial_snake, source.foods)
    replay_states = [replay]
    moves: List[str] = []
    sequence_legal = bool(distinct and _decoded_matches(distinct[0], replay))
    consumed_events = 0
    if sequence_legal:
        for observed in distinct[1:]:
            delta = (
                observed.head[0] - replay.snake[0][0],
                observed.head[1] - replay.snake[0][1],
            )
            move = _DELTA_TO_DIR.get(delta)
            if move is None or not replay.foods:
                sequence_legal = False
                break
            next_state = _advance(replay, move, rows=source.rows, cols=source.cols)
            if next_state is None or not _decoded_matches(observed, next_state):
                sequence_legal = False
                break
            if len(next_state.foods) < len(replay.foods):
                consumed_events += 1
            moves.append(move)
            replay = next_state
            replay_states.append(replay)

    nonterminal_holds = hold_counts[:-1]
    stable_holds = bool(
        hold_counts
        and all(count >= 2 for count in nonterminal_holds)
        and (not nonterminal_holds or max(nonterminal_holds) - min(nonterminal_holds) <= 1)
        and hold_counts[-1] >= 2
    )
    render_distances: List[float] = []
    render_region_distances: List[float] = []
    if sequence_legal and len(replay_states) == len(distinct):
        expected_images = [
            np.asarray(
                render_snake_board(source.cols, source.rows, state.snake, state.foods),
                dtype=np.uint8,
            )
            for state in replay_states
        ]
        render_distances = [
            _image_distance(frame, expected_images[state_index])
            for frame, state_index in zip(frames, frame_state_indices)
        ]
        # A whole-frame mean lets a patch painted over an untouched part of the
        # board hide behind the rest of the board; the same statistic per 32x32
        # region does not.
        render_region_distances = [
            float(region_error_map(frame, expected_images[state_index]).max())
            for frame, state_index in zip(frames, frame_state_indices)
        ]
    visual_stability = bool(
        render_distances
        and max(render_distances) <= 0.18
        and max(render_region_distances, default=1.0) <= REGION_ERROR_CEILING
    )
    final_state = bool(
        sequence_legal
        and not replay.foods
        and consumed_events == len(source.foods)
        and len(replay.snake) == len(source.initial_snake) + len(source.foods)
    )
    process = bool(
        media_valid
        and sequence_legal
        and stable_holds
        and visual_stability
        and final_state
        and len(distinct) >= len(source.foods) + 1
    )
    return {
        "video_media_valid": media_valid,
        "video_fps": media.get("fps"),
        "video_frame_count": len(frames),
        "video_distinct_state_count": len(distinct),
        "video_state_hold_counts": hold_counts,
        "video_stable_discrete_holds": stable_holds,
        "video_source_replay_legal": sequence_legal,
        "video_replayed_moves": moves,
        "video_food_consumption_event_count": consumed_events,
        "video_expected_food_consumption_event_count": len(source.foods),
        "video_final_food_count": len(replay.foods),
        "video_final_snake_length": len(replay.snake),
        "video_max_source_render_distance": max(render_distances) if render_distances else None,
        "video_camera_and_palette_stability_pass": visual_stability,
        "video_whole_process_pass": process,
        "video_final_state_pass": final_state,
        "video_verifier_used_reference_media": False,
    }, process


def verify_snake_path(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: Optional[str] = None,
    training_phase: Optional[str] = None,
) -> List[EvalRecord]:
    """Verify the declared formal delivery without any shortest-path oracle."""

    requested = str(modality or "").strip().lower()
    if requested not in {"ti2v"}:
        raise ValueError("snake_path verifier requires ti2v")
    records: List[EvalRecord] = []
    for sample in samples:
        record = _record_for(sample)
        try:
            source = _source_state(_structured(sample))
            video_path = resolve_candidate_video(pred_root, sample.id, sample.split)
            if video_path is None:
                raise FileNotFoundError("Missing TI2V output video")
            record.prediction_asset = video_path.as_posix()
            video_metrics, passed = _verify_video(video_path, source)
            record.offline_metrics.update(video_metrics)

            record.offline_pass = bool(passed)
            record.offline_metrics.update(
                {
                    "verifier_profile": VERIFIER_PROFILE,
                    "verifier_source_domain": "semantic_input_and_candidate_only",
                    "requested_modality": requested,
                    "shortest_path_required": False,
                    "binary_correct": bool(passed),
                    "failure_reason": (
                        None
                        if passed
                        else f"{requested} multi-food snake replay or formal artifact check failed"
                    ),
                }
            )
        except Exception as exc:
            record.offline_pass = False
            record.error = str(exc)
            record.offline_metrics.update(
                {
                    "verifier_profile": VERIFIER_PROFILE,
                    "verifier_source_domain": "semantic_input_and_candidate_only",
                    "requested_modality": requested,
                    "shortest_path_required": False,
                    "binary_correct": False,
                    "failure_reason": str(exc),
                }
            )
        records.append(record)
    return records


VERIFIER = VerifierRoute(
    "data.dynamics_game.snake_path.evaluator",
    "verify_snake_path",
)


def evaluate(samples: Sequence[Any], pred_root: Path, *, modality: str) -> List[Any]:
    """Evaluate only the formal final delivery for this task."""

    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [
    "TASK_NAME",
    "VERIFIER",
    "VERIFIER_PROFILE",
    "evaluate",
    "verify_snake_path",
]
