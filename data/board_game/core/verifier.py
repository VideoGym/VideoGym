"""Source-only final-delivery verification for all Board Game tasks. Candidate CoT and every canonical answer/reference-media field are outside the
correctness domain."""

from __future__ import annotations

import ast
import re
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.board_game.core.answers import parse_board_matrix_answer
from data.common.grid_coordinates import (
    parse_xy_path_answer,
    parse_xy_position_answer,
    parse_xy_positions_answer,
)
from data.common.scalar_lists import (
    parse_nonnegative_integer_list_answer,
    parse_nonnegative_integer_answer,
)
from data.board_game.core.motion_video_process import verify_motion_process
from data.board_game.core.placement_video_process import verify_placement_process
from data.board_game.core.source_replay import BoardReplay, replay_board_sample
from data.common.progress_alignment import detoned
from data.common.schemas import CanonicalSample, EvalRecord
from data.common.typed_answers import parse_answer_line
from data.common.verifiers.region_gates import (
    terminal_region_gates,
    static_region_errors,
    STATIC_REGION_FLOOR,
)
from data.common.verifiers.candidates import resolve_candidate_video


_CANVAS = 512
Cell = Tuple[int, int]

_VALID_MODALITIES = {"ti2v"}
_PLACEMENT_PROCESS_TASKS = {
    "tic_tac_toe_rule_simulator",
    "go_rule_simulator",
    "go_max_capture_search",
    "gomoku_rule_simulator",
}
_MOTION_PROCESS_TASKS = {
    "chess_l_waypoint_search",
}


def _read_image(path: Path) -> tuple[Optional[np.ndarray], Optional[str]]:
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        return None, str(exc)
    if rgb.shape != (_CANVAS, _CANVAS, 3):
        return None, f"expected {_CANVAS}x{_CANVAS} RGB image, got {rgb.shape}"
    return rgb, None


def _normalized_mae(left: np.ndarray, right: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    # Preserve state separation when ranking; codec slack belongs in the acceptance gates.
    difference = np.abs(left.astype(np.float32) - right.astype(np.float32)) / 255.0
    if mask is None:
        return float(np.mean(difference))
    if not bool(np.any(mask)):
        return float(np.mean(difference))
    return float(np.mean(difference[mask]))


def _dynamic_mask(initial: np.ndarray, images: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack([initial, *images], axis=0).astype(np.int16)
    mask = np.max(np.max(stack, axis=0) - np.min(stack, axis=0), axis=2) > 8
    try:
        import cv2

        mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=1).astype(bool)
    except Exception:
        pass
    return mask


def _board_answer_syntax_valid(
    answer_line: Optional[str],
    *,
    task_name: str,
) -> bool:
    if task_name == "chess_l_waypoint_search":
        return parse_xy_path_answer(answer_line) is not None
    if task_name == "lightsout_single_press":
        return parse_xy_positions_answer(answer_line) is not None
    if task_name == "go_max_capture_search":
        return parse_xy_position_answer(answer_line) is not None
    if task_name == "mancala_distribute_predict":
        values = parse_nonnegative_integer_list_answer(answer_line)
        return values is not None and len(values) == 7
    if task_name == "hex_chain_count":
        return parse_nonnegative_integer_answer(answer_line) is not None
    board_schemas = {
        "tic_tac_toe_rule_simulator": (3, ("X", "O", ".")),
        "go_rule_simulator": (7, ("B", "W", ".")),
        "gomoku_rule_simulator": (9, ("B", "W", ".")),
    }
    if task_name in board_schemas:
        size, symbols = board_schemas[task_name]
        return parse_board_matrix_answer(answer_line, size=size, symbols=symbols) is not None
    payload = parse_answer_line(answer_line)
    if payload is None:
        return False
    if re.fullmatch(r"[A-Z]+(?:_[A-Z]+)*", payload):
        return True
    if re.fullmatch(r"[a-z][1-9][0-9]*", payload):
        return True
    if re.fullmatch(r"[0-9]+", payload):
        return True
    try:
        parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return False
    if not isinstance(parsed, list) or not parsed:
        return False
    return all(
        (isinstance(item, int) and not isinstance(item, bool))
        or (
            isinstance(item, tuple)
            and len(item) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in item)
        )
        for item in parsed
    )


def _text_metrics(
    sample: CanonicalSample,
    replay: BoardReplay,
    answer_line: Optional[str],
) -> tuple[Dict[str, Any], bool]:
    passed = bool(answer_line is not None and answer_line == replay.answer_line)
    return {
        "text_prediction": answer_line,
        "text_expected_from_source_replay": replay.answer_line,
        "text_parse_score": float(
            _board_answer_syntax_valid(answer_line, task_name=sample.task_name)
        ),
        "text_semantic_replay_score": float(passed),
        "text_pass": passed,
    }, passed


def _image_metrics(path: Path, replay: BoardReplay) -> tuple[Dict[str, Any], bool]:
    candidate, error = _read_image(path)
    if candidate is None:
        return {
            "image_artifact_validity_score": 0.0,
            "image_error": error,
            "image_reference_media_used": False,
            "image_pass": False,
        }, False
    initial = np.asarray(replay.initial_image.convert("RGB"), dtype=np.uint8)
    terminal = np.asarray(replay.terminal_image.convert("RGB"), dtype=np.uint8)
    mask = _dynamic_mask(initial, [terminal])
    dynamic_distance = _normalized_mae(candidate, terminal, mask)
    static_distance = _normalized_mae(candidate, terminal, ~mask)
    global_distance = _normalized_mae(candidate, terminal)
    initial_dynamic_distance = _normalized_mae(candidate, initial, mask)
    changed_region_present = bool(np.any(mask))
    # Every distance above is a mean over a mask, so one stone, seed or marker
    # that is missing from the delivery is paid for by all the ones that are
    # present.  The per-region / per-member gates hold each of them to its own
    # source-derived evidence.
    region_metrics, region_pass = terminal_region_gates(
        candidate, initial, terminal, [initial], prefix="image_"
    )
    passed = bool(
        changed_region_present
        and dynamic_distance <= 0.075
        and static_distance <= 0.030
        and global_distance <= 0.035
        and dynamic_distance + 1e-5 < initial_dynamic_distance
        and region_pass
    )
    task_metrics: Dict[str, Any] = {}
    task_pass = True
    if replay.task_name == "mancala_distribute_predict":
        task_metrics, task_pass = _mancala_terminal_metrics(candidate, terminal)
    elif replay.task_name in {"tic_tac_toe_rule_simulator", "gomoku_rule_simulator"}:
        task_metrics, task_pass = _winning_annotation_metrics(candidate, replay)
    elif replay.task_name == "chess_l_waypoint_search":
        task_metrics, task_pass = _chess_no_trail_metrics(candidate)
    passed = bool(passed and task_pass)
    metrics = {
        "image_semantic_verifier": "board_source_render_replay_v1",
        "image_reference_media_used": False,
        "image_artifact_validity_score": 1.0,
        "image_changed_region_present": changed_region_present,
        "image_terminal_dynamic_distance": dynamic_distance,
        "image_terminal_static_distance": static_distance,
        "image_terminal_global_distance": global_distance,
        "image_initial_dynamic_distance": initial_dynamic_distance,
        **region_metrics,
        "image_final_state_score": float(passed),
        "image_pass": passed,
    }
    metrics.update(task_metrics)
    return metrics, passed


def _annotation_yellow_mask(image: np.ndarray) -> np.ndarray:
    """Saturated winning-annotation gold.

    The blue ceiling and green-blue margin are deliberately tight: supersampled
    boards antialias wood/grid and panel/stone boundaries into warm blends
    whose blue channel stays well above the annotation gold's, so those
    transition pixels must never count as annotation evidence.
    """

    red = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    blue = image[:, :, 2].astype(np.int16)
    return (
        (red >= 170)
        & (green >= 120)
        & (blue <= 70)
        & (red - green >= 15)
        & (green - blue >= 80)
    )


def _chess_gold_mask(image: np.ndarray) -> np.ndarray:
    red = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    blue = image[:, :, 2].astype(np.int16)
    return (
        (red >= 145)
        & (green >= 100)
        & (blue <= 140)
        & (red - green >= 15)
        & (green - blue >= 25)
    )


def _chess_no_trail_metrics(candidate: np.ndarray) -> tuple[Dict[str, Any], bool]:
    gold_pixels = int(np.sum(_chess_gold_mask(candidate)))
    passed = gold_pixels <= 24
    return {
        "image_chess_movement_trail_gold_pixel_count": gold_pixels,
        "image_chess_no_movement_trail_score": float(passed),
    }, passed


def _winning_cell_region(task_name: str, cell: Cell) -> np.ndarray:
    """Cell regions derived from the single authoritative renderer geometry."""

    row, column = cell
    yy, xx = np.ogrid[:_CANVAS, :_CANVAS]
    if task_name == "tic_tac_toe_rule_simulator":
        from data.board_game.core.tic_tac_toe import TicTacToeRenderer

        if not 0 <= row < 3 or not 0 <= column < 3:
            raise ValueError("winning Tic-Tac-Toe cell lies outside the board")
        renderer = TicTacToeRenderer()
        x0, y0, x1, y1 = renderer.cell_box((row, column))
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = renderer.cell / 2.0
        return (np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)
    if task_name == "gomoku_rule_simulator":
        from data.board_game.core.intersection_games import NumericIntersectionRenderer

        if not 0 <= row < 9 or not 0 <= column < 9:
            raise ValueError("winning Gomoku cell lies outside the board")
        renderer = NumericIntersectionRenderer(9)
        cx, cy = renderer.intersection_xy((row, column))
        radius = float(renderer.stone_radius) + 6.0
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    raise ValueError(f"unsupported winning annotation task: {task_name}")


def _winning_annotation_metrics(
    candidate: np.ndarray,
    replay: BoardReplay,
) -> tuple[Dict[str, Any], bool]:
    overlay = replay.source_payload.get("terminal_overlay")
    if not isinstance(overlay, dict):
        return {"image_winning_annotation_error": "source replay lacks terminal_overlay"}, False
    raw_cells = overlay.get("cells")
    if not isinstance(raw_cells, list):
        return {"image_winning_annotation_error": "terminal_overlay cells are unavailable"}, False
    try:
        cells = [tuple(int(value) for value in cell) for cell in raw_cells]
    except (TypeError, ValueError):
        return {"image_winning_annotation_error": "terminal_overlay contains an invalid cell"}, False
    expected_count = 3 if replay.task_name == "tic_tac_toe_rule_simulator" else 5
    if len(cells) != expected_count or any(len(cell) != 2 for cell in cells):
        return {"image_winning_annotation_error": "terminal_overlay cell count is invalid"}, False

    terminal = np.asarray(replay.terminal_image.convert("RGB"), dtype=np.uint8)
    plain = np.asarray(replay.video_states[-2].convert("RGB"), dtype=np.uint8)
    endpoint_delta = np.max(np.abs(terminal.astype(np.int16) - plain.astype(np.int16)), axis=2)
    expected_support = _annotation_yellow_mask(terminal) & (endpoint_delta >= 12)
    candidate_yellow = _annotation_yellow_mask(candidate)
    if int(np.sum(expected_support)) < 30:
        return {"image_winning_annotation_error": "source winning annotation has no visible support"}, False

    candidate_margin = _dilate_mask(candidate_yellow, 4)
    expected_margin = _dilate_mask(expected_support, 6)
    global_coverage = float(np.mean(candidate_margin[expected_support]))
    per_cell_coverage: List[float] = []
    for cell in cells:
        support = expected_support & _winning_cell_region(replay.task_name, cell)
        if int(np.sum(support)) < 5:
            return {"image_winning_annotation_error": "one winning cell lacks annotation support"}, False
        per_cell_coverage.append(float(np.mean(candidate_margin[support])))
    unexpected = int(np.sum(candidate_yellow & ~expected_margin))
    passed = bool(
        global_coverage >= 0.82
        and min(per_cell_coverage, default=0.0) >= 0.75
        and unexpected <= 80
    )
    return {
        "image_winning_annotation_verifier": "source_winning_overlay_v1",
        "image_winning_annotation_kind": str(overlay.get("kind")),
        "image_winning_annotation_cells": [list(cell) for cell in cells],
        "image_winning_annotation_global_coverage": global_coverage,
        "image_winning_annotation_per_cell_coverage": per_cell_coverage,
        "image_winning_annotation_unexpected_yellow_pixels": unexpected,
        "image_winning_annotation_score": float(passed),
    }, passed


def _read_video(path: Path) -> tuple[List[np.ndarray], float, Optional[str]]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        return [], 0.0, str(exc)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return [], 0.0, "cannot open video"
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(bgr[:, :, ::-1].copy())
    capture.release()
    if not frames:
        return [], fps, "video has no frames"
    return frames, fps, None


def _media_contract(frames: Sequence[np.ndarray], fps: float) -> bool:
    count = len(frames)
    return bool(
        all(frame.shape == (_CANVAS, _CANVAS, 3) for frame in frames)
        and 15.5 <= fps <= 16.5
        and count > 0
        and count % 4 == 1
        and count <= 157
        and count / fps <= 10.05
    )


def _border_static(
    frames: Sequence[np.ndarray],
    dynamic_mask: Optional[np.ndarray] = None,
) -> tuple[float, bool]:
    """Camera-framing gate on the outer border band.

    Edge-to-edge boards may legally change pixels inside the band (an edge
    cell toggling, a border-row waypoint fill), so the caller passes its
    source-derived dynamic mask and the gate inspects only the static
    remainder.  A pan or zoom still moves that remainder.
    """

    border = np.zeros((_CANVAS, _CANVAS), dtype=bool)
    border[:18, :] = True
    border[-18:, :] = True
    border[:, :18] = True
    border[:, -18:] = True
    if dynamic_mask is not None:
        border &= ~_dilate_mask(dynamic_mask, 2)
    if not bool(np.any(border)):
        return 0.0, True
    reference = frames[0]
    distances = [_normalized_mae(frame, reference, border) for frame in frames]
    maximum = max(distances) if distances else 1.0
    return maximum, maximum <= 0.025


def _source_static_scene_metrics(
    frames: Sequence[np.ndarray],
    source: np.ndarray,
    dynamic_mask: np.ndarray,
) -> tuple[Dict[str, Any], bool]:
    """Inspect every unchanged source region in every frame, including holds."""
    static = ~_dilate_mask(dynamic_mask, 2)
    maximum = max(
        (static_region_errors(frame, source, static)[0] for frame in frames),
        default=1.0,
    )
    passed = maximum <= STATIC_REGION_FLOOR
    return {
        "video_static_region_max_error": maximum,
        "video_static_region_bound": STATIC_REGION_FLOOR,
        "video_static_preserved": passed,
    }, passed


def _endpoint_linearity(
    frames: Sequence[np.ndarray],
    source: np.ndarray,
    terminal: np.ndarray,
    mask: np.ndarray,
) -> tuple[List[float], List[float], float, bool]:
    """Detect a compressed straight RGB dissolve between endpoint renders."""

    source_float = source.astype(np.float32)
    terminal_float = terminal.astype(np.float32)
    endpoint_direction = terminal_float - source_float
    active_rgb = np.repeat(mask[:, :, None], 3, axis=2)
    endpoint_denominator = float(np.sum(endpoint_direction[active_rgb] ** 2))
    projections: List[float] = []
    residuals: List[float] = []
    if endpoint_denominator > 1e-6:
        for frame in frames:
            frame_float = frame.astype(np.float32)
            delta = frame_float - source_float
            progress = float(
                np.sum(delta[active_rgb] * endpoint_direction[active_rgb])
                / endpoint_denominator
            )
            progress = max(0.0, min(1.0, progress))
            interpolation = source_float + progress * endpoint_direction
            residual = float(
                np.mean(np.abs(frame_float[active_rgb] - interpolation[active_rgb]))
                / 255.0
            )
            projections.append(progress)
            residuals.append(residual)
    middle = [
        residual
        for progress, residual in zip(projections, residuals)
        if 0.05 < progress < 0.95
    ]
    median = (
        float(np.median(np.asarray(middle, dtype=np.float32)))
        if middle
        else 1.0
    )
    # Lossless/keyframe WebM still introduces tiny integer-rounding residuals.
    # A true endpoint dissolve remains orders of magnitude closer to the
    # one-dimensional RGB segment than a moving token or a shape-growing
    # marker, including after decoding.
    detected = bool(len(middle) >= 3 and median <= 0.00003)
    return projections, residuals, median, detected


def _dilate_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    """Return a small antialias/compression margin around a semantic mask."""

    try:
        import cv2

        width = max(1, int(pixels) * 2 + 1)
        return cv2.dilate(
            mask.astype(np.uint8),
            np.ones((width, width), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
    except Exception:  # pragma: no cover - cv2 is already required for video
        return mask.astype(bool)


def _angular_masks(
    support: np.ndarray,
    centre: Tuple[float, float],
    count: int,
) -> List[np.ndarray]:
    yy, xx = np.ogrid[:_CANVAS, :_CANVAS]
    angle = (np.arctan2(yy - float(centre[1]), xx - float(centre[0])) + 2.0 * np.pi) % (2.0 * np.pi)
    sectors: List[np.ndarray] = []
    for index in range(int(count)):
        lower = index * 2.0 * np.pi / count
        upper = (index + 1) * 2.0 * np.pi / count
        sector = support & (angle >= lower) & (angle < upper)
        if bool(np.any(sector)):
            sectors.append(sector)
    return sectors


def _hex_component_ring_metrics(
    frames: Sequence[np.ndarray],
    replay: BoardReplay,
    expected: Sequence[np.ndarray],
) -> tuple[Dict[str, Any], bool]:
    """Check the required ring objects at key states, not their opacity curve."""
    if len(expected) != 2:
        return {"video_hex_ring_error": "hex replay needs source and terminal states"}, False
    component = replay.source_payload.get("component")
    if not component:
        return {"video_hex_ring_error": "largest red component is unavailable"}, False

    from data.board_game.hex_chain_count.generator import HexChainCountGenerator

    generator = HexChainCountGenerator(Path(tempfile.gettempdir()))
    offset_x, offset_y = generator._board_offsets()
    yy, xx = np.ogrid[:_CANVAS, :_CANVAS]
    radius = int(round(generator.hex_r * 0.79))
    initial, terminal = expected
    direction = terminal.astype(np.float32) - initial.astype(np.float32)
    changed = np.max(np.abs(direction), axis=2) >= 12
    progress: List[List[float]] = []
    for cell in component:
        x, y = generator._hex_centre(*cell)
        x += offset_x
        y += offset_y
        distance = (xx - x) ** 2 + (yy - y) ** 2
        support = (distance >= (radius - 5) ** 2) & (distance <= (radius + 5) ** 2) & changed
        if not support.any():
            return {"video_hex_ring_error": "source ring support is empty"}, False
        delta = direction[support]
        energy = max(1.0, float(np.square(delta).sum()))
        progress.append([
            float(np.sum((frame[support].astype(np.float32) - initial[support]) * delta) / energy)
            for frame in frames
        ])

    initial_clear = all(values[0] <= 0.20 for values in progress)
    final_complete = all(values[-1] >= 0.60 for values in progress)
    erased = False
    for values in progress:
        start = next((i for i, value in enumerate(values) if value >= 0.65), None)
        if start is not None:
            erased |= any(
                values[i] < 0.30 and values[i + 1] < 0.30
                for i in range(start, len(values) - 1)
            )
    final_metrics, final_regions = terminal_region_gates(
        frames[-1], initial, terminal, [initial], prefix="video_hex_",
    )
    passed = bool(initial_clear and final_complete and final_regions and not erased)
    return {
        **final_metrics,
        "video_hex_ring_keyframe_progress": progress,
        "video_hex_initial_clear": initial_clear,
        "video_hex_final_complete": final_complete,
        "video_hex_committed_ring_erased": erased,
        "video_hex_circular_ring_process_pass": passed,
    }, passed


def _mancala_slot_boxes() -> List[Tuple[int, int, int, int]]:
    from data.board_game.mancala_distribute_predict.generator import slot_count_boxes

    return slot_count_boxes()


def _mancala_slot_errors(candidate: np.ndarray, expected: np.ndarray) -> List[float]:
    return [
        _normalized_mae(candidate[top:bottom, left:right], expected[top:bottom, left:right])
        for left, top, right, bottom in _mancala_slot_boxes()
    ]


def _mancala_numeric_vocabulary(maximum: int) -> Dict[int, np.ndarray]:
    from data.board_game.mancala_distribute_predict.generator import MancalaDistributePredictGenerator
    from data.common.verifiers.numeric_glyphs import numeric_glyph_descriptor

    renderer = MancalaDistributePredictGenerator(Path(tempfile.gettempdir()))
    left, top, right, bottom = _mancala_slot_boxes()[0]
    vocabulary = {}
    for value in range(max(10, maximum + 2) + 1):
        image = np.asarray(renderer._render([value] * 6, value).convert("RGB"))
        descriptor = numeric_glyph_descriptor(image[top:bottom, left:right])
        if descriptor is not None:
            vocabulary[value] = descriptor
    return vocabulary


def _mancala_numeric_vector(frame: np.ndarray, vocabulary: Mapping[int, np.ndarray]) -> tuple[Optional[int], ...]:
    from data.common.verifiers.numeric_glyphs import read_numeric_glyph

    return tuple(
        read_numeric_glyph(frame[top:bottom, left:right], vocabulary)[0]
        for left, top, right, bottom in _mancala_slot_boxes()
    )


def _mancala_terminal_metrics(candidate: np.ndarray, terminal: np.ndarray) -> tuple[Dict[str, Any], bool]:
    vocabulary = _mancala_numeric_vocabulary(20)
    expected = _mancala_numeric_vector(terminal, vocabulary)
    observed = _mancala_numeric_vector(candidate, vocabulary)
    passed = None not in observed and None not in expected and observed == expected
    return {
        "image_mancala_slot_distances": _mancala_slot_errors(candidate, terminal),
        "image_mancala_observed_counts": list(observed),
        "image_mancala_all_seven_numeric_slots_score": float(passed),
    }, passed


def _mancala_outline_masks(
    base: np.ndarray,
    highlighted: np.ndarray,
    destination: int,
) -> tuple[np.ndarray, List[np.ndarray]]:
    from data.board_game.mancala_distribute_predict.generator import slot_centres

    delta = np.max(np.abs(highlighted.astype(np.int16) - base.astype(np.int16)), axis=2)
    support = delta >= 8
    sectors = _angular_masks(support, slot_centres()[destination], 4)
    return support, sectors


def _mancala_video_metrics(path: Path, replay: BoardReplay) -> tuple[Dict[str, Any], bool]:
    """Decode count commits and their receiving-slot highlight key states."""
    import cv2

    from data.board_game.mancala_distribute_predict.generator import MancalaDistributePredictGenerator
    from data.common.verifiers.keyframes import ordered_state_observations

    frames, fps, error = _read_video(path)
    if error is not None:
        return {"video_artifact_validity_score": 0.0, "video_error": error}, False
    expected = [np.asarray(image.convert("RGB"), dtype=np.uint8) for image in replay.video_states]
    vectors = [tuple(map(int, vector)) for vector in replay.source_payload["count_vectors"]]
    destinations = list(map(int, replay.source_payload["destinations"]))
    vocabulary = _mancala_numeric_vocabulary(max(value for vector in vectors for value in vector))
    observed = [_mancala_numeric_vector(frame, vocabulary) for frame in frames]
    labels = [vectors.index(vector) if vector in vectors else -1 for vector in observed]
    first_readable = next((index for index, label in enumerate(labels) if label >= 0), len(labels))
    compact, numeric_process = ordered_state_observations(labels[first_readable:], len(vectors))
    numeric_process = bool(numeric_process and first_readable <= 3)
    terminal_ok = bool(observed[-1] == vectors[-1] and None not in observed[-1])
    renderer = MancalaDistributePredictGenerator(Path(tempfile.gettempdir()))
    base = np.asarray(renderer._render(vectors[1][:6], vectors[1][6]).convert("RGB"))
    ring_masks = []
    ring_sectors = []
    for slot in range(7):
        highlight = np.asarray(renderer._render(vectors[1][:6], vectors[1][6], active_slot=slot, active_depth=1).convert("RGB"))
        support, sectors = _mancala_outline_masks(base, highlight, slot)
        ring_masks.append(support)
        ring_sectors.append(sectors)
    ring_scores = []
    for frame in frames:
        rgb = frame.astype(np.int16)
        gold = (rgb[:, :, 0] > 165) & (rgb[:, :, 1] > 100) & (rgb[:, :, 2] < 150) & (rgb[:, :, 0] - rgb[:, :, 2] > 65)
        nearby = cv2.dilate(gold.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        ring_scores.append([
            min((float(nearby[sector].mean()) for sector in sectors if sector.any()), default=0.0)
            for sectors in ring_sectors
        ])
    # A readable outline may fade or vary in stroke weight. Its location and
    # complete circular footprint matter; a renderer's exact opacity curve does not.
    highlighted_slots = [[slot for slot, score in enumerate(scores) if score >= .35] for scores in ring_scores]
    scope_ok = True
    for label, slots in zip(labels, highlighted_slots):
        if label < 0:
            continue
        permitted = {int(replay.source_payload["source"])} if label == 0 else set()
        if label >= 2:
            permitted.add(destinations[label - 2])
        if 1 <= label < len(vectors) - 1:
            permitted.add(destinations[label - 1])
        scope_ok &= len(slots) <= 1 and set(slots).issubset(permitted)
    steps = []
    for deposit, slot in enumerate(destinations):
        old_stage, new_stage = deposit + 1, deposit + 2
        commit = next((index for index, label in enumerate(labels) if label == new_stage), None)
        local = [index for index, label in enumerate(labels) if label == old_stage]
        if commit is not None:
            local.append(commit)
        visible = any(slot in highlighted_slots[index] for index in local)
        steps.append({"destination": slot, "receiving_outline_visible": visible})
    outlines_ok = bool(scope_ok and all(step["receiving_outline_visible"] for step in steps) and not highlighted_slots[-1])
    # Numeric slots and all declared ring footprints own their changing pixels.
    # The rest of the source board stays subject to the existing local static gate.
    dynamic = _dynamic_mask(expected[0], expected[1:])
    for support in ring_masks:
        dynamic |= _dilate_mask(support, 4)
    border_change, camera_static = _border_static(frames, dynamic)
    static_metrics, static_preserved = _source_static_scene_metrics(frames, expected[0], dynamic)
    contract = _media_contract(frames, fps)
    passed = bool(contract and numeric_process and outlines_ok and terminal_ok and camera_static and static_preserved)
    return {
        "video_semantic_verifier": "mancala_numeric_glyph_and_receiving_outline_replay_v2",
        "video_reference_media_used": False,
        "video_artifact_validity_score": float(contract),
        "video_frame_count": len(frames), "video_fps": fps,
        "video_mancala_observed_count_vectors": [list(vector) for vector in observed],
        "video_mancala_decoded_stage_labels": labels,
        "video_mancala_compact_stage_sequence": compact,
        "video_state_sequence_score": float(numeric_process),
        "video_mancala_outline_slots": highlighted_slots,
        "video_mancala_destination_outline_steps": steps,
        "video_mancala_receiving_outline_process_score": float(outlines_ok),
        "video_transition_visibility_score": float(numeric_process and outlines_ok),
        "video_max_border_change": border_change,
        "video_camera_static_score": float(camera_static),
        **static_metrics,
        "video_mancala_terminal_slot_distances": _mancala_slot_errors(frames[-1], expected[-1]),
        "video_final_state_score": float(terminal_ok), "video_pass": passed,
    }, passed


def _generic_video_metrics(path: Path, replay: BoardReplay) -> tuple[Dict[str, Any], bool]:
    frames, fps, error = _read_video(path)
    if error is not None:
        return {"video_artifact_validity_score": 0.0, "video_error": error}, False
    expected = [np.asarray(image.convert("RGB"), dtype=np.uint8) for image in replay.video_states]
    mask = _dynamic_mask(expected[0], expected[1:])
    if not bool(np.any(mask)) or len(expected) < 2:
        return {"video_artifact_validity_score": 0.0, "video_error": "source replay has no changing video states"}, False
    # Remove a uniform tone offset using the unchanged background. Local
    # changes to stones, counts and markers remain visible to state matching.
    static_region = ~mask if bool(np.any(~mask)) else None
    frames = [
        np.clip(frame, 0.0, 255.0).astype(np.uint8)
        for frame in detoned(frames, expected[0], static_region)
    ]

    stage_distances: List[List[float]] = []
    stage_global_distances: List[List[float]] = []
    for stage in expected:
        stage_distances.append([_normalized_mae(frame, stage, mask) for frame in frames])
        stage_global_distances.append([_normalized_mae(frame, stage) for frame in frames])
    best_positions = [int(np.argmin(distances)) for distances in stage_distances]
    best_dynamic = [float(distances[position]) for distances, position in zip(stage_distances, best_positions)]
    best_global = [float(distances[position]) for distances, position in zip(stage_global_distances, best_positions)]
    coverage = bool(
        all(distance <= 0.095 for distance in best_dynamic)
        and all(distance <= 0.040 for distance in best_global)
        and all(left < right for left, right in zip(best_positions, best_positions[1:]))
    )

    stage_dynamic_tolerances: List[float] = []
    stage_global_tolerances: List[float] = []

    def stage_tolerances(stage_index: int) -> tuple[float, float]:
        stage = expected[stage_index]
        other_indices = [index for index in range(len(expected)) if index != stage_index]
        nearest_dynamic = min(
            (_normalized_mae(stage, expected[index], mask) for index in other_indices),
            default=0.01,
        )
        nearest_global = min(
            (_normalized_mae(stage, expected[index]) for index in other_indices),
            default=0.01,
        )
        dynamic_tolerance = max(.025, min(.075, nearest_dynamic * .35))
        global_tolerance = max(.012, min(.035, nearest_global * .35))
        return dynamic_tolerance, global_tolerance

    for stage_index in range(len(expected)):
        dynamic_tolerance, global_tolerance = stage_tolerances(stage_index)
        stage_dynamic_tolerances.append(dynamic_tolerance)
        stage_global_tolerances.append(global_tolerance)

    def stage_hold_count(stage_index: int) -> int:
        dynamic_tolerance = stage_dynamic_tolerances[stage_index]
        global_tolerance = stage_global_tolerances[stage_index]
        return sum(
            dynamic <= dynamic_tolerance and global_distance <= global_tolerance
            for dynamic, global_distance in zip(
                stage_distances[stage_index],
                stage_global_distances[stage_index],
            )
        )

    stage_hold_counts = [stage_hold_count(index) for index in range(len(expected))]
    initial_hits = stage_hold_counts[0]
    final_hits = stage_hold_counts[-1]
    # Preserve hit-count diagnostics without requiring a fixed presentation hold.
    minimum_intermediate_hits = 2 if replay.task_name in _PLACEMENT_PROCESS_TASKS else 1
    intermediate_holds = all(
        count >= minimum_intermediate_hits for count in stage_hold_counts[1:-1]
    )

    # Frames confidently matching a semantic state must visit those states in
    # source order exactly once.  Merely finding one best position per state
    # misses a terminal->intermediate->terminal rollback.
    stable_stage_labels: List[int] = []
    for frame_index in range(len(frames)):
        matching = [
            stage_index
            for stage_index in range(len(expected))
            if stage_distances[stage_index][frame_index] <= stage_dynamic_tolerances[stage_index]
            and stage_global_distances[stage_index][frame_index] <= stage_global_tolerances[stage_index]
        ]
        if not matching:
            continue

        def distance(stage_index: int) -> float:
            return (
                stage_distances[stage_index][frame_index]
                / max(stage_dynamic_tolerances[stage_index], 1e-8)
                + stage_global_distances[stage_index][frame_index]
                / max(stage_global_tolerances[stage_index], 1e-8)
            )

        ranked = sorted(matching, key=distance)
        label = ranked[0]
        # Ambiguous blended frames are transitions, not evidence of rollback.
        if len(ranked) > 1 and distance(label) > 1e-6 and distance(label) > 0.75 * distance(ranked[1]):
            continue
        if not stable_stage_labels or stable_stage_labels[-1] != label:
            stable_stage_labels.append(label)
    ordered_stable_states = stable_stage_labels == list(range(len(expected)))

    # Keep terminal-hold diagnostics separate from the source-derived state
    # order and actual final-frame checks used for semantic acceptance.
    terminal_matches = [
        stage_distances[-1][index] <= stage_dynamic_tolerances[-1]
        and stage_global_distances[-1][index] <= stage_global_tolerances[-1]
        for index in range(len(frames))
    ]
    terminal_commit = next(
        (
            index
            for index in range(len(terminal_matches) - 1)
            if terminal_matches[index] and terminal_matches[index + 1]
        ),
        None,
    )
    terminal_stays_committed = bool(
        terminal_commit is not None and all(terminal_matches[terminal_commit:])
    )
    holds = bool(
        initial_hits >= 3
        and final_hits >= 4
        and intermediate_holds
        and ordered_stable_states
        and terminal_stays_committed
        and best_positions[0] <= max(3, len(frames) // 4)
        and best_positions[-1] >= len(frames) // 2
    )

    adjacent_dynamic = [_normalized_mae(current, previous, mask) for previous, current in zip(frames, frames[1:])]
    active_count = sum(value >= 1e-5 for value in adjacent_dynamic)
    endpoint_dynamic = _normalized_mae(expected[0], expected[-1], mask)
    max_dynamic_jump = max(adjacent_dynamic) if adjacent_dynamic else 1.0
    # A whole-endpoint dissolve can visit every coarse pixel-distance stage
    # while never moving a token or constructing a marker.  Measure how
    # closely the complete transition stays on the straight RGB segment
    # between the source and terminal renders.  Both multi-state programs and
    # atomic marker updates must leave that line through motion/shape growth.
    (
        endpoint_projection,
        endpoint_linear_residual,
        median_middle_residual,
        straight_endpoint_dissolve,
    ) = _endpoint_linearity(frames, expected[0], expected[-1], mask)
    endpoint_linear_dissolve = bool(straight_endpoint_dissolve)
    endpoint_linearity_allowed = replay.task_name == "hex_chain_count"
    border_change, camera_static = _border_static(frames, mask)
    contract = _media_contract(frames, fps)
    task_process_metrics: Dict[str, Any] = {}
    task_process_pass = True
    if replay.task_name == "hex_chain_count":
        task_process_metrics, task_process_pass = _hex_component_ring_metrics(frames, replay, expected)
        static_metrics, static_preserved = _source_static_scene_metrics(frames, expected[0], mask)
        task_process_metrics.update(static_metrics)
        task_process_pass = bool(task_process_pass and static_preserved)
        progress = task_process_metrics.get("video_hex_ring_keyframe_progress") or []
        ring_labels = []
        for values in zip(*progress):
            label = 0 if max(values) <= .20 else 1 if min(values) >= .60 else None
            if label is not None and (not ring_labels or ring_labels[-1] != label):
                ring_labels.append(label)
        # Ring completion is a local semantic observation. Sparse thin rings
        # need not share a whole-board state classifier's raster threshold.
        stable_stage_labels = ring_labels
        ordered_stable_states = ring_labels == [0, 1]

    source_process_metrics: Dict[str, Any] = {}
    source_process_pass = True
    if replay.task_name in _PLACEMENT_PROCESS_TASKS:
        source_process_metrics, source_process_pass = verify_placement_process(frames, replay)
    elif replay.task_name in _MOTION_PROCESS_TASKS:
        source_process_metrics, source_process_pass = verify_motion_process(frames, replay)

    process_visibility = bool(
        task_process_pass
        and source_process_pass
    )
    # A moving L-token can briefly resemble a later landing while it passes
    # nearby. Its source-owned observer orders every motion probe and commit,
    # checks rollback, waypoint/target order, holds and unrelated scene ink.
    # Nearest *static* stage labels are therefore diagnostic for this task.
    ordered_process = (
        source_process_pass
        if replay.task_name in _MOTION_PROCESS_TASKS
        else ordered_stable_states
    )
    terminal_metrics, terminal_regions = terminal_region_gates(
        frames[-1], expected[0], expected[-1], expected[:-1], prefix="video_terminal_")
    terminal_ok = bool(
        _normalized_mae(frames[-1], expected[-1], mask) <= 0.095
        and _normalized_mae(frames[-1], expected[-1]) <= 0.040
        and terminal_regions
    )
    initial_ok = _normalized_mae(frames[0], expected[0]) <= 0.040
    passed = bool(
        contract
        and coverage
        and ordered_process
        and initial_ok
        and terminal_ok
        and process_visibility
        and camera_static
    )
    metrics = {
        "video_semantic_verifier": "board_source_stage_replay_v1",
        "video_reference_media_used": False,
        "video_artifact_validity_score": float(contract),
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_expected_stage_count": len(expected),
        "video_stage_best_positions": best_positions,
        "video_stage_best_dynamic_distances": best_dynamic,
        "video_stage_best_global_distances": best_global,
        "video_state_sequence_score": float(coverage),
        "video_initial_hold_frames": initial_hits,
        "video_stage_hold_frames": stage_hold_counts,
        "video_final_hold_frames": final_hits,
        "video_intermediate_state_hold_score": float(intermediate_holds),
        "video_minimum_intermediate_state_hits": minimum_intermediate_hits,
        "video_stable_state_sequence": stable_stage_labels,
        "video_stable_state_order_score": float(ordered_stable_states),
        "video_state_order_score": float(ordered_process),
        "video_state_order_observer": (
            "source_motion_timeline" if replay.task_name in _MOTION_PROCESS_TASKS
            else "ordered_static_stages"
        ),
        "video_terminal_commit_frame": terminal_commit,
        "video_terminal_stays_committed_score": float(terminal_stays_committed),
        "video_state_hold_score": float(holds),
        "video_active_change_count": active_count,
        "video_endpoint_dynamic_distance": endpoint_dynamic,
        "video_max_adjacent_dynamic_change": max_dynamic_jump,
        "video_endpoint_projection": endpoint_projection,
        "video_endpoint_linear_residual": endpoint_linear_residual,
        "video_endpoint_linear_middle_median_residual": median_middle_residual,
        "video_endpoint_linear_dissolve_detected": endpoint_linear_dissolve,
        "video_endpoint_linearity_allowed_by_task_owned_local_fade": endpoint_linearity_allowed,
        "video_transition_visibility_score": float(process_visibility),
        "video_max_border_change": border_change,
        "video_camera_static_score": float(camera_static),
        "video_final_state_score": float(terminal_ok),
        **terminal_metrics,
        "video_pass": passed,
    }
    metrics.update(task_process_metrics)
    metrics.update(source_process_metrics)
    return metrics, passed


def _lights_layout(size: int) -> tuple[int, int, int]:
    from data.board_game.lightsout_single_press.generator import lights_layout

    return lights_layout(size)


def _decode_lights(rgb: np.ndarray, size: int) -> Optional[List[List[int]]]:
    if rgb.shape != (_CANVAS, _CANVAS, 3):
        return None
    cell, ox, oy = _lights_layout(size)
    radius = max(4, cell // 12)
    decoded: List[List[int]] = []
    for row in range(size):
        decoded_row: List[int] = []
        for col in range(size):
            cx = ox + col * cell + cell // 4
            cy = oy + row * cell + cell // 4
            patch = rgb[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
            colour = np.median(patch.reshape(-1, 3), axis=0)
            red, green, blue = (float(value) for value in colour)
            if red >= 165 and green >= 145 and blue <= 175 and red - blue >= 30:
                decoded_row.append(1)
            elif max(red, green, blue) <= 155 and max(red, green, blue) - min(red, green, blue) <= 55:
                decoded_row.append(0)
            else:
                return None
        decoded.append(decoded_row)
    return decoded


def _freeze_grid(grid: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def _toggle_lights(grid: List[List[int]], row: int, col: int) -> None:
    size = len(grid)
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = row + dr, col + dc
        if 0 <= rr < size and 0 <= cc < size:
            grid[rr][cc] = 1 - grid[rr][cc]


def _compact(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    for value in values:
        if value is None:
            continue
        if not result or result[-1] != value:
            result.append(value)
    return result


def _lights_subset_grid(
    initial: Sequence[Sequence[int]],
    subset: Iterable[Cell],
) -> List[List[int]]:
    grid = [[int(value) for value in row] for row in initial]
    for row, col in subset:
        _toggle_lights(grid, row, col)
    return grid


def _lights_cell_progress(
    frame: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    cell_position: Cell,
    size: int,
) -> float:
    """Project one cell colour onto its source-to-commit toggle direction."""

    cell, ox, oy = _lights_layout(size)
    row, col = cell_position
    cx = ox + col * cell + cell // 2
    cy = oy + row * cell + cell // 2
    radius = max(5, cell // 9)
    sample = frame[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1].astype(np.float32).mean(axis=(0, 1))
    source = before[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1].astype(np.float32).mean(axis=(0, 1))
    target = after[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1].astype(np.float32).mean(axis=(0, 1))
    delta = target - source
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-6:
        return 0.0
    return float(np.dot(sample - source, delta) / denominator)


def _lights_press_patch(press: Cell, size: int) -> Tuple[int, int, int, int]:
    """Interior crop of the pressed cell used to measure the press shade."""

    cell, ox, oy = _lights_layout(size)
    row, col = press
    inset = max(6, cell // 6)
    return (
        ox + col * cell + inset,
        oy + row * cell + inset,
        ox + (col + 1) * cell - inset,
        oy + (row + 1) * cell - inset,
    )


def _luma(color: np.ndarray) -> float:
    red, green, blue = (float(value) for value in color)
    return 0.299 * red + 0.587 * green + 0.114 * blue


# A press counts as let go below this shade depth.  The generator's release
# ramp renders cue depths 2/3 and 1/3 and then nothing, and over 20 samples
# (280 no-cue frames, 40 releases) the encoded video measured 0.0000 on every
# frame drawn without a cue and never below 0.3293 on a frame drawn at 1/3.
# The gate sits between the two, far enough above the no-cue ceiling to never
# call a released press held and well under half the shallowest cue step the
# generator ever draws.
_LIGHTS_PRESS_RELEASED = 0.15


def _lights_press_shade_depth(
    frame: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    press: Cell,
    size: int,
    toggle_alpha: float,
) -> float:
    """Estimate how deeply the pressed cell is shaded toward black.

    The declared cue multiplies the pressed cell's current fill by one half at
    full depth.  The expected un-shaded fill at this moment is the source
    color blended toward the committed color by the neighbors' toggle alpha,
    so the returned depth is cue-specific and stays near zero for a plain
    toggle without a press shade.
    """

    x0, y0, x1, y1 = _lights_press_patch(press, size)
    observed = _luma(frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0))
    plain_before = _luma(before[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0))
    plain_after = _luma(after[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0))
    alpha = max(0.0, min(1.0, float(toggle_alpha)))
    plain = plain_before * (1.0 - alpha) + plain_after * alpha
    if plain <= 1.0:
        return 0.0
    return float((plain - observed) / (0.5 * plain))


def _lights_direct_transition_metrics(
    frames: Sequence[np.ndarray],
    recognized: Sequence[Optional[frozenset[Cell]]],
    sequence: Sequence[frozenset[Cell]],
    initial: Sequence[Sequence[int]],
) -> tuple[List[Dict[str, Any]], bool]:
    """Verify the declared press grammar of every toggle step.

    Each step must show which cell is pressed: the pressed cell alone darkens
    to a clear shade while every affected cell still has zero toggle
    progress, the shade is held while the plus-neighborhood switches
    synchronously and monotonically, and the shade is fully lifted at the
    committed board.  A toggle without a visible press shade, a shade on the
    wrong cell, or a residual shade after the commit all fail.
    """

    from data.board_game.lightsout_single_press.generator import LightsOutSinglePressGenerator

    size = len(initial)
    renderer = LightsOutSinglePressGenerator(Path(tempfile.gettempdir()), size=size)
    details: List[Dict[str, Any]] = []
    process_pass = True
    previous_commit = 0
    for previous, current in zip(sequence, sequence[1:]):
        added = tuple(current - previous)
        press = added[0] if len(added) == 1 else None
        if press is None:
            details.append({"press": None, "commit_frame": None, "pass": False})
            process_pass = False
            continue

        before_grid = _lights_subset_grid(initial, previous)
        after_grid = _lights_subset_grid(initial, current)
        before = np.asarray(renderer._render_board(before_grid).convert("RGB"), dtype=np.uint8)
        after = np.asarray(renderer._render_board(after_grid).convert("RGB"), dtype=np.uint8)
        affected = tuple(
            sorted(
                (row, col)
                for row, col in (
                    (press[0], press[1]),
                    (press[0] - 1, press[1]),
                    (press[0] + 1, press[1]),
                    (press[0], press[1] - 1),
                    (press[0], press[1] + 1),
                )
                if 0 <= row < size and 0 <= col < size
            )
        )
        neighbors = tuple(cell for cell in affected if cell != press)
        dynamic = _dynamic_mask(before, [after])
        static = ~dynamic
        press_box = np.zeros((_CANVAS, _CANVAS), dtype=bool)
        tile, ox, oy = _lights_layout(size)
        row, col = press
        press_box[oy + row * tile:oy + (row + 1) * tile,
                  ox + col * tile:ox + (col + 1) * tile] = True
        neighbor_dynamic = dynamic & ~press_box
        # Binary decoding can recognize a late blend before a semantic commit.
        # Require the affected neighbours to have reached their target colours
        # and the press shade to be released. A fixed RGB-error floor on the
        # narrow changing mask rejects generated raster/codec differences;
        # the independent per-cell progress and shade observations read the
        # declared toggle directly without diluting it into static board pixels.
        commit = next(
            (
                index
                for index in range(previous_commit + 1, len(recognized))
                if recognized[index] == current
                and all(_lights_cell_progress(frames[index], before, after, cell, size) >= .86
                        for cell in neighbors)
                and _lights_press_shade_depth(frames[index], before, after, press, size, 1.0)
                <= _LIGHTS_PRESS_RELEASED
            ),
            None,
        )
        if commit is None:
            details.append({"press": list(press), "commit_frame": None, "pass": False})
            process_pass = False
            continue
        progress_rows: List[List[float]] = []
        mean_progress: List[float] = []
        shade_depths: List[float] = []
        residuals: List[float] = []
        static_errors: List[float] = []
        unaffected_errors: List[float] = []
        unaffected = [(row, col) for row in range(size) for col in range(size) if (row, col) not in affected]
        for frame_index in range(previous_commit, commit + 1):
            frame = frames[frame_index]
            cell_progress = [
                _lights_cell_progress(frame, before, after, cell_position, size)
                for cell_position in neighbors
            ]
            progress_rows.append(cell_progress)
            mean = float(np.mean(cell_progress)) if cell_progress else 0.0
            mean_progress.append(mean)
            clipped = max(0.0, min(1.0, mean))
            shade_depths.append(
                _lights_press_shade_depth(frame, before, after, press, size, clipped)
            )
            expected = np.clip(
                before.astype(np.float32) * (1.0 - clipped) + after.astype(np.float32) * clipped,
                0,
                255,
            ).astype(np.uint8)
            residuals.append(_normalized_mae(frame, expected, neighbor_dynamic))
            static_errors.append(_normalized_mae(frame, before, static))
            local_errors = []
            for cell_position in unaffected:
                left, top, right, bottom = _lights_press_patch(cell_position, size)
                local_errors.append(_normalized_mae(frame[top:bottom, left:right], before[top:bottom, left:right]))
            unaffected_errors.append(max(local_errors, default=0.0))

        active_indices = [
            index for index, value in enumerate(mean_progress) if 0.08 <= value <= 0.92
        ]
        active_spreads = [
            max(progress_rows[index]) - min(progress_rows[index]) if progress_rows[index] else 1.0
            for index in active_indices
        ]
        monotonic = all(
            later + 0.08 >= earlier
            for earlier, later in zip(mean_progress, mean_progress[1:])
        )
        pre_switch_indices = [
            index
            for index, value in enumerate(mean_progress)
            if value <= 0.08 and index + previous_commit < commit
        ]
        cue_frames = [index for index in pre_switch_indices if shade_depths[index] >= 0.55]
        first_switch = next(
            (index for index, value in enumerate(mean_progress) if value > 0.08),
            len(mean_progress),
        )
        cue_before_switch = bool(cue_frames and min(cue_frames) < first_switch)
        shade_held = all(shade_depths[index] >= 0.30 for index in active_indices)
        step_pass = bool(
            len(active_indices) >= 2
            and mean_progress[-1] >= 0.86
            and monotonic
            and len(cue_frames) >= 2
            and cue_before_switch
            and shade_held
            and max(active_spreads, default=1.0) <= 0.18
            and max((residuals[index] for index in active_indices), default=1.0) <= 0.060
            and max(static_errors, default=1.0) <= 0.020
            and max(unaffected_errors, default=1.0) <= .060
        )
        details.append(
            {
                "press": list(press),
                "affected_cells": [list(cell) for cell in affected],
                "frame_range": [previous_commit, commit],
                "commit_frame": commit,
                "mean_progress": mean_progress,
                "press_shade_depths": shade_depths,
                "press_cue_frames": [previous_commit + index for index in cue_frames],
                "press_cue_before_switch": cue_before_switch,
                "press_shade_held_during_switch": shade_held,
                "active_frame_count": len(active_indices),
                "active_synchrony_spreads": active_spreads,
                "maximum_active_linear_residual": max(
                    (residuals[index] for index in active_indices), default=1.0
                ),
                "maximum_static_error": max(static_errors, default=1.0),
                "maximum_unaffected_cell_error": max(unaffected_errors, default=1.0),
                "pass": step_pass,
            }
        )
        process_pass = bool(process_pass and step_pass)
        previous_commit = commit

    # Nothing may still hold a press once the last commit has landed: the
    # board is finished and the cue has no state left to mark.  The state
    # sequence cannot see this, because a shade does not change which colour
    # a cell decodes as, so a tail that ends part-way down a release ramp --
    # a reversal across the final release, for one -- would otherwise deliver
    # a committed board with a press still on it.  Every pressed cell is read
    # against the settled board, not just the last one.
    if process_pass and details:
        pressed = sorted(sequence[-1])
        residual = max(
            _lights_press_shade_depth(frames[index], after, after, cell, size, 1.0)
            for index in range(previous_commit, len(frames))
            for cell in pressed
        )
        details[-1]["terminal_hold_maximum_press_shade"] = residual
        if residual > _LIGHTS_PRESS_RELEASED:
            details[-1]["pass"] = False
            process_pass = False
    return details, process_pass


def _lights_video_metrics(path: Path, replay: BoardReplay) -> tuple[Dict[str, Any], bool]:
    frames, fps, error = _read_video(path)
    if error is not None:
        return {"video_artifact_validity_score": 0.0, "video_error": error}, False
    initial = [[int(value) for value in row] for row in replay.source_payload["initial_grid"]]
    presses = tuple(tuple(int(value) for value in cell) for cell in replay.source_payload["presses"])
    states: Dict[Tuple[Tuple[int, ...], ...], frozenset[Cell]] = {}
    for count in range(len(presses) + 1):
        for subset in combinations(presses, count):
            grid = [row[:] for row in initial]
            for row, col in subset:
                _toggle_lights(grid, row, col)
            states[_freeze_grid(grid)] = frozenset(subset)
    decoded = [_decode_lights(frame, len(initial)) for frame in frames]
    recognized = [states.get(_freeze_grid(grid)) if grid is not None else None for grid in decoded]
    sequence = _compact(recognized)
    full = frozenset(presses)
    sequence_valid = bool(
        sequence and recognized[0] == frozenset() and recognized[-1] == full
        and sequence[0] == frozenset() and sequence[-1] == full
    )
    if sequence_valid:
        sequence_valid = len(sequence) == len(presses) + 1 and all(
            previous < current and len(current - previous) == 1
            for previous, current in zip(sequence, sequence[1:])
        )
    counts = Counter(value for value in recognized if value is not None)
    holds = bool(counts[frozenset()] >= 3 and counts[full] >= 4 and all(counts[state] >= 2 for state in sequence[1:-1]))
    transition_steps, transition_process = (
        _lights_direct_transition_metrics(frames, recognized, sequence, initial)
        if sequence_valid
        else ([], False)
    )
    adjacent = [_normalized_mae(current, previous) for previous, current in zip(frames, frames[1:])]
    active = sum(value >= 0.0002 for value in adjacent)
    initial_rgb = np.asarray(replay.initial_image.convert("RGB"), dtype=np.uint8)
    terminal_rgb = np.asarray(replay.terminal_image.convert("RGB"), dtype=np.uint8)
    dynamic_mask = _dynamic_mask(initial_rgb, [terminal_rgb])
    (
        endpoint_projection,
        endpoint_residual,
        endpoint_middle_median,
        endpoint_linear_dissolve,
    ) = _endpoint_linearity(frames, initial_rgb, terminal_rgb, dynamic_mask)
    continuous = bool(
        active >= 3 * len(presses)
        and (max(adjacent) if adjacent else 1.0) <= 0.09
        and not endpoint_linear_dissolve
    )
    commit_states = [
        np.asarray(image.convert("RGB"), dtype=np.uint8) for image in replay.video_states
    ]
    changing = _dynamic_mask(commit_states[0], commit_states[1:])
    border_change, camera_static = _border_static(frames, changing)
    # A local foreign object can disappear in a mean over the whole border.
    # Only source-owned changing cells are exempt; all fixed regions are checked.
    max_static_region_error = max(
        (static_region_errors(frame, initial_rgb, ~changing)[0] for frame in frames),
        default=1.0,
    )
    static_preserved = max_static_region_error <= STATIC_REGION_FLOOR
    contract = _media_contract(frames, fps)
    recognized_fraction = sum(value is not None for value in recognized) / len(frames)
    passed = bool(
        contract
        and sequence_valid
        and transition_process
        and camera_static
        and static_preserved
        and recognized_fraction >= 0.40
    )
    return {
        "video_semantic_verifier": "lights_press_cue_plus_replay_v5",
        "video_reference_media_used": False,
        "video_artifact_validity_score": float(contract),
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_recognized_fraction": recognized_fraction,
        "video_state_sequence": [[list(cell) for cell in sorted(state)] for state in sequence],
        "video_state_sequence_score": float(sequence_valid),
        "video_state_hold_score": float(holds),
        "video_direct_plus_transition_steps": transition_steps,
        "video_direct_plus_transition_score": float(transition_process),
        "video_press_cue_score": float(
            transition_process
            and bool(transition_steps)
            and all(step.get("press_cue_before_switch") for step in transition_steps)
        ),
        "video_active_change_count": active,
        "video_endpoint_projection": endpoint_projection,
        "video_endpoint_linear_residual": endpoint_residual,
        "video_endpoint_linear_middle_median_residual": endpoint_middle_median,
        "video_endpoint_linear_dissolve_detected": endpoint_linear_dissolve,
        "video_transition_visibility_score": float(continuous),
        "video_max_border_change": border_change,
        "video_camera_static_score": float(camera_static),
        "video_static_region_max_error": max_static_region_error,
        "video_static_preserved": static_preserved,
        "video_final_state_score": float(bool(recognized and recognized[-1] == full)),
        "video_pass": passed,
    }, passed


def _video_metrics(path: Path, replay: BoardReplay) -> tuple[Dict[str, Any], bool]:
    if replay.task_name == "lightsout_single_press":
        return _lights_video_metrics(path, replay)
    if replay.task_name == "mancala_distribute_predict":
        return _mancala_video_metrics(path, replay)
    return _generic_video_metrics(path, replay)


def verify_board_game(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: str,
) -> List[EvalRecord]:
    """Verify exactly one explicitly declared Board Game final-delivery modality."""

    normalized_modality = str(modality or "").strip().lower()
    if normalized_modality not in _VALID_MODALITIES:
        raise ValueError(f"modality must be one of {sorted(_VALID_MODALITIES)}")
    records: List[EvalRecord] = []
    with tempfile.TemporaryDirectory(prefix="videogym-board-source-replay-") as temporary_root:
        renderer_root = Path(temporary_root)
        for sample in samples:
            record = EvalRecord(
                sample_id=sample.id,
                task_name=sample.task_name,
                input_asset=sample.primary_input_image,
                task_family=sample.task_family,
                split=sample.split,
            )
            try:
                replay = replay_board_sample(sample, renderer_root)
                video_metrics: Dict[str, Any] = {}
                video_pass: Optional[bool] = None

                if normalized_modality == "ti2v":
                    video_path = resolve_candidate_video(pred_root, sample.id, sample.split)
                    if video_path is None:
                        raise FileNotFoundError(f"Missing full output video for {sample.id}")
                    video_metrics, video_pass = _video_metrics(video_path, replay)
                    record.prediction_asset = video_path.as_posix()


                record.offline_pass = bool(video_pass)
                record.offline_metrics = {
                    **video_metrics, "artifact_results": {"video": bool(video_pass)},
                    "canonical_ground_truth_used": False, "canonical_reference_media_used": False,
                    "candidate_cot_semantically_inspected": False, "binary_correct": bool(video_pass),
                }
            except Exception as exc:  # per-sample failure reporting
                record.error = str(exc)
                record.offline_pass = False
                record.offline_metrics = {
                    "requested_modality": normalized_modality,
                    "whole_video_loaded": False,
                    "whole_video_error": str(exc),
                    "canonical_ground_truth_used": False,
                    "canonical_reference_media_used": False,
                    "candidate_cot_semantically_inspected": False,
                    "binary_correct": False,
                    "failure_reason": str(exc),
                }
            records.append(record)
    return records


__all__ = ["verify_board_game"]
