"""Reference-free final-delivery verification for retained Dynamics tasks. Each expected outcome and native process state is replayed from the visible
initial semantic source."""

from __future__ import annotations

import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps

from data.dynamics_game.core.source_replay import SourceReplay, replay_source
from data.common.grid_coordinates import parse_xy_path_answer
from data.common.pipeline.sliding_tiles import _swipe_board
from data.common.schemas import CanonicalSample, EvalRecord
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.common.typed_answers import parse_matrix_answer
from data.dynamics_game.merge_swipe_score.generator import Game2048BestSwipeGenerator
from data.common.verifiers.region_gates import (
    REGION_ERROR_CEILING,
    REGION_ERROR_FLOOR,
    SEPARATION_ALPHA,
    STATIC_REGION_FLOOR,
    region_error_map,
    static_region_errors,
)
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.verifiers.keyframes import visual_delta


Grid = List[List[int]]
_CANVAS = 512
_COUNT_RE = re.compile(r"Answer: (0|[1-9][0-9]*)\.")


def _source_record(sample: CanonicalSample) -> Dict[str, Any]:
    return sample.semantic_input if isinstance(sample.semantic_input, dict) else {}


def _structured(sample: CanonicalSample) -> Dict[str, Any]:
    raw = _source_record(sample)
    value = (
        raw.get("semantic_state")
        or raw.get("structured_answer")
        or raw.get("answer_payload")
        or {}
    )
    return value if isinstance(value, dict) else {}


def _int_grid(value: Any, *, square: bool = False) -> Optional[Grid]:
    if not isinstance(value, list) or not value:
        return None
    grid: Grid = []
    for raw_row in value:
        if not isinstance(raw_row, list) or not raw_row:
            return None
        row: List[int] = []
        for cell in raw_row:
            if isinstance(cell, bool) or not isinstance(cell, int):
                return None
            row.append(cell)
        grid.append(row)
    width = len(grid[0]) if grid else 0
    if not width or any(len(row) != width for row in grid):
        return None
    if square and len(grid) != width:
        return None
    return grid


def _merge_replay(structured: Dict[str, Any]) -> tuple[Optional[List[Grid]], Optional[int], Dict[str, Any]]:
    initial = _int_grid(structured.get("initial_grid"), square=True)
    directions = structured.get("swipe_directions")
    valid_directions = (
        isinstance(directions, list)
        and len(directions) == 2
        and all(str(item).lower() in {"up", "right", "down", "left"} for item in directions)
    )
    if initial is None or len(initial) != 4 or not valid_directions:
        return None, None, {
            "source_initial_grid_valid": initial is not None,
            "source_directions_valid": valid_directions,
        }
    states = [initial]
    counts: List[int] = []
    current = initial
    for raw_direction in directions:
        current, score = _swipe_board(current, str(raw_direction).lower())
        states.append(current)
        counts.append(int(score[0]))
    expected = sum(counts)
    stored_trace = structured.get("state_trace")
    metrics = {
        "source_initial_grid_valid": True,
        "source_directions_valid": True,
        "source_replay_kind": "two_round_2048_swipe_v1",
        "source_replayed_round_merge_counts": counts,
        "source_replayed_merge_total": expected,
        "source_saved_trace_consistent_diagnostic": isinstance(stored_trace, list) and stored_trace == states,
    }
    return states, expected, metrics


def _text_metrics(
    answer_line: Optional[str],
    expected: Any,
    *,
    text_kind: Optional[str] = None,
) -> tuple[Dict[str, Any], bool]:
    kind = text_kind or "count"
    if kind == "option":
        predicted: Any = FOUR_CHOICE_CODEC.parse(answer_line)
    elif kind == "count":
        match = _COUNT_RE.fullmatch(str(answer_line or ""))
        predicted = int(match.group(1)) if match else None
    elif kind == "grid":
        predicted = parse_matrix_answer(answer_line)
    elif kind == "cell_path":
        predicted = parse_xy_path_answer(answer_line)
    else:
        raise ValueError(f"unsupported Dynamics text answer kind: {kind}")
    passed = predicted is not None and expected is not None and predicted == expected
    canonical_kind = {
        "option": "option",
        "count": "typed_text",
        "grid": "grid",
        "cell_path": "path",
    }[kind]
    return {
        "text_prediction": answer_line,
        "text_parse_score": float(predicted is not None),
        "text_predicted_payload": predicted,
        "text_expected_payload_replayed": expected,
        "canonical_answer_kind": canonical_kind,
        "text_answer_score": float(passed),
        "text_pass": passed,
    }, passed


def _median_patch(rgb: np.ndarray, cx: float, cy: float, radius: int) -> np.ndarray:
    x = int(round(cx))
    y = int(round(cy))
    patch = rgb[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    return np.median(patch.reshape(-1, 3), axis=0)


def _decode_merge_grid(rgb: np.ndarray, allowed_values: Iterable[int]) -> Optional[Grid]:
    if rgb.shape != (_CANVAS, _CANVAS, 3):
        return None
    from data.common.pipeline.sliding_tiles import _2048_tile_value

    cell = Game2048BestSwipeGenerator.CELL
    ox = Game2048BestSwipeGenerator.ORIGIN_X
    oy = Game2048BestSwipeGenerator.ORIGIN_Y
    palette: Dict[int, np.ndarray] = {0: np.asarray((235, 240, 247), dtype=np.float32)}
    for value in set(int(item) for item in allowed_values if int(item) > 0):
        palette[value] = np.asarray(_2048_tile_value(value), dtype=np.float32)
    result: Grid = []
    for row in range(4):
        decoded_row: List[int] = []
        for col in range(4):
            colour = _median_patch(rgb, ox + col * cell + 18, oy + row * cell + 18, 4)
            distances = {key: float(np.linalg.norm(colour - target)) for key, target in palette.items()}
            value = min(distances, key=distances.get)
            if distances[value] > 48.0:
                return None
            decoded_row.append(int(value))
        result.append(decoded_row)
    return result


@lru_cache(maxsize=32)
def _canonical_merge_cell_endpoints(value: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render independent absent/full endpoints through the task renderer."""

    absent = np.asarray(
        Game2048BestSwipeGenerator.render_cell_endpoint(0), dtype=np.float32
    )
    terminal = np.asarray(
        Game2048BestSwipeGenerator.render_cell_endpoint(int(value)), dtype=np.float32
    )
    delta = terminal - absent
    # Ignore only antialias/codec-fragile near-zero endpoint differences.  The
    # retained mask covers the complete coloured body, outline, and label.
    footprint = np.linalg.norm(delta, axis=2) >= 24.0
    return absent, terminal, footprint


def _merge_cell_opacity_observation(
    frame: np.ndarray,
    row: int,
    col: int,
    value: int,
) -> Dict[str, Any]:
    cell = Game2048BestSwipeGenerator.CELL
    ox = Game2048BestSwipeGenerator.ORIGIN_X
    oy = Game2048BestSwipeGenerator.ORIGIN_Y
    patch = frame[
        oy + row * cell : oy + (row + 1) * cell,
        ox + col * cell : ox + (col + 1) * cell,
    ].astype(np.float32)
    absent, terminal, footprint = _canonical_merge_cell_endpoints(int(value))
    delta = terminal - absent
    denominator = np.sum(delta * delta, axis=2)
    count = int(np.count_nonzero(footprint))
    if patch.shape != absent.shape or count < 256:
        return {
            "progress": None,
            "spread": 1.0,
            "support": 0.0,
            "fit": 1.0,
            "footprint_pixels": count,
            "pixel_progress": np.empty((0,), dtype=np.float32),
        }
    pixel_progress = np.sum((patch - absent) * delta, axis=2)[footprint] / denominator[footprint]
    progress = float(np.median(pixel_progress))
    q10, q90 = np.quantile(pixel_progress, [0.10, 0.90])
    spread = float(q90 - q10)
    support = float(np.mean(np.abs(pixel_progress - progress) <= 0.10))
    reconstruction = absent + np.clip(progress, 0.0, 1.0) * delta
    fit = float(np.mean(np.abs(patch - reconstruction)[footprint]) / 255.0)
    return {
        "progress": progress,
        "spread": spread,
        "support": support,
        "fit": fit,
        "footprint_pixels": count,
        "pixel_progress": pixel_progress.astype(np.float32),
    }


def _merge_complete_footprint_metrics(
    frames: Sequence[np.ndarray],
    states: Sequence[Grid],
    directions: Sequence[str],
    classified: Sequence[Optional[int]],
) -> tuple[Dict[str, Any], bool]:
    """Hard-gate only the atomic summed-tile appearance after each collision.

    Sliding and collision contraction remain semantic motion and are excluded
    by finding the source-derived absent-to-full cell interpolation run between
    consecutive decoded board holds.
    """

    if len(states) != 3 or len(directions) != 2:
        return {
            "video_merge_complete_footprint_verifier": "source_cell_projection_v1",
            "video_merge_complete_footprint_input_valid": False,
            "video_merge_complete_footprint_rounds": [],
            "video_merge_complete_footprint_score": 0.0,
        }, False

    round_metrics: List[Dict[str, Any]] = []
    all_rounds_pass = True
    for round_index, raw_direction in enumerate(directions):
        direction = str(raw_direction).lower()
        targets = Game2048BestSwipeGenerator.merge_targets_for(states[round_index], direction)
        targets_valid = bool(
            targets
            and all(states[round_index + 1][row][col] == value for row, col, value in targets)
        )
        before_indices = [index for index, state in enumerate(classified) if state == round_index]
        next_indices = [index for index, state in enumerate(classified) if state == round_index + 1]
        first_next = next((index for index in next_indices if any(before < index for before in before_indices)), None)
        if first_next is None:
            segment_start = 0
            segment_end = -1
        else:
            preceding = [index for index in before_indices if index < first_next]
            segment_start = max(preceding) if preceding else max(0, first_next - 16)
            segment_end = min(len(frames) - 1, first_next + 4)

        observations: List[Dict[str, Any]] = []
        for frame_index in range(segment_start, segment_end + 1):
            cells = [
                _merge_cell_opacity_observation(frames[frame_index], row, col, value)
                for row, col, value in targets
            ]
            progresses = [float(cell["progress"]) for cell in cells if cell["progress"] is not None]
            sync_spread = max(progresses) - min(progresses) if progresses else 1.0
            uniform = bool(
                len(progresses) == len(targets)
                and cells
                and all(float(cell["spread"]) <= 0.10 for cell in cells)
                and all(float(cell["support"]) >= 0.90 for cell in cells)
                and all(float(cell["fit"]) <= 0.060 for cell in cells)
                and sync_spread <= 0.08
            )
            observations.append(
                {
                    "frame": frame_index,
                    "progress": float(np.mean(progresses)) if progresses else None,
                    "max_spread": max((float(cell["spread"]) for cell in cells), default=1.0),
                    "min_support": min((float(cell["support"]) for cell in cells), default=0.0),
                    "max_fit": max((float(cell["fit"]) for cell in cells), default=1.0),
                    "sync_spread": sync_spread,
                    "uniform": uniform,
                    "pixel_progress": np.concatenate(
                        [cell["pixel_progress"] for cell in cells]
                    ) if cells else np.empty((0,), dtype=np.float32),
                }
            )

        runs: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for observation in observations:
            if observation["uniform"]:
                if current and int(observation["frame"]) != int(current[-1]["frame"]) + 1:
                    runs.append(current)
                    current = []
                current.append(observation)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)

        run_summaries: List[Dict[str, Any]] = []
        selected: Optional[Dict[str, Any]] = None
        for run in runs:
            progresses = [float(item["progress"]) for item in run]
            partial = [value for value in progresses if 0.12 <= value <= 0.88]
            aggregate_monotonic = bool(
                all(right + 0.05 >= left for left, right in zip(progresses, progresses[1:]))
            )
            pixel_pairs = [
                (left["pixel_progress"], right["pixel_progress"])
                for left, right in zip(run, run[1:])
                if left["pixel_progress"].shape == right["pixel_progress"].shape
                and left["pixel_progress"].size
            ]
            pixel_monotonic = float(
                np.mean(
                    np.concatenate(
                        [(right + 0.08 >= left) for left, right in pixel_pairs]
                    )
                )
            ) if pixel_pairs else 0.0
            distinct_partial = bool(len(partial) >= 2 and max(partial) - min(partial) >= 0.20)
            endpoints = bool(progresses and progresses[0] <= 0.08 and progresses[-1] >= 0.92)
            run_pass = bool(
                len(run) >= 4
                and endpoints
                and distinct_partial
                and aggregate_monotonic
                and pixel_monotonic >= 0.96
            )
            summary = {
                "frames": [int(item["frame"]) for item in run],
                "progress": progresses,
                "max_q10_q90_spread": max(float(item["max_spread"]) for item in run),
                "min_support": min(float(item["min_support"]) for item in run),
                "max_fit": max(float(item["max_fit"]) for item in run),
                "max_target_sync_spread": max(float(item["sync_spread"]) for item in run),
                "partial_level_count": len(partial),
                "distinct_partial_levels": distinct_partial,
                "endpoint_score": float(endpoints),
                "aggregate_monotonic_score": float(aggregate_monotonic),
                "pixel_monotonic_fraction": pixel_monotonic,
                "pass": run_pass,
            }
            run_summaries.append(summary)
            if run_pass and selected is None:
                selected = summary

        round_pass = bool(targets_valid and selected is not None)
        all_rounds_pass = bool(all_rounds_pass and round_pass)
        fallback = max(
            run_summaries,
            key=lambda item: (
                float(item["endpoint_score"]),
                int(item["partial_level_count"]),
                len(item["frames"]),
            ),
            default=None,
        )
        round_metrics.append(
            {
                "round": round_index + 1,
                "direction": direction,
                "targets": [[row, col, value] for row, col, value in targets],
                "targets_source_valid": targets_valid,
                "segment": [segment_start, segment_end],
                "valid_uniform_frames": [
                    int(item["frame"]) for item in observations if item["uniform"]
                ],
                "selected_run": selected or fallback,
                "pass": round_pass,
            }
        )

    return {
        "video_merge_complete_footprint_verifier": "source_cell_projection_v1",
        "video_merge_complete_footprint_input_valid": True,
        "video_merge_opacity_q10_q90_spread_threshold": 0.10,
        "video_merge_opacity_support_threshold": 0.90,
        "video_merge_opacity_fit_threshold": 0.060,
        "video_merge_target_sync_threshold": 0.08,
        "video_merge_pixel_monotonic_threshold": 0.96,
        "video_merge_complete_footprint_rounds": round_metrics,
        "video_merge_complete_footprint_score": float(all_rounds_pass),
    }, all_rounds_pass


def _read_rgb(path: Path) -> tuple[Optional[np.ndarray], Optional[str]]:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8), None
    except Exception as exc:
        return None, str(exc)


def _project_scene(scene: Image.Image) -> np.ndarray:
    native = scene.convert("RGB")
    if native.size != (_CANVAS, _CANVAS):
        native = ImageOps.fit(native, (_CANVAS, _CANVAS), method=Image.Resampling.LANCZOS)
    return np.asarray(native, dtype=np.uint8)


def _semantic_view(rgb: np.ndarray) -> np.ndarray:
    return rgb


def _image_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size == 0:
        return 1.0
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) / 255.0)


def _localized_change_distance_matrix(
    candidates: Sequence[np.ndarray],
    expected_states: Sequence[np.ndarray],
    baseline: np.ndarray,
) -> List[List[float]]:
    """Cache downsampled change masks for whole-video stage matching."""

    stride = 2
    baseline_small = baseline[::stride, ::stride]

    def features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        small = image[::stride, ::stride]
        changed = visual_delta(small, baseline_small).mean(axis=2) >= 6.0
        return small, changed

    candidate_features = [features(image) for image in candidates]
    expected_features = [features(image) for image in expected_states]
    rows: List[List[float]] = []
    for expected_small, expected_mask in expected_features:
        row: List[float] = []
        for candidate_small, candidate_mask in candidate_features:
            mask = expected_mask | candidate_mask
            difference = visual_delta(candidate_small, expected_small).mean(axis=2)
            if int(np.count_nonzero(mask)) < 3:
                row.append(float(np.mean(difference) / 255.0))
            else:
                row.append(float(np.mean(difference[mask]) / 255.0))
        rows.append(row)
    return rows


def _semantic_stage_distance_matrix(
    candidates: Sequence[np.ndarray],
    expected_states: Sequence[np.ndarray],
) -> List[List[float]]:
    """Measure every stage on the pixels that distinguish it from a neighbor.

    A single initial-state baseline can dilute a late local commit, such as a
    shrinking brick finally disappearing.  Each expected state therefore uses
    its preceding state as the baseline.  Stage zero uses stage one as its
    contrast so reversed or terminal-first video cannot masquerade as an
    initial hold merely because most of the canvas is empty.
    """

    if not expected_states:
        return []
    rows: List[List[float]] = []
    for index, expected in enumerate(expected_states):
        baseline = (
            expected_states[1]
            if index == 0 and len(expected_states) > 1
            else expected_states[index - 1]
            if index > 0
            else expected
        )
        rows.append(
            _localized_change_distance_matrix(candidates, [expected], baseline)[0]
        )
    return rows


def _lane_semantic_stage_distance_matrix(
    candidates: Sequence[np.ndarray],
    expected_states: Sequence[np.ndarray],
) -> List[List[float]]:
    """Read complete note/hit identities, allowing local contour raster noise.

    Erasing a moved note can leave a pale background patch without leaving a
    red note. Raw RGB change masks mistake that patch for another moving
    object. Count and locate every coloured component instead, with symmetric
    contour coverage so missing, extra or malformed objects still fail.
    Fixed source ink and the complete scene retain local preservation gates.
    """
    import cv2
    from data.common.verifiers.numeric_glyphs import numeric_glyph_descriptor, read_numeric_glyph
    from data.dynamics_game.lane_tap_step.generator import RhythmGameTapStepGenerator

    renderer = RhythmGameTapStepGenerator(Path(tempfile.gettempdir()), video=False)
    badge_templates = {
        speed: _project_scene(renderer._render([[] for _ in range(4)], [speed] * 4))
        for speed in range(1, 10)
    }
    badge_boxes = [
        (round(renderer.left + lane * renderer.lane_w + renderer.lane_w / 2), 71,
         round(renderer.left + lane * renderer.lane_w + renderer.lane_w / 2 + 15), 92)
        for lane in range(4)
    ]

    def badge_ink(patch: np.ndarray) -> np.ndarray:
        red, green, blue = patch.astype(np.int16).transpose(2, 0, 1)
        ink = (blue - red > 30) & (blue - green > 15) & (green < 180)
        return np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)

    vocabularies = [
        {speed: numeric_glyph_descriptor(badge_ink(image[top:bottom, left:right]))
         for speed, image in badge_templates.items()}
        for left, top, right, bottom in badge_boxes
    ]

    def speeds(image: np.ndarray) -> List[Optional[int]]:
        return [read_numeric_glyph(badge_ink(image[top:bottom, left:right]), vocabulary)[0]
                for (left, top, right, bottom), vocabulary in zip(badge_boxes, vocabularies)]

    required_speeds = speeds(expected_states[0])

    def objects(image: np.ndarray) -> List[List[tuple[np.ndarray, np.ndarray, int]]]:
        red, green, blue = image.astype(np.int16).transpose(2, 0, 1)
        masks = (
            (red >= 160) & (red - green >= 65) & (red - blue >= 55),
            (green >= 125) & (green - red >= 25) & (green - blue >= 5),
        )
        result = []
        for mask in masks:
            count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            result.append([
                (labels == index, centers[index], int(stats[index, cv2.CC_STAT_AREA]))
                for index in range(1, count)
                if stats[index, cv2.CC_STAT_AREA] >= 24
            ])
        return result

    expected_objects = [objects(image) for image in expected_states]
    candidate_objects = [objects(image) for image in candidates]
    changed = np.any(np.stack([
        np.max(np.abs(image.astype(np.int16) - expected_states[0].astype(np.int16)), axis=2) > 8
        for image in expected_states
    ]), axis=0)
    static = ~cv2.dilate(changed.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    preserved = [
        static_region_errors(image, expected_states[0], static, region=16)[0] <= STATIC_REGION_FLOOR
        and None not in required_speeds and speeds(image) == required_speeds
        for image in candidates
    ]

    def match(observed: List[List[tuple[np.ndarray, np.ndarray, int]]],
              expected: List[List[tuple[np.ndarray, np.ndarray, int]]]) -> bool:
        for colour, (actual_group, expected_group) in enumerate(zip(observed, expected)):
            if len(actual_group) != len(expected_group):
                return False
            used = set()
            for required_mask, center, area in expected_group:
                nearest = sorted(
                    (float(np.linalg.norm(observed_center - center)), index)
                    for index, (_mask, observed_center, _area) in enumerate(actual_group)
                    if index not in used
                )
                if not nearest or nearest[0][0] > 4.:
                    return False
                index = nearest[0][1]
                observed_mask, _center, observed_area = actual_group[index]
                if not .65 <= observed_area / area <= 1.40:
                    return False
                if colour == 0:
                    ys, xs = np.where(observed_mask)
                    width, height = np.ptp(xs) + 1, np.ptp(ys) + 1
                    if not (.62 <= observed_area / (width * height) <= .90
                            and .80 <= width / height <= 1.25):
                        return False
                expanded_observed = cv2.dilate(observed_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                expanded_required = cv2.dilate(required_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                if (float(np.mean(expanded_observed[required_mask])) < .80
                        or float(np.mean(expanded_required[observed_mask])) < .80):
                    return False
                used.add(index)
        return True

    return [[
        0. if (preserved[index] and match(observed, expected)
               and static_region_errors(image, required, np.ones(image.shape[:2], dtype=bool))[0]
               <= STATIC_REGION_FLOOR) else 1.
        for index, (image, observed) in enumerate(zip(candidates, candidate_objects))
    ] for required, expected in zip(expected_states, expected_objects)]


def _compact(indices: Iterable[Optional[int]]) -> List[int]:
    result: List[int] = []
    for index in indices:
        if index is None or (result and result[-1] == index):
            continue
        result.append(index)
    return result


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
    return frames, fps, None


def _media_contract(frames: Sequence[np.ndarray], fps: float) -> bool:
    return bool(
        frames
        and all(frame.shape == (_CANVAS, _CANVAS, 3) for frame in frames)
        and 15.5 <= fps <= 16.5
        and len(frames) % 4 == 1
    )


def _source_video_metrics(
    path: Path,
    source_path: Path,
    replay: SourceReplay,
) -> tuple[Dict[str, Any], bool]:
    frames, fps, error = _read_video(path)
    source, source_error = _read_rgb(source_path)
    if not frames or source is None:
        return {"video_artifact_validity_score": 0.0, "video_error": error or source_error}, False
    projected = [_project_scene(scene) for scene in replay.scenes]
    frame_views = [_semantic_view(frame) for frame in frames]
    expected_views = [_semantic_view(frame) for frame in projected]
    lane_semantics = replay.task_name == "lane_tap_step"
    stage_rows = (
        _lane_semantic_stage_distance_matrix(frame_views, expected_views)
        if lane_semantics else _semantic_stage_distance_matrix(frame_views, expected_views)
    )
    stage_thresholds = [0. if lane_semantics else 0.040 for _ in expected_views]
    # Each state must also retain every local object. A changed-mask average
    # alone lets an incorrect cell be paid for by the remaining correct cells.
    # This gate uses complete 32px regions, independently of candidate changes.
    if not lane_semantics:
        for stage_index, expected in enumerate(expected_views):
            alternates = [image for index, image in enumerate(expected_views) if index != stage_index]
            separations = np.stack([region_error_map(expected, image) for image in alternates])
            positive = np.where(separations > 1e-9, separations, np.inf).min(axis=0)
            separation = np.where(np.isfinite(positive), positive, 0.0)
            bounds = np.maximum(REGION_ERROR_FLOOR, np.minimum(REGION_ERROR_CEILING, separation * SEPARATION_ALPHA))
            for frame_index, frame in enumerate(frame_views):
                if np.any(region_error_map(frame, expected) > bounds):
                    stage_rows[stage_index][frame_index] = 1.0
    positions = [int(np.argmin(row)) for row in stage_rows]
    distances = [float(row[position]) for row, position in zip(stage_rows, positions)]
    stage_holds = [
        sum(value <= threshold for value in row)
        for row, threshold in zip(stage_rows, stage_thresholds)
    ]
    classified_stages: List[Optional[int]] = []
    for frame_index in range(len(frame_views)):
        matching = [
            (row[frame_index], stage_index)
            for stage_index, (row, threshold) in enumerate(zip(stage_rows, stage_thresholds))
            if row[frame_index] <= threshold
        ]
        classified_stages.append(min(matching)[1] if matching else None)
    recognized_sequence = _compact(classified_stages)
    stages_valid = bool(
        len(positions) == len(expected_views)
        and all(left < right for left, right in zip(positions, positions[1:]))
        and all(distance <= threshold for distance, threshold in zip(distances, stage_thresholds))
        and all(count >= 1 for count in stage_holds)
        and recognized_sequence == list(range(len(expected_views)))
    )
    initial_distances = list(stage_rows[0])
    final_distances = list(stage_rows[-1])
    initial_holds = sum(value <= stage_thresholds[0] for value in initial_distances)
    final_holds = sum(value <= stage_thresholds[-1] for value in final_distances)
    endpoint_holds = bool(
        initial_distances[0] <= stage_thresholds[0]
        and final_distances[-1] <= stage_thresholds[-1]
        and initial_holds >= 3
        and final_holds >= 4
    )
    adjacent = [
        _image_distance(current, previous)
        for previous, current in zip(frame_views, frame_views[1:])
    ]
    active_count = sum(value >= 0.00018 for value in adjacent)
    discrete = replay.timing == "discrete"
    if discrete:
        # A discrete world switches directly between complete states.  The
        # ordered-stage and endpoint gates already require every state to
        # appear, so the continuous-motion activity floor does not apply;
        # instead every frame must classify as one expected exact state, with
        # no interpolated in-between rendering.
        unclassified = sum(stage is None for stage in classified_stages)
        no_interpolated_frames = unclassified == 0
        transition_visible = True
    else:
        required_active = max(8, min(20, len(frames) // 6))
        no_interpolated_frames = True
        unclassified = None
        transition_visible = active_count >= required_active
    no_hard_cut = bool(adjacent and max(adjacent) <= 0.115)
    media = _media_contract(frames, fps)

    # Semantic stage matching localizes state changes and cannot protect the
    # rest of the canvas. Bound the source-declared moving area and check every
    # unchanged 32px region in every frame, including between commits.
    expected_stack = np.stack(projected).astype(np.int16)
    changed = np.max(expected_stack,axis=0)-np.min(expected_stack,axis=0)
    import cv2
    changing_pixels = cv2.dilate(
        (np.max(changed, axis=2) > 8).astype(np.uint8), np.ones((9, 9), np.uint8),
    ).astype(bool)
    static = ~changing_pixels
    # Continuous motion owns its swept area. Separate it from the union of
    # discrete commits: one cleared row plus a piece travelling on the right
    # must not exempt the entire board (and every surviving block) from scene
    # preservation.
    if replay.motion_scenes:
        motion_stack = np.stack([projected[0]] + [_project_scene(scene) for scene in replay.motion_scenes]).astype(np.int16)
        motion_change = np.max(motion_stack, axis=0) - np.min(motion_stack, axis=0)
        yy, xx = np.where(np.max(motion_change, axis=2) > 8)
        if len(xx):
            static[max(0, int(yy.min()) - 4):int(yy.max()) + 5,
                   max(0, int(xx.min()) - 4):int(xx.max()) + 5] = False
    elif not discrete:
        yy, xx = np.where(changing_pixels)
        if len(xx):
            static[int(yy.min()):int(yy.max()) + 1, int(xx.min()):int(xx.max()) + 1] = False
    static_error = max(static_region_errors(frame.astype(np.float32),source.astype(np.float32),static)[0] for frame in frames)
    static_preserved = static_error <= STATIC_REGION_FLOOR
    piece_metrics = {}
    supply_process = supply_terminal = True
    if replay.task_name == 'block_blast_clear_predict':
        from data.dynamics_game.block_blast_clear_predict.generator import PIECE_FILL
        left, top, right, bottom = replay.metrics['source_piece_card_box']
        color = np.asarray(PIECE_FILL, dtype=float)
        def orange(image):
            return np.linalg.norm(image.astype(float)-color, axis=2) <= 48
        landed = orange(projected[1])
        landed[top:bottom,left:right] = False
        coverages = [float(orange(frame)[landed].mean()) for frame in frames]
        landed_index = next((index for index, coverage in enumerate(coverages) if coverage >= .85), None)
        supply_pixels = [int(orange(frame)[top+4:bottom-4,left+4:right-4].sum()) for frame in frames]
        # A full landed piece cannot have another copy in its supply card.
        # Read that role locally rather than average it with the larger board.
        supply_terminal = supply_pixels[-1] < 24
        supply_process = landed_index is not None and all(value < 24 for value in supply_pixels[landed_index:])
        piece_metrics = {'video_block_piece_landed_frame': landed_index,
                         'video_block_supply_orange_pixels': supply_pixels,
                         'video_block_supply_stays_empty_after_landing': supply_process,
                         'video_block_supply_terminal_empty': supply_terminal}
    passed = bool(media and stages_valid and no_interpolated_frames
                  and endpoint_holds and transition_visible
                  and (discrete or no_hard_cut) and static_preserved
                  and initial_distances[0] <= stage_thresholds[0]
                  and final_distances[-1] <= stage_thresholds[-1]
                  and supply_process and supply_terminal)
    metrics = {
        "video_semantic_verifier": "dynamics_source_replay_whole_video_v1",
        "video_reference_keyframes_used": False,
        "video_saved_process_trace_used": False,
        "video_artifact_validity_score": float(media),
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_static_region_max_error": static_error,
        "video_static_preserved": static_preserved,
        "video_source_stage_positions": positions,
        "video_source_stage_distances": distances,
        "video_source_stage_thresholds": stage_thresholds,
        "video_source_stage_hold_counts": stage_holds,
        "video_recognized_stage_sequence": recognized_sequence,
        "video_expected_stage_sequence": list(range(len(expected_views))),
        "video_source_stage_distance_kind": (
            "lane_note_hit_component_identity" if lane_semantics else "neighbor_localized_change"
        ),
        "video_expected_stage_count": len(expected_views),
        "video_state_sequence_score": float(stages_valid),
        "video_initial_hold_count": initial_holds,
        "video_final_hold_count": final_holds,
        "video_state_hold_score": float(endpoint_holds),
        "video_transition_visibility_score": float(transition_visible),
        "video_timing_model": replay.timing,
        "video_unclassified_frame_count": unclassified,
        "video_no_interpolated_frame_score": float(no_interpolated_frames),
        "video_active_change_count": active_count,
        "video_max_adjacent_change": max(adjacent) if adjacent else None,
        "video_no_hard_cut_score": float(no_hard_cut),
        "video_final_state_score": float(final_distances[-1] <= stage_thresholds[-1] and supply_terminal),
        **piece_metrics,
        "video_pass": passed,
    }
    return metrics, passed


def _merge_video_metrics(
    path: Path,
    states: Sequence[Grid],
    directions: Sequence[str],
) -> tuple[Dict[str, Any], bool]:
    frames, fps, error = _read_video(path)
    if not frames:
        return {"video_artifact_validity_score": 0.0, "video_error": error}, False
    values = {cell for grid in states for row in grid for cell in row}
    decoded = [_decode_merge_grid(frame, values) for frame in frames]
    classified = [states.index(grid) if grid in states else None for grid in decoded]
    sequence = _compact(classified)
    semantic_sequence = sequence == [0, 1, 2]
    hold_counts = [classified.count(index) for index in range(3)]
    holds_valid = hold_counts[0] >= 3 and hold_counts[1] >= 2 and hold_counts[2] >= 5
    adjacent = [
        float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))) / 255.0)
        for previous, current in zip(frames, frames[1:])
    ]
    visible = sum(value >= 0.0002 for value in adjacent) >= 16 and (max(adjacent) if adjacent else 1.0) <= 0.09
    cell = Game2048BestSwipeGenerator.CELL
    ox = Game2048BestSwipeGenerator.ORIGIN_X
    oy = Game2048BestSwipeGenerator.ORIGIN_Y
    outside = np.ones((_CANVAS, _CANVAS), dtype=bool)
    outside[max(0, oy - 30) : oy + 4 * cell + 12, max(0, ox - 30) : ox + 4 * cell + 12] = False
    baseline = frames[0].astype(np.float32)
    outside_change = max(
        float(np.mean(np.abs(frame.astype(np.float32) - baseline)[outside]) / 255.0)
        for frame in frames
    )
    # The mean over the whole outside region dilutes a painted patch to
    # nothing; the same statistic per 32x32 region does not (section 11.8).
    outside_region_change = max(
        static_region_errors(frame.astype(np.float32), baseline, outside)[0]
        for frame in frames
    )
    camera_static = outside_change <= 0.025 and outside_region_change <= STATIC_REGION_FLOOR
    media = _media_contract(frames, fps)
    footprint_metrics, footprint_pass = _merge_complete_footprint_metrics(
        frames,
        states,
        directions,
        classified,
    )
    passed = bool(
        media
        and semantic_sequence
        and holds_valid
        and visible
        and camera_static
        and footprint_pass
    )
    metrics = {
        "video_semantic_verifier": "two_round_merge_whole_video_state_decoder_v1",
        "video_reference_keyframes_used": False,
        "video_artifact_validity_score": float(media),
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_recognized_state_sequence": sequence,
        "video_expected_state_sequence": [0, 1, 2],
        "video_state_hold_counts": hold_counts,
        "video_state_sequence_score": float(semantic_sequence),
        "video_state_hold_score": float(holds_valid),
        "video_transition_visibility_score": float(visible),
        "video_active_change_count": sum(value >= 0.0002 for value in adjacent),
        "video_max_adjacent_change": max(adjacent) if adjacent else None,
        "video_camera_static_score": float(camera_static),
        "video_max_outside_change": outside_change,
        "video_static_region_max_error": outside_region_change,
        "video_static_region_bound": STATIC_REGION_FLOOR,
        "video_final_state_score": float(bool(sequence and sequence[-1] == 2)),
        "video_pass": passed,
    }
    metrics.update(footprint_metrics)
    return metrics, passed


def verify_dynamics_game(samples: Sequence[CanonicalSample], pred_root: Path, *, modality: str = "ti2v") -> List[EvalRecord]:
    """Verify the complete video by replaying only the visible semantic source."""
    if modality != "ti2v":
        raise ValueError("Dynamics Game supports only TI2V delivery")
    records = []
    for sample in samples:
        if sample.task_name == "sokoban":
            from data.dynamics_game.core.sokoban import verify_sokoban_sample
            records.append(verify_sokoban_sample(sample, pred_root, modality=modality))
            continue
        record = EvalRecord(sample_id=sample.id, task_name=sample.task_name, task_family=sample.task_family,
                            input_asset=sample.primary_input_image, split=sample.split)
        try:
            video_path = resolve_candidate_video(pred_root, sample.id, sample.split)
            if video_path is None:
                raise FileNotFoundError(f"Missing output video for {sample.id}")
            structured = _structured(sample)
            if sample.task_name == "merge_swipe_score":
                states, expected, source_metrics = _merge_replay(structured)
                if states is None or expected is None:
                    raise ValueError("Could not replay the merge source")
                metrics, passed = _merge_video_metrics(video_path, states, [str(item).lower() for item in structured.get("swipe_directions", [])])
            else:
                replay = replay_source(sample.task_name, structured)
                source_metrics = dict(replay.metrics)
                metrics, passed = _source_video_metrics(video_path, Path(sample.primary_input_image), replay)
            record.prediction_asset = str(video_path)
            record.offline_pass = bool(passed)
            record.offline_metrics = {**source_metrics, **metrics, "video_pass": bool(passed),
                                      "source_saved_answer_used": False, "source_reference_media_used": False,
                                      "binary_correct": bool(passed)}
        except Exception as error:
            record.error = str(error)
            record.offline_pass = False
        records.append(record)
    return records


__all__ = ["verify_dynamics_game"]
