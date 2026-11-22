"""Whole-video verification for Constraint Grid."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np

from data.constraint_grid.core.render_replay import render_candidate_terminal
from data.constraint_grid.core.scene import standard_board_geometry
from data.constraint_grid.core.terminal_decode import decode_constraint_terminal_payload
from data.common.schemas import CanonicalSample
from data.common.verifiers.region_gates import (
    STATIC_REGION_FLOOR,
    static_region_errors,
)
from data.common.verifiers.media import _read_bgr
from data.common.verifiers.metrics import _verifier_contract_metrics
from data.common.verifiers.keyframes import visual_delta, visual_features


_PIXEL_CHANGE_THRESHOLD = 34.0
_MAX_STATIC_MUTATION_FRACTION = 0.006
_MAX_NQUEENS_FIXED_CELL_MUTATION_FRACTION = 0.015

# Whole-video ordered-commit thresholds.  Reference fades step each committed
# pixel by at most a quarter of its full excursion per frame, so a hard cut
# (near-full excursion in one frame) and a vanishing mark (distance to the
# terminal value rising again) are separated from anti-aliased noise by wide
# margins.
_MAX_PROGRESS_JUMP = 0.30
_HARD_STEP_DELTA = 130.0
_MAX_HARD_STEP_FRACTION = 0.02
_PIXEL_REGRESSION_TOLERANCE = 26.0
_MAX_PIXEL_REGRESSION_FRACTION = 0.004
_ENVELOPE_TOLERANCE = 46.0
_MAX_ENVELOPE_VIOLATION_FRACTION = 0.02


def _nqueens_fixed_cell_masks(
    sample: CanonicalSample,
    shape: tuple[int, ...],
) -> List[np.ndarray]:
    if sample.task_name != "nqueens_completion" or len(shape) < 2:
        return []
    semantic_input = getattr(sample, "semantic_input", {}) or {}
    state = semantic_input.get("semantic_state") if isinstance(semantic_input, dict) else None
    if not isinstance(state, dict):
        return []
    try:
        size = int(state["board_size"])
        fixed = [(int(row), int(col)) for row, col in state["fixed_positions"]]
    except Exception:
        return []
    if size <= 0 or shape[:2] != (512, 512):
        return []
    # Single-sourced from the family scene module (the nqueens renderer uses
    # the standard centered board geometry).
    cell, left, top = standard_board_geometry(size, size)
    masks: List[np.ndarray] = []
    inset = max(2, cell // 12)
    for row, col in fixed:
        if not (0 <= row < size and 0 <= col < size):
            continue
        mask = np.zeros((512, 512), dtype=bool)
        x0 = left + col * cell + inset
        y0 = top + row * cell + inset
        x1 = left + (col + 1) * cell - inset
        y1 = top + (row + 1) * cell - inset
        mask[y0:y1, x0:x1] = True
        masks.append(mask)
    return masks


def _nqueens_fixed_cell_mutation_fractions(
    sample: CanonicalSample,
    candidate: np.ndarray,
    puzzle: np.ndarray,
) -> List[float]:
    if candidate.shape != puzzle.shape:
        return [1.0] if sample.task_name == "nqueens_completion" else []
    source_error = np.max(visual_delta(candidate, puzzle), axis=2)
    return [
        float(np.mean(source_error[mask] > _PIXEL_CHANGE_THRESHOLD))
        for mask in _nqueens_fixed_cell_masks(sample, candidate.shape)
        if np.any(mask)
    ]


def _candidate_expected_arrays(
    sample: CanonicalSample,
    payload: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_image = render_candidate_terminal(sample, payload)
    if expected_image is None:
        raise ValueError("visible renderer state is insufficient for independent terminal reconstruction")
    expected = np.asarray(expected_image.convert("RGB"), dtype=np.uint8)[:, :, ::-1].copy()
    puzzle = _read_bgr(Path(sample.primary_input_image).expanduser().resolve())
    if puzzle.shape != expected.shape:
        raise ValueError("independently rendered terminal and source puzzle do not share the 512x512 geometry")
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for Constraint Grid verification") from exc
    delta = np.max(np.abs(expected.astype(np.int16) - puzzle.astype(np.int16)), axis=2)
    dynamic = (delta > 8).astype(np.uint8)
    dynamic = cv2.dilate(dynamic, np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
    if int(dynamic.sum()) < 16:
        raise ValueError("independently reconstructed terminal has no answer-bearing visual region")
    return puzzle, expected, dynamic


def _decode_video(video_path: Path) -> Tuple[List[np.ndarray], float]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for Constraint Grid verification") from exc
    capture = cv2.VideoCapture(video_path.as_posix())
    if not capture.isOpened():
        raise ValueError("candidate video cannot be decoded")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError("candidate video contains no decodable frames")
    return frames, fps


def verify_constraint_progressive_video(
    sample: CanonicalSample,
    candidate_video: Path,
) -> Tuple[Dict[str, Any], bool]:
    """Verify the whole ordered-commit visual program.

    The contract is: source hold; answer-bearing marks then accumulate through
    one or more element-local fade commits — every answer pixel moves
    monotonically from its source value to its terminal value, committed marks
    never regress or vanish, and no commit lands as a hard cut — then a
    terminal hold whose decoded state is an independently validated legal
    witness, with preserved non-answer content throughout.  A forced chain
    presented in causal waves and a single simultaneous commit are both legal
    shapes of this program; arbitrary re-grouping of exchangeable commits is
    deliberately accepted.
    """
    started_at = perf_counter()
    frames: List[np.ndarray] = []
    fps = 0.0
    media_decode_error = ""
    try:
        frames, fps = _decode_video(candidate_video)
    except Exception as exc:
        media_decode_error = str(exc)

    decoder_metrics: Dict[str, Any] = {}
    terminal_decode_ok = False
    terminal_decode_reason = media_decode_error or "candidate video contains no terminal frame"
    payload: Any = None
    if frames:
        terminal_frame = np.median(np.stack(frames[-min(5, len(frames)):]), axis=0).astype(np.uint8)
        payload, decoder_metrics, terminal_decode_ok, terminal_decode_reason = decode_constraint_terminal_payload(
            sample,
            terminal_frame,
        )

    puzzle = expected = np.zeros((0, 0, 3), dtype=np.uint8)
    dynamic = np.zeros((0, 0), dtype=bool)
    render_available = False
    render_error = ""
    if terminal_decode_ok:
        try:
            puzzle, expected, dynamic = _candidate_expected_arrays(sample, payload)
            render_available = True
        except Exception as exc:
            render_error = str(exc)

    shapes_ok = bool(render_available and all(frame.shape == expected.shape for frame in frames))
    media_ok = bool(
        frames
        and shapes_ok
        and abs(fps - 16.0) <= 0.05
        and len(frames) % 4 == 1
        and len(frames) <= 161
    )

    progresses: List[float] = []
    static_mutations: List[float] = []
    static_region_maxima: List[float] = []
    fixed_cell_mutations_by_frame: List[List[float]] = []
    input_distances: List[float] = []
    final_distances: List[float] = []
    max_pixel_regression_fraction = 0.0
    max_hard_step_fraction = 0.0
    max_envelope_violation_fraction = 0.0
    if render_available and shapes_ok:
        source = visual_features(puzzle)
        # The decoded terminal witness has already been checked against the
        # source rules. Its observed stroke appearance supplies the opacity
        # endpoint only; terminal correctness below still uses the independent
        # renderer. This prevents a slightly softer star or displaced edge from
        # looking like an illegal color excursion while it fades in.
        target = visual_features(terminal_frame)
        direction = target[dynamic] - source[dynamic]
        denominator = float(np.sum(direction * direction))
        static = ~dynamic
        lower = np.minimum(source, target) - _ENVELOPE_TOLERANCE
        upper = np.maximum(source, target) + _ENVELOPE_TOLERANCE
        previous_distance: np.ndarray | None = None
        previous_frame: np.ndarray | None = None
        for frame in frames:
            current = visual_features(frame)
            observed = current[dynamic] - source[dynamic]
            progress = float(np.sum(observed * direction) / denominator) if denominator > 1e-9 else 0.0
            progresses.append(progress)
            source_error = np.max(visual_delta(frame, puzzle), axis=2)
            static_mutations.append(float(np.mean(source_error[static] > _PIXEL_CHANGE_THRESHOLD)) if np.any(static) else 0.0)
            static_region_maxima.append(
                static_region_errors(frame, puzzle, static)[0] if np.any(static) else 0.0
            )
            fixed_cell_mutations_by_frame.append(
                _nqueens_fixed_cell_mutation_fractions(sample, frame, puzzle)
            )
            input_distances.append(float(np.sqrt(np.mean(visual_delta(frame, puzzle)[dynamic] ** 2)) / 255.0))
            final_distances.append(float(np.sqrt(np.mean(visual_delta(frame, expected)[dynamic] ** 2)) / 255.0))

            # Per-pixel process checks over the answer-bearing region.
            envelope_violation = np.max((current < lower) | (current > upper), axis=2)
            max_envelope_violation_fraction = max(
                max_envelope_violation_fraction,
                float(np.mean(envelope_violation[dynamic])),
            )
            distance_to_target = np.max(np.abs(current - target), axis=2)[dynamic]
            if previous_distance is not None:
                regression = distance_to_target > previous_distance + _PIXEL_REGRESSION_TOLERANCE
                max_pixel_regression_fraction = max(
                    max_pixel_regression_fraction, float(np.mean(regression))
                )
            previous_distance = distance_to_target
            if previous_frame is not None:
                step = np.max(np.abs(current - previous_frame), axis=2)[dynamic]
                max_hard_step_fraction = max(
                    max_hard_step_fraction, float(np.mean(step > _HARD_STEP_DELTA))
                )
            previous_frame = current

    monotonic = bool(progresses) and all(right + 0.035 >= left for left, right in zip(progresses, progresses[1:]))
    max_jump = max((right - left for left, right in zip(progresses, progresses[1:])), default=1.0)
    continuous = bool(
        max_jump <= _MAX_PROGRESS_JUMP
        and max_hard_step_fraction <= _MAX_HARD_STEP_FRACTION
    )
    accumulation_ok = bool(
        progresses
        and max_pixel_regression_fraction <= _MAX_PIXEL_REGRESSION_FRACTION
        and max_envelope_violation_fraction <= _MAX_ENVELOPE_VIOLATION_FRACTION
    )
    initial_hold = bool(
        len(progresses) >= 5
        and all(progress <= 0.06 for progress in progresses[:5])
        and all(distance <= 0.035 for distance in input_distances[:5])
    )
    final_hold = bool(
        len(progresses) >= 5
        # The terminal witness is decoded and independently validated above.
        # A softened, fully visible thin mark need not project to the exact
        # opacity of the raster renderer. Judge its held appearance directly.
        and all(distance <= 0.055 for distance in final_distances[-5:])
        and all(float(np.mean(visual_delta(frame, frames[-1])[dynamic])) / 255.0 <= .015
                for frame in frames[-5:])
    )
    max_static_mutation = max(static_mutations, default=1.0)
    max_static_region = max(static_region_maxima, default=0.0)
    fixed_identity_max = max(
        (value for frame_values in fixed_cell_mutations_by_frame for value in frame_values),
        default=(1.0 if sample.task_name == "nqueens_completion" else 0.0),
    )
    fixed_identity_ok = bool(
        sample.task_name != "nqueens_completion"
        or fixed_identity_max <= _MAX_NQUEENS_FIXED_CELL_MUTATION_FRACTION
    )
    preservation_ok = bool(
        max_static_mutation <= _MAX_STATIC_MUTATION_FRACTION
        and max_static_region <= STATIC_REGION_FLOOR
        and fixed_identity_ok
    )
    # The terminal decoder reads answer-bearing variables, not fixed clues or
    # the rest of the scene. Validate preservation in the final hold separately
    # from earlier frames so a repaired process defect cannot erase a correct
    # terminal, and a correct answer cannot excuse a corrupted terminal scene.
    terminal_preservation_ok = bool(
        shapes_ok
        and len(static_mutations) >= 5
        and max(static_mutations[-5:]) <= _MAX_STATIC_MUTATION_FRACTION
        and max(static_region_maxima[-5:]) <= STATIC_REGION_FLOOR
        and (sample.task_name != "nqueens_completion" or max(
            (value for values in fixed_cell_mutations_by_frame[-5:] for value in values),
            default=1.0,
        ) <= _MAX_NQUEENS_FIXED_CELL_MUTATION_FRACTION)
    )
    terminal_ok = bool(terminal_decode_ok and final_hold and terminal_preservation_ok)
    process_ok = bool(monotonic and continuous and accumulation_ok and initial_hold and final_hold)
    passed = bool(terminal_decode_ok and render_available and media_ok and process_ok and preservation_ok)
    if not frames:
        reason, failures = f"candidate video decoding failed: {media_decode_error}", ["media_contract"]
    elif not terminal_decode_ok:
        reason, failures = f"terminal video frame does not decode to a legal source-derived witness: {terminal_decode_reason}", ["semantic_final_state"]
    elif not render_available:
        reason, failures = f"independent terminal reconstruction failed: {render_error}", ["source_replay"]
    elif not media_ok:
        reason, failures = "video must be 512x512, 16fps, at most 10 seconds, and use a 4N+1 frame count", ["media_contract"]
    elif not initial_hold or not final_hold:
        reason, failures = "video does not hold the unchanged source and completed terminal states", ["temporal_process"]
    elif not monotonic or not accumulation_ok:
        reason, failures = "committed answer marks regress, vanish, or leave the source-to-terminal envelope", ["temporal_process"]
    elif not continuous:
        reason, failures = "an answer commit lands as a hard cut instead of an element-local fade", ["temporal_process"]
    elif not preservation_ok:
        reason, failures = "fixed clues, grid, labels, background, or camera mutate during the video", ["fixed_content_preservation"]
    else:
        reason, failures = "ok", []
    component_scores = {
        "media_validity_score": float(media_ok),
        "source_replay_score": float(terminal_decode_ok and render_available),
        "temporal_process_score": float(process_ok),
        "fixed_content_score": max(0.0, 1.0 - max_static_mutation / _MAX_STATIC_MUTATION_FRACTION),
        "final_state_score": float(terminal_ok),
    }
    return {
        "video_candidate_frame_count": len(frames),
        "video_fps": fps,
        "video_dynamic_pixel_count": int(dynamic.sum()),
        "video_progress": progresses,
        "video_max_progress_jump": max_jump,
        "video_progress_monotonic": monotonic,
        "video_update_continuous": continuous,
        "video_commit_accumulation_pass": accumulation_ok,
        "video_max_pixel_regression_fraction": max_pixel_regression_fraction,
        "video_max_hard_step_fraction": max_hard_step_fraction,
        "video_max_envelope_violation_fraction": max_envelope_violation_fraction,
        "video_initial_hold": initial_hold,
        "video_final_hold": final_hold,
        "video_max_static_mutation_fraction": max_static_mutation,
        "video_max_static_region_error": max_static_region,
        "video_static_region_bound": STATIC_REGION_FLOOR,
        "video_nqueens_fixed_cell_mutation_fractions": fixed_cell_mutations_by_frame,
        "video_nqueens_fixed_identity_max_mutation_fraction": fixed_identity_max,
        "video_nqueens_fixed_identity_preservation_pass": fixed_identity_ok,
        "video_terminal_preservation_pass": terminal_preservation_ok,
        "whole_video_process_correct": bool(process_ok and preservation_ok),
        "video_final_state_correct": terminal_ok,
        "video_semantic_process_check_pass": process_ok,
        "video_terminal_payload_decode_pass": terminal_decode_ok,
        "video_terminal_payload_decode_reason": terminal_decode_reason,
        **decoder_metrics,
        **_verifier_contract_metrics(
            verifier_profile="constraint_grid_progressive_commit_full_video_v1",
            artifact_kind="full_video",
            component_scores=component_scores,
            binary_correct=passed,
            failure_reason=reason,
            started_at=started_at,
            hard_gate_failures=failures,
        ),
    }, passed


__all__ = ["verify_constraint_progressive_video"]
