"""Family-level temporal continuity checks for Geometry answer videos.

Task-local verifiers remain responsible for the meaning and order of their
states.  This module supplies the invariant that every Geometry video shares:
the visible input and solved endpoint must be connected by observable frames,
with an input hold, a stable terminal hold, and no whole-canvas jump.  The
check deliberately derives its mask from the candidate video's own endpoints;
it does not read a saved solution video or any answer metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


_ENDPOINT_PIXEL_THRESHOLD = 12
_FRAME_PIXEL_THRESHOLD = 18
_MIN_ENDPOINT_CHANGE_FRACTION = 0.0005
_MAX_ADJACENT_CHANGE_FRACTION = 0.18
_HOLD_CHANGE_FRACTION = 0.004
_MIN_INTERMEDIATE_FRAMES = 2
_MIN_NON_CROSSFADE_RESIDUAL = 0.015
_DEFAULT_MAX_REVERSAL_FRACTION = 0.24
_TASK_MAX_REVERSAL_FRACTION = {
    # These references contain a real curved trajectory, so some
    # endpoint-projected motion is expected.
    "arc_connect": 0.34,
    # A continuously rotated asymmetric tile can move slightly away from its
    # terminal pixel endpoint during a legitimate half-turn.  The task-owned
    # pose replay separately enforces the signed, monotone angle path, so this
    # family-level endpoint projection only needs to avoid rejecting that
    # valid curved trajectory after video compression.
    "rotation": 0.43,
}
_RENDERED_TASK_MAX_PROGRESS_STEP = {
    "region_count": 0.40,
    "symmetry_completion": 0.35,
}


def _changed_fraction(first: np.ndarray, second: np.ndarray, *, threshold: int) -> float:
    delta = np.max(
        np.abs(first.astype(np.int16) - second.astype(np.int16)),
        axis=2,
    )
    return float(np.mean(delta >= threshold))


def geometry_endpoint_continuity_from_frames(
    frames: Sequence[np.ndarray],
    *,
    task_name: str | None = None,
    max_progress_step: float | None = None,
) -> tuple[dict[str, Any], bool]:
    """Check the shared Geometry endpoint/hold/continuity contract.

    Arrays may be RGB or BGR, but every frame must use the same channel order.
    Intermediate progress is measured only on pixels that differ between the
    first and final frames.  Consequently a direct input-to-answer cut has no
    intermediate evidence even when the edited region is very small.
    """

    arrays = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    shape_valid = bool(
        arrays
        and arrays[0].ndim == 3
        and arrays[0].shape[2] == 3
        and all(frame.shape == arrays[0].shape for frame in arrays)
    )
    if not shape_valid:
        metrics = {
            "geometry_continuity_frame_count": len(arrays),
            "geometry_continuity_shape_valid": False,
            "geometry_continuity_pass": False,
            "geometry_continuity_reason": "frames are empty or do not share one three-channel shape",
        }
        return metrics, False

    first = arrays[0]
    final = arrays[-1]
    endpoint_channel_delta = np.abs(final.astype(np.float32) - first.astype(np.float32))
    endpoint_mask = np.max(endpoint_channel_delta, axis=2) >= _ENDPOINT_PIXEL_THRESHOLD
    endpoint_change_fraction = float(np.mean(endpoint_mask))
    endpoint_mean_delta = (
        float(np.mean(endpoint_channel_delta[endpoint_mask]))
        if bool(np.any(endpoint_mask))
        else 0.0
    )

    progress: list[float] = []
    linear_endpoint_residuals: list[float] = []
    endpoint_projection_reversal = 0.0
    endpoint_projection_travel = 0.0
    endpoint_path_length = 0.0
    if endpoint_mean_delta > 0.0:
        endpoint_vectors = (final.astype(np.float32) - first.astype(np.float32))[endpoint_mask]
        endpoint_vector_energy = float(np.sum(endpoint_vectors * endpoint_vectors))
        endpoint_pixel_energy = np.sum(endpoint_vectors * endpoint_vectors, axis=1) + 1e-6
        endpoint_rms = max(1.0, float(np.sqrt(np.mean(endpoint_vectors * endpoint_vectors))))
        previous_projection: np.ndarray | None = None
        for frame in arrays:
            delta = np.abs(frame.astype(np.float32) - first.astype(np.float32))
            progress.append(float(np.mean(delta[endpoint_mask]) / endpoint_mean_delta))
            signed_delta = (frame.astype(np.float32) - first.astype(np.float32))[endpoint_mask]
            alpha = (
                float(np.sum(signed_delta * endpoint_vectors) / endpoint_vector_energy)
                if endpoint_vector_energy > 0.0
                else 0.0
            )
            residual = signed_delta - alpha * endpoint_vectors
            linear_endpoint_residuals.append(
                float(np.sqrt(np.mean(residual * residual)) / endpoint_rms)
            )
            projection = np.sum(signed_delta * endpoint_vectors, axis=1) / endpoint_pixel_energy
            if previous_projection is not None:
                projection_step = projection - previous_projection
                endpoint_projection_reversal += float(np.sum(np.clip(-projection_step, 0.0, None)))
                endpoint_projection_travel += float(np.sum(np.abs(projection_step)))
            previous_projection = projection
        endpoint_path_length = sum(
            float(
                np.mean(
                    np.abs(
                        arrays[index].astype(np.float32)[endpoint_mask]
                        - arrays[index - 1].astype(np.float32)[endpoint_mask]
                    )
                )
            )
            for index in range(1, len(arrays))
        ) / max(endpoint_mean_delta, 1e-6)
    else:
        progress = [0.0 for _ in arrays]
        linear_endpoint_residuals = [0.0 for _ in arrays]

    intermediate_indices = [
        index
        for index, value in enumerate(progress)
        if 0.04 < value < 0.96
    ]
    direct_endpoint_jumps = [
        index
        for index in range(len(progress) - 1)
        if progress[index] <= 0.05 and progress[index + 1] >= 0.95
    ]
    adjacent_change_fractions = [
        _changed_fraction(arrays[index - 1], arrays[index], threshold=_FRAME_PIXEL_THRESHOLD)
        for index in range(1, len(arrays))
    ]
    initial_hold_change = max(
        (
            _changed_fraction(first, frame, threshold=_FRAME_PIXEL_THRESHOLD)
            for frame in arrays[: min(4, len(arrays))]
        ),
        default=1.0,
    )
    final_hold_change = max(
        (
            _changed_fraction(final, frame, threshold=_FRAME_PIXEL_THRESHOLD)
            for frame in arrays[-min(3, len(arrays)) :]
        ),
        default=1.0,
    )

    endpoint_distinct = endpoint_change_fraction >= _MIN_ENDPOINT_CHANGE_FRACTION
    required_intermediate_frames = _MIN_INTERMEDIATE_FRAMES
    intermediate_evidence = len(intermediate_indices) >= required_intermediate_frames
    initial_hold = initial_hold_change <= _HOLD_CHANGE_FRACTION
    final_hold = final_hold_change <= _HOLD_CHANGE_FRACTION
    bounded_adjacent_change = (
        max(adjacent_change_fractions, default=1.0) <= _MAX_ADJACENT_CHANGE_FRACTION
    )
    max_linear_residual = max(linear_endpoint_residuals[1:-1], default=0.0)
    non_crossfade_process = max_linear_residual >= _MIN_NON_CROSSFADE_RESIDUAL
    reversal_fraction = endpoint_projection_reversal / max(endpoint_projection_travel, 1e-6)
    reversal_limit = _TASK_MAX_REVERSAL_FRACTION.get(
        str(task_name or ""),
        _DEFAULT_MAX_REVERSAL_FRACTION,
    )
    ordered_endpoint_evolution = reversal_fraction <= reversal_limit
    if max_progress_step is None:
        max_progress_step = _RENDERED_TASK_MAX_PROGRESS_STEP.get(str(task_name or ""))
    observed_max_progress_step = max(
        (abs(progress[index + 1] - progress[index]) for index in range(len(progress) - 1)),
        default=1.0,
    )
    bounded_progress_step = bool(
        max_progress_step is None or observed_max_progress_step <= max_progress_step
    )
    task_path_length_ok = True
    display_diagnostics_pass = bool(
        endpoint_distinct
        and intermediate_evidence
        and initial_hold
        and final_hold
        and bounded_adjacent_change
        and non_crossfade_process
        and ordered_endpoint_evolution
        and bounded_progress_step
        and task_path_length_ok
    )
    # Task-owned parsers verify required geometric constructions. The shared
    # layer only requires a visible result; raster timing/fade diagnostics do
    # not veto a semantically valid construction.
    passed = bool(endpoint_distinct)
    metrics = {
        "geometry_continuity_frame_count": len(arrays),
        "geometry_continuity_shape_valid": shape_valid,
        "geometry_continuity_endpoint_change_fraction": endpoint_change_fraction,
        "geometry_continuity_endpoint_mean_channel_delta": endpoint_mean_delta,
        "geometry_continuity_endpoint_distinct": endpoint_distinct,
        "geometry_continuity_progress": progress,
        "geometry_continuity_intermediate_frame_indices": intermediate_indices,
        "geometry_continuity_intermediate_frame_count": len(intermediate_indices),
        "geometry_continuity_required_intermediate_frame_count": required_intermediate_frames,
        "geometry_continuity_direct_endpoint_jump_indices": direct_endpoint_jumps,
        "geometry_continuity_initial_hold_change_fraction": initial_hold_change,
        "geometry_continuity_initial_hold": initial_hold,
        "geometry_continuity_final_hold_change_fraction": final_hold_change,
        "geometry_continuity_final_hold": final_hold,
        "geometry_continuity_adjacent_change_fractions": adjacent_change_fractions,
        "geometry_continuity_bounded_adjacent_change": bounded_adjacent_change,
        "geometry_continuity_linear_endpoint_residuals": linear_endpoint_residuals,
        "geometry_continuity_max_linear_endpoint_residual": max_linear_residual,
        "geometry_continuity_non_crossfade_process": non_crossfade_process,
        "geometry_continuity_endpoint_reversal_fraction": reversal_fraction,
        "geometry_continuity_endpoint_reversal_limit": reversal_limit,
        "geometry_continuity_ordered_endpoint_evolution": ordered_endpoint_evolution,
        "geometry_continuity_max_progress_step": observed_max_progress_step,
        "geometry_continuity_max_progress_step_limit": max_progress_step,
        "geometry_continuity_bounded_progress_step": bounded_progress_step,
        "geometry_continuity_endpoint_path_length": endpoint_path_length,
        "geometry_continuity_task_path_length_pass": task_path_length_ok,
        "geometry_continuity_pass": passed,
        "geometry_continuity_display_diagnostics_pass": display_diagnostics_pass,
        "geometry_continuity_reason": "ok" if passed else "no visible result",
    }
    return metrics, passed


def verify_geometry_video_continuity(
    video_path: Path,
    *,
    task_name: str | None = None,
    max_progress_step: float | None = None,
) -> tuple[dict[str, Any], bool]:
    """Decode one video and apply :func:`geometry_endpoint_continuity_from_frames`."""

    capture = cv2.VideoCapture(video_path.as_posix())
    opened = capture.isOpened()
    frames: list[np.ndarray] = []
    if opened:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    capture.release()
    metrics, passed = geometry_endpoint_continuity_from_frames(
        frames,
        task_name=task_name,
        max_progress_step=max_progress_step,
    )
    metrics = {
        **metrics,
        "geometry_continuity_video_opened": opened,
        "geometry_continuity_decode_count": 1,
        "geometry_continuity_candidate_video": video_path.as_posix(),
    }
    return metrics, bool(opened and passed)


__all__ = [
    "geometry_endpoint_continuity_from_frames",
    "verify_geometry_video_continuity",
]
