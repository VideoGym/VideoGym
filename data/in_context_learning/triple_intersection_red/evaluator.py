"""Task-owned source replay and visual observations."""

from __future__ import annotations

import math
import numpy as np
from data.common.verifiers.media import _align_bgr, _read_bgr, _video_fps, _video_frames_bgr
from data.common.verifiers.metrics import _verifier_contract_metrics
from data.common.verifiers.observations import (
    _analyze_fixed_visual_state_sequence,
    _pixels_near_rgb,
)
from data.common.verifiers.source import _semantic_metric
from data.common.verifiers.region_gates import (
    STATIC_REGION_FLOOR, region_delivery_gate, region_ink_gate, static_region_errors,
)
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional, Sequence


_TRIPLE_INTERSECTION_VERIFIER_PROFILE = "triple_intersection_complete_footprint_fade_coordinate_and_region_replay_v6"


_TRIPLE_INTERSECTION_UNITS_TO_PIXELS = 16.0


def _validate_triple_intersection_red(structured: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    replay = _triple_intersection_replay(structured)
    if replay is None:
        return _semantic_metric(False, fail_reason="triple-intersection centroid mismatch")
    return _semantic_metric(
        True,
        fail_reason="triple-intersection centroid mismatch",
        video_expected_centroid=replay["centroid"],
        video_triple_area=replay["triple_area"],
    )


def _triple_intersection_replay(structured: dict[str, Any]) -> Optional[dict[str, Any]]:
    raw_circles = structured.get("circles")
    if not isinstance(raw_circles, list) or len(raw_circles) != 3:
        return None
    try:
        circles = [(float(cx), float(cy), float(radius)) for cx, cy, radius in raw_circles]
    except Exception:
        return None
    radii = [radius for _cx, _cy, radius in circles]
    if (
        max(radii) - min(radii) > 1e-9
        or any(radius <= 0 for _cx, _cy, radius in circles)
        or len({(cx, cy) for cx, cy, _radius in circles}) != 3
        or any(
            abs(cx / 4.0 - round(cx / 4.0)) > 1e-9
            or abs(cy / 4.0 - round(cy / 4.0)) > 1e-9
            for cx, cy, _radius in circles
        )
        or any(
            (cx - radius) * _TRIPLE_INTERSECTION_UNITS_TO_PIXELS < 0
            or (cy - radius) * _TRIPLE_INTERSECTION_UNITS_TO_PIXELS < 0
            or (cx + radius) * _TRIPLE_INTERSECTION_UNITS_TO_PIXELS >= 512
            or (cy + radius) * _TRIPLE_INTERSECTION_UNITS_TO_PIXELS >= 512
            for cx, cy, radius in circles
        )
    ):
        return None
    yy, xx = np.ogrid[:512, :512]
    masks = [
        (xx - cx * _TRIPLE_INTERSECTION_UNITS_TO_PIXELS) ** 2
        + (yy - cy * _TRIPLE_INTERSECTION_UNITS_TO_PIXELS) ** 2
        <= (radius * _TRIPLE_INTERSECTION_UNITS_TO_PIXELS) ** 2
        for cx, cy, radius in circles
    ]
    triple = masks[0] & masks[1] & masks[2]
    ys, xs = np.nonzero(triple)
    if len(xs) == 0:
        return None
    center_x = sum(cx for cx, _cy, _radius in circles) / 3.0
    center_y = sum(cy for _cx, cy, _radius in circles) / 3.0
    centroid = [int(round(center_x)), int(round(center_y))]
    if (
        abs(center_x - centroid[0]) > 1e-8
        or abs(center_y - centroid[1]) > 1e-8
        or any(
            (center_x - cx) ** 2 + (center_y - cy) ** 2 > radius ** 2 + 1e-8
            for cx, cy, radius in circles
        )
    ):
        return None
    contributions = [
        int((masks[(index + 1) % 3] & masks[(index + 2) % 3] & ~masks[index]).sum())
        for index in range(3)
    ]
    triple_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if (
        min(contributions, default=0) < 2000
    ):
        return None
    return {
        "circles": circles,
        "centroid": centroid,
        "triple_mask": triple,
        "triple_area": int(triple.sum()),
        "triple_bbox": triple_bbox,
        "state_trace": ["input", "triple_final"],
    }


def _triple_intersection_render_language() -> tuple[
    tuple[int, int, int], float, tuple[int, int, int], tuple[int, int, int]
]:
    """Import the renderer's fill base colour, alpha, background, and grid.

    The marked region uses the family set-cluster flat translucent red.  Its
    observed colour depends on the pixels underneath, so the verifier derives
    per-pixel expectations from the renderer's own constants instead of
    duplicating a hard-coded observed RGB that can drift.
    """

    from data.in_context_learning.triple_intersection_red.generator import (
        _BG,
        _GRID,
        _TRIPLE_ALPHA,
        _TRIPLE_FILL,
    )

    return tuple(_TRIPLE_FILL), float(_TRIPLE_ALPHA) / 255.0, tuple(_BG), tuple(_GRID)


def _triple_intersection_expected_fill_bgr(base_bgr: np.ndarray) -> np.ndarray:
    """Expected observed colour after the translucent fill covers ``base_bgr``."""

    fill_rgb, alpha, _bg, _grid = _triple_intersection_render_language()
    fill = np.asarray(fill_rgb[::-1], dtype=np.float64)
    return base_bgr.astype(np.float64) * (1.0 - alpha) + fill[None, None, :] * alpha


def _triple_intersection_fill_pixels(
    frame_bgr: np.ndarray,
    *,
    base_bgr: np.ndarray,
    tolerance: float = 30.0,
    min_direction: float = 30.0,
) -> np.ndarray:
    """Pixels whose colour matches the renderer-derived translucent fill.

    Only pixels whose expected blend moves far enough from their source colour
    are classified: where the fill would be pixel-indistinguishable from the
    underlying ink (for example the orange outline), presence or absence of a
    translucent overlay cannot be decided from one raster and is not scored.
    """

    expected = _triple_intersection_expected_fill_bgr(base_bgr)
    direction = np.max(np.abs(expected - base_bgr.astype(np.float64)), axis=2)
    distance = np.sqrt(np.sum((frame_bgr.astype(np.float64) - expected) ** 2, axis=2))
    return (direction >= min_direction) & (distance <= tolerance)


def _triple_intersection_boundary_band(replay: dict[str, Any]) -> np.ndarray:
    yy, xx = np.ogrid[:512, :512]
    boundary_band = np.zeros((512, 512), dtype=bool)
    for cx, cy, radius in replay["circles"]:
        cx_px = cx * _TRIPLE_INTERSECTION_UNITS_TO_PIXELS
        cy_px = cy * _TRIPLE_INTERSECTION_UNITS_TO_PIXELS
        radius_px = radius * _TRIPLE_INTERSECTION_UNITS_TO_PIXELS
        distance = np.sqrt((xx - cx_px) ** 2 + (yy - cy_px) ** 2)
        boundary_band |= np.abs(distance - radius_px) <= 6.0
    return boundary_band


def _triple_intersection_fillable_core(
    semantic_mask: np.ndarray,
    *,
    replay: dict[str, Any],
    input_frame_bgr: np.ndarray,
) -> np.ndarray:
    """Return semantic fill pixels not occluded by fixed source ink.

    Circle outlines, centre dots, and their short labels are deliberately
    redrawn above the region fill.  Counting those fixed pixels as missing
    fill disproportionately penalized the smallest legal intersections (a
    reference radius-six sample reached only 0.72 apparent coverage).  The
    input frame independently identifies pixels that are background or grid
    and therefore actually available to receive the fill; no saved solution
    mask or authored target is consulted.
    """

    _fill, _alpha, background_rgb, grid_rgb = _triple_intersection_render_language()
    boundary_band = _triple_intersection_boundary_band(replay)
    background = _pixels_near_rgb(input_frame_bgr, background_rgb, tolerance=20.0)
    grid = _pixels_near_rgb(input_frame_bgr, grid_rgb, tolerance=20.0)
    return semantic_mask & ~boundary_band & (background | grid)


def _measure_triple_intersection_complete_footprint_fade(
    frames: Sequence[np.ndarray],
    *,
    source_frame_bgr: np.ndarray,
    triple_core: np.ndarray,
) -> dict[str, Any]:
    """Measure one uniform opacity trajectory over the full common lens.

    The target color and mutable region come from task semantics, not saved
    answer media.  Per-pixel source-to-target projection makes a center-out
    mask, directional wipe, scan, scale-in, or one-frame commit distinguishable
    from a complete fixed-footprint fade even when all variants end correctly.
    """

    source = source_frame_bgr.astype(np.float64)
    # The terminal colour is the renderer's translucent blend over each source
    # pixel, so the per-pixel fade direction is alpha * (fill - source).
    direction = _triple_intersection_expected_fill_bgr(source_frame_bgr) - source
    endpoint_magnitude = np.max(np.abs(direction), axis=2)
    active = triple_core & (endpoint_magnitude >= 12.0)
    active_pixels = int(active.sum())
    denominator = np.sum(direction * direction, axis=2)
    endpoint_energy = float(np.sum(direction[active] ** 2)) if active_pixels else 0.0
    if active_pixels < 64 or endpoint_energy <= 1.0:
        return {
            "complete_footprint_fade_ok": False,
            "active_pixels": active_pixels,
            "progress": [],
            "q10": [],
            "q90": [],
            "spread": [],
            "footprint_support": [],
            "linear_fit_error": [],
            "partial_frame_indices": [],
            "distinct_partial_opacity_levels": 0,
            "progress_monotonic": False,
            "projection_bounded": False,
            "source_endpoint_ok": False,
            "terminal_endpoint_ok": False,
            "partial_spread_ok": False,
            "partial_support_ok": False,
            "partial_linear_fit_ok": False,
        }

    progress_values: List[float] = []
    q01_values: List[float] = []
    q10_values: List[float] = []
    q90_values: List[float] = []
    q99_values: List[float] = []
    spread_values: List[float] = []
    support_values: List[float] = []
    fit_errors: List[float] = []
    support_threshold = np.maximum(3.0, endpoint_magnitude * 0.07)
    for frame in frames:
        delta = frame.astype(np.float64) - source
        raw_projection = np.sum(delta * direction, axis=2) / np.maximum(1.0, denominator)
        raw_values = raw_projection[active]
        clipped = np.clip(raw_values, 0.0, 1.0)
        progress = float(np.mean(clipped))
        q01, q10, q90, q99 = np.quantile(raw_values, (0.01, 0.10, 0.90, 0.99))
        residual = delta - progress * direction
        progress_values.append(progress)
        q01_values.append(float(q01))
        q10_values.append(float(np.clip(q10, 0.0, 1.0)))
        q90_values.append(float(np.clip(q90, 0.0, 1.0)))
        q99_values.append(float(q99))
        spread_values.append(float(np.clip(q90, 0.0, 1.0) - np.clip(q10, 0.0, 1.0)))
        support_values.append(float(
            np.mean(np.max(np.abs(delta), axis=2)[active] >= support_threshold[active])
        ))
        fit_errors.append(math.sqrt(float(np.sum(residual[active] ** 2)) / endpoint_energy))

    partial_indices = [
        index for index, value in enumerate(progress_values)
        if 0.07 < value < 0.93
    ]
    distinct_levels = len({round(progress_values[index], 1) for index in partial_indices})
    monotonic = all(
        right + 0.035 >= left
        for left, right in zip(progress_values, progress_values[1:])
    )
    # The public answer calls for a translucent red region, not the renderer's
    # exact alpha. Final mask/color parsing separately bounds that appearance;
    # a slightly stronger legal red must not fail an exact-opacity ceiling.
    bounded = bool(min(q01_values, default=-1.0) >= -0.16)
    source_endpoint_ok = bool(progress_values and max(progress_values[:3]) <= 0.04)
    terminal_endpoint_ok = bool(progress_values and min(progress_values[-3:]) >= 0.94)
    partial_spread_ok = bool(
        partial_indices
        and max(spread_values[index] for index in partial_indices) <= 0.22
    )
    partial_support_ok = bool(
        partial_indices
        and min(support_values[index] for index in partial_indices) >= 0.82
    )
    partial_linear_fit_ok = bool(
        partial_indices
        and max(fit_errors[index] for index in partial_indices) <= 0.22
    )
    complete_footprint_fade_ok = bool(
        len(partial_indices) >= 3
        and distinct_levels >= 3
        and monotonic
        and bounded
        and source_endpoint_ok
        and terminal_endpoint_ok
        and partial_spread_ok
        and partial_support_ok
        and partial_linear_fit_ok
    )
    return {
        "complete_footprint_fade_ok": complete_footprint_fade_ok,
        "active_pixels": active_pixels,
        "progress": progress_values,
        "q10": q10_values,
        "q90": q90_values,
        "spread": spread_values,
        "footprint_support": support_values,
        "linear_fit_error": fit_errors,
        "partial_frame_indices": partial_indices,
        "distinct_partial_opacity_levels": distinct_levels,
        "progress_monotonic": monotonic,
        "projection_bounded": bounded,
        "source_endpoint_ok": source_endpoint_ok,
        "terminal_endpoint_ok": terminal_endpoint_ok,
        "partial_spread_ok": partial_spread_ok,
        "partial_support_ok": partial_support_ok,
        "partial_linear_fit_ok": partial_linear_fit_ok,
    }


def _verify_triple_intersection_candidate_state_video(
    video_path: Path,
    structured: dict[str, Any],
    puzzle_image: Path,
) -> tuple[dict[str, Any], bool]:
    started_at = perf_counter()
    semantic_metrics, semantic_pass = _validate_triple_intersection_red(structured)
    replay = _triple_intersection_replay(structured)
    if replay is None:
        reason = "missing or invalid triple-intersection metadata"
        scores = {
            "source_replay_score": 0.0,
            "mask_exactness_score": 0.0,
            "centroid_score": 0.0,
            "visual_parseability_score": 0.0,
            "state_order_score": 0.0,
            "transition_visibility_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
            "media_contract_score": 0.0,
        }
        return {
            **semantic_metrics,
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": reason,
            **_verifier_contract_metrics(
                verifier_profile=_TRIPLE_INTERSECTION_VERIFIER_PROFILE,
                artifact_kind="full_video",
                component_scores=scores,
                binary_correct=False,
                failure_reason=reason,
                started_at=started_at,
                hard_gate_failures=["source_replay"],
            ),
        }, False
    try:
        frames = _video_frames_bgr(video_path)
    except Exception as exc:
        scores = {
            "source_replay_score": 1.0, "mask_exactness_score": 0.0,
            "centroid_score": 1.0, "visual_parseability_score": 0.0,
            "state_order_score": 0.0, "transition_visibility_score": 0.0,
            "preservation_score": 0.0, "final_state_score": 0.0,
            "media_contract_score": 0.0,
        }
        reason = f"triple-intersection video could not be decoded: {exc}"
        return {**semantic_metrics, "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": reason, "video_candidate_frame_count": 0,
            "video_media_contract_valid": False,
            **_verifier_contract_metrics(
                verifier_profile=_TRIPLE_INTERSECTION_VERIFIER_PROFILE,
                artifact_kind="full_video", component_scores=scores, binary_correct=False,
                failure_reason=reason, started_at=started_at, hard_gate_failures=["video_decode"],
            )}, False
    if not frames or any(frame.shape[:2] != (512, 512) for frame in frames):
        return {
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": "triple-intersection video must remain 512x512",
        }, False

    source_frame = _align_bgr(_read_bgr(puzzle_image), frames[0].shape)
    source_delta = np.max(
        np.abs(frames[0].astype(np.int16) - source_frame.astype(np.int16)),
        axis=2,
    )
    source_changed_fraction = float((source_delta > 18).mean())
    # Raw pixel counts turn codec chroma fringes on all three circle outlines
    # into thousands of alleged scene changes. Verify each local region and
    # the complete source ink with the same raster tolerance used elsewhere.
    source_region_metrics, source_regions_ok = region_delivery_gate(
        frames[0], source_frame, prefix="video_triple_intersection_source_",
    )
    source_ink_metrics, source_ink_ok = region_ink_gate(
        frames[0][..., ::-1], source_frame[..., ::-1], prefix="video_triple_intersection_source_",
    )
    source_frame_ok = source_regions_ok and source_ink_ok

    triple = replay["triple_mask"]
    triple_core = _triple_intersection_fillable_core(
        triple,
        replay=replay,
        input_frame_bgr=frames[0],
    )
    if int(triple_core.sum()) <= 0:
        return {
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": "triple-intersection visible fill core is empty",
        }, False
    fade_metrics = _measure_triple_intersection_complete_footprint_fade(
        frames,
        source_frame_bgr=source_frame,
        triple_core=triple_core,
    )

    recognized: List[tuple[int, str]] = []
    static_mutation_fractions: List[float] = []
    static_local_errors: List[float] = []
    reference = frames[0].astype(np.int16)
    static_region = ~triple
    for frame_index, frame in enumerate(frames):
        red = _triple_intersection_fill_pixels(frame, base_bgr=source_frame)
        red_triple_coverage = float((red & triple_core).sum()) / float(triple_core.sum())
        red_outside = float((red & ~triple).sum()) / float(max(1, triple.sum()))
        red_fraction = float(red.sum()) / float(max(1, triple.sum()))
        fill_change = np.max(
            np.abs(frame.astype(np.int16) - reference),
            axis=2,
        ) > 8
        fill_change_fraction = float((fill_change & triple_core).sum()) / float(triple_core.sum())
        if (
            red_fraction <= 0.03
            and fill_change_fraction <= 0.03
        ):
            recognized.append((frame_index, "input"))
        elif (
            red_triple_coverage >= 0.82
            and red_outside <= 0.03
        ):
            recognized.append((frame_index, "triple_final"))

        changed = np.max(np.abs(frame.astype(np.int16) - reference), axis=2) > 26
        static_mutation_fractions.append(
            float((changed & static_region).sum()) / float(max(1, static_region.sum()))
        )
        static_local_errors.append(static_region_errors(frame, frames[0], static_region)[0])

    sequence = _analyze_fixed_visual_state_sequence(
        frame_count=len(frames),
        recognized=recognized,
        expected_states=["input", "triple_final"],
    )
    max_unrecognized_fraction = 0.35
    motion_gap_counts = [
        current["frame_indices"][0] - previous["frame_indices"][-1] - 1
        for previous, current in zip(sequence["compact_groups"], sequence["compact_groups"][1:])
    ]
    dynamic_steps: List[int] = []
    for frame_index, (previous, current) in enumerate(zip(frames, frames[1:])):
        delta = np.max(
            np.abs(current.astype(np.int16) - previous.astype(np.int16)),
            axis=2,
        )
        changed_core_fraction = float(((delta > 8) & triple_core).sum()) / float(triple_core.sum())
        # The complete overlap changes together.  A local sweep or one-pixel
        # trace is not the required simultaneous region transition.
        if changed_core_fraction >= 0.70:
            dynamic_steps.append(frame_index)
    transition_runs: List[List[int]] = []
    for frame_index in dynamic_steps:
        if not transition_runs or frame_index != transition_runs[-1][-1] + 1:
            transition_runs.append([frame_index])
        else:
            transition_runs[-1].append(frame_index)
    has_visible_transitions = bool(
        len(motion_gap_counts) == 1
        and all(gap >= 2 for gap in motion_gap_counts)
        and len(fade_metrics["partial_frame_indices"]) >= 3
        and fade_metrics["distinct_partial_opacity_levels"] >= 3
    )
    max_static_mutation_fraction = max(static_mutation_fractions, default=1.0)
    static_preserved = bool(
        max_static_mutation_fraction <= 0.003
        and max(static_local_errors, default=1.) <= STATIC_REGION_FLOOR
    )
    fps = _video_fps(video_path)
    media_contract_ok = bool(
        len(frames) > 0 and abs(fps - 16.0) <= 0.05
        and len(frames) % 4 == 1 and len(frames) / max(fps, 1e-9) <= 10.0
    )
    passed = bool(
        semantic_pass
        and source_frame_ok
        and sequence["unrecognized_fraction"] <= max_unrecognized_fraction
        and sequence["starts_at_input"]
        and sequence["ends_at_final"]
        and sequence["has_initial_hold"]
        and sequence["has_final_hold"]
        and sequence["exact_sequence"]
        and has_visible_transitions
        and fade_metrics["complete_footprint_fade_ok"]
        and static_preserved
        and media_contract_ok
    )
    if not source_frame_ok:
        reason = "triple-intersection video does not begin with the supplied source image"
    elif sequence["unrecognized_fraction"] > max_unrecognized_fraction:
        reason = "triple-intersection fill colors are not parseable in enough frames"
    elif not sequence["exact_sequence"]:
        reason = "triple-intersection states are incomplete, reordered, or regressive"
    elif not sequence["starts_at_input"] or not sequence["ends_at_final"]:
        reason = "triple-intersection input or final mask mismatch"
    elif not sequence["has_initial_hold"] or not sequence["has_final_hold"]:
        reason = "triple-intersection initial or final hold is missing"
    elif not has_visible_transitions:
        reason = "triple-intersection states use hard cuts instead of visible mask transitions"
    elif not fade_metrics["complete_footprint_fade_ok"]:
        reason = "the complete triple-intersection footprint must fade uniformly through multiple opacity levels"
    elif not static_preserved:
        reason = "triple-intersection circles, grid, labels, or background mutate"
    elif not media_contract_ok:
        reason = "triple-intersection video must be 16fps, at most 10 seconds, and use a 4N+1 frame count"
    else:
        reason = "ok"
    scores = {
        "source_replay_score": float(semantic_pass and source_frame_ok), "mask_exactness_score": float(sequence["ends_at_final"]),
        "centroid_score": 1.0, "visual_parseability_score": 1.0 - min(1.0, sequence["unrecognized_fraction"]),
        "state_order_score": float(sequence["exact_sequence"]),
        "transition_visibility_score": float(has_visible_transitions),
        "complete_footprint_fade_score": float(fade_metrics["complete_footprint_fade_ok"]),
        "preservation_score": float(static_preserved),
        "final_state_score": float(sequence["ends_at_final"] and sequence["has_final_hold"]),
        "media_contract_score": float(media_contract_ok),
    }
    failures = [] if passed else [
        "source_input_hold" if not source_frame_ok
        else "visual_parseability" if sequence["unrecognized_fraction"] > max_unrecognized_fraction
        else "state_order" if not sequence["exact_sequence"]
        else "visible_transitions" if not has_visible_transitions
        else "complete_footprint_local_fade" if not fade_metrics["complete_footprint_fade_ok"]
        else "fixed_content_preservation" if not static_preserved
        else "media_contract" if not media_contract_ok
        else "semantic_video"
    ]
    return {
        **semantic_metrics,
        "video_candidate_state_check_pass": passed,
        "video_candidate_state_reason": reason,
        "video_candidate_state_verifier": "triple_intersection_complete_footprint_fade_parser",
        "video_candidate_frame_count": len(frames),
        "video_candidate_recognized_frame_count": sequence["recognized_count"],
        "video_candidate_unrecognized_fraction": sequence["unrecognized_fraction"],
        "video_candidate_max_unrecognized_fraction": max_unrecognized_fraction,
        "video_triple_intersection_observed_states": sequence["observed_states"],
        "video_triple_intersection_exact_sequence": sequence["exact_sequence"],
        "video_triple_intersection_skipped_states": sequence["skipped_layers"],
        "video_triple_intersection_has_initial_hold": sequence["has_initial_hold"],
        "video_triple_intersection_has_final_hold": sequence["has_final_hold"],
        "video_triple_intersection_motion_gap_counts": motion_gap_counts,
        "video_triple_intersection_transition_runs": transition_runs,
        "video_triple_intersection_has_visible_transitions": has_visible_transitions,
        "video_triple_intersection_source_changed_fraction": source_changed_fraction,
        "video_triple_intersection_max_static_region_error": max(static_local_errors, default=1.),
        **source_region_metrics,
        **source_ink_metrics,
        "video_triple_intersection_source_frame_valid": source_frame_ok,
        "video_triple_intersection_max_static_mutation_fraction": max_static_mutation_fraction,
        "video_triple_intersection_complete_footprint_fade_pass": fade_metrics["complete_footprint_fade_ok"],
        "video_triple_intersection_fade_active_pixels": fade_metrics["active_pixels"],
        "video_triple_intersection_opacity_progress": fade_metrics["progress"],
        "video_triple_intersection_opacity_q10": fade_metrics["q10"],
        "video_triple_intersection_opacity_q90": fade_metrics["q90"],
        "video_triple_intersection_opacity_spread": fade_metrics["spread"],
        "video_triple_intersection_footprint_support": fade_metrics["footprint_support"],
        "video_triple_intersection_linear_fit_error": fade_metrics["linear_fit_error"],
        "video_triple_intersection_partial_frame_indices": fade_metrics["partial_frame_indices"],
        "video_triple_intersection_distinct_partial_opacity_levels": fade_metrics["distinct_partial_opacity_levels"],
        "video_triple_intersection_opacity_progress_monotonic": fade_metrics["progress_monotonic"],
        "video_triple_intersection_opacity_projection_bounded": fade_metrics["projection_bounded"],
        "video_triple_intersection_source_opacity_endpoint_valid": fade_metrics["source_endpoint_ok"],
        "video_triple_intersection_terminal_opacity_endpoint_valid": fade_metrics["terminal_endpoint_ok"],
        "video_triple_intersection_partial_opacity_spread_valid": fade_metrics["partial_spread_ok"],
        "video_triple_intersection_partial_footprint_support_valid": fade_metrics["partial_support_ok"],
        "video_triple_intersection_partial_linear_fit_valid": fade_metrics["partial_linear_fit_ok"],
        "video_fps": fps,
        "video_media_contract_valid": media_contract_ok,
        **_verifier_contract_metrics(
            verifier_profile=_TRIPLE_INTERSECTION_VERIFIER_PROFILE,
            artifact_kind="full_video", component_scores=scores, binary_correct=passed,
            failure_reason=reason, started_at=started_at, hard_gate_failures=failures,
        ),
    }, passed
