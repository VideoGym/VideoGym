from __future__ import annotations

import math
import re
import time
from itertools import permutations
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import cv2
import numpy as np
from PIL import Image, ImageDraw

from data.common.render_primitives import (
    draw_smooth_polyline,
    finish_scaled_canvas,
    scale_stroke_width,
    stroke_polyline,
)
from data.common.verifiers.candidates import (
    resolve_source_record,
    resolve_source_record_id,
    resolve_source_task_dir,
)
from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.keyframes import visual_delta
from data.common.paths import first_existing_path
from data.common.typed_answers import FIVE_CHOICE_CODEC
from data.common.progress_alignment import (
    # The jump ceiling is read rather than restated: the process-order reading
    # below has to know the largest step the shared gate can ever allow, in order
    # to decline a source it could not walk, and a second copy of the number
    # would be free to drift away from the one that decides.
    _INHERITED_JUMP_CEILING,
    read_process_order,
    tone_offset,
)
from data.common.verifiers.region_gates import (
    REGION,
    STATIC_REGION_FLOOR,
    static_region_errors,
)
from data.geometry.core.point_target_base import (
    PointTargetImageEvaluator,
    PointTargetPuzzleGenerator,
    _explicit_prompt_section,
)
from data.geometry.core.coordinate_replay import (
    CoordinateReplayError,
    derive_coordinate_answer,
)
from data.geometry.core.video_continuity import verify_geometry_video_continuity
from data.geometry.core.phase_observation import phase_onset, selection_onset, serial_construction_observation
from data.geometry.core.construction_paths import (
    NO_CONSTRUCTION_TASKS,
    serial_construction_path_variants,
    terminal_attribution_paths,
    terminal_construction_paths,
)
from data.geometry.core.terminal_scene import (
    final_marker_state_gate,
    observed_added_ink,
    point_target_delivery_ink_state,
    point_target_foreign_ink_measurement,
    verify_point_target_static_regions,
)
from data.geometry.arc_connect.evaluator import (
    ARC_CONNECT_VERIFIER_PROFILE,
    replay_arc_connect_geometry,
)
from data.geometry.circle_tangent_line.evaluator import (
    CIRCLE_TANGENT_LINE_VERIFIER_PROFILE,
    replay_circle_tangent_line_geometry,
)
from data.geometry.circle_tangent_point.evaluator import (
    replay_circle_tangent_point_geometry,
    verify_circle_tangent_point_full_video,
)
from data.geometry.circumcenter.evaluator import (
    foreign_ink_clearance,
    replay_circumcenter_geometry,
    renderer_line_width,
    verify_circumcenter_full_video,
)
from data.geometry.incenter.evaluator import (
    replay_incenter_geometry,
    verify_incenter_full_video,
)
from data.geometry.isosceles_trapezoid.evaluator import (
    replay_isosceles_trapezoid_geometry,
    verify_isosceles_trapezoid_full_video,
)
from data.geometry.midpoint.evaluator import verify_midpoint_full_video
from data.geometry.orthocenter.evaluator import (
    replay_orthocenter_geometry,
    verify_orthocenter_full_video,
)
from data.geometry.parallel.evaluator import (
    replay_parallel_geometry,
    verify_parallel_full_video,
)


SEMANTIC_COORDINATE_TARGET_VIDEO_TASKS = {
    "angle_bisector",
    "arc_connect",
    "circle_tangent_line",
    "circle_tangent_point",
    "circumcenter",
    "incenter",
    "isosceles_trapezoid",
    "midpoint",
    "orthocenter",
    "parallel",
}
DEFAULT_CANDIDATE_VIDEOS = ("generated.webm", "output.webm", "prediction.webm", "result.webm")
ANGLE_BISECTOR_VERIFIER_PROFILE = "angle_bisector_multimodal_semantic_hard_gate_v1"
ANGLE_BISECTOR_STATE_TRACE = [
    "input_hold",
    "bisector_vertex_to_candidate_growth",
    "correct_candidate_highlight",
    "final_hold",
]
def _angle_distance(angle1: float, angle2: float) -> float:
    return abs((angle2 - angle1 + np.pi) % (2.0 * np.pi) - np.pi)


def _angle_close(value: object, expected: float, *, tolerance: float = 1e-5) -> bool:
    try:
        actual = float(value)
    except (TypeError, ValueError):
        return False
    delta = abs((actual - expected + 180.0) % 360.0 - 180.0)
    return delta <= tolerance


def _point_pair(value: object) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(component) for component in point):
        return None
    return point


def _angle_bisector_replay(record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Recompute the answer from source geometry and audit all redundant metadata."""

    structured = record.get("structured_answer")
    metrics: Dict[str, Any] = {
        "angle_bisector_geometry_replay_valid": False,
        "angle_bisector_geometry_replay_reason": "invalid structured geometry",
        "verifier_profile": ANGLE_BISECTOR_VERIFIER_PROFILE,
    }
    if not isinstance(structured, dict):
        return metrics, False
    vertex = _point_pair(structured.get("vertex"))
    endpoints_raw = structured.get("ray_endpoints")
    if vertex is None or not isinstance(endpoints_raw, list) or len(endpoints_raw) != 2:
        return metrics, False
    endpoints = [_point_pair(value) for value in endpoints_raw]
    if any(value is None for value in endpoints):
        return metrics, False
    endpoint1, endpoint2 = endpoints  # type: ignore[misc]
    vectors = [
        (endpoint1[0] - vertex[0], endpoint1[1] - vertex[1]),
        (endpoint2[0] - vertex[0], endpoint2[1] - vertex[1]),
    ]
    ray_lengths = [float(np.hypot(dx, dy)) for dx, dy in vectors]
    if min(ray_lengths) < 80.0:
        metrics["angle_bisector_geometry_replay_reason"] = "degenerate ray segment"
        return metrics, False
    ray_angles = [float(np.arctan2(dy, dx) % (2.0 * np.pi)) for dx, dy in vectors]
    signed_delta = float((ray_angles[1] - ray_angles[0] + np.pi) % (2.0 * np.pi) - np.pi)
    opening = abs(signed_delta)
    if not np.deg2rad(54.0) - 1e-6 <= opening <= np.deg2rad(112.0) + 1e-6:
        metrics["angle_bisector_geometry_replay_reason"] = "smaller opening outside task bounds"
        return metrics, False
    bisector = float((ray_angles[0] + signed_delta / 2.0) % (2.0 * np.pi))

    candidates_raw = structured.get("candidate_centers")
    if not isinstance(candidates_raw, list) or len(candidates_raw) != 5:
        return metrics, False
    candidates: List[Tuple[str, Tuple[float, float]]] = []
    for item in candidates_raw:
        if not isinstance(item, dict):
            return metrics, False
        label = str(item.get("label") or "")
        center = _point_pair(item.get("center"))
        if label not in "ABCDE" or center is None:
            return metrics, False
        candidates.append((label, center))
    if sorted(label for label, _center in candidates) != list("ABCDE"):
        return metrics, False
    candidates.sort(key=lambda item: item[0])

    derived_measurements: List[Dict[str, float | str | list[float]]] = []
    for label, center in candidates:
        bearing = float(np.arctan2(center[1] - vertex[1], center[0] - vertex[0]) % (2.0 * np.pi))
        distance1 = _angle_distance(bearing, ray_angles[0])
        distance2 = _angle_distance(bearing, ray_angles[1])
        derived_measurements.append(
            {
                "label": label,
                "center": [center[0], center[1]],
                "bearing_degrees": float(np.degrees(bearing)),
                "ray1_distance_degrees": float(np.degrees(distance1)),
                "ray2_distance_degrees": float(np.degrees(distance2)),
                "angular_imbalance_degrees": float(np.degrees(abs(distance1 - distance2))),
            }
        )
    ordered = sorted(derived_measurements, key=lambda item: float(item["angular_imbalance_degrees"]))
    best = ordered[0]
    unique = (
        float(best["angular_imbalance_degrees"]) <= 1e-5
        and float(ordered[1]["angular_imbalance_degrees"]) >= 1.0
    )
    derived_correct = str(best["label"])
    derived_index = next(index for index, item in enumerate(derived_measurements) if item["label"] == derived_correct)

    top_candidates = record.get("candidates")
    top_candidate_valid = isinstance(top_candidates, list) and len(top_candidates) == 5
    if top_candidate_valid:
        by_label = {str(item.get("label") or ""): item for item in top_candidates if isinstance(item, dict)}
        top_candidate_valid = sorted(by_label) == list("ABCDE") and all(
            _point_pair([by_label[label].get("x"), by_label[label].get("y")]) is not None
            and float(np.hypot(
                float(by_label[label]["x"]) - candidates[index][1][0],
                float(by_label[label]["y"]) - candidates[index][1][1],
            )) <= 1e-5
            for index, label in enumerate("ABCDE")
        )

    stored_measurements = structured.get("candidate_measurements")
    measurements_valid = isinstance(stored_measurements, list) and len(stored_measurements) == 5
    if measurements_valid:
        stored_by_label = {
            str(item.get("label") or ""): item for item in stored_measurements if isinstance(item, dict)
        }
        measurements_valid = sorted(stored_by_label) == list("ABCDE") and all(
            _point_pair(stored_by_label[label].get("center")) is not None
            and float(np.hypot(
                float(stored_by_label[label]["center"][0]) - float(derived_measurements[index]["center"][0]),
                float(stored_by_label[label]["center"][1]) - float(derived_measurements[index]["center"][1]),
            )) <= 1e-5
            and _angle_close(stored_by_label[label].get("bearing_degrees"), float(derived_measurements[index]["bearing_degrees"]))
            and abs(float(stored_by_label[label].get("ray1_distance_degrees", 1e9)) - float(derived_measurements[index]["ray1_distance_degrees"])) <= 1e-5
            and abs(float(stored_by_label[label].get("ray2_distance_degrees", 1e9)) - float(derived_measurements[index]["ray2_distance_degrees"])) <= 1e-5
            and abs(float(stored_by_label[label].get("angular_imbalance_degrees", 1e9)) - float(derived_measurements[index]["angular_imbalance_degrees"])) <= 1e-5
            for index, label in enumerate("ABCDE")
        )

    answer_schema = structured.get("answer_schema")
    verifier_contract = structured.get("verifier_contract")
    render_geometry = structured.get("render_geometry")
    contracts_valid = (
        isinstance(answer_schema, dict)
        and answer_schema.get("terminal_text_regex") == FIVE_CHOICE_CODEC.full_pattern
        and answer_schema.get("requires_sample_specific_reasoning_before_terminal_answer") is True
        and answer_schema.get("terminal_text_contains_exact_answer") is True
        and answer_schema.get("no_block_after_terminal_image") is True
        and isinstance(verifier_contract, dict)
        and verifier_contract.get("profile") == ANGLE_BISECTOR_VERIFIER_PROFILE
        and verifier_contract.get("strict_binary_correctness") is True
        and isinstance(render_geometry, dict)
        and render_geometry.get("canvas_dimensions") == [512, 512]
        and isinstance(render_geometry.get("bisector"), dict)
        and structured.get("state_trace") == ANGLE_BISECTOR_STATE_TRACE
        and structured.get("expected_video_frame_count") == 21
        and structured.get("video_fps") == 16
    )
    # Correctness is anchored only in independently replayed source geometry.
    # Authored contracts, measurement tables, state traces, and saved answer
    # fields remain diagnostics; canonical manifests intentionally scrub some
    # of them and corrupting them must not change a candidate's reward.
    valid = bool(unique and top_candidate_valid)
    metrics.update(
        {
            "angle_bisector_geometry_replay_valid": valid,
            "angle_bisector_geometry_replay_reason": "ok" if valid else "source geometry is invalid or non-unique",
            "angle_bisector_derived_correct_option": derived_correct,
            "angle_bisector_derived_correct_index": derived_index,
            "angle_bisector_derived_opening_degrees": float(np.degrees(opening)),
            "angle_bisector_derived_bisector_bearing_degrees": float(np.degrees(bisector)),
            "angle_bisector_candidate_imbalances_degrees": {
                str(item["label"]): float(item["angular_imbalance_degrees"])
                for item in derived_measurements
            },
            "angle_bisector_unique_solution": unique,
            "angle_bisector_top_candidate_metadata_valid": top_candidate_valid,
            "angle_bisector_measurement_metadata_valid": measurements_valid,
            "angle_bisector_contract_metadata_valid": contracts_valid,
        }
    )
    return metrics, valid


def _angle_bisector_contract_metrics(
    *,
    artifact_kind: str,
    binary_correct: bool,
    component_scores: Dict[str, float],
    failure_reason: str,
) -> Dict[str, Any]:
    return {
        "verifier_profile": ANGLE_BISECTOR_VERIFIER_PROFILE,
        "artifact_kind": artifact_kind,
        "component_scores": component_scores,
        "binary_correct": bool(binary_correct),
        "strict_binary_correctness": True,
        "failure_reason": "ok" if binary_correct else failure_reason,
        "performance_policy": "latency_throughput_memory_and_decode_counts_are_diagnostics_without_failure_thresholds",
    }


def _angle_bisector_color_masks(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    cyan = (
        (red <= 95)
        & (green >= 105)
        & (blue >= 135)
        & (green - red >= 45)
        & (blue - red >= 70)
    )
    dark_red = (
        (red >= 130)
        & (green <= 125)
        & (red - green >= 45)
        & (red - blue >= 20)
    )
    return cyan, dark_red


def _angle_bisector_red_option(
    frame: np.ndarray,
    record: Dict[str, Any],
) -> Tuple[Optional[str], np.ndarray]:
    _cyan, dark_red = _angle_bisector_color_masks(frame)
    radius = max(12, int(round(float(record.get("point_radius") or 11) * 1.35)))
    height, width = dark_red.shape
    selected: List[str] = []
    yy, xx = np.ogrid[:height, :width]
    for candidate in record.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        try:
            cx = float(candidate["x"])
            cy = float(candidate["y"])
        except (KeyError, TypeError, ValueError):
            continue
        local_circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        if int(np.count_nonzero(dark_red & local_circle)) >= 15:
            selected.append(str(candidate.get("label") or "").strip().upper())
    label = "".join(sorted(item for item in selected if item)) or None
    return label, dark_red


def _angle_bisector_expected_geometry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    replay_metrics, replay_valid = _angle_bisector_replay(record)
    if not replay_valid:
        return None
    structured = record["structured_answer"]
    vertex = tuple(float(value) for value in structured["vertex"])
    endpoints = [tuple(float(value) for value in point) for point in structured["ray_endpoints"]]
    correct = str(replay_metrics["angle_bisector_derived_correct_option"])
    centers = {
        str(item["label"]): tuple(float(value) for value in item["center"])
        for item in structured["candidate_centers"]
    }
    ray_angles = [float(np.arctan2(point[1] - vertex[1], point[0] - vertex[0]) % (2.0 * np.pi)) for point in endpoints]
    signed_delta = float((ray_angles[1] - ray_angles[0] + np.pi) % (2.0 * np.pi) - np.pi)
    bisector = float((ray_angles[0] + signed_delta / 2.0) % (2.0 * np.pi))
    return {
        "vertex": vertex,
        "endpoints": endpoints,
        "correct": correct,
        "target": centers[correct],
        "centers": centers,
        "ray_angles": ray_angles,
        "bisector": bisector,
    }


def _angle_bisector_allowed_dynamic_mask(
    shape: Tuple[int, int],
    geometry: Dict[str, Any],
    point_radius: float,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    vertex = tuple(int(round(value)) for value in geometry["vertex"])
    target = tuple(int(round(value)) for value in geometry["target"])
    cv2.line(mask, vertex, target, 255, thickness=16)
    cv2.circle(mask, target, max(18, int(round(point_radius * 2.0))), 255, thickness=-1)
    return mask


def _read_rgb_image(path: Path, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    width, height = target_size
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb


def _point_sequence_support(
    mask: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    height, width = mask.shape
    hits = 0
    total = 0
    for fraction in np.linspace(0.12, 0.88, 28):
        x = int(round(start[0] + (end[0] - start[0]) * float(fraction)))
        y = int(round(start[1] + (end[1] - start[1]) * float(fraction)))
        x0, x1 = max(0, x - 3), min(width, x + 4)
        y0, y1 = max(0, y - 3), min(height, y + 4)
        total += 1
        hits += int(bool(np.any(mask[y0:y1, x0:x1])))
    return hits / total if total else 0.0


# Span of each declared stroke the sampler reads.  The ends are left out
# because a stroke's endpoint sits under whatever marker the task anchors there.
_PATH_PARAMETER_SPAN: Tuple[float, float] = (0.08, 0.92)


def _path_positions(path: np.ndarray, sample_count: int, *,
                    parameter_span: Tuple[float, float] = _PATH_PARAMETER_SPAN) -> np.ndarray:
    """``sample_count`` evenly indexed positions along ``path``."""

    points = np.asarray(path, dtype=np.float32)
    positions = np.linspace(*parameter_span, sample_count)
    source_positions = np.linspace(0.0, 1.0, len(points))
    xs = np.interp(positions, source_positions, points[:, 0])
    ys = np.interp(positions, source_positions, points[:, 1])
    return np.stack([xs, ys], axis=1)


def overlay_occluders(
    record: Dict[str, Any],
    paths: Sequence[np.ndarray] = (),
) -> List[Tuple[float, float, float]]:
    """Discs the renderer paints *over* the construction layer, from the source.

    ``draw_candidates`` and the per-task anchor markers are re-applied on top of
    every animated frame and on the terminal image, so construction ink under
    them is never visible however correctly it was drawn.  Both families sit at
    source-derived positions.  Candidate-covered endpoints are governed solely
    by the actual candidate disc carried by the record: a renderer-trimmed line
    endpoint inside that disc is construction ink, not another marker centre.
    Declared endpoints outside every candidate disc retain the route's
    non-candidate anchor-marker model.  Task-owned anchor discs that are not path
    endpoints are added separately by the task-aware caller.
    """

    raw_radius = record.get("point_radius")
    try:
        point_radius = float(raw_radius)
    except (TypeError, ValueError):
        point_radius = float(PointTargetPuzzleGenerator.POINT_RADIUS)
    if not math.isfinite(point_radius) or point_radius <= 0.0:
        point_radius = float(PointTargetPuzzleGenerator.POINT_RADIUS)
    candidate_radius = point_radius + float(PointTargetPuzzleGenerator.CANDIDATE_OUTLINE_WIDTH) / 2.0
    discs: List[Tuple[float, float, float]] = []
    candidate_discs: List[Tuple[float, float, float]] = []
    for item in record.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        centre = _point_pair((item.get("x"), item.get("y")))
        if centre is not None:
            candidate_discs.append((centre[0], centre[1], candidate_radius))
    discs.extend(candidate_discs)
    for path in paths:
        points = np.asarray(path, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        for endpoint in (points[0], points[-1]):
            endpoint_x, endpoint_y = float(endpoint[0]), float(endpoint[1])
            # A number of task renderers trim a construction line to the rim of
            # a candidate before recording it.  Such an endpoint lies inside
            # the candidate's real painted disc; adding another point-radius
            # disc at the trimmed endpoint would invent an occluder extending a
            # full radius beyond the marker.  The record's candidate centre and
            # renderer radius are the authoritative overlay in that case.
            if any(
                math.hypot(endpoint_x - centre_x, endpoint_y - centre_y) <= radius
                for centre_x, centre_y, radius in candidate_discs
            ):
                continue
            discs.append((endpoint_x, endpoint_y, point_radius))
    return discs


# Departure from its own image's background at which a pixel counts as input
# ink.  Same constant the construction mask itself uses, so a position is
# excluded exactly when the mask that scores it cannot register a change there.
_SOURCE_INK_DELTA = 24

# Canvas the window exclusion falls back to when no input mask is supplied to
# bound it.  Every task on this route declares 512x512, and the only
# caller without a mask passes overlay discs alone, so the value only decides
# where a sample counts as off-canvas.
_CANVAS_FALLBACK = 512


def source_ink_mask(source: np.ndarray) -> np.ndarray:
    """Pixels the *input* already inked, from the input image alone.

    A construction gate reads the ink a delivery *added*: the position is scored
    against ``candidate != source``.  Wherever the input already carries ink the
    construction layer runs underneath, that difference is structurally zero —
    the reference delivery and a delivery that skipped the stroke there produce
    the same pixels — so such a position carries no evidence in either
    direction.  The mask is derived from the supplied input image alone, never
    from the candidate, which is what keeps a candidate from manufacturing its
    own exclusions.
    """

    frame = np.asarray(source)
    return (
        np.max(np.abs(frame.astype(np.int16) - _frame_background(frame)), axis=2)
        >= _SOURCE_INK_DELTA
    )


def _unreadable_window_positions(
    samples: np.ndarray,
    occluders: Sequence[Tuple[float, float, float]],
    source_ink: Optional[np.ndarray],
    shape: Tuple[int, int],
    *,
    sample_radius: int,
) -> np.ndarray:
    """Samples whose observation window holds no pixel new ink could reach.

    This is the exclusion matched to the measurement.  :func:`_sampled_path_hits`
    reads the box of half-width ``sample_radius`` around a sample and reports
    True if any pixel in it differs from the input; a pixel can only differ there
    if the renderer did not paint an overlay disc over it and the input had not
    already inked it.  When *every* pixel of that box fails both tests the sample
    reports "no ink" for a stroke that was drawn correctly, so it is evidence in
    neither direction.

    The corridor test below asks a different question -- whether the stroke's own
    band is covered along the arc -- and the two are complementary: a covered
    corridor means no ink can *appear* in the window, a covered window means no
    ink can be *seen* in it.  Measured on double_rotation at seed 77003, the
    corridor test alone keeps samples whose 7x7 window sits 6 pixels inside a
    12.5-pixel candidate disc, because the corridor it examines runs 13.5 pixels
    along the arc and leaves the disc at both ends; those samples then read a
    hard zero on a stroke the delivery drew correctly underneath the disc, and
    the terminal-image gate failed its own reference.
    """

    blind = np.zeros(len(samples), dtype=bool)
    if not len(samples) or (not occluders and source_ink is None):
        return blind
    height, width = int(shape[0]), int(shape[1])
    unreadable = (
        np.zeros((height, width), dtype=bool)
        if source_ink is None
        else np.asarray(source_ink, dtype=bool).copy()
    )
    for centre_x, centre_y, radius in occluders:
        if not (math.isfinite(centre_x) and math.isfinite(centre_y) and math.isfinite(radius)):
            continue
        cv2.circle(
            unreadable.view(np.uint8),
            (int(round(float(centre_x))), int(round(float(centre_y)))),
            max(0, int(math.floor(float(radius)))),
            1,
            -1,
        )
    radius = int(sample_radius)
    for index, (x_value, y_value) in enumerate(np.asarray(samples, dtype=np.float64)):
        x = int(round(float(x_value)))
        y = int(round(float(y_value)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            blind[index] = True
            continue
        blind[index] = bool(unreadable[y0:y1, x0:x1].all())
    return blind


def _occluded_positions(
    path: np.ndarray,
    samples: np.ndarray,
    occluders: Sequence[Tuple[float, float, float]],
    *,
    sample_radius: int,
    line_width: float,
) -> np.ndarray:
    """Samples whose whole stroke corridor lies inside overlay discs.

    :func:`_sampled_path_hits` accepts any lit pixel in the box of half-width
    ``sample_radius`` around a sample, so the ink that box can see is the stroke
    corridor within ``sample_radius * sqrt(2) + line_width / 2`` arc-length of
    the sample.  When every point of that corridor lies inside an overlay disc,
    the sample reports "no ink" for a stroke that was drawn correctly, so it is
    not evidence about the stroke either way.  Requiring *complete* coverage
    keeps the exclusion minimal: a sample that can still see one unoccluded
    millimetre of its own corridor is kept.
    """

    blind = np.zeros(len(samples), dtype=bool)
    if not occluders or not len(samples):
        return blind
    points = np.asarray(path, dtype=np.float64)
    segments = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total = float(cumulative[-1])
    if total <= 1e-6:
        return blind
    reach = float(sample_radius) * math.sqrt(2.0) + float(line_width) / 2.0
    span = np.linspace(0.0, total, max(9, int(round(total * 2.0)) + 1))
    corridor = np.stack(
        [np.interp(span, cumulative, points[:, 0]), np.interp(span, cumulative, points[:, 1])],
        axis=1,
    )
    centres = np.asarray([[x, y] for x, y, _radius in occluders], dtype=np.float64)
    radii = np.asarray([radius for _x, _y, radius in occluders], dtype=np.float64)
    covered = np.min(
        np.linalg.norm(corridor[:, None, :] - centres[None, :, :], axis=2) - radii[None, :],
        axis=1,
    ) <= 0.0
    reachable = np.linalg.norm(
        np.asarray(samples, dtype=np.float64)[:, None, :] - corridor[None, :, :], axis=2
    ) <= reach
    for index in range(len(samples)):
        window = reachable[index]
        if not bool(window.any()):
            continue
        blind[index] = bool(covered[window].all())
    return blind


def _source_inked_positions(
    samples: np.ndarray,
    source_ink: Optional[np.ndarray],
) -> np.ndarray:
    """Samples the input has already inked, where new ink cannot register.

    Half of this route's generators re-apply their reference scene *over* the
    construction layer -- ``_video_overlay_extras`` exists for exactly that, and
    34 of the 67 tasks override it -- so the construction is visible only where
    the input left the canvas blank.  A construction gate scores a position on
    ``candidate != source``; at a position the input already inked, that
    difference is structurally zero whether the stroke was drawn there or not, so
    the position is evidence in neither direction and is dropped rather than
    counted as absent ink.

    The test is the sample's own pixel, not a window around it: a position on the
    input's own ink is one the declared stroke can only reach *underneath* that
    ink, and reading a window instead would make the exclusion depend on how far
    the declared centreline happens to sit from where the renderer's primitive
    put the stroke -- PIL strokes an arc inward from its bounding box, so a
    declared arc radius and its drawn band differ by half the stroke width.
    Dropping the position is the conservative direction: it removes evidence,
    never adds it, and the mask is read from the input image alone, so no
    delivery can manufacture its own exclusions.
    """

    if source_ink is None or not len(samples):
        return np.zeros(len(samples), dtype=bool)
    height, width = source_ink.shape[:2]
    positions = np.asarray(samples, dtype=np.float64)
    columns = np.clip(np.rint(positions[:, 0]).astype(np.int64), 0, width - 1)
    rows = np.clip(np.rint(positions[:, 1]).astype(np.int64), 0, height - 1)
    return np.asarray(source_ink[rows, columns], dtype=bool)


# Escalation ladder for the exclusive sampler.  The fraction of a stroke that
# survives exclusion is a geometric constant of the configuration — measured on
# spiral_similarity_center it reads 0.271 at 24, 48, 96 and 144 positions alike
# — so the only way to recover a usable sample budget on a heavily crossed
# stroke is to sample it more finely.  The ladder stops at 192 because
# 192 * 0.0625 = 12, the retained budget, is already below the lowest survival
# fraction any stroke grammar in the registry produces.
_SAMPLE_DENSITY_LADDER: Tuple[int, ...] = (1, 2, 4, 8)
# `_single_endpoint_growth` refuses to judge fewer than eight positions, so a
# stroke that cannot reach eight exclusive, observable samples at the densest
# rung is genuinely unjudgeable and must fail closed rather than be measured on
# ink another stroke or an overlay disc can explain.
_MINIMUM_JUDGEABLE_SAMPLES = 8


def _attributable_path_positions(
    path: np.ndarray,
    *,
    sample_count: int = 24,
    excluded_paths: Sequence[np.ndarray] = (),
    clearance: float = 0.0,
    minimum_retained: int = 12,
    occluders: Sequence[Tuple[float, float, float]] = (),
    sample_radius: int = 3,
    line_width: float = 0.0,
    source_ink: Optional[np.ndarray] = None,
    parameter_span: Tuple[float, float] = _PATH_PARAMETER_SPAN,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Positions along ``path`` whose ink only this stroke can explain.

    Returns ``(samples, parameters)`` where ``parameters`` is each retained
    sample's place along the declared stroke on the same 0..1 scale
    :func:`_path_positions` samples, so a caller can gate equal parts of the
    *stroke* rather than equal counts of an unevenly thinned sample list.

    Declared constructions legally touch each other — an excircle is tangent to
    the side extensions it was built from, concurrent lines share a centre — and
    :func:`_sampled_path_hits` accepts any lit pixel in a small window, so a
    sample sitting on a neighbour reads as this stroke's ink from the moment the
    neighbour is drawn.  Samples within ``clearance`` of another declared path
    are therefore not evidence about this one, and neither are samples the
    observation model cannot read: those buried under an overlay disc the
    renderer paints on top of the construction layer
    (:func:`_occluded_positions`), and those the input has already inked, where
    ``candidate != source`` is structurally zero however the stroke was drawn
    (:func:`_source_inked_positions`).

    Dropping any of those families shrinks the sample budget, so the path is
    resampled more finely until enough exclusive, observable positions remain.
    Returning the *unfiltered* positions instead would knowingly score this
    stroke on ink another stroke drew, which is what makes a legally serial
    construction read as two strokes growing at once; ``None`` is returned when
    even the densest rung cannot supply a judgeable set, and the caller then has
    no evidence about this stroke in either direction.
    """

    points = np.asarray(path, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return None
    filtering = bool(
        (clearance > 0.0 and len(excluded_paths)) or len(occluders) or source_ink is not None
    )
    if not filtering:
        return (_path_positions(points, sample_count, parameter_span=parameter_span),
                np.linspace(*parameter_span, sample_count))
    best: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for multiple in _SAMPLE_DENSITY_LADDER:
        parameters = np.linspace(*parameter_span, sample_count * multiple)
        samples = _path_positions(points, sample_count * multiple, parameter_span=parameter_span)
        keep = np.ones(len(samples), dtype=bool)
        if clearance > 0.0:
            for other in excluded_paths:
                other_points = np.asarray(other, dtype=np.float32)
                if other_points.ndim != 2 or other_points.shape[0] < 2:
                    continue
                for index, (x_value, y_value) in enumerate(samples):
                    if not keep[index]:
                        continue
                    if _polyline_point_distance((float(x_value), float(y_value)), other_points) <= clearance:
                        keep[index] = False
        if len(occluders):
            keep &= ~_occluded_positions(
                points,
                samples,
                occluders,
                sample_radius=sample_radius,
                line_width=line_width,
            )
        if source_ink is not None:
            keep &= ~_source_inked_positions(samples, source_ink)
        if len(occluders) or source_ink is not None:
            keep &= ~_unreadable_window_positions(
                samples,
                occluders,
                source_ink,
                (
                    source_ink.shape[:2]
                    if source_ink is not None
                    else (_CANVAS_FALLBACK, _CANVAS_FALLBACK)
                ),
                sample_radius=sample_radius,
            )
        best = (samples[keep], parameters[keep])
        if int(keep.sum()) >= minimum_retained:
            return best
    if best is not None and len(best[0]) >= _MINIMUM_JUDGEABLE_SAMPLES:
        return best
    return None


def _attributable_path_samples(
    path: np.ndarray,
    **kwargs: Any,
) -> Optional[np.ndarray]:
    """:func:`_attributable_path_positions` without the parameter track."""

    retained = _attributable_path_positions(path, **kwargs)
    return None if retained is None else retained[0]


def _polyline_point_distance(point: Tuple[float, float], polyline: np.ndarray) -> float:
    """Shortest distance from ``point`` to any segment of ``polyline``."""

    starts = polyline[:-1]
    ends = polyline[1:]
    deltas = ends - starts
    lengths = np.sum(deltas * deltas, axis=1)
    offsets = np.asarray(point, dtype=np.float32) - starts
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.sum(offsets * deltas, axis=1) / np.where(lengths > 1e-9, lengths, 1.0)
    parameters = np.clip(np.where(lengths > 1e-9, raw, 0.0), 0.0, 1.0)
    projections = starts + deltas * parameters[:, None]
    return float(np.min(np.linalg.norm(np.asarray(point, dtype=np.float32) - projections, axis=1)))


def _sampled_path_hits(
    mask: np.ndarray,
    path: np.ndarray,
    *,
    radius: int = 4,
    sample_count: int = 24,
    samples: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample one ordered stroke from its declared start toward its end."""

    if samples is None:
        samples = _attributable_path_samples(path, sample_count=sample_count)
    if samples is None:
        return np.zeros(sample_count, dtype=bool)
    height, width = mask.shape
    hits: List[bool] = []
    for x_float, y_float in samples:
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        hits.append(bool(x0 < x1 and y0 < y1 and np.any(mask[y0:y1, x0:x1])))
    return np.asarray(hits, dtype=bool)


def _single_endpoint_growth(
    hit_rows: Sequence[np.ndarray],
    *,
    final_support: float = 0.68,
    require_partial: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """Require a cumulative stroke whose moving tip travels start-to-end.

    When partial growth is observable, this rejects center-out/bidirectional
    growth and whole-stroke opacity reveals: a partial frame must contain a strong start prefix and
    a still-empty end suffix, aggregate support may not regress, and sampled
    activation times must progress predominantly in path order. A source replay
    can waive the partial-frame requirement for a sub-frame visible fragment.
    """

    if not hit_rows:
        return False, {
            "supports": [],
            "start_supports": [],
            "end_supports": [],
            "ordered_activation_fraction": 0.0,
            "directional_partial_frame": False,
        }
    rows = np.asarray(hit_rows, dtype=bool)
    if rows.ndim != 2 or rows.shape[1] < 8:
        return False, {
            "supports": [],
            "start_supports": [],
            "end_supports": [],
            "ordered_activation_fraction": 0.0,
            "directional_partial_frame": False,
        }
    supports = np.mean(rows, axis=1)
    quarter = max(2, rows.shape[1] // 4)
    start_supports = np.mean(rows[:, :quarter], axis=1)
    end_supports = np.mean(rows[:, -quarter:], axis=1)
    directional_partial = any(
        0.05 <= float(total) <= 0.82
        and float(start) >= 0.34
        and float(start) >= float(end) + 0.24
        for total, start, end in zip(supports, start_supports, end_supports)
    )
    cumulative = all(
        float(supports[index + 1]) + 0.13 >= float(supports[index])
        for index in range(len(supports) - 1)
    )
    activation = np.full(rows.shape[1], len(rows), dtype=np.int16)
    for sample_index in range(rows.shape[1]):
        active = np.flatnonzero(rows[:, sample_index])
        if active.size:
            activation[sample_index] = int(active[0])
    comparable = 0
    ordered = 0
    for index in range(len(activation) - 1):
        if activation[index] >= len(rows) or activation[index + 1] >= len(rows):
            continue
        comparable += 1
        ordered += int(activation[index] <= activation[index + 1] + 1)
    ordered_fraction = ordered / max(1, comparable)
    passed = bool(
        float(supports[-1]) >= final_support
        and (directional_partial or not require_partial)
        and cumulative
        and comparable >= rows.shape[1] // 2
        and ordered_fraction >= 0.82
    )
    return passed, {
        "supports": supports.tolist(),
        "start_supports": start_supports.tolist(),
        "end_supports": end_supports.tolist(),
        "activation_frames": activation.tolist(),
        "ordered_activation_fraction": ordered_fraction,
        "directional_partial_frame": directional_partial,
        "directional_partial_required": require_partial,
        "cumulative": cumulative,
    }


def _no_large_regression(values: Sequence[float], start_index: int, *, tolerance: float = 0.20) -> bool:
    if not 0 <= start_index < len(values):
        return False
    running = values[start_index]
    for value in values[start_index + 1 :]:
        if value + tolerance < running:
            return False
        running = max(running, value)
    return True


def _verify_angle_bisector_semantic_video(
    video_path: Path,
    record: Dict[str, Any],
    input_path: Path,
) -> Tuple[Dict[str, Any], bool]:
    replay_metrics, replay_valid = _angle_bisector_replay(record)
    geometry = _angle_bisector_expected_geometry(record) if replay_valid else None
    capture = cv2.VideoCapture(video_path.as_posix())
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    frames: List[np.ndarray] = []
    if opened:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    dims = record.get("canvas_dimensions") or [512, 512]
    width, height = int(dims[0]), int(dims[1])
    source = _read_rgb_image(input_path, (width, height))
    media_valid = bool(opened and len(frames) >= 9 and abs(fps - 16.0) <= 0.25 and source is not None)
    if not media_valid or geometry is None or source is None:
        components = {
            "media_validity_score": float(media_valid),
            "source_replay_score": float(replay_valid),
            "temporal_process_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
        }
        return {
            **replay_metrics,
            "video_process_verifier": "coordinate_target_angle_bisector_full_sequence_semantic",
            "video_frame_count": len(frames),
            "video_fps": fps,
            **_angle_bisector_contract_metrics(
                artifact_kind="full_video",
                binary_correct=False,
                component_scores=components,
                failure_reason="video media contract or source geometry replay failed",
            ),
        }, False
    aligned = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.shape[:2] != (height, width)
        else frame
        for frame in frames
    ]
    allowed = _angle_bisector_allowed_dynamic_mask(
        (height, width), geometry, float(record.get("point_radius") or 11)
    )
    static_region = allowed == 0
    line_supports: List[float] = []
    line_hit_rows: List[np.ndarray] = []
    red_labels: List[Optional[str]] = []
    static_mutations: List[float] = []
    cyan_spills: List[float] = []
    line_path = np.asarray([geometry["vertex"], geometry["target"]], dtype=np.float32)
    for frame in aligned:
        cyan, _dark_red = _angle_bisector_color_masks(frame)
        red_label, _red_mask = _angle_bisector_red_option(frame, record)
        line_supports.append(_point_sequence_support(cyan, geometry["vertex"], geometry["target"]))
        line_hit_rows.append(_sampled_path_hits(cyan, line_path))
        red_labels.append(red_label)
        diff = np.max(np.abs(frame.astype(np.int16) - source.astype(np.int16)), axis=2)
        static_mutations.append(float(np.mean((diff >= 32)[static_region])))
        cyan_count = int(np.count_nonzero(cyan))
        cyan_spills.append(float(np.count_nonzero(cyan & (allowed == 0)) / max(1, cyan_count)))

    growth_start = phase_onset(line_supports, 0.08)
    highlight_start = selection_onset(red_labels)
    final_start = len(frames) - 3
    initial_ok = growth_start >= 2 and all(
        line_supports[index] < 0.08
        and red_labels[index] is None
        and float(np.mean(np.abs(aligned[index].astype(np.int16) - source.astype(np.int16)))) <= 2.0
        for index in range(min(4, growth_start))
    )
    bisector_growth_ok, bisector_growth_metrics = _single_endpoint_growth(line_hit_rows[growth_start:highlight_start])
    construction_ok = bool(
        bisector_growth_ok
        and line_supports[highlight_start - 1] >= 0.78
        and all(label is None for label in red_labels[growth_start:highlight_start])
        and _no_large_regression(line_supports, growth_start)
    )
    highlight_phase_ok = bool(
        all(line_supports[index] >= 0.78 for index in range(highlight_start, final_start))
        and all(label in {None, geometry["correct"]} for label in red_labels[highlight_start:final_start])
        and red_labels[final_start - 1] == geometry["correct"]
    )
    final_hold_ok = bool(
        all(
            line_supports[index] >= 0.78
            and red_labels[index] == geometry["correct"]
            for index in range(final_start, len(frames))
        )
        and all(
            float(np.mean(np.abs(aligned[index].astype(np.int16) - aligned[-1].astype(np.int16)))) <= 2.0
            for index in range(final_start, len(frames))
        )
    )
    valid_labels = all(label in {None, geometry["correct"]} for label in red_labels)
    preservation_ok = (
        max(static_mutations, default=1.0) <= 0.004
        and max(cyan_spills, default=1.0) <= 0.16
    )
    temporal_ok = bool(
        initial_ok and construction_ok and highlight_phase_ok and final_hold_ok and valid_labels
    )
    final_ok = bool(final_hold_ok and red_labels[-1] == geometry["correct"])
    passed = bool(media_valid and replay_valid and temporal_ok and preservation_ok and final_ok)
    components = {
        "media_validity_score": float(media_valid),
        "source_replay_score": float(replay_valid),
        "temporal_process_score": float(temporal_ok),
        "preservation_score": float(preservation_ok),
        "final_state_score": float(final_ok),
    }
    return {
        **replay_metrics,
        "video_process_verifier": "coordinate_target_angle_bisector_full_sequence_semantic",
        "video_semantic_parser": "single_endpoint_bisector_then_highlight_parser",
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_option": geometry["correct"] if final_ok else None,
        "video_angle_bisector_initial_hold": initial_ok,
        "video_angle_bisector_construction_phase": construction_ok,
        "video_angle_bisector_bisector_single_endpoint_growth": bisector_growth_ok,
        "video_angle_bisector_bisector_growth_metrics": bisector_growth_metrics,
        "video_angle_bisector_highlight_phase": highlight_phase_ok,
        "video_angle_bisector_final_hold": final_hold_ok,
        "video_angle_bisector_valid_highlight_labels": valid_labels,
        "video_angle_bisector_max_static_mutation_fraction": max(static_mutations, default=1.0),
        "video_angle_bisector_max_cyan_spill_fraction": max(cyan_spills, default=1.0),
        "video_semantic_process_check_pass": passed,
        **_angle_bisector_contract_metrics(
            artifact_kind="full_video",
            binary_correct=passed,
            component_scores=components,
            failure_reason="bisector growth, highlight, preservation, or final-hold semantics failed",
        ),
    }, passed


def _circle_tangent_line_contract_metrics(
    *,
    artifact_kind: str,
    binary_correct: bool,
    component_scores: Dict[str, float],
    failure_reason: str,
    elapsed_ms: float,
) -> Dict[str, Any]:
    return {
        "verifier_profile": CIRCLE_TANGENT_LINE_VERIFIER_PROFILE,
        "artifact_kind": artifact_kind,
        "component_scores": component_scores,
        "binary_correct": bool(binary_correct),
        "strict_binary_correctness": True,
        "failure_reason": "ok" if binary_correct else failure_reason,
        "verification_elapsed_ms": elapsed_ms,
        "performance_policy": "latency_throughput_memory_and_decode_counts_are_diagnostics_without_failure_thresholds",
    }


def _circle_tangent_line_expected_geometry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    replay_metrics, replay_valid, correct = replay_circle_tangent_line_geometry(record)
    if not replay_valid or correct is None:
        return None
    structured = record.get("structured_answer")
    if not isinstance(structured, dict):
        return None
    center = _point_pair(structured.get("circle_center"))
    contact = _point_pair(structured.get("contact_point"))
    if center is None or contact is None:
        return None
    centers = {
        str(item.get("label") or ""): _point_pair(item.get("center"))
        for item in structured.get("candidate_centers") or []
        if isinstance(item, dict)
    }
    target = centers.get(correct)
    if target is None:
        return None
    radius = float(structured["circle_radius"])
    ux = (contact[0] - center[0]) / radius
    uy = (contact[1] - center[1]) / radius
    tangent_bearing = math.radians(
        float(replay_metrics["circle_tangent_line_derived_tangent_bearing_degrees"])
    )
    tx, ty = math.cos(tangent_bearing), math.sin(tangent_bearing)
    # Fixed family-owned construction grammar.
    marker_size = 24.0
    q1 = (contact[0] - ux * marker_size, contact[1] - uy * marker_size)
    q2 = (q1[0] + tx * marker_size, q1[1] + ty * marker_size)
    q3 = (contact[0] + tx * marker_size, contact[1] + ty * marker_size)
    return {
        "center": center,
        "contact": contact,
        "target": target,
        "correct": correct,
        "radius": radius,
        "right_angle_points": [q1, q2, q3],
    }


def _circle_tangent_line_allowed_dynamic_mask(
    shape: Tuple[int, int],
    geometry: Dict[str, Any],
    point_radius: float,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    center = tuple(int(round(value)) for value in geometry["center"])
    contact = tuple(int(round(value)) for value in geometry["contact"])
    target = tuple(int(round(value)) for value in geometry["target"])
    cv2.line(mask, center, contact, 255, thickness=18)
    cv2.line(mask, contact, target, 255, thickness=18)
    q1, q2, q3 = [
        tuple(int(round(value)) for value in point)
        for point in geometry["right_angle_points"]
    ]
    cv2.line(mask, q1, q2, 255, thickness=16)
    cv2.line(mask, q2, q3, 255, thickness=16)
    cv2.circle(mask, center, 12, 255, thickness=-1)
    cv2.circle(mask, target, max(18, int(round(point_radius * 2.0))), 255, thickness=-1)
    return mask


def _circle_tangent_line_frame_analysis(
    frame: np.ndarray,
    source: np.ndarray,
    record: Dict[str, Any],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    cyan, _dark_red = _angle_bisector_color_masks(frame)
    red_label, red_mask = _angle_bisector_red_option(frame, record)
    q1, q2, q3 = geometry["right_angle_points"]
    radius_support = _point_sequence_support(cyan, geometry["center"], geometry["contact"])
    tangent_support = _point_sequence_support(cyan, geometry["contact"], geometry["target"])
    corner_first = _point_sequence_support(cyan, q1, q2)
    corner_second = _point_sequence_support(cyan, q2, q3)
    cx, cy = (int(round(value)) for value in geometry["center"])
    y0, y1 = max(0, cy - 8), min(cyan.shape[0], cy + 9)
    x0, x1 = max(0, cx - 8), min(cyan.shape[1], cx + 9)
    center_dot_pixels = int(np.count_nonzero(cyan[y0:y1, x0:x1]))
    allowed = _circle_tangent_line_allowed_dynamic_mask(
        cyan.shape,
        geometry,
        float(record.get("point_radius") or 11),
    )
    cyan_count = int(np.count_nonzero(cyan))
    cyan_spill = float(np.count_nonzero(cyan & (allowed == 0)) / max(1, cyan_count))
    diff = np.max(visual_delta(frame, source), axis=2)
    static_mutation = float(np.mean((diff >= 24)[allowed == 0]))
    return {
        "radius_support": radius_support,
        "tangent_support": tangent_support,
        "corner_first_support": corner_first,
        "corner_second_support": corner_second,
        "center_dot_pixels": center_dot_pixels,
        "red_option": red_label,
        "red_pixel_count": int(np.count_nonzero(red_mask)),
        "cyan_spill_fraction": cyan_spill,
        "static_mutation_fraction": static_mutation,
        "black_vacancy": float(np.mean(frame)) < 35.0,
    }


def _circle_tangent_line_final_semantics(
    analysis: Dict[str, Any],
    correct: str,
) -> Tuple[bool, bool]:
    preservation_ok = bool(
        float(analysis["static_mutation_fraction"]) <= 0.004
        and float(analysis["cyan_spill_fraction"]) <= 0.16
        and not analysis["black_vacancy"]
    )
    final_ok = bool(
        float(analysis["radius_support"]) >= 0.80
        and float(analysis["tangent_support"]) >= 0.80
        and float(analysis["corner_first_support"]) >= 0.68
        and float(analysis["corner_second_support"]) >= 0.68
        and int(analysis["center_dot_pixels"]) >= 18
        and analysis["red_option"] == correct
        and int(analysis["red_pixel_count"]) >= 24
    )
    return preservation_ok, final_ok


def _verify_circle_tangent_line_semantic_video(
    video_path: Path,
    record: Dict[str, Any],
    input_path: Path,
) -> Tuple[Dict[str, Any], bool]:
    started = time.perf_counter()
    replay_metrics, replay_valid, correct = replay_circle_tangent_line_geometry(record)
    geometry = _circle_tangent_line_expected_geometry(record) if replay_valid else None
    capture = cv2.VideoCapture(video_path.as_posix())
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    frames: List[np.ndarray] = []
    if opened:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    dims = record.get("canvas_dimensions") or [512, 512]
    width, height = int(dims[0]), int(dims[1])
    source = _read_rgb_image(input_path, (width, height))
    media_valid = bool(
        opened
        and len(frames) >= 9
        and abs(fps - 16.0) <= 0.25
        and source is not None
    )
    if not media_valid or geometry is None or correct is None or source is None:
        components = {
            "media_validity_score": float(media_valid),
            "source_replay_score": float(replay_valid),
            "temporal_process_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
        }
        return {
            **replay_metrics,
            "video_process_verifier": "coordinate_target_circle_tangent_line_full_sequence_semantic",
            "video_frame_count": len(frames),
            "video_fps": fps,
            **_circle_tangent_line_contract_metrics(
                artifact_kind="full_video",
                binary_correct=False,
                component_scores=components,
                failure_reason="video media contract or source geometry replay failed",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            ),
        }, False
    aligned = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.shape[:2] != (height, width)
        else frame
        for frame in frames
    ]
    analyses = [
        _circle_tangent_line_frame_analysis(frame, source, record, geometry)
        for frame in aligned
    ]
    radius_values = [float(item["radius_support"]) for item in analyses]
    tangent_values = [float(item["tangent_support"]) for item in analyses]
    corner_values = [
        min(float(item["corner_first_support"]), float(item["corner_second_support"]))
        for item in analyses
    ]
    labels = [item["red_option"] for item in analyses]
    first_start = phase_onset(radius_values, 0.08)
    second_start = phase_onset(tangent_values, 0.08)
    corner_start = phase_onset([
        max(float(item["corner_first_support"]), float(item["corner_second_support"]))
        for item in analyses
    ], 0.33)
    highlight_start = selection_onset(labels)
    final_start = len(frames) - 3
    growth_metrics, growth_pass = serial_construction_observation(
        aligned, source, record, task_name="circle_tangent_line",
        expected=str(correct), end=highlight_start,
    )
    initial_hold = first_start >= 2 and all(
        radius_values[index] < 0.08
        and tangent_values[index] < 0.08
        and corner_values[index] < 0.08
        and labels[index] is None
        and float(np.mean(np.abs(aligned[index].astype(np.int16) - source.astype(np.int16)))) <= 2.0
        for index in range(min(4, first_start))
    )
    radius_phase = bool(
        all(
            tangent_values[index] < 0.08
            and float(analyses[index]["corner_first_support"]) <= 0.25
            and float(analyses[index]["corner_second_support"]) <= 0.25
            and labels[index] is None
            for index in range(first_start, second_start)
        )
        and radius_values[second_start - 1] >= 0.80
        and _no_large_regression(radius_values[first_start:second_start], 0, tolerance=0.18)
    )
    tangent_phase = bool(
        all(
            radius_values[index] >= 0.78
            # The completed tangent meets the right-angle marker's endpoint
            # neighbourhood.  A few sampled corner points therefore receive
            # a small amount of support before the marker stroke itself starts;
            # keep the early-marker gate strict while ignoring this bounded
            # endpoint overlap.
            and (tangent_values[index] >= 0.78 or (
                float(analyses[index]["corner_first_support"]) <= 0.32
                and float(analyses[index]["corner_second_support"]) <= 0.32
            ))
            and labels[index] is None
            for index in range(second_start, corner_start)
        )
        and tangent_values[corner_start - 1] >= 0.80
        and _no_large_regression(tangent_values[second_start:corner_start], 0, tolerance=0.18)
    )
    corner_phase = bool(
        all(radius_values[index] >= 0.78 and tangent_values[index] >= 0.78 and labels[index] is None for index in range(corner_start, highlight_start))
        and corner_values[highlight_start - 1] >= 0.68
        and _no_large_regression(corner_values[corner_start:highlight_start], 0, tolerance=0.22)
    )
    highlight_phase = bool(
        all(
            radius_values[index] >= 0.78
            and tangent_values[index] >= 0.78
            and corner_values[index] >= 0.66
            and labels[index] in {None, correct}
            for index in range(highlight_start, final_start)
        )
        and labels[final_start - 1] == correct
    )
    final_semantics = [
        _circle_tangent_line_final_semantics(analyses[index], correct)
        for index in range(final_start, len(frames))
    ]
    final_hold = all(item == (True, True) for item in final_semantics) and all(
        float(np.mean(np.abs(aligned[index].astype(np.int16) - aligned[-1].astype(np.int16)))) <= 2.0
        for index in range(final_start, len(frames))
    )
    valid_labels = all(label in {None, correct} for label in labels)
    no_vacancy = not any(bool(item["black_vacancy"]) for item in analyses)
    preservation_ok = bool(
        max(float(item["static_mutation_fraction"]) for item in analyses) <= 0.004
        and max(float(item["cyan_spill_fraction"]) for item in analyses) <= 0.16
    )
    temporal_ok = bool(
        initial_hold
        and growth_pass is not False
        and radius_phase
        and tangent_phase
        and corner_phase
        and highlight_phase
        and final_hold
        and valid_labels
        and no_vacancy
    )
    final_ok = bool(final_hold and labels[-1] == correct)
    passed = bool(media_valid and replay_valid and preservation_ok and temporal_ok and final_ok)
    components = {
        "media_validity_score": float(media_valid),
        "source_replay_score": float(replay_valid),
        "temporal_process_score": float(temporal_ok),
        "preservation_score": float(preservation_ok),
        "final_state_score": float(final_ok),
    }
    return {
        **replay_metrics,
        "video_process_verifier": "coordinate_target_circle_tangent_line_full_sequence_semantic",
        "video_semantic_parser": "radius_tangent_right_angle_unique_highlight_phase_parser",
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_decoded_frame_count": len(frames),
        "video_option": correct if final_ok else None,
        **growth_metrics,
        "video_circle_tangent_line_single_endpoint_growth": growth_pass is not False,
        "video_circle_tangent_line_initial_hold": initial_hold,
        "video_circle_tangent_line_radius_phase": radius_phase,
        "video_circle_tangent_line_tangent_phase": tangent_phase,
        "video_circle_tangent_line_right_angle_phase": corner_phase,
        "video_circle_tangent_line_highlight_phase": highlight_phase,
        "video_circle_tangent_line_final_hold": final_hold,
        "video_circle_tangent_line_valid_highlight_labels": valid_labels,
        "video_circle_tangent_line_no_black_vacancy": no_vacancy,
        "video_circle_tangent_line_max_static_mutation_fraction": max(
            float(item["static_mutation_fraction"]) for item in analyses
        ),
        "video_circle_tangent_line_max_cyan_spill_fraction": max(
            float(item["cyan_spill_fraction"]) for item in analyses
        ),
        "video_semantic_process_check_pass": passed,
        **_circle_tangent_line_contract_metrics(
            artifact_kind="full_video",
            binary_correct=passed,
            component_scores=components,
            failure_reason="ordered radius, tangent, right-angle, highlight, preservation, or final hold failed",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        ),
    }, passed


def resolve_candidate_video(
    pred_root: Path,
    sample_id: str,
    split: Optional[str] = None,
) -> Optional[Path]:
    candidate_dirs = []
    if split:
        candidate_dirs.append(pred_root / split / sample_id)
    candidate_dirs.append(pred_root / sample_id)
    candidates: List[Path] = []
    for sample_dir in candidate_dirs:
        if not sample_dir.is_dir():
            continue
        candidates.extend(sample_dir / name for name in DEFAULT_CANDIDATE_VIDEOS)
        candidates.extend(sample_dir.glob("*.mp4"))
        candidates.extend(sample_dir.glob("*.webm"))
        candidates.extend(sample_dir.glob("*.mov"))
    return first_existing_path(candidates)


def _verify_midpoint_semantic_video(
    video_path: Path,
    record: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    return verify_midpoint_full_video(video_path, record)


def _arc_connect_contract_metrics(
    *,
    artifact_kind: str,
    binary_correct: bool,
    component_scores: Dict[str, float],
    failure_reason: str,
    elapsed_ms: float,
) -> Dict[str, Any]:
    return {
        "verifier_profile": ARC_CONNECT_VERIFIER_PROFILE,
        "artifact_kind": artifact_kind,
        "component_scores": component_scores,
        "binary_correct": bool(binary_correct),
        "strict_binary_correctness": True,
        "failure_reason": "ok" if binary_correct else failure_reason,
        "verification_latency_ms": float(elapsed_ms),
        "performance_policy": "latency_throughput_memory_and_decode_counts_are_diagnostics_without_failure_thresholds",
    }


def _arc_connect_circle_y(circle: Tuple[float, float, float], x: float) -> List[float]:
    cx, cy, radius = circle
    dx = x - cx
    if abs(dx) > radius:
        return []
    dy = float(np.sqrt(max(0.0, radius * radius - dx * dx)))
    return [cy - dy, cy + dy]


def _arc_connect_point(circle: Tuple[float, float, float], theta: float) -> Tuple[float, float]:
    return circle[0] + circle[2] * float(np.cos(theta)), circle[1] + circle[2] * float(np.sin(theta))


def _arc_connect_angle_path(
    circle: Tuple[float, float, float],
    start: float,
    end: float,
    *,
    samples: int = 96,
) -> np.ndarray:
    delta = float((end - start + np.pi) % (2.0 * np.pi) - np.pi)
    angles = start + delta * np.linspace(0.0, 1.0, samples)
    return np.asarray(
        [
            [circle[0] + circle[2] * np.cos(angle), circle[1] + circle[2] * np.sin(angle)]
            for angle in angles
        ],
        dtype=np.float32,
    )


def _arc_connect_expected_geometry(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    replay_metrics, replay_valid, correct = replay_arc_connect_geometry(record)
    if not replay_valid or correct is None:
        return None
    structured = record["structured_answer"]
    mask = structured["mask"]
    center_x = float(mask["center_x"])
    left_x = float(mask["left_x"])
    right_x = float(mask["right_x"])
    span = float(np.deg2rad(float(structured["arc_span_degrees"])))
    candidate_centers = {
        str(item["label"]): (float(item["x"]), float(item["y"]))
        for item in record.get("candidates") or []
    }
    circles = {
        str(item["label"]): (float(item["cx"]), float(item["cy"]), float(item["radius"]))
        for item in structured["candidate_circles"]
    }
    right_paths: Dict[str, np.ndarray] = {}
    right_hidden_paths: Dict[str, np.ndarray] = {}
    right_crossings: Dict[str, float] = {}
    for label in "ABCDE":
        circle = circles[label]
        crossings = _arc_connect_circle_y(circle, right_x)
        if len(crossings) != 2:
            return None
        crossing_y = min(crossings, key=lambda value: abs(value - candidate_centers[label][1]))
        right_crossings[label] = crossing_y
        theta_base = float(np.arctan2(crossing_y - circle[1], right_x - circle[0]))
        test_x, _test_y = _arc_connect_point(circle, theta_base + 0.01)
        direction = 1.0 if test_x > right_x else -1.0
        theta_end = theta_base + direction * span
        center_crossings = _arc_connect_circle_y(circle, center_x)
        if len(center_crossings) != 2:
            return None
        center_y = min(center_crossings, key=lambda value: abs(value - crossing_y))
        theta_center = float(np.arctan2(center_y - circle[1], center_x - circle[0]))
        right_paths[label] = _arc_connect_angle_path(circle, theta_center, theta_end)
        right_hidden_paths[label] = _arc_connect_angle_path(circle, theta_base, theta_center)

    left_circle_raw = structured["left_circle"]
    left_circle = (
        float(left_circle_raw["cx"]),
        float(left_circle_raw["cy"]),
        float(left_circle_raw["radius"]),
    )
    correct_right_y = right_crossings[correct]
    center_crossings = _arc_connect_circle_y(left_circle, center_x)
    center_y = min(center_crossings, key=lambda value: abs(value - correct_right_y))
    theta_center = float(np.arctan2(center_y - left_circle[1], center_x - left_circle[0]))
    left_crossings = _arc_connect_circle_y(left_circle, left_x)
    if len(left_crossings) != 2:
        return None
    left_y = min(left_crossings, key=lambda value: abs(value - center_y))
    theta_base = float(np.arctan2(left_y - left_circle[1], left_x - left_circle[0]))
    test_x, _test_y = _arc_connect_point(left_circle, theta_base - 0.01)
    direction = -1.0 if test_x < left_x else 1.0
    theta_end = theta_base + direction * span
    left_path = _arc_connect_angle_path(left_circle, theta_center, theta_end)
    left_hidden_path = _arc_connect_angle_path(left_circle, theta_base, theta_center)
    return {
        "correct": correct,
        "mask": {
            "center_x": center_x,
            "left_x": left_x,
            "right_x": right_x,
            "top_y": float(mask["top_y"]),
            "bottom_y": float(mask["bottom_y"]),
        },
        "candidate_centers": candidate_centers,
        "right_paths": right_paths,
        "right_hidden_paths": right_hidden_paths,
        "left_path": left_path,
        "left_hidden_path": left_hidden_path,
        # Fixed family-owned construction grammar.
        "line_width": 8,
        "replay_metrics": replay_metrics,
    }


def _arc_connect_path_support(mask: np.ndarray, path: np.ndarray, *, radius: int = 5) -> float:
    if path.size == 0:
        return 0.0
    height, width = mask.shape
    hits = 0
    for x_float, y_float in path[::3]:
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        hits += int(x0 < x1 and y0 < y1 and bool(np.any(mask[y0:y1, x0:x1])))
    return hits / max(1, len(path[::3]))


def _arc_connect_path_continuity(mask: np.ndarray, path: np.ndarray, *, radius: int = 5) -> Tuple[float, float]:
    sampled = path[::3]
    if sampled.size == 0:
        return 0.0, 0.0
    height, width = mask.shape
    hits: List[float] = []
    for x_float, y_float in sampled:
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        hits.append(float(x0 < x1 and y0 < y1 and bool(np.any(mask[y0:y1, x0:x1]))))
    window = min(6, len(hits))
    minimum_window = min(
        sum(hits[index : index + window]) / window
        for index in range(0, len(hits) - window + 1)
    )
    return hits[0], float(minimum_window)


def _arc_connect_dynamic_masks(
    shape: Tuple[int, int],
    geometry: Dict[str, Any],
    point_radius: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    height, width = shape
    band = np.zeros((height, width), dtype=np.uint8)
    left_x = max(0, int(np.floor(geometry["mask"]["left_x"] - 6)))
    right_x = min(width - 1, int(np.ceil(geometry["mask"]["right_x"] + 6)))
    band[:, left_x : right_x + 1] = 255
    correct_disk = np.zeros((height, width), dtype=np.uint8)
    correct_center = geometry["candidate_centers"][geometry["correct"]]
    cv2.circle(
        correct_disk,
        tuple(int(round(value)) for value in correct_center),
        max(18, int(round(point_radius * 2.0))),
        255,
        thickness=-1,
    )
    allowed_dynamic = cv2.bitwise_or(band, correct_disk)
    path_masks: Dict[str, np.ndarray] = {}
    all_paths = np.zeros((height, width), dtype=np.uint8)
    thickness = max(10, int(geometry["line_width"]) + 4)
    for label, path in geometry["right_paths"].items():
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(mask, [np.rint(path).astype(np.int32)], False, 255, thickness=thickness)
        path_masks[f"right_{label}"] = mask
        all_paths = cv2.bitwise_or(all_paths, mask)
    left_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(left_mask, [np.rint(geometry["left_path"]).astype(np.int32)], False, 255, thickness=thickness)
    path_masks["left"] = left_mask
    all_paths = cv2.bitwise_or(all_paths, left_mask)
    hidden_paths = cv2.bitwise_and(all_paths, band)
    return allowed_dynamic, band, path_masks, hidden_paths


def _arc_connect_red_option(frame: np.ndarray, record: Dict[str, Any]) -> Tuple[Optional[str], int]:
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    red_mask = (
        (red >= 130)
        & (green <= 145)
        & (red - green >= 35)
        & (red - blue >= 15)
    )
    height, width = red_mask.shape
    yy, xx = np.ogrid[:height, :width]
    radius = max(15, int(round(float(record.get("point_radius") or 11) * 1.7)))
    labels: List[str] = []
    for item in record.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        cx = float(item["x"])
        cy = float(item["y"])
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        if int(np.count_nonzero(red_mask & disk)) >= 15:
            labels.append(str(item.get("label") or "").strip().upper())
    return "".join(sorted(labels)) or None, int(np.count_nonzero(red_mask))


def _arc_connect_analyze_frame(
    frame: np.ndarray,
    source: np.ndarray,
    record: Dict[str, Any],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    height, width = frame.shape[:2]
    allowed_dynamic, band, path_masks, hidden_paths = _arc_connect_dynamic_masks(
        (height, width), geometry, float(record.get("point_radius") or 11)
    )
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    dark = (red <= 105) & (green <= 105) & (blue <= 110)
    gray = (
        (red >= 185)
        & (red <= 245)
        & (np.max(frame, axis=2).astype(np.int16) - np.min(frame, axis=2).astype(np.int16) <= 28)
    )
    background_band = (band > 0) & (hidden_paths == 0)
    gray_fraction = float(np.mean(gray[background_band])) if np.any(background_band) else 1.0
    bright_fraction = float(np.mean(np.min(frame, axis=2)[background_band] >= 244)) if np.any(background_band) else 0.0
    supports = {
        name: _arc_connect_path_support(dark, path)
        for name, path in {
            **{f"right_{label}": path for label, path in geometry["right_paths"].items()},
            "left": geometry["left_path"],
        }.items()
    }
    continuity = {
        name: _arc_connect_path_continuity(dark, path)
        for name, path in {
            **{f"right_{label}": path for label, path in geometry["right_paths"].items()},
            "left": geometry["left_path"],
        }.items()
    }
    hidden_dark_fraction = float(np.mean(dark[hidden_paths > 0])) if np.any(hidden_paths) else 0.0
    hidden_path_supports = {
        name: float(np.mean(dark[(mask > 0) & (band > 0)]))
        if np.any((mask > 0) & (band > 0))
        else 0.0
        for name, mask in path_masks.items()
    }
    dark_spill = float(np.mean(dark[(band > 0) & (hidden_paths == 0)])) if np.any((band > 0) & (hidden_paths == 0)) else 0.0
    diff = np.max(visual_delta(frame, source), axis=2)
    static_mutation = float(np.mean((diff >= 20)[allowed_dynamic == 0]))
    option, red_count = _arc_connect_red_option(frame, record)
    return {
        "mask_gray_fraction": gray_fraction,
        "former_mask_bright_fraction": bright_fraction,
        "path_supports": supports,
        "minimum_path_support": min(supports.values(), default=0.0),
        "path_center_endpoint_supports": {name: value[0] for name, value in continuity.items()},
        "minimum_center_endpoint_support": min((value[0] for value in continuity.values()), default=0.0),
        "path_minimum_window_supports": {name: value[1] for name, value in continuity.items()},
        "minimum_path_window_support": min((value[1] for value in continuity.values()), default=0.0),
        "hidden_path_dark_fraction": hidden_dark_fraction,
        "hidden_path_supports": hidden_path_supports,
        "band_dark_spill_fraction": dark_spill,
        "static_mutation_fraction": static_mutation,
        "red_option": option,
        "red_pixel_count": red_count,
    }


def _arc_connect_final_semantics(analysis: Dict[str, Any], correct: str) -> Tuple[bool, bool]:
    preservation_ok = bool(
        float(analysis["static_mutation_fraction"]) <= 0.006
        # The former mask includes a small halo around both boundary edges;
        # a few antialiased arc pixels necessarily fall in that halo after
        # the mask disappears.  Keep the gate strict enough to reject
        # unrelated ink while allowing the authored stroke width.  A
        # source arc whose circle center is outside the canvas can leave a
        # slightly larger clipped halo than the centered seeds, so use the
        # same bounded tolerance for all valid source geometries.
        and float(analysis["band_dark_spill_fraction"]) <= 0.02
    )
    final_ok = bool(
        float(analysis["mask_gray_fraction"]) <= 0.03
        and float(analysis["former_mask_bright_fraction"]) >= 0.94
        and float(analysis["minimum_path_support"]) >= 0.72
        and float(analysis["minimum_center_endpoint_support"]) >= 1.0
        and float(analysis["minimum_path_window_support"]) >= 0.66
        and float(analysis["hidden_path_dark_fraction"]) >= 0.32
        and analysis["red_option"] == correct
        and int(analysis["red_pixel_count"]) >= 24
    )
    return preservation_ok, final_ok


def _arc_connect_nonincreasing(values: Sequence[float], *, tolerance: float) -> bool:
    return all(float(values[index + 1]) <= float(values[index]) + tolerance for index in range(len(values) - 1))


def _arc_connect_nondecreasing(values: Sequence[float], *, tolerance: float) -> bool:
    return all(float(values[index + 1]) + tolerance >= float(values[index]) for index in range(len(values) - 1))


def _verify_arc_connect_semantic_video(
    video_path: Path,
    record: Dict[str, Any],
    input_path: Path,
) -> Tuple[Dict[str, Any], bool]:
    started = time.perf_counter()
    replay_metrics, replay_valid, correct = replay_arc_connect_geometry(record)
    geometry = _arc_connect_expected_geometry(record) if replay_valid else None
    capture = cv2.VideoCapture(video_path.as_posix())
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    frames: List[np.ndarray] = []
    if opened:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    dims = record.get("canvas_dimensions") or [512, 512]
    width, height = int(dims[0]), int(dims[1])
    source = _read_rgb_image(input_path, (width, height))
    sizes_valid = bool(frames and all(frame.shape[:2] == (height, width) for frame in frames))
    media_valid = bool(
        opened
        and len(frames) >= 9
        and abs(fps - 16.0) <= 0.25
        and sizes_valid
        and source is not None
    )
    if not media_valid or geometry is None or correct is None:
        elapsed = (time.perf_counter() - started) * 1000.0
        components = {
            "media_validity_score": float(media_valid),
            "source_replay_score": float(replay_valid),
            "temporal_process_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
        }
        return {
            **replay_metrics,
            "video_process_verifier": "coordinate_target_arc_connect_full_sequence_semantic",
            "video_frame_count": len(frames),
            "video_fps": fps,
            **_arc_connect_contract_metrics(
                artifact_kind="full_video",
                binary_correct=False,
                component_scores=components,
                failure_reason="video media contract or source replay failed",
                elapsed_ms=elapsed,
            ),
        }, False

    analyses = [_arc_connect_analyze_frame(frame, source, record, geometry) for frame in frames]
    gray_values = [float(item["mask_gray_fraction"]) for item in analyses]
    bright_values = [float(item["former_mask_bright_fraction"]) for item in analyses]
    hidden_values = [float(item["hidden_path_dark_fraction"]) for item in analyses]
    red_counts = [float(item["red_pixel_count"]) for item in analyses]
    labels = [item["red_option"] for item in analyses]
    reveal_start = phase_onset([gray_values[0] - value for value in gray_values], 0.05)
    reveal_end = next((index for index, item in enumerate(analyses)
                       if float(item["mask_gray_fraction"]) <= 0.03
                       and float(item["former_mask_bright_fraction"]) >= 0.94
                       and float(item["minimum_path_support"]) >= 0.70
                       and min(item["hidden_path_supports"].values()) >= 0.40), len(frames) - 3)
    highlight_start = selection_onset(labels)
    final_start = len(frames) - 3
    initial_hold = bool(reveal_start >= 2 and all(
        gray_values[index] >= 0.70 and hidden_values[index] <= 0.08 and labels[index] is None
        for index in range(reveal_start)
    ))
    mask_clear_gray = gray_values[reveal_start:reveal_end + 1]
    mask_clear_bright = bright_values[reveal_start:reveal_end + 1]
    mask_narrowing_reveal = bool(
        reveal_start < reveal_end < highlight_start
        and len([value for value in mask_clear_gray if 0.08 < value < 0.80]) >= 2
        and gray_values[reveal_end] <= 0.03
        and bright_values[reveal_end] >= 0.94
        and _arc_connect_nonincreasing(mask_clear_gray, tolerance=0.05)
        and _arc_connect_nondecreasing(mask_clear_bright, tolerance=0.05)
    )
    hidden_support_rows = [
        [float(analysis["hidden_path_supports"].get(name, 0.0))
         for name in ("right_A", "right_B", "right_C", "right_D", "right_E", "left")]
        for analysis in analyses
    ]
    synchronous_arc_reveal = bool(
        all(_arc_connect_nondecreasing(
            [row[index] for row in hidden_support_rows[reveal_start:reveal_end + 1]], tolerance=0.08)
            for index in range(6))
        and all(max(row) - min(row) <= 0.35
                for row in hidden_support_rows[reveal_start:reveal_end + 1])
        and len({round(value, 2) for value in hidden_values[reveal_start:reveal_end + 1]}) >= 3
    )
    no_early_highlight = all(item["red_option"] is None for item in analyses[:reveal_end])
    revealed_hold = bool(reveal_end < highlight_start and all(
        float(item["mask_gray_fraction"]) <= 0.03
        and float(item["minimum_path_support"]) >= 0.70
        and min(float(value) for value in item["hidden_path_supports"].values()) >= 0.40
        and item["red_option"] is None
        for item in analyses[reveal_end:highlight_start]
    ))
    highlight_counts = red_counts[highlight_start:final_start + 1]
    highlight_ordered = bool(
        _arc_connect_nondecreasing(highlight_counts, tolerance=max(4.0, red_counts[-1] * 0.15))
        and labels[-1] == correct
        and all(label in {None, correct} for label in labels)
        and red_counts[-1] >= 24
    )
    preservation_ok = max(float(item["static_mutation_fraction"]) for item in analyses) <= 0.009
    final_preservation, final_ok = _arc_connect_final_semantics(analyses[-1], str(correct))
    final_hold_semantic = all(_arc_connect_final_semantics(item, str(correct)) == (True, True)
                              for item in analyses[final_start:])
    final_hold_pixel_stable = all(
        float(np.mean(visual_delta(frame, frames[-1]))) <= 2.0
        for frame in frames[final_start:]
    )
    process_ok = bool(
        initial_hold
        and mask_narrowing_reveal
        and synchronous_arc_reveal
        and no_early_highlight
        and revealed_hold
        and highlight_ordered
        and preservation_ok
        and final_hold_semantic
        and final_hold_pixel_stable
    )
    passed = bool(replay_valid and media_valid and process_ok and final_preservation and final_ok)
    components = {
        "media_validity_score": 1.0,
        "source_replay_score": float(replay_valid),
        "temporal_process_score": float(process_ok),
        "preservation_score": float(preservation_ok and final_preservation),
        "final_state_score": float(final_ok),
    }
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        **replay_metrics,
        "video_process_verifier": "coordinate_target_arc_connect_full_sequence_semantic",
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_arc_connect_initial_hold": initial_hold,
        "video_arc_connect_mask_narrowing_reveal": mask_narrowing_reveal,
        "video_arc_connect_synchronous_arc_reveal": synchronous_arc_reveal,
        "video_arc_connect_no_early_highlight": no_early_highlight,
        "video_arc_connect_revealed_hold": revealed_hold,
        "video_arc_connect_highlight_ordered": highlight_ordered,
        "video_arc_connect_static_geometry_stable": preservation_ok,
        "video_arc_connect_final_hold_semantic": final_hold_semantic,
        "video_arc_connect_final_hold_pixel_stable": final_hold_pixel_stable,
        "video_arc_connect_mask_gray_fractions": gray_values,
        "video_arc_connect_mask_bright_fractions": bright_values,
        "video_arc_connect_hidden_path_dark_fractions": hidden_values,
        "video_arc_connect_red_pixel_counts": red_counts,
        "video_semantic_process_check_pass": process_ok,
        "video_semantic_process_reason": "ok" if process_ok else "symmetric mask reveal, synchronous arc continuity, highlight order, preservation, or final hold mismatch",
        **_arc_connect_contract_metrics(
            artifact_kind="full_video",
            binary_correct=passed,
            component_scores=components,
            failure_reason="full-video two-sided mask reveal, arc continuity, or final arc state failed",
            elapsed_ms=elapsed,
        ),
    }, passed


def _parse_option_label_sequence(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    answer_match = re.search(
        r"\banswer\s*:\s*([A-Za-z](?:\s*(?:[,&+/]\s*)?[A-Za-z])*)\b",
        text,
        flags=re.IGNORECASE,
    )
    if answer_match is None and re.search(r"\banswer\b", text, flags=re.IGNORECASE):
        return None
    cleaned = answer_match.group(1) if answer_match else text
    cleaned = re.sub(r"^answer\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip(" \t\r\n\"'`.,;:!?()[]{}")
    if len(cleaned) > 1 and cleaned.isalpha() and cleaned != cleaned.upper():
        return None
    if not re.fullmatch(r"[A-Za-z](?:\s*[,&+/]\s*[A-Za-z]|\s+[A-Za-z]|[A-Za-z])*", cleaned):
        return None
    labels = re.findall(r"[A-Za-z]", cleaned.upper())
    if not labels or len(labels) != len(set(labels)):
        return None
    return "".join(sorted(labels))


def _normalize_option(value: object) -> Optional[str]:
    return _parse_option_label_sequence(value)


def _formal_artifact_options(metrics: Dict[str, object]) -> List[str]:
    values = [
        _normalize_option(metrics.get("predicted_option")),
        _normalize_option(metrics.get("video_option")),
    ]
    return [value for value in values if value is not None]


def _coordinate_serial_paths(
    task_name: str,
    record: Dict[str, Any],
    expected: str,
) -> Optional[List[np.ndarray]]:
    """Replay ordered pen strokes from canonical, answer-independent inputs.

    ``None`` means that this verifier does not own a stroke grammar for the
    task.  An empty list means that it *does* own the grammar but the supplied
    semantic input is incomplete, which is a fail-closed condition below.
    Point markers and the terminal candidate highlight are deliberately not
    returned: they are commits/overlays, not construction strokes.
    """

    supported = {
        "circle_tangent_intersection",
        "diagonal_intersection",
        "excircle_center",
        "fermat_point",
        "homothety_center",
        "incircle_tangent_point",
        "inscribed_angle_arc",
        "inscribed_square_vertex",
        "kite_vertex",
        "napoleon_vertex",
        "nine_point_center",
        "secant_intersection",
        "spiral_similarity_center",
        "square_vertex_from_diagonal",
        "triangle_center",
        "triangle_from_midpoints",
    }
    if task_name not in supported:
        return None

    def points(key: str, count: int) -> Optional[List[Tuple[float, float]]]:
        raw = record.get(key)
        if not isinstance(raw, list) or len(raw) != count:
            return None
        parsed = [_point_pair(value) for value in raw]
        if any(value is None for value in parsed):
            return None
        return [value for value in parsed if value is not None]

    def ordered_triangle(
        values: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Match the stable cyclic order used by triangle renderers."""

        centroid_x = sum(point[0] for point in values) / len(values)
        centroid_y = sum(point[1] for point in values) / len(values)
        clockwise = sorted(
            values,
            key=lambda point: math.atan2(point[1] - centroid_y, point[0] - centroid_x),
        )
        start = min(range(len(clockwise)), key=lambda index: (clockwise[index][1], clockwise[index][0]))
        return clockwise[start:] + clockwise[:start]

    def candidate(label: str) -> Optional[Tuple[float, float]]:
        raw = record.get("candidates")
        if not isinstance(raw, list) or len(label) != 1:
            return None
        for item in raw:
            if not isinstance(item, dict) or str(item.get("label") or "").upper() != label:
                continue
            return _point_pair((item.get("x"), item.get("y")))
        return None

    def path(*values: Tuple[float, float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    def midpoint(first: Tuple[float, float], second: Tuple[float, float]) -> Tuple[float, float]:
        return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)

    def projection(
        point: Tuple[float, float],
        first: Tuple[float, float],
        second: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        dx, dy = second[0] - first[0], second[1] - first[1]
        denominator = dx * dx + dy * dy
        if denominator <= 1e-9:
            return None
        ratio = ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / denominator
        return (first[0] + ratio * dx, first[1] + ratio * dy)

    def line_intersection(
        first: Tuple[float, float],
        second: Tuple[float, float],
        third: Tuple[float, float],
        fourth: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        dx1, dy1 = second[0] - first[0], second[1] - first[1]
        dx2, dy2 = fourth[0] - third[0], fourth[1] - third[1]
        denominator = dx1 * dy2 - dy1 * dx2
        if abs(denominator) <= 1e-9:
            return None
        ratio = ((third[0] - first[0]) * dy2 - (third[1] - first[1]) * dx2) / denominator
        return (first[0] + ratio * dx1, first[1] + ratio * dy1)

    def reflect(
        point: Tuple[float, float],
        first: Tuple[float, float],
        second: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        foot = projection(point, first, second)
        return None if foot is None else (2.0 * foot[0] - point[0], 2.0 * foot[1] - point[1])

    def circle(center: Tuple[float, float], radius: float) -> np.ndarray:
        # VideoRenderer.animate_circle starts at PIL angle zero (rightmost
        # point) and sweeps in increasing screen-space angle.
        angles = np.linspace(0.0, 2.0 * np.pi, 97)
        return np.asarray(
            [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles],
            dtype=np.float32,
        )

    target = candidate(str(expected).strip().upper())

    if task_name == "circle_tangent_intersection":
        contacts = points("circumference_points", 2)
        if contacts is None or target is None:
            return []
        return [path(contacts[0], target), path(contacts[1], target)]
    if task_name == "diagonal_intersection":
        vertices = points("quadrilateral", 4)
        if vertices is None or target is None:
            return []
        result = []
        for a, b in ((vertices[0], vertices[2]), (vertices[1], vertices[3])):
            u = np.subtract(b, a)
            t = float(np.dot(np.subtract(target, a), u) / np.dot(u, u))
            result.append(path(target if t < 0 else a, target if t > 1 else b))
        return result
    if task_name == "inscribed_square_vertex":
        vertices = points("triangle_vertices", 3)
        if vertices is None:
            return []
        first, second, opposite = vertices
        bx, by = second[0] - first[0], second[1] - first[1]
        base_length = math.hypot(bx, by)
        if base_length <= 1e-9:
            return []
        ux, uy = bx / base_length, by / base_length
        nx, ny = -uy, ux
        height = (opposite[0] - first[0]) * nx + (opposite[1] - first[1]) * ny
        if height < 0.0:
            nx, ny, height = -nx, -ny, -height
        if height <= 1e-9:
            return []
        side = height * base_length / (height + base_length)
        along = side * ((opposite[0] - first[0]) * ux + (opposite[1] - first[1]) * uy) / height
        bottom_left = (first[0] + along * ux, first[1] + along * uy)
        bottom_right = (bottom_left[0] + side * ux, bottom_left[1] + side * uy)
        top_left = (bottom_left[0] + side * nx, bottom_left[1] + side * ny)
        top_right = (bottom_right[0] + side * nx, bottom_right[1] + side * ny)
        return [path(bottom_left, top_left), path(top_left, top_right), path(top_right, bottom_right)]
    if task_name == "kite_vertex":
        shown = points("shown_vertices", 3)
        if shown is None:
            return []
        reflected = reflect(shown[1], shown[0], shown[2])
        return [] if reflected is None else [path(shown[2], reflected, shown[0])]
    if task_name == "napoleon_vertex":
        constructed = points("constructed_equilateral_vertices", 3)
        if constructed is None:
            return []
        first, second, apex = constructed
        return [
            path(first, midpoint(second, apex)),
            path(second, midpoint(first, apex)),
            path(apex, midpoint(first, second)),
        ]
    if task_name == "secant_intersection":
        first_chord = points("chord1_endpoints", 2)
        second_chord = points("chord2_endpoints", 2)
        if first_chord is None or second_chord is None or target is None:
            return []
        nearest = lambda chord: min(chord, key=lambda value: math.hypot(value[0] - target[0], value[1] - target[1]))
        return [path(nearest(first_chord), target), path(nearest(second_chord), target)]
    if task_name == "square_vertex_from_diagonal":
        diagonal = points("diagonal_endpoints", 2)
        if diagonal is None or target is None:
            return []
        return [path(diagonal[0], target), path(diagonal[1], target)]
    if task_name == "excircle_center":
        vertices = points("triangle_vertices", 3)
        if vertices is None or target is None:
            return []
        marked = int(record.get("marked_vertex_index", 0))
        va, vb, vc = vertices[marked], vertices[(marked + 1) % 3], vertices[(marked + 2) % 3]
        extension_length = 0.22 * min(record.get("canvas_dimensions") or [512, 512])
        def extend(start: Tuple[float, float], end: Tuple[float, float]) -> Optional[Tuple[float, float]]:
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            return None if length <= 1e-9 else (end[0] + dx * extension_length / length, end[1] + dy * extension_length / length)
        ext_b, ext_c = extend(va, vb), extend(va, vc)
        opposite_foot = projection(target, vb, vc)
        if ext_b is None or ext_c is None or opposite_foot is None:
            return []
        radius = math.hypot(target[0] - opposite_foot[0], target[1] - opposite_foot[1])
        return [path(vb, ext_b), path(vc, ext_c), circle(target, radius)]
    if task_name == "fermat_point":
        vertices = points("triangle_vertices", 3)
        if vertices is None:
            return []
        def apex(pa: Tuple[float, float], pb: Tuple[float, float], pc: Tuple[float, float]) -> Optional[Tuple[float, float]]:
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                return None
            middle = midpoint(pa, pb)
            offset = (-dy * math.sqrt(3.0) / 2.0, dx * math.sqrt(3.0) / 2.0)
            options = [(middle[0] + offset[0], middle[1] + offset[1]), (middle[0] - offset[0], middle[1] - offset[1])]
            return max(options, key=lambda value: math.hypot(value[0] - pc[0], value[1] - pc[1]))
        sides = [(vertices[1], vertices[2], vertices[0]), (vertices[0], vertices[2], vertices[1]), (vertices[0], vertices[1], vertices[2])]
        result: List[np.ndarray] = []
        for pa, pb, pc in sides:
            outer = apex(pa, pb, pc)
            if outer is None:
                return []
            result.extend([path(pa, outer, pb), path(outer, pc)])
        return result
    if task_name == "homothety_center":
        large, small = points("large_triangle", 3), points("small_triangle", 3)
        if large is None or small is None or target is None:
            return []
        return [path(a, b if np.dot(np.subtract(a, target), np.subtract(b, target)) < 0 else target)
                for a, b in zip(large, small)]
    if task_name == "spiral_similarity_center":
        large, small = points("large_triangle", 3), points("small_triangle", 3)
        if large is None or small is None or target is None:
            return []
        # One polyline per vertex pair, cornered at the spiral centre: the leg
        # to the black vertex is drawn first, then the leg to its gray partner.
        return [path(black, target, gray) for black, gray in zip(large, small)]
    if task_name == "incircle_tangent_point":
        raw_vertices, side = points("triangle_vertices", 3), points("side_vertices", 2)
        incenter = _point_pair(record.get("incenter"))
        if raw_vertices is None or side is None or incenter is None or target is None:
            return []
        # The generator draws bisectors in the same topmost, cyclic order
        # used by its local _ordered_vertices() helper, not raw metadata order.
        vertices = ordered_triangle(raw_vertices)
        bisectors: List[np.ndarray] = []
        for index, vertex in enumerate(vertices):
            opposite_a = vertices[(index + 1) % 3]
            opposite_b = vertices[(index + 2) % 3]
            endpoint = line_intersection(vertex, incenter, opposite_a, opposite_b)
            if endpoint is None:
                return []
            bisectors.append(path(vertex, endpoint))
        perimeter = sum(
            math.hypot(
                vertices[index][0] - vertices[(index + 1) % 3][0],
                vertices[index][1] - vertices[(index + 1) % 3][1],
            )
            for index in range(3)
        )
        area = abs(
            sum(
                vertices[index][0] * vertices[(index + 1) % 3][1]
                - vertices[(index + 1) % 3][0] * vertices[index][1]
                for index in range(3)
            )
        ) / 2.0
        radius = area / max(perimeter / 2.0, 1e-9)
        # The authored incircle sweep starts at the visible tangent-point
        # candidate, rather than at the generic rightmost circle point.
        # Replaying from that same start is required to validate the
        # single-ended reveal across all triangle orientations.
        start_angle = math.atan2(
            target[1] - incenter[1],
            target[0] - incenter[0],
        )
        angles = np.linspace(start_angle, start_angle + 2.0 * np.pi, 97)
        tangent_circle = np.asarray(
            [
                (
                    incenter[0] + radius * math.cos(float(angle)),
                    incenter[1] + radius * math.sin(float(angle)),
                )
                for angle in angles
            ],
            dtype=np.float32,
        )
        return bisectors + [tangent_circle]
    if task_name == "nine_point_center":
        vertices = points("triangle_vertices", 3)
        if vertices is None or target is None:
            return []
        feet = [projection(vertices[index], vertices[(index + 1) % 3], vertices[(index + 2) % 3]) for index in range(3)]
        if any(value is None for value in feet):
            return []
        radius_point = midpoint(vertices[1], vertices[2])
        radius = math.hypot(target[0] - radius_point[0], target[1] - radius_point[1])
        return [path(vertices[index], feet[index]) for index in range(3) if feet[index] is not None] + [circle(target, radius)]
    if task_name == "triangle_center":
        vertices = points("triangle_vertices", 3)
        return [] if vertices is None else [path(vertices[index], midpoint(vertices[(index + 1) % 3], vertices[(index + 2) % 3])) for index in range(3)]

    if task_name == "inscribed_angle_arc":
        center = _point_pair(record.get("circle_center"))
        endpoints_raw = record.get("arc_endpoints")
        if center is None or not isinstance(endpoints_raw, list) or len(endpoints_raw) != 2:
            return []
        endpoints = [_point_pair(value) for value in endpoints_raw]
        if any(endpoint is None for endpoint in endpoints):
            return []
        return [
            np.asarray([center, endpoint], dtype=np.float32)
            for endpoint in endpoints
            if endpoint is not None
        ]
    if task_name == "triangle_from_midpoints":
        midpoints_raw = record.get("midpoints")
        if not isinstance(midpoints_raw, list) or len(midpoints_raw) != 3:
            return []
        midpoints = [_point_pair(value) for value in midpoints_raw]
        if any(point is None for point in midpoints):
            return []
        m1, m2, m3 = midpoints  # type: ignore[misc]
        vertices = [
            (m3[0] + m2[0] - m1[0], m3[1] + m2[1] - m1[1]),
            (m3[0] + m1[0] - m2[0], m3[1] + m1[1] - m2[1]),
            (m2[0] + m1[0] - m3[0], m2[1] + m1[1] - m3[1]),
        ]
        return [
            np.asarray([vertices[0], vertices[1]], dtype=np.float32),
            np.asarray([vertices[1], vertices[2]], dtype=np.float32),
            np.asarray([vertices[2], vertices[0]], dtype=np.float32),
        ]
    return []


def _clip_infinite_line_to_canvas(
    point: Tuple[float, float],
    direction: Tuple[float, float],
    *,
    width: float,
    height: float,
    padding: float,
) -> Optional[np.ndarray]:
    """Clip an infinite line to a padded delivery canvas."""

    dx, dy = direction
    if math.hypot(dx, dy) <= 1e-9:
        return None
    crossings: List[Tuple[float, Tuple[float, float]]] = []
    if abs(dx) > 1e-9:
        for x_value in (padding, width - padding):
            ratio = (x_value - point[0]) / dx
            y_value = point[1] + ratio * dy
            if padding - 1e-6 <= y_value <= height - padding + 1e-6:
                crossings.append((ratio, (x_value, y_value)))
    if abs(dy) > 1e-9:
        for y_value in (padding, height - padding):
            ratio = (y_value - point[1]) / dy
            x_value = point[0] + ratio * dx
            if padding - 1e-6 <= x_value <= width - padding + 1e-6:
                crossings.append((ratio, (x_value, y_value)))
    if len(crossings) < 2:
        return None
    crossings.sort(key=lambda item: item[0])
    return np.asarray(
        [crossings[0][1], crossings[-1][1]],
        dtype=np.float32,
    )


def _bespoke_coordinate_serial_path_variants(
    task_name: str,
    record: Dict[str, Any],
    expected: str,
) -> Optional[List[List[np.ndarray]]]:
    """Ordered construction variants for tasks outside the shared recorder.

    Every coordinate below is recomputed from visible/raw semantic fields and
    the independently replayed option.  The saved solution trace and saved
    answer fields are deliberately outside this function.
    """

    owned = {
        "angle_bisector",
        "circle_tangent_line",
        "circle_tangent_point",
        "circumcenter",
        "dilation_to_circle",
        "incenter",
        "internal_tangent_point",
        "isosceles_trapezoid",
        "orthocenter",
        "parallel",
        "power_of_point",
        "ray_reflect",
        "regular_octagon_vertex",
    }
    if task_name not in owned:
        return None

    def point(value: object) -> Optional[Tuple[float, float]]:
        return _point_pair(value)

    def points(key: str, count: int) -> Optional[List[Tuple[float, float]]]:
        raw = record.get(key)
        if not isinstance(raw, list) or len(raw) != count:
            return None
        parsed = [point(value) for value in raw]
        if any(value is None for value in parsed):
            return None
        return [value for value in parsed if value is not None]

    def candidate() -> Optional[Tuple[float, float]]:
        label = str(expected or "").strip().upper()
        for item in record.get("candidates") or []:
            if (
                isinstance(item, dict)
                and str(item.get("label") or "").strip().upper() == label
            ):
                return point((item.get("x"), item.get("y")))
        return None

    def path(*values: Tuple[float, float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    def ordered_triangle(
        values: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        centroid_x = sum(value[0] for value in values) / len(values)
        centroid_y = sum(value[1] for value in values) / len(values)
        cyclic = sorted(
            values,
            key=lambda value: math.atan2(
                value[1] - centroid_y,
                value[0] - centroid_x,
            ),
        )
        start = min(
            range(len(cyclic)),
            key=lambda index: (cyclic[index][1], cyclic[index][0]),
        )
        return cyclic[start:] + cyclic[:start]

    def projection(
        value: Tuple[float, float],
        first: Tuple[float, float],
        second: Tuple[float, float],
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        dx, dy = second[0] - first[0], second[1] - first[1]
        denominator = dx * dx + dy * dy
        if denominator <= 1e-9:
            return None
        ratio = (
            (value[0] - first[0]) * dx + (value[1] - first[1]) * dy
        ) / denominator
        return (first[0] + ratio * dx, first[1] + ratio * dy), ratio

    def circle(
        centre: Tuple[float, float],
        radius: float,
        *,
        start: float = 0.0,
    ) -> np.ndarray:
        angles = np.linspace(start, start + 2.0 * np.pi, 97)
        return np.asarray(
            [
                (
                    centre[0] + radius * math.cos(float(angle)),
                    centre[1] + radius * math.sin(float(angle)),
                )
                for angle in angles
            ],
            dtype=np.float32,
        )

    def line_intersection(
        first: Tuple[float, float],
        second: Tuple[float, float],
        third: Tuple[float, float],
        fourth: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        dx1, dy1 = second[0] - first[0], second[1] - first[1]
        dx2, dy2 = fourth[0] - third[0], fourth[1] - third[1]
        denominator = dx1 * dy2 - dy1 * dx2
        if abs(denominator) <= 1e-9:
            return None
        ratio = (
            (third[0] - first[0]) * dy2
            - (third[1] - first[1]) * dx2
        ) / denominator
        return (first[0] + ratio * dx1, first[1] + ratio * dy1)

    target = candidate()
    if target is None:
        return [[]]

    if task_name == "angle_bisector":
        vertex = point(record.get("vertex"))
        return [[]] if vertex is None else [[path(vertex, target)]]

    if task_name in {"circle_tangent_line", "circle_tangent_point"}:
        centre = point(record.get("circle_center"))
        if centre is None:
            return [[]]
        marker_size = 24.0
        if task_name == "circle_tangent_line":
            contact = point(record.get("contact_point"))
            if contact is None:
                return [[]]
            radius_length = math.dist(centre, contact)
            tangent_length = math.dist(contact, target)
            if min(radius_length, tangent_length) <= 1e-9:
                return [[]]
            radius_unit = (
                (contact[0] - centre[0]) / radius_length,
                (contact[1] - centre[1]) / radius_length,
            )
            tangent_unit = (
                (target[0] - contact[0]) / tangent_length,
                (target[1] - contact[1]) / tangent_length,
            )
            q1 = (
                contact[0] - radius_unit[0] * marker_size,
                contact[1] - radius_unit[1] * marker_size,
            )
            q2 = (
                q1[0] + tangent_unit[0] * marker_size,
                q1[1] + tangent_unit[1] * marker_size,
            )
            q3 = (
                contact[0] + tangent_unit[0] * marker_size,
                contact[1] + tangent_unit[1] * marker_size,
            )
            return [[
                path(centre, contact),
                path(contact, target),
                path(q1, q2, q3),
            ]]
        external = point(record.get("external_point"))
        radius = float(record.get("circle_radius") or 0.0)
        external_length = math.dist(target, external) if external is not None else 0.0
        if external is None or radius <= 1e-9 or external_length <= 1e-9:
            return [[]]
        inward = (
            (centre[0] - target[0]) / radius,
            (centre[1] - target[1]) / radius,
        )
        toward_external = (
            (external[0] - target[0]) / external_length,
            (external[1] - target[1]) / external_length,
        )
        q1 = (
            target[0] + inward[0] * marker_size,
            target[1] + inward[1] * marker_size,
        )
        q2 = (
            q1[0] + toward_external[0] * marker_size,
            q1[1] + toward_external[1] * marker_size,
        )
        q3 = (
            target[0] + toward_external[0] * marker_size,
            target[1] + toward_external[1] * marker_size,
        )
        return [[
            path(external, target),
            path(centre, target),
            path(q1, q2, q3),
        ]]

    if task_name in {"circumcenter", "incenter", "orthocenter"}:
        raw_vertices = points("triangle_vertices", 3)
        if raw_vertices is None:
            return [[]]
        vertices = ordered_triangle(raw_vertices)

        if task_name == "circumcenter":
            a, b, c = vertices
            denominator = 2.0 * (
                a[0] * (b[1] - c[1])
                + b[0] * (c[1] - a[1])
                + c[0] * (a[1] - b[1])
            )
            if abs(denominator) <= 1e-9:
                return [[]]
            norms = [value[0] ** 2 + value[1] ** 2 for value in vertices]
            centre = (
                (
                    norms[0] * (b[1] - c[1])
                    + norms[1] * (c[1] - a[1])
                    + norms[2] * (a[1] - b[1])
                )
                / denominator,
                (
                    norms[0] * (c[0] - b[0])
                    + norms[1] * (a[0] - c[0])
                    + norms[2] * (b[0] - a[0])
                )
                / denominator,
            )
            dims = record.get("canvas_dimensions") or [512, 512]
            margin = float(record.get("margin") or max(18, round(0.10 * min(dims))))
            padding = margin + renderer_line_width(record) + 5.0
            bisectors: List[np.ndarray] = []
            for index, first in enumerate(vertices):
                second = vertices[(index + 1) % 3]
                midpoint_value = (
                    (first[0] + second[0]) / 2.0,
                    (first[1] + second[1]) / 2.0,
                )
                clipped = _clip_infinite_line_to_canvas(
                    midpoint_value,
                    (-(second[1] - first[1]), second[0] - first[0]),
                    width=float(dims[0]),
                    height=float(dims[1]),
                    padding=padding,
                )
                if clipped is None:
                    return [[]]
                bisectors.append(np.asarray(clipped, dtype=np.float32))
            radius = math.dist(centre, vertices[0])
            start_vertex = min(vertices, key=lambda value: (value[1], value[0]))
            sweep = circle(
                centre,
                radius,
                start=math.atan2(
                    start_vertex[1] - centre[1],
                    start_vertex[0] - centre[0],
                ),
            )
            return [
                [bisectors[index] for index in order] + [sweep]
                for order in permutations(range(3))
            ]

        if task_name == "incenter":
            side_lengths = [
                math.dist(vertices[1], vertices[2]),
                math.dist(vertices[0], vertices[2]),
                math.dist(vertices[0], vertices[1]),
            ]
            perimeter = sum(side_lengths)
            if perimeter <= 1e-9:
                return [[]]
            centre = (
                sum(side_lengths[index] * vertices[index][0] for index in range(3))
                / perimeter,
                sum(side_lengths[index] * vertices[index][1] for index in range(3))
                / perimeter,
            )
            bisectors: List[np.ndarray] = []
            feet: List[Tuple[float, float]] = []
            for index, vertex in enumerate(vertices):
                first, second = vertices[(index + 1) % 3], vertices[(index + 2) % 3]
                endpoint = line_intersection(vertex, centre, first, second)
                foot = projection(centre, first, second)
                if endpoint is None or foot is None:
                    return [[]]
                bisectors.append(path(vertex, endpoint))
                feet.append(foot[0])
            radius = min(math.dist(centre, foot) for foot in feet)
            start_foot = min(feet, key=lambda value: (value[1], value[0]))
            sweep = circle(
                centre,
                radius,
                start=math.atan2(
                    start_foot[1] - centre[1],
                    start_foot[0] - centre[0],
                ),
            )
            return [
                [bisectors[index] for index in order] + [sweep]
                for order in permutations(range(3))
            ]

        # Orthocenter: draw every required opposite-side extension first, then
        # all three altitudes.  Orders inside those two visually equivalent
        # groups are accepted rather than treating vertex array order as a rule.
        a, b, c = vertices
        centre = line_intersection(
            a,
            (a[0] - (c[1] - b[1]), a[1] + (c[0] - b[0])),
            b,
            (b[0] - (c[1] - a[1]), b[1] + (c[0] - a[0])),
        )
        if centre is None:
            return [[]]
        extensions: List[np.ndarray] = []
        altitudes: List[np.ndarray] = []
        for index, vertex in enumerate(vertices):
            first, second = vertices[(index + 1) % 3], vertices[(index + 2) % 3]
            projected = projection(vertex, first, second)
            if projected is None:
                return [[]]
            foot, ratio = projected
            candidates = (vertex, centre, foot)
            segment = max(
                (
                    (left, right)
                    for left in candidates
                    for right in candidates
                    if left is not right
                ),
                key=lambda pair: math.dist(pair[0], pair[1]),
            )
            if ratio < 0.0:
                extensions.append(path(first, foot))
            elif ratio > 1.0:
                extensions.append(path(second, foot))
            altitudes.append(path(*segment))
        extension_orders = list(permutations(range(len(extensions)))) or [()]
        return [
            [extensions[index] for index in extension_order]
            + [altitudes[index] for index in altitude_order]
            for extension_order in extension_orders
            for altitude_order in permutations(range(3))
        ]

    if task_name == "dilation_to_circle":
        first, second = record.get("circle_1"), record.get("circle_2")
        if not isinstance(first, dict) or not isinstance(second, dict):
            return [[]]
        centre1, centre2 = point(first.get("center")), point(second.get("center"))
        radius1 = float(first.get("radius") or 0.0)
        radius2 = float(second.get("radius") or 0.0)
        if centre1 is None or centre2 is None or min(radius1, radius2) <= 0.0:
            return [[]]
        dx, dy = centre2[0] - centre1[0], centre2[1] - centre1[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            return [[]]
        theta = math.atan2(dy, dx)
        alpha = math.asin(max(-1.0, min(1.0, (radius1 - radius2) / distance)))
        variants: List[List[np.ndarray]] = []
        for tangent_angle in (
            theta + math.pi / 2.0 - alpha,
            theta - math.pi / 2.0 + alpha,
        ):
            contact = (
                centre1[0] + radius1 * math.cos(tangent_angle),
                centre1[1] + radius1 * math.sin(tangent_angle),
            )
            variants.append([path(target, contact)])
        return variants

    if task_name == "internal_tangent_point":
        first, second = record.get("circle_1"), record.get("circle_2")
        if not isinstance(first, dict) or not isinstance(second, dict):
            return [[]]
        centre1, centre2 = point(first.get("center")), point(second.get("center"))
        radius1 = float(first.get("radius") or 0.0)
        radius2 = float(second.get("radius") or 0.0)
        if centre1 is None or centre2 is None or min(radius1, radius2) <= 0.0:
            return [[]]
        distance = math.dist(centre1, centre2)
        if distance <= radius1 + radius2:
            return [[]]
        similitude = (
            (radius2 * centre1[0] + radius1 * centre2[0]) / (radius1 + radius2),
            (radius2 * centre1[1] + radius1 * centre2[1]) / (radius1 + radius2),
        )
        angle2 = math.atan2(
            similitude[1] - centre2[1],
            similitude[0] - centre2[0],
        )
        offset2 = math.acos(
            max(-1.0, min(1.0, radius2 / (radius2 * distance / (radius1 + radius2))))
        )
        contact2 = (
            centre2[0] + radius2 * math.cos(angle2 + offset2),
            centre2[1] + radius2 * math.sin(angle2 + offset2),
        )
        return [[path(target, contact2)], [path(contact2, target)]]

    if task_name == "isosceles_trapezoid":
        visible = record.get("visible_vertices")
        if not isinstance(visible, dict):
            return [[]]
        junction = point(visible.get("J"))
        base = point(visible.get("U"))
        leg = point(visible.get("V"))
        if junction is None or base is None or leg is None:
            return [[]]
        return [[path(leg, target), path(target, base)]]

    if task_name == "parallel":
        through = point(record.get("through_point"))
        if through is None:
            return [[]]
        length = math.dist(through, target)
        if length <= 1e-9:
            return [[]]
        unit = (
            (target[0] - through[0]) / length,
            (target[1] - through[1]) / length,
        )
        tail = (
            through[0] - unit[0] * 36.0,
            through[1] - unit[1] * 36.0,
        )
        point_radius = float(record.get("point_radius") or 11.0)
        end = (
            target[0] - unit[0] * (point_radius + 2.0),
            target[1] - unit[1] * (point_radius + 2.0),
        )
        return [[path(tail, end)]]

    if task_name == "power_of_point":
        external = point(record.get("external_point"))
        near = point(record.get("partial_secant_near_chord"))
        full = points("full_secant_chord", 2)
        if external is None or near is None or full is None:
            return [[]]
        product = math.dist(external, full[0]) * math.dist(external, full[1])
        near_distance = math.dist(external, near)
        if near_distance <= 1e-9:
            return [[]]
        far_distance = product / near_distance
        direction = (
            (near[0] - external[0]) / near_distance,
            (near[1] - external[1]) / near_distance,
        )
        chord_span = far_distance - near_distance
        dims = record.get("canvas_dimensions") or [512, 512]
        visible_extra = min(chord_span * 0.22, min(dims) * 0.055)
        anchor = (
            near[0] + direction[0] * visible_extra,
            near[1] + direction[1] * visible_extra,
        )
        return [[path(anchor, target)]]

    if task_name == "ray_reflect":
        if "mirror_segments" in record:
            from data.geometry.core.ray_optics import trace_mirrors
            contacts,_,_ = trace_mirrors(record["light_source"],record["incoming_ray_tip"],record["mirror_segments"])
            return [[path(tuple(record["incoming_ray_tip"]),*contacts,target)]]
        source = point(record.get("light_source"))
        contact = point(record.get("reflection_point"))
        if source is None or contact is None:
            return [[]]
        stub = (
            source[0] + 0.75 * (contact[0] - source[0]),
            source[1] + 0.75 * (contact[1] - source[1]),
        )
        return [[path(stub, contact, target)]]

    if task_name == "regular_octagon_vertex":
        shown = points("shown_vertices", 6)
        if shown is None:
            return [[]]
        previous = (
            target[0] - shown[-1][0],
            target[1] - shown[-1][1],
        )
        cosine = math.cos(math.pi / 4.0)
        sine = math.sin(math.pi / 4.0)
        following = (
            target[0] + previous[0] * cosine - previous[1] * sine,
            target[1] + previous[0] * sine + previous[1] * cosine,
        )
        return [[path(shown[-1], target, following, shown[0])]]

    return [[]]


def _coordinate_serial_path_variants(
    task_name: str,
    record: Dict[str, Any],
    expected: str,
) -> Optional[List[List[np.ndarray]]]:
    """All complete legal physical stroke orders known for one source."""

    line_width = renderer_line_width(record)
    declared = serial_construction_path_variants(
        task_name,
        record,
        expected,
        line_width=line_width,
    )
    if declared is not None:
        return declared
    bespoke = _bespoke_coordinate_serial_path_variants(
        task_name,
        record,
        expected,
    )
    if bespoke is not None:
        return bespoke
    legacy = _coordinate_serial_paths(task_name, record, expected)
    if legacy is None:
        return None
    if not legacy:
        return [[]]

    def permuted(indices: Sequence[int]) -> List[np.ndarray]:
        return [np.asarray(legacy[index], dtype=np.float32).copy() for index in indices]

    if task_name in {
        "circle_tangent_intersection",
        "secant_intersection",
        "square_vertex_from_diagonal",
    }:
        return [permuted(order) for order in permutations(range(2))]
    if task_name == "diagonal_intersection":
        variants: List[List[np.ndarray]] = []
        for order in permutations(range(2)):
            for directions in range(4):
                variant = permuted(order)
                variants.append(
                    [
                        np.asarray(path[::-1], dtype=np.float32).copy()
                        if directions & (1 << index)
                        else path
                        for index, path in enumerate(variant)
                    ]
                )
        return variants
    if task_name == "excircle_center":
        return [
            permuted((*order, 2))
            for order in permutations(range(2))
        ]
    if task_name in {
        "homothety_center",
        "napoleon_vertex",
        "spiral_similarity_center",
        "triangle_center",
    }:
        return [permuted(order) for order in permutations(range(3))]
    if task_name in {"incircle_tangent_point", "nine_point_center"}:
        return [
            permuted((*order, 3))
            for order in permutations(range(3))
        ]
    if task_name == "fermat_point":
        # Each outward equilateral outline is causally followed by its
        # opposite-vertex connector.  The three side-owned pairs are mutually
        # independent, so only the pair order is free.
        return [
            permuted(
                tuple(
                    index
                    for block in block_order
                    for index in (2 * block, 2 * block + 1)
                )
            )
            for block_order in permutations(range(3))
        ]
    if task_name == "inscribed_angle_arc":
        return [permuted(order) for order in permutations(range(2))]
    if task_name == "triangle_from_midpoints":
        forward = list(range(3))
        reverse = [
            np.asarray(path[::-1], dtype=np.float32).copy()
            for path in reversed(legacy)
        ]
        return [
            permuted(forward[start:] + forward[:start])
            for start in range(3)
        ] + [
            [
                np.asarray(path, dtype=np.float32).copy()
                for path in reverse[start:] + reverse[:start]
            ]
            for start in range(3)
        ]
    if task_name == "inscribed_square_vertex":
        return [
            permuted(range(3)),
            [
                np.asarray(path[::-1], dtype=np.float32).copy()
                for path in reversed(legacy)
            ],
        ]
    return [[np.asarray(path, dtype=np.float32).copy() for path in legacy]]


def _coordinate_all_variant_paths(
    variants: Optional[Sequence[Sequence[np.ndarray]]],
) -> Optional[List[np.ndarray]]:
    """Flatten every legal variant for containment-only corridor checks.

    A construction-presence obligation must match one complete variant, never
    a mixture of them.  Static preservation and foreign-ink containment have
    the opposite job: any stroke belonging to any legal complete variant is an
    allowed dynamic corridor.  Keeping this flattening helper separate makes
    that distinction explicit at each call site.
    """

    if variants is None:
        return None
    return [
        np.asarray(path, dtype=np.float32).copy()
        for variant in variants
        for path in variant
    ]


def _coordinate_unique_terminal_variants(
    variants: Optional[Sequence[Sequence[np.ndarray]]],
) -> Optional[List[List[np.ndarray]]]:
    """Deduplicate variants that differ only in terminal-irrelevant order.

    Process verification keeps every directed command order.  A terminal state
    instead carries an unordered set of complete centrelines, so permutations
    and whole-stroke reversals describe the same delivery obligation.  This
    canonicalisation avoids rerunning the frame-wide terminal gates up to
    twelve times for one identical construction while preserving genuinely
    different routes such as the two translation parallelogram paths.
    """

    if variants is None:
        return None
    unique: List[List[np.ndarray]] = []
    seen: set[Tuple[Tuple[Tuple[int, ...], bytes], ...]] = set()
    for variant in variants:
        copied = [
            np.asarray(path, dtype=np.float32).copy()
            for path in variant
        ]
        member_keys: List[Tuple[Tuple[int, ...], bytes]] = []
        for path in copied:
            forward = np.ascontiguousarray(path).tobytes()
            reverse = np.ascontiguousarray(path[::-1]).tobytes()
            member_keys.append((tuple(path.shape), min(forward, reverse)))
        signature = tuple(sorted(member_keys))
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(copied)
    return unique


def _derive_coordinate_route_answer(
    task_name: str,
    record: Dict[str, Any],
) -> Optional[str]:
    """Independently recover the displayed winner for every routed task.

    The generic coordinate replay owns most tasks.  The ten task-owned
    semantic verifiers intentionally keep their more specialized source
    replays, so the route-level process and static-region gates delegate to
    those same independent functions instead of reading a saved answer field.
    """

    try:
        answer, _metrics = derive_coordinate_answer(task_name, record)
        parsed = str(answer or "").strip().upper()
        return parsed or None
    except (
        CoordinateReplayError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        pass

    if task_name == "angle_bisector":
        metrics, valid = _angle_bisector_replay(record)
        correct = metrics.get("angle_bisector_derived_correct_option")
    else:
        replay = {
            "arc_connect": replay_arc_connect_geometry,
            "circle_tangent_line": replay_circle_tangent_line_geometry,
            "circle_tangent_point": replay_circle_tangent_point_geometry,
            "circumcenter": replay_circumcenter_geometry,
            "incenter": replay_incenter_geometry,
            "isosceles_trapezoid": replay_isosceles_trapezoid_geometry,
            "orthocenter": replay_orthocenter_geometry,
            "parallel": replay_parallel_geometry,
        }.get(task_name)
        if replay is None:
            return None
        _metrics, valid, correct = replay(record)
    parsed = str(correct or "").strip().upper()
    return parsed if valid and parsed else None


def _frame_background(frame: np.ndarray) -> np.ndarray:
    """Modal colour of a decimated view of the frame."""

    sample = np.asarray(frame)[::4, ::4].reshape(-1, 3)
    packed = (
        sample[:, 0].astype(np.uint32) << 16
        | sample[:, 1].astype(np.uint32) << 8
        | sample[:, 2].astype(np.uint32)
    )
    values, counts = np.unique(packed, return_counts=True)
    dominant = int(values[int(np.argmax(counts))])
    return np.array(
        [(dominant >> 16) & 0xFF, (dominant >> 8) & 0xFF, dominant & 0xFF], dtype=np.int16
    )


# Equal parts each declared stroke is hard-gated in.  Four keeps every part
# above the eight exclusive samples the attribution model needs at its densest
# rung, and makes a contiguous quarter of a stroke impossible to delete.
TERMINAL_CONSTRUCTION_PARTS = 4

# Per-stroke construction support a terminal image must reach.  The original
# six-seed calibration over the first sixteen tasks with a stroke grammar found a reference
# terminal delivery supports its weakest declared stroke at 0.75 while a
# terminal image belonging to another sample of the same task supports its
# weakest declared stroke at no more than 0.33; the bound is the geometric
# mean of those two measurements, so it clears both by 1.5x.
TERMINAL_CONSTRUCTION_SUPPORT = 0.50

# Bound both the aggregate share and the largest local off-construction
# addition. A long correct stroke cannot compensate for a compact extra branch.
# Source-edge reconstruction is removed by the observation model before these
# bounds are applied; colored additions retain their own spatial evidence.
_VARIANT_FOREIGN_INK_FRACTION_BOUND = 0.039
_VARIANT_FOREIGN_INK_COMPONENT_BOUND = 66
_VARIANT_FOREIGN_INK_TILE_BOUND = 34

# Retained samples a part needs before it can carry a hard gate of its own.
# ``_MINIMUM_JUDGEABLE_SAMPLES`` is the whole stroke's floor, spread over the
# parts: below it, "half this part is missing" and "one observation window fell
# in a gap" are the same reading, which is exactly the resolution the 0.50 floor
# above asks a part to have.  A part under the floor reports no verdict rather
# than a verdict its sample count cannot support.
_MINIMUM_JUDGEABLE_PART_SAMPLES = max(1, _MINIMUM_JUDGEABLE_SAMPLES // TERMINAL_CONSTRUCTION_PARTS)


def _stroke_part_supports(
    hits: np.ndarray,
    parameters: np.ndarray,
) -> List[Optional[float]]:
    """Support of each equal part *of the stroke*, ``None`` where none is readable.

    Splitting the retained hit row into equal counts would only measure equal
    parts of the stroke when the retained samples are spread evenly along it.
    Once the unreadable positions are dropped they are not: a stroke that runs
    under the input's own ink for a third of its length keeps all its samples in
    the readable two thirds, and equal-count parts would then gate the same two
    thirds four times over.  The parts are therefore the stroke's own equal
    spans, and a span no retained sample lands in reports ``None`` -- the
    observation model has nothing to say about it, which is not the same as its
    ink being absent.
    """

    if not len(hits):
        return [None] * TERMINAL_CONSTRUCTION_PARTS
    first, last = float(_PATH_PARAMETER_SPAN[0]), float(_PATH_PARAMETER_SPAN[1])
    scaled = (np.asarray(parameters, dtype=np.float64) - first) / max(1e-9, last - first)
    buckets = np.clip(
        (scaled * TERMINAL_CONSTRUCTION_PARTS).astype(np.int64),
        0,
        TERMINAL_CONSTRUCTION_PARTS - 1,
    )
    supports: List[Optional[float]] = []
    for part in range(TERMINAL_CONSTRUCTION_PARTS):
        selected = np.asarray(hits, dtype=bool)[buckets == part]
        supports.append(
            float(np.mean(selected))
            if len(selected) >= _MINIMUM_JUDGEABLE_PART_SAMPLES
            else None
        )
    return supports


def _stroke_parts_missing(part_supports: Sequence[Optional[float]]) -> bool:
    """Whether any readable part of the stroke falls under the support floor."""

    readable = [value for value in part_supports if value is not None]
    return bool(readable) and min(readable) < TERMINAL_CONSTRUCTION_SUPPORT


def _declared_stroke_evidence(
    paths: Sequence[np.ndarray],
    source: np.ndarray,
    *,
    clearance: float,
    occluders: Sequence[Tuple[float, float, float]],
    line_width: float,
    preceding_only: bool = False,
) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Per-stroke readable evidence, or ``None`` where the stroke cannot be read.

    Terminal evidence excludes every other declared stroke, source ink and
    overlay discs so a present stroke cannot stand in for a missing one.
    During ordered growth, only preceding strokes can already be present;
    future construction corridors must not erase readable temporal evidence.
    No attributable positions means this observation cannot judge the stroke.
    """

    # During an ordered stroke, future strokes are absent. Excluding their
    # corridors can hide nearly the entire current stroke (e.g. a later
    # tangent excircle) and erase its observable intermediate growth. The
    # terminal reader still excludes every other stroke.
    ink = source_ink_mask(source)
    return [
        _attributable_path_positions(
            path,
            excluded_paths=[other for other_index, other in enumerate(paths)
                            if other_index < index or (not preceding_only and other_index != index)],
            clearance=clearance,
            occluders=occluders,
            line_width=line_width,
            source_ink=ink,
        )
        for index, path in enumerate(paths)
    ]


def _delivery_ink_arguments(
    record: Dict[str, Any],
    paths: Sequence[np.ndarray],
    *,
    task_name: str = "",
    expected: str = "",
) -> Dict[str, Any]:
    """Source-derived geometry the ink gates need, from the record alone.

    The corridors a legal delivery may put ink in are the declared strokes plus
    whatever else the task's own renderer restates while it draws them; the
    strokes it must *carry* are only the first set, which is why the two are
    asked for separately.
    """

    line_width = renderer_line_width(record)
    corridors = list(paths)
    if task_name:
        corridors.extend(
            terminal_attribution_paths(task_name, record, expected, line_width=line_width)
        )
    # Declared means the route can name every stroke a legal delivery may add
    # -- through an obligation, an attribution corridor, or the task's stated
    # emptiness -- so any changed member off the discs and off the corridors
    # is unexplained ink.  Without a declaration the corridor test stays off:
    # rejecting ink the route cannot describe would reject legal deliveries.
    declared = bool(corridors) or str(task_name) in NO_CONSTRUCTION_TASKS
    return {
        "paths": corridors,
        # Candidate discs only.  The marker gate owns everything inside them,
        # which is what justifies exempting disc-contained ink from the
        # stroke-thickness bound; the anchor discs at stroke endpoints have no
        # such owner, so an opaque patch there must stay a thickness violation.
        "discs": overlay_occluders(record, ()),
        "clearance": foreign_ink_clearance(line_width),
        "construction_declared": declared,
    }


def _video_delivery_ink(
    candidate_video: Path,
    source_image: Path,
    record: Dict[str, Any],
    paths: Sequence[np.ndarray] = (),
    *,
    task_name: str = "",
    expected: str = "",
) -> Tuple[Dict[str, Any], bool]:
    """Static-region hold plus the terminal-state ink gates for a whole video.

    The final frame is the video's own terminal state and carries the same two
    obligations the terminal image does; every other frame has to leave the
    regions that terminal state never touches exactly as the input drew them.
    """

    capture = cv2.VideoCapture(Path(candidate_video).as_posix())
    frames: List[np.ndarray] = []
    if capture.isOpened():
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        return {"video_static_region_evaluated": False, "video_delivery_ink_evaluated": False}, True
    source_bgr = cv2.imread(Path(source_image).as_posix(), cv2.IMREAD_COLOR)
    if source_bgr is None:
        return {"video_static_region_evaluated": False, "video_delivery_ink_evaluated": False}, True
    source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    if frames[-1].shape != source.shape:
        return {
            "video_delivery_ink_evaluated": True,
            "video_delivery_ink_failure": "frame size does not match the input canvas",
        }, False
    metrics, static_pass = verify_point_target_static_regions(frames, source)
    ink_metrics, ink_pass = point_target_delivery_ink_state(
        frames[-1],
        source,
        record,
        prefix="video_terminal",
        **_delivery_ink_arguments(record, paths, task_name=task_name, expected=expected),
    )
    metrics.update(ink_metrics)
    # The image path's marker gate never sees the video, and the red decode
    # only reads the committed option, so without this a marker interior in
    # the final frame could carry anything.  Answer-free by design: which
    # marker must be committed stays the decode's question.  A task-owned
    # route (no task_name here) commits in its own palette, so its one
    # committed marker's content is left to its own decoder.
    marker_metrics, marker_pass = final_marker_state_gate(
        frames[-1],
        source,
        record,
        committed_style_owned_elsewhere=not task_name,
        prefix="video_terminal",
    )
    metrics.update(marker_metrics)
    metrics["video_static_region_pass"] = static_pass
    metrics["video_terminal_delivery_pass"] = bool(ink_pass and marker_pass)
    metrics["video_delivery_ink_evaluated"] = True
    return metrics, bool(static_pass and ink_pass and marker_pass)


def _construction_presence(
    frame: np.ndarray,
    source: np.ndarray,
    record: Dict[str, Any],
    *,
    task_name: str,
    expected: Optional[str],
    prefix: str = "terminal",
    paths_override: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[Dict[str, Any], Optional[bool]]:
    """Require every construction stroke the task declares to be present.

    The stroke grammar is the same source-derived one the whole-video path
    replays, so the delivery is checked against geometry recomputed from the
    supplied input and the rules -- never against a reference image.  ``None``
    means the dimension carries no verdict -- either the task owns no stroke
    grammar, or the observation model can read none of the strokes it declares;
    ``False`` means a stroke the delivery *can* be read for is missing, or the
    grammar could not be instantiated from the supplied semantic input.

    ``frame`` is one terminal state: the terminal answer image, or the final
    frame of a whole video, which is that video's own terminal state and owes
    the same construction.  Both callers share this one reading so a member the
    still image must carry cannot go missing from the video unnoticed.
    """

    def key(name: str) -> str:
        return f"{prefix}_construction_{name}"

    line_width = renderer_line_width(record)
    variants = (
        None
        if paths_override is not None
        else _coordinate_serial_path_variants(
            task_name,
            record,
            str(expected or ""),
        )
    )
    # The ordered stroke grammar is the whole-video contract; a terminal state
    # only has to *carry* the construction, so the tasks that declare geometry
    # without declaring a drawing order are held to it here as well.
    ordered_grammar = paths_override is not None or variants is not None
    if paths_override is not None:
        candidate_variants = [
            [np.asarray(path, dtype=np.float32).copy() for path in paths_override]
        ]
    elif variants is None:
        fallback = terminal_construction_paths(
            task_name, record, str(expected or ""), line_width=line_width
        )
        candidate_variants = None if fallback is None else [list(fallback)]
    else:
        candidate_variants = _coordinate_unique_terminal_variants(variants)
    if candidate_variants is None:
        return {key("evaluated"): False}, None
    if not candidate_variants or any(not paths for paths in candidate_variants):
        return {
            key("evaluated"): True,
            key("stroke_count"): 0,
            key("failure"): "declared stroke grammar could not be instantiated",
        }, False
    if frame is None or source is None or frame.shape != source.shape:
        return {
            key("evaluated"): True,
            key("failure"): "terminal state could not be compared with the input canvas",
        }, False
    changed_from_source = np.max(np.abs(frame.astype(np.int16) - source.astype(np.int16)), axis=2) >= 24
    # Ink the delivery put on the canvas, whatever palette it used.  The tasks
    # that declare their construction without an ordered grammar draw parts of
    # it in their own indicator colours -- an orbit in the same blue the input
    # arrow uses, say -- so a cyan-only band would report those strokes missing
    # from a legal delivery.
    construction = changed_from_source & (
        np.max(np.abs(frame.astype(np.int16) - _frame_background(frame)), axis=2) >= 24
    )
    if task_name == "circle_inversion" and record.get("_circle_cyan_required"):
        construction = _construction_ink_mask(frame, source, task_name=task_name, record=record)
    clearance = foreign_ink_clearance(line_width)

    def evaluate_variant(
        paths: Sequence[np.ndarray],
    ) -> Tuple[Dict[str, Any], Optional[bool]]:
        occluders = _coordinate_overlay_discs(task_name, record, paths)
        attributable = _declared_stroke_evidence(
            paths,
            source,
            clearance=clearance,
            occluders=occluders,
            line_width=line_width,
        )
        unjudgeable = [
            index for index, entry in enumerate(attributable) if entry is None
        ]
        judgeable = [
            index for index, entry in enumerate(attributable) if entry is not None
        ]
        if not judgeable:
            return {
                key("evaluated"): False,
                key("stroke_count"): len(paths),
                key("ordered_grammar"): ordered_grammar,
                key("unjudgeable_strokes"): unjudgeable,
            }, None
        hit_rows = [
            _sampled_path_hits(
                construction,
                paths[index],
                samples=attributable[index][0],
            )
            for index in judgeable
        ]
        supports = [float(np.mean(row)) for row in hit_rows]
        # A stroke's mean support is an aggregate over the stroke: erasing a
        # contiguous half of a declared line leaves the other half paying for
        # it, which is the same compensating shape one member down.  Every
        # readable part of every stroke is therefore gated on its own samples.
        part_supports = [
            _stroke_part_supports(
                row,
                attributable[judgeable[position]][1],
            )
            for position, row in enumerate(hit_rows)
        ]
        missing = [
            judgeable[position]
            for position, parts in enumerate(part_supports)
            if _stroke_parts_missing(parts)
        ]
        return {
            key("evaluated"): True,
            key("stroke_count"): len(paths),
            key("ordered_grammar"): ordered_grammar,
            key("judgeable_strokes"): judgeable,
            key("supports"): [round(value, 4) for value in supports],
            key("part_supports"): [
                [None if value is None else round(value, 4) for value in parts]
                for parts in part_supports
            ],
            key("part_count"): TERMINAL_CONSTRUCTION_PARTS,
            key("support_floor"): TERMINAL_CONSTRUCTION_SUPPORT,
            key("sample_counts"): [
                len(attributable[index][0]) for index in judgeable
            ],
            key("unjudgeable_strokes"): unjudgeable,
            key("missing_strokes"): missing,
        }, not missing

    attempts = [evaluate_variant(paths) for paths in candidate_variants]
    selected_index = next(
        (
            index
            for index, (_metrics, verdict) in enumerate(attempts)
            if verdict is True
        ),
        next(
            (
                index
                for index, (_metrics, verdict) in enumerate(attempts)
                if verdict is None
            ),
            0,
        ),
    )
    selected_metrics, selected_verdict = attempts[selected_index]
    if paths_override is not None:
        # The caller already owns the outer legal-variant index. Repeating a
        # local 0/1 variant tuple here makes a selected outer variant 2 appear
        # to be variant 1 in the same metrics payload.
        return selected_metrics, selected_verdict
    return {
        **selected_metrics,
        key("variant_index"): selected_index,
        key("variant_count"): len(candidate_variants),
        key("variant_verdicts"): [
            verdict for _metrics, verdict in attempts
        ],
    }, selected_verdict


def _coordinate_variant_foreign_ink(
    frame: np.ndarray,
    source: np.ndarray,
    record: Dict[str, Any],
    paths: Sequence[np.ndarray],
    *,
    task_name: str,
    expected: str,
    prefix: str,
) -> Tuple[Dict[str, Any], bool]:
    """Reject compact or aggregate ink outside one complete legal variant."""

    arguments = _delivery_ink_arguments(
        record,
        paths,
        task_name=task_name,
        expected=expected,
    )
    metrics = point_target_foreign_ink_measurement(
        frame,
        source,
        record,
        paths=arguments["paths"],
        discs=arguments["discs"],
        clearance=arguments["clearance"],
        prefix=prefix,
    )
    evaluated = metrics.get(f"{prefix}_foreign_ink_evaluated") is True
    fraction = float(metrics.get(f"{prefix}_foreign_ink_fraction") or 0.0)
    component = int(
        metrics.get(f"{prefix}_foreign_ink_largest_component_pixels") or 0
    )
    tile = int(metrics.get(f"{prefix}_foreign_ink_tile_pixels_max") or 0)
    passed = bool(
        evaluated
        and fraction <= _VARIANT_FOREIGN_INK_FRACTION_BOUND
        and component <= _VARIANT_FOREIGN_INK_COMPONENT_BOUND
        and tile <= _VARIANT_FOREIGN_INK_TILE_BOUND
    )
    metrics.update(
        {
            f"{prefix}_foreign_ink_fraction_bound": (
                _VARIANT_FOREIGN_INK_FRACTION_BOUND
            ),
            f"{prefix}_foreign_ink_component_bound": (
                _VARIANT_FOREIGN_INK_COMPONENT_BOUND
            ),
            f"{prefix}_foreign_ink_tile_bound": (
                _VARIANT_FOREIGN_INK_TILE_BOUND
            ),
            f"{prefix}_foreign_ink_pass": passed,
        }
    )
    return metrics, passed


def _verify_video_construction(
    frames: Sequence[np.ndarray],
    source: Optional[np.ndarray],
    record: Dict[str, Any],
    *,
    task_name: str,
    expected: Optional[str],
) -> Tuple[Dict[str, Any], Optional[bool]]:
    """:func:`_construction_presence` read off the video's own terminal state.

    The whole-video branch already bounds *where* a delivery may put ink, but
    containment is an aggregate: it cannot tell a construction with one member
    wholly absent from one with every member drawn, because the members that
    are present keep the added ink inside its corridors either way.  Section
    11.8 asks for a hard gate per member, and the final frame is the frame that
    owes every one of them.
    """

    if not frames or source is None:
        return {"video_terminal_construction_evaluated": False}, None
    return _construction_presence(
        np.asarray(frames[-1]),
        np.asarray(source),
        record,
        task_name=task_name,
        expected=expected,
        prefix="video_terminal",
    )


def _verify_coordinate_video_variant_delivery(
    frames: Sequence[np.ndarray],
    source: Optional[np.ndarray],
    record: Dict[str, Any],
    *,
    task_name: str,
    expected: str,
    variants: Sequence[Sequence[np.ndarray]],
) -> Tuple[Dict[str, Any], bool, Optional[bool]]:
    """Video terminal/static gates bound to one complete legal variant."""

    terminal_variants = _coordinate_unique_terminal_variants(variants) or []
    if not frames or source is None or not terminal_variants:
        return {
            "video_delivery_ink_evaluated": bool(frames and source is not None),
            "video_delivery_variant_count": len(terminal_variants),
            "video_delivery_variant_failure": (
                "video terminal state or declared stroke grammar is unavailable"
            ),
        }, False, False
    arrays = [np.asarray(frame) for frame in frames]
    source_array = np.asarray(source)
    if arrays[-1].shape != source_array.shape:
        return {
            "video_delivery_ink_evaluated": True,
            "video_delivery_variant_count": len(terminal_variants),
            "video_delivery_variant_failure": (
                "frame size does not match the input canvas"
            ),
        }, False, False
    static_metrics, static_pass = verify_point_target_static_regions(
        arrays,
        source_array,
    )
    marker_metrics, marker_pass = final_marker_state_gate(
        arrays[-1],
        source_array,
        record,
        committed_style_owned_elsewhere=False,
        prefix="video_terminal",
    )
    attempts: List[Tuple[Dict[str, Any], bool, Optional[bool], bool]] = []
    for paths in terminal_variants:
        ink_metrics, ink_pass = point_target_delivery_ink_state(
            arrays[-1],
            source_array,
            record,
            prefix="video_terminal",
            **_delivery_ink_arguments(
                record,
                paths,
                task_name=task_name,
                expected=expected,
            ),
        )
        construction_metrics, construction_verdict = _construction_presence(
            arrays[-1],
            source_array,
            record,
            task_name=task_name,
            expected=expected,
            prefix="video_terminal",
            paths_override=paths,
        )
        foreign_metrics, foreign_pass = _coordinate_variant_foreign_ink(
            arrays[-1],
            source_array,
            record,
            paths,
            task_name=task_name,
            expected=expected,
            prefix="video_terminal",
        )
        variant_ink_pass = bool(ink_pass and foreign_pass)
        combined = bool(
            variant_ink_pass and construction_verdict is not False
        )
        attempts.append(
            (
                {
                    **ink_metrics,
                    **foreign_metrics,
                    **construction_metrics,
                    "video_terminal_variant_ink_pass": variant_ink_pass,
                },
                variant_ink_pass,
                construction_verdict,
                combined,
            )
        )
    selected_index = next(
        (
            index
            for index, (_metrics, _ink, _construction, passed) in enumerate(
                attempts
            )
            if passed
        ),
        next(
            (
                index
                for index, (_metrics, ink, _construction, _passed) in enumerate(
                    attempts
                )
                if ink
            ),
            0,
        ),
    )
    selected_metrics, _ink_pass, selected_construction, selected_variant_pass = (
        attempts[selected_index]
    )
    passed = bool(static_pass and marker_pass and selected_variant_pass)
    return {
        **static_metrics,
        **selected_metrics,
        **marker_metrics,
        "video_static_region_pass": static_pass,
        "video_delivery_ink_evaluated": True,
        "video_delivery_variant_index": selected_index,
        "video_delivery_variant_count": len(attempts),
        "video_delivery_variant_verdicts": [
            verdict for _metrics, _ink, _construction, verdict in attempts
        ],
        "video_delivery_variant_ink_verdicts": [
            ink for _metrics, ink, _construction, _verdict in attempts
        ],
        "video_delivery_variant_construction_verdicts": [
            construction
            for _metrics, _ink, construction, _verdict in attempts
        ],
        "video_delivery_variant_pass": selected_variant_pass,
        "video_terminal_delivery_pass": bool(marker_pass and selected_variant_pass),
        "video_terminal_construction_pass": selected_construction,
    }, passed, selected_construction


def _circle_inversion_prompt_contract(prompt: str) -> Dict[str, bool]:
    """Read inversion pen constraints from the supplied public instruction."""
    output = _explicit_prompt_section(prompt, "Required Output", "Output", "Deliverable") or prompt
    source_started = re.search(
        r"\b(?:from|at)\s+(?:the\s+)?(?:source(?:\s+(?:point|marker))?|p)(?!['’′])\b",
        output, flags=re.IGNORECASE,
    ) is not None
    return {"_circle_source_started": source_started,
            "_circle_cyan_required": re.search(r"\bcyan\b", output, flags=re.IGNORECASE) is not None}


def _construction_ink_mask(
    frame: np.ndarray,
    source: np.ndarray,
    *,
    task_name: str = "",
    record: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Pixels this frame draws the construction layer in.

    Tasks that prescribe cyan separate the construction from the red highlight
    through that palette. An inversion prompt may leave its radial segment's
    color open, so its source-derived path and marker occluders own that distinction. The
    departure from the supplied input is the same 24/255 the still-image gates
    call this delivery's ink, so "the video drew here" and "the terminal image
    drew here" mean one thing.
    """

    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    changed_from_source = (
        np.max(np.abs(frame.astype(np.int16) - source.astype(np.int16)), axis=2) >= 24
    )
    if task_name == "circle_inversion" and not (record or {}).get("_circle_cyan_required"):
        return changed_from_source & observed_added_ink(frame, source) & (
            np.max(np.abs(frame.astype(np.int16) - _frame_background(frame)), axis=2) >= 24
        )
    cyan_or_blue = (
        (red < 110)
        & (green > 88)
        & (blue > 112)
        & (green - red > 30)
        & (blue - red > 50)
    )
    # Double rotation deliberately keeps the two input-arrow palettes: blue
    # for the first causal turn and green (34, 145, 88) for the second.  The
    # generic cyan/blue predicate above sees the first and cannot see the
    # second.  This second branch is derived from that declared renderer colour
    # with the same codec margin as the first branch; it is task-scoped so green
    # scene semantics elsewhere never become construction evidence.
    green_turn = (
        (red < 110)
        & (green > 92)
        & (blue < 150)
        & (green - red > 30)
        & (green - blue > 20)
    )
    palette = cyan_or_blue | (green_turn if task_name == "double_rotation" else False)
    return palette & changed_from_source


def _verify_serial_coordinate_strokes(
    frames: Sequence[np.ndarray], source: np.ndarray, record: Dict[str, Any], *,
    task_name: str, expected: str, construction_start: int, construction_end: int,
) -> Tuple[Dict[str, Any], Optional[bool]]:
    """Check each complete legal stroke order; alternatives never waive a rule."""
    variants = _coordinate_serial_path_variants(task_name, record, expected)
    if variants is None:
        return {"video_serial_stroke_required": False}, None
    first_paths = variants[0] if variants else ()
    programme = _coordinate_programme(task_name, first_paths)
    # A single arc still exposes growth when occlusion makes the progress bank decline.
    if (programme is not None and len(first_paths) > 1
            and any(str(command.get("kind")) == _SMOOTH for command in programme)):
        return {"video_serial_stroke_required": False,
                "video_serial_stroke_reason": "compound smooth turns are evaluated by the pipeline-rendered progress bank"}, None
    decisions = []
    for index, paths in enumerate(variants):
        if task_name == "circle_inversion":
            # Growth and terminal containment must choose the same complete
            # connector; a short prefix cannot excuse two-ended radial growth.
            terminal_metrics, terminal_pass = _coordinate_variant_foreign_ink(
                frames[-1], source, record, paths,
                task_name=task_name, expected=expected, prefix="video_serial_terminal",
            )
            if not terminal_pass:
                decisions.append((terminal_metrics, False))
                continue
        metrics, passed = _verify_serial_coordinate_variant(
            frames, source, record, task_name=task_name, expected=expected,
            construction_start=construction_start, construction_end=construction_end, paths=paths)
        decisions.append((metrics, passed))
        if passed is True:
            return {**metrics, "video_serial_stroke_variant_count": len(variants),
                    "video_serial_stroke_selected_variant": index}, True
    metrics = decisions[0][0] if decisions else {}
    return {**metrics, "video_serial_stroke_required": True,
            "video_serial_stroke_variant_count": len(variants),
            "video_serial_stroke_variant_verdicts": [decision is True for _, decision in decisions]}, False


def _verify_semantic_line_endpoint(frames, source, record, *, task_name, expected):
    """Require the visible tail of a finite geometric line, independent of pen width."""
    variants = _coordinate_serial_path_variants(task_name, record, expected)
    if not variants or len(variants[0]) != 1:
        return {"video_source_endpoint_evaluated": False}, None
    paths = variants[0]
    path = np.asarray(paths[0], dtype=np.float64)
    length = float(np.linalg.norm(path[-1] - path[0]))
    parameters = np.linspace(0.0, 1.0, max(2, int(np.ceil(length)) + 1))
    samples = path[0] + parameters[:, None] * (path[-1] - path[0])
    pixels = np.rint(samples).astype(np.int32)
    readable = ~source_ink_mask(source)[pixels[:, 1], pixels[:, 0]]
    radius = float(record.get("point_radius") or PointTargetPuzzleGenerator.POINT_RADIUS)
    radius += float(PointTargetPuzzleGenerator.CANDIDATE_OUTLINE_WIDTH) / 2.0
    for candidate in record.get("candidates") or ():
        center = np.asarray([candidate["x"], candidate["y"]], dtype=np.float64)
        readable &= np.linalg.norm(samples - center, axis=1) > radius
    samples, parameters = samples[readable], parameters[readable]
    if not len(samples):
        return {"video_source_endpoint_evaluated": False}, None
    # Read the actual visible endpoint. The generic mid-stroke observer
    # excludes endpoint fractions and uses full-corridor clearances; neither
    # exclusion can determine whether this finite line reaches its marker.
    tail = parameters >= np.max(parameters) - 0.10 * np.ptp(parameters)
    ink = _construction_ink_mask(frames[-1], source, task_name=task_name, record=record)
    hits = _sampled_path_hits(ink, paths[0], radius=3, samples=samples[tail])
    support = float(np.mean(hits))
    passed = support >= 0.70
    return {"video_source_endpoint_evaluated": True,
            "video_source_endpoint_sample_count": int(tail.sum()),
            "video_source_endpoint_support": support,
            "video_source_endpoint_pass": passed}, passed


def _closed_curve_occupancy_growth(hits: np.ndarray, parameters: np.ndarray):
    """Read a retained circular sweep from visible pen occupancy.

    Frame-to-frame RGB changes include VAE edge flicker on already completed
    arcs. They cannot establish where a geometric sweep begins or advances.
    The inputs here are pen hits at fixed, independently derived arc positions.
    """
    bins = np.minimum(7, (parameters * 8).astype(int))
    readable = [index for index in range(8) if np.count_nonzero(bins == index) >= 2]
    if len(readable) < 4 or len(hits) < 3:
        return False, {"readable_bins": len(readable), "supports": []}
    supports = np.asarray([hits[:, bins == index].mean(axis=1) for index in readable]).T
    active = supports >= 0.50
    activation = [next((i for i, row in enumerate(active) if row[j]), len(hits))
                  for j in range(len(readable))]
    # A circle may start at any angle and sweep in either direction. One
    # advancing pen produces a cyclically ordered activation, never two
    # disconnected growing arcs or a simultaneous whole-circle fade.
    orders = [list(range(i, len(readable))) + list(range(i)) for i in range(len(readable))]
    orders += [list(reversed(order)) for order in orders]
    ordered = any(all(activation[order[i]] <= activation[order[i + 1]]
                      for i in range(len(order) - 1)) for order in orders)
    contiguous = all(np.count_nonzero(row != np.roll(row, 1)) <= 2 for row in active)
    partial = sum(0.08 < float(row.mean()) < 0.90 for row in supports)
    terminal = supports[-1].tolist()
    retained = all(min(supports[index:, j], default=0.0) >= 0.36
                   for j, index in enumerate(activation) if index < len(hits))
    gradual = len(set(activation)) >= 3 and partial >= 2
    passed = bool(ordered and contiguous and gradual and retained and min(terminal) >= 0.62)
    return passed, {"observation": "source_derived_pen_occupancy",
                    "readable_bins": len(readable), "supports": supports.tolist(),
                    "activation_frames": activation, "ordered_activation": ordered,
                    "prefix_growth": contiguous, "partial_frame_count": partial,
                    "gradual": gradual, "terminal_supports": terminal, "retained": retained}


def _verify_serial_coordinate_variant(
    frames: Sequence[np.ndarray], source: np.ndarray, record: Dict[str, Any], *,
    task_name: str, expected: str, construction_start: int, construction_end: int,
    paths: Sequence[np.ndarray],
    construction_masks: Optional[Sequence[np.ndarray]] = None,
    source_growth_masks: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[Dict[str, Any], Optional[bool]]:
    """Read source-derived directed strokes, optionally in a task-owned palette."""
    paths = list(paths)
    if not paths or construction_end <= construction_start:
        return {
            "video_serial_stroke_required": True,
            "video_serial_stroke_count": len(paths),
            "video_serial_stroke_start_frames": [],
            "video_serial_stroke_growth_metrics": [],
        }, False
    cyan_masks = (list(construction_masks) if construction_masks is not None else [
        _construction_ink_mask(frame, source, task_name=task_name, record=record)
        for frame in frames
    ])
    line_width = renderer_line_width(record)
    clearance = foreign_ink_clearance(line_width)
    occluders = _coordinate_overlay_discs(task_name, record, paths)
    evidence = _declared_stroke_evidence(
        paths,
        source,
        clearance=clearance,
        occluders=occluders,
        line_width=line_width,
        preceding_only=True,
    )
    declared_count = len(paths)
    unjudgeable = [index for index, entry in enumerate(evidence) if entry is None]
    judgeable = [index for index, entry in enumerate(evidence) if entry is not None]
    if not judgeable:
        return {
            "video_serial_stroke_required": False,
            "video_serial_stroke_count": declared_count,
            "video_serial_stroke_unjudgeable_strokes": unjudgeable,
            "video_serial_stroke_start_frames": [],
            "video_serial_stroke_growth_metrics": [],
        }, None
    # An unreadable stroke is dropped from the ordered analysis rather than
    # failed on: the declared order restricted to the readable strokes is still
    # a real obligation -- the one before an unreadable stroke must finish before
    # the one after it begins. Terminal completion independently requires the
    # final image's evidence after accounting for all declared strokes.
    paths = [paths[index] for index in judgeable]
    attributable = [evidence[index][0] for index in judgeable]
    all_rows = [
        [_sampled_path_hits(mask, path, radius=3, samples=samples) for mask in cyan_masks]
        for path, samples in zip(paths, attributable)
    ]
    support_rows = [
        [float(np.mean(row)) for row in path_rows]
        for path_rows in all_rows
    ]
    start_frames: List[int] = []
    for supports in support_rows:
        start = next(
            (
                index
                for index in range(construction_start, construction_end)
                if supports[index] >= 0.28
            ),
            construction_end,
        )
        start_frames.append(start)
    # A stroke's growth window has to contain the whole stroke.  The 0.28
    # crossing above is the robust "this stroke has clearly begun" signal used
    # for ordering, but by construction it lands more than a quarter of the way
    # into the stroke, and a closed sweep analysed from there has already lost
    # the bins it drew first.  The window therefore opens on the last frame in
    # which the stroke was still absent and closes when the next one first
    # inks, which also makes "the previous stroke was finished" observable.
    first_ink_frames: List[int] = []
    for index, supports in enumerate(support_rows):
        floor = 1.5 / max(1, len(attributable[index]))
        first = next(
            (
                frame_index
                for frame_index in range(construction_start, construction_end)
                if supports[frame_index] >= floor
            ),
            start_frames[index],
        )
        first_ink_frames.append(max(construction_start, min(first, start_frames[index]) - 1))
    growth_metrics: List[Dict[str, Any]] = []
    growth_passes: List[bool] = []
    previous_complete = True
    no_future_early_ink = True
    closed_curve_growth: Dict[str, Any] | None = None
    ordered_starts = bool(
        all(
            construction_start <= start_frames[index] < start_frames[index + 1] < construction_end
            for index in range(len(start_frames) - 1)
        )
        and start_frames[-1] < construction_end
    )
    for path_index, start in enumerate(first_ink_frames):
        end = (
            first_ink_frames[path_index + 1]
            if path_index + 1 < len(first_ink_frames)
            else construction_end
        )
        if start >= end:
            growth_passes.append(False)
            growth_metrics.append({"supports": []})
            previous_complete = False
            continue
        # ``end`` is the last frame before the next stroke first inks. It
        # carries this stroke's completion, so it belongs to both observations:
        # the preceding stroke's finished endpoint and the next stroke's empty
        # start. Dropping it rejects a legal short stroke followed immediately
        # by the next operation.
        path_points = np.asarray(paths[path_index], dtype=np.float32)
        is_closed_curve = bool(
            len(path_points) >= 17
            and float(np.linalg.norm(path_points[0] - path_points[-1])) <= 1.5
        )
        if is_closed_curve:
            dense = _attributable_path_positions(
                path_points, sample_count=192, excluded_paths=paths[:path_index],
                clearance=clearance, occluders=occluders, line_width=line_width,
                source_ink=source_ink_mask(source), parameter_span=(0.0, 1.0),
            )
            if dense is None:
                passed, closed_curve_growth = False, {"readable_bins": 0}
            else:
                samples, parameters = dense
                hits = np.asarray([
                    _sampled_path_hits(mask, path_points, radius=3, samples=samples)
                    for mask in cyan_masks[start:end + 1]
                ])
                passed, closed_curve_growth = _closed_curve_occupancy_growth(hits, parameters)
            metrics = {"closed_curve": True, "closed_curve_growth": closed_curve_growth}
        else:
            require_partial = True
            if source_growth_masks is not None:
                source_rows = [
                    _sampled_path_hits(mask, path_points, radius=3, samples=attributable[path_index])
                    for mask in source_growth_masks
                ]
                _, visibility = _single_endpoint_growth(source_rows, final_support=0.62)
                # A short exposed fragment can appear within one source frame.
                # Its presence, serial order and retention remain obligations.
                require_partial = not (
                    visibility.get("cumulative")
                    and visibility["supports"][-1] >= 0.62
                    and not visibility["directional_partial_frame"]
                )
            passed, metrics = _single_endpoint_growth(
                all_rows[path_index][start:end + 1],
                final_support=0.62,
                require_partial=require_partial,
            )
            if task_name in {"fermat_point", "triangle_from_midpoints"} and not passed:
                # The public task names the endpoints of each undirected
                # construction segment, without choosing the pen's start end.
                reverse_pass, reverse_metrics = _single_endpoint_growth(
                    [row[::-1] for row in all_rows[path_index][start:end + 1]],
                    final_support=0.62,
                )
                if reverse_pass:
                    passed, metrics = True, {**reverse_metrics, "reversed_endpoint": True}
        growth_passes.append(passed)
        growth_metrics.append(metrics)
        # Ordering stays anchored to the robust 0.28 crossing: the first-ink
        # frames above only widen each growth window, they must not shorten the
        # span over which a later stroke has to stay absent.
        declared_start = start_frames[path_index]
        if path_index > 0:
            previous_complete = bool(
                previous_complete
                and support_rows[path_index - 1][max(construction_start, declared_start - 1)] >= 0.60
            )
        for future_index in range(path_index + 1, len(paths)):
            no_future_early_ink = bool(
                no_future_early_ink
                and max(
                    support_rows[future_index][construction_start:declared_start],
                    default=0.0,
                ) < 0.28
            )
    retained = all(
        min(
            supports[
                (
                    start_frames[index + 1]
                    if index + 1 < len(start_frames)
                    else max(start_frames[index], construction_end - 2)
                )
                : construction_end
            ],
            default=0.0,
        )
        >= 0.56
        for index, supports in enumerate(support_rows)
    )
    simultaneous_partial_counts: List[int] = []
    for frame_index in range(construction_start, construction_end):
        partial_count = 0
        for supports in support_rows:
            support = supports[frame_index]
            final_support = max(supports[max(construction_start, construction_end - 2):construction_end], default=0.0)
            if 0.28 <= support <= max(0.30, final_support - 0.10):
                partial_count += 1
        simultaneous_partial_counts.append(partial_count)
    max_simultaneous_partial = max(simultaneous_partial_counts, default=0)
    passed = bool(
        ordered_starts
        and all(growth_passes)
        and previous_complete
        and no_future_early_ink
        and retained
        and max_simultaneous_partial <= 1
    )
    return {
        "video_serial_stroke_required": True,
        "video_serial_stroke_count": declared_count,
        "video_serial_stroke_unjudgeable_strokes": unjudgeable,
        "video_serial_stroke_judgeable_strokes": judgeable,
        "video_serial_stroke_start_frames": start_frames,
        "video_serial_stroke_first_ink_frames": first_ink_frames,
        "video_serial_stroke_sample_counts": [len(samples) for samples in attributable],
        "video_serial_stroke_ordered_starts": ordered_starts,
        "video_serial_stroke_growth_passes": growth_passes,
        "video_serial_stroke_growth_metrics": growth_metrics,
        "video_serial_stroke_previous_complete": previous_complete,
        "video_serial_stroke_no_future_early_ink": no_future_early_ink,
        "video_serial_stroke_retained": retained,
        "video_serial_stroke_partial_counts": simultaneous_partial_counts,
        "video_serial_stroke_max_simultaneous_partial": max_simultaneous_partial,
        "video_serial_stroke_closed_curve_growth": closed_curve_growth,
    }, passed


# The tone estimate below is a per-channel constant, so it is fixed by any large
# sample of the region that carries it; reading every fifth row and column of a
# 512x512 canvas leaves about ten thousand pixels, four orders of magnitude more
# than a median needs to sit well inside the one grey level the mask can
# resolve, and keeps the whole-video pass at one small median per frame instead
# of one over a quarter of a million pixels.
_TONE_PROBE_STRIDE = 5

# A lost pixel counts only where the loss is wider than one pixel.  A pen tip
# that advances a fraction of a pixel per frame re-quantises the antialiased
# pixels at its own end, so a legal frame drops a one-pixel rim from the mask
# without anything having been erased; requiring a lost pixel to have all four
# of its neighbours lost as well removes exactly that rim, because a rim is one
# pixel wide by construction while a stroke segment a video really took back is
# as wide as the stroke.
_INK_REGRESSION_NEIGHBOURHOOD = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
)

# How many such pixels one frame may lose against the frame before it.
# Measured over all 67 routed tasks at sample indices 0-4 and four seeds, a
# legal reference delivery loses at most 3, and a uniform +3/255 tone shift on
# every frame loses the same 3 -- the widest raw rim measured, 96 pixels on
# arc_connect, erodes away entirely, and what survives on four tasks is the
# two-or-three-pixel corner a mitred joint re-quantises as the tip turns.  The
# tightest corruption a battery can build, a six-frame reversal butted against
# the terminal hold, loses 55 at its cheapest and 68 at the median.  The floor
# is the geometric mean of those two measurements, so it clears the legal
# maximum by 4x and sits 4.6x below the cheapest corruption.
_INK_REGRESSION_FLOOR = 12


def _channel_max(image: np.ndarray) -> np.ndarray:
    """Largest of an image's three channels at every pixel.

    ``np.max(..., axis=2)`` reduces along the shortest, fastest-varying axis and
    pays an order of magnitude for it (6.0 ms against 0.34 ms per 512x512 frame,
    measured); two pairwise maxima over contiguous planes read the same value.
    """

    return np.maximum(np.maximum(image[:, :, 0], image[:, :, 1]), image[:, :, 2])


def _added_ink_masks(
    frames: Sequence[np.ndarray],
    source: np.ndarray,
) -> List[np.ndarray]:
    """Pixels each frame adds to the supplied input, with uniform tone removed.

    ``_SOURCE_INK_DELTA`` is the same departure at which the still-image gates
    call a pixel this delivery's ink, so "the video drew here" and "the terminal
    image drew here" mean one thing on this route.

    A whole-video verifier must accept style variance that changes
    no semantics, and a uniform tone change is the whole of that freedom, so the
    shared :func:`detoned` removes it before anything is measured.  Its estimate
    is taken over the pixels *this* candidate never changes, which is where a
    legal frame equals the input and any difference is therefore tone rather
    than drawing; a candidate that repaints the entire canvas leaves no such
    region and is measured against the whole frame, where the mask below then
    reports its repainting rather than absorbing it.
    """

    from data.common.progress_alignment import detoned

    reference = np.ascontiguousarray(source, dtype=np.uint8)
    raw = [
        _channel_max(cv2.absdiff(np.ascontiguousarray(frame, dtype=np.uint8), reference))
        >= _SOURCE_INK_DELTA
        for frame in frames
    ]
    if not raw:
        return []
    untouched = ~np.logical_or.reduce(raw)
    stride = _TONE_PROBE_STRIDE
    sampled = untouched[::stride, ::stride]
    probe: Optional[np.ndarray] = None
    if bool(sampled.any()):
        probe = np.zeros_like(untouched)
        probe[::stride, ::stride] = sampled
    else:
        sampled = None
    probe_source = np.ascontiguousarray(source[::stride, ::stride])
    floor = float(_SOURCE_INK_DELTA)
    scene = np.asarray(source, dtype=np.float32)
    masks: List[np.ndarray] = []
    for frame, untoned in zip(frames, raw):
        # The offset is read off exactly the pixels ``detoned`` reads it off, so
        # this is that function's own estimate taken on a strided copy of them
        # rather than on a full-canvas gather.  A candidate carrying no tone
        # shift estimates exactly zero, and subtracting zero cannot move an
        # integer difference across an integer threshold, so its mask is the
        # untoned one and the float pass is skipped entirely.
        offset = tone_offset(
            np.ascontiguousarray(frame[::stride, ::stride]),
            probe_source,
            sampled,
        )
        if not bool(np.any(offset)):
            masks.append(untoned)
            continue
        masks.append(
            _channel_max(np.abs(detoned([frame], source, probe)[0] - scene)) >= floor
        )
    return masks


def _decode_rgb_frames(video_path: Path) -> List[np.ndarray]:
    """Decode every frame of a candidate video to RGB."""

    capture = cv2.VideoCapture(Path(video_path).as_posix())
    frames: List[np.ndarray] = []
    if capture.isOpened():
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def _verify_cumulative_ink_order(
    frames: Sequence[np.ndarray],
    source: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], bool]:
    """Require the ink a frame added to the input never to be taken back.

    This route's whole-video contract is one cumulative construction: strokes
    grow from a fixed end, commits land in place, and nothing the delivery drew
    is ever removed before the terminal hold.  Reading that as a *set* -- the
    pixels this frame differs from the supplied input in -- is what makes the
    reading order-keyed.  The checks it joins read one scalar per frame instead,
    the fraction of the canvas that differs from the input, and a scalar that
    only has to stay non-decreasing within a tolerance cannot see any corruption
    whose endpoints agree: a repeated segment advances the scalar by nothing at
    all, and a reversal or a permutation spread over enough frames moves it by
    less than the tolerance at every single step however much process it spans.
    A set has no such freedom.  The frames of a repeat, a reversal or a
    permutation must show, at some seam, ink that a frame before it already had,
    and the seam is as visible in a slow phase as in a fast one, so the verdict
    does not depend on where in the process the corruption sits.

    A deletion is deliberately outside what this reading can decide: removing
    frames removes process without ever taking ink back, so the sequence it
    leaves is still cumulative.  Bounding it needs the process rendered at
    ascending progress to compare a jump against, which is what
    :func:`_read_coordinate_process_order` below supplies for the tasks whose
    stroke order and render programme this route can name.  The gate claims only
    what it measures.
    """

    if source is None or len(frames) < 2 or any(
        frame.shape != source.shape for frame in frames
    ):
        return {
            "video_ink_order_available": False,
            "video_ink_order_reason": "the candidate video and its input could not be read as one scene",
            "video_cumulative_ink_order_pass": False,
        }, False

    masks = _added_ink_masks(frames, source)
    regressions: List[int] = []
    for index in range(len(masks) - 1):
        lost = (masks[index] & ~masks[index + 1]).astype(np.uint8)
        regressions.append(
            int(np.count_nonzero(cv2.erode(lost, _INK_REGRESSION_NEIGHBOURHOOD)))
        )
    worst = max(regressions, default=0)
    passed = worst <= _INK_REGRESSION_FLOOR
    return {
        "video_ink_order_available": True,
        "video_ink_order_verifier": "cumulative_added_ink_set_monotonicity_v1",
        "video_ink_order_frame_count": len(frames),
        "video_ink_order_terminal_ink_pixels": int(masks[-1].sum()),
        "video_ink_regression_pixels": regressions,
        "video_ink_regression_worst": worst,
        "video_ink_regression_worst_frame": (
            int(np.argmax(regressions)) + 1 if regressions else -1
        ),
        "video_ink_regression_floor": _INK_REGRESSION_FLOOR,
        "video_cumulative_ink_order_pass": passed,
        "video_ink_order_reason": "ok" if passed else "a frame took back ink an earlier frame had drawn",
    }, passed


# --------------------------------------------------- source static hold -----

# Radius of the structuring element the declared geometry is swept by.  A
# declared corridor is a polyline of no width; the renderer strokes it at
# ``renderer_line_width`` and the WebM raster spreads that further, and the
# candidate markers sit at their own radius.  Sampling the construction at a
# few states cannot give this region: a stroke advances further per frame than
# the pen is wide, so sampled states leave gaps a legal frame lands in.  The
# region is therefore the Minkowski sum of the whole declared geometry with a
# disc -- every position the pen can occupy -- taken once.

# Originally measured over the 52 tasks whose construction the shared tables
# named at that revision, at
# sample indices 0-4 of seeds 4001 and 91733, the furthest a legal reference
# frame inks from its declared geometry is 8.0 px (``nine_point_center``, the
# same reading on both seeds); the value below carries that plus the
# renderer's own half stroke width.  Three tasks -- ``translation_composition``,
# ``vector_translation`` and ``glide_reflection`` -- put 2 to 20 pixels of an
# indicator further out than that, and they are deliberately not bought a
# wider sweep: 20 pixels is 2% of a 32x32 region, which the region statistic
# below carries at 0.0037 against a 0.0314 bound, while widening the sweep for
# them would loosen it route-wide.
_SOURCE_DYNAMIC_SWEEP = 11


def _sweep(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate by an elliptical element of ``radius`` -- the Minkowski sum."""

    if radius <= 0:
        return mask.astype(bool)
    size = 2 * int(radius) + 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), element) > 0


def _polyline(*points: Tuple[float, float]) -> np.ndarray:
    return np.asarray(points, dtype=np.float32)


def _foot_of_perpendicular(
    point: Tuple[float, float],
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> Tuple[float, float]:
    dx, dy = second[0] - first[0], second[1] - first[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-9:
        return first
    ratio = ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / denominator
    return (first[0] + ratio * dx, first[1] + ratio * dy)


def _circle_polyline(centre: Tuple[float, float], radius: float, *, samples: int = 97) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, samples)
    return np.stack(
        [centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)], axis=1
    ).astype(np.float32)


def _containment_corridors(
    task_name: str,
    record: Dict[str, Any],
) -> Optional[List[np.ndarray]]:
    """Where a legal delivery of this task may put ink, for tasks that own their
    verifier and declare no construction to the shared tables.

    This is a *containment* declaration, and deliberately not the same
    obligation as :func:`terminal_construction_paths`, which names the strokes a
    delivery must be shown to *carry* and so feeds the still-image evidence
    gate.  A containment corridor only has to hold everything a legal delivery
    draws; it is never read as evidence that a stroke is present, so a corridor
    wider than the stroke costs nothing but reach and cannot reject a legal
    delivery for a stroke it could not see.

    Only the three tasks that measurably needed it are here.  Before this
    declaration a 24x24 static patch pasted from the third frame onward
    survived at 61 of 64 canvas positions on ``orthocenter``, 62 of 64 on
    ``regular_octagon_vertex`` and 61 of 64 on ``incenter``; the remaining
    undeclared tasks of the route survived at 0 or 1 of 64, so nothing is
    gained by describing geometry for them that their own verifiers already
    refuse.
    """

    if task_name in {"orthocenter", "incenter"}:
        raw = record.get("triangle_vertices")
        if not isinstance(raw, list) or len(raw) != 3:
            return None
        parsed = [_point_pair(value) for value in raw]
        if any(value is None for value in parsed):
            return None
        vertices = [value for value in parsed if value is not None]
        corridors = [
            _polyline(vertices[index], vertices[(index + 1) % 3]) for index in range(3)
        ]
        if task_name == "orthocenter":
            feet = [
                _foot_of_perpendicular(
                    vertices[index], vertices[(index + 1) % 3], vertices[(index + 2) % 3]
                )
                for index in range(3)
            ]
            # An obtuse triangle puts the orthocentre outside itself, and the
            # solution draws each altitude out to meet it, so the corridor has
            # to run past the foot as well as up to it.
            centre = _line_pair_intersection(
                vertices[0], feet[0], vertices[1], feet[1]
            )
            for index in range(3):
                corridors.append(_polyline(vertices[index], feet[index]))
                corridors.append(_polyline(vertices[(index + 1) % 3], feet[index]))
                corridors.append(_polyline(vertices[(index + 2) % 3], feet[index]))
                if centre is not None:
                    corridors.append(_polyline(vertices[index], centre))
        else:
            sides = [
                math.dist(vertices[(index + 1) % 3], vertices[(index + 2) % 3])
                for index in range(3)
            ]
            perimeter = float(sum(sides))
            if perimeter <= 1e-6:
                return None
            centre = (
                sum(sides[index] * vertices[index][0] for index in range(3)) / perimeter,
                sum(sides[index] * vertices[index][1] for index in range(3)) / perimeter,
            )
            area = abs(
                (vertices[1][0] - vertices[0][0]) * (vertices[2][1] - vertices[0][1])
                - (vertices[2][0] - vertices[0][0]) * (vertices[1][1] - vertices[0][1])
            ) / 2.0
            for index in range(3):
                # The bisector is drawn from the vertex all the way to the
                # opposite side, not just as far as the centre it passes
                # through, so the corridor has to run to the side as well.
                opposite = _line_pair_intersection(
                    vertices[index],
                    centre,
                    vertices[(index + 1) % 3],
                    vertices[(index + 2) % 3],
                )
                corridors.append(_polyline(vertices[index], centre))
                if opposite is not None:
                    corridors.append(_polyline(vertices[index], opposite))
            corridors.append(_circle_polyline(centre, 2.0 * area / perimeter))
        return corridors

    if task_name == "regular_octagon_vertex":
        raw = record.get("shown_vertices")
        if not isinstance(raw, list) or len(raw) < 3:
            return None
        parsed = [_point_pair(value) for value in raw]
        if any(value is None for value in parsed):
            return None
        shown = [value for value in parsed if value is not None]
        centre = _circumcentre(shown[0], shown[1], shown[2])
        if centre is None:
            return None
        radius = math.dist(shown[0], centre)
        start = math.atan2(shown[0][1] - centre[1], shown[0][0] - centre[0])
        ring = [
            (
                centre[0] + radius * math.cos(start + step * math.pi / 4.0),
                centre[1] + radius * math.sin(start + step * math.pi / 4.0),
            )
            for step in range(9)
        ]
        return [np.asarray(ring, dtype=np.float32)]
    return None


def _line_pair_intersection(
    first: Tuple[float, float],
    second: Tuple[float, float],
    third: Tuple[float, float],
    fourth: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    x1, y1 = first
    x2, y2 = second
    x3, y3 = third
    x4, y4 = fourth
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) <= 1e-9:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return (
        (a * (x3 - x4) - (x1 - x2) * b) / denominator,
        (a * (y3 - y4) - (y1 - y2) * b) / denominator,
    )


def _circumcentre(
    first: Tuple[float, float],
    second: Tuple[float, float],
    third: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    ax, ay = first
    bx, by = second
    cx, cy = third
    denominator = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(denominator) <= 1e-9:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    return (
        (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / denominator,
        (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / denominator,
    )


def _declared_dynamic_corridors(
    task_name: str,
    record: Dict[str, Any],
    expected: str,
) -> Optional[List[np.ndarray]]:
    """Corridors this route can name for the task, or ``None`` if it cannot.

    Same declaration the added-ink gate runs on (:func:`_delivery_ink_arguments`):
    the ordered stroke grammar where one exists, otherwise the terminal
    construction geometry, plus whatever the task's own renderer restates while
    it draws.  ``None`` is the honest answer where the route can name nothing --
    holding such a task to a region it cannot describe would reject legal
    deliveries -- and an empty list is a task that declares its construction is
    empty, which is a real declaration.
    """

    line_width = renderer_line_width(record)
    variants = _coordinate_serial_path_variants(task_name, record, expected)
    variant_corridors = _coordinate_all_variant_paths(variants)
    corridors = (
        list(variant_corridors)
        if variant_corridors is not None
        else list(
            terminal_construction_paths(
                task_name,
                record,
                expected,
                line_width=line_width,
            )
            or ()
        )
    )
    corridors.extend(
        terminal_attribution_paths(task_name, record, expected, line_width=line_width) or ()
    )
    corridors.extend(_containment_corridors(task_name, record) or ())
    if not corridors and str(task_name) not in NO_CONSTRUCTION_TASKS:
        return None
    return corridors


def _source_dynamic_region(
    task_name: str,
    record: Dict[str, Any],
    expected: str,
    shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Region a legal delivery of this sample may change, from the source alone.

    Nothing here reads the candidate, which is the whole point.  The family
    continuity contract and the video's own static-region hold both take their
    reference from the candidate's own first and last frames, so a mutation
    that is present in the last frame is counted as part of the process and
    neither can see it: a check whose reference is the candidate cannot detect
    the candidate.

    ``None`` means the route cannot name where this task's delivery may draw,
    and the caller reaches no verdict rather than one it cannot support.
    """

    height, width = int(shape[0]), int(shape[1])
    if task_name == "angle_bisector":
        geometry = _angle_bisector_expected_geometry(record)
        if geometry is None:
            return None
        radius = float(record.get("point_radius") or PointTargetPuzzleGenerator.POINT_RADIUS)
        return _angle_bisector_allowed_dynamic_mask((height, width), geometry, radius) > 0
    if task_name == "arc_connect":
        _replay, replay_valid, _correct = replay_arc_connect_geometry(record)
        geometry = _arc_connect_expected_geometry(record) if replay_valid else None
        if geometry is None:
            return None
        radius = float(record.get("point_radius") or PointTargetPuzzleGenerator.POINT_RADIUS)
        allowed, _band, _paths, _hidden = _arc_connect_dynamic_masks(
            (height, width), geometry, radius
        )
        return allowed > 0
    if task_name == "circle_tangent_line":
        geometry = _circle_tangent_line_expected_geometry(record)
        if geometry is None:
            return None
        radius = float(record.get("point_radius") or PointTargetPuzzleGenerator.POINT_RADIUS)
        return _circle_tangent_line_allowed_dynamic_mask((height, width), geometry, radius) > 0

    corridors = _declared_dynamic_corridors(task_name, record, expected)
    if corridors is None:
        return None
    seed = np.zeros((height, width), dtype=np.uint8)
    for path in corridors:
        points = np.asarray(path, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        cv2.polylines(seed, [np.rint(points).astype(np.int32)], False, 255, thickness=1)
    for centre_x, centre_y, radius in _coordinate_overlay_discs(
        task_name,
        record,
        corridors,
    ):
        cv2.circle(
            seed,
            (int(round(centre_x)), int(round(centre_y))),
            max(1, int(np.ceil(radius))),
            255,
            thickness=-1,
        )
    if not bool(seed.any()):
        return None
    return _sweep(seed > 0, _SOURCE_DYNAMIC_SWEEP)


def _verify_source_static_preservation(
    frames: Sequence[np.ndarray],
    source: np.ndarray,
    dynamic: np.ndarray,
) -> Tuple[Dict[str, Any], bool]:
    """Hard-gate every frame's static region against the supplied input.

    Section 11.4 requires the whole-video verifier to reject a static mutation,
    and every reading this route had was blind to one that survives to the last
    frame: the family endpoint contract and the delivery's own static-region
    hold both derive their mask from the candidate's own endpoints, and the
    task-owned semantic verifiers read only the colours their own geometry is
    drawn in.  This gate takes its region from the source instead, so a patch
    that persists is a patch outside what the sample declares.

    The bound is the shared renderer floor, ``STATIC_REGION_FLOOR`` = 8/255 =
    0.0314, which the routes' own change detectors already treat as "not a
    change"; it is not a fitted margin.  In the original 59-task calibration,
    at sample indices 0-4, the worst static region of a legal
    reference video reads 0.0037 and a uniform +3/255 tone shift on every frame
    reads 0.0095 -- 3.3x under the bound -- while a 24x24 block pasted from the
    third frame onward into the region the source declares static reads 0.10 to
    0.28, eleven times the legal tone reading below it.
    """

    static = ~np.asarray(dynamic, dtype=bool)
    if not bool(static.any()):
        return {
            "video_source_static_evaluated": False,
            "video_source_static_reason": "the source declares the whole canvas dynamic",
        }, True
    worst = 0.0
    worst_frame = -1
    violating_frames = 0
    for index, frame in enumerate(frames):
        error, _regions = static_region_errors(np.asarray(frame), source, static)
        if error > STATIC_REGION_FLOOR:
            violating_frames += 1
        if error > worst:
            worst, worst_frame = error, index
    passed = bool(frames) and violating_frames == 0
    return {
        "video_source_static_evaluated": True,
        "video_source_static_declared_dynamic_fraction": round(float(1.0 - static.mean()), 6),
        "video_source_static_region_max_error": round(float(worst), 6),
        "video_source_static_region_worst_frame": worst_frame,
        "video_source_static_region_violation_frames": violating_frames,
        "video_source_static_region_size": REGION,
        "video_source_static_region_bound": round(float(STATIC_REGION_FLOOR), 6),
        "video_source_static_preservation_source": "supplied_input_and_source_declared_construction",
        "video_source_static_region_pass": passed,
    }, passed


# ------------------------------------------------ source process order ------

# How the renderer spends its animation on the declared construction.  Each
# entry is one of its drawing commands, in the order it animates them:

# * ``line``   -- ``animate_line``: the declared stroke revealed from its first
# point, full width, cut flat at the moving tip.
# * ``circle`` -- ``animate_circle``: the finished ellipse revealed through an
# arc mask sweeping from PIL angle zero.
# * ``arc``    -- a task-owned progressive arc, stroked from the declared
# path's own first point rather than from angle zero.
# * ``hold``   -- a command that draws no construction: an anchor disc, or a
# redraw of a stroke that is already complete.  It spends frames without
# advancing the picture, so its frames are counted as non-advancing and it
# contributes no bank state at all.  Giving it progress instead is what breaks
# the reading on ``nine_point_center``, whose six anchor commands would own
# 0.61 of the progress axis as a single plateau -- wider than the search's own
# budget and than the inherited jump ceiling, so the reading cannot cross it
# and stalls before the circle it is meant to check.

# ``width`` is the factor the generator hands ``draw_line``; ``joins`` its
# declared sealing discs.  Both are load-bearing rather than cosmetic: the bank
# is compared against the candidate through the same colour detector, which sees
# only the saturated core of a stroke, so a bank drawn at some other width
# predicts an amount of visible ink the frames never show.  Measured on this
# route, a bank stroked at the declared 5 px and read at full resolution
# overstates the detected ink by 8-19% on twelve tasks and by about 100% on
# ``napoleon_vertex``, whose commands are stroked at 0.55 of it.

# ``frames`` is present where a task renders its own schedule instead of the
# family's equal division.  ``moving_endpoint`` names the bespoke line primitive
# that redraws a line from its fixed start to a newly rounded moving endpoint on
# every frame; it is not the generic renderer's fixed finished stroke revealed
# through a mask.  ``smooth`` and ``partial_arc`` likewise name exact renderer
# primitives.  Approximating any of them changes the pixels the candidate-side
# observation reads.
_LINE = "line"
_SMOOTH = "smooth"
_CIRCLE = "circle"
_ARC = "arc"
_PARTIAL_ARC = "partial_arc"
_HOLD = "hold"

# The render layout's own non-drawing frames: the opening hold on the untouched
# input, the four-frame highlight commit that adds no construction, and the
# closing hold on the terminal state.  Read from the layout, not the candidate.
_COORDINATE_LAYOUT_HOLD_FRAMES = 6 + 4 + 8

# Frames the layout will spend on one command, and the frame budgets it chooses
# between; both are the generator's own, and the whole schedule follows from the
# number of commands the programme below declares.
_COORDINATE_MAX_COMMAND_FRAMES = 10
_COORDINATE_FRAME_BUDGETS = ((3, 49), (7, 65), (None, 97))


def _coordinate_lines(
    count: int,
    *,
    width: float = 1.0,
    joins: Optional[Sequence[Tuple[bool, bool]]] = None,
    moving_endpoint: bool = False,
) -> List[Dict[str, Any]]:
    seals = list(joins or [(False, False)] * count)
    return [
        {
            "kind": _LINE,
            "path": index,
            "width": width,
            "joins": seals[index],
            "moving_endpoint": moving_endpoint,
        }
        for index in range(count)
    ]


_COORDINATE_PROGRAMME: Dict[str, Tuple[Dict[str, Any], ...]] = {
    # Generic family renderer, one physical command unless noted otherwise.
    "angle_trisection": tuple(_coordinate_lines(1, joins=[(True, False)])),
    "arc_center": ({"kind": _PARTIAL_ARC, "path": 0, "width": 1.0},),
    "circle_inversion": tuple(_coordinate_lines(1)),
    "circle_tangent_intersection": tuple(_coordinate_lines(2)),
    "closest_pair": tuple(_coordinate_lines(1)),
    "diagonal_intersection": tuple(_coordinate_lines(2)),
    "diameter_opposite": tuple(_coordinate_lines(1)),
    "dilation_to_circle": tuple(_coordinate_lines(1)),
    "double_reflection": tuple(
        _coordinate_lines(2, joins=[(False, True), (True, False)])
    ),
    "double_rotation": (
        {
            "kind": _SMOOTH,
            "path": 0,
            "width": 1.0,
            "colour": (46, 110, 220),
        },
        {
            "kind": _SMOOTH,
            "path": 1,
            "width": 1.0,
            "colour": (34, 145, 88),
        },
    ),
    "equilateral_triangle_vertex": tuple(
        _coordinate_lines(1, joins=[(False, False)])
    ),
    "excircle_center": tuple(
        _coordinate_lines(2) + [{"kind": _CIRCLE, "path": 2, "width": 1.0}]
    ),
    "excircle_tangent_point": (
        {"kind": _PARTIAL_ARC, "path": 0, "width": 1.0},
    ),
    "external_angle_bisector": tuple(
        _coordinate_lines(2, joins=[(True, False), (True, False)])
    ),
    "external_ratio_point": tuple(_coordinate_lines(1)),
    "external_tangent_point": (
        {"kind": _LINE, "path": 0, "width": 1.0, "joins": (False, False)},
        {"kind": _HOLD},
    ),
    "fermat_point": tuple(_coordinate_lines(6)),
    "glide_reflection": tuple(
        _coordinate_lines(2, joins=[(False, True), (True, False)])
    ),
    "homothety_center": tuple(_coordinate_lines(3)),
    # This task renders its own video rather than the family's: four frames per
    # bisector, one for the incentre marker, seven for an incircle stroked as a
    # progressive arc starting at the visible tangent point.
    "incircle_tangent_point": tuple(
        [
            dict(command, frames=4)
            for command in _coordinate_lines(
                3,
                width=0.72,
                joins=[(True, False)] * 3,
                moving_endpoint=True,
            )
        ]
        + [
            {"kind": _HOLD, "frames": 1},
            {"kind": _ARC, "path": 3, "width": 1.0, "frames": 7},
        ]
    ),
    "inscribed_angle_arc": tuple([{"kind": _HOLD}] + _coordinate_lines(2)),
    "inscribed_square_vertex": tuple(_coordinate_lines(3)),
    "internal_tangent_point": (
        {"kind": _LINE, "path": 0, "width": 1.0, "joins": (False, False)},
        {"kind": _HOLD},
    ),
    "inverse_rotation": (
        {"kind": _SMOOTH, "path": 0, "width": 1.0},
    ),
    # The second command redraws the finished stroke above the candidate discs,
    # so every frame it spends shows the same picture as the first command's last.
    "kite_vertex": (
        {"kind": _LINE, "path": 0, "width": 1.0, "joins": (True, True)},
        {"kind": _HOLD},
    ),
    "midline_of_triangle": tuple(_coordinate_lines(1)),
    "napoleon_vertex": tuple(_coordinate_lines(3, width=0.55)),
    "nine_point_center": tuple(
        _coordinate_lines(3, joins=[(True, False)] * 3)
        + [{"kind": _HOLD}] * 6
        + [{"kind": _CIRCLE, "path": 3, "width": 1.0}]
    ),
    "orthogonal_projection": tuple(_coordinate_lines(1)),
    "parabola_vertex": tuple(_coordinate_lines(1)),
    "parallelogram": tuple(
        _coordinate_lines(1, joins=[(True, True)])
    ),
    "perpendicular": tuple(_coordinate_lines(1)),
    "perpendicular_bisector": tuple(_coordinate_lines(1)),
    "point_rotation": (
        {"kind": _SMOOTH, "path": 0, "width": 1.0},
    ),
    "power_of_point": tuple(_coordinate_lines(1)),
    "ray_intersection": tuple(
        dict(command, width=1.6, frames=4)
        for command in _coordinate_lines(3, moving_endpoint=True)
    ),
    "ray_reflect": tuple(_coordinate_lines(1)),
    "reflection": tuple(_coordinate_lines(1)),
    "regular_octagon_vertex": tuple(
        _coordinate_lines(1, joins=[(True, True)])
    ),
    "rhombus_vertex": tuple(
        _coordinate_lines(1, joins=[(True, True)])
    ),
    "right_triangle": (
        {"kind": _LINE, "path": 0, "width": 1.0, "joins": (False, False)},
        {"kind": _LINE, "path": 1, "width": 0.7, "joins": (False, False)},
    ),
    "secant_intersection": tuple(_coordinate_lines(2)),
    "spiral_continuation": tuple(_coordinate_lines(1, joins=[(True, False)])),
    "spiral_similarity_center": tuple(_coordinate_lines(3)),
    "square_vertex_from_diagonal": tuple(_coordinate_lines(2, joins=[(True, True)] * 2)),
    "symmetric_completion": tuple(
        _coordinate_lines(1, joins=[(True, True)])
    ),
    "tangent_line_pair_angle": (
        {"kind": _LINE, "path": 0, "width": 1.0, "joins": (False, False)},
        {"kind": _LINE, "path": 1, "width": 1.0, "joins": (False, False)},
        {"kind": _PARTIAL_ARC, "path": 2, "width": 0.7},
    ),
    "triangle_center": tuple(_coordinate_lines(3, joins=[(True, True)] * 3)),
    "triangle_from_midpoints": tuple(
        _coordinate_lines(3, joins=[(False, True), (True, True), (True, False)])
    ),
    "translation_composition": tuple(
        _coordinate_lines(2, joins=[(True, True), (True, False)])
    ),
    "trapezoid_median": tuple(_coordinate_lines(1)),
    "vector_translation": tuple(_coordinate_lines(1, joins=[(True, False)])),
    "vertical_angle_pair": (
        {"kind": _PARTIAL_ARC, "path": 0, "width": 0.6},
    ),
    # Bespoke renderers.  Their schedules are task contracts, not inferred from
    # the candidate's frame transitions.
    "angle_bisector": (
        {
            "kind": _LINE,
            "path": 0,
            "width": 1.0,
            "joins": (True, False),
            "frames": 8,
            "moving_endpoint": True,
        },
    ),
    "circle_tangent_line": (
        {
            "kind": _LINE,
            "path": 0,
            "width": 1.0,
            "joins": (True, False),
            "frames": 6,
            "moving_endpoint": True,
        },
        {
            "kind": _LINE,
            "path": 1,
            "width": 1.0,
            "joins": (True, False),
            "frames": 8,
            "moving_endpoint": True,
        },
        {
            "kind": _LINE,
            "path": 2,
            "width": 0.72,
            "joins": (False, False),
            "frames": 4,
            "moving_endpoint": True,
        },
    ),
    "circle_tangent_point": (
        {
            "kind": _LINE,
            "path": 0,
            "width": 1.0,
            "joins": (True, False),
            "frames": 8,
            "moving_endpoint": True,
        },
        {
            "kind": _LINE,
            "path": 1,
            "width": 1.0,
            "joins": (True, False),
            "frames": 6,
            "moving_endpoint": True,
        },
        {
            "kind": _LINE,
            "path": 2,
            "width": 0.72,
            "joins": (False, False),
            "frames": 4,
            "moving_endpoint": True,
        },
    ),
    "circumcenter": tuple(
        [
            dict(command, frames=4)
            for command in _coordinate_lines(
                3,
                width=0.65,
                moving_endpoint=True,
            )
        ]
        + [{"kind": _ARC, "path": 3, "width": 1.0, "frames": 8}]
    ),
    "incenter": tuple(
        [
            dict(command, frames=4)
            for command in _coordinate_lines(
                3,
                width=0.72,
                joins=[(True, False)] * 3,
                moving_endpoint=True,
            )
        ]
        + [
            {"kind": _HOLD, "frames": 1},
            {"kind": _ARC, "path": 3, "width": 1.0, "frames": 7},
        ]
    ),
    "isosceles_trapezoid": (
        {
            "kind": _LINE,
            "path": 0,
            "width": 1.0,
            "joins": (True, False),
            "frames": 6,
            "moving_endpoint": True,
        },
        {
            "kind": _LINE,
            "path": 1,
            "width": 1.0,
            "joins": (False, True),
            "frames": 6,
            "moving_endpoint": True,
        },
    ),
    # Orthocenter is assembled dynamically below because an acute sample has
    # no extension command while an obtuse sample has one or two.
    "parallel": (
        {
            "kind": _LINE,
            "path": 0,
            "width": 0.72,
            "joins": (False, False),
            "frames": 8,
            "moving_endpoint": True,
        },
    ),
}

# Exact fail-closed coverage of the pipeline-rendered progress bank.
# Orthocenter has a source-dependent command count and is therefore resolved
# by :func:`_coordinate_programme` instead of appearing in the static table.
# Discs a ``hold`` command paints over the construction layer that
# :func:`overlay_occluders` does not already name -- it names candidate centres
# and declared stroke endpoints, and these sit at neither.  ``nine_point_center``
# is handled separately below because its six are derived rather than stored.
# ``inscribed_angle_arc``'s single anchor sits on the circle centre, which is
# already the shared start of both its declared strokes, so it needs no entry.
_COORDINATE_MARKER_DISCS: Dict[str, Tuple[str, float]] = {
    "incircle_tangent_point": ("incenter", 7.0),
    "incenter": ("incenter", 7.0),
}

# Radius of the six anchor discs ``nine_point_center`` composites *above* its
# nine-point circle, at the triangle's own midpoints and altitude feet.
_NINE_POINT_MARKER_RADIUS = 7.0


def _coordinate_programme(
    task_name: str,
    paths: Sequence[np.ndarray] = (),
) -> Optional[Tuple[Dict[str, Any], ...]]:
    """Return the source renderer's physical command programme.

    Orthocenter alone has a sample-dependent topology: every required finite
    side extension is drawn first, followed by the three altitudes.  The number
    of extensions is independently recoverable from the raw triangle, and is
    represented by the leading paths supplied by its grammar.
    """

    if task_name == "orthocenter":
        extension_count = len(paths) - 3
        if extension_count not in (0, 1, 2):
            return None
        return tuple(
            [
                {
                    "kind": _LINE,
                    "path": index,
                    "width": 0.5,
                    "joins": (False, False),
                    "frames": 4,
                    "moving_endpoint": True,
                }
                for index in range(extension_count)
            ]
            + [
                {
                    "kind": _LINE,
                    "path": extension_count + index,
                    "width": 0.72,
                    "joins": (False, False),
                    "frames": 4,
                    "moving_endpoint": True,
                }
                for index in range(3)
            ]
        )
    return _COORDINATE_PROGRAMME.get(task_name)


def _coordinate_command_schedule(
    task_name: str,
    programme: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[int]:
    """Frames the renderer spends on each declared command, in order.

    Recomputed from the render layout rather than read off the candidate: a
    schedule taken from where *this* video happens to start and stop drawing
    would move its own budget whenever the video was corrupted.
    """

    resolved = tuple(programme or _COORDINATE_PROGRAMME[task_name])
    declared = [command.get("frames") for command in resolved]
    if any(declared):
        return [int(value or 1) for value in declared]
    budget = _COORDINATE_FRAME_BUDGETS[-1][1]
    for limit, frames in _COORDINATE_FRAME_BUDGETS:
        if limit is not None and len(resolved) <= limit:
            budget = frames
            break
    available = max(1, budget - _COORDINATE_LAYOUT_HOLD_FRAMES)
    per_command = max(
        1, min(_COORDINATE_MAX_COMMAND_FRAMES, available // max(1, len(resolved)))
    )
    return [per_command] * len(resolved)


def _coordinate_drawing_frames(
    task_name: str,
    programme: Optional[Sequence[Dict[str, Any]]] = None,
) -> int:
    """Frames the layout spends advancing the construction.

    Only the drawing commands count: a hold spends its frames on the picture the
    command before it left, so a reference video advances the process once per
    *drawing* frame and the bank's own grid must match that.
    """

    resolved = tuple(programme or _COORDINATE_PROGRAMME[task_name])
    schedule = _coordinate_command_schedule(task_name, resolved)
    return sum(
        frames
        for command, frames in zip(resolved, schedule)
        if str(command["kind"]) != _HOLD
    )


def _coordinate_overlay_discs(
    task_name: str,
    record: Dict[str, Any],
    paths: Sequence[np.ndarray],
) -> List[Tuple[float, float, float]]:
    """Every disc the renderer paints over this task's construction layer."""

    discs = list(overlay_occluders(record, paths))
    declared = _COORDINATE_MARKER_DISCS.get(task_name)
    if declared is not None:
        centre = _point_pair(record.get(declared[0]))
        if centre is not None:
            discs.append((centre[0], centre[1], float(declared[1])))
    if task_name != "nine_point_center":
        return discs
    raw = record.get("triangle_vertices")
    if not isinstance(raw, list) or len(raw) != 3:
        return discs
    vertices = [_point_pair(value) for value in raw]
    if any(vertex is None for vertex in vertices):
        return discs
    for index in range(3):
        first = vertices[(index + 1) % 3]
        second = vertices[(index + 2) % 3]
        opposite = vertices[index]
        if first is None or second is None or opposite is None:
            continue
        discs.append(
            (
                (first[0] + second[0]) / 2.0,
                (first[1] + second[1]) / 2.0,
                _NINE_POINT_MARKER_RADIUS,
            )
        )
        foot = _foot_of_perpendicular(opposite, first, second)
        discs.append((foot[0], foot[1], _NINE_POINT_MARKER_RADIUS))
    return discs


def _coordinate_observable_region(
    task_name: str,
    record: Dict[str, Any],
    paths: Sequence[np.ndarray],
    source: np.ndarray,
) -> np.ndarray:
    """Where construction ink can register at all, from the input alone.

    Two exclusions, the same ones :func:`_declared_stroke_evidence` applies to
    the still-image reading: the discs the renderer repaints over the
    construction layer on every frame, and the pixels the input has already
    inked, where the construction runs underneath and changes nothing.  The bank
    and the frames are read in the same region, so the comparison never asks the
    candidate for evidence the renderer cannot show.

    The input's ink is taken one pixel wider than the still-image gate takes it.
    That gate scores sampled positions and a rim costs it nothing; this one has
    to predict the *rate* at which ink appears, and the scene is repainted over
    the construction with its antialiasing, so the pixels just outside a source
    line -- which fall under the 24/255 ink threshold but are still overwritten
    -- hide construction the bank would otherwise expect to see.
    """

    height, width = source.shape[:2]
    hidden = np.zeros((height, width), dtype=np.uint8)
    for centre_x, centre_y, radius in _coordinate_overlay_discs(task_name, record, paths):
        cv2.circle(
            hidden,
            (int(round(centre_x)), int(round(centre_y))),
            max(1, int(np.ceil(radius))),
            255,
            thickness=-1,
        )
    return ~(hidden > 0) & ~_sweep(source_ink_mask(source), 1)


def _coordinate_stroke_prefix(
    points: np.ndarray,
    fraction: float,
) -> Optional[List[Tuple[int, int]]]:
    """The renderer's own revealed prefix of one stroke, in supersampled pixels.

    ``animate_line`` reveals the finished stroke through a mask two supersampled
    pixels wider than itself, built from the vertices reached so far plus the
    moving tip.  The visible result is therefore the full-width stroke cut flat
    at the tip -- which is what stroking this prefix produces -- and the mask's
    own oversize only guarantees that the miter at a passed joint is uncovered
    whole.  The tip is rounded here exactly as the renderer rounds it.
    """

    lengths = [0.0]
    for index in range(1, len(points)):
        previous, current = points[index - 1], points[index]
        lengths.append(
            lengths[-1]
            + math.hypot(float(current[0] - previous[0]), float(current[1] - previous[1]))
        )
    if lengths[-1] <= 1e-6:
        return None
    target = lengths[-1] * float(np.clip(fraction, 0.0, 1.0))
    reached = 0
    while reached < len(lengths) - 1 and lengths[reached + 1] <= target + 1e-6:
        reached += 1
    if reached >= len(points) - 1:
        tip = (int(points[-1][0]), int(points[-1][1]))
    else:
        span = lengths[reached + 1] - lengths[reached]
        remaining = max(0.0, target - lengths[reached])
        if span > 0.0 and remaining > 0.0:
            ratio = min(1.0, remaining / span)
            first, second = points[reached], points[reached + 1]
            tip = (
                int(round(float(first[0]) + float(second[0] - first[0]) * ratio)),
                int(round(float(first[1]) + float(second[1] - first[1]) * ratio)),
            )
        else:
            tip = (int(points[reached][0]), int(points[reached][1]))
    prefix = [(int(point[0]), int(point[1])) for point in points[: reached + 1]]
    prefix.append(tip)
    return prefix if len(prefix) >= 2 else None


def _coordinate_progress_bank(
    task_name: str,
    paths: Sequence[np.ndarray],
    record: Dict[str, Any],
    observable: np.ndarray,
    *,
    programme: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[np.ndarray], List[float], np.ndarray]:
    """Render the declared construction at ascending progress, through the
    family's own pipeline.

    Every state is drawn on the renderer's own supersampled canvas, at the
    renderer's own per-command stroke width, and brought back to delivery size
    by the renderer's own LANCZOS finish; it is then read with the *same*
    construction-ink detector that reads the candidate's frames.  That is what
    makes a bank state and a reference frame the same measurement of the same
    picture: an approximated bank cannot be, because the detector sees only a
    stroke's saturated core and the width it reaches is a property of the
    supersample and the downsample together, not of the declared line width.

    Progress is *time*, counted over the drawing frames alone: each drawing
    command owns its declared share of them and advances by arc length (or sweep
    angle) inside its own share, because that is how the renderer divides its
    animation.  A bank on that grid *is* the sequence of pictures the renderer
    draws, so two of its states are equal exactly when two reference frames
    rasterise the same, and the reading's own quantum equals the reference's own
    per-frame advance.
    """

    resolved = tuple(programme or _COORDINATE_PROGRAMME[task_name])
    schedule = _coordinate_command_schedule(task_name, resolved)
    total = float(_coordinate_drawing_frames(task_name, resolved))
    if total <= 0.0:
        return [], [], np.zeros(observable.shape[:2], dtype=bool)
    # ``point_rotation`` is the sole 8x task on this route; every other source
    # renderer uses the doctrine's 4x factor.  This is renderer topology, not a
    # candidate-controlled value.
    factor = 8 if task_name == "point_rotation" else int(PointTargetPuzzleGenerator.SUPERSAMPLE)
    line_width = renderer_line_width(record)
    height, width = observable.shape[:2]
    default_colour = tuple(
        int(channel) for channel in PointTargetPuzzleGenerator.CONSTRUCTION_COLOR
    )
    raw_paths = [np.asarray(path, dtype=np.float64) for path in paths]
    scaled = [
        np.rint(path * factor).astype(np.int64)
        for path in raw_paths
    ]
    if any(points.ndim != 2 or points.shape[0] < 2 for points in scaled):
        return [], [], np.zeros((height, width), dtype=bool)

    def rendered(canvas: Image.Image) -> np.ndarray:
        # The bank draws only the construction, so its "departure from the input"
        # is its departure from the background the renderer starts from.  Reading
        # it against that rather than against the supplied scene is what keeps
        # both sides of the comparison the same measurement: the input's own ink
        # is already outside ``observable``, so the only pixels either side can
        # disagree about are ones the construction alone lit.
        finished = np.asarray(finish_scaled_canvas(canvas, (width, height)))
        blank = np.full_like(finished, 255)
        mask = (
            _construction_ink_mask(finished, blank, task_name=task_name)
            & observable
        )
        return np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)

    accumulated = Image.new("RGB", (width * factor, height * factor), (255, 255, 255))
    bank: List[np.ndarray] = [rendered(accumulated.copy())]
    progress: List[float] = [0.0]
    elapsed = 0
    for command, frames in zip(resolved, schedule):
        kind = str(command["kind"])
        if kind == _HOLD:
            continue
        path_index = int(command["path"])
        raw_points = raw_paths[path_index]
        moving_endpoint = bool(command.get("moving_endpoint"))
        # Bespoke task renderers interpolate in delivery coordinates and then
        # ``draw_line`` rounds both endpoints before ScaledDraw multiplies them
        # onto the supersampled canvas.  The generic video renderer instead
        # receives one already-rounded finished path and reveals that fixed
        # raster through a mask.  Preserve the distinction here: multiplying a
        # floating endpoint before rounding creates a third raster that neither
        # source renderer ever draws.
        points = (
            np.rint(raw_points).astype(np.int64) * factor
            if kind == _LINE
            else scaled[path_index]
        )
        if moving_endpoint and len(points) != 2:
            return [], [], np.zeros((height, width), dtype=bool)
        colour = tuple(int(channel) for channel in command.get("colour", default_colour))
        stroke_width = scale_stroke_width(
            max(1, round(line_width * float(command["width"]))), factor
        )
        joins = tuple(command.get("joins") or (False, False))
        complete = accumulated.copy()
        box: Optional[Tuple[float, float, float, float]] = None
        sweep_start = 0.0
        sweep = 0.0
        if kind in (_CIRCLE, _ARC):
            box = (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
            ImageDraw.Draw(complete).ellipse(box, outline=colour, width=stroke_width)
            if kind == _ARC:
                sweep_start = math.degrees(
                    math.atan2(
                        float(points[0][1]) - (box[1] + box[3]) / 2.0,
                        float(points[0][0]) - (box[0] + box[2]) / 2.0,
                    )
                )
                sweep = 360.0
        elif kind == _PARTIAL_ARC:
            # Recover the native Pillow arc parameters from the independently
            # replayed centreline.  Three non-collinear samples determine the
            # circle; the path's unwrapped screen bearings determine both the
            # sweep and which endpoint the moving tip leaves first.
            first = raw_points[0]
            middle = raw_points[len(raw_points) // 2]
            last = raw_points[-1]
            determinant = 2.0 * (
                first[0] * (middle[1] - last[1])
                + middle[0] * (last[1] - first[1])
                + last[0] * (first[1] - middle[1])
            )
            if abs(determinant) <= 1e-6:
                return [], [], np.zeros((height, width), dtype=bool)
            first_norm = float(first[0] ** 2 + first[1] ** 2)
            middle_norm = float(middle[0] ** 2 + middle[1] ** 2)
            last_norm = float(last[0] ** 2 + last[1] ** 2)
            centre_x = (
                first_norm * (middle[1] - last[1])
                + middle_norm * (last[1] - first[1])
                + last_norm * (first[1] - middle[1])
            ) / determinant
            centre_y = (
                first_norm * (last[0] - middle[0])
                + middle_norm * (first[0] - last[0])
                + last_norm * (middle[0] - first[0])
            ) / determinant
            radius = float(
                np.mean(
                    np.hypot(
                        raw_points[:, 0] - centre_x,
                        raw_points[:, 1] - centre_y,
                    )
                )
            )
            unwrapped = np.unwrap(
                np.arctan2(
                    raw_points[:, 1] - centre_y,
                    raw_points[:, 0] - centre_x,
                )
            )
            sweep_start = math.degrees(float(unwrapped[0]))
            sweep = math.degrees(float(unwrapped[-1] - unwrapped[0]))
            if abs(sweep) <= 1e-6:
                return [], [], np.zeros((height, width), dtype=bool)
            box = tuple(
                float(round(value) * factor)
                for value in (
                    centre_x - radius,
                    centre_y - radius,
                    centre_x + radius,
                    centre_y + radius,
                )
            )
            final_start = sweep_start if sweep > 0.0 else sweep_start + sweep
            final_end = sweep_start + sweep if sweep > 0.0 else sweep_start
            ImageDraw.Draw(complete).arc(
                box,
                start=final_start,
                end=final_end,
                fill=colour,
                width=stroke_width,
            )
        elif kind == _SMOOTH:
            draw_smooth_polyline(
                complete,
                raw_points.tolist(),
                color=colour,
                width=max(1, round(line_width * float(command["width"]))),
                factor=factor,
            )
        else:
            stroke_polyline(
                ImageDraw.Draw(complete),
                [(int(point[0]), int(point[1])) for point in points],
                fill=colour,
                width=stroke_width,
                join_start=bool(joins[0]),
                join_end=bool(joins[1]),
            )
        for step in range(frames):
            fraction = (step + 1) / float(frames)
            if step == frames - 1:
                frame = complete
            elif kind == _CIRCLE:
                # ``animate_circle`` reveals the finished ellipse through an arc
                # mask four supersampled pixels wider than the stroke, so the
                # revealed part is the ellipse's own rasterisation rather than a
                # separately stroked arc.
                reveal = Image.new("L", complete.size, 0)
                ImageDraw.Draw(reveal).arc(
                    box, start=0.0, end=360.0 * fraction, fill=255,
                    width=stroke_width + 4 * factor,
                )
                frame = Image.composite(complete, accumulated, reveal)
            elif kind == _ARC:
                frame = accumulated.copy()
                ImageDraw.Draw(frame).arc(
                    box,
                    start=sweep_start,
                    end=sweep_start + 360.0 * fraction,
                    fill=colour,
                    width=stroke_width,
                )
            elif kind == _PARTIAL_ARC:
                frame = accumulated.copy()
                current = sweep_start + sweep * fraction
                start = sweep_start if sweep > 0.0 else current
                end = current if sweep > 0.0 else sweep_start
                ImageDraw.Draw(frame).arc(
                    box,
                    start=start,
                    end=end,
                    fill=colour,
                    width=stroke_width,
                )
            elif kind == _SMOOTH:
                frame = accumulated.copy()
                prefix = _coordinate_stroke_prefix(points, fraction)
                if prefix is not None:
                    draw_smooth_polyline(
                        frame,
                        [
                            [float(point[0]) / factor, float(point[1]) / factor]
                            for point in prefix
                        ],
                        color=colour,
                        width=max(
                            1,
                            round(line_width * float(command["width"])),
                        ),
                        factor=factor,
                    )
            elif moving_endpoint:
                frame = accumulated.copy()
                raw_tip = raw_points[0] + (
                    raw_points[-1] - raw_points[0]
                ) * fraction
                moving_points = np.rint(
                    np.stack([raw_points[0], raw_tip], axis=0)
                ).astype(np.int64) * factor
                if not np.array_equal(moving_points[0], moving_points[1]):
                    stroke_polyline(
                        ImageDraw.Draw(frame),
                        [
                            (int(point[0]), int(point[1]))
                            for point in moving_points
                        ],
                        fill=colour,
                        width=stroke_width,
                        join_start=bool(joins[0]),
                        join_end=False,
                    )
            else:
                prefix = _coordinate_stroke_prefix(points, fraction)
                reveal = Image.new("L", complete.size, 0)
                if prefix is not None:
                    mask_draw = ImageDraw.Draw(reveal)
                    mask_width = stroke_width + 2 * factor
                    for mask_start, mask_end in zip(prefix, prefix[1:]):
                        mask_draw.line(
                            [mask_start, mask_end],
                            fill=255,
                            width=mask_width,
                        )
                    stroke_polyline(
                        mask_draw,
                        prefix,
                        fill=255,
                        width=mask_width,
                    )
                start_x, start_y = points[0]
                mask_radius = stroke_width // 2 + factor
                ImageDraw.Draw(reveal).ellipse(
                    (
                        start_x - mask_radius,
                        start_y - mask_radius,
                        start_x + mask_radius,
                        start_y + mask_radius,
                    ),
                    fill=255,
                )
                frame = Image.composite(complete, accumulated, reveal)
            bank.append(rendered(frame))
            progress.append((elapsed + step + 1) / total)
        elapsed += frames
        accumulated = complete
    terminal = np.asarray(bank[-1])[:, :, 0] > 0
    return bank, progress, _sweep(terminal, _SOURCE_DYNAMIC_SWEEP) & observable


def _coordinate_widest_still_stretch(
    bank: Sequence[np.ndarray],
    bank_progress: Sequence[float],
    dynamic: np.ndarray,
) -> float:
    """Widest span of progress over which the bank shows one single picture.

    The shared reading collapses runs of identical bank states, so this is the
    largest distance between two consecutive pictures the source actually draws.
    Equality is exact, not thresholded, and it is measured inside the same
    dynamic region the reading compares in.
    """

    region = np.asarray(dynamic, dtype=bool)
    widest = 0.0
    previous = 0
    for index in range(1, len(bank)):
        if np.array_equal(
            np.asarray(bank[index])[region], np.asarray(bank[previous])[region]
        ):
            continue
        widest = max(widest, float(bank_progress[index]) - float(bank_progress[previous]))
        previous = index
    return widest


def _read_coordinate_process_order_variant(
    frames: Sequence[np.ndarray],
    source: np.ndarray,
    record: Dict[str, Any],
    *,
    task_name: str,
    expected: str,
    paths: Sequence[np.ndarray],
    variant_index: int,
) -> Tuple[Dict[str, Any], Optional[bool]]:
    """Read the construction as a bounded-step walk over the source's own order.

    Everything else this route reads of a video is one scalar per frame plus a
    media-parity check, and a deletion removes process without ever taking ink
    back, so the cumulative-ink gate cannot see one by construction: measured
    before this gate, a four-frame deletion that keeps the 4N+1 parity was
    accepted on 46 of the 66 videos it could be built for, and a hard cut on 29
    of 60.  The shared bounded-step reading judges every frame against the
    source's own advance and the source's own local rate of visible change at
    the state the frame is read as, so the same corruption is judged the same
    way wherever in the process it sits.

    ``None`` is the reading declining: it needs both an ordered stroke grammar
    and the render programme that spends frames on it.  The route owns both for
    61 of its 67 tasks; the other six have no serial pen programme.
    """

    paths = list(paths)
    programme = _coordinate_programme(task_name, paths)
    if programme is None:
        return {
            "video_process_order_available": False,
            "video_process_order_reason": (
                "the route owns no render programme for this task's construction"
            ),
        }, None
    if not paths:
        return {
            "video_process_order_available": False,
            "video_process_order_reason": "the declared stroke grammar could not be instantiated",
        }, False
    declared_strokes = max(
        (int(command.get("path", -1)) for command in programme), default=-1
    ) + 1
    if len(paths) != declared_strokes:
        # The programme and the grammar disagree about how many strokes this
        # task draws, so the bank cannot be placed on the renderer's own grid.
        # That is a gap in what the route declares, not evidence about the
        # candidate, so the dimension reaches no verdict.
        return {
            "video_process_order_available": False,
            "video_process_order_reason": (
                "the render programme and the stroke grammar disagree on the stroke count"
            ),
            "video_process_order_declared_strokes": declared_strokes,
            "video_process_order_grammar_strokes": len(paths),
        }, None
    if len(frames) < 9:
        return {
            "video_process_order_available": False,
            "video_process_order_reason": "too few decoded frames to read as a sequence",
        }, False
    observable = _coordinate_observable_region(task_name, record, paths, source)
    bank, bank_progress, dynamic = _coordinate_progress_bank(
        task_name,
        paths,
        record,
        observable,
        programme=programme,
    )
    if len(bank) < 3 or not bool(np.any(dynamic)):
        return {
            "video_process_order_available": False,
            "video_process_order_reason": "the source construction could not be rendered as a bank",
        }, False
    # A bounded-step walk needs a bank it can actually walk.  The step gate's
    # allowance is capped at the shared jump ceiling, so a stretch of progress the
    # source renders as one single picture, wider than that ceiling, can never be
    # crossed however legal the delivery is -- and refusing a correct video for a
    # blind spot in the reading's own observation domain is exactly what section
    # 11.6 forbids.  Such a stretch means one of the declared commands draws
    # where the input's own ink already covers it: measured at seed 203791, two
    # ``inscribed_angle_arc`` samples run a radius almost entirely under the
    # scene's existing lines, leaving 0.45 and 0.40 of the progress axis as a
    # single picture, and the reading refuses both legal references at step
    # ratios of 1.88 and 1.67.  Declining is the honest verdict there, and it is
    # the same answer the still-image branch gives a stroke the input leaves
    # without attributable samples.

    # The gaps are measured on the bank's own pictures at full resolution, which
    # is what the shared reading compares at on every task of this route (its
    # coarsening probe selects kernel 0 for all 480 readings measured); a coarser
    # kernel could only merge more states and widen a gap further.
    unwalkable = _coordinate_widest_still_stretch(bank, bank_progress, dynamic)
    if unwalkable > _INHERITED_JUMP_CEILING:
        return {
            "video_process_order_available": False,
            "video_process_order_reason": (
                "the source renders one unbroken stretch of its own construction that no "
                "bounded step may cross, so the reading can reach no verdict"
            ),
            "video_process_order_widest_still_stretch": round(float(unwalkable), 6),
            "video_process_order_step_ceiling": round(float(_INHERITED_JUMP_CEILING), 6),
        }, None
    schedule = _coordinate_command_schedule(task_name, programme)
    advancing = _coordinate_drawing_frames(task_name, programme)
    observed = [
        np.repeat(
            ((
                _construction_ink_mask(
                    frame,
                    source,
                    task_name=task_name,
                    record=record,
                )
                & observable
            ).astype(np.uint8) * 255)[
                :, :, None
            ],
            3,
            axis=2,
        )
        for frame in frames
    ]
    reading = read_process_order(
        observed,
        bank,
        bank_progress,
        dynamic,
        # Every frame the render layout does not spend on a drawing command is a
        # frame that does not advance the construction, and the layout spends
        # exactly ``advancing`` of them on one.  Taking the complement this way
        # makes the reference advance ``1 / advancing`` whatever the candidate's
        # own length is, so a candidate cannot buy a looser budget by being
        # shorter -- which is precisely the corruption being measured.
        non_advancing_frames=max(0, len(frames) - advancing),
        bank_quantum=1.0 / float(advancing),
        # This bank renders exactly one state per source drawing frame.  When
        # source/overlay occlusion makes several consecutive states
        # pixel-identical, retain the first visible transition after that run as
        # the real one-frame source change it is.
        preserve_adjacent_transitions=True,
    )
    # A legal ordering is useful evidence only if the delivery reaches that
    # ordering's final *observable* construction state.  Without this check a
    # candidate following variant A can be compared with a geometrically
    # different variant B whose dynamic corridor it never enters: every frame
    # then aligns to B's blank state, producing zero steps and zero pace
    # violations, so B would incorrectly rescue a corrupted A.  Use the last
    # distinct bank picture rather than raw progress 1.0 because a renderer may
    # finish with source-hidden or overlay-hidden work that cannot change the
    # observable raster.
    observable_terminal_index = 0
    previous_observable_index = 0
    for index in range(1, len(bank)):
        if not np.array_equal(
            np.asarray(bank[index])[dynamic],
            np.asarray(bank[observable_terminal_index])[dynamic],
        ):
            previous_observable_index = observable_terminal_index
            observable_terminal_index = index
    observable_terminal_progress = float(
        bank_progress[observable_terminal_index]
    )
    terminal_progress = float(reading.progress[-1])
    terminal_quantum = 1.0 / float(advancing)
    # The alignment reads a lossy decoded video against a lossless source bank.
    # Its last frame can therefore tie with the immediately preceding bank cell.
    # That does *not* make a whole cell free: a candidate that really omits the
    # last cell of a long tangent is a distinguishable incomplete construction.
    # Read the cell's own newly visible pixels in the candidate terminal frame.
    # The existing source-derived sample/support floors decide whether that
    # increment is judgeable; below eight pixels, the legal codec can erase the
    # increment completely and section 11.6 requires abstention.  At or above
    # that floor, at least half of the increment must remain.  Thus the fallback
    # is a measured codec tie, not a one-quantum omission budget.
    minimum_terminal_progress = max(
        0.0,
        observable_terminal_progress - terminal_quantum,
    )
    terminal_bank_mask = (
        np.asarray(bank[observable_terminal_index])[:, :, 0] > 0
    )
    previous_bank_mask = (
        np.asarray(bank[previous_observable_index])[:, :, 0] > 0
    )
    terminal_increment = (
        terminal_bank_mask & ~previous_bank_mask & np.asarray(dynamic, dtype=bool)
    )
    terminal_increment_pixels = int(terminal_increment.sum())
    terminal_candidate_mask = np.asarray(observed[-1])[:, :, 0] > 0
    terminal_increment_hits = int(
        (terminal_candidate_mask & terminal_increment).sum()
    )
    terminal_increment_judgeable = bool(
        terminal_increment_pixels >= _MINIMUM_JUDGEABLE_SAMPLES
    )
    terminal_increment_support = (
        float(terminal_increment_hits) / terminal_increment_pixels
        if terminal_increment_pixels
        else 0.0
    )
    terminal_increment_pass: Optional[bool] = (
        terminal_increment_support >= TERMINAL_CONSTRUCTION_SUPPORT
        if terminal_increment_judgeable
        else None
    )
    exact_observable_completion = bool(
        terminal_progress + 1e-9 >= observable_terminal_progress
    )
    quantised_terminal_tie = bool(
        not exact_observable_completion
        and terminal_progress + 1e-9 >= minimum_terminal_progress
        and (
            terminal_increment_pass is True
            or terminal_increment_pass is None
        )
    )
    observable_completion = bool(
        (exact_observable_completion or quantised_terminal_tie)
        and terminal_increment_pass is not False
    )
    # The shared reading's ``ordered`` is the conjunction of three gates.  Two of
    # them decide here; the third, the pace *floor*, is recorded as a diagnostic.
    # A source-render tick is not itself a semantic state: timing variation may
    # legitimately omit one uniformly sampled raster prefix while preserving the
    # same continuous stroke.  Command/stroke order is hard-gated separately by
    # the serial semantic reader, so this alignment judges only bounded progress
    # and the source-local pace ceiling.
    passed = bool(
        reading.step_break_count == 0
        and reading.pace_break_count == 0
        and observable_completion
    )
    metrics = {
        "video_process_order_available": True,
        "video_process_order_verifier": "bounded_step_pipeline_rendered_progress_alignment",
        "video_process_order_variant_index": variant_index,
        "video_process_order_stroke_count": len(paths),
        "video_process_order_command_schedule": list(schedule),
        "video_process_order_advancing_frames": advancing,
        "video_process_order_non_advancing_frames": max(0, len(frames) - advancing),
        "video_progress_bank_states": len(bank),
        "video_progress_bank_quantum": round(1.0 / float(advancing), 6),
        "video_reading_reference_pace": round(float(reading.reference_pace), 6),
        "video_reading_distinct_states": reading.distinct_states,
        "video_reading_comparison_kernel": reading.comparison_kernel,
        "video_reading_max_step": round(float(reading.max_step), 6),
        "video_reading_step_ratio": round(float(reading.step_ratio), 6),
        "video_reading_step_violations": reading.step_break_count,
        "video_frame_pace_violations": reading.pace_break_count,
        "video_worst_frame_pace_ratio": round(float(reading.worst_pace_ratio), 6),
        "video_worst_frame_pair_change": round(float(reading.worst_pair_change), 6),
        "video_frame_pace_floor_violations": reading.pace_floor_break_count,
        "video_worst_frame_pace_floor_ratio": round(float(reading.worst_pace_floor_ratio), 6),
        "video_process_order_floor_is_diagnostic": True,
        "video_process_order_terminal_progress": round(terminal_progress, 6),
        "video_process_order_observable_terminal_progress": round(
            observable_terminal_progress,
            6,
        ),
        "video_process_order_minimum_terminal_progress": round(
            minimum_terminal_progress,
            6,
        ),
        "video_process_order_exact_observable_completion": (
            exact_observable_completion
        ),
        "video_process_order_terminal_increment_pixels": (
            terminal_increment_pixels
        ),
        "video_process_order_terminal_increment_hits": (
            terminal_increment_hits
        ),
        "video_process_order_terminal_increment_support": round(
            terminal_increment_support,
            6,
        ),
        "video_process_order_terminal_increment_sample_floor": (
            _MINIMUM_JUDGEABLE_SAMPLES
        ),
        "video_process_order_terminal_increment_support_floor": (
            TERMINAL_CONSTRUCTION_SUPPORT
        ),
        "video_process_order_terminal_increment_judgeable": (
            terminal_increment_judgeable
        ),
        "video_process_order_terminal_increment_pass": (
            terminal_increment_pass
        ),
        "video_process_order_quantised_terminal_tie": (
            quantised_terminal_tie
        ),
        "video_process_order_observable_completion": observable_completion,
        "video_bounded_step_order_pass": passed,
    }
    return metrics, passed


def _read_coordinate_process_order(
    frames: Sequence[np.ndarray],
    source: np.ndarray,
    record: Dict[str, Any],
    *,
    task_name: str,
    expected: str,
) -> Tuple[Dict[str, Any], Optional[bool]]:
    """Accept any complete order the task's visible rules make equivalent."""

    variants = _coordinate_serial_path_variants(task_name, record, expected)
    if variants is None:
        return {"video_process_order_available": False}, None
    if not variants or any(not variant for variant in variants):
        return {
            "video_process_order_available": False,
            "video_process_order_reason": (
                "the declared stroke grammar could not be instantiated"
            ),
        }, False

    attempts: List[Tuple[Dict[str, Any], Optional[bool]]] = []
    for index, paths in enumerate(variants):
        result = _read_coordinate_process_order_variant(
            frames,
            source,
            record,
            task_name=task_name,
            expected=expected,
            paths=paths,
            variant_index=index,
        )
        attempts.append(result)
        if result[1] is True:
            metrics = dict(result[0])
            metrics["video_process_order_variant_count"] = len(variants)
            metrics["video_process_order_variant_verdicts"] = [
                verdict for _values, verdict in attempts
            ]
            return metrics, True

    # A blind legal variant cannot be turned into evidence against a candidate.
    # Prefer its abstention to a different legal ordering's rejection.
    selected = next(
        (result for result in attempts if result[1] is None),
        attempts[0],
    )
    metrics = dict(selected[0])
    metrics["video_process_order_variant_count"] = len(variants)
    metrics["video_process_order_variant_verdicts"] = [
        verdict for _values, verdict in attempts
    ]
    return metrics, selected[1]


def _coordinate_answer_commit_state(
    frames: Sequence[np.ndarray],
    source: np.ndarray,
    record: Dict[str, Any],
    expected: str,
    *,
    task_name: str,
) -> Tuple[Dict[str, Any], bool]:
    """Read the local selection update from color evidence on each marker.

    Geometry and label identity come from the source.  Grayscale reconstruction
    noise does not constitute a selection; the terminal-scene gates separately
    verify retained source objects and the marker's appearance.  The marker's
    precise opacity curve and its antialiasing raster are presentation details.
    """
    if not frames:
        return {"video_answer_commit_evaluated": False}, False
    height, width = source.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    radius = float(record.get("point_radius") or PointTargetPuzzleGenerator.POINT_RADIUS) + 3.0
    readings: List[Dict[str, Any]] = []
    selected_starts: List[int] = []
    valid = True
    targets = set(expected)
    observed_targets: set[str] = set()
    for entry in record.get("candidates") or []:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "").strip().upper()
        center = _point_pair((entry.get("x"), entry.get("y")))
        if center is None:
            continue
        disc = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
        counts = []
        for frame in frames:
            patch = np.asarray(frame)[disc].astype(np.int16)
            red = (patch[:, 0] >= 100) & (patch[:, 0] - patch[:, 1] >= 32) & (patch[:, 0] - patch[:, 2] >= 32)
            counts.append(int(np.count_nonzero(red)))
        terminal_count = counts[-1]
        onset = max(8.0, terminal_count * 0.15)
        start = next((i for i, count in enumerate(counts) if count >= onset), None)
        if label in targets:
            observed_targets.add(label)
            selected = terminal_count >= 24 and start is not None and start >= 4
            monotonic = _no_large_regression(counts, start or 0, tolerance=max(8.0, terminal_count * 0.25))
            valid = bool(valid and selected and monotonic)
            if start is not None:
                selected_starts.append(start)
        else:
            selected = max(counts, default=0) < 8
            monotonic = True
            valid = bool(valid and selected)
        readings.append({"label": label, "selected": label in targets,
                         "red_pixel_counts": counts, "first_commit_frame": start,
                         "state_valid": bool(selected), "retained": bool(monotonic)})
    passed = bool(valid and targets and observed_targets == targets)
    return {
        "video_answer_commit_evaluated": True,
        "video_answer_commit_state_valid": passed,
        "video_answer_commit_start_frame": min(selected_starts) if selected_starts else None,
        "video_answer_commit_marker_readings": readings,
        "video_answer_commit_reason": "ok" if passed else "selection is missing, wrong, premature, or erased",
        "video_answer_commit_task": task_name,
    }, passed


def _verify_generic_coordinate_video(
    video_path: Path,
    record: Dict[str, Any],
    evaluator: PointTargetImageEvaluator,
    input_path: Path,
    expected: str,
    *,
    task_name: str,
) -> tuple[Dict[str, Any], bool]:
    """Verify the complete common point-target visual program.

    This parser is intentionally stricter than a final-frame decoder: it
    requires an unchanged input hold, progressive non-highlight construction,
    no wrong-option highlight at any time, a late unique correct highlight,
    bounded adjacent-frame changes, and a stable final hold.
    """

    capture = cv2.VideoCapture(video_path.as_posix())
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    frames: List[np.ndarray] = []
    if opened:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()

    dims = record.get("canvas_dimensions") or [512, 512]
    width, height = int(dims[0]), int(dims[1])
    source = _read_rgb_image(input_path, (width, height))
    media_valid = bool(
        opened
        and source is not None
        and 9 <= len(frames) <= 161
        and len(frames) % 4 == 1
        and abs(fps - 16.0) <= 0.25
        and all(frame.shape[:2] == (height, width) for frame in frames)
    )
    if not media_valid or source is None:
        return {
            "video_process_verifier": "coordinate_target_common_full_sequence_semantic_v1",
            "video_frame_count": len(frames),
            "video_fps": fps,
            "video_media_valid": False,
            "video_semantic_process_check_pass": False,
            "video_semantic_process_reason": "media contract failed",
            "video_option": None,
        }, False

    options = [evaluator.image_option_from_frame(frame, record)[0] for frame in frames]
    source_diffs = [
        float(np.mean(np.max(visual_delta(frame, source), axis=2) >= 18))
        for frame in frames
    ]
    adjacent_diffs = [
        float(
            np.mean(
                np.max(
                    np.abs(frames[index].astype(np.int16) - frames[index - 1].astype(np.int16)),
                    axis=2,
                )
                >= 24
            )
        )
        for index in range(1, len(frames))
    ]
    initial_hold = bool(
        max(source_diffs[: min(4, len(source_diffs))], default=1.0) <= 0.006
        and all(option is None for option in options[: min(4, len(options))])
    )
    expected_labels = set(str(expected))
    wrong_highlight_absent = all(
        option is None
        or (
            bool(option)
            and set(str(option)).issubset(expected_labels)
            and len(set(str(option))) == len(str(option))
        )
        for option in options
    )
    first_highlight = next((index for index, option in enumerate(options) if option is not None), None)
    answer_commit_metrics, answer_commit_pass = (
        _coordinate_answer_commit_state(
            frames,
            source,
            record,
            expected,
            task_name=task_name,
        )
    )
    raw_commit_start = answer_commit_metrics.get(
        "video_answer_commit_start_frame"
    )
    answer_commit_start = (
        int(raw_commit_start)
        if isinstance(raw_commit_start, int)
        and not isinstance(raw_commit_start, bool)
        else None
    )
    # A route with no construction has only one legal pre-answer phase: the
    # readable unchanged-input hold already hard-gated above.  Requiring its
    # smooth answer onset to wait for one third of the whole clip would make
    # the arbitrary final-hold length decide correctness.  Construction routes
    # retain the broader relative-timing guard; their answer must also follow
    # the independently verified drawing programme.
    minimum_answer_commit_frame = (
        4
        if task_name in NO_CONSTRUCTION_TASKS
        else max(4, len(frames) // 3)
    )
    late_highlight = bool(
        answer_commit_start is not None
        and answer_commit_start >= minimum_answer_commit_frame
    )
    pre_highlight = (
        source_diffs[4:answer_commit_start]
        if answer_commit_start is not None
        else []
    )
    construction_present = bool(pre_highlight and max(pre_highlight) >= 0.001)
    # These tasks have one atomic semantic commit: the input already contains
    # every geometric relation needed to select the target, so inventing a
    # decorative pre-highlight drawing would not be a reasoning state.  Their
    # whole-video contract is therefore source hold -> one global selection
    # update -> final hold.  All other common coordinate tasks retain the
    # cumulative-construction hard gate below.
    atomic_selection_tasks = NO_CONSTRUCTION_TASKS
    construction_required = task_name not in atomic_selection_tasks
    construction_ok = bool(construction_present or not construction_required)
    # Equal semantic status does not mean simultaneous physical drawing.  The
    # cumulative state may contain all lines, while TI2V must draw each
    # declared stroke from one fixed endpoint and finish it before the next.
    replayed_serial_variants = _coordinate_serial_path_variants(
        task_name,
        record,
        expected,
    )
    serial_construction_required = replayed_serial_variants is not None
    serial_construction_metrics: Dict[str, Any] = {
        "video_serial_stroke_required": serial_construction_required,
    }
    serial_construction_ok = not serial_construction_required
    if serial_construction_required and answer_commit_start is not None:
        serial_construction_metrics, serial_verdict = _verify_serial_coordinate_strokes(
            frames,
            source,
            record,
            task_name=task_name,
            expected=expected,
            construction_start=4,
            construction_end=answer_commit_start,
        )
        # ``None`` is the checker declining to judge: no declared stroke has a
        # position the video can be read at, so this dimension contributes no
        # verdict rather than a failure it cannot support.
        serial_construction_ok = serial_verdict is not False
    # A verified source-derived stroke is stronger evidence than the global
    # changed-pixel area floor, which cannot measure a short, thin arc.
    if serial_construction_required and serial_construction_ok:
        construction_ok = True
    # Drawing is cumulative.  Compression may move the measured fraction a
    # little, but a large regression signals a cut, disappearance or reset.
    cumulative_construction = all(
        pre_highlight[index + 1] + 0.010 >= pre_highlight[index]
        for index in range(len(pre_highlight) - 1)
    )
    bounded_frame_changes = max(adjacent_diffs, default=1.0) <= 0.22
    no_blank_frames = all(15.0 <= float(np.mean(frame)) <= 254.9 for frame in frames)
    final_option = options[-1]
    final_hold_count = min(5, max(3, len(frames) // 8))
    final_hold_options = all(option == expected for option in options[-final_hold_count:])
    final_hold_stable = all(
        float(np.mean(np.max(np.abs(frame.astype(np.int16) - frames[-1].astype(np.int16)), axis=2) >= 18)) <= 0.004
        for frame in frames[-final_hold_count:]
    )
    # These tasks ask for a marker selection, without prescribing a shared
    # per-pixel opacity curve. The marker observer already verifies a unique,
    # retained correct selection and absence of premature/wrong commits.
    process_ok = bool(
        initial_hold
        and wrong_highlight_absent
        and late_highlight
        and construction_ok
        and serial_construction_ok
        and answer_commit_pass
        and cumulative_construction
        and bounded_frame_changes
        and no_blank_frames
        and final_hold_options
        and final_hold_stable
        and final_option == expected
    )
    return {
        "video_process_verifier": "coordinate_target_common_full_sequence_semantic_v1",
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_media_valid": media_valid,
        "video_option": final_option,
        "video_highlight_options": options,
        "video_initial_hold": initial_hold,
        "video_no_wrong_highlight": wrong_highlight_absent,
        "video_first_highlight_frame": first_highlight,
        "video_answer_commit_minimum_frame": (
            minimum_answer_commit_frame
        ),
        "video_late_highlight": late_highlight,
        **answer_commit_metrics,
        "video_answer_commit_pass": answer_commit_pass,
        "video_construction_required_before_highlight": construction_required,
        "video_construction_present_before_highlight": construction_present,
        "video_construction_requirement_pass": construction_ok,
        **serial_construction_metrics,
        "video_serial_construction_pass": serial_construction_ok,
        "video_cumulative_construction": cumulative_construction,
        "video_bounded_adjacent_frame_changes": bounded_frame_changes,
        "video_no_blank_frames": no_blank_frames,
        "video_final_hold_options": final_hold_options,
        "video_final_hold_stable": final_hold_stable,
        "video_source_diff_fractions": source_diffs,
        "video_adjacent_diff_fractions": adjacent_diffs,
        "video_semantic_process_check_pass": process_ok,
        "video_semantic_process_reason": "ok" if process_ok else "input hold, serial cumulative construction, highlight order, continuity, or final hold failed",
    }, process_ok


def verify_coordinate_target(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    training_phase: Optional[str] = None,
) -> List[EvalRecord]:
    records: List[EvalRecord] = []
    evaluator_cache: Dict[tuple[str, str], object] = {}

    for sample in samples:
        metadata_record_id = resolve_source_record_id(sample)
        record = EvalRecord(
            sample_id=sample.id,
            task_name=sample.task_name,
            input_asset=sample.primary_input_image,
            task_family=sample.task_family,
            split=sample.split,
        )
        try:
            candidate_video = resolve_candidate_video(pred_root, sample.id, sample.split)
            if candidate_video is None:
                raise FileNotFoundError(f"No prediction video found for sample_id={sample.id}")
            geometry_continuity_metrics: Dict[str, Any] = {}
            geometry_continuity_pass = True
            if candidate_video is not None:
                geometry_continuity_metrics, geometry_continuity_pass = (
                    verify_geometry_video_continuity(
                        candidate_video,
                        task_name=sample.task_name,
                        max_progress_step=0.35,
                    )
                )

            cache_key = (sample.task_name, sample.id)
            evaluator = evaluator_cache.get(cache_key)
            if evaluator is None:
                evaluator = PointTargetImageEvaluator(
                    resolve_source_record(sample),
                    base_dir=resolve_source_task_dir(sample),
                )
                evaluator_cache[cache_key] = evaluator

            puzzle = evaluator.get_record(metadata_record_id)
            if sample.task_name == "circle_inversion":
                # Frozen and paraphrased prompts retain their own public pen
                # contract; today's generator cannot replace that instruction.
                puzzle = {**puzzle, **_circle_inversion_prompt_contract(sample.prompt)}
            # Additive constructions share a cumulative-ink obligation.
            # arc_connect instead removes a moving opaque band; its task-owned
            # observer checks ordered narrowing, arc reveal and highlighting.

            # All three take their reference from the supplied input and the
            # sample's own declared geometry.  The task branches read only the
            # colours their own answer is drawn in, and the family contract and
            # the delivery's static hold both derive their region from the
            # candidate's own endpoints, so none of them can see a mutation
            # that lasts to the final frame or a deletion that removes process
            # without taking ink back.
            if candidate_video is not None:
                decoded_frames = _decode_rgb_frames(candidate_video)
                dims = puzzle.get("canvas_dimensions") or [512, 512]
                video_source = _read_rgb_image(
                    Path(sample.primary_input_image).expanduser().resolve(),
                    (int(dims[0]), int(dims[1])),
                )
                if sample.task_name == "arc_connect":
                    geometry_continuity_metrics.update(
                        video_ink_order_applicable=False,
                        video_ink_order_reason="opaque-band removal is checked by the source-derived reveal observer",
                    )
                else:
                    ink_order_metrics, ink_order_pass = _verify_cumulative_ink_order(
                        decoded_frames,
                        video_source,
                    )
                    geometry_continuity_metrics.update(ink_order_metrics, video_ink_order_applicable=True)
                    geometry_continuity_pass = bool(geometry_continuity_pass and ink_order_pass)
                if video_source is not None and decoded_frames:
                    declared_answer_for_source = str(
                        _derive_coordinate_route_answer(sample.task_name, puzzle)
                        or ""
                    )
                    dynamic_region = _source_dynamic_region(
                        sample.task_name,
                        puzzle,
                        declared_answer_for_source,
                        video_source.shape[:2],
                    )
                    if dynamic_region is None:
                        geometry_continuity_metrics["video_source_static_evaluated"] = False
                        geometry_continuity_metrics["video_source_static_reason"] = (
                            "the route cannot name where this task's delivery may draw"
                        )
                    else:
                        static_metrics, static_pass = _verify_source_static_preservation(
                            decoded_frames,
                            video_source,
                            dynamic_region,
                        )
                        geometry_continuity_metrics.update(static_metrics)
                        geometry_continuity_pass = bool(geometry_continuity_pass and static_pass)
                    order_metrics, order_verdict = _read_coordinate_process_order(
                        decoded_frames,
                        video_source,
                        puzzle,
                        task_name=sample.task_name,
                        expected=declared_answer_for_source,
                    )
                    geometry_continuity_metrics.update(order_metrics)
                    # ``None`` is the reading declining for want of a declared
                    # stroke order or render programme, which is no verdict
                    # rather than a failure.
                    # The dedicated geometric parsers independently read the
                    # required strokes and their order.  Their complete cyan
                    # constructions can differ in thickness from a raster bank;
                    # nearest raster progress is diagnostic for those routes.
                    serial_variants = _coordinate_serial_path_variants(
                        sample.task_name, puzzle, declared_answer_for_source
                    )
                    programme = _coordinate_programme(
                        sample.task_name, serial_variants[0] if serial_variants else ()
                    )
                    semantic_stroke_order = bool(
                        serial_variants
                        and not any(str(command.get("kind")) == _SMOOTH for command in programme or ())
                    )
                    order_is_diagnostic = bool(
                        sample.task_name in SEMANTIC_COORDINATE_TARGET_VIDEO_TASKS
                        or semantic_stroke_order
                    )
                    geometry_continuity_metrics["video_process_order_is_diagnostic"] = order_is_diagnostic
                    semantic_endpoint_owned = sample.task_name in {"angle_bisector", "parallel"}
                    last_stroke = serial_variants[0][-1] if serial_variants and serial_variants[0] else None
                    closed_stroke_owned = bool(
                        order_is_diagnostic and last_stroke is not None and len(last_stroke) >= 17
                        and np.linalg.norm(last_stroke[0] - last_stroke[-1]) <= 1.5
                    )
                    # A complete circle has no privileged final pixel. The
                    # excircle contact arc also has no public stopping angle.
                    # Their source geometry and continuous pen growth own
                    # completion, independently of a renderer's final increment.
                    raster_completion_diagnostic = bool(
                        semantic_endpoint_owned or closed_stroke_owned
                        # A long perpendicular bisector has no public finite
                        # display endpoint. Its source-derived line coverage
                        # and serial growth own completion.
                        or sample.task_name == "perpendicular_bisector"
                        or sample.task_name == "excircle_tangent_point"
                    )
                    geometry_continuity_metrics["video_process_order_terminal_increment_is_diagnostic"] = raster_completion_diagnostic
                    if semantic_endpoint_owned:
                        endpoint_metrics, endpoint_pass = _verify_semantic_line_endpoint(
                            decoded_frames, video_source, puzzle, task_name=sample.task_name,
                            expected=declared_answer_for_source,
                        )
                        geometry_continuity_metrics.update(endpoint_metrics)
                        geometry_continuity_pass &= endpoint_pass is not False
                    geometry_continuity_pass = bool(
                        geometry_continuity_pass
                        and (
                            raster_completion_diagnostic
                            or order_metrics.get("video_process_order_terminal_increment_pass") is not False
                        )
                        and (order_is_diagnostic or order_verdict is not False)
                    )
            dedicated_video_verifiers = {
                "midpoint": verify_midpoint_full_video,
                "isosceles_trapezoid": verify_isosceles_trapezoid_full_video,
                "circle_tangent_line": _verify_circle_tangent_line_semantic_video,
                "circle_tangent_point": verify_circle_tangent_point_full_video,
                "circumcenter": verify_circumcenter_full_video,
                "orthocenter": verify_orthocenter_full_video,
                "parallel": verify_parallel_full_video,
                "incenter": verify_incenter_full_video,
                "arc_connect": _verify_arc_connect_semantic_video,
                "angle_bisector": _verify_angle_bisector_semantic_video,
            }
            if sample.task_name in dedicated_video_verifiers:
                input_path = Path(sample.primary_input_image).expanduser().resolve()
                video_metrics, video_pass = dedicated_video_verifiers[sample.task_name](
                    candidate_video,
                    puzzle,
                    input_path,
                )
                delivery_metrics, delivery_pass = _video_delivery_ink(
                    candidate_video, input_path, puzzle
                )
                primary_metrics = {
                    **video_metrics,
                    **geometry_continuity_metrics,
                    **delivery_metrics,
                }
                offline_pass = bool(
                    video_pass and geometry_continuity_pass and delivery_pass
                )
                artifact_passes = {"video": offline_pass}
                artifact_metrics = {"video": primary_metrics}
                metrics = {
                    "puzzle_id": metadata_record_id,
                    "correct_option": primary_metrics.get(
                        f"{sample.task_name}_derived_correct_option"
                    ),
                    **primary_metrics,
                    f"{sample.task_name}_artifact_passes": artifact_passes,
                    f"{sample.task_name}_metrics_by_artifact": artifact_metrics,
                    "binary_correct": offline_pass,
                    "strict_multimodal_hard_gate": True,
                    "performance_policy": "latency_throughput_memory_and_decode_counts_are_diagnostics_without_failure_thresholds",
                }
                record.prediction_asset = candidate_video.as_posix()
                record.offline_metrics = metrics
                record.offline_pass = offline_pass
                records.append(record)
                continue
            metrics: Dict[str, object] = {
                "puzzle_id": metadata_record_id,
                "correct_option": None,
                "video_option": None,
            }
            try:
                derived_answer, replay_metrics = derive_coordinate_answer(
                    sample.task_name,
                    puzzle,
                )
                source_replay_valid = True
                metrics.update(replay_metrics)
                metrics["correct_option"] = derived_answer
            except (CoordinateReplayError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                derived_answer = None
                source_replay_valid = False
                metrics.update(
                    {
                        "coordinate_source_replay_valid": False,
                        "coordinate_source_replay_reason": str(exc),
                        "coordinate_derived_answer": None,
                    }
                )
            declared_answer = str(derived_answer or "")
            declared_variants = _coordinate_serial_path_variants(
                sample.task_name,
                puzzle,
                declared_answer,
            )
            declared_paths = list(
                terminal_construction_paths(
                    sample.task_name,
                    puzzle,
                    declared_answer,
                    line_width=renderer_line_width(puzzle),
                )
                or ()
            )
            video_semantic_pass: Optional[bool] = None
            if sample.task_name in SEMANTIC_COORDINATE_TARGET_VIDEO_TASKS:
                video_metrics, video_semantic_pass = _verify_midpoint_semantic_video(candidate_video, puzzle)
                metrics.update(video_metrics)
            elif derived_answer is not None:
                video_metrics, video_semantic_pass = _verify_generic_coordinate_video(
                    candidate_video,
                    puzzle,
                    evaluator,
                    Path(sample.primary_input_image).expanduser().resolve(),
                    derived_answer,
                    task_name=sample.task_name,
                )
                metrics.update(video_metrics)
            metrics.update(geometry_continuity_metrics)
            if declared_variants is not None:
                (
                    video_delivery_metrics,
                    video_delivery_pass,
                    video_construction_pass,
                ) = _verify_coordinate_video_variant_delivery(
                    decoded_frames,
                    video_source,
                    puzzle,
                    task_name=sample.task_name,
                    expected=declared_answer,
                    variants=declared_variants,
                )
            else:
                video_delivery_metrics, video_delivery_pass = (
                    _video_delivery_ink(
                        candidate_video,
                        Path(
                            sample.primary_input_image
                        ).expanduser().resolve(),
                        puzzle,
                        declared_paths,
                        task_name=sample.task_name,
                        expected=declared_answer,
                    )
                )
                (
                    video_construction_metrics,
                    video_construction_pass,
                ) = _verify_video_construction(
                    decoded_frames,
                    video_source,
                    puzzle,
                    task_name=sample.task_name,
                    expected=derived_answer,
                )
                video_delivery_metrics.update(video_construction_metrics)
            metrics.update(video_delivery_metrics)
            metrics["video_terminal_construction_pass"] = video_construction_pass
            video_semantic_pass = bool(
                video_semantic_pass is True
                and geometry_continuity_pass
                and video_delivery_pass
                and video_construction_pass is not False
            )
            artifact_options = _formal_artifact_options(metrics)
            artifact_parse_complete = bool(video_semantic_pass is True)
            consistent_option = (
                artifact_options[0]
                if artifact_options and all(value == artifact_options[0] for value in artifact_options)
                else None
            )
            correct = _normalize_option(metrics.get("correct_option"))
            offline_pass = bool(
                source_replay_valid
                and video_semantic_pass
                and artifact_parse_complete
                and all(value == correct for value in artifact_options)
            )
            metrics["artifact_options"] = artifact_options
            metrics["artifact_parse_complete"] = artifact_parse_complete
            metrics["consistent_option"] = consistent_option
            metrics["normalized_correct_option"] = correct
            semantic_artifact_results: Dict[str, bool] = {
                "video": bool(source_replay_valid and video_semantic_pass is True)
            }
            metrics["video_final_state_pass"] = bool(
                source_replay_valid
                and video_delivery_metrics.get("video_terminal_delivery_pass") is True
                and video_construction_pass is not False
                and _normalize_option(metrics.get("video_option")) == correct
            )
            metrics["artifact_results"] = semantic_artifact_results

            record.prediction_asset = candidate_video.as_posix()
            record.offline_metrics = metrics
            record.offline_pass = offline_pass
        except Exception as exc:  # pragma: no cover - guarded by the integration audit
            record.error = str(exc)
            record.offline_pass = False
        records.append(record)

    return records
