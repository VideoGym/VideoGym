"""Full-video verifier for the unified labelled-graph tasks.

Every expected value and visual state is replayed from canonical source fields.
Reference answers and saved terminal media are outside this module's input domain.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.verifiers.region_gates import (
    STATIC_REGION_FLOOR,
    static_region_errors,
    terminal_region_gates,
)
from data.graph_theory.core.rendering import (
    ANSWER_RING_TASKS,
    NODE_RADIUS,
    ORANGE,
    PHASE_SPLIT,
    degree_badge_center,
    solution_ring_labels,
    use_answer_ring_masks,
)
from data.graph_theory.core.replay import UnifiedGraphReplay, replay_unified_graph
from data.graph_theory.core.semantics import edge_key


_MODALITIES = frozenset({"ti2v"})
_FIXED_FOOTPRINT_LOCAL_FADE_TASKS = frozenset({
    "balance_comparison",
    "bipartite_color_completion",
    "bridge_edge_choice",
    "closest_common_ancestor",
    "connected_component_count",
    "degree_missing_value",
    "diameter_endpoints",
    "euler_trail_endpoints",
    "leaf_count",
    "subtree_size",
    "two_hop_neighbor_count",
})
_FADE_FOOTPRINT_MIN_PIXELS = 96
_FADE_PROJECTION_SPREAD_LIMIT = 0.08
_FADE_FOOTPRINT_SUPPORT_LIMIT = 0.80
_FADE_FULL_MASK_FIT_LIMIT = 0.018
_FADE_COMPONENT_SYNC_LIMIT = 0.12
_FADE_ALPHA_REGRESSION_LIMIT = 0.04
# A shared group tint may have small codec/lighting differences. Read every
# node pair in perceptual color space; a median RGB box can put a stable blue
# and a stable purple on opposite sides of one permissive "same color" center.
_COMPONENT_TINT_MAX_PAIR_DELTA_E = 10.0

# ``trace_all_edges_once`` accepts every legal Euler trail, so its TI2V gate
# reconstructs the committed-edge sequence from the candidate's orange stroke
# instead of matching one canonical reference trail.
_TRACE_ORANGE_DISTANCE_LIMIT = 36.0
_TRACE_COMMIT_COVERAGE = 0.80
_TRACE_PARTIAL_LOW = 0.10
_TRACE_GROWING_HIGH = 0.70
_TRACE_COVERAGE_REGRESSION = 0.06
_TRACE_MIN_PARTIAL_FRAMES = 3
_TRACE_MAX_GROUP_SIZE = 2
_TRACE_MIN_SPAN_FRACTION = 0.40


def _fixed_footprint_fade_phases(task_name: str) -> tuple[tuple[str, float, float], ...]:
    """Return the task's opacity-only reveal phases.

    Single-commit tasks share one source-to-terminal opacity.  Two-phase
    tasks first fade their complete evidence overlay (tints or teal routes)
    and then their answer overlay, split at the shared PHASE_SPLIT boundary
    used by both replay checkpoints and the renderer.
    """

    split = PHASE_SPLIT.get(task_name)
    if split is not None:
        return (
            ("evidence_overlay", 0.0, float(split)),
            ("answer_overlay", float(split), 1.0),
        )
    return (("answer_overlay", 0.0, 1.0),)


def _rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return array[:, :, :3]


def _resize(array: np.ndarray, shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    if array.shape[:2] == shape:
        return array
    return cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _dynamic_mask(initial: np.ndarray, expected_states: Sequence[np.ndarray]) -> np.ndarray:
    mask = np.zeros(initial.shape[:2], dtype=bool)
    baseline = initial.astype(np.int16)
    for expected in expected_states:
        mask |= np.max(np.abs(baseline - expected.astype(np.int16)), axis=2) > 9
    if np.any(mask):
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    return mask


def _fixed_label_metrics(frames, initial, source):
    """Read each fixed node label locally so canvas means cannot hide erasure."""
    from data.graph_theory.core.rendering import INK
    scores = []
    for node in source["nodes"]:
        x, y = (int(round(value)) for value in node["position"])
        box = (slice(max(0, y - 8), y + 9), slice(max(0, x - 11), x + 12))
        template = initial[box].astype(np.int16)
        ink = np.max(np.abs(template - np.asarray(INK)), axis=2) <= 35
        # A thin antialiased digit can have fewer than four core-ink pixels.
        # Its readable source strokes still provide the preservation template.
        if not np.any(ink):
            raise ValueError("source node label has no readable ink")
        for frame in frames:
            observed = _resize(frame)[box].astype(np.int16)
            dark = (np.max(np.abs(observed - np.asarray(INK)), axis=2) <= 65).astype(np.uint8)
            nearby = cv2.dilate(dark, np.ones((3, 3), np.uint8)) > 0
            scores.append(float(nearby[ink].mean()))
    minimum = min(scores, default=0.0)
    return {"fixed_node_label_min_ink_recall": minimum,
            "fixed_node_labels_preserved": minimum >= .80}


def _terminal_image_semantics(candidate: np.ndarray, replay: UnifiedGraphReplay, *, observed_answer_set=False) -> dict[str, Any]:
    initial = _rgb(replay.render_input())
    terminal = _rgb(replay.render_progress(1.0))
    candidate = _resize(candidate, terminal.shape[:2])
    delta = np.abs(candidate.astype(np.float32) - terminal.astype(np.float32))
    full_error = float(np.mean(delta)) / 255.0
    changed = _dynamic_mask(initial, [terminal])
    semantic_error = float(np.mean(delta[changed])) / 255.0 if np.any(changed) else full_error
    expected_change = (
        float(np.mean(np.abs(terminal.astype(np.float32) - initial.astype(np.float32))[changed])) / 255.0
        if np.any(changed) else 0.0
    )
    candidate_change = (
        float(np.mean(np.abs(candidate.astype(np.float32) - initial.astype(np.float32))[changed])) / 255.0
        if np.any(changed) else 0.0
    )
    static = ~changed
    source_delta = np.abs(candidate.astype(np.float32) - initial.astype(np.float32))
    static_error = float(np.mean(source_delta[static])) / 255.0 if np.any(static) else full_error
    semantic_tolerance = max(0.025, min(0.06, expected_change * 0.45))
    changed_present = bool(
        np.any(changed)
        and expected_change >= 0.01
        and candidate_change >= expected_change * 0.55
        and semantic_error <= semantic_tolerance
    )
    static_preserved = static_error <= 0.035
    # Every aggregate above is a mean over a mask, so one completely missing
    # node, edge or badge is paid for by the ones that are present.  The
    # per-region / per-member gates hold each object to its own evidence

    region_metrics, region_pass = terminal_region_gates(
        candidate, initial, terminal, [initial], prefix="terminal_image_"
    )
    labels = _fixed_label_metrics([candidate], initial, replay.source)
    if observed_answer_set:
        # The node set is independently decoded. Per-region and per-member
        # delivery gates retain the evidence checks without charging a second
        # aggregate shape error to those already validated ring outlines.
        changed_present = bool(np.any(changed) and expected_change >= .01
                               and candidate_change >= expected_change*.55 and region_pass)
    badge = _degree_badge_semantics(candidate, replay)
    correct = bool(full_error <= 0.035 and changed_present and static_preserved
                   and region_pass and labels["fixed_node_labels_preserved"]
                   and badge.get("degree_badge_correct", True))
    return {
        "terminal_image_source_render_full_error": full_error,
        "terminal_image_semantic_region_error": semantic_error,
        "terminal_image_expected_semantic_change": expected_change,
        "terminal_image_candidate_semantic_change": candidate_change,
        "terminal_image_semantic_tolerance": semantic_tolerance,
        "terminal_image_static_region_error": static_error,
        "terminal_image_semantic_change_present": float(changed_present),
        "terminal_image_static_preserved": float(static_preserved),
        **region_metrics,
        **badge,
        **labels,
        "terminal_image_correct": float(correct),
        "terminal_image_reference_pixels_used": False,
        "terminal_image_saved_final_state_used": False,
    }


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise ValueError("candidate video cannot be opened")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError("candidate video contains no readable frames")
    return frames, fps


def _progress_bank(replay: UnifiedGraphReplay) -> tuple[list[float], list[np.ndarray]]:
    values = {index / 32.0 for index in range(33)}
    values.update(replay.semantic_checkpoints)
    ordered = sorted(max(0.0, min(1.0, float(value))) for value in values)
    return ordered, [_rgb(replay.render_progress(value)) for value in ordered]


def _frame_replay_errors(
    frame: np.ndarray,
    bank_float: Sequence[np.ndarray],
    dynamic: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Measure both errors from one float32 difference per source state."""
    regional, complete = [], []
    for expected in bank_float:
        delta = np.abs(frame - expected)
        regional.append(float(np.mean(delta[dynamic])) / 255.0)
        complete.append(float(np.mean(delta)) / 255.0)
    return regional, complete


def _linear_fade_footprint(
    replay: UnifiedGraphReplay,
    start: float,
    finish: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive one opacity-only terminal footprint from task-owned renders.

    The mask is not taken from a saved reference video.  It starts with the
    source-derived endpoint delta, then retains only pixels that the task
    renderer independently predicts to follow the same 25/50/75% local-alpha
    trajectory.  This excludes legitimate growing path pixels in mixed phases
    while retaining the complete footprint of rings, fills, glyphs, and edge
    emphases that are contractually opacity-only.
    """

    base = _rgb(replay.render_input() if start <= 0.0 else replay.render_progress(start)).astype(np.float32)
    terminal = _rgb(replay.render_progress(finish)).astype(np.float32)
    delta = terminal - base
    denominator = np.sum(delta * delta, axis=2)
    mask = denominator >= 30.0 ** 2
    for fraction in (0.25, 0.5, 0.75):
        progress = start + (finish - start) * fraction
        expected = _rgb(replay.render_progress(progress)).astype(np.float32)
        projection = np.sum((expected - base) * delta, axis=2) / np.maximum(denominator, 1.0)
        reconstruction = np.mean(np.abs(expected - (base + fraction * delta)), axis=2)
        mask &= np.abs(projection - fraction) <= 0.10
        mask &= reconstruction <= 8.0
    return base, delta, denominator, mask


def _fixed_footprint_local_fade_metrics(
    frames: Sequence[np.ndarray],
    replay: UnifiedGraphReplay,
) -> dict[str, Any]:
    """Measure semantic opacity progress and strict local-fit diagnostics.

    Nearest-progress replay alone is insensitive to a center-out reveal: a
    half-visible saturated stroke can be close in mean error to a half-opacity
    complete stroke.  For every declared fade phase this routine projects each
    candidate pixel onto the independently rendered source-to-terminal color
    vector. The process gate consumes readable partial opacity and local
    monotonicity. Full-footprint support, pixel fit, opacity spread and
    component synchronization remain diagnostics because generated raster
    variation can violate their strict limits in readable legal fades.
    """

    if replay.task_name not in _FIXED_FOOTPRINT_LOCAL_FADE_TASKS:
        return {
            "required": False,
            "valid": True,
            "phases": {},
        }

    resized = [_resize(frame).astype(np.float32) for frame in frames]
    phase_metrics: dict[str, Any] = {}
    all_valid = True
    for name, start, finish in _fixed_footprint_fade_phases(replay.task_name):
        base, delta, denominator, mask = _linear_fade_footprint(replay, start, finish)
        footprint_pixels = int(np.count_nonzero(mask))
        if footprint_pixels < _FADE_FOOTPRINT_MIN_PIXELS:
            phase_metrics[name] = {
                "observation_footprint_pixels": footprint_pixels,
                "valid": False,
                "error": "source-derived opacity footprint is too small",
            }
            all_valid = False
            continue

        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        component_masks = [
            component_labels == index
            for index in range(1, component_count)
            if int(component_stats[index, cv2.CC_STAT_AREA]) >= 20
        ]

        alphas: list[float] = []
        partial_spreads: list[float] = []
        partial_supports: list[float] = []
        partial_fits: list[float] = []
        partial_component_spreads: list[float] = []
        partial_levels: set[int] = set()
        for candidate in resized:
            projection = np.sum((candidate - base) * delta, axis=2) / np.maximum(denominator, 1.0)
            footprint_projection = projection[mask]
            alpha = float(np.median(footprint_projection))
            alphas.append(alpha)
            if not 0.10 < alpha < 0.90:
                continue
            partial_levels.add(int(round(max(0.0, min(1.0, alpha)) * 8.0)))
            q10, q90 = np.quantile(footprint_projection, (0.10, 0.90))
            partial_spreads.append(float(q90 - q10))
            visible_floor = max(0.035, alpha * 0.35)
            partial_supports.append(float(np.mean(footprint_projection >= visible_floor)))
            reconstructed = base + alpha * delta
            partial_fits.append(
                float(np.mean(np.abs(candidate[mask] - reconstructed[mask]))) / 255.0
            )
            component_alphas = [
                float(np.median(projection[component_mask]))
                for component_mask in component_masks
            ]
            partial_component_spreads.append(
                max(component_alphas) - min(component_alphas)
                if len(component_alphas) >= 2 else 0.0
            )

        regressions = [
            left - right
            for left, right in zip(alphas, alphas[1:])
            if right + _FADE_ALPHA_REGRESSION_LIMIT < left
        ]
        start_alpha = max(alphas[: min(3, len(alphas))], default=1.0)
        finish_alpha = min(alphas[max(0, len(alphas) - 3) :], default=0.0)
        endpoints_valid = start_alpha <= 0.06 and finish_alpha >= 0.94
        partial_frame_count = len(partial_spreads)
        partial_level_count = len({level for level in partial_levels if 1 <= level <= 7})
        max_spread = max(partial_spreads, default=1.0)
        min_support = min(partial_supports, default=0.0)
        max_fit = max(partial_fits, default=1.0)
        max_component_spread = max(partial_component_spreads, default=1.0)
        phase_valid = bool(
            endpoints_valid
            and not regressions
            and partial_frame_count >= 3
            and partial_level_count >= 3
            and max_spread <= _FADE_PROJECTION_SPREAD_LIMIT
            and min_support >= _FADE_FOOTPRINT_SUPPORT_LIMIT
            and max_fit <= _FADE_FULL_MASK_FIT_LIMIT
            and max_component_spread <= _FADE_COMPONENT_SYNC_LIMIT
        )
        phase_metrics[name] = {
            "source_progress_interval": [start, finish],
            "observation_footprint_pixels": footprint_pixels,
            "observation_component_count": len(component_masks),
            "start_opacity_upper_bound": start_alpha,
            "terminal_opacity_lower_bound": finish_alpha,
            "opacity_endpoints_valid": float(endpoints_valid),
            "opacity_monotonic": float(not regressions),
            "max_opacity_regression": max(regressions, default=0.0),
            "partial_frame_count": partial_frame_count,
            "partial_opacity_level_count": partial_level_count,
            "max_partial_projection_q10_q90_spread": max_spread,
            "min_partial_full_footprint_support": min_support,
            "max_partial_full_mask_fit_error": max_fit,
            "max_partial_component_opacity_spread": max_component_spread,
            "valid": float(phase_valid),
        }
        all_valid &= phase_valid

    return {
        "required": True,
        "valid": bool(all_valid),
        "phases": phase_metrics,
    }


def _degree_badge_box(replay):
    x, y = (int(round(value)) for value in degree_badge_center(replay.source))
    return (slice(max(0, y - 8), min(512, y + 9)),
            slice(max(0, x - 8), min(512, x + 9)))


def _degree_badge_semantics(candidate, replay):
    """Decode the visible digit independently of the solved degree."""
    if replay.task_name != "degree_missing_value":
        return {}
    from data.common.verifiers.numeric_glyphs import numeric_glyph_descriptor, read_numeric_glyph
    box = _degree_badge_box(replay)
    vocabulary = {}
    for value in range(10):
        other = UnifiedGraphReplay(replay.task_name, replay.source, dict(replay.solution, answer=value))
        descriptor = numeric_glyph_descriptor(_rgb(other.render_progress(1.))[box])
        if descriptor is not None:
            vocabulary[value] = descriptor
    glyph_patch = _resize(candidate)[box]
    observed, similarity, margin = read_numeric_glyph(glyph_patch, vocabulary)
    descriptor = numeric_glyph_descriptor(glyph_patch)
    literal = sorted((float(np.sum(descriptor * template) / max(
        1e-6, np.linalg.norm(descriptor) * np.linalg.norm(template))), value)
        for value, template in vocabulary.items()) if descriptor is not None else []
    pixel_similarity, pixel_value = literal[-1] if literal else (0., None)
    pixel_margin = pixel_similarity - (literal[-2][0] if len(literal) > 1 else 0.)
    from data.common.verifiers.printed_digits import read_printed_digit
    patch = _resize(candidate)[box].copy()
    yy, xx = np.indices(patch.shape[:2])
    # Exclude the fixed purple ring using the source badge geometry. Its dark
    # arcs are not part of the digit and confuse single-character OCR.
    patch[(xx-8)**2 + (yy-8)**2 > 7**2] = (255, 248, 214)
    printed = read_printed_digit(patch)
    if pixel_similarity >= .95 and pixel_margin >= .025:
        # A complete source glyph can be confused by one OCR segmentation
        # mode. Its near-identical full silhouette identifies it directly.
        observed = pixel_value
    elif printed["digit"] is not None:
        observed = printed["digit"]
    elif any(value is not None for value in printed["readings"]):
        # Conflicting readable digits are positive evidence of ambiguity;
        # they must not fall back to a different contour-template guess.
        observed = None
    return {"degree_badge_observed": observed, "degree_badge_expected": replay.solution["answer"],
            "degree_badge_similarity": similarity, "degree_badge_margin": margin,
            "degree_badge_printed_reading": printed,
            "degree_badge_pixel_similarity": pixel_similarity,
            "degree_badge_pixel_margin": pixel_margin,
            "degree_badge_correct": observed == replay.solution["answer"]}


def _degree_badge_fade_metrics(frames, replay):
    """Observe question fade-out then digit fade-in, without a renderer clock.

    Both glyph templates come from the public source renderer. Their joint
    coefficients disambiguate overlapping strokes without treating each raw
    antialiased pixel change as a hard cut.
    """
    from data.common.verifiers.keyframes import visual_features
    box = _degree_badge_box(replay)
    blank = visual_features(_rgb(replay.render_progress(.5)))[box]
    question = visual_features(_rgb(replay.render_input()))[box] - blank
    digit = visual_features(_rgb(replay.render_progress(1.)))[box] - blank
    design = np.column_stack((question.ravel(), digit.ravel()))
    reader = np.linalg.pinv(design)
    readings = [reader @ (visual_features(_resize(frame))[box] - blank).ravel() for frame in frames]
    q = np.clip([value[0] for value in readings], 0., 1.)
    d = np.clip([value[1] for value in readings], 0., 1.)
    q_regression = float(np.max(q - np.minimum.accumulate(q)))
    d_regression = float(np.max(np.maximum.accumulate(d) - d))
    question_partials = int(sum(.15 < value < .85 for value in q))
    digit_partials = int(sum(.15 < value < .85 for value in d))
    # Any visibly appearing digit must follow the disappearing question.
    ordered = all(left <= .20 or right <= .20 for left, right in zip(q, d))
    valid = bool(q[0] >= .70 and d[0] <= .20 and q[-1] <= .20 and d[-1] >= .70
                 and q_regression <= .15 and d_regression <= .15
                 and question_partials and digit_partials and ordered)
    return {"valid": valid, "question_opacity": q.tolist(), "digit_opacity": d.tolist(),
            "question_regression": q_regression, "digit_regression": d_regression,
            "question_partial_frames": question_partials, "digit_partial_frames": digit_partials,
            "question_before_digit": ordered}


def _strong_orange(frame: np.ndarray) -> np.ndarray:
    """Saturated answer-orange pixels; source gold and group tints are excluded."""
    hsv = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (hue >= 5) & (hue <= 17) & (saturation >= 115) & (value >= 140)


def _observe_answer_rings(frames, replay):
    """Decode every marked node, independently of the source's answer set.

    A ring must enclose one visible node and stay in its local neighbourhood.
    Only its outline and antialias coverage are observed from the final hold;
    its colour, node identity and complete fade still have independent gates.
    """
    final = np.median(np.stack([_resize(frame) for frame in frames[-3:]]), axis=0).astype(np.uint8)
    initial = _rgb(replay.render_input()).astype(np.float32)
    strong = _strong_orange(final)
    yy, xx = np.indices(strong.shape)
    centers = np.array([node["position"] for node in replay.source["nodes"]], dtype=float)
    distances = np.stack([np.hypot(xx-x, yy-y) for x,y in centers])
    owners = distances.argmin(axis=0)
    masks, covered = {}, np.zeros(strong.shape, dtype=bool)
    # Branch comparison explicitly fills node interiors before adding rings.
    # A saturated but readable pastel orange fill is not an extra answer mark.
    tint_nodes = ({str(label) for component in replay.solution.get("components", []) for label in component}
                  if replay.task_name == "balance_comparison" else set())
    for index, node in enumerate(replay.source["nodes"]):
        distance = distances[index]
        region = (owners == index) & (distance >= NODE_RADIUS-1) & (distance <= 42)
        label = str(node["label"])
        if label in tint_nodes:
            covered |= strong & (distance < NODE_RADIUS - 1)
        ink = strong & region
        if int(ink.sum()) < 30:
            continue
        closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        center = tuple(map(float, node["position"]))
        holes = [contour for contour, relation in zip(contours, hierarchy[0] if hierarchy is not None else [])
                 if relation[3] >= 0 and cv2.pointPolygonTest(contour, center, False) > 0]
        if len(holes) != 1:
            continue
        hole = np.zeros(strong.shape, dtype=np.uint8)
        cv2.drawContours(hole, holes, -1, 1, thickness=cv2.FILLED)
        band = cv2.distanceTransform(1-hole, cv2.DIST_L2, 3) <= 12
        # Outline width may vary, but an attached route is not part of a ring.
        ink &= band
        # A closed outline must surround the node, rather than paint it in.
        if label not in tint_nodes and np.count_nonzero(strong & (distance < NODE_RADIUS-3)) > 8:
            continue
        target = np.asarray(ORANGE, dtype=np.float32)-initial
        denominator = np.maximum(np.sum(target*target, axis=2), 1.)
        alpha = np.clip(np.sum((final.astype(np.float32)-initial)*target, axis=2)/denominator, 0, 1)
        halo = cv2.dilate(ink.astype(np.uint8), np.ones((3,3),np.uint8)) > 0
        mask = np.where(region & halo & (alpha > .12), alpha, 0.).astype(np.float32)
        masks[str(node["label"])] = mask
        covered |= ink
    expected = set(solution_ring_labels(replay.task_name, replay.solution))
    stray = int(np.count_nonzero(strong & ~covered))
    return masks, {"video_answer_ring_nodes": sorted(masks),
                   "video_answer_ring_expected_nodes": sorted(expected),
                   "video_answer_ring_stray_pixels": stray,
                   "video_answer_ring_set_correct": set(masks) == expected and stray <= 24}


def _answer_ring_fade_metrics(frames, replay, masks):
    """Read opacity on each closed outline with the label reader's 1px reach."""
    initial = _rgb(replay.render_input()).astype(np.float32)
    direction = np.asarray(ORANGE, dtype=np.float32)-initial
    denominator = np.maximum(np.sum(direction*direction, axis=2), 1.)
    cores = {label: mask >= .55 for label, mask in masks.items()}
    readings = {label: [] for label in masks}
    supports = {label: [] for label in masks}
    for frame in frames:
        delta = _resize(frame).astype(np.float32)-initial
        alpha = np.sum(delta*direction, axis=2)/denominator
        residual = np.max(np.abs(delta-alpha[:,:,None]*direction), axis=2)
        alpha = np.where(residual <= 65, np.clip(alpha,0,1.25), 0).astype(np.float32)
        nearby = cv2.dilate(alpha, np.ones((3,3),np.uint8))
        for label, core in cores.items():
            values = nearby[core]
            amount = float(np.median(values)) if values.size else 0.
            readings[label].append(amount)
            if .10 < amount < .90:
                supports[label].append(float(np.mean(values >= max(.035,amount*.35))))
    per_node = {}
    for label, values in readings.items():
        regressions = [a-b for a,b in zip(values,values[1:]) if b+.05 < a]
        partial = [v for v in values if .10 < v < .90]
        per_node[label] = dict(start=max(values[:3]),terminal=min(values[-3:]),
                               max_regression=max(regressions,default=0.),
                               partial_frames=len(partial),partial_levels=len({round(v*8) for v in partial}),
                               min_support=min(supports[label],default=0.))
    # Synchrony concerns time, not slight differences in committed orange.
    # Ring identity/color and a complete terminal are checked independently.
    normalized = [[v / max(float(np.median(values[-3:])), .01) for v in values]
                  for values in readings.values()]
    synchronous = max((max(v)-min(v) for v in zip(*normalized)),default=1.)
    valid = bool(per_node) and synchronous <= .12 and all(
        v['start'] <= .10 and v['terminal'] >= .85 and v['max_regression'] <= .05
        and v['partial_frames'] >= 3 and v['partial_levels'] >= 3 and v['min_support'] >= .80
        for v in per_node.values())
    return dict(valid=valid,nodes=per_node,max_node_opacity_spread=synchronous)


def _node_interior_colors(frame, source):
    """Read fill away from the fixed label and outline, using source positions."""
    frame = _resize(frame)
    yy, xx = np.indices(frame.shape[:2])
    colors = {}
    for node in source["nodes"]:
        x, y = node["position"]
        radius = np.hypot(xx - x, yy - y)
        mask = (radius >= 10) & (radius <= NODE_RADIUS - 3)
        colors[str(node["label"])] = np.median(frame[mask], axis=0)
    return colors


def _node_tint_fade_metrics(frames, replay, terminal_colors):
    """Observe every changing node, so a canvas mean cannot hide a late tint.

    Endpoint colors are permissible choices already checked for meaning. Only
    opacity should change during a fade. Median interior readings tolerate edge
    raster noise without allowing one node to change hue or commit separately.
    """
    initial = _node_interior_colors(_rgb(replay.render_input()), replay.source)
    labels = (list(replay.solution["node_color"]) if replay.task_name == "bipartite_color_completion"
              else [str(n["label"]) for n in replay.source["nodes"]])
    observed = [_node_interior_colors(frame, replay.source) for frame in frames]
    readings, fits = {}, []
    for label in labels:
        label = str(label)
        base, final = initial[label], terminal_colors[label]
        direction = final - base
        energy = max(float(np.dot(direction, direction)), 1.)
        colors = np.array([value[label] for value in observed])
        alpha = (colors - base) @ direction / energy
        residual = np.abs(colors - (base + alpha[:, None]*direction)).max(axis=1)
        fits.append(float(residual.max()))
        readings[label] = alpha
    spread = max((float(max(v)-min(v)) for v in zip(*readings.values())), default=1.)
    per_node = {}
    for label, values in readings.items():
        partial = [float(v) for v in values if .10 < v < .90]
        per_node[label] = dict(start=float(max(values[:3])), terminal=float(min(values[-3:])),
            max_regression=max((float(a-b) for a,b in zip(values, values[1:])), default=0.),
            partial_frames=len(partial), partial_levels=len({round(v*8) for v in partial}))
    valid = bool(per_node) and max(fits, default=255.) <= 20 and spread <= .20 and all(
        v["start"] <= .10 and v["terminal"] >= .85 and v["max_regression"] <= .08
        and v["partial_frames"] >= 3 and v["partial_levels"] >= 3 for v in per_node.values())
    return dict(valid=valid, nodes=per_node, max_color_residual=max(fits, default=255.),
                max_node_opacity_spread=spread)


def _observe_node_tints(frames, replay):
    """Check tint meaning before using its permissible palette in source replay.

    Component colors are unnamed in the prompt. They must express exactly the
    independently solved partition; observing a palette never supplies or
    changes that partition. Bipartite RED/BLUE labels retain their fixed meaning.
    Candidate frames are neither recolored nor repaired.
    """
    final = np.median(np.stack([_resize(frame) for frame in frames[-3:]]), axis=0).astype(np.uint8)
    colors = _node_interior_colors(final, replay.source)
    fade = _node_tint_fade_metrics(frames, replay, colors)
    fade_metrics = {"video_node_tint_fade": fade, "video_node_tint_fade_correct": fade["valid"]}
    if replay.task_name == "bipartite_color_completion":
        from data.graph_theory.core.rendering import RED_FILL, BLUE_FILL
        readings = {}
        expected_colors = {str(n["label"]): n["bipartite_color"]
                           for n in replay.source["nodes"] if n.get("bipartite_color")}
        expected_colors.update(replay.solution["node_color"])
        for label, expected in expected_colors.items():
            color = colors[str(label)]
            distances = {name: float(np.linalg.norm(color - fill))
                         for name, fill in (("RED", RED_FILL), ("BLUE", BLUE_FILL))}
            observed = min(distances, key=distances.get)
            readings[str(label)] = dict(expected=expected, observed=observed,
                                        distance=distances[observed], rgb=color.tolist())
        valid = all(v["observed"] == v["expected"] and v["distance"] <= 30 for v in readings.values())
        return replay, {**fade_metrics, "video_node_tint_semantics_correct": valid, "video_node_tint_readings": readings}

    from data.graph_theory.core.rendering import NODE_FILL
    components = replay.solution["components"]
    tints, spreads = [], []
    pair_readings = []
    for component in components:
        values = np.array([colors[str(label)] for label in component])
        tint = np.median(values, axis=0)
        tints.append(tint)
        spreads.append(float(np.max(np.abs(values - tint))))
        labs = cv2.cvtColor((values / 255.).astype(np.float32)[None], cv2.COLOR_RGB2LAB)[0]
        for left in range(len(component)):
            for right in range(left + 1, len(component)):
                distance = float(np.linalg.norm(labs[left] - labs[right]))
                pair_readings.append({
                    "nodes": [str(component[left]), str(component[right])],
                    "rgb": [values[left].tolist(), values[right].tolist()],
                    "delta_e": distance,
                    "same_tint": distance <= _COMPONENT_TINT_MAX_PAIR_DELTA_E,
                })
    separation = min((float(np.max(np.abs(a - b))) for i, a in enumerate(tints) for b in tints[i+1:]), default=255.)
    visible = all(float(np.max(np.abs(tint - NODE_FILL))) >= 30 and float(tint.min()) >= 100 for tint in tints)
    valid = (visible and max(spreads, default=255.) <= 20 and separation >= 28
             and all(pair["same_tint"] for pair in pair_readings))
    observed_replay = UnifiedGraphReplay(replay.task_name, replay.source,
        dict(replay.solution, component_tints=[tuple(map(int, np.rint(c))) for c in tints]))
    return observed_replay, {**fade_metrics, "video_node_tint_semantics_correct": bool(valid),
                            "video_component_tints": [c.tolist() for c in tints],
                            "video_component_max_color_spread": max(spreads, default=255.),
                            "video_component_node_color_pairs": pair_readings,
                            "video_component_max_pair_delta_e": max((pair["delta_e"] for pair in pair_readings), default=0.),
                            "video_component_pair_delta_e_bound": _COMPONENT_TINT_MAX_PAIR_DELTA_E,
                            "video_component_min_color_separation": separation,
                            "video_component_tints_visible": bool(visible)}


def _verify_video(path: Optional[Path], replay: UnifiedGraphReplay) -> dict[str, Any]:
    if path is None:
        return {
            "whole_video_loaded": False,
            "whole_video_process_correct": 0.0,
            "whole_video_error": "candidate video is missing",
        }
    try:
        frames, fps = _read_video(path)
        if replay.task_name in ANSWER_RING_TASKS:
            masks, observed = _observe_answer_rings(frames, replay)
            if observed["video_answer_ring_set_correct"]:
                with use_answer_ring_masks(masks):
                    metrics = _video_metrics(frames, fps, replay, ring_masks=masks)
            else:
                metrics = _video_metrics(frames, fps, replay)
                metrics["whole_video_process_correct"] = 0.
                metrics["video_final_state_correct"] = 0.
            metrics.update(observed)
            metrics["video_answer_ring_shape_observed"] = True
            return metrics
        if replay.task_name in {"connected_component_count", "bipartite_color_completion"}:
            observed_replay, tint_metrics = _observe_node_tints(frames, replay)
            if not tint_metrics["video_node_tint_semantics_correct"]:
                # Invalid or unchanged white fills are a failed observation,
                # not a replacement source program with zero visual change.
                observed_replay = replay
            metrics = _video_metrics(frames, fps, observed_replay, tint_semantics=tint_metrics)
            metrics.update(tint_metrics)
            return metrics
        return _video_metrics(frames, fps, replay)
    except Exception as exc:
        return {
            "whole_video_loaded": False,
            "whole_video_process_correct": 0.0,
            "whole_video_error": str(exc),
        }


def _video_metrics(frames: Sequence[np.ndarray], fps: float, replay: UnifiedGraphReplay, *, ring_masks=None, tint_semantics=None) -> dict[str, Any]:
    progress_bank, bank = _progress_bank(replay)
    initial = bank[0]
    dynamic = _dynamic_mask(initial, bank[1:])
    if not np.any(dynamic):
        raise ValueError("source replay has no visible graph-state change")
    static = ~dynamic
    if ring_masks is not None:
        # Static comparisons smooth RGB with a 3x3 kernel. Between crowded
        # answer rings that kernel can carry legal orange contour pixels into
        # a narrow static gap. Keep the usual two-pixel contour allowance out
        # of the scene gate; ring identity/fade and fixed labels are checked
        # independently, including all their original pixels.
        static = ~cv2.dilate(dynamic.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    observed: list[Optional[float]] = []
    dynamic_errors: list[float] = []
    full_errors: list[float] = []
    static_errors: list[float] = []
    static_region_seen: list[float] = []
    bank_float = [expected.astype(np.float32) for expected in bank]
    initial_float = initial.astype(np.float32)
    for raw in frames:
        frame = _resize(raw, initial.shape[:2]).astype(np.float32)
        regional, complete = _frame_replay_errors(frame, bank_float, dynamic)
        # Quantized local opacity and overlays hidden beneath fixed node
        # disks can make adjacent replay ticks pixel-identical at a hold
        # boundary.  Anchor exact source-like frames to 0 and exact
        # terminal-like frames to 1; only interior ties prefer the later
        # coordinate.  Candidate CoT/reference media are not involved.
        if regional[0] <= 0.0005:
            best = 0
        elif regional[-1] <= 0.0005:
            best = len(bank) - 1
        else:
            best = min(
                range(len(bank)),
                key=lambda index: (regional[index], complete[index], -progress_bank[index]),
            )
        dynamic_errors.append(regional[best])
        full_errors.append(complete[best])
        static_errors.append(float(np.mean(np.abs(frame - initial_float)[static])) / 255.0 if np.any(static) else 0.0)
        # A mean over the whole static mask dilutes a painted patch to
        # nothing; the same statistic per region does not (section 11.8).
        static_region_seen.append(
            static_region_errors(frame, initial_float, static)[0] if np.any(static) else 0.0
        )
        observed.append(progress_bank[best] if regional[best] <= 0.075 and complete[best] <= 0.045 else None)

    recognized = [value for value in observed if value is not None]
    recognition = len(recognized) / len(frames)
    first = [value for value in observed[: min(5, len(observed))] if value is not None]
    initial_hold = len(first) >= 3 and max(first[:3]) <= 0.05
    # Completion is a readable legal endpoint held still, not a nearest
    # canonical opacity coordinate. A thinner/lighter orange stroke can have
    # correct semantics while its best canonical tick remains below .95.
    from data.common.verifiers.keyframes import visual_delta
    observed_semantics = ring_masks is not None or tint_semantics is not None
    final_hold = bool(len(frames) >= 3
        and all(_terminal_image_semantics(frame, replay, observed_answer_set=observed_semantics)["terminal_image_correct"]
                for frame in frames[-3:])
        and all(float(visual_delta(_resize(frame), _resize(frames[-1]))[dynamic].mean()) / 255. <= .015
                for frame in frames[-3:]))
    if tint_semantics is not None:
        final_hold &= tint_semantics["video_node_tint_semantics_correct"]
    transitions = [(left, right) for left, right in zip(observed, observed[1:]) if left is not None and right is not None]
    regressions = [left - right for left, right in transitions if right + 0.045 < left]
    jumps = [right - left for left, right in transitions if right > left]
    max_jump = max(jumps, default=0.0)
    oversized_jumps = [
        (left, right)
        for left, right in transitions
        if right - left > 0.24
    ]
    interior = {round(value * 8) for value in recognized if 0.08 < value < 0.92}
    checkpoints = {
        f"{value:.4f}": any(abs(observed_value - value) <= 0.09 for observed_value in recognized)
        for value in replay.semantic_checkpoints[1:-1]
    }
    dimensions_valid = all(frame.shape[:2] == (512, 512) for frame in frames)
    duration = len(frames) / fps if fps > 0 else None
    media_valid = bool(
        dimensions_valid
        and abs(fps - 16.0) <= 0.2
        and len(frames) >= 5
        and (len(frames) - 1) % 4 == 0
        and len(frames) <= 157
        and duration is not None
        and duration <= 10.05
    )
    max_static_region_error = max(static_region_seen, default=0.0)
    static_preserved = (
        max(static_errors, default=1.0) <= 0.04
        and max_static_region_error <= STATIC_REGION_FLOOR
    )
    local_fade_required = replay.task_name in _FIXED_FOOTPRINT_LOCAL_FADE_TASKS
    local_fade_metrics = _fixed_footprint_local_fade_metrics(frames, replay)
    local_fade_followed = bool(local_fade_metrics["valid"])
    # Opacity-only effects must visibly fade rather than appear in one cut.
    # The stricter footprint fit remains diagnostic: codec noise changes
    # individual edge pixels without changing the readable opacity phase.
    semantic_fade_followed = bool(
        not local_fade_required or all(
            phase.get("partial_frame_count", 0) > 0
            and phase.get("opacity_monotonic", 0)
            for phase in local_fade_metrics["phases"].values()
        )
    )
    if tint_semantics is not None:
        semantic_fade_followed &= tint_semantics["video_node_tint_fade_correct"]
    ring_fade = None
    if ring_masks is not None:
        ring_fade = _answer_ring_fade_metrics(frames,replay,ring_masks)
        evidence_followed = all(phase.get("partial_frame_count",0) > 0 and phase.get("opacity_monotonic",0)
                                for name,phase in local_fade_metrics["phases"].items() if name != "answer_overlay")
        semantic_fade_followed = bool(evidence_followed and ring_fade["valid"])
    max_dynamic_replay_error = max(dynamic_errors, default=1.0)
    badge_fade = (_degree_badge_fade_metrics(frames, replay)
                  if replay.task_name == "degree_missing_value" else None)
    task_owned_render_progress_valid = badge_fade is None or badge_fade["valid"]
    terminal_metrics = _terminal_image_semantics(frames[-1], replay,
        observed_answer_set=ring_masks is not None or tint_semantics is not None)
    if tint_semantics is not None and not tint_semantics["video_node_tint_semantics_correct"]:
        terminal_metrics["terminal_image_correct"] = 0.
    labels = _fixed_label_metrics(frames, initial, replay.source)
    process_correct = bool(media_valid and recognition >= .80
                           and initial_hold and final_hold
                           and not oversized_jumps and len(interior) >= 3
                           and task_owned_render_progress_valid
                           and not regressions and all(checkpoints.values())
                           and semantic_fade_followed
                           and static_preserved and labels["fixed_node_labels_preserved"]
                           and terminal_metrics["terminal_image_correct"])
    return {
        **labels,
        "whole_video_loaded": True,
        "whole_video_error": None,
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_duration_seconds": duration,
        "video_media_valid": float(media_valid),
        "video_recognized_fraction": recognition,
        "video_max_dynamic_replay_error": max_dynamic_replay_error,
        "video_max_full_replay_error": max(full_errors, default=1.0),
        "video_initial_hold": float(initial_hold),
        "video_final_hold": float(final_hold),
        "video_progress_monotonic": float(not regressions),
        "video_max_progress_jump": max_jump,
        "video_oversized_jump_count": len(oversized_jumps),
        "video_continuous_process": float(not oversized_jumps and len(interior) >= 3),
        "video_semantic_checkpoint_hits": checkpoints,
        "video_semantic_checkpoints_covered": float(all(checkpoints.values())),
        "video_static_preservation_error": max(static_errors, default=1.0),
        "video_static_region_max_error": max_static_region_error,
        "video_static_region_bound": STATIC_REGION_FLOOR,
        "video_static_preserved": float(static_preserved),
        "video_fixed_footprint_local_fade_required": local_fade_required,
        "video_fixed_footprint_local_fade_followed": float(local_fade_followed),
        "video_semantic_opacity_fade_followed": float(semantic_fade_followed),
        "video_answer_ring_fade": ring_fade,
        "video_fixed_footprint_local_fade_phases": local_fade_metrics["phases"],
        "video_fade_projection_spread_limit": _FADE_PROJECTION_SPREAD_LIMIT,
        "video_fade_footprint_support_limit": _FADE_FOOTPRINT_SUPPORT_LIMIT,
        "video_fade_full_mask_fit_limit": _FADE_FULL_MASK_FIT_LIMIT,
        "video_fade_component_sync_limit": _FADE_COMPONENT_SYNC_LIMIT,
        "video_degree_badge_fade": badge_fade,
        "video_task_owned_render_progress_valid": float(task_owned_render_progress_valid),
        "video_undeclared_full_frame_change_rejected": float(static_preserved),
        "video_final_state_correct": terminal_metrics["terminal_image_correct"],
        **{f"video_{key}": value for key, value in terminal_metrics.items()
           if key.startswith("degree_badge_")},
        "video_reference_pixels_used": False,
        "video_saved_keyframes_used": False,
        "whole_video_process_correct": float(process_correct),
    }


def _trace_edge_sample_points(replay: UnifiedGraphReplay) -> list[tuple[tuple[str, str], list[tuple[float, float]]]]:
    """Evenly spaced sample points along every source edge's interior band.

    Points cover the middle 76% of each trimmed segment: near a shared node
    the dilated stroke of an adjacent committed edge can bleed onto another
    edge's first samples inside a narrow wedge, so endpoint-adjacent samples
    would spuriously report growth on an untouched edge.  Terminal-state
    verification separately covers the full edge footprint.
    """

    source = replay.source
    positions = {str(node["label"]): node["position"] for node in source["nodes"]}
    sampled: list[tuple[tuple[str, str], list[tuple[float, float]]]] = []
    count = 24
    for edge in source["edges"]:
        left, right = str(edge["u"]), str(edge["v"])
        x0, y0 = map(float, positions[left])
        x1, y1 = map(float, positions[right])
        length = max(1e-9, math.hypot(x1 - x0, y1 - y0))
        trim = min(float(NODE_RADIUS), max(0.0, length / 2.0 - 1.0))
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        ax, ay = x0 + ux * trim, y0 + uy * trim
        bx, by = x1 - ux * trim, y1 - uy * trim
        points = [
            (
                ax + (bx - ax) * (0.12 + 0.76 * (index + 0.5) / count),
                ay + (by - ay) * (0.12 + 0.76 * (index + 0.5) / count),
            )
            for index in range(count)
        ]
        sampled.append((edge_key(left, right), points))
    return sampled


def _trace_frame_coverages(
    frames: Sequence[np.ndarray],
    sampled: Sequence[tuple[tuple[str, str], list[tuple[float, float]]]],
) -> list[dict[tuple[str, str], float]]:
    """Per-frame orange coverage fraction for every source edge."""

    target = np.array(ORANGE, dtype=np.float32)
    kernel = np.ones((7, 7), np.uint8)
    per_frame: list[dict[tuple[str, str], float]] = []
    for raw in frames:
        frame = _resize(raw, (512, 512)).astype(np.float32)
        distance = np.sqrt(np.sum((frame - target) ** 2, axis=2))
        mask = (distance <= _TRACE_ORANGE_DISTANCE_LIMIT).astype(np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        coverages: dict[tuple[str, str], float] = {}
        for key, points in sampled:
            hits = 0
            for x, y in points:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < dilated.shape[0] and 0 <= xi < dilated.shape[1] and dilated[yi, xi]:
                    hits += 1
            coverages[key] = hits / len(points)
        per_frame.append(coverages)
    return per_frame


def _trace_order_legal(groups: Sequence[Sequence[tuple[str, str]]], start: str, goal: str) -> bool:
    """Whether one commit order can realize a legal start-to-goal Euler trail."""

    def extend(group: list[tuple[str, str]], node: str, index: int) -> bool:
        if not group:
            return walk(index + 1, node)
        for position, edge in enumerate(group):
            rest = group[:position] + group[position + 1 :]
            left, right = edge
            if left == node and extend(rest, right, index):
                return True
            if right == node and extend(rest, left, index):
                return True
        return False

    def walk(index: int, node: str) -> bool:
        if index == len(groups):
            return node == goal
        return extend(list(groups[index]), node, index)

    return walk(0, start)


def _verify_trace_video(path: Optional[Path], replay: UnifiedGraphReplay) -> dict[str, Any]:
    """Whole-video gate for the multi-certificate Euler-trail task.

    Any legal trail is acceptable, so this gate never compares against one
    canonical trail.  It reads the candidate's orange stroke directly and
    requires: a blank initial hold, every source edge committed exactly once
    in a monotone single-edge-growing cadence, a commit order that forms a
    legal start-to-goal Euler trail, a fully committed final hold, preserved
    static regions, and the shared source-rendered terminal state.
    """

    if path is None:
        return {
            "whole_video_loaded": False,
            "whole_video_process_correct": 0.0,
            "whole_video_error": "candidate video is missing",
        }
    try:
        frames, fps = _read_video(path)
        dimensions_valid = all(frame.shape[:2] == (512, 512) for frame in frames)
        duration = len(frames) / fps if fps > 0 else None
        media_valid = bool(
            dimensions_valid
            and abs(fps - 16.0) <= 0.2
            and len(frames) >= 5
            and (len(frames) - 1) % 4 == 0
            and len(frames) <= 157
            and duration is not None
            and duration <= 10.05
        )
        source = replay.source
        roles = source["roles"]
        all_edges = [edge_key(edge["u"], edge["v"]) for edge in source["edges"]]
        per_frame = _trace_frame_coverages(frames, _trace_edge_sample_points(replay))

        initial_blank = len(frames) >= 5 and all(
            sum(coverages.values()) <= 0.02 for coverages in per_frame[:3]
        )
        committed_counts = [
            sum(1 for edge in all_edges if coverages[edge] >= _TRACE_COMMIT_COVERAGE)
            for coverages in per_frame
        ]
        final_full = len(frames) >= 5 and all(count == len(all_edges) for count in committed_counts[-3:])

        monotone = True
        for edge in all_edges:
            series = [coverages[edge] for coverages in per_frame]
            greatest = 0.0
            for right in series:
                if right + _TRACE_COVERAGE_REGRESSION < greatest:
                    monotone = False
                    break
                greatest = max(greatest, right)
            committed_index = next(
                (index for index, value in enumerate(series) if value >= _TRACE_COMMIT_COVERAGE),
                None,
            )
            if committed_index is not None and any(
                value < _TRACE_COMMIT_COVERAGE - _TRACE_COVERAGE_REGRESSION
                for value in series[committed_index:]
            ):
                monotone = False
            if not monotone:
                break

        partial_frames = sum(
            1
            for coverages in per_frame
            if any(_TRACE_PARTIAL_LOW <= coverages[edge] < _TRACE_COMMIT_COVERAGE for edge in all_edges)
        )
        edge_growth = {
            f"{edge[0]}-{edge[1]}": any(
                _TRACE_PARTIAL_LOW <= coverage[edge] < .95 for coverage in per_frame
            )
            for edge in all_edges
        }
        single_growth = all(
            sum(1 for edge in all_edges if _TRACE_PARTIAL_LOW <= coverages[edge] < _TRACE_GROWING_HIGH) <= 1
            for coverages in per_frame
        )

        groups: list[tuple[int, list[tuple[str, str]]]] = []
        committed: set[tuple[str, str]] = set()
        for index, coverages in enumerate(per_frame):
            group = [
                edge for edge in all_edges
                if edge not in committed and coverages[edge] >= _TRACE_COMMIT_COVERAGE
            ]
            if group:
                groups.append((index, group))
                committed.update(group)
        group_sizes_ok = all(len(group) <= _TRACE_MAX_GROUP_SIZE for _, group in groups)
        span_ok = bool(groups) and (groups[-1][0] - groups[0][0] + 1) >= _TRACE_MIN_SPAN_FRACTION * len(frames)
        order_legal = (
            _trace_order_legal([group for _, group in groups], str(roles["start"]), str(roles["goal"]))
            if committed == set(all_edges)
            else False
        )

        initial = _rgb(replay.render_input())
        terminal = _rgb(replay.render_progress(1.0))
        dynamic = _dynamic_mask(initial, [terminal])
        static = ~dynamic
        static_errors = []
        static_region_seen = []
        initial_float = initial.astype(np.float32)
        for raw in frames:
            frame = _resize(raw, initial.shape[:2]).astype(np.float32)
            static_errors.append(
                float(np.mean(np.abs(frame - initial_float)[static])) / 255.0 if np.any(static) else 0.0
            )
            static_region_seen.append(
                static_region_errors(frame, initial_float, static)[0] if np.any(static) else 0.0
            )
        max_static_region_error = max(static_region_seen, default=0.0)
        static_preserved = (
            max(static_errors, default=1.0) <= 0.04
            and max_static_region_error <= STATIC_REGION_FLOOR
        )
        terminal_metrics = _terminal_image_semantics(frames[-1], replay)
        labels = _fixed_label_metrics(frames, initial, replay.source)

        process_correct = bool(
            media_valid
            and initial_blank
            and final_full
            and monotone
            and partial_frames >= _TRACE_MIN_PARTIAL_FRAMES
            and all(edge_growth.values())
            and single_growth
            and group_sizes_ok
            and span_ok
            and order_legal
            and static_preserved
            and labels["fixed_node_labels_preserved"]
            and terminal_metrics["terminal_image_correct"]
        )
        return {
            **labels,
            "whole_video_loaded": True,
            "whole_video_error": None,
            "video_frame_count": len(frames),
            "video_fps": fps,
            "video_duration_seconds": duration,
            "video_media_valid": float(media_valid),
            "trace_initial_blank": float(initial_blank),
            "trace_final_full_commit": float(final_full),
            "trace_coverage_monotonic": float(monotone),
            "trace_partial_frame_count": partial_frames,
            "trace_each_edge_grows": edge_growth,
            "trace_single_edge_growth": float(single_growth),
            "trace_commit_group_sizes_valid": float(group_sizes_ok),
            "trace_commit_span_valid": float(span_ok),
            "trace_euler_order_legal": float(order_legal),
            "video_static_preservation_error": max(static_errors, default=1.0),
            "video_static_region_max_error": max_static_region_error,
            "video_static_region_bound": STATIC_REGION_FLOOR,
            "video_static_preserved": float(static_preserved),
            "video_final_state_correct": terminal_metrics["terminal_image_correct"],
            "video_reference_pixels_used": False,
            "video_saved_keyframes_used": False,
            "whole_video_process_correct": float(process_correct),
        }
    except Exception as exc:
        return {
            "whole_video_loaded": False,
            "whole_video_process_correct": 0.0,
            "whole_video_error": str(exc),
        }


def verify_unified_graph(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: str = "ti2v",
) -> list[EvalRecord]:
    """Verify the TI2V full-video delivery for unified labelled-graph tasks."""

    selected = str(modality or "").strip().lower()
    if selected not in _MODALITIES:
        raise ValueError(f"unsupported unified Graph Theory modality: {modality!r}")
    records: list[EvalRecord] = []
    for sample in samples:
        video_path = resolve_candidate_video(pred_root, sample.id, sample.split)
        try:
            replay = replay_unified_graph(sample.task_name, sample.semantic_input)
            replay_error = None
        except Exception as exc:
            replay = None
            replay_error = str(exc)

        video_metrics = (
            (_verify_trace_video(video_path, replay) if sample.task_name == "trace_all_edges_once" else _verify_video(video_path, replay))
            if replay is not None else
            {"whole_video_loaded": False, "whole_video_process_correct": 0.0, "whole_video_error": replay_error}
        )
        video_available = video_path is not None
        binary = bool(video_available and video_metrics["whole_video_process_correct"])
        prediction_asset = video_path.as_posix() if video_path else None
        error = replay_error
        if error is None and not binary and not video_available:
            error = "TI2V candidate video is missing"

        records.append(EvalRecord(
            sample_id=sample.id,
            task_family=sample.task_family,
            task_name=sample.task_name,
            split=sample.split,
            input_asset=sample.primary_input_image,
            prediction_asset=prediction_asset,
            offline_metrics={
                **video_metrics,
                "selected_modality": selected,
                "source_replay_variant": "unified_labelled_graph_source_v2" if replay is not None else None,
                "source_replay_error": replay_error,
                "ground_truth_used": False,
                "reference_artifact_used": False,
                "candidate_cot_inspected": False,
                "raw_reward_vector": {
                    "whole_video_process_correct": float(video_metrics["whole_video_process_correct"]) if video_available else 0.0,
                },
                "reward_availability_mask": {
                    "whole_video_process_correct": True,
                },
                "binary_correct": int(binary),
            },
            offline_pass=binary,
            error=error,
        ))
    order = {sample.id: index for index, sample in enumerate(samples)}
    records.sort(key=lambda record: order.get(record.sample_id, len(order)))
    return records


__all__ = ["verify_unified_graph"]
