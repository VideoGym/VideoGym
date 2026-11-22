"""Source-replay verifier for the Geometry answer-hypothesis task cluster.

Every routed task delivers its problem scene and video as native full-canvas
renders. The verifier consumes only ``CanonicalSample.semantic_input`` and the
supplied problem image, never consulting ``ground_truth`` or reference media.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.verifiers.keyframes import keyframe_indices, visual_delta, visual_features
from data.common.verifiers.region_gates import static_region_errors, STATIC_REGION_FLOOR
from data.common.verifiers.transitions import complete_footprint_projection, uniform_fade_metrics
from data.geometry.core.terminal_scene import observed_added_ink
from data.geometry.core.card_stroke_specs import (
    SERIAL_STROKE_CARD_TASKS,
    geometry_card_strokes,
)
from data.geometry.core.stroke_sequence import (
    EndpointStroke,
    _pixel_terminal_projection,
    _stroke_bin_masks,
    analyze_serial_endpoint_strokes,
)


_MODALITIES = frozenset({"ti2v"})
# Every routed task's expected terminal is re-rendered from the independently
# replayed semantic state through the renderer's own scene function, so pixel
# geometry can never drift from the renderer.
_NATIVE_SCENE_TASKS = frozenset(
    {
        "convex_hull_count",
        "count_intersection_points",
        "lattice_points_strict_inside",
        "mirror_image_select",
        "region_count",
        "right_triangle_selection",
        "rotation",
        "symmetry_completion",
        "triangle_inequality_check",
    }
)
@dataclass(frozen=True)
class CardReplay:
    correct_index: int
    correct_label: str
    direct_value: str
    state: Mapping[str, Any]


def _state(sample: CanonicalSample) -> dict[str, Any]:
    envelope = sample.semantic_input
    state = envelope.get("semantic_state") if isinstance(envelope, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("Geometry card sample has no canonical semantic_state")
    return dict(state)


def _unique_option(options: Sequence[Any], target: Any) -> tuple[int, str]:
    matches = [index for index, value in enumerate(options) if value == target]
    if len(options) != 4 or len(matches) != 1:
        raise ValueError("visible hypotheses do not contain one unique source-replayed answer")
    index = matches[0]
    return index, "ABCD"[index]


def _right_triangle_labels(state: Mapping[str, Any]) -> tuple[list[str], int]:
    from data.geometry.right_triangle_selection.generator import _triangle_dot_products

    triangles = state.get("triangles")
    candidates = state.get("candidates")
    if not isinstance(triangles, list) or not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("malformed right-triangle source")
    right = []
    for index, triangle in enumerate(triangles):
        if not isinstance(triangle, Mapping):
            raise ValueError("malformed visible triangle")
        vertices = triangle.get("vertices")
        dots = _triangle_dot_products(vertices)
        if sum(int(value) == 0 for value in dots) == 1:
            right.append(index)
    candidate_sets = [
        sorted(int(value) for value in candidate["highlighted_triangle_indices"])
        for candidate in candidates
    ]
    index, _label = _unique_option(candidate_sets, right)
    labels = [str(triangles[value].get("label") or value + 1) for value in right]
    return labels, index


def _replay_answer(task_name: str, state: Mapping[str, Any]) -> CardReplay:
    options = state.get("options")
    target: Any

    if task_name == "convex_hull_count":
        from data.geometry.convex_hull_count.generator import _convex_hull

        target = len(_convex_hull(state["points"]))
    elif task_name == "count_intersection_points":
        from data.geometry.count_intersection_points.generator import (
            _cluster_intersections,
            _raw_pair_intersections,
        )

        target = len(_cluster_intersections(_raw_pair_intersections(state["lines"])))
    elif task_name == "lattice_points_strict_inside":
        from data.geometry.lattice_points_strict_inside.generator import _lattice_point_sets

        vertices = [tuple(int(value) for value in point) for point in state["verts"]]
        _boundary, interior = _lattice_point_sets(vertices)
        target = len(interior)
    elif task_name == "mirror_image_select":
        from data.geometry.mirror_image_select.generator import _apply_transform

        source = state["source_vertices"]
        expected = [list(point) for point in _apply_transform(source, "mirror_lr")]
        normalized = [
            [[int(value) for value in point] for point in candidate]
            for candidate in state["candidate_vertices"]
        ]
        index, label = _unique_option(normalized, expected)
        return CardReplay(index, label, label, state)
    elif task_name == "region_count":
        from data.geometry.region_count.generator import _replay_region_count

        replay, _metadata_valid = _replay_region_count(dict(state))
        if not replay:
            raise ValueError("circle arrangement cannot be replayed")
        index = int(replay["correct_index"])
        return CardReplay(index, "ABCD"[index], str(replay["correct_count"]), state)
    elif task_name == "right_triangle_selection":
        labels, index = _right_triangle_labels(state)
        return CardReplay(index, "ABCD"[index], "{" + ", ".join(labels) + "}", state)
    elif task_name == "rotation":
        target = str(state["requested_rotation"])
        candidates = [str(value) for value in state["visible_candidate_poses"]]
        index, label = _unique_option(candidates, target)
        return CardReplay(index, label, label, state)
    elif task_name == "symmetry_completion":
        from data.geometry.symmetry_completion.generator import _candidate_flags

        candidates = [candidate["cells"] for candidate in state["candidates"]]
        flags = _candidate_flags(
            state["source_cells"], candidates, int(state["grid_size"]), str(state["axis"])
        )
        index, label = _unique_option(flags, True)
        return CardReplay(index, label, label, state)
    elif task_name == "triangle_inequality_check":
        triples = [[int(value) for value in triple] for triple in state["triples"]]
        flags = [sum(sorted(triple)[:2]) <= max(triple) for triple in triples]
        index, label = _unique_option(flags, True)
        return CardReplay(index, label, label, state)
    else:
        raise ValueError(f"unsupported Geometry card task: {task_name}")

    if not isinstance(options, list):
        raise ValueError("numeric card task has no visible option values")
    normalized = [int(value) for value in options]
    index, label = _unique_option(normalized, int(target))
    return CardReplay(index, label, str(int(target)), state)


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _native_scene_expected(source: np.ndarray, replay: CardReplay, task_name: str) -> np.ndarray:
    """Re-render the expected native terminal from the replayed semantics.

    The answer always comes from the independent semantic replay, never from
    saved metadata; only the pixel template is produced by the renderer's own
    scene function so verifier geometry cannot drift from the renderer.
    """

    state = dict(replay.state)
    if task_name == "triangle_inequality_check":
        from data.geometry.triangle_inequality_check.generator import (
            _comparison_payload,
            render_scene,
        )

        triples = [[int(value) for value in triple] for triple in state["triples"]]
        task = {
            "triples": triples,
            "comparisons": _comparison_payload(triples),
            "invalid_index": int(replay.correct_index),
        }
        image = render_scene(task, reveal_count=4, selection_progress=1.0)
    elif task_name == "count_intersection_points":
        from data.geometry.count_intersection_points.generator import (
            CountIntersectionPointsPattern,
            _raw_pair_intersections,
        )

        raw_records = _raw_pair_intersections(state["lines"])
        task = {"lines": state["lines"], "intersection_records": raw_records}
        renderer = object.__new__(CountIntersectionPointsPattern)
        renderer.image_size = (512, 512)
        image = renderer._render_scene(task, marked_count=len(raw_records))
    elif task_name == "lattice_points_strict_inside":
        from data.geometry.lattice_points_strict_inside.generator import (
            LatticePointsStrictInsidePattern,
            _lattice_point_sets,
        )

        vertices = [tuple(int(value) for value in point) for point in state["verts"]]
        boundary, interior = _lattice_point_sets(vertices)
        task = {
            "verts": [list(point) for point in vertices],
            "boundary_points": boundary,
            "interior_points": interior,
        }
        renderer = object.__new__(LatticePointsStrictInsidePattern)
        renderer.image_size = (512, 512)
        image = renderer._render_scene(task, marked_count=len(interior))
    elif task_name == "convex_hull_count":
        from data.geometry.convex_hull_count.generator import (
            ConvexHullCountPattern,
            _convex_hull,
            _hull_chains,
        )

        hull = _convex_hull(state["points"])
        primary, remaining = _hull_chains(state["points"])
        task = {
            "points": state["points"],
            "hull": hull,
            "hull_count": len(hull),
            "primary_chain": primary,
            "remaining_chain": remaining,
        }
        renderer = object.__new__(ConvexHullCountPattern)
        renderer.image_size = (512, 512)
        image = renderer._render_scene(task, edge_count=len(hull))
    elif task_name == "region_count":
        from data.geometry.region_count.generator import RegionCountPattern, _replay_region_count

        recovered, _metadata_valid = _replay_region_count(state)
        if not recovered:
            raise ValueError("region-count terminal cannot be rebuilt")
        renderer = object.__new__(RegionCountPattern)
        trace = renderer._state_trace(int(recovered["correct_count"]), replay.correct_label)
        image = renderer._render_scene(recovered, trace[-1])
    elif task_name == "right_triangle_selection":
        from data.geometry.right_triangle_selection.generator import (
            RightTriangleSelectionPattern,
            _replay_right_triangle_selection,
        )

        recovered, _metadata_valid = _replay_right_triangle_selection(state)
        if not recovered:
            raise ValueError("right-triangle terminal cannot be rebuilt")
        renderer = object.__new__(RightTriangleSelectionPattern)
        renderer.image_size = (512, 512)
        image = renderer._draw_scene(
            recovered,
            selected_count=len(recovered["right_triangle_indices"]),
        )
    elif task_name == "mirror_image_select":
        from data.geometry.mirror_image_select.generator import MirrorImageSelectPattern

        task = {
            "source": state["source_vertices"],
            "candidates": state["candidate_vertices"],
            "color": tuple(int(value) for value in state["path_color"]),
        }
        renderer = object.__new__(MirrorImageSelectPattern)
        image = renderer._render_scene(
            task,
            show_endpoint_guides=True,
            mirror_progress=float(len(task["source"]) - 1),
        )
    elif task_name == "symmetry_completion":
        from data.geometry.symmetry_completion.generator import (
            SymmetryCompletionPattern,
            _BOARD_BOX,
            _FILLS,
            _expected_completion,
        )

        grid_size = int(state["grid_size"])
        axis = str(state["axis"])
        source_cells = [tuple(int(value) for value in cell) for cell in state["source_cells"]]
        fill = _nearest_palette_color(source, _FILLS, _BOARD_BOX)
        expected = _expected_completion(source_cells, grid_size, axis)
        task = {
            "grid_size": grid_size,
            "axis": axis,
            "source_cells": source_cells,
            "expected_completion_cells": expected,
            "fill_color": fill,
        }
        renderer = object.__new__(SymmetryCompletionPattern)
        image = renderer._render_scene(task, renderer._state_trace(len(expected))[-1])
    elif task_name == "rotation":
        from data.geometry.rotation.generator import (
            RotationPattern,
            _TILE_SIZE,
            _render_geometry,
            _rotate_tile,
        )

        geometry = _render_geometry()
        source_box = geometry["source_tile_box"]
        # The authored source is a square tile resized into the visible
        # workspace.  Restore that authoring resolution before replaying the
        # rotation; rotating the resized workspace crop directly produces a
        # different antialiased silhouette and rejects a valid terminal image.
        source_tile = Image.fromarray(source).crop(
            tuple(int(value) for value in source_box)
        ).resize((_TILE_SIZE, _TILE_SIZE), Image.Resampling.LANCZOS)
        task = {
            "source_tile": source_tile,
            "target_transform": state["requested_rotation"],
            "poses": {
                pose: _rotate_tile(source_tile, pose)
                for pose in ("identity", "rot90_cw", "rot180", "rot90_ccw")
            },
        }
        renderer = object.__new__(RotationPattern)
        image = renderer._render_solution_image(task)
    else:
        raise ValueError(f"{task_name}: no native-scene expectation is registered")
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _nearest_palette_color(
    source: np.ndarray,
    palette: Sequence[Sequence[int]],
    area: Sequence[int],
) -> tuple[int, int, int]:
    left, top, right, bottom = [int(value) for value in area]
    pixels = source[top:bottom, left:right].reshape(-1, 3).astype(np.int16)
    best_color: tuple[int, int, int] | None = None
    best_count = -1
    for raw in palette:
        color = tuple(int(value) for value in raw[:3])
        distance = np.max(np.abs(pixels - np.asarray(color, dtype=np.int16)), axis=1)
        count = int((distance <= 8).sum())
        if count > best_count:
            best_color, best_count = color, count
    if best_color is None or best_count <= 0:
        raise ValueError("visible source color cannot be recovered")
    return best_color


def _expected_terminal(source: np.ndarray, replay: CardReplay, task_name: str) -> np.ndarray:
    return _native_scene_expected(source, replay, task_name)


def _mask_match(source: np.ndarray, expected: np.ndarray, candidate: np.ndarray,
                *, match_fill_colors: bool = True) -> dict[str, Any]:
    if source.shape != (512, 512, 3) or candidate.shape != source.shape or expected.shape != source.shape:
        return {"template_match": False, "source_preserved": False}
    expected_change = np.max(np.abs(expected.astype(np.int16) - source.astype(np.int16)), axis=2) >= 12
    candidate_change = observed_added_ink(candidate, source)
    kernel = np.ones((7, 7), np.uint8)
    expected_dilated = cv2.dilate(expected_change.astype(np.uint8), kernel).astype(bool)
    candidate_dilated = cv2.dilate(candidate_change.astype(np.uint8), kernel).astype(bool)
    recall = float((expected_change & candidate_dilated).sum() / max(1, expected_change.sum()))
    precision = float((candidate_change & expected_dilated).sum() / max(1, candidate_change.sum()))
    static_mutation = float((candidate_change & ~expected_dilated).mean())
    dynamic_region = expected_dilated
    static_region = ~expected_dilated
    dynamic_delta = visual_delta(candidate, expected)
    static_delta = visual_delta(candidate, source)
    dynamic_rmse = float(
        np.sqrt(np.mean(np.square(dynamic_delta[dynamic_region]))) / 255.0
    ) if np.any(dynamic_region) else 1.0
    static_rmse = float(
        np.sqrt(np.mean(np.square(static_delta[static_region]))) / 255.0
    ) if np.any(static_region) else 1.0
    static_local_error = static_region_errors(candidate, source, static_region)[0]
    match = bool(expected_change.any() and recall >= 0.78 and precision >= 0.72)
    match = bool(match and (not match_fill_colors or dynamic_rmse <= 0.095))
    preserved = bool(static_mutation <= 0.012 and static_rmse <= 0.04
                     and static_local_error <= STATIC_REGION_FLOOR)
    return {
        "template_dynamic_recall": recall,
        "template_dynamic_precision": precision,
        "template_static_mutation_fraction": static_mutation,
        "template_dynamic_rmse": dynamic_rmse,
        "template_static_rmse": static_rmse,
        "template_max_static_region_error": static_local_error,
        "template_match": match,
        "source_preserved": preserved,
    }


def _region_boundary_colors(candidate, source):
    """Keep each circle's identity color independently of the face palette."""
    from data.geometry.region_count.generator import _CURVE_COLORS

    readings = []
    for color in _CURVE_COLORS:
        color = np.asarray(color, dtype=np.int16)
        mask = np.max(np.abs(source.astype(np.int16) - color), axis=2) <= 24
        if not mask.any():
            continue
        matching = np.max(np.abs(candidate.astype(np.int16) - color), axis=2) <= 48
        matching = cv2.dilate(matching.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        recall = float(np.mean(matching[mask]))
        readings.append(recall)
    passed = bool(readings and min(readings) >= 0.75)
    return {"region_boundary_color_recall": readings,
            "region_boundary_colors_preserved": passed}, passed


def _verify_terminal_array(
    candidate: np.ndarray,
    source: np.ndarray,
    replay: CardReplay,
    task_name: str,
) -> dict[str, Any]:
    if candidate.shape != (512, 512, 3) or source.shape != candidate.shape:
        return {"terminal_image_correct": 0.0, "terminal_image_media_valid": 0.0}
    changed = np.max(np.abs(candidate.astype(np.int16) - source.astype(np.int16)), axis=2) >= 18
    changed_fraction = float(changed.mean())
    direct_change = 0.00035 <= changed_fraction <= 0.35
    metrics: dict[str, Any] = {
        "terminal_image_media_valid": 1.0,
        "terminal_image_changed_fraction": changed_fraction,
        "terminal_image_direct_change_present": float(direct_change),
    }
    expected = _expected_terminal(source, replay, task_name)
    template = _mask_match(source, expected, candidate,
                           match_fill_colors=task_name != "region_count")
    if task_name == "region_count":
        boundary_metrics, boundary_pass = _region_boundary_colors(candidate, source)
        metrics.update(boundary_metrics)
        template["source_preserved"] &= boundary_pass
    metrics.update(template)
    objects_pass = True
    if task_name in {"count_intersection_points", "lattice_points_strict_inside",
                     "right_triangle_selection", "region_count"}:
        object_metrics, objects_pass = _required_answer_objects(candidate, source, expected,
                                                               task_name=task_name)
        metrics.update(object_metrics)
    correct = bool(direct_change and template["template_match"] and template["source_preserved"]
                   and objects_pass)
    metrics["terminal_image_correct"] = float(correct)
    metrics["reference_artifact_used"] = False
    metrics["saved_answer_used"] = False
    return metrics


def _answer_object_masks(source: np.ndarray, expected: np.ndarray):
    """Source-owned footprints, separated by unchanged contour pixels."""
    change = np.max(np.abs(expected.astype(np.int16) - source.astype(np.int16)), axis=2) >= 12
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(change.astype(np.uint8), connectivity=8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < 24:
            continue
        mask = labels == label
        interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if interior.sum() >= max(24, mask.sum() * 0.25):
            mask = interior
        yield label, mask


def _required_answer_objects(candidate: np.ndarray, source: np.ndarray, expected: np.ndarray,
                             *, task_name: str) -> tuple[dict[str, Any], bool]:
    """Require every independently replayed marker or bounded face locally.

    These native scenes separate answer objects by unchanged source pixels.
    Their connected components therefore supply one source-owned footprint per
    marker, evidence circle, or bounded face. Candidate pixels never choose the
    footprints, and a large correct object cannot pay for an absent small one.
    """
    observed = visual_features(candidate)
    initial = visual_features(source)
    terminal = visual_features(expected)
    readings = []
    for label, mask in _answer_object_masks(source, expected):
        role_metrics = {}
        if task_name == "region_count":
            # Pastel is a fill role, not a prescribed per-face color. The
            # circle boundaries and their identity colors remain scene-owned.
            support = float(np.mean(np.max(np.abs(observed[mask] - initial[mask]), axis=1) >= 12))
            error_ratio = None
            fill_rgb = np.median(observed[mask], axis=0)
            # A pastel can use any hue: permit a half-white mixture of every
            # RGB color, with one bright channel and an eight-level codec
            # margin. Dark opaque fills cannot substitute for that role.
            pastel = bool(fill_rgb.min() >= 120 and fill_rgb.max() >= 192)
            role_metrics = {"fill_rgb": fill_rgb.tolist(), "pastel_fill": pastel}
            passed = support >= 0.70 and pastel
        else:
            direction = terminal[mask] - initial[mask]
            departure = observed[mask] - initial[mask]
            energy = np.sum(direction * direction, axis=1)
            progress = np.sum(departure * direction, axis=1) / np.maximum(energy, 1.0)
            support = float(np.mean(progress >= 0.35))
            error_ratio = float(np.sqrt(np.sum((observed[mask] - terminal[mask]) ** 2)
                                        / max(1.0, float(np.sum(energy)))))
            passed = bool(support >= 0.70 and error_ratio <= 0.65)
            if task_name == "right_triangle_selection":
                # A black ring projects strongly onto a dark-red endpoint.
                # Read red chroma too, so the color role cannot be replaced
                # by an equally dark achromatic annotation.
                pixels = observed[mask]
                red_fraction = float(np.mean((pixels[:, 0] - pixels[:, 1] >= 30)
                                             & (pixels[:, 0] - pixels[:, 2] >= 15)))
                passed = bool(passed and red_fraction >= 0.50)
        readings.append({"object": label, "pixels": int(mask.sum()), "support": support,
                         "endpoint_error_ratio": error_ratio, **role_metrics, "pass": passed})
    passed = bool(readings and all(row["pass"] for row in readings))
    return {"required_answer_objects": readings, "required_answer_object_count": len(readings),
            "required_answer_objects_pass": passed}, passed


def _region_reveal_metrics(frames, source, expected):
    """Observe the joint face reveal, then retain every filled footprint.

    The source circles' colors carry identity; output pastel hues do not.
    Register opacity to the first complete, stable face set, rather than to a
    later decorative palette. Geometry and per-face presence remain fixed by
    independently replayed source masks throughout the video.
    """
    objects = list(_answer_object_masks(source, expected))
    if not objects:
        return {"video_region_reveal_pass": False, "video_uniform_fade_pass": False}, False
    initial = visual_features(source)
    supports, colors = [], []
    for frame in frames:
        observed = visual_features(frame)
        departure = np.max(np.abs(observed - initial), axis=2)
        supports.append([float(np.mean(departure[mask] >= 12)) for _label, mask in objects])
        colors.append([observed[mask].mean(axis=0) for _label, mask in objects])
    colors = np.asarray(colors)
    first_complete = next((index for index, values in enumerate(supports)
                           if min(values) >= 0.90), None)
    if first_complete is None:
        return {"video_region_reveal_pass": False, "video_uniform_fade_pass": False,
                "video_region_fill_retained": False}, False
    stable_ends = (index for index in range(first_complete, len(frames) - 2)
                   if all(min(supports[step]) >= 0.90 for step in range(index, index + 3))
                   and np.max(np.mean(np.abs(colors[index + 1] - colors[index]), axis=1)) <= 3.0
                   and np.max(np.mean(np.abs(colors[index + 2] - colors[index + 1]), axis=1)) <= 3.0)
    for end in stable_ends:
        anchor_index = end + 2
        anchor = frames[anchor_index]
        progress_by_face, malformed, intermediate, monotone = [], [], [], True
        for label, mask in objects:
            ys, xs = np.where(mask)
            crop = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            readings = [complete_footprint_projection(frame[crop], source[crop], anchor[crop], mask=mask[crop])
                        for frame in frames[:anchor_index + 1]]
            progress = [row["progress"] for row in readings]
            progress_by_face.append(progress)
            intermediate.append(any(0.10 < value < 0.90 for value in progress))
            monotone &= all(right >= left - 0.06 for left, right in zip(progress, progress[1:]))
            malformed.extend((label, index) for index, row in enumerate(readings)
                             if 0.08 < row["progress"] < 0.92
                             and (row["progress_spread"] > 0.35 or row["linear_fit_error"] > 0.25
                                  or row["footprint_support"] < 0.72))
        # Holding a partial opacity does not finish the reveal. Skip only
        # candidates with no observable intermediate opacity on any face;
        # the first observable candidate owns every other process verdict.
        if any(intermediate):
            break
    else:
        return {"video_region_reveal_pass": False, "video_uniform_fade_pass": False,
                "video_region_observable_reveal": False}, False
    progress = np.asarray(progress_by_face).T
    synchronous = bool(np.max(np.ptp(progress, axis=1)) <= 0.35)
    endpoints = bool(np.max(progress[:2]) <= 0.08 and np.min(progress[-2:]) >= 0.85)
    retained = all(min(values) >= 0.70 for values in supports[end:])
    final = visual_features(frames[-1])
    steady = True
    for frame in frames[-3:]:
        difference = np.abs(visual_features(frame) - final)
        steady &= all(float(np.mean(difference[mask])) <= 3.0 for _label, mask in objects)
    passed = bool(all(intermediate) and not malformed and monotone and synchronous
                  and endpoints and retained and steady)
    return {"video_region_reveal_pass": passed, "video_uniform_fade_pass": passed,
            "video_region_first_complete_frame": first_complete,
            "video_region_palette_anchor_frame": anchor_index,
            "video_region_face_progress": progress.tolist(),
            "video_region_malformed_face_frames": malformed,
            "video_region_reveal_synchronous": synchronous,
            "video_region_reveal_monotone": bool(monotone),
            "video_region_fill_retained": retained,
            "video_region_final_hold": steady}, passed


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


def _mirror_serial_sequence(
    frames: Sequence[np.ndarray], source: np.ndarray, replay: CardReplay,
    trace_strokes: Sequence[EndpointStroke],
) -> tuple[dict[str, Any], bool]:
    """Separate cyan/pink ink from source-derived colors, then check both phases.

    The two layers can occupy the same pixels. Their RGB directions distinguish
    a guide crossing from premature pink ink without selecting reference frame
    numbers. Once pink begins, guide persistence is measured on support that
    the legitimate pink construction cannot cover.
    """
    from data.geometry.mirror_image_select.generator import MirrorImageSelectPattern, _expected_render_geometry

    state = replay.state
    task = {"source": state["source_vertices"], "candidates": state["candidate_vertices"],
            "color": tuple(int(value) for value in state["path_color"])}
    renderer = object.__new__(MirrorImageSelectPattern)
    guide_scene = np.asarray(renderer._render_scene(task, show_endpoint_guides=True).convert("RGB"))
    pink_scene = np.asarray(renderer._render_scene(task, mirror_progress=len(task["source"]) - 1).convert("RGB"))
    # Serial masks only consume this source-derived ink support. Cropping its
    # enclosing rectangle preserves every tested pixel while avoiding three
    # extra full-canvas copies of every decoded video frame.
    ys, xs = np.where(np.any(guide_scene != source, axis=2) | np.any(pink_scene != source, axis=2))
    top, bottom, left, right = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    crop = (slice(top, bottom), slice(left, right))
    source, guide_scene, pink_scene = source[crop], guide_scene[crop], pink_scene[crop]
    trace_strokes = [EndpointStroke(stroke.label, tuple((x - left, y - top) for x, y in stroke.points), stroke.width)
                     for stroke in trace_strokes]
    guide_delta = guide_scene.astype(np.float64) - source
    pink_delta = pink_scene.astype(np.float64) - source
    support = np.any(guide_delta != 0, axis=2) | np.any(pink_delta != 0, axis=2)
    directions = np.stack([guide_delta[support], pink_delta[support]], axis=-1)
    inverse = np.linalg.pinv(directions)
    geometry = _expected_render_geometry(task["source"], task["candidates"], task["color"])
    endpoint_masks = []
    for point in (geometry["expected_screen_vertices"][0], geometry["expected_screen_vertices"][-1]):
        mask = np.zeros(source.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(point[0]) - left, int(point[1]) - top),
                   int(round(trace_strokes[0].width + 3)), 1, thickness=-1)
        endpoint_masks.append((mask.astype(bool) & np.any(pink_delta != 0, axis=2))[support])
    endpoint_progress = []
    visible_guides = guide_delta.copy()
    visible_guides[np.any(pink_delta != 0, axis=2)] = 0
    visible_guide_scene = np.rint(source + visible_guides).clip(0, 255).astype(np.uint8)

    def layer(indices, plane, delta):
        direction = delta[support]
        for index in indices:
            observed = frames[index][crop][support].astype(np.float64) - source[support]
            coefficients = np.einsum("nij,nj->ni", inverse, observed)[:, plane].clip(0, 1)
            if plane == 1:
                endpoint_progress.append([float(coefficients[mask].mean()) for mask in endpoint_masks])
            image = source.copy()
            image[support] = np.rint(source[support] + coefficients[:, None] * direction).clip(0, 255).astype(np.uint8)
            yield image

    trace_metrics, trace_valid = analyze_serial_endpoint_strokes(
        layer(range(len(frames)), 1, pink_delta), source, pink_scene, trace_strokes)
    trace_progress = trace_metrics.get("serial_stroke_progress_by_frame", [])
    trace_start = next((index for index, values in enumerate(trace_progress)
                        if max(values + endpoint_progress[index]) > 0.22), None)
    guide_strokes = [EndpointStroke(f"endpoint_guide_{index + 1}",
                                   tuple((float(point[0]) - left, float(point[1]) - top) for point in endpoints), 2.0)
                     for index, endpoints in enumerate(geometry["endpoint_guide_segments"])]
    # Before the actual first pink stroke, every guide pixel remains observable.
    # Later pink may cover crossings/endpoints, so separately retain checks on
    # the source-derived guide support that cannot legitimately be occluded.
    guide_stop = trace_start if trace_start is not None else len(frames)
    guide_metrics, guide_valid = analyze_serial_endpoint_strokes(
        layer(range(guide_stop), 0, guide_delta), source, guide_scene, guide_strokes, pixel_reach=1)
    # A bin wholly covered by legitimate pink has no later guide evidence.
    # Its complete guide was already checked in the full prefix above. Measure
    # the still-observable bins directly, without holding past values by cummax.
    visible_dynamic = np.max(np.abs(visible_guide_scene.astype(np.int16) - source), axis=2) >= 12
    visible_bins = [[mask for mask in _stroke_bin_masks(source.shape[:2], stroke, visible_dynamic) if mask.any()]
                    for stroke in guide_strokes]
    guide_progress = []
    for image in layer(range(len(frames)), 0, visible_guides):
        projection = _pixel_terminal_projection(image, source, visible_guide_scene, pixel_reach=1)
        guide_progress.append([float(np.mean([projection[mask].mean() for mask in masks])) if masks else 1.0
                               for masks in visible_bins])
    complete_frame = next((index for index, values in enumerate(
        guide_metrics.get("serial_stroke_progress_by_frame", [])) if min(values) >= 0.78), None)
    visible_valid = bool(complete_frame is not None and all(min(values) >= 0.65
                         for values in guide_progress[complete_frame:]))
    phase_order = bool(trace_progress and guide_progress and all(
        max(pink + endpoints) <= 0.22 or min(guides) >= 0.65
        for pink, endpoints, guides in zip(trace_progress, endpoint_progress, guide_progress)))
    passed = bool(trace_valid and guide_valid and visible_valid and phase_order)
    metrics = dict(trace_metrics)
    metrics.update({"serial_stroke_check_pass": passed,
                    "mirror_guide_serial_metrics": guide_metrics,
                    "mirror_visible_guide_progress_by_frame": guide_progress,
                    "mirror_visible_guide_bin_counts": [[int(mask.sum()) for mask in masks] for masks in visible_bins],
                    "mirror_guide_complete_frame": complete_frame,
                    "mirror_reflected_endpoint_progress_by_frame": endpoint_progress,
                    "mirror_guide_preservation_valid": visible_valid,
                    "mirror_guides_before_trace_valid": phase_order,
                    "mirror_trace_start_frame": trace_start})
    if not guide_valid:
        metrics["serial_stroke_reason"] = "endpoint guides: " + guide_metrics["serial_stroke_reason"]
    elif not visible_valid:
        metrics["serial_stroke_reason"] = "a completed endpoint guide is erased on still-visible support"
    elif not phase_order:
        metrics["serial_stroke_reason"] = "pink construction begins before both endpoint guides are complete"
    return metrics, passed


def _verify_video(path: Optional[Path], source: np.ndarray, replay: CardReplay, task_name: str) -> dict[str, Any]:
    if path is None:
        return {"whole_video_loaded": False, "whole_video_process_correct": 0.0}
    try:
        frames, fps = _read_video(path)
        media_ok = bool(len(frames) >= 5 and len(frames) % 4 == 1 and len(frames) <= 157
                        and abs(fps - 16) <= .05 and all(f.shape == (512, 512, 3) for f in frames))
        if not media_ok:
            raise ValueError("invalid video dimensions, frame count or FPS")
        indices = keyframe_indices(frames)
        initial_error = float(visual_delta(frames[0], source).mean() / 255)
        source_ok = initial_error <= .045
        terminal = _expected_terminal(source, replay, task_name)
        terminal_metrics = _verify_terminal_array(frames[-1], source, replay, task_name)
        terminal_ok = terminal_metrics["terminal_image_correct"] == 1
        dynamic = np.max(np.abs(terminal.astype(np.int16) - source.astype(np.int16)), axis=2) >= 12
        dynamic = cv2.dilate(dynamic.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        if task_name == "rotation":
            # A moving tile can sweep pixels absent from both endpoint poses.
            # Its public workspace, not the endpoint XOR, defines its extent.
            from data.geometry.rotation.generator import _render_geometry
            left, top, right, bottom = _render_geometry()["target_tile_box"]
            dynamic[top:bottom, left:right] = True
        static_error = max(static_region_errors(frame, source, ~dynamic)[0] for frame in frames)
        static_preserved = static_error <= STATIC_REGION_FLOOR
        boundary_metrics = {}
        if task_name == "region_count":
            boundaries_preserved = all(_region_boundary_colors(frame, source)[1] for frame in frames)
            boundary_metrics["whole_video_region_boundary_colors_preserved"] = boundaries_preserved
            static_preserved = bool(static_preserved and boundaries_preserved)
        serial_metrics, serial_ok = {}, True
        fade_metrics, fade_ok = {}, True
        if task_name == "region_count":
            fade_metrics, fade_ok = _region_reveal_metrics(frames, source, terminal)
        elif task_name == "count_intersection_points":
            from data.geometry.core.intersection_marker_observation import marker_fade_metrics
            fade_metrics, fade_ok = marker_fade_metrics(frames, source, replay.state)
        elif task_name == "lattice_points_strict_inside":
            from data.geometry.core.lattice_marker_observation import marker_fade_metrics
            fade_metrics, fade_ok = marker_fade_metrics(frames, source, replay.state)
        elif task_name in {
            "symmetry_completion", "right_triangle_selection",
        }:
            fade_metrics, fade_ok = uniform_fade_metrics(frames, source, terminal)
        if task_name in SERIAL_STROKE_CARD_TASKS:
            strokes = geometry_card_strokes(task_name, replay.state, correct_index=replay.correct_index)
            if task_name == "mirror_image_select":
                serial_metrics, serial_ok = _mirror_serial_sequence(frames, source, replay, strokes)
            else:
                serial_metrics, serial_ok = analyze_serial_endpoint_strokes(frames, source, terminal, strokes)
                if task_name == "convex_hull_count" and not serial_ok:
                    reverse_strokes = geometry_card_strokes(
                        task_name, replay.state, correct_index=replay.correct_index,
                        reverse_hull=True,
                    )
                    reverse_metrics, reverse_ok = analyze_serial_endpoint_strokes(
                        frames, source, terminal, reverse_strokes)
                    if reverse_ok:
                        serial_metrics, serial_ok = reverse_metrics, reverse_ok
                        serial_metrics["serial_hull_direction"] = "reverse"
        passed = bool(source_ok and terminal_ok and static_preserved and serial_ok and fade_ok)
        return {
            "whole_video_loaded": True, "whole_video_frame_count": len(frames),
            "whole_video_fps": fps, "whole_video_media_valid": float(media_ok),
            "whole_video_source_error": initial_error, "whole_video_source_correct": float(source_ok),
            "video_keyframe_indices": indices, "whole_video_static_preserved": float(static_preserved),
            "whole_video_max_static_region_error": static_error,
            "whole_video_serial_stroke_valid": float(serial_ok),
            "whole_video_final_state_correct": float(terminal_ok),
            "video_final_state_correct": float(terminal_ok),
            "whole_video_process_correct": float(passed), **serial_metrics, **fade_metrics,
            **boundary_metrics,
            **{f"video_{key}": value for key, value in terminal_metrics.items()},
        }
    except Exception as exc:
        return {"whole_video_loaded": False, "whole_video_process_correct": 0.0, "whole_video_error": str(exc)}


def verify_geometry_card_delivery(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: str = "ti2v",
) -> list[EvalRecord]:
    """Verify the TI2V full-video delivery for Geometry card tasks."""

    selected = str(modality or "").strip().lower()
    if selected not in _MODALITIES:
        raise ValueError(f"unsupported Geometry card modality: {modality!r}")
    records: list[EvalRecord] = []
    for sample in samples:
        video_path = resolve_candidate_video(pred_root, sample.id, sample.split)
        try:
            if sample.task_name not in _NATIVE_SCENE_TASKS:
                raise ValueError(f"task {sample.task_name!r} is not a routed Geometry native-scene task")
            replay = _replay_answer(sample.task_name, _state(sample))
            source = _rgb(Path(sample.primary_input_image))
            replay_error = None
        except Exception as exc:
            replay = None
            source = np.empty((0, 0, 3), dtype=np.uint8)
            replay_error = str(exc)

        video_metrics = (
            _verify_video(video_path, source, replay, sample.task_name)
            if replay is not None
            else {"whole_video_loaded": False, "whole_video_process_correct": 0.0, "whole_video_error": replay_error}
        )
        binary = bool(video_path is not None and video_metrics["whole_video_process_correct"])
        records.append(
            EvalRecord(
                sample_id=sample.id,
                task_family=sample.task_family,
                task_name=sample.task_name,
                split=sample.split,
                input_asset=sample.primary_input_image,
                prediction_asset=video_path.as_posix() if video_path is not None else None,
                offline_metrics={
                    **video_metrics,
                    "selected_modality": selected,
                    "source_replay_error": replay_error,
                    "ground_truth_used": False,
                    "reference_artifact_used": False,
                    "candidate_cot_inspected": False,
                    "raw_reward_vector": {
                        "whole_video_process_correct": float(video_metrics.get("whole_video_process_correct", 0.0)) if video_path is not None else 0.0,
                    },
                    "reward_availability_mask": {
                        "whole_video_process_correct": True,
                    },
                    "binary_correct": int(binary),
                },
                offline_pass=binary,
                error=replay_error if replay_error is not None else (None if binary else "formal delivery failed source-replay verification"),
            )
        )
    return records


__all__ = ["CardReplay", "verify_geometry_card_delivery"]
