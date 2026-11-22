"""Final-delivery verifier for all Spatial 3D tasks.

Every branch replays the sample forward from its visible source semantics via
``data.spatial_3d.core.source_replay`` and the task's own renderer.  No branch
reads ground truth, saved final matrices, authored reasoning, solution pixels,
or a reference video.  Candidate CoT is never inspected: the verifier judges
the whole delivered video and nothing else.

Two whole-video models cover the family:

* motion replays — a continuous fold/rotation process matched against a
  progress bank with hold, monotonicity, continuity, and checkpoint gates;
* staged replays — an ordered chain of discrete complete states in which every
  transition must be a synchronized element-local opacity fade (a visible
  partial state, complete footprint, static surroundings), never a
  hard cut or a full-frame dissolve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence


import cv2
import numpy as np
from PIL import Image

from data.common.progress_alignment import detoned, read_process_order
from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.verifiers.keyframes import visual_delta, visual_features
from data.spatial_3d.core.source_replay import Spatial3DReplay, replay_spatial_3d


_MODALITIES = frozenset({"ti2v"})

# ``Spatial3DScenePattern._render_video`` spends five frames holding the input
# and five holding the terminal state, so a motion reference video of ``N``
# frames advances the process by ``1 / (N - 10)`` per remaining frame.
_REFERENCE_NON_ADVANCING_FRAMES = 10

# Preserved-region gates.  A mean over the whole preserved region
# is a compensating aggregate: a solid foreign block moves it by only
# (block area / region area), so a 100x100 mutation reads 0.045 against a 0.04
# ceiling and passes.  The gates below are local, so their reading does not
# shrink as the behaving part of the region grows.

# Guard radius, measured on genuine reference media (all six family tasks x
# seeds 3/11/42/137 = 24 vp9 reference videos): the only renderer whose
# preserved region moves at all under compression is cube_net_fold, whose
# motion-compensation bleed reaches 11 px out of the source's own changed union
# (peak deviation 224/255 at 4 px, 173/255 at 10 px, nothing at 12 px); every
# other task is bit-exact even at 4 px.  Guarding the union by 16 px - the next
# whole 8-px vp9 transform block above the measured 11 - leaves the checked
# region bit-exact on all 24 videos.
_STATIC_GUARD_PX = 16
# One vp9 largest-transform block: the smallest area over which the codec's own
# error is organized, hence the finest window whose mean is still a codec
# statistic rather than a single-pixel outlier (genuine single pixels reach
# 224/255 next to motion, so a per-pixel max is not usable).
_STATIC_WINDOW_PX = 32
# Ceilings are placed against the worst noise measured at HALF the adopted guard
# (8 px: window mean 0.0399, cluster 32 px), so an unseen encode with ~2x the
# measured bleed still passes.  The weakest deliberate mutation probed - a solid
# 20x20 block, the smallest size this family cares about - reads 0.7534 and
# 400 px, i.e. 15x and 8x above these ceilings.
_STATIC_LOCAL_CEILING = 0.05
_STATIC_CLUSTER_LEVEL = 0.20
_STATIC_CLUSTER_PIXELS = 48


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


def _raw_changed(initial: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Pixels the source itself moves between two of its own renders."""

    return np.max(np.abs(initial.astype(np.int16) - other.astype(np.int16)), axis=2) > 10


def _changed_mask(initial: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    mask = _raw_changed(initial, terminal)
    if np.any(mask):
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    return mask


def _preserved_mask(raw_changed: np.ndarray) -> np.ndarray:
    """The region the source keeps constant, minus the codec's bleed band.

    ``raw_changed`` is the source's own undilated changed union; guarding it by
    ``_STATIC_GUARD_PX`` keeps the codec's motion-compensation bleed around legal
    motion from vetoing a legal process.
    """

    guard = np.ones((2 * _STATIC_GUARD_PX + 1, 2 * _STATIC_GUARD_PX + 1), np.uint8)
    return ~cv2.dilate(raw_changed.astype(np.uint8), guard, iterations=1).astype(bool)


def _preserved_gates(
    candidate: np.ndarray,
    source: np.ndarray,
    preserved: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    """Hard-gate the preserved region locally, per window and per cluster.

    Everything outside the guarded band must reproduce ``source``.  Two
    independent readings hard-gate it: the worst ``_STATIC_WINDOW_PX`` window
    mean catches broad low-amplitude repaints, and the largest cluster of pixels
    past ``_STATIC_CLUSTER_LEVEL`` catches compact high-contrast ones.  Neither
    is diluted by the size of the part of the region that is behaving.
    """

    if not np.any(preserved):
        return {
            f"{prefix}_preserved_area_fraction": 0.0,
            f"{prefix}_static_local_error": 0.0,
            f"{prefix}_static_mutation_cluster_px": 0,
            f"{prefix}_static_local_gates_pass": 1.0,
        }
    deviation = np.max(visual_delta(candidate, source), axis=2) / 255.0
    deviation = np.where(preserved, deviation, 0.0)
    weight = preserved.astype(np.float32)
    window = (_STATIC_WINDOW_PX, _STATIC_WINDOW_PX)
    total = cv2.boxFilter(deviation * weight, -1, window, normalize=False, borderType=cv2.BORDER_ISOLATED)
    count = cv2.boxFilter(weight, -1, window, normalize=False, borderType=cv2.BORDER_ISOLATED)
    # Only windows that are mostly inside the preserved region are scored, so a
    # window clipped by the guard band cannot manufacture a reading.
    scored = count >= 0.5 * _STATIC_WINDOW_PX * _STATIC_WINDOW_PX
    local = float((total[scored] / count[scored]).max()) if np.any(scored) else 0.0
    hot = (deviation >= _STATIC_CLUSTER_LEVEL).astype(np.uint8)
    found, _labels, stats, _centroids = cv2.connectedComponentsWithStats(hot, connectivity=8)
    cluster = int(stats[1:, cv2.CC_STAT_AREA].max()) if found > 1 else 0
    gates_pass = local <= _STATIC_LOCAL_CEILING and cluster <= _STATIC_CLUSTER_PIXELS
    return {
        f"{prefix}_preserved_area_fraction": float(preserved.mean()),
        f"{prefix}_static_local_error": local,
        f"{prefix}_static_mutation_cluster_px": cluster,
        f"{prefix}_static_local_gates_pass": float(gates_pass),
    }


def _terminal_frame(replay: Spatial3DReplay) -> np.ndarray:
    if replay.motion:
        assert replay.render_progress is not None
        return _rgb(replay.render_progress(1.0))
    assert replay.render_stage is not None
    return _rgb(replay.render_stage(replay.stage_count - 1))


def _initial_frame(replay: Spatial3DReplay) -> np.ndarray:
    if replay.motion:
        assert replay.render_progress is not None
        return _rgb(replay.render_progress(0.0))
    assert replay.render_stage is not None
    return _rgb(replay.render_stage(0))


def _member_region_gates(candidate: np.ndarray, terminal: np.ndarray, replay: Spatial3DReplay) -> dict[str, Any]:
    """Hard-gate every source-declared member region individually."""

    regions = tuple(replay.member_regions or ())
    if not regions:
        return {"terminal_member_gates_pass": 1.0, "terminal_member_gate_count": 0}
    height, width = terminal.shape[:2]
    observed_delta = visual_delta(candidate, terminal)
    errors: dict[str, float] = {}
    for name, box in regions:
        x0 = max(0, int(np.floor(float(box[0]))))
        y0 = max(0, int(np.floor(float(box[1]))))
        x1 = min(width, int(np.ceil(float(box[2]))))
        y1 = min(height, int(np.ceil(float(box[3]))))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"member region {name!r} is empty")
        delta = observed_delta[y0:y1, x0:x1]
        errors[str(name)] = float(np.mean(delta)) / 255.0
    worst = max(errors, key=errors.__getitem__)
    gates_pass = all(value <= 0.055 for value in errors.values())
    return {
        "terminal_member_gates_pass": float(gates_pass),
        "terminal_member_gate_count": len(errors),
        "terminal_member_region_errors": errors,
        "terminal_member_worst_region": worst,
        "terminal_member_worst_region_error": errors[worst],
    }


def _panel_geometry_metrics(candidate, terminal, replay):
    """Read the flat section's shape and proportions within its source panel.

    Absolute pixel scale and the thin 3D cut outline are not the flat shape's
    aspect ratio. Read rigid-rotation-invariant proportions and silhouette
    moments, with a four-pixel span allowance for the two raster edges.
    """
    panel = next((box for name, box in replay.member_regions if name == "panel_shape"), None)
    if panel is None:
        return {"terminal_panel_evaluated": False, "terminal_panel_geometry_correct": True}
    x0, y0, x1, y1 = map(int, panel)
    def shape(image):
        crop = image[y0:y1, x0:x1].astype(np.int16)
        orange = ((crop[:, :, 0] - crop[:, :, 1] >= 10)
                  & (crop[:, :, 1] - crop[:, :, 2] >= 8) & (crop[:, :, 0] >= 120))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(orange.astype(np.uint8), 8)
        if count <= 1:
            return None
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        _x, _y, _width, _height, area = stats[label]
        if area < 24:
            return None
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        height, width = sorted(map(float, cv2.minAreaRect(contour)[1]))
        fill = float(area) / max(1., width*height)
        extras = int(orange.sum()) - int(area)
        return width, height, contour, fill, extras <= max(12, .01*int(area))
    observed, expected = shape(candidate), shape(terminal)
    if observed is None or expected is None:
        return {"terminal_panel_evaluated": True, "terminal_panel_geometry_correct": False}
    width, height, contour, fill, one_shape = observed
    expected_ratio = expected[0] / expected[1]
    ratio_ok = bool(height > 4 and width > 4
                    and (width-4)/(height+4) <= expected_ratio <= (width+4)/(height-4))
    shape_error = float(cv2.matchShapes(contour, expected[2], cv2.CONTOURS_MATCH_I1, 0.))
    fill_error = abs(fill - expected[3])
    return {"terminal_panel_evaluated": True, "terminal_panel_observed_size": [width, height],
            "terminal_panel_expected_size": [expected[0], expected[1]],
            "terminal_panel_proportions_correct": ratio_ok, "terminal_panel_shape_error": shape_error,
            "terminal_panel_fill_error": fill_error,
            "terminal_panel_geometry_correct": bool(one_shape and ratio_ok and shape_error <= .05 and fill_error <= .08)}


def _image_semantics(candidate: np.ndarray, replay: Spatial3DReplay, *,
                     preserved_source: Optional[np.ndarray] = None) -> dict[str, Any]:
    initial = _initial_frame(replay)
    static_source = initial if preserved_source is None else preserved_source
    terminal = _terminal_frame(replay)
    candidate = _resize(candidate, terminal.shape[:2])
    delta = visual_delta(candidate, terminal)
    full_error = float(np.mean(delta)) / 255.0
    raw_changed = _raw_changed(initial, terminal)
    changed = _changed_mask(initial, terminal)
    changed_error = float(np.mean(delta[changed])) / 255.0 if np.any(changed) else full_error
    static = ~changed
    source_delta = visual_delta(candidate, static_source)
    static_error = float(np.mean(source_delta[static])) / 255.0 if np.any(static) else full_error
    changed_present = bool(np.any(changed) and changed_error <= 0.055)
    local_metrics = _preserved_gates(candidate, static_source, _preserved_mask(raw_changed), "terminal_image")
    static_preserved = bool(
        static_error <= 0.035 and local_metrics["terminal_image_static_local_gates_pass"]
    )
    member_metrics = _member_region_gates(candidate, terminal, replay)
    panel_metrics = _panel_geometry_metrics(candidate, terminal, replay)
    correct = bool(
        full_error <= 0.035
        and changed_present
        and static_preserved
        and member_metrics["terminal_member_gates_pass"]
        and panel_metrics["terminal_panel_geometry_correct"]
    )
    return {
        "terminal_image_correct": float(correct),
        "terminal_image_source_render_full_error": full_error,
        "terminal_image_semantic_region_error": changed_error,
        "terminal_image_static_region_error": static_error,
        "terminal_image_semantic_change_present": float(changed_present),
        "terminal_image_static_preserved": float(static_preserved),
        **local_metrics,
        **member_metrics,
        **panel_metrics,
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


def _media_validity(frames: list[np.ndarray], fps: float) -> tuple[bool, Optional[float]]:
    duration = len(frames) / fps if fps > 0 else None
    valid = bool(
        all(frame.shape[:2] == (512, 512) for frame in frames)
        and abs(fps - 16.0) <= 0.2
        and len(frames) >= 5
        and (len(frames) - 1) % 4 == 0
        and duration is not None
        and duration > 0
    )
    return valid, duration


def _fade_observation(frame: np.ndarray, base: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Project one frame onto the element-local fade between two states."""

    base_float = visual_features(base)
    target_float = visual_features(target)
    frame_float = visual_features(frame)
    direction = target_float - base_float
    energy_map = np.sum(direction * direction, axis=2)
    magnitude = np.max(np.abs(direction), axis=2)
    # Include the filled face as well as its darker border. The native orange
    # panel fill differs from white by less than 100 levels; keeping only the
    # border makes a two-pixel contour displacement look like a missing shape.
    active = (magnitude >= 40.0) & (energy_map >= 1.0)
    if int(active.sum()) < 20:
        return {"recognized": False, "pass": False, "error": "transition footprint is empty"}
    total_energy = float(np.sum(direction[active] ** 2))
    frame_delta = frame_float - base_float
    opacity = float(np.sum(frame_delta[active] * direction[active]) / total_energy)
    projection_map = np.sum(frame_delta * direction, axis=2) / np.maximum(energy_map, 1.0)
    projection = projection_map[active]
    q10, q90 = np.quantile(projection, (0.10, 0.90))
    support_floor = max(0.025, 0.30 * max(0.0, opacity))
    # Read the neighbouring contour before restricting to the source strong
    # core. Restricting first discards a displaced, complete thin curve merely
    # because its ink lands on the source's antialiased fringe. Two native
    # pixels retain the complete observed slice outline without filling a
    # missing edge or a missing local patch.
    supported = ((projection_map >= support_floor) & (magnitude >= 10.0)).astype(np.uint8)
    supported = cv2.dilate(supported, np.ones((5, 5), np.uint8)).astype(bool) & active
    window_count = cv2.boxFilter(active.astype(np.float32), -1, (32, 32), normalize=False)
    window_support = cv2.boxFilter(supported.astype(np.float32), -1, (32, 32), normalize=False)
    geometric_support = float(np.mean(supported[active]))
    local_windows = (window_count >= 20) & active
    local_support = (float(np.min(window_support[local_windows] / window_count[local_windows]))
                     if np.any(local_windows) else geometric_support)
    residual = frame_delta[active] - opacity * direction[active]
    linear_fit = float(np.sqrt(np.sum(residual * residual) / total_energy))
    expected = base_float + direction * max(0.0, min(1.0, opacity))
    footprint_error = float(np.mean(np.abs(frame_float - expected)[active])) / 255.0
    footprint = cv2.dilate(active.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    outside = ~footprint
    static_error = (
        float(np.mean(np.abs(frame_float - base_float)[outside])) / 255.0 if np.any(outside) else 0.0
    )
    full_error = float(np.mean(np.abs(frame_float - expected))) / 255.0
    recognized = bool(
        -0.08 <= opacity <= 1.08
        and footprint_error <= 0.065
        and static_error <= 0.040
        and full_error <= 0.045
    )
    partial = bool(recognized and 0.08 <= opacity <= 0.92)
    complete_footprint = bool(
        not partial
        or (
            geometric_support >= 0.90
            and local_support >= 0.70
            and static_error <= 0.040
        )
    )
    return {
        "recognized": recognized,
        "partial": partial,
        "pass": complete_footprint,
        "opacity": opacity,
        "q10_q90_spread": float(q90 - q10),
        "footprint_support": float(np.mean(projection >= support_floor)),
        "geometric_footprint_support": geometric_support,
        "minimum_local_footprint_support": local_support,
        "linear_fit_error": linear_fit,
        "footprint_replay_error": footprint_error,
        "outside_footprint_static_error": static_error,
        "full_replay_error": full_error,
    }


def _verify_staged_video(frames: list[np.ndarray], fps: float, replay: Spatial3DReplay, *,
                         preserved_source: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Verify an ordered chain of complete states joined by element-local fades."""

    assert replay.render_stage is not None and replay.stage_count >= 2
    states = [_rgb(replay.render_stage(index)) for index in range(replay.stage_count)]
    static_source = states[0] if preserved_source is None else preserved_source
    shape = states[0].shape[:2]
    # Every state agrees with the first outside the union of its own commits, so
    # that guarded complement is the region every frame must reproduce.  Reading
    # a frame by its whole-frame mean error alone would let a foreign block ride
    # along inside one interior frame: a 40x40 repaint moves the mean by
    # 0.004 against a 0.020 recognition band.
    raw_changed = np.zeros(shape, dtype=bool)
    for state in states[1:]:
        raw_changed |= _raw_changed(states[0], state)
    preserved = _preserved_mask(raw_changed)
    reading_regions = [cv2.dilate(_raw_changed(before, after).astype(np.uint8),
                       np.ones((5, 5), np.uint8)).astype(bool) for before, after in zip(states, states[1:])]
    def reading_error(candidate, expected):
        delta = visual_delta(candidate, expected)
        return max(float(delta[region].mean()) / 255. for region in reading_regions if region.any())
    scores: list[Optional[float]] = []
    exact_hits: set[int] = set()
    transition_partials: dict[int, set[int]] = {index: set() for index in range(len(states) - 1)}
    partial_failures = 0
    unrecognized = 0
    static_locals: list[float] = []
    static_clusters: list[int] = []
    for frame in frames:
        resized = _resize(frame, shape)
        frame_gates = _preserved_gates(resized, static_source, preserved, "frame")
        static_locals.append(float(frame_gates["frame_static_local_error"]))
        static_clusters.append(int(frame_gates["frame_static_mutation_cluster_px"]))
        exact_errors = [
            reading_error(resized, state)
            for state in states
        ]
        best_exact = int(np.argmin(exact_errors))
        best_fade: Optional[tuple[int, dict[str, Any]]] = None
        for index in range(len(states) - 1):
            observation = _fade_observation(resized, states[index], states[index + 1])
            if not observation.get("recognized"):
                continue
            alpha = max(0., min(1., observation["opacity"]))
            expected = states[index].astype(float) * (1 - alpha) + states[index + 1] * alpha
            observation["semantic_region_error"] = reading_error(resized, expected)
            if best_fade is None or observation["semantic_region_error"] < best_fade[1]["semantic_region_error"]:
                best_fade = (index, observation)
        # A low-opacity fade frame is also a numerically close match to the
        # previous complete state. Interpretations compete over every declared
        # changed region separately, so a large panel cannot dilute a small
        # section trace. Endpoint opacities collapse back to complete states.
        fade_error = float(best_fade[1]["semantic_region_error"]) if best_fade is not None else float("inf")
        if best_fade is None or exact_errors[best_exact] <= fade_error:
            if exact_errors[best_exact] <= 0.065:
                scores.append(float(best_exact))
                exact_hits.add(best_exact)
            else:
                scores.append(None)
                unrecognized += 1
            continue
        index, observation = best_fade
        opacity = max(0.0, min(1.0, float(observation["opacity"])))
        # The public contract requires the whole element to appear together.
        # Check the footprint even when readable contrast is high enough to
        # classify this frame as the committed endpoint below.
        if observation.get("partial"):
            transition_partials[index].add(int(round(opacity * 8)))
            if not observation.get("pass"):
                partial_failures += 1
        if opacity <= 0.05:
            scores.append(float(index))
            exact_hits.add(index)
        elif opacity >= 0.80 or (scores and scores[-1] == float(index + 1) and opacity >= 0.73):
            # Keep the same 0.07 observation tolerance at the commit boundary;
            # tiny contrast jitter must not turn .80 -> .79 into 1 -> .79.
            scores.append(float(index + 1))
            exact_hits.add(index + 1)
        else:
            scores.append(index + opacity)

    recognized = [value for value in scores if value is not None]
    recognition = len(recognized) / len(frames)
    initial_hold = bool(len(scores) >= 2 and scores[0] == 0.0 and scores[1] == 0.0)
    final_value = float(len(states) - 1)
    final_hold = bool(
        len(scores) >= 2 and scores[-1] == final_value and scores[-2] == final_value
    )
    greatest_progress = np.maximum.accumulate(recognized)
    monotonic = all(value + 0.07 >= greatest
                    for value, greatest in zip(recognized, greatest_progress))
    all_states_held = exact_hits == set(range(len(states)))
    used_transitions = {
        index for index, levels in transition_partials.items() if levels
    }
    # At least one observable partial proves a fade rather than a hard cut.
    # The prompt does not prescribe a sampled opacity count or fade velocity.
    fades_progressive = all(transition_partials[index] for index in range(len(states) - 1))
    media_valid, duration = _media_validity(frames, fps)
    terminal_metrics = _image_semantics(frames[-1], replay, preserved_source=preserved_source)
    worst_local = max(static_locals, default=1.0)
    worst_cluster = max(static_clusters, default=10 ** 6)
    static_preserved = bool(
        worst_local <= _STATIC_LOCAL_CEILING and worst_cluster <= _STATIC_CLUSTER_PIXELS
    )
    correct = bool(
        media_valid
        and unrecognized == 0
        and monotonic
        and all_states_held
        and initial_hold
        and final_hold
        and fades_progressive
        and partial_failures == 0
        and static_preserved
        and terminal_metrics["terminal_image_correct"]
    )
    return {
        "whole_video_loaded": True,
        "whole_video_error": None,
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_duration_seconds": duration,
        "video_media_valid": float(media_valid),
        "video_staged_process": True,
        "video_stage_count": len(states),
        "video_stage_progress": scores,
        "video_recognized_fraction": recognition,
        "video_unrecognized_frames": unrecognized,
        "video_initial_hold": float(initial_hold),
        "video_final_hold": float(final_hold),
        "video_progress_monotonic": float(monotonic),
        "video_all_states_held": float(all_states_held),
        "video_transitions_with_partials": len(used_transitions),
        "video_fades_progressive": float(fades_progressive),
        "video_partial_footprint_failures": partial_failures,
        "video_static_local_error": worst_local,
        "video_static_mutation_cluster_px": worst_cluster,
        "video_static_preserved": float(static_preserved),
        "video_full_frame_endpoint_dissolve_used": False,
        "video_final_state_correct": terminal_metrics["terminal_image_correct"],
        **{f"video_{key}": value for key, value in terminal_metrics.items()},
        "video_reference_pixels_used": False,
        "video_saved_keyframes_used": False,
        "whole_video_process_correct": float(correct),
    }


def _verify_motion_video(frames: list[np.ndarray], fps: float, replay: Spatial3DReplay) -> dict[str, Any]:
    """Verify one continuous motion process against a source progress bank.

    Committed pauses render pixel-identical frames over a progress interval,
    so consecutive identical bank entries are grouped and every match yields a
    progress interval for appearance checks. Process order and checkpoint
    coverage use the whole-sequence alignment so recurring poses cannot create
    false jumps or reversals in a legal program.
    """

    assert replay.render_progress is not None
    # Sample the bank at least as finely as the artifact advances.  A video of
    # N frames cannot move by less than one N-th of the process per frame, so a
    # bank of N steps puts every emitted pose within half a step of a bank
    # entry.  A coarser bank leaves most frames sitting *between* two entries,
    # and a solid whose silhouette repeats can then land nearer some distant
    # pose than either neighbour - which reads as a jump and a reversal in an
    # otherwise legal process.
    steps = max(32, len(frames))
    progress = {index / steps for index in range(steps + 1)}
    progress.update(float(value) for value in replay.semantic_checkpoints)
    ordered = sorted(max(0.0, min(1.0, value)) for value in progress)
    bank = [_rgb(replay.render_progress(value)) for value in ordered]
    groups: list[tuple[float, float, np.ndarray]] = []
    for value, image in zip(ordered, bank):
        if groups and np.array_equal(groups[-1][2], image):
            groups[-1] = (groups[-1][0], value, groups[-1][2])
        else:
            groups.append((value, value, image))
    initial = bank[0]
    raw_dynamic = np.zeros(initial.shape[:2], dtype=bool)
    for expected in bank[1:]:
        raw_dynamic |= _raw_changed(initial, expected)
    if not np.any(raw_dynamic):
        raise ValueError("source replay has no visible dynamic region")
    dynamic = cv2.dilate(raw_dynamic.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    static = ~dynamic
    preserved = _preserved_mask(raw_dynamic)

    observed: list[Optional[tuple[float, float]]] = []
    dynamic_errors: list[float] = []
    full_errors: list[float] = []
    static_errors: list[float] = []
    static_locals: list[float] = []
    static_clusters: list[int] = []
    initial_float = initial.astype(np.float32)
    running = 0.0
    # Comparing every frame against a bank this fine costs a lot of full-image
    # passes, so score the dynamic region for all states at once and pay for the
    # whole-frame score only where it can matter: a state whose dynamic error is
    # already worse than the best by more than the codec noise band can neither
    # win nor enter the near-tie band below.
    banked_dynamic = np.stack([group[2][dynamic].astype(np.int16) for group in groups])
    # a uniform tone shift carries no positional information, but it can
    # tip the near-tie band - distances to inky states move more than distances
    # to bright ones - and one tipped tie at the end reads as a failed terminal
    # hold.  Classify detoned frames; a genuine candidate's offset is zero, so
    # nothing changes for it.
    resized = [_resize(raw, initial.shape[:2]) for raw in frames]
    detoned_frames = detoned(resized, initial, static if np.any(static) else None)
    for frame_float in detoned_frames:
        frame = frame_float
        dynamic_scores = (
            np.abs(banked_dynamic.astype(np.float32) - frame[dynamic]).mean(axis=(1, 2)) / 255.0
        )
        candidates = [int(index) for index in np.flatnonzero(dynamic_scores <= dynamic_scores.min() + 0.004)]
        errors = {index: float(dynamic_scores[index]) for index in candidates}
        full = {
            index: float(np.mean(np.abs(frame_float - groups[index][2].astype(np.float32)))) / 255.0
            for index in candidates
        }
        best = min(candidates, key=lambda index: (errors[index], full[index]))
        static_errors.append(float(np.mean(np.abs(frame_float - initial_float)[static])) / 255.0 if np.any(static) else 0.0)
        frame_gates = _preserved_gates(frame, initial, preserved, "frame")
        static_locals.append(float(frame_gates["frame_static_local_error"]))
        static_clusters.append(int(frame_gates["frame_static_mutation_cluster_px"]))
        # A small moving element inside a large dynamic region - or fast motion
        # between two bank states - leaves several states within codec noise of
        # each other, so the single nearest state is not a reliable reading.
        # Take the whole near-tie band as the states this frame could be, drop
        # the ones that contradict the progress already established, and read
        # the frame as the surviving state nearest that progress for appearance
        # scoring. The sequence alignment below determines process order.
        feasible = [
            index
            for index in candidates
            if errors[index] <= errors[best] + 0.004 and full[index] <= full[best] + 0.004
        ]
        forward = [index for index in feasible if groups[index][1] + 0.045 >= running]
        if not forward:
            forward = feasible

        def _distance(index: int) -> float:
            lo, hi = groups[index][0], groups[index][1]
            if lo <= running <= hi:
                return 0.0
            return lo - running if lo > running else running - hi

        best = min(
            forward,
            key=lambda index: (
                _distance(index) > 0.05,
                _distance(index) if _distance(index) > 0.05 else 0.0,
                errors[index],
                groups[index][0],
            ),
        )
        dynamic_errors.append(errors[best])
        full_errors.append(full[best])
        if errors[best] <= 0.075 and full[best] <= 0.045:
            interval = (groups[best][0], groups[best][1])
            running = min(interval[1], max(running, interval[0]))
            observed.append(interval)
        else:
            observed.append(None)

    recognized = [interval for interval in observed if interval is not None]
    recognition = len(recognized) / len(frames)
    first = [interval for interval in observed[: min(4, len(observed))] if interval is not None]
    last = [interval for interval in observed[max(0, len(observed) - 4):] if interval is not None]
    initial_hold = len(first) >= 2 and max(interval[0] for interval in first[:2]) <= 0.05
    final_hold = len(last) >= 2 and min(interval[1] for interval in last[-2:]) >= 0.95
    # A fixed progress-jump ceiling only notices a deletion whose gap happens to
    # exceed that constant, so the same deletion is caught in a fast phase and
    # missed in a slow one, and a reversal spread over frames that each advance
    # less than the backward tolerance above is invisible however much process it
    # spans.  The shared bounded-step reading judges both against the source's own
    # advance at the place they sit, which is what makes the verdict independent
    # of where the splice was made.
    # The reading runs on the ungrouped bank: grouping equal states onto their
    # first progress value is what the per-frame reading above needs to keep a
    # plateau frame from aliasing to an arbitrary end of it, but it also leaves
    # the progress axis with a gap the width of every plateau, and a step gate
    # cannot tell that gap from a deleted commit.  The sequence reading needs no
    # such help: its motion term places a frame inside a plateau from how fast
    # the picture is moving there, not from how it looks.
    ordering = read_process_order(
        resized,
        bank,
        ordered,
        dynamic,
        non_advancing_frames=_REFERENCE_NON_ADVANCING_FRAMES,
        bank_quantum=1.0 / steps,
    )
    continuous = ordering.ordered
    assigned = ordering.progress
    interior = {round(value * 8) for value in assigned if 0.08 < value < 0.92}
    intermediate_visible = len(interior) >= 3
    checkpoint_hits = {
        f"{checkpoint:.4f}": any(
            abs(value - checkpoint) <= 0.09 for value in assigned
        )
        for checkpoint in replay.semantic_checkpoints[1:-1]
    }
    checkpoints_covered = all(checkpoint_hits.values())
    # Per-frame local hard gates, never an aggregate over frames or over area:
    # one mutated frame cannot be averaged away by the frames that behave, and
    # one mutated window cannot be averaged away by the region that behaves.
    worst_local = max(static_locals, default=1.0)
    worst_cluster = max(static_clusters, default=10 ** 6)
    static_preserved = bool(
        max(static_errors, default=1.0) <= 0.04
        and worst_local <= _STATIC_LOCAL_CEILING
        and worst_cluster <= _STATIC_CLUSTER_PIXELS
    )
    media_valid, duration = _media_validity(frames, fps)
    terminal_metrics = _image_semantics(frames[-1], replay)
    correct = bool(
        media_valid
        and recognition >= 0.82
        and continuous
        and intermediate_visible
        and checkpoints_covered
        and static_preserved
        and terminal_metrics["terminal_image_correct"]
    )
    return {
        "whole_video_loaded": True,
        "whole_video_error": None,
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_duration_seconds": duration,
        "video_media_valid": float(media_valid),
        "video_staged_process": False,
        "video_recognized_fraction": recognition,
        "video_max_dynamic_replay_error": max(dynamic_errors, default=1.0),
        "video_max_full_replay_error": max(full_errors, default=1.0),
        "video_initial_hold": float(initial_hold),
        "video_final_hold": float(final_hold),
        "video_max_progress_jump": ordering.max_step,
        "video_reading_max_step": ordering.max_step,
        "video_reading_distinct_states": ordering.distinct_states,
        "video_reading_step_ratio": ordering.step_ratio,
        "video_reading_reference_pace": ordering.reference_pace,
        "video_reading_comparison_kernel": ordering.comparison_kernel,
        "video_reading_step_violations": ordering.step_break_count,
        "video_frame_pace_violations": ordering.pace_break_count,
        "video_worst_frame_pace_ratio": ordering.worst_pace_ratio,
        "video_frame_pace_floor_violations": ordering.pace_floor_break_count,
        "video_worst_frame_pace_floor_ratio": ordering.worst_pace_floor_ratio,
        "video_bounded_step_order_pass": float(ordering.ordered),
        "video_continuous_process": float(continuous and intermediate_visible),
        "video_semantic_checkpoint_hits": checkpoint_hits,
        "video_semantic_checkpoints_covered": float(checkpoints_covered),
        "video_static_preservation_error": max(static_errors, default=1.0),
        "video_static_local_error": worst_local,
        "video_static_mutation_cluster_px": worst_cluster,
        "video_static_preserved": float(static_preserved),
        "video_full_frame_endpoint_dissolve_used": False,
        "video_final_state_correct": terminal_metrics["terminal_image_correct"],
        "video_reference_pixels_used": False,
        "video_saved_keyframes_used": False,
        "whole_video_process_correct": float(correct),
    }


def _verify_video(path: Optional[Path], replay: Spatial3DReplay, *,
                  source_image: Optional[Path] = None) -> dict[str, Any]:
    if path is None:
        return {
            "whole_video_loaded": False,
            "whole_video_process_correct": 0.0,
            "whole_video_error": "candidate video is missing",
        }
    try:
        frames, fps = _read_video(path)
        if replay.motion:
            return _verify_motion_video(frames, fps, replay)
        if source_image is not None:
            with Image.open(source_image) as image:
                source = _rgb(image)
            return _verify_staged_video(frames, fps, replay, preserved_source=source)
        return _verify_staged_video(frames, fps, replay)
    except Exception as exc:
        return {
            "whole_video_loaded": False,
            "whole_video_process_correct": 0.0,
            "whole_video_error": str(exc),
        }


def verify_spatial_3d(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: str = "ti2v",
) -> list[EvalRecord]:
    """Verify exactly the declared Spatial 3D final-delivery modality."""

    selected = str(modality or "").strip().lower()
    if selected not in _MODALITIES:
        raise ValueError(f"unsupported Spatial 3D modality: {modality!r}")
    records: list[EvalRecord] = []
    for sample in samples:
        video_path = resolve_candidate_video(pred_root, sample.id, sample.split)
        try:
            replay = replay_spatial_3d(sample, modality=selected)
            replay_error = None
        except Exception as exc:
            replay = None
            replay_error = str(exc)

        if replay is not None and sample.task_name == "slice_cross_section":
            # The visible solid, plane frame and panel frame are givens from
            # this question, not from a later renderer's line-join choices.
            video_metrics = _verify_video(video_path, replay, source_image=Path(sample.primary_input_image))
        else:
            video_metrics = (
                _verify_video(video_path, replay)
                if replay is not None
                else {"whole_video_loaded": False, "whole_video_process_correct": 0.0, "whole_video_error": replay_error}
            )
        video_available = video_path is not None
        binary = bool(video_available and video_metrics["whole_video_process_correct"])
        prediction_asset = video_path.as_posix() if video_path else None
        availability = {
            "whole_video_process_correct": True,
        }
        raw_rewards = {
            "whole_video_process_correct": float(video_metrics["whole_video_process_correct"]) if video_available else 0.0,
        }
        error = replay_error
        if error is None and not binary:
            if not video_available:
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
                "source_replay_variant": replay.source_variant if replay is not None else None,
                "source_replay_error": replay_error,
                "ground_truth_used": False,
                "reference_artifact_used": False,
                "candidate_cot_inspected": False,
                "raw_reward_vector": raw_rewards,
                "reward_availability_mask": availability,
                "binary_correct": int(binary),
            },
            offline_pass=binary,
            error=error,
        ))
    order = {sample.id: index for index, sample in enumerate(samples)}
    records.sort(key=lambda record: order.get(record.sample_id, len(order)))
    return records


__all__ = ["verify_spatial_3d"]
