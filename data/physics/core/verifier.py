"""Full-video verifier for all 12 Physics tasks.

Every task is replayed from visible source semantics via
``data.physics.core.source_replay`` and the task's own renderer without
consulting ground truth, saved final states, or reference media.
"""

from __future__ import annotations


import math
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.region_gates import (
    STATIC_REGION_FLOOR,
    static_region_errors,
)
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.verifiers.keyframes import visual_delta
from data.common.verifiers.numeric_glyphs import numeric_glyph_descriptor, read_numeric_glyph
from data.common.progress_alignment import (
    align_progress,
    at_scale,
    comparison_scale,
    crossed_pair_changes,
    detoned,
    dynamic_probe,
    emission_costs,
    local_pair_change_bounds,
    observed_pair_changes,
    pace_violations,
    reference_pair_changes,
)
from data.physics.core.progress_alignment import (
    backward_rotation_degrees,
    carrier_advance_bound,
    carrier_backward_travel,
    carrier_reading_floor,
    carrier_readings,
    carrier_sampling_noise,
    carrier_settled_points,
    carrier_total_variation,
    profile_rotations,
    reference_advance,
    rotation_profiles,
    rotation_reading_floor,
    step_budget,
)
from data.physics.core.source_replay import PhysicsReplay, replay_physics


_MODALITIES = frozenset({"ti2v"})

# Largest dynamic-region disagreement, in mean absolute channel error over 255,
# at which a frame still counts as a reading of its aligned source state.
# Measured worst aligned disagreement on genuine reference media across the
# seven Physics tasks at ten seeds is 0.029 (gear_train_direction, whose rim
# teeth are the finest structure in the family and therefore the most sensitive
# to the bank's progress quantisation); the ceiling keeps a 2.6x margin for
# codec noise on top of that.
_EMISSION_CEILING = 0.075
# The same reading judged over the whole frame; the static complement dominates
# the average, so the measured worst is an order of magnitude smaller (0.0034).
_FULL_FRAME_CEILING = 0.045
# A declared semantic checkpoint must be shown, not merely straddled: some frame
# has to land within one legal frame advance of it.  For the longer videos in
# the family that advance is tighter than the inherited 0.09 ceiling, and the
# ceiling caps the tolerance for the shortest ones so this can never loosen.
_CHECKPOINT_TOLERANCE_CEILING = 0.09



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


def _dynamic_error(frame: np.ndarray, expected: np.ndarray, dynamic: np.ndarray) -> float:
    """Mean absolute disagreement over the source-derived dynamic region."""

    return float(
        np.abs(frame[dynamic].astype(np.int16) - expected[dynamic].astype(np.int16)).mean()
    ) / 255.0


def _changed_mask(initial: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    mask = np.max(np.abs(initial.astype(np.int16) - terminal.astype(np.int16)), axis=2) > 10
    if np.any(mask):
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    return mask


def _member_region_gates(
    candidate: np.ndarray,
    terminal: np.ndarray,
    replay: PhysicsReplay,
) -> dict[str, Any]:
    """Hard-gate every source-declared member region individually.

    Each named member's candidate pixels must match the source-rendered
    terminal on that member's own evidence box, so a fully wrong domino,
    container level, gear arrow, ball position, or detector state cannot be
    compensated by the aggregate metrics.
    """

    regions = tuple(replay.member_regions or ())
    if not regions:
        return {"terminal_member_gates_pass": 1.0, "terminal_member_gate_count": 0}
    height, width = terminal.shape[:2]
    observed_delta = visual_delta(candidate, terminal)
    errors: dict[str, float] = {}
    for name, box in regions:
        x0 = max(0, int(math.floor(float(box[0]))))
        y0 = max(0, int(math.floor(float(box[1]))))
        x1 = min(width, int(math.ceil(float(box[2]))))
        y1 = min(height, int(math.ceil(float(box[3]))))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"member region {name!r} is empty")
        region_delta = observed_delta[y0:y1, x0:x1]
        errors[str(name)] = float(np.mean(region_delta)) / 255.0
    worst = max(errors, key=errors.__getitem__)
    gates_pass = all(value <= 0.055 for value in errors.values())
    return {
        "terminal_member_gates_pass": float(gates_pass),
        "terminal_member_gate_count": len(errors),
        "terminal_member_region_errors": errors,
        "terminal_member_worst_region": worst,
        "terminal_member_worst_region_error": errors[worst],
    }


def _numeric_fields(candidate: np.ndarray, terminal: np.ndarray, replay: PhysicsReplay) -> dict[str, Any]:
    vocabulary = {}
    for value, patch in replay.numeric_templates:
        descriptor = numeric_glyph_descriptor(_rgb(patch))
        if descriptor is None:
            raise ValueError("source-rendered numeric template is unreadable")
        vocabulary[value] = descriptor
    for _name, (x0, y0, x1, y1), value in replay.numeric_regions:
        descriptor = numeric_glyph_descriptor(terminal[y0:y1, x0:x1])
        if descriptor is None:
            raise ValueError("source-rendered numeric field is unreadable")
        vocabulary[value] = descriptor
    readings = {}
    for name, (x0, y0, x1, y1), expected in replay.numeric_regions:
        value, similarity, margin = read_numeric_glyph(candidate[y0:y1, x0:x1], vocabulary)
        readings[name] = {"expected": expected, "observed": value,
                          "similarity": similarity, "margin": margin, "pass": value == expected}
    return readings


def _clash_marks(candidate: np.ndarray, terminal: np.ndarray) -> dict[str, Any]:
    """Read each red clash cross, allowing two pixels of stroke displacement."""
    def red(frame):
        r, g, b = frame.astype(np.int16).transpose(2, 0, 1)
        return (r > 110) & (r > g + 40) & (r > b + 30)

    observed, expected = red(candidate), red(terminal)
    kernel = np.ones((5, 5), np.uint8)
    nearby = cv2.dilate(observed.astype(np.uint8), kernel).astype(bool)
    allowed = cv2.dilate(expected.astype(np.uint8), kernel).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(expected.astype(np.uint8), 8)
    supports = [float(nearby[labels == index].mean()) for index in range(1, count)]
    shapes = []
    overfills = []
    def chroma(frame):
        r, g, b = frame.astype(np.float32).transpose(2, 0, 1)
        return np.maximum(r - np.maximum(g, b) - 35, 0)
    observed_chroma, expected_chroma = chroma(candidate), chroma(terminal)
    for index in range(1, count):
        x, y, width, height, _area = stats[index]
        crop = np.s_[max(0, y - 4):y + height + 4, max(0, x - 4):x + width + 4]
        required = cv2.GaussianBlur(expected_chroma[crop], (3, 3), .7)
        visible = observed_chroma[crop]
        similarities = []
        shifted_patches = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                shifted = cv2.warpAffine(visible, np.float32([[1, 0, dx], [0, 1, dy]]),
                                         (visible.shape[1], visible.shape[0]))
                shifted_patches.append(shifted)
                shifted = cv2.GaussianBlur(shifted, (3, 3), .7)
                similarities.append(float(np.sum(required * shifted) / max(1e-6, np.linalg.norm(required) * np.linalg.norm(shifted))))
        shapes.append(max(similarities))
        aligned = shifted_patches[int(np.argmax(similarities))]
        local_expected = expected[crop]
        negative = ~cv2.dilate(local_expected.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        inside = np.zeros_like(negative)
        x0, y0 = x - max(0, x - 4), y - max(0, y - 4)
        inside[y0:y0 + height, x0:x0 + width] = True
        negative &= inside
        overfills.append(float((aligned[negative] > max(20., float(aligned.max()) * .3)).mean()) if negative.any() else 0.)
    precision = float(allowed[observed].mean()) if observed.any() else 0.0
    return {"terminal_clash_mark_supports": supports, "terminal_clash_mark_precision": precision,
            "terminal_clash_mark_shape_similarities": shapes,
            "terminal_clash_mark_negative_space_filled": overfills,
            "terminal_clash_mark_pixels": int(observed.sum()),
            "terminal_clash_marks_pass": bool(supports and min(supports) >= .70 and precision >= .80
                                               and min(shapes) >= .83 and max(overfills) <= .20)}


def _image_semantics(candidate: np.ndarray, replay: PhysicsReplay) -> dict[str, Any]:
    initial = _rgb(replay.render_progress(0.0, False))
    terminal = _rgb(replay.render_progress(1.0, True))
    candidate = _resize(candidate, terminal.shape[:2])
    delta = visual_delta(candidate, terminal)
    full_error = float(np.mean(delta)) / 255.0
    changed = _changed_mask(initial, terminal)
    changed_error = float(np.mean(delta[changed])) / 255.0 if np.any(changed) else full_error
    static = ~changed
    source_delta = visual_delta(candidate, initial)
    static_error = float(np.mean(source_delta[static])) / 255.0 if np.any(static) else full_error
    changed_present = bool(np.any(changed) and changed_error <= 0.055)
    clash_metrics = {}
    if replay.source_variant == "gear_train_direction_visible_source_v2" and replay.answer == "JAM":
        clash_metrics = _clash_marks(candidate, terminal)
        changed_present = clash_metrics["terminal_clash_marks_pass"]
    static_preserved = static_error <= 0.035
    member_metrics = _member_region_gates(candidate, terminal, replay)
    numeric_readings = _numeric_fields(candidate, terminal, replay)
    numeric_ok = all(reading["pass"] for reading in numeric_readings.values())
    correct = bool(
        full_error <= 0.035
        and changed_present
        and static_preserved
        and member_metrics["terminal_member_gates_pass"]
        and numeric_ok
    )
    return {
        "terminal_image_correct": float(correct),
        "terminal_image_source_render_full_error": full_error,
        "terminal_image_semantic_region_error": changed_error,
        "terminal_image_static_region_error": static_error,
        "terminal_image_semantic_change_present": float(changed_present),
        "terminal_image_static_preserved": float(static_preserved),
        **member_metrics,
        **clash_metrics,
        "terminal_numeric_fields_pass": float(numeric_ok),
        "terminal_numeric_field_readings": numeric_readings,
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


def _progress_bank(replay: PhysicsReplay, steps: int = 32) -> tuple[list[float], list[np.ndarray]]:
    """Render the source process at ``steps`` progress samples plus checkpoints.

    ``steps`` is chosen by the caller from the candidate's own frame count: a
    video of N frames cannot advance by less than one N-th of the process per
    frame, so a bank of N steps puts every emitted state within half a step of a
    bank entry.  A coarser bank leaves most frames sitting *between* two entries,
    and a scene whose appearance repeats can then land nearer some distant state
    than either neighbour - which reads as a jump and a reversal in an otherwise
    legal process.
    """

    steps = max(32, int(steps))
    progress = {index / steps for index in range(steps + 1)}
    progress.update(float(value) for value in replay.semantic_checkpoints)
    ordered = sorted(max(0.0, min(1.0, value)) for value in progress)
    images = [
        _rgb(
            replay.render_progress(
                value,
                1.0
                if (
                    replay.answer_overlay_mode == "semantic_terminal"
                    and value >= 1.0 - 1e-9
                )
                else 0.0,
            )
        )
        for value in ordered
    ]
    return ordered, images


def _answer_overlay_observation(
    frame: np.ndarray,
    base: np.ndarray,
    terminal: np.ndarray,
) -> dict[str, Any]:
    """Project one decoded frame onto a source-derived answer footprint."""

    base_float = base.astype(np.float32)
    terminal_float = terminal.astype(np.float32)
    frame_float = frame.astype(np.float32)
    direction = terminal_float - base_float
    energy_map = np.sum(direction * direction, axis=2)
    magnitude = np.max(np.abs(direction), axis=2)
    # Weak antialias deltas amplify codec noise in a per-pixel projection and
    # are not needed to establish whether the visible footprint is complete.
    # Every declared Physics answer overlay has a source-derived strong core.
    active = (magnitude >= 100.0) & (energy_map >= 1.0)
    if int(active.sum()) < 20:
        return {
            "recognized": False,
            "pass": False,
            "error": "source-derived answer footprint is empty",
        }
    total_energy = float(np.sum(direction[active] ** 2))
    frame_delta = frame_float - base_float
    opacity = float(np.sum(frame_delta[active] * direction[active]) / total_energy)
    projection_map = np.sum(frame_delta * direction, axis=2) / np.maximum(energy_map, 1.0)
    projection = projection_map[active]
    q10, q90 = np.quantile(projection, (0.10, 0.90))
    support_floor = max(0.025, 0.30 * max(0.0, opacity))
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
    footprint = cv2.dilate(
        active.astype(np.uint8),
        np.ones((5, 5), np.uint8),
        iterations=1,
    ).astype(bool)
    outside = ~footprint
    static_error = (
        float(np.mean(np.abs(frame_float - base_float)[outside])) / 255.0
        if np.any(outside)
        else 0.0
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
        "active_pixel_count": int(active.sum()),
    }


def _rotation_reading_gates(
    frames: Sequence[np.ndarray],
    bank: Sequence[np.ndarray],
    bank_progress: Sequence[float],
    replay: PhysicsReplay,
    *,
    budget: float,
) -> dict[str, Any]:
    """Hard-gate each declared rotating body's own turning.

    Order read out of pixel *distance* is hostage to how far the scene moves per
    frame: too little and a short rewind fits inside the emission ceiling, too
    much and adjacent frames share no ink at all.  A body whose phase the source
    draws escapes that trade, because its rotation is read in degrees the source
    itself declares.  Shrinking the per-frame step shrinks the reading and the
    bound on it by the same factor, so nothing here is bought back by pacing the
    scene differently.

    Each carrier is read frame by frame and gated on both sides:

    * **Backwards.**  The readings telescope into a cumulative phase whose worst
      backwards travel must stay under the reader's own floor.  A gear that
      visibly turns back contradicts the very quantity the answer states, and it
      is caught at any step size, however short the rewind.
    * **Forwards.**  One frame may not turn a body further than the process
      advances in one legal frame - ``budget`` converted into that body's own
      degrees, plus the reader's floor.  This is the same bound the alignment
      places on progress, restated in the units the phase is actually legible in:
      where a mean pixel difference cannot separate a state from its neighbour
      three states away, ten degrees of rotation are unmistakable.

    Both floors are measured, per body, by reading the source's own bank against
    the rotation the rule declares between its states.
    """

    carriers = tuple(replay.rotation_carriers or ())
    if not carriers or len(frames) < 2:
        return {
            "video_rotation_carrier_count": len(carriers),
            "video_rotation_direction_consistent": 1.0,
            "video_rotation_advance_within_budget": 1.0,
        }
    declared_span = np.diff(np.asarray(bank_progress, dtype=np.float64))
    excursions: dict[str, float] = {}
    advances: dict[str, float] = {}
    floors: dict[str, float] = {}
    allowances: dict[str, float] = {}
    for carrier in carriers:
        floor = rotation_reading_floor(
            rotation_profiles(
                bank, carrier.center, carrier.inner_radius, carrier.outer_radius
            ),
            declared_span * carrier.degrees_per_progress,
        )
        turned = profile_rotations(
            rotation_profiles(
                frames, carrier.center, carrier.inner_radius, carrier.outer_radius
            )
        ) * (1.0 if carrier.degrees_per_progress >= 0.0 else -1.0)
        floors[carrier.name] = floor
        allowances[carrier.name] = (
            float(budget) * abs(float(carrier.degrees_per_progress)) + floor
        )
        excursions[carrier.name] = backward_rotation_degrees(turned, 1)
        advances[carrier.name] = float(np.max(turned)) if turned.size else 0.0
    reversed_bodies = sorted(name for name in excursions if excursions[name] > floors[name])
    overrunning = sorted(name for name in advances if advances[name] > allowances[name])
    worst_backward = max(excursions, key=lambda name: excursions[name] / floors[name])
    worst_forward = max(advances, key=lambda name: advances[name] / allowances[name])
    return {
        "video_rotation_carrier_count": len(carriers),
        "video_rotation_direction_consistent": float(not reversed_bodies),
        "video_rotation_advance_within_budget": float(not overrunning),
        "video_rotation_reversed_bodies": reversed_bodies,
        "video_rotation_overrunning_bodies": overrunning,
        "video_rotation_backward_excursion_degrees": excursions,
        "video_rotation_reading_floor_degrees": floors,
        "video_rotation_max_advance_degrees": advances,
        "video_rotation_advance_allowance_degrees": allowances,
        "video_rotation_worst_backward_body": worst_backward,
        "video_rotation_worst_backward_ratio": (
            excursions[worst_backward] / floors[worst_backward]
        ),
        "video_rotation_worst_advance_body": worst_forward,
        "video_rotation_worst_advance_ratio": (
            advances[worst_forward] / allowances[worst_forward]
        ),
    }


def _progress_carrier_gates(
    frames: Sequence[np.ndarray],
    bank: Sequence[np.ndarray],
    bank_progress: Sequence[float],
    replay: PhysicsReplay,
    *,
    budget: float,
) -> dict[str, Any]:
    """Hard-gate each declared one-way extent on its own.

    The alignment reads order out of how much the picture changes, which is a
    magnitude, and a magnitude can always be made small enough for a short rewind
    to hide inside the emission ceiling.  What a rewind cannot hide from is a
    quantity the rule forbids from decreasing.  Each carrier is read as the count
    of its declared sample points the frame has reached (:func:`carrier_readings`)
    and gated three ways:

    * **Backwards.**  A ``+1`` carrier's reading may not travel backwards past
      the reader's own floor.  A trace that shortens, water that drains out of a
      filling chamber, a fallen domino standing back up: each contradicts the
      rule, at any step size and for any length of rewind.
    * **Distance travelled.**  Every carrier - including one that legitimately
      moves both ways, like a container that receives water and later gives it
      away - travels exactly as far as the source does.  Total variation does not
      depend on how the path is sampled, so a candidate that dwells travels
      exactly as far as the source and one that rewinds, repeats or swaps frames
      travels further.
    * **Forwards.**  One frame may not move a carrier further than the source
      moves it across one legal progress advance, plus the floor.  This is the
      alignment's step budget restated in the carrier's own units, and it is what
      a cut or a skipped stretch runs into.

    All three thresholds are read off the source's own progress bank, per
    carrier: the floor from the backwards travel the source's own renders show,
    the travel allowance from the noise finer sampling manufactures, and the
    advance bound from the largest move the source makes across one budgeted
    advance.  None of them is a chosen number.

    Coloured thin traces are read in the source terminal ink's chromatic
    direction, so a moving ball or neutral antialias halo does not impersonate
    already drawn ink. Solid-body carriers retain their declared fill reader.

    The backwards floor also owns the total-variation floor: an allowed
    one-point reading excursion costs two points when it retreats and returns.
    Counting only one leg would reject the same raster jitter that the direction
    gate accepts. Larger reversals and excess travel remain hard failures.
    """

    carriers = tuple(replay.progress_carriers or ())
    if not carriers or len(frames) < 2:
        return {
            "video_progress_carrier_count": len(carriers),
            "video_carrier_direction_consistent": 1.0,
            "video_carrier_travel_within_declared": 1.0,
            "video_carrier_advance_within_budget": 1.0,
        }
    backward: dict[str, float] = {}
    floors: dict[str, float] = {}
    travel: dict[str, float] = {}
    travel_allowance: dict[str, float] = {}
    advance: dict[str, float] = {}
    advance_allowance: dict[str, float] = {}
    settled_counts: dict[str, int] = {}
    for carrier in carriers:
        # read only the points the source's own terminal state settles, so
        # the bank's travel really is a bound on any legal candidate's rather than
        # an artefact of which frames each grid happened to sample.
        keep = carrier_settled_points(bank[-1], carrier.samples, carrier.rest_rgb, carrier.fill_rgb)
        samples = tuple(
            point for point, settled in zip(carrier.samples, keep) if settled
        ) or carrier.samples
        settled_counts[carrier.name] = len(samples)
        thin_rgb = None
        if carrier.fill_rgb is None:
            pixel_points = np.rint(np.asarray(samples)).astype(int)
            thin_rgb = np.median(bank[-1][pixel_points[:, 1], pixel_points[:, 0]], axis=0)
        source = carrier_readings(bank, samples, carrier.rest_rgb, carrier.fill_rgb, thin_rgb=thin_rgb)
        observed = carrier_readings(frames, samples, carrier.rest_rgb, carrier.fill_rgb, thin_rgb=thin_rgb)
        floor = carrier_reading_floor(source, carrier.direction)
        # Water-height carriers sample two parallel pixel columns. A one-pixel
        # contour change moves both readings together, so its measurement floor
        # is two points, rather than the one point of a single trace. Keep this
        # tied to the declared sampling geometry; a one-column trace retains
        # its original one-point floor.
        pixel_samples = np.rint(np.asarray(samples)).astype(int)
        cross_section = min(len(np.unique(pixel_samples[:, axis])) for axis in (0, 1))
        if cross_section <= 2:
            floor = max(floor, float(cross_section))
        floors[carrier.name] = floor
        backward[carrier.name] = (
            carrier_backward_travel(observed, carrier.direction) if carrier.direction else 0.0
        )
        travel[carrier.name] = carrier_total_variation(observed)
        travel_allowance[carrier.name] = carrier_total_variation(source) + max(
            2.0 * floor, 2.0 * carrier_sampling_noise(source)
        )
        advance[carrier.name] = (
            float(np.max(np.abs(np.diff(observed)))) if observed.size > 1 else 0.0
        )
        advance_allowance[carrier.name] = (
            carrier_advance_bound(source, bank_progress, budget=float(budget)) + floor
        )
    reversed_carriers = sorted(name for name in backward if backward[name] > floors[name])
    overtravelling = sorted(name for name in travel if travel[name] > travel_allowance[name])
    overrunning = sorted(name for name in advance if advance[name] > advance_allowance[name])
    worst_backward = max(backward, key=lambda name: backward[name] / floors[name])
    worst_travel = max(travel, key=lambda name: travel[name] / max(travel_allowance[name], 1e-9))
    worst_advance = max(
        advance, key=lambda name: advance[name] / max(advance_allowance[name], 1e-9)
    )
    return {
        "video_progress_carrier_count": len(carriers),
        "video_carrier_direction_consistent": float(not reversed_carriers),
        "video_carrier_travel_within_declared": float(not overtravelling),
        "video_carrier_advance_within_budget": float(not overrunning),
        "video_carrier_settled_point_counts": settled_counts,
        "video_carrier_reversed": reversed_carriers,
        "video_carrier_overtravelling": overtravelling,
        "video_carrier_overrunning": overrunning,
        "video_carrier_backward_travel_points": backward,
        "video_carrier_reading_floor_points": floors,
        "video_carrier_travel_points": travel,
        "video_carrier_travel_allowance_points": travel_allowance,
        "video_carrier_max_advance_points": advance,
        "video_carrier_advance_allowance_points": advance_allowance,
        "video_carrier_worst_backward": worst_backward,
        "video_carrier_worst_backward_ratio": (
            backward[worst_backward] / floors[worst_backward]
        ),
        "video_carrier_worst_travel": worst_travel,
        "video_carrier_worst_travel_ratio": (
            travel[worst_travel] / max(travel_allowance[worst_travel], 1e-9)
        ),
        "video_carrier_worst_advance": worst_advance,
        "video_carrier_worst_advance_ratio": (
            advance[worst_advance] / max(advance_allowance[worst_advance], 1e-9)
        ),
    }



def _mirror_answer_order_gates(
    frames: Sequence[np.ndarray],
    observations: Sequence[dict[str, Any]],
    bank: Sequence[np.ndarray],
    replay: PhysicsReplay,
) -> dict[str, Any]:
    """A detector cannot light before the complete beam or turn gray again.

    Read the detector throughout the video, including before the final reveal
    tail. A brief premature green flash otherwise disappears when that tail is
    split at the last gray frame. Beam points inside the detector's own changed
    footprint are excluded because its legal green fill can cover their ink;
    the remainder must already show the full source-derived ray on the first
    visibly lit frame. No sampled opacity count or gray endpoint hold is imposed.
    """
    opacities = np.clip([float(item.get("opacity", 0.0)) for item in observations], 0.0, 1.0)
    lit = np.flatnonzero(opacities >= 0.08)
    retained = bool(not len(opacities) or np.max(np.maximum.accumulate(opacities) - opacities) <= 0.06)
    first_lit = int(lit[0]) if len(lit) else -1
    terminal = _rgb(replay.render_progress(1.0, 1.0))
    base = _rgb(replay.render_progress(1.0, 0.0))
    overlay = np.max(np.abs(terminal.astype(np.int16) - base.astype(np.int16)), axis=2) > 10
    overlay = cv2.dilate(overlay.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    missing: dict[str, float] = {}
    complete = first_lit >= 0
    for carrier in replay.progress_carriers:
        points = np.rint(np.asarray(carrier.samples)).astype(int)
        keep = ~overlay[points[:, 1], points[:, 0]]
        samples = tuple(point for point, retain in zip(carrier.samples, keep) if retain)
        if not samples:
            continue
        points = np.rint(np.asarray(samples)).astype(int)
        ink = np.median(bank[-1][points[:, 1], points[:, 0]], axis=0)
        source = carrier_readings(bank, samples, carrier.rest_rgb, carrier.fill_rgb, thin_rgb=ink)
        observed = carrier_readings(frames[:first_lit + 1], samples, carrier.rest_rgb,
                                    carrier.fill_rgb, thin_rgb=ink) if first_lit >= 0 else np.zeros(1)
        missing[carrier.name] = max(0.0, float(source[-1] - observed[-1]))
        complete = complete and missing[carrier.name] <= carrier_reading_floor(source, carrier.direction)
    return {
        "video_answer_overlay_first_visible_frame": first_lit,
        "video_answer_overlay_retained_throughout": float(retained),
        "video_answer_beam_missing_when_detector_lit": missing,
        "video_answer_beam_complete_when_detector_lit": float(complete),
        "video_answer_overlay_order_correct": float(retained and complete),
    }


def _verify_static_process_video(
    frames: list[np.ndarray],
    fps: float,
    replay: PhysicsReplay,
) -> dict[str, Any]:
    """Verify a declared static physical process (for example a jammed train).

    The only legal change over the whole video is the element-local answer
    reveal: every frame must be a recognized opacity state of the source
    answer footprint, starting from the untouched input, rising monotonically,
    and holding the complete terminal state.  Motion-progress gates do not
    apply because the source replay proves no object may move.
    """

    if replay.answer_overlay_mode != "local_opacity":
        raise ValueError("static-process replay requires a local-opacity answer overlay")
    base = _rgb(replay.render_progress(1.0, 0.0))
    terminal = _rgb(replay.render_progress(1.0, 1.0))
    observations = [
        _answer_overlay_observation(_resize(frame, base.shape[:2]).astype(np.float32).astype(np.uint8), base, terminal)
        for frame in frames
    ]
    recognized_flags = [bool(observation.get("recognized")) for observation in observations]
    recognition = sum(recognized_flags) / len(frames)
    opacities = [float(observation.get("opacity", -1.0)) for observation in observations]
    initial_hold = bool(
        len(frames) >= 2
        and all(recognized_flags[:2])
        and max(opacities[:2]) <= 0.05
    )
    terminal_hold = bool(
        len(frames) >= 2
        and all(recognized_flags[-2:])
        and min(opacities[-2:]) >= 0.92
    )
    recognized_opacities = [
        opacity for opacity, flag in zip(opacities, recognized_flags) if flag
    ]
    monotonic = all(
        right + 0.06 >= left
        for left, right in zip(recognized_opacities, recognized_opacities[1:])
    )
    partial_indices = [
        index
        for index, observation in enumerate(observations)
        if observation.get("partial")
    ]
    partial_levels = {round(opacities[index] * 8) for index in partial_indices}
    partials_complete = all(bool(observations[index].get("pass")) for index in partial_indices)
    duration = len(frames) / fps if fps > 0 else None
    media_valid = bool(
        all(frame.shape[:2] == (512, 512) for frame in frames)
        and abs(fps - 16.0) <= 0.2
        and len(frames) >= 5
        and (len(frames) - 1) % 4 == 0
        and duration is not None
        and duration > 0
    )
    terminal_metrics = _image_semantics(frames[-1], replay)
    correct = bool(
        media_valid
        and recognition >= 0.9
        and monotonic
        and terminal_metrics["terminal_image_correct"]
    )
    semantic_clash_metrics = {}
    if replay.source_variant == "gear_train_direction_visible_source_v2" and replay.answer == "JAM":
        # A jam has no physical motion. Each declared cross must appear and
        # remain; its opacity curve need not reproduce the renderer's pixels.
        marks = [_clash_marks(frame, terminal) for frame in frames]
        red = terminal.astype(np.int16)
        dynamic = (red[:, :, 0] > red[:, :, 1] + 40) & (red[:, :, 0] > red[:, :, 2] + 30)
        dynamic = cv2.dilate(dynamic.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        static_errors = [static_region_errors(frame, base, ~dynamic)[0] for frame in frames]
        progress = [item["terminal_clash_mark_supports"] for item in marks]
        starts_blank = bool(progress and max(progress[0], default=1) <= .08)
        retained = all(right + .08 >= left for before, after in zip(progress, progress[1:])
                       for left, right in zip(before, after))
        clean = all(item["terminal_clash_mark_pixels"] < 12 or item["terminal_clash_mark_precision"] >= .80
                    for item in marks)
        preserved = max(static_errors, default=1) <= STATIC_REGION_FLOOR
        correct = bool(media_valid and starts_blank and retained and clean and preserved
                       and terminal_metrics["terminal_image_correct"])
        semantic_clash_metrics = {"video_clash_mark_progress": progress,
                                  "video_process_observation": "source_preservation_and_clash_mark_states",
                                  "video_renderer_opacity_recognized_fraction": recognition,
                                  "video_progress_monotonic": float(retained),
                                  "video_clash_marks_retained": retained,
                                  "video_clash_marks_initially_absent": starts_blank,
                                  "video_clash_marks_no_extra_ink": clean,
                                  "video_static_preserved": float(preserved),
                                  "video_static_region_max_error": max(static_errors, default=1)}
    return {
        "whole_video_loaded": True,
        "whole_video_error": None,
        "video_frame_count": len(frames),
        "video_fps": fps,
        "video_duration_seconds": duration,
        "video_media_valid": float(media_valid),
        "video_static_process": True,
        "video_recognized_fraction": recognition,
        "video_initial_hold": float(initial_hold),
        "video_final_hold": float(terminal_hold),
        "video_progress_monotonic": float(monotonic),
        "video_answer_overlay_mode": replay.answer_overlay_mode,
        "video_answer_overlay_partial_level_count": len(partial_levels),
        "video_answer_overlay_complete_footprint_pass": float(partials_complete),
        "video_local_element_opacity_reveal": float(partials_complete and bool(partial_levels)),
        "video_full_frame_endpoint_dissolve_used": False,
        "video_final_state_correct": terminal_metrics["terminal_image_correct"],
        **{f"video_{key}": value for key, value in terminal_metrics.items()},
        **semantic_clash_metrics,
        "video_reference_pixels_used": False,
        "video_saved_keyframes_used": False,
        "whole_video_process_correct": float(correct),
    }


def _verify_video(path: Optional[Path], replay: PhysicsReplay) -> dict[str, Any]:
    if path is None:
        return {
            "whole_video_loaded": False,
            "whole_video_process_correct": 0.0,
            "whole_video_error": "candidate video is missing",
        }
    try:
        frames, fps = _read_video(path)
        if not replay.motion:
            return _verify_static_process_video(frames, fps, replay)
        bank_steps = max(32, len(frames))
        progress_bank, bank = _progress_bank(replay, steps=bank_steps)
        initial = bank[0]
        terminal = _rgb(replay.render_progress(1.0, 1.0))
        answer_base = _rgb(replay.render_progress(1.0, 0.0))
        require_answer_opacity = replay.answer_overlay_mode == "local_opacity"
        dynamic = np.zeros(initial.shape[:2], dtype=bool)
        for expected in [*bank[1:], terminal]:
            dynamic |= np.max(np.abs(initial.astype(np.int16) - expected.astype(np.int16)), axis=2) > 10
        if not np.any(dynamic):
            raise ValueError("source replay has no visible dynamic region")
        raw_dynamic = dynamic.copy()
        dynamic = cv2.dilate(dynamic.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
        static = ~dynamic
        if replay.preserved_regions:
            # The body observer owns domino shape, length and causal travel.
            # Its slightly displaced contour must not also be classified as a
            # background mutation. Fixed labels and blocked bodies are checked
            # explicitly inside this guard as well as everywhere outside it.
            static = ~cv2.dilate(raw_dynamic.astype(np.uint8), np.ones((33, 33), np.uint8)).astype(bool)
            for x0, y0, x1, y1 in replay.preserved_regions:
                static[max(0, int(y0)):min(512, int(np.ceil(y1))),
                       max(0, int(x0)):min(512, int(np.ceil(x1)))] = True
        # remove each frame's uniform per-channel tone offset before any
        # reading.  A tone shift carries no positional information, but it moves
        # antialiased pixels near a carrier's rest colour across the departure
        # threshold (a +2 shift grew mirror_aim_detector's beam reading by six
        # sample points and tripped the travel bound at 1.003), and it consumes
        # the answer overlay's linear-fit allowance.  The offset is a median
        # over unsaturated static pixels, so a locally wrong object or painted
        # patch moves it by a fraction of its own area and still rejects; a
        # genuine candidate's offset is zero and nothing changes.
        decoded = [
            np.clip(frame, 0.0, 255.0).astype(np.uint8)
            for frame in detoned(
                [_resize(raw, initial.shape[:2]) for raw in frames],
                initial,
                static if np.any(static) else None,
            )
        ]
        initial_float = initial.astype(np.float32)

        # The element-local answer reveal is not part of the physical
        # process: the source may only raise it once progress is complete, so it
        # is split off the tail of the frame sequence and keeps its own
        # footprint, monotonicity and completeness gates.  The split point is the
        # last frame that still shows the completed process with the overlay at
        # zero opacity; without such a frame the reveal has no base endpoint and
        # its own gate fails, so every frame is aligned as motion instead.
        answer_opacity_observations = [
            _answer_overlay_observation(frame, answer_base, terminal)
            if require_answer_opacity
            else {"recognized": False, "partial": False, "pass": True, "opacity": 0.0}
            for frame in decoded
        ]
        terminal_overlay_start = len(answer_opacity_observations)
        for index in range(len(answer_opacity_observations) - 1, -1, -1):
            observation = answer_opacity_observations[index]
            if (
                bool(observation.get("recognized"))
                and float(observation.get("opacity", 0.0)) >= 0.92
            ):
                terminal_overlay_start = index
                continue
            break
        answer_base_index = max(
            (
                index
                for index, observation in enumerate(answer_opacity_observations)
                if index < terminal_overlay_start
                and bool(observation.get("recognized"))
                and float(observation.get("opacity", 1.0)) <= 0.05
            ),
            default=-1,
        )
        reveal_start = (
            answer_base_index + 1
            if require_answer_opacity and answer_base_index >= 0
            else len(decoded)
        )

        # read the frames as a sequence, not one at a time.  A
        # process whose visible state recurs - equal-mass exchange reversing a
        # ball, a ray retracing its corridor - is non-injective in time, so no
        # per-frame nearest-state reading can describe it.  One Viterbi pass over
        # the progress bank finds the best monotone reading whose per-frame
        # advance stays inside a budget derived from the reference video's own
        # advance, and every property below is then checked on that path.
        step_allowance = step_budget(len(decoded), 1.0 / float(bank_steps))
        bank_quantum = 1.0 / float(bank_steps)
        # Order can only be read at a spatial scale where the source's own
        # consecutive states are still more alike than unrelated ones; a scene
        # whose finest moving strokes clear their own width between frames has to
        # be compared more coarsely or its frames carry no order at all.
        scale, resolving_ratio = comparison_scale(
            bank,
            progress_bank,
            dynamic,
            span=reference_advance(len(decoded)) + 2.0 * bank_quantum,
        )
        probe_bank = dynamic_probe(at_scale(bank, scale), dynamic)
        probe_frames = dynamic_probe(at_scale(decoded[:reveal_start], scale), dynamic)
        pair_changes = observed_pair_changes(probe_frames)
        source_changes = reference_pair_changes(
            probe_bank,
            progress_bank,
            budget=step_allowance,
            bank_quantum=bank_quantum,
        )
        alignment = align_progress(
            emission_costs(probe_frames, probe_bank),
            progress_bank,
            budget=step_allowance,
            pace=reference_advance(len(decoded)),
            pair_changes=pair_changes,
            reference_changes=source_changes,
        )

        observed: list[Optional[float]] = []
        dynamic_errors: list[float] = []
        full_errors: list[float] = []
        static_errors: list[float] = []
        static_region_maxima: list[float] = []
        for index, frame in enumerate(decoded):
            frame_float = frame.astype(np.float32)
            if index >= reveal_start:
                observation = answer_opacity_observations[index]
                dynamic_error = float(observation["footprint_replay_error"])
                full_error = float(observation["full_replay_error"])
                recognized_value = 1.0 if observation.get("recognized") else None
            else:
                state = alignment.state_indices[index]
                # The state is chosen at the comparison scale, but the frame must
                # still be that state at full resolution: coarsening decides
                # order, it never excuses a wrong picture.
                dynamic_error = max(
                    alignment.emission_errors[index],
                    _dynamic_error(frame, bank[state], dynamic) if scale else 0.0,
                )
                full_error = float(
                    np.mean(np.abs(frame_float - bank[state].astype(np.float32)))
                ) / 255.0
                recognized_value = (
                    alignment.progress[index]
                    if dynamic_error <= _EMISSION_CEILING and full_error <= _FULL_FRAME_CEILING
                    else None
                )
            dynamic_errors.append(dynamic_error)
            full_errors.append(full_error)
            static_errors.append(
                float(np.mean(np.abs(frame_float - initial_float)[static])) / 255.0
                if np.any(static)
                else 0.0
            )
            # The mean over the whole static mask dilutes a painted patch to
            # nothing; the same statistic per 32x32 region does not (11.8).
            static_region_maxima.append(
                static_region_errors(frame_float, initial_float, static)[0]
                if np.any(static)
                else 0.0
            )
            observed.append(recognized_value)

        recognized = [value for value in observed if value is not None]
        recognition = len(recognized) / len(frames)
        # Endpoint holds are checked against the source states directly rather
        # than through the reading: on a process whose appearance plateaus at an
        # endpoint the reading is free inside the plateau, but "the first frames
        # are the untouched input" and "the last frames are the terminal state"
        # are pixel facts that do not depend on any alignment.
        initial_hold_errors = [_dynamic_error(frame, initial, dynamic) for frame in decoded[:2]]
        final_hold_errors = [_dynamic_error(frame, terminal, dynamic) for frame in decoded[-2:]]
        initial_hold = bool(
            len(decoded) >= 2 and max(initial_hold_errors) <= _EMISSION_CEILING
        )
        final_hold = bool(
            len(decoded) >= 2 and max(final_hold_errors) <= _EMISSION_CEILING
        )
        # A rewind, a duplicated segment or a splice can leave every frame
        # resembling *some* legal state, so the alignment alone may still find a
        # bounded-step reading.  What it cannot hide is the seam, and the seam
        # can be wrong in either direction: the visible change between two
        # consecutive frames must not exceed what the source itself produces
        # across one budgeted advance, and must not fall short of what the source
        # puts between the two states the reading claims those frames are.  The
        # second half is what refuses a deletion that hides in a catch-up - the
        # reading advancing several bank samples over a pair of frames whose
        # picture is identical.
        pace_bounds = local_pair_change_bounds(source_changes, progress_bank)
        (
            pace_breaks,
            worst_pace_ratio,
            worst_pair_change,
            pace_floor_breaks,
            worst_pace_floor_ratio,
        ) = pace_violations(
            pair_changes,
            alignment.state_indices,
            pace_bounds,
            crossed_pair_changes(probe_bank, alignment.state_indices),
        )
        pace_consistent = pace_breaks == 0 and pace_floor_breaks == 0
        max_jump = alignment.max_step
        # The declared bodies' own turning, read straight out of the candidate's
        # pixels in the units the source declares it in.  This is the # evidence the alignment cannot supply, because it is the only reading
        # here that does not scale with how far the scene moves between frames.
        rotation_metrics = _rotation_reading_gates(
            decoded[:reveal_start], bank, progress_bank, replay, budget=step_allowance
        )
        rotation_consistent = bool(
            rotation_metrics["video_rotation_direction_consistent"]
            and rotation_metrics["video_rotation_advance_within_budget"]
        )
        # The declared one-way extents, read the same way: in units the source
        # owns, so shrinking the per-frame step shrinks the reading and its bound
        # together and buys a tampered sequence nothing.
        carrier_metrics = _progress_carrier_gates(
            decoded[:reveal_start], bank, progress_bank, replay, budget=step_allowance
        )
        carrier_consistent = bool(
            carrier_metrics["video_carrier_direction_consistent"]
            and carrier_metrics["video_carrier_travel_within_declared"]
            and carrier_metrics["video_carrier_advance_within_budget"]
        )
        continuous = (
            alignment.feasible and pace_consistent and rotation_consistent and carrier_consistent
        )
        interior = {round(value * 8) for value in recognized if 0.08 < value < 0.92}
        intermediate_visible = len(interior) >= 3
        checkpoint_tolerance = min(_CHECKPOINT_TOLERANCE_CEILING, step_allowance)
        checkpoint_hits = {
            f"{checkpoint:.4f}": any(
                abs(value - checkpoint) <= checkpoint_tolerance for value in recognized
            )
            for checkpoint in replay.semantic_checkpoints[1:-1]
        }
        checkpoints_covered = all(checkpoint_hits.values())
        answer_partial_indices = [
            index
            for index, observation in enumerate(answer_opacity_observations)
            if answer_base_index < index < terminal_overlay_start
            and bool(observation.get("partial"))
        ]
        answer_partial_levels = {
            round(float(answer_opacity_observations[index]["opacity"]) * 8)
            for index in answer_partial_indices
        }
        answer_base_endpoint = answer_base_index >= 0
        answer_terminal_hold = bool(
            len(answer_opacity_observations) >= 2
            and all(
                bool(observation.get("recognized"))
                and float(observation.get("opacity", 0.0)) >= 0.92
                for observation in answer_opacity_observations[-2:]
            )
        )
        answer_phase_values = [
            float(observation["opacity"])
            for observation in answer_opacity_observations[
                answer_base_index + 1 : terminal_overlay_start
            ]
            if bool(observation.get("recognized"))
            and float(observation.get("opacity", 0.0)) >= 0.05
        ]
        answer_monotonic = all(
            right + 0.06 >= left
            for left, right in zip(answer_phase_values, answer_phase_values[1:])
        )
        answer_opacity_pass = bool(
            not require_answer_opacity
            or (
                bool(answer_partial_levels)
                and answer_base_endpoint
                and answer_terminal_hold
                and answer_monotonic
                and all(
                    bool(answer_opacity_observations[index].get("pass"))
                    for index in answer_partial_indices
                )
            )
        )
        answer_order_metrics = {}
        if replay.source_variant == "mirror_aim_detector_visible_source_v2":
            answer_order_metrics = _mirror_answer_order_gates(
                decoded, answer_opacity_observations, bank, replay
            )
        static_preserved = (
            max(static_errors, default=1.0) <= 0.04
            and max(static_region_maxima, default=0.0) <= STATIC_REGION_FLOOR
        )
        dimensions_valid = all(frame.shape[:2] == (512, 512) for frame in frames)
        duration = len(frames) / fps if fps > 0 else None
        media_valid = bool(
            dimensions_valid
            and abs(fps - 16.0) <= 0.2
            and len(frames) >= 5
            and (len(frames) - 1) % 4 == 0
            and duration is not None
            and duration > 0
        )
        terminal_metrics = _image_semantics(frames[-1], replay)
        numeric_violations = []
        if replay.numeric_regions:
            for index, frame in enumerate(decoded):
                for name, reading in _numeric_fields(frame, terminal, replay).items():
                    if reading["observed"] is not None and not reading["pass"]:
                        numeric_violations.append({"frame": index, "field": name, **reading})
        # Alignment ambiguity must not turn repeated endpoint pixels into motion.
        # This counts visible intermediate states, independent of reference pace.
        visible_intermediate_count = sum(
            float(np.mean(np.abs(frame.astype(np.float32)-initial.astype(np.float32))[dynamic]))/255 > .015
            and float(np.mean(np.abs(frame.astype(np.float32)-terminal.astype(np.float32))[dynamic]))/255 > .015
            for frame in decoded
        )
        correct = bool(
            media_valid
            and visible_intermediate_count >= 3
            # Required source events and their order carry physical meaning;
            # ambiguous in-between codec frames are not themselves failures.
            and recognition >= .80
            and intermediate_visible
            and checkpoints_covered
            # Direction and total travel express physical rules independently
            # of renderer cadence: filling cannot drain, a fallen domino cannot
            # stand up, and a driven gear cannot reverse. Pace-derived advance
            # diagnostics remain separate from these observable violations.
            and rotation_metrics["video_rotation_direction_consistent"]
            and carrier_metrics["video_carrier_direction_consistent"]
            and carrier_metrics["video_carrier_travel_within_declared"]
            and static_preserved
            and terminal_metrics["terminal_image_correct"]
            and not numeric_violations
            and answer_order_metrics.get("video_answer_overlay_order_correct", 1.0)
        )
        return {
            "whole_video_loaded": True,
            "whole_video_error": None,
            "video_frame_count": len(frames),
            "video_fps": fps,
            "video_duration_seconds": duration,
            "video_media_valid": float(media_valid),
            "video_recognized_fraction": recognition,
            "video_max_dynamic_replay_error": max(dynamic_errors, default=1.0),
            "video_max_full_replay_error": max(full_errors, default=1.0),
            "video_initial_hold": float(initial_hold),
            "video_initial_hold_error": max(initial_hold_errors, default=1.0),
            "video_final_hold": float(final_hold),
            "video_final_hold_error": max(final_hold_errors, default=1.0),
            # Monotonicity is structural under the alignment model, so what the
            # reading can still fail is existence: whether any monotone
            # bounded-step path through the source states explains these frames.
            "video_alignment_feasible": float(alignment.feasible),
            "video_alignment_overstep_frames": alignment.overstep_frames,
            "video_alignment_step_budget": step_allowance,
            "video_alignment_reference_pace": reference_advance(len(decoded)),
            "video_alignment_motion_frame_count": reveal_start,
            "video_alignment_progress": list(alignment.progress),
            "video_alignment_comparison_scale": scale,
            "video_source_state_resolving_ratio": resolving_ratio,
            "video_frame_pace_violations": pace_breaks,
            "video_worst_frame_pace_ratio": worst_pace_ratio,
            "video_worst_frame_pair_change": worst_pair_change,
            "video_frame_pace_floor_violations": pace_floor_breaks,
            "video_worst_frame_pace_floor_ratio": worst_pace_floor_ratio,
            "video_frame_pace_consistent": float(pace_consistent),
            **rotation_metrics,
            **carrier_metrics,
            "video_max_progress_jump": max_jump,
            "video_continuous_process": float(continuous and intermediate_visible),
            "video_visible_intermediate_frames": visible_intermediate_count,
            "video_semantic_checkpoint_tolerance": checkpoint_tolerance,
            "video_semantic_checkpoint_hits": checkpoint_hits,
            "video_semantic_checkpoints_covered": float(checkpoints_covered),
            **answer_order_metrics,
            "video_local_element_opacity_reveal": float(answer_opacity_pass),
            "video_full_frame_endpoint_dissolve_used": False,
            "video_answer_overlay_mode": replay.answer_overlay_mode,
            "video_answer_overlay_complete_footprint_pass": float(answer_opacity_pass),
            "video_answer_overlay_partial_level_count": len(answer_partial_levels),
            "video_answer_overlay_base_endpoint": float(
                answer_base_endpoint if require_answer_opacity else True
            ),
            "video_answer_overlay_terminal_hold": float(
                answer_terminal_hold if require_answer_opacity else True
            ),
            "video_answer_overlay_opacity_monotonic": float(
                answer_monotonic if require_answer_opacity else True
            ),
            "video_answer_overlay_opacity_observations": answer_opacity_observations,
            "video_answer_overlay_worst_q10_q90_spread": max(
                (
                    float(answer_opacity_observations[index]["q10_q90_spread"])
                    for index in answer_partial_indices
                ),
                default=(0.0 if not require_answer_opacity else 1.0),
            ),
            "video_answer_overlay_min_support": min(
                (
                    float(answer_opacity_observations[index]["footprint_support"])
                    for index in answer_partial_indices
                ),
                default=(1.0 if not require_answer_opacity else 0.0),
            ),
            "video_answer_overlay_max_linear_fit_error": max(
                (
                    float(answer_opacity_observations[index]["linear_fit_error"])
                    for index in answer_partial_indices
                ),
                default=(0.0 if not require_answer_opacity else 1.0),
            ),
            "video_answer_overlay_max_outside_static_error": max(
                (
                    float(answer_opacity_observations[index]["outside_footprint_static_error"])
                    for index in answer_partial_indices
                ),
                default=(0.0 if not require_answer_opacity else 1.0),
            ),
            "video_static_preservation_error": max(static_errors, default=1.0),
            "video_static_region_max_error": max(static_region_maxima, default=0.0),
            "video_static_region_bound": STATIC_REGION_FLOOR,
            "video_static_preserved": float(static_preserved),
            "video_final_state_correct": terminal_metrics["terminal_image_correct"],
            "video_numeric_field_violations": numeric_violations,
            **{f"video_{key}": value for key, value in terminal_metrics.items()},
            "video_reference_pixels_used": False,
            "video_saved_keyframes_used": False,
            "whole_video_process_correct": float(correct),
        }
    except Exception as exc:
        return {
            "whole_video_loaded": False,
            "whole_video_process_correct": 0.0,
            "whole_video_error": str(exc),
        }


def verify_physics(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: str = "ti2v",
) -> list[EvalRecord]:
    """Verify exactly the declared Physics final-delivery modality."""

    selected = str(modality or "").strip().lower()
    if selected not in _MODALITIES:
        raise ValueError(f"unsupported Physics modality: {modality!r}")
    records: list[EvalRecord] = []
    for sample in samples:
        video_path = resolve_candidate_video(pred_root, sample.id, sample.split)
        try:
            replay = replay_physics(sample, modality=selected)
            replay_error = None
        except Exception as exc:
            replay = None
            replay_error = str(exc)

        video_metrics = (
            _verify_video(video_path, replay)
            if replay is not None else
            {"whole_video_loaded": False, "whole_video_process_correct": 0.0, "whole_video_error": replay_error}
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


__all__ = ["verify_physics"]
