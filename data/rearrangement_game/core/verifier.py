"""Full-video verifier for all Rearrangement Game tasks.

Validates the complete ordered motion program against independently solved
identity, contact, and commit templates without loading saved answers, saved
traces, terminal reference images, or reference videos.
"""

from __future__ import annotations

import math

from pathlib import Path
from copy import deepcopy
from itertools import permutations

import tempfile
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

from data.common.progress_alignment import tone_offset
from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.verifiers.keyframes import visual_delta
from data.common.verifiers.region_gates import (
    region_error_map,
    static_region_errors,
    region_delivery_gate,
    member_commit_gate,
    STATIC_REGION_FLOOR,
    REGION_ERROR_CEILING,
)
from data.rearrangement_game.core.source_replay import (
    replay_plan,
    solve_plan,
    source_record,
)


CANVAS_SIZE = 512
VIDEO_FPS = 16.0
VIDEO_FPS_TOLERANCE = 0.1
VIDEO_MIN_RECOGNIZED_FRACTION = 0.97
# Ceilings on the terminal-image bounds.  The bounds themselves are derived
# per sample from the measured separation between the terminal arrangement and
# the other arrangements the same plan passes through; these values only stop
# a very distinguishable plan from buying an arbitrarily loose bound.
TERMINAL_DYNAMIC_ERROR = 0.060
TERMINAL_FULL_ERROR = 0.040
# Scaling convention shared with the whole-video landmark gate below:
# ``max(floor, min(ceiling, distinctiveness * alpha))``.
TERMINAL_SEPARATION_ALPHA = 0.45
# Floor under the aggregate terminal bounds. Local regions separately allow
# a codec/raster floor so static stroke edges do not determine correctness.
TERMINAL_ERROR_FLOOR = 0.006
MOTION_DYNAMIC_ERROR = 0.070
MOTION_FULL_ERROR = 0.045
# Direct (non-monotonic) time-order gate: (min progress correlation, max
# backwards fraction).  Tasks whose single action legally shifts several
# objects at once through revisited arrangements (park-then-place survivor
# compaction) alias the frame-wise nearest-template classifier, so they carry
# a bounded task-specific tolerance per ; the monotonic
# coverage, landmark, hold, and terminal gates stay at full strength, and a
# reversed clip still scores a negative correlation far below every floor.
DIRECT_TIME_ORDER_DEFAULT = (0.80, 0.025)
DIRECT_TIME_ORDER_TOLERANCES = {
    "train_couple_swap": (0.50, 0.10),
}


def _resize(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape[:2] == shape:
        return array
    import cv2

    height, width = shape
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)


def _dynamic_mask(images: Sequence[np.ndarray], *, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    resized = [_resize(image, shape).astype(np.int16) for image in images]
    base = resized[0]
    mask = np.zeros(shape, dtype=bool)
    for image in resized[1:]:
        mask |= np.max(np.abs(image - base), axis=2) > 8
    if int(mask.sum()) < 32:
        mask[:] = True
    else:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    return mask


def _image_array(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as image:
        original_size = image.size
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb, original_size


def _channel_identity(
    candidate: np.ndarray,
    expected: np.ndarray,
) -> tuple[bool, list[float], list[int]]:
    """Reject global channel permutations masquerading as object identities."""

    permutations = (
        (0, 1, 2),
        (1, 0, 2),
        (0, 2, 1),
        (2, 1, 0),
        (1, 2, 0),
        (2, 0, 1),
    )
    # High-chroma pixels carry palette/identity information; using the whole
    # canvas would let a large neutral background hide a colour swap.
    chroma = np.ptp(expected.astype(np.int16), axis=2) >= 18
    if int(chroma.sum()) < 32:
        chroma[:] = True
    errors = [
        float(np.mean(np.abs(candidate[:, :, permutation] - expected)[chroma])) / 255.0
        for permutation in permutations
    ]
    best = int(np.argmin(errors))
    improvement = errors[0] - errors[best]
    passed = bool(best == 0 or improvement <= 0.0025)
    return passed, errors, list(permutations[best])


def _derived_bound(separation: float, ceiling: float) -> float:
    """Scale a bound with a measured separation, as the video path does."""

    if separation <= 0.0:
        return TERMINAL_ERROR_FLOOR
    return max(TERMINAL_ERROR_FLOOR, min(ceiling, separation * TERMINAL_SEPARATION_ALPHA))


def _arrangement_region_gates(candidate, initial, terminal, alternates=(), *, prefix=""):
    # Multi-panel scenes have several legitimate background colours. A modal
    # whole-canvas ink mask treats one panel's background as an object and can
    # label it erased after a small generated tone shift. Match changed object
    # members directly; local delivery regions preserve the rest of the scene.
    regions, region_pass = region_delivery_gate(candidate, terminal, alternates, prefix=prefix)
    members, member_pass = member_commit_gate(candidate, initial, terminal, prefix=prefix)
    return {**regions, **members,
            f"{prefix}region_delivery_pass": region_pass,
            f"{prefix}member_commit_pass": member_pass}, bool(region_pass and member_pass)


def _terminal_image_semantics(
    candidate_path: Optional[Path],
    stable: Sequence[np.ndarray],
) -> tuple[dict[str, Any], bool]:
    if candidate_path is None:
        return {
            "image_semantic_parser": "rearrangement_source_terminal_v2",
            "image_loaded": False,
            "image_media_contract_valid": False,
            "image_terminal_state_pass": False,
            "image_static_preservation_pass": False,
        }, False
    try:
        candidate, size = _image_array(candidate_path)
        shape = stable[-1].shape[:2]
        candidate = _resize(candidate, shape).astype(np.float32)
        expected = [_resize(image, shape).astype(np.float32) for image in stable]
        mask = _dynamic_mask(stable, shape=shape)
        # Style variance that changes no semantics must be accepted,
        # and a uniform tone shift is the whole of that freedom - it is the same
        # at every pixel and so says nothing about where anything is.  Every bound
        # below sits at the renderer floor (two renders of one arrangement are
        # byte-identical), so without this a candidate two levels brighter fails
        # on a difference that carries no information.  The offset is measured on
        # the region no commit of this plan touches, where a legal image equals
        # the source exactly, and as a median it does not move for a locally
        # wrong object.
        static_region = ~mask
        candidate = candidate - tone_offset(
            candidate, expected[0], static_region if np.any(static_region) else None
        )
        dynamic_errors: list[float] = []
        full_errors: list[float] = []
        for image in expected:
            difference = visual_delta(candidate, image)
            dynamic_errors.append(float(np.mean(difference[mask])) / 255.0)
            full_errors.append(float(np.mean(difference)) / 255.0)
        classified = int(np.argmin(dynamic_errors))
        final_index = len(expected) - 1
        static_mask = ~mask
        static_error = (
            float(np.mean(np.abs(candidate - expected[0])[static_mask])) / 255.0
            if np.any(static_mask)
            else full_errors[final_index]
        )
        media_valid = size == (CANVAS_SIZE, CANVAS_SIZE)
        channel_identity_pass, channel_errors, best_channel_permutation = _channel_identity(
            candidate,
            expected[final_index],
        )
        # How far the terminal arrangement sits from the nearest *other*
        # arrangement this plan passes through.  Both aggregate bounds are
        # scaled by it instead of being fixed, so a plan whose commits barely
        # move the picture cannot be satisfied by an image that is equally
        # close to a different arrangement.
        dynamic_separation = min(
            (
                float(np.mean(np.abs(expected[final_index] - image)[mask])) / 255.0
                for image in expected[:-1]
            ),
            default=0.0,
        )
        full_separation = min(
            (
                float(np.mean(np.abs(expected[final_index] - image))) / 255.0
                for image in expected[:-1]
            ),
            default=0.0,
        )
        dynamic_bound = _derived_bound(dynamic_separation, TERMINAL_DYNAMIC_ERROR)
        full_bound = _derived_bound(full_separation, TERMINAL_FULL_ERROR)
        region_metrics, region_pass = _arrangement_region_gates(candidate, expected[0], expected[-1], expected[:-1], prefix="image_")
        terminal_pass = bool(
            classified == final_index
            and dynamic_errors[final_index] <= dynamic_bound
            and full_errors[final_index] <= full_bound
            and channel_identity_pass
            and region_pass
        )
        static_region_error, static_region_violations = static_region_errors(
            candidate, expected[0], static_mask,
        ) if np.any(static_mask) else (0.0, 0)
        preservation_pass = static_region_violations == 0
        passed = bool(media_valid and terminal_pass and preservation_pass)
        return {
            "image_semantic_parser": "rearrangement_source_terminal_v2",
            "image_loaded": True,
            "image_original_size": list(size),
            "image_media_contract_valid": media_valid,
            "image_classified_commit_index": classified,
            "image_expected_terminal_commit_index": final_index,
            "image_commit_dynamic_errors": dynamic_errors,
            "image_terminal_dynamic_error": dynamic_errors[final_index],
            "image_terminal_full_error": full_errors[final_index],
            "image_static_preservation_error": static_error,
            "image_terminal_dynamic_separation": round(dynamic_separation, 6),
            "image_terminal_full_separation": round(full_separation, 6),
            "image_terminal_dynamic_bound": round(dynamic_bound, 6),
            "image_terminal_full_bound": round(full_bound, 6),
            "image_static_preservation_bound": round(STATIC_REGION_FLOOR, 6),
            "image_static_region_max_error": round(static_region_error, 6),
            "image_static_region_violations": static_region_violations,
            **region_metrics,
            "image_region_pass": region_pass,
            "image_channel_permutation_errors": channel_errors,
            "image_best_channel_permutation": best_channel_permutation,
            "image_channel_identity_pass": channel_identity_pass,
            "image_terminal_state_pass": terminal_pass,
            "image_static_preservation_pass": preservation_pass,
            "image_failure_reason": "ok" if passed else "terminal arrangement, identity-preserving static region, or media contract mismatch",
        }, passed
    except Exception as exc:
        return {
            "image_semantic_parser": "rearrangement_source_terminal_v2",
            "image_loaded": False,
            "image_media_contract_valid": False,
            "image_terminal_state_pass": False,
            "image_static_preservation_pass": False,
            "image_failure_reason": str(exc),
        }, False


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    import cv2

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
        raise ValueError("candidate video has no readable frames")
    return frames, fps


def _template_distances(
    frames: Sequence[np.ndarray],
    templates: Sequence[np.ndarray],
    mask: np.ndarray,
) -> tuple[list[int], list[float], list[float]]:
    """Classify each frame against independently rendered motion progress."""

    template_float = [image.astype(np.float32) for image in templates]
    labels: list[int] = []
    dynamic_errors: list[float] = []
    full_errors: list[float] = []
    for raw in frames:
        frame = raw.astype(np.float32)
        best_index = -1
        best_dynamic = float("inf")
        best_full = float("inf")
        for index, expected in enumerate(template_float):
            difference = np.abs(frame - expected)
            dynamic_error = float(np.mean(difference[mask])) / 255.0
            if dynamic_error >= best_dynamic:
                continue
            best_index = index
            best_dynamic = dynamic_error
            best_full = float(np.mean(difference)) / 255.0
        recognized = best_dynamic <= MOTION_DYNAMIC_ERROR and best_full <= MOTION_FULL_ERROR
        labels.append(best_index if recognized else -1)
        dynamic_errors.append(best_dynamic)
        full_errors.append(best_full)
    return labels, dynamic_errors, full_errors


def _monotonic_template_distances(
    frames: Sequence[np.ndarray],
    templates: Sequence[np.ndarray],
    mask: np.ndarray,
) -> tuple[list[int], list[float], list[float], list[int]]:
    """Align candidate frames to the source timeline without time reversal.

    Some legal rearrangements revisit an earlier *visible* arrangement.  A
    frame-wise nearest-template classifier then aliases the later visit to the
    earlier one and rejects a correct identity-preserving process.  This small
    dynamic program keeps the visual score but chooses one globally monotonic
    source-time path, with a tiny tempo regularizer only to break exact-image
    ties caused by holds or revisited states.
    """

    frame_float = [image.astype(np.float32) for image in frames]
    template_float = [image.astype(np.float32) for image in templates]
    frame_count = len(frame_float)
    template_count = len(template_float)
    if frame_count < 2 or template_count < 2:
        labels, dynamic_errors, full_errors = _template_distances(frames, templates, mask)
        return labels, dynamic_errors, full_errors, labels

    dynamic = np.empty((frame_count, template_count), dtype=np.float64)
    full = np.empty_like(dynamic)
    template_bank=np.stack(template_float)
    # Bounded batches remove the quadratic Python/NumPy-call overhead while
    # retaining the same float32 pixel means followed by float64 division.
    for frame_index, frame in enumerate(frame_float):
        for start in range(0,template_count,64):
            batch=template_bank[start:start+64]
            difference=np.abs(batch-frame)
            pixels=np.ascontiguousarray(difference[:,mask]).reshape(len(batch),-1)
            dynamic[frame_index,start:start+len(batch)]=pixels.mean(axis=1).astype(np.float64)/255.0
            full[frame_index,start:start+len(batch)]=difference.reshape(len(batch),-1).mean(axis=1).astype(np.float64)/255.0

    visual_cost = dynamic + 0.35 * full
    direct_indices = np.argmin(dynamic, axis=1)
    direct_labels = [
        int(label)
        if dynamic[frame_index, label] <= MOTION_DYNAMIC_ERROR
        and full[frame_index, label] <= MOTION_FULL_ERROR
        else -1
        for frame_index, label in enumerate(direct_indices)
    ]
    infinite = float("inf")
    costs = np.full((frame_count, template_count), infinite, dtype=np.float64)
    parents = np.full((frame_count, template_count), -1, dtype=np.int32)
    costs[0, 0] = visual_cost[0, 0]
    expected_advance = (template_count - 1) / (frame_count - 1)
    max_advance = max(4, int(np.ceil(expected_advance * 3.0)) + 2)
    tempo_weight = 1e-4
    # Ascending predecessor indices preserve the scalar implementation's
    # argmin tie order. Vectorize only the independent template columns;
    # candidate-time rows retain the same recurrence and endpoint constraint.
    columns = np.arange(template_count, dtype=np.int32)
    predecessors = columns[:, None] - np.arange(max_advance, -1, -1, dtype=np.int32)
    valid = predecessors >= 0
    safe = np.maximum(predecessors, 0)
    penalties = tempo_weight * np.abs(columns[:, None] - predecessors - expected_advance)
    for frame_index in range(1, frame_count):
        choices = costs[frame_index - 1, safe] + penalties
        choices[~valid] = infinite
        offsets = np.argmin(choices, axis=1)
        best = choices[columns, offsets]
        reachable = np.isfinite(best)
        costs[frame_index, reachable] = visual_cost[frame_index, reachable] + best[reachable]
        parents[frame_index, reachable] = predecessors[columns, offsets][reachable]

    if not np.isfinite(costs[-1, -1]):
        labels, dynamic_errors, full_errors = _template_distances(frames, templates, mask)
        return labels, dynamic_errors, full_errors, direct_labels
    labels = [0] * frame_count
    labels[-1] = template_count - 1
    for frame_index in range(frame_count - 1, 0, -1):
        labels[frame_index - 1] = int(parents[frame_index, labels[frame_index]])
    dynamic_errors = [float(dynamic[index, label]) for index, label in enumerate(labels)]
    full_errors = [float(full[index, label]) for index, label in enumerate(labels)]
    recognized = [
        label if dynamic_error <= MOTION_DYNAMIC_ERROR and full_error <= MOTION_FULL_ERROR else -1
        for label, dynamic_error, full_error in zip(labels, dynamic_errors, full_errors)
    ]
    return recognized, dynamic_errors, full_errors, direct_labels


def _ordered_milestone_indices(
    stable: Sequence[np.ndarray],
    templates: Sequence[np.ndarray],
    mask: np.ndarray,
) -> tuple[list[int], list[float]]:
    """Locate commits in order, disambiguating later repeated states."""

    template_float = [image.astype(np.float32) for image in templates]
    indices: list[int] = []
    errors: list[float] = []
    cursor = 0
    for stable_image in stable:
        target = stable_image.astype(np.float32)
        candidates = [
            float(np.mean(np.abs(target - template)[mask])) / 255.0
            for template in template_float[cursor:]
        ]
        if not candidates:
            return indices, [*errors, float("inf")]
        relative = int(np.argmin(candidates))
        index = cursor + relative
        indices.append(index)
        errors.append(float(candidates[relative]))
        cursor = index + 1
    return indices, errors


def _ordered_subsequence(required: Sequence[int], observed: Sequence[int], *, tolerance: int = 2) -> bool:
    cursor = 0
    for target in required:
        while cursor < len(observed) and abs(observed[cursor] - target) > tolerance:
            cursor += 1
        if cursor >= len(observed):
            return False
        cursor += 1
    return True


def _ordered_local_motion_landmarks(
    frames: Sequence[np.ndarray],
    stable: Sequence[np.ndarray],
    motion: Sequence[np.ndarray],
    milestones: Sequence[int],
) -> tuple[list[bool], list[float], list[int], list[float], list[float]]:
    """Match per-action interior motion on a local, non-diluted pixel mask.

    A family-wide union mask lets a tiny moved object disappear among all
    pixels changed by a long plan.  Per-action masks make hard cuts, global
    crossfades, and reversed clips pay for the exact object trajectory.
    """

    checks: list[bool] = []
    errors: list[float] = []
    matched_frames: list[int] = []
    thresholds: list[float] = []
    distinctiveness_values: list[float] = []
    cursor = 0
    for action_index, (start, end) in enumerate(zip(milestones, milestones[1:])):
        span = end - start
        if span < 2:
            continue
        local_images = [stable[action_index], *motion[start : end + 1], stable[action_index + 1]]
        local_mask = _dynamic_mask(local_images, shape=stable[0].shape[:2])
        start_image = stable[action_index].astype(np.float32)
        end_image = stable[action_index + 1].astype(np.float32)
        distinctive: list[tuple[int, float]] = []
        for template_index in range(start + 1, end):
            template = motion[template_index].astype(np.float32)
            start_error = float(np.mean(np.abs(template - start_image)[local_mask])) / 255.0
            end_error = float(np.mean(np.abs(template - end_image)[local_mask])) / 255.0
            distinctiveness = min(start_error, end_error)
            if distinctiveness >= 0.018:
                distinctive.append((template_index, distinctiveness))
        if not distinctive:
            continue
        positions = sorted({
            min(len(distinctive) - 1, int(round((len(distinctive) - 1) * fraction)))
            for fraction in (0.20, 0.50, 0.80)
        })
        landmarks = [distinctive[position] for position in positions]
        for landmark_index, distinctiveness in landmarks:
            expected = motion[landmark_index].astype(np.float32)
            threshold = max(0.020, min(0.035, distinctiveness * 0.45))
            found = -1
            best_error = float("inf")
            # Greedy earliest acceptable match preserves action and phase order.
            for frame_index in range(cursor, len(frames)):
                difference = np.abs(frames[frame_index].astype(np.float32) - expected)
                error = float(np.mean(difference[local_mask])) / 255.0
                best_error = min(best_error, error)
                if error <= threshold:
                    found = frame_index
                    break
            passed = found >= 0
            checks.append(passed)
            errors.append(best_error)
            matched_frames.append(found)
            thresholds.append(threshold)
            distinctiveness_values.append(distinctiveness)
            if passed:
                cursor = found + 1
    return checks, errors, matched_frames, thresholds, distinctiveness_values


def _bookshelf_unselected_preservation(frames, offsets, labels, stable, milestones):
    """Keep non-background support for every source-unselected book body.

    Books share one native panel background and have solid coloured bodies.
    Actual foreground may cover an unselected body during a legal occlusion;
    exposed background cannot. This reads candidate evidence without matching
    a rendered moving pose or forcing noisy pixels into exclusive colour IDs.
    """
    import cv2
    from data.common.pipeline.object_plans import ITEM_PALETTE
    from data.common.verifiers.region_gates import _background

    background = _background(stable[0]).astype(np.float32)
    palette = np.asarray(ITEM_PALETTE, dtype=np.float32)

    def colour_labels(image, tolerance):
        # Palette classification is an observation of the supplied pixels.
        # There is no candidate repair or fitted colour/geometry mapping.
        distances = np.stack([np.linalg.norm(image.astype(np.float32)-colour, axis=2)
                              for colour in palette])
        closest = distances.argmin(axis=0)
        return np.where(distances.min(axis=0) <= tolerance, closest, -1)

    source = [colour_labels(image, 25) for image in stable]
    supports = []
    for before, after in zip(source, source[1:]):
        static = (before == after) & (before >= 0)
        supports.append((before, static))
    failed = []
    progress = 0
    for index, (frame, offset, label) in enumerate(zip(frames, offsets, labels)):
        if label >= 0:
            progress = label
        action = min(len(supports)-1, max(0, int(np.searchsorted(milestones, progress, side="right"))-1))
        before, static = supports[action]
        foreground = np.linalg.norm(frame.astype(np.float32)-offset-background, axis=2) >= 40
        cover = cv2.dilate(foreground.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        for colour in np.unique(before[static]):
            required = cv2.erode(((before == colour) & static).astype(np.uint8),
                                 np.ones((3, 3), np.uint8)) > 0
            if int(required.sum()) < 120:
                continue
            count, members, stats, _centers = cv2.connectedComponentsWithStats(required.astype(np.uint8))
            damaged = any(
                stats[member, cv2.CC_STAT_AREA] >= 120
                and np.count_nonzero((members == member) & ~cover)
                    > max(24, .25*int(stats[member, cv2.CC_STAT_AREA]))
                for member in range(1, count)
            )
            if damaged:
                failed.append(index)
                break
    return {"video_bookshelf_unselected_preservation_pass": not failed,
            "video_bookshelf_unselected_failed_frames": failed[:24]}, not failed


def _video_semantics(
    candidate_path: Optional[Path],
    stable_full: Sequence[np.ndarray],
    motion_full: Sequence[np.ndarray],
    *,
    task_name: str,
    decoded_video: Optional[tuple[Sequence[np.ndarray], float]] = None,
    observed_terminal_pass: Optional[bool] = None,
) -> tuple[dict[str, Any], bool]:
    process_profile = "rearrangement_source_identity_contact_commit_v1"
    if candidate_path is None:
        return {
            "video_process_verifier": process_profile,
            "video_process_available": False,
            "video_media_contract_valid": False,
            "video_process_pass": False,
        }, False
    try:
        if len(stable_full) < 2 or len(motion_full) < 5:
            raise ValueError("source replay lacks semantic commits or continuous motion")
        frames_full, fps = _read_video(candidate_path) if decoded_video is None else decoded_video
        comparison_shape = (64, 64)
        stable = [_resize(image, comparison_shape) for image in stable_full]
        motion = [_resize(image, comparison_shape) for image in motion_full]
        frames = [_resize(image, comparison_shape) for image in frames_full]
        mask = _dynamic_mask([*stable, *motion], shape=comparison_shape)
        # Same reason as the terminal image above: the landmark threshold below
        # floors at the renderer's own reproduction error, so a uniform tone shift
        # - which carries no spatial information at all - would otherwise consume
        # the whole allowance.  Removing it here leaves every structured
        # difference, including the ones the landmark gate exists to find.
        static_region = ~mask
        offsets = [
            tone_offset(frame, stable[0], static_region if np.any(static_region) else None)
            for frame in frames
        ]
        frames = [frame.astype(np.float32) - offset for frame, offset in zip(frames, offsets)]
        labels, dynamic_errors, full_errors, direct_labels = _monotonic_template_distances(
            frames,
            motion,
            mask,
        )
        recognized_labels = [label for label in labels if label >= 0]
        recognized_fraction = len(recognized_labels) / len(labels)
        direct_pairs = [
            (frame_index, label)
            for frame_index, label in enumerate(direct_labels)
            if label >= 0
        ]
        direct_recognized_fraction = len(direct_pairs) / len(direct_labels)
        if len(direct_pairs) >= 3:
            frame_positions = np.asarray([pair[0] for pair in direct_pairs], dtype=np.float64)
            source_positions = np.asarray([pair[1] for pair in direct_pairs], dtype=np.float64)
            if float(np.std(frame_positions)) > 0 and float(np.std(source_positions)) > 0:
                direct_progress_correlation = float(np.corrcoef(frame_positions, source_positions)[0, 1])
            else:
                direct_progress_correlation = 0.0
        else:
            direct_progress_correlation = 0.0
        direct_sequence = [pair[1] for pair in direct_pairs]
        direct_backwards_fraction = (
            sum(current + 2 < previous for previous, current in zip(direct_sequence, direct_sequence[1:]))
            / max(1, len(direct_sequence) - 1)
        )
        min_direct_correlation, max_direct_backwards = DIRECT_TIME_ORDER_TOLERANCES.get(
            task_name, DIRECT_TIME_ORDER_DEFAULT
        )
        revisited_commits = len({image.tobytes() for image in stable}) < len(stable)
        # A repeated legal arrangement maps to the first identical direct
        # template regardless of its occurrence in the program. Ordered commit
        # coverage, local motion landmarks and monotonic alignment disambiguate
        # those source-visible revisits; direct nearest-label order cannot.
        direct_time_order_pass = bool(
            direct_recognized_fraction >= VIDEO_MIN_RECOGNIZED_FRACTION
            and (revisited_commits or (
                direct_progress_correlation >= min_direct_correlation
                and direct_backwards_fraction <= max_direct_backwards
            ))
        )
        milestones, milestone_render_errors = _ordered_milestone_indices(stable, motion, mask)
        milestones_available = bool(
            len(milestones) == len(stable)
            and milestones == sorted(milestones)
            and len(set(milestones)) == len(milestones)
            and max(milestone_render_errors, default=1.0) <= 0.02
        )
        coverage = milestones_available and _ordered_subsequence(milestones, recognized_labels, tolerance=3)

        # Motion must visit the interior of every pick/slide/contact interval;
        # a hard cut or a sequence of static screenshots cannot satisfy this.
        interior_checks: list[bool] = []
        for start, end in zip(milestones, milestones[1:]):
            low = start + max(1, int((end - start) * 0.20))
            high = end - max(1, int((end - start) * 0.20))
            interior_checks.append(
                low <= high and any(low <= label <= high for label in recognized_labels)
            )
        contact_motion_complete = bool(interior_checks and all(interior_checks))
        (
            local_landmark_checks,
            local_landmark_errors,
            local_landmark_frames,
            local_landmark_thresholds,
            local_landmark_distinctiveness,
        ) = (
            _ordered_local_motion_landmarks(frames, stable, motion, milestones)
            if milestones_available
            else ([], [], [], [], [])
        )
        # Some tiny motions have no source frame sufficiently distinct from
        # both adjacent commits at comparison resolution.  Those actions are
        # covered by direct time order and commit checks; available landmarks
        # remain hard requirements.
        local_motion_complete = bool(all(local_landmark_checks))

        # Ignore unrecognized frames for order only after bounding them to 15%;
        # the former 80/90% loophole is therefore impossible.
        backwards = [
            previous - current
            for previous, current in zip(recognized_labels, recognized_labels[1:])
            if current + 2 < previous
        ]
        monotonic = not backwards
        progress_span = bool(
            recognized_labels
            and recognized_labels[0] <= max(2, milestones[0] + 2)
            and recognized_labels[-1] >= min(len(motion) - 3, milestones[-1] - 2)
        )
        def stable_match(frame: np.ndarray, expected: np.ndarray) -> bool:
            difference = np.abs(frame.astype(np.float32) - expected.astype(np.float32))
            return bool(
                float(np.mean(difference[mask])) / 255.0 <= MOTION_DYNAMIC_ERROR
                and float(np.mean(difference)) / 255.0 <= MOTION_FULL_ERROR
            )

        # A hold consists of visually stable source/terminal frames.  Do not
        # key it to one timeline index: the source renderer intentionally
        # repeats the same terminal image over several distinct template
        # indices, and legal processes may revisit an earlier visible state.
        input_hold = sum(
            stable_match(frame, stable[0]) for frame in frames[: min(8, len(frames))]
        ) >= 2
        final_hold = sum(
            stable_match(frame, stable[-1]) for frame in frames[max(0, len(frames) - 8) :]
        ) >= 2

        static_mask = ~mask
        if np.any(static_mask):
            initial = stable[0].astype(np.float32)
            static_error = max(
                float(np.mean(np.abs(frame.astype(np.float32) - initial)[static_mask])) / 255.0
                for frame in frames
            )
        else:
            static_error = 0.0
        full_static_mask = ~_dynamic_mask([*stable_full, *motion_full], shape=(512,512))
        static_region_error = max(
            (static_region_errors(frame.astype(np.float32)-offset, stable_full[0], full_static_mask)[0]
             for frame,offset in zip(frames_full,offsets)),
            default=1.0,
        ) if np.any(full_static_mask) else 0.0
        object_preservation_metrics, object_preservation_ok = (
            _bookshelf_unselected_preservation(frames_full, offsets, labels, stable_full, milestones)
            if task_name == "bookshelf_sort_by_height" and milestones_available else ({}, True)
        )
        static_preserved = bool(static_error <= 0.055 and static_region_error <= STATIC_REGION_FLOOR
                                and object_preservation_ok)

        dynamic_changes = [
            float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))[mask])) / 255.0
            for left, right in zip(frames, frames[1:])
        ]
        expected_changes = [
            float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))[mask])) / 255.0
            for left, right in zip(motion, motion[1:])
        ]
        expected_max_change = max(expected_changes, default=0.0)
        max_candidate_change = max(dynamic_changes, default=1.0)
        no_hard_jump = bool(
            dynamic_changes
            and max_candidate_change <= max(0.015, expected_max_change * 1.60 + 0.004)
        )
        source_activity_threshold = max(0.0015, expected_max_change * 0.08)
        source_active_fraction = (
            sum(change >= source_activity_threshold for change in expected_changes)
            / max(1, len(expected_changes))
        )
        candidate_active_fraction = (
            sum(change >= source_activity_threshold for change in dynamic_changes)
            / max(1, len(dynamic_changes))
        )
        # Repeating legal holds changes an activity fraction without removing
        # any action. Require observable transitions per semantic action; the
        # local landmarks still enforce each action's complete interior.
        required_active_changes = max(2, len(stable) - 1)
        motion_activity_pass = sum(
            change >= source_activity_threshold for change in dynamic_changes
        ) >= required_active_changes

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            terminal_path = Path(handle.name)
        try:
            Image.fromarray(frames_full[-1].astype(np.uint8), mode="RGB").save(terminal_path)
            terminal_metrics, terminal_pass = _terminal_image_semantics(terminal_path, stable_full)
        finally:
            terminal_path.unlink(missing_ok=True)

        if observed_terminal_pass is not None:
            terminal_pass = bool(observed_terminal_pass)
            terminal_metrics["image_terminal_state_source"] = "observed_task_object_identities"
            terminal_metrics["image_terminal_state_pass"] = terminal_pass

        dimensions_valid = all(frame.shape[:2] == (CANVAS_SIZE, CANVAS_SIZE) for frame in frames_full)
        fps_valid = abs(fps - VIDEO_FPS) <= VIDEO_FPS_TOLERANCE
        frame_pattern_valid = (len(frames_full) - 1) % 4 == 0
        duration = len(frames_full) / fps if fps > 0 else None
        duration_valid = bool(duration is not None and duration > 0)
        media_valid = bool(
            dimensions_valid
            and fps_valid
            and frame_pattern_valid
            and duration_valid
        )
        process_pass = bool(
            recognized_fraction >= VIDEO_MIN_RECOGNIZED_FRACTION
            and coverage
            and contact_motion_complete
            and local_motion_complete
            and direct_time_order_pass
            and monotonic
            and progress_span
            and static_preserved
            and terminal_pass
            and no_hard_jump
            and motion_activity_pass
            and input_hold
            and final_hold
            and media_valid
        )
        return {
            "video_process_verifier": process_profile,
            "video_process_available": True,
            "video_expected_commit_count": len(stable),
            "video_source_motion_template_count": len(motion),
            "video_commit_template_indices": milestones,
            "video_commit_template_errors": milestone_render_errors,
            "video_commit_templates_available": milestones_available,
            "video_recognized_frame_fraction": recognized_fraction,
            "video_min_recognized_frame_fraction": VIDEO_MIN_RECOGNIZED_FRACTION,
            "video_unrecognized_frame_fraction": 1.0 - recognized_fraction,
            "video_max_unrecognized_frame_fraction": 1.0 - VIDEO_MIN_RECOGNIZED_FRACTION,
            "video_direct_recognized_frame_fraction": direct_recognized_fraction,
            "video_direct_progress_correlation": direct_progress_correlation,
            "video_direct_backwards_fraction": direct_backwards_fraction,
            "video_direct_time_order_pass": direct_time_order_pass,
            "video_source_revisited_commit_states": revisited_commits,
            "video_complete_ordered_commit_coverage": coverage,
            "video_contact_interval_checks": interior_checks,
            "video_identity_contact_commit_complete": contact_motion_complete,
            "video_local_motion_landmark_checks": local_landmark_checks,
            "video_local_motion_landmark_errors": local_landmark_errors,
            "video_local_motion_landmark_frames": local_landmark_frames,
            "video_local_motion_landmark_thresholds": local_landmark_thresholds,
            "video_local_motion_landmark_distinctiveness": local_landmark_distinctiveness,
            "video_local_motion_complete": local_motion_complete,
            "video_progress_monotonic": monotonic,
            "video_progress_spans_source_to_goal": progress_span,
            "video_input_hold": input_hold,
            "video_final_hold": final_hold,
            "video_static_preservation_error": static_error,
            "video_static_region_max_error": static_region_error,
            "video_static_preserved": static_preserved,
            **object_preservation_metrics,
            "video_max_adjacent_dynamic_change": max_candidate_change,
            "video_source_max_adjacent_dynamic_change": expected_max_change,
            "video_no_hard_jump": no_hard_jump,
            "video_source_active_transition_fraction": source_active_fraction,
            "video_candidate_active_transition_fraction": candidate_active_fraction,
            "video_motion_activity_pass": motion_activity_pass,
            "video_required_active_changes": required_active_changes,
            "video_frame_dynamic_errors": dynamic_errors,
            "video_frame_full_errors": full_errors,
            "video_terminal_metrics": terminal_metrics,
            "video_terminal_state_pass": terminal_pass,
            "video_frame_count": len(frames_full),
            "video_fps": fps,
            "video_duration_seconds": duration,
            "video_dimensions_valid": dimensions_valid,
            "video_fps_valid": fps_valid,
            "video_frame_count_pattern_valid": frame_pattern_valid,
            "video_duration_valid": duration_valid,
            "video_media_contract_valid": media_valid,
            "video_process_pass": process_pass,
        }, process_pass
    except Exception as exc:
        return {
            "video_process_verifier": process_profile,
            "video_process_available": False,
            "video_media_contract_valid": False,
            "video_process_pass": False,
            "video_failure_reason": str(exc),
        }, False


# These frozen TI2V contracts explicitly require a shortest/minimum program.
# Typed-text training answers do not add requirements to a video submission.
_SHORTEST_VIDEO_PLAN_TASKS = frozenset({
    "jumping_frog_swap", "maze_rush_hour", "meal_train_pass_serve",
    "office_inbox_priority_pull", "pipeline_hold_cells_3", "sliding_tile_15",
    "train_couple_swap",
})


def _bookshelf_observed_plan(
    sample: CanonicalSample,
    frames: Sequence[np.ndarray],
    shortest_length: Optional[int],
) -> tuple[Optional[list[tuple[int, int]]], dict[str, Any]]:
    """Read complete shelf arrangements, then replay the observed swaps.

    Swaps on disjoint slots commute, and the public video contract accepts
    complete legal programs. Colour probes propose identities; a full rendered-state
    check confirms each proposal, so a book merely crossing a probe during
    continuous motion cannot fabricate a commit.
    """
    from data.rearrangement_game.bookshelf_sort_by_height.generator import (
        BookshelfSortByHeightGenerator,
    )

    record = source_record(sample)
    initial = tuple(record["initial_state"])
    target = tuple(record["target_state"])
    with tempfile.TemporaryDirectory(prefix="videogym-shelf-observation-") as tmp:
        generator = BookshelfSortByHeightGenerator(output_dir=tmp, video=False)
        panel = generator._panel_geometry()
        layout = generator._layout(panel["panel_w"], panel["panel_h"], len(initial))
        palette = np.asarray([generator._book_color(value) for value in initial], dtype=np.float32)
        centers = [
            (int(panel["left_panel_x0"] + layout["margin_x"] + layout["slot_w"] * (slot + .5)),
             int(panel["left_panel_y0"] + layout["shelf_y"] - layout["base_h"] * .5))
            for slot in range(len(initial))
        ]
        from data.common.prompt_paraphrase import active_profile
        if active_profile("height_columns"):
            centers=[]
            for slot in range(len(initial)):
                source_x=layout["margin_x"]+layout["slot_w"]*(slot+.5)
                cx,cy,base,scale,angle=generator._carousel_geometry(layout,source_x)
                probe=base+layout["base_h"]*scale*.5
                centers.append((int(panel["left_panel_x0"]+cx+probe*math.cos(angle)),
                                int(panel["left_panel_y0"]+cy+probe*math.sin(angle))))
        states: list[tuple[int, ...]] = []
        commit_frames: list[int] = []
        terminal_readings: list[bool] = []
        cache: dict[tuple[int, ...], np.ndarray] = {}
        for index, frame in enumerate(frames):
            if index >= len(frames) - 2:
                terminal_readings.append(False)
            identities = []
            for x, y in centers:
                observed = np.median(frame[y-2:y+3, x-2:x+3], axis=(0, 1))
                distances = np.linalg.norm(palette - observed, axis=1)
                if float(distances.min()) > 45:
                    break
                identities.append(initial[int(np.argmin(distances))])
            state = tuple(identities)
            if len(state) != len(initial) or set(state) != set(initial):
                continue
            if state not in cache:
                cache[state] = np.asarray(generator._render_static_scene(state, target), dtype=np.uint8)
            if float(region_error_map(frame, cache[state]).max()) > REGION_ERROR_CEILING:
                continue
            if index >= len(frames) - 2 and state == target:
                # Slot colors identify every book; its own top edge verifies
                # height without shrinking the allowance with the final swap.
                # At larger n that swap can change only two similar heights.
                heights_ok = True
                if not active_profile("height_columns"):
                    shelf = int(panel["left_panel_y0"] + layout["shelf_y"])
                    gap = (layout["max_h"] - layout["base_h"]) / max(1, len(initial) - 1)
                    tolerance = max(3., .30 * gap)
                    for slot, height_id in enumerate(target):
                        x, _ = centers[slot]
                        half_width = max(1, int(layout["book_w"] * .25))
                        region = frame[int(panel["left_panel_y0"]):shelf,
                                       x-half_width:x+half_width+1].astype(float)
                        color = np.asarray(generator._book_color(height_id))
                        colored_rows = np.any(np.linalg.norm(region - color, axis=2) <= 45, axis=1)
                        ys = np.flatnonzero(colored_rows)
                        top = (int(panel["left_panel_y0"]) + int(ys[0])) if len(ys) else None
                        expected_top = shelf - generator._book_height(layout, height_id)
                        heights_ok &= top is not None and abs(top - expected_top) <= tolerance
                terminal_readings[-1] = bool(heights_ok)
            if not states or state != states[-1]:
                states.append(state)
                commit_frames.append(index)
    metrics = {
        "video_observed_terminal_state_pass": len(terminal_readings) == 2 and all(terminal_readings),
        "video_observed_terminal_state_source": "slot_identities_heights_and_full_rendered_regions",
        "video_observed_plan_parser": "bookshelf_slot_identity_commits_v1",
        "video_observed_commit_states": [list(state) for state in states],
        "video_observed_commit_frames": commit_frames,
    }
    plan: list[tuple[int, int]] = []
    valid = bool(states and states[0] == initial and states[-1] == target)
    for before, after in zip(states, states[1:]):
        changed = [slot for slot, (a, b) in enumerate(zip(before, after)) if a != b]
        if len(changed) != 2:
            valid = False
            break
        a, b = changed
        if before[a] != after[b] or before[b] != after[a]:
            valid = False
            break
        plan.append((a, b))
    valid = bool(valid and (shortest_length is None or len(plan) == shortest_length))
    metrics["video_observed_plan_legal"] = valid
    metrics["video_observed_plan_requires_shortest"] = shortest_length is not None
    metrics["video_observed_plan"] = [list(move) for move in plan]
    return (plan if valid else None), metrics


_OBSERVED_OBJECT_PLAN_TASKS = frozenset({
    "crane_block_stack", "dock_yard_crane_swap", "filing_cabinet_alphabet_lock",
    "pancake_flip", "stack_reversal_minimal_moves", "tower_of_hanoi",
    "chess_pawn_swap", "color_group_with_locks",
    "parking_2d_n_by_n", "parking_lot_swap_minimal_moves", "token_trade_matrix",
    "bus_stops_dispatch", "container_ship_loading_order",
    "lever_crane_swap", "office_inbox_priority_pull", "pipeline_hold_cells_3",
    "train_couple_swap", "vending_dispense_fifo",
})


def _object_terminal_members(sample, frames, terminal):
    """Read each colored disk/block in its declared goal position."""
    import cv2
    from data.rearrangement_game.core.source_replay import _new_generator

    generator = _new_generator(sample.task_name, Path(tempfile.gettempdir()), video=False)
    record = source_record(sample)
    target = record["target_state"]
    identities = sorted(value for column in target for value in column)
    color_for = generator._block_color if sample.task_name == "crane_block_stack" else generator._disk_color
    panel = generator._panel_geometry()
    left, top = int(panel["left_panel_x0"]), int(panel["left_panel_y0"])
    width, height = int(panel["panel_w"]), int(panel["panel_h"])
    expected = terminal[top:top+height, left:left+width].astype(float)
    tolerance = np.ones((5, 5), np.uint8)
    readings = []
    palette = np.asarray([color_for(identity) for identity in identities])
    expected_distances = np.linalg.norm(expected[:, :, None, :] - palette, axis=3)
    expected_labels = expected_distances.argmin(axis=2)
    for frame in frames[-2:]:
        candidate = frame[top:top+height, left:left+width].astype(float)
        distances = np.linalg.norm(candidate[:, :, None, :] - palette, axis=3)
        labels = distances.argmin(axis=2)
        for index, identity in enumerate(identities):
            support = (expected_labels == index) & (expected_distances[:, :, index] <= 12)
            observed = (labels == index) & (distances[:, :, index] <= 45)
            near_observed = cv2.dilate(observed.astype(np.uint8), tolerance).astype(bool)
            near_expected = cv2.dilate(support.astype(np.uint8), tolerance).astype(bool)
            coverage = float(near_observed[support].mean()) if support.any() else 0.
            precision = float(near_expected[observed].mean()) if observed.any() else 0.
            readings.append({"identity": identity, "coverage": coverage, "precision": precision,
                             "pass": coverage >= .65 and precision >= .65})
    passed = len(frames) >= 2 and bool(readings) and all(row["pass"] for row in readings)
    return {"video_terminal_object_members_pass": passed,
            "video_terminal_object_member_readings": readings}, passed


def _object_candidate_moves(task: str, state: Any) -> list[Any]:
    if task == "chess_pawn_swap":
        from data.rearrangement_game.chess_pawn_swap.generator import _all_moves
        return list(_all_moves(tuple(state)))
    if task == "token_trade_matrix":
        from data.rearrangement_game.token_trade_matrix.generator import _validate_state
        return sorted(_validate_state(state)[1])
    if task == "parking_2d_n_by_n":
        return [(car, slot) for car in state if car is not None
                for slot, value in enumerate(state) if value is None]
    if task == "parking_lot_swap_minimal_moves":
        return [car for car in state if car is not None]
    if task == "color_group_with_locks":
        return list(range(len(state) - 1))
    if task == "bus_stops_dispatch":
        return ["L", "R"]
    if task == "lever_crane_swap":
        return ["PL", "PR", "DL", "DR"]
    if task == "pipeline_hold_cells_3":
        return [(action, cell) for action in ("push", "pop") for cell in range(1, 4)]
    if task == "train_couple_swap":
        if state["parked"] is None:
            return [("park", car) for car in state["track"]]
        return [("place", slot) for slot in range(len(state["track"]) + 1)]
    if task == "office_inbox_priority_pull":
        from data.rearrangement_game.office_inbox_priority_pull.generator import _legal_successors, _state_key
        key = _state_key(state["inbox"], state["work"], state["buffer"])
        return [move for _state, move in _legal_successors(key)]
    if task == "vending_dispense_fifo":
        return ([("dispense", state["chute"][0])] if state["chute"] else []) + [
            ("reinsert", can) for can in state["tray"]
        ]
    if task == "container_ship_loading_order":
        moves = []
        if state["dock"]:
            moves.extend([(action, state["dock"][-1]) for action in ("ship", "buffer")])
        if state.get("buffer") is not None:
            moves.extend([(action, state["buffer"]) for action in ("ship", "unbuffer")])
        return moves
    if task == "dock_yard_crane_swap":
        return [(crane, a, b) for crane in ("L", "R") for a, b in permutations(range(3), 2)]
    if task == "filing_cabinet_alphabet_lock":
        return list(permutations(state["drawers"], 2))
    if task == "pancake_flip":
        return list(range(2, len(state) + 1))
    return list(permutations(range(len(state)), 2))


def _observe_successor_commits(frames, initial, shortest_length, *, render, successors, goal, merge_move=None):
    """Read complete commits proposed by a source-owned transition system."""
    current = initial
    current_image = np.asarray(render(current), dtype=np.uint8)
    plan: list[Any] = []
    commits = [0]

    def proposals():
        result = []
        before = _resize(current_image, (64, 64)).astype(np.float32)
        for move, state in successors(current):
            image = np.asarray(render(state), dtype=np.uint8)
            reduced = _resize(image, (64, 64)).astype(np.float32)
            mask = _dynamic_mask([before, reduced], shape=(64, 64))
            separation = float(np.mean(visual_delta(before, reduced)[mask])) / 255.0
            if separation > 0:
                result.append((move, state, image, reduced, before, mask))
        return result

    neighbors = proposals()
    for index, frame in enumerate(frames[1:], start=1):
        reduced = _resize(frame, (64, 64)).astype(np.float32)
        candidates = []
        for proposal in neighbors:
            move, state, image, after, before, mask = proposal
            error = float(np.mean(visual_delta(reduced, after)[mask])) / 255.0
            stale = float(np.mean(visual_delta(reduced, before)[mask])) / 255.0
            if error <= MOTION_DYNAMIC_ERROR and error < .45 * stale:
                candidates.append((error, proposal))
        for _error, proposal in sorted(candidates, key=lambda item: item[0]):
            move, state, image, _after, _before, _mask = proposal
            _metrics, committed = _arrangement_region_gates(frame, current_image, image)
            if not committed:
                continue
            current, current_image = state, image
            plan = [*plan, move] if merge_move is None else merge_move(plan, move)
            commits.append(index)
            neighbors = proposals() if shortest_length is None or len(plan) <= shortest_length else []
            break
        if shortest_length is not None and len(plan) > shortest_length:
            break
    valid = bool(goal(current) and (shortest_length is None or len(plan) == shortest_length))
    metrics = {
        "video_observed_plan_parser": "source_legal_successor_commits_v1",
        "video_observed_commit_frames": commits,
        "video_observed_plan": plan,
        "video_observed_plan_legal": valid,
        "video_observed_plan_requires_shortest": shortest_length is not None,
    }
    return (plan if valid else None), metrics


def _object_observed_plan(sample, frames, shortest_length):
    """Recover a complete source-legal program under its frozen video contract."""
    from data.rearrangement_game.core.source_replay import _new_generator

    record = source_record(sample)
    initial = deepcopy(record["initial_state"])
    target = deepcopy(record["target_state"])
    with tempfile.TemporaryDirectory(prefix="videogym-object-observation-") as tmp:
        generator = _new_generator(sample.task_name, Path(tmp), video=False)
        for key, value in record.items():
            if key not in {"id", "initial_state", "target_state"} and hasattr(generator, key):
                setattr(generator, key, deepcopy(value))

        def successors(state):
            result = []
            for move in _object_candidate_moves(sample.task_name, state):
                try:
                    result.append((move, generator._apply_move(deepcopy(state), move)))
                except (ValueError, IndexError):
                    continue
            return result

        def goal(state):
            reached = generator._state_signature(state) == generator._state_signature(target)
            if sample.task_name == "lever_crane_swap":
                reached = bool(state[:2] == target[:2] and (len(state) == 2 or state[2] is None))
            return reached

        return _observe_successor_commits(
            frames, initial, shortest_length,
            render=lambda state: generator._render_static_scene(state, target),
            successors=successors, goal=goal,
        )


def _rush_observed_plan(sample, frames, shortest_length):
    """Read legal fixed-axis vehicle commits, accepting every shortest exit plan."""
    from data.rearrangement_game.core.source_replay import _new_generator, _rush_styled_cars
    from data.rearrangement_game.maze_rush_hour.evaluator import _visible_problem, _successors, _goal

    record = source_record(sample)
    problem = _visible_problem(record)
    if problem is None:
        raise ValueError("rush-hour visible source violates the vehicle contract")
    initial, meta, red_id, exit_row = problem
    source_cars = _rush_styled_cars(record["cars"])
    with tempfile.TemporaryDirectory(prefix="videogym-rush-observation-") as tmp:
        generator = _new_generator(sample.task_name, Path(tmp), video=False)

        def render(state):
            positions = {car_id: (row, col) for car_id, row, col in state}
            cars = [dict(car, row=positions[car["id"]][0], col=positions[car["id"]][1])
                    for car in source_cars]
            return generator._render_lot(
                cars=cars, draw_cell_ids=False, draw_car_badges=True,
                padding=tuple(generator.padding), canvas_dims=tuple(generator.canvas_dimensions),
                exit_row=exit_row, red_extra_east_offset=generator._red_exit_offset_px(cars),
            )

        def merge_slide(plan, move):
            # A continuous multi-cell slide passes through other integer heads.
            # Those observations are not separate moves. Consecutive advance
            # of the same vehicle in one direction is one source-legal slide.
            if plan and all(plan[-1][key] == move[key] for key in ("car_id", "direction")):
                return [*plan[:-1], {**move, "cells": plan[-1]["cells"] + move["cells"]}]
            return [*plan, dict(move)]

        plan, metrics = _observe_successor_commits(
            frames, initial, shortest_length, render=render,
            successors=lambda state: [(move, following) for following, move in _successors(state, meta, red_id)],
            goal=lambda state: _goal(state, red_id), merge_move=merge_slide,
        )
    metrics["video_observed_plan_parser"] = "rush_hour_source_legal_vehicle_commits_v1"
    return plan, metrics


def _sliding_observed_plan(sample, frames, shortest_length):
    """Read the empty-slot trajectory, then replay the actual blank moves.

    The blank has neither a dark tile outline nor a numeral. Reading that
    source-visible occupancy signal avoids treating a generated numeral's
    stroke variation as a different grid state; full motion verification still
    checks every tile identity against the replayed program.
    """
    from data.rearrangement_game.core.source_replay import _new_generator
    from data.common.pipeline.sliding_tiles import _apply_blank_move, _find_blank, _goal_state, _state_key

    record = source_record(sample)
    initial = record["initial_grid"]
    size = int(record["board_size"])
    target = _goal_state(size)
    with tempfile.TemporaryDirectory(prefix="videogym-sliding-observation-") as tmp:
        generator = _new_generator(sample.task_name, Path(tmp), video=False)
        cell, ox, oy = generator._layout(size)
        source = np.asarray(generator._render_board(initial), dtype=np.uint8)

    def ink_counts(frame):
        return np.asarray([
            np.count_nonzero(frame[oy+row*cell+3:oy+(row+1)*cell-3,
                                   ox+col*cell+3:ox+(col+1)*cell-3].min(axis=2) < 170)
            for row in range(size) for col in range(size)
        ])

    source_counts = ink_counts(source)
    ink_limit = max(24., .25*float(np.median(source_counts)))
    positions = []
    commits = []
    for index, frame in enumerate(frames):
        counts = ink_counts(frame)
        order = np.argsort(counts)
        best, second = int(order[0]), int(order[1])
        if counts[best] > ink_limit or counts[best] > .25*max(1, counts[second]):
            continue
        position = divmod(best, size)
        if not positions or positions[-1] != position:
            positions.append(position)
            commits.append(index)
    current = initial
    plan = []
    valid = bool(positions and positions[0] == _find_blank(initial))
    directions = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
    for before, after in zip(positions, positions[1:]):
        if _find_blank(current) != before:
            valid = False
            break
        move = directions.get((after[0]-before[0], after[1]-before[1]))
        if move is None:
            valid = False
            break
        current = _apply_blank_move(current, move)
        plan.append(move)
    valid = bool(valid and _state_key(current) == _state_key(target) and len(plan) == shortest_length)
    return (plan if valid else None), {
        "video_observed_plan_parser": "sliding_tile_source_empty_slot_commits_v1",
        "video_observed_commit_frames": commits,
        "video_observed_blank_positions": [list(position) for position in positions],
        "video_observed_plan": plan,
        "video_observed_plan_legal": valid,
        "video_observed_plan_requires_shortest": True,
    }


def _record_shell(sample: CanonicalSample) -> EvalRecord:
    return EvalRecord(
        sample_id=sample.id,
        task_name=sample.task_name,
        input_asset=sample.primary_input_image,
        task_family=sample.task_family,
        split=sample.split,
    )


def _verify_sample(
    sample: CanonicalSample,
    pred_root: Path,
    *,
    modality: str,
    training_phase: Optional[str],
) -> EvalRecord:
    record = _record_shell(sample)
    if sample.task_family != "rearrangement_game":
        record.error = "Rearrangement verifier received a different task family"
        record.offline_pass = False
        return record
    try:
        if modality != "ti2v":
            raise ValueError("modality must be exactly ti2v")
        source_plan = solve_plan(sample)
        candidate_video = resolve_candidate_video(pred_root, sample.id, sample.split)
        decoded_video = _read_video(candidate_video) if candidate_video is not None else None
        if candidate_video is not None:
            record.prediction_asset = candidate_video.as_posix()
        bookshelf_metrics = {}
        if sample.task_name == "bookshelf_sort_by_height" and decoded_video is not None:
            observed_plan, bookshelf_metrics = _bookshelf_observed_plan(sample, decoded_video[0], None)
            if observed_plan is not None:
                source_plan = observed_plan
        source_stable, source_motion, _ = replay_plan(sample, source_plan, include_motion=True)
        object_terminal = None
        object_terminal_metrics = {}
        if sample.task_name in {"crane_block_stack", "tower_of_hanoi"} and decoded_video is not None:
            object_terminal_metrics, object_terminal = _object_terminal_members(
                sample, decoded_video[0], source_stable[-1])
        video_metrics, video_pass = _video_semantics(
            candidate_video, source_stable, source_motion,
            task_name=sample.task_name, decoded_video=decoded_video,
            observed_terminal_pass=bookshelf_metrics.get("video_observed_terminal_state_pass", object_terminal),
        )
        video_metrics.update(bookshelf_metrics)
        video_metrics.update(object_terminal_metrics)
        shortest_required = sample.task_name in _SHORTEST_VIDEO_PLAN_TASKS
        observer_tasks = {"maze_rush_hour", "sliding_tile_15",
                          "meal_train_pass_serve", "jumping_frog_swap", *_OBSERVED_OBJECT_PLAN_TASKS}
        if ((not video_pass or shortest_required) and decoded_video is not None
                and sample.task_name in observer_tasks):
            from data.rearrangement_game.core.native_program_observation import observe_meal_program, observe_frog_program
            observer = {"meal_train_pass_serve": observe_meal_program,
                        "jumping_frog_swap": observe_frog_program,
                        "maze_rush_hour": _rush_observed_plan,
                        "sliding_tile_15": _sliding_observed_plan}.get(sample.task_name, _object_observed_plan)
            required_length = len(source_plan) if shortest_required else None
            observed_plan, observation_metrics = observer(sample, decoded_video[0], required_length)
            matched_source_pass = video_pass
            if shortest_required:
                # Fuzzy template proximity alone cannot count actions: a
                # redundant legal excursion may stay close to an early pose.
                # These public contracts require the observed program count.
                video_pass = False
            if observed_plan is not None:
                if observed_plan == source_plan:
                    observed_metrics, observed_pass = video_metrics, matched_source_pass
                else:
                    stable, motion, _ = replay_plan(sample, observed_plan, include_motion=True)
                    observed_metrics, observed_pass = _video_semantics(
                        candidate_video, stable, motion, task_name=sample.task_name,
                        decoded_video=decoded_video,
                        observed_terminal_pass=object_terminal,
                    )
                observation_metrics["video_observed_plan_process_pass"] = observed_pass
                if observed_pass or shortest_required:
                    video_metrics, video_pass = observed_metrics, observed_pass
            video_metrics.update(observation_metrics)
            video_metrics["video_process_pass"] = video_pass


        binary = {
            "ti2v": video_pass,
        }[modality]
        raw_rewards = {
            "whole_video_process": float(video_pass) if modality == "ti2v" else 0.0,
            "video_final_state": float(video_metrics.get("video_terminal_state_pass", False)) if modality == "ti2v" else 0.0,
            "media_contract": float(
                video_metrics.get("video_media_contract_valid", False)
            ),
        }
        availability = {
            "whole_video_process": modality == "ti2v",
            "video_final_state": modality == "ti2v",
            "media_contract": True,
        }
        record.offline_metrics.update({
            **video_metrics,
            "selected_modality": modality,
            "video_pass": video_pass,
            "raw_reward_vector": raw_rewards,
            "reward_availability_mask": availability,
            "candidate_cot_inspected": False,
            "saved_oracle_fields_used": [],
            "ground_truth_used": False,
            "reference_artifacts_used": False,
            "source_only_plan_recomputed": True,
            "binary_correct": bool(binary),
        })
        record.offline_pass = bool(binary)
        if not binary:
            record.error = f"formal {modality.upper()} delivery failed source-only semantic verification"
    except Exception as exc:
        record.offline_pass = False
        record.error = str(exc)
        record.offline_metrics.update({
            "selected_modality": modality,
            "candidate_cot_inspected": False,
            "saved_oracle_fields_used": [],
            "ground_truth_used": False,
            "reference_artifacts_used": False,
            "source_only_plan_recomputed": False,
            "binary_correct": False,
            "failure_reason": str(exc),
        })
    return record


def verify_rearrangement(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: str,
    training_phase: Optional[str] = None,
) -> list[EvalRecord]:
    """Verify exactly the caller-declared formal delivery modality."""

    normalized = str(modality or "").strip().lower()
    if normalized not in {"ti2v"}:
        raise ValueError("modality must be exactly ti2v")
    return [
        _verify_sample(
            sample,
            pred_root,
            modality=normalized,
            training_phase=training_phase,
        )
        for sample in samples
    ]


__all__ = ["verify_rearrangement"]
