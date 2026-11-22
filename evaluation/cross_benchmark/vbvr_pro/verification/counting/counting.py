"""Exact inventories and complete one-at-a-time counting observations."""

from collections import Counter

import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _layers, _mask, _result, _static
from evaluation.cross_benchmark.vbvr_pro.vision.text import read_count
from evaluation.cross_benchmark.vbvr_pro.vision.masks import ink, mask_coverage


def _glyph_template_matches(actual, expected):
    """Match each literal glyph; a correct prefix cannot dilute a wrong digit."""

    def spans(mask):
        columns = np.flatnonzero(np.any(mask, axis=0))
        if not len(columns):
            return []
        boundaries = np.r_[0, np.flatnonzero(np.diff(columns) > 1) + 1, len(columns)]
        return [(int(columns[a]), int(columns[b - 1]) + 1) for a, b in zip(boundaries[:-1], boundaries[1:])]

    given, observed = spans(expected), spans(actual)
    if len(given) != len(observed):
        return False
    for (left, right), (actual_left, actual_right) in zip(given, observed):
        ys = np.flatnonzero(np.any(expected[:, left:right], axis=1))
        height = int(ys[-1] - ys[0] + 1)
        radius = max(1, round(height * 0.025))
        if max(abs(left - actual_left), abs(right - actual_right)) > radius + 1:
            return False
        xa, xb = max(0, min(left, actual_left) - 1), min(expected.shape[1], max(right, actual_right) + 1)
        if min(mask_coverage(actual[:, xa:xb], expected[:, xa:xb], radius=radius)) < 0.97:
            return False
    return True


def _stationary_scene(frames, expected, allowed):
    """Preserve complete RGB observations at a one-pixel codec edge phase.

    Both directions need an actual nearby RGB pixel. A deleted border cannot
    be excused just because the remaining fill matches another source pixel.
    """
    reference = cv2.GaussianBlur(expected, (3, 3), 0).astype(np.int16)
    height, width = expected.shape[:2]
    worst = 0
    for frame in frames:
        actual = cv2.GaussianBlur(frame, (3, 3), 0).astype(np.int16)
        changed = (np.max(np.abs(actual - reference), axis=2) > 65) & ~allowed
        ys, xs = np.nonzero(changed)
        if not len(xs):
            continue
        forward = np.full(len(xs), 32767, np.int16)
        backward = forward.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = np.clip(ys + dy, 0, height - 1), np.clip(xs + dx, 0, width - 1)
                forward = np.minimum(forward, np.max(np.abs(actual[ys, xs] - reference[yy, xx]), axis=1))
                backward = np.minimum(backward, np.max(np.abs(reference[ys, xs] - actual[yy, xx]), axis=1))
        unexpected = np.zeros((height, width), np.uint8)
        unexpected[ys, xs] = (forward > 65) | (backward > 65)
        _, _, stats, _ = cv2.connectedComponentsWithStats(unexpected, 8)
        worst = max(worst, int(np.max(stats[1:, cv2.CC_STAT_AREA], initial=0)))
    return worst <= 30, worst


def _majority(frames, first, final, metadata):
    p = metadata["parameters"]
    counts = Counter(tuple(obj["color"]) for obj in p["objects"])
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ValueError("Majority-color task has a tied maximum")
    color, count = ranked[0]
    if list(color) != p["majority_color"] or count != p["majority_count"]:
        raise ValueError("Majority annotation contradicts independently counted input objects")
    keep = _mask(final, color)
    if np.count_nonzero(keep) < 50:
        raise ValueError("Reference retained objects unavailable")
    # Check each connected retained object, rather than allowing a large object
    # or a white canvas to dilute deletion of a small one.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8))
    minority_colors = [c for c in counts if c != color]
    initial_minority = np.zeros(first.shape[:2], np.uint8)
    for minority in minority_colors:
        initial_minority |= _mask(first, minority).astype(np.uint8)
    occluded = cv2.dilate(initial_minority, np.ones((9, 9), np.uint8)).astype(bool)
    checks = {}
    for index in range(1, n):
        if stats[index, cv2.CC_STAT_AREA] < 30:
            continue
        region = labels == index
        x, y, w, h = stats[index, :4]
        expected = (region & ~occluded)[y : y + h, x : x + w].astype(np.uint8)
        checks[f"majority_object_{index}_stationary_throughout"] = all(
            mask_coverage(
                (_mask(frame[y : y + h, x : x + w], color) & ~occluded[y : y + h, x : x + w]).astype(np.uint8),
                expected,
                radius=2,
            )[1]
            >= 0.97
            for frame in frames
        )
    final_mask = _mask(frames[-1], color).astype(np.uint8)
    checks["exact_retained_shape_set"] = min(mask_coverage(final_mask, keep.astype(np.uint8), radius=2)) >= 0.98
    checks["all_minority_objects_removed"] = all(np.count_nonzero(_mask(frames[-1], c)) <= 20 for c in minority_colors)
    checks["no_other_objects_or_marks"] = min(mask_coverage(ink(frames[-1]), ink(final), radius=2)) >= 0.98
    # A retained object's black border and interior must also remain intact.
    allowed = occluded
    checks["retained_objects_and_background_preserved"], worst = _stationary_scene(frames, first, allowed)
    # The removal regions are mutable while minority objects disappear, but
    # the final frame must show their empty canvas or revealed retained shape.
    # Otherwise a new gray/black ghost can evade both color-count and global
    # foreground-overlap checks inside the permanently exempt source region.
    checks["removed_regions_match_retained_scene"], final_worst = _stationary_scene([frames[-1]], final, ~allowed)
    return _result(
        checks,
        {
            "independent_color_counts": {str(k): v for k, v in counts.items()},
            "majority_rgb": list(color),
            "unexpected_static_component": worst,
            "unexpected_final_removed_region_component": final_worst,
        },
    )


def _count(frames, first, final, metadata):
    p = metadata["parameters"]
    number = len(p["positions"])
    if number != p["num_objects"] or len(p["shapes"]) != number:
        raise ValueError("Count annotation disagrees with original object inventory")
    texts = [
        layer
        for layer in _layers(metadata, "final_frame")
        if layer.get("type") == "text" and "Count:" in str(layer.get("content", ""))
    ]
    if not texts:
        raise ValueError("Count answer observation geometry unavailable")
    text = texts[0]
    if str(text["content"]) != f"Count: {number}":
        raise ValueError("Reference count label contradicts independently counted objects")
    # Recover the answer panel from the changed central text region. This only
    # defines an OCR template, never the answer itself (computed above).
    layers = _layers(metadata, "final_frame")
    panels = [
        layer
        for layer in layers
        if layer.get("kind") == "rect"
        and layer.get("fill_rgb") in ([255, 255, 255], [240, 240, 240])
        and layer.get("w_px", 0) > 150
    ]
    if panels:
        panel = min(panels, key=lambda b: abs(b["x_px"] + b["w_px"] / 2 - first.shape[1] / 2))
        x, y, w, h = (round(panel[key]) for key in ("x_px", "y_px", "w_px", "h_px"))
    else:
        x, y, w, h = (
            first.shape[1] // 4,
            round(first.shape[0] * 0.43),
            first.shape[1] // 2,
            round(first.shape[0] * 0.14),
        )
    expected_text = (cv2.cvtColor(final[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY) < 100).astype(np.uint8)
    actual_text = (cv2.cvtColor(frames[-1][y : y + h, x : x + w], cv2.COLOR_BGR2GRAY) < 100).astype(np.uint8)
    literal_count = read_count(frames[-1][y : y + h, x : x + w])
    final_matches = (
        literal_count == number if literal_count is not None else _glyph_template_matches(actual_text, expected_text)
    )
    checks = {"exact_count_text": bool(final_matches)}
    # Identify every object's highlight separately using the provided object
    # geometry. A valid count may enumerate objects in any order.
    positions = np.array(p["positions"], float)
    yy, xx = np.indices(first.shape[:2])
    owner = np.argmin(np.stack([(xx - cx) ** 2 + (yy - cy) ** 2 for cx, cy in positions]), axis=0)
    original_border = cv2.Canny(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), 20, 60) > 0
    border_zone = cv2.dilate(original_border.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    observations, event_times = [], []
    previous = set()
    static = True
    answer_time = None
    last_text_mask = None
    last_count = None
    original_text = (cv2.cvtColor(first[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY) < 100).astype(np.uint8)
    panel_allowed = np.zeros(first.shape[:2], bool)
    panel_allowed[y : y + h, x : x + w] = True
    panel_allowed = cv2.dilate(panel_allowed.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    for frame_index, frame in enumerate(frames):
        changed = np.max(np.abs(frame.astype(np.int16) - first.astype(np.int16)), axis=2) > 65
        saturated = (np.ptp(frame.astype(np.int16), axis=2) > 70) & (np.max(frame, axis=2) > 150)
        marks = changed & saturated & border_zone
        selected = set()
        for i in range(number):
            if np.count_nonzero(marks & (owner == i)) >= max(15, p["sizes"][i] * 2):
                selected.add(i)
        new = selected - previous
        if new:
            observations.append(sorted(new))
            event_times.append(frame_index)
        previous = selected
        actual = (cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY) < 100).astype(np.uint8)
        if min(mask_coverage(actual, original_text, radius=2)) >= 0.98:
            observed_count = None
        elif _glyph_template_matches(actual, expected_text):
            observed_count = number
        elif last_text_mask is not None and _glyph_template_matches(actual, last_text_mask):
            observed_count = last_count
        else:
            observed_count = read_count(frame[y : y + h, x : x + w])
            last_text_mask, last_count = actual, observed_count
        if answer_time is None and observed_count == number:
            answer_time = frame_index
        permissible = (panel_allowed if observed_count == number else np.zeros(first.shape[:2], bool)) | (
            border_zone & cv2.dilate(marks.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        )
        static &= _static([frame], first, permissible)[0]
    counted = [i for state in observations for i in state]
    checks["one_new_object_counted_at_a_time"] = all(len(state) == 1 for state in observations)
    checks["every_object_counted_exactly_once"] = len(counted) == number and sorted(counted) == list(range(number))
    checks["count_text_after_all_objects"] = bool(
        event_times and answer_time is not None and max(event_times) <= answer_time
    )
    checks["original_objects_preserved_throughout"] = static
    return _result(
        checks,
        {
            "expected_count": number,
            "observed_count_order": counted,
            "count_event_frames": event_times,
            "answer_box": [x, y, w, h],
            "literal_ocr_count": literal_count,
        },
    )


def verify(task, frames, first, final, metadata, reference):
    prefix = task.split("_", 1)[0]
    if prefix == "O-38":
        return _majority(frames, first, final, metadata)
    if prefix == "O-33":
        return _count(frames, first, final, metadata)
    return None
