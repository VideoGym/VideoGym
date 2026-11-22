"""Exact public three-circle intersection contract for G-250."""

import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.verification.annotation.selection import (
    _background_reference,
    _certified_fade_mask,
    _foreign_background_component,
    _mark_mask,
    _preservation,
    _source_pixel_support,
)


def _largest_component(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return int(np.max(stats[1:count, cv2.CC_STAT_AREA])) if count > 1 else 0


def verify_triple_intersection(frames, first, metadata, observation_resolution=None):
    parameters = metadata["parameters"]
    circles = parameters["circles"][:3]
    if len(circles) != 3:
        raise ValueError("G-250 requires three publicly displayed circles")
    width, height = observation_resolution or first.shape[1::-1]
    canvas_width, canvas_height = parameters["canvas_size"]
    yy, xx = np.indices((height, width), dtype=np.float32)
    xx *= canvas_width / width
    yy *= canvas_height / height
    truth = np.logical_and.reduce([
        (xx - circle["center"][0]) ** 2 + (yy - circle["center"][1]) ** 2 <= circle["radius"] ** 2
        for circle in circles
    ])
    if not np.any(truth):
        raise ValueError("Source circles have no observable triple intersection")
    # Boundary tolerance is measured in original candidate pixels, rather than
    # inferred from the reference fill or from the candidate's painted region.
    allowed = cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2, 5) <= 2
    core = cv2.distanceTransform(truth.astype(np.uint8), cv2.DIST_L2, 5) > 2
    minimum = max(6, round(height * width * 6 / (320 * 320)))
    source = cv2.resize(first, (width, height), interpolation=cv2.INTER_AREA)
    background = _background_reference(source)
    color = np.asarray([0, 0, 255.0])
    mutable = allowed.astype(np.uint8) * 255
    corrupted, outside, foreign = [], [], []
    largest_extra = largest_foreign = 0
    last_ink = None
    for index, frame in enumerate(frames):
        observed = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        opaque = _mark_mask(observed, source, color)
        fade = _certified_fade_mask(observed, source, color)
        sy, sx = np.nonzero((fade > 0) & ~background[1])
        if len(sx):
            supported = _source_pixel_support(source, observed, np.column_stack([sx, sy]))
            fade[sy[supported], sx[supported]] = 0
        ink = (opaque | fade) > 0
        extra = _largest_component(ink & ~allowed)
        largest_extra = max(largest_extra, extra)
        if extra >= minimum:
            outside.append(index)
        if not _preservation(source, observed, mutable)["passed"]:
            corrupted.append(index)
        clean, area = _foreign_background_component(observed, background, mutable, chromatic_ink=True)
        largest_foreign = max(largest_foreign, area)
        if not clean:
            foreign.append(index)
        last_ink = ink
    missing = _largest_component(core & ~last_ink)
    checks = {
        "annotation_present": bool(np.any(last_ink & truth)),
        "complete_triple_intersection_filled": missing < minimum,
        "red_stays_in_triple_intersection": not outside,
        "source_objects_preserved_throughout": not corrupted,
        "no_new_background_objects": not foreign,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "family": "triple_circle_intersection_fill",
            "target_source": "independent_first_three_circle_geometry",
            "observation_resolution": [width, height],
            "boundary_tolerance_pixels": 2,
            "observable_component_pixels": minimum,
            "truth_pixels": int(np.count_nonzero(truth)),
            "filled_truth_fraction": float(np.mean(last_ink[truth])),
            "largest_missing_body_component": missing,
            "largest_extra_red_component": largest_extra,
            "off_target_frame_indices": outside,
            "corrupted_frame_indices": corrupted,
            "foreign_background_frame_indices": foreign,
            "largest_foreign_background_component": largest_foreign,
        },
    }
