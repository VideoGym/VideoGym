"""Observe source-derived local transitions without prescribing their duration."""

from typing import Optional
import cv2
import numpy as np


def resize_rgb(array, shape):
    return array if array.shape[:2] == shape else cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def complete_footprint_projection(
    current: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    box: Optional[tuple[int, int, int, int]] = None,
    mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Measure one source-derived, fixed-footprint local opacity step.

    ``source`` and ``target`` define the registered appearance endpoints.
    Their semantic validity is checked by the calling verifier. A complete-footprint fade advances
    all materially changing pixels together, so their individual projections
    onto the endpoint delta stay tightly clustered.  Centre-first, radial, and
    directional reveals instead leave a material share of the same footprint
    near zero while another share is already near one.
    """

    shape = source.shape[:2]
    observed_full = resize_rgb(current, shape).astype(np.float32)
    source_full = source.astype(np.float32)
    target_full = target.astype(np.float32)
    if box is not None:
        left, top, right, bottom = box
        observed_full = observed_full[top:bottom, left:right]
        source_full = source_full[top:bottom, left:right]
        target_full = target_full[top:bottom, left:right]
    delta_patch = target_full - source_full
    observed_patch = observed_full - source_full
    pixel_energy = np.sum(delta_patch * delta_patch, axis=2)
    measurable = pixel_energy >= 144.0
    if mask is not None:
        measurable &= mask[top:bottom, left:right] if box is not None else mask
    if not np.any(measurable):
        raise ValueError("local fade footprint has no measurable endpoint pixels")

    delta = delta_patch[measurable]
    observed = observed_patch[measurable]
    total_energy = float(np.sum(delta * delta))
    progress = float(np.sum(observed * delta) / max(1.0, total_energy))
    pixel_progress = np.sum(observed * delta, axis=1) / pixel_energy[measurable]
    low, high = np.quantile(pixel_progress, (0.10, 0.90))
    predicted = progress * delta
    fit_error = float(
        np.sqrt(np.sum((observed - predicted) ** 2) / max(1.0, total_energy))
    )

    # At a partial opacity, even weak anti-aliased terminal pixels must already
    # participate.  Scale the threshold with the observed mean so codec noise
    # near zero does not count as real support while an early legitimate fade
    # is not penalized for being faint.
    support_threshold = max(0.025, min(0.20, 0.35 * max(0.0, progress)))
    support = float(np.mean(pixel_progress >= support_threshold))
    bounded = float(np.mean((pixel_progress >= -0.18) & (pixel_progress <= 1.18)))
    return {
        "progress": progress,
        "progress_q10": float(low),
        "progress_q90": float(high),
        "progress_spread": float(high - low),
        "footprint_support": support,
        "linear_fit_error": fit_error,
        "bounded_pixel_fraction": bounded,
        "measurable_pixels": float(np.count_nonzero(measurable)),
    }


def uniform_fade_metrics(frames, source, terminal, *, erasure=False):
    """A local whole-footprint fade needs observable intermediate opacity.

    A shared opacity is required across all changed pixels. A spatial scan,
    growing object or all-at-once cut cannot borrow a correct endpoint. The
    number of frames and the particular monotone opacity curve remain free.
    """
    # Object deletion explicitly selects fading-out source ink; other tasks
    # read newly added answer ink. Removing the old question mark is a
    # separate terminal obligation; it must not become a second opacity clock
    # whose relative area depends on whether the answer is a tiny polygon or
    # a large height card. Source grid lines and antialias edges are excluded.
    background, ink = (terminal, source) if erasure else (source, terminal)
    added = (np.min(background, axis=2) >= 220) & (np.max(background.astype(float) - ink, axis=2) >= 24)
    interior = cv2.erode(added.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if interior.sum() >= max(24, .25 * added.sum()):
        added = interior
    high = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    low = cv2.erode(ink, np.ones((5, 5), np.uint8))
    flat = added & (np.max(high.astype(float) - low, axis=2) <= 8)
    if flat.sum() >= max(24, .25 * added.sum()):
        added = flat
    # The task verifier checks the terminal's answer independently. Register
    # that accepted observation's paint/edge bandwidth for the temporal fit;
    # comparing every intermediate pixel with an ideal raster would turn a
    # constant harmless silhouette displacement into a time-varying opacity.
    observed_terminal = np.asarray(frames[-1])
    coverage = float(np.mean(np.max(np.abs(observed_terminal.astype(float) - source), axis=2)[added] >= 12)) if added.any() else 0.
    endpoint_delta = observed_terminal.astype(np.float32) - source.astype(np.float32)
    measurable = added & (np.sum(endpoint_delta * endpoint_delta, axis=2) >= 144.)
    if not measurable.any():
        return {"video_uniform_fade_pass": False,
                "video_fade_expected_footprint_coverage": coverage,
                "video_fade_failure_reason": "candidate has no measurable answer fade"}, False
    readings = [complete_footprint_projection(frame, source, observed_terminal, mask=added) for frame in frames]
    progress = [row["progress"] for row in readings]
    intermediate = [i for i, value in enumerate(progress) if .10 < value < .90]
    malformed = [i for i, row in enumerate(readings)
                 if .08 < row["progress"] < .92
                 and (row["progress_spread"] > .35 or row["linear_fit_error"] > .25
                      or row["footprint_support"] < .72)]
    monotone = all(right >= left - .06 for left, right in zip(progress, progress[1:]))
    endpoints = bool(progress and max(progress[:2]) <= .08 and min(progress[-2:]) >= .85)
    passed = bool(intermediate and not malformed and monotone and endpoints and coverage >= .85)
    return {"video_uniform_fade_pass": passed, "video_fade_progress": progress,
            "video_fade_intermediate_frames": intermediate,
            "video_fade_malformed_frames": malformed, "video_fade_monotone": monotone,
            "video_fade_endpoint_holds": endpoints,
            "video_fade_expected_footprint_coverage": coverage,
            "video_fade_max_progress_spread": max((r["progress_spread"] for r in readings), default=0)}, passed


