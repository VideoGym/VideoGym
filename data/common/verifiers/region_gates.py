"""Codec-tolerant checks for task-relevant regions and object evidence.

Color differences use mildly smoothed observations with a quantization
noise deadband. Source-derived regions still prevent a missing or wrong
answer object from disappearing in a whole-image average. Ink coverage
allows two pixels of raster displacement; committed objects must remain
closer to their required state than to their unchanged initial state.
No saved reference image/video or authored answer is consulted.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from data.common.verifiers.keyframes import visual_delta, visual_features

try:  # cv2 is already a hard dependency of every route that calls in here
    import cv2
except Exception:  # pragma: no cover - exercised only on a broken install
    cv2 = None  # type: ignore[assignment]


# Region size for every per-region gate.  512 / 16 gives 32x32 regions, small
# enough that one erased or added object dominates its own region instead of
# being averaged away over the canvas.  Shared with the rearrangement terminal
# gate (``TERMINAL_REGION``) and the geometry scene gate (``SCENE_TILE``).
REGION = 32

# Raster-noise allowance after smoothing and the color deadband.
REGION_ERROR_FLOOR = 12.0 / 255.0

# Cap the allowance so an incorrect object/color still fails locally.
REGION_ERROR_CEILING = 0.080

# Scaling convention shared with the rearrangement terminal/video gates and the
# geometry marker gate: ``max(floor, min(ceiling, separation * alpha))``.
SEPARATION_ALPHA = 0.45

# A pixel counts as ink when it departs from its own frame's background by at
# least this much on some channel.  Same constant as the geometry scene gate:
# well below every palette separation the families draw with and well above
# the residue antialiasing leaves behind.
INK_DELTA = 24

# The candidate's ink is read with half that contrast.  A hard threshold on
# both sides is brittle at a saturated background: a uniform brightness shift
# leaves the background clipped at 255 while it moves the palest strokes
# towards it, so a stroke drawn at exactly INK_DELTA drops out of the
# candidate's mask and reads as erased.  Hysteresis removes that failure
# without weakening the test -- an object that is actually gone measures 0.
INK_COVER_DELTA = INK_DELTA // 2

# A region takes part in the ink gate once it carries at least this much ink;
# below it the denominator is too small for a fraction to mean anything.  It
# doubles as the absolute floor on the per-region uncovered-ink budget: a
# region may lose at most as much ink as would have kept it out of the gate in
# the first place.
REGION_MIN_INK = 24

# Share of one region's terminal ink that may go uncovered.  Same constant as
# the geometry scene gate, where a legal delivery leaves 0.0 of every region
# uncovered and a delivery belonging to a different sample of the same task
# leaves at least 0.62 of its worst region uncovered.
REGION_ERASED_FRACTION = 0.25

# A member of the source-derived change must be at least this many pixels
# before it is hard-gated on its own.  Smaller components are antialiasing
# fringe around a larger member rather than an independent object; they remain
# covered by the region gates.
MEMBER_MIN_PIXELS = 120


def _as_float(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def region_error_map(first: np.ndarray, second: np.ndarray, *, region: int = REGION) -> np.ndarray:
    """Mean absolute difference per square region, normalised to [0, 1]."""

    difference = visual_delta(first, second).mean(axis=2) / 255.0
    height, width = difference.shape
    rows, cols = height // region, width // region
    if rows == 0 or cols == 0:
        return np.array([[float(difference.mean())]], dtype=np.float64)
    cropped = difference[: rows * region, : cols * region]
    return cropped.reshape(rows, region, cols, region).mean(axis=(1, 3))


def _region_counts(mask: np.ndarray, *, region: int = REGION) -> np.ndarray:
    height, width = mask.shape
    rows, cols = height // region, width // region
    if rows == 0 or cols == 0:
        return np.array([[int(mask.sum())]], dtype=np.int64)
    cropped = mask[: rows * region, : cols * region]
    return cropped.reshape(rows, region, cols, region).sum(axis=(1, 3), dtype=np.int64)


def _background(image: np.ndarray) -> np.ndarray:
    sample = np.asarray(image)[::4, ::4].reshape(-1, 3)
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


def ink_mask(image: np.ndarray, *, delta: int = INK_DELTA) -> np.ndarray:
    """Pixels that depart from the frame's own modal background colour."""

    return np.max(np.abs(np.asarray(image).astype(np.int16) - _background(image)), axis=2) >= delta


def _dilate(mask: np.ndarray, size: int = 5) -> np.ndarray:
    if cv2 is None:  # pragma: no cover
        return mask
    return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)) > 0


def _labelled_members(mask: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Label ``mask``'s connected members and return the retained label ids.

    Components below :data:`MEMBER_MIN_PIXELS` are antialiasing fringe around a
    larger member rather than independent objects; they stay covered by the
    region gates.
    """

    if not bool(mask.any()) or cv2 is None:
        return None, np.empty(0, dtype=np.int64)
    _count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    sizes = np.bincount(labels.ravel())
    keep = np.nonzero(sizes >= MEMBER_MIN_PIXELS)[0]
    keep = keep[keep != 0]
    order = np.argsort(-sizes[keep])
    return labels, keep[order]


def region_delivery_gate(
    candidate: np.ndarray,
    terminal: np.ndarray,
    alternates: Sequence[np.ndarray] = (),
    *,
    region: int = REGION,
    prefix: str = "",
) -> Tuple[Dict[str, Any], bool]:
    """Hard-gate every region of the delivery against the re-rendered terminal.

    ``alternates`` are the other states this sample legitimately passes
    through -- the input state and any intermediate commit.  Each region is
    held to its own measured separation, so a region that the sample never
    changes collapses to the renderer floor while a region that carries a large
    commit is allowed proportionally more slack, never more than the ceiling.
    """

    observed = region_error_map(candidate, terminal, region=region)
    if alternates:
        stacked = np.stack([region_error_map(terminal, other, region=region) for other in alternates])
        positive = np.where(stacked > 1e-9, stacked, np.inf)
        separation = positive.min(axis=0)
        separation = np.where(np.isfinite(separation), separation, 0.0)
    else:
        separation = np.zeros_like(observed)
    bounds = np.maximum(
        REGION_ERROR_FLOOR,
        np.minimum(REGION_ERROR_CEILING, separation * SEPARATION_ALPHA),
    )
    violations = observed > bounds
    worst = int(np.argmax(observed - bounds))
    row, col = divmod(worst, observed.shape[1])
    metrics = {
        f"{prefix}region_size": int(region),
        f"{prefix}region_count": int(observed.size),
        f"{prefix}region_max_error": round(float(observed.max()), 6),
        f"{prefix}region_worst": [int(row), int(col)],
        f"{prefix}region_worst_error": round(float(observed.flat[worst]), 6),
        f"{prefix}region_worst_bound": round(float(bounds.flat[worst]), 6),
        f"{prefix}region_violations": int(violations.sum()),
    }
    return metrics, not bool(violations.any())


def region_ink_gate(
    candidate: np.ndarray,
    terminal: np.ndarray,
    *,
    region: int = REGION,
    prefix: str = "",
) -> Tuple[Dict[str, Any], bool]:
    """Require every load-bearing region of the terminal ink to survive.

    Additions are handled by :func:`region_delivery_gate`; this gate exists for
    removals, where a thin member spread over many regions is too sparse to
    move any region's mean but is still a whole object that has gone missing.
    Two pixels of growth are granted to the candidate's ink before subtracting so
    subpixel rounding cannot be decisive; that cannot hide a missing primitive,
    which is at least a stroke wide.
    """

    terminal_ink = ink_mask(terminal)
    covered = _dilate(ink_mask(candidate, delta=INK_COVER_DELTA))
    erased = terminal_ink & ~covered

    region_ink = _region_counts(terminal_ink, region=region)
    region_erased = _region_counts(erased, region=region)
    load_bearing = region_ink >= REGION_MIN_INK
    budget = np.maximum(
        float(REGION_MIN_INK), region_ink.astype(np.float64) * REGION_ERASED_FRACTION
    )
    violations = load_bearing & (region_erased.astype(np.float64) > budget)
    fractions = np.where(load_bearing, region_erased / np.maximum(region_ink, 1), 0.0)

    # A region budget is still an aggregate inside its own region: a small
    # object spread across several densely inked regions is a small share of
    # each of them.  Every connected member of the terminal ink is therefore
    # also held to the same share of *its own* pixels, which is what makes a
    # complete object impossible to delete no matter where it sits.
    labels, member_ids = _labelled_members(terminal_ink)
    member_worst = 0.0
    member_violations = 0
    if labels is not None and member_ids.size:
        flat = labels.ravel()
        sizes = np.bincount(flat)[member_ids]
        gone = np.bincount(flat, weights=erased.ravel().astype(np.float64),
                           minlength=int(labels.max()) + 1)[member_ids]
        member_fractions = gone / np.maximum(sizes, 1)
        member_worst = float(member_fractions.max())
        member_violations = int((member_fractions > REGION_ERASED_FRACTION).sum())

    total = int(terminal_ink.sum())
    metrics = {
        f"{prefix}region_ink_pixels": total,
        f"{prefix}region_ink_erased_fraction": round(float(erased.sum()) / total, 6) if total else 0.0,
        f"{prefix}region_ink_worst_fraction": round(float(fractions.max()), 6) if load_bearing.any() else 0.0,
        f"{prefix}region_ink_budget_fraction": REGION_ERASED_FRACTION,
        f"{prefix}region_ink_violations": int(violations.sum()),
        f"{prefix}region_ink_load_bearing_regions": int(load_bearing.sum()),
        f"{prefix}member_ink_count": int(member_ids.size),
        f"{prefix}member_ink_worst_erased_fraction": round(member_worst, 6),
        f"{prefix}member_ink_violations": member_violations,
    }
    return metrics, not bool(violations.any()) and member_violations == 0


def member_commit_gate(
    candidate: np.ndarray,
    initial: np.ndarray,
    terminal: np.ndarray,
    *,
    prefix: str = "",
) -> Tuple[Dict[str, Any], bool]:
    """Hard-gate every connected member of the source-derived change.

    The change mask is derived by re-rendering the sample's own initial and
    terminal states, never from the candidate.  Each connected member of that
    mask is an object the delivery has to update; the candidate must sit on the
    terminal side of that member's own measured separation rather than being
    left in the initial state there.  Checking the union instead would let one
    completely un-committed member be paid for by every committed one.
    """

    candidate_f = _as_float(candidate)
    initial_f = _as_float(initial)
    terminal_f = _as_float(terminal)
    change = np.max(np.abs(terminal_f - initial_f), axis=2) >= INK_DELTA
    labels, member_ids = _labelled_members(change)
    if labels is None or member_ids.size == 0:
        return {f"{prefix}member_evaluated": False, f"{prefix}member_count": 0}, True

    flat = labels.ravel()
    size = np.bincount(flat)

    def per_member(delta: np.ndarray) -> np.ndarray:
        total = np.bincount(flat, weights=delta.mean(axis=2).ravel(), minlength=size.size)
        return total[member_ids] / np.maximum(size[member_ids], 1) / 255.0

    separations = per_member(np.abs(terminal_f - initial_f))
    observed = per_member(visual_delta(candidate_f, terminal_f))
    stale = per_member(visual_delta(candidate_f, initial_f))
    bounds = np.maximum(
        REGION_ERROR_FLOOR, np.minimum(REGION_ERROR_CEILING, separations * SEPARATION_ALPHA)
    )
    ok = (observed <= bounds) & (observed < stale)
    failures = [int(index) for index in np.nonzero(~ok)[0]]
    readings = [
        {
            "pixels": int(size[member_ids[index]]),
            "separation": round(float(separations[index]), 6),
            "terminal_error": round(float(observed[index]), 6),
            "initial_error": round(float(stale[index]), 6),
            "bound": round(float(bounds[index]), 6),
            "pass": bool(ok[index]),
        }
        for index in range(min(24, member_ids.size))
    ]
    metrics = {
        f"{prefix}member_evaluated": True,
        f"{prefix}member_count": int(member_ids.size),
        f"{prefix}member_worst_error": round(float(observed.max()), 6),
        f"{prefix}member_failures": failures[:24],
        f"{prefix}member_readings": readings,
    }
    return metrics, not failures


def terminal_region_gates(
    candidate: np.ndarray,
    initial: np.ndarray,
    terminal: np.ndarray,
    alternates: Sequence[np.ndarray] = (),
    *,
    prefix: str = "",
) -> Tuple[Dict[str, Any], bool]:
    """Run the three terminal-delivery gates and combine them as a conjunction."""

    others = list(alternates) or [initial]
    delivery_metrics, delivery_pass = region_delivery_gate(
        candidate, terminal, others, prefix=prefix
    )
    ink_metrics, ink_pass = region_ink_gate(candidate, terminal, prefix=prefix)
    member_metrics, member_pass = member_commit_gate(
        candidate, initial, terminal, prefix=prefix
    )
    metrics: Dict[str, Any] = {}
    metrics.update(delivery_metrics)
    metrics.update(ink_metrics)
    metrics.update(member_metrics)
    passed = bool(delivery_pass and ink_pass and member_pass)
    metrics[f"{prefix}region_delivery_pass"] = delivery_pass
    metrics[f"{prefix}region_ink_pass"] = ink_pass
    metrics[f"{prefix}member_commit_pass"] = member_pass
    metrics[f"{prefix}per_member_pass"] = passed
    return metrics, passed


# --------------------------------------------------------------- video ------

# Static regions use the same codec-tolerant observation metric.

STATIC_REGION_FLOOR = 12.0 / 255.0

# A region takes part in the static gate once this many of its pixels are
# declared static; below it the region is dominated by the moving object and
# the frame-level dynamic checks already own it.
STATIC_REGION_MIN_PIXELS = 64


def static_region_errors(
    frame: np.ndarray,
    baseline: np.ndarray,
    static_mask: np.ndarray,
    *,
    region: int = REGION,
    raster_tolerance: int = 0,
) -> Tuple[float, int]:
    """Worst per-region static error of one frame, and the violation count.

    The frame-wide mean over the static mask is a compensating aggregate: a
    24x24 patch painted over an untouched part of the scene is 0.2% of a
    512x512 static region and vanishes into it.  Restricting the same statistic
    to each region makes that patch 20-90% of the region it lands in.

    ``raster_tolerance`` permits a bounded edge displacement. Each neighbour
    is compared as one RGB colour before minimising; the two matching
    directions retain the larger residual so additions and erasures both count.
    """

    if raster_tolerance:
        observed, source = visual_features(frame), visual_features(baseline)
        radius = int(raster_tolerance)
        padding = ((radius, radius), (radius, radius), (0, 0))
        observed_near = np.pad(observed, padding, mode="edge")
        source_near = np.pad(source, padding, mode="edge")
        height, width = source.shape[:2]
        forward = np.full((height, width), np.inf, dtype=np.float32)
        backward = np.full_like(forward, np.inf)
        for row in range(2 * radius + 1):
            for col in range(2 * radius + 1):
                near_observed = observed_near[row:row+height, col:col+width]
                near_source = source_near[row:row+height, col:col+width]
                forward_error = np.maximum(np.abs(near_observed-source)-4.0, 0.0).mean(axis=2)
                backward_error = np.maximum(np.abs(observed-near_source)-4.0, 0.0).mean(axis=2)
                np.minimum(forward, forward_error, out=forward)
                np.minimum(backward, backward_error, out=backward)
        # Both directions are required: preserving nearby source ink cannot
        # excuse candidate ink added into the neighbouring empty region.
        delta = np.maximum(forward, backward) / 255.0
    else:
        delta = visual_delta(frame, baseline).mean(axis=2) / 255.0
    masked = np.where(static_mask, delta, 0.0)
    sums = _region_counts_float(masked, region=region)
    counts = _region_counts(static_mask, region=region)
    eligible = counts >= STATIC_REGION_MIN_PIXELS
    if not bool(eligible.any()):
        return 0.0, 0
    errors = np.where(eligible, sums / np.maximum(counts, 1), 0.0)
    return float(errors.max()), int((errors > STATIC_REGION_FLOOR).sum())


def _region_counts_float(values: np.ndarray, *, region: int = REGION) -> np.ndarray:
    height, width = values.shape
    rows, cols = height // region, width // region
    if rows == 0 or cols == 0:
        return np.array([[float(values.sum())]], dtype=np.float64)
    cropped = values[: rows * region, : cols * region]
    return cropped.reshape(rows, region, cols, region).sum(axis=(1, 3), dtype=np.float64)


__all__ = [
    "REGION",
    "REGION_ERROR_FLOOR",
    "REGION_ERROR_CEILING",
    "SEPARATION_ALPHA",
    "STATIC_REGION_FLOOR",
    "region_error_map",
    "static_region_errors",
    "terminal_region_gates",
]
