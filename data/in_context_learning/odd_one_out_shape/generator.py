"""Find the unique shape that differs on one explicitly named property.

The prompt names exactly one comparison axis: side count, convexity,
horizontal mirror symmetry, or boundary closure.  Three candidates share one
value on that axis and exactly one candidate has the other value, and the
non-target qualitative properties are held fixed wherever the target property
allows it.

Candidates are sampled in two independent layers.  Every candidate, the outlier
included, draws its answer-irrelevant presentation - rotation, anisotropy,
vertex-radius spread, angular clustering, vertex jitter, reflection, size and
slot offset - from one shared distribution.  The four are then placed on the
scale-free magnitudes a solver could rank them by, in three steps: the region
of that space every one of the four can reach is found by screening proposed
targets against all four candidate clouds, each candidate draws its own target
i.i.d. from that region, and a short hill climb over the presentation budget
walks it onto that target.  Only after that does the outlier receive its
property-bearing deviation, exaggerated well past the majority so the
discriminating feature clears the perception margin.  Both halves matter: three
near-identical clones beside one different shape would let the outlier be found
without evaluating the property at all, and an outlier that is simply the most
irregular shape would replace that shortcut with another.

The named property changes what a candidate can reach, not only what it happens
to look like: a reflex notch, a missing pair of sides and an unpainted break
each cost boundary length at equal area.  That leaves a real offset between the
outlier's reachable region and the majority's, and the offset is a single
direction, so it cannot be cancelled - only spread.  Mismatch on the two
magnitudes carrying it is priced above the rest so the matcher spends its
freedom there and distributes the residual across all five rather than leaving
it all where a solver would look.
"""

from __future__ import annotations

from data.common.base import RenderedTaskGenerator

import math
import random
import json
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.in_context_learning.core.option_delivery import (
    OPTION_MODALITY_OUTPUTS,
    OPTION_PRESERVATION_CONSTRAINTS,
    OPTION_TRANSITION_CONSTRAINTS,
)
from data.common.pipeline.media import (
    pad_image,
    save_visual_reasoning_video,
)


pattern_instructions = {
    "odd_one_out_shape": (
        "One 512 by 512 problem image shows four unlabelled candidate outlines, "
        "followed by four independent labelled plates. The prompt names the one "
        "property on which three candidates agree and one differs."
    ),
}

_LABELS = ("A", "B", "C", "D")
# Delivered media is drawn by data.in_context_learning.core.direct_delivery,
# the single renderer for this task.  Import its collection geometry and mark
# colour so the self-verifier can never drift from the raster it decodes.
from data.in_context_learning.core.direct_delivery import (  # noqa: E402
    _ODD_MARK as _RING_RGB,
    _ODD_RING_RADIUS as _RING_RADIUS,
    _ODD_SLOT_CENTERS as _SLOT_CENTERS,
)

_SLOT_HALF = 112.0

_RULE_LABELS = {
    "side_count": "SIDE COUNT",
    "convexity": "CONVEXITY",
    "horizontal_symmetry": "HORIZONTAL SYMMETRY",
    "closure": "CLOSED / OPEN",
}

_RULE_DEFINITIONS = {
    "side_count": "Count the straight boundary segments of each shape.",
    "convexity": "A convex polygon has no inward notch; every interior angle is below 180 degrees.",
    "horizontal_symmetry": "A shape has horizontal symmetry when reflection across a horizontal line through its center leaves it unchanged.",
    "closure": "A closed boundary has no gap and its final endpoint reconnects to its first endpoint.",
}

# --- geometry contract -------------------------------------------------------
# Candidate vertices are authored in plate units and rendered through one shared
# scale per view, so a candidate that is genuinely larger stays larger.
_BASE_AREA = 8000.0
_COORD_CAP = 132.0
_MAX_ANISO_RATIO = 2.4
_MAX_BBOX_ASPECT = 2.7
_MIN_BASE_EDGE = 14.0
_MIN_HALF_PROFILE_EDGE = 16.0
_MIN_RENDERED_EDGE = 14.0
# No non-feature vertex may sit this close to straight, so a hexagon can never
# raster as a pentagon (perception margin).  This is the floor validation
# re-asserts on the finished task.
_MIN_ANGLE_FROM_STRAIGHT = 12.0
# The reserve the sampler and the local search must keep above that floor.
# A floor is not a safe target for an optimiser: the search is free to trade
# corner angle for a better presentation match, so if it is allowed right down
# to the validation floor it will sit there, and validation - which only asks
# whether the floor was crossed - reports nothing.  Searching against a reserve
# and validating against the floor means the delivered shapes keep a margin
# nobody had to notice.  Without it the search parked corners at 12.0 degrees
# exactly and put 5.2 percent of all corners within 15 degrees of straight,
# which is a shape whose sides cannot be counted - fatal for side_count, where
# counting the sides IS the rule.
_SEARCH_ANGLE_FROM_STRAIGHT = 19.0
# Serialized vertex precision, and the mirror tolerance that must stay well
# above it so replay reproduces generation exactly.
_VERTEX_PRECISION = 6
_SYMMETRY_TOLERANCE = 1e-2
# Length of the break in an open boundary, as a fraction of sqrt(area).
# Mirrors direct_delivery._ODD_OPEN_GAP, asserted against it at validation.
_OPEN_GAP_FRACTION = 0.14
_OPEN_GAP_MAX_EDGE_FRACTION = 0.55

# --- property-bearing margins ------------------------------------------------
# The outlier's deviation is exaggerated well past the majority's, so the
# discriminating feature clears the perception margin by a checkable gap.
# Every majority corner stays clearly convex.  This is the straightness reserve
# expressed as an interior angle, not a second independent number: a corner the
# base builder accepts but the placement gate would reject is wasted work.
_MAX_MAJORITY_ANGLE = 180.0 - _SEARCH_ANGLE_FROM_STRAIGHT
# ...but still carries a matched feature vertex.  The window runs up to the
# straightness reserve, so it widened when that reserve tightened the cap: a
# 13-degree window would have made the majority's feature vertex nearly a fixed
# angle, which is a regularity of its own.
_MIN_DENT_ANGLE = 138.0
_MIN_REFLEX_ANGLE = 200.0       # the outlier's notch is unmistakably reflex
_MAX_REFLEX_ANGLE = 245.0
# Perpendicular depth of the notch as a share of the shape's diameter.  At the
# delivered collection scale one diameter is roughly 165 pixels against a
# 7-pixel stroke, so this floor is about two and a half stroke widths.
_MIN_NOTCH_DEPTH_RATIO = 0.105
_MAX_OUTLIER_MIRROR_IOU = 0.80  # asymmetry is visible over the whole silhouette
# ...and the reserve the search keeps below it, for the same reason the
# straightness floor needs one: matching is easier the less asymmetric the
# outlier is, so the search creeps up to whatever ceiling it is given.  Measured
# against the bare 0.80 ceiling the delivered outliers reached 0.791.
_SEARCH_OUTLIER_MIRROR_IOU = 0.76
_DENT_PULL = (0.28, 0.74)
_REFLEX_PULL = (1.18, 1.95)
# Answer-irrelevant sampling ranges shared by every candidate of a task.
_CLUSTER_RANGE = (0.7, 2.2)
_SPREAD_RANGE = (0.0, 0.46)
# Side gap is always exactly two, and no pairing may involve a triangle.  At
# equal area a triangle sits at a compactness ceiling and a diameter floor that
# no pentagon or hexagon reaches without extreme elongation, so triangle
# pairings cannot hold presentation parity at all: measured over 900-variant
# clouds the best target reachable by both sides of a 3-vs-5 split is still
# 0.97 pooled standard deviations from one of them, against 0.32 for 4-vs-6,
# and the residual is a full standard deviation of compactness.  A triangle
# outlier would therefore rank on silhouette alone.
_SIDE_PAIRS = ((4, 6), (6, 4))

# --- presentation parity -----------------------------------------------------
# Answer-irrelevant magnitudes matched to one shared per-task target.  All are
# scale-free; absolute size is matched separately by the i.i.d. area gain.
# ``iso`` is 4*pi*A/painted^2 and ``ink_norm`` is painted/sqrt(A), so the two
# are algebraically equivalent; matching ``iso`` matches painted ink too.
_MATCH_KEYS = ("iso", "radius_cv", "bbox_aspect", "edge_cv", "diam_norm")
# Relative price of a mismatch on each magnitude.  Not every magnitude is
# equally hard to place: the property itself moves compactness and, through it,
# normalised diameter, while radius and edge spread can be matched almost
# exactly.  Under equal weights the matcher spends its freedom where it is
# cheap and leaves the whole residual sitting on the two hard magnitudes, which
# is precisely where a solver would look.  Pricing the hard ones up spreads the
# same unavoidable residual across all five, so no single one of them ranks the
# outlier.
_MATCH_WEIGHTS = (3.0, 1.0, 1.0, 1.0, 2.5)
_VARIANTS_PER_ROUND = 260
# Rounds of cloud growth a task may take before it is abandoned.  The
# straightness reserve costs the matcher freedom, so hard pairings need more
# cloud rather than fewer: at six rounds the reserve dropped convexity's
# acceptance to 0.10, and growing the cloud instead restores it to 1.00.
_MATCH_ATTEMPTS = 14
_MIN_CLOUD = 40
# Variants walked onto admissible targets per distinct slot kind, and the
# hill-climb budget each walk gets.  Sampling alone cannot beat discretisation:
# nearest-neighbour distance in five matched magnitudes falls only as the fifth
# root of the sample, so removing a third of a standard deviation of it by
# sampling would need sixteen times the cloud.  The walk removes it directly.
_REFINE_COUNT = 130
_POLISH_STEPS = 26
# Presentation targets are proposed around real cloud points - one slot drawn at
# random per proposal, so no slot's own region is over-represented - and then
# screened against EVERY slot's cloud.  A proposal survives only when all four
# candidates can reach it, and each candidate then draws its own target i.i.d.
# from that jointly reachable set.  Aiming instead at one shared target and
# spreading around it is what left the residual: wherever the spread ran past
# the edge of a candidate's reachable region, that candidate got clipped and the
# others did not, which parks it at one end of the four on the clipped
# magnitude.  A reflex notch has a compactness floor and a four-sided shape has a
# compactness ceiling, so the clipping was systematic, not incidental.
# Proposals are the cloud points themselves plus displacements of them, at a
# spread of scales.  Free-floating proposals are wasted - a point drawn at
# random in the statistic space is far from every cloud - but the widely
# displaced proposals that do survive the screen are the valuable ones: they
# stretch the admitted region out to the edge of what all four candidates can
# reach.  That width matters as much as the screen does.  Matching the four
# candidates tightly does not remove a residual offset, it amplifies it: at a
# spread of zero any systematic bias whatsoever makes the outlier the extreme
# of the four every single time.
_PROPOSAL_JITTERS = (0.0, 0.10, 0.20, 0.34, 0.52, 0.75)
# Escalating admission radius, in units of the pooled weighted cloud spread.
# The first entry that yields enough targets wins, so an easy task is matched
# tightly and a hard one still completes instead of being rejected.  The ladder
# deliberately stops well short of the weighted spread: when nothing on it
# admits enough targets the task grows its clouds instead, and a denser cloud
# is what a hard pairing actually needs.  Extending the ladder upward instead
# was measured and gives the parity back - convexity's worst statistic goes
# from 0.356 to 0.422 - because a loose target is accepted rather than earned.
_ADMIT_TOLERANCES = (0.14, 0.20, 0.28, 0.40, 0.56, 0.78)
_MIN_ADMISSIBLE = 60
# Forcing a magnitude's RANK uniform for the outlier, on top of matching its
# value, was tried here and makes the task worse rather than better.  The
# residual offset the named property leaves is a single direction in the matched
# space, so compactness and normalised diameter move together: pinning one
# rank does not remove the offset, it relocates all of it into the other.
# Measured on the shared audit at n=60, worst statistic over convexity:
# unforced 0.400, compactness forced 0.467, all five forced 0.467.

_SCHEMA_VERSION = "odd_one_out_shape.v3"
_VERIFIER_PROFILE = "odd_one_out_named_property_replay_v3"
_VERIFIER_BUDGET = {
    "text_warm_p95_ms": 3.5,
    "text_min_samples_per_second": 400.0,
    "image_warm_p95_ms": 65.0,
    "image_min_samples_per_second": 15.0,
    "video_warm_p95_ms": 225.0,
    "video_min_samples_per_second": 5.0,
    "video_max_length_ms": 1500.0,
    "max_video_frames": 157,
}


Point = Tuple[float, float]
Candidate = Dict[str, Any]


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


def _is_convex(vertices: Sequence[Point]) -> bool:
    if len(vertices) < 3:
        return False
    signs: List[bool] = []
    for index in range(len(vertices)):
        value = _cross(vertices[index - 1], vertices[index], vertices[(index + 1) % len(vertices)])
        if abs(value) > 1e-6:
            signs.append(value > 0)
    return bool(signs) and all(value == signs[0] for value in signs)


def _has_horizontal_symmetry(vertices: Sequence[Point], *, tolerance: float = _SYMMETRY_TOLERANCE) -> bool:
    """Whether reflecting across the shape's own horizontal centre line is a no-op.

    The tolerance is far above the rounding of the serialized vertices, so the
    verifier's replay classifies exactly as generation did, and far below any
    asymmetry this task samples: the outlier must clear a whole-silhouette
    mirror-mismatch margin, so there are no near-misses to confuse.
    """

    if not vertices:
        return False
    center_y = sum(y for _, y in vertices) / len(vertices)
    remaining = list(vertices)
    for x, y in vertices:
        reflected = (x, 2.0 * center_y - y)
        match_index = next(
            (
                index
                for index, point in enumerate(remaining)
                if abs(point[0] - reflected[0]) <= tolerance and abs(point[1] - reflected[1]) <= tolerance
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return True


def _centroid(vertices: Sequence[Point]) -> Point:
    count = len(vertices)
    return (sum(x for x, _ in vertices) / count, sum(y for _, y in vertices) / count)


def _edge_lengths(vertices: Sequence[Point]) -> List[float]:
    count = len(vertices)
    return [
        math.hypot(
            vertices[(index + 1) % count][0] - vertices[index][0],
            vertices[(index + 1) % count][1] - vertices[index][1],
        )
        for index in range(count)
    ]


def _diameter(vertices: Sequence[Point]) -> float:
    return max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in vertices for b in vertices)


def _signed_area(vertices: Sequence[Point]) -> float:
    count = len(vertices)
    return sum(
        vertices[index][0] * vertices[(index + 1) % count][1]
        - vertices[(index + 1) % count][0] * vertices[index][1]
        for index in range(count)
    ) / 2.0


def _interior_angles(vertices: Sequence[Point]) -> List[float]:
    """Interior angle in degrees at each vertex; above 180 means reflex."""

    count = len(vertices)
    counter_clockwise = _signed_area(vertices) > 0
    angles: List[float] = []
    for index in range(count):
        previous, current, following = (
            vertices[index - 1],
            vertices[index],
            vertices[(index + 1) % count],
        )
        first = (previous[0] - current[0], previous[1] - current[1])
        second = (following[0] - current[0], following[1] - current[1])
        angle = math.degrees(
            math.atan2(
                abs(first[0] * second[1] - first[1] * second[0]),
                first[0] * second[0] + first[1] * second[1],
            )
        )
        turn = (current[0] - previous[0]) * (following[1] - current[1]) - (
            current[1] - previous[1]
        ) * (following[0] - current[0])
        reflex = (turn < 0) if counter_clockwise else (turn > 0)
        angles.append(360.0 - angle if reflex else angle)
    return angles


def _normalize_area(vertices: Sequence[Point], target: float) -> List[Point]:
    current = _polygon_area(vertices)
    if current <= 1e-9:
        return list(vertices)
    factor = math.sqrt(target / current)
    cx, cy = _centroid(vertices)
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in vertices]


def _display_stats(vertices: Sequence[Point], *, closed: bool = True) -> Dict[str, float]:
    """Every answer-irrelevant magnitude a solver could rank candidates on.

    ``ink_norm`` is scale-free painted boundary length.  An open boundary omits
    part of its closing edge, so without matching this a solver could pick the
    open shape purely as the one carrying the least ink.
    """

    lengths = _edge_lengths(vertices)
    perimeter = sum(lengths)
    area = _polygon_area(vertices)
    cx, cy = _centroid(vertices)
    radii = [math.hypot(x - cx, y - cy) for x, y in vertices]
    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    mean_edge = sum(lengths) / len(lengths)
    mean_radius = sum(radii) / len(radii)
    edge_dev = math.sqrt(sum((value - mean_edge) ** 2 for value in lengths) / len(lengths))
    radius_dev = math.sqrt(sum((value - mean_radius) ** 2 for value in radii) / len(radii))
    gap = 0.0
    if not closed:
        gap = min(
            _OPEN_GAP_MAX_EDGE_FRACTION * lengths[-1],
            _OPEN_GAP_FRACTION * math.sqrt(max(area, 1e-9)),
        )
    painted = perimeter - gap
    # Compactness is measured on the PAINTED boundary, because that is the only
    # boundary a solver can see.  Scoring an open shape against a closing edge
    # nobody drew would make compactness and ink irreconcilable, and would leave
    # the open candidate permanently the least compact of the four.
    return {
        "iso": 4.0 * math.pi * area / (painted * painted) if painted > 0 else 0.0,
        "radius_cv": radius_dev / (mean_radius or 1.0),
        "bbox_aspect": max(width, height) / max(min(width, height), 1e-9),
        "edge_cv": edge_dev / (mean_edge or 1.0),
        "diam_norm": _diameter(vertices) / math.sqrt(max(area, 1e-9)),
        "ink_norm": painted / math.sqrt(max(area, 1e-9)),
    }


def _ring_polygon(
    rng: random.Random,
    n_sides: int,
    *,
    cluster: float,
    spread: float,
    radius: float = 58.0,
) -> List[Point]:
    """Vertices on a ring with controllable angular clustering and radius spread."""

    gaps = [rng.random() ** cluster + 0.18 for _ in range(n_sides)]
    total = sum(gaps)
    angles: List[float] = []
    accumulated = rng.uniform(0.0, 2.0 * math.pi)
    for gap in gaps:
        angles.append(accumulated)
        accumulated += 2.0 * math.pi * gap / total
    return [
        (
            radius * (1.0 - spread * rng.random()) * math.cos(angle),
            radius * (1.0 - spread * rng.random()) * math.sin(angle),
        )
        for angle in angles
    ]


def _convex_base(rng: random.Random, n_sides: int, *, cluster: float, spread: float) -> List[Point] | None:
    for _ in range(60):
        vertices = _ring_polygon(rng, n_sides, cluster=cluster, spread=spread)
        if (
            _is_convex(vertices)
            and min(_edge_lengths(vertices)) > _MIN_BASE_EDGE
            and max(_interior_angles(vertices)) <= _MAX_MAJORITY_ANGLE
        ):
            return vertices
    return None


def _notched_base(
    rng: random.Random,
    n_sides: int,
    *,
    cluster: float,
    spread: float,
    reflex: bool,
) -> List[Point] | None:
    """A polygon with one feature vertex.

    Every candidate of a convexity task carries a feature vertex.  On the
    outlier it is pulled past the chord of its neighbours into an unmistakable
    reflex notch; on the majority it either stops well short of that chord as a
    dent that stops well short, so the discriminating evidence is a wide
    angular gap rather than a borderline corner.

    Giving the majority its own compactness-lowering convex feature was tried
    here and does not work.  A deep convex spike, and one to three further
    convex dents, each lower compactness exactly as claimed, but neither moves
    the majority cloud onto the outlier's: measured as the excess of the
    majority-to-outlier nearest-neighbour distance over the same-law
    nearest-neighbour distance, separation held at 0.26 to 0.28 pooled standard
    deviations across spike shares 0 to 1 and dent counts 0 to 3, symmetric or
    majority-only.  The features do buy compactness, but they buy diameter and
    radius spread with it and so land somewhere else in the matched space.  The
    residual is discretisation, not geometry, and is removed by the local
    search in ``_polish`` instead.
    """

    for _ in range(60):
        vertices = _convex_base(rng, n_sides, cluster=cluster, spread=spread)
        if vertices is None:
            continue
        index = rng.randrange(n_sides)
        previous, current, following = (
            vertices[index - 1],
            vertices[index],
            vertices[(index + 1) % n_sides],
        )
        mid = ((previous[0] + following[0]) / 2.0, (previous[1] + following[1]) / 2.0)
        reach = math.hypot(current[0] - mid[0], current[1] - mid[1])
        if reach < 1e-6:
            continue
        ux, uy = (current[0] - mid[0]) / reach, (current[1] - mid[1]) / reach
        low, high = _REFLEX_PULL if reflex else _DENT_PULL
        for _ in range(24):
            pull = rng.uniform(low, high)
            candidate = list(vertices)
            candidate[index] = (current[0] - ux * reach * pull, current[1] - uy * reach * pull)
            if not _is_simple_polygon(candidate) or min(_edge_lengths(candidate)) < _MIN_BASE_EDGE:
                continue
            angles = _interior_angles(candidate)
            if any(value > _MAX_MAJORITY_ANGLE for position, value in enumerate(angles) if position != index):
                continue
            if reflex:
                if not _MIN_REFLEX_ANGLE <= angles[index] <= _MAX_REFLEX_ANGLE:
                    continue
                if _notch_depth(candidate, index) < _MIN_NOTCH_DEPTH_RATIO * _diameter(candidate):
                    continue
            else:
                if not _MIN_DENT_ANGLE <= angles[index] <= _MAX_MAJORITY_ANGLE:
                    continue
                if not _is_convex(candidate):
                    continue
            return candidate
    return None


def _notch_depth(vertices: Sequence[Point], index: int) -> float:
    """How far a vertex sits inside the straight line joining its neighbours.

    This is the perpendicular offset, which is what the eye actually reads as
    the depth of a notch.  Measuring instead to the midpoint of the two
    neighbours - the obvious formula, and the one this task used - overstates a
    notch by however far the vertex is displaced ALONG the chord, which for an
    oblique corner is most of the distance.  On delivered rasters that gap was
    close to a factor of two: a gate nominally at eleven percent of the diameter
    was passing notches five percent deep, one stroke width on the plate.
    """

    count = len(vertices)
    previous, current, following = vertices[index - 1], vertices[index], vertices[(index + 1) % count]
    dx, dy = following[0] - previous[0], following[1] - previous[1]
    span = math.hypot(dx, dy)
    if span < 1e-9:
        return 0.0
    return abs((current[0] - previous[0]) * dy - (current[1] - previous[1]) * dx) / span


def _decisive_reflex(vertices: Sequence[Point]) -> bool:
    """Exactly one reflex corner, deep and sharply turned enough to see.

    Checked after the presentation budget is applied, not only on the base
    shape: an anisotropic stretch can flatten a notch that was decisive before
    placement, which would ship a convexity outlier nobody can read.
    """

    angles = _interior_angles(vertices)
    reflex = [value for value in angles if value > 180.0]
    if len(reflex) != 1 or not _MIN_REFLEX_ANGLE <= reflex[0] <= _MAX_REFLEX_ANGLE:
        return False
    index = next(position for position, value in enumerate(angles) if value > 180.0)
    return _notch_depth(vertices, index) >= _MIN_NOTCH_DEPTH_RATIO * _diameter(vertices)


def _mirror_iou(vertices: Sequence[Point], resolution: int = 128) -> float:
    """Silhouette overlap between the shape and its horizontal reflection."""

    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0 * 1.08 + 1e-6
    grid_x = np.linspace(cx - half, cx + half, resolution)
    grid_y = np.linspace(cy - half, cy + half, resolution)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    inside = np.zeros_like(mesh_x, dtype=bool)
    count = len(vertices)
    for index in range(count):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % count]
        straddles = (y1 > mesh_y) != (y2 > mesh_y)
        crossing = (x2 - x1) * (mesh_y - y1) / (1e-12 if y2 == y1 else (y2 - y1)) + x1
        inside ^= straddles & (mesh_x < crossing)
    mirrored = inside[::-1, :]
    union = int((inside | mirrored).sum())
    return float((inside & mirrored).sum() / union) if union else 1.0


def _half_profile_base(
    rng: random.Random,
    *,
    n_inner: int,
    asymmetric: bool,
    cluster: float,
    spread: float,
) -> List[Point] | None:
    """Convex polygon assembled from an upper and a lower half profile.

    The symmetric variant mirrors one profile exactly.  The asymmetric variant
    samples the lower profile independently and must clear a silhouette-level
    mirror-mismatch margin, so the outlier differs over its whole outline
    rather than at one nudged vertex.
    """

    for _ in range(80):
        left = -58.0 * (1.0 - spread * rng.random())
        right = 58.0 * (1.0 - spread * rng.random())

        def profile() -> List[Tuple[float, float]]:
            if n_inner == 1:
                return [(rng.uniform(-24.0, 24.0), rng.uniform(30.0, 56.0) * (1.0 - 0.4 * spread * rng.random()))]
            span = 34.0 * (0.55 + 0.45 * cluster / 2.2)
            return [
                (rng.uniform(-span, -6.0), rng.uniform(28.0, 54.0) * (1.0 - 0.4 * spread * rng.random())),
                (rng.uniform(6.0, span), rng.uniform(28.0, 54.0) * (1.0 - 0.4 * spread * rng.random())),
            ]

        upper = profile()
        lower = list(upper) if not asymmetric else profile()
        vertices: List[Point] = (
            [(left, 0.0)]
            + [(x, -y) for x, y in upper]
            + [(right, 0.0)]
            + [(x, y) for x, y in reversed(lower)]
        )
        if len(vertices) != 2 + 2 * n_inner:
            continue
        if not _is_convex(vertices) or not _is_simple_polygon(vertices):
            continue
        if _has_horizontal_symmetry(vertices) != (not asymmetric):
            continue
        if max(_interior_angles(vertices)) > _MAX_MAJORITY_ANGLE:
            continue
        if min(_edge_lengths(vertices)) < _MIN_HALF_PROFILE_EDGE:
            continue
        if asymmetric and _mirror_iou(vertices) > _SEARCH_OUTLIER_MIRROR_IOU:
            continue
        return vertices
    return None


def _candidate(vertices: Sequence[Point], *, closed: bool = True) -> Candidate:
    return {"vertices": [(float(x), float(y)) for x, y in vertices], "closed": bool(closed)}


def _structured_candidates(task: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The candidate payload carried in the structured answer and rendered from."""

    return [
        {
            "label": _LABELS[index],
            "vertices": [[round(x, _VERTEX_PRECISION), round(y, _VERTEX_PRECISION)]
                         for x, y in candidate["vertices"]],
            "closed": bool(candidate["closed"]),
            "side_count": len(candidate["vertices"]),
            "convex": _is_convex(candidate["vertices"]),
            "horizontal_symmetry": _has_horizontal_symmetry(candidate["vertices"]),
            "property_value": task["property_values"][index],
        }
        for index, candidate in enumerate(task["candidates"])
    ]


def _structured_source(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Minimal semantic source the shared renderer draws from."""

    return {"candidates": _structured_candidates(task)}


def _placement_budget(rng: random.Random, *, rotate: bool) -> Dict[str, Any]:
    """One answer-irrelevant presentation budget, drawn from the shared law.

    The vertex noise is carried as a seed rather than as sampled values so that
    a budget is a small reproducible vector that can be perturbed and replayed.
    """

    return {
        "aniso": math.sqrt(rng.uniform(1.0, _MAX_ANISO_RATIO)) ** rng.choice((-1.0, 1.0)),
        "axis": rng.uniform(0.0, math.pi) if rotate else 0.0,
        "theta": rng.uniform(0.0, 2.0 * math.pi) if rotate else 0.0,
        "flip": rng.choice((1.0, -1.0)),
        "jitter": rng.uniform(0.030, 0.100),
        "area_gain": rng.uniform(0.84, 1.18),
        "offset": (rng.uniform(-6.0, 6.0), rng.uniform(-6.0, 6.0)),
        "seed": rng.getrandbits(30),
    }


def _perturb_budget(
    rng: random.Random, budget: Mapping[str, Any], step: float, *, rotate: bool
) -> Dict[str, Any]:
    """A neighbouring budget, for the local search that closes on a target.

    ``area_gain`` is deliberately left alone.  It moves no scale-free statistic,
    so the search has no opinion about it, and holding it fixed keeps absolute
    size an untouched i.i.d. draw instead of a quantity the search wanders in.
    """

    limit = math.sqrt(_MAX_ANISO_RATIO)
    moved = dict(budget)
    moved["aniso"] = min(max(budget["aniso"] * math.exp(rng.gauss(0.0, 0.20 * step)), 1.0 / limit), limit)
    if rotate:
        moved["axis"] = (budget["axis"] + rng.gauss(0.0, 0.55 * step)) % math.pi
        moved["theta"] = (budget["theta"] + rng.gauss(0.0, 0.90 * step)) % (2.0 * math.pi)
    moved["jitter"] = min(max(budget["jitter"] + rng.gauss(0.0, 0.030 * step), 0.030), 0.100)
    if rng.random() < 0.4 * step:
        moved["seed"] = rng.getrandbits(30)
    return moved


def _place(
    vertices: Sequence[Point],
    budget: Mapping[str, Any],
    *,
    mirror_jitter: bool,
) -> List[Point]:
    """Apply one answer-irrelevant presentation budget to a base shape."""

    rng = random.Random(budget["seed"])
    aniso = budget["aniso"]
    axis = budget["axis"]
    theta = budget["theta"]
    flip = budget["flip"]
    jitter = budget["jitter"]
    area_gain = budget["area_gain"]
    offset = budget["offset"]
    count = len(vertices)
    cx0, cy0 = _centroid(vertices)
    # Jitter is a fraction of the shape's own mean vertex radius, so the budget
    # means the same thing at every size.
    reference = sum(math.hypot(x - cx0, y - cy0) for x, y in vertices) / count
    noise = [
        (rng.uniform(-jitter, jitter) * reference, rng.uniform(-jitter, jitter) * reference)
        for _ in range(count)
    ]
    if mirror_jitter:
        # Horizontal symmetry is frame-dependent, so the jitter must respect the
        # mirror axis or it would silently flip the sampled property.
        cy = sum(y for _, y in vertices) / count
        for index, (x, y) in enumerate(vertices):
            if abs(y - cy) < 1e-9:
                noise[index] = (noise[index][0], 0.0)
                continue
            partner = next(
                (
                    other
                    for other, (qx, qy) in enumerate(vertices)
                    if abs(qx - x) < 1e-6 and abs(qy - (2.0 * cy - y)) < 1e-6
                ),
                None,
            )
            if partner is not None and partner > index:
                noise[partner] = (noise[index][0], -noise[index][1])
    ca, sa = math.cos(axis), math.sin(axis)
    ct, st = math.cos(theta), math.sin(theta)
    placed: List[Point] = []
    for (x, y), (jx, jy) in zip(vertices, noise):
        x, y = (x + jx) * flip, y + jy
        u, v = x * ca + y * sa, -x * sa + y * ca
        u, v = u * aniso, v / aniso
        x, y = u * ca - v * sa, u * sa + v * ca
        placed.append((x * ct - y * st, x * st + y * ct))
    placed = _normalize_area(placed, _BASE_AREA * area_gain)
    cx, cy = _centroid(placed)
    return [(x - cx + offset[0], y - cy + offset[1]) for x, y in placed]


def _structure_ok(vertices: Sequence[Point]) -> bool:
    """The cheap structural gates every placed candidate must clear.

    The local search re-places shapes outside the sampling loop, so these gates
    have to be callable from both or a polished shape could ship without them.
    Angles are held to the search RESERVE, not to the floor validation asserts,
    so a shape that improves its presentation match by flattening a corner runs
    out of room well before the corner stops being countable.
    """

    if max(max(abs(x), abs(y)) for x, y in vertices) > _COORD_CAP:
        return False
    if not _is_simple_polygon(vertices) or min(_edge_lengths(vertices)) < _MIN_RENDERED_EDGE:
        return False
    # No candidate may be a sliver.  Elongation is the cheapest way to buy
    # boundary length, so an unbounded matcher reaches for it first and the
    # candidate that needs the most - the open one, whose break costs it visible
    # ink - ends up the only splinter on the board.  That is a silhouette-level
    # giveaway no matched magnitude would report.
    xs = [x for x, _ in vertices]
    ys = [y for _, y in vertices]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    if max(width, height) > _MAX_BBOX_ASPECT * max(min(width, height), 1e-9):
        return False
    # Reflex corners are held to the same reserve as convex ones: a notch that
    # is barely past straight is as unreadable as a corner that is barely short
    # of it.
    return not any(
        abs(180.0 - value) < _SEARCH_ANGLE_FROM_STRAIGHT for value in _interior_angles(vertices)
    )


def _placement_ok(vertices: Sequence[Point], validate) -> bool:
    return _structure_ok(vertices) and bool(validate(vertices))


def _open_boundary(vertices: Sequence[Point]) -> List[Point]:
    """Rotate an outline so its break falls on its longest edge.

    The break is a fixed share of the root area but is also capped as a share of
    the edge it sits on, so on a short closing edge it shrinks.  That coupling
    is a lever: matching painted compactness could then be done by reshaping
    whichever edge happened to be last, which showed up as the open candidate
    carrying by far the most uneven edge lengths of the four.  Putting the break
    on the longest edge makes the cap inactive and the gap a pure function of
    area, so compactness can only be matched by the shape itself.
    """

    lengths = _edge_lengths(vertices)
    cut = max(range(len(lengths)), key=lengths.__getitem__)
    return list(vertices[cut + 1 :]) + list(vertices[: cut + 1])


def _variant_cloud(
    build_base,
    rng: random.Random,
    *,
    count: int,
    rotate: bool,
    mirror_jitter: bool,
    validate,
) -> List[Tuple[List[Point], Dict[str, Any], List[Point]]]:
    """Admissible placements of independently sampled bases.

    Every candidate of a task draws its budget from this one distribution, so
    answer-irrelevant irregularity carries no signal about which is the outlier.
    Slots built from the same base builder share one cloud: they are i.i.d.
    draws from the same law either way, and pooling them is what pays for the
    denser sampling the target screening needs.  Each entry keeps its base and
    its budget so the local search can re-place it.
    """

    cloud: List[Tuple[List[Point], Dict[str, Any], List[Point]]] = []
    for _ in range(count):
        cluster = rng.uniform(*_CLUSTER_RANGE)
        spread = rng.uniform(*_SPREAD_RANGE)
        base = build_base(rng, cluster=cluster, spread=spread)
        if base is None:
            continue
        budget = _placement_budget(rng, rotate=rotate)
        vertices = _place(base, budget, mirror_jitter=mirror_jitter)
        if not _placement_ok(vertices, validate):
            continue
        cloud.append((base, budget, vertices))
    return cloud


def _oriented(vertices: Sequence[Point], *, closed: bool) -> List[Point]:
    """The vertex order actually delivered, which an open outline rotates."""

    return list(vertices) if closed else _open_boundary(vertices)


def _stat_row(vertices: Sequence[Point], *, closed: bool, scales) -> np.ndarray:
    stats = _display_stats(_oriented(vertices, closed=closed), closed=closed)
    return np.array([stats[key] / scales[key] for key in _MATCH_KEYS], dtype=float)


def _slot_matrix(cloud, *, closed: bool, scales) -> np.ndarray:
    """One slot's cloud as a scale-free matrix, one row per variant.

    Openness changes the painted boundary and therefore the statistics, so a
    geometry cloud shared between a closed and an open slot still yields one
    matrix per slot.
    """

    return np.asarray(
        [_stat_row(entry[2], closed=closed, scales=scales) for entry in cloud], dtype=float
    )


def _polish(
    entry,
    target: np.ndarray,
    rng: random.Random,
    *,
    scales,
    rotate: bool,
    mirror_jitter: bool,
    validate,
    closed: bool,
):
    """Walk one variant's presentation budget downhill onto a target.

    Random sampling cannot put a candidate on a given target: in five matched
    magnitudes the nearest of a few hundred sampled variants still sits about a
    third of a standard deviation away, and that discretisation - not any
    property-bearing difference - was most of the residual the matcher was
    left holding.  A short self-adapting hill climb over the same presentation
    budget the sampler draws removes it, and every structural gate is re-checked
    on each trial so nothing reaches a target by leaving the admissible set.
    """

    base, budget, vertices = entry
    best = (budget, vertices)
    best_distance = float(np.linalg.norm(_stat_row(vertices, closed=closed, scales=scales) - target))
    current, current_distance = best, best_distance
    step = 1.0
    for _ in range(_POLISH_STEPS):
        trial = _perturb_budget(rng, current[0], step, rotate=rotate)
        placed = _place(base, trial, mirror_jitter=mirror_jitter)
        # The property check is the expensive one - the symmetry margin rasters
        # a silhouette - so it runs only on a trial that would be accepted.
        if _structure_ok(placed):
            distance = float(np.linalg.norm(_stat_row(placed, closed=closed, scales=scales) - target))
            if distance < current_distance and validate(placed):
                current, current_distance = (trial, placed), distance
                if distance < best_distance:
                    best, best_distance = current, distance
                step = min(step * 1.15, 1.0)
                continue
        step = max(step * 0.90, 0.12)
    return (base, best[0], best[1])


def _squared_distances(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Pairwise squared distances, expanded so the whole screen is one matmul."""

    return np.maximum(
        np.einsum("ij,ij->i", points, points)[:, None]
        - 2.0 * points @ matrix.T
        + np.einsum("ij,ij->i", matrix, matrix)[None, :],
        0.0,
    )


def _admissible_targets(
    matrices: Sequence[np.ndarray],
    sources: Sequence[np.ndarray],
    rng: random.Random,
):
    """Presentation targets every candidate can reach, and the radius used.

    A target only three candidates can reach leaves the fourth systematically
    offset on that magnitude, which is exactly the shortcut this task must not
    contain.  Screening every proposal against every slot removes those targets
    before any geometry is picked, so the four draws come from the region the
    four clouds jointly cover rather than from one cloud's own region.  Each
    distinct cloud contributes the same proposals, so the surviving region is
    not tilted toward whichever cloud happens to fill more slots.
    """

    noise = np.random.default_rng(rng.getrandbits(64))
    blocks: List[np.ndarray] = []
    for matrix in sources:
        for jitter in _PROPOSAL_JITTERS:
            blocks.append(matrix if jitter == 0.0 else matrix + noise.normal(0.0, jitter, matrix.shape))
    proposals = np.vstack(blocks)
    reach = np.zeros(proposals.shape[0], dtype=float)
    for matrix in matrices:
        np.maximum(reach, np.sqrt(_squared_distances(proposals, matrix).min(axis=1)), out=reach)
    for tolerance in _ADMIT_TOLERANCES:
        admitted = proposals[reach <= tolerance]
        if admitted.shape[0] >= _MIN_ADMISSIBLE:
            return admitted, tolerance
    return None, None


def _property_value(candidate: Candidate, rule_kind: str) -> int | bool:
    vertices = candidate["vertices"]
    if rule_kind == "side_count":
        return len(vertices)
    if rule_kind == "convexity":
        return bool(candidate["closed"]) and _is_convex(vertices)
    if rule_kind == "horizontal_symmetry":
        return _has_horizontal_symmetry(vertices)
    if rule_kind == "closure":
        return bool(candidate["closed"])
    raise ValueError(f"Unknown odd-one-out rule: {rule_kind}")


def _non_target_signature(candidate: Candidate, rule_kind: str) -> Tuple[Any, ...]:
    values = {
        "side_count": len(candidate["vertices"]),
        "convexity": _is_convex(candidate["vertices"]),
        "horizontal_symmetry": _has_horizontal_symmetry(candidate["vertices"]),
        "closure": bool(candidate["closed"]),
    }
    return tuple(values[name] for name in ("side_count", "convexity", "horizontal_symmetry", "closure") if name != rule_kind)


def _validate_task(task: Dict[str, Any]) -> None:
    candidates = task["candidates"]
    if len(candidates) != 4:
        raise RuntimeError("odd_one_out_shape requires exactly four candidates")
    values = [_property_value(candidate, task["rule_kind"]) for candidate in candidates]
    target_value = task["target_value"]
    matching = [index for index, value in enumerate(values) if value == target_value]
    odd = [index for index, value in enumerate(values) if value != target_value]
    if len(matching) != 3 or len(odd) != 1 or odd[0] != task["correct_index"]:
        raise RuntimeError("odd_one_out_shape construction is not uniquely decidable")

    # For every rule, hold all qualitative non-target properties fixed when
    # doing so is geometrically meaningful.  Side-count necessarily changes
    # the exact vertex count, so compare the remaining three properties.
    signatures = [_non_target_signature(candidate, task["rule_kind"]) for candidate in candidates]
    if len(set(signatures)) != 1:
        raise RuntimeError(
            f"odd_one_out_shape {task['rule_kind']} candidate changes a non-target property: {signatures}"
        )

    from data.in_context_learning.core.direct_delivery import _ODD_OPEN_GAP

    if abs(_ODD_OPEN_GAP - _OPEN_GAP_FRACTION) > 1e-9:
        raise RuntimeError(
            "odd_one_out_shape open-boundary gap disagrees with the delivery renderer"
        )

    rule_kind = task["rule_kind"]
    odd_index = int(task["correct_index"])
    for index, candidate in enumerate(candidates):
        vertices = candidate["vertices"]
        # Structural validity has to hold at generation, not only in the
        # verifier's parser: a self-intersecting or degenerate jittered shape
        # would otherwise ship as a silently unverifiable sample.
        if not _is_simple_polygon(vertices):
            raise RuntimeError("odd_one_out_shape candidate is self-intersecting or degenerate")
        if max(max(abs(x), abs(y)) for x, y in vertices) > _COORD_CAP:
            raise RuntimeError("odd_one_out_shape candidate leaves the authored coordinate bound")
        if min(_edge_lengths(vertices)) < _MIN_RENDERED_EDGE:
            raise RuntimeError("odd_one_out_shape candidate has an edge too short to raster")
        angles = _interior_angles(vertices)
        # No non-feature vertex may approach straight, or a six-sided shape
        # could read as five-sided in the final raster.
        if any(abs(180.0 - value) < _MIN_ANGLE_FROM_STRAIGHT for value in angles if value <= 180.0):
            raise RuntimeError("odd_one_out_shape candidate has a vertex too close to straight")
        reflex = [value for value in angles if value > 180.0]
        if rule_kind == "convexity" and index == odd_index:
            if not _decisive_reflex(vertices):
                raise RuntimeError("odd_one_out_shape convexity outlier lacks one decisive reflex notch")
        elif reflex:
            raise RuntimeError("odd_one_out_shape majority candidate is unexpectedly concave")
        if rule_kind == "horizontal_symmetry" and index == odd_index:
            if _mirror_iou(vertices) > _MAX_OUTLIER_MIRROR_IOU:
                raise RuntimeError("odd_one_out_shape symmetry outlier is not visibly asymmetric")

    if rule_kind == "side_count":
        counts = sorted({len(candidate["vertices"]) for candidate in candidates})
        if len(counts) != 2 or counts[1] - counts[0] < 2:
            raise RuntimeError("odd_one_out_shape side-count split must differ by at least two sides")


def _polygon_area(vertices: Sequence[Point]) -> float:
    return abs(
        sum(
            vertices[index][0] * vertices[(index + 1) % len(vertices)][1]
            - vertices[(index + 1) % len(vertices)][0] * vertices[index][1]
            for index in range(len(vertices))
        )
    ) / 2.0


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) <= 1e-7:
        return 0
    return 1 if value > 0 else -1


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    return _orientation(a, b, c) != _orientation(a, b, d) and _orientation(c, d, a) != _orientation(c, d, b)


def _is_simple_polygon(vertices: Sequence[Point]) -> bool:
    if len(vertices) < 3:
        return False
    edge_count = len(vertices)
    for left in range(edge_count):
        a, b = vertices[left], vertices[(left + 1) % edge_count]
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 8.0:
            return False
        for right in range(left + 1, edge_count):
            if right in {left, (left + 1) % edge_count} or left == (right + 1) % edge_count:
                continue
            c, d = vertices[right], vertices[(right + 1) % edge_count]
            if _segments_cross(a, b, c, d):
                return False
    return _polygon_area(vertices) >= 500.0


def _parse_structured_candidate(raw: Any, expected_label: str) -> Candidate | None:
    if not isinstance(raw, dict) or raw.get("label") != expected_label:
        return None
    raw_vertices = raw.get("vertices")
    if not isinstance(raw_vertices, list) or not 3 <= len(raw_vertices) <= 6:
        return None
    vertices: List[Point] = []
    for point in raw_vertices:
        if not isinstance(point, list) or len(point) != 2:
            return None
        x, y = point
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
        ):
            return None
        x_value, y_value = float(x), float(y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            return None
        if abs(x_value) > _COORD_CAP or abs(y_value) > _COORD_CAP:
            return None
        vertices.append((x_value, y_value))
    if not _is_simple_polygon(vertices) or not isinstance(raw.get("closed"), bool):
        return None
    return {"vertices": vertices, "closed": bool(raw["closed"])}


def _expected_trace(matching_indices: Sequence[int]) -> List[Dict[str, Any]]:
    # The named-property comparison is atomic: marking all three majority
    # cards is redundant answer annotation, not a natural intermediate state.
    # Keep only input and one committed selection.
    return [
        {"selected": False},
        {"selected": True},
    ]


def _expected_render_geometry() -> Dict[str, Any]:
    """The geometry the delivered media actually uses.

    The four boxes are the collection slots, not the retired in-image answer
    cards; they are what the shared option-card plumbing counts candidates by.
    """

    return {
        "slot_centers": [list(center) for center in _SLOT_CENTERS],
        "panel_bboxes": [
            [cx - _SLOT_HALF, cy - _SLOT_HALF, cx + _SLOT_HALF, cy + _SLOT_HALF]
            for cx, cy in _SLOT_CENTERS
        ],
        "ring_radius": _RING_RADIUS,
        "ring_rgb": list(_RING_RGB),
    }


def _expected_difficulty(rule_kind: str) -> str:
    if rule_kind in {"side_count", "closure"}:
        return "minimum"
    if rule_kind == "convexity":
        return "typical"
    return "maximum"


@lru_cache(maxsize=4096)
def _semantic_replay_cached(serialized_structured: str) -> Dict[str, Any]:
    try:
        structured = json.loads(serialized_structured)
    except Exception:
        return {
            "semantic_source_valid": False,
            "rule_replay_valid": False,
            "metadata_integrity_valid": False,
            "failure_reason": "structured semantic source is not valid JSON",
        }
    if not isinstance(structured, dict):
        return {
            "semantic_source_valid": False,
            "rule_replay_valid": False,
            "metadata_integrity_valid": False,
            "failure_reason": "structured semantic source must be an object",
        }
    rule_kind = structured.get("rule_kind")
    raw_candidates = structured.get("candidates")
    if rule_kind not in _RULE_LABELS or not isinstance(raw_candidates, list) or len(raw_candidates) != 4:
        return {
            "semantic_source_valid": False,
            "rule_replay_valid": False,
            "metadata_integrity_valid": False,
            "failure_reason": "missing rule kind or four candidate geometries",
        }
    candidates = [
        _parse_structured_candidate(raw_candidate, _LABELS[index])
        for index, raw_candidate in enumerate(raw_candidates)
    ]
    if any(candidate is None for candidate in candidates):
        return {
            "semantic_source_valid": False,
            "rule_replay_valid": False,
            "metadata_integrity_valid": False,
            "failure_reason": "candidate geometry schema is malformed or degenerate",
        }
    parsed_candidates = [candidate for candidate in candidates if candidate is not None]
    property_values = [_property_value(candidate, str(rule_kind)) for candidate in parsed_candidates]
    frequencies = Counter(property_values)
    majority_values = [value for value, count in frequencies.items() if count == 3]
    if len(majority_values) != 1:
        return {
            "semantic_source_valid": True,
            "rule_replay_valid": False,
            "metadata_integrity_valid": False,
            "property_values": property_values,
            "failure_reason": "named property does not have a unique three-to-one split",
        }
    target_value = majority_values[0]
    matching_indices = [index for index, value in enumerate(property_values) if value == target_value]
    odd_indices = [index for index, value in enumerate(property_values) if value != target_value]
    signatures = [_non_target_signature(candidate, str(rule_kind)) for candidate in parsed_candidates]
    rule_replay_valid = len(matching_indices) == 3 and len(odd_indices) == 1 and len(set(signatures)) == 1
    if not rule_replay_valid:
        return {
            "semantic_source_valid": True,
            "rule_replay_valid": False,
            "metadata_integrity_valid": False,
            "property_values": property_values,
            "failure_reason": "odd candidate changes a non-target property or is not unique",
        }
    odd_index = odd_indices[0]
    metadata_checks: List[bool] = [
        structured.get("schema_version") == _SCHEMA_VERSION,
        structured.get("rule_label") == _RULE_LABELS[str(rule_kind)],
        structured.get("rule_definition") == _RULE_DEFINITIONS[str(rule_kind)],
        structured.get("difficulty") == _expected_difficulty(str(rule_kind)),
        structured.get("target_value") == target_value,
        structured.get("correct_index") == odd_index,
        structured.get("matching_indices") == matching_indices,
        structured.get("state_trace") == _expected_trace(matching_indices),
        structured.get("render_geometry") == _expected_render_geometry(),
        "fill_rgb" not in structured,
    ]
    for index, (raw_candidate, candidate, property_value) in enumerate(
        zip(raw_candidates, parsed_candidates, property_values)
    ):
        metadata_checks.extend(
            [
                raw_candidate.get("label") == _LABELS[index],
                raw_candidate.get("side_count") == len(candidate["vertices"]),
                raw_candidate.get("convex") is _is_convex(candidate["vertices"]),
                raw_candidate.get("horizontal_symmetry") is _has_horizontal_symmetry(candidate["vertices"]),
                raw_candidate.get("property_value") == property_value,
            ]
        )
    metadata_integrity_valid = all(metadata_checks)
    return {
        "semantic_source_valid": True,
        "rule_replay_valid": True,
        "metadata_integrity_valid": metadata_integrity_valid,
        "rule_kind": rule_kind,
        "target_value": target_value,
        "property_values": property_values,
        "matching_indices": matching_indices,
        "odd_index": odd_index,
        "expected_option": _LABELS[odd_index],
        "failure_reason": "ok" if metadata_integrity_valid else "redundant structured metadata disagrees with independent replay",
    }


def replay_odd_one_out_semantics(structured: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        serialized = json.dumps(dict(structured), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return {
            "semantic_source_valid": False,
            "rule_replay_valid": False,
            "metadata_integrity_valid": False,
            "cache_hit": False,
            "failure_reason": "structured semantic source is not JSON-serializable",
        }
    before = _semantic_replay_cached.cache_info()
    replay = dict(_semantic_replay_cached(serialized))
    after = _semantic_replay_cached.cache_info()
    replay["cache_hit"] = after.hits > before.hits
    return replay


def _read_rgb(path: str | Path) -> np.ndarray | None:
    try:
        import cv2
    except Exception:
        return None
    image = cv2.imread(Path(path).as_posix(), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _ring_band(slot_index: int) -> np.ndarray:
    """Annulus where the answer ring is drawn around one collection slot."""

    cx, cy = _SLOT_CENTERS[slot_index]
    ys, xs = np.ogrid[:512, :512]
    distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return (distance >= _RING_RADIUS - 7.0) & (distance <= _RING_RADIUS + 7.0)


_RING_BANDS = tuple(_ring_band(index) for index in range(4))


def _preservation_mask() -> np.ndarray:
    """Everything outside every legal ring band must stay pixel-identical."""

    mask = np.ones((512, 512), dtype=bool)
    for band in _RING_BANDS:
        mask &= ~band
    return mask


_PRESERVATION_MASK = _preservation_mask()
_PRESERVATION_SAMPLE_MASK = _PRESERVATION_MASK.copy()
_PRESERVATION_SAMPLE_MASK[:] = False
_PRESERVATION_SAMPLE_MASK[::4, ::4] = _PRESERVATION_MASK[::4, ::4]
_PRESERVATION_SAMPLE_ROWS, _PRESERVATION_SAMPLE_COLS = np.nonzero(_PRESERVATION_SAMPLE_MASK)


def _static_mutation_fraction(input_image: np.ndarray, candidate_image: np.ndarray) -> float:
    if input_image.shape != candidate_image.shape or input_image.shape[:2] != (512, 512):
        return 1.0
    input_pixels = input_image[_PRESERVATION_SAMPLE_ROWS, _PRESERVATION_SAMPLE_COLS].astype(np.int16)
    return _static_mutation_fraction_from_pixels(input_pixels, candidate_image)


def _static_mutation_fraction_from_pixels(input_pixels: np.ndarray, candidate_image: np.ndarray) -> float:
    if candidate_image.shape[:2] != (512, 512):
        return 1.0
    candidate_pixels = candidate_image[_PRESERVATION_SAMPLE_ROWS, _PRESERVATION_SAMPLE_COLS].astype(np.int16)
    delta = np.max(np.abs(candidate_pixels - input_pixels), axis=1)
    return float(np.mean(delta > 35))


class OddOneOutShapePattern(RenderedTaskGenerator):
    task_name = "odd_one_out_shape"
    task_family = "in_context_learning"
    video_fps = 16

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        difficulty: str | int | None = None,
        video_fps: int = 16,
        generate_videos: bool = True,
        **kwargs: Any,
    ) -> None:
        effective_seed = random_seed if random_seed is not None else seed
        self.rng = random.Random(effective_seed)
        self.difficulty = difficulty
        self.video_fps = int(video_fps)
        self.generate_videos = bool(kwargs.get("video", generate_videos))
        self.image_size = (512, 512)
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / "videogym_odd_one_out_shape"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    def _rule_pool(self) -> Tuple[str, ...]:
        normalized = str(self.difficulty).strip().lower() if self.difficulty is not None else ""
        if normalized in {"0", "min", "minimum", "easy"}:
            return ("side_count", "closure")
        if normalized in {"1", "typical", "medium"}:
            return ("convexity",)
        if normalized in {"2", "max", "maximum", "hard"}:
            return ("horizontal_symmetry",)
        return ("side_count", "convexity", "horizontal_symmetry", "closure")

    def _candidate_specs(
        self, rule_kind: str
    ) -> Tuple[List[Tuple[Any, Any]], Tuple[int, int, int, int], int | bool]:
        """Base builders, the slot-to-builder map, and the majority value.

        Slots 0-2 carry the majority value and slot 3 carries the outlier.  Two
        slots sharing a builder share one cloud, because they are i.i.d. draws
        from the same law either way; they are never allowed to pick the same
        variant, so the three majority shapes stay independent rather than
        cloned - three near-identical shapes beside one different shape would
        let the outlier be found without evaluating the named property.
        """

        rng = self.rng
        if rule_kind == "side_count":
            majority_sides, odd_sides = rng.choice(_SIDE_PAIRS)
            majority = (lambda r, **kw: _convex_base(r, majority_sides, **kw), _is_convex)
            outlier = (lambda r, **kw: _convex_base(r, odd_sides, **kw), _is_convex)
            return [majority, outlier], (0, 0, 0, 1), majority_sides

        if rule_kind == "convexity":
            n_sides = rng.choice((5, 6))
            majority = (
                lambda r, **kw: _notched_base(r, n_sides, reflex=False, **kw),
                _is_convex,
            )
            outlier = (
                lambda r, **kw: _notched_base(r, n_sides, reflex=True, **kw),
                lambda vertices: not _is_convex(vertices) and _decisive_reflex(vertices),
            )
            return [majority, outlier], (0, 0, 0, 1), True

        if rule_kind == "horizontal_symmetry":
            n_inner = rng.choice((1, 2))
            majority = (
                lambda r, **kw: _half_profile_base(r, n_inner=n_inner, asymmetric=False, **kw),
                _has_horizontal_symmetry,
            )
            outlier = (
                lambda r, **kw: _half_profile_base(r, n_inner=n_inner, asymmetric=True, **kw),
                # The margin is re-checked after placement: an anisotropic
                # stretch can compress the vertical difference and quietly
                # shrink an asymmetry that was decisive in the base shape.
                lambda vertices: (
                    _is_convex(vertices)
                    and not _has_horizontal_symmetry(vertices)
                    and _mirror_iou(vertices) <= _SEARCH_OUTLIER_MIRROR_IOU
                ),
            )
            return [majority, outlier], (0, 0, 0, 1), True

        if rule_kind == "closure":
            n_sides = rng.choice((4, 5, 6))
            slot = (lambda r, **kw: _convex_base(r, n_sides, **kw), _is_convex)
            # Openness is applied at draw time, not at sampling time, so the
            # closed and open candidates come from one identical shape
            # distribution.  Biasing the open candidate's shapes to compensate
            # for the ink its break removes only moved the leak: it never
            # touches the diameter, so the bias showed up there instead.
            return [slot], (0, 0, 0, 0), True

        raise ValueError(f"Unknown rule kind: {rule_kind}")

    def _build_candidates(self, rule_kind: str) -> Tuple[List[Candidate], int | bool] | None:
        """Sample four candidates from one jointly reachable presentation region.

        Bounded rejection grows the candidate clouds until enough targets are
        reachable by all four slots; each candidate then draws its own target
        i.i.d. from that set.  Identically distributed targets inside a region
        every candidate covers is what makes presentation parity executable:
        the four draws are exchangeable by construction, so no answer-irrelevant
        magnitude can rank the outlier beyond the residual match error.
        """

        rng = self.rng
        builders, groups, target_value = self._candidate_specs(rule_kind)
        rotate = rule_kind != "horizontal_symmetry"
        mirror_jitter = rule_kind == "horizontal_symmetry"
        closed_flags = [not (rule_kind == "closure" and slot == 3) for slot in range(4)]
        clouds: List[List[List[Point]]] = [[] for _ in builders]
        for _ in range(_MATCH_ATTEMPTS):
            for index, (build_base, validate) in enumerate(builders):
                clouds[index].extend(
                    _variant_cloud(
                        build_base,
                        rng,
                        count=_VARIANTS_PER_ROUND,
                        rotate=rotate,
                        mirror_jitter=mirror_jitter,
                        validate=validate,
                    )
                )
            if any(len(cloud) < _MIN_CLOUD for cloud in clouds):
                continue
            pooled = [
                _display_stats(
                    _oriented(entry[2], closed=closed_flags[slot]), closed=closed_flags[slot]
                )
                for slot, group in enumerate(groups)
                for entry in clouds[group]
            ]
            scales = {}
            for index, key in enumerate(_MATCH_KEYS):
                column = [stats[key] for stats in pooled]
                mean = sum(column) / len(column)
                spread = math.sqrt(sum((value - mean) ** 2 for value in column) / len(column))
                scales[key] = max(spread, 1e-6) / _MATCH_WEIGHTS[index]
            matrices = [
                _slot_matrix(clouds[group], closed=closed_flags[slot], scales=scales)
                for slot, group in enumerate(groups)
            ]
            keys = [(groups[slot], closed_flags[slot]) for slot in range(4)]
            sources = [matrices[slot] for slot in range(4) if keys[slot] not in keys[:slot]]
            admissible, _ = _admissible_targets(matrices, sources, rng)
            if admissible is None:
                continue

            # Each distinct slot kind is re-sampled as a set of variants walked
            # onto targets drawn i.i.d. from the admissible region.  This is the
            # step that makes the targets binding rather than aspirational:
            # picking the nearest of a few hundred randomly placed variants
            # leaves a third of a standard deviation of pure discretisation
            # error, which the four-way comparison then reads as signal.
            refined: Dict[Tuple[int, bool], List[Any]] = {}
            for slot in range(4):
                if keys[slot] in refined:
                    continue
                group, closed = keys[slot]
                validate = builders[group][1]
                entries = []
                for _ in range(_REFINE_COUNT):
                    target = admissible[rng.randrange(admissible.shape[0])]
                    ancestor = int(np.argmin(_squared_distances(target[None, :], matrices[slot])[0]))
                    entries.append(
                        (
                            ancestor,
                            _polish(
                                clouds[group][ancestor],
                                target,
                                rng,
                                scales=scales,
                                rotate=rotate,
                                mirror_jitter=mirror_jitter,
                                validate=validate,
                                closed=closed,
                            ),
                        )
                    )
                refined[keys[slot]] = entries
            # One draw per slot, each aimed at its own target from the shared
            # admissible set.  Two slots of the same kind never share a variant,
            # nor two variants walked from one sampled shape: either would put a
            # near-clone on the board and let the outlier be found as the one
            # shape without a twin.
            taken: Dict[Tuple[int, bool], set] = {key: set() for key in keys}
            picked: List[int] = []
            for slot in range(4):
                entries = refined[keys[slot]]
                used = taken[keys[slot]]
                ancestors = {entries[index][0] for index in used}
                choices = [
                    index
                    for index in range(len(entries))
                    if index not in used and entries[index][0] not in ancestors
                ]
                if not choices:
                    picked = []
                    break
                choice = rng.choice(choices)
                used.add(choice)
                picked.append(choice)
            if not picked:
                continue
            candidates = [
                _candidate(
                    _oriented(refined[keys[slot]][picked[slot]][1][2], closed=closed_flags[slot]),
                    closed=closed_flags[slot],
                )
                for slot in range(4)
            ]
            return candidates, target_value
        return None

    def _generate_task(self) -> Dict[str, Any]:
        for _ in range(20):
            rule_kind = self.rng.choice(self._rule_pool())
            built = self._build_candidates(rule_kind)
            if built is None:
                continue
            candidates, target_value = built
            self.rng.shuffle(candidates)
            values = [_property_value(candidate, rule_kind) for candidate in candidates]
            odd_indices = [index for index, value in enumerate(values) if value != target_value]
            if len(odd_indices) != 1:
                continue
            task = {
                "rule_kind": rule_kind,
                "rule_label": _RULE_LABELS[rule_kind],
                "rule_definition": _RULE_DEFINITIONS[rule_kind],
                "target_value": target_value,
                "candidates": candidates,
                "property_values": values,
                "correct_index": odd_indices[0],
                "difficulty": (
                    "minimum" if rule_kind in {"side_count", "closure"}
                    else "typical" if rule_kind == "convexity"
                    else "maximum"
                ),
            }
            _validate_task(task)
            return task
        raise RuntimeError("odd_one_out_shape: failed to generate a valid unique instance")

    @staticmethod
    def _matching_indices(task: Dict[str, Any]) -> List[int]:
        return [index for index in range(4) if index != int(task["correct_index"])]

    @staticmethod
    def _state_trace(task: Dict[str, Any]) -> List[Dict[str, Any]]:
        return _expected_trace(OddOneOutShapePattern._matching_indices(task))

    def _render_scene(self, task: Dict[str, Any], *, selected: bool = False) -> Image.Image:
        """Render through the single delivery renderer.

        ``direct_delivery`` owns every pixel this task ships, so the generator
        renders the same collection rather than maintaining a parallel drawing
        path that could drift from what the model and the verifier see.
        """

        from data.in_context_learning.core.direct_delivery import _odd_collection

        return _odd_collection(
            _structured_source(task),
            selected_index=int(task["correct_index"]) if selected else None,
        )

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        # The comparison is atomic: hold the unmarked collection, then commit the
        # single ring.  The family postprocessor rewrites this into the shared
        # local-opacity fade, so both states must come from the same renderer.
        holds = (9, 16)
        frames: List[np.ndarray] = []
        for selected, hold in zip((False, True), holds):
            frame = np.asarray(self._render_scene(task, selected=selected), dtype=np.uint8)
            frames.extend(frame.copy() for _ in range(hold))
        if len(frames) != 25 or (len(frames) - 1) % 4 != 0:
            raise RuntimeError("odd_one_out_shape video must contain exactly 25 = 4N+1 frames")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    @staticmethod
    def _contract(task: Dict[str, Any]) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The supplied problem image shows four unlabelled candidate shapes as one comparison set, "
                "followed by four independent candidate plates labelled A-D."
            ),
            symbol_legend=(
                f"The compared property {task['rule_label']} means: {task['rule_definition']}",
                "The source scene carries no majority or answer marks; the selected outlier is indicated by one complete ring around it.",
            ),
            rules=(
                "Compare only the named property; do not choose by position, size, orientation, or how irregular a shape looks.",
                "Exactly three candidates share one value of that property and exactly one candidate has a different value.",
                "The candidate with the different value is the odd one out.",
                "Each candidate is one simple polygonal outline. For SIDE COUNT, count straight boundary segments; for CONVEXITY, any inward notch makes a polygon non-convex; for HORIZONTAL SYMMETRY, reflect across that shape's own horizontal centerline; for CLOSED / OPEN, a visible break in the outline makes the boundary open.",
                "The four shapes deliberately differ in size, orientation and outline irregularity; only the named property decides the answer. Exactly one three-to-one split exists, so ties and multiple valid options do not occur.",
            ),
            modality_outputs=OPTION_MODALITY_OUTPUTS,
            preservation_constraints=OPTION_PRESERVATION_CONSTRAINTS,
            transition_constraints=OPTION_TRANSITION_CONSTRAINTS,
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="uppercase_option_letter",
                verifier_profile=_VERIFIER_PROFILE,
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_odd_one_out_named_property_trace_v3",
        )

    def _prompts(self, task: Dict[str, Any]) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract(task)
        prompts = {
            "question": (
                f"Compare the four candidates on {task['rule_label'].lower()} and identify the unique one "
                "whose value differs from the other three."
            ),
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid odd-one-out prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        sample_id = f'{self.task_name}-{self._sample_index:05d}'
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        answer_label = _LABELS[int(task['correct_index'])]
        FOUR_CHOICE_CODEC.format(answer_label)
        contract, prompts = self._prompts(task)
        task_contract = contract.to_dict()
        task_contract['answer'] = {**dict(task_contract['answer']), 'schema_version': _SCHEMA_VERSION, 'intermediate_supervision_schema': 'Single-step atomic comparison without intermediate board mutations.', 'final_deliverable': {'ti2v': 'input hold, simultaneous whole-border green fade, and final hold'}}
        task_contract['verifier'] = {'profile': _VERIFIER_PROFILE, 'semantic_source': 'rule_kind plus four candidate polygon geometries and closure flags', 'component_scores': {'ti2v': ['video_decode_score', 'semantic_source_score', 'rule_replay_score', 'metadata_integrity_score', 'media_contract_score', 'state_recognition_score', 'state_order_score', 'hold_score', 'preservation_score', 'final_state_score']}, 'binary_predicate': 'all modality-specific hard-gate component scores equal 1.0'}
        puzzle_image = self._render_scene(task)
        solution_image = self._render_scene(task, selected=True)
        puzzle_path = sample_dir / f'{sample_id}__puzzle.png'
        solution_path = sample_dir / f'{sample_id}__solution.png'
        video_path = sample_dir / f'{sample_id}__solution.webm'
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)
        solution_video_path, video_num_frames = self._render_video(task, video_path)
        candidates = _structured_candidates(task)
        sample: Dict[str, Any] = {'id': sample_id, 'task_name': self.task_name, 'task_family': self.task_family, 'task_contract': task_contract, 'question': prompts['question'], 'answer': answer_label, 'correct_option': answer_label, 'structured_answer': {'schema_version': _SCHEMA_VERSION, 'rule_kind': task['rule_kind'], 'rule_label': task['rule_label'], 'rule_definition': task['rule_definition'], 'difficulty': task['difficulty'], 'target_value': task['target_value'], 'candidates': candidates, 'correct_index': int(task['correct_index']), 'matching_indices': self._matching_indices(task), 'state_trace': self._state_trace(task), 'intermediate_supervision_schema': {'ti2v': ['input_hold', 'complete_outlier_ring_fade', 'final_hold']}, 'render_geometry': _expected_render_geometry(), 'verifier_contract': {'profile': _VERIFIER_PROFILE}}, 'ti2v_prompt': prompts['ti2v_prompt'],  'image': puzzle_path.as_posix(), 'reasoning_image': puzzle_path.as_posix(), 'reasoning_images': [puzzle_path.as_posix()], 'solution_image_path': solution_path.as_posix(), 'solution_video_path': solution_video_path, 'video_fps': self.video_fps if solution_video_path else None, 'video_num_frames': video_num_frames}
        return (sample, puzzle_image, solution_image)

    def create_puzzle(self) -> Dict[str, Any]:
        return self.make_sample()[0]


from data.in_context_learning.core.polish import polished_icl_generator

OddOneOutShapePattern = polished_icl_generator(OddOneOutShapePattern)

__all__ = [
    'OddOneOutShapePattern',
    'pad_image',
    'save_visual_reasoning_video',
    '_has_horizontal_symmetry',
    '_is_convex',
    '_property_value',
    '_validate_task',
    'np',
    'replay_odd_one_out_semantics',
]
