"""Source-derived semantic decoding of a point-target terminal answer image.

A point-target terminal delivery is the *input scene* plus two declared
additions: the construction the task asks for, and the highlight that marks
the chosen candidate.  Reading back only "which marker turned red" therefore
verifies a single token of that state and leaves the entire scene unchecked,
so a terminal image belonging to a different problem is accepted whenever its
highlight happens to land near this problem's answer marker.

This module recovers the rest of the delivered state and compares it against
the scene that is visible in the sample's own input image plus the marker
state implied by the independently replayed answer.  Nothing here reads a
saved ``correct_option``, a reference terminal image, or a solver trace: the
only inputs are the supplied input image, the task-owned candidate geometry,
the renderer's own palette constants, and the caller's independently derived
answer label(s).

Four hard gates, each evaluated per object / per region rather than as one
frame-wide average:

``scene``
    Every load-bearing region *and* every connected member of the input scene
    must survive into the terminal image.  A region budget is still an
    aggregate inside its own region: a thin object spread over several densely
    inked regions is a small share of each of them, so a rim, a ray or a long
    axis can be deleted whole without any region noticing.

``added members``
    Every connected member of the change the delivery makes must be something
    the renderer can draw.  The construction layer is strokes of the declared
    ``line_width`` and discs of the declared ``point_radius``; an opaque patch
    wider than that is not a construction however plausible its colour, and
    when the task declares its stroke grammar the change must also sit on one
    of those strokes, so a second copy of a legal stroke pasted elsewhere is
    not an answer mark.

``markers``
    Every declared candidate marker is checked individually against the two
    states the renderer can put it in.  Exactly the independently derived
    answer label(s) must read as highlighted and every other marker must read
    as untouched.

``static regions`` (whole video)
    Every region of every frame that the delivery's own terminal state leaves
    at the input value must stay at the input value throughout, worst region
    rather than frame mean.  Regions the delivery does change are owned by the
    final-frame gates above, which is what stops a persistent patch from
    defining its own way out of the static mask.

Highlight bounds scale with the source marker's palette separation.  Untouched
markers are read through smoothed color/appearance evidence, so achromatic
source reconstruction noise does not masquerade as an answer commit.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from data.common.verifiers.region_gates import (
    INK_COVER_DELTA,
    MEMBER_MIN_PIXELS,
    REGION_ERASED_FRACTION,
    STATIC_REGION_FLOOR,
    static_region_errors,
)
from data.geometry.core.point_target_base import PointTargetPuzzleGenerator
from data.common.verifiers.keyframes import visual_delta, visual_features


# A pixel counts as ink when it departs from its own image's background by at
# least this much on some channel.  Well below the palette separations the
# family draws with (the faintest family stroke, dark gray 144 on white, is
# 111 apart) and well above the residue that antialiasing leaves behind.
INK_DELTA = 24

# Region size for the per-region scene gate.  512 / 32 = 16x16 regions, small
# enough that a single erased primitive dominates its own region instead of
# being averaged away across the canvas.
SCENE_TILE = 32

# A region takes part in the gate once it carries at least this much input
# ink; below it the denominator is too small for a fraction to mean anything.
SCENE_TILE_MIN_INK = 24

# Share of one region's input ink that may go uncovered.  Measured across the
# family, a legal terminal delivery leaves 0.0 of every region uncovered while
# a terminal image belonging to a different sample of the same task leaves at
# least 0.62 of its worst region uncovered, so the gate sits with margin on
# both sides.  Same constant as the shared region gate, and it doubles as the
# share of one *member* of the input ink that may go uncovered: measured over
# the route, a legal delivery leaves 0.0 of its worst scene member uncovered
# and erasing one whole scene object leaves 0.79 of it.
SCENE_TILE_ERASED_FRACTION = REGION_ERASED_FRACTION

# How much of its own footprint an added member may sink into before it stops
# being something the construction layer can draw, as a multiple of the
# renderer's stroke half-width.  A stroke of the declared ``line_width``
# inscribes ``line_width / 2``; where strokes cross, the inscribed radius grows
# with how many meet and how shallowly, and the densest legal crossing in the
# family -- three concurrent cevians -- measures 2.4 half-widths.  Four clears
# that by 1.7x and still rejects any opaque region more than ``2 * line_width``
# across, which is what an inserted patch is.
ADDED_BLOB_HALF_WIDTHS = 4.0

# Output pixels the delivered ink reaches beyond a stroke's nominal edge, on
# each side.  The family draws on a supersampled canvas and LANCZOS-downscales
# it, which spreads each edge over the output pixel it falls inside and about one
# more, and a pixel at that edge already departs from the background by more than
# :data:`INK_DELTA`.  Same one-pixel spread ``foreign_ink_clearance`` derives.
DELIVERED_EDGE_SPREAD = 1

# Slack when a member's thickness is compared against the thickness the
# *declared* construction has in the same place.  With the reference rasterised
# at the delivered footprint, what is left is the integer-pixel rounding of the
# declared centrelines: ``np.round`` moves a vertex by up to half a pixel per
# axis, so the reference's own bands can sit that much inside the geometry they
# describe.  Added to a locally measured reference, never to the flat bound.
ADDED_DECLARED_THICKNESS_MARGIN = 1.0

# Share of an added member that must lie inside the renderer's declared discs
# before the member is read as a marker commit rather than a construction
# stroke, and is therefore exempt from the stroke-width bound.  A disc of
# ``point_radius`` cannot contain more than 0.62 of a square patch wide enough
# to trip that bound, so the exemption cannot be borrowed by one.
ADDED_DISC_CONTAINED_SHARE = 0.85

# Share of a changed member that must sit within the foreign-ink clearance of a
# declared stroke, or inside a declared disc, before the member counts as that
# construction.  Measured over every task that declares its construction, three
# seeds each, a legal delivery keeps its worst member at 1.00 inside those
# corridors while a duplicated stroke pasted into empty canvas reaches 0.13, so
# the bound sits with margin on both sides.
ADDED_CORRIDOR_SHARE = 0.60

# Output pixels a declared edge can spread over once the supersampled canvas is
# downscaled: the pixel the edge falls inside, and the one next to it.
_DECLARED_EDGE_MARGIN = 2

# A named answer marker must visibly move toward its highlight palette.
# Untouched-source appearance is read separately from this relative bound.
MARKER_HIGHLIGHTED_ALPHA = 0.20

# Absolute floor under the relative marker bound, for the degenerate case
# where a marker sits on ink that the highlight palette barely changes.
# Measured renderer reproduction noise on an untouched marker is 2e-5.
MARKER_ERROR_FLOOR = 0.004

# Model-decoded source edges differ from the input PNG even when the marker
# remains the same.  Read a commit from red chroma rather than achromatic edge
# residuals; retain a local appearance bound to reject covered/erased markers.
MARKER_RED_CHROMA_FLOOR = 0.03
MARKER_BASE_APPEARANCE_FLOOR = 0.065

# How far a marker that claims to be highlighted may sit from the projected
# highlight state, as a share of the base/highlight separation.  Measured over
# the route at three seeds, a legal delivered highlight reaches 0.60 of the
# separation (the supersampled downscale rings across the marker's interior)
# while an opaque patch covering the marker reads 6.5 of it, so the bound at
# their geometric mean clears both by 3.3x.  This is what makes the two-state
# reading two-sided: a marker may be untouched or highlighted, but nothing
# else, and on the video path it needs no answer to say so.
MARKER_HIGHLIGHT_MATCH_ALPHA = 2.0

# How close a marker pixel must sit to one of the renderer's flat marker
# colours before it is used to measure the base/highlight separation.  The two
# anchors are 223 apart per channel and the label glyph is 32 away from the
# outline anchor, so this tolerance keeps antialiased blends and glyph ink out
# of the measurement without discarding the flat interior.
MARKER_ANCHOR_TOLERANCE = 20

_PALETTE_SOURCE = PointTargetPuzzleGenerator


def _background(image: np.ndarray) -> np.ndarray:
    """Modal colour of a decimated view of the frame."""

    sample = image[::4, ::4].reshape(-1, 3)
    packed = (
        sample[:, 0].astype(np.uint32) << 16
        | sample[:, 1].astype(np.uint32) << 8
        | sample[:, 2].astype(np.uint32)
    )
    values, counts = np.unique(packed, return_counts=True)
    dominant = int(values[int(np.argmax(counts))])
    return np.array(
        [(dominant >> 16) & 0xFF, (dominant >> 8) & 0xFF, dominant & 0xFF],
        dtype=np.int16,
    )


def _ink(image: np.ndarray, background: np.ndarray, *, delta: int = INK_DELTA) -> np.ndarray:
    return np.max(np.abs(image.astype(np.int16) - background), axis=2) >= delta


def _tile_counts(mask: np.ndarray, tile: int = SCENE_TILE) -> np.ndarray:
    height, width = mask.shape
    rows, cols = height // tile, width // tile
    if rows == 0 or cols == 0:
        return np.array([[int(mask.sum())]], dtype=np.int64)
    cropped = mask[: rows * tile, : cols * tile]
    return cropped.reshape(rows, tile, cols, tile).sum(axis=(1, 3), dtype=np.int64)


def _members(mask: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Label ``mask``'s connected members and return the retained label ids.

    Components below :data:`MEMBER_MIN_PIXELS` are antialiasing fringe around a
    larger member rather than independent objects; the region gates keep
    covering them.
    """

    if not bool(mask.any()):
        return None, np.empty(0, dtype=np.int64)
    _count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    sizes = np.bincount(labels.ravel())
    keep = np.nonzero(sizes >= MEMBER_MIN_PIXELS)[0]
    keep = keep[keep != 0]
    return labels, keep[np.argsort(-sizes[keep])]


def _member_erased_fractions(
    labels: Optional[np.ndarray], member_ids: np.ndarray, erased: np.ndarray
) -> np.ndarray:
    if labels is None or member_ids.size == 0:
        return np.zeros(0, dtype=np.float64)
    flat = labels.ravel()
    sizes = np.bincount(flat)[member_ids]
    gone = np.bincount(
        flat, weights=erased.ravel().astype(np.float64), minlength=int(labels.max()) + 1
    )[member_ids]
    return gone / np.maximum(sizes, 1)


def _scene_gate(
    candidate: np.ndarray,
    source: np.ndarray,
    *,
    line_width: int,
    exempt: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Require every region *and* every member of the input scene to survive.

    ``exempt`` is a source-derived region the task's own semantics allow the
    delivery to clear -- a declared occluding mask, say.  It is subtracted from
    the scene before either statistic is taken rather than being paid for by a
    looser bound, so nothing outside it is held to a weaker standard
   .
    """

    source_ink = _ink(source, _background(source))
    if exempt is not None:
        source_ink = source_ink & ~exempt
    # The candidate's ink is read with half the contrast.  A hard threshold on
    # both sides is brittle at a saturated background: a uniform brightness
    # shift leaves the background clipped at 255 while it moves the palest
    # strokes towards it, so a stroke drawn at exactly ``INK_DELTA`` drops out
    # of the candidate's mask and reads as erased.
    candidate_ink = _ink(candidate, _background(candidate), delta=INK_COVER_DELTA)
    # One pixel of growth absorbs the subpixel rounding that redrawing the
    # same primitive over a supersampled canvas can produce.  It cannot hide a
    # missing primitive, which is at least ``line_width`` pixels wide.
    covered = cv2.dilate(candidate_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    erased = source_ink & ~covered

    tile_ink = _tile_counts(source_ink)
    tile_erased = _tile_counts(erased)
    load_bearing = tile_ink >= SCENE_TILE_MIN_INK
    # The tolerated loss per region scales with that region's own measured ink
    # content; the floor keeps one stroke width of rounding from ever being
    # decisive in a sparsely inked region.
    budget = np.maximum(
        float(max(1, int(line_width))),
        tile_ink.astype(np.float64) * SCENE_TILE_ERASED_FRACTION,
    )
    violations = load_bearing & (tile_erased.astype(np.float64) > budget)
    fractions = np.where(
        load_bearing, tile_erased / np.maximum(tile_ink, 1), 0.0
    ).astype(np.float64)
    worst = float(fractions.max()) if load_bearing.any() else 0.0

    # A region budget is an aggregate inside its own region: an axis, a rim or
    # a ray spread over a dozen densely inked regions is a small share of each
    # of them and can be deleted whole without any region noticing.  Every
    # connected member of the scene is therefore held to the same share of its
    # *own* pixels.
    labels, member_ids = _members(source_ink)
    member_fractions = _member_erased_fractions(labels, member_ids, erased)
    member_violations = int((member_fractions > SCENE_TILE_ERASED_FRACTION).sum())
    member_worst = float(member_fractions.max()) if member_fractions.size else 0.0

    total_ink = int(source_ink.sum())
    metrics = {
        "terminal_scene_source_ink_pixels": total_ink,
        "terminal_scene_erased_fraction": round(
            float(erased.sum()) / total_ink, 6
        )
        if total_ink
        else 0.0,
        "terminal_scene_region_erased_max": round(worst, 6),
        "terminal_scene_region_erased_budget": round(
            float(SCENE_TILE_ERASED_FRACTION), 6
        ),
        "terminal_scene_region_violations": int(violations.sum()),
        "terminal_scene_load_bearing_regions": int(load_bearing.sum()),
        "terminal_scene_member_count": int(member_ids.size),
        "terminal_scene_member_erased_max": round(member_worst, 6),
        "terminal_scene_member_violations": member_violations,
    }
    passed = bool(load_bearing.any() and not violations.any() and member_violations == 0)
    return metrics, passed


def _disc_mask(shape: Tuple[int, int], discs: Sequence[Tuple[float, float, float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if not discs:
        return mask
    height, width = shape
    for centre_x, centre_y, radius in discs:
        if not all(math.isfinite(value) for value in (centre_x, centre_y, radius)) or radius <= 0:
            continue
        x0 = max(0, int(math.floor(centre_x - radius)))
        x1 = min(width, int(math.ceil(centre_x + radius)) + 1)
        y0 = max(0, int(math.floor(centre_y - radius)))
        y1 = min(height, int(math.ceil(centre_y + radius)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= ((xx - centre_x) ** 2 + (yy - centre_y) ** 2) <= radius * radius
    return mask


def _corridor_mask(
    shape: Tuple[int, int],
    paths: Sequence[np.ndarray],
    clearance: float,
) -> np.ndarray:
    """Pixels within ``clearance`` of any declared stroke's centreline."""

    height, width = shape
    lines = np.zeros(shape, dtype=np.uint8)
    for path in paths:
        points = np.asarray(path, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        pixels = np.round(points).astype(np.int32)
        cv2.polylines(lines, [pixels.reshape(-1, 1, 2)], False, 1, thickness=1)
    if not lines.any():
        return np.zeros(shape, dtype=bool)
    distance = cv2.distanceTransform((1 - lines).astype(np.uint8), cv2.DIST_L2, 5)
    return distance <= float(clearance)


def observed_added_ink(candidate, source):
    """New ink excluding reconstruction of nearby retained source colours."""
    changed = np.max(visual_delta(candidate, source), axis=2) >= INK_DELTA
    source_ink = np.min(source, axis=2) < 200
    source_edge = cv2.dilate(source_ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    candidate_chroma = np.ptp(candidate.astype(np.int16), axis=2)
    retained = source_edge & (candidate_chroma < 30)
    ys, xs = np.nonzero(changed & source_edge & (candidate_chroma >= 30))
    if len(ys):
        colours = candidate[ys, xs].astype(np.int16)
        matched = np.zeros(len(ys), dtype=bool)
        height, width = changed.shape
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                original = source[np.clip(ys+dy, 0, height-1), np.clip(xs+dx, 0, width-1)].astype(np.int16)
                matched |= (original.min(axis=1) < 231) & (np.max(np.abs(colours-original), axis=1) <= INK_DELTA)
        retained[ys[matched], xs[matched]] = True
    return changed & ~retained

def point_target_foreign_ink_measurement(
    candidate_rgb: np.ndarray,
    source_rgb: np.ndarray,
    record: Mapping[str, Any],
    *,
    paths: Sequence[np.ndarray],
    discs: Sequence[Tuple[float, float, float]],
    clearance: float,
    prefix: str = "terminal",
) -> Dict[str, Any]:
    """Measure changed pixels outside one declared construction variant.

    This is deliberately a measurement, not a policy: callers with mutually
    exclusive construction variants can calibrate their own hard bound while
    using source-derived corridors, candidate discs, and declared clearing
    regions with codec-tolerant added-ink observations.  Reporting local connected/tile maxima beside
    the aggregate prevents a long legal stroke from paying for one compact
    foreign branch.
    """

    candidate = np.asarray(candidate_rgb)
    source = np.asarray(source_rgb)
    if candidate.shape != source.shape or candidate.ndim != 3:
        return {
            f"{prefix}_foreign_ink_evaluated": False,
            f"{prefix}_foreign_ink_failure": (
                "candidate and source canvases do not match"
            ),
        }
    # VAE/codec reconstruction changes the antialiasing of retained black
    # source ink. Those changes are not new construction ink. Compare at the
    # observation bandwidth used by the scene gate, and attribute achromatic
    # edge residuals to nearby source ink. New colored strokes and ink on empty
    # canvas still participate in the foreign-construction check.
    changed = observed_added_ink(candidate, source)
    exempt = declared_clearing_region(record, changed.shape)
    if exempt is not None:
        changed &= ~exempt
    allowed = _corridor_mask(changed.shape, paths, clearance)
    allowed |= _disc_mask(changed.shape, discs)
    # A rendered edge spreads one output pixel beyond its attributed stroke.
    # Grant that pixel only next to observed ink already inside the declared
    # region; an empty corridor grants no extra permission to foreign marks.
    raster_edge = cv2.dilate(
        (changed & allowed).astype(np.uint8),
        np.ones((2 * DELIVERED_EDGE_SPREAD + 1,) * 2, np.uint8),
    ) > 0
    foreign = changed & ~allowed & ~raster_edge
    changed_pixels = int(changed.sum())
    foreign_pixels = int(foreign.sum())
    component_count, component_labels = cv2.connectedComponents(
        foreign.astype(np.uint8),
        connectivity=8,
    )
    component_sizes = (
        np.bincount(component_labels.ravel(), minlength=component_count)[1:]
        if component_count > 1
        else np.zeros(0, dtype=np.int64)
    )
    tile_foreign_max = 0
    tile_changed_max = 0
    tile_fraction_max = 0.0
    height, width = changed.shape
    for y0 in range(0, height, SCENE_TILE):
        for x0 in range(0, width, SCENE_TILE):
            foreign_count = int(
                foreign[y0 : y0 + SCENE_TILE, x0 : x0 + SCENE_TILE].sum()
            )
            changed_count = int(
                changed[y0 : y0 + SCENE_TILE, x0 : x0 + SCENE_TILE].sum()
            )
            fraction = (
                float(foreign_count) / changed_count
                if changed_count
                else 0.0
            )
            if foreign_count > tile_foreign_max:
                tile_foreign_max = foreign_count
                tile_changed_max = changed_count
            tile_fraction_max = max(tile_fraction_max, fraction)
    return {
        f"{prefix}_foreign_ink_evaluated": True,
        f"{prefix}_foreign_ink_changed_pixels": changed_pixels,
        f"{prefix}_foreign_ink_pixels": foreign_pixels,
        f"{prefix}_foreign_ink_raster_edge_pixels": int(np.count_nonzero(changed & ~allowed & raster_edge)),
        f"{prefix}_foreign_ink_fraction": round(
            float(foreign_pixels) / changed_pixels
            if changed_pixels
            else 0.0,
            6,
        ),
        f"{prefix}_foreign_ink_component_count": int(component_sizes.size),
        f"{prefix}_foreign_ink_largest_component_pixels": int(
            component_sizes.max()
            if component_sizes.size
            else 0
        ),
        f"{prefix}_foreign_ink_tile_pixels_max": tile_foreign_max,
        f"{prefix}_foreign_ink_tile_changed_pixels_at_max": tile_changed_max,
        f"{prefix}_foreign_ink_tile_fraction_max": round(
            tile_fraction_max,
            6,
        ),
    }


def _declared_thickness(
    shape: Tuple[int, int],
    paths: Sequence[np.ndarray],
    disc_mask: np.ndarray,
    *,
    line_width: float,
    span: int,
) -> np.ndarray:
    """How thick the *declared* construction is at each pixel, in inscribed radius.

    The flat stroke-width bound below asks how deep a member sinks into itself,
    which is the right question for an inserted patch and the wrong one where the
    declared strokes themselves run close together: the gate closes its mask over
    one stroke width first, so two declared strokes a few pixels apart are bridged
    into a single block whose inscribed radius is the gap plus both widths.  That
    thickness belongs to the geometry the task declares, not to the delivery, and
    the renderer knows it exactly -- stroke width and the declared centrelines are
    both source-derived.  Rasterising them through the same discs and the same
    closing therefore measures the thickness a legal delivery is entitled to, per
    pixel, without loosening the bound anywhere else: a lone declared stroke
    inscribes well under the flat bound, so only a place the task's own strokes
    make thick gets more than that bound allows.

    The reference is rasterised at the width the *delivered* ink occupies, which
    is the declared width plus one pixel on each side: the family draws on a
    supersampled canvas and downscales it, spreading each edge over the output
    pixel it falls inside and about one more -- the same one-pixel spread
    ``foreign_ink_clearance`` derives -- and a pixel at that edge already departs
    from the background by more than :data:`INK_DELTA`.  Rasterising the
    reference at the bare declared width instead leaves wider gaps between
    neighbouring bands than the delivery has, so the same closing bridges the
    delivery's bands and not the reference's, and the reference under-reads the
    thickness by several pixels exactly where two strokes nearly meet.
    """

    ink = np.zeros(shape, dtype=np.uint8)
    for path in paths:
        points = np.asarray(path, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        cv2.polylines(
            ink,
            [np.round(points).astype(np.int32).reshape(-1, 1, 2)],
            False,
            1,
            thickness=max(1, int(round(line_width)) + 2 * DELIVERED_EDGE_SPREAD),
        )
    ink |= disc_mask.astype(np.uint8)
    if not ink.any():
        return np.zeros(shape, dtype=np.float32)
    solid = (
        cv2.morphologyEx(
            ink,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span)),
        )
        > 0
    )
    return cv2.distanceTransform((solid & ~disc_mask).astype(np.uint8), cv2.DIST_L2, 5)


def _added_member_gate(
    candidate: np.ndarray,
    source: np.ndarray,
    *,
    line_width: float,
    discs: Sequence[Tuple[float, float, float]],
    paths: Sequence[np.ndarray],
    clearance: float,
    construction_declared: Optional[bool] = None,
    exempt: Optional[np.ndarray] = None,
    prefix: str = "terminal",
) -> Tuple[Dict[str, Any], bool]:
    """Hard-gate every connected member of the change the delivery makes.

    The scene gate above only forbids removals -- the construction and the
    highlight *are* additions -- which leaves a frame free to carry anything
    it likes as long as it does not rub the input out.  Two source-derived
    obligations close that, per member rather than per frame:

    * the construction layer is strokes ``line_width`` wide and discs
      ``point_radius`` across, so a member that sinks deeper into itself than
      :data:`ADDED_BLOB_HALF_WIDTHS` half-widths *and* deeper than the declared
      construction itself reaches in the same place
      (:func:`_declared_thickness`) is not something the renderer can draw,
      whatever colour it is.  The allowance is read per pixel, so a crossing the
      task's own strokes make stays legal while a patch a few pixels away from it
      does not;
    * when the task declares its stroke grammar, the change also has to sit on
      one of those strokes, so a second copy of a legal stroke pasted into
      empty canvas is not a construction either.

    Tasks without a declared grammar keep only the first obligation; a member
    that clears the stroke-width bound there is left accepted rather than
    guessed at.

    The members are taken over everything the frame *changes* rather than over
    the ink it adds.  An opaque patch dropped on a figure is only added ink
    where the figure was blank, so a stroke running through it splits it into
    pieces thin enough to pass for construction; measured against the input it
    stays one solid block whatever it was painted over.  For the same reason
    the mask is closed over one stroke width first: a patch painted in a colour
    the figure underneath already carried leaves the figure's own pixels
    unchanged, cutting the block into slices that are each only as thick as the
    gap between two strokes.  Closing cannot thicken an isolated stroke, so a
    legal construction measures the same either way.
    """

    changed = np.max(visual_delta(candidate, source), axis=2) >= INK_DELTA
    new_ink = observed_added_ink(candidate, source)
    if exempt is not None:
        changed = changed & ~exempt
    span = 2 * int(math.ceil(max(1.0, float(line_width)) / 2.0)) + 1
    solid = (
        cv2.morphologyEx(
            changed.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span)),
        )
        > 0
    )
    labels, member_ids = _members(solid)
    blob_bound = ADDED_BLOB_HALF_WIDTHS * max(1.0, float(line_width)) / 2.0
    disc_mask = _disc_mask(solid.shape, discs)
    corridor = (
        _corridor_mask(solid.shape, paths, clearance)
        if paths
        else np.zeros(solid.shape, dtype=bool)
    )
    declared = bool(paths) if construction_declared is None else bool(construction_declared)
    readings: List[Dict[str, Any]] = []
    failures: List[int] = []
    if labels is not None:
        # The thickness is read outside the declared discs.  The highlighted
        # marker is a filled disc of the declared point_radius -- far thicker
        # than a stroke, and closing fills the label glyph's holes in it -- but
        # the marker gate owns everything inside those discs, so ink there says
        # nothing about whether the rest of the member is drawable.
        distance = cv2.distanceTransform((solid & ~disc_mask).astype(np.uint8), cv2.DIST_L2, 5)
        # What the declared construction is entitled to be, pixel by pixel: the
        # flat bound everywhere, and wherever the task's own strokes are thicker
        # than that, their measured thickness plus the downscale's one pixel.
        allowance = np.maximum(
            blob_bound,
            _declared_thickness(
                solid.shape, paths, disc_mask, line_width=line_width, span=span
            )
            + ADDED_DECLARED_THICKNESS_MARGIN,
        )
        # Declared junctions may shift by two raster pixels after VAE/codec
        # reconstruction. Register their local thickness allowance without
        # raising the flat off-geometry blob bound.
        allowance = cv2.dilate(allowance, np.ones((5, 5), np.uint8))
        for index, identifier in enumerate(member_ids):
            member = labels == identifier
            pixels = int(member.sum())
            inscribed = float(distance[member].max())
            member_allowance = float(allowance[member].max())
            # Pointwise, not per member: a member that happens to touch a legal
            # crossing must not inherit that crossing's allowance everywhere
            # else, which is exactly how a patch would borrow it.
            over_allowance = int(np.count_nonzero(member & (distance > allowance)))
            disc_share = float((member & disc_mask).sum()) / pixels
            # A marker commit and a declared stroke explain the same member,
            # so the two are one mask: a highlight whose ring reaches a little
            # past the disc it is drawn on is still that marker's own change.
            corridor_share = float((member & (corridor | disc_mask)).sum()) / pixels
            on_disc = disc_share >= ADDED_DISC_CONTAINED_SHARE
            drawable = over_allowance == 0 or on_disc
            attributed = (not declared) or on_disc or corridor_share >= ADDED_CORRIDOR_SHARE
            # A closed source-edge residual can join a few isolated codec
            # pixels into a large changed member. Identify actual additions
            # at the same component scale as every other scene member,
            # before attributing the reconstruction to a new stroke.
            _new_labels, new_members = _members(member & new_ink)
            source_reconstruction = new_members.size == 0
            ok = source_reconstruction or (drawable and attributed)
            if not ok:
                failures.append(index)
            if index < 12:
                readings.append(
                    {
                        "pixels": pixels,
                        "inscribed_radius": round(inscribed, 3),
                        "allowed_radius": round(member_allowance, 3),
                        "over_allowance_pixels": over_allowance,
                        "disc_share": round(disc_share, 4),
                        "corridor_share": round(corridor_share, 4),
                        "source_reconstruction_only": source_reconstruction,
                        "pass": bool(ok),
                    }
                )
    metrics = {
        f"{prefix}_added_member_count": int(member_ids.size),
        f"{prefix}_added_member_pixels": int(changed.sum()),
        f"{prefix}_added_member_close_span": int(span),
        f"{prefix}_added_member_blob_bound": round(blob_bound, 3),
        f"{prefix}_added_member_declared_margin": ADDED_DECLARED_THICKNESS_MARGIN,
        f"{prefix}_added_member_corridor_declared": bool(paths),
        f"{prefix}_added_member_readings": readings,
        f"{prefix}_added_member_failures": failures[:12],
    }
    return metrics, not failures


def _highlight_projection(patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project a marker patch onto the two states the renderer can draw.

    Returns ``(support, expected)`` where ``support`` selects the pixels that
    sit on one of the renderer's flat marker colours, and ``expected`` is the
    colour those pixels would carry if this marker were the highlighted one.
    Antialiased blends and the label glyph are deliberately left out: they
    have no closed-form image under the palette substitution, and including
    them would inflate the measured separation instead of describing it.

    ``expected`` is used only as the *scale* of the base/highlight separation,
    never as an exact target: the marker is small enough that the family's
    supersampled downscale rings across its whole interior, so the delivered
    highlight is a bounded fraction of the flat palette step rather than the
    flat palette colour itself.
    """

    pairs = (
        (_PALETTE_SOURCE.CANDIDATE_BASE_FILL, _PALETTE_SOURCE.CANDIDATE_HIGHLIGHT_FILL),
        (
            _PALETTE_SOURCE.CANDIDATE_OUTLINE_COLOR,
            _PALETTE_SOURCE.CANDIDATE_HIGHLIGHT_COLOR,
        ),
    )
    flat = patch.reshape(-1, 3).astype(np.float32)
    support = np.zeros(flat.shape[0], dtype=bool)
    expected = flat.copy()
    for anchor, target in pairs:
        anchor_arr = np.array(anchor, dtype=np.float32)
        near = np.max(np.abs(flat - anchor_arr), axis=1) <= MARKER_ANCHOR_TOLERANCE
        support |= near
        expected[near] = np.array(target, dtype=np.float32)
    return support, expected


def _marker_reconstruction_error(candidate, source, cx, cy, radius):
    """Match retained marker edges within the shared two-pixel raster reach.

    Read both directions so erasure cannot match an unchanged white interior.
    The central label region is checked separately from the outer housing;
    a large unchanged border cannot conceal a missing identifier.
    """
    height, width = source.shape[:2]
    y0, y1 = max(0, int(math.floor(cy-radius))-2), min(height, int(math.ceil(cy+radius))+3)
    x0, x1 = max(0, int(math.floor(cx-radius))-2), min(width, int(math.ceil(cx+radius))+3)
    before, after = source[y0:y1, x0:x1], candidate[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = (xx-cx)**2 + (yy-cy)**2
    neutral = np.ptp(after, axis=2) <= 20.0
    errors = []
    # Housing contours share the scene's two-pixel reconstruction reach.
    # The much smaller identifier glyph retains its one-pixel edge reach.
    for reach, edge in ((radius, 2), (min(5.0, radius/2.0), 1)):
        mask = neutral & (distance <= reach**2)
        window = np.ones((2*edge+1, 2*edge+1), np.uint8)
        for observed, other in ((after, before), (before, after)):
            lower, upper = cv2.erode(other, window), cv2.dilate(other, window)
            error = np.maximum(np.maximum(lower-observed, observed-upper)-4.0, 0.0)
            errors.append(float(error[mask].mean()) / 255.0 if mask.any() else 1.0)
    return max(errors)


def _marker_base_reading(candidate_patch: np.ndarray, source_patch: np.ndarray,
                         *, reconstruction_error: Optional[float] = None) -> Tuple[bool, float]:
    red_chroma = float(np.mean(np.maximum(
        candidate_patch[:, 0] - np.maximum(candidate_patch[:, 1], candidate_patch[:, 2]), 0.0
    ))) / 255.0
    # An incident cyan construction may cross the rim of an untouched marker.
    # Judge its retained neutral scene pixels, while requiring that they still
    # occupy most of the footprint so a colored covering patch cannot abstain.
    neutral = np.max(candidate_patch, axis=1) - np.min(candidate_patch, axis=1) <= 20.0
    appearance = (float(np.mean(np.maximum(
        np.abs(candidate_patch[neutral] - source_patch[neutral]) - 4.0, 0.0
    ))) / 255.0 if bool(neutral.any()) else 1.0)
    if reconstruction_error is not None:
        appearance = min(appearance, reconstruction_error)
    return bool(red_chroma <= MARKER_RED_CHROMA_FLOOR
                and float(neutral.mean()) >= 0.50
                and appearance <= MARKER_BASE_APPEARANCE_FLOOR), red_chroma


def _marker_gate(
    candidate: np.ndarray,
    source: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
    answer_labels: Iterable[str],
    *,
    point_radius: float,
) -> Tuple[Dict[str, Any], bool]:
    """Hard-gate every declared marker against the two states it can be in."""

    height, width = source.shape[:2]
    # Include the outline as well as the fill: a red outline alone is a visible
    # commit.  Chroma distinguishes it from an incident cyan construction.
    radius = max(3.0, float(point_radius) + 1.0)
    expected = {str(label).strip().upper() for label in answer_labels if str(label).strip()}
    readings: List[Dict[str, Any]] = []
    failures: List[str] = []
    source_observed = visual_features(source)
    candidate_observed = visual_features(candidate)

    for entry in candidates:
        if not isinstance(entry, Mapping):
            continue
        label_raw = entry.get("label")
        x_raw, y_raw = entry.get("x"), entry.get("y")
        if label_raw is None or x_raw is None or y_raw is None:
            continue
        label = str(label_raw).strip().upper()
        cx, cy = float(x_raw), float(y_raw)
        y0 = max(0, int(math.floor(cy - radius)))
        y1 = min(height, int(math.ceil(cy + radius)) + 1)
        x0 = max(0, int(math.floor(cx - radius)))
        x1 = min(width, int(math.ceil(cx + radius)) + 1)
        if y0 >= y1 or x0 >= x1:
            failures.append(f"{label}:offcanvas")
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius
        if not disk.any():
            failures.append(f"{label}:empty")
            continue
        source_patch = source_observed[y0:y1, x0:x1][disk]
        candidate_patch = candidate_observed[y0:y1, x0:x1][disk]
        support, highlighted = _highlight_projection(source_patch)
        if not support.any():
            # The marker's flat interior is entirely overdrawn in the input,
            # so the two renderer states are not separable here.  Report it
            # instead of inventing a bound.
            readings.append(
                {
                    "label": label,
                    "expected_highlighted": label in expected,
                    "separation": 0.0,
                    "pass": True,
                    "note": "no flat marker pixels to measure",
                }
            )
            continue
        source_support = source_patch[support]
        # How far highlighting *this* marker would move *these* pixels: the
        # measured separation between the two states the renderer can draw.
        separation = float(np.mean(np.abs(highlighted[support] - source_support))) / 255.0
        observed = float(np.mean(np.abs(candidate_patch[support] - source_support))) / 255.0
        highlight_error = (
            float(np.mean(np.abs(candidate_patch[support] - highlighted[support]))) / 255.0
        )
        highlighted_bound = max(MARKER_ERROR_FLOOR, separation * MARKER_HIGHLIGHTED_ALPHA)
        highlight_match_bound = max(
            MARKER_ERROR_FLOOR, separation * MARKER_HIGHLIGHT_MATCH_ALPHA
        )
        if label in expected:
            # The added red support already establishes which option the
            # commit selects; this gate establishes that the named marker
            # carries a commit at all, and that what it carries is the
            # renderer's highlight state rather than arbitrary ink parked on
            # the marker.
            ok = observed >= highlighted_bound and highlight_error <= highlight_match_bound
            if not ok:
                failures.append(f"{label}:not_highlighted")
        else:
            ok, _red_chroma = _marker_base_reading(
                candidate_patch, source_patch,
                reconstruction_error=_marker_reconstruction_error(candidate_observed, source_observed, cx, cy, radius),
            )
            if not ok:
                failures.append(f"{label}:altered")
        readings.append(
            {
                "label": label,
                "expected_highlighted": label in expected,
                "observed_error": round(observed, 6),
                "highlight_error": round(highlight_error, 6),
                "separation": round(separation, 6),
                "support_pixels": int(support.sum()),
                "unchanged_appearance_bound": MARKER_BASE_APPEARANCE_FLOOR,
                "highlighted_bound": round(highlighted_bound, 6),
                "pass": bool(ok),
            }
        )

    if not readings:
        return {"terminal_marker_readings": [], "terminal_marker_evaluated": False}, True
    observed_labels = {reading["label"] for reading in readings}
    missing = sorted(expected - observed_labels)
    failures.extend(f"{label}:undeclared" for label in missing)
    metrics = {
        "terminal_marker_evaluated": True,
        "terminal_marker_radius": round(radius, 3),
        "terminal_marker_readings": readings,
        "terminal_marker_failures": failures,
    }
    return metrics, not failures


def declared_clearing_region(
    record: Mapping[str, Any], shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """The band the sample's own semantics allow the delivery to clear.

    One point-target task states an occluding mask in its record and asks the
    delivery to reveal what is behind it, so the ink inside that band is not
    scene the terminal image has to preserve.  Reading the band from the
    record keeps the exemption source-derived and bounded to the declared
    rectangle instead of loosening the gate everywhere.
    """

    mask = record.get("mask")
    if not isinstance(mask, Mapping):
        return None
    try:
        left = float(mask["left_x"])
        right = float(mask["right_x"])
        top = float(mask["top_y"])
        bottom = float(mask["bottom_y"])
    except (KeyError, TypeError, ValueError):
        return None
    height, width = shape
    region = np.zeros(shape, dtype=bool)
    # The band is rasterised through the family's supersampled downscale, which
    # spreads each declared edge over the output pixel it falls inside and the
    # one next to it; the declared rectangle is grown by that margin so the
    # exemption covers the band the renderer actually drew.
    margin = _DECLARED_EDGE_MARGIN
    x0 = max(0, int(math.floor(left)) - margin)
    x1 = min(width, int(math.ceil(right)) + 1 + margin)
    y0 = max(0, int(math.floor(top)) - margin)
    y1 = min(height, int(math.ceil(bottom)) + 1 + margin)
    if x0 >= x1 or y0 >= y1:
        return None
    region[y0:y1, x0:x1] = True
    return region


def point_target_delivery_ink_state(
    candidate_rgb: np.ndarray,
    source_rgb: np.ndarray,
    record: Mapping[str, Any],
    *,
    paths: Sequence[np.ndarray] = (),
    discs: Sequence[Tuple[float, float, float]] = (),
    clearance: float = 0.0,
    construction_declared: Optional[bool] = None,
    prefix: str = "terminal",
) -> Tuple[Dict[str, Any], bool]:
    """Run the scene and added-member gates over one delivered frame."""

    raw_line_width = record.get("line_width")
    line_width = (
        float(raw_line_width)
        if isinstance(raw_line_width, (int, float)) and raw_line_width > 0
        else float(_PALETTE_SOURCE.LINE_WIDTH)
    )
    exempt = declared_clearing_region(record, candidate_rgb.shape[:2])
    scene_metrics, scene_pass = _scene_gate(
        candidate_rgb, source_rgb, line_width=int(round(line_width)), exempt=exempt
    )
    added_metrics, added_pass = _added_member_gate(
        candidate_rgb,
        source_rgb,
        line_width=line_width,
        discs=discs,
        paths=paths,
        clearance=clearance,
        construction_declared=construction_declared,
        exempt=exempt,
        prefix=prefix,
    )
    metrics: Dict[str, Any] = {}
    metrics.update(scene_metrics)
    metrics.update(added_metrics)
    metrics["terminal_scene_preservation_pass"] = scene_pass
    metrics[f"{prefix}_added_member_pass"] = added_pass
    return metrics, bool(scene_pass and added_pass)


def verify_point_target_static_regions(
    frames: Sequence[np.ndarray],
    source_rgb: np.ndarray,
) -> Tuple[Dict[str, Any], bool]:
    """Hold every region the delivery's own terminal state never changes.

    The frame-wide statistics this route already carries -- the adjacent-frame
    fraction and the final-hold fraction -- are compensating aggregates: a
    24x24 patch is 0.2% of a 512x512 canvas and sits under both of them.  The
    same measurement restricted to each 32x32 region makes that patch 20-90%
    of the region it lands in.

    The static set is the complement of what the delivery's own last frame
    moves, which is the only description of "untouched" available before the
    construction geometry is known.  It is not a way out for a patch that
    stays to the end: such a patch is ink the final frame added, and the
    added-member gate above owns it.
    """

    if len(frames) < 2:
        return {"video_static_region_evaluated": False}, True
    final = np.asarray(frames[-1])
    changed = np.max(np.abs(final.astype(np.int16) - source_rgb.astype(np.int16)), axis=2) >= INK_COVER_DELTA
    static = ~(cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    if not bool(static.any()):
        return {
            "video_static_region_evaluated": False,
            "video_static_region_note": "the delivery changes every pixel",
        }, True
    worst = 0.0
    worst_frame = 0
    violations = 0
    for index, frame in enumerate(frames):
        error, count = static_region_errors(np.asarray(frame), source_rgb, static)
        violations += int(count > 0)
        if error > worst:
            worst, worst_frame = error, index
    metrics = {
        "video_static_region_evaluated": True,
        "video_static_region_pixels": int(static.sum()),
        "video_static_region_max_error": round(float(worst), 6),
        "video_static_region_bound": round(float(STATIC_REGION_FLOOR), 6),
        "video_static_region_worst_frame": int(worst_frame),
        "video_static_region_violating_frames": int(violations),
    }
    return metrics, violations == 0


def final_marker_state_gate(
    candidate: np.ndarray,
    source: np.ndarray,
    record: Mapping[str, Any],
    *,
    committed_style_owned_elsewhere: bool = False,
    prefix: str = "terminal",
) -> Tuple[Dict[str, Any], bool]:
    """Require every declared marker to read as one of its two renderer states.

    Answer-free: the red decode and the replayed answer own *which* marker is
    committed, but they are blind to what sits on the other markers -- and on
    the answer marker itself a one-sided "moved at least this much" test lets
    any opaque ink through.  Each marker's interior must therefore be close to
    its input state or close to the projected highlight state; a patch is
    neither (measured: 6.5 separations from the highlight where a legal
    commit reads 0.6).

    ``committed_style_owned_elsewhere`` is for the tasks with their own final
    verifiers, whose commit palette is not the shared one: exactly one marker
    may then depart from its base state without matching the shared highlight,
    because that task's own decoder reads the committed marker's content.  Two
    departures are still two commits, which no single-answer delivery can
    justify.
    """

    raw_radius = record.get("point_radius")
    point_radius = (
        float(raw_radius)
        if isinstance(raw_radius, (int, float)) and raw_radius > 0
        else float(_PALETTE_SOURCE.POINT_RADIUS)
    )
    radius = max(3.0, point_radius + 1.0)
    height, width = source.shape[:2]
    readings: List[Dict[str, Any]] = []
    failures: List[str] = []
    source_observed = visual_features(source)
    candidate_observed = visual_features(candidate)
    for entry in record.get("candidates") or []:
        if not isinstance(entry, Mapping):
            continue
        label_raw, x_raw, y_raw = entry.get("label"), entry.get("x"), entry.get("y")
        if label_raw is None or x_raw is None or y_raw is None:
            continue
        label = str(label_raw).strip().upper()
        cx, cy = float(x_raw), float(y_raw)
        y0, y1 = max(0, int(math.floor(cy - radius))), min(height, int(math.ceil(cy + radius)) + 1)
        x0, x1 = max(0, int(math.floor(cx - radius))), min(width, int(math.ceil(cx + radius)) + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius
        if not disk.any():
            continue
        source_patch = source_observed[y0:y1, x0:x1][disk]
        candidate_patch = candidate_observed[y0:y1, x0:x1][disk]
        support, highlighted = _highlight_projection(source_patch)
        if not support.any():
            continue
        separation = float(np.mean(np.abs(highlighted[support] - source_patch[support]))) / 255.0
        observed = float(np.mean(np.abs(candidate_patch[support] - source_patch[support]))) / 255.0
        highlight_error = (
            float(np.mean(np.abs(candidate_patch[support] - highlighted[support]))) / 255.0
        )
        as_base, red_chroma = _marker_base_reading(
            candidate_patch, source_patch,
            reconstruction_error=_marker_reconstruction_error(candidate_observed, source_observed, cx, cy, radius),
        )
        as_highlight = highlight_error <= max(
            MARKER_ERROR_FLOOR, separation * MARKER_HIGHLIGHT_MATCH_ALPHA
        )
        ok = as_base or as_highlight
        readings.append(
            {
                "label": label,
                "observed_error": round(observed, 6),
                "highlight_error": round(highlight_error, 6),
                "separation": round(separation, 6),
                "as_base": bool(as_base),
                "red_chroma": round(red_chroma, 6),
                "cyan_chroma": round(float(np.mean(np.maximum(
                    np.minimum(candidate_patch[:, 1], candidate_patch[:, 2]) - candidate_patch[:, 0], 0.0
                ))) / 255.0, 6),
                "shared_style": bool(ok),
            }
        )
    departures = [reading for reading in readings if not reading["as_base"]]
    if committed_style_owned_elsewhere:
        # One departure is the commit that route's own decoder will read; a
        # second is a commit nothing can justify, and then neither can be
        # trusted to be the real one.
        # A cyan stroke crossing a source marker is owned by the task's
        # construction parser.  Only red chroma supplies an answer commit;
        # arbitrary covering ink still fails the independent style reading.
        departures = [reading for reading in departures
                      if reading["red_chroma"] > MARKER_RED_CHROMA_FLOOR]
        failures = (
            [f"{reading['label']}:invalid_state" for reading in departures]
            if len(departures) > 1
            else []
        )
        failures.extend(f"{reading['label']}:invalid_state" for reading in readings
                        if not reading["as_base"]
                        and reading["red_chroma"] <= MARKER_RED_CHROMA_FLOOR
                        and (not reading["shared_style"] or reading["cyan_chroma"] < 0.01))
    else:
        failures = [
            f"{reading['label']}:invalid_state"
            for reading in readings
            if not reading["shared_style"]
        ]
    metrics = {
        f"{prefix}_marker_state_readings": readings,
        f"{prefix}_marker_state_failures": failures,
        f"{prefix}_marker_state_departures": len(departures),
    }
    return metrics, not failures


__all__ = [
    "declared_clearing_region",
    "final_marker_state_gate",
    "point_target_foreign_ink_measurement",
    "point_target_delivery_ink_state",
    "verify_point_target_static_regions",
]
