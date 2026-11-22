"""Physics-specific readings the shared progress alignment cannot supply.

``data.common.progress_alignment`` owns the family-agnostic part of the
whole-video ordering model: the source progress bank, the bounded-step monotone
Viterbi reading of a decoded sequence over it, and the three gates whose
allowances are measured on the source at the state being judged.  This module
keeps only what is particular to the Physics family - its reference render
layout, and two readings that do not scale with how far the scene moves between
frames.

Everything the shared model reads, it reads out of *how much* the picture
changes, and that is exactly what a scene of small, fast-moving strokes cannot
supply: shrink the per-frame step until adjacent states are alike and a short
rewind starts fitting inside the emission ceiling, enlarge it until they are
unlike and appearance stops carrying order at all.  A rotating rigid body escapes
that trade entirely, because its *direction* is readable without measuring any
magnitude.  Everything static inside a gear's annulus is rotationally symmetric
and everything that carries the phase - the shaded sector, the bold spoke, the
aperiodic tooth dashes - rotates rigidly with the body's own declared angle, so
the mean-subtracted angular profile (:func:`rotation_profiles`) is a purely
rotating signal and the circular cross-correlation of two frames' profiles peaks
at the angle turned between them (:func:`profile_rotations`).  Only the *sign* of
that peak enters the gate, so a reversal is caught however small the step that
renders it.

Most physical processes carry no rotating body, but most of them do carry
something else the appearance distance cannot see: a quantity the rule itself
forbids from going backwards.  A traced path only ever lengthens, a toppling
domino only ever reaches further, a filling chamber's water only ever rises, and
water poured on a stated schedule only ever moves in that pour's declared
direction.  :func:`carrier_readings` reads such a quantity straight out of the
picture as the number of source-declared sample points at which the frame has
left its declared rest colour - the traced ink along a declared polyline, the
water standing in a declared column, the ground a declared domino has covered.
The count is in sample points, a unit the source owns, so like the rotation
reading it does not move when the scene is paced differently; and because a
legal candidate's reading is bounded by what the source's own bank does
(:func:`carrier_advance_bound`, :func:`carrier_sampling_noise`), the thresholds
are read off the source rather than chosen.

The observations expose their source calibration and pixel-resolution floors;
colour occupancy uses the declared palette and an explicit undecided band.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from data.common.progress_alignment import (
    reference_advance as _shared_reference_advance,
    step_budget as _shared_step_budget,
)


# ``PhysicalScenePattern._render_video`` spends a fixed part of every reference
# video not advancing the physical process: five frames holding the untouched
# input, four frames on the terminal commit (the element-local answer reveal,
# ), and five frames holding the terminal state.  A reference video of
# ``N`` frames therefore advances the process by exactly ``1 / (N - 14)`` per
# motion frame in the local-opacity layout and by ``1 / (N - 10)`` in the plain
# layout; the larger of the two is the family's reference advance.
_REFERENCE_NON_ADVANCING_FRAMES = 14


def reference_advance(frame_count: int) -> float:
    """Process advance one reference Physics motion frame carries."""

    return _shared_reference_advance(
        frame_count, non_advancing_frames=_REFERENCE_NON_ADVANCING_FRAMES
    )


def step_budget(frame_count: int, bank_quantum: float) -> float:
    """Largest process advance one candidate Physics frame may legally carry."""

    return _shared_step_budget(reference_advance(frame_count), bank_quantum)


# Angular resolution of the rotation profile, in bins per turn.  The residual of
# the read rotation against the source's declared one is set by the rasterisation
# of the finest rotating stroke - a 3px tooth dash - not by this grid: measured
# over a 320-pose sweep of five gear trains, the worst residual is 0.284 degrees
# at 360 bins and 0.307 degrees at 720, so the coarser grid resolves everything
# the picture actually carries and is half the work.
_ROTATION_PROFILE_BINS = 360

# Radial samples across the annulus.  The band is at most about 30px deep, so 28
# samples read it at roughly one per pixel; 16 samples measure a worst residual of
# 0.306 degrees against 0.284 for 28, so the denser reading is kept.
_ROTATION_PROFILE_RADII = 28

# No reading finer than one bin is evidence that anything turned: the correlation
# is computed on the bin grid and the sub-bin peak interpolation below is an
# estimate of where the true peak lies inside that bin, not a measurement at finer
# resolution.  This is the floor's structural term; :func:`rotation_reading_floor`
# takes the larger of it and the measured one.
_ROTATION_READING_BIN = 360.0 / _ROTATION_PROFILE_BINS


# A declared sample point counts as occupied once the frame differs from the
# carrier's declared rest colour by more than this in any channel.  It is the
# same departure test the Verifier already uses to derive the dynamic region, so
# a pixel that counts as "the process changed something here" for one gate counts
# for the other; every carrier below is drawn in ink that clears it by an order
# of magnitude (the narrowest margin in the family is the domino body's 42 against
# its board, the widest the ball trace's 199 against the arena).
_CARRIER_DEPARTURE = 10

# No reading finer than one declared sample point is evidence that anything
# moved: the count is an integer over the declared points, and a single point
# flipping is the smallest change the reading can express.  This is the floor's
# structural term; the functions below take the larger of it and a measured one.
_CARRIER_QUANTUM = 1.0


def rotation_profiles(
    images: Sequence[np.ndarray],
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    """Mean-subtracted angular brightness profile of one annulus, per image.

    The annulus is sampled on a fixed polar grid, so a rigid rotation of the
    scene is a circular shift of the returned row, and subtracting each row's mean
    removes the rotationally symmetric part - the rim circle, the disk fill, the
    flat background - which would otherwise contribute a constant to every bin and
    pull every correlation towards zero.  Only the sampled points are ever read,
    so a whole-frame intermediate is never materialised.
    """

    if outer_radius <= inner_radius:
        raise ValueError("rotation annulus must have positive depth")
    angles = (np.arange(_ROTATION_PROFILE_BINS) + 0.5) * (
        2.0 * np.pi / _ROTATION_PROFILE_BINS
    )
    radii = np.linspace(float(inner_radius), float(outer_radius), _ROTATION_PROFILE_RADII)
    columns = np.rint(float(center[0]) + np.outer(np.cos(angles), radii)).astype(np.int32)
    rows = np.rint(float(center[1]) + np.outer(np.sin(angles), radii)).astype(np.int32)
    height, width = np.asarray(images[0]).shape[:2]
    np.clip(columns, 0, width - 1, out=columns)
    np.clip(rows, 0, height - 1, out=rows)
    profiles = np.empty((len(images), _ROTATION_PROFILE_BINS), dtype=np.float64)
    for index, image in enumerate(images):
        gathered = np.asarray(image)[rows, columns]
        sampled = gathered.reshape(_ROTATION_PROFILE_BINS, -1).mean(axis=1, dtype=np.float64)
        profiles[index] = sampled - sampled.mean()
    return profiles


def profile_rotations(profiles: np.ndarray) -> np.ndarray:
    """Signed rotation between each consecutive pair of profiles, in degrees.

    The circular cross-correlation of two mean-subtracted profiles peaks at the
    angle the body turned between them; the peak is located to sub-bin precision
    by fitting a parabola through it and its two neighbours, and reported in
    ``(-180, 180]`` so that the sign is the turning direction.
    """

    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    bins = profiles.shape[1]
    spectra = np.fft.rfft(profiles, axis=1)
    correlation = np.fft.irfft(spectra[1:] * np.conjugate(spectra[:-1]), n=bins, axis=1)
    peak = np.argmax(correlation, axis=1)
    pairs = np.arange(correlation.shape[0])
    left = correlation[pairs, (peak - 1) % bins]
    right = correlation[pairs, (peak + 1) % bins]
    denominator = left - 2.0 * correlation[pairs, peak] + right
    # A flat peak - two identical frames, whose correlation is symmetric - has no
    # sub-bin position to fit, and reads as exactly zero rotation.
    offset = np.zeros_like(denominator)
    np.divide(
        0.5 * (left - right),
        denominator,
        out=offset,
        where=np.abs(denominator) > 1e-12,
    )
    degrees = (peak + offset) * (360.0 / bins)
    return (degrees + 180.0) % 360.0 - 180.0


def rotation_reading_floor(
    bank_profiles: np.ndarray,
    declared_degrees: Sequence[float],
) -> float:
    """Largest backwards rotation a *legal* frame pair can still read as.

    ``declared_degrees`` is the rotation the source rule puts between each
    consecutive pair of bank states, so the residual measured here is the
    reader's own error on this scene, at this scale, for this body.  A legal
    candidate pair reads as its true advance plus the difference of the two poses'
    residuals, and the advance may be arbitrarily small - a candidate is allowed
    to dwell - so a legal pair can read backwards by up to twice the worst
    residual.  Below one bin nothing has been read at all, so the floor is the
    larger of the two.
    """

    declared = np.asarray(declared_degrees, dtype=np.float64)
    observed = profile_rotations(bank_profiles)
    if observed.shape[0] != declared.shape[0]:
        raise ValueError("declared rotations must cover every bank transition")
    residual = float(np.max(np.abs(observed - declared))) if observed.size else 0.0
    return max(_ROTATION_READING_BIN, 2.0 * residual)


def backward_rotation_degrees(rotations: Sequence[float], sign: int) -> float:
    """Worst backwards travel of the cumulative phase, in degrees.

    Reading the whole excursion rather than one pair at a time is what makes the
    gate independent of how the reversal is spread: the per-pair readings
    telescope into a cumulative phase whose error stays bounded by the endpoints'
    residuals, so a rewind of any length shows up at its full size while a legal
    sequence stays flat.
    """

    travelled = np.cumsum(np.asarray(rotations, dtype=np.float64) * float(sign))
    travelled = np.concatenate(([0.0], travelled))
    return float(np.max(np.maximum.accumulate(travelled) - travelled))


def carrier_settled_points(
    terminal: np.ndarray,
    samples: Sequence[Sequence[float]],
    rest_rgb: Sequence[int],
    fill_rgb: Sequence[int] | None,
) -> np.ndarray:
    """Declared points whose reading the source's own terminal state settles.

    A solid-body carrier's reading latches (:func:`carrier_readings`), and a latch
    is only sampling-independent for a point the terminal state decides.  Consider
    a point the terminal state shows neither bare nor covered - one straddling the
    body's edge, or one a dark label is drawn over the body at.  The source's
    progress bank ends at the terminal state, so the bank never latches it; but a
    frame sequence that happens to sample the moment the body's edge sits square on
    it does.  Whether that point is counted is then a property of the grid rather
    than of the process, and the bank's own travel stops being a bound on a legal
    candidate's: measured over 99 carriers of 16 domino samples at three seeds, one
    carrier's reading latches one such point and one latches two, the second of
    which is one point past what the one-point floor can absorb.  Dropping them
    makes the two agree exactly: every point any grid can latch is a point the
    terminal state shows covered, and the bank latches every one of those at its
    own last sample.

    This is a characterisation and not a threshold: it needs no level beyond the
    two colours the carrier declares, and points the terminal state does decide -
    covered, or bare because the cascade never reached them - are all kept, so the
    standing-still gate keeps its full evidence.
    """

    if fill_rgb is None:
        return np.ones(len(samples), dtype=bool)
    points = np.asarray(samples, dtype=np.float64)
    height, width = np.asarray(terminal).shape[:2]
    columns = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, width - 1)
    rows = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, height - 1)
    gathered = np.asarray(terminal)[rows, columns].astype(np.int16)
    at_rest = np.max(
        np.abs(gathered - np.asarray(rest_rgb, dtype=np.int16).reshape(1, -1)), axis=1
    ) <= _CARRIER_DEPARTURE
    at_fill = np.max(
        np.abs(gathered - np.asarray(fill_rgb, dtype=np.int16).reshape(1, -1)), axis=1
    ) <= _CARRIER_DEPARTURE
    return at_rest | at_fill


def carrier_readings(
    images: Sequence[np.ndarray],
    samples: Sequence[Sequence[float]],
    rest_rgb: Sequence[int],
    fill_rgb: Sequence[int] | None = None,
    *,
    thin_rgb: Sequence[float] | None = None,
) -> np.ndarray:
    """Reached count of one declared carrier's sample points, per image.

    ``samples`` are points in 512-space that the source declares the process
    advances *through*, ``rest_rgb`` the colour they show while the process has
    not reached them, and ``fill_rgb`` the colour the process leaves where it has
    covered them.  The count is the extent the process has reached: the length of
    trace already drawn, the height of water already standing, the ground a domino
    has already covered.

    Points the scene permanently covers with something else - a label plate over a
    trace, a link dash under a domino - read the same in every frame alike and
    contribute a constant, which no gate below can see.  Only the points that
    change carry the reading.

    A thin stroke declares no solid-body fill because its antialiased centreline
    does not have one exact RGB value. When ``thin_rgb`` supplies the ink colour
    measured from the source terminal, chromatic projection separates that ink
    from a moving ball, a static arrow and neutral codec halos. A one-pixel
    contour neighbourhood and an undecided band retain the reached state until
    the point visibly returns to rest. Neutral-coloured carriers without such a
    chromatic signal retain the direct departure reading.

    A carrier whose ink is a solid body declares the body's own colour, and each
    point is read three ways instead of two: it is at **rest** when it is within
    ``_CARRIER_DEPARTURE`` of the rest colour, **covered** when it is within
    ``_CARRIER_DEPARTURE`` of the fill, and otherwise **undecided**, in which case
    it carries whatever it carried in the previous frame - starting from rest,
    because the source declares nothing reached before the process starts.  No
    second threshold is introduced: it is the same sameness tolerance, applied to
    each of the two colours the carrier declares.

    The undecided band is what a single threshold cannot supply.  These scenes are
    LANCZOS downsampled from a 4x canvas, and that filter's negative lobes lift a
    neighbourhood *above* its own content while a high-contrast edge approaches,
    so a pixel's departure is not monotone in progress anywhere the edge is
    partway across it: a corridor pixel holding only a foreign sliver sits a few
    units above a bare threshold and is pushed back under it for the one frame
    before the body arrives, and a mid-ramp or plateau pixel ripples by up to a
    tenth of the body's contrast. Every such wobble costs total variation two
    sample points, one out and one back. The source-calibrated reading floor
    allows an unresolved excursion; the travel bound counts both of its legs
    (:func:`_progress_carrier_gates`). The three-state reader limits how much
    raster noise reaches that floor.
    Under the three-state rule a ramp goes rest -> undecided -> covered and a
    plateau ripple stays covered, so the wobble never reaches the count, while a
    genuine retreat still does: a body that gives ground back returns its pixels
    to within tolerance of the rest colour.

    Measured on ``domino_reach_cascade`` by reading the source at eight times the
    bank's progress density, over 16 samples at three seeds and 84 carriers: the
    three-state reading's total variation equals its net rise, and the bank agrees
    with the refined grid, for every carrier.  Bare departure does not: it rejects
    2 of 65 reference samples (3.1%) at three seeds, at departures 19 -> 9 (a
    dashed link's tail) and 15 -> 10 (an already-settled body's antialiased
    boundary), and leaves every accepted sample within one sample point of the same
    bound.  A single mid-ramp level does not either - the clean levels are
    fragmented by the ripple, so any one of them is a fit to the sampled
    geometries rather than a property of the scene.

    What the undecided band costs is resolution, and the cost is stated rather than
    hidden: a reading that a one-pixel halo cannot move is also one that a pair of
    frames differing by less than a sample point cannot move.  Scanning every
    distinct adjacent-frame swap of three domino samples, 8 of 90 are now accepted,
    and they are exactly the pairs whose frames differ in 71 to 187 pixels of
    262144, inside the per-wave easing plateau; every swap of frames differing in
    275 or more is still rejected.  Bare departure caught those eight - on the same
    tip pixel crossing its boundary that rejected 3.1% of reference media.  The
    signal and its own false positive were one quantity, so no reading can keep the
    first and drop the second, and the ambiguity is resolved by favoring a false
    reject corrupts the supervision signal, an unobservable reorder does not.
    """

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        raise ValueError("a carrier must declare a non-empty list of sample points")
    height, width = np.asarray(images[0]).shape[:2]
    columns = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, width - 1)
    rows = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, height - 1)
    rest = np.asarray(rest_rgb, dtype=np.int16).reshape(1, -1)
    fill = None if fill_rgb is None else np.asarray(fill_rgb, dtype=np.int16).reshape(1, -1)
    covered = np.zeros(points.shape[0], dtype=bool)
    readings = np.empty(len(images), dtype=np.float64)
    # A coloured thin trail is an ink carrier, not arbitrary departure from
    # white: the moving blue ball, its navy arrow and codec halos also depart
    # from white. Read the source-declared ink's chromatic direction with a
    # one-pixel contour neighbourhood. The undecided band suppresses edge
    # quantisation while a true erased stroke returns to rest and is unlatched.
    chroma = None
    if fill is None and thin_rgb is not None:
        colour = np.asarray(thin_rgb, dtype=np.float64)
        chroma = np.array((colour[0] - colour[1], colour[1] - colour[2]))
        if float(np.dot(chroma, chroma)) <= _CARRIER_DEPARTURE ** 2:
            chroma = None
    for index, image in enumerate(images):
        gathered = np.asarray(image)[rows, columns].astype(np.int16)
        departed = np.max(np.abs(gathered - rest), axis=1) > _CARRIER_DEPARTURE
        if fill is None and chroma is not None:
            pixels = np.asarray(image, dtype=np.float32)
            projection = ((pixels[:, :, 0] - pixels[:, :, 1]) * chroma[0]
                          + (pixels[:, :, 1] - pixels[:, :, 2]) * chroma[1]) / float(np.dot(chroma, chroma))
            projection = cv2.dilate(projection, np.ones((3, 3), np.uint8))
            ink = projection[rows, columns]
            rest_floor = _CARRIER_DEPARTURE / max(abs(chroma).max(), _CARRIER_DEPARTURE)
            covered = np.where(ink >= 0.5, True, np.where(ink <= rest_floor, False, covered))
        elif fill is None:
            covered = departed
        else:
            at_fill = np.max(np.abs(gathered - fill), axis=1) <= _CARRIER_DEPARTURE
            covered = np.where(at_fill, True, np.where(departed, covered, False))
        readings[index] = float(np.count_nonzero(covered))
    return readings


def carrier_backward_travel(readings: Sequence[float], sign: int) -> float:
    """Worst backwards travel of a carrier's reading, in sample points.

    Reading the whole excursion rather than one pair at a time is what makes the
    gate independent of how a rewind is spread across frames: a rewind of any
    length shows up at its full size, while a legal sequence - which may dwell,
    but may not retreat - stays flat.
    """

    travelled = np.asarray(readings, dtype=np.float64) * float(sign)
    if travelled.size == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(travelled) - travelled))


def carrier_total_variation(readings: Sequence[float]) -> float:
    """Total distance a carrier's reading travels, in sample points.

    A carrier that is not monotone over the whole process - a container that
    receives water and later gives it away - still travels exactly as far as its
    schedule says, because it moves only in each step's declared direction.  Total
    variation is the same for any sampling of that path, so a candidate that
    rewinds, repeats or swaps frames travels further than the source ever does
    while a candidate that merely dwells travels exactly as far.
    """

    values = np.asarray(readings, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(np.abs(np.diff(values)).sum())


def carrier_sampling_noise(readings: Sequence[float]) -> float:
    """Total variation this reading manufactures purely by sampling more finely.

    Rasterisation makes an occupancy count jitter by a point or two around the
    extent it stands for, and that jitter inflates total variation in proportion
    to how many samples are taken.  Halving the sampling of the *source's own*
    bank removes the jitter it added without removing any of the travel the
    process actually makes - every monotone stretch in the family spans many bank
    states - so the difference is the reading's own noise at bank density, which
    is the density a candidate video is read at too.
    """

    values = np.asarray(readings, dtype=np.float64)
    if values.size < 3:
        return 0.0
    return max(0.0, carrier_total_variation(values) - carrier_total_variation(values[::2]))


def carrier_advance_bound(
    bank_readings: Sequence[float],
    bank_progress: Sequence[float],
    *,
    budget: float,
) -> float:
    """Largest reading change the source itself makes across one legal advance.

    This is the step budget restated in the carrier's own units, and it is
    measured over a progress *window* rather than divided by a spacing: the bank
    interleaves semantic checkpoints into its uniform grid, so consecutive
    entries can sit arbitrarily close together and a rate computed from one pair
    would diverge.
    """

    readings = np.asarray(bank_readings, dtype=np.float64)
    progress = np.asarray(bank_progress, dtype=np.float64)
    span = max(0.0, float(budget))
    bound = 0.0
    for index in range(readings.shape[0]):
        reach = int(np.searchsorted(progress, progress[index] + span + 1e-12, side="right"))
        if reach > index + 1:
            bound = max(
                bound, float(np.max(np.abs(readings[index + 1 : reach] - readings[index])))
            )
    return bound


def carrier_reading_floor(bank_readings: Sequence[float], sign: int) -> float:
    """Largest backwards travel a *legal* frame pair can still read as.

    The residual measured here is the reader's own error on this scene, for this
    carrier, at bank density: whatever backwards travel the source's own renders
    show is rasterisation, not motion.  A legal candidate pair reads as its true
    advance plus the difference of the two frames' residuals, and the advance may
    be arbitrarily small - a candidate is allowed to dwell - so a legal pair can
    read backwards by up to twice the worst residual.  Below one sample point
    nothing has been read at all, so the floor is the larger of the two.
    """

    residual = carrier_backward_travel(bank_readings, sign) if sign else 0.0
    return max(_CARRIER_QUANTUM, 2.0 * residual)


__all__ = [
    "backward_rotation_degrees",
    "carrier_advance_bound",
    "carrier_backward_travel",
    "carrier_readings",
    "carrier_reading_floor",
    "carrier_settled_points",
    "carrier_sampling_noise",
    "carrier_total_variation",
    "profile_rotations",
    "reference_advance",
    "rotation_profiles",
    "rotation_reading_floor",
    "step_budget",
]
