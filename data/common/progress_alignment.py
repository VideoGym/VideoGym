"""Align complete candidate videos to a source-derived process bank.

A monotone dynamic program compares states and inter-frame motion over the
source's dynamic region. Distinct bank entries allow visually identical states
to recur at different process times. Transition budgets use the source's local
motion rate, reference cadence and bank sampling interval; local evidence gates
reject omitted steps, rollbacks and source-ink mutations. A per-channel tone
offset removes uniform brightness changes before comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


# How many reference advances a candidate frame may carry.  One advance is the
# real motion and the bank quantum below is the reading's quantisation, so 1.0 is
# the exact bound on a legal reading rather than an empirical margin: measured
# across the Physics tasks at ten seeds, the aligned reading of genuine reference
# media never advances by more than two bank quanta per frame, against a budget of
# 2.3 to 2.6 quanta.  The cost of that exactness is that a candidate which paced
# the same process faster than the reference does is rejected; the corruption
# battery shows that any looser value lets a duplicated or rewound segment be
# re-read as legal progress on more than half of a family.
_TIMING_SLACK = 1.0

# Frame-to-frame visible change may not exceed what the source itself produces
# locally across one legal advance (see :func:`local_pair_change_bounds`).  The
# bound is measured on the bank rather than assumed; this slack covers the codec,
# the bank's sampling and the unevenness of the source's own pace.  Measured worst
# ratio of observed to locally allowed pair change is 0.76 over genuine reference
# media at ten seeds and single-frame timing variants, against 1.07 for the
# tightest corruption a battery can construct, so 1.35 sits between them with a
# 1.8x margin below and a 1.4x separation above.
_PAIR_CHANGE_SLACK = 1.35

# Frame-to-frame visible change may not fall *below* this share of what the
# source itself puts between the two states the reading crosses.  The structural
# expectation is 1/2: a legal frame advances the process by one reference pace
# ``p``, the reading rounds both endpoints onto the bank grid ``q``, so the pair
# may be read as spanning up to ``p + q`` of process while its picture moved only
# ``p`` worth, and every route here samples the bank at least as finely as it
# paces (``q <= p``).  The source's rate is not uniform across that span - it
# eases out of every authored hold and into every terminal one - and that, not
# the quantisation, sets the real worst case: measured over 9158 judged
# transitions of genuine reference media across all three routes, two seeds and
# five samples of every task, the smallest legal share is 0.375, on the last
# partial step ``maze_hexagon`` draws before its terminal hold, against 0.526 on
# spatial_3d and 0.584 on Physics.  The tightest route is deliberately not the
# one the defect was found on: a floor derived only on spatial_3d's 31%-dead-time
# axis would have been calibrated against a route whose legal transitions stop
# moving, and applied to eight maze tasks whose banks have no plateau at all.

# The constant is placed by the accept side alone, and deliberately so: the
# corruption it exists to refuse reads exactly 0.000 - a catch-up transition
# advances the reading across states the candidate never drew, so its two frames
# are identical and there is nothing between them to measure - so nothing
# whatever is bought by moving the floor up, while every step up spends margin
# against a legal process that decelerates harder than any measured here.  0.20
# keeps a 1.9x margin under the worst legal reading and refuses every seam
# catch-up in the battery outright.
_PAIR_CHANGE_FLOOR = 0.20

# ...but only where the reading claimed enough to be held to it.  A floor applied
# to every transition is not available: genuine reference media contain
# transitions whose share is exactly 0.000 (``domino_reach_cascade`` crosses two
# bank states 0.0009 apart between two pixel-identical reference frames,
# ``cube_net_fold`` the same at 0.00002), and any positive floor applied
# unconditionally rejects them.  The line is structural rather than fitted to
# those: the reading rounds each endpoint onto the bank grid, so it can claim up
# to one whole grid step of change that is not there, and one grid step is
# ``q/(p+q) <= 1/2`` of what ``local_pair_change_bounds`` measures across one
# advance.  A claim of at most half of that bound is therefore inside what
# rounding alone manufactures and carries no obligation.  Measured, the line
# lands exactly where legal transitions stop needing it - the largest legal claim
# share it skips is 0.4994 to 0.500 on all three routes, so legal claims run
# right up to the line and stop - and the weakest catch-up the battery constructs
# still claims 0.632 (path_trace), 0.649 (spatial_3d) and 0.818 (Physics), a 1.3x
# separation above.  Erring here is graceful in the one direction that matters: a
# claim wrongly skipped is a corruption missed, which is the behaviour this gate
# replaces, whereas a claim wrongly judged rejects legal media.
_FLOOR_CLAIM_SHARE = 0.5

# The fixed per-frame jump ceiling this budget replaces, inherited from the three
# progress-bank Verifiers that shared it.  The derived budget is below it for
# every frame count the media contract permits (0.157 at the shortest video the
# families emit, N=37), and clamping guarantees that shortening a candidate can
# never buy a looser continuity gate than the old model applied.
_INHERITED_JUMP_CEILING = 0.24

# Slack on the paced part of a legal reading step.  This one is derived
# analytically rather than fitted, because the reading is quantised: a legal
# frame advances the process by one reference pace ``p``, the reading rounds
# that to the bank grid ``q``, so past whatever stretch the source renders
# identically (the ``gap`` term of the allowance, which is exact and carries no
# slack - the source really does advance that far showing nothing) a legal step
# carries at most ``p + q``, and the beat between a uniform grid and a uniform
# pace guarantees the bound is reached, not merely approached.  The smallest
# deletion the battery can build that removes real process - two motion frames -
# forces ``3p`` through the same transition, a ratio of ``3p / (p + q)``, at
# least 1.5 at the coarsest bank the routes use (``q = p``).  1.15 leaves the
# exact legal worst case a 15% margin below and that smallest real deletion a
# 30% separation above; a single deleted motion frame (``2p``, ratio at most
# 1.0 for ``q >= p``) stays accepted, which is right - it is one frame of
# timing, below the reading's own quantum.
_STEP_SLACK = 1.15

# How far the *search* may step, in reference advances.  The overstep penalty
# below is lexicographic, so a reading offered a budget it cannot meet will fall
# arbitrarily far behind the picture rather than admit one long step - which
# hides exactly the step the gate above is looking for.  Letting the search reach
# eight advances lets a deleted commit be read as the jump it is; a candidate
# that jumps further than that is clamped here and fails the gate on the clamp,
# which is the same verdict.
_SEARCH_ADVANCES = 8.0

# A move that exceeds the step budget must never be traded against pixel
# agreement: each emission term is at most 1.0, so a penalty of 1e6 per unit
# of overshoot dominates the finite clips in this benchmark and makes the search
# lexicographic - it finds a bounded-step reading whenever one exists, and
# otherwise the reading with the least total overshoot, which is the diagnostic
# we want to report.
_OVERSTEP_PENALTY = 1.0e6

# The pacing term only breaks ties.  Reading a frame one bank state away from
# the state it actually shows costs at least about 0.005 in emission error
# (measured: 0.005 for the smoothest renderer in Physics, up to 0.03 for
# gear_train_direction's rim teeth), whereas departing a full reference advance
# from the source's pace costs 1.0 * advance^2, at most 1.4e-3 for the shortest
# video (N=37, advance 1/27).  The pacing term therefore cannot outvote real
# pixel evidence, but it does decide the reading inside a plateau where the
# emission evidence is exactly tied.
_PACING_WEIGHT = 1.0

# The motion term is real evidence, not a tie-break, so it carries the same
# weight as the appearance term: both are mean absolute disagreements over the
# same dynamic region, in the same units.
_MOTION_WEIGHT = 1.0


# A reading can only order frames if the source's consecutive states are still
# more alike than unrelated states are.  ``gear_train_direction`` breaks that at
# full resolution: its 3px rim strokes move further than their own width between
# frames, so adjacent states already disagree about as much as distant ones and
# the mean absolute distance carries no order.  Measured ratio of adjacent to
# decorrelated source distance is 0.05-0.35 for the other six Physics tasks and
# 0.76 for gear_train_direction, so 0.45 leaves the six comparing at full
# resolution with a 1.3x margin and coarsens only the scene that needs it.
_RESOLVING_RATIO = 0.45

# Gaussian kernel ladder for that coarsening.  Odd sizes only; 0 means full
# resolution.  gear_train_direction resolves at 25, the rest at 0.
_COMPARISON_SCALES = (0, 9, 17, 25, 33)


# The scale probe below only has to rank blur radii, not measure correctness, so
# it reads every third dynamic pixel: the ratios it compares differ by factors of
# two or more, far above the sampling noise of a third of tens of thousands of
# pixels, and the full-resolution work is left to the gates that decide.
_SCALE_PROBE_STRIDE = 3


def reference_advance(frame_count: int, *, non_advancing_frames: int) -> float:
    """Process advance one reference motion frame carries.

    ``non_advancing_frames`` is how many frames of a reference video of
    ``frame_count`` frames do not advance the process - the opening hold on the
    untouched input, any terminal commit rendered in place, and the closing hold
    on the terminal state.  It is a property of the family's own render layout,
    which is why it is supplied rather than assumed: a family that holds longer
    advances faster over the frames that do move, and a budget that ignored that
    would be loose on exactly the families that pace themselves most tightly.
    """

    return 1.0 / float(max(1, int(frame_count) - int(non_advancing_frames)))


def step_budget(reference_pace: float, bank_quantum: float) -> float:
    """Largest process advance one candidate frame may legally carry.

    ``bank_quantum`` is the bank's progress sampling spacing: a reading carries
    up to half a quantum of quantisation error, so two consecutive readings can
    differ by a full quantum even when the underlying advance is smaller.  The
    budget is that quantum plus the reference advance times the timing slack.
    """

    return min(
        _INHERITED_JUMP_CEILING,
        _TIMING_SLACK * max(0.0, float(reference_pace)) + max(0.0, float(bank_quantum)),
    )


def tone_offset(frame: np.ndarray, reference: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Per-channel constant by which ``frame`` is uniformly toned against ``reference``.

    A whole-video verifier must accept style variance that does not
    change semantics, and a uniform tone change is the whole of that freedom: it
    is the only difference an image can carry that is the same at every pixel and
    therefore encodes nothing about *where* anything is.  Estimating it as the
    per-channel median of the difference over ``region`` makes it robust in both
    directions that matter - a locally wrong object moves the median by a fraction
    of its own area and so is not absorbed, while a genuinely uniform shift is
    removed exactly.  ``region`` is normally the static complement, where every
    legal frame equals the input and any difference is therefore either tone or a
    mutation; the caller falls back to the whole frame when the process leaves no
    static pixels.

    Saturated pixels cannot express the shift - white stays white under +2 - so
    they are excluded from the estimate per channel; and a region whose every
    pixel is saturated (a pure-white background around a busy board) says
    nothing at all, so the estimate then widens to the whole frame's
    unsaturated pixels, where unchanged structure still dominates any moving
    minority and the median still reads the tone.
    """

    frame = np.asarray(frame)
    difference = frame.astype(np.float32) - np.asarray(reference, dtype=np.float32)
    flat_samples = difference.reshape(-1, difference.shape[-1])
    flat_values = frame.reshape(-1, frame.shape[-1])
    if region is None:
        samples, values = flat_samples, flat_values
    else:
        samples, values = difference[region], frame[region]
    if samples.size == 0:
        samples, values = flat_samples, flat_values
    offsets = np.zeros(samples.shape[-1], dtype=np.float32)
    for channel in range(samples.shape[-1]):
        expressive = (values[:, channel] > 0) & (values[:, channel] < 255)
        if np.any(expressive):
            column = samples[expressive, channel]
        else:
            widened = (flat_values[:, channel] > 0) & (flat_values[:, channel] < 255)
            column = flat_samples[widened, channel] if np.any(widened) else samples[:, channel]
        offsets[channel] = float(np.median(column))
    return offsets


def detoned(frames: Sequence[np.ndarray], reference: np.ndarray, region: np.ndarray) -> list[np.ndarray]:
    """Return ``frames`` with each one's uniform tone offset removed.

    The offset is subtracted only from pixels that are unsaturated in the
    channel concerned: a saturated pixel never received the shift - white stays
    white under +2 - so subtracting there would manufacture exactly the error
    the operation exists to remove.  The result is float32 and deliberately
    unclipped: clipping would reintroduce part of the offset, and a candidate
    shifted far enough to clip mid-tones has destroyed its own contrast and
    fails the very same measurements on its structure.
    """

    out: list[np.ndarray] = []
    for frame in frames:
        frame = np.asarray(frame)
        offset = tone_offset(frame, reference, region)
        expressive = (frame > 0) & (frame < 255)
        out.append(np.where(expressive, frame.astype(np.float32) - offset, frame.astype(np.float32)))
    return out


def comparison_scale(
    bank: Sequence[np.ndarray],
    bank_progress: Sequence[float],
    dynamic: np.ndarray,
    *,
    span: float,
) -> tuple[int, float]:
    """Smallest blur at which the source's own states are still time-ordered.

    Returns ``(kernel, ratio)``: the Gaussian kernel size to compare at, and the
    ratio of adjacent-state to decorrelated-state disagreement achieved there.
    ``span`` is the widest advance a legal reading may cross, so states further
    apart than that count as unrelated.
    """

    progress = np.asarray(bank_progress, dtype=np.float64)
    anchors = np.unique(np.linspace(0, len(bank) - 1, 12).astype(int))
    rows, columns = np.nonzero(dynamic)
    rows = rows[::_SCALE_PROBE_STRIDE]
    columns = columns[::_SCALE_PROBE_STRIDE]
    ratio = float("inf")
    kernel = _COMPARISON_SCALES[-1]
    for candidate in _COMPARISON_SCALES:
        states = np.stack([
            (
                cv2.GaussianBlur(np.asarray(state), (candidate, candidate), 0)
                if candidate
                else np.asarray(state)
            )[rows, columns].astype(np.int16)
            for state in bank
        ])
        adjacent = float(np.abs(states[:-1] - states[1:]).mean()) / 255.0
        distant: list[float] = []
        for anchor in anchors:
            far = np.abs(progress - progress[anchor]) > span
            if not np.any(far):
                continue
            distant.extend(
                (np.abs(states[far] - states[anchor]).mean(axis=(1, 2)) / 255.0).tolist()
            )
        decorrelated = float(np.median(distant)) if distant else 0.0
        ratio = adjacent / decorrelated if decorrelated > 0.0 else float("inf")
        kernel = candidate
        if ratio <= _RESOLVING_RATIO:
            break
    return kernel, ratio


def at_scale(images: Sequence[np.ndarray], kernel: int) -> list[np.ndarray]:
    """Return ``images`` at the comparison scale ``kernel`` (0 = unchanged)."""

    if kernel <= 0:
        return list(images)
    return [cv2.GaussianBlur(np.asarray(image), (kernel, kernel), 0) for image in images]


def dynamic_probe(images: Sequence[np.ndarray], dynamic: np.ndarray) -> np.ndarray:
    """Stack the dynamic-region pixels of ``images`` once, for reuse by the pass.

    Every measurement in this module scores the same region of the same images, so
    extracting it once and passing the stack around keeps the decoder from paying
    for the copy four times over.  Integer images are kept at 16 bits, which holds
    every difference the comparisons take exactly and halves the stack; images
    already detoned into floats keep their fractional offset.
    """

    stack = [np.asarray(image) for image in images]
    dtype = np.float32 if any(np.issubdtype(image.dtype, np.floating) for image in stack) else np.int16
    return np.stack([image[dynamic].astype(dtype) for image in stack])


@dataclass(frozen=True)
class ProgressAlignment:
    """One monotone bounded-step reading of a decoded frame sequence."""

    state_indices: tuple[int, ...]
    progress: tuple[float, ...]
    emission_errors: tuple[float, ...]
    steps: tuple[float, ...]
    step_budget: float
    max_step: float
    overstep_frames: int

    @property
    def feasible(self) -> bool:
        """True when every frame-to-frame advance fits inside the budget."""

        return self.overstep_frames == 0


def emission_costs(frames: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Mean absolute disagreement of every frame against every bank state.

    Both arguments are :func:`dynamic_probe` stacks.  Only the source-derived
    dynamic region is scored: the complement is where no bank state differs from
    the input at all, and the Verifier checks it separately as static
    preservation, so scoring it here would only dilute the signal with a constant.
    """

    costs = np.empty((frames.shape[0], bank.shape[0]), dtype=np.float64)
    for index in range(frames.shape[0]):
        costs[index] = np.abs(bank - frames[index]).mean(axis=(1, 2)) / 255.0
    return costs


def reference_pair_changes(
    bank: np.ndarray,
    bank_progress: Sequence[float],
    *,
    budget: float,
    bank_quantum: float,
) -> np.ndarray:
    """Source disagreement between every pair of states one advance apart.

    ``bank`` is a :func:`dynamic_probe` stack.  Entry ``(a, b)`` is how much the
    process visibly changes going from state ``a`` to state ``b``, filled only for
    the pairs a legal reading may cross; outside that window the entry stays zero
    because the step budget already forbids the transition outright.
    """

    progress = np.asarray(bank_progress, dtype=np.float64)
    span = float(budget) + max(0.0, float(bank_quantum))
    count = bank.shape[0]
    changes = np.zeros((count, count), dtype=np.float64)
    for index in range(count):
        reach = int(np.searchsorted(progress, progress[index] + span + 1e-12, side="right"))
        if reach > index + 1:
            changes[index, index + 1 : reach] = (
                np.abs(bank[index + 1 : reach] - bank[index]).mean(axis=(1, 2)) / 255.0
            )
    return changes


def local_pair_change_bounds(
    reference_changes: np.ndarray,
    bank_progress: Sequence[float],
) -> np.ndarray:
    """Visible change the source produces near each state across one advance.

    Entry ``a`` is the largest disagreement between state ``a`` and any state one
    advance away from it, over the window :func:`reference_pair_changes` already
    filled, so both share the same widening by one bank quantum: the bank samples
    progress discretely, and a window aligned to bank samples can be up to a
    quantum narrower than the advance it stands for.  The window looks forward,
    and reaches backward only by however much of it runs off the end of the
    process - otherwise the last states before the terminal would be allowed just
    the sliver of change that remains ahead of them, and a perfectly paced final
    frame would read as outrunning the process.

    This is the whole-video counterpart to the step budget, and it closes the
    case the alignment alone cannot see: a rewind, a duplicated segment or a
    splice whose frames all still resemble *some* legal state, but whose
    frame-to-frame jump is larger than the process ever makes *there*.  Bounding
    locally rather than globally matters wherever the source's own pace is
    uneven - a cascade that eases into each wave, a fill that accelerates - since
    a global maximum would license a rewind inside every slow phase.
    """

    progress = np.asarray(bank_progress, dtype=np.float64)
    count = reference_changes.shape[0]
    forward = reference_changes.max(axis=1)
    # The filled window's width, recovered from the matrix itself so the two
    # functions cannot drift apart.
    reach = np.array(
        [
            int(np.flatnonzero(reference_changes[index] > 0.0).max())
            if np.any(reference_changes[index] > 0.0)
            else index
            for index in range(count)
        ]
    )
    span = float(np.max(progress[reach] - progress)) if count else 0.0
    bounds = forward.astype(np.float64).copy()
    for index in range(count):
        deficit = span - (progress[reach[index]] - progress[index])
        if deficit <= 1e-12:
            continue
        start = int(np.searchsorted(progress, progress[index] - deficit - 1e-12, side="left"))
        if start < index:
            bounds[index] = max(
                bounds[index], float(reference_changes[start:index, index].max())
            )
    # A state with no window at all in either direction - only possible on a bank
    # of one or two entries - inherits its neighbour rather than zero.
    for index in range(count - 2, -1, -1):
        if bounds[index] <= 0.0:
            bounds[index] = bounds[index + 1]
    for index in range(1, count):
        if bounds[index] <= 0.0:
            bounds[index] = bounds[index - 1]
    return bounds


def neighbour_widened(bounds: np.ndarray) -> np.ndarray:
    """Widen each state's allowance to its two bank neighbours.

    The bank samples progress on a uniform grid; a renderer that advances its
    scene in visually discrete steps - a fold that snaps a face to the next
    crease, a graph that commits an edge - does not.  A state whose own forward
    window happens to fall between two such steps then measures almost no change,
    while the state one sample away measures a whole one, so the same transition
    is judged by wildly different allowances depending only on where the grid's
    phase landed.  That is a positional artefact of the sampling, not a property
    of the process, and it is exactly the shape of defect this module exists to
    remove.  Widening by one state on each side makes the allowance the fastest
    the source moves anywhere the reading could plausibly sit, which is
    phase-independent and still local.

    Renderers whose motion is continuous do not need this - Physics measures the
    same allowance without it - so the widening is applied by the caller rather
    than folded into :func:`local_pair_change_bounds`.
    """

    values = np.asarray(bounds, dtype=np.float64)
    if values.size < 2:
        return values.copy()
    widened = values.copy()
    widened[:-1] = np.maximum(widened[:-1], values[1:])
    widened[1:] = np.maximum(widened[1:], values[:-1])
    return widened


def observed_pair_changes(frames: np.ndarray) -> list[float]:
    """Visible change between each consecutive pair of decoded frames.

    ``frames`` is a :func:`dynamic_probe` stack.
    """

    return [
        float(np.abs(frames[index] - frames[index + 1]).mean()) / 255.0
        for index in range(frames.shape[0] - 1)
    ]


def crossed_pair_changes(
    bank: np.ndarray,
    state_indices: Sequence[int],
) -> list[float]:
    """Source disagreement between exactly the two states each transition crosses.

    ``bank`` is a :func:`dynamic_probe` stack.  This is deliberately *not*
    :func:`local_pair_change_bounds`: that one answers "how fast may the picture
    move near here", a question about a neighbourhood, and it is measured over a
    window on the collapsed bank, which is why it can under-read a source that
    turns several things on inside one advance.  This one answers "how far apart
    are these two pictures", which has an exact answer and needs no window, no
    neighbourhood and no new render - both states are already rendered, and
    collapsing pixel-identical entries never changed what either one looks like.
    """

    stack = np.asarray(bank)
    indices = np.asarray(state_indices, dtype=np.intp)
    if indices.size < 2:
        return []
    # One gathered difference rather than a Python loop of pairwise casts: the
    # stack is already the narrowest dtype that holds every difference exactly,
    # so subtracting in place costs a fraction of what promoting each pair does.
    differences = np.abs(stack[indices[:-1]] - stack[indices[1:]])
    return (differences.mean(axis=tuple(range(1, differences.ndim))) / 255.0).tolist()


def pace_violations(
    pair_changes: Sequence[float],
    state_indices: Sequence[int],
    bounds: np.ndarray,
    crossed_changes: Sequence[float],
    *,
    slack: float = _PAIR_CHANGE_SLACK,
    floor: float = _PAIR_CHANGE_FLOOR,
    claim_share: float = _FLOOR_CLAIM_SHARE,
) -> tuple[int, float, float, int, float]:
    """Count transitions whose visible change leaves the local source band.

    Returns ``(violations, worst_ratio, worst_observed, floor_violations,
    worst_floor_ratio)``.

    **Above.**  The allowance for one transition is the largest local bound over
    the states the reading crosses, so a transition that straddles the onset of a
    fast phase is judged by that phase, never by the slow state it started from.

    **Below.**  The reading is an assertion that these two frames are those two
    bank states, and the floor asks the picture to show it: the frame-to-frame
    change may not fall below ``floor`` times the source's own disagreement
    between exactly the two states crossed.  ``crossed_changes`` is required
    rather than optional so that no caller can apply half the band by omission.

    The two halves read different quantities on purpose, and the asymmetry is the
    point rather than an inconsistency.  The ceiling asks whether the picture
    moved faster than the process can move *around here*, so it must be widened
    to the neighbourhood - a bound read off one grid-aligned pair would swing
    with the sampling phase and judge the same transition differently depending
    on where the grid landed.  The floor asks whether the picture moved as far as
    the two asserted pictures are apart, which is a fact about those two states
    and nothing else; widening it to the neighbourhood would charge a transition
    for change that happens somewhere it was never read as being, and the
    reference's own ease-out - a real, legal deceleration into every authored
    hold - is exactly where that mistake lands.  Measured: the slowest legal
    transition on each route reads 0.216 (path_trace), 0.300 (spatial_3d) and
    0.330 (Physics) of the widened envelope, all of which a floor read off that
    envelope would reject, against 0.375, 0.526 and 0.584 of the states those
    same transitions actually cross.

    **What the floor may not judge.**  A floor applied to every transition is not
    available, and the reason is measured rather than theoretical: genuine
    reference media contain transitions whose ratio is exactly 0.000.
    ``domino_reach_cascade`` renders two consecutive bank states 0.0009 apart
    while two genuine reference frames are pixel-identical, and ``cube_net_fold``
    does the same at 0.00002.  Nothing is wrong with either - the reading crossed
    two states that are the same picture to within a rounding of the bank grid,
    so it claimed nothing the picture could have shown - but any positive floor
    applied unconditionally rejects both.  ``claim_share`` is therefore the share
    of one advance's worth of local source change a transition must claim before
    the floor will judge it at all; see :data:`_FLOOR_CLAIM_SHARE`.

    ``worst_floor_ratio`` is the smallest ratio of observed to crossed-state
    change over the transitions actually judged, and reports 1.0 when none were,
    so a smaller number always means a larger shortfall.
    """

    violations = 0
    worst_ratio = 0.0
    worst_observed = 0.0
    floor_violations = 0
    floor_ratios: list[float] = []
    for index, observed in enumerate(pair_changes):
        low, high = sorted((int(state_indices[index]), int(state_indices[index + 1])))
        local = float(bounds[low : high + 1].max())
        allowance = local * float(slack)
        ratio = float(observed) / allowance if allowance > 0.0 else float("inf")
        if ratio > 1.0:
            violations += 1
        if ratio > worst_ratio:
            worst_ratio, worst_observed = ratio, float(observed)
        crossed = float(crossed_changes[index])
        # The reading stayed on one picture, or crossed two that are barely
        # different pictures: either way it claimed too little to be held to it,
        # and a candidate is in any case allowed to dwell.
        if crossed <= 0.0 or crossed < float(claim_share) * local:
            continue
        floor_ratio = float(observed) / crossed
        floor_ratios.append(floor_ratio)
        if floor_ratio < float(floor):
            floor_violations += 1
    return (
        violations,
        worst_ratio,
        worst_observed,
        floor_violations,
        min(floor_ratios) if floor_ratios else 1.0,
    )


def align_progress(
    emission: np.ndarray,
    bank_progress: Sequence[float],
    *,
    budget: float,
    pace: float,
    pair_changes: Sequence[float] | None = None,
    reference_changes: np.ndarray | None = None,
    pacing_weight: float = _PACING_WEIGHT,
    motion_weight: float = _MOTION_WEIGHT,
) -> ProgressAlignment:
    """Best monotone bounded-step reading of ``emission`` over the bank.

    ``emission`` is the ``(frames, states)`` cost matrix from
    :func:`emission_costs`, ``bank_progress`` its ascending progress axis, and
    ``pace`` the advance the pacing term is centred on - the reference video's
    own per-frame advance from :func:`reference_advance`.

    ``pair_changes`` and ``reference_changes`` add the motion term: how much the
    candidate visibly changes between two frames, against how much the source
    changes between the two states they are read as.  Appearance alone cannot
    place a frame inside a stretch of progress over which nothing changes, so
    without this term a reading is free to sit anywhere in such a plateau; with
    it, the observed change pins the reading to the part of the plateau where the
    process is actually moving that fast.
    """

    emission = np.asarray(emission, dtype=np.float64)
    if emission.ndim != 2 or emission.size == 0:
        raise ValueError("emission costs must be a non-empty (frames, states) matrix")
    progress = np.asarray(bank_progress, dtype=np.float64)
    if progress.shape[0] != emission.shape[1]:
        raise ValueError("emission costs and bank progress disagree on the state count")
    if np.any(np.diff(progress) < -1e-12):
        raise ValueError("bank progress must be ascending")

    advance = progress[None, :] - progress[:, None]
    transition = np.where(
        advance < -1e-12,
        np.inf,
        pacing_weight * np.square(np.maximum(advance, 0.0) - float(pace))
        + _OVERSTEP_PENALTY * np.maximum(0.0, advance - float(budget)),
    )

    frame_count, state_count = emission.shape
    if (pair_changes is None) != (reference_changes is None):
        raise ValueError("the motion term needs both observed and reference pair changes")
    observed = None
    if pair_changes is not None:
        observed = np.asarray(pair_changes, dtype=np.float64)
        if observed.shape[0] != frame_count - 1:
            raise ValueError("observed pair changes must cover every frame transition")
        reference_changes = np.asarray(reference_changes, dtype=np.float64)
        if reference_changes.shape != (state_count, state_count):
            raise ValueError("reference pair changes must be a (states, states) matrix")

    predecessor = np.zeros((frame_count, state_count), dtype=np.int32)
    predecessor[0] = -1
    cost = emission[0].copy()
    columns = np.arange(state_count)
    for index in range(1, frame_count):
        total = cost[:, None] + transition
        if observed is not None:
            total = total + motion_weight * np.abs(reference_changes - observed[index - 1])
        best = np.argmin(total, axis=0)
        predecessor[index] = best
        cost = total[best, columns] + emission[index]

    path = np.empty(frame_count, dtype=np.int64)
    path[-1] = int(np.argmin(cost))
    for index in range(frame_count - 1, 0, -1):
        path[index - 1] = predecessor[index, path[index]]

    read = progress[path]
    steps = np.diff(read) if frame_count > 1 else np.zeros(0)
    return ProgressAlignment(
        state_indices=tuple(int(value) for value in path),
        progress=tuple(float(value) for value in read),
        emission_errors=tuple(float(emission[n, path[n]]) for n in range(frame_count)),
        steps=tuple(float(value) for value in steps),
        step_budget=float(budget),
        max_step=float(steps.max()) if steps.size else 0.0,
        overstep_frames=int(np.count_nonzero(steps > float(budget) + 1e-9)),
    )


@dataclass(frozen=True)
class OrderingReading:
    """The whole-video ordering verdict for one decoded sequence."""

    state_indices: tuple[int, ...]
    progress: tuple[float, ...]
    max_step: float
    step_ratio: float
    reference_pace: float
    bank_quantum: float
    distinct_states: int
    comparison_kernel: int
    step_break_count: int
    pace_break_count: int
    worst_pace_ratio: float
    worst_pair_change: float
    pace_floor_break_count: int
    worst_pace_floor_ratio: float

    @property
    def ordered(self) -> bool:
        """True when every frame's advance and every frame's visible change sit
        inside the source's own local band."""

        return (
            self.step_break_count == 0
            and self.pace_break_count == 0
            and self.pace_floor_break_count == 0
        )


def read_process_order(
    frames: Sequence[np.ndarray],
    bank: Sequence[np.ndarray],
    bank_progress: Sequence[float],
    dynamic: np.ndarray,
    *,
    non_advancing_frames: int,
    bank_quantum: float,
    preserve_adjacent_transitions: bool = False,
) -> OrderingReading:
    """Read a decoded sequence as a bounded-step monotone walk over the bank.

    ``non_advancing_frames`` is the family's render layout (see
    :func:`reference_advance`); ``bank_quantum`` is the bank's progress spacing.
    ``preserve_adjacent_transitions`` is for a bank sampled once per real source
    render tick: after exact-state collapse, it retains the first visible
    transition following an occluded plateau as a one-tick local pace bound.

    Two gates decide, and both are local to where the corruption sits:

    **The reading's own advance.**  A legal frame moves the process by one
    reference advance, and the reading quantises that to the bank grid, so the
    largest legal reading step is ``pace + quantum`` - reached, not approached,
    since a uniform grid and a uniform pace beat against each other.  A deletion
    of ``k`` frames asks for ``k + 1`` advances at one place; a reversal asks the
    reading to clear the window it could not enter.  Neither fits.  The search
    itself runs on a deliberately wide budget: with an overwhelming overstep
    penalty the reading would rather fall behind the picture than admit a jump,
    which hides the very thing being measured, so the jump is *permitted* during
    the search and *judged* afterwards.

    **The pace of the picture**, bounded on both sides.  Between two frames the
    video may not change by more than the source itself changes across one
    advance near there, which catches a splice whose frames all still match legal
    states in order; and it may not change by less than the source puts between
    the two states the reading claims those frames are, which catches the
    complementary case the ceiling is blind to.  A deletion at the edge of a hold
    does not present its missing process as one long jump: the reading absorbs it
    as two shorter transitions, each inside the step budget, one of which
    advances several bank samples while the picture stands perfectly still.  A
    ceiling has nothing to say about a candidate that moved too little.

    ``ordered`` is their conjunction.  All of them scale with the source's own
    local rate of change, so the same corruption is judged the same way in a fast
    phase and in a slow one, and a permutation or deletion inside an exact
    plateau - where the source changes nothing, the reading therefore crosses no
    distinct states and the spliced video is frame-identical - passes all of
    them, which is the correct verdict.

    Candidate frames are compared after their uniform tone offset is removed
    (:func:`tone_offset`), so a semantics-preserving global brightness change
    reads exactly as the original does.
    """

    pace = reference_advance(len(frames), non_advancing_frames=non_advancing_frames)
    quantum = max(0.0, float(bank_quantum))
    kernel, _ = comparison_scale(bank, bank_progress, dynamic, span=pace + 2.0 * quantum)
    scaled_bank = at_scale(bank, kernel)
    # Bank entries that render identically carry no order between them, and a
    # reading that lands on either end of such a run is equally right.  Left in,
    # they make the reading's own step ambiguous by the width of the run, which is
    # indistinguishable from a deleted commit.  Collapsing each run to its first
    # entry removes the ambiguity exactly - equality is exact, not thresholded -
    # and the run's width is then handed to the gate below as the distance to the
    # next picture the source actually draws.
    probe_all = dynamic_probe(scaled_bank, dynamic)
    keep = [0]
    for index in range(1, probe_all.shape[0]):
        if not np.array_equal(probe_all[index], probe_all[keep[-1]]):
            keep.append(index)
    probe_bank = probe_all[keep]
    progress = np.asarray([float(bank_progress[index]) for index in keep], dtype=np.float64)
    static = ~np.asarray(dynamic)
    toned = detoned(frames, np.asarray(bank[0]), static if static.any() else None)
    probe_frames = dynamic_probe(at_scale(toned, kernel), dynamic)
    pair = observed_pair_changes(probe_frames)
    source_changes = reference_pair_changes(
        probe_bank, progress, budget=pace, bank_quantum=quantum
    )
    # A caller whose bank has one state per real source-renderer tick can lose a
    # tick transition when the exact-state collapse above removes several
    # pixel-identical frames: the next distinct picture may then sit more than
    # ``pace + quantum`` away even though the renderer exposes it in one frame.
    # Such callers can preserve that exact adjacent source transition in the
    # local pace model.  The progress step remains unchanged and is still judged
    # independently, so this does not turn a multi-frame deletion into one free
    # step.
    if preserve_adjacent_transitions and len(progress) > 1:
        adjacent = crossed_pair_changes(
            probe_bank,
            tuple(range(len(progress))),
        )
        for index, change in enumerate(adjacent):
            source_changes[index, index + 1] = max(
                float(source_changes[index, index + 1]),
                float(change),
            )
    alignment = align_progress(
        emission_costs(probe_frames, probe_bank),
        progress,
        budget=_SEARCH_ADVANCES * pace + quantum,
        pace=pace,
        pair_changes=pair,
        reference_changes=source_changes,
    )
    # A legal frame advances the process by one reference pace, and the reading
    # rounds that onto the bank, so from state ``b`` it may reach at most one
    # pace past the next picture the source draws: the distance to that picture
    # (``gap``) is exact and carries no slack - across a stretch the source
    # renders identically it really does advance that far while showing nothing,
    # and there is nothing to verify inside it - while the paced part carries the
    # quantisation slack.  Keeping the two apart is what closes the seam case: a
    # deletion at the edge of a hold removes motion frames but starts its jump
    # inside the plateau, and an allowance that multiplied the whole sum by the
    # slack would hand the deletion ``0.15 * gap`` of unearned advance.
    gaps = np.diff(progress, append=progress[-1])
    step_breaks = 0
    worst_step_ratio = 0.0
    for index, step in enumerate(alignment.steps):
        allowance = min(
            _INHERITED_JUMP_CEILING,
            float(gaps[alignment.state_indices[index]])
            + (pace + quantum) * _STEP_SLACK,
        )
        ratio = float(step) / allowance if allowance > 0.0 else float("inf")
        if ratio > 1.0:
            step_breaks += 1
        worst_step_ratio = max(worst_step_ratio, ratio)
    bounds = neighbour_widened(local_pair_change_bounds(source_changes, progress))
    breaks, worst_ratio, worst_change, floor_breaks, worst_floor_ratio = pace_violations(
        pair,
        alignment.state_indices,
        bounds,
        crossed_pair_changes(probe_bank, alignment.state_indices),
    )
    return OrderingReading(
        state_indices=alignment.state_indices,
        progress=alignment.progress,
        max_step=alignment.max_step,
        step_ratio=worst_step_ratio,
        reference_pace=pace,
        bank_quantum=quantum,
        distinct_states=len(keep),
        comparison_kernel=kernel,
        step_break_count=step_breaks,
        pace_break_count=breaks,
        worst_pace_ratio=worst_ratio,
        worst_pair_change=worst_change,
        pace_floor_break_count=floor_breaks,
        worst_pace_floor_ratio=worst_floor_ratio,
    )


__all__ = [
    "OrderingReading",
    "ProgressAlignment",
    "align_progress",
    "at_scale",
    "comparison_scale",
    "crossed_pair_changes",
    "detoned",
    "dynamic_probe",
    "emission_costs",
    "local_pair_change_bounds",
    "neighbour_widened",
    "observed_pair_changes",
    "pace_violations",
    "read_process_order",
    "reference_advance",
    "reference_pair_changes",
    "step_budget",
    "tone_offset",
]
