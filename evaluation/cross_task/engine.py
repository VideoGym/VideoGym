"""Semantic construction verification, independent of stored answers and references.

Slots describe publicly located answer marks, not multiple-choice whole solutions.
Decoding uses local appearances; legality is recomputed by the family solver.
"""

import json
from dataclasses import replace
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from data.common.render_primitives import ScaledDraw, finish_scaled_canvas
from evaluation.cross_task.drawing import FACTOR

CONNECTED_MARK_GOALS = frozenset(
    (
        "halfplane_intersection",
        "taut_obstacle_path",
        "layered_refraction",
        "disk_union_boundary",
        "rectangle_partition",
    )
)

STROKE_WIDTHS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0)


def observation_widths(slot):
    return STROKE_WIDTHS if slot.kind in ("line", "ring", "arc") else (3.5,)


def observation_styles(slot, source):
    return (
        ("badge", "text")
        if slot.kind == "number" and source["goal"] != "projective_landmarks"
        else ("badge",)
    )


def family(source):
    import importlib

    from evaluation.cross_task.catalog import DESIGNS

    return importlib.import_module(
        "evaluation.cross_task.families." + DESIGNS[source["goal"]].module
    )


def solve(source):
    return family(source).solve(source)


def scene(source):
    canvas, slots = family(source).scene(source)
    for slot in slots:
        if slot.kind == "number" and slot.guide:
            x, y = slot.shape
            canvas.d.rounded_rectangle(
                (x - 18, y - 17, x + 18, y + 17), radius=4, outline="#b4c6d4"
            )
    return canvas, slots


def render(source, state):
    canvas, slots = scene(source)
    image = canvas.surface.copy()
    for slot, value in zip(slots, state):
        slot.paint(image, value)
    return finish_scaled_canvas(image, (512, 512))


def reference_states(source, answer=None):
    answer = solve(source)[0] if answer is None else answer
    state = [-1] * len(answer)
    states = [tuple(state)]
    for i, value in enumerate(answer):
        if value == -1:
            continue
        state[i] = value
        states.append(tuple(state))
    return states


def semantic_mark_groups(source, state, slots=None):
    """Group observed connected paths/boundaries, without consulting solutions.

    A polyline's raster segments are not separately declared public actions.
    Only observed marks participate: unused grid cuts or visibility edges must
    not connect independent constructions through invisible candidate strokes.
    """
    slots = scene(source)[1] if slots is None else slots
    selected = [i for i, value in enumerate(state) if value != -1]
    if source["goal"] not in CONNECTED_MARK_GOALS:
        return tuple((i,) for i in selected)

    def endpoints(slot):
        if slot.kind == "line":
            return np.asarray([slot.shape[0], slot.shape[-1]], dtype=float)
        bounds, start, end = slot.shape
        left, top, right, bottom = bounds
        center = np.array([(left + right) / 2, (top + bottom) / 2])
        radii = np.array([(right - left) / 2, (bottom - top) / 2])
        angles = np.deg2rad([start, end])
        return center + np.column_stack((np.cos(angles), np.sin(angles))) * radii

    parent = {i: i for i in selected}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tips = {i: endpoints(slots[i]) for i in selected}
    for n, i in enumerate(selected):
        for j in selected[:n]:
            if np.min(np.linalg.norm(tips[i][:, None] - tips[j][None, :], axis=2)) <= 1:
                parent[root(i)] = root(j)
    groups = {}
    for i in selected:
        groups.setdefault(root(i), []).append(i)
    return tuple(tuple(group) for group in groups.values())


@lru_cache(16)
def templates(serialized):
    source = json.loads(serialized)
    canvas, slots = scene(source)
    base_image = canvas.image
    base = np.asarray(base_image).astype(float)

    def variant_image(slot, value, width, style="badge"):
        if slot.kind in ("line", "ring", "arc"):
            # Paint before downsampling, including the original colored line
            # beneath an answer. AA overlay on an already downsampled source
            # otherwise turns opaque thin arcs into apparently faint marks.
            if slot.kind == "line":
                points = np.asarray(slot.shape)
                lo, hi = points.min(axis=0), points.max(axis=0)
            else:
                lo, hi = np.asarray(
                    slot.shape[0] if slot.kind == "arc" else slot.shape
                ).reshape(2, 2)
            left, top = np.maximum(np.floor(lo).astype(int) - 8, 0)
            right, bottom = np.minimum(np.ceil(hi).astype(int) + 9, 512)
            offset = np.array([left, top])
            if slot.kind == "line":
                shape = (points - offset).tolist()
            elif slot.kind == "arc":
                shape = [
                    (np.asarray(slot.shape[0]).reshape(2, 2) - offset).ravel().tolist(),
                    *slot.shape[1:],
                ]
            else:
                shape = (np.asarray(slot.shape).reshape(2, 2) - offset).ravel().tolist()
            patch = canvas.surface.crop(
                (left * FACTOR, top * FACTOR, right * FACTOR, bottom * FACTOR)
            )
            # The patch is supersampled even if its width happens to be <512.
            replace(slot, shape=shape)._paint(
                ScaledDraw(ImageDraw.Draw(patch), FACTOR), value, FACTOR, width
            )
            image = base_image.copy()
            observed = finish_scaled_canvas(patch, (right - left, bottom - top))
            # Discard the crop edge's Lanczos footprint; source content beyond
            # the crop must not turn into a spurious answer observation mask.
            inset = (
                4 if left else 0,
                4 if top else 0,
                right - left - (4 if right < 512 else 0),
                bottom - top - (4 if bottom < 512 else 0),
            )
            image.paste(observed.crop(inset), (left + inset[0], top + inset[1]))
            return image
        image = base_image.copy()
        slot.paint(image, value, mark_style=style)
        return image

    all_masks = []
    for slot in slots:
        mask = np.zeros(base.shape[:2], bool)
        for value in slot.options:
            for width in observation_widths(slot):
                im = variant_image(slot, value, width)
                mask |= np.any(np.asarray(im) != base, axis=2)
        all_masks.append(mask)
    coverage = np.sum(all_masks, axis=0)
    exclusive = [m & (coverage == 1) for m in all_masks]
    core_masks = []
    for slot in slots:
        im = variant_image(slot, slot.options[0], observation_widths(slot)[0])
        core_masks.append(
            np.max(np.abs(np.asarray(im).astype(float) - base), axis=2) > 3
        )
    core_coverage = np.sum(core_masks, axis=0)
    # Broad alternative strokes can consume every useful observation pixel of
    # a near-collinear candidate. Use its disjoint thin core in that case.
    for i, (slot, mask) in enumerate(zip(slots, exclusive)):
        widest = variant_image(slot, slot.options[0], observation_widths(slot)[-1])
        visible = np.max(np.abs(np.asarray(widest).astype(float) - base), axis=2) > 3
        if np.count_nonzero(mask & visible) < 12:
            exclusive[i] = core_masks[i] & (core_coverage == 1)
    # A short arc or a nested line can lack an exclusive raster footprint.
    # Observe its visible core; overlapping line hypotheses are pruned jointly.
    for i, mask in enumerate(exclusive):
        if mask.sum() < 12 and slots[i].kind in ("line", "arc"):
            exclusive[i] = core_masks[i]
    if any(not m.any() for m in exclusive):
        raise ValueError("answer marks have no visible pixels")
    if source["goal"] in ("taut_obstacle_path", "bichromatic_matching"):
        exclusive = core_masks
    variations = []
    for slot, mask in zip(slots, exclusive):
        variants = {}
        for value in slot.options:
            for width in observation_widths(slot):
                for style in observation_styles(slot, source):
                    im = variant_image(slot, value, width, style)
                    direction = (
                        np.asarray(im)[mask].astype(float) - base[mask]
                    ).astype("float32")
                    if np.any(direction):
                        variants[value, width, style] = direction
        variations.append(variants)
    return base, slots, variations, exclusive


def appearance_matches(frame, expected, answer_mask=None, *, construction_base=None):
    """Local scene checks tolerate codecs and one-pixel stroke variation.

    Blur removes raster phase sensitivity; local tiles prevent a small changed
    clue from being averaged away across a mostly unchanged whole image.
    """
    import cv2

    a = cv2.GaussianBlur(frame.astype("float32"), (3, 3), 0.7)
    b = cv2.GaussianBlur(expected.astype("float32"), (3, 3), 0.7)
    delta = np.abs(a - b).mean(axis=2)
    tiles = lambda d: np.max(d.reshape(32, 16, 32, 16).mean(axis=(1, 3)))
    if answer_mask is None or not np.any(answer_mask):
        return bool(tiles(delta) <= 13)
    # Generated contours can move by two pixels while retaining the same mark.
    # Give their footprint a small raster tolerance while fixed clues retain
    # the original local preservation bound.
    ink = cv2.dilate(answer_mask.astype("uint8"), np.ones((5, 5), "uint8")).astype(bool)
    if construction_base is not None:
        # A connected contour may emerge piece by piece. Its public contract
        # does not require one uniform opacity across a raster segment. Observe
        # the allowed source-to-orange color interval at each ink pixel; the
        # geometry footprint and fixed scene still come from the public input.
        from PIL import ImageColor
        from evaluation.cross_task.drawing import ORANGE

        origin = construction_base.astype("float32")
        direction = np.array(ImageColor.getrgb(ORANGE), dtype="float32") - origin
        alpha = np.clip(
            np.sum((frame - origin) * direction, axis=2)
            / np.maximum(np.sum(direction**2, axis=2), 1),
            0,
            1,
        )
        variable = np.asarray(expected).astype("float32").copy()
        variable[ink] = (origin + alpha[..., None] * direction)[ink]
        flexible = np.abs(a - cv2.GaussianBlur(variable, (3, 3), 0.7)).mean(axis=2)
        delta = np.where(ink, np.minimum(delta, flexible), delta)
    return bool(tiles(delta) <= 17 and tiles(delta * ~ink) <= 13)


def grade(source, frames):
    if not frames:
        return False, False, {"error": "empty video"}
    base, slots, variants, masks = templates(json.dumps(source, sort_keys=True))
    valid = set(map(tuple, solve(source)))
    accept = getattr(family(source), "accepts", lambda s, a: tuple(a) in valid)

    previous_frame = None
    previous_decoding = None

    def decode(frame):
        nonlocal previous_frame, previous_decoding
        if previous_frame is not None and np.array_equal(frame, previous_frame):
            return previous_decoding
        if frame.shape != base.shape:
            return None, None, False, None
        observed = frame.astype(float)
        state = []
        fractions = []
        widths = []
        styles = []
        ink_levels = []
        for i, mask in enumerate(masks):
            delta = observed[mask] - base[mask]
            candidates = [(float(np.mean(delta**2)), -1, 0.0, 3.5, "badge")]
            for (value, width, style), direction in variants[i].items():
                # Fit opacity independently of the legal stroke width. A thin
                # opaque line is complete, rather than a stalled faint line.
                fraction = float(
                    np.clip(np.sum(delta * direction) / np.sum(direction**2), 0, 1)
                )
                residual = float(np.mean((delta - fraction * direction) ** 2))
                candidates.append((residual, value, fraction, width, style))
            _, value, fraction, width, style = min(candidates, key=lambda c: c[0])
            # Subthreshold source/VAE edge changes are not visible answer ink.
            if fraction < 0.18:
                value = -1
                fraction = 0.0
            elif fraction > 0.88:
                fraction = 1.0
            state.append(value)
            fractions.append(fraction)
            widths.append(width)
            styles.append(style)
            if value == -1:
                ink_levels.append(0.0)
            else:
                # A fixed template measures accumulation across changes in the
                # fitted width; alpha from width5 and width7 is not comparable.
                fixed = variants[i][value, observation_widths(slots[i])[-1], style]
                ink_levels.append(
                    float(max(0.0, np.sum(delta * fixed) / np.sum(fixed**2)))
                )

        def reconstruction(indices):
            expected = Image.fromarray(base.astype("uint8"))
            for i in indices:
                value, fraction = state[i], fractions[i]
                if value == -1:
                    continue
                before = expected.copy()
                slots[i].paint(
                    expected, value, stroke_width=widths[i], mark_style=styles[i]
                )
                if fraction < 1:
                    expected = Image.blend(before, expected, fraction)
            return expected

        if source["goal"] in ("taut_obstacle_path", "bichromatic_matching"):
            # Choose the observed stroke subset by reconstruction alone, before
            # checking the source-derived answer or optimization objective.
            selected = [i for i, v in enumerate(state) if v != -1]
            def residual(indices):
                return float(np.mean((observed - np.asarray(reconstruction(indices))) ** 2))
            best = residual(selected)
            while selected:
                candidates = [
                    (residual([j for j in selected if j != i]), i) for i in selected
                ]
                error, remove = min(candidates)
                if error > best + 1e-6:
                    break
                selected.remove(remove)
                state[remove], fractions[remove], ink_levels[remove] = -1, 0.0, 0.0
                best = error

        if source["goal"] == "minimum_width_strip":
            # A strip has one support pair. Nearly parallel polygon edges can
            # share their widened ink, so decode the pair jointly, considering
            # every public orientation rather than the solver's optimum.
            groups = [()] + [tuple(range(i, i + 2)) for i in range(0, len(slots), 2)]
            selected = min(
                groups,
                key=lambda g: np.mean((observed - np.asarray(reconstruction(g))) ** 2),
            )
            for i in range(len(slots)):
                if i not in selected:
                    state[i], fractions[i], ink_levels[i] = -1, 0.0, 0.0
        expected = reconstruction(range(len(slots)))
        previous_frame = frame
        previous_decoding = (
            tuple(state),
            fractions,
            appearance_matches(
                frame,
                np.asarray(expected),
                np.any(np.asarray(expected) != base, axis=2),
                construction_base=(
                    base
                    if source["goal"] in CONNECTED_MARK_GOALS
                    and any(0 < a < 1 for a in fractions)
                    else None
                ),
            ),
            ink_levels,
        )
        return previous_decoding

    terminal, terminal_alpha, terminal_scene, _ = decode(frames[-1])
    answer_correct = terminal is not None and bool(accept(source, terminal))
    terminal_complete = terminal_alpha is not None and all(
        p in (0.0, 1.0) for p in terminal_alpha
    )
    final = bool(terminal_scene and answer_correct and terminal_complete)
    groups = (
        semantic_mark_groups(source, terminal, slots) if terminal is not None else ()
    )
    group_for = {i: n for n, group in enumerate(groups) for i in group}
    # Marks that disappear before the terminal state stay independent; the
    # immutable-content checks below reject their disappearance regardless.
    group_for.update(
        (i, len(groups) + i) for i in range(len(slots)) if i not in group_for
    )
    previous = [-1] * len(slots)
    previous_alpha = [0.0] * len(slots)
    peak_ink = [0.0] * len(slots)
    completed = [False] * len(slots)
    active = None
    error = None
    reason = None
    for index, frame in enumerate(frames):
        state, fractions, intact, ink_levels = decode(frame)
        if not intact:
            error, reason = index, "scene_or_unrecognized_mark"
            break
        if index == 0 and any(v != -1 for v in state):
            error, reason = index, "answer_in_first_frame"
            break
        if any(a != -1 and a != b for a, b in zip(previous, state)):
            error, reason = index, "replaced_or_simultaneous_marks"
            break
        erased = [
            i
            for i, (peak, now) in enumerate(zip(peak_ink, ink_levels))
            if now + 0.08 * max(peak, 0.18) < peak
        ]
        if erased:
            error, reason = index, (
                "completed_mark_faded"
                if any(completed[i] for i in erased)
                else "nonmonotone_mark"
            )
            break
        progressing = {
            group_for[i]
            for i, (old, new) in enumerate(zip(previous_alpha, fractions))
            if new > old + 0.08
        }
        if len(progressing) > 1 or (
            active is not None and any(group != active for group in progressing)
        ):
            error, reason = index, "replaced_or_simultaneous_marks"
            break
        unfinished = {
            group_for[i] for i, alpha in enumerate(fractions) if 0 < alpha < 1
        }
        if len(unfinished) > 1:
            error, reason = index, "replaced_or_simultaneous_marks"
            break
        active = next(iter(unfinished), None)
        previous, previous_alpha = state, fractions
        peak_ink = [max(a, b) for a, b in zip(peak_ink, ink_levels)]
        completed = [old or alpha == 1 for old, alpha in zip(completed, fractions)]
    return (
        error is None and active is None and final,
        final,
        {
            "first_invalid_frame": error,
            "semantic_mark_component_count": len(groups),
            "invalid_reason": reason,
            "answer_correct": answer_correct,
            "terminal_scene_preserved": bool(terminal_scene),
            "terminal_complete": terminal_complete,
            "solver_source_only": True,
            "enumerated_witnesses": len(valid),
            "decoded_answer": terminal,
        },
    )
