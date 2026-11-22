"""Shared ruled-height card content for the two ICL size-sequence tasks.

Sizes are made typed and perceptually reliable by drawing every shape on the
same ruled card: a baseline plus numbered unit lines 1-8.  A shape stands on
the baseline and its top edge lands exactly on one numbered line, so its
height IS that line's whole number.  Both ``alternating_size_sequence`` and
``increasing_size_sequence`` paint through this module, and their verifier
templates re-render through the same functions.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from data.common.pipeline.media import load_font
from data.common.render_primitives import DOCTRINE_SUPERSAMPLE, ScaledDraw

RULER_UNITS = 8
_LABEL_BAND = 20.0
_BASELINE_INSET = 14.0
_TOP_INSET = 10.0

_BASELINE_COLOR = (116, 122, 132)
_LINE_COLOR = (208, 213, 221)
_LABEL_COLOR = (122, 128, 138)

SIZE_SHAPES = ("circle", "square", "triangle", "diamond", "hexagon")
SIZE_COLORS = (
    (63, 111, 212),
    (36, 150, 100),
    (224, 120, 60),
    (149, 88, 192),
    (190, 82, 82),
)


def ruler_geometry(box) -> Tuple[float, float, float]:
    """Return (baseline_y, unit, shape_center_x) for one content box."""

    left, top, right, bottom = (float(value) for value in box)
    baseline_y = bottom - _BASELINE_INSET
    unit = (baseline_y - (top + _TOP_INSET)) / float(RULER_UNITS)
    center_x = left + _LABEL_BAND + (right - left - _LABEL_BAND) / 2.0
    return baseline_y, unit, center_x


def paint_ruler(sdraw: ScaledDraw, box) -> None:
    """Paint the shared baseline and numbered unit lines of one card."""

    left, _top, right, _bottom = (float(value) for value in box)
    baseline_y, unit, _center_x = ruler_geometry(box)
    label_font = load_font(10 * DOCTRINE_SUPERSAMPLE)
    sdraw.line((left + 4.0, baseline_y, right - 4.0, baseline_y), fill=_BASELINE_COLOR, width=2)
    for level in range(1, RULER_UNITS + 1):
        line_y = baseline_y - level * unit
        sdraw.line((left + _LABEL_BAND, line_y, right - 4.0, line_y), fill=_LINE_COLOR, width=1)
        sdraw.text((left + 9.0, line_y), str(level), fill=_LABEL_COLOR, font=label_font, anchor="mm")


def paint_sized_shape(sdraw: ScaledDraw, box, item: Mapping[str, Any], position: int) -> None:
    """Paint the ruled lines and one shape whose height is ``item['height']`` units."""

    del position
    baseline_y, unit, center_x = ruler_geometry(box)
    paint_ruler(sdraw, box)

    height_units = int(item["height"])
    if not 1 <= height_units <= RULER_UNITS:
        raise ValueError(f"shape height {height_units} is outside the ruled card")
    shape = str(item["shape"])
    color = tuple(int(value) for value in item["color"])
    outline = (44, 48, 54)
    h = height_units * unit
    top_y = baseline_y - h
    from data.common.prompt_paraphrase import active_profile, draw_column, draw_gem, draw_crate
    if active_profile("height_solids"):
        width=h*({"circle":.65,"square":.75,"triangle":.62,"diamond":.58,"hexagon":.70}[shape])
        bounds=(center_x-width/2,top_y,center_x+width/2,baseline_y)
        if shape in {"circle","hexagon"}: draw_column(sdraw,bounds,color,outline)
        elif shape=="square": draw_crate(sdraw,bounds,color,outline)
        else: draw_gem(sdraw,bounds,color,outline)
        return
    if shape == "circle":
        sdraw.ellipse((center_x - h / 2.0, top_y, center_x + h / 2.0, baseline_y), fill=color, outline=outline, width=2)
    elif shape == "square":
        sdraw.rectangle((center_x - h / 2.0, top_y, center_x + h / 2.0, baseline_y), fill=color, outline=outline, width=2)
    elif shape == "triangle":
        half = h * 0.56
        sdraw.polygon(
            [(center_x, top_y), (center_x - half, baseline_y), (center_x + half, baseline_y)],
            fill=color,
            outline=outline,
            width=2,
        )
    elif shape == "diamond":
        half = h * 0.34
        middle_y = baseline_y - h / 2.0
        sdraw.polygon(
            [
                (center_x, top_y),
                (center_x - half, middle_y),
                (center_x, baseline_y),
                (center_x + half, middle_y),
            ],
            fill=color,
            outline=outline,
            width=2,
        )
    elif shape == "hexagon":
        half_w = h * 0.44
        quarter = h * 0.25
        sdraw.polygon(
            [
                (center_x - half_w * 0.55, top_y),
                (center_x + half_w * 0.55, top_y),
                (center_x + half_w, top_y + quarter * 2.0),
                (center_x + half_w * 0.55, baseline_y),
                (center_x - half_w * 0.55, baseline_y),
                (center_x - half_w, top_y + quarter * 2.0),
            ],
            fill=color,
            outline=outline,
            width=2,
        )
    else:
        raise ValueError(f"unknown size-card shape: {shape}")


SIZE_CARD_GLYPHS = {
    "schema_version": "icl_rendered_glyph_inventory_v1",
    "static_whitelist": ["1", "2", "3", "4", "5", "6", "7", "8", "?"],
    "bounded_dynamic_schema": [],
    "observed_dynamic_values": [],
    "answer_fields_excluded": True,
}

SIZE_INPUT_COPY = (
    "Four numbered cards show one height sequence: cards 1-3 each hold the same shape standing on a "
    "baseline under ruled lines numbered 1-8, and card 4 holds only a question mark over the same "
    "ruled lines."
)
SIZE_RULER_SENTENCE = (
    "Every shape stands on the baseline and its top edge lands exactly on one numbered line, so a "
    "shape's height is that line's number."
)
SIZE_TERMINAL_COPY = (
    "the same shape at the same color standing on the baseline with its top edge exactly on the "
    "required numbered line."
)


def make_ruled_height_decoder(templates, box, color):
    """Read a ruler height without treating a small glyph shift as a new value."""
    import cv2
    import numpy as np
    from data.common.verifiers.region_gates import static_region_errors, STATIC_REGION_FLOOR

    left, top, right, bottom = map(lambda value: int(round(value)), box)
    color = np.asarray(color, dtype=float)
    def mask(image):
        crop = np.asarray(image)[top:bottom, left:right].astype(float)
        return np.linalg.norm(crop-color, axis=2) <= 45
    expected = {value: mask(image) for value, image in templates.items()}
    crops = {value: np.asarray(image)[top:bottom, left:right] for value, image in templates.items()}
    def bounds(ink):
        yy, xx = np.where(ink)
        return np.array([xx.min(), yy.min(), xx.max(), yy.max()]) if len(xx) else None
    boxes = {value: bounds(ink) for value, ink in expected.items()}
    tolerance = np.ones((7, 7), np.uint8)
    def decode(candidate):
        observed = mask(candidate)
        observed_box = bounds(observed)
        readings = {}
        for value, ink in expected.items():
            target = boxes[value]
            if observed_box is None or target is None:
                continue
            # Top and baseline encode the integer; horizontal centre and
            # contour coverage retain the source's visible symbol identity.
            endpoints = max(abs(observed_box[1]-target[1]), abs(observed_box[3]-target[3]))
            centered = abs((observed_box[0]+observed_box[2])-(target[0]+target[2])) <= 8
            near_observed = cv2.dilate(observed.astype(np.uint8), tolerance).astype(bool)
            near_expected = cv2.dilate(ink.astype(np.uint8), tolerance).astype(bool)
            # The ruler and card are fixed, and the question must disappear.
            # Allow the three-pixel shift plus its outline and smoothing halo.
            symbol_neighbourhood = cv2.dilate(ink.astype(np.uint8), np.ones((13,13),np.uint8)).astype(bool)
            background_error = static_region_errors(
                np.asarray(candidate)[top:bottom,left:right], crops[value], ~symbol_neighbourhood,
            )[0]
            coverage = float(near_observed[ink].mean()) if ink.any() else 0.
            precision = float(near_expected[observed].mean()) if observed.any() else 0.
            readings[value] = {'endpoint_error_pixels': int(endpoints),
                               'centered': bool(centered), 'coverage': coverage, 'precision': precision,
                               'background_region_error': background_error,
                               'pass': bool(endpoints <= 3 and centered and coverage >= .80 and precision >= .80
                                            and background_error <= STATIC_REGION_FLOOR)}
        matches = [value for value, reading in readings.items() if reading['pass']]
        return (matches[0] if len(matches) == 1 else None), {
            'ruled_height_readings': readings,
            'ruled_height_observed_color_bbox': observed_box.tolist() if observed_box is not None else None,
        }
    return decode


__all__ = [
    "RULER_UNITS",
    "SIZE_CARD_GLYPHS",
    "SIZE_COLORS",
    "SIZE_INPUT_COPY",
    "SIZE_RULER_SENTENCE",
    "SIZE_TERMINAL_COPY",
    "SIZE_SHAPES",
    "paint_ruler",
    "paint_sized_shape",
    "make_ruled_height_decoder",
    "ruler_geometry",
]
