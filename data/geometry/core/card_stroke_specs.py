"""Source-derived serial-stroke specifications for Geometry card deliveries."""

from __future__ import annotations

from typing import Any, Mapping

from data.geometry.core.stroke_sequence import EndpointStroke


SERIAL_STROKE_CARD_TASKS = frozenset(
    {
        "convex_hull_count",
        "mirror_image_select",
        "triangle_inequality_check",
    }
)


def _point(raw: Any) -> tuple[float, float]:
    return float(raw[0]), float(raw[1])


def geometry_card_strokes(
    task_name: str,
    state: Mapping[str, Any],
    *,
    correct_index: int,
    reverse_hull: bool = False,
) -> list[EndpointStroke]:
    """Derive the ordered answer-bearing strokes without saved media."""

    if task_name == "convex_hull_count":
        from data.geometry.convex_hull_count.generator import _convex_hull

        hull = [_point(point) for point in _convex_hull(state["points"])]
        if reverse_hull:
            # Both boundary chains may be drawn first. The public contract
            # fixes the leftmost start, but does not choose an orientation.
            hull = [hull[0], *reversed(hull[1:])]
        return [
            EndpointStroke(
                f"hull_edge_{index + 1}",
                (hull[index], hull[(index + 1) % len(hull)]),
                5.0,
            )
            for index in range(len(hull))
        ]

    if task_name == "mirror_image_select":
        from data.geometry.mirror_image_select.generator import (
            MirrorImageSelectPattern,
            _MIRROR_CENTER,
            _SOURCE_SCALE,
            _apply_transform,
        )

        source_vertices = state["source_vertices"]
        expected = MirrorImageSelectPattern._map_points(
            _apply_transform(source_vertices, "mirror_lr"),
            _MIRROR_CENTER,
            _SOURCE_SCALE,
        )
        # Endpoint correspondence guides are evidence overlays, not
        # answer-bearing construction strokes.  They can cross the first
        # mirrored segment in the terminal footprint, so including them in
        # the serial pen-stroke grammar creates a false future-ink signal.
        # The mirror polyline itself remains a strict ordered construction.
        strokes = []
        strokes.extend(
            EndpointStroke(
                f"mirror_segment_{index + 1}",
                (_point(expected[index]), _point(expected[index + 1])),
                6.0,
            )
            for index in range(len(expected) - 1)
        )
        return strokes

    if task_name == "triangle_inequality_check":
        from data.geometry.triangle_inequality_check.generator import (
            _comparison_payload,
            comparison_stroke_geometry,
        )

        triples = [[int(value) for value in triple] for triple in state["triples"]]
        task = {"triples": triples, "comparisons": _comparison_payload(triples)}
        return [
            EndpointStroke(
                str(stroke["label"]),
                (stroke["start"], stroke["end"]),
                float(stroke["width"]),
            )
            for stroke in comparison_stroke_geometry(task)
        ]

    return []


__all__ = ["SERIAL_STROKE_CARD_TASKS", "geometry_card_strokes"]
