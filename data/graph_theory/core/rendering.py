"""One low-text renderer for the unified labelled-graph task suite.

Visual system v3: a cool minimal gray canvas, compact numeric-label nodes,
and one shared answer language.  Every stroke passes through the shared
``ScaledDraw`` proxy at the doctrine 4x supersample factor and is finished
with one LANCZOS downsample, so callers stay in unscaled 512-space.

The family output vocabulary is deliberately small:

* orange stroke ........ a committed answer route or trail;
* orange ring .......... an answer-set node;
* teal stroke .......... intermediate evidence chains (ancestor chains, the
                         longest tree route before its endpoints are ringed);
* pale interior tint ... group membership evidence (components, the two root
                         branches, the one-hop frontier);
* badge digit .......... the resolved degree replacing the input `?`.

Role rings on the input side are green (start/root), red (goal), gold
(checkpoint), purple (query), and blue (path start); a blocked vertex is the
one solid gray node.  Nothing else may appear in any state.
"""

from __future__ import annotations

from data.common.assets import render_font
from data.common.visual_style import current_style

from contextlib import contextmanager
from contextvars import ContextVar
import math
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.graph_theory.core.semantics import parse_graph_source

FACTOR = DOCTRINE_SUPERSAMPLE
CANVAS = 512

# Answer-ring radius and stroke width are presentation geometry, not answer
# semantics. Geometry overrides author varied control videos. The verifier
# observes each outline separately; the node set and other overlays stay source-derived.
_ANSWER_RING_GEOMETRY: ContextVar = ContextVar("videogym_answer_ring_geometry", default=None)
_ANSWER_RING_MASKS: ContextVar = ContextVar("videogym_answer_ring_masks", default=None)


@contextmanager
def use_answer_ring_masks(masks):
    """Observe source-derived answer nodes with candidate-specific ring outlines."""
    token = _ANSWER_RING_MASKS.set(masks)
    try:
        yield
    finally:
        _ANSWER_RING_MASKS.reset(token)



BACKGROUND = (244, 245, 248)
INK = (35, 43, 53)
EDGE = (117, 128, 139)
ORANGE = (238, 126, 45)
TEAL = (22, 157, 151)
GREEN = (36, 158, 92)
RED = (211, 67, 67)
GOLD = (218, 166, 42)
PURPLE = (143, 83, 190)
BLUE = (52, 111, 184)

NODE_FILL = (255, 255, 252)
BLOCKED_FILL = (189, 193, 198)
RED_FILL = (245, 185, 185)
BLUE_FILL = (181, 211, 244)
# Group-membership tints shared by every set-evidence task: pale blue, pale
# orange, pale green, pale purple, in component order.
GROUP_TINTS = ((172, 201, 236), (245, 199, 158), (172, 217, 186), (212, 188, 232))
TEAL_TINT = (168, 215, 211)
BADGE_FILL = (255, 248, 214)

NODE_RADIUS = 16
RING_RADIUS = 22
SOLUTION_RING_RADIUS = 26
BADGE_RADIUS = 10
BADGE_DISTANCE = 38.0

EDGE_WIDTH = 4
ROUTE_WIDTH = 8
EMPHASIS_WIDTH = 9
ROLE_RING_WIDTH = 4
SOLUTION_RING_WIDTH = 5

# Tasks whose answer overlay appears in two sequential local-fade phases; the
# boundary is the shared semantic checkpoint used by replay and verifier.
PHASE_SPLIT = {
    "two_hop_neighbor_count": 0.5,
    "closest_common_ancestor": 0.6,
    "diameter_endpoints": 0.6,
    "balance_comparison": 0.65,
}

_ROUTE_GROWTH_TASKS = frozenset({
    "shortest_path_choice",
    "route_via_checkpoint",
    "avoid_blocked_vertex",
    "trace_all_edges_once",
    "depth_lookup",
    "path_between_nodes",
})


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    if style := current_style():
        from data.common.assets import font_path
        return ImageFont.truetype(str(font_path(style.font)), size * FACTOR)
    return render_font(size * FACTOR, bold=bold)


def _mix(left: Sequence[int], right: Sequence[int], amount: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, float(amount)))
    return tuple(int(round((1.0 - value) * left[i] + value * right[i])) for i in range(3))


def _phase_alpha(progress: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0 if progress >= end else 0.0
    return max(0.0, min(1.0, (progress - start) / (end - start)))


def _trim_segment(
    start: Sequence[float],
    end: Sequence[float],
    radius: float = float(NODE_RADIUS),
) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, y0 = map(float, start)
    x1, y1 = map(float, end)
    length = max(1e-9, math.hypot(x1 - x0, y1 - y0))
    trim = min(radius, max(0.0, length / 2.0 - 1.0))
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    return (x0 + ux * trim, y0 + uy * trim), (x1 - ux * trim, y1 - uy * trim)


def _centered_text(
    sd: ScaledDraw,
    center: Sequence[float],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: Sequence[int] = INK,
) -> None:
    x, y = float(center[0]) * sd.factor, float(center[1]) * sd.factor
    box = sd.raw.textbbox((0, 0), text, font=font)
    sd.raw.text(
        (x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - box[1]),
        text,
        font=font,
        fill=tuple(fill),
    )


def _partial_route(
    sd: ScaledDraw,
    positions: Mapping[str, Sequence[float]],
    path: Sequence[str],
    progress: float,
    *,
    color: Sequence[int],
    width: int = ROUTE_WIDTH,
) -> None:
    """Grow one route stroke edge by edge; one partial stroke at most."""

    edge_count = max(0, len(path) - 1)
    travelled = max(0.0, min(1.0, float(progress))) * edge_count
    for index, (left, right) in enumerate(zip(path, path[1:])):
        amount = max(0.0, min(1.0, travelled - index))
        if amount <= 0.0:
            continue
        (x0, y0), (x1, y1) = _trim_segment(positions[left], positions[right])
        stop = (x0 + (x1 - x0) * amount, y0 + (y1 - y0) * amount)
        sd.line(((x0, y0), stop), fill=tuple(color), width=width)


def _composite_overlay(image: Image.Image, overlay: Image.Image, opacity: float) -> None:
    """Composite one complete local overlay with a shared opacity.

    The overlay is drawn at its terminal footprint first; scaling only its
    alpha keeps every visible intermediate on that same footprint.  This is
    deliberately different from changing a radius or growing a stroke.
    """

    amount = max(0.0, min(1.0, float(opacity)))
    if amount <= 0.0:
        return
    if amount < 1.0:
        alpha = overlay.getchannel("A").point(lambda value: int(round(value * amount)))
        overlay.putalpha(alpha)
    composited = Image.alpha_composite(image.convert("RGBA"), overlay)
    image.paste(composited.convert("RGB"))


def _new_overlay(image: Image.Image) -> tuple[Image.Image, ScaledDraw]:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    return overlay, ScaledDraw(ImageDraw.Draw(overlay), FACTOR)


def _overlay_edge_emphasis(
    image: Image.Image,
    positions: Mapping[str, Sequence[float]],
    edges: Sequence[Sequence[str]],
    *,
    color: Sequence[int],
    opacity: float,
    width: int = EMPHASIS_WIDTH,
) -> None:
    """Fade complete trimmed edge strokes in together at a fixed footprint."""

    if opacity <= 0.0 or not edges:
        return
    overlay, sd = _new_overlay(image)
    for left, right in edges:
        start, end = _trim_segment(positions[str(left)], positions[str(right)])
        sd.line((start, end), fill=(*tuple(color), 255), width=width)
    _composite_overlay(image, overlay, opacity)


def _overlay_route_fade(
    image: Image.Image,
    positions: Mapping[str, Sequence[float]],
    paths: Sequence[Sequence[str]],
    *,
    color: Sequence[int],
    opacity: float,
    width: int = ROUTE_WIDTH,
) -> None:
    """Fade one or more complete route footprints in as one evidence unit."""

    if opacity <= 0.0 or not paths:
        return
    overlay, sd = _new_overlay(image)
    for path in paths:
        for left, right in zip(path, path[1:]):
            start, end = _trim_segment(positions[str(left)], positions[str(right)])
            sd.line((start, end), fill=(*tuple(color), 255), width=width)
    _composite_overlay(image, overlay, opacity)


def _overlay_node_rings(
    image: Image.Image,
    positions: Mapping[str, Sequence[float]],
    labels: Sequence[str],
    *,
    color: Sequence[int],
    opacity: float,
    radius: float = float(SOLUTION_RING_RADIUS),
    width: int = SOLUTION_RING_WIDTH,
) -> None:
    """Fade complete answer rings in together with no enumeration order."""

    if opacity <= 0.0 or not labels:
        return
    if (masks := _ANSWER_RING_MASKS.get()) is not None:
        alpha = np.maximum.reduce([masks[str(label)] for label in labels])
        overlay = Image.new("RGBA", image.size, (*tuple(color), 255))
        opacity_mask = Image.fromarray(np.round(alpha*255).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
        overlay.putalpha(opacity_mask)
        _composite_overlay(image, overlay, opacity)
        return
    if (geometry := _ANSWER_RING_GEOMETRY.get()) is not None:
        radius, width = geometry
    overlay, sd = _new_overlay(image)
    for label in labels:
        x, y = map(float, positions[str(label)])
        sd.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(*tuple(color), 255),
            width=width,
        )
    _composite_overlay(image, overlay, opacity)


# ---------------------------------------------------------------------------
# Degree badge geometry (shared with the verifier ROI)
# ---------------------------------------------------------------------------


def _point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length_sq))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def degree_badge_center(source: Mapping[str, Any]) -> tuple[float, float]:
    """Deterministic badge anchor for the query node, clear of edges and nodes.

    The direction is a pure function of the visible geometry (a rendering
    rule, not stored metadata): among sixteen candidate directions around the
    query node, pick the one whose badge disk keeps the largest clearance
    from every edge segment and every other node, preferring the upper-right
    on ties.
    """

    positions = {str(node["label"]): node["position"] for node in source["nodes"]}
    query = str(source["roles"].get("query") or "")
    cx, cy = map(float, positions[query])
    segments = [
        (positions[str(edge["u"])], positions[str(edge["v"])])
        for edge in source["edges"]
    ]
    others = [
        tuple(map(float, node["position"]))
        for node in source["nodes"]
        if str(node["label"]) != query
    ]
    # Candidate order encodes the tie preference: start at the upper-right
    # diagonal and walk outward alternately.
    base = -math.pi / 4.0
    offsets = [0]
    for step in range(1, 8):
        offsets.extend((step, -step))
    offsets.append(8)
    best: tuple[float, float] | None = None
    best_score = -1.0
    for rank, offset in enumerate(offsets):
        angle = base + offset * (math.pi / 8.0)
        bx = cx + BADGE_DISTANCE * math.cos(angle)
        by = cy + BADGE_DISTANCE * math.sin(angle)
        margin = 30.0
        if not (margin <= bx <= CANVAS - margin and margin <= by <= CANVAS - margin):
            continue
        clearance = min(
            [_point_segment_distance((bx, by), start, end) for start, end in segments]
            + [math.hypot(bx - ox, by - oy) for ox, oy in others]
            or [999.0]
        )
        score = clearance - rank * 1e-3
        if score > best_score:
            best_score = score
            best = (bx, by)
    if best is None:
        best = (cx + BADGE_DISTANCE, cy - BADGE_DISTANCE)
    return best


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _group_fill_alpha(task: str, progress: float) -> float:
    split = PHASE_SPLIT.get(task)
    if split is None:
        return progress
    return _phase_alpha(progress, 0.0, split)


def _node_fill(
    node: Mapping[str, Any],
    source: Mapping[str, Any],
    solution: Mapping[str, Any] | None,
    progress: float,
) -> tuple[int, int, int]:
    label = str(node["label"])
    task = str(source["task_name"])
    roles = source["roles"]
    fill = NODE_FILL
    color = node.get("bipartite_color")
    if color == "RED":
        fill = RED_FILL
    elif color == "BLUE":
        fill = BLUE_FILL
    if label == str(roles.get("blocked") or ""):
        fill = BLOCKED_FILL
    if not solution or progress <= 0.0:
        return fill

    if task == "bipartite_color_completion" and label in (solution.get("node_color") or {}):
        target = RED_FILL if solution["node_color"][label] == "RED" else BLUE_FILL
        return _mix(fill, target, progress)
    if task in {"connected_component_count", "balance_comparison"}:
        components = solution.get("components") or []
        alpha = _group_fill_alpha(task, progress)
        for index, component in enumerate(components):
            if label in {str(value) for value in component}:
                tints = solution.get("component_tints") or GROUP_TINTS
                return _mix(fill, tints[index % len(tints)], alpha)
    if task == "two_hop_neighbor_count":
        components = solution.get("components") or []
        if components and label in {str(value) for value in components[0]}:
            return _mix(fill, TEAL_TINT, _group_fill_alpha(task, progress))
    return fill


def _draw_node(
    sd: ScaledDraw,
    node: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    solution: Mapping[str, Any] | None,
    progress: float,
) -> None:
    label = str(node["label"])
    x, y = map(float, node["position"])
    roles = source["roles"]
    task = str(source["task_name"])
    fill = _node_fill(node, source, solution, progress)
    box = (x - NODE_RADIUS, y - NODE_RADIUS, x + NODE_RADIUS, y + NODE_RADIUS)
    if style := current_style():
        style.token(sd, box, fill, INK, width=3)
    else:
        sd.ellipse(box, fill=fill, outline=INK, width=3)
    from data.common.prompt_paraphrase import current_prompt_paraphrase
    display_label = (current_prompt_paraphrase() or {}).get('display_labels', {}).get(label, label)
    _centered_text(sd, (x, y), display_label, font=_font(14, bold=True))

    # Role rings share one base radius; when one node carries two roles the
    # later ring sits concentrically inside the first instead of colliding.
    role_rings: list[tuple[Sequence[int], int]] = []
    # The path task marks its two endpoints blue/red; its root ring is
    # suppressed so the start ring never merges with a concentric green ring
    # into an ambiguous two-tone band.
    show_root = task != "path_between_nodes"
    if label == str(roles.get("start") or "") or (show_root and label == str(roles.get("root") or "")):
        role_rings.append((GREEN, ROLE_RING_WIDTH))
    if label == str(roles.get("goal") or ""):
        role_rings.append((RED, ROLE_RING_WIDTH))
    if label == str(roles.get("checkpoint") or ""):
        role_rings.append((GOLD, 5))
    if label == str(roles.get("query_start") or ""):
        role_rings.append((BLUE, ROLE_RING_WIDTH))
    if label == str(roles.get("query_goal") or ""):
        role_rings.append((RED, ROLE_RING_WIDTH))
    if label == str(roles.get("query") or "") and task != "bipartite_color_completion":
        role_rings.append((PURPLE, ROLE_RING_WIDTH))
    if label in {str(value) for value in roles.get("queries", [])}:
        role_rings.append((PURPLE, ROLE_RING_WIDTH))
    for index, (ring_color, ring_width) in enumerate(role_rings):
        radius = RING_RADIUS - 3 * index
        sd.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=tuple(ring_color),
            width=ring_width,
        )

    if task == "degree_missing_value" and label == str(roles.get("query") or ""):
        _draw_degree_badge(sd, source, solution, progress)


def _draw_degree_badge(
    sd: ScaledDraw,
    source: Mapping[str, Any],
    solution: Mapping[str, Any] | None,
    progress: float,
) -> None:
    bx, by = degree_badge_center(source)
    sd.ellipse(
        (bx - BADGE_RADIUS, by - BADGE_RADIUS, bx + BADGE_RADIUS, by + BADGE_RADIUS),
        fill=BADGE_FILL,
        outline=PURPLE,
        width=2,
    )
    font = _font(12, bold=True)
    if not solution or progress <= 0.0:
        _centered_text(sd, (bx, by), "?", font=font)
        return
    # The `?` fades away first; only then does the complete answer digit fade
    # in on the same fixed footprint.  Exactly one glyph is drawn per state so
    # neither glyph clips the other mid-transition.
    if progress < 0.5:
        visibility = 1.0 - _phase_alpha(progress, 0.0, 0.5)
        _centered_text(sd, (bx, by), "?", font=font, fill=_mix(BADGE_FILL, INK, visibility))
    else:
        visibility = _phase_alpha(progress, 0.5, 1.0)
        _centered_text(
            sd,
            (bx, by),
            str(solution["answer"]),
            font=font,
            fill=_mix(BADGE_FILL, INK, visibility),
        )


# ---------------------------------------------------------------------------
# Solution overlays
# ---------------------------------------------------------------------------


def _draw_solution_strokes(
    image: Image.Image,
    sd: ScaledDraw,
    source: Mapping[str, Any],
    positions: Mapping[str, Sequence[float]],
    solution: Mapping[str, Any],
    progress: float,
) -> None:
    task = str(source["task_name"])
    if progress <= 0.0:
        return
    paths = [[str(value) for value in path] for path in (solution.get("paths") or [])]
    if task in _ROUTE_GROWTH_TASKS and paths:
        _partial_route(sd, positions, paths[0], progress, color=ORANGE)
        return
    if task == "closest_common_ancestor" and paths:
        _overlay_route_fade(
            image, positions, paths,
            color=TEAL, opacity=_phase_alpha(progress, 0.0, PHASE_SPLIT[task]),
        )
        return
    if task == "diameter_endpoints" and paths:
        _overlay_route_fade(
            image, positions, paths[:1],
            color=TEAL, opacity=_phase_alpha(progress, 0.0, PHASE_SPLIT[task]),
        )
        return
    solution_edges = [edge for edge in (solution.get("edges") or []) if len(edge) == 2]
    if task == "bridge_edge_choice" and solution_edges:
        _overlay_edge_emphasis(image, positions, solution_edges, color=ORANGE, opacity=progress)
    elif task == "degree_missing_value" and solution_edges:
        _overlay_edge_emphasis(image, positions, solution_edges, color=ORANGE, opacity=progress)


ANSWER_RING_TASKS = frozenset({
    "euler_trail_endpoints", "leaf_count", "subtree_size", "closest_common_ancestor",
    "diameter_endpoints", "two_hop_neighbor_count", "balance_comparison",
})


def solution_ring_labels(task: str, solution: Mapping[str, Any]) -> list[str]:
    """The answer-set nodes that receive an orange ring in the terminal state."""

    if task in {"euler_trail_endpoints", "leaf_count", "subtree_size",
                "closest_common_ancestor", "diameter_endpoints"}:
        return [str(value) for value in solution.get("nodes") or []]
    if task == "two_hop_neighbor_count":
        components = solution.get("components") or []
        return [str(value) for value in components[1]] if len(components) > 1 else []
    if task == "balance_comparison":
        components = solution.get("components") or []
        answer = str(solution.get("answer"))
        chosen = {"LEFT": (0,), "RIGHT": (1,), "EQUAL": (0, 1)}.get(answer, ())
        return [str(value) for index in chosen if index < len(components) for value in components[index]]
    return []


def _draw_solution_rings(
    image: Image.Image,
    source: Mapping[str, Any],
    positions: Mapping[str, Sequence[float]],
    solution: Mapping[str, Any],
    progress: float,
) -> None:
    task = str(source["task_name"])
    if progress <= 0.0 or task not in ANSWER_RING_TASKS:
        return
    split = PHASE_SPLIT.get(task)
    opacity = progress if split is None else _phase_alpha(progress, split, 1.0)
    _overlay_node_rings(image, positions, solution_ring_labels(task, solution), color=ORANGE, opacity=opacity)


def render_unified_graph(
    raw_source: Mapping[str, Any],
    solution: Mapping[str, Any] | None = None,
    *,
    progress: float = 0.0,
) -> Image.Image:
    """Render source state or one cumulative solution progress state."""

    task = str(raw_source.get("task_name") or raw_source.get("semantic_state", {}).get("task_name") or "")
    source = parse_graph_source(task, raw_source)
    progress = max(0.0, min(1.0, float(progress)))
    image, sd = new_scaled_canvas(CANVAS, CANVAS, FACTOR, background=BACKGROUND)
    style = current_style()
    if style:
        style.surface(sd, (0, 0, CANVAS, CANVAS), style.background)
    positions = {str(node["label"]): node["position"] for node in source["nodes"]}

    for edge in source["edges"]:
        points = (tuple(map(float, positions[edge["u"]])), tuple(map(float, positions[edge["v"]])))
        if style:
            style.edge(sd, points, EDGE_WIDTH)
        else:
            sd.line(points, fill=EDGE, width=EDGE_WIDTH)
    if solution:
        _draw_solution_strokes(image, sd, source, positions, solution, progress)
        # Alpha overlays composite onto the canvas image; re-bind the draw
        # proxy so node rendering continues on the composited pixels.
        sd = ScaledDraw(ImageDraw.Draw(image), FACTOR)
    for node in source["nodes"]:
        _draw_node(sd, node, source, solution=solution, progress=progress)
    if solution:
        _draw_solution_rings(image, source, positions, solution, progress)

    return finish_scaled_canvas(image, (CANVAS, CANVAS))


__all__ = [
    "BACKGROUND",
    "CANVAS",
    "NODE_RADIUS",
    "ORANGE",
    "PHASE_SPLIT",
    "RING_RADIUS",
    "SOLUTION_RING_RADIUS",
    "SOLUTION_RING_WIDTH",
    "ANSWER_RING_TASKS",
    "degree_badge_center",
    "solution_ring_labels",
    "use_answer_ring_masks",
    "render_unified_graph",
]
