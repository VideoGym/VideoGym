"""Branching domino-network topple: shared physics-family implementation.

Top-down world: numbered upright dominoes stand on a board, joined by dashed
topple links.  Every domino prints its height ``H`` and every link prints its
gap ``G`` in the same length unit, and the drawn base-to-base distance of a
link equals ``G`` at the sample's uniform pixel scale.  Domino 1 is pushed;
a toppling domino lies flat as an elongated rectangle reaching exactly ``H``
along its topple direction, so reach versus shortfall is directly visible.
A linked neighbor topples exactly when the faller's ``H`` is at least the
link's ``G``; both children of one faller start together.
"""

from __future__ import annotations

from data.common.assets import render_font

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageFont

from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract
from data.common.scalar_lists import (
    NONNEGATIVE_INTEGER_LIST_ANSWER_EXAMPLE,
    NONNEGATIVE_INTEGER_LIST_ANSWER_SCHEMA,
    format_nonnegative_integer_list_payload,
)
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.physics.core.physical_scene_common import PhysicalScenePattern
from data.physics.core.source_replay import carrier_polyline_samples


BG = (246, 244, 239)
CARD = (251, 250, 247)
CARD_EDGE = (218, 216, 210)
INK = (42, 46, 54)
BLUE = (72, 116, 176)
LINK = (168, 172, 178)
_WIDTH_UNITS = 26.0        # domino width across the topple direction
_THICK_UNITS = 13.0        # upright footprint thickness along the topple direction
_MIN_HEIGHT, _MAX_HEIGHT = 42, 108
_MIN_GAP, _MAX_GAP = 50, 100
_MARGIN_UNITS = 8          # margin for successful links (H - G >= 8)
_FAIL_MARGIN_UNITS = 12    # margin for failed links (G - H >= 12): the
                           # drawn shortfall stays visible at every layout scale
_MAX_OVERSHOOT = 26        # cap H - G so a landing tip stays near its target
_CANVAS_LOW, _CANVAS_HIGH = 30.0, 482.0


def _font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def _unit(vector: Tuple[float, float]) -> Tuple[float, float]:
    length = math.hypot(*vector)
    return vector[0] / length, vector[1] / length


def _rect_corners(
    base: Tuple[float, float],
    direction: Tuple[float, float],
    back: float,
    front: float,
    half_width: float,
) -> List[Tuple[float, float]]:
    """Corners of a rectangle spanning [-back, front] along ``direction``."""

    dx, dy = direction
    px, py = -dy, dx
    bx, by = base
    return [
        (bx - dx * back + px * half_width, by - dy * back + py * half_width),
        (bx + dx * front + px * half_width, by + dy * front + py * half_width),
        (bx + dx * front - px * half_width, by + dy * front - py * half_width),
        (bx - dx * back - px * half_width, by - dy * back - py * half_width),
    ]


def _polygons_clear(first: Sequence[Tuple[float, float]], second: Sequence[Tuple[float, float]], gap: float) -> bool:
    """Separating-axis test: True when the convex polygons stay ``gap`` apart."""

    for polygon in (first, second):
        count = len(polygon)
        for index in range(count):
            ax, ay = polygon[index]
            bx, by = polygon[(index + 1) % count]
            nx, ny = -(by - ay), bx - ax
            norm = math.hypot(nx, ny)
            if norm <= 1e-9:
                continue
            nx, ny = nx / norm, ny / norm
            first_dots = [nx * x + ny * y for x, y in first]
            second_dots = [nx * x + ny * y for x, y in second]
            if min(second_dots) >= max(first_dots) + gap or min(first_dots) >= max(second_dots) + gap:
                return True
    return False


def _box_of(points: Sequence[Tuple[float, float]], pad: float = 0.0) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def simulate_topple_waves(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> tuple[List[List[int]], List[Tuple[int, int]], set[int]]:
    """Replay the cascade from visible facts only.

    Returns ``(waves, failed_links, fallen)`` where waves hold display ids in
    landing order, and failed links are ``(faller, blocked_child)`` pairs whose
    comparison failed while the faller actually fell.
    """

    heights = {int(node["id"]): float(node["height"]) for node in nodes}
    children: Dict[int, List[Tuple[int, float]]] = {}
    for edge in edges:
        children.setdefault(int(edge["from"]), []).append((int(edge["to"]), float(edge["gap"])))
    if 1 not in heights:
        raise ValueError("domino network requires pushed domino 1")
    waves: List[List[int]] = [[1]]
    fallen = {1}
    failed: List[Tuple[int, int]] = []
    frontier = [1]
    while frontier:
        next_wave: List[int] = []
        for faller in frontier:
            for child, gap in sorted(children.get(faller, ())):
                if child in fallen:
                    raise ValueError("domino network must be a tree")
                if heights[faller] >= gap:
                    next_wave.append(child)
                    fallen.add(child)
                else:
                    failed.append((faller, child))
        if not next_wave:
            break
        waves.append(sorted(next_wave))
        frontier = next_wave
    return waves, failed, fallen


class DominoReachCascadePattern(PhysicalScenePattern):
    task_name = "domino_reach_cascade"
    task_mechanic = "branching_height_gap_topple_network"
    answer_overlay_mode = "none"
    verifier_profile = "domino_network_visible_geometry_multimodal_hard_gate_v1"
    video_verifier_profile = "domino_network_full_video_semantic_single_decode"

    # ------------------------------------------------------------------ sampling

    def _tree_shape(self, count: int) -> Dict[int, List[int]] | None:
        parents: Dict[int, int] = {}
        child_count = {0: 0}
        for node in range(1, count):
            open_nodes = [index for index, taken in child_count.items() if taken < 2]
            parent = self.rng.choice(open_nodes)
            parents[node] = parent
            child_count[parent] = child_count[parent] + 1
            child_count[node] = 0
        children: Dict[int, List[int]] = {index: [] for index in range(count)}
        for node, parent in parents.items():
            children[parent].append(node)
        if not any(len(kids) == 2 for kids in children.values()):
            return None
        depth = {0: 0}
        for node in range(1, count):
            depth[node] = depth[parents[node]] + 1
        if max(depth.values()) > 4:
            return None
        return children

    def _sample_task(self) -> Dict[str, Any]:
        counts = {"min": (5, 5), "typical": (6, 7), "max": (8, 8)}.get(self.difficulty or "", (5, 8))
        # Fix the domino count first so acceptance speed cannot skew the
        # sampled size distribution toward small scenes.
        count = self.rng.randint(*counts)
        for attempt in range(9000):
            if attempt == 6000:
                count = self.rng.randint(counts[0], min(counts[1], 6))
            children = self._tree_shape(count)
            if children is None:
                continue
            task = self._try_layout(count, children)
            if task is not None:
                return task
        raise RuntimeError("unable to sample a valid domino network layout")

    def _try_layout(self, count: int, children: Dict[int, List[int]]) -> Dict[str, Any] | None:
        rng = self.rng
        heights = {index: rng.randint(_MIN_HEIGHT, _MAX_HEIGHT) for index in range(count)}
        gaps: Dict[Tuple[int, int], int] = {}
        for parent, kids in children.items():
            height = heights[parent]
            # every printed comparison keeps a clear numeric margin, and
            # a successful landing never overshoots absurdly far, so gaps are
            # sampled directly from the feasible success/fail ranges.
            success_low, success_high = max(_MIN_GAP, height - _MAX_OVERSHOOT), min(_MAX_GAP, height - _MARGIN_UNITS)
            fail_low, fail_high = max(_MIN_GAP, height + _FAIL_MARGIN_UNITS), _MAX_GAP
            for kid in kids:
                feasible = []
                if success_low <= success_high:
                    feasible.append("success")
                if fail_low <= fail_high:
                    feasible.append("fail")
                if not feasible:
                    return None
                intent = (
                    "success"
                    if ("success" in feasible and ("fail" not in feasible or rng.random() < 0.80))
                    else "fail"
                )
                if intent == "success":
                    gaps[(parent, kid)] = rng.randint(success_low, success_high)
                else:
                    gaps[(parent, kid)] = rng.randint(fail_low, fail_high)

        display = self._display_ids(count)
        sim_nodes = [
            {"id": display[index], "height": heights[index]} for index in range(count)
        ]
        sim_edges = [
            {"from": display[parent], "to": display[kid], "gap": gap}
            for (parent, kid), gap in gaps.items()
        ]
        waves, failed, fallen = simulate_topple_waves(sim_nodes, sim_edges)
        if not (3 <= len(fallen) <= count - 1):
            return None
        if not failed:
            return None
        if len(waves) > 5:
            return None
        # Larger networks must also cascade deeper, so size raises real
        # dependency length instead of only adding blocked decoration.
        if count >= 6 and len(waves) < 3:
            return None
        fallen_internal = {index for index in range(count) if display[index] in fallen}
        if not any(
            len(children[index]) == 2 and index in fallen_internal for index in range(count)
        ):
            return None

        # Unit-space geometry: place children inside the parent's fall corridor
        # (lateral offset at most 21 units, so the landing tip visibly covers
        # each struck child's base).
        positions: Dict[int, Tuple[float, float]] = {0: (0.0, 0.0)}
        directions: Dict[int, Tuple[float, float]] = {}
        root_angle = rng.uniform(0.0, 2.0 * math.pi)
        order = [0]
        angles: Dict[int, float] = {0: root_angle}
        while order:
            parent = order.pop(0)
            kids = children[parent]
            base_angle = angles[parent]
            if len(kids) == 1:
                gap = gaps[(parent, kids[0])]
                cap = min(0.26, math.asin(min(1.0, 21.0 / gap)))
                offsets = [rng.uniform(-cap, cap)]
            elif len(kids) == 2:
                caps = [
                    math.asin(min(1.0, 21.0 / gaps[(parent, kid)]))
                    for kid in kids
                ]
                magnitudes = [rng.uniform(0.78, 1.0) * cap for cap in caps]
                sign = rng.choice((1.0, -1.0))
                offsets = [sign * magnitudes[0], -sign * magnitudes[1]]
            else:
                offsets = []
            for kid, offset in zip(kids, offsets):
                gap = gaps[(parent, kid)]
                angle = base_angle + offset
                px, py = positions[parent]
                positions[kid] = (px + gap * math.cos(angle), py + gap * math.sin(angle))
                angles[kid] = angle
                order.append(kid)
        for index in range(count):
            directions[index] = (math.cos(angles[index]), math.sin(angles[index]))

        # Terminal unit-space footprints for overlap checks and canvas fit.
        footprints: Dict[int, List[Tuple[float, float]]] = {}
        for index in range(count):
            length = heights[index] if index in fallen_internal else _THICK_UNITS / 2.0
            footprints[index] = _rect_corners(
                positions[index], directions[index], _THICK_UNITS / 2.0, length, _WIDTH_UNITS / 2.0
            )
        linked = set(gaps) | {(kid, parent) for parent, kid in gaps}
        parent_of = {kid: parent for parent, kid in gaps}
        for first in range(count):
            for second in range(first + 1, count):
                if (first, second) in linked:
                    continue
                # Siblings legitimately stand side by side inside their
                # parent's strike corridor; they only need visual separation.
                siblings = parent_of.get(first) is not None and parent_of.get(first) == parent_of.get(second)
                clearance = 4.0 if siblings else 14.0
                if not _polygons_clear(footprints[first], footprints[second], clearance):
                    return None

        boxes = [_box_of(corners) for corners in footprints.values()]
        min_x = min(box[0] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_x = max(box[2] for box in boxes)
        max_y = max(box[3] for box in boxes)
        span = max(max_x - min_x, max_y - min_y, 1.0)
        scale = min(1.18, 380.0 / span)
        if scale < 0.52:
            return None
        offset_x = 256.0 - scale * (min_x + max_x) / 2.0
        offset_y = 256.0 - scale * (min_y + max_y) / 2.0

        nodes: List[Dict[str, Any]] = []
        for index in range(count):
            base = positions[index]
            nodes.append(
                {
                    "id": display[index],
                    "base": [
                        round(base[0] * scale + offset_x, 2),
                        round(base[1] * scale + offset_y, 2),
                    ],
                    "dir": [round(directions[index][0], 5), round(directions[index][1], 5)],
                    "height": heights[index],
                }
            )
        edges = [
            {"from": display[parent], "to": display[kid], "gap": gap}
            for (parent, kid), gap in sorted(gaps.items())
        ]
        task = {"nodes": nodes, "edges": edges, "px_per_unit": round(scale, 5)}
        if not self._layout_fits(task):
            return None
        return task

    def _display_ids(self, count: int) -> Dict[int, int]:
        rest = list(range(2, count + 1))
        self.rng.shuffle(rest)
        display = {0: 1}
        for index, value in enumerate(rest, start=1):
            display[index] = value
        return display

    # ------------------------------------------------------- geometry derivation

    @staticmethod
    def _scene(task: Mapping[str, Any]) -> Dict[str, Any]:
        """Derive all deterministic drawing geometry from the visible facts.

        Badge anchors (number discs, bare ``H``/``G`` labels) are placed by a
        deterministic greedy scan over fixed candidate rings, so the sampler,
        renderer, and verifier replay always agree without storing render
        decisions in the semantic source.
        """

        scale = float(task["px_per_unit"])
        nodes = {int(node["id"]): node for node in task["nodes"]}
        waves, failed, fallen = simulate_topple_waves(task["nodes"], task["edges"])
        geometry: Dict[str, Any] = {
            "scale": scale,
            "nodes": {},
            "edges": [],
            "waves": waves,
            "failed": failed,
            "fallen": fallen,
            "badges_ok": True,
        }
        for node_id, node in nodes.items():
            base = (float(node["base"][0]), float(node["base"][1]))
            direction = _unit((float(node["dir"][0]), float(node["dir"][1])))
            geometry["nodes"][node_id] = {
                "base": base,
                "dir": direction,
                "height_px": float(node["height"]) * scale,
                "half_width_px": _WIDTH_UNITS * scale / 2.0,
                "half_thick_px": _THICK_UNITS * scale / 2.0,
                "fallen": node_id in fallen,
            }
        root = geometry["nodes"][1]
        tail = 34.0 + root["half_thick_px"]
        geometry["push_arrow"] = {
            "tip": (
                root["base"][0] - root["dir"][0] * (root["half_thick_px"] + 7.0),
                root["base"][1] - root["dir"][1] * (root["half_thick_px"] + 7.0),
            ),
            "tail": (
                root["base"][0] - root["dir"][0] * tail,
                root["base"][1] - root["dir"][1] * tail,
            ),
        }

        # Obstacles every badge must clear: terminal footprints, link lines,
        # the push arrow, and previously placed badges.
        obstacles: List[List[Tuple[float, float]]] = []
        for node_id in sorted(geometry["nodes"]):
            node = geometry["nodes"][node_id]
            length = node["height_px"] if node["fallen"] else node["half_thick_px"]
            obstacles.append(
                _rect_corners(node["base"], node["dir"], node["half_thick_px"], length, node["half_width_px"])
            )
        for edge in task["edges"]:
            source = geometry["nodes"][int(edge["from"])]["base"]
            target = geometry["nodes"][int(edge["to"])]["base"]
            axis = _unit((target[0] - source[0], target[1] - source[1]))
            mid = ((source[0] + target[0]) / 2.0, (source[1] + target[1]) / 2.0)
            half = math.hypot(target[0] - source[0], target[1] - source[1]) / 2.0
            obstacles.append(_rect_corners(mid, axis, half, half, 2.0))
        arrow = geometry["push_arrow"]
        arrow_axis = _unit((arrow["tip"][0] - arrow["tail"][0], arrow["tip"][1] - arrow["tail"][1]))
        arrow_mid = ((arrow["tip"][0] + arrow["tail"][0]) / 2.0, (arrow["tip"][1] + arrow["tail"][1]) / 2.0)
        arrow_half = math.hypot(arrow["tip"][0] - arrow["tail"][0], arrow["tip"][1] - arrow["tail"][1]) / 2.0
        obstacles.append(_rect_corners(arrow_mid, arrow_axis, arrow_half, arrow_half, 7.0))

        def _badge_rect(center: Tuple[float, float], half_w: float, half_h: float) -> List[Tuple[float, float]]:
            return [
                (center[0] - half_w, center[1] - half_h),
                (center[0] + half_w, center[1] - half_h),
                (center[0] + half_w, center[1] + half_h),
                (center[0] - half_w, center[1] + half_h),
            ]

        def _place(anchor: Tuple[float, float], radii: Sequence[float], half_w: float, half_h: float) -> Tuple[float, float] | None:
            for radius in radii:
                for step in range(8):
                    angle = step * math.pi / 4.0
                    center = (anchor[0] + radius * math.cos(angle), anchor[1] + radius * math.sin(angle))
                    rect = _badge_rect(center, half_w, half_h)
                    if not all(
                        _CANVAS_LOW <= x <= _CANVAS_HIGH and _CANVAS_LOW <= y <= _CANVAS_HIGH
                        for x, y in rect
                    ):
                        continue
                    if all(_polygons_clear(rect, obstacle, 2.0) for obstacle in obstacles):
                        obstacles.append(rect)
                        return center
            return None

        for node_id in sorted(geometry["nodes"]):
            node = geometry["nodes"][node_id]
            ring = node["half_width_px"] + 16.0
            disc = _place(node["base"], (ring, ring + 13.0, ring + 26.0), 12.0, 12.0)
            if disc is None:
                geometry["badges_ok"] = False
                disc = (node["base"][0], node["base"][1] - ring)
            label = _place(disc, (25.0, 32.0), 21.0, 8.0)
            if label is None:
                geometry["badges_ok"] = False
                label = (disc[0], disc[1] + 24.0)
            node["disc_center"] = disc
            node["height_label"] = label
        for edge in sorted(task["edges"], key=lambda item: (int(item["from"]), int(item["to"]))):
            source = geometry["nodes"][int(edge["from"])]["base"]
            target = geometry["nodes"][int(edge["to"])]["base"]
            mid = ((source[0] + target[0]) / 2.0, (source[1] + target[1]) / 2.0)
            label = _place(mid, (20.0, 30.0), 20.0, 8.0)
            if label is None:
                geometry["badges_ok"] = False
                label = (mid[0], mid[1] - 20.0)
            geometry["edges"].append(
                {
                    "from": int(edge["from"]),
                    "to": int(edge["to"]),
                    "gap": float(edge["gap"]),
                    "gap_label": label,
                }
            )
        return geometry

    def _layout_fits(self, task: Mapping[str, Any]) -> bool:
        geometry = self._scene(task)
        if not geometry["badges_ok"]:
            return False
        points: List[Tuple[float, float]] = []
        for node in geometry["nodes"].values():
            length = node["height_px"] if node["fallen"] else node["half_thick_px"]
            points.extend(
                _rect_corners(node["base"], node["dir"], node["half_thick_px"], length, node["half_width_px"])
            )
        points.extend([geometry["push_arrow"]["tail"], geometry["push_arrow"]["tip"]])
        if not all(
            _CANVAS_LOW <= x <= _CANVAS_HIGH and _CANVAS_LOW <= y <= _CANVAS_HIGH
            for x, y in points
        ):
            return False
        # label binding: every G label must sit clearly nearest to its own
        # link and every disc/H label nearest to its own domino base, so no
        # printed value can be read against the wrong object.
        def _to_segment(point, start, end):
            vx, vy = end[0] - start[0], end[1] - start[1]
            length_sq = vx * vx + vy * vy
            t = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / max(length_sq, 1e-9)))
            proj = (start[0] + vx * t, start[1] + vy * t)
            return math.hypot(point[0] - proj[0], point[1] - proj[1])

        segments = {
            (edge["from"], edge["to"]): (
                geometry["nodes"][edge["from"]]["base"],
                geometry["nodes"][edge["to"]]["base"],
            )
            for edge in geometry["edges"]
        }
        for edge in geometry["edges"]:
            own = _to_segment(edge["gap_label"], *segments[(edge["from"], edge["to"])])
            for key, segment in segments.items():
                if key == (edge["from"], edge["to"]):
                    continue
                if _to_segment(edge["gap_label"], *segment) < own + 8.0:
                    return False
        for node_id, node in geometry["nodes"].items():
            for anchor in (node["disc_center"], node["height_label"]):
                own = math.hypot(anchor[0] - node["base"][0], anchor[1] - node["base"][1])
                for other_id, other in geometry["nodes"].items():
                    if other_id == node_id:
                        continue
                    if math.hypot(anchor[0] - other["base"][0], anchor[1] - other["base"][1]) < own + 8.0:
                        return False
        return True

    @classmethod
    def progress_carriers(
        cls, task: Mapping[str, Any]
    ) -> List[Tuple[str, List[Tuple[float, float]], Tuple[int, int, int], Tuple[int, int, int] | None, int]]:
        """Per domino, the ground it has covered - which only ever grows.

        A toppling domino is drawn as its own rectangle reaching ``front`` along
        its topple direction, and ``front`` runs from the upright footprint out
        to the domino's full height without ever coming back: a fallen domino
        never stands up again.  Each carrier samples that domino's own corridor,
        from just past its upright footprint out to its stated height, so the
        reading is how far it has reached.  A domino the cascade never touches
        declares the same corridor and reaches nothing, which makes its carrier a
        standing-still gate rather than a reaching one.

        The corridor is board first and domino body afterwards, and the body is a
        solid fill, so the carrier declares both colours and a point counts as
        covered once it has actually taken the body's blue.  Bare departure from
        the board would instead count what merely brushes the corridor: a dashed
        link's tail sits at departure 19 of the body's 179, and the antialiased
        boundary of an already-settled neighbour at 15.  Those slivers are not
        constant - the downsampling halo of the approaching body lifts them back
        under a bare threshold for exactly one frame - so read that way the ground
        appears to give a point back and take it again, which is travel this
        cascade never makes.
        """

        geometry = cls._scene(task)
        carriers: List[
            Tuple[str, List[Tuple[float, float]], Tuple[int, int, int], Tuple[int, int, int] | None, int]
        ] = []
        for node_id in sorted(geometry["nodes"]):
            node = geometry["nodes"][node_id]
            base, direction = node["base"], node["dir"]
            near = float(node["half_thick_px"]) + 2.0
            far = max(near + 2.0, float(node["height_px"]))
            carriers.append(
                (
                    f"domino_{node_id}",
                    carrier_polyline_samples(
                        [
                            (base[0] + direction[0] * near, base[1] + direction[1] * near),
                            (base[0] + direction[0] * far, base[1] + direction[1] * far),
                        ]
                    ),
                    CARD,
                    BLUE,
                    1,
                )
            )
        return carriers

    @classmethod
    def terminal_member_boxes(cls, task: Mapping[str, Any]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        """Per-domino evidence boxes over each terminal footprint."""

        geometry = cls._scene(task)
        boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for node_id in sorted(geometry["nodes"]):
            node = geometry["nodes"][node_id]
            length = max(node["height_px"], node["half_thick_px"] + 2.0)
            corners = _rect_corners(
                node["base"], node["dir"], node["half_thick_px"], length, node["half_width_px"]
            )
            boxes.append((f"domino_{node_id}", _box_of(corners, pad=5.0)))
        return boxes

    # ------------------------------------------------------------------ contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single image is a top-down view of numbered upright dominoes standing on a board. "
                "Each blue rectangle is one upright domino seen from above; its white disc holds its number and the pill beside it holds its height as `H` with a value. "
                "Thin dashed lines are the topple links between neighboring dominoes, and the pill on each link holds that link's gap as `G` with a value. "
                "The blue arrow shows the push applied to domino 1."
            ),
            symbol_legend=(
                "All H and G values use one shared length unit, and each link's drawn base-to-base distance matches its printed G, so every printed pair is directly comparable.",
                "A number disc only names its domino and carries no order information.",
            ),
            rules=(
                "Domino 1 is pushed along the arrow and topples.",
                "A toppling domino pivots at its base point, and its tip lands exactly its height H away from that base along its topple direction.",
                "It knocks over a linked neighbor exactly when its H value is at least the G value printed on their link; if H is smaller than G, the tip settles short and that neighbor never falls.",
                "Every struck domino topples onward away from the domino that hit it and applies the same comparison along each of its own outgoing links; two neighbors struck by the same falling domino start toppling at the same moment.",
                "A domino that is never struck stays upright, and the cascade ends when no newly struck domino remains.",
            ),
            modality_outputs={
                "ti2v": (
                    "Hold the input, then rotate domino 1 flat along the arrow. For each successful link, the falling tip visibly reaches the next upright domino before that neighbor starts toppling, and two neighbors of one faller start together. "
                    "At a failed link the tip settles short and that branch stops for good. Continue wave by wave until no struck domino remains, then hold the settled scene."
                ),
                "solution_image": (
                    "Render the settled top-down scene: every fallen domino lies flat as an elongated blue rectangle reaching exactly its height along its topple direction, and every unstruck domino remains a short upright rectangle. Keep all discs, pills, links, and the arrow unchanged."
                ),
            },
            preservation_constraints=(
                "Keep the board, number discs, H and G pills, dashed links, push arrow, and palette fixed; only toppling dominoes change shape.",
            ),
            transition_constraints=(
                "Topple continuously about fixed base points in causal contact order; never skip a landing, revive a blocked branch, or move an upright domino.",
            ),
            ti2v_continuity={
                "moving_nouns": "the toppling dominoes",
                "static_nouns": "the number discs, H and G pills, dashed links, push arrow, board, and camera",
                "hold_noun": "the settled cascade with every blocked domino upright",
            },
            answer=AnswerContract(
                kind="nonnegative_integer_list",
                format_template=NONNEGATIVE_INTEGER_LIST_ANSWER_SCHEMA,
                parser_profile="strict_ascending_domino_list",
                verifier_profile=self.verifier_profile,
                format_example=NONNEGATIVE_INTEGER_LIST_ANSWER_EXAMPLE,
            ),
            render_profile="top_down_branching_domino_network_v1",
        )

    def _question(self, task: Mapping[str, Any]) -> str:
        return "Which dominoes have fallen when the cascade stops?"

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        _waves, _failed, fallen = simulate_topple_waves(task["nodes"], task["edges"])
        return format_nonnegative_integer_list_payload(sorted(fallen))

    # ------------------------------------------------------------------- states

    def _wave_text(self, task: Mapping[str, Any], wave_index: int) -> str:
        geometry = self._scene(task)
        waves = geometry["waves"]
        heights = {int(node["id"]): int(node["height"]) for node in task["nodes"]}
        gaps = {(int(edge["from"]), int(edge["to"])): int(edge["gap"]) for edge in task["edges"]}
        children: Dict[int, List[int]] = {}
        for source, target in gaps:
            children.setdefault(source, []).append(target)
        sentences: List[str] = []
        for faller in waves[wave_index]:
            height = heights[faller]
            if wave_index == 0:
                sentences.append(
                    f"Domino 1 takes the push and topples flat; its height is H {height}."
                )
            else:
                sentences.append(
                    f"Domino {faller} was struck, so it topples flat with H {height}."
                )
            kids = sorted(children.get(faller, ()))
            if not kids:
                sentences.append(f"Domino {faller} has no outgoing link, so this branch simply ends.")
                continue
            for kid in kids:
                gap = gaps[(faller, kid)]
                if height >= gap:
                    sentences.append(
                        f"Its link to domino {kid} has G {gap}, and {height} >= {gap}, so the tip reaches domino {kid} and knocks it over."
                    )
                else:
                    sentences.append(
                        f"Its link to domino {kid} has G {gap}, but {height} < {gap}, so the tip settles short and domino {kid} stays upright."
                    )
        return " ".join(sentences)

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[float, str]]:
        geometry = self._scene(task)
        waves = geometry["waves"]
        total = len(waves)
        states: List[tuple[float, str]] = []
        for index in range(total):
            text = self._wave_text(task, index)
            if index == total - 1:
                fallen = ", ".join(str(value) for value in sorted(geometry["fallen"]))
                text += (
                    f" No newly struck domino remains, so the cascade is over and the fallen dominoes in increasing order are {fallen}."
                )
            states.append(((index + 1) / total, text))
        return states

    def _motion_frame_count(self, task: Mapping[str, Any]) -> int:
        waves, _failed, _fallen = simulate_topple_waves(task["nodes"], task["edges"])
        frames = max(23, len(waves) * 10)
        while frames % 4 != 3:
            frames += 1
        return frames

    # ---------------------------------------------------------------- structured

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        geometry = self._scene(task)
        waves = geometry["waves"]
        comparisons = []
        heights = {int(node["id"]): int(node["height"]) for node in task["nodes"]}
        for edge in task["edges"]:
            source = int(edge["from"])
            comparisons.append(
                {
                    "from_domino": source,
                    "to_domino": int(edge["to"]),
                    "height": heights[source],
                    "gap": int(edge["gap"]),
                    "reaches": heights[source] >= int(edge["gap"]),
                    "evaluated": source in geometry["fallen"],
                }
            )
        return {
            "semantic_state": {
                "nodes": [dict(node) for node in task["nodes"]],
                "edges": [dict(edge) for edge in task["edges"]],
                "px_per_unit": float(task["px_per_unit"]),
            },
            "answer_schema": {
                "schema_version": "domino_network_fallen_list_v1",
                "kind": "ascending_domino_id_list",
                "parser_profile": "strict_ascending_domino_list",
                "format_template": NONNEGATIVE_INTEGER_LIST_ANSWER_SCHEMA,
                "format_example": NONNEGATIVE_INTEGER_LIST_ANSWER_EXAMPLE,
                "judged_artifacts": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "verifier_contract": {
                "profile": self.verifier_profile,
                "video_profile": self.video_verifier_profile,
                "semantic_source": "visible node layout, printed H and G values, dashed links, and the push on domino 1",
                "strict_binary_correctness": True,
                "strict_text_parser": r"^Answer: \[([1-9][0-9]*(?:, [1-9][0-9]*)*)\]\.$",
                "untrusted_redundant_fields": [
                    "waves",
                    "fallen",
                    "failed_links",
                    "state_trace",
                    "semantic_state_trace",
                ],
            },
            "multiple_solution_policy": "deterministic cascade; exactly one fallen set",
            "waves": waves,
            "fallen": sorted(geometry["fallen"]),
            "failed_links": [list(pair) for pair in geometry["failed"]],
            "semantic_state_trace": comparisons,
        }

    # ------------------------------------------------------------------- render

    def _render_progress(self, task: Mapping[str, Any], progress: float, *, mark_answer: bool) -> Image.Image:
        del mark_answer  # terminal state is pure settled geometry
        supersample = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(512, 512, supersample, background=BG)
        geometry = self._scene(task)
        waves = geometry["waves"]
        total_waves = len(waves)
        elapsed = max(0.0, min(1.0, float(progress))) * total_waves

        draw.rounded_rectangle((16, 16, 496, 496), radius=24, fill=CARD, outline=CARD_EDGE, width=2)

        # Dashed links under everything else.
        for edge in geometry["edges"]:
            source = geometry["nodes"][edge["from"]]["base"]
            target = geometry["nodes"][edge["to"]]["base"]
            length = math.hypot(target[0] - source[0], target[1] - source[1])
            axis = _unit((target[0] - source[0], target[1] - source[1]))
            dash, hole, cursor = 7.0, 6.0, 0.0
            while cursor < length:
                end = min(length, cursor + dash)
                draw.line(
                    (
                        source[0] + axis[0] * cursor,
                        source[1] + axis[1] * cursor,
                        source[0] + axis[0] * end,
                        source[1] + axis[1] * end,
                    ),
                    fill=LINK,
                    width=2,
                )
                cursor = end + hole
        # Bare G labels beside each link.
        gap_font = _font(12 * supersample)
        for edge in geometry["edges"]:
            label = edge["gap_label"]
            draw.text((label[0], label[1]), f"G {int(edge['gap'])}", anchor="mm", fill=INK, font=gap_font)

        # Wave start times: wave k (0-based) starts at k, lands at k+1.
        wave_of: Dict[int, int] = {}
        for index, wave in enumerate(waves):
            for node_id in wave:
                wave_of[node_id] = index
        for node_id in sorted(geometry["nodes"]):
            node = geometry["nodes"][node_id]
            if node["fallen"]:
                tau = max(0.0, min(1.0, elapsed - wave_of[node_id]))
            else:
                tau = 0.0
            reach = node["height_px"] * tau * tau
            front = max(node["half_thick_px"], reach)
            corners = _rect_corners(node["base"], node["dir"], node["half_thick_px"], front, node["half_width_px"])
            draw.polygon(corners, fill=BLUE, outline=INK, width=1)

        # Number discs and bare H labels stay in fixed slots.
        disc_font = _font(14 * supersample)
        label_font = _font(12 * supersample)
        heights = {int(item["id"]): int(item["height"]) for item in task["nodes"]}
        for node_id in sorted(geometry["nodes"]):
            node = geometry["nodes"][node_id]
            disc = node["disc_center"]
            draw.ellipse(
                (disc[0] - 12, disc[1] - 12, disc[0] + 12, disc[1] + 12),
                fill=(255, 255, 253),
                outline=(181, 185, 184),
                width=2,
            )
            draw.text((disc[0], disc[1]), str(node_id), anchor="mm", fill=INK, font=disc_font)
            label = node["height_label"]
            draw.text((label[0], label[1]), f"H {heights[node_id]}", anchor="mm", fill=INK, font=label_font)

        arrow = geometry["push_arrow"]
        tail, tip = arrow["tail"], arrow["tip"]
        draw.line((*tail, *tip), fill=BLUE, width=4)
        axis = _unit((tip[0] - tail[0], tip[1] - tail[1]))
        for delta in (math.radians(150), math.radians(-150)):
            angle = math.atan2(axis[1], axis[0]) + delta
            draw.line(
                (tip[0], tip[1], tip[0] + 10 * math.cos(angle), tip[1] + 10 * math.sin(angle)),
                fill=BLUE,
                width=4,
            )
        return finish_scaled_canvas(image, (512, 512))


__all__ = ["DominoReachCascadePattern", "simulate_topple_waves"]
