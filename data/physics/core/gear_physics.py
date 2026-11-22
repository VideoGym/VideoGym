"""Gear-train direction propagation: shared physics-family implementation.

A numbered train of gears is driven at gear 1, whose turning direction is
fixed by a blue curved arrow.  Two meshing gears (rims touching) counter-
rotate; two belt-linked gears (joined by a straight open belt) co-rotate.
Path samples ask for the last gear's direction; ring samples close a loop of
mutually meshing gears, and an odd loop makes the whole train jam so that no
gear can turn at all.  Directions are always read as seen by the viewer.
"""

from __future__ import annotations

from data.common.assets import render_font

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageFont

from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.physics.core.physical_scene_common import PhysicalScenePattern


BG = (245, 247, 249)
CARD = (252, 253, 254)
CARD_EDGE = (216, 221, 226)
INK = (40, 47, 56)
GEAR_FILL = (226, 233, 240)
GEAR_SECTOR = (198, 212, 226)
GEAR_RIM = (96, 112, 128)
HUB = (150, 162, 174)
SPOKE = (96, 112, 128)
DRIVER_ARROW = (57, 115, 182)
DIR_ARROW = (222, 133, 73)
CLASH = (198, 64, 58)
BELT = (128, 138, 148)

_MIN_R, _MAX_R = 27.0, 43.0
_MESH_TOL = 0.5          # tangency tolerance when deriving mesh adjacency
_CLEAR_GAP = 16.0        # non-linked rims stay at least this far apart
_CANVAS_LOW, _CANVAS_HIGH = 34.0, 478.0
# Total driver rotation over the video.  The rim carries the phase in 3px tooth
# dashes, so a sweep that moves the rim further than a dash is wide between
# frames leaves adjacent frames sharing no ink, and the whole-video decoder's
# appearance distance stops carrying any order at all.  Worst
# adjacent-to-decorrelated source-state disagreement over five seeds, measured
# with the decoder's own probe, against the 0.45 it needs to order frames at
# full resolution: 0.76 at 288 degrees, 0.61 at 180, 0.46 at 120, 0.44 at 108,
# 0.42 at 100.  108 clears it by 0.01 and 100 by 0.035, so 100 is the setting
# with real margin - and it still turns the driver more than a quarter turn,
# which the 70-degree shaded sector and the bold spoke make plain to a viewer.
_DRIVER_SWEEP_DEG = 100.0
def _font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def derive_mesh_edges(gears: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    """Meshing pairs derived from visible tangency alone."""

    edges: List[Tuple[int, int]] = []
    for first_index in range(len(gears)):
        for second_index in range(first_index + 1, len(gears)):
            first, second = gears[first_index], gears[second_index]
            distance = math.hypot(
                float(first["center"][0]) - float(second["center"][0]),
                float(first["center"][1]) - float(second["center"][1]),
            )
            if abs(distance - (float(first["radius"]) + float(second["radius"]))) <= _MESH_TOL:
                edges.append((int(first["id"]), int(second["id"])))
    return edges


def solve_directions(
    gears: Sequence[Mapping[str, Any]],
    belts: Sequence[Sequence[int]],
    driver_direction: str,
) -> tuple[Dict[int, int], List[Tuple[int, int]]] | None:
    """Propagate turning signs; ``None`` plus the conflicting mesh set on jam.

    Returns ``(signs, [])`` when consistent (+1 = CW, -1 = CCW) or
    ``(None_marker)``... on conflict returns ``({}, conflict_edges)`` where
    conflict edges are every mesh contact inside a conflicting cycle.
    """

    sign = {"CW": 1, "CCW": -1}[driver_direction]
    mesh = derive_mesh_edges(gears)
    neighbors: Dict[int, List[Tuple[int, int]]] = {}
    for a, b in mesh:
        neighbors.setdefault(a, []).append((b, -1))
        neighbors.setdefault(b, []).append((a, -1))
    for a, b in belts:
        neighbors.setdefault(int(a), []).append((int(b), 1))
        neighbors.setdefault(int(b), []).append((int(a), 1))
    signs: Dict[int, int] = {1: sign}
    queue = [1]
    conflict = False
    while queue:
        current = queue.pop(0)
        for other, factor in sorted(neighbors.get(current, ())):
            required = signs[current] * factor
            if other in signs:
                if signs[other] != required:
                    conflict = True
            else:
                signs[other] = required
                queue.append(other)
    if len(signs) != len(gears):
        raise ValueError("gear train is not connected")
    if not conflict:
        return signs, []
    # Every mesh contact on a conflicting cycle: for these compact trains the
    # conflicting cycle is the set of edges whose endpoints both lie on some
    # cycle; report every mesh edge that belongs to any cycle (bridge test).
    conflict_edges = [edge for edge in mesh if not _is_bridge(edge, mesh, belts, len(gears))]
    if not conflict_edges:
        raise ValueError("jam without a visible conflicting loop")
    return {}, conflict_edges


def _is_bridge(edge: Tuple[int, int], mesh: Sequence[Tuple[int, int]], belts: Sequence[Sequence[int]], count: int) -> bool:
    links = [tuple(sorted(item)) for item in [*mesh, *[(int(a), int(b)) for a, b in belts]]]
    target = tuple(sorted(edge))
    remaining = [item for item in links if item != target]
    adjacency: Dict[int, set[int]] = {index: set() for index in range(1, count + 1)}
    for a, b in remaining:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen = {target[0]}
    stack = [target[0]]
    while stack:
        node = stack.pop()
        for other in adjacency[node]:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return target[1] not in seen


class GearTrainDirectionPattern(PhysicalScenePattern):
    task_name = "gear_train_direction"
    task_family = "physics"
    task_mechanic = "gear_belt_direction_propagation"
    answer_overlay_mode = "local_opacity"
    verifier_profile = "gear_train_visible_geometry_multimodal_hard_gate_v1"
    video_verifier_profile = "gear_train_full_video_semantic_single_decode"

    # ------------------------------------------------------------------ sampling

    def _sample_task(self) -> Dict[str, Any]:
        variants = {
            "min": ("path4",),
            "typical": ("path5", "ring4in"),
            "max": ("ring5in", "path6"),
        }.get(self.difficulty or "", ("path4", "path5", "path6", "ring3in", "ring4in", "ring5in"))
        variant = self.rng.choice(variants)
        for attempt in range(9000):
            if attempt == 6000:
                variant = "path4"
            task = self._try_scene(variant)
            if task is not None:
                return task
        raise RuntimeError("unable to sample a valid gear train")

    def _try_scene(self, variant: str) -> Dict[str, Any] | None:
        rng = self.rng
        if variant.startswith("path"):
            count = int(variant[4:])
            centers, radii = self._path_layout(count)
            ring_ids: List[int] = []
        else:
            ring_size = int(variant[4])
            entry = rng.randint(1, 2)
            layout = self._ring_layout(ring_size, entry)
            if layout is None:
                return None
            centers, radii, ring_ids = layout
            count = len(centers)
        if centers is None:
            return None
        # Canvas fit and non-linked clearance.
        for (cx, cy), radius in zip(centers, radii):
            if not (
                _CANVAS_LOW + radius + 14.0 <= cx <= _CANVAS_HIGH - radius - 14.0
                and _CANVAS_LOW + radius + 14.0 <= cy <= _CANVAS_HIGH - radius - 14.0
            ):
                return None
        gears = [
            {"id": index + 1, "center": [round(c[0], 2), round(c[1], 2)], "radius": round(r, 2)}
            for index, (c, r) in enumerate(zip(centers, radii))
        ]
        mesh = derive_mesh_edges(gears)
        expected_mesh = self._expected_links(variant, count, ring_ids)
        if sorted(tuple(sorted(edge)) for edge in mesh) != sorted(expected_mesh["mesh"]):
            return None
        for first_index in range(count):
            for second_index in range(first_index + 1, count):
                pair = (first_index + 1, second_index + 1)
                if tuple(sorted(pair)) in expected_mesh["mesh"]:
                    continue
                surface = (
                    math.hypot(
                        centers[first_index][0] - centers[second_index][0],
                        centers[first_index][1] - centers[second_index][1],
                    )
                    - radii[first_index]
                    - radii[second_index]
                )
                low = 4.0 if tuple(sorted(pair)) in expected_mesh["belt"] else _CLEAR_GAP
                if surface < low:
                    return None
        belts = [list(pair) for pair in expected_mesh["belt"]]
        driver_direction = rng.choice(("CW", "CCW"))
        task = {
            "gears": gears,
            "belts": belts,
            "driver_direction": driver_direction,
            "target": count,
        }
        signs, conflict = solve_directions(gears, belts, driver_direction)
        if variant in ("ring3in", "ring5in"):
            if not conflict:
                return None
        elif conflict:
            return None
        if not self._scene_fits(task):
            return None
        return task

    def _path_layout(self, count: int) -> tuple[List[Tuple[float, float]] | None, List[float]]:
        rng = self.rng
        radii = [rng.uniform(_MIN_R, _MAX_R) for _ in range(count)]
        # Two belt links at most, never adjacent to each other.
        belt_slots: List[int] = []
        for slot in range(count - 1):
            if len(belt_slots) < 2 and (not belt_slots or slot - belt_slots[-1] >= 2) and rng.random() < 0.34:
                belt_slots.append(slot)
        self._path_belt_slots = belt_slots
        centers: List[Tuple[float, float]] = [(0.0, 0.0)]
        heading = rng.uniform(0.0, 2.0 * math.pi)
        for index in range(1, count):
            heading += rng.uniform(-0.75, 0.75)
            spacing = radii[index - 1] + radii[index]
            if (index - 1) in belt_slots:
                spacing *= rng.uniform(1.75, 2.2)
            centers.append(
                (
                    centers[-1][0] + spacing * math.cos(heading),
                    centers[-1][1] + spacing * math.sin(heading),
                )
            )
        return self._fit(centers, radii), radii

    def _ring_layout(self, ring_size: int, entry: int) -> tuple[List[Tuple[float, float]], List[float], List[int]] | None:
        rng = self.rng
        ring_radius_gear = rng.uniform(_MIN_R + 3.0, _MAX_R - 3.0)
        ring_big = ring_radius_gear / math.sin(math.pi / ring_size)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        ring_centers = [
            (
                ring_big * math.cos(phase + 2.0 * math.pi * index / ring_size),
                ring_big * math.sin(phase + 2.0 * math.pi * index / ring_size),
            )
            for index in range(ring_size)
        ]
        entry_radii = [rng.uniform(_MIN_R, _MAX_R) for _ in range(entry)]
        attach = ring_centers[0]
        outward = math.atan2(attach[1], attach[0])
        centers_entry: List[Tuple[float, float]] = []
        previous = attach
        previous_radius = ring_radius_gear
        for index in range(entry):
            angle = outward + rng.uniform(-0.55, 0.55)
            spacing = previous_radius + entry_radii[index]
            point = (previous[0] + spacing * math.cos(angle), previous[1] + spacing * math.sin(angle))
            centers_entry.append(point)
            previous = point
            previous_radius = entry_radii[index]
        # Gear ids: driver 1 at the free end of the entry path, then along the
        # path into the ring and around it.
        centers = [*reversed(centers_entry), *ring_centers]
        radii = [*reversed(entry_radii), *[ring_radius_gear] * ring_size]
        fitted = self._fit(centers, radii)
        if fitted is None:
            return None
        ring_ids = list(range(entry + 1, entry + ring_size + 1))
        return fitted, radii, ring_ids

    @staticmethod
    def _fit(centers: List[Tuple[float, float]], radii: List[float]) -> List[Tuple[float, float]] | None:
        min_x = min(c[0] - r for c, r in zip(centers, radii))
        max_x = max(c[0] + r for c, r in zip(centers, radii))
        min_y = min(c[1] - r for c, r in zip(centers, radii))
        max_y = max(c[1] + r for c, r in zip(centers, radii))
        span = max(max_x - min_x, max_y - min_y)
        if span > 402.0:
            return None
        offset_x = 256.0 - (min_x + max_x) / 2.0
        offset_y = 256.0 - (min_y + max_y) / 2.0
        return [(c[0] + offset_x, c[1] + offset_y) for c in centers]

    def _expected_links(self, variant: str, count: int, ring_ids: List[int]) -> Dict[str, List[Tuple[int, int]]]:
        if variant.startswith("path"):
            belts = {(slot + 1, slot + 2) for slot in getattr(self, "_path_belt_slots", [])}
            mesh = [
                (index, index + 1)
                for index in range(1, count)
                if (index, index + 1) not in belts
            ]
            return {"mesh": sorted(mesh), "belt": sorted(belts)}
        mesh = []
        entry = ring_ids[0] - 1
        for index in range(1, entry + 1):
            mesh.append((index, index + 1))
        ring = ring_ids
        for position in range(len(ring)):
            mesh.append(tuple(sorted((ring[position], ring[(position + 1) % len(ring)]))))
        return {"mesh": sorted(set(mesh)), "belt": []}

    # ------------------------------------------------------- geometry derivation

    def _scene_fits(self, task: Mapping[str, Any]) -> bool:
        try:
            geometry = self._scene(task)
        except ValueError:
            return False
        for badge in geometry["clash_marks"]:
            x, y = badge
            if not (_CANVAS_LOW <= x <= _CANVAS_HIGH and _CANVAS_LOW <= y <= _CANVAS_HIGH):
                return False
        arrow = geometry["driver_arc"]
        if not (
            _CANVAS_LOW <= arrow["center"][0] - arrow["radius"]
            and arrow["center"][0] + arrow["radius"] <= _CANVAS_HIGH
            and _CANVAS_LOW <= arrow["center"][1] - arrow["radius"]
            and arrow["center"][1] + arrow["radius"] <= _CANVAS_HIGH
        ):
            return False
        return True

    @staticmethod
    def _scene(task: Mapping[str, Any]) -> Dict[str, Any]:
        gears = list(task["gears"])
        belts = [list(pair) for pair in task["belts"]]
        driver_direction = str(task["driver_direction"])
        signs, conflict_edges = solve_directions(gears, belts, driver_direction)
        by_id = {int(gear["id"]): gear for gear in gears}
        clash_marks: List[Tuple[float, float]] = []
        for a, b in conflict_edges:
            first, second = by_id[a], by_id[b]
            r1 = float(first["radius"])
            cx = float(first["center"][0])
            cy = float(first["center"][1])
            dx = float(second["center"][0]) - cx
            dy = float(second["center"][1]) - cy
            distance = math.hypot(dx, dy)
            clash_marks.append((cx + dx / distance * r1, cy + dy / distance * r1))
        driver = by_id[1]
        # Place the driver arc in the angular gap farthest from every linked
        # neighbor so the blue annotation never crosses another gear body.
        driver_center = (float(driver["center"][0]), float(driver["center"][1]))
        neighbor_angles: List[float] = []
        for edge in derive_mesh_edges(gears):
            if 1 in edge:
                other = by_id[edge[0] if edge[1] == 1 else edge[1]]
                neighbor_angles.append(
                    math.atan2(
                        float(other["center"][1]) - driver_center[1],
                        float(other["center"][0]) - driver_center[0],
                    )
                )
        for a, b in belts:
            if 1 in (int(a), int(b)):
                other = by_id[int(b) if int(a) == 1 else int(a)]
                neighbor_angles.append(
                    math.atan2(
                        float(other["center"][1]) - driver_center[1],
                        float(other["center"][0]) - driver_center[0],
                    )
                )
        if neighbor_angles:
            ordered = sorted(angle % (2.0 * math.pi) for angle in neighbor_angles)
            gaps = [
                (
                    (ordered[(index + 1) % len(ordered)] - angle) % (2.0 * math.pi),
                    angle,
                )
                for index, angle in enumerate(ordered)
            ]
            best_gap, best_start = max(gaps)
            arc_mid = math.degrees(best_start + best_gap / 2.0)
        else:
            arc_mid = 270.0
        return {
            "signs": signs,
            "conflict_edges": conflict_edges,
            "jam": bool(conflict_edges),
            "clash_marks": clash_marks,
            "driver_arc": {
                "center": driver_center,
                "radius": float(driver["radius"]) + 13.0,
                "mid_deg": arc_mid,
            },
        }

    @staticmethod
    def _display_task(task):
        """Source-only uniform view transform shared by drawing and observation."""
        from data.common.prompt_paraphrase import active_profile
        if not active_profile("open_gears"):
            return task,1.0,(0.0,0.0)
        low_x=min(float(g["center"][0])-float(g["radius"])-18 for g in task["gears"])
        high_x=max(float(g["center"][0])+float(g["radius"])+18 for g in task["gears"])
        low_y=min(float(g["center"][1])-float(g["radius"])-18 for g in task["gears"])
        high_y=max(float(g["center"][1])+float(g["radius"])+18 for g in task["gears"])
        scale=min(1.6,430/max(high_x-low_x,high_y-low_y))
        offset=(256-(low_x+high_x)/2*scale,256-(low_y+high_y)/2*scale)
        mapped=dict(task,gears=[dict(g,center=(float(g["center"][0])*scale+offset[0],float(g["center"][1])*scale+offset[1]),radius=float(g["radius"])*scale) for g in task["gears"]])
        return mapped,scale,offset

    @classmethod
    def rotation_carriers(
        cls, task: Mapping[str, Any]
    ) -> List[Tuple[str, Tuple[float, float], float, float, float]]:
        """Each gear's phase-carrying annulus and the angle the rule turns it.

        ``_render_progress`` draws the shaded sector, the bold spoke and every
        tooth dash of a gear at that gear's own ``theta``, and draws nothing else
        inside the band returned here that is not rotationally symmetric: the
        white hub and the gear number end at radius 12.5, and the outer bound
        stays inside the gear's own rim stroke, so a meshing neighbour - tangent,
        with its rim stroke reaching 1.5px inside its own radius - contributes at
        most a hairline at one fixed angle.  The band is therefore a rigidly
        rotating picture of the gear's phase, and the signed sweep below is the
        same one ``_render_progress`` applies.
        """

        gears = list(task["gears"])
        signs, conflict = solve_directions(gears, task["belts"], str(task["driver_direction"]))
        if conflict:
            return []
        gears=list(cls._display_task(task)[0]["gears"])
        driver_radius = next(float(g["radius"]) for g in gears if int(g["id"]) == 1)
        carriers: List[Tuple[str, Tuple[float, float], float, float, float]] = []
        for gear in gears:
            gear_id = int(gear["id"])
            radius = float(gear["radius"])
            carriers.append(
                (
                    f"gear_{gear_id}",
                    (float(gear["center"][0]), float(gear["center"][1])),
                    13.0,
                    radius - 1.0,
                    _DRIVER_SWEEP_DEG * signs[gear_id] * (driver_radius / radius),
                )
            )
        return carriers

    @classmethod
    def terminal_member_boxes(cls, task: Mapping[str, Any]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for gear in cls._display_task(task)[0]["gears"]:
            cx, cy = float(gear["center"][0]), float(gear["center"][1])
            pad = float(gear["radius"]) + 4.0
            boxes.append((f"gear_{int(gear['id'])}", (cx - pad, cy - pad, cx + pad, cy + pad)))
        return boxes

    # ---------------------------------------------------------------- semantics

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        geometry = self._scene(task)
        if geometry["jam"]:
            return "JAM"
        return "CW" if geometry["signs"][int(task["target"])] > 0 else "CCW"

    def _question(self, task: Mapping[str, Any]) -> str:
        return (
            f"Does the train turn or jam, and which way does gear {int(task['target'])} turn?"
        )

    def _direction_word(self, sign: int) -> str:
        return "clockwise" if sign > 0 else "counterclockwise"

    def _deduction_order(self, task: Mapping[str, Any]) -> List[Tuple[int, int, str]]:
        """(from_gear, to_gear, link_kind) in breadth-first deduction order."""

        mesh = {tuple(sorted(edge)) for edge in derive_mesh_edges(task["gears"])}
        belts = {tuple(sorted((int(a), int(b)))) for a, b in task["belts"]}
        neighbors: Dict[int, List[Tuple[int, str]]] = {}
        for a, b in mesh:
            neighbors.setdefault(a, []).append((b, "mesh"))
            neighbors.setdefault(b, []).append((a, "mesh"))
        for a, b in belts:
            neighbors.setdefault(a, []).append((b, "belt"))
            neighbors.setdefault(b, []).append((a, "belt"))
        seen = {1}
        queue = [1]
        order: List[Tuple[int, int, str]] = []
        while queue:
            current = queue.pop(0)
            for other, kind in sorted(neighbors.get(current, ())):
                if other not in seen:
                    seen.add(other)
                    order.append((current, other, kind))
                    queue.append(other)
        return order

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[float, str]]:
        geometry = self._scene(task)
        target = int(task["target"])
        if geometry["jam"]:
            ring = sorted({gear for edge in geometry["conflict_edges"] for gear in edge})
            steps = len(geometry["conflict_edges"])
            text = (
                f"Gear 1 is driven {self._direction_word(1 if task['driver_direction'] == 'CW' else -1)}. "
                f"Following the contacts, gears {', '.join(str(g) for g in ring)} close a loop of {steps} meshing contacts. "
                "Each meshing contact must reverse the turning direction, but an odd number of reversals around a closed loop would force one gear to turn both ways at once. "
                "No assignment works, so the whole train locks and nothing can turn: every contact of that loop is jammed."
            )
            return ((1.0, text),)
        signs = geometry["signs"]
        order = self._deduction_order(task)
        states: List[tuple[float, str]] = []
        total = len(order)
        for index, (source, dest, kind) in enumerate(order):
            if kind == "mesh":
                sentence = (
                    f"Gear {dest} meshes with gear {source}, so it turns the opposite way: {self._direction_word(signs[dest])}."
                )
            else:
                sentence = (
                    f"Gear {dest} is joined to gear {source} by the straight belt, so it turns the same way: {self._direction_word(signs[dest])}."
                )
            if index == 0:
                sentence = (
                    f"Gear 1 is driven {self._direction_word(signs[1])} by the blue arrow. " + sentence
                )
            if index == total - 1:
                sentence += (
                    f" That settles every gear, and gear {target} turns {self._direction_word(signs[target])}."
                )
            states.append(((index + 1) / total, sentence))
        return states

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        geometry = self._scene(task)
        return {
            "semantic_state": {
                "gears": [dict(gear) for gear in task["gears"]],
                "belts": [list(pair) for pair in task["belts"]],
                "driver_direction": str(task["driver_direction"]),
                "target": int(task["target"]),
            },
            "answer_schema": {
                "schema_version": "gear_train_direction_v1",
                "kind": "direction_or_jam_token",
                "parser_profile": "strict_cw_ccw_jam_token",
                "format_template": "Answer: <uppercase result token>.",
                "judged_artifacts": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "verifier_contract": {
                "profile": self.verifier_profile,
                "video_profile": self.video_verifier_profile,
                "semantic_source": "visible gear tangency, belt bands, and the driver arrow direction",
                "strict_binary_correctness": True,
                "strict_text_parser": r"^Answer: (CW|CCW|JAM)\.$",
                "untrusted_redundant_fields": [
                    "signs",
                    "jam",
                    "conflict_edges",
                    "state_trace",
                    "semantic_state_trace",
                ],
            },
            "multiple_solution_policy": "deterministic sign propagation; exactly one payload",
            "signs": {str(key): value for key, value in geometry["signs"].items()},
            "jam": geometry["jam"],
            "conflict_edges": [list(edge) for edge in geometry["conflict_edges"]],
            "semantic_state_trace": [
                f"{kind}:{source}->{dest}" for source, dest, kind in self._deduction_order(task)
            ],
        }

    # ------------------------------------------------------------------- render

    def _render_progress(self, task: Mapping[str, Any], progress: float, *, mark_answer: bool) -> Image.Image:
        supersample = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(512, 512, supersample, background=BG)
        geometry = self._scene(task)
        jam = geometry["jam"]
        signs = geometry["signs"]
        amount = max(0.0, min(1.0, float(progress)))
        driver_radius = next(float(g["radius"]) for g in task["gears"] if int(g["id"]) == 1)

        from data.common.prompt_paraphrase import active_profile
        if active_profile("open_gears"):
            draw.rectangle((0,0,512,512),fill=(229,222,191))
            draw.rectangle((12,12,500,500),fill=(249,239,204),outline=(104,99,74),width=3)
            task,scale,offset=self._display_task(task)
            def point(p):return (p[0]*scale+offset[0],p[1]*scale+offset[1])
            driver_radius*=scale
            geometry=dict(geometry,driver_arc=dict(geometry["driver_arc"],center=point(geometry["driver_arc"]["center"]),radius=geometry["driver_arc"]["radius"]*scale),
                          clash_marks=[point(p) for p in geometry["clash_marks"]])
        else:
            draw.rounded_rectangle((16, 16, 496, 496), radius=24, fill=CARD, outline=CARD_EDGE, width=2)

        # Belts under the gears.
        by_id = {int(gear["id"]): gear for gear in task["gears"]}
        for a, b in task["belts"]:
            first, second = by_id[int(a)], by_id[int(b)]
            c1 = (float(first["center"][0]), float(first["center"][1]))
            c2 = (float(second["center"][0]), float(second["center"][1]))
            r1, r2 = float(first["radius"]), float(second["radius"])
            axis = math.atan2(c2[1] - c1[1], c2[0] - c1[0])
            for side in (1.0, -1.0):
                # open-belt outer tangent (approximation for near-equal radii)
                offset = math.asin(max(-0.9, min(0.9, (r1 - r2) / math.hypot(c2[0] - c1[0], c2[1] - c1[1]))))
                angle = axis + side * (math.pi / 2.0 - offset)
                p1 = (c1[0] + r1 * math.cos(angle), c1[1] + r1 * math.sin(angle))
                p2 = (c2[0] + r2 * math.cos(angle), c2[1] + r2 * math.sin(angle))
                draw.line((*p1, *p2), fill=BELT, width=4)

        number_font = _font(15 * supersample)
        for gear in task["gears"]:
            gear_id = int(gear["id"])
            cx, cy = float(gear["center"][0]), float(gear["center"][1])
            radius = float(gear["radius"])
            if jam:
                theta = 0.0
            else:
                theta = (
                    math.radians(_DRIVER_SWEEP_DEG)
                    * amount
                    * signs[gear_id]
                    * (driver_radius / radius)
                )
            if active_profile("open_gears"):
                teeth=max(10,int(radius/3.8))
                contour=[]
                for i in range(teeth*4):
                    angle=theta+2*math.pi*i/(teeth*4)
                    rr=radius*(1.0 if i%4 in (1,2) else .86)
                    contour.append((cx+rr*math.cos(angle),cy+rr*math.sin(angle)))
                palette=((137,191,176),(165,193,212),(174,177,212),(206,211,164))
                draw.polygon(contour,fill=palette[(gear_id-1)%len(palette)],outline=(64,103,103),width=2)
                draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=(86,124,117),width=1.5)
                inner=radius*.67
                draw.ellipse((cx-inner,cy-inner,cx+inner,cy+inner),fill=(253,247,226),outline=(64,103,103),width=2)
                for blade,offset in enumerate((0,1.6,3.45,4.9)):
                    angle=theta+offset
                    points=[]
                    for rr,aa in ((9,angle-.2),(radius*.9,angle-.34),(radius*.76,angle+.35),(radius*.24,angle+.63)):
                        points.append((cx+rr*math.cos(aa),cy+rr*math.sin(aa)))
                    draw.polygon(points,fill=(51,116,125) if blade==0 else (115,162,145),outline=(43,83,88),width=2)
            else:
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=GEAR_FILL, outline=GEAR_RIM, width=3)
                # One large shaded sector rotates with the gear: its area makes the
                # rotation phase pixel-dominant and unique within one revolution,
                # so replay progress matching cannot alias across tooth pitches.
                draw.pieslice(
                    (cx - radius + 3.0, cy - radius + 3.0, cx + radius - 3.0, cy + radius - 3.0),
                    math.degrees(theta),
                    math.degrees(theta) + 70.0,
                    fill=GEAR_SECTOR,
                )
            # Rotating tooth dashes on the rim.  Tooth depths follow a fixed
            # aperiodic pattern, so the rim has no rotational symmetry and a
            # frame's rotation phase is pixel-unique within one revolution.
            tooth_count = max(8,int(round(radius/3.4)))
            for tooth in range(0 if active_profile("open_gears") else tooth_count):
                angle = theta + 2.0 * math.pi * tooth / tooth_count
                depth = 4.5 + 4.0 * (((tooth * 7 + gear_id * 3) % tooth_count) / max(1, tooth_count - 1))
                inner = radius - 2.0 - depth
                outer = radius - 1.5
                draw.line(
                    (
                        cx + inner * math.cos(angle),
                        cy + inner * math.sin(angle),
                        cx + outer * math.cos(angle),
                        cy + outer * math.sin(angle),
                    ),
                    fill=GEAR_RIM,
                    width=3,
                )
            # One bold spoke makes the rotation phase unique within a turn.
            draw.line(
                (
                    cx + 13.0 * math.cos(theta),
                    cy + 13.0 * math.sin(theta),
                    cx + (radius - 8.0) * math.cos(theta),
                    cy + (radius - 8.0) * math.sin(theta),
                ),
                fill=SPOKE,
                width=4,
            )
            draw.ellipse((cx - 11.5, cy - 11.5, cx + 11.5, cy + 11.5), fill=(255, 255, 254), outline=HUB, width=2)
            draw.text((cx, cy), str(gear_id), anchor="mm", fill=INK, font=number_font)

        # Driver arrow: curved blue arc outside gear 1 (static annotation).
        arc = geometry["driver_arc"]
        clockwise = task["driver_direction"] == "CW"
        start = arc["mid_deg"] - 60.0 if clockwise else arc["mid_deg"] + 60.0
        self._draw_arc_arrow(
            draw,
            arc["center"],
            arc["radius"],
            start_deg=start,
            sweep_deg=120.0,
            clockwise=clockwise,
            color=DRIVER_ARROW,
            width=4,
        )

        if mark_answer:
            if jam:
                for x, y in geometry["clash_marks"]:
                    for delta in (45.0, 135.0):
                        angle = math.radians(delta)
                        draw.line(
                            (
                                x - 9.0 * math.cos(angle),
                                y - 9.0 * math.sin(angle),
                                x + 9.0 * math.cos(angle),
                                y + 9.0 * math.sin(angle),
                            ),
                            fill=CLASH,
                            width=5,
                        )
            else:
                for gear in task["gears"]:
                    gear_id = int(gear["id"])
                    if gear_id == 1:
                        continue
                    cx, cy = float(gear["center"][0]), float(gear["center"][1])
                    self._draw_arc_arrow(
                        draw,
                        (cx, cy),
                        float(gear["radius"]) * 0.62,
                        start_deg=300.0,
                        sweep_deg=120.0,
                        clockwise=geometry["signs"][gear_id] > 0,
                        color=DIR_ARROW,
                        width=4,
                    )
        return finish_scaled_canvas(image, (512, 512))

    @staticmethod
    def _draw_arc_arrow(
        draw: Any,
        center: Tuple[float, float],
        radius: float,
        *,
        start_deg: float,
        sweep_deg: float,
        clockwise: bool,
        color: Tuple[int, int, int],
        width: float,
    ) -> None:
        cx, cy = center
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        if clockwise:
            begin, end = start_deg, start_deg + sweep_deg
            head_angle = math.radians(end)
            tangent = head_angle + math.pi / 2.0
        else:
            begin, end = start_deg - sweep_deg, start_deg
            head_angle = math.radians(begin)
            tangent = head_angle - math.pi / 2.0
        draw.arc(bbox, begin, end, fill=color, width=width)
        tip = (cx + radius * math.cos(head_angle), cy + radius * math.sin(head_angle))
        for spread in (math.radians(153.0), math.radians(-153.0)):
            angle = tangent + spread
            draw.line(
                (tip[0], tip[1], tip[0] + 9.5 * math.cos(angle), tip[1] + 9.5 * math.sin(angle)),
                fill=color,
                width=width,
            )

    # ------------------------------------------------------------------ contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        has_belt = bool(task["belts"])
        target = int(task["target"])
        belt_input = (
            " A pair of straight gray lines joining two separated gears is an open belt around both."
            if has_belt
            else ""
        )
        belt_rule = (
            " Two gears joined by the straight open belt turn in the same direction."
            if has_belt
            else ""
        )
        return VisualReasoningContract(
            input_description=(
                "The single image shows a train of numbered gears. Each gear is a toothed disk with one shaded sector, one bold spoke, a white hub holding its number, and a rim; "
                "two gears whose rims touch are meshed."
                + belt_input
                + " The blue curved arrow on gear 1 fixes the direction in which gear 1 is driven."
            ),
            symbol_legend=(
                "All turning directions are read as seen by the viewer: CW means clockwise and CCW means counterclockwise.",
                "Gear numbers are identities only and carry no order information beyond naming the driven gear 1 and the asked gear.",
            ),
            rules=(
                "Gear 1 is driven in the direction of the blue arrow.",
                "Two meshed gears (rims touching) always turn in opposite directions."
                + belt_rule,
                "Every gear takes exactly one turning direction through its contacts. If the contacts of a closed loop of gears cannot all be satisfied at once, the whole train jams and no gear can turn at all.",
                f"The question asks about gear {target}: report CW, CCW, or JAM.",
            ),
            modality_outputs={
                "ti2v": (
                    "If the train turns, all gears rotate together continuously in their true directions with speeds inversely proportional to size, come to rest, and then the orange direction arrows appear on every gear except gear 1; hold that final state. "
                    "If the train jams, nothing rotates at any time: the red clash crosses appear on every contact of the conflicting loop over the still scene, then hold."
                ),
                "solution_image": (
                    "If the train turns, add one orange curved direction arrow to every gear except gear 1 and change nothing else. "
                    "If it jams, add one red clash cross at every contact of the conflicting loop and change nothing else."
                ),
            },
            preservation_constraints=(
                "Keep every gear's position, size, teeth, shaded sector and spoke pose at rest, hub, number, the belt lines if present, the blue driver arrow, and the background fixed; only rotation, the orange arrows, or the red clash crosses may appear.",
            ),
            transition_constraints=(
                "Rotate rigidly about fixed centers with meshed neighbors counter-rotating, or stay perfectly still when jammed; never teleport a spoke, reverse mid-video, or move a gear center.",
            ),
            ti2v_continuity={
                "moving_nouns": "the rotating gear teeth and spokes",
                "static_nouns": "the gear centers, hubs, numbers, belt lines, blue driver arrow, board, and camera",
                "hold_noun": "the resting train with its orange direction arrows, or the still jammed train with its red clash crosses",
            },
            answer=AnswerContract(
                kind="typed_text",
                format_template="Answer: <uppercase result token>.",
                parser_profile="strict_cw_ccw_jam_token",
                verifier_profile=self.verifier_profile,
                format_example="Answer: UNKNOWN.",
            ),
            render_profile="gear_train_direction_v1",
        )


__all__ = ["GearTrainDirectionPattern", "derive_mesh_edges", "solve_directions"]
