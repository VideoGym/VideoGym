"""Slice cross-section: name the true shape of a plane cut through a solid.

One simple solid (box, cylinder, cone, sphere, square pyramid, or triangular
prism) floats in the family three-quarter camera; a thin blue wireframe
rectangle marks the cutting plane (horizontal, vertical, or clearly tilted).
The typed answer is the most specific name of the flat cut face from a closed
six-word vocabulary; the visual answer shows the cut outline on the solid and
then the true face-on shape in the answer panel.  Every sampled combination
carries an explicit perceptual margin (tilt angle, side ratio, or offset) so
no two vocabulary words ever compete for the same picture.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple


from PIL import Image, ImageDraw

from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
)
from data.spatial_3d.core.iso_geometry import (
    Face,
    IsoProjection,
    fit_projection,
    render_faces,
)
from data.spatial_3d.core.scene_base import Spatial3DScenePattern

BG = (246, 248, 251)
CARD = (255, 255, 253)
CARD_EDGE = (216, 221, 226)
SOLID_COLOR = (198, 206, 218)
PLANE_COLOR = (57, 115, 182)
CUT_COLOR = (222, 133, 73)
CUT_FILL = (247, 205, 166)
PANEL_FRAME = (90, 96, 108)

_WORDS = ("circle", "ellipse", "square", "rectangle", "triangle", "trapezoid")
_RATIO_GATE = 1.30          # rectangle vs square margin
_TILT_MIN_DEG = 30.0        # ellipse vs circle margin (axis ratio >= 1.155)
_TRAPEZOID_TOP_RATIO = (0.35, 0.70)

_PANEL_BOX = (156, 356, 356, 492)

# The curated cut table: (solid, cut, answer word).
_COMBOS: Tuple[Tuple[str, str, str], ...] = (
    ("sphere", "tilted", "circle"),
    ("cylinder", "horizontal", "circle"),
    ("cone", "horizontal", "circle"),
    ("cylinder", "tilted", "ellipse"),
    ("box", "horizontal", "rectangle"),
    ("box", "vertical", "rectangle"),
    ("cylinder", "vertical", "rectangle"),
    ("prism", "vertical", "rectangle"),
    ("cube", "horizontal", "square"),
    ("pyramid", "horizontal", "square"),
    ("cone", "vertical_apex", "triangle"),
    ("pyramid", "vertical_apex", "triangle"),
    ("prism", "horizontal", "triangle"),
    ("pyramid", "vertical_offset", "trapezoid"),
)


def replay_slice(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Independent forward replay from the visible solid and cut only."""

    solid = str(source["solid"])
    cut = dict(source["cut"])
    kind = str(cut["kind"])
    combo = next(((s, k, w) for s, k, w in _COMBOS if s == solid and k == kind), None)
    if combo is None:
        raise ValueError(f"unsupported solid/cut combination: {solid}/{kind}")
    section = _section_shape(solid, cut)
    if section["word"] != combo[2]:
        raise ValueError("replayed section word disagrees with the cut table")
    return {"solid": solid, "cut": cut, "word": section["word"], "section": section}


def _section_shape(solid: str, cut: Mapping[str, Any]) -> Dict[str, Any]:
    """True 2D shape (face-on) and its 3D boundary loop for one solid and cut."""

    kind = str(cut["kind"])
    if solid == "sphere":
        radius = float(cut["radius"])
        offset = float(cut["offset"])
        section_radius = math.sqrt(max(1e-9, radius * radius - offset * offset))
        theta = math.radians(float(cut["tilt_deg"]))
        center = (0.0, 0.0, radius)
        normal = (0.0, -math.sin(theta), math.cos(theta))
        loop = _disk_loop(center=_offset_point(center, normal, offset), normal=normal, r_u=section_radius, r_v=section_radius)
        return {"word": "circle", "outline_2d": _ellipse_outline(section_radius, section_radius), "loop_3d": loop}
    if solid == "cylinder":
        radius, height = float(cut["radius"]), float(cut["height"])
        if kind == "horizontal":
            z = float(cut["z"])
            loop = _disk_loop(center=(0.0, 0.0, z), normal=(0.0, 0.0, 1.0), r_u=radius, r_v=radius)
            return {"word": "circle", "outline_2d": _ellipse_outline(radius, radius), "loop_3d": loop}
        if kind == "tilted":
            theta = math.radians(float(cut["tilt_deg"]))
            z = height / 2.0
            major = radius / math.cos(theta)
            # Boundary: points on the lateral surface where the tilted plane
            # z = z0 + tan(theta) * y meets x^2 + y^2 = r^2.
            loop = [
                (radius * math.cos(a), radius * math.sin(a), z + math.tan(theta) * radius * math.sin(a))
                for a in _angles()
            ]
            return {"word": "ellipse", "outline_2d": _ellipse_outline(radius, major), "loop_3d": loop}
        chord_offset = float(cut["offset"])
        half_chord = math.sqrt(max(1e-9, radius * radius - chord_offset * chord_offset))
        loop = [
            (-half_chord, chord_offset, 0.0), (half_chord, chord_offset, 0.0),
            (half_chord, chord_offset, height), (-half_chord, chord_offset, height),
        ]
        return {"word": "rectangle", "outline_2d": _rect_outline(2 * half_chord, height), "loop_3d": loop}
    if solid == "cone":
        radius, height = float(cut["radius"]), float(cut["height"])
        if kind == "horizontal":
            z = float(cut["z"])
            section_radius = radius * (1.0 - z / height)
            loop = _disk_loop(center=(0.0, 0.0, z), normal=(0.0, 0.0, 1.0), r_u=section_radius, r_v=section_radius)
            return {"word": "circle", "outline_2d": _ellipse_outline(section_radius, section_radius), "loop_3d": loop}
        loop = [(-radius, 0.0, 0.0), (radius, 0.0, 0.0), (0.0, 0.0, height)]
        return {"word": "triangle", "outline_2d": [(-radius, 0.0), (radius, 0.0), (0.0, height)], "loop_3d": loop}
    if solid in ("box", "cube"):
        w, d, h = float(cut["w"]), float(cut["d"]), float(cut["h"])
        if kind == "horizontal":
            z = float(cut["z"])
            loop = [(-w / 2, -d / 2, z), (w / 2, -d / 2, z), (w / 2, d / 2, z), (-w / 2, d / 2, z)]
            word = "square" if solid == "cube" else "rectangle"
            return {"word": word, "outline_2d": _rect_outline(w, d), "loop_3d": loop}
        x = float(cut["x"])
        loop = [(x, -d / 2, 0.0), (x, d / 2, 0.0), (x, d / 2, h), (x, -d / 2, h)]
        return {"word": "rectangle", "outline_2d": _rect_outline(d, h), "loop_3d": loop}
    if solid == "pyramid":
        half, height = float(cut["half"]), float(cut["height"])
        if kind == "horizontal":
            z = float(cut["z"])
            s = half * (1.0 - z / height)
            loop = [(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)]
            return {"word": "square", "outline_2d": _rect_outline(2 * s, 2 * s), "loop_3d": loop}
        if kind == "vertical_apex":
            loop = [(-half, 0.0, 0.0), (half, 0.0, 0.0), (0.0, 0.0, height)]
            return {"word": "triangle", "outline_2d": [(-half, 0.0), (half, 0.0), (0.0, height)], "loop_3d": loop}
        y = float(cut["y"])                       # offset plane y = const, 0 < y < half
        c = y / half                              # top width fraction
        top_z = height * (1.0 - c)
        x_at = lambda z: half * (1.0 - z / height)
        loop = [
            (-x_at(0.0), y, 0.0), (x_at(0.0), y, 0.0),
            (x_at(top_z), y, top_z), (-x_at(top_z), y, top_z),
        ]
        return {
            "word": "trapezoid",
            "outline_2d": [(-half, 0.0), (half, 0.0), (half * c, top_z), (-half * c, top_z)],
            "loop_3d": loop,
        }
    if solid == "prism":
        side, height = float(cut["side"]), float(cut["height"])
        tri = _prism_triangle(side)
        if kind == "horizontal":
            z = float(cut["z"])
            loop = [(px, py, z) for px, py in tri]
            return {"word": "triangle", "outline_2d": list(tri), "loop_3d": loop}
        t = float(cut["t"])                       # fraction from the front edge toward the back vertex
        chord = side * (1.0 - t)
        y0, y1 = tri[0][1], tri[2][1]
        y = y0 + (y1 - y0) * t
        loop = [(-chord / 2, y, 0.0), (chord / 2, y, 0.0), (chord / 2, y, height), (-chord / 2, y, height)]
        return {"word": "rectangle", "outline_2d": _rect_outline(chord, height), "loop_3d": loop}
    raise ValueError(f"unsupported solid: {solid}")


def _angles(count: int = 48) -> List[float]:
    return [2.0 * math.pi * i / count for i in range(count)]


def _disk_loop(center, normal, r_u, r_v) -> List[Tuple[float, float, float]]:
    nx, ny, nz = normal
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    if abs(nz) < 0.9:
        u = _normalize(_cross((nx, ny, nz), (0.0, 0.0, 1.0)))
    else:
        u = _normalize(_cross((nx, ny, nz), (1.0, 0.0, 0.0)))
    v = _cross((nx, ny, nz), u)
    return [
        (
            center[0] + r_u * math.cos(a) * u[0] + r_v * math.sin(a) * v[0],
            center[1] + r_u * math.cos(a) * u[1] + r_v * math.sin(a) * v[1],
            center[2] + r_u * math.cos(a) * u[2] + r_v * math.sin(a) * v[2],
        )
        for a in _angles()
    ]


def _offset_point(point, normal, distance):
    n = _normalize(normal)
    return (point[0] + n[0] * distance, point[1] + n[1] * distance, point[2] + n[2] * distance)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return (v[0] / n, v[1] / n, v[2] / n)


def _ellipse_outline(r_u: float, r_v: float) -> List[Tuple[float, float]]:
    return [(r_u * math.cos(a), r_v * math.sin(a)) for a in _angles()]


def _rect_outline(w: float, h: float) -> List[Tuple[float, float]]:
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]


def _prism_triangle(side: float) -> List[Tuple[float, float]]:
    height = side * math.sqrt(3.0) / 2.0
    return [(-side / 2, -height / 3), (side / 2, -height / 3), (0.0, 2 * height / 3)]


class SliceCrossSectionPattern(Spatial3DScenePattern):
    task_name = "slice_cross_section"
    task_family = "spatial_3d"
    task_mechanic = "plane_solid_cross_section"
    motion = False
    # The Rule lists the complete six-word answer vocabulary (8.9).
    answer_vocabulary_in_prompt = True
    verifier_profile = "slice_cross_section_visible_cut_v1"

    # ------------------------------------------------------------- sampling

    def _sample_task(self) -> Dict[str, Any]:
        words = {
            "min": ("circle", "square", "rectangle", "triangle"),
            "typical": _WORDS,
            "max": ("ellipse", "trapezoid", "triangle", "rectangle"),
        }.get(self.difficulty or "", _WORDS)
        word = self.rng.choice(words)
        solid, kind, _ = self.rng.choice([combo for combo in _COMBOS if combo[2] == word])
        cut = self._sample_cut(solid, kind)
        replayed = replay_slice({"solid": solid, "cut": cut})
        section = replayed["section"]
        evidence = self._margin_evidence(solid, cut)
        ratio = float(evidence["bbox_aspect_ratio"])
        # 7.5 hard gates: the named word must beat its nearest neighbor by a
        # visible margin, never by sub-pixel proportions.
        if word in ("rectangle", "ellipse") and ratio < 1.15:
            raise AssertionError(f"{word} section sampled without its aspect margin: {ratio}")
        if word in ("square", "circle") and ratio > 1.02:
            raise AssertionError(f"{word} section sampled off-square: {ratio}")
        if word == "rectangle" and ratio < _RATIO_GATE - 0.02:
            raise AssertionError(f"rectangle section under the ratio gate: {ratio}")
        task = {
            "solid": solid,
            "cut": cut,
            "word": replayed["word"],
            "section": section,
            "margin_evidence": evidence,
        }
        task["projection"] = self._fit_camera(task)
        return task

    def _sample_cut(self, solid: str, kind: str) -> Dict[str, Any]:
        rng = self.rng
        if solid == "sphere":
            radius = rng.uniform(1.15, 1.5)
            return {
                "kind": "tilted",
                "radius": round(radius, 3),
                "tilt_deg": round(rng.uniform(18.0, 55.0), 1),
                "offset": round(rng.uniform(-0.35, 0.35) * radius, 3),
            }
        if solid == "cylinder":
            radius = rng.uniform(0.85, 1.2)
            height = rng.uniform(1.9, 2.6)
            if kind == "horizontal":
                return {"kind": kind, "radius": round(radius, 3), "height": round(height, 3),
                        "z": round(rng.uniform(0.3, 0.7) * height, 3)}
            if kind == "tilted":
                tilt = rng.uniform(_TILT_MIN_DEG, 46.0)
                # keep the tilted plane strictly on the lateral surface
                while math.tan(math.radians(tilt)) * radius > height * 0.44:
                    tilt -= 2.0
                return {"kind": kind, "radius": round(radius, 3), "height": round(height, 3),
                        "tilt_deg": round(max(_TILT_MIN_DEG, tilt), 1)}
            offset = rng.uniform(0.0, 0.45) * radius
            while 2 * math.sqrt(radius**2 - offset**2) > height / _RATIO_GATE:
                offset += 0.08 * radius
            return {"kind": "vertical", "radius": round(radius, 3), "height": round(height, 3),
                    "offset": round(offset, 3)}
        if solid == "cone":
            radius = rng.uniform(0.95, 1.3)
            height = rng.uniform(1.7, 2.4)
            if kind == "horizontal":
                return {"kind": kind, "radius": round(radius, 3), "height": round(height, 3),
                        "z": round(rng.uniform(0.3, 0.62) * height, 3)}
            return {"kind": "vertical_apex", "radius": round(radius, 3), "height": round(height, 3)}
        if solid in ("box", "cube"):
            if solid == "cube":
                edge = rng.uniform(1.4, 1.9)
                w = d = h = edge
            else:
                w = rng.uniform(1.5, 2.2)
                d = rng.uniform(w / 2.4, w / _RATIO_GATE)
                h = rng.uniform(1.0, 1.6)
                if kind == "vertical" and max(d, h) / min(d, h) < _RATIO_GATE:
                    # The vertical section is d x h: force its own ratio gate.
                    h = min(2.2, max(0.8, d * (1.45 if rng.random() < 0.5 else 1.0 / 1.45)))
            base = {"w": round(w, 3), "d": round(d, 3), "h": round(h, 3)}
            if kind == "horizontal":
                return {"kind": kind, **base, "z": round(rng.uniform(0.3, 0.7) * h, 3)}
            return {"kind": "vertical", **base, "x": round(rng.uniform(-0.3, 0.3) * w, 3)}
        if solid == "pyramid":
            half = rng.uniform(0.95, 1.3)
            height = rng.uniform(1.6, 2.2)
            base = {"half": round(half, 3), "height": round(height, 3)}
            if kind == "horizontal":
                return {"kind": kind, **base, "z": round(rng.uniform(0.28, 0.6) * height, 3)}
            if kind == "vertical_apex":
                return {"kind": kind, **base}
            c = rng.uniform(*_TRAPEZOID_TOP_RATIO)
            return {"kind": "vertical_offset", **base, "y": round(c * half, 3)}
        if solid == "prism":
            side = rng.uniform(1.6, 2.2)
            height = rng.uniform(1.7, 2.4)
            base = {"side": round(side, 3), "height": round(height, 3)}
            if kind == "horizontal":
                return {"kind": kind, **base, "z": round(rng.uniform(0.3, 0.7) * height, 3)}
            t = rng.uniform(0.15, 0.45)
            while side * (1.0 - t) > height / _RATIO_GATE:
                t += 0.05
            return {"kind": "vertical", **base, "t": round(t, 3)}
        raise ValueError(solid)

    def _margin_evidence(self, solid: str, cut: Mapping[str, Any]) -> Dict[str, float]:
        section = _section_shape(solid, cut)
        outline = section["outline_2d"]
        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        ratio = max(w, h) / max(1e-9, min(w, h))
        evidence = {"bbox_aspect_ratio": round(ratio, 3)}
        if section["word"] == "ellipse":
            evidence["tilt_deg"] = float(cut["tilt_deg"])
        if section["word"] == "rectangle":
            evidence["rectangle_ratio_gate"] = _RATIO_GATE
        if section["word"] == "trapezoid":
            evidence["top_over_bottom"] = round(float(cut["y"]) / float(cut["half"]), 3)
        return evidence

    # ------------------------------------------------------------- solids

    def _solid_faces(self, task: Mapping[str, Any]) -> List[Face]:
        solid = task["solid"]
        cut = task["cut"]
        faces: List[Face] = []
        if solid == "sphere":
            radius = float(cut["radius"])
            faces.extend(_uv_sphere_faces((0.0, 0.0, radius), radius))
        elif solid == "cylinder":
            faces.extend(_cylinder_faces(float(cut["radius"]), float(cut["height"])))
        elif solid == "cone":
            faces.extend(_cone_faces(float(cut["radius"]), float(cut["height"])))
        elif solid in ("box", "cube"):
            w, d, h = float(cut["w"]), float(cut["d"]), float(cut["h"])
            faces.extend(_box_faces(w, d, h))
        elif solid == "pyramid":
            faces.extend(_pyramid_faces(float(cut["half"]), float(cut["height"])))
        elif solid == "prism":
            faces.extend(_prism_faces(float(cut["side"]), float(cut["height"])))
        return faces

    def _plane_frame(self, task: Mapping[str, Any]) -> List[Tuple[float, float, float]]:
        loop = task["section"]["loop_3d"]
        cx = sum(p[0] for p in loop) / len(loop)
        cy = sum(p[1] for p in loop) / len(loop)
        cz = sum(p[2] for p in loop) / len(loop)
        normal = _loop_normal(loop)
        # In-plane basis aligned with the world: u horizontal, v as upright
        # as the plane allows, so vertical planes show vertical frame edges.
        if abs(normal[2]) < 0.95:
            u = _normalize(_cross(normal, (0.0, 0.0, 1.0)))
        else:
            u = _normalize(_cross(normal, (0.0, 1.0, 0.0)))
        v = _cross(normal, u)
        extents_u = [sum((p[i] - (cx, cy, cz)[i]) * u[i] for i in range(3)) for p in loop]
        extents_v = [sum((p[i] - (cx, cy, cz)[i]) * v[i] for i in range(3)) for p in loop]
        margin = 0.42
        u0, u1 = min(extents_u) - margin, max(extents_u) + margin
        v0, v1 = min(extents_v) - margin, max(extents_v) + margin
        return [
            (
                cx + su * u[0] + sv * v[0],
                cy + su * u[1] + sv * v[1],
                cz + su * u[2] + sv * v[2],
            )
            for su, sv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
        ]

    def _fit_camera(self, task: Mapping[str, Any]) -> IsoProjection:
        points = [p for face in self._solid_faces(task) for p in face.points]
        points.extend(self._plane_frame(task))
        return fit_projection(points, box=(76.0, 40.0, 436.0, 330.0), max_scale=92.0)

    # ------------------------------------------------------------- rendering

    def _render_stage(self, task: Mapping[str, Any], index: int) -> Image.Image:
        supersample = DOCTRINE_SUPERSAMPLE
        canvas = Image.new("RGB", (512 * supersample, 512 * supersample), BG)
        draw = ImageDraw.Draw(canvas)
        stroke = ScaledDraw(draw, supersample)
        draw.rounded_rectangle(
            (8 * supersample, 8 * supersample, 504 * supersample, 504 * supersample),
            radius=20 * supersample,
            fill=CARD,
            outline=CARD_EDGE,
            width=2 * supersample,
        )
        projection: IsoProjection = task["projection"]
        render_faces(canvas, self._solid_faces(task), projection, supersample)
        # Cutting-plane wireframe: a load-bearing annotation drawn above the
        # solid with a deliberately thin stroke.
        frame = [projection.project(p) for p in self._plane_frame(task)]
        stroke.closed_outline(frame, fill=PLANE_COLOR, width=2)
        if index >= 1:
            loop = [projection.project(p) for p in task["section"]["loop_3d"]]
            stroke.closed_outline(loop, fill=CUT_COLOR, width=3)
        # Blank answer panel; the true shape appears at stage 2.
        draw.rounded_rectangle(
            tuple(c * supersample for c in _PANEL_BOX),
            radius=10 * supersample,
            fill=(255, 255, 255),
            outline=PANEL_FRAME,
            width=2 * supersample,
        )
        if index >= 2:
            outline = task["section"]["outline_2d"]
            xs = [p[0] for p in outline]
            ys = [p[1] for p in outline]
            width, height = max(xs) - min(xs), max(ys) - min(ys)
            pad = 18
            scale = min(
                (_PANEL_BOX[2] - _PANEL_BOX[0] - 2 * pad) / width,
                (_PANEL_BOX[3] - _PANEL_BOX[1] - 2 * pad) / height,
            )
            cx = (_PANEL_BOX[0] + _PANEL_BOX[2]) / 2.0
            cy = (_PANEL_BOX[1] + _PANEL_BOX[3]) / 2.0
            mx = (max(xs) + min(xs)) / 2.0
            my = (max(ys) + min(ys)) / 2.0
            points = [
                (cx + (px - mx) * scale, cy - (py - my) * scale)
                for px, py in outline
            ]
            stroke.polygon(points, fill=CUT_FILL)
            stroke.closed_outline(points, fill=CUT_COLOR, width=3)
        return finish_scaled_canvas(canvas, (512, 512))

    def _stage_count(self, task: Mapping[str, Any]) -> int:
        return 3

    # ------------------------------------------------------------- semantics

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        return str(task["word"])

    def _question(self, task: Mapping[str, Any]) -> str:
        return "What is the true shape of the flat face the plane cuts through the solid?"

    _SOLID_PROSE = {
        "sphere": "a solid ball",
        "cylinder": "an upright round cylinder",
        "cone": "an upright cone with its point at the top",
        "box": "a rectangular box",
        "cube": "a cube",
        "pyramid": "an upright square pyramid with its point at the top",
        "prism": "an upright triangular prism",
    }

    def _cut_prose(self, task: Mapping[str, Any]) -> str:
        kind = task["cut"]["kind"]
        if kind == "horizontal":
            return "level, cutting straight across"
        if kind == "vertical":
            return "upright, cutting straight down"
        if kind == "vertical_apex":
            return "upright, passing exactly through the top point"
        if kind == "vertical_offset":
            return "upright, passing clearly to one side of the top point"
        return "clearly tilted"

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[Any, str]]:
        solid = task["solid"]
        word = task["word"]
        reasons = {
            ("sphere", "circle"): "every flat cut through a ball meets it in a circle, whatever the tilt",
            ("cylinder", "circle"): "a level cut across the round tube is exactly its round footprint",
            ("cone", "circle"): "a level cut across the cone is a smaller copy of its round base",
            ("cylinder", "ellipse"): "the tilt stretches the tube's round footprint one way, so the cut face is a stretched circle",
            ("box", "rectangle"): "the cut runs parallel to a pair of box faces, so it repeats their rectangular outline",
            ("cube", "square"): "the level cut repeats the cube's square footprint",
            ("cylinder", "rectangle"): "the upright plane takes a straight chord of the base and carries it up the full height",
            ("prism", "rectangle"): "the upright plane takes a straight chord of the triangular base and carries it up the full height",
            ("pyramid", "square"): "a level cut across the pyramid is a smaller copy of its square base",
            ("cone", "triangle"): "through the apex the cut takes the base diameter and the slanting sides straight to the point",
            ("pyramid", "triangle"): "through the apex the cut takes a base line and the slanting faces straight to the point",
            ("prism", "triangle"): "a level cut across the prism repeats its triangular footprint",
            ("pyramid", "trapezoid"): "off the apex the slanting faces pull the top edge in shorter than the bottom edge, while the two edges stay parallel",
        }
        key = (solid, word)
        section = task["section"]
        outline = section["outline_2d"]
        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        ratio = round(max(max(xs) - min(xs), max(ys) - min(ys)) / max(1e-9, min(max(xs) - min(xs), max(ys) - min(ys))), 2)
        trace_text = (
            f"The solid is {self._SOLID_PROSE[solid]}, and the blue frame shows the cutting plane is "
            f"{self._cut_prose(task)}. I trace where the plane meets the surface: {reasons[key]}."
        )
        detail = ""
        if word == "rectangle":
            detail = f" Its two side lengths clearly differ (about {ratio} to 1), so it is a rectangle rather than a square."
        elif word == "ellipse":
            detail = f" Its long axis clearly beats its short axis (about {ratio} to 1), so it is an ellipse rather than a circle."
        elif word == "trapezoid":
            detail = " Its top edge is clearly shorter than its bottom edge, so it is a trapezoid rather than a rectangle."
        panel_text = (
            f"Laying that cut face flat gives the true shape for the panel: a {word}.{detail}"
        )
        return [(1, trace_text), (2, panel_text)]

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "semantic_state": {
                "solid": task["solid"],
                "cut": dict(task["cut"]),
            },
            "section_word": task["word"],
            "margin_evidence": dict(task["margin_evidence"]),
            "multiple_solution_policy": "the closed vocabulary asks the most specific name; exactly one applies",
            "difficulty_evidence": {
                "solid": task["solid"],
                "cut_kind": task["cut"]["kind"],
            },
            "answer_schema": {
                "schema_version": "slice_section_word_v1",
                "kind": "closed_shape_word",
                "parser_profile": "strict_shape_word_six",
                "format_template": "Answer: <lowercase shape>.",
                "judged_artifacts": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "verifier_contract": {
                "profile": self.verifier_profile,
                "semantic_source": "visible solid, cutting-plane frame pose, and the cut table",
                "strict_binary_correctness": True,
                "strict_text_parser": r"^Answer: (circle|ellipse|square|rectangle|triangle|trapezoid)\.$",
                "untrusted_redundant_fields": ["section_word"],
            },
        }

    def _glyph_inventory(self, task: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
        del task
        inventory = {
            "schema_version": "slice_rendered_glyph_inventory_v1",
            "fixed_labels": [],
            "dynamic_labels": [],
            "all_rendered_text": [],
        }
        schema = {
            "fixed_labels": "none: the scene carries only geometry",
            "dynamic_labels": "none",
            "exclusions": "no task title, rule prose, coordinates, status sentence, or answer banner",
        }
        return inventory, schema

    # -------------------------------------------------------------- contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                f"The single image shows one solid - {self._SOLID_PROSE[task['solid']]} - with a thin blue "
                "wireframe rectangle marking a flat cutting plane that passes through it, and a blank "
                "answer panel below."
            ),
            symbol_legend=(
                "The blue rectangle is only the visible frame of the endless flat cutting plane; the plane "
                "continues through the whole solid.",
                f"For this sample the plane is {self._cut_prose(task)}.",
            ),
            rules=(
                "The plane slices straight through the solid, leaving one flat cut face.",
                "The answer is the true shape of that flat face, seen square-on, named with the most "
                "specific word that applies from exactly this list: circle, ellipse, square, rectangle, "
                "triangle, trapezoid.",
            ),
            modality_outputs={
                "ti2v": (
                    "Two commits: first the orange cut outline appears on the solid exactly where the "
                    "plane meets its surface, fading in as one complete curve; then the answer panel "
                    "shows the cut face flat-on as an orange-filled shape at its true proportions, fading "
                    "in as one complete shape. The finished picture then holds."
                ),
                "solution_image": (
                    "Show the orange cut outline on the solid where the plane meets its surface, and the "
                    "answer panel completed with the flat-on orange-filled true shape of the cut face."
                ),
            },
            preservation_constraints=(
                "the solid, the blue plane frame, the answer panel frame, and the background unchanged; "
                "only the orange cut outline and the panel shape may appear.",
            ),
            transition_constraints=(
                "Commit each of the two additions as one synchronized element-local fade of its complete "
                "geometry; never draw the outline stroke by stroke, reshape the panel figure, or dissolve "
                "the full frame.",
            ),
            ti2v_continuity={
                "moving_nouns": "the appearing orange cut outline and panel shape",
                "static_nouns": "the solid, the blue plane frame, the answer panel frame, the background, "
                "and the camera",
                "hold_noun": "the traced solid with its completed answer panel",
            },
            answer=AnswerContract(
                kind="typed_text",
                format_template="Answer: <lowercase shape>.",
                parser_profile="strict_shape_word_six",
                verifier_profile=self.verifier_profile,
                format_example="Answer: octagon.",
            ),
            render_profile="slice_cross_section_v1",
        )


def _loop_normal(loop: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    a, b, c = loop[0], loop[1], loop[2 if len(loop) < 8 else len(loop) // 4]
    return _normalize(_cross((b[0] - a[0], b[1] - a[1], b[2] - a[2]), (c[0] - a[0], c[1] - a[1], c[2] - a[2])))


def _box_faces(w: float, d: float, h: float) -> List[Face]:
    x0, x1, y0, y1, z0, z1 = -w / 2, w / 2, -d / 2, d / 2, 0.0, h
    quads = (
        ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),
        ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)),
        ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)),
        ((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)),
        ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)),
    )
    return [Face(points=quad, color=SOLID_COLOR) for quad in quads]


def _pyramid_faces(half: float, height: float) -> List[Face]:
    apex = (0.0, 0.0, height)
    corners = ((-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0))
    faces = [Face(points=corners[::-1], color=SOLID_COLOR)]
    for index in range(4):
        a, b = corners[index], corners[(index + 1) % 4]
        faces.append(Face(points=(a, b, apex), color=SOLID_COLOR))
    return faces


def _prism_faces(side: float, height: float) -> List[Face]:
    tri = _prism_triangle(side)
    top = [(x, y, height) for x, y in tri]
    bottom = [(x, y, 0.0) for x, y in tri]
    faces = [Face(points=tuple(top), color=SOLID_COLOR)]
    for index in range(3):
        a, b = bottom[index], bottom[(index + 1) % 3]
        ta, tb = top[index], top[(index + 1) % 3]
        faces.append(Face(points=(a, b, tb, ta), color=SOLID_COLOR))
    return faces


def _cylinder_faces(radius: float, height: float, segments: int = 48) -> List[Face]:
    faces: List[Face] = []
    ring = [(radius * math.cos(a), radius * math.sin(a)) for a in _angles(segments)]
    faces.append(Face(points=tuple((x, y, height) for x, y in ring), color=SOLID_COLOR))
    for index in range(segments):
        x0, y0 = ring[index]
        x1, y1 = ring[(index + 1) % segments]
        faces.append(
            Face(
                points=((x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, height), (x0, y0, height)),
                color=SOLID_COLOR,
                edge_color=None,
            )
        )
    return faces


def _cone_faces(radius: float, height: float, segments: int = 48) -> List[Face]:
    faces: List[Face] = []
    ring = [(radius * math.cos(a), radius * math.sin(a)) for a in _angles(segments)]
    apex = (0.0, 0.0, height)
    for index in range(segments):
        x0, y0 = ring[index]
        x1, y1 = ring[(index + 1) % segments]
        faces.append(Face(points=((x0, y0, 0.0), (x1, y1, 0.0), apex), color=SOLID_COLOR, edge_color=None))
    faces.append(Face(points=tuple((x, y, 0.0) for x, y in reversed(ring)), color=SOLID_COLOR))
    return faces


def _uv_sphere_faces(center, radius: float, rings: int = 16, segments: int = 28) -> List[Face]:
    faces: List[Face] = []
    for ring_index in range(rings):
        phi0 = math.pi * ring_index / rings
        phi1 = math.pi * (ring_index + 1) / rings
        for seg_index in range(segments):
            theta0 = 2 * math.pi * seg_index / segments
            theta1 = 2 * math.pi * (seg_index + 1) / segments
            def point(phi, theta):
                return (
                    center[0] + radius * math.sin(phi) * math.cos(theta),
                    center[1] + radius * math.sin(phi) * math.sin(theta),
                    center[2] + radius * math.cos(phi),
                )
            quad = (point(phi0, theta0), point(phi1, theta0), point(phi1, theta1), point(phi0, theta1))
            faces.append(Face(points=quad, color=SOLID_COLOR, edge_color=None))
    return faces


__all__ = ["SliceCrossSectionPattern", "replay_slice"]
