"""Strict-interior lattice-point counting with a semantic reveal trace."""

from __future__ import annotations

from data.common.base import RenderedTaskGenerator
from data.geometry.core.rendered_contract import finalize_geometry_sample

import math
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.video_encoding import encode_rgb_frames_to_webm, pad_frames_to_wan_compatible_length
from data.common.pipeline.media import load_font
from data.common.choice_cards import attach_text_choice_card_inputs
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.geometry.core.geometry_counting import augment_counting_contract


pattern_instructions = {
    "lattice_points_strict_inside": (
        "A full-canvas coordinate lattice shows one integer-vertex polygon. "
        "The task is to count the visible lattice points strictly inside the polygon while excluding every boundary point."
    ),
}

_LABELS = ("A", "B", "C", "D")
_GRID_RADIUS = 4
_GRID_CENTER = (264.0, 264.0)
_CELL_SIZE = 46.0
_GRID_LINE_OVERHANG = 10.0
_BG = (255, 255, 255)
_GRID_RGB = (218, 222, 228)
_DOT_RGB = (134, 142, 152)
_POLYGON_FILL = (224, 241, 249)
_OUTLINE_RGB = (42, 45, 52)
_TEXT_RGB = (52, 54, 60)
_BOUNDARY_RGB = (232, 146, 58)
_INTERIOR_RGB = (20, 151, 188)
_MARKER_RADIUS = 8.0
_CANVAS_SIZE = (512, 512)


def _shoelace_twice(verts: Sequence[Tuple[int, int]]) -> int:
    return abs(
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(verts, [*verts[1:], verts[0]])
        )
    )


def _point_on_segment(point: Tuple[int, int], start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    px, py = point
    x0, y0 = start
    x1, y1 = end
    return bool(
        (x1 - x0) * (py - y0) == (y1 - y0) * (px - x0)
        and min(x0, x1) <= px <= max(x0, x1)
        and min(y0, y1) <= py <= max(y0, y1)
    )


def _orientation(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 0 if value == 0 else 1 if value > 0 else -1


def _segments_intersect(
    a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int], d: Tuple[int, int]
) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return bool(
        (o1 == 0 and _point_on_segment(c, a, b))
        or (o2 == 0 and _point_on_segment(d, a, b))
        or (o3 == 0 and _point_on_segment(a, c, d))
        or (o4 == 0 and _point_on_segment(b, c, d))
    )


def _is_simple_polygon(verts: Sequence[Tuple[int, int]]) -> bool:
    n = len(verts)
    if n < 3 or len(set(verts)) != n or _shoelace_twice(verts) == 0:
        return False
    if any(_orientation(verts[index - 1], verts[index], verts[(index + 1) % n]) == 0 for index in range(n)):
        return False
    for left in range(n):
        a, b = verts[left], verts[(left + 1) % n]
        for right in range(left + 1, n):
            if right in {left, (left + 1) % n} or left in {right, (right + 1) % n}:
                continue
            c, d = verts[right], verts[(right + 1) % n]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def _is_convex_polygon(verts: Sequence[Tuple[int, int]]) -> bool:
    signs = {
        _orientation(verts[index - 1], verts[index], verts[(index + 1) % len(verts)])
        for index in range(len(verts))
    }
    signs.discard(0)
    return len(signs) == 1


def _point_in_polygon(point: Tuple[int, int], verts: Sequence[Tuple[int, int]]) -> bool:
    if any(_point_on_segment(point, verts[index], verts[(index + 1) % len(verts)]) for index in range(len(verts))):
        return False
    y = point[1]
    winding_number = 0
    for index, start in enumerate(verts):
        end = verts[(index + 1) % len(verts)]
        turn = _orientation(start, end, point)
        if start[1] <= y < end[1] and turn > 0:
            winding_number += 1
        elif end[1] <= y < start[1] and turn < 0:
            winding_number -= 1
    return winding_number != 0


def _lattice_point_sets(
    verts: Sequence[Tuple[int, int]], *, grid_radius: int = _GRID_RADIUS
) -> Tuple[List[List[int]], List[List[int]]]:
    boundary: List[List[int]] = []
    interior: List[List[int]] = []
    for y in range(grid_radius, -grid_radius - 1, -1):
        for x in range(-grid_radius, grid_radius + 1):
            point = (x, y)
            if any(_point_on_segment(point, verts[index], verts[(index + 1) % len(verts)]) for index in range(len(verts))):
                boundary.append([x, y])
            elif _point_in_polygon(point, verts):
                interior.append([x, y])
    return boundary, interior


def _pick_interior_count(verts: Sequence[Tuple[int, int]]) -> int:
    boundary_count = sum(
        math.gcd(abs(x1 - x0), abs(y1 - y0))
        for (x0, y0), (x1, y1) in zip(verts, [*verts[1:], verts[0]])
    )
    return (_shoelace_twice(verts) - boundary_count + 2) // 2


def _difficulty_evidence(task: Dict[str, Any]) -> Dict[str, Any]:
    vertex_count = len(task["verts"])
    interior_count = int(task["interior_count"])
    nonvertex_boundary_count = int(task["boundary_count"]) - vertex_count
    if interior_count <= 6 and vertex_count <= 4:
        tier = "minimum"
    elif interior_count >= 16 and vertex_count >= 5:
        tier = "maximum"
    else:
        tier = "typical"
    return {
        "tier": tier,
        "polygon_kind": task["polygon_kind"],
        "vertex_count": vertex_count,
        "boundary_point_count": int(task["boundary_count"]),
        "nonvertex_boundary_point_count": nonvertex_boundary_count,
        "strict_interior_point_count": interior_count,
        "enumerated_lattice_decisions": int(task["boundary_count"]) + interior_count,
    }


def _screen_point(point: Sequence[int]) -> List[float]:
    return [
        float(round(_GRID_CENTER[0] + int(point[0]) * _CELL_SIZE, 3)),
        float(round(_GRID_CENTER[1] - int(point[1]) * _CELL_SIZE, 3)),
    ]


def _distance_point_to_edge(
    point: Tuple[int, int],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _minimum_offedge_clearance(
    verts: Sequence[Tuple[int, int]],
    boundary: Sequence[Tuple[int, int]],
    *,
    grid_radius: int = _GRID_RADIUS,
) -> float:
    """Smallest edge distance over lattice points that are NOT on an edge.

    Every off-edge lattice point (interior or exterior) must keep a visible
    clearance from the outline, otherwise counting strictly-inside points
    becomes an eyesight test near shallow-slope edges (perceptual margin).
    """

    on_edge = {tuple(point) for point in boundary}
    edges = list(zip(verts, [*verts[1:], verts[0]]))
    clearance = float("inf")
    for x in range(-grid_radius, grid_radius + 1):
        for y in range(-grid_radius, grid_radius + 1):
            if (x, y) in on_edge:
                continue
            nearest = min(_distance_point_to_edge((x, y), a, b) for a, b in edges)
            clearance = min(clearance, nearest)
    return clearance


def _minimum_vertex_angle(verts: Sequence[Tuple[int, int]]) -> float:
    smallest = 180.0
    count = len(verts)
    for index in range(count):
        ax, ay = verts[index - 1]
        bx, by = verts[index]
        cx, cy = verts[(index + 1) % count]
        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 <= 1e-9 or n2 <= 1e-9:
            return 0.0
        cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        smallest = min(smallest, math.degrees(math.acos(cosine)))
    return smallest


def _generate_polygon(rng: random.Random, interior_range=(4, 6), vertex_range=(3, 6)) -> Dict[str, Any]:
    want_convex = rng.random() < 0.55
    coordinates = [(x, y) for x in range(-4, 5) for y in range(-4, 5)]
    for _ in range(5000):
        vertex_count = rng.randint(*vertex_range)
        verts = rng.sample(coordinates, vertex_count)
        cx = sum(point[0] for point in verts) / vertex_count
        cy = sum(point[1] for point in verts) / vertex_count
        verts.sort(key=lambda point: math.atan2(point[1] - cy, point[0] - cx))
        if not _is_simple_polygon(verts) or _is_convex_polygon(verts) != want_convex:
            continue
        span_x = max(x for x, _ in verts) - min(x for x, _ in verts)
        span_y = max(y for _, y in verts) - min(y for _, y in verts)
        if span_x < 5 or span_y < 5:
            continue
        # Fatness: the polygon must fill at least a quarter of its bounding
        # box, and no corner may collapse into a needle tip.  Both guards
        # reject the degenerate sliver wedges that made earlier renders
        # unreadable.
        if _shoelace_twice(verts) < 0.5 * span_x * span_y:
            continue
        if _minimum_vertex_angle(verts) < 20.0:
            continue
        boundary, interior = _lattice_point_sets(verts)
        if not (interior_range[0] <= len(interior) <= interior_range[1]):
            continue
        if len(boundary) <= vertex_count or len(boundary) > 24:
            continue
        if len(interior) != _pick_interior_count(verts):
            continue
        # Perceptual margin: every lattice point that is not exactly on the
        # outline stays visibly clear of it, so inside/outside judgements
        # never hinge on a hairline gap.
        if _minimum_offedge_clearance(verts, boundary) < 0.15:
            continue
        return {
            "verts": [list(point) for point in verts],
            "polygon_kind": "convex" if want_convex else "concave",
            "boundary_points": boundary,
            "interior_points": interior,
            "boundary_count": len(boundary),
            "interior_count": len(interior),
            "twice_area": _shoelace_twice(verts),
        }
    raise RuntimeError("Failed to generate a valid lattice polygon.")


class LatticePointsStrictInsidePattern(RenderedTaskGenerator):
    TASK_NAME = "lattice_points_strict_inside"
    TASK_FAMILY = "geometry"
    task_name = "lattice_points_strict_inside"
    task_family = "geometry"
    video_fps = 16

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        image_size: Tuple[int, int] | int | None = None,
        target_size: Tuple[int, int] | int | None = None,
        video_fps: int | None = None,
        generate_videos: bool = True,
        video: bool | None = None,
        unique: bool = True,
        interior_range=(4, 6),
        vertex_range=(3, 6),
    ) -> None:
        del unique
        requested_size = image_size if image_size is not None else target_size
        normalized_size = (
            (int(requested_size), int(requested_size))
            if isinstance(requested_size, int)
            else tuple(requested_size or _CANVAS_SIZE)
        )
        if normalized_size != _CANVAS_SIZE:
            raise ValueError("lattice_points_strict_inside renders are fixed at 512x512")
        if video_fps not in (None, 16):
            raise ValueError("lattice_points_strict_inside videos are fixed at 16 fps")
        seed = random_seed if random_seed is not None else seed
        self.rng = random.Random(seed)
        self.interior_range = tuple(interior_range)
        self.vertex_range = tuple(vertex_range)
        self.video_fps = 16
        self.generate_videos = bool(generate_videos if video is None else video)
        self.image_size = _CANVAS_SIZE
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / "videogym_lattice_points_strict_inside"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    @staticmethod
    def _state_trace(task: Dict[str, Any]) -> List[Dict[str, Any]]:
        # The strict-interior points form one unordered, equal-status evidence
        # set, so the semantic trace has exactly one global marker commit.
        return [
            {"marked_count": 0},
            {"marked_count": int(task["true_count"])},
        ]

    def _generate_task(self) -> Dict[str, Any]:
        task = _generate_polygon(self.rng, self.interior_range, self.vertex_range)
        verts = [tuple(point) for point in task["verts"]]
        replayed_boundary, replayed_interior = _lattice_point_sets(verts)
        replayed_pick_count = _pick_interior_count(verts)
        if replayed_boundary != task["boundary_points"]:
            raise AssertionError("boundary lattice-point replay disagrees with generation")
        if replayed_interior != task["interior_points"]:
            raise AssertionError("strict-interior lattice-point replay disagrees with generation")
        if replayed_pick_count != len(replayed_interior):
            raise AssertionError("point-in-polygon enumeration and Pick cross-check disagree")
        if int(task["interior_count"]) != replayed_pick_count:
            raise AssertionError("saved strict-interior count disagrees with semantic replay")
        true_count = replayed_pick_count
        trap_values = {
            task["boundary_count"],
            task["twice_area"] // 2,
            true_count - 3,
            true_count - 2,
            true_count - 1,
            true_count + 1,
            true_count + 2,
            true_count + 3,
            true_count + task["boundary_count"],
        }
        trap_values = {value for value in trap_values if value >= 0 and value != true_count}
        options = [true_count, *self.rng.sample(sorted(trap_values), 3)]
        self.rng.shuffle(options)
        return {**task, "true_count": true_count, "options": options, "correct_index": options.index(true_count)}

    @staticmethod
    def _geometry(task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "grid_radius": _GRID_RADIUS,
            "grid_center": list(_GRID_CENTER),
            "cell_size": _CELL_SIZE,
            "grid_point_centers": [
                _screen_point([x, y])
                for y in range(_GRID_RADIUS, -_GRID_RADIUS - 1, -1)
                for x in range(-_GRID_RADIUS, _GRID_RADIUS + 1)
            ],
            "polygon_screen_vertices": [_screen_point(point) for point in task["verts"]],
            "boundary_centers": [_screen_point(point) for point in task["boundary_points"]],
            "interior_centers": [_screen_point(point) for point in task["interior_points"]],
            "marker_radius": _MARKER_RADIUS,
            "grid_rgb": list(_GRID_RGB),
            "dot_rgb": list(_DOT_RGB),
            "outline_rgb": list(_OUTLINE_RGB),
            "boundary_rgb": list(_BOUNDARY_RGB),
            "interior_rgb": list(_INTERIOR_RGB),
        }

    def _render_scene(
        self,
        task: Dict[str, Any],
        *,
        marked_count: int = 0,
        marker_progress: float = 1.0,
    ) -> Image.Image:
        factor = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(self.image_size[0], self.image_size[1], factor, background=_BG)
        vertices = [tuple(_screen_point(point)) for point in task["verts"]]
        draw.polygon(vertices, fill=_POLYGON_FILL)
        grid_low = -_GRID_RADIUS * _CELL_SIZE - _GRID_LINE_OVERHANG
        grid_high = _GRID_RADIUS * _CELL_SIZE + _GRID_LINE_OVERHANG
        label_font = load_font(10 * factor)
        for offset in range(-_GRID_RADIUS, _GRID_RADIUS + 1):
            x = _GRID_CENTER[0] + offset * _CELL_SIZE
            y = _GRID_CENTER[1] + offset * _CELL_SIZE
            draw.line((x, _GRID_CENTER[1] + grid_low, x, _GRID_CENTER[1] + grid_high), fill=_GRID_RGB, width=1)
            draw.line((_GRID_CENTER[0] + grid_low, y, _GRID_CENTER[0] + grid_high, y), fill=_GRID_RGB, width=1)
            draw.text(
                (x, _GRID_CENTER[1] + grid_low - 16.0),
                str(offset),
                fill=_TEXT_RGB,
                font=label_font,
                anchor="mm",
            )
            draw.text(
                (_GRID_CENTER[0] + grid_low - 16.0, y),
                str(-offset),
                fill=_TEXT_RGB,
                font=label_font,
                anchor="mm",
            )
        for center in self._geometry(task)["grid_point_centers"]:
            x, y = center
            draw.ellipse((x - 3.5, y - 3.5, x + 3.5, y + 3.5), fill=_DOT_RGB)
        for index, start in enumerate(vertices):
            draw.line((start, vertices[(index + 1) % len(vertices)]), fill=_OUTLINE_RGB, width=4)
        if int(marked_count) > 0:
            opacity = max(0.0, min(1.0, float(marker_progress)))
            alpha = int(round(255 * opacity))
            if alpha > 0:
                marker_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
                marker_draw = ScaledDraw(ImageDraw.Draw(marker_overlay), factor)
                for x, y in self._geometry(task)["interior_centers"]:
                    marker_draw.ellipse(
                        (x - _MARKER_RADIUS, y - _MARKER_RADIUS, x + _MARKER_RADIUS, y + _MARKER_RADIUS),
                        fill=(*_INTERIOR_RGB, alpha),
                        outline=(*_OUTLINE_RGB, alpha),
                        width=1,
                    )
                image = Image.alpha_composite(image.convert("RGBA"), marker_overlay).convert("RGB")
        return finish_scaled_canvas(image, self.image_size)

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        frames: List[np.ndarray] = []
        input_frame = np.asarray(self._render_scene(task), dtype=np.uint8)
        frames.extend(input_frame.copy() for _ in range(5))
        for progress in (0.25, 0.5, 0.75, 1.0):
            frames.append(
                np.asarray(
                    self._render_scene(
                        task,
                        marked_count=int(task["true_count"]),
                        marker_progress=progress,
                    ),
                    dtype=np.uint8,
                )
            )
        final_frame = np.asarray(
            self._render_scene(task, marked_count=int(task["true_count"])),
            dtype=np.uint8,
        )
        frames.extend(final_frame.copy() for _ in range(8))
        frames = [np.asarray(frame, dtype=np.uint8) for frame in pad_frames_to_wan_compatible_length(frames)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    @staticmethod
    def _contract() -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single image shows one light-blue polygon on a full-canvas 9 by 9 lattice with coordinate labels "
                "along the top and left edges. Every small gray dot is one lattice point. Polygon vertices use those "
                "same ordinary dots and are identified only where the polygon outline passes through them; no extra "
                "vertex markers are added."
            ),
            symbol_legend=(
                "The grid-edge numbers identify lattice coordinates.",
            ),
            rules=(
                "A lattice point counts only when it lies inside the polygon and on no polygon edge.",
                "Exclude every vertex and every other lattice point lying on an edge, including points between two vertices.",
                "Count each remaining interior lattice point exactly once; that total is the required count.",
            ),
            modality_outputs={
                "ti2v": "Hold the input, place the complete cyan interior-marker set at its fixed terminal size and position while raising only its shared opacity, then hold the marked scene.",
                "solution_image": "Preserve the lattice, coordinate labels, and polygon; add only one cyan disc on every strict-interior lattice point.",
            },
            preservation_constraints=(
                "Keep every lattice line, lattice point, coordinate label, polygon edge, framing, and camera unchanged.",
                "Only the cyan strict-interior markers may be added.",
            ),
            transition_constraints=(
                "Begin with a readable hold of the exact unmarked input image.",
                "Fade all equal-status strict-interior markers in together at their complete final size; none may appear earlier or later than the rest.",
                "Do not omit, add an invalid point, recolor, remove, move, or regress any marker state.",
            ),
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="strict_uppercase_option_letter_a_to_d",
                verifier_profile="lattice_strict_interior_visible_count_multimodal_hard_gate_v1",
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_full_canvas_lattice_interior_local_opacity_v7",
        )

    @staticmethod
    def _options_sentence(task: Dict[str, Any]) -> str:
        entries = "; ".join(
            f"{_LABELS[index]} = {value}" for index, value in enumerate(task["options"])
        )
        return f"Options: {entries}."

    def _prompts(self, task: Dict[str, Any]) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Count the lattice points strictly inside the polygon and choose the matching numeric option.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        options_sentence = self._options_sentence(task)
        for key in ("ti2v_prompt",):
            prompt = prompts[key]
            if "\n\nRequired Output:" not in prompt:
                raise ValueError("lattice_points_strict_inside reasoning prompt is missing its Required Output section")
            prompts[key] = prompt.replace(
                "\n\nRequired Output:",
                f" {options_sentence}\n\nRequired Output:",
                1,
            )
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid lattice-point prompt contract: {'; '.join(issues)}")
        return contract, prompts

    @staticmethod
    def _authored_reasoning(task: Dict[str, Any]) -> str:
        vertex_lookup = {tuple(point) for point in task["verts"]}
        interior = [tuple(point) for point in task["interior_points"]]
        boundary = [tuple(point) for point in task["boundary_points"]]
        edge_only = [point for point in boundary if point not in vertex_lookup]
        interior_text = ", ".join(f"({x}, {y})" for x, y in interior)
        if edge_only:
            edge_text = ", ".join(f"({x}, {y})" for x, y in edge_only)
            edge_clause = (
                f" Besides the {len(vertex_lookup)} vertices, the outline also passes exactly through "
                f"{edge_text}, so those are boundary points and do not count."
            )
        else:
            edge_clause = (
                f" The outline passes through no lattice point other than the {len(vertex_lookup)} vertices."
            )
        label = _LABELS[task["correct_index"]]
        return (
            f"The polygon is a {task['polygon_kind']} shape whose vertices sit on lattice points, so I test each "
            f"lattice point against the outline.{edge_clause} Scanning the enclosed area row by row, the points "
            f"strictly inside are {interior_text}. That is {task['true_count']} interior points, matching "
            f"Options entry {label} = {task['true_count']}."
        )

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        contract, prompts = self._prompts(task)
        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"
        puzzle = self._render_scene(task)
        solution = self._render_scene(task, **self._state_trace(task)[-1])
        puzzle.save(puzzle_path)
        solution.save(solution_path)
        solution_video_path, video_num_frames = self._render_video(task, video_path)
        correct_option = _LABELS[task["correct_index"]]
        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": {
                **contract.to_dict(),
                "input_count": 1,
                "canvas_size": [512, 512],
            },
            "question": prompts["question"],
            "answer": correct_option,
            "correct_option": correct_option,
            "structured_answer": {
                "correct_option": correct_option,
                "correct_index": task["correct_index"],
                "verts": task["verts"],
                "polygon_kind": task["polygon_kind"],
                "twice_area": task["twice_area"],
                "boundary_points": task["boundary_points"],
                "interior_points": task["interior_points"],
                "boundary_count": task["boundary_count"],
                "interior_count": task["interior_count"],
                "true_count": task["true_count"],
                "options": task["options"],
                "difficulty_evidence": _difficulty_evidence(task),
                "multiple_solution_policy": "exactly one option equals the replayed strict-interior count",
                "replay_rule": "enumerate all lattice points by exact boundary and winding tests, then cross-check the count with Pick's theorem",
                "state_trace": self._state_trace(task),
                "render_geometry": self._geometry(task),
            },
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }
        augment_counting_contract(
            sample,
            verifier_profile="lattice_strict_interior_visible_count_multimodal_hard_gate_v1",
            semantic_source=(
                "visible integer-lattice polygon vertices, fixed 9x9 grid geometry, and the prompt-serialized numeric "
                "options; the verifier independently enumerates boundary and strict-interior points and cross-checks Pick's theorem"
            ),
        )
        sample["structured_answer"]["verifier_contract"]["binary_predicate"] = (
            "strict option parse, independent exact lattice-point replay with a Pick-theorem cross-check, one unique "
            "numeric candidate, one global equal-status strict-interior marker set with synchronized fixed-footprint opacity, "
            "fixed-content preservation, and the declared media contract"
        )
        attach_text_choice_card_inputs(
            sample,
            problem_image=puzzle_path.as_posix(),
            option_labels=_LABELS,
            ti2v_problem_image=puzzle_path.as_posix(),
        )
        return finalize_geometry_sample(sample), puzzle, solution


__all__ = [
    "LatticePointsStrictInsidePattern",
    "_BOUNDARY_RGB",
    "_INTERIOR_RGB",
    "_generate_polygon",
    "_is_simple_polygon",
    "_lattice_point_sets",
    "_pick_interior_count",
]
