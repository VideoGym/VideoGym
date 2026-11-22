(
    "Select the unique candidate label set containing every right triangle from a single problem-only image "
    "of numbered triangles. Exact "
    "integer-vector geometry drives generation, rendering, answers, and downstream semantic replay."
)

from __future__ import annotations

from data.common.base import RenderedTaskGenerator
from data.common.choice_cards import attach_text_choice_card_inputs
from data.common.render_primitives import DOCTRINE_SUPERSAMPLE, stroke_closed_ring
from data.geometry.core.rendered_contract import finalize_geometry_sample

import math
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from PIL import Image, ImageDraw

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.pipeline.media import LETTER_OPTIONS, downsample, load_font
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC


pattern_instructions = {
    "right_triangle_selection": (
        "A clean square canvas shows several numbered colored triangles. The four candidate label sets are "
        "Prompt text values, while the direct scene shows only the triangle field and its red evidence circles."
    ),
}

# Supersample at the family-wide line-quality floor rather
# than the historical 2x, so triangle outlines and label glyphs resolve as
# cleanly here as in the rest of the Geometry family.
_SCALE = DOCTRINE_SUPERSAMPLE
_BACKGROUND_COLOR = (250, 251, 253)
_CARD_COLOR = (255, 255, 255)
_CELL_COLOR = (248, 250, 252)
_GRID_COLOR = (220, 225, 231)
_OUTLINE_COLOR = (31, 36, 43)
_TEXT_COLOR = (43, 49, 57)
_SELECTION_COLOR = (224, 50, 54)
_TRIANGLE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (124, 160, 221),
    (137, 179, 128),
    (201, 139, 95),
    (172, 125, 196),
    (224, 120, 146),
    (107, 171, 194),
    (196, 176, 108),
    (149, 148, 148),
)

_PLOT_BBOX = (18, 18, 494, 494)
_DISTRACTOR_RIGHT_ANGLE_GAP_DEGREES = 15.0
_VERIFIER_PROFILE = "right_triangle_visible_vertex_set_multimodal_hard_gate_v2"
_RENDER_PROFILE = "native_right_triangle_selection_cardless_v1"


def _normalize_canvas_size(value: object) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return 512, 512


def _triangle_centroid(vertices: Sequence[Sequence[int]]) -> Tuple[int, int]:
    return (
        int(round(sum(int(point[0]) for point in vertices) / 3.0)),
        int(round(sum(int(point[1]) for point in vertices) / 3.0)),
    )


def _triangle_dot_products(vertices: Sequence[Sequence[int]]) -> List[int]:
    if len(vertices) != 3:
        return []
    points = [(int(x), int(y)) for x, y in vertices]
    dots: List[int] = []
    for index in range(3):
        previous = points[(index - 1) % 3]
        current = points[index]
        following = points[(index + 1) % 3]
        left = (previous[0] - current[0], previous[1] - current[1])
        right = (following[0] - current[0], following[1] - current[1])
        dots.append(left[0] * right[0] + left[1] * right[1])
    return dots


def _triangle_right_angle_index(vertices: Sequence[Sequence[int]]) -> int | None:
    dots = _triangle_dot_products(vertices)
    return next((index for index, dot in enumerate(dots) if dot == 0), None)


def _triangle_interior_angles(vertices: Sequence[Sequence[int]]) -> List[float]:
    if len(vertices) != 3:
        return []
    points = [(float(x), float(y)) for x, y in vertices]
    angles: List[float] = []
    for index in range(3):
        previous = points[(index - 1) % 3]
        current = points[index]
        following = points[(index + 1) % 3]
        left = (previous[0] - current[0], previous[1] - current[1])
        right = (following[0] - current[0], following[1] - current[1])
        denominator = math.hypot(*left) * math.hypot(*right)
        if denominator == 0.0:
            return []
        cosine = max(-1.0, min(1.0, (left[0] * right[0] + left[1] * right[1]) / denominator))
        angles.append(math.degrees(math.acos(cosine)))
    return angles


def _right_angle_gap_degrees(vertices: Sequence[Sequence[int]]) -> float:
    angles = _triangle_interior_angles(vertices)
    if len(angles) != 3:
        return 0.0
    return min(abs(angle - 90.0) for angle in angles)


def _polygon_area2(vertices: Sequence[Sequence[int]]) -> int:
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    return abs(int(x1) * (int(y2) - int(y3)) + int(x2) * (int(y3) - int(y1)) + int(x3) * (int(y1) - int(y2)))


def _minimum_side_length(vertices: Sequence[Sequence[int]]) -> float:
    return min(
        math.hypot(
            float(vertices[(index + 1) % 3][0]) - float(vertex[0]),
            float(vertices[(index + 1) % 3][1]) - float(vertex[1]),
        )
        for index, vertex in enumerate(vertices)
    )


def _rotated_primitive_directions(max_component: int = 4) -> List[Tuple[int, int]]:
    return [
        (dx, dy)
        for dx in range(1, max_component + 1)
        for dy in range(1, max_component + 1)
        if math.gcd(dx, dy) == 1
    ]


def _answer_schema() -> Dict[str, Any]:
    return {
        "kind": "option_letter",
        "pattern": FOUR_CHOICE_CODEC.capture_pattern,
        "strict_full_match": True,
        "surrounding_whitespace_allowed": False,
        "extra_text_allowed": False,
    }


def _verifier_contract() -> Dict[str, Any]:
    return {
        "profile": _VERIFIER_PROFILE,
        "judged_artifacts": ["text", "final_image", "full_video"],
        "strict_text": True,
        "independent_vertex_dot_product_replay": True,
        "independent_candidate_set_replay": True,
        "full_video_required": True,
        "performance_diagnostics_only": True,
        "automatic_failure_thresholds": False,
    }


# Every sampled triangle count tiles the full plot with exactly that many
# cells (row lengths per row, top to bottom), so no empty placeholder panels
# ever appear and each triangle gets the largest possible cell.
_CELL_LAYOUTS: Dict[int, Tuple[int, ...]] = {
    5: (3, 2),
    6: (3, 3),
    7: (4, 3),
    8: (4, 4),
}


def _layout_cell_boxes(triangle_count: int) -> List[Tuple[int, int, int, int]]:
    rows_spec = _CELL_LAYOUTS[int(triangle_count)]
    left, top, right, bottom = _PLOT_BBOX
    row_count = len(rows_spec)
    y_edges = [int(round(top + (bottom - top) * index / row_count)) for index in range(row_count + 1)]
    boxes: List[Tuple[int, int, int, int]] = []
    for row, cols in enumerate(rows_spec):
        x_edges = [int(round(left + (right - left) * index / cols)) for index in range(cols + 1)]
        boxes.extend(
            (x_edges[col], y_edges[row], x_edges[col + 1], y_edges[row + 1])
            for col in range(cols)
        )
    return boxes


def _render_geometry(triangle_count: int) -> Dict[str, Any]:
    return {
        "canvas_size": [512, 512],
        "plot_bbox": list(_PLOT_BBOX),
        "cell_bboxes": [list(box) for box in _layout_cell_boxes(triangle_count)],
        "selection_rgb": list(_SELECTION_COLOR),
        "selection_width": 5,
    }


def _expected_state_trace(right_indices: Sequence[int], correct_option: str) -> List[Dict[str, Any]]:
    ordered = [int(index) for index in right_indices]
    return [
        {"phase": "input", "selected_triangle_indices": [], "selected_option": None},
        {
            "phase": "global_triangle_evidence",
            "selected_triangle_indices": ordered,
            "selected_option": None,
        },
        {
            "phase": "final",
            "selected_triangle_indices": ordered,
            "selected_option": correct_option,
        },
    ]


def _replay_right_triangle_selection(structured: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    triangles = structured.get("triangles")
    candidates = structured.get("candidates")
    if not isinstance(triangles, list) or not 5 <= len(triangles) <= 8:
        return {}, False
    if not isinstance(candidates, list) or len(candidates) != 4:
        return {}, False
    normalized_triangles: List[Dict[str, Any]] = []
    try:
        for index, triangle in enumerate(triangles):
            vertices = [[int(value) for value in point] for point in triangle["vertices"]]
            if len(vertices) != 3 or any(len(point) != 2 for point in vertices):
                return {}, False
            dots = _triangle_dot_products(vertices)
            zero_indices = [dot_index for dot_index, dot in enumerate(dots) if dot == 0]
            if len(zero_indices) > 1 or _polygon_area2(vertices) <= 0:
                return {}, False
            right_index = zero_indices[0] if zero_indices else None
            centroid = list(_triangle_centroid(vertices))
            selection_radius = int(
                math.ceil(
                    max(
                        math.hypot(int(vertex[0]) - centroid[0], int(vertex[1]) - centroid[1])
                        for vertex in vertices
                    )
                    + 8
                )
            )
            cell_index = int(triangle["cell_index"])
            if not 0 <= cell_index < len(triangles):
                return {}, False
            cell_bbox = _render_geometry(len(triangles))["cell_bboxes"][cell_index]
            left, top, right, bottom = cell_bbox
            if any(not (left <= point[0] <= right and top <= point[1] <= bottom) for point in vertices):
                return {}, False
            if not (
                centroid[0] - selection_radius >= left + 4
                and centroid[0] + selection_radius <= right - 4
                and centroid[1] - selection_radius >= top + 29
                and centroid[1] + selection_radius <= bottom - 4
            ):
                return {}, False
            color = [int(value) for value in triangle["color"]]
            if tuple(color) not in _TRIANGLE_COLORS:
                return {}, False
            normalized_triangles.append(
                {
                    "vertices": vertices,
                    "centroid": centroid,
                    "label": str(index + 1),
                    "label_center": [left + 16, top + 16],
                    "cell_index": cell_index,
                    "cell_bbox": cell_bbox,
                    "right_angle_index": right_index,
                    "right_angle_gap_degrees": float(_right_angle_gap_degrees(vertices)),
                    "interior_angles": [float(value) for value in _triangle_interior_angles(vertices)],
                    "dot_products": dots,
                    "is_right_triangle": right_index is not None,
                    "selection_radius": selection_radius,
                    "color": color,
                }
            )
        candidate_sets = [
            sorted({int(value) for value in candidate["highlighted_triangle_indices"]})
            for candidate in candidates
        ]
    except (KeyError, TypeError, ValueError):
        return {}, False
    if any(not values or any(value < 0 or value >= len(triangles) for value in values) for values in candidate_sets):
        return {}, False
    if len({triangle["cell_index"] for triangle in normalized_triangles}) != len(normalized_triangles):
        return {}, False
    if len({tuple(triangle["color"]) for triangle in normalized_triangles}) != len(normalized_triangles):
        return {}, False
    right_indices = [
        index for index, triangle in enumerate(normalized_triangles) if triangle["right_angle_index"] is not None
    ]
    if not 1 <= len(right_indices) <= 3 or len({tuple(values) for values in candidate_sets}) != 4:
        return {}, False
    matching_indices = [index for index, values in enumerate(candidate_sets) if values == right_indices]
    if len(matching_indices) != 1:
        return {}, False
    correct_index = matching_indices[0]
    correct_option = LETTER_OPTIONS[correct_index]
    expected_candidates = [
        {
            "highlighted_triangle_indices": values,
            "triangle_labels": [str(index + 1) for index in values],
        }
        for values in candidate_sets
    ]
    replay = {
        "triangles": normalized_triangles,
        "right_triangle_indices": right_indices,
        "right_triangle_centroids": [normalized_triangles[index]["centroid"] for index in right_indices],
        "candidate_sets": candidate_sets,
        "candidates": expected_candidates,
        "correct_index": correct_index,
        "correct_option": correct_option,
        "state_trace": _expected_state_trace(right_indices, correct_option),
        "render_geometry": _render_geometry(len(triangles)),
        "answer_schema": _answer_schema(),
        "verifier_contract": _verifier_contract(),
    }
    float_fields_valid = all(
        abs(float(source.get("right_angle_gap_degrees", -1.0)) - target["right_angle_gap_degrees"]) <= 1e-7
        and len(source.get("interior_angles") or []) == 3
        and all(
            abs(float(source_value) - target_value) <= 1e-7
            for source_value, target_value in zip(source.get("interior_angles") or [], target["interior_angles"])
        )
        for source, target in zip(triangles, normalized_triangles)
    )
    metadata_valid = bool(
        float_fields_valid
        and triangles == normalized_triangles
        and candidates == expected_candidates
        and structured.get("right_triangle_indices") == right_indices
        and structured.get("right_triangle_centroids") == replay["right_triangle_centroids"]
        and structured.get("triangle_centroids") == [triangle["centroid"] for triangle in normalized_triangles]
        and structured.get("correct_option") == correct_option
        and structured.get("state_trace") == replay["state_trace"]
        and structured.get("render_geometry") == replay["render_geometry"]
        and structured.get("answer_schema") == replay["answer_schema"]
        and structured.get("verifier_contract") == replay["verifier_contract"]
        and structured.get("multiple_solution_policy") == "exactly_one_candidate_equals_the_complete_right_triangle_label_set"
    )
    return replay, metadata_valid


class RightTriangleSelectionPattern(RenderedTaskGenerator):
    task_name = "right_triangle_selection"
    task_family = "geometry"
    video_fps: int = 16
    generate_videos: bool = True

    def __init__(self, output_dir: str | Path | None = None, **kwargs: Any) -> None:
        image_size = _normalize_canvas_size(kwargs.get("image_size", 512))
        if image_size != (512, 512):
            raise ValueError("right_triangle_selection requires a 512x512 canvas")
        self.image_size = image_size
        self.video_fps = int(kwargs.get("video_fps", self.video_fps))
        self.generate_videos = bool(kwargs.get("generate_videos", kwargs.get("video", self.generate_videos)))
        seed = kwargs.get("random_seed", kwargs.get("seed"))
        self.rng = random.Random(seed)
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / "videogym_right_triangle_selection"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0
        self._rotated_directions = _rotated_primitive_directions()

    def _choose_triangle_colors(self, count: int) -> List[Tuple[int, int, int]]:
        return [tuple(color) for color in self.rng.sample(_TRIANGLE_COLORS, k=count)]

    @staticmethod
    def _selection_radius(vertices: Sequence[Sequence[int]], centroid: Tuple[int, int]) -> int:
        return int(
            math.ceil(
                max(math.hypot(int(vertex[0]) - centroid[0], int(vertex[1]) - centroid[1]) for vertex in vertices) + 8
            )
        )

    @staticmethod
    def _selection_fits_cell(
        centroid: Tuple[int, int],
        radius: int,
        box: Tuple[int, int, int, int],
    ) -> bool:
        left, top, right, bottom = box
        return (
            centroid[0] - radius >= left + 4
            and centroid[0] + radius <= right - 4
            and centroid[1] - radius >= top + 29
            and centroid[1] + radius <= bottom - 4
        )

    def _finish_triangle(
        self,
        vertices: Sequence[Tuple[int, int]],
        box: Tuple[int, int, int, int],
        *,
        expected_right: bool,
    ) -> Dict[str, Any] | None:
        if _polygon_area2(vertices) < 1000 or _minimum_side_length(vertices) < 30.0:
            return None
        angles = _triangle_interior_angles(vertices)
        if len(angles) != 3 or min(angles) < 24.0:
            return None
        right_index = _triangle_right_angle_index(vertices)
        if expected_right != (right_index is not None):
            return None
        right_gap = _right_angle_gap_degrees(vertices)
        if not expected_right and right_gap < _DISTRACTOR_RIGHT_ANGLE_GAP_DEGREES:
            return None
        centroid = _triangle_centroid(vertices)
        radius = self._selection_radius(vertices, centroid)
        if not self._selection_fits_cell(centroid, radius, box):
            return None
        return {
            "vertices": [tuple(point) for point in vertices],
            "centroid": centroid,
            "right_angle_index": right_index,
            "right_angle_gap_degrees": float(right_gap),
            "interior_angles": [float(angle) for angle in angles],
            "dot_products": _triangle_dot_products(vertices),
            "selection_radius": radius,
            "is_right_triangle": expected_right,
        }

    def _make_rotated_right_triangle(self, box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        left, top, right, bottom = box
        for _ in range(1200):
            base_dx, base_dy = self.rng.choice(self._rotated_directions)
            direction = (base_dx * self.rng.choice((-1, 1)), base_dy * self.rng.choice((-1, 1)))
            perpendicular = self.rng.choice(((-direction[1], direction[0]), (direction[1], -direction[0])))
            # Scale the legs with the available cell so larger layouts render
            # larger, easier-to-judge triangles instead of small glyphs
            # floating in empty panels.
            direction_norm = max(1.0, math.hypot(*direction))
            usable = min(right - left - 30, bottom - top - 54)
            scale_cap = max(10, min(24, int(usable / direction_norm) - 2))
            scale_floor = max(8, int(round(scale_cap * 0.55)))
            scale_a = self.rng.randint(scale_floor, scale_cap)
            scale_b = self.rng.randint(scale_floor, scale_cap)
            # Keep the two legs comparable so the right angle never sits at
            # the tip of a needle-thin triangle.
            if not 0.55 <= scale_a / scale_b <= 1.0 / 0.55:
                continue
            offsets = (
                (0, 0),
                (direction[0] * scale_a, direction[1] * scale_a),
                (perpendicular[0] * scale_b, perpendicular[1] * scale_b),
            )
            min_dx = min(point[0] for point in offsets)
            max_dx = max(point[0] for point in offsets)
            min_dy = min(point[1] for point in offsets)
            max_dy = max(point[1] for point in offsets)
            x_min = left + 15 - min_dx
            x_max = right - 15 - max_dx
            y_min = top + 39 - min_dy
            y_max = bottom - 15 - max_dy
            if x_min > x_max or y_min > y_max:
                continue
            anchor_x = self.rng.randint(x_min, x_max)
            anchor_y = self.rng.randint(y_min, y_max)
            vertices = [
                (anchor_x, anchor_y),
                (anchor_x + offsets[1][0], anchor_y + offsets[1][1]),
                (anchor_x + offsets[2][0], anchor_y + offsets[2][1]),
            ]
            triangle = self._finish_triangle(vertices, box, expected_right=True)
            if triangle is not None:
                return triangle
        raise RuntimeError("Failed to generate a clear rotated right triangle.")

    def _make_distractor_triangle(self, box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        left, top, right, bottom = box
        # Size floors grow with the cell so distractors stay visually
        # comparable to the right triangles instead of shrinking into small
        # glyphs surrounded by empty panel space.
        usable = min(right - left - 30, bottom - top - 54)
        side_floor = max(30.0, usable * 0.42)
        area2_floor = max(1000, int(0.16 * usable * usable))
        for _ in range(3000):
            vertices = [
                (
                    self.rng.randint(left + 15, right - 15),
                    self.rng.randint(top + 39, bottom - 15),
                )
                for _ in range(3)
            ]
            if _minimum_side_length(vertices) < side_floor:
                continue
            if _polygon_area2(vertices) < area2_floor:
                continue
            triangle = self._finish_triangle(vertices, box, expected_right=False)
            if triangle is not None:
                return triangle
        raise RuntimeError("Failed to generate a clearly non-right distractor triangle.")

    def _generate_task(self) -> Dict[str, Any]:
        num_triangles = self.rng.randint(5, 8)
        boxes = _layout_cell_boxes(num_triangles)
        num_right = self.rng.randint(1, min(3, num_triangles - 1))
        selected_cells = self.rng.sample(range(len(boxes)), k=num_triangles)
        right_cell_indices = set(self.rng.sample(selected_cells, k=num_right))
        colors = self._choose_triangle_colors(num_triangles)

        triangles: List[Dict[str, Any]] = []
        for triangle_index, (cell_index, color) in enumerate(zip(selected_cells, colors)):
            triangle = (
                self._make_rotated_right_triangle(boxes[cell_index])
                if cell_index in right_cell_indices
                else self._make_distractor_triangle(boxes[cell_index])
            )
            triangle["color"] = list(color)
            triangle["cell_index"] = cell_index
            triangle["cell_bbox"] = list(boxes[cell_index])
            triangle["label"] = str(triangle_index + 1)
            triangle["label_center"] = [boxes[cell_index][0] + 16, boxes[cell_index][1] + 16]
            triangles.append(triangle)

        right_triangle_indices = [
            index for index, triangle in enumerate(triangles) if _triangle_right_angle_index(triangle["vertices"]) is not None
        ]
        return {
            "triangles": triangles,
            "triangle_centroids": [list(triangle["centroid"]) for triangle in triangles],
            "right_triangle_indices": right_triangle_indices,
            "right_triangle_centroids": [list(triangles[index]["centroid"]) for index in right_triangle_indices],
        }

    def _draw_scene(
        self,
        task: Dict[str, Any],
        *,
        selected_count: int = 0,
        selection_opacity: float = 1.0,
        highlight_triangle_indices: Sequence[int] | None = None,
        answer_progress: float = 0.0,
    ) -> Image.Image:
        scale = _SCALE
        width, height = self.image_size
        image = Image.new("RGB", (width * scale, height * scale), color=_BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)
        plot_left, plot_top, plot_right, plot_bottom = [value * scale for value in _PLOT_BBOX]
        draw.rounded_rectangle(
            (plot_left, plot_top, plot_right, plot_bottom),
            radius=18 * scale,
            fill=_CARD_COLOR,
            outline=_GRID_COLOR,
            width=2 * scale,
        )

        for box in _layout_cell_boxes(len(task["triangles"])):
            left, top, right, bottom = [value * scale for value in box]
            draw.rounded_rectangle(
                (left + 5 * scale, top + 5 * scale, right - 5 * scale, bottom - 5 * scale),
                radius=12 * scale,
                fill=_CELL_COLOR,
                outline=(232, 236, 241),
                width=1 * scale,
            )

        for triangle in task["triangles"]:
            vertices = [(int(v[0]) * scale, int(v[1]) * scale) for v in triangle["vertices"]]
            draw.polygon(vertices, fill=tuple(triangle["color"]))
            stroke_closed_ring(draw, vertices, fill=_OUTLINE_COLOR, width=3 * scale)
            label_x, label_y = (int(value) * scale for value in triangle["label_center"])
            draw.ellipse(
                (label_x - 11 * scale, label_y - 11 * scale, label_x + 11 * scale, label_y + 11 * scale),
                fill=_CARD_COLOR,
                outline=_OUTLINE_COLOR,
                width=2 * scale,
            )
            draw.text(
                (label_x, label_y),
                triangle["label"],
                fill=_TEXT_COLOR,
                font=load_font(14 * scale),
                anchor="mm",
            )

        if highlight_triangle_indices is not None:
            selected_indices = [int(index) for index in highlight_triangle_indices]
        else:
            selected_indices = list(task["right_triangle_indices"][:selected_count])

        selection_depth = max(0.0, min(1.0, float(selection_opacity)))
        if selected_indices and selection_depth > 0.0:
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            for triangle_index in selected_indices:
                triangle = task["triangles"][triangle_index]
                cx = int(triangle["centroid"][0]) * scale
                cy = int(triangle["centroid"][1]) * scale
                radius = int(triangle["selection_radius"]) * scale
                overlay_draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    outline=(*_SELECTION_COLOR, 255),
                    width=5 * scale,
                )
            alpha = overlay.getchannel("A").point(
                lambda value: int(round(value * selection_depth))
            )
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

        # The direct artifact has no answer text or candidate border.  The
        # answer_progress argument remains accepted for renderer compatibility
        # but intentionally has no visual effect.

        return downsample(image, scale)

    def _render_video(self, task: Dict[str, Any], *, output_path: Path) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None

        frames: List[np.ndarray] = []
        puzzle_frame = np.asarray(self._draw_scene(task).convert("RGB"), dtype=np.uint8)
        frames.extend(puzzle_frame.copy() for _ in range(5))
        for opacity in (0.25, 0.50, 0.75, 1.0):
            frame = self._draw_scene(
                task,
                selected_count=len(task["right_triangle_indices"]),
                selection_opacity=opacity,
            )
            frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
        for progress_step in range(1, 5):
            frame = self._draw_scene(
                task,
                selected_count=len(task["right_triangle_indices"]),
                answer_progress=progress_step / 4.0,
            )
            frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
        solution_frame = np.asarray(
            self._draw_scene(
                task,
                selected_count=len(task["right_triangle_indices"]),
                answer_progress=1.0,
            ).convert("RGB"),
            dtype=np.uint8,
        )
        frames.extend(solution_frame.copy() for _ in range(4))
        expected_count = 17
        if len(frames) != expected_count or (len(frames) - 1) % 4 != 0:
            raise AssertionError("right_triangle_selection video violates its 17-frame phase contract")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    @staticmethod
    def _contract() -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single image shows a centered field of numbered colored triangles."
            ),
            symbol_legend=(
                "Each white numeric badge identifies exactly one triangle.",
            ),
            rules=(
                "Judge whether one visible corner is unmistakably perpendicular: its two incident sides meet as a clear right angle. Every non-right corner is separated from 90 degrees by a clearly visible angular margin, so no borderline visual judgment is required.",
                "The complete right-triangle label set collects the numeric labels of every triangle with such a perpendicular corner and of no other triangle.",
            ),
            modality_outputs={
                "ti2v": "Hold the input, place all correct red triangle circles at their complete fixed terminal geometry and raise only their shared local opacity, then hold the solved triangle scene.",
                "solution_image": "Draw complete red circular outlines around every and only the exact right triangles. Do not add answer text or a selection border.",
            },
            preservation_constraints=(
                "Keep the canvas, triangle vertices, fills, outlines, numeric labels, spacing, framing, and camera unchanged.",
                "Only the centered red evidence circles may be added; all triangles, labels, layout, and camera stay fixed.",
                "Keep each circle inside its triangle field without covering the numeric badge.",
            ),
            transition_constraints=(
                "Begin with an unmarked hold; all correct circles occupy their complete terminal radii and line widths from their first visible frame and share one local opacity progression, with no circle order.",
                "Keep non-right triangles unmarked and never regress a committed circle.",
            ),
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="uppercase_option_letter",
                verifier_profile=_VERIFIER_PROFILE,
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile=_RENDER_PROFILE,
        )

    @staticmethod
    def _options_sentence(task: Dict[str, Any]) -> str:
        """Serialize the complete sample-specific candidate sets as Prompt text."""

        entries = "; ".join(
            f"{LETTER_OPTIONS[index]} = "
            + "{"
            + ", ".join(task["triangles"][int(value)]["label"] for value in subset)
            + "}"
            for index, subset in enumerate(task["candidate_subsets"])
        )
        return f"Options: {entries}."

    def _build_prompts(self, task: Dict[str, Any]) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Choose the unique candidate label set containing every and only the right triangles.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid right-triangle-selection prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _build_candidate_subsets(
        self,
        triangles: Sequence[Dict[str, Any]],
    ) -> Tuple[Tuple[int, ...], List[Tuple[int, ...]]]:
        """Build one exact set and three distinct near-miss sets."""
        right_indices = tuple(
            sorted(index for index, triangle in enumerate(triangles) if _triangle_right_angle_index(triangle["vertices"]) is not None)
        )
        non_right_indices = [
            index for index, triangle in enumerate(triangles) if _triangle_right_angle_index(triangle["vertices"]) is None
        ]
        if not right_indices or not non_right_indices:
            raise RuntimeError("Task must contain both right triangles and non-right triangles.")

        distractor_sets: List[Tuple[int, ...]] = []
        seen: set[Tuple[int, ...]] = {right_indices}
        strategies = ("swap_one", "add_extra", "drop_one", "replace_all")
        for _ in range(500):
            if len(distractor_sets) >= 3:
                break
            strategy = self.rng.choice(strategies)
            candidate: Tuple[int, ...]
            if strategy == "swap_one":
                values = list(right_indices)
                values[self.rng.randrange(len(values))] = self.rng.choice(non_right_indices)
                candidate = tuple(sorted(set(values)))
            elif strategy == "add_extra":
                candidate = tuple(sorted((*right_indices, self.rng.choice(non_right_indices))))
            elif strategy == "drop_one" and len(right_indices) >= 2:
                values = list(right_indices)
                values.pop(self.rng.randrange(len(values)))
                candidate = tuple(sorted(values))
            elif strategy == "replace_all":
                count = min(len(non_right_indices), max(1, len(right_indices)))
                candidate = tuple(sorted(self.rng.sample(non_right_indices, k=count)))
            else:
                continue
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            distractor_sets.append(candidate)
        if len(distractor_sets) != 3:
            raise RuntimeError("Failed to build three unique right-triangle distractor sets.")
        return right_indices, distractor_sets


    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        correct_subset, distractor_subsets = self._build_candidate_subsets(task["triangles"])
        candidate_subsets: List[Tuple[int, ...]] = [correct_subset, *distractor_subsets]
        self.rng.shuffle(candidate_subsets)
        correct_index = candidate_subsets.index(correct_subset)
        correct_option = LETTER_OPTIONS[correct_index]
        task["candidate_subsets"] = [list(subset) for subset in candidate_subsets]
        task["correct_index"] = correct_index
        task["correct_option"] = correct_option
        contract, prompts = self._build_prompts(task)

        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"

        puzzle_image = self._draw_scene(task)
        solution_image = self._draw_scene(
            task,
            selected_count=len(task["right_triangle_indices"]),
            answer_progress=1.0,
        )
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        FOUR_CHOICE_CODEC.format(correct_option)
        solution_video_path, video_num_frames = self._render_video(task, output_path=video_path)

        structured_triangles = []
        for triangle in task["triangles"]:
            structured_triangles.append(
                {
                    "vertices": [list(vertex) for vertex in triangle["vertices"]],
                    "centroid": list(triangle["centroid"]),
                    "label": triangle["label"],
                    "label_center": triangle["label_center"],
                    "cell_index": triangle["cell_index"],
                    "cell_bbox": triangle["cell_bbox"],
                    "right_angle_index": triangle["right_angle_index"],
                    "right_angle_gap_degrees": triangle["right_angle_gap_degrees"],
                    "interior_angles": triangle["interior_angles"],
                    "dot_products": triangle["dot_products"],
                    "is_right_triangle": triangle["is_right_triangle"],
                    "selection_radius": triangle["selection_radius"],
                    "color": triangle["color"],
                }
            )

        structured_answer: Dict[str, Any] = {
            "correct_option": correct_option,
            "candidates": [
                {
                    "highlighted_triangle_indices": list(subset),
                    "triangle_labels": [task["triangles"][int(index)]["label"] for index in subset],
                }
                for subset in candidate_subsets
            ],
            "true_coordinate": {
                "right_triangle_indices": list(correct_subset),
                "right_triangle_centroids": task["right_triangle_centroids"],
                "correct_option": correct_option,
            },
            "triangle_centroids": task["triangle_centroids"],
            "right_triangle_indices": list(correct_subset),
            "right_triangle_centroids": task["right_triangle_centroids"],
            "triangles": structured_triangles,
            "distractor_min_right_angle_gap_degrees": _DISTRACTOR_RIGHT_ANGLE_GAP_DEGREES,
            "render_geometry": _render_geometry(len(task["triangles"])),
            "state_trace": _expected_state_trace(correct_subset, correct_option),
            "answer_schema": _answer_schema(),
            "verifier_contract": _verifier_contract(),
            "multiple_solution_policy": "exactly_one_candidate_equals_the_complete_right_triangle_label_set",
        }
        _, replay_valid = _replay_right_triangle_selection(structured_answer)
        if not replay_valid:
            raise RuntimeError("Generated right-triangle-selection sample failed independent semantic replay.")

        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": {
                **contract.to_dict(),
                "input_media_policy": "text_choice_cards_v1",
                "input_count_by_modality": { "ti2v": 1},
            },
            "question": prompts["question"],
            "answer": correct_option,
            "correct_option": correct_option,
            "structured_answer": structured_answer,
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": [],
            "option_images": [],
            "option_image_paths": [],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }
        attach_text_choice_card_inputs(
            sample,
            problem_image=puzzle_path.as_posix(),
            option_labels=LETTER_OPTIONS[:4],
            ti2v_problem_image=puzzle_path.as_posix(),
        )
        return finalize_geometry_sample(sample), puzzle_image, solution_image


__all__ = [
    "RightTriangleSelectionPattern",
    "_DISTRACTOR_RIGHT_ANGLE_GAP_DEGREES",
    "_PLOT_BBOX",
    "_SELECTION_COLOR",
    "_VERIFIER_PROFILE",
    "_expected_state_trace",
    "_replay_right_triangle_selection",
    "_right_angle_gap_degrees",
    "_triangle_dot_products",
    "_triangle_interior_angles",
    "_triangle_right_angle_index",
    "np",
]
