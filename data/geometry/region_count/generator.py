"""Count connected bounded faces using circle topology and raster replay.

ID samples use one connected crossing network. Nested/disconnected samples
exercise the same face definition on multiple boundary components, including
faces with holes. Tangencies, coincident circles and triple crossings are excluded.
"""

from __future__ import annotations

from data.common.base import RenderedTaskGenerator
from data.geometry.core.rendered_contract import finalize_geometry_sample

import math
import random
import tempfile
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.pipeline.media import LETTER_OPTIONS
from data.common.choice_cards import attach_text_choice_card_inputs
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC


pattern_instructions = {
    "region_count": (
        "A full-canvas scene shows an arrangement of closed curves. Count every connected bounded "
        "face enclosed by the visible boundaries and exclude the one unbounded outside face."
    ),
}

Circle = Tuple[float, float, float]
Point = Tuple[float, float]

_LABELS = ("A", "B", "C", "D")
_LOGICAL_SIZE = 240

_BG = (255, 255, 255)
_PANEL = (255, 255, 255)
_INTERSECTION = (18, 166, 182)
_CURVE_COLORS = (
    (45, 99, 163),
    (179, 71, 83),
    (55, 139, 103),
    (127, 82, 169),
)
_REGION_COLORS = (
    (255, 226, 154),
    (190, 228, 242),
    (217, 205, 241),
    (198, 235, 211),
    (248, 205, 218),
)

# Square full-canvas content box: uniform logical-to-screen scale keeps
# circles round while the arrangement fills the whole 512x512 scene.
_SOURCE_BOX = (36, 36, 476, 476)

_VERIFIER_PROFILE = "region_count_visible_bounded_face_multimodal_hard_gate_v1"
_RENDER_PROFILE = "native_full_canvas_bounded_face_global_reveal_v5"
_MAX_BOUNDED_REGIONS = 7


def _video_frame_count(region_count: int) -> int:
    """One global face-set reveal between the two endpoint holds."""

    del region_count
    return 17


def _answer_schema(region_count: int) -> Dict[str, Any]:
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
        "independent_circle_topology_replay": True,
        "independent_raster_face_replay": True,
        "full_video_required": True,
        "performance_diagnostics_only": True,
        "automatic_failure_thresholds": False,
    }


def _expected_state_trace(correct_option: str, region_count: int) -> List[Dict[str, Any]]:
    del correct_option
    return [
        {
            "phase": "input",
            "filled_regions": 0,
            "global_fill_progress": 0.0,
        },
        {
            "phase": "bounded_faces_complete",
            "filled_regions": int(region_count),
            "global_fill_progress": 1.0,
        },
    ]


def _render_geometry(region_count: int) -> Dict[str, Any]:
    frame_count = _video_frame_count(region_count)
    return {
        "canvas_size": [512, 512],
        "logical_source_size": _LOGICAL_SIZE,
        "source_box": list(_SOURCE_BOX),
        "region_rgbs": [list(color) for color in _REGION_COLORS],
        "curve_rgbs": [list(color) for color in _CURVE_COLORS],
        "video_frame_count": frame_count,
        "video_fps": 16,
        "maximum_bounded_regions": _MAX_BOUNDED_REGIONS,
        "phase_frames": {
            "input_hold": [1, 5],
            "global_bounded_face_reveal": [6, 9],
            "final_hold": [10, frame_count],
        },
    }


def _normalise_circles(circles: Iterable[Sequence[float]]) -> Tuple[Circle, ...]:
    return tuple((float(circle[0]), float(circle[1]), float(circle[2])) for circle in circles)


def _circle_pair_geometry(circle_a: Sequence[float], circle_b: Sequence[float]) -> Dict[str, Any]:
    ax, ay, ar = (float(value) for value in circle_a)
    bx, by, br = (float(value) for value in circle_b)
    dx, dy = bx - ax, by - ay
    distance = math.hypot(dx, dy)
    if distance < 1e-9 and abs(ar - br) < 1e-9:
        return {"relation": "coincident", "distance": distance, "points": []}
    outer_gap = distance - (ar + br)
    inner_gap = distance - abs(ar - br)
    if abs(outer_gap) < 1e-7:
        return {"relation": "external_tangent", "distance": distance, "points": []}
    if abs(inner_gap) < 1e-7:
        return {"relation": "internal_tangent", "distance": distance, "points": []}
    if outer_gap > 0:
        return {"relation": "disjoint", "distance": distance, "points": []}
    if inner_gap < 0:
        return {"relation": "contained", "distance": distance, "points": []}
    if distance <= 0:
        return {"relation": "coincident", "distance": distance, "points": []}
    along = (ar * ar - br * br + distance * distance) / (2.0 * distance)
    height_sq = ar * ar - along * along
    if height_sq <= 0:
        return {"relation": "tangent", "distance": distance, "points": []}
    height = math.sqrt(height_sq)
    mid_x = ax + along * dx / distance
    mid_y = ay + along * dy / distance
    offset_x = -dy * height / distance
    offset_y = dx * height / distance
    points = [(mid_x + offset_x, mid_y + offset_y), (mid_x - offset_x, mid_y - offset_y)]
    return {"relation": "intersect_two", "distance": distance, "points": points}


def _analyze_arrangement(
    circles: Iterable[Sequence[float]],
    *,
    require_connected: bool = False,
    minimum_clearance: float = 5.0,
) -> Dict[str, Any]:
    normalized = _normalise_circles(circles)
    if not 2 <= len(normalized) <= 4:
        raise ValueError("region_count requires two to four circles")
    for cx, cy, radius in normalized:
        if not 34 <= radius <= 68:
            raise ValueError("region_count circle radius must stay in the readable range")
        if cx - radius < 10 or cy - radius < 10 or cx + radius > _LOGICAL_SIZE - 10 or cy + radius > _LOGICAL_SIZE - 10:
            raise ValueError("region_count circles must stay inside the source frame with a safe margin")

    parent = list(range(len(normalized)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    pair_relations: List[Dict[str, Any]] = []
    all_points: List[Tuple[Point, Tuple[int, int]]] = []
    intersections_per_circle = [0 for _ in normalized]
    intersecting_pairs = 0
    for left, right in combinations(range(len(normalized)), 2):
        geometry = _circle_pair_geometry(normalized[left], normalized[right])
        relation = str(geometry["relation"])
        distance = float(geometry["distance"])
        ar, br = normalized[left][2], normalized[right][2]
        outer_clearance = abs(distance - (ar + br))
        inner_clearance = abs(distance - abs(ar - br))
        if relation in {"coincident", "external_tangent", "internal_tangent", "tangent"}:
            raise ValueError("region_count excludes coincident or tangent circles")
        if min(outer_clearance, inner_clearance) < minimum_clearance:
            raise ValueError("region_count excludes near-tangent circle pairs")
        points = [(float(point[0]), float(point[1])) for point in geometry["points"]]
        if relation == "intersect_two":
            intersecting_pairs += 1
            intersections_per_circle[left] += 2
            intersections_per_circle[right] += 2
            union(left, right)
            all_points.extend((point, (left, right)) for point in points)
        pair_relations.append(
            {
                "circles": [left, right],
                "relation": relation,
                "intersection_points": [[round(point[0], 6), round(point[1], 6)] for point in points],
            }
        )

    for index, (point, pair) in enumerate(all_points):
        for other_point, other_pair in all_points[index + 1 :]:
            if math.hypot(point[0] - other_point[0], point[1] - other_point[1]) < minimum_clearance:
                if pair != other_pair:
                    raise ValueError("region_count excludes triple or visually merged intersections")
                raise ValueError("region_count excludes near-tangent paired intersections")

    component_count = len({find(index) for index in range(len(normalized))})
    if require_connected and component_count != 1:
        raise ValueError("region_count generated arrangements must form one connected crossing network")
    isolated_circles = sum(count == 0 for count in intersections_per_circle)
    actual_vertices = len(all_points)
    effective_vertices = actual_vertices + isolated_circles
    edge_count = sum(count if count else 1 for count in intersections_per_circle)
    bounded_count = edge_count - effective_vertices + component_count
    if bounded_count <= 0:
        raise ValueError("region_count arrangement must create at least one bounded face")
    return {
        "circle_count": len(normalized),
        "pair_relations": pair_relations,
        "intersecting_pair_count": intersecting_pairs,
        "intersection_points": [
            [round(point[0], 6), round(point[1], 6)]
            for point, _pair in sorted(all_points, key=lambda item: (item[0][1], item[0][0]))
        ],
        "intersections_per_circle": intersections_per_circle,
        "actual_vertex_count": actual_vertices,
        "effective_vertex_count": effective_vertices,
        "edge_count": edge_count,
        "component_count": component_count,
        "bounded_region_count": bounded_count,
        "unbounded_region_count": 1,
    }


def _raster_bounded_regions(
    circles: Iterable[Sequence[float]],
    *,
    resolution: int = 240,
    return_labels: bool = False,
) -> Dict[str, Any]:
    normalized = _normalise_circles(circles)
    scale = resolution / float(_LOGICAL_SIZE)
    barrier_image = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(barrier_image)
    line_width = max(3, int(round(resolution / 100.0)))
    for cx, cy, radius in normalized:
        draw.ellipse(
            (
                (cx - radius) * scale,
                (cy - radius) * scale,
                (cx + radius) * scale,
                (cy + radius) * scale,
            ),
            outline=255,
            width=line_width,
        )
    barrier = np.asarray(barrier_image, dtype=np.uint8) > 0
    visited = np.zeros((resolution, resolution), dtype=bool)
    labels = np.zeros((resolution, resolution), dtype=np.int16)
    bounded: List[Dict[str, Any]] = []
    # Rasterized antialiased-scale contours can leave one-pixel diagonal gaps
    # at some resolutions, creating a fake one-pixel "face" that does not
    # exist in the analytic circle arrangement.  Legitimate generated faces
    # are required to be at least 90 pixels at 240px resolution, so this much
    # smaller scale-normalized floor removes only discretization artifacts.
    minimum_component_area = max(4, int(round(12.0 * (resolution / 240.0) ** 2)))
    for start_y in range(resolution):
        for start_x in range(resolution):
            if barrier[start_y, start_x] or visited[start_y, start_x]:
                continue
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            pixels: List[Tuple[int, int]] = []
            touches_border = False
            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))
                if y in {0, resolution - 1} or x in {0, resolution - 1}:
                    touches_border = True
                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= next_y < resolution
                        and 0 <= next_x < resolution
                        and not barrier[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
            if touches_border or len(pixels) < minimum_component_area:
                continue
            mean_y = sum(pixel[0] for pixel in pixels) / len(pixels)
            mean_x = sum(pixel[1] for pixel in pixels) / len(pixels)
            bounded.append(
                {
                    "pixels": pixels,
                    "area": len(pixels),
                    "centroid": [mean_x * _LOGICAL_SIZE / resolution, mean_y * _LOGICAL_SIZE / resolution],
                }
            )
    bounded.sort(key=lambda region: (region["centroid"][1], region["centroid"][0]))
    summaries: List[Dict[str, Any]] = []
    for index, region in enumerate(bounded, start=1):
        for y, x in region["pixels"]:
            labels[y, x] = index
        summaries.append(
            {
                "index": index,
                "centroid": [round(float(value), 4) for value in region["centroid"]],
                "pixel_area": int(region["area"]),
            }
        )
    result: Dict[str, Any] = {"count": len(summaries), "regions": summaries, "barrier_width": line_width}
    if return_labels:
        result["labels"] = labels
    return result




def _validate_task(task: Dict[str, Any]) -> None:
    analysis = _analyze_arrangement(task["circles"])
    raster = _raster_bounded_regions(task["circles"], resolution=240)
    if raster["count"] != analysis["bounded_region_count"]:
        raise ValueError("region_count raster face enumeration must agree with planar topology")
    if raster["regions"] and min(int(region["pixel_area"]) for region in raster["regions"]) < 90:
        raise ValueError("region_count bounded faces must remain visually readable")
    if task.get("analysis") != analysis:
        raise ValueError("region_count saved topology does not replay from circle primitives")
    if task.get("bounded_regions") != raster["regions"]:
        raise ValueError("region_count saved bounded faces do not replay from circle primitives")
    options = [int(value) for value in task["options"]]
    expected = int(analysis["bounded_region_count"])
    if len(options) != 4 or len(set(options)) != 4:
        raise ValueError("region_count requires four distinct numeric options")
    if options.count(expected) != 1 or int(task["correct_index"]) != options.index(expected):
        raise ValueError("region_count requires one uniquely indexed correct option")


def _replay_region_count(structured: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    try:
        raw_circles = structured.get("circles")
        raw_options = structured.get("options")
        if not isinstance(raw_circles, list) or not isinstance(raw_options, list):
            return {}, False
        circles = _normalise_circles(raw_circles)
        options = [int(value) for value in raw_options]
        if len(options) != 4 or len(set(options)) != 4:
            return {}, False
        analysis = _analyze_arrangement(circles)
        raster = _raster_bounded_regions(circles, resolution=240)
        correct_count = int(analysis["bounded_region_count"])
        if raster["count"] != correct_count or options.count(correct_count) != 1:
            return {}, False
        correct_index = options.index(correct_count)
        correct_option = _LABELS[correct_index]
        circles_payload = [[round(value, 6) for value in circle] for circle in circles]
        candidates = [
            {"label": _LABELS[index], "value": value, "is_correct": index == correct_index}
            for index, value in enumerate(options)
        ]
        state_trace = _expected_state_trace(correct_option, correct_count)
        geometry = _render_geometry(correct_count)
        answer_schema = _answer_schema(correct_count)
        verifier_contract = _verifier_contract()
        distractor_counts = [value for index, value in enumerate(options) if index != correct_index]
        task_meta = {
            "circles": circles_payload,
            "correct_count": correct_count,
            "distractor_counts": distractor_counts,
            "correct_value": correct_count,
            "distractor_values": distractor_counts,
            "candidate_values_by_label": {
                label: options[index] for index, label in enumerate(_LABELS)
            },
            "analysis": analysis,
            "bounded_regions": raster["regions"],
        }
        replay = {
            "circles": circles_payload,
            "options": options,
            "analysis": analysis,
            "bounded_regions": raster["regions"],
            "correct_count": correct_count,
            "correct_index": correct_index,
            "correct_option": correct_option,
            "candidates": candidates,
            "state_trace": state_trace,
            "render_geometry": geometry,
            "answer_schema": answer_schema,
            "verifier_contract": verifier_contract,
            "expected_video_frame_count": _video_frame_count(correct_count),
            "task_meta": task_meta,
        }
        metadata_valid = bool(
            structured.get("circles") == circles_payload
            and structured.get("options") == options
            and structured.get("analysis") == analysis
            and structured.get("bounded_regions") == raster["regions"]
            and structured.get("correct_count") == correct_count
            and structured.get("correct_index") == correct_index
            and structured.get("correct_option") == correct_option
            and structured.get("candidates") == candidates
            and structured.get("state_trace") == state_trace
            and structured.get("render_geometry") == geometry
            and structured.get("answer_schema") == answer_schema
            and structured.get("verifier_contract") == verifier_contract
            and structured.get("expected_video_frame_count") == _video_frame_count(correct_count)
            and structured.get("task_meta") == task_meta
            and structured.get("multiple_solution_policy") == "unique"
        )
        return replay, metadata_valid
    except (KeyError, TypeError, ValueError):
        return {}, False


def _random_arrangement(rng: random.Random, circle_count: int) -> List[Circle]:
    """Chain-sample circles so multi-circle connected arrangements are common.

    Independent uniform placement almost never yields a connected 3-4 circle
    crossing network, which silently collapsed the task to the two-circle
    three-face template.  Each circle after the first is therefore anchored at
    a crossing distance from one already-placed circle; general-position and
    connectivity rejection still applies downstream.
    """

    circles: List[Circle] = []
    for _ in range(circle_count):
        radius = float(rng.randint(42, 62))
        if not circles:
            circles.append(
                (
                    float(rng.randint(int(radius + 13), int(_LOGICAL_SIZE - radius - 13))),
                    float(rng.randint(int(radius + 13), int(_LOGICAL_SIZE - radius - 13))),
                    radius,
                )
            )
            continue
        anchor_x, anchor_y, anchor_radius = circles[rng.randrange(len(circles))]
        low = abs(anchor_radius - radius) + 12.0
        high = anchor_radius + radius - 12.0
        if high <= low:
            distance = (abs(anchor_radius - radius) + anchor_radius + radius) / 2.0
        else:
            distance = rng.uniform(low, high)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        # Integer centers round-trip exactly through the canonical payload,
        # so the metadata replay recomputes the identical analysis.
        circles.append(
            (
                float(round(anchor_x + distance * math.cos(angle))),
                float(round(anchor_y + distance * math.sin(angle))),
                radius,
            )
        )
    # Recenter the sampled cluster on the logical square so the rendered
    # arrangement never hugs one side of the source panel.  Translation
    # preserves the region topology; the clamp keeps every circle within the
    # validator's safe margin.
    min_x = min(cx - radius for cx, _cy, radius in circles)
    max_x = max(cx + radius for cx, _cy, radius in circles)
    min_y = min(cy - radius for _cx, cy, radius in circles)
    max_y = max(cy + radius for _cx, cy, radius in circles)
    desired_dx = _LOGICAL_SIZE / 2.0 - (min_x + max_x) / 2.0
    desired_dy = _LOGICAL_SIZE / 2.0 - (min_y + max_y) / 2.0
    dx = max(11.0 - min_x, min(desired_dx, (_LOGICAL_SIZE - 11.0) - max_x))
    dy = max(11.0 - min_y, min(desired_dy, (_LOGICAL_SIZE - 11.0) - max_y))
    return [(cx + dx, cy + dy, radius) for cx, cy, radius in circles]


class RegionCountPattern(RenderedTaskGenerator):
    task_name = "region_count"
    task_family = "geometry"
    video_fps = 16
    generate_videos = True

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        image_size: int | Sequence[int] = 512,
        circle_count: int | None = None,
        nested_disconnected: bool = False,
        video_fps: int = 16,
        generate_videos: bool | None = None,
        video: bool | None = None,
        **_kwargs: Any,
    ) -> None:
        size = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(int(value) for value in image_size)
        if size != (512, 512):
            raise ValueError("region_count requires a 512x512 canvas")
        if int(video_fps) != 16:
            raise ValueError("region_count requires video_fps=16")
        if circle_count is not None and int(circle_count) not in {2, 3, 4}:
            raise ValueError("region_count circle_count must be 2, 3, or 4")
        resolved_seed = random_seed if random_seed is not None else seed
        self.rng = random.Random(resolved_seed)
        self.image_size = size
        self.circle_count = int(circle_count) if circle_count is not None else None
        self.nested_disconnected = bool(nested_disconnected)
        self.video_fps = 16
        if generate_videos is None:
            generate_videos = self.generate_videos if video is None else bool(video)
        self.generate_videos = bool(generate_videos)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_region_count"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    def _generate_task(self) -> Dict[str, Any]:
        # Commit to one sampled circle count for most of the retry budget so
        # rejection difficulty cannot silently skew the population toward the
        # easiest two-circle template; a late fallback keeps seeds solvable.
        committed_count = self.circle_count or self.rng.choice((2, 3, 3, 3, 4, 4))
        for _attempt in range(2400):
            circle_count = (
                committed_count
                if _attempt < 1600 or self.circle_count
                else self.rng.choice((2, 3, 3, 3, 4, 4))
            )
            if self.nested_disconnected:
                radius = self.rng.uniform(65, 68)
                inner = self.rng.uniform(34, 36)
                offset = self.rng.uniform(14, 18)
                circles = [(82, 120, radius), (82-offset, 120, inner), (82+offset, 120, inner), (194, 120, 34)]
                angle = self.rng.uniform(0, math.tau)
                circles = [(120+(x-120)*math.cos(angle)-(y-120)*math.sin(angle),
                            120+(x-120)*math.sin(angle)+(y-120)*math.cos(angle), r) for x,y,r in circles]
            else:
                circles = _random_arrangement(self.rng, circle_count)
            circles = [tuple(round(v, 6) for v in circle) for circle in circles]
            try:
                analysis = _analyze_arrangement(circles)
            except ValueError:
                continue
            if not self.nested_disconnected and (analysis["component_count"] != 1 or any(p["relation"] == "contained" for p in analysis["pair_relations"])):
                continue
            raster = _raster_bounded_regions(circles, resolution=240)
            if raster["count"] != analysis["bounded_region_count"]:
                continue
            if not raster["regions"] or min(int(region["pixel_area"]) for region in raster["regions"]) < 90:
                continue
            correct_count = int(analysis["bounded_region_count"])
            if correct_count > _MAX_BOUNDED_REGIONS:
                continue
            distractor_pool = {
                correct_count - 2,
                correct_count - 1,
                correct_count + 1,
                correct_count + 2,
                int(analysis["actual_vertex_count"]),
                int(analysis["intersecting_pair_count"]),
                correct_count + 1,
            }
            distractor_pool = {value for value in distractor_pool if value > 0 and value != correct_count}
            if len(distractor_pool) < 3:
                continue
            options = [correct_count, *self.rng.sample(sorted(distractor_pool), 3)]
            self.rng.shuffle(options)
            task = {
                "circles": circles,
                "analysis": analysis,
                "bounded_regions": raster["regions"],
                "options": options,
                "correct_index": options.index(correct_count),
            }
            _validate_task(task)
            return task
        raise RuntimeError("region_count: failed to generate a valid readable arrangement")

    @staticmethod
    def _state_trace(region_count: int, correct_option: str = "A") -> List[Dict[str, Any]]:
        return _expected_state_trace(correct_option, region_count)

    _RASTER_RENDER_RESOLUTION = 880
    _raster_label_cache: Dict[Tuple[Tuple[float, float, float], ...], Dict[str, Any]] = {}

    @classmethod
    def _cached_render_raster(cls, circles: Any) -> Dict[str, Any]:
        key = _normalise_circles(circles)
        cached = cls._raster_label_cache.get(key)
        if cached is None:
            cached = _raster_bounded_regions(
                key, resolution=cls._RASTER_RENDER_RESOLUTION, return_labels=True
            )
            cls._raster_label_cache.clear()
            cls._raster_label_cache[key] = cached
        return cached

    @classmethod
    def _render_source_regions(
        cls,
        image: Image.Image,
        factor: int,
        task: Dict[str, Any],
        filled_regions: int,
        global_fill_progress: float = 1.0,
    ) -> None:
        left, top, right, bottom = _SOURCE_BOX
        target = (right - left) * factor
        raster = cls._cached_render_raster(task["circles"])
        labels = raster["labels"]
        resolution = labels.shape[0]
        pixels = np.full((resolution, resolution, 3), _PANEL, dtype=np.uint8)
        progress = max(0.0, min(1.0, float(global_fill_progress)))
        for index in range(1, min(filled_regions, raster["count"]) + 1):
            color = np.asarray(
                _REGION_COLORS[(index - 1) % len(_REGION_COLORS)],
                dtype=np.float64,
            )
            blended = np.rint(
                np.asarray(_PANEL, dtype=np.float64) * (1.0 - progress) + color * progress
            ).astype(np.uint8)
            pixels[labels == index] = blended
        region_image = Image.fromarray(pixels, mode="RGB").resize(
            (target, target), Image.Resampling.NEAREST
        )
        image.paste(region_image, (left * factor, top * factor))

    def _render_scene(self, task: Dict[str, Any], state: Dict[str, Any]) -> Image.Image:
        factor = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(512, 512, factor, background=_BG)
        self._render_source_regions(
            image,
            factor,
            task,
            int(state.get("filled_regions", 0)),
            float(state.get("global_fill_progress", 1.0)),
        )
        left, top, right, bottom = _SOURCE_BOX
        scale_x = (right - left) / _LOGICAL_SIZE
        scale_y = (bottom - top) / _LOGICAL_SIZE
        for index, (cx, cy, radius) in enumerate(task["circles"]):
            center_x = left + cx * scale_x
            center_y = top + cy * scale_y
            radius_x = radius * scale_x
            radius_y = radius * scale_y
            draw.ellipse(
                (
                    center_x - radius_x,
                    center_y - radius_y,
                    center_x + radius_x,
                    center_y + radius_y,
                ),
                outline=_CURVE_COLORS[index],
                width=4,
            )
        return finish_scaled_canvas(image, (512, 512))

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        frames: List[np.ndarray] = []
        initial = self._state_trace(int(task["analysis"]["bounded_region_count"]))[0]
        frames.extend(np.asarray(self._render_scene(task, initial), dtype=np.uint8).copy() for _ in range(5))
        region_count = int(task["analysis"]["bounded_region_count"])
        for step in range(1, 5):
            frames.append(
                np.asarray(
                    self._render_scene(
                        task,
                        {
                            "phase": "bounded_face_global_reveal",
                            "filled_regions": region_count,
                            "global_fill_progress": step / 4.0,
                        },
                    ),
                    dtype=np.uint8,
                )
            )
        solution = self._render_scene(
            task,
            self._state_trace(region_count, _LABELS[int(task["correct_index"])])[-1],
        )
        frames.extend(np.asarray(solution, dtype=np.uint8).copy() for _ in range(8))
        expected_count = _video_frame_count(region_count)
        if len(frames) != expected_count or (len(frames) - 1) % 4 != 0:
            raise AssertionError("region_count video must follow its semantic phases and contain 4N+1 frames")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    @staticmethod
    def _contract(nested_disconnected=False) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single image shows two to four colored closed circles on a white canvas."
            ),
            symbol_legend=(
                "Colors distinguish circle identity only. Pastel face fills are output marks; the unbounded outside stays white.",
            ),
            rules=(
                "Count every connected bounded face whose complete boundary consists of visible circle arcs; exclude the single unbounded outside face. Lenses, lobes, overlap subdivisions, and enclosed interiors each count once.",
                ("Circle boundaries may form separate components and contain other circles. A connected face with several boundary loops counts once; a nested circle splits an existing face. There are no tangencies, coincidences, or triple crossings."
                 if nested_disconnected else "The arrangement is connected and in general position: each pair is disjoint or crosses twice, with no tangency, coincidence, containment, shared arc, endpoint, or triple intersection. Regions sharing an arc remain separate."),
            ),
            modality_outputs={
                "ti2v": "Hold the input, reveal all bounded faces uniformly as one global semantic update without a row, column, or face order, then hold the filled scene. Never fill the unbounded outside face.",
                "solution_image": "Fill all and only connected bounded faces in pastel, leave the unbounded outside white, and preserve the contour arrangement.",
            },
            preservation_constraints=(
                "Keep every circle, color, position, background, and camera fixed; add only bounded-face fills, and leave the outside white.",
            ),
            transition_constraints=(
                "Begin with the unmarked input, then reveal the entire bounded-face set uniformly with no face or spatial order.",
                "Keep every fill inside its arc boundary and end on the solved hold.",
            ),
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="strict_uppercase_option_letter_a_to_d",
                verifier_profile=_VERIFIER_PROFILE,
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile=_RENDER_PROFILE,
        )

    @staticmethod
    def _options_sentence(task: Dict[str, Any]) -> str:
        entries = "; ".join(
            f"{_LABELS[index]} = {int(value)}" for index, value in enumerate(task["options"])
        )
        return f"Options: {entries}."

    def _prompts(self, task: Dict[str, Any]) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract(self.nested_disconnected)
        prompts = {
            "question": "Count all connected bounded faces formed by the visible circle arcs, exclude the outside face, and choose the matching total.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        options_sentence = self._options_sentence(task)
        for key in ("ti2v_prompt",):
            prompt = prompts[key]
            if "\n\nRequired Output:" not in prompt:
                raise ValueError("region_count reasoning prompt is missing its Required Output section")
            prompts[key] = prompt.replace(
                "\n\nRequired Output:",
                f" {options_sentence}\n\nRequired Output:",
                1,
            )
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid region-count prompt contract: {'; '.join(issues)}")
        return contract, prompts

    _CIRCLE_COLOR_NAMES = ("blue", "red", "green", "purple")

    @classmethod
    def _authored_reasoning(cls, task: Dict[str, Any]) -> str:
        analysis = task["analysis"]
        names = cls._CIRCLE_COLOR_NAMES[: int(analysis["circle_count"])]
        pair_sentences: List[str] = []
        for pair in analysis["pair_relations"]:
            left, right = (int(value) for value in pair["circles"])
            if pair["relation"] == "intersect_two":
                pair_sentences.append(f"the {names[left]} and {names[right]} circles cross twice")
            elif pair["relation"] == "contained":
                pair_sentences.append(f"one of the {names[left]} and {names[right]} circles encloses the other without crossing")
            else:
                pair_sentences.append(f"the {names[left]} and {names[right]} circles stay apart")
        region_phrases = [
            f"one {cls._relative_position(float(region['centroid'][0]), float(region['centroid'][1]))}"
            for region in task["bounded_regions"]
        ]
        count = int(analysis["bounded_region_count"])
        label = _LABELS[int(task["correct_index"])]
        return (
            f"The scene shows {len(names)} circles: {'; '.join(pair_sentences)}. Every crossing pair carves its "
            f"overlap into separate enclosed areas, so I walk the arrangement and count each bounded face once: "
            f"{', '.join(region_phrases)}. The outside area is unbounded and never counts. That gives {count} "
            f"bounded faces, matching Options entry {label} = {count}."
        )

    @staticmethod
    def _relative_position(x: float, y: float) -> str:
        horizontal = "left" if x < 80 else "center" if x < 160 else "right"
        vertical = "upper" if y < 80 else "middle" if y < 160 else "lower"
        if horizontal == "center" and vertical == "middle":
            return "near the center"
        if horizontal == "center":
            return f"in the {vertical}-center"
        if vertical == "middle":
            return f"at the middle-{horizontal}"
        return f"in the {vertical}-{horizontal}"

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        contract, prompts = self._prompts(task)
        correct_index = int(task["correct_index"])
        correct_option = _LABELS[correct_index]
        correct_count = int(task["analysis"]["bounded_region_count"])
        states = self._state_trace(correct_count, correct_option)
        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"
        puzzle = self._render_scene(task, states[0])
        solution = self._render_scene(task, states[-1])
        puzzle.save(puzzle_path)
        solution.save(solution_path)
        solution_video_path, video_num_frames = self._render_video(task, video_path)
        distractor_counts = [value for index, value in enumerate(task["options"]) if index != correct_index]
        candidate_values = {label: int(task["options"][index]) for index, label in enumerate(_LABELS)}
        task_meta = {
            "circles": [[round(value, 6) for value in circle] for circle in task["circles"]],
            "correct_count": correct_count,
            "distractor_counts": distractor_counts,
            "correct_value": correct_count,
            "distractor_values": distractor_counts,
            "candidate_values_by_label": candidate_values,
            "analysis": task["analysis"],
            "bounded_regions": task["bounded_regions"],
        }
        structured_answer = {
            "correct_option": correct_option,
            "correct_index": correct_index,
            "correct_count": correct_count,
            "task_meta": task_meta,
            "circles": task_meta["circles"],
            "analysis": task["analysis"],
            "bounded_regions": task["bounded_regions"],
            "options": [int(value) for value in task["options"]],
            "candidates": [
                {"label": LETTER_OPTIONS[index], "value": int(value), "is_correct": index == correct_index}
                for index, value in enumerate(task["options"])
            ],
            "state_trace": states,
            "multiple_solution_policy": "unique",
            "answer_schema": _answer_schema(correct_count),
            "verifier_contract": _verifier_contract(),
            "render_geometry": _render_geometry(correct_count),
            "expected_video_frame_count": _video_frame_count(correct_count),
        }
        _replay, replay_valid = _replay_region_count(structured_answer)
        if not replay_valid:
            raise RuntimeError("Generated region_count sample failed independent semantic replay")
        task_contract = contract.to_dict()
        task_contract["input_count"] = 1
        task_contract["canvas_size"] = [512, 512]
        record: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": task_contract,
            "question": prompts["question"],
            "answer": correct_option,
            "correct_option": correct_option,
            "structured_answer": structured_answer,
            "verifier_contract": structured_answer["verifier_contract"],
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
            record,
            problem_image=puzzle_path.as_posix(),
            option_labels=_LABELS,
            ti2v_problem_image=puzzle_path.as_posix(),
        )
        return finalize_geometry_sample(record), puzzle, solution

    def create_puzzle(self) -> Dict[str, Any]:
        return self.make_sample()[0]


__all__ = [
    'RegionCountPattern',
    '_analyze_arrangement',
    '_circle_pair_geometry',
    '_raster_bounded_regions',
    '_replay_region_count',
    '_validate_task',
]
