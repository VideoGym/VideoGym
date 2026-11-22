"""Visible line-segment intersection counting with a semantic marker trace."""

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
    "count_intersection_points": (
        "A full-canvas scene shows several finite straight segments. "
        "The task is to count the distinct visible interior crossings and pick the matching prompt option."
    ),
}

_LABELS = ("A", "B", "C", "D")
_VERIFIER_PROFILE = "segment_intersection_visible_count_multimodal_hard_gate_v1"
_BG = (255, 255, 255)
_LINE_RGB = (38, 75, 164)
_TEXT = (52, 54, 60)
_PRIMARY_RGB = (20, 151, 188)
_LINE_BOUNDS = (36.0, 36.0, 476.0, 476.0)
_MARKER_RADIUS = 9.0
_MIN_ANGLE_GAP = math.radians(14.0)
_INTERSECTION_CLUSTER_TOLERANCE = 0.01
_INTERSECTION_POLICY = {
    "objects": "finite_straight_segments",
    "crossing": "strict_interior_only",
    "endpoints": "excluded",
    "parallel_pairs": "zero_points",
    "collinear_overlap": "rejected_by_generation",
    "near_parallel": "rejected_by_generation",
    "multi_segment_concurrency": "count_once_but_rejected_by_generation",
    "tangency": "not_applicable_to_straight_segments",
}


def _angle_gap(left: float, right: float) -> float:
    gap = abs(float(left) - float(right)) % math.pi
    return min(gap, math.pi - gap)


def _clip_line_to_box(
    px: float,
    py: float,
    dx: float,
    dy: float,
    bounds: Sequence[float] = _LINE_BOUNDS,
) -> List[List[float]]:
    x_min, y_min, x_max, y_max = [float(value) for value in bounds]
    candidates: List[Tuple[float, float, float]] = []
    if abs(dx) > 1e-9:
        for x in (x_min, x_max):
            t = (x - px) / dx
            y = py + t * dy
            if y_min - 1e-6 <= y <= y_max + 1e-6:
                candidates.append((t, x, y))
    if abs(dy) > 1e-9:
        for y in (y_min, y_max):
            t = (y - py) / dy
            x = px + t * dx
            if x_min - 1e-6 <= x <= x_max + 1e-6:
                candidates.append((t, x, y))
    unique: List[Tuple[float, float, float]] = []
    for value in sorted(candidates):
        if not any(math.hypot(value[1] - other[1], value[2] - other[2]) < 1e-5 for other in unique):
            unique.append(value)
    if len(unique) < 2:
        raise ValueError("line does not cross the drawing bounds")
    first, last = unique[0], unique[-1]
    return [
        [float(round(first[1], 3)), float(round(first[2], 3))],
        [float(round(last[1], 3)), float(round(last[2], 3))],
    ]


def _segment_intersection(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> Tuple[float, float] | None:
    (x1, y1), (x2, y2) = left
    (x3, y3), (x4, y4) = right
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-8:
        return None
    qx, qy = x3 - x1, y3 - y1
    t = (qx * dy2 - qy * dx2) / denom
    u = (qx * dy1 - qy * dx1) / denom
    if not (0.015 < t < 0.985 and 0.015 < u < 0.985):
        return None
    return float(x1 + t * dx1), float(y1 + t * dy1)


def _infinite_intersection(
    line_a: Sequence[Sequence[float]],
    line_b: Sequence[Sequence[float]],
) -> Tuple[float, float] | None:
    """Intersection of the two carrier lines through the clipped segments."""

    (ax0, ay0), (ax1, ay1) = (line_a[0], line_a[1])
    (bx0, by0), (bx1, by1) = (line_b[0], line_b[1])
    d1x, d1y = float(ax1) - float(ax0), float(ay1) - float(ay0)
    d2x, d2y = float(bx1) - float(bx0), float(by1) - float(by0)
    denominator = d1x * d2y - d1y * d2x
    if abs(denominator) <= 1e-9:
        return None
    t = ((float(bx0) - float(ax0)) * d2y - (float(by0) - float(ay0)) * d2x) / denominator
    return float(ax0) + t * d1x, float(ay0) + t * d1y


def _raw_pair_intersections(lines: Sequence[Sequence[Sequence[float]]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for left in range(len(lines)):
        for right in range(left + 1, len(lines)):
            point = _segment_intersection(lines[left], lines[right])
            if point is None:
                continue
            records.append({"pair": [left, right], "point": [float(round(point[0], 3)), float(round(point[1], 3))]})
    return sorted(records, key=lambda item: (item["point"][1], item["point"][0], item["pair"]))


def _cluster_intersections(
    records: Sequence[Dict[str, Any]],
    *,
    tolerance: float = _INTERSECTION_CLUSTER_TOLERANCE,
) -> List[Dict[str, Any]]:
    """Collapse pairwise intersections that occupy one visible coordinate."""

    clusters: List[Dict[str, Any]] = []
    for record in records:
        point = [float(record["point"][0]), float(record["point"][1])]
        pair = [int(record["pair"][0]), int(record["pair"][1])]
        match = next(
            (
                cluster
                for cluster in clusters
                if math.hypot(point[0] - cluster["point"][0], point[1] - cluster["point"][1]) <= tolerance
            ),
            None,
        )
        if match is None:
            clusters.append({"point": point, "pairs": [pair]})
            continue
        count = len(match["pairs"])
        match["point"] = [
            (match["point"][0] * count + point[0]) / (count + 1),
            (match["point"][1] * count + point[1]) / (count + 1),
        ]
        match["pairs"].append(pair)
    normalized = [
        {
            "point": [float(round(cluster["point"][0], 3)), float(round(cluster["point"][1], 3))],
            "pairs": sorted(cluster["pairs"]),
        }
        for cluster in clusters
    ]
    return sorted(normalized, key=lambda item: (item["point"][1], item["point"][0], item["pairs"]))


def _line_angle(line: Sequence[Sequence[float]]) -> float:
    return math.atan2(float(line[1][1]) - float(line[0][1]), float(line[1][0]) - float(line[0][0])) % math.pi


def _has_safe_angle_gaps(lines: Sequence[Sequence[Sequence[float]]]) -> bool:
    angles = [_line_angle(line) for line in lines]
    return all(
        _angle_gap(angles[left], angles[right]) >= _MIN_ANGLE_GAP - math.radians(0.02)
        for left in range(len(lines))
        for right in range(left + 1, len(lines))
    )


def _point_close(left: Sequence[float], right: Sequence[float], *, tolerance: float = 0.01) -> bool:
    return abs(float(left[0]) - float(right[0])) <= tolerance and abs(float(left[1]) - float(right[1])) <= tolerance


def _line_anchor_points(
    lines: Sequence[Sequence[Sequence[float]]],
    intersection_points: Sequence[Sequence[float]],
) -> List[List[List[float]]]:
    fractions = (0.08, 0.18, 0.30, 0.70, 0.82, 0.92)
    output: List[List[List[float]]] = []
    for start, end in lines:
        anchors: List[List[float]] = []
        for fraction in fractions:
            point = [
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            ]
            if all(math.hypot(point[0] - crossing[0], point[1] - crossing[1]) >= 18.0 for crossing in intersection_points):
                anchors.append([float(round(point[0], 3)), float(round(point[1], 3))])
            if len(anchors) == 3:
                break
        if len(anchors) < 2:
            anchors = [
                [float(round(start[0] + (end[0] - start[0]) * fraction, 3)), float(round(start[1] + (end[1] - start[1]) * fraction, 3))]
                for fraction in (0.04, 0.96)
            ]
        output.append(anchors)
    return output


def _generate_lines(rng: random.Random, n: int, *, concurrent: bool = False) -> Tuple[List[List[List[float]]], List[Dict[str, Any]]]:
    x_min, y_min, x_max, y_max = _LINE_BOUNDS
    center_x, center_y = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    for _ in range(500):
        lines: List[List[List[float]]] = []
        angles: List[float] = []
        common_x, common_y = (center_x + rng.uniform(-40, 40), center_y + rng.uniform(-40, 40)) if concurrent else (center_x, center_y)
        for _index in range(n):
            for _attempt in range(200):
                angle = rng.uniform(0.0, math.pi)
                if any(_angle_gap(angle, other) < _MIN_ANGLE_GAP for other in angles):
                    continue
                dx, dy = math.cos(angle), math.sin(angle)
                offset = rng.uniform(-138.0, 138.0)
                px = center_x - dy * offset
                py = center_y + dx * offset
                if concurrent and _index < 3:
                    px, py = common_x, common_y
                line = _clip_line_to_box(px, py, dx, dy)
                lines.append(line)
                angles.append(angle)
                break
            else:
                break
        if len(lines) != n:
            continue
        records = _raw_pair_intersections(lines)
        clusters = _cluster_intersections(records)
        if not (3 <= len(clusters) <= 8):
            continue
        # ID has distinct pair crossings; extrapolation requires a shared
        # coordinate, counted once despite three contributing segment pairs.
        if (len(records) != len(clusters)) != concurrent:
            continue
        points = [cluster["point"] for cluster in clusters]
        # Perceptual margins: every counted crossing sits well inside the
        # canvas, and no two crossings sit close enough to read as one marker
        # blob.
        if any(
            min(point[0] - x_min, x_max - point[0], point[1] - y_min, y_max - point[1]) < 24.0
            for point in points
        ):
            continue
        if any(
            math.hypot(points[left][0] - points[right][0], points[left][1] - points[right][1]) < 34.0
            for left in range(len(points))
            for right in range(left + 1, len(points))
        ):
            continue
        # Outside near-miss guard: if two infinite carrier lines meet just
        # beyond the canvas, their clipped segments converge at the border and
        # can look like one more touching pair.  Reject such layouts.
        outside_near_miss = False
        for left in range(len(lines)):
            for right in range(left + 1, len(lines)):
                if any(sorted(pair["pair"]) == [left, right] for pair in records):
                    continue
                meet = _infinite_intersection(lines[left], lines[right])
                if meet is None:
                    continue
                mx, my = meet
                clamped_x = min(max(mx, x_min), x_max)
                clamped_y = min(max(my, y_min), y_max)
                if math.hypot(mx - clamped_x, my - clamped_y) < 40.0:
                    outside_near_miss = True
                    break
            if outside_near_miss:
                break
        if outside_near_miss:
            continue
        return lines, records
    raise RuntimeError("Failed to construct a readable line-intersection task.")


class CountIntersectionPointsPattern(RenderedTaskGenerator):
    task_name = "count_intersection_points"
    task_family = "geometry"
    video_fps = 16

    def __init__(self, output_dir: str | Path | None = None, **kwargs: Any) -> None:
        seed = kwargs.get("random_seed", kwargs.get("seed"))
        self.rng = random.Random(seed)
        self.concurrent = bool(kwargs.get("concurrent", False))
        self.video_fps = int(kwargs.get("video_fps", self.video_fps))
        self.generate_videos = bool(kwargs.get("generate_videos", kwargs.get("video", True)))
        self.image_size = (512, 512)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_count_intersection_points"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    def _generate_task(self) -> Dict[str, Any]:
        n = 4
        lines, records = _generate_lines(self.rng, n, concurrent=self.concurrent)
        true_count = len(_cluster_intersections(records))
        distractors = {max(1, true_count - 2), max(1, true_count - 1), true_count + 1, true_count + 2, n, n * (n - 1) // 2}
        distractors.discard(true_count)
        options = [true_count, *self.rng.sample(sorted(distractors), k=3)]
        self.rng.shuffle(options)
        return {
            "n": n,
            "lines": lines,
            "intersection_records": records,
            "true_count": true_count,
            "options": options,
            "correct_index": options.index(true_count),
        }

    @staticmethod
    def _state_trace(task: Dict[str, Any]) -> List[Dict[str, Any]]:
        # The crossings form one unordered, equal-status evidence set.  The
        # semantic trace therefore has one global marker commit, not an
        # arbitrary permutation of otherwise equivalent points.
        return [
            {"marked_count": 0},
            {"marked_count": int(task["true_count"])},
        ]

    @staticmethod
    def _geometry(task: Dict[str, Any]) -> Dict[str, Any]:
        points = [group["point"] for group in _cluster_intersections(task["intersection_records"])]
        pairs = [record["pair"] for record in task["intersection_records"]]
        return {
            "line_bounds": list(_LINE_BOUNDS),
            "lines": task["lines"],
            "line_anchor_points": _line_anchor_points(task["lines"], points),
            "intersection_points": points,
            "intersection_pairs": pairs,
            "marker_radius": _MARKER_RADIUS,
            "line_rgb": list(_LINE_RGB),
            "primary_rgb": list(_PRIMARY_RGB),
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
        for line in task["lines"]:
            draw.line([tuple(line[0]), tuple(line[1])], fill=_LINE_RGB, width=4)
        if int(marked_count) > 0:
            opacity = max(0.0, min(1.0, float(marker_progress)))
            alpha = int(round(255 * opacity))
            if alpha > 0:
                overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
                overlay_draw = ScaledDraw(ImageDraw.Draw(overlay), factor)
                for point in self._geometry(task)["intersection_points"]:
                    x, y = float(point[0]), float(point[1])
                    overlay_draw.ellipse(
                        (x - _MARKER_RADIUS, y - _MARKER_RADIUS, x + _MARKER_RADIUS, y + _MARKER_RADIUS),
                        fill=(*_PRIMARY_RGB, alpha),
                        outline=(*_TEXT, alpha),
                        width=1,
                    )
                image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
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

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description="The single image shows four straight blue line segments crossing a white canvas.",
            symbol_legend=(
                "Every blue stroke is one finite straight segment.",
            ),
            rules=(
                "Treat every blue object as one finite straight segment, not as an infinite line, and count only a strict interior crossing shared by two segments.",
                "A segment endpoint never counts; parallel or non-crossing pairs add no point, including pairs whose infinite extensions meet outside either segment.",
                "No two visible segments overlap collinearly or form an ambiguous near-parallel crossing; a collinear overlap would not define one finite crossing point.",
                "If several segment pairs meet at exactly the same coordinate, that coordinate counts once; all crossing coordinates in the supplied image are distinct.",
                "Tangency does not apply because the scene contains only straight segments; the required count is the number of distinct interior crossings.",
            ),
            modality_outputs={
                "ti2v": "Hold the input, place the complete cyan crossing-marker set at its fixed terminal size and position while raising only its shared opacity, then hold the marked scene.",
                "solution_image": "Preserve every segment and add one plain cyan disc at every distinct crossing; change nothing else.",
            },
            preservation_constraints=(
                "Keep every segment endpoint, segment path, background, framing, and camera unchanged.",
                "Only the plain cyan crossing markers may be added.",
            ),
            transition_constraints=(
                "Begin with a readable hold of the exact unmarked input.",
                "Fade all equal-status crossing markers in together at their complete final size; none may appear earlier or later than the rest.",
                "After every distinct interior crossing has exactly one marker, hold the final state without regression.",
            ),
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="strict_uppercase_option_letter_a_to_d",
                verifier_profile=_VERIFIER_PROFILE,
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_full_canvas_intersection_marker_local_opacity_v7",
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
            "question": "Count the distinct visible interior crossings and choose the matching numeric option.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid intersection-count prompt contract: {'; '.join(issues)}")
        return contract, prompts

    @staticmethod
    def _segment_descriptions(lines: Sequence[Sequence[Sequence[float]]]) -> List[str]:
        """Reader-facing names for each segment from its visible slope."""

        descriptions: List[str] = []
        for line in lines:
            (x0, y0), (x1, y1) = line
            angle = math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 180.0
            if angle < 15.0 or angle > 165.0:
                kind = "nearly horizontal segment"
            elif 75.0 <= angle <= 105.0:
                kind = "nearly vertical segment"
            elif angle < 75.0:
                kind = f"segment rising at roughly {5 * round(angle / 5.0):.0f} degrees"
            else:
                kind = f"segment falling at roughly {5 * round((180.0 - angle) / 5.0):.0f} degrees"
            descriptions.append(kind)
        for index, description in enumerate(descriptions):
            twins = [other for other, text in enumerate(descriptions) if text == description]
            if len(twins) > 1:
                for rank, other in enumerate(
                    sorted(twins, key=lambda item: min(lines[item][0][1], lines[item][1][1]))
                ):
                    qualifier = ("upper", "lower", "third", "fourth")[rank]
                    descriptions[other] = f"{qualifier} {description}"
        return descriptions

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        contract, prompts = self._prompts(task)
        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__input.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"
        puzzle = self._render_scene(task)
        solution = self._render_scene(task, marked_count=int(task["true_count"]))
        puzzle.save(puzzle_path)
        solution.save(solution_path)
        solution_video_path, video_num_frames = self._render_video(task, video_path)
        correct_option = _LABELS[task["correct_index"]]
        structured_answer = {
            "correct_option": correct_option,
            "correct_index": task["correct_index"],
            "n": task["n"],
            "lines": task["lines"],
            "intersection_points": [record["point"] for record in task["intersection_records"]],
            "intersection_pairs": [record["pair"] for record in task["intersection_records"]],
            "intersection_groups": _cluster_intersections(task["intersection_records"]),
            "intersection_policy": dict(_INTERSECTION_POLICY),
            "true_count": task["true_count"],
            "options": task["options"],
            "state_trace": self._state_trace(task),
            "render_geometry": self._geometry(task),
        }
        if _replay_count_intersection_points(structured_answer) is None:
            raise ValueError("Generated intersection-count sample failed independent task-local replay.")
        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": prompts["question"],
            "answer": correct_option,
            "correct_option": correct_option,
            "structured_answer": structured_answer,
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": [puzzle_path.as_posix()],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }
        augment_counting_contract(sample, verifier_profile=_VERIFIER_PROFILE, semantic_source="visible finite segment endpoints and the prompt-serialized numeric options")
        sample["structured_answer"]["verifier_contract"]["binary_predicate"] = (
            "strict option parse, independent finite-segment intersection replay, one unique numeric candidate, "
            "one global equal-status crossing-marker set with synchronized fixed-footprint opacity, "
            "fixed-content preservation, and the declared media contract"
        )
        attach_text_choice_card_inputs(
            sample,
            problem_image=puzzle_path.as_posix(),
            option_labels=_LABELS,
            ti2v_problem_image=puzzle_path.as_posix(),
        )
        return finalize_geometry_sample(sample), puzzle, solution


def _replay_count_intersection_points(structured: Dict[str, Any]) -> Dict[str, Any] | None:
    """Independently rebuild every visible crossing and generated-domain guard."""

    try:
        n = int(structured["n"])
        lines = [
            [[float(coordinate) for coordinate in point] for point in line]
            for line in structured["lines"]
        ]
        reported_points = [
            [float(coordinate) for coordinate in point]
            for point in structured["intersection_points"]
        ]
        reported_pairs = [
            [int(value) for value in pair]
            for pair in structured["intersection_pairs"]
        ]
        reported_groups = [
            {
                "point": [float(coordinate) for coordinate in group["point"]],
                "pairs": [[int(value) for value in pair] for pair in group["pairs"]],
            }
            for group in structured["intersection_groups"]
        ]
        true_count = int(structured["true_count"])
        options = [int(value) for value in structured["options"]]
        correct_index = int(structured["correct_index"])
        correct_option = str(structured["correct_option"]).strip().upper()
        state_trace = [
            {"marked_count": int(state["marked_count"])}
            for state in structured["state_trace"]
        ]
    except (KeyError, TypeError, ValueError, IndexError):
        return None

    if n not in {4, 5} or len(lines) != n:
        return None
    if any(len(line) != 2 or any(len(point) != 2 for point in line) for line in lines):
        return None
    if any(not math.isfinite(coordinate) for line in lines for point in line for coordinate in point):
        return None

    x_min, y_min, x_max, y_max = _LINE_BOUNDS
    for line in lines:
        if math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1]) < 1.0:
            return None
        for x, y in line:
            if not (x_min - 0.01 <= x <= x_max + 0.01 and y_min - 0.01 <= y <= y_max + 0.01):
                return None
            if min(abs(x - x_min), abs(x - x_max), abs(y - y_min), abs(y - y_max)) > 0.01:
                return None

    if not _has_safe_angle_gaps(lines):
        return None

    records = _raw_pair_intersections(lines)
    groups = _cluster_intersections(records)
    # Count visible coordinates, including multi-line concurrency, once.
    if not 3 <= len(groups) <= 8:
        return None
    points = [record["point"] for record in records]
    pairs = [record["pair"] for record in records]
    if any(
        min(point[0] - x_min, x_max - point[0], point[1] - y_min, y_max - point[1]) < 18.0
        for point in points
    ):
        return None
    unique_points = [group["point"] for group in groups]
    if any(
        math.dist(a, b) < 22.0
        for i, a in enumerate(unique_points) for b in unique_points[i + 1:]
    ):
        return None
    if reported_pairs != pairs or len(reported_points) != len(points):
        return None
    if any(not _point_close(reported, expected) for reported, expected in zip(reported_points, points)):
        return None
    if reported_groups != groups or structured.get("intersection_policy") != _INTERSECTION_POLICY:
        return None

    expected_trace = CountIntersectionPointsPattern._state_trace({"true_count": true_count})
    if state_trace != expected_trace:
        return None
    if (
        true_count != len(groups)
        or len(options) != 4
        or len(set(options)) != 4
        or options.count(true_count) != 1
        or not 0 <= correct_index < 4
        or options[correct_index] != true_count
        or correct_option != _LABELS[correct_index]
    ):
        return None

    replay_task = {
        "n": n,
        "lines": lines,
        "intersection_records": records,
        "true_count": true_count,
        "options": options,
        "correct_index": correct_index,
    }
    expected_geometry = CountIntersectionPointsPattern._geometry(replay_task)
    if structured.get("render_geometry") != expected_geometry:
        return None
    return {
        "n": n,
        "lines": lines,
        "intersection_records": records,
        "intersection_groups": groups,
        "intersection_points": unique_points,
        "intersection_pairs": pairs,
        "true_count": true_count,
        "options": options,
        "correct_index": correct_index,
        "correct_option": correct_option,
        "state_trace": expected_trace,
        "render_geometry": expected_geometry,
    }


__all__ = [
    'CountIntersectionPointsPattern',
    '_INTERSECTION_POLICY',
    '_PRIMARY_RGB',
    '_cluster_intersections',
    '_generate_lines',
    '_has_safe_angle_gaps',
    '_line_anchor_points',
    '_raw_pair_intersections',
    '_replay_count_intersection_points',
    '_segment_intersection',
]
