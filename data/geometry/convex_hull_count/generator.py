"""Convex-hull vertex counting with a semantic boundary-construction trace."""

from __future__ import annotations

from data.common.base import RenderedTaskGenerator
from data.geometry.core.rendered_contract import finalize_geometry_sample

import math
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.choice_cards import attach_text_choice_card_inputs
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.geometry.core.geometry_counting import augment_counting_contract


pattern_instructions = {
    "convex_hull_count": (
        "A clean square canvas shows a well-separated scatter of black dots and four numeric options. "
        "The task is to construct the tight convex boundary and count its corner dots."
    ),
}

_LABELS = ("A", "B", "C", "D")
_VERIFIER_PROFILE = "convex_hull_visible_count_multimodal_hard_gate_v1"
_BG = (255, 255, 255)
_DOT_RGB = (38, 42, 48)
_ANCHOR_RGB = (142, 74, 210)
_PRIMARY_RGB = (20, 151, 188)
_REMAINING_RGB = (232, 146, 58)
_HULL_RGB = (214, 54, 62)
_DRAW_BOUNDS = (40.0, 40.0, 472.0, 472.0)
_MIN_POINT_DISTANCE = 30.0
_MIN_INTERIOR_EDGE_CLEARANCE = 28.0
_MIN_CORNER_CLEARANCE = 24.0


def _cross(origin: Sequence[float], left: Sequence[float], right: Sequence[float]) -> float:
    return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])


def _hull_chains(points: Sequence[Sequence[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    """Return the two strict monotone hull chains, each including both anchors."""
    pts = sorted({(int(point[0]), int(point[1])) for point in points})
    if len(pts) <= 1:
        singleton = [[x, y] for x, y in pts]
        return singleton, singleton
    primary: List[Tuple[int, int]] = []
    for point in pts:
        while len(primary) >= 2 and _cross(primary[-2], primary[-1], point) <= 0:
            primary.pop()
        primary.append(point)
    remaining: List[Tuple[int, int]] = []
    for point in reversed(pts):
        while len(remaining) >= 2 and _cross(remaining[-2], remaining[-1], point) <= 0:
            remaining.pop()
        remaining.append(point)
    return [[x, y] for x, y in primary], [[x, y] for x, y in remaining]


def _convex_hull(points: Sequence[Sequence[int]]) -> List[List[int]]:
    primary, remaining = _hull_chains(points)
    if len(primary) <= 1:
        return primary
    return primary[:-1] + remaining[:-1]


def _point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    vx, vy = end[0] - start[0], end[1] - start[1]
    wx, wy = point[0] - start[0], point[1] - start[1]
    denom = vx * vx + vy * vy
    if denom <= 0:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    return math.hypot(point[0] - (start[0] + t * vx), point[1] - (start[1] + t * vy))


def _inside_convex_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    signs = [_cross(polygon[index], polygon[(index + 1) % len(polygon)], point) for index in range(len(polygon))]
    return all(value > 0 for value in signs) or all(value < 0 for value in signs)


def _edge_centers(chain: Sequence[Sequence[int]]) -> List[List[float]]:
    return [
        [float(round((left[0] + right[0]) / 2.0, 3)), float(round((left[1] + right[1]) / 2.0, 3))]
        for left, right in zip(chain, chain[1:])
    ]


def _minimum_pairwise_distance(points: Sequence[Sequence[float]]) -> float:
    return min(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for index, left in enumerate(points)
        for right in points[index + 1 :]
    )


def _hull_clearances(
    points: Sequence[Sequence[int]], hull: Sequence[Sequence[int]]
) -> Tuple[List[List[int]], float, float]:
    hull_keys = {tuple(point) for point in hull}
    interior = [list(point) for point in points if tuple(point) not in hull_keys]
    if not interior:
        raise ValueError("convex_hull_count requires interior points")
    interior_clearance = min(
        _point_segment_distance(point, hull[index], hull[(index + 1) % len(hull)])
        for point in interior
        for index in range(len(hull))
    )
    corner_clearance = min(
        _point_segment_distance(hull[index], hull[index - 1], hull[(index + 1) % len(hull)])
        for index in range(len(hull))
    )
    return interior, interior_clearance, corner_clearance


class ConvexHullCountPattern(RenderedTaskGenerator):
    task_name = "convex_hull_count"
    task_family = "geometry"
    video_fps = 16

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        image_size: int | Tuple[int, int] | None = None,
        video_fps: int | None = None,
        generate_videos: bool = True,
        video: bool | None = None,
        **kwargs: Any,
    ) -> None:
        target_size = kwargs.get("target_size")
        requested_size = target_size if target_size is not None else image_size
        normalized_size = (
            (int(requested_size), int(requested_size))
            if isinstance(requested_size, int)
            else tuple(requested_size or (512, 512))
        )
        if normalized_size != (512, 512):
            raise ValueError("convex_hull_count renders are fixed at 512x512")
        if video_fps not in (None, 16):
            raise ValueError("convex_hull_count videos are fixed at 16 fps")
        resolved_seed = seed if seed is not None else random_seed
        self.rng = random.Random(resolved_seed)
        self.video_fps = 16
        self.generate_videos = bool(generate_videos if video is None else video)
        self.image_size = (512, 512)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_convex_hull_count"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0
        self.point_count = kwargs.get("point_count")
        self.hull_count = kwargs.get("hull_count")

    def _generate_task(self) -> Dict[str, Any]:
        for _ in range(200):
            # TI2V reveals every boundary edge; its duration grows with the hull.
            hull_count = (
                self.rng.randint(4, 6)
                if self.hull_count is None
                else int(self.hull_count)
            )
            step = 2.0 * math.pi / hull_count
            rotation = self.rng.uniform(0.0, 2.0 * math.pi)
            boundary: List[List[int]] = []
            for index in range(hull_count):
                angle = (
                    rotation
                    + index * step
                    + self.rng.uniform(-0.11 * step, 0.11 * step)
                )
                radius = self.rng.uniform(148.0, 168.0)
                boundary.append(
                    [
                        int(round(256.0 + math.cos(angle) * radius * 1.18)),
                        int(round(256.0 + math.sin(angle) * radius * 1.18)),
                    ]
                )
            if len(_convex_hull(boundary)) != hull_count:
                continue
            ordered_boundary = _convex_hull(boundary)
            if any(
                not (_DRAW_BOUNDS[0] + 12 <= point[0] <= _DRAW_BOUNDS[2] - 12)
                or not (_DRAW_BOUNDS[1] + 12 <= point[1] <= _DRAW_BOUNDS[3] - 12)
                for point in ordered_boundary
            ):
                continue
            interior: List[List[int]] = []
            interior_count = (
                self.rng.randint(4, 7)
                if self.point_count is None
                else int(self.point_count) - hull_count
            )
            for _candidate in range(2000):
                if len(interior) == interior_count:
                    break
                point = [self.rng.randint(72, 440), self.rng.randint(72, 440)]
                if not _inside_convex_polygon(point, ordered_boundary):
                    continue
                edge_clearance = min(
                    _point_segment_distance(
                        point,
                        ordered_boundary[index],
                        ordered_boundary[(index + 1) % hull_count],
                    )
                    for index in range(hull_count)
                )
                if edge_clearance < 28.0:
                    continue
                if any(
                    math.hypot(point[0] - other[0], point[1] - other[1]) < 30.0
                    for other in [*boundary, *interior]
                ):
                    continue
                interior.append(point)
            if len(interior) != interior_count:
                continue
            points = [*boundary, *interior]
            self.rng.shuffle(points)
            primary_chain, remaining_chain = _hull_chains(points)
            hull = _convex_hull(points)
            if len(hull) != hull_count or len(points) - len(hull) < 4:
                continue
            interior_points, interior_clearance, corner_clearance = _hull_clearances(
                points, hull
            )
            minimum_point_distance = _minimum_pairwise_distance(points)
            if (
                minimum_point_distance < _MIN_POINT_DISTANCE
                or interior_clearance < _MIN_INTERIOR_EDGE_CLEARANCE
                or corner_clearance < _MIN_CORNER_CLEARANCE
            ):
                continue
            options = self._options(hull_count, len(points))
            return {
                "points": points,
                "hull": hull,
                "hull_count": hull_count,
                "point_count": len(points),
                "primary_chain": primary_chain,
                "remaining_chain": remaining_chain,
                "interior_points": interior_points,
                "minimum_point_distance": minimum_point_distance,
                "minimum_interior_edge_clearance": interior_clearance,
                "minimum_corner_clearance": corner_clearance,
                "options": options,
                "correct_index": options.index(hull_count),
            }
        raise RuntimeError("Failed to construct a clear convex-hull counting task.")

    def _options(self, true_count: int, point_count: int) -> List[int]:
        pool = {
            max(3, true_count - 2),
            max(3, true_count - 1),
            true_count + 1,
            true_count + 2,
            point_count - true_count,
            point_count,
        }
        pool.discard(true_count)
        options = [true_count, *self.rng.sample(sorted(pool), 3)]
        self.rng.shuffle(options)
        return options

    @staticmethod
    def _state_trace(task: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"edge_count": edge_count}
            for edge_count in range(int(task["hull_count"]) + 1)
        ]

    @staticmethod
    def _geometry(task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "draw_bounds": list(_DRAW_BOUNDS),
            "primary_chain": [list(point) for point in task["primary_chain"]],
            "remaining_chain": [list(point) for point in task["remaining_chain"]],
            "primary_sample_centers": _edge_centers(task["primary_chain"]),
            "remaining_sample_centers": _edge_centers(task["remaining_chain"]),
            "anchor_points": [list(task["primary_chain"][0]), list(task["primary_chain"][-1])],
            "hull_points": [list(point) for point in task["hull"]],
            "dot_rgb": list(_DOT_RGB),
            "anchor_rgb": list(_ANCHOR_RGB),
            "primary_rgb": list(_PRIMARY_RGB),
            "remaining_rgb": list(_REMAINING_RGB),
            "hull_rgb": list(_HULL_RGB),
        }

    def _render_scene(
        self,
        task: Dict[str, Any],
        *,
        edge_count: int = 0,
        active_edge_progress: float = 0.0,
    ) -> Image.Image:
        image, draw = new_scaled_canvas(512, 512, DOCTRINE_SUPERSAMPLE, background=_BG)
        geometry = self._geometry(task)
        hull = geometry["hull_points"]
        for index in range(min(max(0, int(edge_count)), len(hull))):
            draw.line(
                [tuple(hull[index]), tuple(hull[(index + 1) % len(hull)])],
                fill=_PRIMARY_RGB,
                width=5,
            )
        active_index = min(max(0, int(edge_count)), len(hull))
        active_progress = max(0.0, min(1.0, float(active_edge_progress)))
        if active_index < len(hull) and active_progress > 0.0:
            start = hull[active_index]
            end = hull[(active_index + 1) % len(hull)]
            tip = (
                float(start[0]) + (float(end[0]) - float(start[0])) * active_progress,
                float(start[1]) + (float(end[1]) - float(start[1])) * active_progress,
            )
            draw.line([tuple(start), tip], fill=_PRIMARY_RGB, width=5)
        for x, y in task["points"]:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=_DOT_RGB)
        return finish_scaled_canvas(image, (512, 512))

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        frames: List[np.ndarray] = []
        puzzle = np.asarray(self._render_scene(task), dtype=np.uint8)
        frames.extend(puzzle.copy() for _ in range(5))
        hull_count = int(task["hull_count"])
        for edge_index in range(hull_count):
            for progress in (0.25, 0.50, 0.75, 1.0):
                frames.append(
                    np.asarray(
                        self._render_scene(
                            task,
                            edge_count=edge_index,
                            active_edge_progress=progress,
                        ),
                        dtype=np.uint8,
                    )
                )
        final_frame = np.asarray(
            self._render_scene(task, edge_count=hull_count),
            dtype=np.uint8,
        )
        frames.extend(final_frame.copy() for _ in range(8))
        while len(frames) % 4 != 1:
            frames.append(frames[-1].copy())
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single image shows one well-separated scatter of black point centers on a white canvas. "
                "Every point is visible; each point is either a convex-hull corner or clearly inside, and no point lies along a hull edge between two corners."
            ),
            symbol_legend=(
                "Black disks mark the point centers to classify.",
            ),
            rules=(
                "The convex hull is the tightest convex polygon whose boundary and interior contain every point center.",
                "Starting at the leftmost anchor, trace one outside boundary chain to the rightmost anchor, then trace the other outside chain back; together the two chains form the complete convex boundary.",
                "Count each distinct point center that is a corner of this closed boundary exactly once; do not count interior points.",
            ),
            modality_outputs={
                "ti2v": "Hold the input, connect one cyan convex-boundary edge at a time until the boundary closes, then hold the completed scene.",
                "solution_image": "Preserve the scatter and retain the single closed cyan convex boundary; change nothing else.",
            },
            preservation_constraints=(
                "Keep every point center, background, framing, and camera unchanged.",
                "Only the cyan hull edges may be added.",
            ),
            transition_constraints=(
                "Begin with a readable hold of the exact unmarked scatter.",
                "Add exactly one new adjacent hull edge per semantic step until the boundary closes.",
                "Never skip, remove, reorder, or relocate points or boundary evidence during the sequence.",
            ),
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="strict_uppercase_option_letter_a_to_d",
                verifier_profile=_VERIFIER_PROFILE,
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_full_canvas_convex_hull_edge_trace_v4",
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
            "question": "Construct the complete convex boundary of the point set and choose the number of point centers that are its corners.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid convex-hull prompt contract: {'; '.join(issues)}")
        return contract, prompts

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
                "point_count": task["point_count"],
                "hull_count": task["hull_count"],
                "options": task["options"],
                "points": task["points"],
                "hull": task["hull"],
                "primary_chain": task["primary_chain"],
                "remaining_chain": task["remaining_chain"],
                "interior_points": task["interior_points"],
                "minimum_point_distance": task["minimum_point_distance"],
                "minimum_interior_edge_clearance": task["minimum_interior_edge_clearance"],
                "minimum_corner_clearance": task["minimum_corner_clearance"],
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
        augment_counting_contract(sample, verifier_profile=_VERIFIER_PROFILE, semantic_source="visible point cloud and the prompt-serialized numeric options")
        sample["structured_answer"]["verifier_contract"]["binary_predicate"] = (
            "strict option parse, independent convex-hull replay from the visible point cloud, one unique numeric "
            "candidate, one serial adjacent-edge boundary construction, fixed-content preservation, and the declared "
            "media contract"
        )
        attach_text_choice_card_inputs(
            sample,
            problem_image=puzzle_path.as_posix(),
            option_labels=_LABELS,
            ti2v_problem_image=puzzle_path.as_posix(),
        )
        return finalize_geometry_sample(sample), puzzle, solution


__all__ = [
    'ConvexHullCountPattern',
    '_ANCHOR_RGB',
    '_DOT_RGB',
    '_HULL_RGB',
    '_PRIMARY_RGB',
    '_REMAINING_RGB',
    '_convex_hull',
    '_edge_centers',
    '_hull_chains',
    '_hull_clearances',
    '_minimum_pairwise_distance',
]
