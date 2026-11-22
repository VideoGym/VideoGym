"""Ray intersection puzzle generator.

Three partial rays originate from a hidden intersection point near the canvas
center. Only the edge-adjacent segments of the rays are drawn so solvers must
extend them mentally to locate the true intersection. Five circled options
(A–E) are rendered near the hidden point; exactly one is positioned at the
actual intersection. Prompt instructs respondents to extend the lines, mark the
intersection in red, and report the option using the phonetic alphabet.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.geometry.core.point_target_base import PointTargetPuzzleGenerator, PointCandidate
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.typed_answers import FIVE_CHOICE_CODEC


@dataclass
class RaySegment:
    angle: float
    start: Tuple[float, float]
    end: Tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "angle": self.angle,
            "start": list(self.start),
            "end": list(self.end),
        }


@dataclass
class RayIntersectionPuzzleRecord:
    id: str
    ti2v_prompt: str
    canvas_dimensions: Tuple[int, int]
    margin: int
    intersection: Tuple[float, float]
    rays: List[RaySegment]
    candidates: List[PointCandidate]
    point_radius: int
    correct_option: str
    image: str
    solution_image_path: str
    solution_image_prompt: Optional[str] = None
    seed: Optional[int] = None
    reasoning_image: Optional[str] = None
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None
    construction_trace: Optional[List[str]] = None
    expected_video_frame_count: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ti2v_prompt": self.ti2v_prompt,
            "canvas_dimensions": list(self.canvas_dimensions),
            "margin": self.margin,
            "intersection": list(self.intersection),
            "rays": [ray.to_dict() for ray in self.rays],
            "candidates": [c.to_dict() for c in self.candidates],
            "point_radius": self.point_radius,
            "correct_option": self.correct_option,
            "answer": FIVE_CHOICE_CODEC.format(self.correct_option),
            "image": self.image,
            "reasoning_image": self.reasoning_image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "construction_trace": self.construction_trace,
            "expected_video_frame_count": self.expected_video_frame_count,
            "seed": self.seed,
            "type": "ray_intersection",
        }


CandidatePoint = PointCandidate


class RayIntersectionGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles with partially hidden ray intersections."""

    DEFAULT_RULE_PROMPT=(
        "A white canvas shows three dark gray line fragments that should be mentally extended inward and five labeled "
        "candidate circles A-E near the hidden center. Determine which candidate is the single point where the three full "
        "lines intersect. Answer with one option from A-E."
    )
    DEFAULT_OUTPUT_DIR = "outputs/geometry/ray_intersection"
    EXPECTED_VIDEO_FRAMES = 25
    STATE_TRACE = [
        "input_hold",
        "first_ray_extension_single_end_growth",
        "second_ray_extension_single_end_growth",
        "third_ray_extension_single_end_growth",
        "correct_candidate_highlight",
        "final_hold",
    ]
    DEFAULT_TI2V_PROMPT=(
        "Input Image: A square white canvas shows three thick dark gray line fragments near the outer part of the "
        "image and five candidate markers A-E near their hidden common intersection. Rule: Extend each fragment "
        "inward along its existing straight direction; the unique candidate at the common intersection is correct. "
        "Required Output: Hold the unchanged input. Extend the three fragments inward one at a time in cyan, always "
        "from the visible inner endpoint toward the common intersection. Finish one extension before starting the "
        "next; never grow multiple extensions together or reveal a whole extension at once. After all three meet, "
        "change only the correct candidate to pale red with a dark red outline. Keep the other markers, source "
        "fragments, framing, and camera unchanged."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self, *, puzzle_id: Optional[str]=None) -> RayIntersectionPuzzleRecord:
        intersection = self.pick_target_point(0.46, padding=self.candidate_anchor_padding(extra=self.canvas_short_side * 0.12))
        rays = self._build_rays(intersection.to_list())
        point_radius = self.point_radius
        self.place_candidates(intersection)
        intersection = intersection.to_list()
        self._intersection_coords = intersection
        self._rays = rays
        pid = puzzle_id or str(uuid.uuid4())
        puzzle_img = self._downscale(self._render(highlight_label=None))
        solution_img = self._downscale(self._render(highlight_label=self.correct_label))
        puzzle_path = self.puzzle_dir / f'{pid}_puzzle.png'
        solution_path = self.solution_dir / f'{pid}_solution.png'
        puzzle_img.save(puzzle_path)
        solution_img.save(solution_path)
        video_rel_path: Optional[str] = None
        video_num_frames: Optional[int] = None
        if self.record_video:
            video_num_frames = self.save_video_solution(pid)
            video_abs = self.solution_dir / f'{pid}_solution.webm'
            video_rel_path = self.relativize_path(video_abs)
        return RayIntersectionPuzzleRecord(id=pid, ti2v_prompt=self.ti2v_prompt, canvas_dimensions=self.canvas_dimensions, margin=self.margin, intersection=intersection, rays=rays, candidates=self.candidates, point_radius=point_radius, correct_option=self.correct_label, image=self.relativize_path(puzzle_path), solution_image_path=self.relativize_path(solution_path), seed=self.seed, reasoning_image=self.relativize_path(puzzle_path), solution_video_path=video_rel_path, video_fps=16 if video_rel_path else None, video_num_frames=video_num_frames if video_rel_path else None, construction_trace=list(self.STATE_TRACE), expected_video_frame_count=self.EXPECTED_VIDEO_FRAMES)

    def _ray_to_bounds(
        self,
        origin: Tuple[float, float],
        angle: float,
    ) -> Tuple[float, float]:
        left, top, right, bottom = self.canvas_bounds()
        ox, oy = origin
        dx = math.cos(angle)
        dy = math.sin(angle)
        best_t = float("inf")
        hit_x = ox
        hit_y = oy

        if dx > 0:
            t = (right - ox) / dx
            y_intercept = oy + t * dy
            if t > 0 and top <= y_intercept <= bottom and t < best_t:
                best_t = t
                hit_x = right
                hit_y = y_intercept
        if dx < 0:
            t = (left - ox) / dx
            y_intercept = oy + t * dy
            if t > 0 and top <= y_intercept <= bottom and t < best_t:
                best_t = t
                hit_x = left
                hit_y = y_intercept
        if dy > 0:
            t = (bottom - oy) / dy
            x_intercept = ox + t * dx
            if t > 0 and left <= x_intercept <= right and t < best_t:
                best_t = t
                hit_x = x_intercept
                hit_y = bottom
        if dy < 0:
            t = (top - oy) / dy
            x_intercept = ox + t * dx
            if t > 0 and left <= x_intercept <= right and t < best_t:
                best_t = t
                hit_x = x_intercept
                hit_y = top
        return (hit_x, hit_y)

    def _edge_segment_start(self, origin: Tuple[float, float], end_point: Tuple[float, float]) -> Tuple[float, float]:
        ox, oy = origin
        ex, ey = end_point
        dx = ex - ox
        dy = ey - oy
        span = math.hypot(dx, dy)
        if span <= 1.0:
            return (ox, oy)
        draw_fraction = self._rng.uniform(0.28, 0.4)
        start_dist = span * (1.0 - draw_fraction)
        ratio = start_dist / span
        sx = ox + dx * ratio
        sy = oy + dy * ratio
        return (sx, sy)

    def _render(
        self,
        *,
        highlight_label: Optional[str],
        construction_progresses: Optional[Sequence[float]] = None,
        # Explicit geometry overrides support deterministic rendering checks.
        intersection: Optional[Tuple[float, float]] = None,
        rays: Optional[Sequence[RaySegment]] = None,
    ) -> Image.Image:
        width, height = self.canvas_dimensions
        draw, base = self.get_draw_base()

        # Resolve args or state
        intersection_val = intersection if intersection is not None else getattr(self, "_intersection_coords", None)
        rays_val = rays if rays is not None else getattr(self, "_rays", None)

        if intersection_val is None or rays_val is None:
             raise RuntimeError("Render called without geometry state")

        stroke_color = (40, 40, 40)
        construction_color = self.CONSTRUCTION_COLOR
        stroke_width = max(3, int(round(min(width, height) * 0.015)))

        progresses = (
            (1.0, 1.0, 1.0)
            if construction_progresses is None and highlight_label
            else (0.0, 0.0, 0.0)
            if construction_progresses is None
            else tuple(max(0.0, min(1.0, float(value))) for value in construction_progresses)
        )
        for ray, progress in zip(rays_val, progresses):
            if progress > 0.0:
                endpoint = (
                    ray.start[0] + (intersection_val[0] - ray.start[0]) * progress,
                    ray.start[1] + (intersection_val[1] - ray.start[1]) * progress,
                )
                draw.line(
                    [
                        (int(round(ray.start[0])), int(round(ray.start[1]))),
                        (int(round(endpoint[0])), int(round(endpoint[1]))),
                    ],
                    fill=construction_color,
                    width=stroke_width,
                )

        for ray in rays_val:
            draw.line(
                [
                    (int(round(ray.start[0])), int(round(ray.start[1]))),
                    (int(round(ray.end[0])), int(round(ray.end[1]))),
                ],
                fill=stroke_color,
                width=stroke_width,
            )

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )

        return base

    def save_video_solution(self, pid: str) -> Optional[int]:
        """Reveal the three inward extensions as separate single-end strokes."""

        input_image = self._downscale(
            self._render(highlight_label=None, construction_progresses=(0.0, 0.0, 0.0))
        ).convert("RGB")
        constructed = self._downscale(
            self._render(highlight_label=None, construction_progresses=(1.0, 1.0, 1.0))
        ).convert("RGB")
        solved = self._downscale(
            self._render(highlight_label=self.correct_label, construction_progresses=(1.0, 1.0, 1.0))
        ).convert("RGB")
        frames: List[np.ndarray] = [np.asarray(input_image, dtype=np.uint8).copy() for _ in range(4)]
        progresses = [0.0, 0.0, 0.0]
        for ray_index in range(3):
            for progress in (0.25, 0.50, 0.75, 1.0):
                current = list(progresses)
                current[ray_index] = progress
                frame = self._downscale(
                    self._render(highlight_label=None, construction_progresses=current)
                ).convert("RGB")
                frames.append(np.asarray(frame, dtype=np.uint8))
            progresses[ray_index] = 1.0
        for alpha in (0.25, 0.50, 0.75, 1.0):
            frames.append(np.asarray(Image.blend(constructed, solved, alpha), dtype=np.uint8))
        frames.extend(np.asarray(solved, dtype=np.uint8).copy() for _ in range(5))
        if len(frames) != self.EXPECTED_VIDEO_FRAMES or (len(frames) - 1) % 4:
            raise AssertionError(
                f"ray_intersection video must contain exactly {self.EXPECTED_VIDEO_FRAMES} frames"
            )
        video_path = self.solution_dir / f"{pid}_solution.webm"
        if not encode_rgb_frames_to_webm(frames, video_path, fps=16):
            return None
        return len(frames)

    @staticmethod
    def _angle_distance(a: float, b: float) -> float:
        diff = abs(a - b) % math.tau
        if diff > math.pi:
            diff = math.tau - diff
        return diff

    def _build_rays(
        self,
        intersection: Tuple[float, float],
    ) -> List[RaySegment]:
        left, top, right, bottom = self.canvas_bounds()
        min_sep = math.radians(48.0)
        angles: List[float] = []
        attempts = 0
        while len(angles) < 3 and attempts < 400:
            attempts += 1
            angle = self._rng.uniform(0.0, math.tau)
            if all(self._angle_distance(angle, existing) >= min_sep for existing in angles):
                angles.append(angle)
        if len(angles) < 3:
            base = [0.0, 2.09439510239, 4.18879020479]
            shift = self._rng.uniform(-0.3, 0.3)
            angles = [(value + shift) % math.tau for value in base]

        segments: List[RaySegment] = []
        for angle in angles:
            end_point = self._ray_to_bounds(intersection, angle)
            start_point = self._edge_segment_start(intersection, end_point)
            segments.append(RaySegment(angle=angle, start=start_point, end=end_point))
        return segments


__all__ = [
    "RayIntersectionGenerator",
    "RayIntersectionPuzzleRecord",
    "RaySegment",
    "CandidatePoint",
]
