from __future__ import annotations

from data.common.assets import render_font

from data.common.base import RenderedTaskGenerator

import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.cartesian_coordinates import CARTESIAN_POINT_ANSWER_EXAMPLE, CARTESIAN_POINT_ANSWER_SCHEMA, format_cartesian_point_payload


pattern_instructions = {
    "triple_intersection_red": (
        "A coordinate plane shows three labeled overlapping circles A, B, and C. "
        "Color exactly their common region red and return the coordinate of its marked central grid point."
    ),
}

_CANVAS = (512, 512)
_BG = (255, 255, 255)
_GRID = (226, 229, 226)
_TEXT = (47, 53, 58)
_MUTED = (102, 108, 112)
_OUTLINE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (63, 111, 212),
    (224, 120, 60),
    (149, 88, 192),
)
# Family set-diagram cluster language: one marked region is one flat
# translucent red fill.  The base colour and alpha match the Venn tasks'
# shared encoding; the verifier derives its expected observed colours from
# these two values instead of duplicating them.
_TRIPLE_FILL = (222, 60, 60)
_TRIPLE_ALPHA = 96
# Geometry is expressed in task coordinates, not screen pixels.  The renderer
# uses a fixed affine map solely to draw those semantics.
_UNITS_TO_PIXELS = 16.0
_COORDINATE_MAX = 32
# Every center lies on one of the labeled four-unit grid intersections.  The
# three offsets in each template sum to zero, so their visible arithmetic mean
# is the labeled anchor without relying on sub-grid renderer coordinates.
_TEMPLATES: Tuple[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], ...] = (
    ((0, -4), (-4, 0), (4, 4)),
    ((0, -4), (-4, 4), (4, 0)),
)
_RADII: Tuple[float, ...] = (6.0, 6.5, 7.0, 7.5)
_ANCHORS: Tuple[int, ...] = (12, 16, 20)
_VERIFIER_PROFILE = "triple_intersection_complete_footprint_fade_coordinate_and_region_replay_v6"
_VERIFIER_COMPONENTS = (
    "parse_score",
    "source_replay_score",
    "mask_exactness_score",
    "centroid_score",
    "visual_parseability_score",
    "state_order_score",
    "transition_visibility_score",
    "preservation_score",
    "final_state_score",
    "media_contract_score",
)
def _load_font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def _rotate_offset(dx: int, dy: int, quarter_turns: int) -> tuple[int, int]:
    for _ in range(quarter_turns % 4):
        dx, dy = -dy, dx
    return dx, dy


def _circle_masks(
    circles: Sequence[Sequence[float]],
    *,
    width: int = 512,
    height: int = 512,
) -> List[np.ndarray]:
    yy, xx = np.ogrid[:height, :width]
    return [
        (xx - float(cx) * _UNITS_TO_PIXELS) ** 2
        + (yy - float(cy) * _UNITS_TO_PIXELS) ** 2
        <= (float(radius) * _UNITS_TO_PIXELS) ** 2
        for cx, cy, radius in circles
    ]


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


class TripleIntersectionRedPattern(RenderedTaskGenerator):
    task_name = "triple_intersection_red"
    task_family = "in_context_learning"
    video_fps: int = 16
    generate_videos: bool = True

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        image_size: int | tuple[int, int] = _CANVAS,
        video_fps: int = 16,
        generate_videos: bool = True,
        video: bool | None = None,
        **_: Any,
    ) -> None:
        resolved_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if resolved_size != _CANVAS:
            raise ValueError("triple_intersection_red requires a 512x512 canvas")
        if int(video_fps) != 16:
            raise ValueError("triple_intersection_red videos must use 16fps")
        resolved_seed = seed if seed is not None else random_seed
        self.rng = random.Random(resolved_seed)
        self.image_size = _CANVAS
        self.video_fps = 16
        self.generate_videos = bool(video if video is not None else generate_videos)
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / "videogym_triple_intersection_red"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    def _generate_task(self) -> Dict[str, Any]:
        anchor = (self.rng.choice(_ANCHORS), self.rng.choice(_ANCHORS))
        template = self.rng.choice(_TEMPLATES)
        radius = self.rng.choice(_RADII)
        rotation_quarters = self.rng.randrange(4)
        centers: list[tuple[float, float]] = [
            (
                anchor[0] + _rotate_offset(dx, dy, rotation_quarters)[0],
                anchor[1] + _rotate_offset(dx, dy, rotation_quarters)[1],
            )
            for dx, dy in template
        ]
        if any(int(cx) % 4 or int(cy) % 4 for cx, cy in centers):
            raise AssertionError("every circle center must land on a labeled grid intersection")
        circles = [(float(cx), float(cy), float(radius)) for cx, cy in centers]
        masks = _circle_masks(circles)
        triple = masks[0] & masks[1] & masks[2]
        centroid = (
            int(round(sum(cx for cx, _cy in centers) / 3.0)),
            int(round(sum(cy for _cx, cy in centers) / 3.0)),
        )
        if centroid != anchor or any(
            (centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2 > radius ** 2
            for cx, cy in centers
        ):
            raise AssertionError("the semantic central grid point must lie inside all three circles")
        contributions = [
            int((masks[(index + 1) % 3] & masks[(index + 2) % 3] & ~masks[index]).sum())
            for index in range(3)
        ]
        if min(contributions) < 2000:
            raise AssertionError("every circle must materially constrain the triple intersection")
        return {
            "circles": circles,
            "outline_colors": [list(color) for color in _OUTLINE_COLORS],
            "anchor": list(anchor),
            "centroid": centroid,
            "template": [list(offset) for offset in template],
            "radius": radius,
            "rotation_quarters": rotation_quarters,
            "coordinate_system": {
                "x_direction": "right",
                "y_direction": "down",
                "coordinate_min": 0,
                "coordinate_max": _COORDINATE_MAX,
                "central_point_rule": "rounded_arithmetic_mean_of_three_circle_centers",
            },
            "triple_area": int(triple.sum()),
            "circle_contributions": contributions,
            "triple_bbox": _mask_bbox(triple),
            "state_trace": ["input", "triple_final"],
        }

    @staticmethod
    def _phase_layer(task: Dict[str, Any], phase: str, progress: float = 1.0) -> np.ndarray:
        masks = _circle_masks(task["circles"])
        triple = masks[0] & masks[1] & masks[2]
        progress = min(1.0, max(0.0, float(progress)))
        red = np.zeros(triple.shape, dtype=np.float32)
        if phase == "final_transition":
            # This task has one atomic visual answer: fade the complete
            # common lens together rather than inventing a union checkpoint.
            red[triple] = progress
        elif phase == "final":
            red[triple] = 1.0
        return red

    def _render_scene(
        self,
        task: Dict[str, Any],
        *,
        phase: str = "input",
        progress: float = 1.0,
    ) -> Image.Image:
        factor = DOCTRINE_SUPERSAMPLE
        image, sdraw = new_scaled_canvas(512, 512, factor, _BG)
        grid_font = _load_font(9 * factor)
        legend_font = _load_font(11 * factor)
        for coordinate in range(64, 512, 64):
            sdraw.line((coordinate, 0, coordinate, 512), fill=_GRID, width=1)
            sdraw.line((36, coordinate, 512, coordinate), fill=_GRID, width=1)
            task_coordinate = int(round(coordinate / _UNITS_TO_PIXELS))
            sdraw.text((coordinate, 6), str(task_coordinate), font=grid_font, fill=_MUTED, anchor="ma")
            sdraw.text((6, coordinate), str(task_coordinate), font=grid_font, fill=_MUTED, anchor="lm")
        sdraw.text((14, 14), "x →", font=legend_font, fill=_TEXT, anchor="lm")
        sdraw.text((14, 31), "y ↓", font=legend_font, fill=_TEXT, anchor="lm")
        red = self._phase_layer(task, phase, progress)
        if float(red.max()) > 0.0:
            # One flat translucent fill, per the family set-cluster language.
            # The per-pixel opacity is the fade progress times the cluster
            # alpha; the complete footprint appears together.
            high_alpha = np.kron(
                np.rint(red * float(_TRIPLE_ALPHA)).astype(np.uint8),
                np.ones((factor, factor), dtype=np.uint8),
            )
            solid = Image.new("RGB", image.size, _TRIPLE_FILL)
            image.paste(solid, (0, 0), Image.fromarray(high_alpha))
            sdraw = ScaledDraw(ImageDraw.Draw(image), factor)

        for label, ((cx, cy, radius), outline_color) in enumerate(zip(task["circles"], _OUTLINE_COLORS)):
            cx_px, cy_px, radius_px = (
                float(cx) * _UNITS_TO_PIXELS,
                float(cy) * _UNITS_TO_PIXELS,
                float(radius) * _UNITS_TO_PIXELS,
            )
            sdraw.ellipse(
                (
                    cx_px - radius_px,
                    cy_px - radius_px,
                    cx_px + radius_px,
                    cy_px + radius_px,
                ),
                outline=outline_color,
                width=4,
            )
            sdraw.ellipse(
                (cx_px - 3, cy_px - 3, cx_px + 3, cy_px + 3),
                fill=outline_color,
            )
            sdraw.text(
                (cx_px + 7, cy_px - 7),
                "ABC"[label],
                font=legend_font,
                fill=outline_color,
                anchor="lm",
            )
        return finish_scaled_canvas(image, _CANVAS)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The input is a fixed coordinate plane with three labeled circle outlines A, B, and C. "
                "The horizontal coordinate increases to the right and the vertical coordinate increases downward."
            ),
            symbol_legend=(
                "Each circle's letter label sits beside its center mark in the circle's own outline color: A blue, B orange, C purple.",
                "Small colored dots mark the three circle centers.",
            ),
            rules=(
                "Boundary points count as inside. Treat A, B, and C symmetrically and identify the region common to all three.",
                "The target region consists exactly of points inside all three circles; any point outside one or more circles is excluded.",
                "Obtain the central grid point by averaging the three visible circle-center coordinates and rounding each coordinate to the nearest integer.",
            ),
            modality_outputs={
                "ti2v": "Hold the input, fade the complete A intersect B intersect C lens into one flat translucent red as one global update, then hold the terminal state.",
                "solution_image": "Fill exactly A intersect B intersect C with one flat translucent red and preserve every fixed element.",
            },
            preservation_constraints=(
                "Keep all circle centers, radii, outline colors, grid lines, coordinate labels, framing, and camera fixed.",
                "Only the region inside all three circles may change, and it may become red only.",
            ),
            transition_constraints=(
                "Begin with a readable hold of the exact unfilled input.",
                "Fade the complete red triple intersection uniformly at its fixed terminal footprint; do not grow it from the center, privilege a pair, or sweep from one direction.",
                "End with a readable hold of the exact red triple intersection.",
                "Do not use a crossfade, global recolor, teleportation, brief disappearance, overfill, underfill, pan, zoom, crop, camera drift, or flicker.",
            ),
            answer=AnswerContract(
                kind="cartesian_point",
                format_template=CARTESIAN_POINT_ANSWER_SCHEMA,
                format_example=CARTESIAN_POINT_ANSWER_EXAMPLE,
                parser_profile="strict_cartesian_integer_point",
                verifier_profile=_VERIFIER_PROFILE,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_triple_intersection_complete_footprint_fade_v6",
        )

    def _build_prompts(self) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": (
                "Identify the region inside all three circles, color exactly that region red, and return its central grid point."
            ),
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid triple-intersection prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(self, task: Dict[str, Any], *, output_path: Path) -> tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        frames: List[np.ndarray] = []
        input_frame = np.asarray(self._render_scene(task), dtype=np.uint8)
        frames.extend(input_frame.copy() for _ in range(9))
        # The translucent cluster fill moves fewer than 100 intensity levels in
        # total, so the fade uses 0.2-progress steps: each step stays a clearly
        # visible simultaneous update of the complete footprint instead of a
        # sub-threshold shimmer.
        for progress in (0.2, 0.4, 0.6, 0.8):
            frames.append(
                np.asarray(self._render_scene(task, phase="final_transition", progress=progress), dtype=np.uint8)
            )
        final = np.asarray(self._render_scene(task, phase="final"), dtype=np.uint8)
        frames.extend(final.copy() for _ in range(8))
        if len(frames) != 21:
            raise AssertionError("triple_intersection_red video must contain 21 frames")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        contract, prompts = self._build_prompts()
        puzzle_image = self._render_scene(task)
        solution_image = self._render_scene(task, phase="final")

        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        answer_raw = format_cartesian_point_payload(task["centroid"])
        solution_video_path, video_num_frames = self._render_video(task, output_path=video_path)

        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": prompts["question"],
            "answer": answer_raw,
            "structured_answer": {
                "circles": [list(circle) for circle in task["circles"]],
                "outline_colors": task["outline_colors"],
                "anchor": task["anchor"],
                "centroid": list(task["centroid"]),
                "template": task["template"],
                "radius": task["radius"],
                "rotation_quarters": task["rotation_quarters"],
                "coordinate_system": task["coordinate_system"],
                "triple_area": task["triple_area"],
                "triple_bbox": task["triple_bbox"],
                "state_trace": ["input", "triple_final"],
                "region_boundary_policy": "closed_circle_regions_in_task_coordinates",
                "triple_fill_rgb": list(_TRIPLE_FILL),
                "triple_fill_alpha": _TRIPLE_ALPHA,
                "answer_schema": {
                    "schema_version": "cartesian_integer_point_v1",
                    "kind": "cartesian_point",
                    "format_template": CARTESIAN_POINT_ANSWER_SCHEMA,
                    "format_example": CARTESIAN_POINT_ANSWER_EXAMPLE,
                    "parser_profile": "strict_cartesian_integer_point",
                    "final_deliverable": {
                        "ti2v": ["full_video"],
                    },
                    "accepts_multiple_valid_answers": False,
                    "extra_text_allowed": False,
                },
                "intermediate_supervision_schema": {
                    "ti2v": ["input", "triple_final_complete_footprint_fade"],
                },
                "verifier_contract": {
                    "profile": _VERIFIER_PROFILE,
                    "semantic_source": "three_visible_circle_geometries_in_labeled_task_coordinates",
                    "component_scores": list(_VERIFIER_COMPONENTS),
                    "hard_gates": [
                        "strict_parse",
                        "independent_circle_region_replay",
                        "central_coordinate_replay",
                        "semantic_final_image",
                        "whole_video_complete_footprint_triple_fade",
                        "visible_transitions",
                        "fixed_content_preservation",
                        "media_contract",
                    ],
                },
            },
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }
        return sample, puzzle_image, solution_image


from data.in_context_learning.core.polish import polished_icl_generator

TripleIntersectionRedPattern = polished_icl_generator(TripleIntersectionRedPattern)

__all__ = [
    "TripleIntersectionRedPattern",
    "_ANCHORS",
    "_BG",
    "_GRID",
    "_OUTLINE_COLORS",
    "_RADII",
    "_TEMPLATES",
    "_TRIPLE_ALPHA",
    "_TRIPLE_FILL",
    "_circle_masks",
    "_VERIFIER_COMPONENTS",
    "_VERIFIER_PROFILE",
    "np",
]
