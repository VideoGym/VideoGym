from __future__ import annotations

from data.common.base import RenderedTaskGenerator

import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.common.assets import font_path
from data.common.render_primitives import DOCTRINE_SUPERSAMPLE, finish_scaled_canvas, new_scaled_canvas
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts

pattern_instructions = {
    "shape_outline_fill": (
        "A clean square canvas shows a two-row analogy A:B :: C:? where the top row demonstrates a style transformation "
        "between filled and outlined shapes. The task is to apply the same style change to the bottom-row query shape."
    ),
}

_SHAPES: Tuple[str, ...] = ("circle", "square", "triangle", "diamond", "hexagon")
_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (61, 110, 183),
    (190, 86, 86),
    (73, 144, 110),
    (156, 95, 192),
)
_PREVIEW_COLOR = (151, 158, 168)
# Layout in unscaled 512-square coordinates; the verifier derives its slot
# centres and radius from these fractions instead of duplicating pixel values.
_SLOT_FRACTIONS: Dict[str, Tuple[float, float]] = {
    "A": (0.28, 0.28),
    "B": (0.72, 0.28),
    "C": (0.28, 0.72),
    "D": (0.72, 0.72),
}
_RADIUS_FRACTION = 0.08
_VERIFIER_PROFILE = "shape_identity_and_fill_style_visual_replay_v3"
_VERIFIER_COMPONENTS = (
    "parse_score",
    "source_replay_score",
    "shape_identity_score",
    "style_direction_score",
    "visual_parseability_score",
    "state_order_score",
    "transition_visibility_score",
    "preservation_score",
    "final_state_score",
    "media_contract_score",
)
def _normalize_canvas_size(value: object) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return 512, 512

def _load_font(size: int) -> ImageFont.ImageFont:
    return ImageFont.truetype(str(font_path()), size=size)

def _draw_shape(draw: ImageDraw.ImageDraw, *, shape: str, center: Tuple[float, float], radius: float, color: Tuple[int, int, int], style: str, line_width: int = 4) -> None:
    cx, cy = center
    fill = color if style == "filled" else None
    outline = color
    if shape == "circle":
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=outline, width=line_width)
        return
    if shape == "square":
        draw.rectangle((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=outline, width=line_width)
        return
    if shape == "triangle":
        points = [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)]
    elif shape == "diamond":
        points = [(cx, cy - radius), (cx - radius, cy), (cx, cy + radius), (cx + radius, cy)]
    else:
        points = [
            (cx + radius * np.cos(angle), cy + radius * np.sin(angle))
            for angle in [np.pi / 3 * index - np.pi / 6 for index in range(6)]
        ]
    draw.polygon(points, fill=fill, outline=outline, width=line_width)

class ShapeOutlineFillPattern(RenderedTaskGenerator):
    task_name = "shape_outline_fill"
    task_family = "in_context_learning"
    video_fps: int = 16
    generate_videos: bool = True

    def __init__(self, output_dir: Path | str | None = None, **kwargs: Any) -> None:
        self.image_size = _normalize_canvas_size(kwargs.get("image_size", 512))
        if self.image_size != (512, 512):
            raise ValueError("shape_outline_fill requires a 512x512 canvas")
        self.video_fps = int(kwargs.get("video_fps", self.video_fps))
        if self.video_fps != 16:
            raise ValueError("shape_outline_fill requires 16fps video")
        self.generate_videos = bool(kwargs.get("generate_videos", self.generate_videos))
        seed = kwargs.get("random_seed", kwargs.get("seed"))
        self.rng = random.Random(seed)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_shape_outline_fill"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    def _generate_task(self) -> Dict[str, Any]:
        shape_a, shape_c = self.rng.sample(_SHAPES, 2)
        color = self.rng.choice(_COLORS)
        style_from, style_to = self.rng.choice((("filled", "outline"), ("outline", "filled")))
        answer = f"{shape_c}:{style_to}"
        return {
            "shape_a": shape_a,
            "shape_c": shape_c,
            "style_from": style_from,
            "style_to": style_to,
            "color": color,
            "answer": answer,
        }

    @staticmethod
    def _state_trace() -> list[Dict[str, str]]:
        return [
            {"answer_state": "hidden"},
            {"answer_state": "shape_preview"},
            {"answer_state": "final"},
        ]

    def _render_board(self, task: Dict[str, Any], *, answer_state: str) -> Image.Image:
        if answer_state not in {"hidden", "shape_preview", "final"}:
            raise ValueError(f"unsupported answer state: {answer_state}")
        factor = DOCTRINE_SUPERSAMPLE
        width, height = self.image_size
        image, sdraw = new_scaled_canvas(width, height, factor, (255, 255, 255))
        label_font = _load_font(max(16, width // 24) * factor)
        positions = {
            label: (width * fx, height * fy)
            for label, (fx, fy) in _SLOT_FRACTIONS.items()
        }
        radius = width * _RADIUS_FRACTION
        lw = 4
        _draw_shape(sdraw, shape=task["shape_a"], center=positions["A"], radius=radius, color=task["color"], style=task["style_from"], line_width=lw)
        _draw_shape(sdraw, shape=task["shape_a"], center=positions["B"], radius=radius, color=task["color"], style=task["style_to"], line_width=lw)
        _draw_shape(sdraw, shape=task["shape_c"], center=positions["C"], radius=radius, color=task["color"], style=task["style_from"], line_width=lw)
        if answer_state == "final":
            _draw_shape(sdraw, shape=task["shape_c"], center=positions["D"], radius=radius, color=task["color"], style=task["style_to"], line_width=lw)
        elif answer_state == "shape_preview":
            _draw_shape(sdraw, shape=task["shape_c"], center=positions["D"], radius=radius, color=_PREVIEW_COLOR, style="outline", line_width=lw)
        else:
            sdraw.text(positions["D"], "?", anchor="mm", font=_load_font(max(28, width // 10) * factor), fill=(96, 92, 86))
        for label, pos in positions.items():
            sdraw.text((pos[0], pos[1] + radius + 28), label, anchor="mm", font=label_font, fill=(92, 88, 82))
        return finish_scaled_canvas(image, (width, height))

    def _render_video(self, task: Dict[str, Any], *, output_path: Path) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        input_image = self._render_board(task, answer_state="hidden").convert("RGB")
        preview_image = self._render_board(task, answer_state="shape_preview").convert("RGB")
        final_image = self._render_board(task, answer_state="final").convert("RGB")

        # Only slot D changes.  Blend that bounded patch through each
        # semantic transition so A/B/C, labels, and the frame remain exact;
        # a whole-frame crossfade or a hard state cut is never authored.
        change_box = (315, 315, 423, 423)

        def local_blend(before: Image.Image, after: Image.Image, fraction: float) -> np.ndarray:
            result = before.copy()
            blended = Image.blend(
                before.crop(change_box),
                after.crop(change_box),
                min(1.0, max(0.0, float(fraction))),
            )
            result.paste(blended, (change_box[0], change_box[1]))
            return np.asarray(result, dtype=np.uint8)

        frames: list[np.ndarray] = [np.asarray(input_image, dtype=np.uint8).copy() for _ in range(6)]
        for fraction in (0.2, 0.4, 0.6, 0.8):
            frames.append(local_blend(input_image, preview_image, fraction))
        preview = np.asarray(preview_image, dtype=np.uint8)
        frames.extend(preview.copy() for _ in range(5))
        for fraction in (0.2, 0.4, 0.6, 0.8):
            frames.append(local_blend(preview_image, final_image, fraction))
        final = np.asarray(final_image, dtype=np.uint8)
        frames.extend(final.copy() for _ in range(6))
        if len(frames) != 25:
            raise AssertionError("shape_outline_fill video must contain 25 frames")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The supplied image shows a two-row analogy A:B :: C:D. A and B have the same shape and color but opposite fill styles; "
                "C has a different shape in the same starting style as A, and D is the only missing slot."
            ),
            symbol_legend=(
                "A, B, C, and D label the four analogy slots from top-left to bottom-right.",
                "A solid interior means filled; a white interior with a colored border means outline.",
                "A neutral gray outline in an intermediate state is only a shape-preservation preview, not the final style.",
            ),
            rules=(
                "Infer whether A changes from filled to outline or from outline to filled when producing B.",
                "Preserve C's shape, color, size, and position, and apply exactly the same fill-style change to produce D.",
                "The final payload names C's shape followed by D's final style; the only allowed styles are filled and outline.",
            ),
            modality_outputs={
                "ti2v": (
                    "Generate a fixed-camera video. First hold the exact input. "
                    "Then replace only D's question mark with a neutral gray outline of C's unchanged shape, holding that preview "
                    "briefly. Next transform only that preview into C's original source color and the fill style demonstrated by A to B: "
                    "for a filled result, grow the colored interior continuously while keeping the outline aligned; for an outline result, "
                    "replace the neutral stroke continuously with the source-colored stroke while the interior remains white. Hold the exact "
                    "solved state. A, B, C, labels, slot geometry, background, framing, and camera stay fixed throughout."
                ),
                "solution_image": (
                    "Render one solved analogy. Replace only D's question mark with C's exact shape, size, center, and source "
                    "color in the final filled-or-outline style demonstrated by A to B. Keep A, B, C, all labels, slot positions, white "
                    "background, framing, spacing, and camera unchanged. Do not add a preview, arrows, formulas, highlights, candidates, "
                    "or an answer banner."
                ),
            },
            preservation_constraints=(
                "Keep A, B, and C, all four slot labels, the white background, framing, object positions, object sizes, and source color unchanged.",
                "Only the question mark and the content of slot D may change; do not add candidates, titles, formulas, or answer banners.",
            ),
            transition_constraints=(
                "Begin with a readable hold of the exact puzzle image.",
                "Replace the question mark with one neutral outline preview that preserves C's shape before applying the inferred style.",
                "Replace the preview with the final colored shape without moving or mutating A, B, C, labels, or camera, then hold the final state.",
                "Do not hard-cut, crossfade the frame, teleport the final shape, flicker, zoom, pan, crop, or briefly remove any fixed source object.",
            ),
            answer=AnswerContract(
                kind="typed_text",
                format_template="Answer: <shape>:<style>.",
                parser_profile="strict_lowercase_shape_colon_style",
                verifier_profile=_VERIFIER_PROFILE,
                format_example="Answer: octagon:striped.",
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_shape_outline_fill_trace_v3",
        )

    def _build_prompts(self, answer_raw: str) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Infer the fill-style transformation from A to B and apply it to C.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts, answer_payload=answer_raw)
        if issues:
            raise ValueError(f"Invalid shape-outline-fill prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        contract, prompts = self._build_prompts(task["answer"])
        puzzle_image = self._render_board(task, answer_state="hidden")
        solution_image = self._render_board(task, answer_state="final")

        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        answer_raw = task["answer"]
        solution_video_path, video_num_frames = self._render_video(task, output_path=sample_dir / f"{sample_id}__solution.webm")
        state_trace = self._state_trace()

        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": prompts["question"],
            "answer": answer_raw,
            "structured_answer": {
                "shape_a": task["shape_a"],
                "shape_c": task["shape_c"],
                "style_from": task["style_from"],
                "style_to": task["style_to"],
                "source_color": list(task["color"]),
                "state_trace": state_trace,
                "multiple_solution_policy": "unique",
                "answer_schema": {
                    "schema_version": "shape_style_v1",
                    "kind": "shape_style_pair",
                    "format_template": "Answer: <shape>:<style>.",
                    "parser_profile": "strict_lowercase_shape_colon_style",
                    "final_deliverable": {
                        "ti2v": ["full_video"],
                    },
                    "accepts_multiple_valid_answers": False,
                    "extra_text_allowed": False,
                },
                "intermediate_supervision_schema": {
                    "ti2v": ["input", "shape_preview", "final"],
                },
                "verifier_contract": {
                    "profile": _VERIFIER_PROFILE,
                    "semantic_source": "visible_A_to_B_fill_style_change_and_C_shape_identity",
                    "component_scores": list(_VERIFIER_COMPONENTS),
                    "hard_gates": [
                        "strict_parse",
                        "independent_shape_and_style_replay",
                        "semantic_final_image",
                        "whole_video_state_order",
                        "visible_local_transitions",
                        "fixed_content_preservation",
                        "media_contract",
                    ],
                },
                "difficulty_parameters": {
                    "shape_vocabulary_size": len(_SHAPES),
                    "style_direction": f"{task['style_from']}_to_{task['style_to']}",
                },
                "render_geometry": {
                    # Derived from the renderer's own layout fractions so the
                    # verifier and the raster can never drift apart.
                    "slot_centers": {
                        label: [512 * fx, 512 * fy]
                        for label, (fx, fy) in _SLOT_FRACTIONS.items()
                    },
                    "shape_radius": 512 * _RADIUS_FRACTION,
                    "preview_rgb": list(_PREVIEW_COLOR),
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

ShapeOutlineFillPattern = polished_icl_generator(ShapeOutlineFillPattern)

__all__ = [
    "ShapeOutlineFillPattern",
    "np",
    "_VERIFIER_COMPONENTS",
    "_VERIFIER_PROFILE",
]
