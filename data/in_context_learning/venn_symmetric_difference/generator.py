"""Pick the Venn diagram that represents the symmetric difference ``A △ B``.

The task uses one native 512x512 card.  The unshaded source diagram and all
four candidates share exactly the same universe and circle geometry.  Each
candidate is described by a complete mask over the four atomic regions
``outside``, ``a_only``, ``ab_both``, and ``b_only``; exactly one candidate
shades both exclusive crescents and leaves the overlap and exterior unshaded.

If reference authoring uses a visual evidence state, both exclusive crescents
appear together because A-only and B-only have no intrinsic order.  Formal
TI2V uses the shared input-preserving option-border fade.
"""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.pipeline.media import LETTER_OPTIONS, load_font
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.in_context_learning.core.option_delivery import (
    OPTION_MODALITY_OUTPUTS,
    OPTION_PRESERVATION_CONSTRAINTS,
    OPTION_TRANSITION_CONSTRAINTS,
    render_terminal_image,
    render_selection_video,
)
from data.common.pipeline.choice_generator import ChoiceTaskGenerator
from data.common.pipeline.choice_layout import (
    ACCENT_RED,
    EDGE_PADDING,
    GUTTER,
    SOURCE_HEIGHT,
    assemble_choice_order,
    compose_tile_panel,
)


_SHADE_COLOR = (222, 60, 60)
_SHADE_ALPHA = 96
_A_OUTLINE = (63, 111, 212)
_B_OUTLINE = (224, 120, 60)
_UNIVERSE_FILL = (255, 255, 255)
_UNIVERSE_EDGE = (150, 156, 168)
_ATOMIC_REGIONS = ("outside", "a_only", "ab_both", "b_only")
_REGION_MASKS: Mapping[str, Tuple[int, int, int, int]] = {
    "symdiff": (0, 1, 0, 1),
    "a_only": (0, 1, 0, 0),
    "b_only": (0, 0, 0, 1),
    "ab_both": (0, 0, 1, 0),
}
_VERIFIER_PROFILE = "venn_symmetric_difference_atomic_mask_and_visual_trace_replay_v2"
_VERIFIER_COMPONENTS = (
    "parse_score", "source_replay_score", "atomic_mask_score",
    "candidate_uniqueness_score", "visual_parseability_score",
    "state_order_score", "transition_visibility_score", "preservation_score",
    "final_state_score", "media_contract_score",
)
# Sampled layout bounds.  Every sample keeps the same rule, candidate set,
# and answer schema; only the circle geometry varies so the diagram content
# is no longer one constant memorizable raster.
# offset_scale is bounded so the shared lens keeps at least 0.44 of A's
# radius in width even against the smallest sampled B (perceptual margin at
# candidate-plate scale).
_LAYOUT_RANGES = {
    "radius_scale": (0.84, 1.06),
    "offset_scale": (0.46, 0.68),
    "b_radius_ratio": (0.80, 1.0),
    "cy_shift": (-0.03, 0.05),
}
_DEFAULT_LAYOUT = {
    "radius_scale": 1.0,
    "offset_scale": 0.58,
    "b_radius_ratio": 1.0,
    "cy_shift": 0.0,
}


def _venn_geometry(
    box: Tuple[int, int, int, int],
    layout: Mapping[str, float] | None = None,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Return the visible universe, A disc, and B disc boxes.

    The universe frame makes the exterior atomic region explicit.  Circle
    bounds retain clearance from that frame at every supported
    source/candidate size; the sampled ``layout`` scales the base radius,
    centre offset, B-to-A radius ratio, and vertical centre within bounds
    that keep every atomic region readable.
    """

    resolved = {**_DEFAULT_LAYOUT, **dict(layout or {})}
    universe = (box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2)
    width = universe[2] - universe[0]
    height = universe[3] - universe[1]
    cx = (universe[0] + universe[2]) // 2
    cy = (
        (universe[1] + universe[3]) // 2
        + max(1, height // 24)
        + int(round(float(resolved["cy_shift"]) * height))
    )
    base_radius = max(12, int(min(width / 3.45, height / 2.35)))
    radius_a = max(12, int(round(base_radius * float(resolved["radius_scale"]))))
    radius_b = max(10, int(round(radius_a * float(resolved["b_radius_ratio"]))))
    offset = max(5, int(round(radius_a * float(resolved["offset_scale"]))))
    # Clamp so both discs keep at least three pixels inside the frame.
    max_radius = min(cy - universe[1] - 3, universe[3] - cy - 3)
    radius_a = min(radius_a, max_radius)
    radius_b = min(radius_b, max_radius)
    half_span = (offset * 2 + radius_a + radius_b) / 2.0
    span_budget = width / 2.0 - 5.0
    if half_span > span_budget:
        offset = max(5, int(offset - math.ceil(half_span - span_budget)))
    mid_a = cx - offset
    mid_b = cx + offset
    a_box = (mid_a - radius_a, cy - radius_a, mid_a + radius_a, cy + radius_a)
    b_box = (mid_b - radius_b, cy - radius_b, mid_b + radius_b, cy + radius_b)
    return universe, a_box, b_box


def _atomic_region_arrays(
    size: Tuple[int, int],
    box: Tuple[int, int, int, int],
    layout: Mapping[str, float] | None = None,
) -> Dict[str, np.ndarray]:
    """Rasterize a complete, disjoint partition of the visible universe."""

    universe, a_box, b_box = _venn_geometry(box, layout)
    universe_layer = Image.new("L", size, 0)
    ImageDraw.Draw(universe_layer).rectangle(universe, fill=255)
    a_layer = Image.new("L", size, 0)
    ImageDraw.Draw(a_layer).ellipse(a_box, fill=255)
    b_layer = Image.new("L", size, 0)
    ImageDraw.Draw(b_layer).ellipse(b_box, fill=255)
    universe_arr = np.asarray(universe_layer) > 0
    a_arr = (np.asarray(a_layer) > 0) & universe_arr
    b_arr = (np.asarray(b_layer) > 0) & universe_arr
    return {
        "outside": universe_arr & ~a_arr & ~b_arr,
        "a_only": a_arr & ~b_arr,
        "ab_both": a_arr & b_arr,
        "b_only": b_arr & ~a_arr,
    }


def _semantic_mask(region: str) -> Tuple[int, int, int, int]:
    try:
        return _REGION_MASKS[region]
    except KeyError as exc:
        raise ValueError(f"Unsupported Venn region: {region}") from exc


def _combined_region_mask(
    size: Tuple[int, int],
    box: Tuple[int, int, int, int],
    semantic_mask: Sequence[int],
    *,
    a_only_fraction: float = 1.0,
    b_only_fraction: float = 1.0,
    layout: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return a rendered mask, optionally wiping in the exclusive crescents."""

    if len(semantic_mask) != len(_ATOMIC_REGIONS):
        raise ValueError("Venn semantic masks must cover all four atomic regions")
    atoms = _atomic_region_arrays(size, box, layout)
    output = np.zeros((size[1], size[0]), dtype=bool)
    for include, name in zip(semantic_mask, _ATOMIC_REGIONS):
        if int(include) not in (0, 1):
            raise ValueError("Venn semantic-mask entries must be binary")
        if not include:
            continue
        atom = atoms[name].copy()
        if name in {"a_only", "b_only"}:
            ys, xs = np.where(atom)
            fraction = a_only_fraction if name == "a_only" else b_only_fraction
            fraction = min(1.0, max(0.0, float(fraction)))
            if xs.size and fraction < 1.0:
                if fraction <= 0.0:
                    atom[:] = False
                elif name == "a_only":
                    cutoff = int(np.floor(xs.min() + fraction * (xs.max() - xs.min() + 1)))
                    atom[:, cutoff:] = False
                else:
                    cutoff = int(np.ceil(xs.max() - fraction * (xs.max() - xs.min() + 1)))
                    atom[:, :cutoff] = False
        output |= atom
    return output


def _paint_mask(image: Image.Image, mask: np.ndarray) -> None:
    """Paint the marked region as one flat translucent red fill.

    The family set-diagram language uses exactly one encoding for a marked
    region; the former amber fill plus hatch texture double-encoded it."""

    mask_image = Image.fromarray((mask.astype(np.uint8) * _SHADE_ALPHA))
    solid = Image.new("RGB", image.size, _SHADE_COLOR)
    image.paste(solid, (0, 0), mask_image)


def _draw_venn_outline(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    *,
    semantic_mask: Sequence[int] | None = None,
    a_only_fraction: float = 1.0,
    b_only_fraction: float = 1.0,
    layout: Mapping[str, float] | None = None,
) -> None:
    """Render one complete Venn field with labels and optional region shading."""

    from data.common.render_primitives import (
        DOCTRINE_SUPERSAMPLE,
        ScaledDraw,
        finish_scaled_canvas,
    )

    factor = DOCTRINE_SUPERSAMPLE
    # The target canvas may itself be supersampled (compose_tile_panel at the
    # shared floor); paste the native high-resolution layer directly in that
    # case instead of downsampling twice.
    target_factor = max(1, image.width // 512)
    universe, a_box, b_box = _venn_geometry(box, layout)
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    background = tuple(
        int(value)
        for value in image.getpixel((box[0] * target_factor, box[1] * target_factor))
    )
    layer = Image.new("RGB", (box_width * factor, box_height * factor), background)
    sdraw = ScaledDraw(ImageDraw.Draw(layer), factor)

    def _local(rect):
        return (rect[0] - box[0], rect[1] - box[1], rect[2] - box[0], rect[3] - box[1])

    sdraw.rounded_rectangle(_local(universe), radius=6, fill=_UNIVERSE_FILL, outline=_UNIVERSE_EDGE, width=2)
    if semantic_mask is not None:
        mask = _combined_region_mask(
            (box_width, box_height),
            _local(box),
            semantic_mask,
            a_only_fraction=a_only_fraction,
            b_only_fraction=b_only_fraction,
            layout=layout,
        )
        scaled_mask = np.kron(mask, np.ones((factor, factor), dtype=bool))
        _paint_mask(layer, scaled_mask)
        sdraw = ScaledDraw(ImageDraw.Draw(layer), factor)
    circle_width = 3 if box_height >= 120 else 2
    sdraw.ellipse(_local(a_box), fill=None, outline=_A_OUTLINE, width=circle_width)
    sdraw.ellipse(_local(b_box), fill=None, outline=_B_OUTLINE, width=circle_width)
    font_size = 17 if box_height >= 120 else 13
    font = load_font(font_size * factor)
    label_y = min(a_box[1], b_box[1]) - box[1] + max(9, font_size // 2 + 2)
    sdraw.text((a_box[0] - box[0] + 12, label_y), "A", fill=_A_OUTLINE, font=font, anchor="mm")
    sdraw.text((b_box[2] - box[0] - 12, label_y), "B", fill=_B_OUTLINE, font=font, anchor="mm")
    if target_factor == factor:
        image.paste(layer, (box[0] * factor, box[1] * factor))
    else:
        image.paste(
            finish_scaled_canvas(layer, (box_width * target_factor, box_height * target_factor)),
            (box[0] * target_factor, box[1] * target_factor),
        )


def _candidate_boxes() -> Dict[str, Tuple[int, int, int, int]]:
    width = 512
    source_bottom = EDGE_PADDING + SOURCE_HEIGHT
    candidate_top = source_bottom + GUTTER
    candidate_bottom = 512 - EDGE_PADDING
    candidate_height = (candidate_bottom - candidate_top - GUTTER) // 2
    half_width = (width - 2 * EDGE_PADDING - GUTTER) // 2
    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    for index, letter in enumerate(LETTER_OPTIONS):
        row, column = divmod(index, 2)
        left = EDGE_PADDING + column * (half_width + GUTTER)
        top = candidate_top + row * (candidate_height + GUTTER)
        boxes[letter] = (left, top, left + half_width, top + candidate_height)
    return boxes


def _draw_selection_progress(image: Image.Image, letter: str, progress: float) -> None:
    """Trace the answer frame around one candidate without a one-frame hard cut."""

    progress = min(1.0, max(0.0, float(progress)))
    if progress <= 0.0:
        return
    box = _candidate_boxes()[letter]
    if progress >= 1.0:
        ImageDraw.Draw(image).rounded_rectangle(box, radius=8, outline=ACCENT_RED, width=5)
        return
    left, top, right, bottom = box
    points = ((left, top), (right, top), (right, bottom), (left, bottom), (left, top))
    lengths = (right - left, bottom - top, right - left, bottom - top)
    remaining = progress * float(sum(lengths))
    draw = ImageDraw.Draw(image)
    for (start, end, length) in zip(points, points[1:], lengths):
        if remaining <= 0:
            break
        segment = min(float(length), remaining)
        ratio = segment / float(length)
        endpoint = (
            int(round(start[0] + (end[0] - start[0]) * ratio)),
            int(round(start[1] + (end[1] - start[1]) * ratio)),
        )
        draw.line((start, endpoint), fill=ACCENT_RED, width=5)
        remaining -= segment


class VennSymmetricDifferencePattern(ChoiceTaskGenerator):
    task_name = "venn_symmetric_difference"
    task_family = "in_context_learning"
    video_fps = 16

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        video_fps: int = 16,
        generate_videos: bool = True,
        **_kwargs: Any,
    ) -> None:
        resolved_seed = random_seed if random_seed is not None else seed
        self.rng = random.Random(resolved_seed)
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / f"videogym_{self.task_name}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_fps = int(video_fps)
        self.generate_videos = bool(generate_videos)
        self.image_size = (512, 512)
        self._sample_index = 0

    def _generate_task(self) -> Dict[str, Any]:
        layout = {
            name: round(self.rng.uniform(low, high), 4)
            for name, (low, high) in _LAYOUT_RANGES.items()
        }
        return {
            "correct_region": "symdiff",
            "regions": ["symdiff", "a_only", "b_only", "ab_both"],
            "venn_layout": layout,
        }

    @staticmethod
    def _state_trace() -> List[Dict[str, Any]]:
        return [
            {"a_only_fraction": 0.0, "b_only_fraction": 0.0, "selection_progress": 0.0},
            {"a_only_fraction": 0.0, "b_only_fraction": 0.0, "selection_progress": 1.0},
        ]

    def _draw_source(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int],
        task: Dict[str, Any],
    ) -> None:
        state = task.get("_render_state") or self._state_trace()[0]
        show_mask = bool(float(state["a_only_fraction"]) or float(state["b_only_fraction"]))
        _draw_venn_outline(
            image,
            box,
            semantic_mask=_semantic_mask("symdiff") if show_mask else None,
            a_only_fraction=float(state["a_only_fraction"]),
            b_only_fraction=float(state["b_only_fraction"]),
            layout=task.get("venn_layout"),
        )

    @staticmethod
    def _candidate_drawer(
        region: str,
        layout: Mapping[str, float] | None = None,
    ) -> Callable[[Image.Image, Tuple[int, int, int, int]], None]:
        semantic_mask = _semantic_mask(region)

        def drawer(image: Image.Image, box: Tuple[int, int, int, int]) -> None:
            _draw_venn_outline(image, box, semantic_mask=semantic_mask, layout=layout)

        return drawer

    def _make_candidate_drawers(self, task: Dict[str, Any]):
        layout = task.get("venn_layout")
        correct = self._candidate_drawer("symdiff", layout)
        distractors = [self._candidate_drawer(region, layout) for region in ("a_only", "b_only", "ab_both")]
        return correct, distractors, {
            "correct_region": "symdiff",
            "all_regions": ["symdiff", "a_only", "b_only", "ab_both"],
            "correct_value": "symdiff",
            "distractor_values": ["a_only", "b_only", "ab_both"],
        }

    def _render_scene(
        self,
        ordered_regions: Sequence[str],
        correct_letter: str,
        state: Mapping[str, float],
        *,
        layout: Mapping[str, float] | None = None,
    ) -> Image.Image:
        task = {"_render_state": dict(state), "venn_layout": layout}

        def source_drawer(image: Image.Image, box: Tuple[int, int, int, int]) -> None:
            self._draw_source(image, box, task)

        from data.common.render_primitives import DOCTRINE_SUPERSAMPLE

        image = compose_tile_panel(
            draw_source=source_drawer,
            candidate_drawers=[self._candidate_drawer(region, layout) for region in ordered_regions],
            supersample=DOCTRINE_SUPERSAMPLE,
        )
        _draw_selection_progress(image, correct_letter, float(state["selection_progress"]))
        return image

    @staticmethod
    def _contract() -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The supplied input is one card. Its source panel contains an unshaded two-circle Venn diagram, "
                "and the lower half contains four labeled candidates A-D with identical circle and universe geometry."
            ),
            symbol_legend=(
                "The left circle is set A, the right circle is set B, and the light rectangular field is the visible universe containing the exterior region.",
                "A flat translucent red fill means that an atomic region is selected; an unshaded region is excluded.",
                "The lens is A intersect B, while the two non-overlapping crescents are A-only and B-only; the selected candidate is indicated by its complete green outer border.",
            ),
            rules=(
                "A symmetric difference B contains every point that belongs to exactly one of A and B.",
                "Therefore include the complete A-only and B-only crescents, but exclude the overlap lens and the exterior of both circles.",
                "Every candidate uses the same boundaries and differs only in which complete atomic regions are shaded; exactly one candidate satisfies the rule.",
                "Black set outlines remain visible on top of any fill and are not separate selectable regions.",
            ),
            modality_outputs=OPTION_MODALITY_OUTPUTS,
            preservation_constraints=OPTION_PRESERVATION_CONSTRAINTS,
            transition_constraints=OPTION_TRANSITION_CONSTRAINTS,
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="strict_uppercase_option_letter_a_to_d",
                verifier_profile=_VERIFIER_PROFILE,
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_venn_symmetric_difference_trace_v2",
        )

    def _prompts(self) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Choose the candidate whose shaded atomic regions equal A symmetric difference B.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid Venn symmetric-difference prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(
        self,
        correct_letter: str,
        puzzle_path: Path,
        output_path: Path,
    ) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        render_selection_video(
            output_path,
            puzzle_path=puzzle_path,
            boxes=[_candidate_boxes()[label] for label in LETTER_OPTIONS],
            selected_index=LETTER_OPTIONS.index(correct_letter),
        )
        return output_path.as_posix(), 21

    def create_puzzle(self) -> Dict[str, Any]:
        task = self._generate_task()
        contract, prompts = self._prompts()
        correct_drawer, distractor_drawers, meta = self._make_candidate_drawers(task)
        _drawers, correct_letter, choice_indices = assemble_choice_order(
            self.rng, correct_drawer, distractor_drawers
        )
        source_regions = ["symdiff", "a_only", "b_only", "ab_both"]
        ordered_regions = [source_regions[index] for index in choice_indices]
        candidate_values_by_label = {
            LETTER_OPTIONS[index]: region for index, region in enumerate(ordered_regions)
        }
        candidate_masks_by_label = {
            label: list(_semantic_mask(region)) for label, region in candidate_values_by_label.items()
        }

        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"
        puzzle_image = self._render_scene(
            ordered_regions,
            correct_letter,
            self._state_trace()[0],
            layout=task.get("venn_layout"),
        )
        puzzle_image.save(puzzle_path)
        puzzle_image.save(solution_path)
        render_terminal_image(
            solution_path,
            puzzle_path=puzzle_path,
            boxes=[_candidate_boxes()[label] for label in LETTER_OPTIONS],
            selected_index=LETTER_OPTIONS.index(correct_letter),
        )
        solution_video_path, video_num_frames = self._render_video(
            correct_letter, puzzle_path, video_path
        )

        task_meta = {
            **meta,
            "candidate_values_by_label": candidate_values_by_label,
            "atomic_region_order": list(_ATOMIC_REGIONS),
            "candidate_masks_by_label": candidate_masks_by_label,
            "correct_mask": list(_semantic_mask("symdiff")),
            "venn_layout": dict(task.get("venn_layout") or {}),
            "outside_policy": "outside_A_union_B_is_unshaded",
            "boundary_policy": "set_outlines_are_preserved_on_top_of_region_fill",
            "answer_schema": {
                "schema_version": "option_letter_v1",
                "kind": "option_letter",
                "format_template": FOUR_CHOICE_CODEC.schema,
                "format_example": FOUR_CHOICE_CODEC.example,
                "parser_profile": "strict_uppercase_option_letter_a_to_d",
                "final_deliverable": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "intermediate_supervision_schema": {
                "ti2v": ["input_hold", "uniform_complete_green_border_fade", "final_hold"],
            },
            "verifier_contract": {
                "profile": _VERIFIER_PROFILE,
                "semantic_source": "visible_universe_circle_geometry_and_atomic_region_masks",
                "component_scores": list(_VERIFIER_COMPONENTS),
                "hard_gates": [
                    "strict_parse", "independent_atomic_mask_replay", "candidate_uniqueness",
                    "semantic_final_image", "whole_video_uniform_complete_border_fade",
                    "fixed_content_preservation", "media_contract",
                ],
            },
        }
        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": prompts["question"],
            "answer": correct_letter,
            "correct_option": correct_letter,
            "structured_answer": {
                "correct_option": correct_letter,
                "task_meta": task_meta,
                "state_trace": self._state_trace(),
                "render_geometry": {
                    "canvas_size": [512, 512],
                    "venn_layout": dict(task.get("venn_layout") or {}),
                    "source_content_box": [30, 48, 482, 236],
                    "candidate_content_boxes": {
                        "A": [26, 294, 239, 362],
                        "B": [273, 294, 486, 362],
                        "C": [26, 418, 239, 486],
                        "D": [273, 418, 486, 486],
                    },
                    "candidate_panel_boxes": {
                        label: list(_candidate_boxes()[label]) for label in "ABCD"
                    },
                    "atomic_region_order": list(_ATOMIC_REGIONS),
                    "shade_rgb": list(_SHADE_COLOR),
                    "shade_alpha": _SHADE_ALPHA,
                    "selection_rgb": list(ACCENT_RED),
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


from data.in_context_learning.core.polish import polished_icl_generator

VennSymmetricDifferencePattern = polished_icl_generator(VennSymmetricDifferencePattern)

__all__ = [
    "VennSymmetricDifferencePattern",
    "_ATOMIC_REGIONS",
    "_REGION_MASKS",
    "_atomic_region_arrays",
    "_semantic_mask",
    "_venn_geometry",
    "_VERIFIER_PROFILE",
    "_VERIFIER_COMPONENTS",
]
