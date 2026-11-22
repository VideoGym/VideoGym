"""Object-subtraction pattern induction with native single-card media.

Three before/after pairs demonstrate one invariant deletion rule.  Shape,
colour, position, and bounding-box size vary across examples, while the
filled geometric area is the only property that consistently identifies the
missing object.  The target has four candidates, each of which removes one
different object from the same preserved slot layout.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import encode_rgb_frames_to_webm, pad_frames_to_wan_compatible_length
from data.common.pipeline.media import LETTER_OPTIONS, load_font
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.in_context_learning.core.option_delivery import (
    OPTION_MODALITY_OUTPUTS,
    OPTION_PRESERVATION_CONSTRAINTS,
    OPTION_TRANSITION_CONSTRAINTS,
)
from data.common.pipeline.rendered_task import RenderedTaskGenerator
from data.in_context_learning.core.pattern_drawing import draw_shape


pattern_instructions = {
    "object_subtraction": (
        "A single card shows three before-and-after examples, one target scene, and four labeled candidate "
        "after-scenes. Infer the one object-deletion rule demonstrated by every example and choose the matching target result."
    )
}

_BG = (248, 247, 244)
_PANEL = (255, 255, 255)
_PANEL_EDGE = (188, 184, 178)
_SCENE_EDGE = (220, 217, 211)
_TEXT = (48, 45, 42)
_SUBTEXT = (108, 103, 97)
_EVIDENCE = (232, 143, 48)
_TARGET_MARK = (31, 158, 168)
_CORRECT = (52, 158, 91)
_PALETTE = (
    (225, 76, 73),
    (62, 143, 218),
    (79, 166, 94),
    (239, 177, 43),
    (153, 82, 184),
    (236, 111, 55),
    (35, 177, 191),
    (126, 96, 82),
)
_SHAPES = ("circle", "square", "triangle", "diamond")
_SLOT_FRACTIONS = ((0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72))
_EXAMPLE_PANELS = ((18, 44, 168, 252), (181, 44, 331, 252), (344, 44, 494, 252))
_TARGET_PANEL = (18, 263, 256, 494)
_OPTION_PANELS = ((274, 281, 378, 382), (390, 281, 494, 382), (274, 389, 378, 489), (390, 389, 494, 489))

_AREA_COEFFICIENTS = {
    "circle": math.pi / 4.0,
    "square": 1.0,
    "triangle": 0.5,
    "diamond": 0.5,
}

_VERIFIER_PROFILE = "object_subtraction_area_candidate_and_visual_state_replay_v1"
_VERIFIER_COMPONENTS = {
    "ti2v": (
        "semantic_replay_score",
        "frame_parse_score",
        "state_sequence_score",
        "target_removal_score",
        "static_preservation_score",
        "frame_format_score",
    ),
}
def _object_area(obj: Dict[str, Any]) -> float:
    return _AREA_COEFFICIENTS[str(obj["shape"])] * float(obj["size"]) ** 2


def _largest_index(objects: Sequence[Dict[str, Any]]) -> int:
    """Return the unique maximum-area slot, rejecting ambiguous ties."""

    if len(objects) != 4:
        raise ValueError("object_subtraction requires exactly four source objects")
    ranked = sorted(((_object_area(obj), index) for index, obj in enumerate(objects)), reverse=True)
    if math.isclose(ranked[0][0], ranked[1][0], rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("object_subtraction does not admit equal maximum areas")
    return ranked[0][1]


def _without_slot(objects: Sequence[Dict[str, Any]], slot: int) -> List[Dict[str, Any]]:
    return [dict(obj) for obj in objects if int(obj["slot"]) != int(slot)]


def _draw_centered(sdraw: ScaledDraw, xy: Tuple[float, float], text: str, *, size: int, fill: Tuple[int, int, int]) -> None:
    sdraw.text(xy, text, font=load_font(size * sdraw.factor), fill=fill, anchor="mm")


def _render_scene(
    sdraw: ScaledDraw,
    box: Sequence[float],
    objects: Sequence[Dict[str, Any]],
    *,
    highlight_slot: int | None = None,
    empty_slot_mark: int | None = None,
) -> None:
    left, top, right, bottom = (float(value) for value in box)
    sdraw.rounded_rectangle(tuple(box), radius=8, fill=_PANEL, outline=_SCENE_EDGE, width=1)
    width = right - left
    height = bottom - top
    scale = min(width, height) / 100.0
    object_by_slot = {int(obj["slot"]): obj for obj in objects}
    for slot, (fx, fy) in enumerate(_SLOT_FRACTIONS):
        center = (left + width * fx, top + height * fy)
        obj = object_by_slot.get(slot)
        if obj is not None:
            rendered_size = float(obj["size"]) * scale
            draw_shape(
                sdraw,
                str(obj["shape"]),
                center,
                rendered_size,
                fill=tuple(int(value) for value in obj["color"]),
                outline=(39, 37, 35),
                width=2,
            )
            if highlight_slot == slot:
                half = rendered_size / 2.0 + 5
                sdraw.rounded_rectangle(
                    (center[0] - half, center[1] - half, center[0] + half, center[1] + half),
                    radius=7,
                    outline=_TARGET_MARK if empty_slot_mark is None else _EVIDENCE,
                    width=3,
                )
        elif empty_slot_mark == slot:
            radius = 7
            sdraw.ellipse(
                (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
                outline=_EVIDENCE,
                width=3,
            )


class ObjectSubtractionPattern(RenderedTaskGenerator):
    TASK_NAME = "object_subtraction"
    TASK_FAMILY = "in_context_learning"

    def _generate_scene(
        self,
        *,
        removed_slot: int,
        removed_shape: str | None = None,
        force_extent_trap: bool = False,
    ) -> List[Dict[str, Any]]:
        colors = self.rng.sample(list(_PALETTE), k=4)
        if force_extent_trap:
            # The removed square has greater filled area, while a surviving
            # triangle/diamond has the larger bounding extent.  This rules out
            # a shortcut based on width/height alone.
            removed_shape = "square"
            removed_size = self.rng.randint(31, 33)
            trap_slot = self.rng.choice([slot for slot in range(4) if slot != removed_slot])
            trap_shape = self.rng.choice(("triangle", "diamond"))
            trap_size = self.rng.randint(38, 40)
        else:
            removed_shape = removed_shape or self.rng.choice(_SHAPES)
            removed_size = self.rng.randint(39, 44)
            trap_slot = -1
            trap_shape = "triangle"
            trap_size = 0

        objects: List[Dict[str, Any]] = []
        for slot in range(4):
            if slot == removed_slot:
                shape = removed_shape
                size = removed_size
            elif slot == trap_slot:
                shape = trap_shape
                size = trap_size
            else:
                shape = self.rng.choice(_SHAPES)
                size = self.rng.randint(17, 24)
            objects.append(
                {
                    "slot": slot,
                    "shape": shape,
                    "color": list(colors[slot]),
                    "size": int(size),
                }
            )

        actual_largest = _largest_index(objects)
        if actual_largest != removed_slot:
            raise RuntimeError("constructed removal object is not the unique maximum-area object")
        areas = sorted((_object_area(obj) for obj in objects), reverse=True)
        if areas[0] / areas[1] < 1.08:
            raise RuntimeError("maximum-area gap is too small for stable visual comparison")
        return objects

    def _generate_task(self) -> Dict[str, Any]:
        example_slots = [self.rng.randrange(4) for _ in range(3)]
        while len(set(example_slots)) < 2:
            example_slots = [self.rng.randrange(4) for _ in range(3)]

        removed_shapes = ["square", *self.rng.sample(["circle", "triangle", "diamond"], k=2)]
        examples: List[Dict[str, Any]] = []
        for index, removed_slot in enumerate(example_slots):
            before = self._generate_scene(
                removed_slot=removed_slot,
                removed_shape=removed_shapes[index],
                force_extent_trap=index == 0,
            )
            after = _without_slot(before, removed_slot)
            examples.append(
                {
                    "before": before,
                    "after": after,
                    "removed_slot": removed_slot,
                }
            )

        target_removed = self.rng.randrange(4)
        # Half of the targets carry the same extent trap as demonstration one:
        # a surviving triangle/diamond spans a wider bounding box than the
        # removed square, so ranking objects by width or height alone stays
        # uninformative on the query, not just inside the demonstrations.
        target_before = self._generate_scene(
            removed_slot=target_removed,
            force_extent_trap=self.rng.random() < 0.5,
        )
        target_after = _without_slot(target_before, target_removed)
        candidate_slots = list(range(4))
        self.rng.shuffle(candidate_slots)
        candidates = [_without_slot(target_before, slot) for slot in candidate_slots]
        answer_index = candidate_slots.index(target_removed)
        return {
            "examples": examples,
            "target_before": target_before,
            "target_after": target_after,
            "target_removed_slot": target_removed,
            "candidate_removed_slots": candidate_slots,
            "candidates": candidates,
            "answer_index": answer_index,
        }

    @staticmethod
    def _state_trace() -> List[Dict[str, Any]]:
        return [
            {"evidence_count": 0, "target_highlight": False, "solved": False},
            {"evidence_count": 1, "target_highlight": False, "solved": False},
            {"evidence_count": 2, "target_highlight": False, "solved": False},
            {"evidence_count": 3, "target_highlight": False, "solved": False},
            {"evidence_count": 3, "target_highlight": True, "solved": False},
            {"evidence_count": 3, "target_highlight": False, "solved": True},
        ]

    def _render_state(
        self,
        task: Dict[str, Any],
        *,
        evidence_count: int = 0,
        target_highlight: bool = False,
        solved: bool = False,
    ) -> Image.Image:
        image, sdraw = new_scaled_canvas(512, 512, DOCTRINE_SUPERSAMPLE, _BG)

        for index, panel in enumerate(_EXAMPLE_PANELS):
            left, top, right, bottom = panel
            sdraw.rounded_rectangle(tuple(panel), radius=10, fill=_PANEL, outline=_PANEL_EDGE, width=2)
            _draw_centered(sdraw, ((left + right) / 2.0, top + 14), str(index + 1), size=12, fill=_SUBTEXT)
            before_box = (left + 10, top + 27, right - 10, top + 101)
            after_box = (left + 10, top + 124, right - 10, bottom - 9)
            example = task["examples"][index]
            marked = index < int(evidence_count)
            removed_slot = int(example["removed_slot"])
            _render_scene(
                sdraw,
                before_box,
                example["before"],
                highlight_slot=removed_slot if marked else None,
                empty_slot_mark=removed_slot if marked else None,
            )
            _draw_centered(sdraw, ((left + right) / 2.0, top + 112), "↓", size=16, fill=_SUBTEXT)
            _render_scene(
                sdraw,
                after_box,
                example["after"],
                empty_slot_mark=removed_slot if marked else None,
            )

        target_panel = _TARGET_PANEL
        sdraw.rounded_rectangle(tuple(target_panel), radius=10, fill=_PANEL, outline=_PANEL_EDGE, width=2)
        target_objects = task["target_after"] if solved else task["target_before"]
        _render_scene(
            sdraw,
            (target_panel[0] + 16, target_panel[1] + 36, target_panel[2] - 16, target_panel[3] - 11),
            target_objects,
            highlight_slot=int(task["target_removed_slot"]) if target_highlight else None,
        )

        for index, panel in enumerate(_OPTION_PANELS):
            left, top, right, bottom = panel
            border = _CORRECT if solved and index == int(task["answer_index"]) else _PANEL_EDGE
            border_width = 4 if solved and index == int(task["answer_index"]) else 2
            sdraw.rounded_rectangle(tuple(panel), radius=9, fill=_PANEL, outline=border, width=border_width)
            _draw_centered(sdraw, (left + 13, top + 13), LETTER_OPTIONS[index], size=12, fill=_TEXT)
            _render_scene(sdraw, (left + 8, top + 22, right - 8, bottom - 7), task["candidates"][index])
        return finish_scaled_canvas(image, (512, 512))

    def _render_puzzle_image(self, task: Dict[str, Any]) -> Image.Image:
        return self._render_state(task)

    def _render_solution_image(self, task: Dict[str, Any]) -> Image.Image:
        return self._render_state(task, **self._state_trace()[-1])

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        holds = (5, 3, 3, 3, 4, 7)  # 25 frames: 4N+1 at 16 fps.
        frames: List[np.ndarray] = []
        for state, hold in zip(self._state_trace(), holds):
            frame = np.asarray(self._render_state(task, **state), dtype=np.uint8)
            frames.extend(frame.copy() for _ in range(hold))
        frames = [np.asarray(frame, dtype=np.uint8) for frame in pad_frames_to_wan_compatible_length(frames)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single card contains three numbered before/after examples across the top, a larger target scene in "
                "the lower left, and four candidates A-D in the lower right. "
                "Every scene uses four fixed 2x2 slots so a removed object's location remains visible as an empty slot."
            ),
            symbol_legend=(
                "Filled circles, squares, triangles, and diamonds are the objects; colour identifies an object within one scene but does not define the shared rule.",
                "A downward arrow separates each example's before scene from its after scene.",
                "In intermediate evidence, orange marks identify removed objects and a cyan outline previews the target result without changing the candidates.",
            ),
            rules=(
                "Each example removes exactly one object and leaves every other object's shape, colour, size, and slot unchanged.",
                "Infer the one deterministic visible-property rule that accounts for all three example removals, then apply that same rule once to the target.",
                "Each candidate removes a different target object; choose the unique candidate whose remaining objects and empty slot match the inferred rule.",
                "The visible demonstrations and target have no ties under the demonstrated property, so exactly one target object is valid.",
            ),
            modality_outputs=OPTION_MODALITY_OUTPUTS,
            preservation_constraints=OPTION_PRESERVATION_CONSTRAINTS,
            transition_constraints=OPTION_TRANSITION_CONSTRAINTS,
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="uppercase_option_letter",
                verifier_profile="unique_maximum_filled_area_and_candidate_slot_replay",
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="native_object_area_subtraction_trace_v3_low_text",
        )

    def _prompts(self) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Infer the one object-deletion rule demonstrated by all three example pairs and choose the matching target after-scene.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid object-subtraction prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        contract, prompts = self._prompts()
        answer_raw = LETTER_OPTIONS[int(task["answer_index"])]
        sample_id = f"{self.TASK_NAME}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"

        puzzle_image = self._render_puzzle_image(task)
        solution_image = self._render_solution_image(task)
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)
        solution_video_path, video_num_frames = self._render_video(task, video_path)

        structured_examples = []
        for example in task["examples"]:
            structured_examples.append(
                {
                    **example,
                    "areas": [round(_object_area(obj), 6) for obj in example["before"]],
                }
            )
        structured_answer = {
            "answer_schema": {
                "schema_version": "option_letter_v1", "kind": "option_letter",
                "format_template": FOUR_CHOICE_CODEC.schema,
                "format_example": FOUR_CHOICE_CODEC.example,
                "parser_profile": "strict_uppercase_option_letter_a_to_d",
                "final_deliverable": { "ti2v": ["full_video"]},
                "accepts_multiple_valid_answers": False, "extra_text_allowed": False,
            },
            "intermediate_supervision_schema": {
                "ti2v": ["input_hold", "uniform_complete_green_border_fade", "final_hold"],
            },
            "rule": "remove_unique_maximum_filled_area",
            "tie_policy": "reject_equal_maximum_area_instances",
            "examples": structured_examples,
            "target_before": task["target_before"],
            "target_areas": [round(_object_area(obj), 6) for obj in task["target_before"]],
            "target_after": task["target_after"],
            "target_removed_slot": int(task["target_removed_slot"]),
            "candidate_removed_slots": [int(value) for value in task["candidate_removed_slots"]],
            "candidates": task["candidates"],
            "answer_index": int(task["answer_index"]),
            "correct_option": answer_raw,
            "state_trace": self._state_trace(),
            "render_geometry": {
                "example_panels": [list(box) for box in _EXAMPLE_PANELS],
                "target_panel": list(_TARGET_PANEL),
                "option_panels": [list(box) for box in _OPTION_PANELS],
                "slot_fractions": [list(value) for value in _SLOT_FRACTIONS],
                "area_coefficients": dict(_AREA_COEFFICIENTS),
                "evidence_rgb": list(_EVIDENCE),
                "target_mark_rgb": list(_TARGET_MARK),
                "correct_rgb": list(_CORRECT),
            },
            "verifier_profile": _VERIFIER_PROFILE,
            "verifier_component_scores": {
                mode: list(names) for mode, names in _VERIFIER_COMPONENTS.items()
            },
        }
        return (
            {
                "id": sample_id,
                "task_name": self.TASK_NAME,
                "task_family": self.TASK_FAMILY,
                "task_contract": contract.to_dict(),
                "question": prompts["question"],
                "answer": answer_raw,
                "correct_option": answer_raw,
                "structured_answer": structured_answer,
                "verifier_contract": {
                    "profile": _VERIFIER_PROFILE,
                    "semantic_source": "examples_before_after_and_target_candidates",
                    "component_scores": {
                        mode: list(names) for mode, names in _VERIFIER_COMPONENTS.items()
                    },
                    "binary_predicate": (
                        "strict artifact parse and independent maximum-filled-area replay and exact candidate/final-state agreement"
                    ),
                },
                "ti2v_prompt": prompts["ti2v_prompt"],
                "image": puzzle_path.as_posix(),
                "reasoning_image": puzzle_path.as_posix(),
                "solution_image_path": solution_path.as_posix(),
                "solution_video_path": solution_video_path,
                "video_fps": self.video_fps if solution_video_path else None,
                "video_num_frames": video_num_frames,
            },
            puzzle_image,
            solution_image,
        )

    def _question_text(self, task: Dict[str, Any]) -> str:
        return "Infer the demonstrated deletion rule and choose the matching target after-scene."

    def _answer_raw(self, task: Dict[str, Any]) -> str:
        return LETTER_OPTIONS[int(task["answer_index"])]

    def _structured_answer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return dict(task)

    def _visual_input_description(self, task: Dict[str, Any]) -> str:
        return self._contract().input_description

    def _visual_output_description(self, task: Dict[str, Any]) -> str:
        return self._contract().modality_outputs["solution_image"]


from data.in_context_learning.core.polish import polished_icl_generator

ObjectSubtractionPattern = polished_icl_generator(ObjectSubtractionPattern)

__all__ = [
    "ObjectSubtractionPattern",
    "_AREA_COEFFICIENTS",
    "_VERIFIER_COMPONENTS",
    "_VERIFIER_PROFILE",
    "_largest_index",
    "_object_area",
]
