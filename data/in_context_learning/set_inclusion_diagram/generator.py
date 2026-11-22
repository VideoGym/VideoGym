"""Infer a two-set relation from visual membership lists.

The source contains two rows of numbered element tokens for sets X and Y.
The same number and colour denote the same element in both rows.  The four
candidate panels are pure diagrams for proper subset in either direction,
partial overlap, and disjointness; no candidate contains a textual relation
name that can shortcut the visual comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.video_encoding import encode_rgb_frames_to_webm, pad_frames_to_wan_compatible_length
from data.common.pipeline.media import LETTER_OPTIONS, downsample, load_font
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.in_context_learning.core.option_delivery import (
    OPTION_MODALITY_OUTPUTS,
    OPTION_PRESERVATION_CONSTRAINTS,
    OPTION_TRANSITION_CONSTRAINTS,
)
from data.common.pipeline.rendered_task import RenderedTaskGenerator
from data.in_context_learning.core.transitions import changed_region_opacity_frames

_SCALE = 4
_VERIFIER_PROFILE = "set_membership_relation_and_visual_trace_replay_v2"

pattern_instructions = {
    "set_inclusion_diagram": (
        "A single card lists the numbered elements of sets X and Y and shows four unlabeled candidate diagrams. "
        "Compare the two membership rows and choose the diagram with the same exact set relation."
    )
}

_REL_TO_DIAGRAM = {
    "all_A_are_B": "subset_A_in_B",
    "all_B_are_A": "subset_B_in_A",
    "some_A_some_B": "overlap",
    "no_A_are_B": "disjoint",
}
_DIAGRAM_KINDS = tuple(_REL_TO_DIAGRAM.values())
_BG = (255, 255, 255)
_PANEL = (255, 255, 255)
_SOURCE_FILL = (255, 255, 255)
_PANEL_EDGE = (150, 156, 168)
_TEXT = (45, 48, 53)
_SUBTEXT = (104, 108, 116)
_X_OUTLINE = (63, 111, 212)
_Y_OUTLINE = (224, 120, 60)
_SHARED_MARK = (15, 128, 140)
_EXCLUSIVE_MARK = (232, 126, 48)
_CORRECT = (48, 158, 89)
# Token identity is carried by the printed number alone; every token uses
# one family token color so number and color never double-encode identity.
_TOKEN_FILL = (63, 111, 212)

_SOURCE_PANEL = (18, 32, 494, 238)
_X_ROW_BOX = (65, 72, 482, 130)
_Y_ROW_BOX = (65, 151, 482, 209)
_OPTION_PANELS = ((18, 254, 250, 367), (262, 254, 494, 367), (18, 375, 250, 488), (262, 375, 494, 488))


def _classify_relation(x_elements: Sequence[int], y_elements: Sequence[int]) -> str:
    x_set = {int(value) for value in x_elements}
    y_set = {int(value) for value in y_elements}
    if not x_set or not y_set:
        raise ValueError("both sets must be non-empty")
    if x_set < y_set:
        return "all_A_are_B"
    if y_set < x_set:
        return "all_B_are_A"
    if x_set.isdisjoint(y_set):
        return "no_A_are_B"
    if x_set & y_set and x_set - y_set and y_set - x_set:
        return "some_A_some_B"
    raise ValueError("equal or otherwise ambiguous sets are not admitted")


def _scaled_box(box: Sequence[float]) -> Tuple[int, int, int, int]:
    return tuple(int(round(float(value) * _SCALE)) for value in box)  # type: ignore[return-value]


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    *,
    size: int,
    fill: Tuple[int, int, int],
) -> None:
    draw.text(
        (xy[0] * _SCALE, xy[1] * _SCALE),
        text,
        font=load_font(size * _SCALE),
        fill=fill,
        anchor="mm",
    )


def _row_token_centers(box: Sequence[float], elements: Sequence[int]) -> Dict[int, Tuple[float, float]]:
    values = sorted(int(value) for value in elements)
    left, top, right, bottom = (float(value) for value in box)
    if not values:
        return {}
    usable_left = left + 30.0
    usable_right = right - 30.0
    if len(values) == 1:
        xs = [(usable_left + usable_right) / 2.0]
    else:
        step = (usable_right - usable_left) / float(len(values) - 1)
        xs = [usable_left + step * index for index in range(len(values))]
    center_y = (top + bottom) / 2.0
    return {value: (xs[index], center_y) for index, value in enumerate(values)}


def _draw_membership_row(
    image: Image.Image,
    box: Sequence[float],
    elements: Sequence[int],
    *,
    shared_elements: set[int],
    mark_shared: bool,
    mark_exclusive: bool,
) -> Dict[int, Tuple[float, float]]:
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        _scaled_box(box),
        radius=11 * _SCALE,
        fill=_PANEL,
        outline=(205, 208, 213),
        width=2 * _SCALE,
    )
    centers = _row_token_centers(box, elements)
    for token, (cx, cy) in centers.items():
        center = (cx * _SCALE, cy * _SCALE)
        radius = 18 * _SCALE
        fill = _TOKEN_FILL
        draw.ellipse(
            (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
            fill=fill,
            outline=(38, 40, 44),
            width=2 * _SCALE,
        )
        _draw_centered(draw, (cx, cy - 1), str(token), size=13, fill=(255, 255, 255))
        mark_color: Tuple[int, int, int] | None = None
        if mark_shared and token in shared_elements:
            mark_color = _SHARED_MARK
        elif mark_exclusive and token not in shared_elements:
            mark_color = _EXCLUSIVE_MARK
        if mark_color is not None:
            outer = radius + 5 * _SCALE
            draw.ellipse(
                (center[0] - outer, center[1] - outer, center[0] + outer, center[1] + outer),
                outline=mark_color,
                width=3 * _SCALE,
            )
    return centers


def _draw_set_diagram(image: Image.Image, box: Sequence[float], kind: str) -> None:
    left, top, right, bottom = (float(value) for value in box)
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0 + 1.0
    width = right - left
    height = bottom - top

    circles: List[Tuple[str, Tuple[float, float, float, float], Tuple[int, int, int]]] = []
    if kind == "subset_A_in_B":
        big_r = min(width * 0.28, height * 0.43)
        small_r = big_r * 0.48
        circles = [
            ("Y", (cx - big_r, cy - big_r, cx + big_r, cy + big_r), _Y_OUTLINE),
            ("X", (cx - small_r - 7, cy - small_r + 2, cx + small_r - 7, cy + small_r + 2), _X_OUTLINE),
        ]
    elif kind == "subset_B_in_A":
        big_r = min(width * 0.28, height * 0.43)
        small_r = big_r * 0.48
        circles = [
            ("X", (cx - big_r, cy - big_r, cx + big_r, cy + big_r), _X_OUTLINE),
            ("Y", (cx - small_r + 7, cy - small_r + 2, cx + small_r + 7, cy + small_r + 2), _Y_OUTLINE),
        ]
    elif kind == "overlap":
        radius = min(width * 0.23, height * 0.40)
        offset = radius * 0.62
        circles = [
            ("X", (cx - offset - radius, cy - radius, cx - offset + radius, cy + radius), _X_OUTLINE),
            ("Y", (cx + offset - radius, cy - radius, cx + offset + radius, cy + radius), _Y_OUTLINE),
        ]
    elif kind == "disjoint":
        radius = min(width * 0.20, height * 0.36)
        offset = radius * 1.25
        circles = [
            ("X", (cx - offset - radius, cy - radius, cx - offset + radius, cy + radius), _X_OUTLINE),
            ("Y", (cx + offset - radius, cy - radius, cx + offset + radius, cy + radius), _Y_OUTLINE),
        ]
    else:
        raise ValueError(f"unknown set diagram: {kind}")

    outline_draw = ImageDraw.Draw(image)
    for label, raw_bbox, color in circles:
        bbox = _scaled_box(raw_bbox)
        outline_draw.ellipse(bbox, outline=color, width=3 * _SCALE)
        label_x = (raw_bbox[0] + raw_bbox[2]) / 2.0
        label_y = (raw_bbox[1] + raw_bbox[3]) / 2.0
        if kind == "overlap":
            label_x += -0.38 * (raw_bbox[2] - raw_bbox[0]) if label == "X" else 0.38 * (raw_bbox[2] - raw_bbox[0])
        elif kind in {"subset_A_in_B", "subset_B_in_A"} and (raw_bbox[2] - raw_bbox[0]) > height * 0.6:
            label_x = raw_bbox[2] - 13
            label_y = raw_bbox[1] + 13
        _draw_centered(outline_draw, (label_x, label_y), label, size=13, fill=color)


class SetInclusionDiagramPattern(RenderedTaskGenerator):
    TASK_NAME = "set_inclusion_diagram"
    TASK_FAMILY = "in_context_learning"

    def _generate_membership(self, relation: str) -> Tuple[List[int], List[int]]:
        pool = list(range(1, 9))
        self.rng.shuffle(pool)
        if relation == "all_A_are_B":
            y_count = self.rng.randint(4, 6)
            x_count = self.rng.randint(2, y_count - 1)
            y_values = pool[:y_count]
            x_values = self.rng.sample(y_values, k=x_count)
        elif relation == "all_B_are_A":
            x_count = self.rng.randint(4, 6)
            y_count = self.rng.randint(2, x_count - 1)
            x_values = pool[:x_count]
            y_values = self.rng.sample(x_values, k=y_count)
        elif relation == "some_A_some_B":
            shared_count = self.rng.randint(1, 2)
            x_only_count = self.rng.randint(1, 3)
            y_only_count = self.rng.randint(1, 3)
            shared = pool[:shared_count]
            x_only = pool[shared_count : shared_count + x_only_count]
            y_only = pool[shared_count + x_only_count : shared_count + x_only_count + y_only_count]
            x_values = shared + x_only
            y_values = shared + y_only
        elif relation == "no_A_are_B":
            x_count = self.rng.randint(2, 4)
            y_count = self.rng.randint(2, min(4, 8 - x_count))
            x_values = pool[:x_count]
            y_values = pool[x_count : x_count + y_count]
        else:
            raise ValueError(relation)

        x_values = sorted(int(value) for value in x_values)
        y_values = sorted(int(value) for value in y_values)
        if _classify_relation(x_values, y_values) != relation:
            raise RuntimeError("generated membership lists do not match the requested relation")
        return x_values, y_values

    def _generate_task(self) -> Dict[str, Any]:
        relation = self.rng.choice(list(_REL_TO_DIAGRAM))
        x_elements, y_elements = self._generate_membership(relation)
        candidate_kinds = list(_DIAGRAM_KINDS)
        self.rng.shuffle(candidate_kinds)
        correct_diagram = _REL_TO_DIAGRAM[relation]
        answer_index = candidate_kinds.index(correct_diagram)
        return {
            "relation": relation,
            "x_elements": x_elements,
            "y_elements": y_elements,
            "shared_elements": sorted(set(x_elements) & set(y_elements)),
            "x_only": sorted(set(x_elements) - set(y_elements)),
            "y_only": sorted(set(y_elements) - set(x_elements)),
            "candidate_kinds": candidate_kinds,
            "correct_diagram": correct_diagram,
            "answer_index": answer_index,
        }

    @staticmethod
    def _state_trace() -> List[Dict[str, bool]]:
        return [
            {"mark_shared": False, "mark_exclusive": False, "solved": False},
            {"mark_shared": True, "mark_exclusive": False, "solved": False},
            {"mark_shared": True, "mark_exclusive": True, "solved": False},
            {"mark_shared": True, "mark_exclusive": True, "solved": True},
        ]

    def _render_state(
        self,
        task: Dict[str, Any],
        *,
        mark_shared: bool = False,
        mark_exclusive: bool = False,
        solved: bool = False,
    ) -> Image.Image:
        image = Image.new("RGB", (512 * _SCALE, 512 * _SCALE), _BG)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            _scaled_box(_SOURCE_PANEL),
            radius=12 * _SCALE,
            fill=_SOURCE_FILL,
            outline=_PANEL_EDGE,
            width=2 * _SCALE,
        )
        _draw_centered(draw, (41, 101), "X", size=18, fill=_TEXT)
        _draw_centered(draw, (41, 180), "Y", size=18, fill=_TEXT)
        shared = set(int(value) for value in task["shared_elements"])
        x_centers = _draw_membership_row(
            image,
            _X_ROW_BOX,
            task["x_elements"],
            shared_elements=shared,
            mark_shared=mark_shared,
            mark_exclusive=mark_exclusive,
        )
        y_centers = _draw_membership_row(
            image,
            _Y_ROW_BOX,
            task["y_elements"],
            shared_elements=shared,
            mark_shared=mark_shared,
            mark_exclusive=mark_exclusive,
        )
        if mark_shared:
            draw.rounded_rectangle(
                _scaled_box((225, 132, 287, 150)),
                radius=8 * _SCALE,
                fill=(231, 249, 250),
                outline=_SHARED_MARK,
                width=2 * _SCALE,
            )
            _draw_centered(draw, (256, 141), f"∩ {len(shared)}", size=11, fill=_TEXT)

        for index, panel in enumerate(_OPTION_PANELS):
            border = _CORRECT if solved and index == int(task["answer_index"]) else _PANEL_EDGE
            width = 4 if solved and index == int(task["answer_index"]) else 2
            draw.rounded_rectangle(
                _scaled_box(panel),
                radius=10 * _SCALE,
                fill=_PANEL,
                outline=border,
                width=width * _SCALE,
            )
            _draw_centered(draw, (panel[0] + 15, panel[1] + 14), LETTER_OPTIONS[index], size=13, fill=_TEXT)
            _draw_set_diagram(
                image,
                (panel[0] + 34, panel[1] + 8, panel[2] - 8, panel[3] - 8),
                str(task["candidate_kinds"][index]),
            )
            if solved and index == int(task["answer_index"]):
                _draw_centered(draw, (panel[2] - 15, panel[1] + 14), "✓", size=13, fill=_CORRECT)

        rendered = downsample(image, _SCALE)
        task.setdefault(
            "token_centers",
            {
                "X": {str(token): [center[0], center[1]] for token, center in x_centers.items()},
                "Y": {str(token): [center[0], center[1]] for token, center in y_centers.items()},
            },
        )
        return rendered

    def _render_puzzle_image(self, task: Dict[str, Any]) -> Image.Image:
        return self._render_state(task)

    def _render_solution_image(self, task: Dict[str, Any]) -> Image.Image:
        return self._render_state(task, **self._state_trace()[-1])

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        holds = (5, 4, 4, 10)
        frames: List[np.ndarray] = []
        states = self._state_trace()
        previous = np.asarray(self._render_state(task, **states[0]), dtype=np.uint8)
        frames.extend(previous.copy() for _ in range(holds[0]))
        for state, hold in zip(states[1:], holds[1:]):
            stable = np.asarray(self._render_state(task, **state), dtype=np.uint8)
            # Each membership layer is one semantic global commit. Fade only
            # the newly added rings/badge/border; never scan the canvas in an
            # arbitrary row-major or left-to-right order.
            frames.extend(changed_region_opacity_frames(previous, stable))
            frames.extend(stable.copy() for _ in range(hold))
            previous = stable
        frames = [np.asarray(frame, dtype=np.uint8) for frame in pad_frames_to_wan_compatible_length(frames)]
        if len(frames) != 29:
            raise AssertionError("set_inclusion_diagram video must contain 29 = 4N+1 frames")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single card shows two membership rows labeled X and Y and four candidate circle diagrams A-D. "
                "The candidates contain only set labels X and Y, not textual relation names."
            ),
            symbol_legend=(
                "Each numbered token is one element; the same number in both rows denotes the same element.",
                "A token in row X belongs to set X and a token in row Y belongs to set Y.",
                "In intermediate evidence, cyan rings mark shared members, orange rings mark exclusive members, and a compact badge shows the intersection count.",
            ),
            rules=(
                "X is a proper subset of Y when every X token also appears in Y and Y has at least one additional token; choose the diagram with X fully inside Y.",
                "Y is a proper subset of X when every Y token also appears in X and X has at least one additional token; choose the diagram with Y fully inside X.",
                "The sets partially overlap when they share at least one token and each row also has at least one exclusive token; choose the two partially overlapping circles.",
                "The sets are disjoint when no token appears in both rows; choose the two separated circles.",
                "Exactly one of these four relations holds and each candidate depicts a different relation.",
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
            render_profile="native_membership_to_set_relation_trace_v2",
        )

    def _prompts(self) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Compare the two visual membership rows and choose the candidate diagram with the same exact relation between sets X and Y.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid set-inclusion prompt contract: {'; '.join(issues)}")
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

        candidate_values = {
            LETTER_OPTIONS[index]: str(kind)
            for index, kind in enumerate(task["candidate_kinds"])
        }
        task_meta = {
            "relation": task["relation"],
            "correct_diagram": task["correct_diagram"],
            "correct_value": task["correct_diagram"],
            "distractor_values": [
                kind for kind in _DIAGRAM_KINDS if kind != task["correct_diagram"]
            ],
            "candidate_values_by_label": candidate_values,
            "x_elements": task["x_elements"],
            "y_elements": task["y_elements"],
            "shared_elements": task["shared_elements"],
            "x_only": task["x_only"],
            "y_only": task["y_only"],
            "candidate_kinds": task["candidate_kinds"],
            "answer_index": int(task["answer_index"]),
            "state_trace": self._state_trace(),
            "render_geometry": {
                "source_panel": list(_SOURCE_PANEL),
                "x_row_box": list(_X_ROW_BOX),
                "y_row_box": list(_Y_ROW_BOX),
                "option_panels": [list(panel) for panel in _OPTION_PANELS],
                "token_centers": task.get("token_centers", {}),
                "shared_mark_rgb": list(_SHARED_MARK),
                "exclusive_mark_rgb": list(_EXCLUSIVE_MARK),
                "correct_rgb": list(_CORRECT),
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
                "structured_answer": {
                    "correct_option": answer_raw,
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
                        "semantic_source": "visible_x_and_y_membership_lists_plus_four_relation_diagrams",
                        "component_scores": ["parse_score", "source_replay_score", "relation_score", "candidate_uniqueness_score", "visual_parseability_score", "state_order_score", "preservation_score", "final_state_score", "media_contract_score"],
                        "hard_gates": ["strict_parse", "independent_membership_set_replay", "complete_unique_relation_candidates", "semantic_visual_trace", "fixed_content_preservation", "media_contract"],
                    },
                    "task_meta": task_meta,
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
        return "Choose the candidate diagram matching the visual set membership relation."

    def _answer_raw(self, task: Dict[str, Any]) -> str:
        return LETTER_OPTIONS[int(task["answer_index"])]

    def _structured_answer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return dict(task)

    def _visual_input_description(self, task: Dict[str, Any]) -> str:
        return self._contract().input_description

    def _visual_output_description(self, task: Dict[str, Any]) -> str:
        return self._contract().modality_outputs["solution_image"]


from data.in_context_learning.core.polish import polished_icl_generator

SetInclusionDiagramPattern = polished_icl_generator(SetInclusionDiagramPattern)

__all__ = [
    "SetInclusionDiagramPattern",
    "_DIAGRAM_KINDS",
    "_REL_TO_DIAGRAM",
    "_classify_relation",
]
