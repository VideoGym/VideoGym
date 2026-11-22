"""Count the markers in the strict two-set intersection.

The input is one native 512x512 card.  All dot centres are generated with a
clear margin from both circle boundaries, so membership never depends on
antialiasing or on whether a boundary stroke is treated as inside.  The
reasoning trace identifies the overlap and marks every qualifying dot in one
global evidence state before selecting the matching numeric candidate.  It
does not privilege an arbitrary scan or counting order.
"""

from __future__ import annotations

from data.common.base import RenderedTaskGenerator

import math
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from data.common.pipeline.media import LETTER_OPTIONS, downsample, load_font
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.in_context_learning.core.option_delivery import (
    OPTION_MODALITY_OUTPUTS,
    OPTION_PRESERVATION_CONSTRAINTS,
    OPTION_TRANSITION_CONSTRAINTS,
    render_selection_video,
    render_terminal_image,
)
from data.common.pipeline.media import (
    pad_image,
    save_visual_reasoning_video,
)


pattern_instructions = {
    "venn_intersection_count": (
        "A single card shows two overlapping sets A and B, clearly separated dot markers, "
        "and four numeric candidates. Count only the markers whose centres lie inside both sets."
    )
}


_SCALE = 4
_CANVAS_SIZE = (512, 512)
_DIAGRAM_PANEL = (18, 18, 494, 326)
_VENN_BOX = (52, 54, 460, 310)
_OPTION_BOXES = (
    (21, 348, 131, 486),
    (141, 348, 251, 486),
    (261, 348, 371, 486),
    (381, 348, 491, 486),
)
_MEMBERSHIP_MARGIN = 16.0
_MIN_DOT_SPACING = 24.0
_DOT_RADIUS = 7

_BACKGROUND = (255, 255, 255)
_PANEL = (255, 255, 255)
_PANEL_EDGE = (150, 156, 168)
_TEXT = (48, 46, 43)
_SUBTEXT = (105, 101, 95)
_DOT = (44, 48, 54)
_A_OUTLINE = (63, 111, 212)
_B_OUTLINE = (224, 120, 60)
_INTERSECTION = (222, 60, 60, 96)
_COUNTED = (222, 60, 60)
_CORRECT = (45, 153, 85)
_VERIFIER_PROFILE = "venn_intersection_membership_count_and_final_delivery_v3"
_VERIFIER_COMPONENTS = (
    "parse_score", "source_replay_score", "membership_score", "count_score",
    "candidate_uniqueness_score", "visual_parseability_score", "state_order_score",
    "preservation_score", "final_state_score", "media_contract_score",
)
def _scaled_box(box: Sequence[float]) -> Tuple[int, int, int, int]:
    return tuple(int(round(float(value) * _SCALE)) for value in box)  # type: ignore[return-value]


def _venn_geometry(box: Tuple[int, int, int, int] = _VENN_BOX) -> Dict[str, Any]:
    """Return the semantic geometry used by both placement and rendering."""

    width = box[2] - box[0]
    height = box[3] - box[1]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    radius = float(min(width / 3.25, height / 2.0 - 17.0))
    offset = radius * 0.50
    return {
        "cx": cx,
        "cy": cy,
        "radius": radius,
        "offset": offset,
        "a_center": (cx - offset, cy),
        "b_center": (cx + offset, cy),
    }


def _region_margin(
    point: Tuple[int, int],
    region: str,
    geometry: Dict[str, Any],
) -> float:
    """Signed distance to the nearest required membership boundary.

    Positive values mean the point is in the requested open region.  For an
    A-only point, for example, this is the smaller of its inside-A clearance
    and its outside-B clearance.
    """

    radius = float(geometry["radius"])
    distance_a = math.dist(point, geometry["a_center"])
    distance_b = math.dist(point, geometry["b_center"])
    if region == "a_only":
        return min(radius - distance_a, distance_b - radius)
    if region == "b_only":
        return min(radius - distance_b, distance_a - radius)
    if region == "ab_both":
        return min(radius - distance_a, radius - distance_b)
    raise ValueError(f"unsupported Venn region: {region}")


def _generate_dot_positions(
    box: Tuple[int, int, int, int],
    counts: Dict[str, int],
    rng: random.Random,
    *,
    membership_margin: float = _MEMBERSHIP_MARGIN,
    minimum_spacing: float = _MIN_DOT_SPACING,
) -> Dict[str, List[Tuple[int, int]]] | None:
    """Place dots in strict semantic regions with stable visual clearance."""

    geometry = _venn_geometry(box)
    radius = float(geometry["radius"])
    centers = (geometry["a_center"], geometry["b_center"])
    x_min = int(math.floor(min(center[0] for center in centers) - radius))
    x_max = int(math.ceil(max(center[0] for center in centers) + radius))
    y_min = int(math.floor(geometry["cy"] - radius))
    y_max = int(math.ceil(geometry["cy"] + radius))

    placed: Dict[str, List[Tuple[int, int]]] = {}
    occupied: List[Tuple[int, int]] = []
    # The overlap is the most constrained region, so allocate it first.
    for region in ("ab_both", "a_only", "b_only"):
        requested = int(counts[region])
        region_points: List[Tuple[int, int]] = []
        for _ in range(12000):
            if len(region_points) == requested:
                break
            point = (
                rng.randint(x_min + _DOT_RADIUS + 2, x_max - _DOT_RADIUS - 2),
                rng.randint(y_min + _DOT_RADIUS + 2, y_max - _DOT_RADIUS - 2),
            )
            if _region_margin(point, region, geometry) < membership_margin:
                continue
            if any(math.dist(point, other) < minimum_spacing for other in occupied + region_points):
                continue
            region_points.append(point)
        if len(region_points) != requested:
            return None
        region_points.sort(key=lambda point: (point[1], point[0]))
        placed[region] = region_points
        occupied.extend(region_points)
    return placed


def _validate_dot_positions(
    positions: Dict[str, List[Tuple[int, int]]],
    counts: Dict[str, int],
    *,
    box: Tuple[int, int, int, int] = _VENN_BOX,
    membership_margin: float = _MEMBERSHIP_MARGIN,
    minimum_spacing: float = _MIN_DOT_SPACING,
) -> None:
    geometry = _venn_geometry(box)
    all_points: List[Tuple[int, int]] = []
    for region in ("a_only", "b_only", "ab_both"):
        points = positions.get(region)
        if not isinstance(points, list) or len(points) != int(counts[region]):
            raise ValueError(f"incorrect number of dots in {region}")
        for point in points:
            if _region_margin(point, region, geometry) < membership_margin - 1e-6:
                raise ValueError(f"dot is too close to a Venn boundary in {region}")
            if any(math.dist(point, other) < minimum_spacing - 1e-6 for other in all_points):
                raise ValueError("dots are too close to each other")
            all_points.append(point)


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


def _circle_bbox(center: Tuple[float, float], radius: float) -> Tuple[int, int, int, int]:
    return _scaled_box((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius))


def _draw_intersection_lens(image: Image.Image, geometry: Dict[str, Any]) -> None:
    size = image.size
    mask_a = Image.new("L", size, 0)
    mask_b = Image.new("L", size, 0)
    ImageDraw.Draw(mask_a).ellipse(_circle_bbox(geometry["a_center"], geometry["radius"]), fill=255)
    ImageDraw.Draw(mask_b).ellipse(_circle_bbox(geometry["b_center"], geometry["radius"]), fill=255)
    lens_mask = ImageChops.multiply(mask_a, mask_b)
    overlay = Image.new("RGBA", size, _INTERSECTION)
    image.paste(overlay.convert("RGB"), (0, 0), lens_mask.point(lambda value: value * _INTERSECTION[3] // 255))


class VennIntersectionCountPattern(RenderedTaskGenerator):
    task_name = "venn_intersection_count"
    task_family = "in_context_learning"
    video_fps = 16
    generate_videos = True

    def __init__(
        self,
        output_dir: Path | str | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        image_size: Tuple[int, int] | int = _CANVAS_SIZE,
        video_fps: int = 16,
        generate_videos: bool = True,
        **_: Any,
    ) -> None:
        if isinstance(image_size, int):
            resolved_size = (image_size, image_size)
        else:
            resolved_size = (int(image_size[0]), int(image_size[1]))
        if resolved_size != _CANVAS_SIZE:
            raise ValueError("venn_intersection_count requires a 512x512 canvas")
        if int(video_fps) != 16:
            raise ValueError("venn_intersection_count requires 16fps video")
        resolved_seed = seed if seed is not None else random_seed
        self.rng = random.Random(resolved_seed)
        self.image_size = resolved_size
        self.video_fps = 16
        self.generate_videos = bool(generate_videos)
        self.output_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_venn_intersection_count"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    def _generate_task(self) -> Dict[str, Any]:
        for _ in range(80):
            # Balance the answer over the smallest/middle/largest rank among
            # the three disjoint Venn regions.  The union remains the fourth,
            # always-largest distractor.
            region_values = sorted(self.rng.sample(range(2, 8), 3))
            intersection_rank = self.rng.randrange(3)
            ab_both = region_values[intersection_rank]
            remaining = [value for value in region_values if value != ab_both]
            self.rng.shuffle(remaining)
            a_only, b_only = remaining
            counts = {
                "a_only": int(a_only),
                "b_only": int(b_only),
                "ab_both": int(ab_both),
                "a_union_b": int(a_only + b_only + ab_both),
            }
            placement_seed = self.rng.randrange(1 << 31)
            positions = _generate_dot_positions(_VENN_BOX, counts, random.Random(placement_seed))
            if positions is None:
                continue
            _validate_dot_positions(positions, counts)

            source_values = [counts["ab_both"], counts["a_only"], counts["b_only"], counts["a_union_b"]]
            order = list(range(4))
            self.rng.shuffle(order)
            candidates = [source_values[index] for index in order]
            answer_index = order.index(0)
            return {
                "counts": counts,
                "dot_positions": positions,
                "placement_seed": placement_seed,
                "intersection_rank": intersection_rank,
                "candidates": candidates,
                "answer_index": answer_index,
            }
        raise RuntimeError("venn_intersection_count: failed to generate a strict instance")

    @staticmethod
    def _state_trace(task: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"phase": "input", "all_intersection_dots_marked": False, "selected": False},
            {"phase": "global", "all_intersection_dots_marked": True, "selected": False},
            {"phase": "final", "all_intersection_dots_marked": False, "selected": True},
        ]

    def _render_state(
        self,
        task: Dict[str, Any],
        *,
        phase: str,
        selected: bool,
    ) -> Image.Image:
        if phase not in {"input", "global", "final"}:
            raise ValueError(f"unsupported Venn trace phase: {phase}")
        image = Image.new("RGB", (_CANVAS_SIZE[0] * _SCALE, _CANVAS_SIZE[1] * _SCALE), _BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            _scaled_box(_DIAGRAM_PANEL),
            radius=14 * _SCALE,
            fill=_PANEL,
            outline=_PANEL_EDGE,
            width=2 * _SCALE,
        )

        geometry = _venn_geometry(_VENN_BOX)
        a_bbox = _circle_bbox(geometry["a_center"], geometry["radius"])
        b_bbox = _circle_bbox(geometry["b_center"], geometry["radius"])
        if phase == "global":
            _draw_intersection_lens(image, geometry)
        draw = ImageDraw.Draw(image)
        draw.ellipse(a_bbox, outline=_A_OUTLINE, width=4 * _SCALE)
        draw.ellipse(b_bbox, outline=_B_OUTLINE, width=4 * _SCALE)
        _draw_centered(draw, (geometry["a_center"][0] - 70, 40), "A", size=17, fill=_A_OUTLINE)
        _draw_centered(draw, (geometry["b_center"][0] + 70, 40), "B", size=17, fill=_B_OUTLINE)

        for region in ("a_only", "b_only", "ab_both"):
            for point in task["dot_positions"][region]:
                cx, cy = point[0] * _SCALE, point[1] * _SCALE
                radius = _DOT_RADIUS * _SCALE
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=_DOT,
                    outline=(255, 255, 255),
                    width=2 * _SCALE,
                )

        if phase == "global":
            # All exchangeable intersection markers commit together.  The
            # overlay intentionally carries no numbers or spatial order.
            for point in task["dot_positions"]["ab_both"]:
                cx, cy = point[0] * _SCALE, point[1] * _SCALE
                outer = 11 * _SCALE
                draw.ellipse(
                    (cx - outer, cy - outer, cx + outer, cy + outer),
                    fill=_COUNTED,
                    outline=(255, 255, 255),
                    width=2 * _SCALE,
                )

        for index, box in enumerate(_OPTION_BOXES):
            is_selected = selected and index == int(task["answer_index"])
            draw.rounded_rectangle(
                _scaled_box(box),
                radius=12 * _SCALE,
                fill=_PANEL,
                outline=_CORRECT if is_selected else _PANEL_EDGE,
                width=(5 if is_selected else 2) * _SCALE,
            )
            _draw_centered(draw, (box[0] + 17, box[1] + 18), LETTER_OPTIONS[index], size=15, fill=_TEXT)
            _draw_centered(
                draw,
                ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0 + 8),
                str(task["candidates"][index]),
                size=38,
                fill=_TEXT,
            )
            if is_selected:
                _draw_centered(draw, (box[2] - 17, box[1] + 18), "✓", size=16, fill=_CORRECT)
        return downsample(image, _SCALE)

    def _render_puzzle_image(self, task: Dict[str, Any]) -> Image.Image:
        return self._render_state(task, phase="input", selected=False)

    def _render_solution_image(self, task: Dict[str, Any]) -> Image.Image:
        return self._render_state(task, phase="final", selected=True)

    def _render_video(
        self,
        task: Dict[str, Any],
        puzzle_path: Path,
        output_path: Path,
    ) -> Tuple[str | None, int | None]:
        if not self.generate_videos:
            return None, None
        render_selection_video(
            output_path,
            puzzle_path=puzzle_path,
            boxes=_OPTION_BOXES,
            selected_index=int(task["answer_index"]),
        )
        return output_path.as_posix(), 21

    @staticmethod
    def _contract() -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single card contains two partially overlapping translucent circles labelled A and B, "
                "black dot markers in their three disjoint regions, and four numeric candidate panels A-D."
            ),
            symbol_legend=(
                "A dot belongs to a set when its centre lies inside that set's circular boundary.",
                "The lens-shaped area covered by both circles is A intersection B; dots in either outer cap belong to only one set.",
                "Generated dot centres stay visibly away from every circle boundary, so no boundary convention or partial dot is ambiguous.",
            ),
            rules=(
                "Count each black marker exactly once if and only if its centre is inside both circle A and circle B.",
                "Do not count markers in A-only or B-only, and do not infer membership from circle colour opacity.",
                "The four candidate integers are distinct and exactly one equals the intersection count.",
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
            render_profile="native_strict_venn_intersection_count_trace_v2",
        )

    def _prompts(self) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {
            "question": "Count the dot markers in the intersection of sets A and B and choose the matching candidate.",
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid Venn-intersection prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        contract, prompts = self._prompts()
        answer_raw = LETTER_OPTIONS[int(task['answer_index'])]
        FOUR_CHOICE_CODEC.format(answer_raw)
        sample_id = f'{self.task_name}-{self._sample_index:05d}'
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f'{sample_id}__puzzle.png'
        solution_path = sample_dir / f'{sample_id}__solution.png'
        video_path = sample_dir / f'{sample_id}__solution.webm'
        puzzle_image = self._render_puzzle_image(task)
        solution_image = self._render_solution_image(task)
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)
        render_terminal_image(solution_path, puzzle_path=puzzle_path, boxes=_OPTION_BOXES, selected_index=int(task['answer_index']))
        solution_image = Image.open(solution_path).convert('RGB')
        solution_video_path, video_num_frames = self._render_video(task, puzzle_path, video_path)
        candidate_values_by_label = {LETTER_OPTIONS[index]: int(value) for index, value in enumerate(task['candidates'])}
        counts = {key: int(value) for key, value in task['counts'].items()}
        task_meta = {'counts': counts, 'correct_count': counts['ab_both'], 'distractor_counts': [counts['a_only'], counts['b_only'], counts['a_union_b']], 'correct_value': counts['ab_both'], 'distractor_values': [counts['a_only'], counts['b_only'], counts['a_union_b']], 'candidate_values_by_label': candidate_values_by_label, 'dot_positions': {region: [list(point) for point in task['dot_positions'][region]] for region in ('a_only', 'b_only', 'ab_both')}, 'intersection_rank_among_disjoint_regions': int(task['intersection_rank']), 'placement_seed': int(task['placement_seed']), 'multiple_solution_policy': 'unique_distinct_candidate_values', 'render_geometry': {'diagram_panel': list(_DIAGRAM_PANEL), 'venn_box': list(_VENN_BOX), 'option_boxes': [list(box) for box in _OPTION_BOXES], 'a_center': list(_venn_geometry()['a_center']), 'b_center': list(_venn_geometry()['b_center']), 'radius': _venn_geometry()['radius'], 'membership_margin': _MEMBERSHIP_MARGIN, 'minimum_dot_spacing': _MIN_DOT_SPACING, 'intersection_rgb': list(_INTERSECTION[:3]), 'global_marker_rgb': list(_COUNTED), 'correct_rgb': list(_CORRECT)}, 'state_trace': self._state_trace(task), 'answer_schema': {'schema_version': 'option_letter_v1', 'kind': 'option_letter', 'format_template': FOUR_CHOICE_CODEC.schema, 'format_example': FOUR_CHOICE_CODEC.example, 'parser_profile': 'strict_uppercase_option_letter_a_to_d', 'final_deliverable': {'ti2v': ['full_video']}, 'accepts_multiple_valid_answers': False, 'extra_text_allowed': False}, 'intermediate_supervision_schema': {'ti2v': ['input_hold', 'uniform_complete_green_border_fade', 'final_hold']}, 'verifier_contract': {'profile': _VERIFIER_PROFILE, 'semantic_source': 'visible_circle_geometry_and_dot_center_membership', 'component_scores': list(_VERIFIER_COMPONENTS), 'hard_gates': ['strict_parse', 'independent_membership_replay', 'candidate_uniqueness', 'semantic_final_image', 'whole_video_uniform_complete_border_fade', 'fixed_content_preservation', 'media_contract']}}
        sample: Dict[str, Any] = {'id': sample_id, 'task_name': self.task_name, 'task_family': self.task_family, 'task_contract': contract.to_dict(), 'question': prompts['question'], 'answer': answer_raw, 'correct_option': answer_raw, 'structured_answer': {'correct_option': answer_raw, 'task_meta': task_meta}, 'ti2v_prompt': prompts['ti2v_prompt'],  'image': puzzle_path.as_posix(), 'reasoning_image': puzzle_path.as_posix(), 'solution_image_path': solution_path.as_posix(), 'solution_video_path': solution_video_path, 'video_fps': self.video_fps if solution_video_path else None, 'video_num_frames': video_num_frames}
        return (sample, puzzle_image, solution_image)

    def create_puzzle(self) -> Dict[str, Any]:
        return self.make_sample()[0]


from data.in_context_learning.core.polish import polished_icl_generator

VennIntersectionCountPattern = polished_icl_generator(VennIntersectionCountPattern)

__all__ = [
    "VennIntersectionCountPattern",
    "pad_image",
    "save_visual_reasoning_video",
    "_MEMBERSHIP_MARGIN",
    "_MIN_DOT_SPACING",
    "_VENN_BOX",
    "_generate_dot_positions",
    "_region_margin",
    "_validate_dot_positions",
    "_venn_geometry",
    "_VERIFIER_COMPONENTS",
    "_VERIFIER_PROFILE",
    "np",
]
