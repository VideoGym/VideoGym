"""Shared per-position card scene for ICL sequence-completion tasks.

Every sequence task in this family shows one card per sequence position:
cards 1..N-1 hold the visible terms and the final card holds only a question
mark in the supplied input.  The deliverable is that same scene with the final
card's question mark replaced by the correct content — there are no candidate
plates and no option labels anywhere in the format.

This module owns the card geometry (single source for renderers and pixel
verifiers), the supersampled scene renderer, and the reference-video author
(input hold -> complete-footprint local fade of the final card content ->
final hold).  Task packages contribute only their semantics and one content
painter per card.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.grid_coordinates import (
    GRID_POSITION_LIST_ANSWER_EXAMPLE,
    GRID_POSITION_LIST_ANSWER_SCHEMA,
)
from data.common.pipeline.media import build_complete_footprint_fade_frames, load_font
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import encode_rgb_frames_to_webm

RGB = Tuple[int, int, int]
Box = Tuple[float, float, float, float]

CANVAS = (512, 512)
BACKGROUND: RGB = (255, 255, 255)
CARD_FILL: RGB = (251, 252, 254)
CARD_EDGE: RGB = (150, 156, 168)
BADGE_TEXT: RGB = (96, 102, 112)
QUESTION_TEXT: RGB = (88, 92, 100)

# Arrangements.  All coordinates are authored in unscaled 512-square space and
# consumed identically by the renderer and by pixel verifiers.
_ROW4_CARD_W, _ROW4_CARD_H, _ROW4_GAP = 116.0, 196.0, 10.0
_ROW6_CARD_W, _ROW6_CARD_H, _ROW6_GAP = 78.0, 148.0, 8.0
_GRID4_CARD_W, _GRID4_CARD_H, _GRID4_GAP = 228.0, 212.0, 14.0

_BADGE_STRIP = 26.0  # vertical space at the card top reserved for the 1-based position number


def card_bboxes(arrangement: str, total: int) -> list[Box]:
    """Card outlines for one arrangement, in position order (1-based positions)."""

    from data.common.prompt_paraphrase import active_profile
    if active_profile("colour_gems") and total == 6:
        centres = [(112,102),(360,102),(414,256),(360,410),(112,410),(58,256)]
        return [(x-48,y-62,x+48,y+62) for x,y in centres]
    if active_profile("height_solids") and total == 4:
        return [(8+126*i,158,126+126*i,354) for i in range(4)]
    if arrangement == "row4":
        if total != 4:
            raise ValueError("row4 arrangement holds exactly four cards")
        width = 4 * _ROW4_CARD_W + 3 * _ROW4_GAP
        x0 = (512.0 - width) / 2.0
        y0 = (512.0 - _ROW4_CARD_H) / 2.0
        return [
            (
                x0 + index * (_ROW4_CARD_W + _ROW4_GAP),
                y0,
                x0 + index * (_ROW4_CARD_W + _ROW4_GAP) + _ROW4_CARD_W,
                y0 + _ROW4_CARD_H,
            )
            for index in range(4)
        ]
    if arrangement == "row6":
        if total != 6:
            raise ValueError("row6 arrangement holds exactly six cards")
        width = 6 * _ROW6_CARD_W + 5 * _ROW6_GAP
        x0 = (512.0 - width) / 2.0
        y0 = (512.0 - _ROW6_CARD_H) / 2.0
        return [
            (
                x0 + index * (_ROW6_CARD_W + _ROW6_GAP),
                y0,
                x0 + index * (_ROW6_CARD_W + _ROW6_GAP) + _ROW6_CARD_W,
                y0 + _ROW6_CARD_H,
            )
            for index in range(6)
        ]
    if arrangement == "grid4":
        if total != 4:
            raise ValueError("grid4 arrangement holds exactly four cards")
        width = 2 * _GRID4_CARD_W + _GRID4_GAP
        height = 2 * _GRID4_CARD_H + _GRID4_GAP
        x0 = (512.0 - width) / 2.0
        y0 = (512.0 - height) / 2.0
        return [
            (
                x0 + (index % 2) * (_GRID4_CARD_W + _GRID4_GAP),
                y0 + (index // 2) * (_GRID4_CARD_H + _GRID4_GAP),
                x0 + (index % 2) * (_GRID4_CARD_W + _GRID4_GAP) + _GRID4_CARD_W,
                y0 + (index // 2) * (_GRID4_CARD_H + _GRID4_GAP) + _GRID4_CARD_H,
            )
            for index in range(4)
        ]
    raise ValueError(f"unknown sequence-card arrangement: {arrangement!r}")


def content_bbox(card_box: Sequence[float]) -> Box:
    """Interior content region of one card, below its position badge."""

    left, top, right, bottom = (float(value) for value in card_box)
    return (left + 7.0, top + _BADGE_STRIP, right - 7.0, bottom - 8.0)


def final_card_bbox(arrangement: str, total: int) -> Box:
    return card_bboxes(arrangement, total)[-1]


def final_content_bbox(arrangement: str, total: int) -> Box:
    return content_bbox(final_card_bbox(arrangement, total))


Painter = Callable[[ScaledDraw, Box, Mapping[str, object], int], None]
PlaceholderPainter = Callable[[ScaledDraw, Box], None]


def render_sequence_scene(
    *,
    arrangement: str,
    items: Sequence[Mapping[str, object]],
    answer_item: Mapping[str, object] | None,
    painter: Painter,
    solved: bool,
    placeholder_painter: PlaceholderPainter | None = None,
) -> Image.Image:
    """Render the per-position card scene.

    ``items`` are the visible terms for positions 1..N-1.  When ``solved`` is
    false the final card holds only a centered question mark (drawn over the
    task's static card furniture when ``placeholder_painter`` supplies one);
    when true, ``answer_item`` is painted into the final card by the same task
    painter that painted the visible terms.
    """

    total = len(items) + 1
    boxes = card_bboxes(arrangement, total)
    factor = DOCTRINE_SUPERSAMPLE
    canvas, sdraw = new_scaled_canvas(512, 512, factor, background=BACKGROUND)
    badge_font = load_font(15 * factor)
    question_font = load_font(48 * factor)
    from data.common.prompt_paraphrase import active_profile
    styled = active_profile("colour_gems") or active_profile("height_solids")
    if styled:
        sdraw.rectangle((0,0,512,512),fill=(225,237,217))
        centres=[((b[0]+b[2])/2,(b[1]+b[3])/2) for b in boxes]
        for first,second in zip(centres,centres[1:]):
            sdraw.line((*first,*second),fill=(123,147,114),width=6)
    for index, box in enumerate(boxes):
        if styled:
            sdraw.rounded_rectangle(box,radius=28,fill=(255,247,221),outline=(82,111,94),width=3)
        else:
            sdraw.rounded_rectangle(box, radius=9, fill=CARD_FILL, outline=CARD_EDGE, width=2)
        cx = (box[0] + box[2]) / 2.0
        sdraw.text((cx, box[1] + _BADGE_STRIP / 2.0 - 1.0), str(index + 1), fill=BADGE_TEXT, font=badge_font, anchor="mm")
        content = content_bbox(box)
        if index < len(items):
            painter(sdraw, content, items[index], index + 1)
        elif solved:
            if answer_item is None:
                raise ValueError("solved sequence scene requires the answer item")
            painter(sdraw, content, answer_item, index + 1)
        else:
            if placeholder_painter is not None:
                placeholder_painter(sdraw, content)
            sdraw.text(
                ((content[0] + content[2]) / 2.0, (content[1] + content[3]) / 2.0),
                "?",
                fill=QUESTION_TEXT,
                font=question_font,
                anchor="mm",
            )
    return finish_scaled_canvas(canvas, CANVAS)


def _decode_video(path) -> tuple[list[np.ndarray], float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, fps


def author_final_card_fade_video(problem: Image.Image, terminal: Image.Image, path) -> int:
    """Author input hold -> complete-footprint fade of the final card -> hold.

    Every changed pixel fades together at its complete terminal footprint, so
    no scan direction or solver order is encoded.  The encoded file is decoded
    back and hard-checked against the authored state sequence.
    """

    frames = build_complete_footprint_fade_frames(
        problem,
        terminal,
        hold_frames=5,
        fade_frames=5,
        transition_repeats=2,
        final_hold_frames=10,
    )
    while len(frames) % 4 != 1:
        frames.append(frames[-1].copy())
    path.parent.mkdir(parents=True, exist_ok=True)
    if not encode_rgb_frames_to_webm(frames, path, fps=16):
        raise RuntimeError(f"failed to encode sequence-card reference video: {path}")
    _verify_reference_fade_video(path, start=problem, final=terminal, expected_frames=frames)
    return len(frames)


def _verify_reference_fade_video(
    target,
    *,
    start: Image.Image,
    final: Image.Image,
    expected_frames: Sequence[np.ndarray],
) -> None:
    """Hard-check the authored reference after lossy video encoding."""

    frames, fps = _decode_video(target)
    if not frames or len(frames) % 4 != 1 or abs(float(fps) - 16.0) > 0.25:
        raise AssertionError("sequence-card reference video violates decode/fps/4N+1 contract")
    first = np.asarray(start.convert("RGB"), dtype=np.int16)
    last = np.asarray(final.convert("RGB"), dtype=np.int16)
    if frames[0].shape != first.shape or frames[-1].shape != last.shape:
        raise AssertionError("sequence-card reference video changed the canvas")
    if float(np.mean(np.abs(frames[0].astype(np.int16) - first))) > 5.0:
        raise AssertionError("sequence-card reference video does not start at the supplied input")
    if float(np.mean(np.abs(frames[-1].astype(np.int16) - last))) > 5.0:
        raise AssertionError("sequence-card reference video does not end at the completed scene")
    bank: list[np.ndarray] = []
    for expected in expected_frames:
        expected_array = np.asarray(expected, dtype=np.uint8)
        if not bank or not np.array_equal(expected_array, bank[-1]):
            bank.append(expected_array)
    recognized_states: list[int | None] = []
    for frame in frames:
        errors = [
            float(np.mean(np.abs(frame.astype(np.float32) - state.astype(np.float32)))) / 255.0
            for state in bank
        ]
        best = min(range(len(bank)), key=errors.__getitem__)
        recognized_states.append(best if errors[best] <= 0.045 else None)
    recognized = [value for value in recognized_states if value is not None]
    if len(recognized) / len(frames) < 0.9:
        raise AssertionError("sequence-card reference video leaves its authored local-fade state bank")
    transitions = [
        (left, right)
        for left, right in zip(recognized_states, recognized_states[1:])
        if left is not None and right is not None
    ]
    if any(right < left for left, right in transitions):
        raise AssertionError("sequence-card reference video reverses its element fade")
    interior_states = {value for value in recognized if 0 < value < len(bank) - 1}
    if len(interior_states) < 2:
        raise AssertionError("sequence-card reference video hard-commits instead of visibly fading")


_TI2V_PROCESS = (
    "Return one video of the solving process. Hold the supplied scene, fade the final card's completed "
    "content in as one whole at its complete final footprint, then hold the completed sequence."
)

_SEQUENCE_ANSWER_EXAMPLES = {
    "Answer: <height>.": "Answer: 99.",
    "Answer: <count>.": "Answer: 99.",
    "Answer: <color>.": "Answer: example-color.",
    "Answer: <operation>.": "Answer: example-operation.",
    GRID_POSITION_LIST_ANSWER_SCHEMA: GRID_POSITION_LIST_ANSWER_EXAMPLE,
}


def build_sequence_prompts(
    *,
    input_copy: str,
    rule_copy: str,
    answer_template: str,
    answer_meaning: str,
    terminal_copy: str,
    answer_example: str,
) -> dict[str, str]:
    """Assemble the three modality prompts for one sequence-completion sample.

    ``answer_template`` is the symbolic typed schema (for example
    ``Answer: <count>.``); ``answer_meaning`` finishes the sentence "…, " +
    meaning; ``terminal_copy`` describes the completed final card content that
    the terminal image and video must show.
    """

    from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

    continuity = build_ti2v_continuity_paragraph(
        timing="continuous",
        moving_nouns="the content that fills the final card",
        static_nouns="the completed cards, their position numbers, the background, and the camera",
        hold_noun="the completed sequence",
    )
    body = {
        "ti2v": (
            f"Input Image: {input_copy}\n\nRule: {rule_copy}\n\nRequired Output: {_TI2V_PROCESS} "
            f"The completed content is {terminal_copy}\n\n{continuity}"
        ),
    }
    return body


class SequenceCompletionGenerator(AbstractPuzzleGenerator):
    """Shared create-puzzle flow for per-position-card sequence tasks.

    Subclasses supply semantics and one content painter; this base renders the
    problem/terminal scenes, authors the reference fade video, and assembles
    the record with the family's CoT-enabled serialization fields.
    """

    task_name: str = ""
    task_family: str = "in_context_learning"
    arrangement: str = "row4"
    answer_kind: str = "typed_value"
    video_fps: int = 16
    generate_videos: bool = True

    def __init__(
        self,
        output_dir=None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        image_size=(512, 512),
        video_fps: int = 16,
        generate_videos: bool = True,
        video: bool | None = None,
        **_: object,
    ) -> None:
        import random
        import tempfile
        from pathlib import Path

        if isinstance(image_size, int):
            resolved = (image_size, image_size)
        else:
            resolved = (int(image_size[0]), int(image_size[1]))
        if resolved != (512, 512):
            raise ValueError(f"{self.task_name} requires a 512x512 canvas")
        if int(video_fps) != 16:
            raise ValueError(f"{self.task_name} requires 16fps video")
        self.rng = random.Random(seed if seed is not None else random_seed)
        self.generate_videos = bool(video if video is not None else generate_videos)
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / f"videogym_{self.task_name}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = resolved
        self._sample_index = 0

    # -- task hooks ---------------------------------------------------------

    def sample_semantics(self) -> dict:
        raise NotImplementedError

    def visible_items(self, semantics: Mapping[str, object]) -> list[dict]:
        raise NotImplementedError

    def answer_item(self, semantics: Mapping[str, object]) -> dict:
        raise NotImplementedError

    def paint_content(self, sdraw: ScaledDraw, box: Box, item: Mapping[str, object], position: int) -> None:
        raise NotImplementedError

    def answer_payload(self, semantics: Mapping[str, object]) -> str:
        raise NotImplementedError

    def answer_template(self) -> str:
        raise NotImplementedError

    def author_reasoning(self, semantics: Mapping[str, object]) -> str:
        raise NotImplementedError

    def prompt_copy(self, semantics: Mapping[str, object]) -> dict:
        """Return input_copy / rule_copy / answer_meaning / terminal_copy."""

        raise NotImplementedError

    def structured_semantics(self, semantics: Mapping[str, object]) -> dict:
        raise NotImplementedError

    def glyph_inventory(self, semantics: Mapping[str, object]) -> dict:
        raise NotImplementedError

    def verifier_profile(self) -> str:
        return "icl_sequence_completion_source_replay_v1"

    def paint_placeholder(self, sdraw: ScaledDraw, box: Box) -> None:
        """Static final-card furniture drawn under the question mark (default none)."""

        return None

    # -- shared flow --------------------------------------------------------

    def render_pair(self, semantics: Mapping[str, object]) -> tuple[Image.Image, Image.Image]:
        items = self.visible_items(semantics)
        answer = self.answer_item(semantics)
        problem = render_sequence_scene(
            arrangement=self.arrangement,
            items=items,
            answer_item=None,
            painter=self.paint_content,
            solved=False,
            placeholder_painter=self.paint_placeholder,
        )
        terminal = render_sequence_scene(
            arrangement=self.arrangement,
            items=items,
            answer_item=answer,
            painter=self.paint_content,
            solved=True,
        )
        return problem, terminal

    def create_puzzle(self) -> dict:
        semantics = self.sample_semantics()
        problem, terminal = self.render_pair(semantics)
        delta = np.max(
            np.abs(
                np.asarray(terminal, dtype=np.int16) - np.asarray(problem, dtype=np.int16)
            ),
            axis=2,
        )
        if int(np.count_nonzero(delta > 8)) < 24:
            raise AssertionError(f"{self.task_name} terminal scene is semantically empty")

        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        problem_path = sample_dir / f"{sample_id}__problem.png"
        terminal_path = sample_dir / f"{sample_id}__terminal_answer.png"
        problem.save(problem_path)
        terminal.save(terminal_path)

        video_path_value = None
        video_frames = None
        if self.generate_videos:
            video_path = sample_dir / f"{sample_id}__solution.webm"
            video_frames = author_final_card_fade_video(problem, terminal, video_path)
            video_path_value = video_path.as_posix()

        payload = self.answer_payload(semantics)
        reasoning = self.author_reasoning(semantics)
        if not reasoning or "{" in reasoning or "}" in reasoning:
            raise AssertionError(f"{self.task_name} authored reasoning is empty or templated")

        copy = self.prompt_copy(semantics)
        prompts = build_sequence_prompts(
            input_copy=str(copy["input_copy"]),
            rule_copy=str(copy["rule_copy"]),
            answer_template=self.answer_template(),
            answer_meaning=str(copy["answer_meaning"]),
            terminal_copy=str(copy["terminal_copy"]),
            answer_example=_SEQUENCE_ANSWER_EXAMPLES[self.answer_template()],
        )

        structured = {
            "schema_version": f"{self.task_name}.v4",
            "answer_schema": {
                "schema_version": f"{self.task_name}.v4",
                "kind": self.answer_kind,
                "format_template": self.answer_template(),
                "format_example": _SEQUENCE_ANSWER_EXAMPLES[self.answer_template()],
                "final_deliverable": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "intermediate_supervision_schema": {
                "ti2v": ["input_hold", "final_card_complete_footprint_fade", "final_hold"],
            },
            "sequence_arrangement": self.arrangement,
            "verifier_profile": self.verifier_profile(),
            **self.structured_semantics(semantics),
        }
        glyphs = self.glyph_inventory(semantics)
        structured["rendered_glyph_inventory"] = glyphs

        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "answer": payload,
            "structured_answer": structured,
            "rendered_glyph_inventory": glyphs,
            "ti2v_prompt": prompts["ti2v"],
            "image": problem_path.as_posix(),
            "reasoning_image": problem_path.as_posix(),
            "reasoning_images": [],
            "solution_image_path": terminal_path.as_posix(),
            "solution_video_path": video_path_value,
            "video_fps": 16 if video_path_value else None,
            "video_num_frames": video_frames,
        }


__all__ = [
    "BACKGROUND",
    "CANVAS",
    "SequenceCompletionGenerator",
    "author_final_card_fade_video",
    "build_sequence_prompts",
    "card_bboxes",
    "content_bbox",
    "final_card_bbox",
    "final_content_bbox",
    "render_sequence_scene",
]
