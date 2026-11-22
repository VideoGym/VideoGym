"""Neutral one-loop token distribution with a direct final-state answer."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import CANVAS_SIZE, DEFAULT_FPS
from data.common.pipeline.discrete_dynamics import encode_video
from data.common.pipeline.board_records import BoardChoiceRecord, write_board_choice_record
from data.common.pipeline.board_rendering import load_font
from data.common.scalar_lists import (
    format_nonnegative_integer_list_answer,
    format_nonnegative_integer_list_payload,
)
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the changing slot integers, the source ring, and the gold destination outlines",
    static_nouns="the seven slots, their P1-P6 and B labels, the arrows, the panel, background, and camera",
    hold_noun="the final seven counts with no highlight left",
)


_BG = (247, 249, 252)
_PANEL = (229, 235, 243)
_PIT = (255, 255, 255)
_BANK = (219, 232, 247)
_INK = (37, 51, 70)
_MUTED = (103, 117, 137)
_PATH = (111, 130, 154)
_GOLD = (224, 166, 48)
_COUNT = (38, 101, 175)

_INPUT = (
    "The single input image shows a row of six circular pits labeled P1-P6 and one rounded bank labeled B. The blue "
    "integer in each slot is its token count, the thick gold ring marks the source pit, and the thin arrows show the "
    "loop P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> B -> P1."
)
_RULES = (
    "Take all tokens from the source pit, leaving it at 0, then hand them on one at a time along the arrows: the "
    "first token goes to the slot after the source, each next token to the following slot, and B passes on to P1. "
    "Stop when every token from the source is placed; no slot is skipped, and slots the loop never reaches keep "
    "their counts."
)
_TEXT_OUTPUT = (
    "Return the seven final counts in P1, P2, P3, P4, P5, P6, B order as nonnegative decimal integers. "
    "For example: `Answer: [98, 99, 100, 101, 102, 103, 104].` Use one space after every comma and no extra text."
)
_IMAGE_OUTPUT = (
    "Render the final board: each of the seven slots shows its final blue integer and the gold source ring is gone. "
    "Keep the slots, labels, arrows, panel, colors, positions, and camera unchanged; draw no tokens and no extra "
    "board."
)
_VIDEO_OUTPUT = (
    "First change the gold-ringed source integer to 0 and remove that ring at the same commit. Then, for each token "
    "in order, fade the complete gold outline of the receiving slot in at its full footprint, increase only that "
    "slot's blue integer by 1 at the commit, and fade the same outline back out — one slot at a time along the "
    "arrows, including B to P1 when the loop wraps. Finish with every highlight gone. Do not draw or move tokens, "
    "change two counts at once, or touch an unrelated slot."
)


def _prompts() -> Dict[str, str]:
    return {
        "ti2v_prompt": build_sectioned_prompt(
            _INPUT,
            _VIDEO_OUTPUT,
            mode="ti2v",
            rule_description=_RULES,
            continuity_paragraph=_TI2V_CONTINUITY,
        ),
    }


def slot_centres() -> List[Tuple[float, float]]:
    """The seven slot centres (six pits then the bank) of the fixed panel."""

    return [(58.0 + index * 65.0, 245.0) for index in range(6)] + [(450.0, 245.0)]


def slot_count_boxes() -> List[Tuple[int, int, int, int]]:
    """Crop boxes that contain each slot's complete count glyph.

    The crop excludes the optional source ring and destination outline, so a
    count decode stays independent of the process highlight while still
    hard-gating all seven slots.  The verifier consumes this function instead
    of duplicating pixel geometry.
    """

    # Two bold digits at the native 31px size exceed the former 39px crop.
    # Keep their complete contours inside the crop; the pit border still
    # touches its boundary and is excluded by the glyph descriptor.
    return [(int(cx) - 24, int(cy) - 22, int(cx) + 25, int(cy) + 23) for cx, cy in slot_centres()]


def distribute_counts(pits: Sequence[int], bank: int, source: int) -> Tuple[List[int], int, List[int]]:
    """Return final pits, final bank, and the ordered destination indices (B is 6)."""

    if len(pits) != 6 or source not in range(6):
        raise ValueError("Expected six P slots and a source index from 0 through 5.")
    state = [int(value) for value in pits] + [int(bank)]
    token_count = state[source]
    if token_count < 1 or any(value < 0 for value in state):
        raise ValueError("Counts must be non-negative and the source must be non-empty.")
    state[source] = 0
    position = source
    destinations: List[int] = []
    for _ in range(token_count):
        position = (position + 1) % 7
        state[position] += 1
        destinations.append(position)
    return state[:6], state[6], destinations


class MancalaDistributePredictGenerator(AbstractPuzzleGenerator[BoardChoiceRecord]):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        video_fps: int = DEFAULT_FPS,
        video: bool = False,
    ) -> None:
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.video_fps = int(video_fps)
        self.video = bool(video)
        self._counter = 0

    def record_to_dict(self, record: BoardChoiceRecord | Dict[str, Any]) -> Dict[str, Any]:
        return record.to_dict() if hasattr(record, "to_dict") else dict(record)

    @staticmethod
    def _centres() -> List[Tuple[float, float]]:
        return slot_centres()

    def _construct(self) -> Tuple[List[int], int, int, List[int], int, List[int]]:
        pits = [self.rng.randint(0, 7) for _ in range(6)]
        # Four through six tokens from a late source pit make B -> P1 wrap
        # load-bearing in every sample without creating a long, repetitive clip.
        source = self.rng.randint(3, 5)
        pits[source] = self.rng.randint(4, 6)
        bank = self.rng.randint(0, 8)
        final_pits, final_bank, destinations = distribute_counts(pits, bank, source)
        return pits, bank, source, final_pits, final_bank, destinations

    @staticmethod
    def _arrow(draw: ScaledDraw, start: Tuple[float, float], end: Tuple[float, float]) -> None:
        sx, sy = start
        ex, ey = end
        draw.line((sx, sy, ex, ey), fill=_PATH, width=3)
        dx, dy = ex - sx, ey - sy
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        tip = (ex, ey)
        left = (ex - 10 * ux + 5 * px, ey - 10 * uy + 5 * py)
        right = (ex - 10 * ux - 5 * px, ey - 10 * uy - 5 * py)
        draw.polygon((tip, left, right), fill=_PATH)

    def _render(
        self,
        pits: Sequence[int],
        bank: int,
        *,
        source: Optional[int] = None,
        active_slot: Optional[int] = None,
        active_depth: float = 0.0,
    ) -> Image.Image:
        supersampled, draw = new_scaled_canvas(
            CANVAS_SIZE, CANVAS_SIZE, DOCTRINE_SUPERSAMPLE, _BG
        )
        factor = draw.factor
        draw.rounded_rectangle((24, 157, 488, 334), radius=26, fill=_PANEL, outline=(188, 199, 214), width=3)
        centres = self._centres()
        for index in range(6):
            self._arrow(draw, (centres[index][0] + 29, 245), (centres[index + 1][0] - 33, 245))
        # The return arrow is routed below the slots so it cannot be mistaken
        # for a direct B -> P6 edge.
        draw.open_path(
            ((450, 288), (450, 310), (58, 310), (58, 287)), fill=_PATH, width=3
        )
        self._arrow(draw, (58, 310), (58, 284))

        label_font = load_font(17 * factor)
        count_font = load_font(31 * factor)
        depth = max(0.0, min(1.0, float(active_depth)))
        for index, (cx, cy) in enumerate(centres[:6]):
            radius = 27
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=_PIT, outline=_INK, width=3)
            if source == index:
                draw.ellipse((cx - radius - 6, cy - radius - 6, cx + radius + 6, cy + radius + 6), outline=_GOLD, width=6)
            draw.text((cx, cy - 1), str(int(pits[index])), fill=_COUNT, font=count_font, anchor="mm")
            # Pit names are model-facing ordinals numbered from 1 at the left;
            # the rendered blue integer is a token count, not an ordinal.
            draw.text((cx, cy - 47), f"P{index + 1}", fill=_MUTED, font=label_font, anchor="mm")

        bx, by = centres[6]
        draw.rounded_rectangle((bx - 31, by - 43, bx + 31, by + 43), radius=17, fill=_BANK, outline=_INK, width=3)
        draw.text((bx, by - 1), str(int(bank)), fill=_COUNT, font=count_font, anchor="mm")
        draw.text((bx, by - 62), "B", fill=_MUTED, font=label_font, anchor="mm")
        if active_slot is not None and depth > 0.0:
            overlay = Image.new("RGBA", supersampled.size, (0, 0, 0, 0))
            overlay_draw = ScaledDraw(ImageDraw.Draw(overlay), factor)
            if active_slot < 6:
                cx, cy = centres[active_slot]
                radius = 27
                overlay_draw.ellipse(
                    (cx - radius - 6, cy - radius - 6, cx + radius + 6, cy + radius + 6),
                    outline=(*_GOLD, 255),
                    width=5,
                )
            elif active_slot == 6:
                overlay_draw.rounded_rectangle(
                    (bx - 37, by - 49, bx + 37, by + 49),
                    radius=21,
                    outline=(*_GOLD, 255),
                    width=5,
                )
            else:
                raise ValueError("active Mancala slot must be one of the six pits or the bank")
            alpha = overlay.getchannel("A").point(lambda value: int(round(value * depth)))
            overlay.putalpha(alpha)
            supersampled = Image.alpha_composite(supersampled.convert("RGBA"), overlay).convert("RGB")
        return finish_scaled_canvas(supersampled, (CANVAS_SIZE, CANVAS_SIZE))

    def _write_video(
        self,
        sample_id: str,
        pits: Sequence[int],
        bank: int,
        source: int,
        destinations: Sequence[int],
        puzzle: Image.Image,
        solution: Image.Image,
    ) -> Tuple[Optional[Path], int]:
        if not self.video:
            return None, 0
        frames: List[np.ndarray] = [np.array(puzzle, dtype=np.uint8) for _ in range(7)]
        state = [int(value) for value in pits] + [int(bank)]
        state[source] = 0
        # Emptying the selected slot is the first discrete commit, before any
        # carried token starts toward the next slot.  Hold that cumulative
        # state so it cannot be hidden inside the first motion frame.
        source_empty = np.array(self._render(state[:6], state[6]), dtype=np.uint8)
        frames.extend(source_empty.copy() for _ in range(4))
        for destination in destinations:
            # The complete destination outline stays fixed while only its
            # local opacity changes. This makes the next numeric commit explicit
            # without a hard pale outline or an abstract travelling stone.
            for opacity in (1.0 / 3.0, 2.0 / 3.0, 1.0):
                frame = self._render(
                    state[:6],
                    state[6],
                    active_slot=destination,
                    active_depth=opacity,
                )
                frames.append(np.array(frame, dtype=np.uint8))
            state[destination] += 1
            committed = np.array(
                self._render(state[:6], state[6], active_slot=destination, active_depth=1.0),
                dtype=np.uint8,
            )
            frames.append(committed.copy())
            for opacity in (2.0 / 3.0, 1.0 / 3.0, 0.0):
                frame = self._render(
                    state[:6],
                    state[6],
                    active_slot=destination if opacity > 0.0 else None,
                    active_depth=opacity,
                )
                frames.append(np.array(frame, dtype=np.uint8))
        frames.extend(np.array(solution, dtype=np.uint8) for _ in range(10))
        video_path = self.output_dir / "videos" / f"{sample_id}.webm"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        return video_path, encode_video(frames, video_path, fps=self.video_fps)

    def create_puzzle(self) -> BoardChoiceRecord:
        sample_id = f"mancala_distribute_predict-{self._counter:05d}"
        self._counter += 1
        pits, bank, source, final_pits, final_bank, destinations = self._construct()
        puzzle = self._render(pits, bank, source=source)
        solution = self._render(final_pits, final_bank)
        initial_grid = [list(pits) + [bank]]
        final_grid = [list(final_pits) + [final_bank]]
        final_vector = final_grid[0]
        answer_text = format_nonnegative_integer_list_answer(final_vector)
        answer_payload = format_nonnegative_integer_list_payload(final_vector)
        state_vectors: List[List[int]] = [list(pits) + [bank]]
        running = list(state_vectors[0])
        running[source] = 0
        state_vectors.append(list(running))
        for destination in destinations:
            running[destination] += 1
            state_vectors.append(list(running))
        payload = {
            "pits": list(pits),
            "bank": bank,
            "source": source,
            "distribution_destinations": list(destinations),
            "initial_grid": initial_grid,
            "state_grid": final_grid,
            "state_trace": [[list(vector)] for vector in state_vectors],
            "process_state_trace": [[list(vector)] for vector in state_vectors],
        }
        cumulative_states: List[Tuple[str, Image.Image]] = []
        running = [int(value) for value in pits] + [int(bank)]
        removed = running[source]
        running[source] = 0
        # ``source`` and ``destination`` stay internal 0-based list positions;
        # the authored trace names pits with the 1-based labels drawn on the board.
        cumulative_states.append(
            (
                f"The gold-ringed source pit P{source + 1} holds {removed} token{'s' if removed != 1 else ''}, so it "
                f"empties to 0 first and its ring disappears; every other count stays put.",
                self._render(running[:6], running[6]),
            )
        )
        for placement_index, destination in enumerate(destinations, start=1):
            running[destination] += 1
            destination_label = "B" if destination == 6 else f"P{destination + 1}"
            wrap_note = " — the loop wraps from B back to P1 here" if destination == 0 else ""
            cumulative_states.append(
                (
                    f"Token {placement_index} of {removed} follows the arrow to {destination_label}{wrap_note}: its "
                    f"count rises from {running[destination] - 1} to {running[destination]} while the other six stay "
                    "put.",
                    self._render(running[:6], running[6]),
                )
            )
        record = write_board_choice_record(
            generator=self,
            sample_id=sample_id,
            task_name="mancala_distribute_predict",
            task_family="board_game",
            question_description="neutral_seven_slot_cyclic_distribution",
            puzzle_image=puzzle,
            solution_image=solution,
            option_tiles=(),
            correct_option_index=None,
            prompts=_prompts(),
            structured_payload=payload,
            question_text="Return the seven final slot counts in the required order.",
            video_states=None,
            video_verifier_states=tuple(
                self._render(vector[:6], vector[6])
                if index
                else puzzle
                for index, vector in enumerate(state_vectors)
            ),
            single_input=True,
            direct_answer_text=answer_text,
            direct_answer_value=answer_payload,

        )
        video_path, frame_count = self._write_video(
            sample_id,
            pits,
            bank,
            source,
            destinations,
            puzzle,
            solution,
        )
        if video_path is not None and frame_count > 0:
            record.solution_video_path = self.relativize_path(video_path)
            record.video_fps = self.video_fps
            record.video_num_frames = frame_count
        return record


__all__ = ["MancalaDistributePredictGenerator", "distribute_counts"]
