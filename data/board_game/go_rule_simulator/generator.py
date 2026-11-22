"""Explicit-colour 7x7 Go rule simulation with source-derived commits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.board_game.core.answers import (
    encode_board_matrix,
    format_board_matrix_answer,
)
from data.board_game.core.intersection_games import (
    GoSimulatorScenario,
    NumericIntersectionRenderer,
    StoneMove,
    build_go_simulator_scenario,
)
from data.board_game.core.coordinates import GRID_XY_CONVENTION_SENTENCE, visible_cell_text
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import DEFAULT_FPS
from data.common.pipeline.board_records import BoardChoiceRecord, write_board_choice_record
from data.common.prompts import build_sectioned_prompt
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.video_encoding import encode_rgb_frames_to_webm

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the appearing stones and the fading captured stones",
    static_nouns="the wood board, grid lines, coordinate numbers, and every unaffected stone",
    hold_noun="the final board after the last operation",
)


TASK_NAME = "go_rule_simulator"
_STATE_CODE = {None: 0, "B": 1, "W": 2}
_BOARD_SYMBOL = {None: ".", "B": "B", "W": "W"}


def _encode_board(board: Sequence[Sequence[Optional[str]]]) -> List[List[int]]:
    return [[_STATE_CODE[cell] for cell in row] for row in board]


def _schedule_text(moves: Sequence[StoneMove]) -> str:
    return "; ".join(
        f"{index}) place {move.colour} at {visible_cell_text(move.cell, size=7)}"
        for index, move in enumerate(moves, start=1)
    )


def _cell_list(cells: Sequence[Tuple[int, int]]) -> str:
    return ", ".join(visible_cell_text(cell, size=7) for cell in cells)


def _prompts(scenario: GoSimulatorScenario) -> Dict[str, str]:
    input_description = (
        "The single input image shows one 7 by 7 Go-style intersection grid on a wood board, with black and white "
        "stones on some intersections. Column numbers 1-7 run above the grid and row numbers 1-7 run at its left; the "
        f"grid uses the same numbering in both directions. {GRID_XY_CONVENTION_SENTENCE}"
    )
    rules = (
        "Execute this sequence exactly as written: "
        f"{_schedule_text(scenario.moves)}. Each operation names its own colour (B for black, W for white); place that "
        "stone on the named empty intersection and do not infer turns from the stone counts. Stones of one colour "
        "connected through horizontal or vertical adjacency form a group, and a liberty is an empty intersection "
        "orthogonally adjacent to any stone of the group; diagonal contact and the board edge give no liberties. After "
        "each placement, every opposing group orthogonally adjacent to the new stone that has no liberty left is "
        "removed, all such groups at the same time; the new stone's own group must then still have a liberty, and "
        "every listed operation is legal when reached. There is no pass, ko, or history rule."
    )
    video_output = (
        "For each listed operation in order, fade one complete fixed-size stone of the named colour in at its "
        "intersection and hold the board. When a placement leaves adjacent opposing groups without liberties, fade all "
        "their stones out together at that same commit. Do not slide stones in from outside, grow them from the "
        "centre, or remove captured stones one at a time."
    )
    return {
        "ti2v_prompt": build_sectioned_prompt(
            input_description,
            video_output,
            mode="ti2v",
            rule_description=rules,
            continuity_paragraph=_TI2V_CONTINUITY,
        ),
    }


def _append_local_transition(frames: List[np.ndarray], start: np.ndarray, end: np.ndarray) -> None:
    changed = np.any(start != end, axis=2)
    for index in range(1, 5):
        depth = index / 4.0
        smooth = depth * depth * (3.0 - 2.0 * depth)
        frame = start.copy()
        frame[changed] = np.rint(
            start[changed].astype(np.float32) * (1.0 - smooth)
            + end[changed].astype(np.float32) * smooth
        ).astype(np.uint8)
        frames.append(frame)


def _write_video(path: Path, states: Sequence[Image.Image]) -> int:
    arrays = [np.asarray(image.convert("RGB"), dtype=np.uint8) for image in states]
    frames: List[np.ndarray] = [arrays[0].copy() for _ in range(5)]
    for target in arrays[1:]:
        _append_local_transition(frames, frames[-1], target)
        frames.extend(target.copy() for _ in range(4))
    frames.extend(arrays[-1].copy() for _ in range(4))
    if len(frames) > 157 or (len(frames) - 1) % 4:
        raise ValueError("Go simulator video violates the 4N+1 media budget")
    if not encode_rgb_frames_to_webm(frames, path, fps=16, extra_args=["-g", "1", "-bf", "0"]):
        return 0
    return len(frames)


class GoRuleSimulatorGenerator(AbstractPuzzleGenerator[BoardChoiceRecord]):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        video_fps: int = DEFAULT_FPS,
        video: bool = False,
        program_length: int | None = None,
    ) -> None:
        super().__init__(output_dir)
        if int(video_fps) != 16:
            raise ValueError("Board Game source videos use exactly 16 fps")
        self.seed = 0 if seed is None else int(seed)
        self.video_fps = 16
        self.video = bool(video)
        self._counter = 0
        self.program_length = program_length
        # Deliberately no geometry parameters: search and simulator share this exact constructor.
        self.renderer = NumericIntersectionRenderer(7)

    def record_to_dict(self, record: BoardChoiceRecord | Dict[str, Any]) -> Dict[str, Any]:
        return record.to_dict() if hasattr(record, "to_dict") else dict(record)

    def create_puzzle(self) -> BoardChoiceRecord:
        sample_id = f"{TASK_NAME}-{self._counter:05d}"
        scenario_seed = self.seed * 1_000_003 + self._counter
        self._counter += 1
        scenario = build_go_simulator_scenario(
            scenario_seed, program_length=self.program_length
        )
        simulation = scenario.simulation
        puzzle = self.renderer.render(scenario.initial_board)
        solution = self.renderer.render(simulation.final_board)

        video_states: List[Image.Image] = [puzzle]
        state_trace = [_encode_board(scenario.initial_board)]
        process_trace = [_encode_board(scenario.initial_board)]
        for step_index, step in enumerate(simulation.steps, start=1):
            placed_plain = self.renderer.render(step.placed_board)
            video_states.append(placed_plain)
            process_trace.append(_encode_board(step.placed_board))
            if step.captured_cells:
                captured_image = self.renderer.render(step.final_board)
                video_states.append(captured_image)
                process_trace.append(_encode_board(step.final_board))
            state_trace.append(_encode_board(step.final_board))

        answer_payload = encode_board_matrix(simulation.final_board, symbols=_BOARD_SYMBOL)
        answer_text = format_board_matrix_answer(simulation.final_board, symbols=_BOARD_SYMBOL)
        semantic_state = {
            "initial_board": [list(row) for row in scenario.initial_board],
            "operations": [
                {"piece": move.colour, "cell": list(move.visible_cell(size=7))}
                for move in scenario.moves
            ],
        }
        payload = {
            "semantic_state": semantic_state,
            "initial_grid": state_trace[0],
            "state_grid": state_trace[-1],
            "state_trace": state_trace,
            "process_state_trace": process_trace,
            "reference_final_board": [list(row) for row in simulation.final_board],
            "reference_capture_counts": [step.captured_count for step in simulation.steps],
            "reference_captured_cells": [
                [list(cell) for cell in step.captured_cells] for step in simulation.steps
            ],
        }
        return write_board_choice_record(
            generator=self,
            sample_id=sample_id,
            task_name=TASK_NAME,
            task_family="board_game",
            question_description="explicit_colour_sequence_with_simultaneous_group_capture",
            puzzle_image=puzzle,
            solution_image=solution,
            option_tiles=(),
            correct_option_index=None,
            prompts=_prompts(scenario),
            structured_payload=payload,
            question_text="Execute the supplied B/W placement program and return the final 7 by 7 board.",
            video_states=video_states,
            video_verifier_states=video_states,
            video_writer=lambda path: _write_video(path, video_states),
            single_input=True,
            direct_answer_text=answer_text,
            direct_answer_value=answer_payload,
            qa_type="State Prediction",
            qa_level="Medium",
        )


__all__ = ["GoRuleSimulatorGenerator"]
