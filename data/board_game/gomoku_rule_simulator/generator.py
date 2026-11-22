"""Explicit-colour 9x9 Gomoku simulation with first-terminal stopping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from data.board_game.core.answers import (
    encode_board_matrix,
    format_board_matrix_answer,
)
from data.board_game.core.coordinates import GRID_XY_CONVENTION_SENTENCE, visible_cell_text
from data.board_game.core.intersection_games import (
    GomokuScenario,
    NumericIntersectionRenderer,
    StoneMove,
    build_gomoku_simulator_scenario,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import DEFAULT_FPS
from data.common.pipeline.board_records import BoardChoiceRecord, write_board_choice_record
from data.common.prompts import build_sectioned_prompt
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.video_encoding import encode_rgb_frames_to_webm

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the appearing stones and the yellow winning line",
    static_nouns="the wood board, grid lines, coordinate numbers, and every earlier stone",
    hold_noun="the final board with the yellow winning line",
)


TASK_NAME = "gomoku_rule_simulator"
_STATE_CODE = {None: 0, "B": 1, "W": 2}
_BOARD_SYMBOL = {None: ".", "B": "B", "W": "W"}


def _encode_board(board: Sequence[Sequence[Optional[str]]]) -> List[List[int]]:
    return [[_STATE_CODE[cell] for cell in row] for row in board]


def _schedule_text(moves: Sequence[StoneMove]) -> str:
    return "; ".join(
        f"{index}) place {move.colour} at {visible_cell_text(move.cell, size=9)}"
        for index, move in enumerate(moves, start=1)
    )


def _prompts(scenario: GomokuScenario) -> Dict[str, str]:
    input_description = (
        "The single input image shows one 9 by 9 intersection grid on a wood board, with black and white stones on "
        "some intersections. Column numbers 1-9 run above the grid and row numbers 1-9 run at its left; the top-left "
        f"intersection uses the same numbering in both directions. {GRID_XY_CONVENTION_SENTENCE}"
    )
    rules = (
        "Execute this sequence exactly as written: "
        f"{_schedule_text(scenario.moves)}. Each operation names its own colour (B for black, W for white); place that "
        "stone on the named empty intersection, leave every earlier stone unchanged, and do not infer turns from the "
        "stone counts. After each placement, check every horizontal, vertical, and diagonal contiguous run: the first "
        "run of five or more stones of one colour ends the game at once, and every listed operation after it is "
        "ignored. Exactly one placement in this sequence creates that first run. There is no capture, movement, or "
        "other unstated rule."
    )
    video_output = (
        "For each executed operation up to and including the winning placement, fade one complete fixed-size stone of "
        "the named colour in at its intersection and hold the board; never place an ignored suffix operation. After "
        "the winning stone commits, fade in one complete straight yellow line through all five winning stone centres "
        "at once. Do not draw the line from one end, slide stones in, or alter an existing stone."
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
        raise ValueError("Gomoku video violates the 4N+1 media budget")
    if not encode_rgb_frames_to_webm(frames, path, fps=16, extra_args=["-g", "1", "-bf", "0"]):
        return 0
    return len(frames)


class GomokuRuleSimulatorGenerator(AbstractPuzzleGenerator[BoardChoiceRecord]):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        video_fps: int = DEFAULT_FPS,
        video: bool = False,
    ) -> None:
        super().__init__(output_dir)
        if int(video_fps) != 16:
            raise ValueError("Board Game source videos use exactly 16 fps")
        self.seed = 0 if seed is None else int(seed)
        self.video_fps = 16
        self.video = bool(video)
        self._counter = 0
        self.renderer = NumericIntersectionRenderer(9)

    def record_to_dict(self, record: BoardChoiceRecord | Dict[str, Any]) -> Dict[str, Any]:
        return record.to_dict() if hasattr(record, "to_dict") else dict(record)

    def create_puzzle(self) -> BoardChoiceRecord:
        sample_id = f'{TASK_NAME}-{self._counter:05d}'
        scenario_seed = self.seed * 1000003 + self._counter
        self._counter += 1
        scenario = build_gomoku_simulator_scenario(scenario_seed)
        simulation = scenario.simulation
        if simulation.stop_reason != 'win' or len(simulation.winning_cells) != 5:
            raise AssertionError('Gomoku generator must author one terminal five-stone win')
        puzzle = self.renderer.render(scenario.initial_board)
        solution = self.renderer.render(simulation.final_board, winning_line=simulation.winning_cells)
        placement_states = [self.renderer.render(step.board_after) for step in simulation.steps]
        video_states = [puzzle, *placement_states, solution]
        answer_payload = encode_board_matrix(simulation.final_board, symbols=_BOARD_SYMBOL)
        answer_text = format_board_matrix_answer(simulation.final_board, symbols=_BOARD_SYMBOL)
        for step_index, step in enumerate(simulation.steps, start=1):
            if not step.winning_cells and (step.board_full or step_index == len(simulation.steps)):
                raise AssertionError('Gomoku authored trace must end only on the winning step')
        semantic_state = {'initial_board': [list(row) for row in scenario.initial_board], 'operations': [{'piece': move.colour, 'cell': list(move.visible_cell(size=9))} for move in scenario.moves]}
        state_trace = [_encode_board(scenario.initial_board)] + [_encode_board(step.board_after) for step in simulation.steps]
        payload = {'semantic_state': semantic_state, 'initial_grid': state_trace[0], 'state_grid': state_trace[-1], 'state_trace': state_trace, 'process_state_trace': state_trace, 'reference_stop_reason': simulation.stop_reason, 'reference_winner': simulation.winner, 'reference_executed_move_count': len(simulation.steps), 'reference_ignored_move_count': len(simulation.ignored_moves), 'reference_winning_cells': [list(cell) for cell in simulation.winning_cells], 'reference_final_board': [list(row) for row in simulation.final_board]}
        return write_board_choice_record(generator=self, sample_id=sample_id, task_name=TASK_NAME, task_family='board_game', question_description='explicit_colour_sequence_with_terminal_five_stop', puzzle_image=puzzle, solution_image=solution, option_tiles=(), correct_option_index=None, prompts=_prompts(scenario), structured_payload=payload, question_text='Execute the supplied B/W placement program and return the final 9 by 9 board.', video_states=video_states, video_verifier_states=video_states, video_writer=lambda path: _write_video(path, video_states), single_input=True, direct_answer_text=answer_text, direct_answer_value=answer_payload, qa_type='State Prediction', qa_level='Medium')


__all__ = ["GomokuRuleSimulatorGenerator"]
