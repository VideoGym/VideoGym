"""Deterministic explicit-schedule Tic-Tac-Toe rule simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.board_game.core.answers import (
    encode_board_matrix,
    format_board_matrix_answer,
)
from data.board_game.core.coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    to_visible_cell,
    visible_cell_text,
)
from data.board_game.core.tic_tac_toe import (
    MarkMove,
    TicTacToeRenderer,
    TicTacToeScenario,
    build_ttt_simulator_scenario,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import DEFAULT_FPS
from data.common.pipeline.board_records import BoardChoiceRecord, write_board_choice_record
from data.common.prompts import build_sectioned_prompt
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.video_encoding import encode_rgb_frames_to_webm

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the appearing marks and the yellow winning borders",
    static_nouns="the grid, the row and column numbers, every earlier mark, background, and camera",
    hold_noun="the final board with the three yellow winning borders",
)


TASK_NAME = "tic_tac_toe_rule_simulator"
_STATE_CODE = {None: 0, "X": 1, "O": 2}
_BOARD_SYMBOL = {None: ".", "X": "X", "O": "O"}


def _encode_board(board: Sequence[Sequence[Optional[str]]]) -> List[List[int]]:
    return [[_STATE_CODE[cell] for cell in row] for row in board]


def _visible_cell(cell: Tuple[int, int]) -> str:
    return visible_cell_text(cell, size=3)


def _schedule_text(moves: Sequence[MarkMove]) -> str:
    return "; ".join(
        f"{index}) place {move.mark} at {_visible_cell(move.cell)}"
        for index, move in enumerate(moves, start=1)
    )


def _prompts(scenario: TicTacToeScenario) -> Dict[str, str]:
    input_description = (
        "The single input image shows one 3 by 3 board of white cells. Column numbers 1-3 run above the board and row "
        f"numbers 1-3 run at its left. {GRID_XY_CONVENTION_SENTENCE} "
        "Blue X marks and red O marks occupy some cells; the rest are empty."
    )
    rules = (
        "Execute this sequence exactly as written: "
        f"{_schedule_text(scenario.moves)}. Each operation names its own mark; write it into the named empty cell, "
        "leave every other cell unchanged, and do not infer turns from the mark counts. After each placement, check "
        "the three rows, three columns, and both corner-to-corner diagonals: the first line of three equal marks ends "
        "the game at once, and every listed operation after it is ignored. Exactly one placement in this sequence "
        "completes that first line."
    )
    video_output = (
        "For each executed operation up to and including the winning placement, fade one complete fixed-size X or O in "
        "at its named cell and hold the board; never place an ignored suffix operation. When the winning mark commits, "
        "fade in all three complete yellow rounded borders around the winning cells at once. Do not draw a mark or "
        "border stroke by stroke, reveal the borders one at a time, or alter another cell."
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
        source = frames[-1]
        _append_local_transition(frames, source, target)
        frames.extend(target.copy() for _ in range(4))
    frames.extend(arrays[-1].copy() for _ in range(4))
    if len(frames) > 157 or (len(frames) - 1) % 4:
        raise ValueError("Tic-Tac-Toe video violates the 4N+1 media budget")
    if not encode_rgb_frames_to_webm(frames, path, fps=16, extra_args=["-g", "1", "-bf", "0"]):
        return 0
    return len(frames)


class TicTacToeRuleSimulatorGenerator(AbstractPuzzleGenerator[BoardChoiceRecord]):
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
        self.renderer = TicTacToeRenderer()

    def record_to_dict(self, record: BoardChoiceRecord | Dict[str, Any]) -> Dict[str, Any]:
        return record.to_dict() if hasattr(record, "to_dict") else dict(record)

    def create_puzzle(self) -> BoardChoiceRecord:
        sample_id = f'{TASK_NAME}-{self._counter:05d}'
        scenario_seed = self.seed * 1000003 + self._counter
        self._counter += 1
        scenario = build_ttt_simulator_scenario(scenario_seed)
        simulation = scenario.simulation
        if simulation.stop_reason != 'win' or len(simulation.winning_cells) != 3:
            raise AssertionError('Tic-Tac-Toe generator must author one terminal three-mark win')
        puzzle = self.renderer.render(scenario.initial_board)
        solution = self.renderer.render(simulation.final_board, winning_cells=simulation.winning_cells)
        placement_states = [self.renderer.render(step.board_after) for step in simulation.steps]
        video_states = [puzzle, *placement_states, solution]
        answer_payload = encode_board_matrix(simulation.final_board, symbols=_BOARD_SYMBOL)
        answer_text = format_board_matrix_answer(simulation.final_board, symbols=_BOARD_SYMBOL)
        for step_index, step in enumerate(simulation.steps, start=1):
            if not step.winning_cells and (step.board_full or step_index == len(simulation.steps)):
                raise AssertionError('Tic-Tac-Toe authored trace must end only on the winning step')
        semantic_state = {'initial_board': [list(row) for row in scenario.initial_board], 'operations': [{'piece': move.mark, 'cell': list(to_visible_cell(move.cell, size=3))} for move in scenario.moves]}
        state_trace = [_encode_board(scenario.initial_board)] + [_encode_board(step.board_after) for step in simulation.steps]
        payload = {'semantic_state': semantic_state, 'initial_grid': state_trace[0], 'state_grid': state_trace[-1], 'state_trace': state_trace, 'process_state_trace': state_trace, 'reference_stop_reason': simulation.stop_reason, 'reference_winner': simulation.winner, 'reference_executed_move_count': len(simulation.steps), 'reference_ignored_move_count': len(simulation.ignored_moves), 'reference_final_board': [list(row) for row in simulation.final_board]}
        return write_board_choice_record(generator=self, sample_id=sample_id, task_name=TASK_NAME, task_family='board_game', question_description='explicit_mark_schedule_with_terminal_three_line_stop', puzzle_image=puzzle, solution_image=solution, option_tiles=(), correct_option_index=None, prompts=_prompts(scenario), structured_payload=payload, question_text='Execute the supplied X/O placement program and return the final board.', video_states=video_states, video_verifier_states=video_states, video_writer=lambda path: _write_video(path, video_states), single_input=True, direct_answer_text=answer_text, direct_answer_value=answer_payload, qa_type='State Prediction', qa_level='Medium')


__all__ = ["TicTacToeRuleSimulatorGenerator"]
