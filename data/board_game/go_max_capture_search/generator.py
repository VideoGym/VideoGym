"""Unique maximum-capture Black-placement search on the shared 7x7 board."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from PIL import Image
from data.board_game.core.coordinates import GRID_XY_CONVENTION_SENTENCE, visible_cell_text
from data.common.grid_coordinates import format_xy_position_answer
from data.board_game.core.intersection_games import NumericIntersectionRenderer, build_go_max_capture_scenario
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import DEFAULT_FPS
from data.common.pipeline.board_records import BoardChoiceRecord, write_board_choice_record
from data.common.prompts import build_sectioned_prompt
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.video_encoding import encode_rgb_frames_to_webm
_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(timing='continuous', moving_nouns='the appearing Black stone and the fading captured White stones', static_nouns='the wood board, grid lines, coordinate numbers, and every uninvolved stone', hold_noun='the final board after the capture')
TASK_NAME = 'go_max_capture_search'
_STATE_CODE = {None: 0, 'B': 1, 'W': 2}

def _encode_board(board: Sequence[Sequence[Optional[str]]]) -> List[List[int]]:
    return [[_STATE_CODE[cell] for cell in row] for row in board]

def _prompts() -> Dict[str, str]:
    input_description = f'The single input image shows one 7 by 7 Go-style intersection grid on a wood board, with black and white stones on some intersections. Column numbers 1-7 run above the grid and row numbers 1-7 run at its left; the intersection grid uses the same numbering in both directions. {GRID_XY_CONVENTION_SENTENCE}'
    rules = 'Consider placing one new Black stone on an empty intersection. Stones of one colour connected through horizontal or vertical adjacency form a group, and a liberty is an empty intersection orthogonally adjacent to any stone of the group; diagonal contact and the board edge give no liberties. After a placement, every White group orthogonally adjacent to the new stone that has no liberty left is removed, all such groups at the same time. A placement whose own Black group would still have no liberty after those removals is forbidden. There is no pass, ko, or history rule. Find the placement that removes the most White stones; exactly one placement achieves the maximum, and the maximum is at least one stone.'
    image_output = 'Render the board after the best Black placement with all captured White stones removed. Keep the wood board, grid, coordinate numbers, remaining stones, and camera unchanged; add no ring, badge, arrow, or extra board.'
    video_output = 'Make one complete fixed-size Black stone appear at the best intersection by fading in at its full footprint, hold it, then remove all captured White stones by fading them out together in place. Do not slide a stone in from outside, grow it from its centre, remove captures one at a time, or show any trial placement.'
    return {'ti2v_prompt': build_sectioned_prompt(input_description, video_output, mode='ti2v', rule_description=rules, continuity_paragraph=_TI2V_CONTINUITY), 'solution_image_prompt': build_sectioned_prompt(input_description, image_output, mode='solution_image', rule_description=rules)}

def _append_local_transition(frames: List[np.ndarray], start: np.ndarray, end: np.ndarray) -> None:
    changed = np.any(start != end, axis=2)
    for index in range(1, 5):
        depth = index / 4.0
        smooth = depth * depth * (3.0 - 2.0 * depth)
        frame = start.copy()
        frame[changed] = np.rint(start[changed].astype(np.float32) * (1.0 - smooth) + end[changed].astype(np.float32) * smooth).astype(np.uint8)
        frames.append(frame)

def _write_video(path: Path, states: Sequence[Image.Image]) -> int:
    arrays = [np.asarray(image.convert('RGB'), dtype=np.uint8) for image in states]
    frames: List[np.ndarray] = [arrays[0].copy() for _ in range(5)]
    for target in arrays[1:]:
        _append_local_transition(frames, frames[-1], target)
        frames.extend((target.copy() for _ in range(4)))
    frames.extend((arrays[-1].copy() for _ in range(4)))
    if len(frames) > 157 or (len(frames) - 1) % 4:
        raise ValueError('Go max-capture video violates the 4N+1 media budget')
    if not encode_rgb_frames_to_webm(frames, path, fps=16, extra_args=['-g', '1', '-bf', '0']):
        return 0
    return len(frames)

class GoMaxCaptureSearchGenerator(AbstractPuzzleGenerator[BoardChoiceRecord]):

    def __init__(self, output_dir: str | Path, *, seed: Optional[int]=None, video_fps: int=DEFAULT_FPS, video: bool=False) -> None:
        super().__init__(output_dir)
        if int(video_fps) != 16:
            raise ValueError('Board Game source videos use exactly 16 fps')
        self.seed = 0 if seed is None else int(seed)
        self.video_fps = 16
        self.video = bool(video)
        self._counter = 0
        self.renderer = NumericIntersectionRenderer(7)

    def record_to_dict(self, record: BoardChoiceRecord | Dict[str, Any]) -> Dict[str, Any]:
        return record.to_dict() if hasattr(record, 'to_dict') else dict(record)

    def create_puzzle(self) -> BoardChoiceRecord:
        sample_id = f'{TASK_NAME}-{self._counter:05d}'
        scenario_seed = self.seed * 1000003 + self._counter
        self._counter += 1
        scenario = build_go_max_capture_scenario(scenario_seed)
        solution = scenario.solution
        if len(solution.best_cells) != 1 or solution.max_captured <= 0:
            raise AssertionError('search source must prove one strictly positive maximum')
        best_cell = solution.best_cells[0]
        best_candidate = next((candidate for candidate in solution.legal_candidates if candidate.cell == best_cell))
        puzzle = self.renderer.render(scenario.initial_board)
        placed_plain = self.renderer.render(best_candidate.step.placed_board)
        terminal = self.renderer.render(best_candidate.step.final_board)
        answer_payload = visible_cell_text(best_cell, size=7)
        answer_text = format_xy_position_answer(best_cell)
        video_states = [puzzle, placed_plain, terminal]
        semantic_state = {'initial_board': [list(row) for row in scenario.initial_board], 'search_piece': 'B'}
        payload = {'semantic_state': semantic_state, 'initial_grid': _encode_board(scenario.initial_board), 'state_grid': _encode_board(best_candidate.step.final_board), 'state_trace': [_encode_board(scenario.initial_board), _encode_board(best_candidate.step.final_board)], 'process_state_trace': [_encode_board(scenario.initial_board), _encode_board(best_candidate.step.placed_board), _encode_board(best_candidate.step.final_board)], 'reference_best_cells': [list(cell) for cell in solution.best_cells], 'reference_max_captured': solution.max_captured, 'reference_captured_cells': [list(cell) for cell in best_candidate.step.captured_cells], 'reference_final_board': [list(row) for row in best_candidate.step.final_board]}
        return write_board_choice_record(generator=self, sample_id=sample_id, task_name=TASK_NAME, task_family='board_game', question_description='unique_black_placement_with_maximum_white_capture', puzzle_image=puzzle, solution_image=terminal, option_tiles=(), correct_option_index=None, prompts=_prompts(), structured_payload=payload, question_text='Find the unique legal Black placement that captures the most White stones.', video_states=video_states, video_verifier_states=video_states, video_writer=lambda path: _write_video(path, video_states), single_input=True, direct_answer_text=answer_text, direct_answer_value=answer_payload, qa_type='Search', qa_level='Medium')
__all__ = ['GoMaxCaptureSearchGenerator']
