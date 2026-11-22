"""Strict independent evaluator for ``chess_pawn_swap`` coordinate moves."""

from __future__ import annotations

import re
from collections import deque
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.core.answer_indexing import from_visible_plan

TASK_NAME = "chess_pawn_swap"

ROWS = 5
COLS = 5
EMPTY = "."
MIN_OPTIMAL_MOVES = 8
MAX_OPTIMAL_MOVES = 13

State = Tuple[str, ...]
Move = Tuple[int, int]

_XY_PATTERN = r"\([1-5], [1-5]\)"
_MOVE_PATTERN = rf"\({_XY_PATTERN}, {_XY_PATTERN}\)"
_ANSWER_RE = re.compile(
    rf"Answer: \[({_MOVE_PATTERN}(?:, {_MOVE_PATTERN})*)\]\."
)
_MOVE_RE = re.compile(
    r"\(\(([1-5]), ([1-5])\), \(([1-5]), ([1-5])\)\)"
)


def _parse_moves(text: str) -> Optional[List[Move]]:
    """Accept only the exact nonempty grammar, returning 0-based squares."""

    if not isinstance(text, str):
        return None
    match = _ANSWER_RE.fullmatch(text)
    if match is None or not match.group(1):
        return None
    submitted = [
        ((int(source_x), int(source_y)), (int(destination_x), int(destination_y)))
        for source_x, source_y, destination_x, destination_y in _MOVE_RE.findall(match.group(1))
    ]
    if not submitted:
        return None
    recovered = from_visible_plan(TASK_NAME, submitted)
    if recovered is None:
        return None
    return [(int(source), int(destination)) for source, destination in recovered]


def _validated_states(record: Dict[str, Any]) -> Optional[Tuple[State, State]]:
    if not isinstance(record, dict):
        return None
    if record.get("rows", ROWS) != ROWS or record.get("cols", COLS) != COLS:
        return None
    initial_raw = record.get("initial_state")
    target_raw = record.get("target_state")
    if not isinstance(initial_raw, list) or not isinstance(target_raw, list):
        return None
    if len(initial_raw) != ROWS * COLS or len(target_raw) != ROWS * COLS:
        return None
    if any(not isinstance(cell, str) or cell not in {EMPTY, "W", "B"} for cell in initial_raw + target_raw):
        return None
    initial = tuple(initial_raw)
    target = tuple(target_raw)
    if initial == target:
        return None

    white_initial = [index for index, cell in enumerate(initial) if cell == "W"]
    black_initial = [index for index, cell in enumerate(initial) if cell == "B"]
    if len(white_initial) not in {2, 3} or len(black_initial) != len(white_initial):
        return None
    if any(index // COLS != ROWS - 1 for index in white_initial):
        return None
    if any(index // COLS != 0 for index in black_initial):
        return None

    white_target = [index for index, cell in enumerate(target) if cell == "W"]
    black_target = [index for index, cell in enumerate(target) if cell == "B"]
    if not white_target or not black_target:
        return None
    if any(index // COLS != 0 for index in white_target):
        return None
    if any(index // COLS != ROWS - 1 for index in black_target):
        return None
    if len(white_target) > len(white_initial) or len(black_target) > len(black_initial):
        return None
    if len(white_target) + len(black_target) >= len(white_initial) + len(black_initial):
        return None
    return initial, target


def _legal_moves(state: State, *, allow_capture: bool = True) -> List[Move]:
    moves: List[Move] = []
    for src, color in enumerate(state):
        if color not in {"W", "B"}:
            continue
        row, col = divmod(src, COLS)
        row_step = -1 if color == "W" else 1
        destination_row = row + row_step
        if not 0 <= destination_row < ROWS:
            continue
        straight = destination_row * COLS + col
        if state[straight] == EMPTY:
            moves.append((src, straight))
        if not allow_capture:
            continue
        opponent = "B" if color == "W" else "W"
        for destination_col in (col - 1, col + 1):
            if not 0 <= destination_col < COLS:
                continue
            destination = destination_row * COLS + destination_col
            if state[destination] == opponent:
                moves.append((src, destination))
    return moves


def _apply(state: State, move: Move, *, allow_capture: bool = True) -> Optional[State]:
    if move not in _legal_moves(state, allow_capture=allow_capture):
        return None
    src, dst = move
    next_state = list(state)
    next_state[dst] = next_state[src]
    next_state[src] = EMPTY
    return tuple(next_state)


@lru_cache(maxsize=65536)
def _minimum_survivor_steps(state: State, target: State) -> Optional[int]:
    """Optimistic move lower bound for matching the immutable terminal cells."""

    total = 0
    for color in ("W", "B"):
        current = [index for index, cell in enumerate(state) if cell == color]
        goals = [index for index, cell in enumerate(target) if cell == color]
        if len(current) < len(goals):
            return None
        terminal_row = 0 if color == "W" else ROWS - 1
        terminal_current = {
            index for index in current if index // COLS == terminal_row
        }
        if not terminal_current.issubset(set(goals)):
            return None

        best: Optional[int] = None
        for assigned in permutations(current, len(goals)):
            if not terminal_current.issubset(set(assigned)):
                continue
            steps = 0
            feasible = True
            for source, destination in zip(assigned, goals):
                source_row, source_col = divmod(source, COLS)
                _, destination_col = divmod(destination, COLS)
                forward_steps = source_row if color == "W" else ROWS - 1 - source_row
                if abs(source_col - destination_col) > forward_steps:
                    feasible = False
                    break
                steps += forward_steps
            if feasible and (best is None or steps < best):
                best = steps
        if best is None:
            return None
        total += best
    return total


@lru_cache(maxsize=512)
def _shortest_stats(
    initial: State,
    target: State,
    *,
    allow_capture: bool = True,
    max_depth: int = MAX_OPTIMAL_MOVES,
) -> Tuple[Optional[int], int]:
    if initial == target:
        return 0, 1
    if not allow_capture:
        if sum(cell != EMPTY for cell in initial) != sum(cell != EMPTY for cell in target):
            return None, 0
    queue = deque([initial])
    distance = {initial: 0}
    path_count = {initial: 1}
    target_distance: Optional[int] = None
    while queue:
        state = queue.popleft()
        current_distance = distance[state]
        if current_distance >= max_depth:
            continue
        if target_distance is not None and current_distance >= target_distance:
            continue
        for move in _legal_moves(state, allow_capture=allow_capture):
            successor = _apply(state, move, allow_capture=allow_capture)
            if successor is None:
                continue
            next_distance = current_distance + 1
            previous = distance.get(successor)
            if previous is not None:
                if previous == next_distance:
                    path_count[successor] += path_count[state]
                continue
            lower_bound = _minimum_survivor_steps(successor, target)
            if lower_bound is None or next_distance + lower_bound > max_depth:
                continue
            distance[successor] = next_distance
            path_count[successor] = path_count[state]
            queue.append(successor)
            if successor == target:
                target_distance = next_distance
    if target not in distance:
        return None, 0
    return distance[target], path_count[target]


def _replay(initial: State, moves: List[Move]) -> Optional[State]:
    state = initial
    for move in moves:
        next_state = _apply(state, move)
        if next_state is None:
            return None
        state = next_state
    return state


class ChessPawnSwapEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = (
            candidate_path.read_text(encoding="utf-8")
            if candidate_path.exists() and candidate_path.is_file()
            else str(candidate)
        )
        moves = _parse_moves(text)
        states = _validated_states(record)
        if moves is None or states is None:
            return {
                "puzzle_id": puzzle_id,
                "moves_parsed": moves,
                "format_valid": moves is not None,
                "source_valid": states is not None,
                "valid": False,
                "result_state": None,
            }
        initial, target = states
        shortest_distance, shortest_count = _shortest_stats(initial, target)
        no_capture_distance, no_capture_count = _shortest_stats(
            initial,
            target,
            allow_capture=False,
        )
        source_valid = (
            shortest_distance is not None
            and MIN_OPTIMAL_MOVES <= shortest_distance <= MAX_OPTIMAL_MOVES
            and shortest_count >= 1
            and no_capture_distance is None
            and no_capture_count == 0
        )
        result = _replay(initial, moves) if source_valid else None
        valid = (
            source_valid
            and result == target
            and shortest_distance is not None
            and len(moves) == shortest_distance
        )
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": moves,
            "format_valid": True,
            "source_valid": source_valid,
            "shortest_distance": shortest_distance,
            "shortest_path_count": shortest_count,
            "no_capture_distance": no_capture_distance,
            "no_capture_path_count": no_capture_count,
            "valid": valid,
            "result_state": list(result) if result is not None else None,
        }


__all__ = [
    "ChessPawnSwapEvaluator",
    "_parse_moves",
    "_replay",
    "_shortest_stats",
    "_validated_states",
]

VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
