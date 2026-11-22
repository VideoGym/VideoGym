"""Strict independent evaluator for ``stack_reversal_minimal_moves``.

A submitted move pairs two column positions counted from 1.  The BFS, the
replay and the renderer below count columns from 0, so :func:`_parse_moves`
performs the single conversion and fails closed — returning ``None`` rather
than raising — on any column number outside the published domain, including an
old zero-based payload.
"""

from __future__ import annotations

import re
from collections import deque
from functools import lru_cache
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.core.answer_indexing import from_visible_plan

TASK_NAME = "stack_reversal_minimal_moves"

BLOCK_COUNT_RANGE = (3, 5)
BUFFER_COLUMNS_RANGE = (2, 3)
MIN_OPTIMAL_MOVES = 6
MAX_OPTIMAL_MOVES = 13
MAX_COLUMNS = BUFFER_COLUMNS_RANGE[1] + 1

State = Tuple[Tuple[int, ...], ...]
Move = Tuple[int, int]

_ANSWER_RE = re.compile(r"Answer: \[((?:\(\d+, \d+\)(?:, \(\d+, \d+\))*)?)\]\.")
_MOVE_RE = re.compile(r"\((\d+), (\d+)\)")


def _parse_moves(text: str) -> Optional[List[Move]]:
    """Parse the formal payload and recover internal 0-based column moves."""

    if not isinstance(text, str):
        return None
    match = _ANSWER_RE.fullmatch(text)
    if match is None or not match.group(1):
        return None
    submitted = [(int(src), int(dst)) for src, dst in _MOVE_RE.findall(match.group(1))]
    if not submitted or any(
        not (1 <= src <= MAX_COLUMNS and 1 <= dst <= MAX_COLUMNS) for src, dst in submitted
    ):
        return None
    canonical = "Answer: [" + ", ".join(f"({src}, {dst})" for src, dst in submitted) + "]."
    if canonical != text:
        return None
    recovered = from_visible_plan(TASK_NAME, submitted)
    if recovered is None:
        return None
    return [(int(src), int(dst)) for src, dst in recovered]


def _parse_state(raw: Any) -> Optional[State]:
    if not isinstance(raw, list) or not 3 <= len(raw) <= 4:
        return None
    parsed: List[Tuple[int, ...]] = []
    values: List[int] = []
    for stack in raw:
        if not isinstance(stack, list):
            return None
        parsed_stack: List[int] = []
        for value in stack:
            if type(value) is not int:
                return None
            parsed_stack.append(value)
            values.append(value)
        parsed.append(tuple(parsed_stack))
    if not BLOCK_COUNT_RANGE[0] <= len(values) <= BLOCK_COUNT_RANGE[1]:
        return None
    if sorted(values) != list(range(1, len(values) + 1)):
        return None
    return tuple(parsed)


def _validated_states(record: Dict[str, Any]) -> Optional[Tuple[State, State]]:
    if not isinstance(record, dict):
        return None
    initial = _parse_state(record.get("initial_state"))
    target = _parse_state(record.get("target_state"))
    if initial is None or target is None or len(initial) != len(target):
        return None
    nonempty_initial = [index for index, stack in enumerate(initial) if stack]
    nonempty_target = [index for index, stack in enumerate(target) if stack]
    if len(nonempty_initial) != 1 or nonempty_target != nonempty_initial:
        return None
    primary = nonempty_initial[0]
    if target[primary] != tuple(reversed(initial[primary])):
        return None
    if any(stack for index, stack in enumerate(target) if index != primary):
        return None
    buffer_count = len(initial) - 1
    if not BUFFER_COLUMNS_RANGE[0] <= buffer_count <= BUFFER_COLUMNS_RANGE[1]:
        return None
    return initial, target


def _successors(state: State) -> List[Tuple[State, Move]]:
    out: List[Tuple[State, Move]] = []
    for src, stack in enumerate(state):
        if not stack:
            continue
        block = stack[-1]
        for dst in range(len(state)):
            if src == dst:
                continue
            successor = list(state)
            successor[src] = stack[:-1]
            successor[dst] = state[dst] + (block,)
            out.append((tuple(successor), (src, dst)))
    return out


@lru_cache(maxsize=512)
def _shortest_stats(initial: State, target: State) -> Tuple[Optional[int], int]:
    if initial == target:
        return 0, 1
    queue = deque([initial])
    distance = {initial: 0}
    path_count = {initial: 1}
    target_distance: Optional[int] = None
    while queue:
        state = queue.popleft()
        current_distance = distance[state]
        if target_distance is not None and current_distance >= target_distance:
            continue
        for successor, _ in _successors(state):
            next_distance = current_distance + 1
            previous = distance.get(successor)
            if previous is None:
                distance[successor] = next_distance
                path_count[successor] = path_count[state]
                queue.append(successor)
            elif previous == next_distance:
                path_count[successor] += path_count[state]
            if successor == target:
                target_distance = next_distance
    if target not in distance:
        return None, 0
    return distance[target], path_count[target]


@lru_cache(maxsize=16)
def _canonical_reversal_stats(block_count: int, buffer_count: int) -> Tuple[Optional[int], int]:
    initial: State = (tuple(range(1, block_count + 1)),) + (tuple(),) * buffer_count
    target: State = (tuple(reversed(initial[0])),) + (tuple(),) * buffer_count
    return _shortest_stats(initial, target)


def _replay(initial: State, moves: List[Move]) -> Optional[State]:
    state = initial
    for src, dst in moves:
        if not (0 <= src < len(state) and 0 <= dst < len(state)) or src == dst or not state[src]:
            return None
        block = state[src][-1]
        successor = list(state)
        successor[src] = state[src][:-1]
        successor[dst] = state[dst] + (block,)
        state = tuple(successor)
    return state


class StackReversalMinimalMovesEvaluator(AbstractPuzzleEvaluator):
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
        shortest_distance, shortest_path_count = _shortest_stats(initial, target)
        block_count = sum(len(stack) for stack in initial)
        buffer_count = len(initial) - 1
        canonical_distance, canonical_path_count = _canonical_reversal_stats(block_count, buffer_count)
        one_buffer_distance, one_buffer_path_count = _canonical_reversal_stats(block_count, 1)
        other_buffer_count = 3 if buffer_count == 2 else 2
        other_buffer_distance, other_buffer_path_count = _canonical_reversal_stats(
            block_count, other_buffer_count
        )
        source_valid = (
            shortest_distance is not None
            and shortest_distance == canonical_distance
            and MIN_OPTIMAL_MOVES <= shortest_distance <= MAX_OPTIMAL_MOVES
            and shortest_path_count == canonical_path_count
            and shortest_path_count >= 1
            and one_buffer_distance is None
            and one_buffer_path_count == 0
            and other_buffer_distance is not None
            and abs(shortest_distance - other_buffer_distance) == 1
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
            "shortest_path_count": shortest_path_count,
            "one_buffer_distance": one_buffer_distance,
            "other_buffer_distance": other_buffer_distance,
            "other_buffer_path_count": other_buffer_path_count,
            "valid": valid,
            "result_state": [list(stack) for stack in result] if result is not None else None,
        }


__all__ = [
    "StackReversalMinimalMovesEvaluator",
    "_canonical_reversal_stats",
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
