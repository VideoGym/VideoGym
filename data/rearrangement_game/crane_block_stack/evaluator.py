"""Evaluator for the crane_block_stack rearrangement task.

A submitted plan counts the columns 1 for the leftmost column onward; the
replay and the independent BFS below index them from 0.  The single conversion
happens where the payload is parsed, and an ordinal outside the published
one-based domain — an old zero-based ``0``, for instance — rejects the whole
plan instead of raising.  The numbers printed on the blocks are identities and
never appear in the payload.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.core.answer_indexing import from_visible_plan

def _parse_moves(text: str) -> Optional[List[Tuple[int, int]]]:
    if not text:
        return None
    payload = text
    if "answer:" in payload.lower():
        payload = payload[payload.lower().index("answer:") + len("answer:"):]
    match = re.fullmatch(r"\s*(\[.*\])\s*\.?\s*", payload, flags=re.DOTALL)
    if match is None:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    submitted: List[Tuple[int, int]] = []
    for entry in parsed:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return None
        if type(entry[0]) is not int or type(entry[1]) is not int:
            return None
        submitted.append((entry[0], entry[1]))
    return from_visible_plan(TASK_NAME, submitted)


def _replay(state: List[List[int]], moves: List[Tuple[int, int]]) -> Optional[List[List[int]]]:
    cur = [list(s) for s in state]
    n = len(cur)
    for src, dst in moves:
        if not (0 <= src < n and 0 <= dst < n) or src == dst:
            return None
        if not cur[src]:
            return None
        cur[dst].append(cur[src].pop())
    return cur


def _validated_states(
    record: Dict[str, Any],
) -> Optional[Tuple[List[List[int]], List[List[int]]]]:
    raw_initial = record.get("initial_state")
    raw_target = record.get("target_state")
    if not isinstance(raw_initial, list) or not isinstance(raw_target, list):
        return None
    if not 3 <= len(raw_initial) <= 5 or len(raw_target) != len(raw_initial):
        return None

    def parse(raw: List[Any]) -> Optional[List[List[int]]]:
        if any(not isinstance(stack, list) for stack in raw):
            return None
        values = [value for stack in raw for value in stack]
        if not 4 <= len(values) <= 6 or any(type(value) is not int for value in values):
            return None
        if sorted(values) != list(range(1, len(values) + 1)):
            return None
        return [list(stack) for stack in raw]

    initial = parse(raw_initial)
    target = parse(raw_target)
    if initial is None or target is None:
        return None
    if sorted(value for stack in initial for value in stack) != sorted(
        value for stack in target for value in stack
    ):
        return None
    return initial, target


def _shortest_length(initial: List[List[int]], target: List[List[int]]) -> Optional[int]:
    start = tuple(tuple(stack) for stack in initial)
    goal = tuple(tuple(stack) for stack in target)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        state, distance = queue.popleft()
        if state == goal:
            return distance
        for src, stack in enumerate(state):
            if not stack:
                continue
            top = stack[-1]
            for dst in range(len(state)):
                if src == dst:
                    continue
                successor = [list(column) for column in state]
                successor[src].pop()
                successor[dst].append(top)
                key = tuple(tuple(column) for column in successor)
                if key in seen:
                    continue
                seen.add(key)
                queue.append((key, distance + 1))
    return None


class CraneBlockStackEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        if candidate_path.exists() and candidate_path.is_file():
            text = candidate_path.read_text(encoding="utf-8")
        else:
            text = str(candidate)
        moves = _parse_moves(text)
        states = _validated_states(record)
        initial, target = states if states is not None else (None, None)
        optimal_length = (
            _shortest_length(initial, target)
            if initial is not None and target is not None
            else None
        )
        result = _replay(initial, moves) if initial is not None and moves is not None else None
        success = (
            result is not None
            and target is not None
            and moves is not None
            and optimal_length is not None
            and len(moves) == optimal_length
            and result == target
        )
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": [[a, b] for a, b in moves] if moves is not None else None,
            "optimal_length": optimal_length,
            "valid": success,
            "result_state": result,
            "target_state": target,
        }


__all__ = ["CraneBlockStackEvaluator"]

TASK_NAME = "crane_block_stack"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
