"""Evaluator for the dock_yard_crane_swap rearrangement task.

A submitted plan names the lanes 1, 2, and 3 from left to right; the replay
and the independent BFS below index them 0, 1, 2.  The crane name is not an
ordinal and is preserved.  Conversion happens once, where the payload is
parsed, and a lane number outside the published one-based domain — an old
zero-based ``0``, for instance — rejects the plan instead of raising.
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

Move = Tuple[str, int, int]
State = Tuple[Tuple[int, ...], ...]


def _parse_moves(text: str) -> Optional[List[Move]]:
    """Recover internal crane/lane triples from one submitted one-based payload."""

    if not text:
        return None
    match = re.fullmatch(r"\s*Answer:\s*(\[.*\])\s*\.?\s*", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    submitted: List[Move] = []
    for entry in parsed:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            return None
        crane, src, dst = entry
        if type(crane) is not str or crane not in {"L", "R"}:
            return None
        if type(src) is not int or type(dst) is not int:
            return None
        submitted.append((crane, src, dst))
    recovered = from_visible_plan(TASK_NAME, submitted)
    if recovered is None:
        return None
    return [(str(crane), int(src), int(dst)) for crane, src, dst in recovered]


def _replay(initial: List[List[int]], moves: List[Move]) -> Optional[List[List[int]]]:
    cur = [list(stack) for stack in initial]
    for crane, src, dst in moves:
        if not (0 <= src < 3 and 0 <= dst < 3) or src == dst:
            return None
        if crane == "R":
            if not cur[src]:
                return None
            cur[dst].append(cur[src].pop())
        elif crane == "L":
            if len(cur[src]) < 2:
                return None
            top = cur[src].pop()
            second = cur[src].pop()
            cur[dst].extend([top, second])
        else:
            return None
    return cur


def _validated_states(
    record: Dict[str, Any],
) -> Optional[Tuple[List[List[int]], List[List[int]]]]:
    raw_initial = record.get("initial_state")
    raw_target = record.get("target_state")
    if not isinstance(raw_initial, list) or not isinstance(raw_target, list):
        return None
    if len(raw_initial) != 3 or len(raw_target) != 3:
        return None

    def parse(raw: List[Any]) -> Optional[List[List[int]]]:
        if any(not isinstance(stack, list) for stack in raw):
            return None
        values = [value for stack in raw for value in stack]
        if not 4 <= len(values) <= 5 or any(type(value) is not int for value in values):
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
    if initial == target:
        return None
    return initial, target


def _state_key(state: List[List[int]]) -> State:
    return tuple(tuple(stack) for stack in state)


def _shortest_length(
    initial: List[List[int]],
    target: List[List[int]],
    *,
    allowed_cranes: frozenset[str] = frozenset({"L", "R"}),
) -> Optional[int]:
    start = _state_key(initial)
    goal = _state_key(target)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        state, distance = queue.popleft()
        if state == goal:
            return distance
        for src in range(3):
            for dst in range(3):
                if src == dst:
                    continue
                if "R" in allowed_cranes and state[src]:
                    successor = [list(stack) for stack in state]
                    successor[dst].append(successor[src].pop())
                    key = _state_key(successor)
                    if key not in seen:
                        seen.add(key)
                        queue.append((key, distance + 1))
                if "L" in allowed_cranes and len(state[src]) >= 2:
                    successor = [list(stack) for stack in state]
                    top = successor[src].pop()
                    second = successor[src].pop()
                    successor[dst].extend([top, second])
                    key = _state_key(successor)
                    if key not in seen:
                        seen.add(key)
                        queue.append((key, distance + 1))
    return None


class DockYardCraneSwapEvaluator(AbstractPuzzleEvaluator):
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
        r_only_length = (
            _shortest_length(initial, target, allowed_cranes=frozenset({"R"}))
            if initial is not None and target is not None
            else None
        )
        l_only_length = (
            _shortest_length(initial, target, allowed_cranes=frozenset({"L"}))
            if initial is not None and target is not None
            else None
        )
        source_contract_valid = (
            optimal_length is not None
            and 3 <= optimal_length <= 7
            and (r_only_length is None or r_only_length > optimal_length)
            and (l_only_length is None or l_only_length > optimal_length)
        )
        result = (
            _replay(initial, moves)
            if initial is not None and moves is not None
            else None
        )
        valid = (
            source_contract_valid
            and result is not None
            and moves is not None
            and target is not None
            and len(moves) == optimal_length
            and result == target
        )
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": [[c, s, d] for c, s, d in moves] if moves is not None else None,
            "optimal_length": optimal_length,
            "r_only_length": r_only_length,
            "l_only_length": l_only_length,
            "source_contract_valid": source_contract_valid,
            "valid": valid,
            "result_state": result,
            "target_state": target,
        }


__all__ = ["DockYardCraneSwapEvaluator"]

TASK_NAME = "dock_yard_crane_swap"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
