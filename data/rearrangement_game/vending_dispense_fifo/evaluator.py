"""Strict semantic evaluator for :mod:`vending_dispense_fifo`.

The prompt requires a *shortest* legal sequence, so merely reaching the target
is insufficient.  This evaluator parses the complete typed payload, replays
every action, and independently solves the small FIFO state graph to recover
the minimum move count.  It never trusts ``optimal_moves`` or ``move_count``
metadata written by the generator.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike

Move = Tuple[str, int]
State = Tuple[Tuple[int, ...], Tuple[int, ...]]

_ANSWER_PATTERN = re.compile(r"\A\s*Answer\s*:\s*(\[.*\])\s*\.?\s*\Z", re.DOTALL)


def _parse_moves(text: str) -> Tuple[Optional[List[Move]], str]:
    """Parse the complete answer without silently dropping malformed entries."""

    if not text:
        return None, "missing_answer"
    match = _ANSWER_PATTERN.fullmatch(str(text))
    if match is None:
        return None, "answer_format"
    try:
        parsed = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None, "answer_syntax"
    if not isinstance(parsed, list):
        return None, "moves_not_list"

    moves: List[Move] = []
    for entry in parsed:
        if not isinstance(entry, tuple) or len(entry) != 2:
            return None, "move_not_two_field_tuple"
        action, value = entry
        if not isinstance(action, str) or action not in {"dispense", "reinsert"}:
            return None, "unknown_action"
        if isinstance(value, bool) or not isinstance(value, int):
            return None, "move_value_not_integer"
        if action == "dispense" and value != 0:
            return None, "dispense_value_must_be_zero"
        if action == "reinsert" and value <= 0:
            return None, "reinsert_id_must_be_positive"
        moves.append((action, value))
    return moves, "ok"


def _normalize_state(raw: Any) -> Optional[State]:
    if not isinstance(raw, dict):
        return None
    chute = raw.get("chute")
    tray = raw.get("tray")
    if not isinstance(chute, (list, tuple)) or not isinstance(tray, (list, tuple)):
        return None
    if any(isinstance(value, bool) or not isinstance(value, int) for value in [*chute, *tray]):
        return None
    all_cans = [int(value) for value in [*chute, *tray]]
    if any(can_id <= 0 for can_id in all_cans) or len(set(all_cans)) != len(all_cans):
        return None
    return tuple(int(value) for value in chute), tuple(int(value) for value in tray)


def _replay(initial: State, moves: Sequence[Move]) -> Tuple[Optional[State], int, str]:
    chute = list(initial[0])
    tray = list(initial[1])
    for index, (action, value) in enumerate(moves):
        if action == "dispense":
            if not chute:
                return None, index, "dispense_from_empty_chute"
            tray.append(chute.pop(0))
        elif action == "reinsert":
            if value not in tray:
                return None, index, "reinsert_can_not_in_tray"
            tray.remove(value)
            chute.append(value)
        else:  # Defensive: strict parsing should make this unreachable.
            return None, index, "unknown_action"
    return (tuple(chute), tuple(tray)), len(moves), "ok"


def _minimum_move_count(initial: State, target: State) -> Optional[int]:
    """Breadth-first search the complete finite state graph."""

    if sorted((*initial[0], *initial[1])) != sorted((*target[0], *target[1])):
        return None
    queue = deque([(initial, 0)])
    seen = {initial}
    while queue:
        state, distance = queue.popleft()
        if state == target:
            return distance
        chute, tray = state
        if chute:
            dispensed = (chute[1:], tray + (chute[0],))
            if dispensed not in seen:
                seen.add(dispensed)
                queue.append((dispensed, distance + 1))
        for index, can_id in enumerate(tray):
            reinserted = (chute + (can_id,), tray[:index] + tray[index + 1 :])
            if reinserted not in seen:
                seen.add(reinserted)
                queue.append((reinserted, distance + 1))
    return None


def _state_payload(state: Optional[State]) -> Optional[Dict[str, List[int]]]:
    if state is None:
        return None
    return {"chute": list(state[0]), "tray": list(state[1])}


class VendingDispenseFifoEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        started = perf_counter()
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        try:
            is_candidate_file = candidate_path.exists() and candidate_path.is_file()
        except OSError:
            is_candidate_file = False
        text = candidate_path.read_text(encoding="utf-8") if is_candidate_file else str(candidate)
        moves, parse_reason = _parse_moves(text)
        initial = _normalize_state(record.get("initial_state"))
        target = _normalize_state(record.get("target_state"))

        if initial is None or target is None:
            failure_reason = "invalid_semantic_record"
            result = None
            applied_count = 0
            replay_reason = "not_run"
            optimal_count = None
        elif moves is None:
            failure_reason = parse_reason
            result = None
            applied_count = 0
            replay_reason = "not_run"
            optimal_count = _minimum_move_count(initial, target)
        else:
            result, applied_count, replay_reason = _replay(initial, moves)
            optimal_count = _minimum_move_count(initial, target)
            if result is None:
                failure_reason = replay_reason
            elif result != target:
                failure_reason = "wrong_final_state"
            elif optimal_count is None:
                failure_reason = "target_unreachable"
            elif len(moves) != optimal_count:
                failure_reason = "not_shortest"
            else:
                failure_reason = "ok"

        parse_score = 1.0 if moves is not None else 0.0
        if moves is None:
            move_legality_score = 0.0
        elif not moves:
            move_legality_score = 1.0
        else:
            move_legality_score = applied_count / len(moves)
        target_score = 1.0 if result is not None and result == target else 0.0
        optimality_score = (
            1.0
            if moves is not None and optimal_count is not None and len(moves) == optimal_count
            else 0.0
        )
        success = (
            parse_score == 1.0
            and move_legality_score == 1.0
            and target_score == 1.0
            and optimality_score == 1.0
        )
        latency_ms = (perf_counter() - started) * 1000.0
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": [[action, value] for action, value in (moves or [])],
            "result_state": _state_payload(result),
            "target_state": _state_payload(target),
            "optimal_move_count": optimal_count,
            "parse_score": parse_score,
            "move_legality_score": move_legality_score,
            "target_state_score": target_score,
            "optimality_score": optimality_score,
            "binary_correct": success,
            "valid": success,
            "failure_reason": failure_reason,
            "verifier_profile": "vending_fifo_shortest_replay_v1",
            "latency_ms": latency_ms,
        }


__all__ = [
    "VendingDispenseFifoEvaluator",
    "_minimum_move_count",
    "_parse_moves",
    "_replay",
]

TASK_NAME = "vending_dispense_fifo"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
