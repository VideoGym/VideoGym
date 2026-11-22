"""Evaluator for Tower of Hanoi rearrangement puzzles. We accept any
plan that respects the Hanoi disk-size constraint and reaches the target
state — not only the BFS-optimal one — so a longer-but-valid plan still
scores positive. A submitted plan names the pegs 1, 2, and 3 from left to right; the replay
below indexes them 0, 1, 2. The single conversion happens where the payload is
parsed, and an ordinal outside the published one-based domain — an old
zero-based ``0``, for instance — rejects the whole plan instead of raising."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.core.answer_indexing import from_visible_plan

_MOVE_PATTERN = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def _parse_moves(answer: str) -> Optional[List[Tuple[int, int]]]:
    """Recover internal peg pairs from one submitted one-based payload.

    Returns ``None`` — never raises — when a submitted peg ordinal falls
    outside the published one-based domain, so the whole plan fails closed
    instead of silently replaying a shifted arrangement.
    """

    return from_visible_plan(TASK_NAME, _submitted_moves(answer))


def _submitted_moves(answer: str) -> List[Tuple[Any, Any]]:
    """Tolerantly read the visible one-based pairs out of the answer text.

    Submitted values keep their parsed type. Coercing them to ``int`` here
    would turn a ``True`` or a ``2.0`` into a peg ordinal and so bypass the
    fail-closed type check at the one-based boundary.
    """

    if not answer:
        return []
    payload = answer
    # Strip an "Answer:" prefix if present.
    if "answer:" in payload.lower():
        idx = payload.lower().index("answer:")
        payload = payload[idx + len("answer:"):]
    payload = payload.strip()
    # Trailing punctuation, brackets, whitespace.
    payload = payload.strip(".? \n\t")
    # First try ast.literal_eval on the bracketed form.
    bracket_match = re.search(r"\[.*\]", payload, flags=re.DOTALL)
    if bracket_match:
        try:
            parsed = ast.literal_eval(bracket_match.group(0))
            if isinstance(parsed, (list, tuple)):
                moves: List[Tuple[Any, Any]] = []
                for entry in parsed:
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        moves.append((entry[0], entry[1]))
                if moves:
                    return moves
        except (ValueError, SyntaxError):
            pass
    # Regex fallback.
    return [(int(a), int(b)) for a, b in _MOVE_PATTERN.findall(payload)]


def _replay_moves(initial_state: List[List[int]], moves: List[Tuple[int, int]]) -> Optional[List[List[int]]]:
    state = [list(stack) for stack in initial_state]
    for src, dst in moves:
        if not (0 <= src < 3 and 0 <= dst < 3) or src == dst:
            return None
        if not state[src]:
            return None
        top = state[src][-1]
        if state[dst] and state[dst][-1] < top:
            return None
        state[dst].append(state[src].pop())
    return state


class TowerOfHanoiEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate_text_or_path: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate_text_or_path)
        if candidate_path.exists() and candidate_path.is_file():
            answer_text = candidate_path.read_text(encoding="utf-8")
        else:
            answer_text = str(candidate_text_or_path)
        initial_state = [list(stack) for stack in record.get("initial_state", [])]
        target_state = [list(stack) for stack in record.get("target_state", [])]
        moves = _parse_moves(answer_text)
        result_state = _replay_moves(initial_state, moves) if moves is not None else None
        success = result_state is not None and result_state == target_state
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": (
                [[int(a), int(b)] for a, b in moves] if moves is not None else None
            ),
            "valid": success,
            "result_state": result_state,
            "target_state": target_state,
        }


__all__ = ["TowerOfHanoiEvaluator"]

TASK_NAME = "tower_of_hanoi"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
