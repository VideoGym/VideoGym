"""Evaluator for meal_train_pass_serve."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.meal_train_pass_serve.generator import (
    _format_seat_path,
    _moves_from_seat_path,
    _solve_min_passes_with_count,
)

_SEAT_PATH_RE = re.compile(r"Answer: \[[1-9](?:, [1-9])+\]\.\Z")


def _parse_seat_path(text: str, num_seats: int) -> Optional[List[int]]:
    if not isinstance(text, str) or not text.splitlines():
        return None
    line = text.splitlines()[-1]
    if _SEAT_PATH_RE.fullmatch(line) is None:
        return None
    try:
        parsed = ast.literal_eval(line[len("Answer: ") : -1])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or any(type(seat) is not int for seat in parsed):
        return None
    seats = [int(seat) for seat in parsed]
    try:
        return seats if _format_seat_path(seats, num_seats) == line else None
    except ValueError:
        return None


def _parse_moves(
    text: str,
    start: int,
    num_seats: int,
) -> Optional[List[str]]:
    seats = _parse_seat_path(text, num_seats)
    return (
        _moves_from_seat_path(start, seats, num_seats)
        if seats is not None
        else None
    )


class MealTrainPassServeEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = (
            candidate_path.read_text(encoding="utf-8")
            if candidate_path.exists() and candidate_path.is_file()
            else str(candidate)
        )
        initial = record.get("initial_state")
        source_valid = isinstance(initial, dict)
        try:
            start = int(initial["current_seat"])
            demands = {int(seat): str(value) for seat, value in initial["demands"].items()}
            # The table size is part of the visible input and varies per
            # sample, so it is read from the record rather than assumed.
            num_seats = len(demands) + len(initial.get("empty_seats") or [])
            source_valid = bool(
                source_valid
                and num_seats >= 3
                and 0 <= start < num_seats
                and demands
                and all(0 <= seat < num_seats for seat in demands)
            )
            gold, shortest_count = _solve_min_passes_with_count(start, demands, num_seats)
            seats = _parse_seat_path(text, num_seats)
            guess = (
                _moves_from_seat_path(start, seats, num_seats)
                if seats is not None
                else None
            )
        except (KeyError, TypeError, ValueError):
            gold, shortest_count, source_valid, seats, guess = None, 0, False, None, None
        valid = bool(source_valid and shortest_count == 1 and gold is not None and guess == gold)
        return {
            "puzzle_id": puzzle_id,
            "seat_path": seats,
            "guess": guess or [],
            "expected": gold or [],
            "shortest_path_count": shortest_count,
            "source_contract_valid": source_valid,
            "valid": valid,
        }


__all__ = ["MealTrainPassServeEvaluator"]

TASK_NAME = "meal_train_pass_serve"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
