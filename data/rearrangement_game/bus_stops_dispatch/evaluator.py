"""Evaluator for bus_stops_dispatch."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.bus_stops_dispatch.generator import (
    CAPACITY,
    _apply_direction_state,
    _format_stop_path,
    _is_complete,
    _moves_from_stop_path,
    _shortest_plan,
)


_STOP_PATH_RE = re.compile(r"Answer: \[[1-5](?:, [1-5])+\]\.\Z")


def _parse_stop_path(text: str) -> Optional[List[int]]:
    if not isinstance(text, str) or not text.splitlines():
        return None
    line = text.splitlines()[-1]
    if _STOP_PATH_RE.fullmatch(line) is None:
        return None
    try:
        parsed = ast.literal_eval(line[len("Answer: ") : -1])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or any(type(stop) is not int for stop in parsed):
        return None
    stops = [int(stop) for stop in parsed]
    return stops if _format_stop_path(stops) == line else None


def _parse_moves(text: str, initial: Dict[str, Any]) -> Optional[List[str]]:
    stops = _parse_stop_path(text)
    return _moves_from_stop_path(initial, stops) if stops is not None else None


def _replay(initial: Dict[str, Any], moves: List[str]) -> Optional[bool]:
    state = initial
    try:
        for move in moves:
            state = _apply_direction_state(state, move, CAPACITY)
    except (TypeError, ValueError):
        return None
    return _is_complete(state)




class BusStopsDispatchEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() and candidate_path.is_file() else str(candidate)
        initial = record.get("initial_state", {})
        stops = _parse_stop_path(text)
        moves = (
            _moves_from_stop_path(initial, stops)
            if stops is not None and isinstance(initial, dict)
            else None
        )
        optimal = _shortest_plan(initial, CAPACITY, 30)
        result = _replay(initial, moves) if moves is not None else None
        shortest = (
            moves is not None
            and optimal is not None
            and len(moves) == len(optimal)
        )
        return {
            "puzzle_id": puzzle_id,
            "stops_parsed": stops,
            "moves_parsed": moves,
            "optimal_length": len(optimal) if optimal is not None else None,
            "reaches_goal": bool(result),
            "valid": bool(result) and shortest,
        }


__all__ = ["BusStopsDispatchEvaluator"]

TASK_NAME = "bus_stops_dispatch"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
