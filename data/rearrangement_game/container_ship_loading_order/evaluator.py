"""Evaluator for container_ship_loading_order."""

from __future__ import annotations

import ast
import re
from collections import deque
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, List, Optional, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike

def _parse_moves(text: str) -> Optional[List[Tuple[str, int]]]:
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
    moves: List[Tuple[str, int]] = []
    for entry in parsed:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return None
        if not isinstance(entry[0], str) or type(entry[1]) is not int:
            return None
        action = entry[0].lower()
        if action not in {"ship", "buffer", "unbuffer"}:
            return None
        moves.append((action, entry[1]))
    return moves


def _replay(initial: Dict[str, Any], moves: List[Tuple[str, int]]) -> Optional[Dict[str, Any]]:
    dock = list(initial.get("dock", []))
    ship = list(initial.get("ship", []))
    buf = initial.get("buffer")
    for action, value in moves:
        if action == "ship":
            if buf == value:
                ship.append(buf)
                buf = None
            elif dock and dock[-1] == value:
                dock.pop()
                ship.append(value)
            else:
                return None
        elif action == "buffer":
            if not dock or dock[-1] != value or buf is not None:
                return None
            dock.pop()
            buf = value
        elif action == "unbuffer":
            if buf != value:
                return None
            dock.append(value)
            buf = None
        else:
            return None
    return {"dock": dock, "ship": ship, "buffer": buf}


def _source_state(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = record.get("initial_state")
    if not isinstance(raw, dict):
        return None
    dock = raw.get("dock")
    if not isinstance(dock, list) or not 4 <= len(dock) <= 5:
        return None
    if raw.get("ship") not in ([], tuple()) or raw.get("buffer") is not None:
        return None
    try:
        crates = [int(value) for value in dock]
    except (TypeError, ValueError):
        return None
    if any(isinstance(value, bool) for value in dock):
        return None
    if sorted(crates) != list(range(1, len(crates) + 1)):
        return None
    return {"dock": crates, "ship": [], "buffer": None}


def _shortest_plan(initial: Dict[str, Any]) -> Optional[List[Tuple[str, int]]]:
    dock = tuple(initial["dock"])
    target = (tuple(), tuple(sorted(dock, reverse=True)), None)
    start = (dock, tuple(), None)
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        (cur_dock, cur_ship, buffer), path = queue.popleft()
        if (cur_dock, cur_ship, buffer) == target:
            return path

        successors: List[Tuple[Tuple[Tuple[int, ...], Tuple[int, ...], Optional[int]], Tuple[str, int]]] = []
        if cur_dock:
            top = cur_dock[-1]
            successors.append(((cur_dock[:-1], cur_ship + (top,), buffer), ("ship", top)))
            if buffer is None:
                successors.append(((cur_dock[:-1], cur_ship, top), ("buffer", top)))
        if buffer is not None:
            successors.append(((cur_dock, cur_ship + (buffer,), None), ("ship", buffer)))
            successors.append(((cur_dock + (buffer,), cur_ship, None), ("unbuffer", buffer)))

        for state, move in successors:
            if state in seen:
                continue
            seen.add(state)
            queue.append((state, path + [move]))
    return None


class ContainerShipLoadingOrderEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = candidate_path.read_text(encoding="utf-8") if candidate_path.exists() and candidate_path.is_file() else str(candidate)
        moves = _parse_moves(text)
        initial = _source_state(record)
        optimal = _shortest_plan(initial) if initial is not None else None
        target = (
            {"dock": [], "ship": sorted(initial["dock"], reverse=True), "buffer": None}
            if initial is not None
            else None
        )
        result = _replay(initial, moves) if initial is not None and moves is not None else None
        success = (
            result is not None
            and moves is not None
            and optimal is not None
            and len(moves) == len(optimal)
            and target is not None
            and result.get("dock") == target.get("dock")
            and result.get("ship") == target.get("ship")
            and result.get("buffer") == target.get("buffer")
        )
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": [[m, v] for m, v in moves] if moves is not None else None,
            "optimal_length": len(optimal) if optimal is not None else None,
            "valid": success,
            "result_state": result,
            "target_state": target,
        }


__all__ = ["ContainerShipLoadingOrderEvaluator"]

TASK_NAME = "container_ship_loading_order"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
