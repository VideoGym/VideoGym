"""Strict visible-source evaluator for three load-bearing LIFO hold cells."""

from __future__ import annotations

import ast
from collections import deque
from functools import lru_cache
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike

MIN_PRODUCTS = 5
MAX_PRODUCTS = 6
NUM_CELLS = 3

Move = Tuple[str, int]
CellState = Tuple[Tuple[int, ...], ...]
SearchState = Tuple[int, CellState, int]
State = Dict[str, Any]

_MOVE_TOKEN = r"\('(?:push|pop)', [1-3]\)"
_ANSWER_RE = re.compile(rf"Answer: \[{_MOVE_TOKEN}(?:, {_MOVE_TOKEN})*\]\.\Z")
def _format_moves(moves: Sequence[Move]) -> str:
    return "Answer: [" + ", ".join(f"('{action}', {cell_id})" for action, cell_id in moves) + "]."


def _parse_answer_line(text: Any) -> Optional[Tuple[Move, ...]]:
    if not isinstance(text, str) or _ANSWER_RE.fullmatch(text) is None:
        return None
    payload = text[len("Answer: ") : -1]
    try:
        parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    moves: List[Move] = []
    for raw_move in parsed:
        if not isinstance(raw_move, tuple) or len(raw_move) != 2:
            return None
        action, cell_id = raw_move
        if type(action) is not str or action not in ("push", "pop"):
            return None
        if type(cell_id) is not int or not 1 <= cell_id <= NUM_CELLS:
            return None
        moves.append((action, int(cell_id)))
    result = tuple(moves)
    return result if _format_moves(result) == text else None


def _parse_moves(text: Any) -> Optional[Tuple[Move, ...]]:
    if not isinstance(text, str):
        return None
    lines = text.splitlines()
    return _parse_answer_line(lines[-1] if lines else text)


def _validate_permutation(values: Any) -> Optional[Tuple[int, ...]]:
    if not isinstance(values, list) or not MIN_PRODUCTS <= len(values) <= MAX_PRODUCTS:
        return None
    parsed: List[int] = []
    for value in values:
        if type(value) is not int:
            return None
        parsed.append(int(value))
    return tuple(parsed) if tuple(sorted(parsed)) == tuple(range(1, len(parsed) + 1)) else None


def _validate_visible_state(raw: Any) -> Optional[State]:
    if not isinstance(raw, dict) or set(raw) != {"arrival", "cells", "outfeed", "target"}:
        return None
    target = _validate_permutation(raw.get("target"))
    if target is None:
        return None
    n = len(target)
    arrival_raw = raw.get("arrival")
    cells_raw = raw.get("cells")
    outfeed_raw = raw.get("outfeed")
    if not isinstance(arrival_raw, list) or len(arrival_raw) != n:
        return None
    if not isinstance(cells_raw, list) or len(cells_raw) != NUM_CELLS:
        return None
    if not isinstance(outfeed_raw, list) or len(outfeed_raw) > n:
        return None
    arrival: List[Optional[int]] = []
    seen_remaining = False
    for value in arrival_raw:
        if value is None:
            if seen_remaining:
                return None
            arrival.append(None)
        elif type(value) is int and 1 <= value <= n:
            seen_remaining = True
            arrival.append(int(value))
        else:
            return None
    cells: List[List[int]] = []
    for raw_cell in cells_raw:
        if not isinstance(raw_cell, list):
            return None
        cell: List[int] = []
        for value in raw_cell:
            if type(value) is not int or not 1 <= value <= n:
                return None
            cell.append(int(value))
        cells.append(cell)
    outfeed: List[int] = []
    for value in outfeed_raw:
        if type(value) is not int or not 1 <= value <= n:
            return None
        outfeed.append(int(value))
    combined = [value for value in arrival if value is not None]
    combined.extend(value for cell in cells for value in cell)
    combined.extend(outfeed)
    if len(combined) != n or tuple(sorted(combined)) != tuple(range(1, n + 1)):
        return None
    if tuple(outfeed) != target[: len(outfeed)]:
        return None
    return {
        "arrival": arrival,
        "cells": cells,
        "outfeed": outfeed,
        "target": list(target),
    }


def _is_solvable(arrival: Sequence[int], target: Sequence[int], num_cells: int) -> bool:
    source = tuple(arrival)
    destination = tuple(target)
    if (
        len(source) not in (MIN_PRODUCTS, MAX_PRODUCTS)
        or tuple(sorted(source)) != tuple(range(1, len(source) + 1))
        or tuple(sorted(destination)) != tuple(range(1, len(source) + 1))
        or type(num_cells) is not int
        or not 1 <= num_cells <= NUM_CELLS
    ):
        return False

    @lru_cache(maxsize=None)
    def visit(arrival_index: int, cells: CellState, output_index: int) -> bool:
        if output_index == len(destination):
            return arrival_index == len(source) and all(not cell for cell in cells)
        needed = destination[output_index]
        for cell_index, cell in enumerate(cells):
            if cell and cell[-1] == needed:
                updated = list(cells)
                updated[cell_index] = cell[:-1]
                if visit(arrival_index, tuple(sorted(updated)), output_index + 1):
                    return True
        if arrival_index < len(source):
            product = source[arrival_index]
            distinct_cells: set[Tuple[int, ...]] = set()
            for cell_index, cell in enumerate(cells):
                if cell in distinct_cells:
                    continue
                distinct_cells.add(cell)
                updated = list(cells)
                updated[cell_index] = cell + (product,)
                if visit(arrival_index + 1, tuple(sorted(updated)), output_index):
                    return True
        return False

    return visit(0, tuple(() for _ in range(num_cells)), 0)


def _independent_shortest(
    arrival: Sequence[int],
    target: Sequence[int],
    num_cells: int = NUM_CELLS,
) -> Optional[Tuple[int, int, Tuple[Move, ...]]]:
    source = tuple(arrival)
    destination = tuple(target)
    if (
        len(source) not in (MIN_PRODUCTS, MAX_PRODUCTS)
        or tuple(sorted(source)) != tuple(range(1, len(source) + 1))
        or tuple(sorted(destination)) != tuple(range(1, len(source) + 1))
        or type(num_cells) is not int
        or not 1 <= num_cells <= NUM_CELLS
    ):
        return None
    start: SearchState = (0, tuple(() for _ in range(num_cells)), 0)
    queue = deque([start])
    distance = {start: 0}
    path_count = {start: 1}
    parent: Dict[SearchState, Tuple[SearchState, Move]] = {}
    goal: Optional[SearchState] = None
    goal_distance: Optional[int] = None
    while queue:
        state = queue.popleft()
        depth = distance[state]
        arrival_index, cells, output_index = state
        if goal_distance is not None and depth >= goal_distance:
            continue
        successors: List[Tuple[SearchState, Move]] = []
        if arrival_index < len(source):
            product = source[arrival_index]
            for cell_index in range(num_cells):
                updated = list(cells)
                updated[cell_index] = cells[cell_index] + (product,)
                successors.append(
                    ((arrival_index + 1, tuple(updated), output_index), ("push", cell_index + 1))
                )
        if output_index < len(destination):
            needed = destination[output_index]
            for cell_index, cell in enumerate(cells):
                if cell and cell[-1] == needed:
                    updated = list(cells)
                    updated[cell_index] = cell[:-1]
                    successors.append(
                        ((arrival_index, tuple(updated), output_index + 1), ("pop", cell_index + 1))
                    )
        for next_state, move in successors:
            next_depth = depth + 1
            if next_state not in distance:
                distance[next_state] = next_depth
                path_count[next_state] = path_count[state]
                parent[next_state] = state, move
                queue.append(next_state)
            elif distance[next_state] == next_depth:
                path_count[next_state] += path_count[state]
            if (
                next_state[0] == len(source)
                and next_state[2] == len(destination)
                and all(not cell for cell in next_state[1])
                and goal is None
            ):
                goal = next_state
                goal_distance = next_depth
    if goal is None:
        return None
    path: List[Move] = []
    cursor = goal
    while cursor != start:
        previous, move = parent[cursor]
        path.append(move)
        cursor = previous
    path.reverse()
    return distance[goal], path_count[goal], tuple(path)


def _parse_visible_pair(record: Any) -> Optional[Tuple[State, State]]:
    if not isinstance(record, dict):
        return None
    source = _validate_visible_state(record.get("initial_state"))
    target = _validate_visible_state(record.get("target_state"))
    if source is None or target is None:
        return None
    n = len(source["target"])
    if target["target"] != source["target"] or len(target["target"]) != n:
        return None
    if any(value is None for value in source["arrival"]):
        return None
    if source["cells"] != [[], [], []] or source["outfeed"]:
        return None
    if target["arrival"] != [None] * n or target["cells"] != [[], [], []]:
        return None
    if target["outfeed"] != source["target"]:
        return None
    arrival = tuple(int(value) for value in source["arrival"] if value is not None)
    target_order = tuple(source["target"])
    if _is_solvable(arrival, target_order, 2) or not _is_solvable(arrival, target_order, 3):
        return None
    return source, target


def _replay(source: State, moves: Sequence[Move]) -> Dict[str, Any]:
    arrival = list(source["arrival"])
    cells = [list(cell) for cell in source["cells"]]
    outfeed = list(source["outfeed"])
    states: List[State] = [
        {
            "arrival": list(arrival),
            "cells": [list(cell) for cell in cells],
            "outfeed": list(outfeed),
            "target": list(source["target"]),
        }
    ]
    legal_count = 0
    illegal_index: Optional[int] = None
    illegal_reason: Optional[str] = None
    used_cells: set[int] = set()
    for index, raw_move in enumerate(moves):
        if not isinstance(raw_move, tuple) or len(raw_move) != 2:
            illegal_index, illegal_reason = index, "move_not_typed_pair"
            break
        action, cell_id = raw_move
        if type(action) is not str or action not in ("push", "pop"):
            illegal_index, illegal_reason = index, "unknown_action"
            break
        if type(cell_id) is not int or not 1 <= cell_id <= NUM_CELLS:
            illegal_index, illegal_reason = index, "invalid_cell"
            break
        used_cells.add(cell_id)
        if action == "push":
            front_index = next((i for i, value in enumerate(arrival) if value is not None), None)
            if front_index is None:
                illegal_index, illegal_reason = index, "empty_infeed"
                break
            product = int(arrival[front_index])
            arrival[front_index] = None
            cells[cell_id - 1].append(product)
        else:
            if not cells[cell_id - 1]:
                illegal_index, illegal_reason = index, "empty_cell"
                break
            outfeed.append(cells[cell_id - 1].pop())
            if len(outfeed) > len(source["target"]):
                illegal_index, illegal_reason = index, "outfeed_overflow"
                break
        legal_count += 1
        states.append(
            {
                "arrival": list(arrival),
                "cells": [list(cell) for cell in cells],
                "outfeed": list(outfeed),
                "target": list(source["target"]),
            }
        )
    return {
        "arrival": arrival,
        "cells": cells,
        "outfeed": outfeed,
        "states": tuple(states),
        "used_cells": tuple(sorted(used_cells)),
        "legal_move_count": legal_count,
        "illegal_move_index": illegal_index,
        "illegal_reason": illegal_reason,
    }


def _structured_format_valid(text: Any, moves: Optional[Tuple[Move, ...]]) -> bool:
    return bool(
        isinstance(text, str)
        and moves is not None
        and text.splitlines()
        and text.splitlines()[-1] == _format_moves(moves)
    )


class PipelineHoldCells3Evaluator(AbstractPuzzleEvaluator):
    """Accept every exact three-cell solution and verify the full typed replay."""

    VERIFIER_PROFILE = "pipeline_visible_three_lifo_structured_bfs_v2"

    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        started = perf_counter()
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        try:
            is_file = candidate_path.exists() and candidate_path.is_file()
        except OSError:
            is_file = False
        text = candidate_path.read_text(encoding="utf-8") if is_file else str(candidate)
        moves = _parse_moves(text)
        format_valid = _structured_format_valid(text, moves)
        pair = _parse_visible_pair(record)
        source_valid = pair is not None
        arrival = (
            tuple(int(value) for value in pair[0]["arrival"] if value is not None)
            if pair is not None
            else tuple()
        )
        target_order = tuple(pair[0]["target"]) if pair is not None else tuple()
        oracle = _independent_shortest(arrival, target_order, NUM_CELLS) if pair is not None else None
        three_cells_required = (
            pair is not None
            and not _is_solvable(arrival, target_order, 2)
            and _is_solvable(arrival, target_order, 3)
        )
        source_solvable = oracle is not None and three_cells_required
        replay = _replay(pair[0], moves) if pair is not None and moves is not None else None
        legal_count = int(replay["legal_move_count"]) if replay is not None else 0
        no_illegal_move = replay is not None and replay["illegal_move_index"] is None
        optimal_distance = int(oracle[0]) if oracle is not None else None
        shortest_path_count = int(oracle[1]) if oracle is not None else None
        canonical = oracle[2] if oracle is not None else tuple()
        goal_reached = bool(
            pair is not None
            and replay is not None
            and replay["arrival"] == [None] * len(pair[0]["target"])
            and replay["cells"] == [[], [], []]
            and replay["outfeed"] == pair[0]["target"]
        )
        minimum_length = (
            moves is not None
            and optimal_distance is not None
            and len(moves) == optimal_distance
        )
        all_cells_used = replay is not None and replay["used_cells"] == (1, 2, 3)

        binary_correct = int(
            format_valid
            and source_valid
            and source_solvable
            and three_cells_required
            and no_illegal_move
            and goal_reached
            and minimum_length
            and all_cells_used
        )
        if not format_valid:
            failure_reason = "format_error"
        elif not source_valid:
            failure_reason = "invalid_visible_source_or_target"
        elif not source_solvable or not three_cells_required:
            failure_reason = "visible_target_not_load_bearing_three_cell_case"
        elif not no_illegal_move:
            failure_reason = str(replay["illegal_reason"])
        elif not goal_reached:
            failure_reason = "target_outfeed_not_reached"
        elif not minimum_length:
            failure_reason = "non_minimum_action_count"
        elif not all_cells_used:
            failure_reason = "all_three_cells_not_used"
        else:
            failure_reason = "ok"

        return {
            "puzzle_id": puzzle_id,
            "verifier_profile": self.VERIFIER_PROFILE,
            "format_valid": format_valid,
            "source_valid": source_valid,
            "source_solvable": source_solvable,
            "three_cells_required": three_cells_required,
            "moves_parsed": [list(move) for move in moves] if moves is not None else None,
            "result_arrival": replay["arrival"] if replay is not None else None,
            "result_cells": replay["cells"] if replay is not None else None,
            "result_outfeed": replay["outfeed"] if replay is not None else None,
            "used_cells": list(replay["used_cells"]) if replay is not None else None,
            "legal_move_count": legal_count,
            "illegal_move_index": replay["illegal_move_index"] if replay is not None else None,
            "illegal_reason": replay["illegal_reason"] if replay is not None else None,
            "goal_reached": goal_reached,
            "minimum_length": minimum_length,
            "all_cells_used": all_cells_used,
            "optimal_distance": optimal_distance,
            "shortest_path_count": shortest_path_count,
            "canonical_shortest_path_diagnostic": [list(move) for move in canonical],
            "binary_correct": binary_correct,
            "valid": bool(binary_correct),
            "failure_reason": failure_reason,
            "latency_ms": (perf_counter() - started) * 1000.0,
            "performance_is_diagnostic_only": True,
        }


__all__ = [
    'PipelineHoldCells3Evaluator',
    '_independent_shortest',
    '_is_solvable',
    '_parse_moves',
    '_parse_visible_pair',
    '_replay',
]

TASK_NAME = "pipeline_hold_cells_3"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
