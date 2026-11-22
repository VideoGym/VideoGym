"""Independent text replay and image evaluation for sliding vehicles."""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
import re
from data.common.verifiers.routes import VerifierRoute
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from data.common.base import PathLike
from data.common.pipeline.path_tasks import PathTaskEvaluator

BOARD_SIZE = 6
MIN_MOVES = 6
MAX_MOVES = 10
State = Tuple[Tuple[int, int, int], ...]
Meta = Dict[int, Dict[str, Any]]


def _car_cells(position: Tuple[int, int, int], meta: Meta) -> List[Tuple[int, int]]:
    car_id, row, col = position
    orientation = str(meta[car_id]["orientation"])
    length = int(meta[car_id]["length"])
    if orientation == "H":
        return [(row, col + offset) for offset in range(length)]
    return [(row + offset, col) for offset in range(length)]


def _updated(state: State, car_id: int, row: int, col: int) -> State:
    return tuple(
        (cid, row if cid == car_id else old_row, col if cid == car_id else old_col)
        for cid, old_row, old_col in state
    )


def _goal(state: State, red_id: int) -> bool:
    return next(col == BOARD_SIZE for cid, _, col in state if cid == red_id)


def _successors(state: State, meta: Meta, red_id: int) -> List[Tuple[State, Dict[str, Any]]]:
    grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for position in state:
        car_id, _, col = position
        if car_id == red_id and col == BOARD_SIZE:
            continue
        for row, cell_col in _car_cells(position, meta):
            if 0 <= row < BOARD_SIZE and 0 <= cell_col < BOARD_SIZE:
                if grid[row][cell_col] != 0:
                    raise ValueError("overlapping_visible_vehicles")
                grid[row][cell_col] = car_id
    output: List[Tuple[State, Dict[str, Any]]] = []
    for car_id, row, col in state:
        orientation = str(meta[car_id]["orientation"])
        length = int(meta[car_id]["length"])
        if car_id == red_id and col == BOARD_SIZE:
            continue
        if orientation == "H":
            for distance in range(1, BOARD_SIZE + 1):
                next_col = col - distance
                if next_col < 0 or grid[row][next_col] != 0:
                    break
                output.append(
                    (_updated(state, car_id, row, next_col), {"car_id": car_id, "direction": "W", "cells": distance})
                )
            for distance in range(1, BOARD_SIZE + 1):
                next_col = col + distance
                leading_col = col + length - 1 + distance
                if leading_col < BOARD_SIZE and grid[row][leading_col] != 0:
                    break
                if car_id == red_id:
                    if next_col > BOARD_SIZE:
                        break
                    if next_col < BOARD_SIZE and next_col + length > BOARD_SIZE:
                        continue
                elif leading_col >= BOARD_SIZE:
                    break
                output.append(
                    (_updated(state, car_id, row, next_col), {"car_id": car_id, "direction": "E", "cells": distance})
                )
        else:
            for distance in range(1, BOARD_SIZE + 1):
                next_row = row - distance
                if next_row < 0 or grid[next_row][col] != 0:
                    break
                output.append(
                    (_updated(state, car_id, next_row, col), {"car_id": car_id, "direction": "N", "cells": distance})
                )
            for distance in range(1, BOARD_SIZE + 1):
                next_tail = row + length - 1 + distance
                if next_tail >= BOARD_SIZE or grid[next_tail][col] != 0:
                    break
                output.append(
                    (_updated(state, car_id, row + distance, col), {"car_id": car_id, "direction": "S", "cells": distance})
                )
    return output


def _visible_problem(record: Any) -> Optional[Tuple[State, Meta, int, int]]:
    if not isinstance(record, Mapping) or record.get("grid_size") != [6, 6]:
        return None
    raw_cars = record.get("cars")
    if not isinstance(raw_cars, list) or not 4 <= len(raw_cars) <= 6:
        return None
    meta: Meta = {}
    positions: List[Tuple[int, int, int]] = []
    colors: List[str] = []
    try:
        for raw in raw_cars:
            if not isinstance(raw, Mapping):
                return None
            car_id = raw.get("id", raw.get("car_id"))
            row, col, length = raw.get("row"), raw.get("col"), raw.get("length")
            orientation, color = raw.get("orientation"), raw.get("color")
            if (
                type(car_id) is not int
                or type(row) is not int
                or type(col) is not int
                or type(length) is not int
                or orientation not in {"H", "V"}
                or length not in {2, 3}
                or type(color) is not str
                or not color
            ):
                return None
            if car_id in meta:
                return None
            meta[car_id] = {"orientation": orientation, "length": length, "color": color}
            positions.append((car_id, row, col))
            colors.append(color)
    except Exception:
        return None
    if sorted(meta) != list(range(1, len(meta) + 1)) or len(colors) != len(set(colors)):
        return None
    positions.sort()
    if [car_id for car_id, _, _ in sorted(positions, key=lambda item: (item[1], item[2]))] != list(range(1, len(meta) + 1)):
        return None
    red_ids = [car_id for car_id, item in meta.items() if item["color"] == "red"]
    if len(red_ids) != 1:
        return None
    red_id = red_ids[0]
    if meta[red_id]["orientation"] != "H" or meta[red_id]["length"] != 2:
        return None
    state: State = tuple(positions)
    occupied: set[Tuple[int, int]] = set()
    for position in state:
        for cell in _car_cells(position, meta):
            if not (0 <= cell[0] < BOARD_SIZE and 0 <= cell[1] < BOARD_SIZE) or cell in occupied:
                return None
            occupied.add(cell)
    exit_row = next(row for car_id, row, _ in state if car_id == red_id)
    if record.get("exit_row") != exit_row:
        return None
    red_col = next(col for car_id, _, col in state if car_id == red_id)
    red_east = red_col + 2
    if all((exit_row, col) not in occupied for col in range(red_east, BOARD_SIZE)):
        return None
    return state, meta, red_id, exit_row


def _independent_analysis(state: State, meta: Meta, red_id: int) -> Optional[Tuple[int, int, Tuple[Dict[str, Any], ...], int]]:
    queue: deque[State] = deque([state])
    distance = {state: 0}
    path_count = {state: 1}
    parent: Dict[State, Tuple[State, Dict[str, Any]]] = {}
    goal_state: Optional[State] = None
    goal_depth: Optional[int] = None
    while queue:
        current = queue.popleft()
        depth = distance[current]
        if goal_depth is not None and depth >= goal_depth:
            continue
        try:
            successors = _successors(current, meta, red_id)
        except ValueError:
            return None
        for next_state, move in successors:
            next_depth = depth + 1
            if next_state not in distance:
                distance[next_state] = next_depth
                path_count[next_state] = path_count[current]
                parent[next_state] = current, move
                queue.append(next_state)
            elif distance[next_state] == next_depth:
                path_count[next_state] += path_count[current]
            if _goal(next_state, red_id) and goal_depth is None:
                goal_depth, goal_state = next_depth, next_state
    if goal_state is None or goal_depth is None:
        return None
    count = sum(path_count[item] for item, depth in distance.items() if depth == goal_depth and _goal(item, red_id))
    plan: List[Dict[str, Any]] = []
    cursor = goal_state
    while cursor != state:
        previous, move = parent[cursor]
        plan.append(move)
        cursor = previous
    plan.reverse()
    return goal_depth, count, tuple(plan), len(distance)


_POSITIVE = r"[1-9][0-9]*"
_HEAD_MOVE = rf"\({_POSITIVE}, \({_POSITIVE}, {_POSITIVE}\)\)"
_ANSWER_RE = re.compile(rf"Answer: \[{_HEAD_MOVE}(?:, {_HEAD_MOVE})*\]\.\Z")


def _format_head_moves(moves: Sequence[Tuple[int, Tuple[int, int]]]) -> str:
    """Serialize visible ``(car_id, new head (x, y))`` moves canonically."""

    items = [
        f"({int(car_id)}, ({int(head[0])}, {int(head[1])}))"
        for car_id, head in moves
    ]
    return f"Answer: [{', '.join(items)}]."


def _head_moves_from_internal(
    state: State,
    meta: Meta,
    red_id: int,
    moves: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[int, Tuple[int, int]], ...]:
    """Replay internal axis moves and expose only identity plus new head."""

    current = state
    visible: List[Tuple[int, Tuple[int, int]]] = []
    for raw_move in moves:
        move = {
            "car_id": int(raw_move["car_id"]),
            "direction": str(raw_move["direction"]),
            "cells": int(raw_move["cells"]),
        }
        match = next(
            (
                next_state
                for next_state, legal_move in _successors(current, meta, red_id)
                if legal_move == move
            ),
            None,
        )
        if match is None:
            raise ValueError("cannot serialize an illegal rush-hour move")
        car_id = int(move["car_id"])
        _, row, col = next(position for position in match if position[0] == car_id)
        visible.append((car_id, (row + 1, col + 1)))
        current = match
    return tuple(visible)


def _format_answer(
    state: State,
    meta: Meta,
    red_id: int,
    moves: Sequence[Mapping[str, Any]],
) -> str:
    return _format_head_moves(_head_moves_from_internal(state, meta, red_id, moves))


def _parse_head_moves(text: Any) -> Optional[Tuple[Tuple[int, Tuple[int, int]], ...]]:
    if not isinstance(text, str) or not text.splitlines():
        return None
    line = text.splitlines()[-1]
    if _ANSWER_RE.fullmatch(line) is None:
        return None
    try:
        payload = ast.literal_eval(line[len("Answer: ") : -1])
    except (SyntaxError, ValueError, TypeError):
        return None
    if not isinstance(payload, list) or not MIN_MOVES <= len(payload) <= MAX_MOVES:
        return None
    moves: List[Tuple[int, Tuple[int, int]]] = []
    for raw in payload:
        if not isinstance(raw, tuple) or len(raw) != 2:
            return None
        car_id, head = raw
        if type(car_id) is not int or not isinstance(head, tuple) or len(head) != 2:
            return None
        if any(type(value) is not int or value < 1 for value in head):
            return None
        moves.append((car_id, (int(head[0]), int(head[1]))))
    return tuple(moves) if _format_head_moves(moves) == line else None


def _parse_moves(
    text: Any,
    state: State,
    meta: Meta,
    red_id: int,
) -> Optional[Tuple[Dict[str, Any], ...]]:
    """Recover internal axis moves from submitted new-head coordinates."""

    submitted = _parse_head_moves(text)
    if submitted is None:
        return None
    current = state
    moves: List[Dict[str, Any]] = []
    for car_id, visible_head in submitted:
        internal_head = (visible_head[0] - 1, visible_head[1] - 1)
        matches = [
            (next_state, legal_move)
            for next_state, legal_move in _successors(current, meta, red_id)
            if int(legal_move["car_id"]) == car_id
            and next(
                (row, col)
                for cid, row, col in next_state
                if cid == car_id
            )
            == internal_head
        ]
        if len(matches) != 1:
            return None
        current, legal_move = matches[0]
        moves.append(dict(legal_move))
    return tuple(moves)


def _replay(state: State, meta: Meta, red_id: int, moves: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    current = state
    states = [state]
    legal_count = 0
    illegal_index: Optional[int] = None
    illegal_reason: Optional[str] = None
    for index, move in enumerate(moves):
        if _goal(current, red_id):
            illegal_index, illegal_reason = index, "move_after_full_exit"
            break
        match = next(
            (
                next_state
                for next_state, legal_move in _successors(current, meta, red_id)
                if legal_move == move
            ),
            None,
        )
        if match is None:
            illegal_index, illegal_reason = index, "illegal_axis_distance_boundary_or_collision"
            break
        current = match
        states.append(current)
        legal_count += 1
    return {
        "state": current,
        "states": tuple(states),
        "legal_move_count": legal_count,
        "illegal_move_index": illegal_index,
        "illegal_reason": illegal_reason,
    }


def _format_valid(
    text: Any,
    state: State,
    meta: Meta,
    red_id: int,
    moves: Optional[Sequence[Dict[str, Any]]],
) -> bool:
    return bool(
        isinstance(text, str)
        and moves is not None
        and text.splitlines()
        and text.splitlines()[-1] == _format_answer(state, meta, red_id, moves)
    )


class MazeRushHourEvaluator(PathTaskEvaluator):
    """Text verifier uses only visible vehicle geometry and duplicated rules."""

    RED_THRESHOLD = 150
    RED_DOMINANCE = 40
    VERIFIER_PROFILE = "fixed_axis_vehicle_head_sequence_independent_bfs_v2"

    def _build_generator(self, record: Dict[str, Any]):
        # The inherited image evaluator needs a renderer hydrated from the
        # source record. Text correctness below never calls generator logic.
        from .generator import MazeRushHourGenerator

        canvas_dims = record.get("canvas_dimensions")
        width = int(canvas_dims[0]) if isinstance(canvas_dims, (list, tuple)) and len(canvas_dims) >= 2 else 512
        cell_size = int(record.get("cell_size", MazeRushHourGenerator.DEFAULT_CELL_SIZE))
        generator = MazeRushHourGenerator(
            output_dir=self.base_dir,
            canvas_width=width,
            size=cell_size,
            grid_size=BOARD_SIZE,
            cell_size=cell_size,
            ti2v_prompt=str(record.get("ti2v_prompt") or ""),
            show_cell_id=False,
            video=False,
        )
        generator.rehydrate_from_record(record)
        return generator

    def evaluate_text(self, record_or_id: Any, candidate: PathLike) -> Dict[str, Any]:
        started = perf_counter()
        record = record_or_id if isinstance(record_or_id, Mapping) else self.get_record(str(record_or_id))
        candidate_path = Path(candidate)
        try:
            is_file = candidate_path.exists() and candidate_path.is_file()
        except OSError:
            is_file = False
        text = candidate_path.read_text(encoding="utf-8") if is_file else str(candidate)
        problem = _visible_problem(record)
        moves = (
            _parse_moves(text, problem[0], problem[1], problem[2])
            if problem is not None
            else None
        )
        format_valid = bool(
            problem is not None
            and _format_valid(text, problem[0], problem[1], problem[2], moves)
        )
        analysis = _independent_analysis(problem[0], problem[1], problem[2]) if problem is not None else None
        source_valid = problem is not None
        source_solvable = bool(analysis is not None and MIN_MOVES <= analysis[0] <= MAX_MOVES)
        replay = _replay(problem[0], problem[1], problem[2], moves) if problem is not None and moves is not None else None
        optimal_distance = int(analysis[0]) if analysis is not None else None
        shortest_path_count = int(analysis[1]) if analysis is not None else None
        canonical = analysis[2] if analysis is not None else tuple()
        explored_states = int(analysis[3]) if analysis is not None else None
        legal_count = int(replay["legal_move_count"]) if replay is not None else 0
        no_illegal = replay is not None and replay["illegal_move_index"] is None
        reached_exit = bool(replay is not None and problem is not None and _goal(replay["state"], problem[2]))
        minimum_length = bool(moves is not None and optimal_distance is not None and len(moves) == optimal_distance)
        binary_correct = int(format_valid and source_solvable and no_illegal and reached_exit and minimum_length)
        if not format_valid:
            failure_reason = "format_error"
        elif not source_valid:
            failure_reason = "invalid_visible_vehicle_state"
        elif not source_solvable:
            failure_reason = "visible_state_outside_published_contract"
        elif not no_illegal:
            failure_reason = str(replay["illegal_reason"])
        elif not reached_exit:
            failure_reason = "red_vehicle_not_fully_exited"
        elif not minimum_length:
            failure_reason = "non_minimum_plan"
        else:
            failure_reason = "ok"
        return {
            "verifier_profile": self.VERIFIER_PROFILE,
            "format_valid": format_valid,
            "source_valid": source_valid,
            "source_solvable": source_solvable,
            "moves_parsed": list(moves) if moves is not None else None,
            "legal_move_count": legal_count,
            "illegal_move_index": replay["illegal_move_index"] if replay is not None else None,
            "illegal_reason": replay["illegal_reason"] if replay is not None else None,
            "red_vehicle_fully_exited": reached_exit,
            "minimum_length": minimum_length,
            "optimal_distance": optimal_distance,
            "shortest_path_count": shortest_path_count,
            "explored_state_count": explored_states,
            "canonical_shortest_path_diagnostic": list(canonical),
            "saved_oracle_fields_used": [],
            "binary_correct": binary_correct,
            "valid": bool(binary_correct),
            "failure_reason": failure_reason,
            "latency_ms": (perf_counter() - started) * 1000.0,
            "performance_is_diagnostic_only": True,
        }


__all__ = [
    'MazeRushHourEvaluator',
    'PathEvaluationResult',
    '_format_answer',
    '_format_head_moves',
    '_format_valid',
    '_head_moves_from_internal',
    '_independent_analysis',
    '_parse_head_moves',
    '_parse_moves',
    '_replay',
    '_visible_problem',
]

TASK_NAME = "maze_rush_hour"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
