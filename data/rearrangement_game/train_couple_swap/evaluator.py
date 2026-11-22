"""Strict independent source-replay evaluator for ``train_couple_swap``.

``("park", car_id)`` names the identity printed on a railcar and is read as
submitted.  ``("place", slot)`` names an insertion boundary: a submitted plan
counts boundaries from 1, while the replay and the independent BFS below count
them from 0.  That single conversion happens where the payload is parsed, and a
slot outside the published one-based domain — an old zero-based ``0``, for
instance — rejects the plan instead of raising.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.core.answer_indexing import from_visible_plan, to_visible_move

CAR_COUNT_RANGE = (4, 6)
MAX_OPTIMAL_MOVES = 6
Move = Tuple[str, int]
StateKey = Tuple[Tuple[int, ...], Optional[int]]

_MOVE_TEXT = r'\("(?:park|place)", \d+\)'
_ANSWER_RE = re.compile(rf"Answer: \[((?:{_MOVE_TEXT})(?:, {_MOVE_TEXT})*)\]\.\Z")
_MOVE_RE = re.compile(r'\("(park|place)", (\d+)\)')


def _format_moves(moves: Sequence[Move]) -> str:
    """Serialize internal moves with the visible one-based insertion slots."""

    visible = [to_visible_move(TASK_NAME, move) for move in moves]
    return "Answer: [" + ", ".join(f'(\"{action}\", {value})' for action, value in visible) + "]."


def _parse_moves(text: Any) -> Optional[Tuple[Move, ...]]:
    """Recover internal moves from one submitted one-based payload."""

    if not isinstance(text, str) or _ANSWER_RE.fullmatch(text) is None:
        return None
    submitted = tuple((action, int(value)) for action, value in _MOVE_RE.findall(text))
    if not submitted:
        return None
    recovered = from_visible_plan(TASK_NAME, submitted)
    if recovered is None:
        return None
    moves = tuple((str(action), int(value)) for action, value in recovered)
    return moves if _format_moves(moves) == text else None


def _parse_source_tracks(record: Any) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    if not isinstance(record, dict):
        return None
    initial = record.get("initial_state")
    target = record.get("target_state")
    if (
        not isinstance(initial, dict)
        or not isinstance(target, dict)
        or set(initial) != {"track", "parked"}
        or set(target) != {"track", "parked"}
        or initial.get("parked") is not None
        or target.get("parked") is not None
    ):
        return None
    initial_track = initial.get("track")
    target_track = target.get("track")
    if (
        not isinstance(initial_track, list)
        or not isinstance(target_track, list)
        or not CAR_COUNT_RANGE[0] <= len(initial_track) <= CAR_COUNT_RANGE[1]
        or len(target_track) != len(initial_track)
        or any(type(value) is not int for value in initial_track + target_track)
    ):
        return None
    expected = list(range(1, len(initial_track) + 1))
    source = tuple(int(value) for value in initial_track)
    target_order = tuple(int(value) for value in target_track)
    if sorted(source) != expected or sorted(target_order) != expected:
        return None
    return source, target_order


def _successors(state: StateKey) -> Tuple[Tuple[StateKey, Move], ...]:
    track, parked = state
    successors: List[Tuple[StateKey, Move]] = []
    if parked is None:
        for index, car in enumerate(track):
            successors.append(((track[:index] + track[index + 1 :], car), ("park", car)))
    else:
        for slot in range(len(track) + 1):
            successors.append(((track[:slot] + (parked,) + track[slot:], None), ("place", slot)))
    return tuple(successors)


def _independent_shortest(
    initial_track: Sequence[int],
    target_track: Sequence[int],
) -> Optional[Tuple[int, int, Tuple[Move, ...]]]:
    start: StateKey = (tuple(initial_track), None)
    goal: StateKey = (tuple(target_track), None)
    queue = deque([start])
    distance = {start: 0}
    path_count = {start: 1}
    parent: Dict[StateKey, Tuple[StateKey, Move]] = {}
    goal_distance: Optional[int] = 0 if start == goal else None
    while queue:
        state = queue.popleft()
        depth = distance[state]
        if goal_distance is not None and depth >= goal_distance:
            continue
        if depth >= MAX_OPTIMAL_MOVES:
            continue
        for next_state, move in _successors(state):
            next_depth = depth + 1
            if next_state not in distance:
                distance[next_state] = next_depth
                path_count[next_state] = path_count[state]
                parent[next_state] = state, move
                queue.append(next_state)
            elif distance[next_state] == next_depth:
                path_count[next_state] += path_count[state]
            if next_state == goal:
                goal_distance = next_depth if goal_distance is None else min(goal_distance, next_depth)
    if goal not in distance:
        return None
    path: List[Move] = []
    cursor = goal
    while cursor != start:
        previous, move = parent[cursor]
        path.append(move)
        cursor = previous
    path.reverse()
    return distance[goal], path_count[goal], tuple(path)


def _replay(initial_track: Sequence[int], moves: Sequence[Move]) -> Dict[str, Any]:
    track = list(initial_track)
    parked: Optional[int] = None
    legal_count = 0
    illegal_index: Optional[int] = None
    illegal_reason: Optional[str] = None
    for index, (action, value) in enumerate(moves):
        if action == "park":
            if parked is not None:
                illegal_index, illegal_reason = index, "siding_already_occupied"
                break
            if value not in track:
                illegal_index, illegal_reason = index, "park_car_not_on_main_track"
                break
            track.remove(value)
            parked = value
        elif action == "place":
            if parked is None:
                illegal_index, illegal_reason = index, "siding_is_empty"
                break
            if not 0 <= value <= len(track):
                illegal_index, illegal_reason = index, "place_slot_out_of_range"
                break
            track = track[:value] + [parked] + track[value:]
            parked = None
        else:
            illegal_index, illegal_reason = index, "unknown_action"
            break
        legal_count += 1
    return {
        "state": {"track": track, "parked": parked},
        "legal_move_count": legal_count,
        "illegal_move_index": illegal_index,
        "illegal_reason": illegal_reason,
    }


class TrainCoupleSwapEvaluator(AbstractPuzzleEvaluator):
    """Accept every legal shortest plan recomputed from the visible two-panel source."""

    VERIFIER_PROFILE = "train_couple_swap_visible_target_bfs_v2"

    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        started = perf_counter()
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = (
            candidate_path.read_text(encoding="utf-8")
            if candidate_path.exists() and candidate_path.is_file()
            else str(candidate)
        )
        moves = _parse_moves(text)
        format_valid = moves is not None
        tracks = _parse_source_tracks(record)
        source_valid = tracks is not None
        oracle = _independent_shortest(*tracks) if tracks is not None else None
        source_solvable = oracle is not None
        replay = _replay(tracks[0], moves) if tracks is not None and moves is not None else None
        state = replay["state"] if replay is not None else None
        legal_count = int(replay["legal_move_count"]) if replay is not None else 0
        no_illegal_move = replay is not None and replay["illegal_move_index"] is None
        optimal_distance = int(oracle[0]) if oracle is not None else None
        shortest_path_count = int(oracle[1]) if oracle is not None else None
        canonical = oracle[2] if oracle is not None else ()
        goal = {"track": list(tracks[1]), "parked": None} if tracks is not None else None
        goal_reached = state == goal if state is not None and goal is not None else False
        minimum_length = moves is not None and optimal_distance is not None and len(moves) == optimal_distance

        binary_correct = int(
            format_valid
            and source_valid
            and source_solvable
            and no_illegal_move
            and goal_reached
            and minimum_length
        )
        if not format_valid:
            failure_reason = "format_error"
        elif not source_valid:
            failure_reason = "invalid_visible_source"
        elif not source_solvable:
            failure_reason = "unsolved_visible_target_within_contract"
        elif not no_illegal_move:
            failure_reason = str(replay["illegal_reason"])
        elif not goal_reached:
            failure_reason = "goal_not_reached"
        elif not minimum_length:
            failure_reason = "non_minimum_plan"
        else:
            failure_reason = "ok"

        latency_ms = (perf_counter() - started) * 1000.0
        return {
            "puzzle_id": puzzle_id,
            "verifier_profile": self.VERIFIER_PROFILE,
            "format_valid": format_valid,
            "source_valid": source_valid,
            "source_solvable": source_solvable,
            "moves_parsed": [[action, value] for action, value in moves] if moves is not None else None,
            "result_state": state,
            "legal_move_count": legal_count,
            "illegal_move_index": replay["illegal_move_index"] if replay is not None else None,
            "illegal_reason": replay["illegal_reason"] if replay is not None else None,
            "goal_reached": goal_reached,
            "minimum_length": minimum_length,
            "optimal_distance": optimal_distance,
            "shortest_path_count": shortest_path_count,
            "canonical_shortest_path_diagnostic": [[action, value] for action, value in canonical],
            "binary_correct": binary_correct,
            "valid": bool(binary_correct),
            "failure_reason": failure_reason,
            "latency_ms": latency_ms,
            "performance_is_diagnostic_only": True,
        }


__all__ = [
    "TrainCoupleSwapEvaluator",
    "_format_moves",
    "_independent_shortest",
    "_parse_moves",
    "_parse_source_tracks",
    "_replay",
]

TASK_NAME = "train_couple_swap"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
