"""Strict source/target and structured-reasoning evaluator for frog moves.

A submitted move names the pad the frog occupies, counted from 1 at the left.
The BFS, the replay and the visible states below index pads from 0, so the
submitted payload is converted once, immediately after parsing; an ordinal
outside the published one-based domain rejects the whole plan instead of
raising.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.core.answer_indexing import from_visible_plan

TASK_NAME = "jumping_frog_swap"
GROUP_COUNT_PAIRS = {(2, 2), (2, 3), (3, 2)}
MIN_OPTIMAL_MOVES = 5
MAX_OPTIMAL_MOVES = 10
State = Tuple[str, ...]

_ANSWER_RE = re.compile(r"Answer: \[(?:\d+(?:, \d+)*)\]\.\Z")


def _format_moves(moves: Sequence[int]) -> str:
    return "Answer: [" + ", ".join(str(int(move)) for move in moves) + "]."


def _parse_answer_line(text: Any) -> Optional[Tuple[int, ...]]:
    if not isinstance(text, str) or _ANSWER_RE.fullmatch(text) is None:
        return None
    payload = text[len("Answer: [") : -2]
    moves = tuple(int(part) for part in payload.split(", "))
    return moves if moves and _format_moves(moves) == text else None


def _parse_moves(text: Any) -> Optional[Tuple[int, ...]]:
    """Read the submitted pad numbers exactly as written, counted from 1."""

    if not isinstance(text, str):
        return None
    return _parse_answer_line(text.splitlines()[-1] if text.splitlines() else text)


def _internal_moves(submitted: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    """Recover 0-based source pads, or ``None`` for an out-of-domain ordinal."""

    if submitted is None:
        return None
    recovered = from_visible_plan(TASK_NAME, list(submitted))
    return None if recovered is None else tuple(int(move) for move in recovered)


def _validate_visible_state(raw: Any) -> Optional[State]:
    if not isinstance(raw, list) or not 5 <= len(raw) <= 6:
        return None
    state = tuple(raw)
    if any(type(value) is not str or value not in {"G", "O", "."} for value in state):
        return None
    if state.count(".") != 1 or (state.count("G"), state.count("O")) not in GROUP_COUNT_PAIRS:
        return None
    return state


def _parse_visible_pair(record: Any) -> Optional[Tuple[State, State]]:
    if not isinstance(record, dict):
        return None
    source = _validate_visible_state(record.get("initial_state"))
    target = _validate_visible_state(record.get("target_state"))
    if source is None or target is None:
        return None
    if (source.count("G"), source.count("O")) != (target.count("G"), target.count("O")):
        return None
    return source, target


def _apply_without_validation(state: State, source: int) -> State:
    current = list(state)
    empty = current.index(".")
    current[empty] = current[source]
    current[source] = "."
    return tuple(current)


def _successors(state: State) -> Tuple[Tuple[State, int], ...]:
    empty = state.index(".")
    out: List[Tuple[State, int]] = []
    if empty - 1 >= 0 and state[empty - 1] == "G":
        out.append((_apply_without_validation(state, empty - 1), empty - 1))
    if empty - 2 >= 0 and state[empty - 2] == "G" and state[empty - 1] == "O":
        out.append((_apply_without_validation(state, empty - 2), empty - 2))
    if empty + 1 < len(state) and state[empty + 1] == "O":
        out.append((_apply_without_validation(state, empty + 1), empty + 1))
    if empty + 2 < len(state) and state[empty + 2] == "O" and state[empty + 1] == "G":
        out.append((_apply_without_validation(state, empty + 2), empty + 2))
    return tuple(out)


def _independent_shortest(source: State, target: State) -> Optional[Tuple[int, int, Tuple[int, ...]]]:
    if source == target:
        return 0, 1, tuple()
    queue = deque([source])
    distance = {source: 0}
    path_count = {source: 1}
    parent: Dict[State, Tuple[State, int]] = {}
    goal_distance: Optional[int] = None
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
            if next_state == target:
                goal_distance = next_depth if goal_distance is None else min(goal_distance, next_depth)
    if target not in distance:
        return None
    path: List[int] = []
    cursor = target
    while cursor != source:
        previous, move = parent[cursor]
        path.append(move)
        cursor = previous
    path.reverse()
    return distance[target], path_count[target], tuple(path)


def _replay(source: State, moves: Sequence[int]) -> Dict[str, Any]:
    state = source
    states = [source]
    legal_count = 0
    illegal_index: Optional[int] = None
    illegal_reason: Optional[str] = None
    leap_count = 0
    for move_index, raw_source in enumerate(moves):
        if type(raw_source) is not int:
            illegal_index, illegal_reason = move_index, "source_index_not_integer"
            break
        selected = int(raw_source)
        if not 0 <= selected < len(state):
            illegal_index, illegal_reason = move_index, "source_index_out_of_range"
            break
        species = state[selected]
        if species not in {"G", "O"}:
            illegal_index, illegal_reason = move_index, "source_is_empty"
            break
        empty = state.index(".")
        delta = empty - selected
        if species == "G" and delta not in (1, 2):
            illegal_index, illegal_reason = move_index, "green_wrong_direction_or_distance"
            break
        if species == "O" and delta not in (-1, -2):
            illegal_index, illegal_reason = move_index, "orange_wrong_direction_or_distance"
            break
        if abs(delta) == 2:
            middle = state[(selected + empty) // 2]
            required = "O" if species == "G" else "G"
            if middle != required:
                illegal_index, illegal_reason = move_index, "leap_does_not_cross_opposite_color"
                break
            leap_count += 1
        state = _apply_without_validation(state, selected)
        states.append(state)
        legal_count += 1
    return {
        "state": state,
        "states": tuple(states),
        "legal_move_count": legal_count,
        "leap_count": leap_count,
        "illegal_move_index": illegal_index,
        "illegal_reason": illegal_reason,
    }


def _structured_format_valid(text: Any, moves: Optional[Tuple[int, ...]]) -> bool:
    return bool(
        isinstance(text, str)
        and moves is not None
        and text.splitlines()
        and text.splitlines()[-1] == _format_moves(moves)
    )


class JumpingFrogSwapEvaluator(AbstractPuzzleEvaluator):
    """Verify structured replay and accept every legal globally shortest plan."""

    VERIFIER_PROFILE = "jumping_frog_visible_target_structured_bfs_v3"

    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        started = perf_counter()
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        try:
            is_file = candidate_path.exists() and candidate_path.is_file()
        except OSError:
            is_file = False
        text = candidate_path.read_text(encoding="utf-8") if is_file else str(candidate)
        submitted = _parse_moves(text)
        format_valid = _structured_format_valid(text, submitted)
        # Typed syntax is checked against what the model wrote; everything
        # below replays 0-based pads, so convert once and fail closed on an
        # ordinal outside the published one-based domain.
        moves = _internal_moves(submitted)
        index_domain_valid = submitted is not None and moves is not None
        pair = _parse_visible_pair(record)
        source_valid = pair is not None
        oracle = _independent_shortest(*pair) if pair is not None else None
        source_solvable = oracle is not None and MIN_OPTIMAL_MOVES <= oracle[0] <= MAX_OPTIMAL_MOVES
        replay = _replay(pair[0], moves) if pair is not None and moves is not None else None
        result_state = replay["state"] if replay is not None else None
        no_illegal_move = replay is not None and replay["illegal_move_index"] is None
        legal_count = int(replay["legal_move_count"]) if replay is not None else 0
        leap_count = int(replay["leap_count"]) if replay is not None else 0
        optimal_distance = int(oracle[0]) if oracle is not None else None
        shortest_path_count = int(oracle[1]) if oracle is not None else None
        canonical = oracle[2] if oracle is not None else tuple()
        goal_reached = result_state == pair[1] if result_state is not None and pair is not None else False
        minimum_length = moves is not None and optimal_distance is not None and len(moves) == optimal_distance

        binary_correct = int(
            format_valid
            and index_domain_valid
            and source_valid
            and source_solvable
            and no_illegal_move
            and goal_reached
            and minimum_length
        )
        if not format_valid:
            failure_reason = "format_error"
        elif not index_domain_valid:
            failure_reason = "pad_number_out_of_one_based_domain"
        elif not source_valid:
            failure_reason = "invalid_visible_source_or_target"
        elif not source_solvable:
            failure_reason = "visible_target_outside_published_contract"
        elif not no_illegal_move:
            failure_reason = str(replay["illegal_reason"])
        elif not goal_reached:
            failure_reason = "goal_not_reached"
        elif not minimum_length:
            failure_reason = "non_minimum_plan"
        else:
            failure_reason = "ok"

        return {
            "puzzle_id": puzzle_id,
            "verifier_profile": self.VERIFIER_PROFILE,
            "format_valid": format_valid,
            "answer_index_domain_valid": index_domain_valid,
            "source_valid": source_valid,
            "source_solvable": source_solvable,
            "submitted_pad_numbers": list(submitted) if submitted is not None else None,
            "moves_parsed": list(moves) if moves is not None else None,
            "result_state": list(result_state) if result_state is not None else None,
            "legal_move_count": legal_count,
            "leap_count": leap_count,
            "illegal_move_index": replay["illegal_move_index"] if replay is not None else None,
            "illegal_reason": replay["illegal_reason"] if replay is not None else None,
            "goal_reached": goal_reached,
            "minimum_length": minimum_length,
            "optimal_distance": optimal_distance,
            "shortest_path_count": shortest_path_count,
            "canonical_shortest_path_diagnostic": list(canonical),
            "binary_correct": binary_correct,
            "valid": bool(binary_correct),
            "failure_reason": failure_reason,
            "latency_ms": (perf_counter() - started) * 1000.0,
            "performance_is_diagnostic_only": True,
        }


__all__ = [
    'JumpingFrogSwapEvaluator',
    '_format_moves',
    '_independent_shortest',
    '_internal_moves',
    '_parse_moves',
    '_parse_visible_pair',
    '_replay',
    '_structured_format_valid',
    '_successors',
]

VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
