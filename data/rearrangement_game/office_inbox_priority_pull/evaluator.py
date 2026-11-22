"""Strict source-replay evaluator for ``office_inbox_priority_pull``."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike

PAPER_COUNT_RANGE = (4, 5)
Move = Tuple[str, int]
StateKey = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]

_MOVE_TEXT = r'\("(?:work|buffer)", [1-5]\)'
_ANSWER_RE = re.compile(rf"Answer: \[((?:{_MOVE_TEXT})(?:, {_MOVE_TEXT})*)\]\.\Z")
_MOVE_RE = re.compile(r'\("(work|buffer)", ([1-5])\)')


def _format_moves(moves: Sequence[Move]) -> str:
    return "Answer: [" + ", ".join(f'(\"{action}\", {paper})' for action, paper in moves) + "]."


def _parse_moves(text: Any) -> Optional[Tuple[Move, ...]]:
    if not isinstance(text, str) or _ANSWER_RE.fullmatch(text) is None:
        return None
    moves = tuple((action, int(paper)) for action, paper in _MOVE_RE.findall(text))
    return moves if moves and _format_moves(moves) == text else None


def _parse_initial_inbox(record: Any) -> Optional[Tuple[int, ...]]:
    if not isinstance(record, dict):
        return None
    initial = record.get("initial_state")
    if (
        not isinstance(initial, dict)
        or set(initial) != {"inbox", "work", "buffer"}
        or initial.get("work") != []
        or initial.get("buffer") != []
    ):
        return None
    inbox = initial.get("inbox")
    if not isinstance(inbox, list) or not PAPER_COUNT_RANGE[0] <= len(inbox) <= PAPER_COUNT_RANGE[1]:
        return None
    if any(type(value) is not int for value in inbox):
        return None
    parsed = tuple(int(value) for value in inbox)
    if sorted(parsed) != list(range(1, len(parsed) + 1)):
        return None
    return parsed


def _successors(state: StateKey) -> Tuple[Tuple[StateKey, Move], ...]:
    inbox, work, buffer = state
    successors: List[Tuple[StateKey, Move]] = []
    if inbox:
        paper = inbox[-1]
        if not work or work[-1] > paper:
            successors.append(((inbox[:-1], work + (paper,), buffer), ("work", paper)))
        successors.append(((inbox[:-1], work, buffer + (paper,)), ("buffer", paper)))
    if buffer:
        paper = buffer[-1]
        if not work or work[-1] > paper:
            successors.append(((inbox, work + (paper,), buffer[:-1]), ("work", paper)))
    return tuple(successors)


def _independent_shortest(inbox: Sequence[int]) -> Optional[Tuple[int, int, Tuple[Move, ...]]]:
    papers = tuple(inbox)
    goal: StateKey = ((), tuple(range(len(papers), 0, -1)), ())
    initial: StateKey = (papers, (), ())
    queue = deque([initial])
    distance = {initial: 0}
    path_count = {initial: 1}
    parent: Dict[StateKey, Tuple[StateKey, Move]] = {}
    goal_distance: Optional[int] = None
    while queue:
        state = queue.popleft()
        depth = distance[state]
        if goal_distance is not None and depth >= goal_distance:
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
    while cursor != initial:
        previous, move = parent[cursor]
        path.append(move)
        cursor = previous
    path.reverse()
    return distance[goal], path_count[goal], tuple(path)


def _replay(inbox: Sequence[int], moves: Sequence[Move]) -> Dict[str, Any]:
    state_inbox = list(inbox)
    work: List[int] = []
    buffer: List[int] = []
    legal_count = 0
    illegal_index: Optional[int] = None
    illegal_reason: Optional[str] = None
    for index, (action, paper) in enumerate(moves):
        if not 1 <= paper <= len(inbox):
            illegal_index, illegal_reason = index, "paper_id_out_of_range"
            break
        if action == "buffer":
            if not state_inbox or state_inbox[-1] != paper:
                illegal_index, illegal_reason = index, "trash_source_is_not_inbox_top"
                break
            state_inbox.pop()
            buffer.append(paper)
        elif action == "work":
            if work and work[-1] <= paper:
                illegal_index, illegal_reason = index, "work_order_violation"
                break
            if state_inbox and state_inbox[-1] == paper:
                state_inbox.pop()
            elif buffer and buffer[-1] == paper:
                buffer.pop()
            else:
                illegal_index, illegal_reason = index, "work_source_is_not_accessible_top"
                break
            work.append(paper)
        else:
            illegal_index, illegal_reason = index, "unknown_action"
            break
        legal_count += 1
    return {
        "state": {"inbox": state_inbox, "work": work, "buffer": buffer},
        "legal_move_count": legal_count,
        "illegal_move_index": illegal_index,
        "illegal_reason": illegal_reason,
    }


class OfficeInboxPriorityPullEvaluator(AbstractPuzzleEvaluator):
    """Recompute the legal state graph from visible source data only."""

    VERIFIER_PROFILE = "office_inbox_priority_pull_source_bfs_v2"

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
        inbox = _parse_initial_inbox(record)
        source_valid = inbox is not None
        oracle = _independent_shortest(inbox) if inbox is not None else None
        source_solvable = oracle is not None

        replay = _replay(inbox, moves) if inbox is not None and moves is not None else None
        state = replay["state"] if replay is not None else None
        legal_count = int(replay["legal_move_count"]) if replay is not None else 0
        no_illegal_move = replay is not None and replay["illegal_move_index"] is None
        optimal_distance = int(oracle[0]) if oracle is not None else None
        shortest_path_count = int(oracle[1]) if oracle is not None else None
        canonical_path = oracle[2] if oracle is not None else ()
        goal = (
            {"inbox": [], "work": list(range(len(inbox), 0, -1)), "buffer": []}
            if inbox is not None
            else None
        )
        goal_reached = state == goal if state is not None and goal is not None else False
        minimum_length = (
            moves is not None and optimal_distance is not None and len(moves) == optimal_distance
        )

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
            failure_reason = "unsolvable_visible_source"
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
            "moves_parsed": [[action, paper] for action, paper in moves] if moves is not None else None,
            "result_state": state,
            "legal_move_count": legal_count,
            "illegal_move_index": replay["illegal_move_index"] if replay is not None else None,
            "illegal_reason": replay["illegal_reason"] if replay is not None else None,
            "goal_reached": goal_reached,
            "minimum_length": minimum_length,
            "optimal_distance": optimal_distance,
            "shortest_path_count": shortest_path_count,
            "canonical_shortest_path_diagnostic": [[action, paper] for action, paper in canonical_path],
            "binary_correct": binary_correct,
            "valid": bool(binary_correct),
            "failure_reason": failure_reason,
            "latency_ms": latency_ms,
            "performance_is_diagnostic_only": True,
        }


__all__ = [
    "OfficeInboxPriorityPullEvaluator",
    "_format_moves",
    "_independent_shortest",
    "_parse_initial_inbox",
    "_parse_moves",
    "_replay",
]

TASK_NAME = "office_inbox_priority_pull"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
