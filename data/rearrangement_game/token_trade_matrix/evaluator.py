"""Strict independent evaluator for ``token_trade_matrix``.

A submitted move names two sockets by their printed position numbers, which run
1 through 5.  Holdings, edges and replay index the same sockets 0 through 4, so
the payload is converted once, immediately after parsing.  Anything outside the
published 1..5 domain — an old zero-based endpoint, for instance — fails the
strict grammar or the conversion and is rejected without raising.
"""

from __future__ import annotations

import re
from collections import deque
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from data.common.verifiers.routes import VerifierRoute
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from data.common.base import AbstractPuzzleEvaluator, PathLike
from data.rearrangement_game.core.answer_indexing import from_visible_plan

N_POSITIONS = 5
EDGE_COUNT_RANGE = (5, 7)
MIN_OPTIMAL_MOVES = 3
MAX_OPTIMAL_MOVES = 6

State = Tuple[int, ...]
Edge = Tuple[int, int]
Move = Tuple[int, int]

_PAIR = r"\(([1-5]), ([1-5])\)"
_ANSWER_RE = re.compile(rf"Answer: \[({_PAIR}(?:, {_PAIR})*)\]\.")
_MOVE_RE = re.compile(_PAIR)


def _parse_moves(text: str) -> Optional[List[Move]]:
    """Recover internal 0-based socket pairs from the printed 1-based payload."""

    if not isinstance(text, str) or _ANSWER_RE.fullmatch(text) is None:
        return None
    submitted = [(int(a), int(b)) for a, b in _MOVE_RE.findall(text)]
    if not submitted or any(a == b for a, b in submitted):
        return None
    canonical = "Answer: [" + ", ".join(f"({a}, {b})" for a, b in submitted) + "]."
    if canonical != text:
        return None
    recovered = from_visible_plan(TASK_NAME, submitted)
    return [(int(a), int(b)) for a, b in recovered] if recovered is not None else None


def _parse_holdings(raw: Any) -> Optional[State]:
    if not isinstance(raw, list) or len(raw) != N_POSITIONS:
        return None
    if any(type(value) is not int for value in raw):
        return None
    if sorted(raw) != list(range(1, N_POSITIONS + 1)):
        return None
    return tuple(raw)


def _parse_edges(raw: Any) -> Optional[FrozenSet[Edge]]:
    if not isinstance(raw, list) or not EDGE_COUNT_RANGE[0] <= len(raw) <= EDGE_COUNT_RANGE[1]:
        return None
    parsed: List[Edge] = []
    for edge in raw:
        if not isinstance(edge, list) or len(edge) != 2:
            return None
        a, b = edge
        if type(a) is not int or type(b) is not int:
            return None
        if not (0 <= a < N_POSITIONS and 0 <= b < N_POSITIONS) or a >= b:
            return None
        parsed.append((a, b))
    if len(set(parsed)) != len(parsed):
        return None
    edges = frozenset(parsed)
    adjacency = {index: set() for index in range(N_POSITIONS)}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return edges if len(seen) == N_POSITIONS else None


def _validated_states(record: Dict[str, Any]) -> Optional[Tuple[State, State, FrozenSet[Edge]]]:
    if not isinstance(record, dict):
        return None
    initial_raw = record.get("initial_state")
    target_raw = record.get("target_state")
    if not isinstance(initial_raw, dict) or not isinstance(target_raw, dict):
        return None
    initial = _parse_holdings(initial_raw.get("holdings"))
    target = _parse_holdings(target_raw.get("holdings"))
    edges = _parse_edges(initial_raw.get("edges"))
    target_edges = _parse_edges(target_raw.get("edges"))
    if initial is None or target is None or edges is None or target_edges != edges or initial == target:
        return None
    return initial, target, edges


def _successors(state: State, edges: FrozenSet[Edge]) -> List[State]:
    out: List[State] = []
    for a, b in sorted(edges):
        successor = list(state)
        successor[a], successor[b] = successor[b], successor[a]
        out.append(tuple(successor))
    return out


@lru_cache(maxsize=4096)
def _shortest_stats(initial: State, target: State, edges: FrozenSet[Edge]) -> Tuple[Optional[int], int]:
    if initial == target:
        return 0, 1
    queue = deque([initial])
    distances = {initial: 0}
    path_counts = {initial: 1}
    target_distance: Optional[int] = None
    while queue:
        state = queue.popleft()
        distance = distances[state]
        if target_distance is not None and distance >= target_distance:
            continue
        for successor in _successors(state, edges):
            next_distance = distance + 1
            previous = distances.get(successor)
            if previous is None:
                distances[successor] = next_distance
                path_counts[successor] = path_counts[state]
                queue.append(successor)
            elif previous == next_distance:
                path_counts[successor] += path_counts[state]
            if successor == target:
                target_distance = next_distance
    if target not in distances:
        return None, 0
    return distances[target], path_counts[target]


def _replay(initial: State, moves: List[Move], edges: FrozenSet[Edge]) -> Optional[State]:
    state = initial
    for a, b in moves:
        if (min(a, b), max(a, b)) not in edges:
            return None
        successor = list(state)
        successor[a], successor[b] = successor[b], successor[a]
        state = tuple(successor)
    return state


class TokenTradeMatrixEvaluator(AbstractPuzzleEvaluator):
    def evaluate(self, puzzle_id: str, candidate: PathLike) -> Dict[str, Any]:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate)
        text = (
            candidate_path.read_text(encoding="utf-8")
            if candidate_path.exists() and candidate_path.is_file()
            else str(candidate)
        )
        moves = _parse_moves(text)
        states = _validated_states(record)
        if moves is None or states is None:
            return {
                "puzzle_id": puzzle_id,
                "moves_parsed": moves,
                "format_valid": moves is not None,
                "source_valid": False,
                "valid": False,
                "result_state": None,
            }

        initial, target, edges = states
        restricted_distance, restricted_path_count = _shortest_stats(initial, target, edges)
        complete_edges = frozenset(combinations(range(N_POSITIONS), 2))
        complete_distance, complete_path_count = _shortest_stats(initial, target, complete_edges)
        source_valid = (
            restricted_distance is not None
            and MIN_OPTIMAL_MOVES <= restricted_distance <= MAX_OPTIMAL_MOVES
            and restricted_path_count >= 1
            and complete_distance is not None
            and complete_distance < restricted_distance
        )
        result = _replay(initial, moves, edges) if source_valid else None
        valid = (
            source_valid
            and result == target
            and restricted_distance is not None
            and len(moves) == restricted_distance
        )
        return {
            "puzzle_id": puzzle_id,
            "moves_parsed": moves,
            "format_valid": True,
            "source_valid": source_valid,
            "restricted_distance": restricted_distance,
            "restricted_shortest_path_count": restricted_path_count,
            "complete_graph_distance": complete_distance,
            "complete_graph_shortest_path_count": complete_path_count,
            "valid": valid,
            "result_state": list(result) if result is not None else None,
        }


__all__ = [
    "TokenTradeMatrixEvaluator",
    "_parse_edges",
    "_parse_holdings",
    "_parse_moves",
    "_replay",
    "_shortest_stats",
    "_validated_states",
]

TASK_NAME = "token_trade_matrix"
VERIFIER = VerifierRoute("data.rearrangement_game.core.verifier", "verify_rearrangement")


def evaluate(samples, pred_root: Path, *, modality: str):
    """Evaluate exactly one declared canonical final-delivery modality."""
    return VERIFIER(samples, pred_root, modality=modality)


__all__ = [*__all__, "TASK_NAME", "VERIFIER", "evaluate"]
