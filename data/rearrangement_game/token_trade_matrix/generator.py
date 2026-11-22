"""Shortest token redistribution over a visible restricted trade graph.

Five fixed positions are arranged on a ring.  Each position holds exactly one
uniquely numbered token, and a visible undirected edge means that the two
endpoint positions may trade their current tokens.  One move swaps both tokens
simultaneously.  Admitted instances use connected, non-complete graphs whose
restricted optimum is strictly longer than the complete-graph optimum, so the
visible edge restriction is load-bearing rather than decorative.

Two independent numberings live here.  A *token identity* is the digit inside a
disc and runs 1..5 everywhere, image and payload alike.  A *position* is a
socket on the ring: the printed socket digit and the formal ``Answer:`` payload
count positions 1..5 clockwise from the top, while holdings, edges, solver
states and renderer geometry index the same sockets 0..4.  The two systems meet
only in ``data.rearrangement_game.core.answer_indexing``.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

import math
from collections import deque
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.pipeline.object_plans import ITEM_PALETTE
from data.common.video_encoding import pad_frames_to_wan_compatible_length
from data.rearrangement_game.core.answer_indexing import to_visible_plan

N_POSITIONS = 5
EDGE_COUNT_RANGE = (5, 7)
MIN_OPTIMAL_MOVES = 3
MAX_OPTIMAL_MOVES = 6
MOVE_FRAMES = 8

State = Tuple[int, ...]
Edge = Tuple[int, int]
Move = Tuple[int, int]

PANEL_BG = (248, 248, 246)
EDGE_COLOR = (80, 151, 139)
SOCKET_FILL = (236, 239, 239)
SOCKET_OUTLINE = (177, 184, 184)
POSITION_TEXT = (70, 76, 78)


def _normalize_edges(raw_edges: Any, *, require_lists: bool = False) -> FrozenSet[Edge]:
    if not isinstance(raw_edges, list) or not EDGE_COUNT_RANGE[0] <= len(raw_edges) <= EDGE_COUNT_RANGE[1]:
        raise ValueError("edges must be a list containing five to seven pairs")
    parsed: List[Edge] = []
    for edge in raw_edges:
        if require_lists and not isinstance(edge, list):
            raise ValueError("serialized edges must be lists")
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("every edge must be a two-position pair")
        a, b = edge
        if type(a) is not int or type(b) is not int:
            raise ValueError("edge endpoints must be integers")
        if not (0 <= a < N_POSITIONS and 0 <= b < N_POSITIONS) or a >= b:
            raise ValueError("edges must be normalized distinct pairs in the range 0..4")
        parsed.append((a, b))
    if len(set(parsed)) != len(parsed):
        raise ValueError("edges must be unique")
    edges = frozenset(parsed)
    if len(edges) == len(tuple(combinations(range(N_POSITIONS), 2))):
        raise ValueError("the visible graph must be non-complete")
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
    if len(seen) != N_POSITIONS:
        raise ValueError("the visible graph must connect all five positions")
    return edges


def _validate_holdings(raw: Any) -> State:
    if not isinstance(raw, list) or len(raw) != N_POSITIONS:
        raise ValueError("holdings must be a five-item list")
    if any(type(value) is not int for value in raw):
        raise ValueError("token identities must be integers")
    if sorted(raw) != list(range(1, N_POSITIONS + 1)):
        raise ValueError("holdings must contain each token identity 1..5 exactly once")
    return tuple(raw)


def _validate_state(raw: Any) -> Tuple[State, FrozenSet[Edge]]:
    if not isinstance(raw, dict):
        raise ValueError("state must be a dictionary")
    holdings = _validate_holdings(raw.get("holdings"))
    edges = _normalize_edges(raw.get("edges"))
    return holdings, edges


def _validate_move(move: Any, edges: FrozenSet[Edge]) -> Move:
    if not isinstance(move, (list, tuple)) or len(move) != 2:
        raise ValueError("move must be a two-position pair")
    a, b = move
    if type(a) is not int or type(b) is not int:
        raise ValueError("move endpoints must be integers")
    if not (0 <= a < N_POSITIONS and 0 <= b < N_POSITIONS) or a == b:
        raise ValueError("move endpoints must be distinct positions in the range 0..4")
    edge = (min(a, b), max(a, b))
    if edge not in edges:
        raise ValueError("the chosen pair is not joined by a visible allowed edge")
    return a, b


def _successors(state: State, edges: FrozenSet[Edge]) -> List[Tuple[State, Move]]:
    out: List[Tuple[State, Move]] = []
    for a, b in sorted(edges):
        successor = list(state)
        successor[a], successor[b] = successor[b], successor[a]
        out.append((tuple(successor), (a, b)))
    return out


@lru_cache(maxsize=4096)
def _shortest_stats(
    initial: State,
    target: State,
    edges: FrozenSet[Edge],
) -> Tuple[Optional[int], int, Optional[Tuple[Move, ...]]]:
    """Return exact distance, shortest-path count, and one shortest plan."""

    if initial == target:
        return 0, 1, tuple()
    queue = deque([initial])
    distances = {initial: 0}
    path_counts = {initial: 1}
    parent: Dict[State, Tuple[State, Move]] = {}
    target_distance: Optional[int] = None
    while queue:
        state = queue.popleft()
        distance = distances[state]
        if target_distance is not None and distance >= target_distance:
            continue
        for successor, move in _successors(state, edges):
            next_distance = distance + 1
            previous = distances.get(successor)
            if previous is None:
                distances[successor] = next_distance
                path_counts[successor] = path_counts[state]
                parent[successor] = (state, move)
                queue.append(successor)
            elif previous == next_distance:
                path_counts[successor] += path_counts[state]
            if successor == target:
                target_distance = next_distance
    if target not in distances:
        return None, 0, None

    plan: List[Move] = []
    cursor = target
    while cursor != initial:
        previous, move = parent[cursor]
        plan.append(move)
        cursor = previous
    plan.reverse()
    return distances[target], path_counts[target], tuple(plan)


class TokenTradeMatrixGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "token_trade_matrix"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/token_trade_matrix"
    MOTION_VIDEO_MOVE_LIMIT = MAX_OPTIMAL_MOVES
    SCENE_OUTER_MARGIN = 8
    SCENE_CHEVRON_BAND = 28
    SCENE_CHEVRON_SCALE = 0.76

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        move_frames: int = MOVE_FRAMES,
    ) -> None:
        if type(move_frames) is not int or move_frames != MOVE_FRAMES:
            raise ValueError("move_frames must be exactly 8")
        super().__init__(
            output_dir=output_dir,
            seed=seed,
            video=video,
            move_frames=move_frames,
        )

    @staticmethod
    def _make_state(holdings: Sequence[int], edges: FrozenSet[Edge]) -> Dict[str, Any]:
        return {
            "holdings": [int(value) for value in holdings],
            "edges": [[a, b] for a, b in sorted(edges)],
        }

    def _make_problem(
        self,
        initial: Sequence[int],
        target: Sequence[int],
        edges: FrozenSet[Edge],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Move]]:
        initial_state = tuple(initial)
        target_state = tuple(target)
        distance, path_count, plan = _shortest_stats(initial_state, target_state, edges)
        complete_edges = frozenset(combinations(range(N_POSITIONS), 2))
        complete_distance, _, _ = _shortest_stats(initial_state, target_state, complete_edges)
        if (
            distance is None
            or plan is None
            or path_count < 1
            or not MIN_OPTIMAL_MOVES <= distance <= MAX_OPTIMAL_MOVES
            or complete_distance is None
            or complete_distance >= distance
        ):
            raise RuntimeError("token-trade source contract produced an invalid problem")
        return self._make_state(initial, edges), self._make_state(target, edges), list(plan)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        all_pairs = list(combinations(range(N_POSITIONS), 2))
        for _ in range(4000):
            initial = list(range(1, N_POSITIONS + 1))
            target = list(range(1, N_POSITIONS + 1))
            self.rng.shuffle(initial)
            self.rng.shuffle(target)
            if initial == target:
                continue
            edge_count = self.rng.randint(*EDGE_COUNT_RANGE)
            edges = frozenset(self.rng.sample(all_pairs, edge_count))
            try:
                _normalize_edges([[a, b] for a, b in sorted(edges)])
                return self._make_problem(initial, target, edges)
            except (ValueError, RuntimeError):
                continue
        raise RuntimeError(
            "token_trade_matrix rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state: Any) -> Any:
        holdings, edges = _validate_state(state)
        return holdings, tuple(sorted(edges))

    def _serialize_state(self, state: Any) -> Dict[str, Any]:
        holdings, edges = _validate_state(state)
        return self._make_state(holdings, edges)

    def _apply_move(self, state: Any, move: Any) -> Dict[str, Any]:
        holdings, edges = _validate_state(state)
        a, b = _validate_move(move, edges)
        successor = list(holdings)
        successor[a], successor[b] = successor[b], successor[a]
        return self._make_state(successor, edges)

    def _token_color(self, token_id: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(token_id - 1) % len(ITEM_PALETTE)]

    def _layout(self, panel_w: int, panel_h: int) -> Dict[str, float]:
        socket_r = min(panel_w * 0.115, 28.0)
        token_r = socket_r * 0.66
        label_offset = socket_r + max(11.0, panel_w * 0.048)
        radius = min(
            panel_w / 2 - label_offset - 4,
            panel_h / 2 - label_offset - 8,
        )
        return {
            "cx": panel_w / 2,
            "cy": panel_h / 2,
            "radius": radius,
            "socket_r": socket_r,
            "token_r": token_r,
            "label_offset": label_offset,
        }

    def _position_center(self, layout: Dict[str, float], index: int) -> Tuple[float, float]:
        angle = -math.pi / 2 + index * 2 * math.pi / N_POSITIONS
        return (
            layout["cx"] + layout["radius"] * math.cos(angle),
            layout["cy"] + layout["radius"] * math.sin(angle),
        )

    def _position_label_center(self, layout: Dict[str, float], index: int) -> Tuple[float, float]:
        center_x, center_y = self._position_center(layout, index)
        angle = -math.pi / 2 + index * 2 * math.pi / N_POSITIONS
        return (
            center_x + layout["label_offset"] * math.cos(angle),
            center_y + layout["label_offset"] * math.sin(angle),
        )

    def _draw_token(self, draw: ImageDraw.ImageDraw, center: Tuple[float, float], radius: float, token_id: int) -> None:
        color = self._token_color(token_id)
        self._draw_disc_icon(draw, center, radius, color, label=str(token_id))

    def _render_panel(
        self,
        state: Any,
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        holdings, edges = _validate_state(state)
        layout = self._layout(panel_w, panel_h)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)

        edge_width = max(3, int(round(panel_w * 0.014)))
        for a, b in sorted(edges):
            draw.line(
                (*self._position_center(layout, a), *self._position_center(layout, b)),
                fill=EDGE_COLOR,
                width=edge_width,
            )

        hidden = set(moving.get("hide_positions", ())) if moving else set()
        socket_r = layout["socket_r"]
        token_r = layout["token_r"]
        label_font = self._draw_font(draw, max(11, int(round(socket_r * 0.72))))
        for index in range(N_POSITIONS):
            center_x, center_y = self._position_center(layout, index)
            draw.ellipse(
                (center_x - socket_r, center_y - socket_r, center_x + socket_r, center_y + socket_r),
                fill=SOCKET_FILL,
                outline=SOCKET_OUTLINE,
                width=2,
            )
            label_x, label_y = self._position_label_center(layout, index)
            # Sockets are printed the way the answer names them: 1 through 5.
            draw.text((label_x, label_y), str(index + 1), font=label_font, fill=POSITION_TEXT, anchor="mm")
            if index not in hidden:
                self._draw_token(draw, (center_x, center_y), token_r, holdings[index])

        if moving:
            for token in moving.get("tokens", ()):
                self._draw_token(draw, tuple(token["center"]), token_r, int(token["token_id"]))
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        panel_h = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, panel_h, moving=animation)

    def _token_center_unscaled(self, panel_w: int, panel_h: int, index: int) -> Tuple[float, float]:
        return self._position_center(self._layout(panel_w, panel_h), index)

    def _animate_move(self, state, move, target_state):
        holdings, edges = _validate_state(state)
        a, b = _validate_move(move, edges)
        geometry = self._panel_geometry()
        panel_w = geometry["panel_w"]
        panel_h = geometry["panel_h"]
        start_a = self._token_center_unscaled(panel_w, panel_h, a)
        start_b = self._token_center_unscaled(panel_w, panel_h, b)
        right_image = self._render_panel(target_state, panel_w, panel_h)

        frames: List[Image.Image] = [self._render_static_scene(state, target_state)]
        for step in range(1, self.move_frames - 1):
            progress = step / (self.move_frames - 1)
            moving = {
                "hide_positions": {a, b},
                "tokens": [
                    {
                        "center": self._interpolate_pixel_linear(start_a, start_b, progress),
                        "token_id": holdings[a],
                    },
                    {
                        "center": self._interpolate_pixel_linear(start_b, start_a, progress),
                        "token_id": holdings[b],
                    },
                ],
            }
            left_image = self._render_panel(state, panel_w, panel_h, moving=moving)
            frames.append(self._compose_scene(left_image, right_image))
        after_state = self._apply_move(state, move)
        frames.append(self._render_static_scene(after_state, target_state))
        return frames

    def _build_video_frames(self, initial_state, target_state, moves):
        initial_image = self._render_static_scene(initial_state, target_state)
        frames = [np.array(initial_image, dtype=np.uint8) for _ in range(self.hold_frames)]
        current = initial_state
        for move in moves:
            frames.extend(np.array(frame, dtype=np.uint8) for frame in self._animate_move(current, move, target_state))
            current = self._apply_move(current, move)
            static_after = self._render_static_scene(current, target_state)
            frames.extend(np.array(static_after, dtype=np.uint8) for _ in range(self.pause_frames))
        final_image = self._render_static_scene(current, target_state)
        frames.extend(np.array(final_image, dtype=np.uint8) for _ in range(self.hold_frames))
        return pad_frames_to_wan_compatible_length(frames)

    def _describe_input(self) -> str:
        return (
            "The input image has two clean side-by-side ring panels separated by a small blue left-to-right "
            "chevron. The left panel is the current state and the right panel is the exact target. Both panels show "
            "the same five fixed circular sockets on the ring, and a teal line joining two socket centers is an "
            "undirected allowed trade edge while a missing line forbids that pair. The small grey digit just outside "
            "a socket is that socket's fixed position number, 1 through 5, and it never moves. The digit inside a "
            "socket's colored token is that token's own identity, also 1 through 5, and it travels with the token, so "
            "a position number and the identity currently sitting there are independent."
        )

    def _describe_rules(self) -> str:
        return (
            "A legal move is one pair `(a, b)` of distinct position numbers from 1 through 5. The pair is legal "
            "only when the input shows a teal edge directly joining those two positions; edge direction does not matter, "
            "so `(a, b)` and `(b, a)` perform the same trade. A legal trade simultaneously swaps the two tokens currently "
            "held at its endpoints. It creates no empty socket: both selected sockets immediately receive the other token. "
            "Every unselected token remains in its current socket, and every socket, position number, edge, and panel remains "
            "fixed. There is no self-trade, out-of-range endpoint, non-edge trade, one-token move, gravity, compaction, refill, "
            "reordering, duplication, disappearance, identity change, or chain reaction. The final five holdings read at "
            "positions 1 through 5 must equal the entire right panel exactly."
        )

    def _describe_output(self) -> str:
        return (
            "Return any shortest legal trade sequence; when several shortest sequences exist, any one is accepted. Use "
            "only `Answer: [(<endpoint_a>, <endpoint_b>), ...].` naming the two sockets by their printed position numbers "
            "1 through 5 rather than by the tokens they hold, with comma-space separators inside and between tuples, "
            "square brackets, exactly one final period, and no prefix, suffix, explanation, empty list, or extra field."
        )

    def _build_prompts(self) -> Dict[str, str]:
        """Use the family-owned natural three-section prompt contract."""
        return super()._build_prompts()

    def _format_move(self, move):
        """Render one internal 0-based socket pair for diagnostics."""

        return f"({int(move[0])}, {int(move[1])})"

    def _format_answer(self, moves):
        """Serialize the formal payload, naming sockets by their printed 1..5."""

        visible = to_visible_plan(self.TASK_NAME, moves)
        items = ", ".join(f"({int(a)}, {int(b)})" for a, b in visible)
        return f"[{items}]"

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    def _author_observation(self, initial_state, target_state, moves) -> str:
        holdings, edges = _validate_state(initial_state)
        target, _ = _validate_state(target_state)
        edge_list = ", ".join(f"{min(a, b) + 1}-{max(a, b) + 1}" for a, b in sorted(edges))
        misplaced = [index + 1 for index in range(N_POSITIONS) if holdings[index] != target[index]]
        return (
            f"Reading the sockets clockwise from the top, positions 1 through 5 hold tokens "
            f"{', '.join(str(t) for t in holdings)}, and the target wants "
            f"{', '.join(str(t) for t in target)}. Trades run only along the drawn edges "
            f"({edge_list}), swapping both endpoint tokens at once, so positions "
            f"{', '.join(str(p) for p in misplaced)} have to be fixed through the wiring that actually "
            "exists rather than by any direct swap."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        before, edges = _validate_state(previous_state)
        after, _ = _validate_state(next_state)
        target, _ = _validate_state(target_state)
        a, b = _validate_move(move, edges)
        left, right = min(a, b), max(a, b)
        settled = [
            position + 1
            for position in (a, b)
            if after[position] == target[position]
        ]
        if len(settled) == 2:
            gain = "which puts both tokens exactly where the target wants them"
        elif settled:
            gain = f"which settles position {settled[0]}"
        else:
            gain = "moving both tokens along the wiring toward their target sockets"
        return (
            f"Along the {left + 1}-{right + 1} edge, token {before[left]} and token {before[right]} trade "
            f"places, {gain}. Positions 1 through 5 now hold "
            f"{', '.join(str(t) for t in after)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Every socket now holds its target token after {len(moves)} edge trades, which is as short "
            "as the visible restricted graph allows."
        )

    def _extra_payload(self, initial_state, target_state, moves):
        initial, edges = _validate_state(initial_state)
        target, target_edges = _validate_state(target_state)
        if target_edges != edges:
            raise RuntimeError("target graph does not match the source graph")
        distance, path_count, _ = _shortest_stats(initial, target, edges)
        complete_edges = frozenset(combinations(range(N_POSITIONS), 2))
        complete_distance, complete_path_count, _ = _shortest_stats(initial, target, complete_edges)
        return {
            "n_positions": N_POSITIONS,
            "edges": [[a, b] for a, b in sorted(edges)],
            "optimal_moves": [[int(a), int(b)] for a, b in moves],
            "shortest_distance": distance,
            "shortest_path_count": path_count,
            "complete_graph_distance": complete_distance,
            "complete_graph_shortest_path_count": complete_path_count,
        }


__all__ = [
    "EDGE_COUNT_RANGE",
    "MAX_OPTIMAL_MOVES",
    "MIN_OPTIMAL_MOVES",
    "MOVE_FRAMES",
    "N_POSITIONS",
    "TokenTradeMatrixGenerator",
    "_normalize_edges",
    "_shortest_stats",
    "_validate_holdings",
    "_validate_move",
    "_validate_state",
]
