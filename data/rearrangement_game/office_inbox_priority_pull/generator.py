"""Priority-ordered work stack with one LIFO side buffer.

Four or five uniquely numbered papers begin in the left inbox stack.  Only
the top paper of the inbox or the top paper of the right buffer is accessible.
The middle work stack must stay strictly decreasing from bottom to top and the
published answer is a shortest legal move sequence that transfers every paper
to work.  The protocol token ``buffer`` denotes a temporary push from inbox to
the side buffer; no paper is discarded.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.pipeline.object_plans import ITEM_PALETTE
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import pad_frames_to_wan_compatible_length

PAPER_COUNT_RANGE = (4, 5)
MOVE_FRAMES = 9
RENDER_SCALE = DOCTRINE_SUPERSAMPLE

Move = Tuple[str, int]
State = Dict[str, List[int]]
StateKey = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]

CANVAS_FILL = (249, 248, 245)
BOARD_FILL = (246, 243, 236)
BOARD_OUTLINE = (211, 204, 191)
LANE_FILLS = ((244, 235, 218), (229, 241, 228), (229, 237, 245))
LANE_OUTLINES = ((205, 179, 139), (157, 190, 158), (155, 181, 205))
CARD_FILL = (255, 254, 250)
CARD_OUTLINE = (91, 91, 87)
TEXT_DARK = (42, 43, 42)


@dataclass(frozen=True)
class SolveResult:
    distance: int
    path_count: int
    path: Tuple[Move, ...]


def _state_key(inbox: Sequence[int], work: Sequence[int], buffer: Sequence[int]) -> StateKey:
    return tuple(inbox), tuple(work), tuple(buffer)


def _validate_move(raw: Any, paper_count: int) -> Move:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("a move must be a two-entry tuple")
    action, paper = raw
    if action not in ("work", "buffer"):
        raise ValueError("move action must be exactly 'work' or 'buffer'")
    if type(paper) is not int or not 1 <= paper <= paper_count:
        raise ValueError("paper id must be a non-boolean integer in 1..N")
    return str(action), int(paper)


def _validate_state(raw: Any) -> State:
    if not isinstance(raw, dict):
        raise ValueError("state must be an object")
    if set(raw) != {"inbox", "work", "buffer"}:
        raise ValueError("state must contain exactly inbox, work, and buffer")
    parsed: State = {}
    all_papers: List[int] = []
    for key in ("inbox", "work", "buffer"):
        values = raw.get(key)
        if not isinstance(values, list):
            raise ValueError(f"state.{key} must be a list")
        stack: List[int] = []
        for value in values:
            if type(value) is not int:
                raise ValueError("paper ids must be integers rather than booleans")
            stack.append(int(value))
        parsed[key] = stack
        all_papers.extend(stack)
    if not PAPER_COUNT_RANGE[0] <= len(all_papers) <= PAPER_COUNT_RANGE[1]:
        raise ValueError("state must contain four or five papers")
    if sorted(all_papers) != list(range(1, len(all_papers) + 1)):
        raise ValueError("state must contain each paper id 1..N exactly once")
    if parsed["work"] != sorted(parsed["work"], reverse=True):
        raise ValueError("work must be strictly decreasing from bottom to top")
    return parsed


def _goal_state(paper_count: int) -> State:
    if type(paper_count) is not int or not PAPER_COUNT_RANGE[0] <= paper_count <= PAPER_COUNT_RANGE[1]:
        raise ValueError("paper count must be four or five")
    return {"inbox": [], "work": list(range(paper_count, 0, -1)), "buffer": []}


def _legal_successors(key: StateKey) -> Tuple[Tuple[StateKey, Move], ...]:
    inbox, work, buffer = key
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


def _shortest_solution(inbox: Sequence[int], *, max_moves: Optional[int] = None) -> Optional[SolveResult]:
    papers = tuple(inbox)
    if not PAPER_COUNT_RANGE[0] <= len(papers) <= PAPER_COUNT_RANGE[1]:
        raise ValueError("inbox must contain four or five papers")
    if sorted(papers) != list(range(1, len(papers) + 1)) or any(type(value) is not int for value in papers):
        raise ValueError("inbox must be a permutation of 1..N")
    if max_moves is not None and (type(max_moves) is not int or max_moves < 0):
        raise ValueError("max_moves must be a non-negative integer or None")

    initial: StateKey = (papers, (), ())
    goal: StateKey = ((), tuple(range(len(papers), 0, -1)), ())
    queue = deque([initial])
    distance = {initial: 0}
    path_count = {initial: 1}
    parent: Dict[StateKey, Tuple[StateKey, Move]] = {}
    goal_distance: Optional[int] = None

    while queue:
        current = queue.popleft()
        depth = distance[current]
        if goal_distance is not None and depth >= goal_distance:
            continue
        if max_moves is not None and depth >= max_moves:
            continue
        for next_state, move in _legal_successors(current):
            next_depth = depth + 1
            if max_moves is not None and next_depth > max_moves:
                continue
            if next_state not in distance:
                distance[next_state] = next_depth
                path_count[next_state] = path_count[current]
                parent[next_state] = current, move
                queue.append(next_state)
            elif distance[next_state] == next_depth:
                path_count[next_state] += path_count[current]
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
    return SolveResult(distance[goal], path_count[goal], tuple(path))


class OfficeInboxPriorityPullGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "office_inbox_priority_pull"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/office_inbox_priority_pull"
    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        paper_count_range: Tuple[int, int] = PAPER_COUNT_RANGE,
        max_optimal_moves: int = 9,
        move_frames: int = MOVE_FRAMES,
    ) -> None:
        if (
            not isinstance(paper_count_range, tuple)
            or len(paper_count_range) != 2
            or any(type(value) is not int for value in paper_count_range)
            or not PAPER_COUNT_RANGE[0] <= paper_count_range[0] <= paper_count_range[1] <= PAPER_COUNT_RANGE[1]
        ):
            raise ValueError("paper_count_range must be an integer subrange of (4, 5)")
        if type(max_optimal_moves) is not int or not 5 <= max_optimal_moves <= 9:
            raise ValueError("max_optimal_moves must be an integer from 5 through 9")
        if type(move_frames) is not int or move_frames < 5 or move_frames % 2 == 0:
            raise ValueError("move_frames must be an odd integer of at least five")
        super().__init__(
            output_dir=output_dir,
            seed=seed,
            video=video,
            move_frames=move_frames,
        )
        self.paper_count_range = paper_count_range
        self.max_optimal_moves = max_optimal_moves

    def _make_problem(self, inbox: Sequence[int]) -> Tuple[State, State, List[Move]]:
        papers = tuple(inbox)
        solution = _shortest_solution(papers, max_moves=self.max_optimal_moves)
        if solution is None:
            raise ValueError("inbox order has no solution within the configured move limit")
        if solution.path_count != 1:
            raise ValueError("published inbox order must have one unique shortest solution")
        if not any(action == "buffer" for action, _ in solution.path):
            raise ValueError("published inbox order must require the side buffer")
        initial = {"inbox": list(papers), "work": [], "buffer": []}
        target = _goal_state(len(papers))
        replay: State = initial
        for move in solution.path:
            replay = self._apply_move(replay, move)
        if self._state_signature(replay) != self._state_signature(target):
            raise RuntimeError("independently solved path does not reach the derived goal")
        self._office_prompt_moves = tuple(solution.path)
        return initial, target, list(solution.path)

    def _sample_problem(self) -> Tuple[State, State, List[Move]]:
        counts = list(range(self.paper_count_range[0], self.paper_count_range[1] + 1))
        self.rng.shuffle(counts)
        for paper_count in counts:
            eligible: List[Tuple[int, ...]] = []
            for order in permutations(range(1, paper_count + 1)):
                result = _shortest_solution(order, max_moves=self.max_optimal_moves)
                if result is None or result.path_count != 1:
                    continue
                if any(action == "buffer" for action, _ in result.path):
                    eligible.append(order)
            if eligible:
                return self._make_problem(self.rng.choice(eligible))
        raise RuntimeError("no uniquely solvable buffer-dependent instance matches the configured limits")

    def _state_signature(self, state: Any) -> StateKey:
        parsed = _validate_state(state)
        return _state_key(parsed["inbox"], parsed["work"], parsed["buffer"])

    def _serialize_state(self, state: Any) -> Any:
        return _validate_state(state)

    def _apply_move(self, state: Any, move: Any) -> State:
        parsed = _validate_state(state)
        paper_count = sum(len(parsed[key]) for key in ("inbox", "work", "buffer"))
        action, paper = _validate_move(move, paper_count)
        inbox = list(parsed["inbox"])
        work = list(parsed["work"])
        buffer = list(parsed["buffer"])
        if action == "buffer":
            if not inbox or inbox[-1] != paper:
                raise ValueError("buffer may move only the current inbox top")
            inbox.pop()
            buffer.append(paper)
        else:
            if work and work[-1] <= paper:
                raise ValueError("work placement must remain strictly decreasing bottom-to-top")
            if inbox and inbox[-1] == paper:
                inbox.pop()
            elif buffer and buffer[-1] == paper:
                buffer.pop()
            else:
                raise ValueError("work may move only an accessible top: the current inbox top or buffer top")
            work.append(paper)
        return _validate_state({"inbox": inbox, "work": work, "buffer": buffer})

    # ------------------------------------------------------------------
    # Clean single-board rendering and fixed-size direct motion
    # ------------------------------------------------------------------

    def _layout(self, size: int, paper_count: int) -> Dict[str, Any]:
        if type(paper_count) is not int or not PAPER_COUNT_RANGE[0] <= paper_count <= PAPER_COUNT_RANGE[1]:
            raise ValueError("paper_count must be four or five")
        scale = size / self.canvas_size
        board = (size * 0.06, size * 0.06, size * 0.94, size * 0.94)
        lane_centers = (size * 0.22, size * 0.50, size * 0.78)
        lane_width = size * 0.225
        lane_top = size * 0.13
        lane_bottom = size * 0.88
        card_w = size * 0.19
        card_h = min(size * 0.105, (lane_bottom - lane_top) / (paper_count + 0.65))
        gap = size * 0.018
        base_y = lane_bottom - size * 0.035
        return {
            "board": board,
            "lane_centers": lane_centers,
            "lane_width": lane_width,
            "lane_top": lane_top,
            "lane_bottom": lane_bottom,
            "card_w": card_w,
            "card_h": card_h,
            "gap": gap,
            "base_y": base_y,
            "scale": scale,
        }

    def _paper_center(self, layout: Dict[str, Any], lane_index: int, depth: int) -> Tuple[float, float]:
        return (
            layout["lane_centers"][lane_index],
            layout["base_y"] - layout["card_h"] / 2 - depth * (layout["card_h"] + layout["gap"]),
        )

    def _paper_color(self, paper: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(paper - 1) % len(ITEM_PALETTE)]

    def _draw_paper(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        card_w: float,
        card_h: float,
        paper: int,
        *,
        scale: float,
    ) -> None:
        cx, cy = center
        x0, y0 = cx - card_w / 2, cy - card_h / 2
        x1, y1 = cx + card_w / 2, cy + card_h / 2
        radius = max(8, int(round(card_h * 0.18)))
        line_width = max(2, int(round(1.4 * scale)))
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=radius,
            fill=CARD_FILL,
            outline=CARD_OUTLINE,
            width=line_width,
        )
        stripe = max(9 * scale, card_w * 0.105)
        draw.rounded_rectangle(
            (x0 + 4 * scale, y0 + 4 * scale, x0 + stripe, y1 - 4 * scale),
            radius=max(4, int(round(card_h * 0.10))),
            fill=self._paper_color(paper),
        )
        font = self._load_font(max(20, int(round(card_h * 0.47))))
        draw.text((cx + card_w * 0.07, cy - scale), str(paper), font=font, fill=TEXT_DARK, anchor="mm")

    def _render_scene_raw(self, state: Any, size: int, *, moving: Optional[Dict[str, Any]] = None) -> Image.Image:
        parsed = _validate_state(state)
        paper_count = sum(len(parsed[key]) for key in ("inbox", "work", "buffer"))
        layout = self._layout(size, paper_count)
        # Shared supersample allocation; this renderer multiplies its own
        # coordinates by the layout scale, so it draws through the raw handle.
        image, scaled_draw = new_scaled_canvas(
            size // RENDER_SCALE, size // RENDER_SCALE, RENDER_SCALE, CANVAS_FILL
        )
        draw = scaled_draw.raw
        line_width = max(2, int(round(1.4 * layout["scale"])))
        draw.rounded_rectangle(
            layout["board"],
            radius=max(24, int(round(24 * layout["scale"]))),
            fill=BOARD_FILL,
            outline=BOARD_OUTLINE,
            width=line_width,
        )
        for lane_index, center_x in enumerate(layout["lane_centers"]):
            x0 = center_x - layout["lane_width"] / 2
            x1 = center_x + layout["lane_width"] / 2
            draw.rounded_rectangle(
                (x0, layout["lane_top"], x1, layout["lane_bottom"]),
                radius=max(18, int(round(18 * layout["scale"]))),
                fill=LANE_FILLS[lane_index],
                outline=LANE_OUTLINES[lane_index],
                width=line_width,
            )
            draw.rounded_rectangle(
                (x0 + size * 0.018, layout["base_y"], x1 - size * 0.018, layout["base_y"] + size * 0.018),
                radius=max(4, int(round(4 * layout["scale"]))),
                fill=LANE_OUTLINES[lane_index],
            )

        hidden_lane = moving.get("source_lane") if isinstance(moving, dict) else None
        stacks = (parsed["inbox"], parsed["work"], parsed["buffer"])
        for lane_index, stack in enumerate(stacks):
            for depth, paper in enumerate(stack):
                if hidden_lane == lane_index and depth == len(stack) - 1:
                    continue
                self._draw_paper(
                    draw,
                    self._paper_center(layout, lane_index, depth),
                    layout["card_w"],
                    layout["card_h"],
                    paper,
                    scale=layout["scale"],
                )
        if isinstance(moving, dict):
            center = moving.get("center")
            paper = moving.get("paper")
            if isinstance(center, tuple) and len(center) == 2 and type(paper) is int:
                self._draw_paper(
                    draw,
                    (center[0] * layout["scale"], center[1] * layout["scale"]),
                    layout["card_w"],
                    layout["card_h"],
                    paper,
                    scale=layout["scale"],
                )
        return image

    def _render_scene(self, state: Any, *, moving: Optional[Dict[str, Any]] = None) -> Image.Image:
        raw = self._render_scene_raw(state, self.canvas_size * RENDER_SCALE, moving=moving)
        return finish_scaled_canvas(raw, (self.canvas_size, self.canvas_size))

    def _render_static_scene(self, initial_state: Any, target_state: Any) -> Image.Image:
        del target_state
        return self._render_scene(initial_state)

    def _render_state(
        self,
        state: Any,
        panel_width: int,
        *,
        panel_height: Optional[int] = None,
        animation: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        panel_h = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_scene(state, moving=animation).resize((int(panel_width), panel_h), Image.Resampling.LANCZOS)

    def _movement_centers(self, state: Any, move: Any) -> Tuple[Tuple[float, float], Tuple[float, float], int, int]:
        parsed = _validate_state(state)
        paper_count = sum(len(parsed[key]) for key in ("inbox", "work", "buffer"))
        action, paper = _validate_move(move, paper_count)
        if parsed["inbox"] and parsed["inbox"][-1] == paper:
            source_lane = 0
            source_depth = len(parsed["inbox"]) - 1
        elif action == "work" and parsed["buffer"] and parsed["buffer"][-1] == paper:
            source_lane = 2
            source_depth = len(parsed["buffer"]) - 1
        else:
            raise ValueError("move source is not the current top paper")
        destination_lane = 1 if action == "work" else 2
        destination_stack = parsed["work"] if action == "work" else parsed["buffer"]
        layout = self._layout(self.canvas_size, paper_count)
        source = self._paper_center(layout, source_lane, source_depth)
        destination = self._paper_center(layout, destination_lane, len(destination_stack))
        return source, destination, source_lane, paper

    def _animate_move(self, state: Any, move: Any, target_state: Any) -> List[Image.Image]:
        del target_state
        parsed = _validate_state(state)
        next_state = self._apply_move(parsed, move)
        source, destination, source_lane, paper = self._movement_centers(parsed, move)
        frames = [self._render_scene(parsed)]
        for step in range(1, self.move_frames - 1):
            progress = step / (self.move_frames - 1)
            center = self._interpolate_pixel_linear(source, destination, progress)
            frames.append(
                self._render_scene(
                    parsed,
                    moving={"source_lane": source_lane, "paper": paper, "center": center},
                )
            )
        frames.append(self._render_scene(next_state))
        return frames

    def _build_video_frames(self, initial_state: Any, target_state: Any, moves: Sequence[Any]) -> List[np.ndarray]:
        frames = [np.asarray(self._render_scene(initial_state), dtype=np.uint8) for _ in range(self.hold_frames)]
        current = initial_state
        for move in moves:
            frames.extend(np.asarray(frame, dtype=np.uint8) for frame in self._animate_move(current, move, target_state))
            current = self._apply_move(current, move)
            frames.extend(np.asarray(self._render_scene(current), dtype=np.uint8) for _ in range(self.pause_frames))
        frames.extend(np.asarray(self._render_scene(current), dtype=np.uint8) for _ in range(self.hold_frames))
        return pad_frames_to_wan_compatible_length(frames)

    # ------------------------------------------------------------------
    # Complete Prompt and Answer contracts
    # ------------------------------------------------------------------

    def _describe_input(self) -> str:
        return (
            "The fixed board has three open vertical lanes: warm amber inbox on the left, pale green work in the centre, "
            "and pale blue temporary buffer on the right. Stacks are read bottom-to-top, so the visibly "
            "highest paper is the only accessible top. Each of the four or five papers shows one priority from 1 through N and a "
            "stable identity-color stripe. No solved target preview or explanatory text appears."
        )

    def _describe_rules(self) -> str:
        return (
            "Every paper id from 1 through N exists once and must finish on work. `('buffer', p)` moves only the current inbox top to "
            "the buffer top. `('work', p)` moves only the "
            "current top of inbox or buffer to work, which must be empty or have a larger current top. Only the buffer top "
            "is accessible, and no paper can return to inbox or leave work. An empty source cannot be used. Each action "
            "moves the same named paper, merely exposing any paper below it, while every unselected paper stays fixed. The "
            "goal is empty inbox and buffer with work=[N, N-1, ..., 1] bottom-to-top. Return a shortest legal sequence; all "
            "shortest solutions are accepted, and the visible instance has a unique one."
        )

    def _describe_output(self) -> str:
        return (
            "Return only `Answer: [(\"action\", p), ...].` using lowercase double-quoted `work` or `buffer`, decimal "
            "paper ids, comma-space separators, parentheses around each move, square brackets, and one final period."
        )

    def _build_prompts(self) -> Dict[str, str]:
        """Use the family-owned natural three-section prompt contract."""
        return super()._build_prompts()

    def _format_move(self, move: Any) -> str:
        action, paper = _validate_move(move, PAPER_COUNT_RANGE[1])
        return f'(\"{action}\", {paper})'

    def _format_answer(self, moves: Sequence[Any]) -> str:
        return "[" + ", ".join(self._format_move(move) for move in moves) + "]"

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _lane_phrase(values) -> str:
        return ", ".join(str(v) for v in values) if values else "empty"

    def _desk_readout(self, state) -> str:
        parsed = _validate_state(state)
        return (
            f"inbox: {self._lane_phrase(parsed['inbox'])}; work: {self._lane_phrase(parsed['work'])}; "
            f"buffer: {self._lane_phrase(parsed['buffer'])}"
        )

    def _author_observation(self, initial_state, target_state, moves) -> str:
        parsed = _validate_state(initial_state)
        return (
            f"The inbox stacks papers {self._lane_phrase(parsed['inbox'])} bottom to top, and only a "
            "stack's top paper is ever reachable. Every paper must end on the work stack, which has to "
            "stay strictly decreasing from bottom to top, so higher numbers must go down first; the "
            "one-lane buffer is the only place to park an inbox top that is not ready yet."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        parsed = _validate_state(previous_state)
        paper_count = sum(len(parsed[key]) for key in ("inbox", "work", "buffer"))
        action, paper = _validate_move(move, paper_count)
        if action == "buffer":
            below = _validate_state(next_state)["inbox"]
            expose = f", exposing paper {below[-1]}" if below else ""
            detail = (
                f"The inbox top is paper {paper}, which cannot go onto work yet, so it parks on the "
                f"buffer{expose}"
            )
        else:
            source = "inbox" if parsed["inbox"] and parsed["inbox"][-1] == paper else "buffer"
            prior = parsed["work"][-1] if parsed["work"] else None
            base = (
                f"under paper {prior}, keeping work strictly decreasing"
                if prior is not None
                else "as the base of the work stack"
            )
            detail = f"Paper {paper} is now reachable on the {source} top, so it settles onto work {base}"
        return f"{detail}. Now {self._desk_readout(next_state)}."

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Every paper now sits on the work stack in strictly decreasing order with the inbox and "
            f"buffer empty, after {len(moves)} pulls of the unique shortest plan."
        )

    def _extra_payload(self, initial_state: Any, target_state: Any, moves: Sequence[Any]) -> Dict[str, Any]:
        initial = _validate_state(initial_state)
        target = _validate_state(target_state)
        solution = _shortest_solution(initial["inbox"])
        if solution is None:
            raise RuntimeError("visible source state is not solvable")
        if tuple(moves) != solution.path or len(moves) != solution.distance or solution.path_count != 1:
            raise RuntimeError("record does not contain the unique independently shortest plan")
        if target != _goal_state(len(initial["inbox"])):
            raise RuntimeError("record target does not equal the goal derived from the visible source")
        replay: State = initial
        for move in moves:
            replay = self._apply_move(replay, move)
        if replay != target:
            raise RuntimeError("record target does not match strict move replay")
        return {
            "paper_count": len(initial["inbox"]),
            "initial_inbox_bottom_to_top": list(initial["inbox"]),
            "optimal_distance": solution.distance,
            "shortest_path_count": solution.path_count,
            "buffer_move_count": sum(1 for action, _ in solution.path if action == "buffer"),
            "work_checkpoint_count": sum(1 for action, _ in solution.path if action == "work"),
            "optimal_moves": [[action, paper] for action, paper in solution.path],
        }


__all__ = [
    "MOVE_FRAMES",
    "PAPER_COUNT_RANGE",
    "RENDER_SCALE",
    "OfficeInboxPriorityPullGenerator",
    "SolveResult",
    "_goal_state",
    "_legal_successors",
    "_shortest_solution",
    "_state_key",
    "_validate_move",
    "_validate_state",
]
