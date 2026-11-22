"""Shortest full reversal of one stack using two or three empty buffers.

The visible source contains one primary stack of three to five distinct blocks
and either two or three empty auxiliary columns.  A move pops exactly the top
block of one non-empty column and pushes it onto another column.  The target
reverses the primary stack in place and leaves every auxiliary column empty.

One auxiliary column cannot solve this task.  With the admitted three-to-five
block range, a third auxiliary column reduces the optimum by exactly one move,
so the number of visible buffers is a load-bearing planning variable.

Column positions are 0-based in the solver, the BFS state space, the state
trace and the renderer geometry, and are published to the model counted from 1.
The number printed on a block is its persistent identity, not a position, so it
is never shifted.  The single conversion lives in
:mod:`data.rearrangement_game.core.answer_indexing`.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    SlotStackMixin,
)
from data.common.video_encoding import pad_frames_to_wan_compatible_length
from data.rearrangement_game.core.answer_indexing import to_visible_move

BLOCK_COUNT_RANGE = (3, 5)
BUFFER_COLUMNS_RANGE = (2, 3)
MIN_OPTIMAL_MOVES = 6
MAX_OPTIMAL_MOVES = 13

State = Tuple[Tuple[int, ...], ...]
Move = Tuple[int, int]

PANEL_BG = (247, 247, 245)
SLOT_FILL = (239, 241, 243)
SLOT_OUTLINE = (213, 216, 219)
FLOOR_FILL = (158, 151, 139)
FLOOR_TOP = (201, 196, 187)


def _state_key(state: Sequence[Sequence[int]]) -> State:
    return tuple(tuple(stack) for stack in state)


def _validate_state(state: Any) -> State:
    if not isinstance(state, (list, tuple)) or not 3 <= len(state) <= 4:
        raise ValueError("state must contain exactly three or four columns")
    parsed: List[Tuple[int, ...]] = []
    values: List[int] = []
    for stack in state:
        if not isinstance(stack, (list, tuple)):
            raise ValueError("every column must be a list or tuple")
        parsed_stack: List[int] = []
        for value in stack:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("block IDs must be integers")
            parsed_stack.append(value)
            values.append(value)
        parsed.append(tuple(parsed_stack))
    if not BLOCK_COUNT_RANGE[0] <= len(values) <= BLOCK_COUNT_RANGE[1]:
        raise ValueError("state must contain three to five blocks")
    if sorted(values) != list(range(1, len(values) + 1)):
        raise ValueError("block IDs must be unique consecutive integers starting at 1")
    return tuple(parsed)


def _validate_move(move: Any, column_count: int) -> Move:
    if not isinstance(move, (list, tuple)) or len(move) != 2:
        raise ValueError("move must be a two-field source/destination pair")
    src, dst = move
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (src, dst)):
        raise ValueError("column indices must be integers")
    if not (0 <= src < column_count and 0 <= dst < column_count) or src == dst:
        raise ValueError("source and destination must be distinct in-range columns")
    return int(src), int(dst)


def _successors(state: State) -> List[Tuple[State, Move]]:
    out: List[Tuple[State, Move]] = []
    for src, stack in enumerate(state):
        if not stack:
            continue
        top = stack[-1]
        for dst in range(len(state)):
            if src == dst:
                continue
            successor = list(state)
            successor[src] = stack[:-1]
            successor[dst] = state[dst] + (top,)
            out.append((tuple(successor), (src, dst)))
    return out


@lru_cache(maxsize=512)
def _shortest_stats(initial: State, target: State) -> Tuple[Optional[int], int, Optional[Tuple[Move, ...]]]:
    """Return exact distance, exact shortest-path count, and one shortest plan."""

    if len(initial) != len(target):
        return None, 0, None
    if initial == target:
        return 0, 1, tuple()
    queue = deque([initial])
    distance = {initial: 0}
    path_count = {initial: 1}
    parent: Dict[State, Tuple[State, Move]] = {}
    target_distance: Optional[int] = None
    while queue:
        state = queue.popleft()
        current_distance = distance[state]
        if target_distance is not None and current_distance >= target_distance:
            continue
        for successor, move in _successors(state):
            next_distance = current_distance + 1
            previous_distance = distance.get(successor)
            if previous_distance is None:
                distance[successor] = next_distance
                path_count[successor] = path_count[state]
                parent[successor] = (state, move)
                queue.append(successor)
            elif previous_distance == next_distance:
                path_count[successor] += path_count[state]
            if successor == target:
                target_distance = next_distance
    if target not in distance:
        return None, 0, None

    plan: List[Move] = []
    cursor = target
    while cursor != initial:
        previous, move = parent[cursor]
        plan.append(move)
        cursor = previous
    plan.reverse()
    return distance[target], path_count[target], tuple(plan)


class StackReversalMinimalMovesGenerator(SlotStackMixin, RearrangementObjectPlanGenerator):
    TASK_NAME = "stack_reversal_minimal_moves"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/stack_reversal_minimal_moves"
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
        block_count_range: Tuple[int, int] = BLOCK_COUNT_RANGE,
        buffer_columns_range: Tuple[int, int] = BUFFER_COLUMNS_RANGE,
        min_optimal_moves: int = MIN_OPTIMAL_MOVES,
        max_optimal_moves: int = MAX_OPTIMAL_MOVES,
        move_frames: int = 8,
    ) -> None:
        if (
            not isinstance(block_count_range, tuple)
            or len(block_count_range) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in block_count_range)
            or block_count_range != BLOCK_COUNT_RANGE
        ):
            raise ValueError("block_count_range must be exactly (3, 5)")
        if (
            not isinstance(buffer_columns_range, tuple)
            or len(buffer_columns_range) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in buffer_columns_range)
            or buffer_columns_range != BUFFER_COLUMNS_RANGE
        ):
            raise ValueError("buffer_columns_range must be exactly (2, 3)")
        if isinstance(min_optimal_moves, bool) or not isinstance(min_optimal_moves, int):
            raise ValueError("min_optimal_moves must be an integer")
        if isinstance(max_optimal_moves, bool) or not isinstance(max_optimal_moves, int):
            raise ValueError("max_optimal_moves must be an integer")
        if (min_optimal_moves, max_optimal_moves) != (MIN_OPTIMAL_MOVES, MAX_OPTIMAL_MOVES):
            raise ValueError("optimal-move bounds must be exactly 6..13")
        if isinstance(move_frames, bool) or not isinstance(move_frames, int) or move_frames != 8:
            raise ValueError("move_frames must be exactly 8")
        super().__init__(
            output_dir=output_dir,
            seed=seed,
            video=video,
            move_frames=move_frames,
        )
        self.block_count_range = BLOCK_COUNT_RANGE
        self.buffer_columns_range = BUFFER_COLUMNS_RANGE
        self.min_optimal_moves = MIN_OPTIMAL_MOVES
        self.max_optimal_moves = MAX_OPTIMAL_MOVES
        self.block_count = BLOCK_COUNT_RANGE[1]

    def _make_problem(self, block_count: int, buffer_count: int, primary_column: int, order: Sequence[int]):
        column_count = buffer_count + 1
        initial = [[] for _ in range(column_count)]
        initial[primary_column] = list(order)
        target = [[] for _ in range(column_count)]
        target[primary_column] = list(reversed(order))
        distance, path_count, plan = _shortest_stats(_state_key(initial), _state_key(target))
        if (
            distance is None
            or plan is None
            or not self.min_optimal_moves <= distance <= self.max_optimal_moves
            or path_count < 1
        ):
            raise RuntimeError("stack-reversal source contract produced an invalid problem")
        return initial, target, list(plan)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        block_count = self.rng.randint(*self.block_count_range)
        buffer_count = self.rng.randint(*self.buffer_columns_range)
        primary_column = self.rng.randrange(buffer_count + 1)
        order = list(range(1, block_count + 1))
        self.rng.shuffle(order)
        return self._make_problem(block_count, buffer_count, primary_column, order)

    def _state_signature(self, state: Any) -> Any:
        return _state_key(state)

    def _apply_move(self, state: Any, move: Any) -> List[List[int]]:
        validated_state = _validate_state(state)
        src, dst = _validate_move(move, len(validated_state))
        if not validated_state[src]:
            raise ValueError("source column is empty")
        successor = [list(stack) for stack in validated_state]
        successor[dst].append(successor[src].pop())
        return successor

    def _block_color(self, block_id: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(block_id - 1) % len(ITEM_PALETTE)]

    def _layout(self, panel_w: int, panel_h: int, column_count: int) -> Dict[str, Any]:
        layout = self._slot_layout(
            panel_w,
            slot_count=column_count,
            item_size=panel_w * 0.42,
            max_stack=BLOCK_COUNT_RANGE[1],
            panel_height=panel_h,
        )
        # The slot group is intentionally compact, but it should still sit near
        # the visual centre rather than leaving an accidental half-frame void.
        desired_pedestal_y = int(round(panel_h * 0.69))
        vertical_shift = desired_pedestal_y - layout["pedestal_y"]
        layout["pedestal_y"] += vertical_shift
        layout["slot_floor_y"] += vertical_shift
        item_h = min(
            layout["item_w"] * 0.62,
            (layout["slot_floor_y"] - layout["margin_top"] - 16) / BLOCK_COUNT_RANGE[1] - 3,
        )
        layout["item_h"] = max(18.0, item_h)
        return layout

    def _block_center(self, layout: Dict[str, Any], column: int, depth: int) -> Tuple[float, float]:
        item_h = layout["item_h"]
        return (
            layout["slot_centers_x"][column],
            layout["slot_floor_y"] - depth * (item_h + 3) - item_h / 2,
        )

    def _render_panel(
        self,
        state: Sequence[Sequence[int]],
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        layout = self._layout(panel_w, panel_h, len(state))
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)

        item_h = layout["item_h"]
        slot_top = layout["slot_floor_y"] - BLOCK_COUNT_RANGE[1] * (item_h + 3) - 10
        slot_half_w = layout["item_w"] * 0.58
        for center_x in layout["slot_centers_x"]:
            draw.rounded_rectangle(
                (
                    center_x - slot_half_w,
                    slot_top,
                    center_x + slot_half_w,
                    layout["slot_floor_y"] + 3,
                ),
                radius=max(8, int(layout["item_h"] * 0.28)),
                fill=SLOT_FILL,
                outline=SLOT_OUTLINE,
                width=2,
            )
        floor_y = layout["pedestal_y"]
        draw.rounded_rectangle(
            (layout["margin_x"] / 2, floor_y, panel_w - layout["margin_x"] / 2, floor_y + layout["pedestal_h"]),
            radius=max(4, layout["pedestal_h"] // 3),
            fill=FLOOR_FILL,
        )
        draw.rectangle(
            (
                layout["margin_x"] / 2,
                floor_y,
                panel_w - layout["margin_x"] / 2,
                floor_y + max(2, layout["pedestal_h"] // 4),
            ),
            fill=FLOOR_TOP,
        )

        item_w = layout["item_w"]
        for column, stack in enumerate(state):
            for depth, block_id in enumerate(stack):
                if (
                    moving is not None
                    and moving.get("hide_top")
                    and column == moving.get("src")
                    and depth == len(stack) - 1
                ):
                    continue
                center_x, center_y = self._block_center(layout, column, depth)
                bbox = (
                    center_x - item_w / 2,
                    center_y - item_h / 2,
                    center_x + item_w / 2,
                    center_y + item_h / 2,
                )
                self._draw_block_icon(
                    draw,
                    bbox,
                    self._block_color(int(block_id)),
                    label=str(int(block_id)),
                    radius=max(5, int(item_h * 0.25)),
                )

        if moving is not None and "block" in moving and "center" in moving:
            center_x, center_y = moving["center"]
            bbox = (
                center_x - item_w / 2,
                center_y - item_h / 2,
                center_x + item_w / 2,
                center_y + item_h / 2,
            )
            self._draw_block_icon(
                draw,
                bbox,
                self._block_color(int(moving["block"])),
                label=str(int(moving["block"])),
                radius=max(5, int(item_h * 0.25)),
            )
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        panel_h = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, panel_h, moving=animation)

    def _animate_move(self, state, move, target_state):
        validated_state = _validate_state(state)
        src, dst = _validate_move(move, len(validated_state))
        if not validated_state[src]:
            raise ValueError("source column is empty")
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h, len(validated_state))
        block = validated_state[src][-1]
        source_center = self._block_center(layout, src, len(validated_state[src]) - 1)
        destination_center = self._block_center(layout, dst, len(validated_state[dst]))
        right_image = self._render_panel(target_state, panel_w, panel_h)
        frames = [self._render_static_scene(state, target_state)]
        for step in range(1, self.move_frames - 1):
            progress = step / (self.move_frames - 2)
            center = self._interpolate_pixel_linear(source_center, destination_center, progress)
            moving = {"block": block, "center": center, "src": src, "hide_top": True}
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
            after_image = self._render_static_scene(current, target_state)
            frames.extend(np.array(after_image, dtype=np.uint8) for _ in range(self.pause_frames))
        final_image = self._render_static_scene(current, target_state)
        frames.extend(np.array(final_image, dtype=np.uint8) for _ in range(self.hold_frames))
        return pad_frames_to_wan_compatible_length(frames)

    def _describe_input(self) -> str:
        return (
            "The input image contains two clean side-by-side stack panels separated by a small blue "
            "left-to-right chevron. The left panel is the current state and the right panel is the exact target. "
            "Each panel has three or four vertical columns. Columns are numbered 1 through C from left to right, "
            "although those numbers are not printed in the image. Exactly one left-panel column is non-empty at the "
            "start; it is the primary column and contains three to five equal-size colored blocks. The integer on each "
            "block is its persistent identity, and stack order is read bottom to top. The other two or three columns "
            "are empty buffers. The right panel shows the same blocks reversed in the same primary column and every "
            "buffer empty. No title, rule text, move list, answer banner, or hidden object appears in the image."
        )

    def _describe_rules(self) -> str:
        return (
            "A legal move is one ordered pair `(src, dst)`. Both numbers must name visible columns, `src` and `dst` "
            "must differ, and `src` must be non-empty. Pop exactly the top block of `src` and append that same block "
            "to the top of `dst`; no block below a top block may be selected. Columns have unlimited capacity. One "
            "move changes no other block: lower and unselected blocks keep their identity, size, order, and column; "
            "nothing falls, compacts, refills, duplicates, disappears, or changes color or number. An emptied position "
            "shows the normal slot background, and an empty column remains empty until a later listed move places a "
            "block there. The final state must match the entire right panel exactly, including the reversed bottom-to-top "
            "primary order and the absence of every block from every buffer."
        )

    def _describe_output(self) -> str:
        return (
            "Return any shortest legal move sequence; if several shortest sequences exist, any one is accepted. Use "
            "only `Answer: [(src, dst), ...].` with decimal integer pairs, comma-space separators between fields and "
            "moves, square brackets, exactly one final period, and no prefix, suffix, explanation, or extra field."
        )

    def _build_prompts(self) -> Dict[str, str]:
        """Use the family-owned natural three-section prompt contract."""
        return super()._build_prompts()

    def _format_move(self, move):
        # Solver, BFS and renderer columns are 0-based; the formal payload
        # counts columns from 1 at the leftmost column.
        source, destination = to_visible_move(self.TASK_NAME, (int(move[0]), int(move[1])))
        return f"({source}, {destination})"

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _column_phrase(stack: Sequence[int]) -> str:
        if not stack:
            return "clear"
        if len(stack) == 1:
            return f"just block {stack[0]}"
        listing = ", ".join(str(block) for block in stack)
        return f"{listing} bottom to top"

    def _columns_readout(self, state: Sequence[Sequence[int]]) -> str:
        return "; ".join(
            f"column {index} is {self._column_phrase(stack)}"
            if not stack
            else f"column {index} holds {self._column_phrase(stack)}"
            for index, stack in enumerate(state, start=1)
        )

    def _author_observation(self, initial_state, target_state, moves) -> str:
        initial = _validate_state(initial_state)
        primary = next(index for index, stack in enumerate(initial) if stack)
        order = list(initial[primary])
        return (
            f"Column {primary + 1} holds the whole stack, {', '.join(str(b) for b in order)} bottom to top, "
            f"and the other {len(initial) - 1} columns are empty buffers. The target shows the same column "
            f"reversed, {', '.join(str(b) for b in reversed(order))} bottom to top, with every buffer empty again. "
            "Only a column's top block can move, so the stack has to be dealt out across the buffers and "
            "rebuilt in the opposite order."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        before = _validate_state(previous_state)
        after = _validate_state(next_state)
        target = _validate_state(target_state)
        src, dst = to_visible_move(self.TASK_NAME, _validate_move(move, len(before)))
        block = before[src - 1][-1]
        settled = (
            len(after[dst - 1]) <= len(target[dst - 1])
            and tuple(target[dst - 1][: len(after[dst - 1])]) == tuple(after[dst - 1])
        )
        uncovered = before[src - 1][:-1]
        if settled and target[dst - 1]:
            purpose = "which is exactly where the reversed order wants it next"
        elif uncovered:
            purpose = f"which uncovers block {uncovered[-1]}"
        else:
            purpose = "keeping it parked out of the way"
        return (
            f"Block {block} sits on top of column {src}, so I set it onto column {dst}, {purpose}. "
            f"Now {self._columns_readout(after)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        initial = _validate_state(initial_state)
        return (
            f"The primary column now reads exactly in the reversed order and every buffer is clear again; "
            f"all {len(moves)} placements moved only exposed top blocks, which is as short as the visible "
            f"{len(initial) - 1}-buffer setup allows."
        )

    def _extra_payload(self, initial_state, target_state, moves):
        initial = _validate_state(initial_state)
        target = _validate_state(target_state)
        distance, path_count, _ = _shortest_stats(initial, target)
        primary = next(index for index, stack in enumerate(initial) if stack)
        return {
            "block_count": sum(len(stack) for stack in initial),
            "buffer_count": len(initial) - 1,
            "primary_column": primary,
            "optimal_moves": [[int(src), int(dst)] for src, dst in moves],
            "shortest_distance": distance,
            "shortest_path_count": path_count,
        }


__all__ = [
    "BLOCK_COUNT_RANGE",
    "BUFFER_COLUMNS_RANGE",
    "MAX_OPTIMAL_MOVES",
    "MIN_OPTIMAL_MOVES",
    "StackReversalMinimalMovesGenerator",
    "_shortest_stats",
    "_state_key",
    "_validate_move",
    "_validate_state",
]
