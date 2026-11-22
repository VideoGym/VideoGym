"""Crane-based block rearrangement.

Three to five vertical columns of colored construction blocks. The crane
moves only the top block of any non-empty column to the top of any other
column. Each column has unlimited capacity. The solver returns the shortest
sequence of (from_column, to_column) crane moves that turns the starting
arrangement into the target arrangement.

This task is the generalised LIFO column rearrangement that the previous
``construction_stack`` task tried to be — same mechanic but now with cleaner
visuals (no captions, no panel labels), smooth-motion video, and a randomised
column count between 3 and 5 to broaden mechanic coverage.

A submitted answer counts the columns 1 for the leftmost column onward; the
sampler, BFS, replay and renderer below index them from 0.  The single
conversion lives in :mod:`data.rearrangement_game.core.answer_indexing` and
happens only where a move is serialized for the model.  The numbers printed on
the blocks are identities, so they are never shifted in either direction.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    SlotStackMixin,
)
from data.rearrangement_game.core.answer_indexing import to_visible_move

PANEL_BG = (250, 246, 238)
PANEL_BORDER = (216, 207, 188)
COLUMN_RAIL = (208, 192, 168)
GANTRY_FILL = (88, 102, 124)
GANTRY_OUTLINE = (52, 62, 80)
TROLLEY_FILL = (224, 88, 76)
TROLLEY_OUTLINE = (138, 36, 28)
CABLE_COLOR = (52, 56, 64)
CLAW_FILL = (216, 198, 132)
CLAW_OUTLINE = (118, 92, 36)


def _state_key(state: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(stack) for stack in state)


class CraneBlockStackGenerator(SlotStackMixin, RearrangementObjectPlanGenerator):
    TASK_NAME = "crane_block_stack"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/crane_block_stack"

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        block_count_range: Tuple[int, int] = (4, 6),
        column_count_range: Tuple[int, int] = (3, 5),
        max_optimal_moves: int = 8,
        min_optimal_moves: int = 2,
        move_frames: int = 24,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            seed=seed,
            video=video,
            move_frames=move_frames,
        )
        if len(block_count_range) != 2 or not (
            4 <= int(block_count_range[0]) <= int(block_count_range[1]) <= 9
        ):
            raise ValueError(
                "block_count_range must stay within the inclusive 4..9 range"
            )
        if len(column_count_range) != 2 or not (
            3 <= int(column_count_range[0]) <= int(column_count_range[1]) <= 5
        ):
            raise ValueError("column_count_range must stay within the inclusive 3..5 contract")
        self.block_count_range = tuple(int(value) for value in block_count_range)
        self.column_count_range = tuple(int(value) for value in column_count_range)
        self.max_optimal_moves = int(max_optimal_moves)
        self.min_optimal_moves = int(min_optimal_moves)

    def _choose_columns(self) -> int:
        lo, hi = self.column_count_range
        return self.rng.randint(lo, hi)

    def _choose_blocks(self) -> int:
        lo, hi = self.block_count_range
        return self.rng.randint(lo, hi)

    def _random_state(self, column_count: int, blocks: List[int]) -> List[List[int]]:
        cols: List[List[int]] = [[] for _ in range(column_count)]
        order = list(blocks)
        self.rng.shuffle(order)
        for block in order:
            cols[self.rng.randrange(column_count)].append(block)
        return cols

    def _shortest_plan(
        self,
        initial: Sequence[Sequence[int]],
        target: Sequence[Sequence[int]],
        column_count: int,
        *,
        max_moves: Optional[int] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        init_key = _state_key(initial)
        tgt_key = _state_key(target)
        if init_key == tgt_key:
            return []
        move_limit = self.max_optimal_moves if max_moves is None else int(max_moves)
        # Every top-block transfer has an exact inverse. Search both ends
        # without copying a complete path into every queued state; larger
        # object sets otherwise exhaust memory before the same depth bound.
        frontiers = [{init_key}, {tgt_key}]
        parents = [{init_key: None}, {tgt_key: None}]
        depths = [0, 0]
        while all(frontiers) and sum(depths) < move_limit:
            side = 0 if len(frontiers[0]) <= len(frontiers[1]) else 1
            following = set()
            for state in sorted(frontiers[side]):
                for src in range(column_count):
                    if not state[src]:
                        continue
                    for dst in range(column_count):
                        if src == dst:
                            continue
                        stacks = list(state)
                        stacks[src] = state[src][:-1]
                        stacks[dst] = state[dst] + (state[src][-1],)
                        nxt = tuple(stacks)
                        if nxt in parents[side]:
                            continue
                        parents[side][nxt] = (state, (src, dst))
                        following.add(nxt)
                        if nxt in parents[1 - side]:
                            left, right = [], []
                            node = nxt
                            while parents[0][node] is not None:
                                node, move = parents[0][node]
                                left.append(move)
                            node = nxt
                            while parents[1][node] is not None:
                                node, move = parents[1][node]
                                right.append((move[1], move[0]))
                            return left[::-1] + right
            frontiers[side] = following
            depths[side] += 1
        return None

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            cols = self._choose_columns()
            blocks_n = self._choose_blocks()
            block_ids = list(range(1, blocks_n + 1))
            initial = self._random_state(cols, block_ids)
            target = self._random_state(cols, block_ids)
            if _state_key(initial) == _state_key(target):
                continue
            plan = self._shortest_plan(initial, target, cols)
            if plan is None or not plan:
                continue
            if self.min_optimal_moves <= len(plan) <= self.max_optimal_moves:
                return initial, target, plan
        raise RuntimeError(
            "crane_block_stack rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state: Any) -> Any:
        return _state_key(state)

    def _apply_move(self, state: Any, move: Any) -> Any:
        src, dst = move
        new_state = [list(stack) for stack in state]
        if isinstance(src, bool) or isinstance(dst, bool):
            raise ValueError("Column indices must be integers")
        src = int(src)
        dst = int(dst)
        if not (0 <= src < len(new_state) and 0 <= dst < len(new_state)) or src == dst:
            raise ValueError(f"Illegal crane move: {(src, dst)}")
        if not new_state[src]:
            raise ValueError(f"Source column {src} is empty")
        new_state[dst].append(new_state[src].pop())
        return new_state

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _block_color(self, block_id: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(block_id - 1) % len(ITEM_PALETTE)]

    def _crane_geometry(self, panel_w: int, panel_h: int) -> Dict[str, Any]:
        """Reserve a top zone for the gantry beam plus a hanging trolley arm.

        The gantry beam runs full-width near the top; the trolley arm dangles
        below it down to a claw resting just above the topmost block of the
        active column.
        """

        gantry_h = max(14, int(round(panel_h * 0.05)))
        # Crane zone: gantry beam + a short clearance band so the cable + claw
        # have somewhere to live before reaching the blocks below.
        crane_zone_h = gantry_h + max(20, int(round(panel_h * 0.04)))
        return {
            "gantry_top": 6,
            "gantry_h": gantry_h,
            "gantry_bot": 6 + gantry_h,
            "crane_zone_h": crane_zone_h,
            # bottom of the reserved crane zone is where layout's margin_top should sit
            "layout_margin_top": 6 + crane_zone_h,
        }

    def _draw_crane(
        self,
        draw: ImageDraw.ImageDraw,
        panel_w: int,
        panel_h: int,
        crane_geom: Dict[str, Any],
        trolley_cx: float,
        cable_bottom_y: float,
        *,
        carrying: Optional[Tuple[int, int, int]] = None,
        item_w: float = 0.0,
        item_h: float = 0.0,
    ) -> None:
        """Draw one flat rail, a compact trolley, cable, and U-clamp."""

        gantry_top = crane_geom["gantry_top"]
        gantry_h = crane_geom["gantry_h"]
        gantry_bot = crane_geom["gantry_bot"]
        draw.rounded_rectangle(
            (4, gantry_top, panel_w - 4, gantry_bot),
            radius=max(4, gantry_h // 3),
            fill=GANTRY_FILL,
            outline=GANTRY_OUTLINE,
            width=2,
        )

        trolley_w = max(22, int(round(panel_w * 0.10)))
        trolley_h = max(12, int(round(gantry_h * 0.72)))
        tx0 = trolley_cx - trolley_w / 2
        tx1 = trolley_cx + trolley_w / 2
        ty0 = gantry_bot - trolley_h / 2
        ty1 = ty0 + trolley_h
        draw.rounded_rectangle(
            (tx0, ty0, tx1, ty1),
            radius=max(3, int(trolley_h // 3)),
            fill=TROLLEY_FILL,
            outline=TROLLEY_OUTLINE,
            width=2,
        )

        cable_top_y = ty1
        cable_x = trolley_cx
        claw_w = max(18, int(round(item_w * 0.55)) if item_w > 0 else int(round(panel_w * 0.08)))
        claw_h = max(10, int(round(claw_w * 0.36)))
        claw_bottom_y = cable_bottom_y
        claw_top_y = claw_bottom_y - claw_h
        if claw_top_y > cable_top_y:
            draw.line(
                (cable_x, cable_top_y, cable_x, claw_top_y),
                fill=CABLE_COLOR,
                width=2,
            )
        clamp_x0 = cable_x - claw_w / 2
        clamp_x1 = cable_x + claw_w / 2
        bar_h = max(3, int(claw_h * 0.28))
        draw.rounded_rectangle(
            (clamp_x0, claw_top_y, clamp_x1, claw_top_y + bar_h),
            radius=2,
            fill=CLAW_FILL,
            outline=CLAW_OUTLINE,
            width=1,
        )
        draw.line((clamp_x0, claw_top_y + bar_h, clamp_x0, claw_bottom_y), fill=CLAW_OUTLINE, width=3)
        draw.line((clamp_x1, claw_top_y + bar_h, clamp_x1, claw_bottom_y), fill=CLAW_OUTLINE, width=3)

    def _trolley_resting_x(self, layout: Dict[str, Any], state: Sequence[Sequence[int]]) -> float:
        """Default trolley x for the static puzzle image: hover over the tallest
        non-empty column so the crane visibly "owns" a stack.
        """

        slot_centers = layout["slot_centers_x"]
        max_h = -1
        choice = 0
        for idx, stack in enumerate(state):
            if len(stack) > max_h:
                max_h = len(stack)
                choice = idx
        return slot_centers[choice]

    def _panel_layout(
        self,
        panel_w: int,
        panel_h: int,
        state: Sequence[Sequence[int]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        crane_geom = self._crane_geometry(panel_w, panel_h)
        reserved = crane_geom["crane_zone_h"]
        layout = self._slot_layout(
            panel_w,
            slot_count=len(state),
            item_size=panel_w * 0.6,
            # The same total block count is present in every legal state, so
            # this keeps block width and height invariant across the video.
            max_stack=max(1, sum(len(stack) for stack in state)),
            panel_height=panel_h - reserved,
        )
        for key in ("margin_top", "pedestal_y", "slot_floor_y"):
            layout[key] = layout[key] + reserved
        layout["panel_height"] = panel_h
        return crane_geom, layout

    def _resting_crane_pose(
        self,
        layout: Dict[str, Any],
        state: Sequence[Sequence[int]],
    ) -> Tuple[float, float]:
        trolley_cx = self._trolley_resting_x(layout, state)
        col_idx = layout["slot_centers_x"].index(trolley_cx)
        stack_height = len(state[col_idx])
        top_block_y = layout["slot_floor_y"] - stack_height * (layout["item_h"] + 3)
        claw_bottom_y = top_block_y - max(8, int(round(layout["item_h"] * 0.22)))
        return trolley_cx, claw_bottom_y

    def _render_panel(
        self,
        state: Sequence[Sequence[int]],
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        crane_geom, layout = self._panel_layout(panel_w, panel_h, state)

        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        # Column rails (vertical guides) — start just below the gantry beam.
        rail_w = max(4, int(round(panel_w * 0.012)))
        for cx in layout["slot_centers_x"]:
            cx = int(round(cx))
            draw.rounded_rectangle(
                (cx - rail_w // 2, layout["margin_top"], cx + rail_w // 2, layout["pedestal_y"]),
                radius=max(2, rail_w // 2),
                fill=COLUMN_RAIL,
            )
        self._slot_pedestal(draw, layout)

        item_w = layout["item_w"]
        item_h = layout["item_h"]

        # Render stacks.
        for col_index, stack in enumerate(state):
            cx = layout["slot_centers_x"][col_index]
            for depth, block in enumerate(stack):
                if moving is not None and moving.get("hide_top", False) and\
                        depth == len(stack) - 1 and col_index == moving["src"]:
                    continue
                bottom_y = layout["slot_floor_y"] - depth * (item_h + 3)
                top_y = bottom_y - item_h
                bbox = (cx - item_w / 2, top_y, cx + item_w / 2, bottom_y)
                color = self._block_color(block)
                self._draw_block_icon(draw, bbox, color, label=str(block), radius=int(item_h * 0.26))

        # Floating block (during animation: held by the claw).
        carry_color: Optional[Tuple[int, int, int]] = None
        floating_cx: Optional[float] = None
        floating_cy: Optional[float] = None
        if moving is not None and "block" in moving and "center" in moving:
            block = moving["block"]
            cx, cy = moving["center"]
            bbox = (cx - item_w / 2, cy - item_h / 2, cx + item_w / 2, cy + item_h / 2)
            color = self._block_color(block)
            self._draw_block_icon(draw, bbox, color, label=str(block), radius=int(item_h * 0.26))
            carry_color = color
            floating_cx = cx
            floating_cy = cy

        # ------------------------------------------------------------
        # Crane glyph: gantry beam + trolley + cable + claw.
        # ------------------------------------------------------------
        if moving is not None and "trolley_cx" in moving and "claw_bottom_y" in moving:
            trolley_cx = float(moving["trolley_cx"])
            claw_bottom_y = float(moving["claw_bottom_y"])
        elif moving is not None and floating_cx is not None and floating_cy is not None:
            # Trolley follows the carried block; claw bottom sits just above
            # the block top.
            trolley_cx = floating_cx
            claw_bottom_y = floating_cy - item_h / 2 - 2
        else:
            trolley_cx, claw_bottom_y = self._resting_crane_pose(layout, state)
        self._draw_crane(
            draw,
            panel_w,
            panel_h,
            crane_geom,
            trolley_cx,
            claw_bottom_y,
            carrying=carry_color,
            item_w=item_w,
            item_h=item_h,
        )

        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(
        self,
        state: Any,
        panel_width: int,
        *,
        panel_height: Optional[int] = None,
        animation: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _animate_move(
        self,
        state: Any,
        move: Any,
        target_state: Any,
    ) -> List[Image.Image]:
        src, dst = move
        moving_block = state[src][-1]
        after_state = self._apply_move(state, move)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        crane_geom, layout = self._panel_layout(panel_w, panel_h, state)
        _, after_layout = self._panel_layout(panel_w, panel_h, after_state)
        item_h = layout["item_h"]
        src_cx = layout["slot_centers_x"][src]
        dst_cx = layout["slot_centers_x"][dst]
        src_top_depth = len(state[src]) - 1
        dst_top_depth = len(state[dst])
        src_y = layout["slot_floor_y"] - src_top_depth * (item_h + 3) - item_h / 2
        dst_y = layout["slot_floor_y"] - dst_top_depth * (item_h + 3) - item_h / 2
        left_col = min(src, dst)
        right_col = max(src, dst)
        obstacle_tops = []
        for col in range(left_col, right_col + 1):
            remaining_height = len(state[col]) - (1 if col == src else 0)
            obstacle_tops.append(
                layout["slot_floor_y"] - remaining_height * (item_h + 3)
            )
        minimal_clearance = min(obstacle_tops) - item_h / 2 - 6
        travel_claw_y = crane_geom["gantry_bot"] + max(10, int(round(item_h * 0.35)))
        source_grip_y = src_y - item_h / 2 - 2
        destination_grip_y = dst_y - item_h / 2 - 2
        start_trolley_x, start_claw_y = self._resting_crane_pose(layout, state)
        end_trolley_x, end_claw_y = self._resting_crane_pose(after_layout, after_state)

        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        frames: List[Image.Image] = []

        def append_crane_phase(
            base_state: Sequence[Sequence[int]],
            trolley_start: float,
            trolley_end: float,
            claw_start: float,
            claw_end: float,
        ) -> None:
            for k in range(1, 4):
                trolley_x, claw_y = self._interpolate_pixel_linear(
                    (trolley_start, claw_start), (trolley_end, claw_end), k / 3
                )
                left_image = self._render_panel(
                    base_state,
                    panel_w,
                    panel_h,
                    moving={"trolley_cx": trolley_x, "claw_bottom_y": claw_y},
                )
                frames.append(self._compose_scene(left_image, right_image))

        def append_block_phase(
            block_start: Tuple[float, float],
            block_end: Tuple[float, float],
        ) -> None:
            for k in range(1, 4):
                cx, cy = self._interpolate_pixel_linear(block_start, block_end, k / 3)
                left_image = self._render_panel(
                    state,
                    panel_w,
                    panel_h,
                    moving={
                        "block": moving_block,
                        "center": (cx, cy),
                        "src": src,
                        "hide_top": True,
                        "trolley_cx": cx,
                        "claw_bottom_y": cy - item_h / 2 - 2,
                    },
                )
                frames.append(self._compose_scene(left_image, right_image))

        append_crane_phase(state, start_trolley_x, start_trolley_x, start_claw_y, travel_claw_y)
        append_crane_phase(state, start_trolley_x, src_cx, travel_claw_y, travel_claw_y)
        append_crane_phase(state, src_cx, src_cx, travel_claw_y, source_grip_y)
        append_block_phase((src_cx, src_y), (src_cx, minimal_clearance))
        append_block_phase((src_cx, minimal_clearance), (dst_cx, minimal_clearance))
        append_block_phase((dst_cx, minimal_clearance), (dst_cx, dst_y))
        append_crane_phase(after_state, dst_cx, dst_cx, destination_grip_y, travel_claw_y)
        append_crane_phase(after_state, dst_cx, end_trolley_x, travel_claw_y, end_claw_y)
        if frames:
            frames[-1] = self._render_static_scene(after_state, target_state)
        return frames

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side construction yards on a beige canvas. Each yard has a "
            "flat overhead rail spanning the top, with a compact red trolley, one cable, and a simple yellow U-clamp. "
            "Below the rail, several quiet vertical guides hold stacks of colored "
            "numbered blocks. The left yard shows the starting arrangement; the right yard shows the desired final "
            "arrangement. A blue chevron arrow between the two yards points from left to right."
        )

    def _describe_rules(self) -> str:
        return (
            "Each move uses an overhead crane to pick up the topmost block of any non-empty column and place it on top "
            "of any other column. Columns are numbered 1, 2, 3, ... from left to right. Block stacking has unlimited "
            "vertical capacity and there are no size constraints between blocks."
        )

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _column_phrase(stack: Sequence[int]) -> str:
        if not stack:
            return "empty"
        if len(stack) == 1:
            return f"holding just block {stack[0]}"
        return f"holding {', '.join(str(block) for block in stack)} bottom to top"

    def _yard_readout(self, state: Sequence[Sequence[int]]) -> str:
        return "; ".join(
            f"column {index} is {self._column_phrase(stack)}"
            for index, stack in enumerate(state, start=1)
        )

    def _author_observation(self, initial_state, target_state, moves) -> str:
        mismatched = [
            index + 1
            for index, (mine, wanted) in enumerate(zip(initial_state, target_state))
            if list(mine) != list(wanted)
        ]
        if len(mismatched) == 1:
            mismatch_fact = f"only column {mismatched[0]} differs from the target"
        else:
            listing = ", ".join(str(column) for column in mismatched)
            mismatch_fact = f"columns {listing} still differ from the target"
        return (
            f"In the left yard, {self._yard_readout(initial_state)}. "
            f"The right yard wants this instead: {self._yard_readout(target_state)}. "
            f"The crane can only lift a column's exposed top block, and right now {mismatch_fact}, "
            "so the buried misplaced blocks have to be uncovered before they can travel."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        src, dst = int(move[0]), int(move[1])
        visible_src, visible_dst = to_visible_move(self.TASK_NAME, (src, dst))
        block = previous_state[src][-1]
        settled = (
            len(next_state[dst]) <= len(target_state[dst])
            and list(target_state[dst][: len(next_state[dst])]) == list(next_state[dst])
        )
        uncovered = list(previous_state[src][:-1])
        if settled and target_state[dst]:
            purpose = "which is exactly where the target stacks it"
        elif uncovered:
            purpose = f"which uncovers block {uncovered[-1]}"
        else:
            purpose = "parking it out of the way for now"
        return (
            f"The crane lifts block {block} from the top of column {visible_src} and sets it on "
            f"column {visible_dst}, {purpose}. Now {self._yard_readout(next_state)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Every column now matches the right yard, and all {len(moves)} crane lifts took only "
            "exposed top blocks."
        )

    def _describe_output(self) -> str:
        return (
            "Return the shortest valid sequence of crane moves that turns the left yard into the right yard. Format "
            "the answer as `Answer: [(<source_column>, <destination_column>), ...].`"
        )

    def _format_move(self, move: Any) -> str:
        """Publish one internal column pair with the visible one-based numbering."""

        src, dst = to_visible_move(self.TASK_NAME, (int(move[0]), int(move[1])))
        return f"({src}, {dst})"

    def _extra_payload(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> Optional[Dict[str, Any]]:
        return {
            "column_count": len(initial_state),
            "block_count": sum(len(s) for s in initial_state),
            "optimal_moves": [list(m) for m in moves],
        }


__all__ = ["CraneBlockStackGenerator"]
