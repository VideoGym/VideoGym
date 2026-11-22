"""Pancake-flip rearrangement.

A single stack of N labelled pancakes (4..6) rests on a plate. Each move
selects a top-K prefix of the stack and flips that prefix in place. The
solver must reorder the starting stack into the target stack with the
shortest sequence of flips.

The video rotates the selected stack segment through 180 degrees around its
geometric centre while keeping every pancake's thickness, width, shape, and
color constant. This makes the reversal readable without squashing the
pancakes or collapsing every centre onto one point.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    SlotStackMixin,
    _shade,
)

PANEL_BG = (252, 247, 232)
PANEL_BORDER = (220, 198, 152)
PLATE_COLOR = (162, 132, 92)
PLATE_RIM = (118, 90, 56)


def _flip_prefix(stack: Tuple[int, ...], k: int) -> Tuple[int, ...]:
    if k <= 1 or k > len(stack):
        return stack
    # Top-K prefix is the LAST K elements (top of stack = end of list).
    body = list(stack)
    body[-k:] = list(reversed(body[-k:]))
    return tuple(body)


def _bfs_optimal(
    initial: Tuple[int, ...],
    target: Tuple[int, ...],
    max_moves: int,
) -> Optional[List[int]]:
    if initial == target:
        return []
    frontiers = [{initial}, {target}]
    parents = [{initial: None}, {target: None}]
    depths = [0, 0]
    while all(frontiers) and sum(depths) < max_moves:
        side = 0 if len(frontiers[0]) <= len(frontiers[1]) else 1
        following = set()
        for state in sorted(frontiers[side]):
            for k in range(2, len(state) + 1):
                nxt = _flip_prefix(state, k)
                if nxt in parents[side]:
                    continue
                parents[side][nxt] = (state, k)
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
                        right.append(move)
                    return left[::-1] + right
        frontiers[side] = following
        depths[side] += 1
    return None


class PancakeFlipGenerator(SlotStackMixin, RearrangementObjectPlanGenerator):
    TASK_NAME = "pancake_flip"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/pancake_flip"
    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        pancake_count_range: Tuple[int, int] = (4, 6),
        min_optimal_moves: int = 2,
        max_optimal_moves: int = 4,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.pancake_count_range = pancake_count_range
        self.min_optimal_moves = int(min_optimal_moves)
        self.max_optimal_moves = int(max_optimal_moves)
        if self.min_optimal_moves < 1 or self.max_optimal_moves < self.min_optimal_moves:
            raise ValueError("Require 1 <= min_optimal_moves <= max_optimal_moves for pancake_flip.")

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            n = self.rng.randint(*self.pancake_count_range)
            ids = list(range(1, n + 1))
            initial = list(ids)
            self.rng.shuffle(initial)
            target = list(ids)
            self.rng.shuffle(target)
            if tuple(initial) == tuple(target):
                continue
            plan = _bfs_optimal(tuple(initial), tuple(target), self.max_optimal_moves)
            if plan is None:
                continue
            if self.min_optimal_moves <= len(plan) <= self.max_optimal_moves:
                return tuple(initial), tuple(target), plan
        raise RuntimeError(
            "pancake_flip rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state: Any) -> Any:
        return tuple(state)

    def _apply_move(self, state: Any, move: Any) -> Any:
        k = int(move)
        if k < 2 or k > len(state):
            raise ValueError(f"invalid pancake prefix length {k}")
        return _flip_prefix(tuple(state), k)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _pancake_color(self, pid: int, n: int) -> Tuple[int, int, int]:
        # Use a graded warm-yellow gradient indexed by pancake id.
        base = ITEM_PALETTE[(pid - 1) % len(ITEM_PALETTE)]
        return base

    def _layout(self, panel_w: int, panel_h: int, n: int) -> Dict[str, Any]:
        layout = self._slot_layout(
            panel_w,
            slot_count=1,
            item_size=panel_w * 0.78,
            max_stack=n + 1,
            panel_height=panel_h,
        )
        cx = layout["slot_centers_x"][0]
        # We override item_w to be wider — pancakes go wider than tall.
        max_w = panel_w * 0.78
        # Item height tuned so all n pancakes plus the plate fit.
        usable_h = layout["slot_floor_y"] - layout["margin_top"]
        item_h = usable_h / (n + 1)
        item_h = max(14.0, min(item_h, 40.0))
        # Pancakes have varied widths according to their size id (smallest = 1, largest = n)
        # so the visual is unmistakably a pancake stack.
        layout["item_h"] = item_h
        layout["max_item_w"] = max_w
        layout["cx"] = cx
        return layout

    def _pancake_width(self, pid: int, n: int, max_w: float) -> float:
        if n <= 1:
            return max_w * 0.6
        t = (pid - 1) / (n - 1)
        return max_w * (0.42 + 0.58 * t)

    def _draw_rotated_pancake(
        self,
        image: Image.Image,
        *,
        center: Tuple[float, float],
        width: float,
        height: float,
        color: Tuple[int, int, int],
        angle_degrees: float,
        factor: int = 1,
    ) -> None:
        """Draw one fixed-size pancake as a rigidly rotated sprite.

        ``center``, ``width``, and ``height`` stay in unscaled panel
        coordinates; the sprite is rasterized and pasted at ``factor`` times
        that size on the supersampled canvas.
        """

        padding = 4 * factor
        sprite_w = max(4, int(round(width * factor))) + 2 * padding
        sprite_h = max(4, int(round(height * factor))) + 2 * padding
        sprite = Image.new("RGBA", (sprite_w, sprite_h), (0, 0, 0, 0))
        sprite_draw = ImageDraw.Draw(sprite)
        sprite_draw.rounded_rectangle(
            (padding, padding, sprite_w - padding - 1, sprite_h - padding - 1),
            radius=max(2, int(height * 0.45 * factor)),
            fill=color + (255,),
            outline=_shade(color, 0.55) + (255,),
            width=2 * factor,
        )
        rotated = sprite.rotate(
            angle_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
        x = int(round(center[0] * factor - rotated.width / 2))
        y = int(round(center[1] * factor - rotated.height / 2))
        image.paste(rotated, (x, y), rotated)

    def _render_panel(
        self,
        state: Sequence[int],
        panel_w: int,
        panel_h: int,
        *,
        flipping: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        n = len(state)
        layout = self._layout(panel_w, panel_h, n)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        # Draw plate.
        cx = layout["cx"]
        plate_y = layout["pedestal_y"] + layout["pedestal_h"] // 2
        plate_w = panel_w * 0.85
        plate_h = layout["pedestal_h"] + 4
        draw.ellipse(
            (cx - plate_w / 2, plate_y - plate_h, cx + plate_w / 2, plate_y + plate_h),
            fill=PLATE_COLOR,
            outline=PLATE_RIM,
            width=2,
        )

        item_h = layout["item_h"]
        max_w = layout["max_item_w"]

        # Render the stack from bottom (state[0]) to top (state[-1]).
        if flipping is None:
            for depth, pid in enumerate(state):
                w = self._pancake_width(pid, n, max_w)
                bottom_y = layout["slot_floor_y"] - depth * (item_h + 1)
                top_y = bottom_y - item_h
                bbox = (cx - w / 2, top_y, cx + w / 2, bottom_y)
                color = self._pancake_color(pid, n)
                self._draw_block_icon(draw, bbox, color, label=None, radius=int(item_h * 0.45))
        else:
            k = int(flipping["k"])
            progress = float(flipping["progress"])
            # Bottom (n - k) pancakes are static.
            for depth in range(n - k):
                pid = state[depth]
                w = self._pancake_width(pid, n, max_w)
                bottom_y = layout["slot_floor_y"] - depth * (item_h + 1)
                top_y = bottom_y - item_h
                bbox = (cx - w / 2, top_y, cx + w / 2, bottom_y)
                color = self._pancake_color(pid, n)
                self._draw_block_icon(draw, bbox, color, label=None, radius=int(item_h * 0.45))
            # Top K pancakes rotate around the selected segment's geometric
            # centre. Each pancake is rendered as a rigid sprite, so width,
            # thickness, shape, and color never change during the flip.
            top_segment = list(state[-k:])
            segment_centers = [
                layout["slot_floor_y"]
                - ((n - k) + offset) * (item_h + 1)
                - item_h / 2
                for offset in range(k)
            ]
            segment_cy = sum(segment_centers) / len(segment_centers)
            angle = math.pi * progress
            for offset, pid in enumerate(top_segment):
                source_cy = segment_centers[offset]
                radial_y = source_cy - segment_cy
                # A half-scale horizontal orbit keeps the rotating sprites
                # inside the narrow panel at intermediate angles.
                pancake_cx = cx + 0.5 * radial_y * math.sin(angle)
                cy = segment_cy + radial_y * math.cos(angle)
                w = self._pancake_width(pid, n, max_w)
                color = self._pancake_color(pid, n)
                if progress >= 1.0 - 1e-9:
                    bbox = (
                        pancake_cx - w / 2,
                        cy - item_h / 2,
                        pancake_cx + w / 2,
                        cy + item_h / 2,
                    )
                    self._draw_block_icon(
                        draw,
                        bbox,
                        color,
                        label=None,
                        radius=int(item_h * 0.45),
                    )
                else:
                    self._draw_rotated_pancake(
                        image,
                        center=(pancake_cx, cy),
                        width=w,
                        height=item_h,
                        color=color,
                        angle_degrees=math.degrees(angle),
                        factor=draw.factor,
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
        return self._render_panel(state, panel_width, ph, flipping=animation)

    def _animate_move(
        self,
        state: Any,
        move: Any,
        target_state: Any,
    ) -> List[Image.Image]:
        k = int(move)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        right_image = self._render_panel(target_state, panel_w, panel_h, flipping=None)
        frames: List[Image.Image] = []
        for step in range(1, self.move_frames + 1):
            t = step / self.move_frames
            anim = {"k": k, "progress": t}
            left_image = self._render_panel(state, panel_w, panel_h, flipping=anim)
            frames.append(self._compose_scene(left_image, right_image))
        return frames

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side plates of stacked round pancakes. Every pancake has a unique "
            "width and matching color, so no printed size number is needed. The left plate carries the starting stack "
            "and the right plate carries the target stack. A blue "
            "chevron arrow between the plates points from left to right."
        )

    def _describe_rules(self) -> str:
        return (
            "Each move chooses an integer K (with K from 2 to the stack height) and flips the top K pancakes in place "
            "so that their order is reversed. Pancakes outside the top K stay where they are. Track each pancake by "
            "its persistent width and color across flips."
        )

    def _describe_output(self) -> str:
        return (
            "Return the shortest sequence of flips that turns the left stack into the right stack. Format the answer "
            "as `Answer: [K1, K2, ...].`, where each Ki is the decimal length of that flip in execution order."
        )

    def _format_move(self, move: Any) -> str:
        return str(int(move))

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    PANCAKE_COLOR_NAMES = ("red", "orange", "yellow", "green", "sky-blue", "blue")

    def _pancake_name(self, pid: int) -> str:
        return f"the {self.PANCAKE_COLOR_NAMES[(pid - 1) % len(self.PANCAKE_COLOR_NAMES)]} pancake"

    def _order_listing(self, state: Sequence[int]) -> str:
        return ", ".join(
            self.PANCAKE_COLOR_NAMES[(pid - 1) % len(self.PANCAKE_COLOR_NAMES)]
            for pid in state
        )

    @staticmethod
    def _matched_base(state: Sequence[int], target: Sequence[int]) -> int:
        matched = 0
        for mine, wanted in zip(state, target):
            if mine != wanted:
                break
            matched += 1
        return matched

    def _author_observation(self, initial_state, target_state, moves) -> str:
        matched = self._matched_base(initial_state, target_state)
        if matched > 1:
            base_fact = f"The bottom {matched} pancakes already agree with the target"
        elif matched == 1:
            base_fact = "Only the bottom pancake already agrees with the target"
        else:
            base_fact = "Nothing at the bottom matches yet"
        return (
            f"Reading the left plate bottom to top, the stack is {self._order_listing(initial_state)}; "
            f"the right plate wants {self._order_listing(target_state)}. "
            "A flip lifts a whole top prefix and reverses it in place while every pancake keeps its "
            f"own width and color. {base_fact}."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        k = int(move)
        prefix_before = list(previous_state[-k:])
        new_top = self._pancake_name(next_state[-1])
        before_match = self._matched_base(previous_state, target_state)
        after_match = self._matched_base(next_state, target_state)
        if after_match > before_match:
            gain = (
                f"the bottom {after_match} pancakes now agree with the target"
                if after_match > 1
                else "the bottom pancake now agrees with the target"
            )
        else:
            gain = f"{new_top} comes to the top where the next flip needs it"
        return (
            f"I flip the top {k}: the prefix {self._order_listing(prefix_before)} reverses to "
            f"{self._order_listing(list(reversed(prefix_before)))}, and {gain}. "
            f"The stack now reads {self._order_listing(next_state)} from the bottom up."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        flips = "flip" if len(moves) == 1 else "flips"
        return (
            f"After {len(moves)} {flips} the stack matches the right plate exactly, and no pancake "
            "outside a chosen prefix ever moved."
        )

    def _extra_payload(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> Optional[Dict[str, Any]]:
        return {
            "pancake_count": len(initial_state),
            "optimal_moves": [int(m) for m in moves],
        }


__all__ = ["PancakeFlipGenerator"]
