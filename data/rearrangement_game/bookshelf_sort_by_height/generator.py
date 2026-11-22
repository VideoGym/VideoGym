"""Bookshelf height sort.

A horizontal bookshelf holds 5-7 books standing upright, each at a
DIFFERENT height. The solver must sort the books from TALLEST on the
left to SHORTEST on the right. Each move SWAPS two books at chosen
positions, not necessarily adjacent. The solver returns the shortest
sequence of swap pairs.

Move notation: ``(i, j)`` swaps the books occupying two shelf slots. Solving,
replay and rendering count slots from 0; the formal answer counts them from 1,
so the leftmost slot is written as 1. That single conversion happens at the
serialization boundary in :meth:`_format_move` and, for a submitted answer, in
the evaluator.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    _shade,
)
from data.rearrangement_game.core.answer_indexing import to_visible_move


def minimum_swap_plan(
    initial: Tuple[int, ...], target: Tuple[int, ...], max_moves: int
) -> Optional[List[Tuple[int, int]]]:
    if initial == target:
        return []
    if len(initial) != len(target) or set(initial) != set(target):
        return None
    current = list(initial)
    positions = {value: index for index, value in enumerate(current)}
    plan = []
    for index, wanted in enumerate(target):
        if current[index] == wanted:
            continue
        other = positions[wanted]
        positions[current[index]] = other
        current[index], current[other] = current[other], current[index]
        positions[wanted] = index
        plan.append((index, other))
    return plan if len(plan) <= max_moves else None


class BookshelfSortByHeightGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "bookshelf_sort_by_height"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/bookshelf_sort_by_height"

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        book_count_range: Tuple[int, int] = (5, 6),
        min_optimal_moves: int = 2,
        max_optimal_moves: int = 4,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.book_count_range = book_count_range
        self.min_optimal_moves = int(min_optimal_moves)
        self.max_optimal_moves = int(max_optimal_moves)
        if self.book_count_range[0] < 2 or self.book_count_range[1] < self.book_count_range[0]:
            raise ValueError("Require 2 <= min_book_count <= max_book_count.")
        if self.min_optimal_moves < 1 or self.max_optimal_moves < self.min_optimal_moves:
            raise ValueError("Require 1 <= min_optimal_moves <= max_optimal_moves.")

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            n = self.rng.randint(*self.book_count_range)
            heights = list(range(1, n + 1))
            initial = list(heights)
            self.rng.shuffle(initial)
            target = sorted(heights, reverse=True)
            if tuple(initial) == tuple(target):
                continue
            plan = minimum_swap_plan(
                tuple(initial), tuple(target), self.max_optimal_moves
            )
            if plan is None:
                continue
            if self.min_optimal_moves <= len(plan) <= self.max_optimal_moves:
                return initial, target, plan
        raise RuntimeError(
            "bookshelf_sort_by_height rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state: Any) -> Any:
        return tuple(state)

    def _apply_move(self, state: Any, move: Any) -> Any:
        if not isinstance(move, (tuple, list)) or len(move) != 2:
            raise ValueError(f"invalid bookshelf swap {move!r}")
        i, j = (int(move[0]), int(move[1]))
        if not (0 <= i < len(state) and 0 <= j < len(state)) or i == j:
            raise ValueError(f"invalid bookshelf swap ({i}, {j})")
        new_state = list(state)
        new_state[i], new_state[j] = new_state[j], new_state[i]
        return new_state

    def _book_color(self, h: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(h - 1) % len(ITEM_PALETTE)]

    def _layout(self, panel_w: int, panel_h: int, n: int) -> Dict[str, Any]:
        margin_x = max(8, int(panel_w * 0.05))
        shelf_y = int(panel_h * 0.78)
        shelf_h = max(8, int(panel_h * 0.04))
        usable_w = panel_w - 2 * margin_x
        slot_w = usable_w / max(1, n)
        book_w = min(slot_w * 0.78, panel_w * 0.16)
        # Heights: shortest book is base_h, tallest is max_h.
        base_h = max(40.0, panel_h * 0.18)
        max_h = min(panel_h * 0.62, base_h * 3.2)
        return {
            "panel_w": panel_w,
            "panel_h": panel_h,
            "margin_x": margin_x,
            "shelf_y": shelf_y,
            "shelf_h": shelf_h,
            "slot_w": slot_w,
            "book_w": book_w,
            "base_h": base_h,
            "max_h": max_h,
            "n": n,
        }

    def _book_height(self, layout: Dict[str, Any], h: int) -> float:
        n = layout["n"]
        # h ranges 1..n. h=1 -> base_h; h=n -> max_h.
        t = (h - 1) / max(1, n - 1)
        return layout["base_h"] + (layout["max_h"] - layout["base_h"]) * t

    def _draw_book(self, draw: ImageDraw.ImageDraw, cx: float, bottom_y: float, book_w: float, book_h: float, height_id: int) -> None:
        x0 = cx - book_w / 2
        x1 = cx + book_w / 2
        y0 = bottom_y - book_h
        y1 = bottom_y
        color = self._book_color(height_id)
        outline_color = _shade(color, 0.55)
        draw.rectangle((x0,y0,x1,y1),fill=color,outline=outline_color,width=2)

    @staticmethod
    def _carousel_geometry(layout, source_x):
        """Map one source slot coordinate to a display angle; lengths stay rigid."""
        diameter=min(layout["panel_w"],layout["panel_h"])
        slot=(source_x-layout["margin_x"])/layout["slot_w"]-.5
        angle=-math.pi/2+2*math.pi*slot/layout["n"]
        return (layout["panel_w"]/2,layout["panel_h"]/2,diameter*.09,diameter*.31/layout["max_h"],angle)

    def _render_carousel(self,state,layout,moving):
        width,height=layout["panel_w"],layout["panel_h"]
        image,draw=self._panel_canvas(width,height,(248,248,239))
        cx,cy,base,scale,_=self._carousel_geometry(layout,layout["margin_x"]+layout["slot_w"]*.5)
        radius=min(width,height)*.43
        draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),fill=(232,243,232),outline=(151,179,159),width=2)
        for slot in range(layout["n"]):
            angle=-math.pi/2+2*math.pi*slot/layout["n"]
            draw.line((cx+base*math.cos(angle),cy+base*math.sin(angle),cx+radius*math.cos(angle),cy+radius*math.sin(angle)),fill=(193,211,194),width=1)
        def rod(source_x,height_id,tangent_offset=0.0):
            _,_,_,_,angle=self._carousel_geometry(layout,source_x)
            length=self._book_height(layout,int(height_id))*scale
            ux,uy=math.cos(angle),math.sin(angle);vx,vy=-uy,ux
            start=(cx+base*ux+tangent_offset*vx,cy+base*uy+tangent_offset*vy)
            tip=(cx+(base+length)*ux+tangent_offset*vx,cy+(base+length)*uy+tangent_offset*vy)
            color=self._book_color(int(height_id));edge=_shade(color,.55)
            thick=max(7.0,min(width,height)*.042)
            draw.line((*start,*tip),fill=edge,width=thick+3)
            draw.line((*start,*tip),fill=color,width=thick)
            head=min(9.0,length*.25)
            draw.polygon([tip,(tip[0]-head*ux+head*.6*vx,tip[1]-head*uy+head*.6*vy),(tip[0]-head*ux-head*.6*vx,tip[1]-head*uy-head*.6*vy)],fill=color,outline=edge,width=1)
        for slot,height_id in enumerate(state):
            if moving is not None and slot in moving.get("hide_idxs",[]):continue
            rod(layout["margin_x"]+layout["slot_w"]*(slot+.5),height_id)
        for token in (moving.get("floats",[]) if moving else []):
            rod(token["center"][0],token["h"],token.get("tangent_offset",0.0))
        draw.ellipse((cx-base*.85,cy-base*.85,cx+base*.85,cy+base*.85),fill=(248,248,239),outline=(112,150,135),width=2)
        draw.polygon([(cx,cy-base*.65),(cx-base*.3,cy-base*.1),(cx+base*.3,cy-base*.1)],fill=(74,118,105))
        return self._panel_at_scene_scale(image,width,height)

    def _render_panel(
        self,
        state: Sequence[int],
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        n = len(state)
        layout = self._layout(panel_w, panel_h, n)
        from data.common.prompt_paraphrase import active_profile
        if active_profile("height_columns"):
            return self._render_carousel(state,layout,moving)
        image,draw=self._panel_canvas(panel_w,panel_h,(240,230,210))
        draw.rectangle((0,0,panel_w-1,panel_h-1),outline=(200,178,132),width=2)
        # Shelf.
        margin_x = layout["margin_x"]
        shelf_y = layout["shelf_y"]
        draw.rectangle(
            (margin_x // 2, shelf_y, panel_w - margin_x // 2, shelf_y + layout["shelf_h"]),
            fill=(142,102,66),
        )
        slot_w = layout["slot_w"]
        for slot_idx, h in enumerate(state):
            if moving is not None and slot_idx in moving.get("hide_idxs", []):
                continue
            cx = margin_x + slot_w * (slot_idx + 0.5)
            book_h = self._book_height(layout, int(h))
            self._draw_book(draw, cx, shelf_y, layout["book_w"], book_h, int(h))
        for floating in (moving.get("floats", []) if moving else []):
            cx, cy = floating["center"]
            book_h = self._book_height(layout, int(floating["h"]))
            self._draw_book(draw, cx, cy + book_h / 2, layout["book_w"], book_h, int(floating["h"]))
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _animate_move(self, state, move, target_state):
        i, j = move
        n = len(state)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h, n)
        margin_x = layout["margin_x"]
        slot_w = layout["slot_w"]
        shelf_y = layout["shelf_y"]
        ax = margin_x + slot_w * (i + 0.5)
        bx = margin_x + slot_w * (j + 0.5)
        ay = shelf_y
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        h_i = state[i]
        h_j = state[j]
        ih = self._book_height(layout, int(h_i))
        jh = self._book_height(layout, int(h_j))
        frames = []
        for step in range(1, self.move_frames + 1):
            t = step / self.move_frames
            # A swap is shown literally: both books travel toward the other
            # book's original slot at fixed size. Midpoint overlap is valid
            # occlusion, not a reason to invent high/low detours.
            bi_x, bi_y = self._interpolate_pixel_linear((ax, ay), (bx, ay), t)
            bj_x, bj_y = self._interpolate_pixel_linear((bx, ay), (ax, ay), t)
            anim = {
                "hide_idxs": [i, j],
                "floats": [
                    {"h": h_i, "center": (bi_x, bi_y - ih / 2)},
                    {"h": h_j, "center": (bj_x, bj_y - jh / 2)},
                ],
            }
            from data.common.prompt_paraphrase import active_profile
            if active_profile("height_columns"):
                # Separate the exchanged rods at a shared angular midpoint.
                # This changes only their travel paths, never lengths or slots.
                clearance=min(panel_w,panel_h)*.065*math.sin(math.pi*t)
                anim["floats"][0]["tangent_offset"]=clearance
                anim["floats"][1]["tangent_offset"]=-clearance
            left = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left, right_image))
        return frames

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side wall-mounted bookshelves on a warm-cream canvas. Each "
            "shelf is a horizontal wooden plank holding upright books of visibly different heights and colors. Height "
            "and color, rather than printed numbers, identify each book. The left shelf shows the starting arrangement and the "
            "right shelf shows the desired final arrangement (heights descending from the leftmost book to the "
            "rightmost). A blue chevron arrow between the two shelves points from left to right."
        )

    def _describe_rules(self) -> str:
        return (
            "Each move SWAPS the two books occupying a chosen pair of shelf slots. Slots are numbered 1 for the "
            "leftmost slot onward, and each pair names the slots as they stand just before that swap. The two slots "
            "do NOT have to be adjacent — any pair of distinct slots can be swapped in one move."
        )

    def _describe_output(self) -> str:
        return (
            "Return the shortest sequence of swaps that produces the target descending-height arrangement (tallest "
            "leftmost, shortest rightmost). Format the answer as a Python-style list of slot pairs, "
            "`Answer: [(<left_slot>, <right_slot>), ...].`, counting the leftmost slot as 1."
        )

    def _format_move(self, move):
        # Solver and replay slots are 0-based; the formal payload is 1-based.
        left, right = to_visible_move(self.TASK_NAME, (int(move[0]), int(move[1])))
        return f"({left}, {right})"

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    BOOK_COLOR_NAMES = ("red", "orange", "yellow", "green", "sky-blue", "blue", "purple")

    def _book_name(self, height_id: int, n: int) -> str:
        color = self.BOOK_COLOR_NAMES[(height_id - 1) % len(self.BOOK_COLOR_NAMES)]
        if height_id == n:
            rank = "tallest"
        elif height_id == n - 1:
            rank = "second-tallest"
        elif height_id == 1:
            rank = "shortest"
        else:
            words = ("", "", "", "third", "fourth", "fifth", "sixth")
            rank = f"{words[n - height_id + 1]}-tallest"
        return f"the {rank} ({color}) book"

    def _shelf_reading(self, state: Sequence[int]) -> str:
        n = len(state)
        return ", then ".join(self._book_name(int(h), n) for h in state)

    def _author_observation(self, initial_state, target_state, moves) -> str:
        misplaced = [
            slot + 1
            for slot, (mine, wanted) in enumerate(zip(initial_state, target_state))
            if mine != wanted
        ]
        slots = ", ".join(str(slot) for slot in misplaced)
        return (
            f"Reading the shelf left to right, it holds {self._shelf_reading(initial_state)}. "
            f"The target shelf runs tallest to shortest, so slots {slots} still hold the wrong books. "
            "A move swaps the books in any two chosen slots, so each swap should send at least one book "
            "straight to its sorted place."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        i, j = int(move[0]), int(move[1])
        visible_left, visible_right = to_visible_move(self.TASK_NAME, (i, j))
        n = len(previous_state)
        book_a = self._book_name(int(previous_state[i]), n)
        book_b = self._book_name(int(previous_state[j]), n)
        placed = [
            slot
            for slot in (i, j)
            if next_state[slot] == target_state[slot]
        ]
        if len(placed) == 2:
            purpose = "and both land exactly in their sorted slots"
        elif placed:
            landed = self._book_name(int(next_state[placed[0]]), n)
            purpose = f"and {landed} lands in its sorted slot"
        else:
            purpose = "setting both closer to their sorted slots"
        return (
            f"Swapping slots {visible_left} and {visible_right} exchanges {book_a} with {book_b}, "
            f"{purpose}. The shelf now reads {self._shelf_reading(next_state)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        swaps = "swap" if len(moves) == 1 else "swaps"
        return (
            f"The books now descend from tallest at the left to shortest at the right, reached in "
            f"{len(moves)} {swaps}."
        )

    def _extra_payload(self, initial_state, target_state, moves):
        return {
            "book_count": len(initial_state),
            "optimal_moves": [list(m) for m in moves],
        }


__all__ = ["BookshelfSortByHeightGenerator"]
