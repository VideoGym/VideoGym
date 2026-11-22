"""Parking lot swap with one free slot.

A 1xN parking row holds N-1 numbered cars and 1 EMPTY slot (rendered as a
blank pavement slot). Each move drives any single car into the adjacent
EMPTY slot, leaving its old slot empty (slide-puzzle mechanic — moves are
restricted to neighbours of the empty slot). The solver returns the
shortest sequence of car ids whose move turns the starting layout into
the target layout.

Move notation: a single integer ``car_id`` per move.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    _shade,
)

PANEL_BG = (244, 246, 246)
PANEL_BORDER = (202, 207, 210)
PAVEMENT = (207, 214, 216)
LANE_LINE = (250, 250, 247)
EMPTY_SLOT = (232, 236, 237)


def _bfs(initial: Tuple[Optional[int], ...], target: Tuple[Optional[int], ...], max_moves: int) -> Optional[List[int]]:
    if initial == target:
        return []
    queue = deque([(initial, [])])
    seen = {initial}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_moves:
            continue
        empty_idx = state.index(None)
        for nb in (empty_idx - 1, empty_idx + 1):
            if 0 <= nb < len(state):
                car_id = state[nb]
                if car_id is None:
                    continue
                new_state = list(state)
                new_state[empty_idx], new_state[nb] = new_state[nb], None
                new_state_t = tuple(new_state)
                if new_state_t in seen:
                    continue
                seen.add(new_state_t)
                np_path = path + [int(car_id)]
                if new_state_t == target:
                    return np_path
                queue.append((new_state_t, np_path))
    return None


class ParkingLotSwapMinimalMovesGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "parking_lot_swap_minimal_moves"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/parking_lot_swap_minimal_moves"
    SCENE_OUTER_MARGIN = 8
    SCENE_CHEVRON_BAND = 26
    SCENE_CHEVRON_SCALE = 0.72

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        slot_count_range: Tuple[int, int] = (5, 6),
        max_optimal_moves: int = 8,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.slot_count_range = slot_count_range
        self.max_optimal_moves = int(max_optimal_moves)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            n = self.rng.randint(*self.slot_count_range)
            cars = list(range(1, n))  # n-1 cars
            initial = list(cars) + [None]
            self.rng.shuffle(initial)
            target = list(cars) + [None]
            self.rng.shuffle(target)
            if tuple(initial) == tuple(target):
                continue
            plan = _bfs(tuple(initial), tuple(target), self.max_optimal_moves)
            if plan is None:
                continue
            if 3 <= len(plan) <= self.max_optimal_moves:
                return initial, target, plan
        raise RuntimeError(
            "parking_lot_swap_minimal_moves rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state: Any) -> Any:
        return tuple(state)

    def _serialize_state(self, state: Any) -> Any:
        return [int(c) if c is not None else None for c in state]

    def _apply_move(self, state: Any, move: Any) -> Any:
        car_id = int(move)
        cur = list(state)
        if car_id not in cur:
            raise ValueError(f"car {car_id} not in lot")
        car_idx = cur.index(car_id)
        empty_idx = cur.index(None)
        if abs(car_idx - empty_idx) != 1:
            raise ValueError(f"car {car_id} is not adjacent to empty slot")
        cur[empty_idx], cur[car_idx] = cur[car_idx], None
        return cur

    def _car_color(self, car_id: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(car_id - 1) % len(ITEM_PALETTE)]

    def _layout(self, panel_w: int, panel_h: int, n: int) -> Dict[str, Any]:
        margin_x = max(8, int(panel_w * 0.04))
        usable_w = panel_w - 2 * margin_x
        slot_w = usable_w / max(1, n)
        car_w = slot_w * 0.92
        car_h = min(panel_h * 0.34, car_w * 2.5)
        slot_y = int((panel_h - car_h) / 2)
        return {
            "panel_w": panel_w,
            "panel_h": panel_h,
            "margin_x": margin_x,
            "slot_w": slot_w,
            "car_w": car_w,
            "car_h": car_h,
            "slot_y": slot_y,
        }

    def _draw_pavement(self, draw: ImageDraw.ImageDraw, layout: Dict[str, Any], n: int) -> None:
        margin_x = layout["margin_x"]
        slot_w = layout["slot_w"]
        slot_y = layout["slot_y"]
        car_h = layout["car_h"]
        # Pavement strip.
        x0 = margin_x
        x1 = layout["panel_w"] - margin_x
        y0 = slot_y - 4
        y1 = slot_y + car_h + 12
        draw.rectangle((x0, y0, x1, y1), fill=PAVEMENT)
        # Slot dividers.
        for i in range(1, n):
            xs = margin_x + slot_w * i
            draw.line((xs, y0 + 4, xs, y1 - 4), fill=LANE_LINE, width=2)
        # Border lines.
        draw.rectangle((x0, y0, x1, y0 + 2), fill=LANE_LINE)
        draw.rectangle((x0, y1 - 2, x1, y1), fill=LANE_LINE)

    def _draw_car(self, draw: ImageDraw.ImageDraw, cx: float, cy: float, car_w: float, car_h: float, car_id: int) -> None:
        """Draw one flat fixed-size car token with only its required identity."""

        color = self._car_color(car_id)
        outline_color = _shade(color, 0.55)
        x0 = cx - car_w / 2
        x1 = cx + car_w / 2
        y0 = cy - car_h / 2
        y1 = cy + car_h / 2
        draw.rounded_rectangle((x0, y0, x1, y1), radius=max(6, int(car_w * 0.24)),
                               fill=color, outline=outline_color, width=2)
        font_size = max(13, int(car_w * 0.42))
        font = self._draw_font(draw, font_size)
        text_color = (32, 30, 26) if (sum(color) > 380) else (250, 248, 244)
        draw.text((cx, cy), str(car_id), font=font, fill=text_color, anchor="mm")

    def _render_panel(
        self,
        state: Sequence[Optional[int]],
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        n = len(state)
        layout = self._layout(panel_w, panel_h, n)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        self._draw_pavement(draw, layout, n)
        margin_x = layout["margin_x"]
        slot_w = layout["slot_w"]
        slot_y = layout["slot_y"]
        car_w = layout["car_w"]
        car_h = layout["car_h"]
        for idx, car_id in enumerate(state):
            cx = margin_x + slot_w * (idx + 0.5)
            cy = slot_y + car_h / 2
            if moving is not None and moving.get("hide_idx") == idx:
                continue
            if car_id is None:
                draw.rounded_rectangle(
                    (cx - car_w / 2, cy - car_h / 2, cx + car_w / 2, cy + car_h / 2),
                    radius=max(6, int(car_w * 0.24)),
                    fill=EMPTY_SLOT,
                    outline=(176, 184, 188),
                    width=2,
                )
            else:
                self._draw_car(draw, cx, cy, car_w, car_h, int(car_id))
        if moving is not None and "car_id" in moving and "center" in moving:
            cx, cy = moving["center"]
            self._draw_car(draw, cx, cy, car_w, car_h, int(moving["car_id"]))
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _animate_move(self, state, move, target_state):
        car_id = int(move)
        n = len(state)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h, n)
        margin_x = layout["margin_x"]
        slot_w = layout["slot_w"]
        slot_y = layout["slot_y"]
        car_h = layout["car_h"]
        car_idx = list(state).index(car_id)
        empty_idx = list(state).index(None)
        src = (margin_x + slot_w * (car_idx + 0.5), slot_y + car_h / 2)
        dst = (margin_x + slot_w * (empty_idx + 0.5), slot_y + car_h / 2)
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        frames = []
        after = self._apply_move(state, move)
        for step in range(1, self.move_frames + 1):
            t = step / self.move_frames
            if step == self.move_frames:
                frames.append(self._render_static_scene(after, target_state))
                continue
            cx, cy = self._interpolate_pixel_linear(src, dst, t)
            anim = {"hide_idx": car_idx, "car_id": car_id, "center": (cx, cy)}
            left = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left, right_image))
        return frames

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side strip parking lots on a grey canvas. Each parking lot has "
            "a row of clean slots separated by pale lane stripes. One slot is EMPTY (shown as a pale outlined bay), "
            "the rest hold uniquely-numbered flat car tokens. The left lot shows the starting arrangement and the "
            "right lot shows the desired final arrangement. A blue chevron arrow between the lots points from left "
            "to right. Only car IDs and slot occupancy are encoded; decorative wheels, windows, lights, and shadows "
            "are omitted."
        )

    def _describe_rules(self) -> str:
        return (
            "Each move drives one car into the adjacent EMPTY slot — leaving the car's old slot empty. A car may move "
            "only when its slot is directly next to the empty slot (i.e. their slot indices differ by 1). Cars cannot "
            "jump over other cars."
        )

    def _describe_output(self) -> str:
        return (
            "Return any shortest plan as `Answer: [<car_id>, ...].`, listing the car moved into the empty slot "
            "each time."
        )

    def _format_move(self, move):
        return str(int(move))

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _row_reading(state: Sequence[Optional[int]]) -> str:
        return ", ".join("the empty bay" if c is None else f"car {c}" for c in state)

    def _author_observation(self, initial_state, target_state, moves) -> str:
        return (
            f"Reading the slots left to right, the lot holds {self._row_reading(initial_state)}; the target "
            f"lot wants {self._row_reading(target_state)}. Only a car parked directly beside the empty bay "
            "can move, and it always rolls into that bay, so the bay has to be walked along the row to let "
            "each car reach its place."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        car_id = int(move)
        car_slot = list(previous_state).index(car_id)
        empty_slot = list(previous_state).index(None)
        settled = list(target_state)[empty_slot] == car_id
        if settled:
            purpose = "which is exactly the slot the target assigns it"
        elif move_index < len(moves):
            next_car = int(moves[move_index])
            purpose = f"which brings the empty bay next to car {next_car}"
        else:
            purpose = "completing the arrangement"
        return (
            f"Car {car_id} in slot {car_slot + 1} sits beside the empty bay, so it rolls into "
            f"slot {empty_slot + 1}, {purpose}. The row now reads {self._row_reading(next_state)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Every car now occupies its target slot with the empty bay where the target shows it, after "
            f"{len(moves)} single-slot rolls."
        )

    def _extra_payload(self, initial_state, target_state, moves):
        return {
            "slot_count": len(initial_state),
            "car_count": sum(1 for c in initial_state if c is not None),
            "optimal_moves": [int(m) for m in moves],
        }


__all__ = ["ParkingLotSwapMinimalMovesGenerator"]
