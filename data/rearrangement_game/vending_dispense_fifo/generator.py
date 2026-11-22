"""Single-chute FIFO dispensing with tray-to-chute reinsertion.

All distinct numbered cans begin in the upper chute.  A legal move either
dispenses the leftmost (front) can to the right end of the lower delivered
tray, or removes one chosen can from the delivered tray and reinserts it at
the rightmost (back) end of the chute.  The target panel fixes the required
final tray order.  The submitted artifact is a shortest legal action list.
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

PANEL_BG = (236, 240, 248)
PANEL_BORDER = (188, 198, 218)
CHUTE_OUTLINE = (88, 100, 130)
CHUTE_FILL = (208, 218, 234)
TRAY_FILL = (216, 200, 168)
TRAY_RIM = (160, 138, 92)




def _bfs(
    initial_chute: Tuple[int, ...],
    target_tray: Tuple[int, ...],
    max_moves: int,
) -> Optional[List[Tuple[str, Optional[int]]]]:
    state = (tuple(initial_chute), tuple())
    target = (tuple(), tuple(target_tray))
    if state == target:
        return []
    queue = deque([(state, [])])
    seen = {state}
    while queue:
        (chute, tray), path = queue.popleft()
        if len(path) >= max_moves:
            continue
        # Dispense front of chute into tray (append to tray).
        if chute:
            new_state = (chute[1:], tray + (chute[0],))
            if new_state not in seen:
                seen.add(new_state)
                np_path = path + [("dispense", None)]
                if new_state == target:
                    return np_path
                queue.append((new_state, np_path))
        # Re-insert any can from tray to back of chute.
        for idx, can in enumerate(tray):
            new_tray = tray[:idx] + tray[idx + 1:]
            new_chute = chute + (can,)
            new_state = (new_chute, new_tray)
            if new_state not in seen:
                seen.add(new_state)
                np_path = path + [("reinsert", int(can))]
                if new_state == target:
                    return np_path
                queue.append((new_state, np_path))
    return None


class VendingDispenseFifoGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "vending_dispense_fifo"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/vending_dispense_fifo"
    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        can_count_range: Tuple[int, int] = (4, 5),
        max_optimal_moves: int = 10,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.can_count_range = can_count_range
        self.max_optimal_moves = int(max_optimal_moves)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            n = self.rng.randint(*self.can_count_range)
            cans = list(range(1, n + 1))
            initial_chute = list(cans)
            self.rng.shuffle(initial_chute)
            target_tray = list(cans)
            self.rng.shuffle(target_tray)
            if tuple(initial_chute) == tuple(target_tray):
                continue
            plan = _bfs(tuple(initial_chute), tuple(target_tray), self.max_optimal_moves)
            if plan is None:
                continue
            if any(action == "reinsert" for action, _ in plan):
                return ({"chute": initial_chute, "tray": []},
                        {"chute": [], "tray": list(target_tray)},
                        plan)
        raise RuntimeError(
            "vending_dispense_fifo rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state: Any) -> Any:
        return (tuple(state["chute"]), tuple(state["tray"]))

    def _serialize_state(self, state: Any) -> Any:
        if isinstance(state, dict):
            return {"chute": list(state["chute"]), "tray": list(state["tray"])}
        return super()._serialize_state(state)

    def _apply_move(self, state: Any, move: Any) -> Any:
        action, value = move
        chute = list(state["chute"])
        tray = list(state["tray"])
        if action == "dispense":
            tray.append(chute.pop(0))
        elif action == "reinsert":
            tray.remove(int(value))
            chute.append(int(value))
        else:
            raise ValueError(f"Unknown action: {action}")
        return {"chute": chute, "tray": tray}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _can_color(self, cid: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(cid - 1) % len(ITEM_PALETTE)]

    def _layout(self, panel_w: int, panel_h: int, n: int) -> Dict[str, Any]:
        margin_x = max(8, int(panel_w * 0.05))
        chute_y = int(panel_h * 0.30)
        tray_y = int(panel_h * 0.72)
        usable_w = panel_w - 2 * margin_x
        slot_w = usable_w / max(4, n + 1)
        can_w = min(slot_w * 0.9, panel_w * 0.18)
        can_h = can_w * 1.4
        return {
            "panel_w": panel_w,
            "panel_h": panel_h,
            "margin_x": margin_x,
            "chute_y": chute_y,
            "tray_y": tray_y,
            "slot_w": slot_w,
            "can_w": can_w,
            "can_h": can_h,
        }

    def _draw_can(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        can_w: float,
        can_h: float,
        cid: int,
    ) -> None:
        cx, cy = center
        color = self._can_color(cid)
        outline_color = _shade(color, 0.55)
        x0 = cx - can_w / 2
        x1 = cx + can_w / 2
        y0 = cy - can_h / 2
        y1 = cy + can_h / 2
        draw.rounded_rectangle((x0, y0, x1, y1), radius=int(can_w * 0.22), fill=color, outline=outline_color, width=2)
        # Top rim ellipse.
        draw.ellipse((x0, y0 - 2, x1, y0 + can_h * 0.18), fill=_shade(color, 0.85), outline=outline_color, width=2)
        # Pull tab.
        tab_w = can_w * 0.4
        draw.ellipse(
            (cx - tab_w / 2, y0 + can_h * 0.04, cx + tab_w / 2, y0 + can_h * 0.10),
            fill=_shade(color, 0.7),
        )
        # Label.
        text_color = (32, 30, 26) if (sum(color) > 380) else (250, 248, 244)
        font_size = max(11, int(can_w * 0.55))
        font = self._draw_font(draw, font_size)
        draw.text((cx, cy + can_h * 0.08), str(cid), font=font, fill=text_color, anchor="mm")

    def _render_panel(
        self,
        state: Dict[str, Any],
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        n = len(state["chute"]) + len(state["tray"])
        layout = self._layout(panel_w, panel_h, max(1, n))
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        # Chute (upper rectangle, glass-styled).
        margin_x = layout["margin_x"]
        chute_w = panel_w - 2 * margin_x
        chute_y = layout["chute_y"]
        chute_h = layout["can_h"] + 16
        draw.rounded_rectangle(
            (margin_x, chute_y - chute_h / 2, margin_x + chute_w, chute_y + chute_h / 2),
            radius=10,
            fill=CHUTE_FILL,
            outline=CHUTE_OUTLINE,
            width=2,
        )
        # Tray (lower rectangle, wood-styled).
        tray_y = layout["tray_y"]
        tray_h = layout["can_h"] + 18
        draw.rounded_rectangle(
            (margin_x, tray_y - tray_h / 2, margin_x + chute_w, tray_y + tray_h / 2),
            radius=8,
            fill=TRAY_FILL,
            outline=TRAY_RIM,
            width=2,
        )
        # Front-of-chute arrow indicator.
        front_x = margin_x + 12
        draw.polygon(
            [
                (front_x, chute_y - chute_h / 2 + 4),
                (front_x + 10, chute_y - chute_h / 2 + 4),
                (front_x + 5, chute_y - chute_h / 2 - 6),
            ],
            fill=(60, 80, 130),
        )

        # Position cans in chute (front=left). Animation may provide
        # interpolated centers so stationary cans close gaps smoothly rather
        # than jumping or overlapping the moving can at the end of a move.
        chute_n = len(state["chute"])
        slot_w = chute_w / max(1, max(chute_n, 1))
        chute_centers = moving.get("chute_centers") if moving is not None else None
        for idx, cid in enumerate(state["chute"]):
            if moving is not None and moving.get("hide_chute_idx") == idx:
                continue
            cx = (
                float(chute_centers[idx])
                if isinstance(chute_centers, (list, tuple)) and idx < len(chute_centers)
                else margin_x + slot_w * (idx + 0.5)
            )
            self._draw_can(draw, (cx, chute_y), layout["can_w"], layout["can_h"], int(cid))

        # Position cans in tray (left=delivered first).
        tray_n = len(state["tray"])
        if tray_n > 0:
            slot_w = chute_w / max(1, tray_n)
            tray_centers = moving.get("tray_centers") if moving is not None else None
            for idx, cid in enumerate(state["tray"]):
                if moving is not None and moving.get("hide_tray_idx") == idx:
                    continue
                cx = (
                    float(tray_centers[idx])
                    if isinstance(tray_centers, (list, tuple)) and idx < len(tray_centers)
                    else margin_x + slot_w * (idx + 0.5)
                )
                self._draw_can(draw, (cx, tray_y), layout["can_w"], layout["can_h"], int(cid))

        # Floating moving can.
        if moving is not None and "can_id" in moving and "center" in moving:
            cx, cy = moving["center"]
            self._draw_can(draw, (cx, cy), layout["can_w"], layout["can_h"], int(moving["can_id"]))

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
        action, value = move
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        n = len(state["chute"]) + len(state["tray"])
        layout = self._layout(panel_w, panel_h, max(1, n))
        margin_x = layout["margin_x"]
        chute_w = panel_w - 2 * margin_x
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        next_state = self._apply_move(state, move)
        frames: List[Image.Image] = []

        def centers(count: int) -> List[float]:
            if count <= 0:
                return []
            width = chute_w / count
            return [margin_x + width * (index + 0.5) for index in range(count)]

        old_chute_centers = centers(len(state["chute"]))
        old_tray_centers = centers(len(state["tray"]))
        new_chute_centers = centers(len(next_state["chute"]))
        new_tray_centers = centers(len(next_state["tray"]))

        def interpolated_centers(
            new_items: Sequence[int],
            old_items: Sequence[int],
            old_positions: Sequence[float],
            new_positions: Sequence[float],
            progress: float,
        ) -> List[float]:
            result: List[float] = []
            for index, can_id in enumerate(new_items):
                new_x = float(new_positions[index])
                if can_id in old_items:
                    old_x = float(old_positions[list(old_items).index(can_id)])
                    result.append(old_x + (new_x - old_x) * progress)
                else:
                    result.append(new_x)
            return result

        if action == "dispense":
            cid = int(state["chute"][0])
            src = (old_chute_centers[0], layout["chute_y"])
            dst = (new_tray_centers[-1], layout["tray_y"])
            for step in range(1, self.move_frames + 1):
                t = step / self.move_frames
                cx, cy = self._interpolate_pixel_linear(src, dst, t)
                anim = {
                    "hide_tray_idx": len(next_state["tray"]) - 1,
                    "chute_centers": interpolated_centers(
                        next_state["chute"], state["chute"], old_chute_centers, new_chute_centers, t
                    ),
                    "tray_centers": interpolated_centers(
                        next_state["tray"], state["tray"], old_tray_centers, new_tray_centers, t
                    ),
                    "can_id": cid,
                    "center": (cx, cy),
                }
                left = self._render_panel(next_state, panel_w, panel_h, moving=anim)
                frames.append(self._compose_scene(left, right_image))
        elif action == "reinsert":
            cid = int(value)
            tray_idx = state["tray"].index(cid)
            src = (old_tray_centers[tray_idx], layout["tray_y"])
            dst = (new_chute_centers[-1], layout["chute_y"])
            for step in range(1, self.move_frames + 1):
                t = step / self.move_frames
                cx, cy = self._interpolate_pixel_linear(src, dst, t)
                anim = {
                    "hide_chute_idx": len(next_state["chute"]) - 1,
                    "chute_centers": interpolated_centers(
                        next_state["chute"], state["chute"], old_chute_centers, new_chute_centers, t
                    ),
                    "tray_centers": interpolated_centers(
                        next_state["tray"], state["tray"], old_tray_centers, new_tray_centers, t
                    ),
                    "can_id": cid,
                    "center": (cx, cy),
                }
                left = self._render_panel(next_state, panel_w, panel_h, moving=anim)
                frames.append(self._compose_scene(left, right_image))
        return frames

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _row_phrase(values) -> str:
        return ", ".join(str(v) for v in values) if values else "empty"

    def _author_observation(self, initial_state, target_state, moves) -> str:
        return (
            f"The chute queues cans {self._row_phrase(initial_state['chute'])} from front to back, and the "
            f"tray must finish as {self._row_phrase(target_state['tray'])} left to right. Only the chute "
            "front can dispense, always landing at the tray's right end, and the only correction is to send "
            "a tray can back to the chute tail, so out-of-order cans have to take that longer loop."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        action = str(move[0])
        # A dispense move stores no explicit can (the chute front is implied).
        can = int(previous_state["chute"][0]) if action == "dispense" else int(move[1])
        if action == "dispense":
            position = len(next_state["tray"])
            wanted = list(target_state["tray"])[: position]
            if list(next_state["tray"]) == wanted:
                purpose = "exactly the next can the target tray needs"
            else:
                purpose = "clearing the front even though it will need the return loop later"
            detail = f"Can {can} at the chute front dispenses to the tray's right end, {purpose}"
        else:
            detail = (
                f"Can {can} leaves the tray and rejoins the chute at the tail, so the cans behind it can "
                "reach the tray first"
            )
        return (
            f"{detail}. Chute: {self._row_phrase(next_state['chute'])}; tray: "
            f"{self._row_phrase(next_state['tray'])}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"The tray now reads exactly the target order with the chute empty, after {len(moves)} "
            "dispense and reinsert actions."
        )

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side vending stations on a pale-blue canvas. Each station has an "
            "upper glass chute holding a left-to-right queue of numbered cans and a lower wooden tray showing the cans "
            "that have been delivered. A small dark-blue triangular marker at the left end of each chute marks its "
            "front. "
            "The left station shows the starting state and the right station shows the desired final state. A blue "
            "chevron arrow between the two stations points from left to right."
        )

    def _describe_rules(self) -> str:
        return (
            "Every numbered can is distinct. Each move is exactly one of: DISPENSE removes the leftmost can from the "
            "chute and appends it to the right end of the delivered tray; REINSERT removes one chosen can that is "
            "currently in the delivered tray and appends it to the right (back) end of the chute. A DISPENSE is illegal "
            "when the chute is empty, and a REINSERT is illegal when the named can is not currently in the tray. Both "
            "moves preserve the relative order of all cans they do not move. No can may be created, duplicated, or "
            "discarded. The final chute and tray must exactly match the right-hand target station."
        )

    def _describe_output(self) -> str:
        return (
            "Return the shortest move sequence that empties the chute and produces the target tray order. Format the "
            "answer as exactly `Answer: [(action, value), ...].` with no extra text. Each action and value must be a "
            "two-field Python-style tuple. Use `(\"dispense\", 0)` for DISPENSE and "
            "`(\"reinsert\", can_id)` for REINSERT, where `can_id` is a positive integer printed on a can. Any legal "
            "shortest sequence is accepted if more than one exists."
        )

    def _build_prompts(self) -> Dict[str, str]:
        """Use the family-owned natural three-section prompt contract."""
        return super()._build_prompts()

    def _format_move(self, move: Any) -> str:
        action, value = move
        return f"(\"{action}\", {int(value) if value is not None else 0})"

    def _extra_payload(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> Optional[Dict[str, Any]]:
        state = {"chute": list(initial_state["chute"]), "tray": list(initial_state["tray"])}
        stack_trace: List[List[List[int]]] = [
            [list(state["chute"]), list(state["tray"])]
        ]
        for move in moves:
            state = self._apply_move(state, move)
            stack_trace.append([list(state["chute"]), list(state["tray"])])
        return {
            "can_count": len(initial_state["chute"]),
            "optimal_moves": [[m[0], int(m[1]) if m[1] is not None else 0] for m in moves],
            # The shared stack video verifier consumes an ordered list of
            # stack-like lanes. The schema label keeps that compact form
            # unambiguous while named initial/target states remain available
            # to the task-specific text evaluator.
            "state_trace": stack_trace,
            "state_trace_stack_order": ["chute", "tray"],
        }


__all__ = ["VendingDispenseFifoGenerator"]
