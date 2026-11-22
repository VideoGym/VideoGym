"""Container ship loading-order rearrangement.

The dock holds a STACK of containers — the crane can pick up only the TOP
container (right-most in the rendered row). The ship needs to be filled
in descending port-number order from left to right (largest container
loaded first). A single quay-side BUFFER pad holds at most one container.

Because the dock is a stack, when the dock's top container is not the
next-needed one, the solver must temporarily move it to the buffer. The
buffer is therefore mechanically load-bearing.

Moves:
* ``("ship", crate_id)`` — from dock-top OR from buffer onto the ship.
* ``("buffer", crate_id)`` — from dock-top onto the empty buffer.
* ``("unbuffer", crate_id)`` — from buffer back to dock-top.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.typed_answers import format_answer_line
from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
)

PANEL_BG = (218, 232, 244)
PANEL_BORDER = (170, 196, 220)
WATER = (118, 168, 210)
HULL_FILL = (76, 96, 124)
DOCK_FILL = (148, 122, 84)
BUFFER_FILL = (210, 192, 152)
BUFFER_RIM = (170, 144, 96)


def _state_key(dock: Sequence[int], ship: Sequence[int], buf: Optional[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...], Optional[int]]:
    return (tuple(dock), tuple(ship), buf)


def _bfs(dock_init: Tuple[int, ...], n: int, max_moves: int) -> Optional[List[Tuple[str, int]]]:
    """BFS where the dock is a stack — only the right-most (top) container is accessible."""

    target_ship = tuple(sorted(dock_init, reverse=True))  # Largest loaded first.
    init = (tuple(dock_init), tuple(), None)
    target = (tuple(), target_ship, None)
    if init == target:
        return []
    queue = deque([(init, [])])
    seen = {init}
    while queue:
        (dock, ship, buf), path = queue.popleft()
        if len(path) >= max_moves:
            continue
        # Move dock-top to ship.
        if dock:
            top = dock[-1]
            new_dock = dock[:-1]
            new_ship = ship + (top,)
            new_state = (new_dock, new_ship, buf)
            if new_state not in seen:
                seen.add(new_state)
                np_path = path + [("ship", int(top))]
                if new_state == target:
                    return np_path
                queue.append((new_state, np_path))
        # Move dock-top to buffer.
        if dock and buf is None:
            top = dock[-1]
            new_dock = dock[:-1]
            new_state = (new_dock, ship, int(top))
            if new_state not in seen:
                seen.add(new_state)
                np_path = path + [("buffer", int(top))]
                queue.append((new_state, np_path))
        # Move buffer to ship.
        if buf is not None:
            new_state = (dock, ship + (buf,), None)
            if new_state not in seen:
                seen.add(new_state)
                np_path = path + [("ship", int(buf))]
                if new_state == target:
                    return np_path
                queue.append((new_state, np_path))
            # Move buffer back to dock-top.
            new_state = (dock + (buf,), ship, None)
            if new_state not in seen:
                seen.add(new_state)
                np_path = path + [("unbuffer", int(buf))]
                queue.append((new_state, np_path))
    return None


class ContainerShipLoadingOrderGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "container_ship_loading_order"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/container_ship_loading_order"

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        crate_count: Optional[int] = None,
        crate_count_range: Tuple[int, int] = (4, 5),
        max_optimal_moves: int = 8,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        if crate_count is not None:
            fixed_count = int(crate_count)
            self.crate_count_range = (fixed_count, fixed_count)
        else:
            if len(crate_count_range) != 2:
                raise ValueError("crate_count_range must contain exactly two bounds")
            self.crate_count_range = tuple(int(value) for value in crate_count_range)
        if not (4 <= self.crate_count_range[0] <= self.crate_count_range[1] <= 8):
            raise ValueError(
                "crate_count_range must stay within the inclusive 4..8 range"
            )
        self.max_optimal_moves = int(max_optimal_moves)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            n = self.rng.randint(*self.crate_count_range)
            dock = list(range(1, n + 1))
            self.rng.shuffle(dock)
            # Filter out trivially-sorted starting docks.
            if dock == sorted(dock, reverse=True):
                continue
            plan = _bfs(tuple(dock), n, self.max_optimal_moves)
            if plan is None:
                continue
            # Require buffer is used at least once for a non-trivial puzzle.
            if any(action == "buffer" for action, _ in plan) and\
               n + 1 <= len(plan) <= self.max_optimal_moves:
                init_state = {"dock": dock, "ship": [], "buffer": None}
                tgt_state = {"dock": [], "ship": sorted(dock, reverse=True), "buffer": None}
                return init_state, tgt_state, plan
        raise RuntimeError(
            "container_ship_loading_order rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state: Any) -> Any:
        return _state_key(state["dock"], state["ship"], state.get("buffer"))

    def _serialize_state(self, state: Any) -> Any:
        if isinstance(state, dict):
            return {"dock": list(state["dock"]), "ship": list(state["ship"]), "buffer": state.get("buffer")}
        return super()._serialize_state(state)

    def _apply_move(self, state: Any, move: Any) -> Any:
        action, value = move
        dock = list(state["dock"])
        ship = list(state["ship"])
        buf = state.get("buffer")
        v = int(value)
        if action == "ship":
            if buf == v:
                ship.append(buf)
                buf = None
            else:
                if not dock or dock[-1] != v:
                    raise ValueError(f"Crate {v} is not on top of the dock; dock top is {dock[-1] if dock else None}")
                dock.pop()
                ship.append(v)
        elif action == "buffer":
            if not dock or dock[-1] != v:
                raise ValueError(f"Crate {v} is not on top of the dock; dock top is {dock[-1] if dock else None}")
            if buf is not None:
                raise ValueError("Buffer already occupied")
            dock.pop()
            buf = v
        elif action == "unbuffer":
            if buf != v:
                raise ValueError(f"Buffer holds {buf}, not {v}")
            dock.append(v)
            buf = None
        else:
            raise ValueError(f"Unknown action: {action}")
        return {"dock": dock, "ship": ship, "buffer": buf}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _container_color(self, port: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(port - 1) % len(ITEM_PALETTE)]

    def _layout(self, panel_w: int, panel_h: int, n: int) -> Dict[str, Any]:
        margin_x = max(8, int(panel_w * 0.05))
        # Push the dock down toward the bottom and pull the buffer up toward
        # the top so the three zones (buffer / ship / dock) stack to fill the
        # panel vertically.
        dock_y = int(panel_h * 0.86)
        dock_h = max(8, int(panel_h * 0.04))
        ship_y = int(panel_h * 0.50)
        ship_h = max(40, int(panel_h * 0.30))
        buffer_y = int(panel_h * 0.14)
        # A single stable slot grid is shared by dock, ship, buffer, and
        # animation.  Existing containers never reflow when one is moved.
        slot_gap = max(3.0, panel_w * 0.015)
        row_avail = panel_w * 0.78
        crate_w = min(panel_w * 0.19, (row_avail - slot_gap * (n - 1)) / max(1, n))
        crate_w = max(26.0, crate_w)
        crate_h = crate_w * 0.78
        row_w = n * crate_w + max(0, n - 1) * slot_gap
        row_x0 = (panel_w - row_w) / 2
        return {
            "panel_w": panel_w,
            "panel_h": panel_h,
            "margin_x": margin_x,
            "dock_y": dock_y,
            "dock_h": dock_h,
            "ship_y": ship_y,
            "ship_h": ship_h,
            "buffer_y": buffer_y,
            "crate_w": crate_w,
            "crate_h": crate_h,
            "slot_gap": slot_gap,
            "row_x0": row_x0,
            "capacity": n,
        }

    def _slot_center(self, layout: Dict[str, Any], idx: int) -> float:
        return (
            layout["row_x0"]
            + layout["crate_w"] / 2
            + idx * (layout["crate_w"] + layout["slot_gap"])
        )

    def _dock_center(self, layout: Dict[str, Any], idx: int) -> Tuple[float, float]:
        return (self._slot_center(layout, idx), layout["dock_y"] - layout["crate_h"] / 2)

    def _ship_center(self, layout: Dict[str, Any], idx: int) -> Tuple[float, float]:
        return (
            self._slot_center(layout, idx),
            layout["ship_y"] + layout["ship_h"] * 0.27,
        )

    def _buffer_center(self, layout: Dict[str, Any]) -> Tuple[float, float]:
        return (layout["panel_w"] / 2, layout["buffer_y"])

    def _draw_container(
        self,
        draw: ImageDraw.ImageDraw,
        center_x: float,
        bottom_y: float,
        crate_w: float,
        crate_h: float,
        port: int,
    ) -> None:
        x0 = center_x - crate_w / 2
        x1 = center_x + crate_w / 2
        y0 = bottom_y - crate_h
        y1 = bottom_y
        color = self._container_color(port)
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=max(4, int(crate_h * 0.16)),
            fill=color,
            outline=(54, 58, 64),
            width=2,
        )
        # Port label.
        font = self._draw_font(draw, max(11, int(crate_w * 0.50)))
        text_color = (32, 30, 26) if (sum(color) > 380) else (250, 248, 244)
        draw.text((center_x, (y0 + y1) / 2 - 1), str(port), font=font, fill=text_color, anchor="mm")

    def _render_panel(
        self,
        state: Dict[str, Any],
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        n_total = len(state["dock"]) + len(state["ship"]) + (1 if state.get("buffer") is not None else 0)
        n_total = max(n_total, 4)
        layout = self._layout(panel_w, panel_h, n_total)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        # One flat water field; no decorative wave texture.
        margin_x = layout["margin_x"]
        water_y0 = layout["ship_y"] - 4
        water_y1 = panel_h - 4
        draw.rectangle((4, water_y0, panel_w - 4, water_y1), fill=WATER)
        # Dock platform near the bottom.
        dock_y = layout["dock_y"]
        dock_h = layout["dock_h"]
        draw.rounded_rectangle(
            (margin_x // 2, dock_y, panel_w - margin_x // 2, dock_y + dock_h),
            radius=max(3, dock_h // 3),
            fill=DOCK_FILL,
        )
        # Ship hull.
        ship_y = layout["ship_y"]
        ship_h = layout["ship_h"]
        hull_w = panel_w * 0.86
        hull_x0 = (panel_w - hull_w) / 2
        hull_x1 = hull_x0 + hull_w
        # Trapezoidal hull silhouette.
        draw.polygon(
            [
                (hull_x0, ship_y + ship_h * 0.20),
                (hull_x1, ship_y + ship_h * 0.20),
                (hull_x1 - hull_w * 0.06, ship_y + ship_h),
                (hull_x0 + hull_w * 0.06, ship_y + ship_h),
            ],
            fill=HULL_FILL,
        )
        # Buffer pad near top.
        buf_w = layout["crate_w"] * 1.4
        buf_h = layout["crate_h"] * 1.2
        buf_cx, buf_cy = self._buffer_center(layout)
        buf_y_top = buf_cy - buf_h / 2
        draw.rounded_rectangle(
            (buf_cx - buf_w / 2, buf_y_top, buf_cx + buf_w / 2, buf_y_top + buf_h),
            radius=8,
            fill=BUFFER_FILL,
            outline=BUFFER_RIM,
            width=2,
        )

        # Render crates on dock.
        for idx, crate in enumerate(state["dock"]):
            if moving is not None and moving.get("hide_dock_idx") == idx:
                continue
            cx, cy = self._dock_center(layout, idx)
            self._draw_container(
                draw, cx, cy + layout["crate_h"] / 2,
                layout["crate_w"], layout["crate_h"], int(crate)
            )

        # Render crates on ship (left = loaded first = deepest in stack).
        for idx, crate in enumerate(state["ship"]):
            cx, cy = self._ship_center(layout, idx)
            self._draw_container(
                draw, cx, cy + layout["crate_h"] / 2,
                layout["crate_w"], layout["crate_h"], int(crate)
            )

        # Render buffer.
        buf = state.get("buffer")
        if buf is not None and (moving is None or not moving.get("hide_buffer", False)):
            self._draw_container(
                draw, buf_cx, buf_cy + layout["crate_h"] / 2,
                layout["crate_w"], layout["crate_h"], int(buf)
            )

        # Floating crate.
        if moving is not None and "crate" in moving and "center" in moving:
            cx, cy = moving["center"]
            self._draw_container(draw, cx, cy + layout["crate_h"] / 2,
                                 layout["crate_w"], layout["crate_h"], int(moving["crate"]))

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
        crate = int(value)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        n_total = len(state["dock"]) + len(state["ship"]) + (1 if state.get("buffer") is not None else 0)
        n_total = max(n_total, 4)
        layout = self._layout(panel_w, panel_h, n_total)
        # Compute source.
        src: Optional[Tuple[float, float]] = None
        anim_hide: Dict[str, Any] = {}
        if state.get("buffer") == crate and action == "ship":
            # buffer -> ship.
            src = self._buffer_center(layout)
            anim_hide["hide_buffer"] = True
        else:
            # dock -> ship/buffer or buffer -> dock.
            if action == "unbuffer":
                src = self._buffer_center(layout)
                anim_hide["hide_buffer"] = True
            else:
                # dock-top -> ...
                idx = len(state["dock"]) - 1
                src = self._dock_center(layout, idx)
                anim_hide["hide_dock_idx"] = idx

        # Compute destination.
        dst: Tuple[float, float]
        if action == "ship":
            dst = self._ship_center(layout, len(state["ship"]))
        elif action == "buffer":
            dst = self._buffer_center(layout)
        else:  # unbuffer
            dst = self._dock_center(layout, len(state["dock"]))

        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        frames: List[Image.Image] = []
        for step in range(1, self.move_frames + 1):
            t = step / self.move_frames
            cx, cy = self._interpolate_pixel_linear(src, dst, t)
            anim = dict(anim_hide)
            anim["crate"] = crate
            anim["center"] = (cx, cy)
            left = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left, right_image))
        after_state = self._apply_move(state, move)
        if frames:
            frames[-1] = self._render_static_scene(after_state, target_state)
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
        dock = list(initial_state["dock"])
        order = ", ".join(str(v) for v in sorted(dock, reverse=True))
        return (
            f"The dock stack reads {self._row_phrase(dock)} from bottom to top, and only the top container "
            f"can be lifted. The ship must be loaded {order} in that exact order, largest first, and the "
            "single buffer pad is the only place to set a container aside, so out-of-order tops have to "
            "detour through it."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        action, crate = str(move[0]), int(move[1])
        if action == "ship":
            source = "the buffer" if previous_state.get("buffer") == crate else "the dock top"
            detail = (
                f"Container {crate} on {source} is exactly the next number the ship needs, so it loads; "
                f"the ship row now reads {self._row_phrase(next_state['ship'])}"
            )
        elif action == "buffer":
            below = next_state["dock"][-1] if next_state["dock"] else None
            expose = f", exposing container {below}" if below is not None else ""
            detail = (
                f"The dock top is container {crate}, but the ship is not ready for it yet, so it parks on "
                f"the buffer pad{expose}"
            )
        else:
            detail = (
                f"Container {crate} returns from the buffer to the dock top, freeing the pad for the "
                "container that must detour next"
            )
        return f"{detail}. Dock: {self._row_phrase(next_state['dock'])}; buffer: " + (
            str(next_state.get("buffer")) if next_state.get("buffer") is not None else "empty"
        ) + "."

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"The ship now holds every container in descending order with the dock and buffer both empty, "
            f"after {len(moves)} crane actions."
        )

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side maritime scenes on a sea-blue canvas. Each scene has, from "
            "bottom to top, a flat dock platform holding a horizontal row of numbered cargo containers, a simple ship hull on a flat water field that "
            "holds a row of containers in loading order from left (loaded first) to right (loaded last), and a single "
            "quay-side BUFFER pad. The left scene shows the starting "
            "state and the right scene shows the desired final state — all containers loaded onto the ship in "
            "DESCENDING port order from left to right with both the dock and the buffer empty. A blue chevron arrow "
            "between the scenes points from left to right."
        )

    def _describe_rules(self) -> str:
        return (
            "The dock is a STACK: only the right-most container of the dock is accessible. Each move is one of: (a) "
            "`(\"ship\", crate_id)` loads the dock-top container OR the buffered container onto the ship by appending "
            "it to the right of the ship's current row; (b) `(\"buffer\", crate_id)` removes the dock-top container "
            "and places it on the empty buffer; (c) `(\"unbuffer\", crate_id)` returns the buffered container back to "
            "the dock-top. The buffer can hold at most one container."
        )

    def _build_prompts(self) -> Dict[str, str]:
        """Use the family contract and reject sample-answer prompt leakage."""

        prompts = super()._build_prompts()
        moves = list(getattr(self, "_prompt_moves", []) or [])
        if moves:
            formal_answer = format_answer_line(self._format_answer(moves))
            leaked_forms = {formal_answer, formal_answer.rstrip(".")}
            for mode in ("ti2v_prompt",):
                prompt = str(prompts.get(mode) or "")
                if any(leaked and leaked in prompt for leaked in leaked_forms):
                    raise RuntimeError(
                        f"container_ship_loading_order {mode} leaks this sample's formal answer"
                    )
        return prompts

    def _describe_output(self) -> str:
        return (
            "Return the shortest sequence of moves that ends with all containers loaded onto the ship in descending "
            "port-number order from left to right. Format the answer as a Python-style list of (action, container_id) "
            "tuples using the schema `Answer: [(<action>, <container_id>), ...].`, where each `<action>` is a quoted "
            "literal from `\"ship\"`, `\"buffer\"`, or `\"unbuffer\"`, and each `<container_id>` is the positive "
            "integer printed on the moved container. Replace the metavariables with the submitted moves."
        )

    def _format_move(self, move: Any) -> str:
        action, value = move
        return f"(\"{action}\", {int(value)})"

    def _extra_payload(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> Optional[Dict[str, Any]]:
        return {
            "container_count": len(initial_state["dock"]),
            "optimal_moves": [[m[0], int(m[1])] for m in moves],
        }


__all__ = ["ContainerShipLoadingOrderGenerator"]
