"""Dock yard two-crane swap with capacity rules.

Two cranes (LEFT crane has capacity 2, RIGHT crane has capacity 1) move
heavy boxes between three numbered LANES of a freight yard. Each lane is
a vertical column of stacked numbered boxes. A move uses one of the
cranes:
* LEFT crane move: ``("L", from, to)`` — moves the TOP TWO boxes from
  the source lane onto the top of the destination lane in REVERSED order
  (because the crane's claw flips them top-to-bottom).
* RIGHT crane move: ``("R", from, to)`` — moves only the TOP BOX, no flip.

The capacity asymmetry is what makes the puzzle non-trivial: the LEFT
crane is more efficient per move (2 boxes at a time) but inverts the
order; the RIGHT crane preserves order but is slower.

The solver returns the shortest move sequence to reach the target. It
indexes the lanes 0, 1, 2 from left to right, while the formal answer
counts them 1, 2, 3; the box numbers printed on the cargo are identities
and never shift. :mod:`data.rearrangement_game.core.answer_indexing` owns
that single conversion in both directions.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    SlotStackMixin,
)
from data.rearrangement_game.core.answer_indexing import to_visible_move


def _state_key(state: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(s) for s in state)


def _bfs(
    initial: Sequence[Sequence[int]],
    target: Sequence[Sequence[int]],
    max_moves: int,
    *,
    allowed_cranes: FrozenSet[str] = frozenset({"L", "R"}),
) -> Optional[List[Tuple[str, int, int]]]:
    init_key = _state_key(initial)
    tgt_key = _state_key(target)
    if init_key == tgt_key:
        return []
    queue = deque([(init_key, [])])
    seen = {init_key}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_moves:
            continue
        for src in range(3):
            if not state[src]:
                continue
            for dst in range(3):
                if src == dst:
                    continue
                # RIGHT crane (top-1, no flip).
                if "R" in allowed_cranes:
                    top1 = state[src][-1]
                    ns_r = tuple(
                        tuple(list(state[i])[:-1]) if i == src
                        else tuple(list(state[i]) + [top1]) if i == dst
                        else state[i]
                        for i in range(3)
                    )
                    if ns_r not in seen:
                        seen.add(ns_r)
                        np = path + [("R", src, dst)]
                        if ns_r == tgt_key:
                            return np
                        queue.append((ns_r, np))
                # LEFT crane (top-2, flipped).
                if "L" in allowed_cranes and len(state[src]) >= 2:
                    a = state[src][-1]
                    b = state[src][-2]
                    ns_l = tuple(
                        tuple(list(state[i])[:-2]) if i == src
                        else tuple(list(state[i]) + [a, b]) if i == dst
                        else state[i]
                        for i in range(3)
                    )
                    if ns_l not in seen:
                        seen.add(ns_l)
                        np = path + [("L", src, dst)]
                        if ns_l == tgt_key:
                            return np
                        queue.append((ns_l, np))
    return None


class DockYardCraneSwapGenerator(SlotStackMixin, RearrangementObjectPlanGenerator):
    TASK_NAME = "dock_yard_crane_swap"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/dock_yard_crane_swap"

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        box_count_range: Tuple[int, int] = (4, 5),
        min_optimal_moves: int = 3,
        max_optimal_moves: int = 7,
        move_frames: int = 24,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            seed=seed,
            video=video,
            move_frames=move_frames,
        )
        if (
            not isinstance(box_count_range, tuple)
            or len(box_count_range) != 2
            or any(type(value) is not int for value in box_count_range)
            or not 4 <= box_count_range[0] <= box_count_range[1] <= 5
        ):
            raise ValueError("box_count_range must stay within the inclusive 4..5 contract")
        if type(min_optimal_moves) is not int or type(max_optimal_moves) is not int:
            raise ValueError("optimal-move bounds must be integers")
        if not 3 <= min_optimal_moves <= max_optimal_moves <= 7:
            raise ValueError("optimal-move bounds must stay within the inclusive 3..7 contract")
        if type(move_frames) is not int or move_frames < 9:
            raise ValueError("move_frames must be an integer of at least 9")
        self.box_count_range = box_count_range
        self.min_optimal_moves = min_optimal_moves
        self.max_optimal_moves = max_optimal_moves

    def _random_state(self, boxes: List[int]) -> List[List[int]]:
        lanes: List[List[int]] = [[], [], []]
        order = list(boxes)
        self.rng.shuffle(order)
        for box in order:
            lanes[self.rng.randrange(3)].append(box)
        return lanes

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(4000):
            n = self.rng.randint(*self.box_count_range)
            boxes = list(range(1, n + 1))
            initial = self._random_state(boxes)
            target = self._random_state(boxes)
            if _state_key(initial) == _state_key(target):
                continue
            plan = _bfs(initial, target, self.max_optimal_moves)
            if plan is None:
                continue
            if not self.min_optimal_moves <= len(plan) <= self.max_optimal_moves:
                continue
            # Both actions must be load-bearing for shortest planning. If a
            # same-length solution exists with only one crane type, the other
            # marker would be decorative rather than semantic.
            r_only = _bfs(
                initial,
                target,
                len(plan),
                allowed_cranes=frozenset({"R"}),
            )
            l_only = _bfs(
                initial,
                target,
                len(plan),
                allowed_cranes=frozenset({"L"}),
            )
            if r_only is None and l_only is None:
                return initial, target, plan
        raise RuntimeError(
            "dock_yard_crane_swap rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _apply_move(self, state: Any, move: Any) -> Any:
        if not isinstance(state, (list, tuple)) or len(state) != 3:
            raise ValueError("Dock-yard state must contain exactly three lanes")
        if any(not isinstance(stack, (list, tuple)) for stack in state):
            raise ValueError("Each dock-yard lane must be a sequence")
        if not isinstance(move, (list, tuple)) or len(move) != 3:
            raise ValueError("A dock-yard move must be a crane/source/destination triple")
        crane, src, dst = move
        if type(crane) is not str or crane not in {"L", "R"}:
            raise ValueError(f"Unknown crane {crane!r}")
        if type(src) is not int or type(dst) is not int:
            raise ValueError("Lane indices must be integers")
        if not (0 <= src < 3 and 0 <= dst < 3) or src == dst:
            raise ValueError(f"Illegal dock-yard move: {(crane, src, dst)}")
        new_state = [list(stack) for stack in state]
        if crane == "R":
            if not new_state[src]:
                raise ValueError(f"Source lane {src} is empty")
            new_state[dst].append(new_state[src].pop())
        else:
            if len(new_state[src]) < 2:
                raise ValueError(f"Source lane {src} has fewer than two boxes")
            a = new_state[src].pop()
            b = new_state[src].pop()
            new_state[dst].extend([a, b])
        return new_state

    def _box_color(self, b: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(b - 1) % len(ITEM_PALETTE)]

    def _crane_geometry(self, panel_w: int, panel_h: int) -> Dict[str, Any]:
        """Reserve a compact top band for two crane-type markers."""

        gantry_h = max(7, int(round(panel_h * 0.022)))
        crane_zone_h = gantry_h + max(28, int(round(panel_h * 0.065)))
        return {
            "gantry_top": 6,
            "gantry_h": gantry_h,
            "gantry_bot": 6 + gantry_h,
            "crane_zone_h": crane_zone_h,
        }

    def _layout(self, state: Sequence[Sequence[int]], panel_w: int, panel_h: int) -> Dict[str, Any]:
        crane_geom = self._crane_geometry(panel_w, panel_h)
        reserved = crane_geom["crane_zone_h"]
        total_boxes = max(1, sum(len(stack) for stack in state))
        layout = self._slot_layout(
            panel_w,
            slot_count=3,
            item_size=panel_w * 0.48,
            max_stack=total_boxes + 1,
            panel_height=panel_h - reserved,
        )
        for key in ("margin_top", "pedestal_y", "slot_floor_y"):
            layout[key] += reserved
        return layout

    def _draw_two_cranes(
        self,
        draw: ImageDraw.ImageDraw,
        panel_w: int,
        crane_geom: Dict[str, Any],
        layout: Dict[str, Any],
        state: Sequence[Sequence[int]],
        moving: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Draw only the rail and the two load-bearing crane-type markers."""

        gantry_top = crane_geom["gantry_top"]
        gantry_bot = crane_geom["gantry_bot"]
        draw.rounded_rectangle(
            (5, gantry_top, panel_w - 5, gantry_bot),
            radius=3,
            fill=(86, 98, 116),
        )
        specs = (
            ("L", 0, (78, 126, 214), (34, 68, 132), 2),
            ("R", 2, (224, 92, 78), (142, 42, 34), 1),
        )
        marker_y = gantry_bot + 11
        marker_h = 20
        for label, lane_idx, fill, outline, capacity in specs:
            cx = layout["slot_centers_x"][lane_idx]
            marker_w = 42 if capacity == 2 else 30
            draw.rounded_rectangle(
                (cx - marker_w / 2, marker_y, cx + marker_w / 2, marker_y + marker_h),
                radius=5,
                fill=fill,
                outline=outline,
                width=2,
            )
            font = self._draw_font(draw, 11)
            draw.text((cx - 7, marker_y + marker_h / 2), label, font=font,
                      fill=(255, 255, 255), anchor="mm")
            unit_w = 6
            units_x = cx + 5
            for index in range(capacity):
                x0 = units_x + index * (unit_w + 2)
                draw.rectangle((x0, marker_y + 7, x0 + unit_w, marker_y + 13),
                               fill=(255, 255, 255))

    def _render_panel(
        self,
        state: Sequence[Sequence[int]],
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        crane_geom = self._crane_geometry(panel_w, panel_h)
        layout = self._layout(state, panel_w, panel_h)
        image, draw = self._panel_canvas(panel_w, panel_h, (240, 232, 220))
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=(212, 196, 162), width=2)
        # Lane rails (vertical guides) — start below the gantry beam.
        for cx in layout["slot_centers_x"]:
            cx = int(round(cx))
            draw.rectangle((cx - 2, layout["margin_top"], cx + 2, layout["pedestal_y"]), fill=(208, 192, 168))
        self._slot_pedestal(draw, layout)
        item_w = layout["item_w"]
        item_h = layout["item_h"]
        # Render boxes.
        hide_top_n: Dict[int, int] = {}
        if moving is not None and "src" in moving:
            hide_top_n[int(moving["src"])] = int(moving.get("hide_top_n", 1))
        for slot_idx, stack in enumerate(state):
            cx = layout["slot_centers_x"][slot_idx]
            n_hidden = hide_top_n.get(slot_idx, 0)
            for depth, b in enumerate(stack):
                if depth >= len(stack) - n_hidden:
                    continue
                bottom_y = layout["slot_floor_y"] - depth * (item_h + 3)
                top_y = bottom_y - item_h
                bbox = (cx - item_w / 2, top_y, cx + item_w / 2, bottom_y)
                self._draw_block_icon(draw, bbox, self._box_color(int(b)), label=str(int(b)),
                                      radius=int(item_h * 0.26))
        if moving is not None and "boxes" in moving:
            for bx_info in moving["boxes"]:
                cx, cy = bx_info["center"]
                b = int(bx_info["block"])
                bbox = (cx - item_w / 2, cy - item_h / 2, cx + item_w / 2, cy + item_h / 2)
                self._draw_block_icon(draw, bbox, self._box_color(b), label=str(b),
                                      radius=int(item_h * 0.26))
        # Crane glyph: gantry beam + LEFT trolley (blue, cap-2) + RIGHT trolley (red, cap-1).
        self._draw_two_cranes(draw, panel_w, crane_geom, layout, state, moving=moving)
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _animate_move(self, state, move, target_state):
        crane, src, dst = move
        n_boxes = 1 if crane == "R" else 2
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(state, panel_w, panel_h)
        item_h = layout["item_h"]
        src_cx = layout["slot_centers_x"][src]
        dst_cx = layout["slot_centers_x"][dst]
        # Source positions: top-most boxes of the src lane.
        src_top_depth = len(state[src]) - 1
        # Pre-compute the n boxes being lifted (they're at depths len-1 and len-2).
        moving_boxes = [int(state[src][-(i + 1)]) for i in range(n_boxes)]
        # For LEFT crane: the order comes out flipped — top-of-src becomes bottom-of-dst-stack-after-move.
        dst_y_targets = []
        dst_top_depth = len(state[dst])
        if crane == "R":
            dst_y_targets.append(layout["slot_floor_y"] - dst_top_depth * (item_h + 3) - item_h / 2)
        else:
            # The two boxes land in order (a, b) where a = src top, b = src below-top, and stacking is (a then b).
            dst_y_targets.append(layout["slot_floor_y"] - dst_top_depth * (item_h + 3) - item_h / 2)
            dst_y_targets.append(layout["slot_floor_y"] - (dst_top_depth + 1) * (item_h + 3) - item_h / 2)
        remaining = [list(stack) for stack in state]
        remaining[src] = remaining[src][:-n_boxes]
        left_lane, right_lane = sorted((src, dst))
        obstacle_tops = [
            layout["slot_floor_y"] - (len(remaining[lane]) - 1) * (item_h + 3) - item_h
            for lane in range(left_lane, right_lane + 1)
            if remaining[lane]
        ]
        obstacle_top = min(obstacle_tops, default=layout["slot_floor_y"])
        minimal_clearance = (
            obstacle_top - 6 - (n_boxes - 1) * (item_h + 2) - item_h / 2
        )
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        frames = []
        n_lift = max(3, self.move_frames // 3)
        n_traverse = max(3, self.move_frames // 3)
        n_drop = self.move_frames - n_lift - n_traverse
        if n_drop < 3:
            raise RuntimeError("Dock-yard animation requires at least three frames per phase")
        # Phase 1: lift n_boxes from src.
        for k in range(n_lift):
            t = k / (n_lift - 1)
            box_anims = []
            for i in range(n_boxes):
                src_y = layout["slot_floor_y"] - (src_top_depth - i) * (item_h + 3) - item_h / 2
                _, cy = self._interpolate_pixel_linear(
                    (src_cx, src_y),
                    (src_cx, minimal_clearance + i * (item_h + 2)),
                    t,
                )
                box_anims.append({"block": moving_boxes[i], "center": (src_cx, cy)})
            anim = {"src": src, "hide_top_n": n_boxes, "boxes": box_anims, "crane_name": crane}
            frames.append(self._compose_scene(self._render_panel(state, panel_w, panel_h, moving=anim), right_image))
        # Phase 2: traverse to dst.
        for k in range(1, n_traverse + 1):
            t = k / n_traverse
            cx, _ = self._interpolate_pixel_linear(
                (src_cx, minimal_clearance),
                (dst_cx, minimal_clearance),
                t,
            )
            box_anims = []
            for i in range(n_boxes):
                cy = minimal_clearance + i * (item_h + 2)
                box_anims.append({"block": moving_boxes[i], "center": (cx, cy)})
            anim = {"src": src, "hide_top_n": n_boxes, "boxes": box_anims, "crane_name": crane}
            frames.append(self._compose_scene(self._render_panel(state, panel_w, panel_h, moving=anim), right_image))
        # Phase 3: drop into dst.
        for k in range(1, n_drop + 1):
            t = k / n_drop
            box_anims = []
            for i in range(n_boxes):
                # For LEFT crane (n=2): the FIRST lifted box goes to higher depth (gets covered),
                # so during descent the order reverses visually too.
                # We just animate each box from its phase-2 (apex+i*delta) position to its dst_y_target.
                start_y = minimal_clearance + i * (item_h + 2)
                end_y = dst_y_targets[i]
                _, cy = self._interpolate_pixel_linear(
                    (dst_cx, start_y),
                    (dst_cx, end_y),
                    t,
                )
                box_anims.append({"block": moving_boxes[i], "center": (dst_cx, cy)})
            anim = {"src": src, "hide_top_n": n_boxes, "boxes": box_anims, "crane_name": crane}
            frames.append(self._compose_scene(self._render_panel(state, panel_w, panel_h, moving=anim), right_image))
        after_state = self._apply_move(state, move)
        if frames:
            frames[-1] = self._render_static_scene(after_state, target_state)
        return frames


    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _lane_phrase(stack) -> str:
        if not stack:
            return "empty"
        if len(stack) == 1:
            return f"just box {stack[0]}"
        return ", ".join(str(b) for b in stack) + " bottom to top"

    def _yard_readout(self, state) -> str:
        return "; ".join(
            f"lane {index} holds {self._lane_phrase(stack)}" if stack else f"lane {index} is empty"
            for index, stack in enumerate(state, start=1)
        )

    def _author_observation(self, initial_state, target_state, moves) -> str:
        return (
            f"In the yard, {self._yard_readout(initial_state)}. In the target arrangement, "
            f"{self._yard_readout(target_state)}. The wide L crane always carries the top pair and flips "
            "their order on landing, while the slim R crane carries a single box unflipped; this layout "
            "cannot be solved at shortest length by either crane alone, so both have to take part."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        crane, src_lane, dst_lane = str(move[0]), int(move[1]), int(move[2])
        _, visible_src, visible_dst = to_visible_move(self.TASK_NAME, (crane, src_lane, dst_lane))
        if crane == "R":
            box = previous_state[src_lane][-1]
            settled = (
                len(next_state[dst_lane]) <= len(target_state[dst_lane])
                and list(target_state[dst_lane][: len(next_state[dst_lane])]) == list(next_state[dst_lane])
            )
            purpose = (
                "which matches the target stack there"
                if settled and target_state[dst_lane]
                else "keeping its order untouched"
            )
            action = (
                f"The R crane lifts box {box} from lane {visible_src} onto lane {visible_dst}, {purpose}"
            )
        else:
            upper = previous_state[src_lane][-1]
            lower = previous_state[src_lane][-2]
            action = (
                f"The L crane grabs the top pair of lane {visible_src} and flips it onto lane {visible_dst}, "
                f"so box {upper} now sits under box {lower}"
            )
        return f"{action}. Now {self._yard_readout(next_state)}."

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"All three lanes now match the target exactly after {len(moves)} crane moves, using both the "
            "flipping pair crane and the single-box crane."
        )

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side dock yards on a beige canvas. Each yard has three vertical "
            "lanes, numbered 1, 2, and 3 from left to right, holding stacks of numbered cargo boxes. A compact rail at the "
            "top carries two flat crane markers: a wide BLUE `L` marker with two white capacity units (moves the top "
            "two boxes with a flip) and a slim RED `R` marker with one white capacity unit (moves only the top "
            "box). The markers intentionally omit cables, claws, wheels, and decorative machinery. The left yard shows "
            "the starting arrangement and the right yard shows the desired final arrangement. A blue chevron arrow "
            "between the yards points from left to right."
        )

    def _describe_rules(self) -> str:
        return (
            "The yard always has exactly three unlimited-capacity lanes numbered 1, 2, and 3 from left to right, and every "
            "move must use different source and destination lanes. The LEFT crane is legal only when the source has "
            "at least two boxes: it removes exactly the source's TOP TWO boxes together and places them on the "
            "destination top in reversed vertical order, so the originally top box finishes directly below the "
            "originally second box. The RIGHT crane is legal only when the source is non-empty: it removes exactly "
            "the TOP ONE box and places it on the destination top without flipping. No other box moves, falls, "
            "compacts, refills, changes lane, changes identity, or changes appearance after either action."
        )

    def _describe_output(self) -> str:
        return (
            "Return one shortest valid move sequence that transforms the complete left arrangement into the complete "
            "right arrangement. Use exactly `Answer: [(<crane>, <source_lane>, <destination_lane>), ...].` with "
            "uppercase quoted crane names and lane numbers 1, 2, or 3; include no explanation or extra fields."
        )

    def _format_move(self, move):
        """Serialize one internal move with visible one-based lane numbers."""

        crane, source, destination = to_visible_move(
            self.TASK_NAME, (str(move[0]), int(move[1]), int(move[2]))
        )
        return f"(\"{crane}\", {source}, {destination})"

    def _extra_payload(self, initial_state, target_state, moves):
        return {
            "lane_count": 3,
            "box_count": sum(len(s) for s in initial_state),
            "optimal_moves": [[m[0], int(m[1]), int(m[2])] for m in moves],
        }


__all__ = ["DockYardCraneSwapGenerator"]
