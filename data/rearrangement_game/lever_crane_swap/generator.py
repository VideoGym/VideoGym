"""Lever-crane alternating-side rearrangement with a 1-block buffer.

A central LEVER CRANE swings between TWO column stacks (LEFT column L and
RIGHT column R) and a single BUFFER cell that holds ONE block under the
crane head. Each move is one of FOUR actions:

* ``"PL"`` — pick top of L into the buffer (buffer must be empty).
* ``"PR"`` — pick top of R into the buffer (buffer must be empty).
* ``"DL"`` — drop the buffer block onto L (buffer must be holding one block).
* ``"DR"`` — drop the buffer block onto R (buffer must be holding one block).

Alternation constraint: SUCCESSIVE PICKS must come from DIFFERENT SIDES.
The side of the previous pick is what matters, however many drops have
happened since — a drop never resets it.  Since the buffer holds one block,
every pick is followed by a drop anyway, so a constraint that a drop reset
would be vacuous and the whole mechanic would carry nothing.

Move notation: a single 2-character string per move from {PL, PR, DL, DR}.

Mechanic load-bearing check: we generate puzzles where the optimal
alternation-respecting plan is strictly longer than the optimal plan that
ignores alternation (i.e. a hypothetical agent that can pick twice from the
same side in a row).  Targets are not fished for: one breadth-first sweep
from the sampled start yields the optimum to every reachable configuration
under both rules at once, and the target is drawn from those that satisfy
the gate, one plan length at a time.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    _shade,
)

PANEL_BG = (246, 245, 241)
PANEL_BORDER = (205, 202, 194)
COLUMN_FILL = (232, 231, 226)
COLUMN_BORDER = (170, 169, 163)
BUFFER_FILL = (226, 240, 246)
BUFFER_BORDER = (105, 160, 184)
CRANE_BEAM = (94, 102, 108)
CRANE_PIVOT = (72, 78, 84)


#: One crane state: (left stack, right stack, buffered block or None, side of
#: the previous pick or None).
CraneState = Tuple[Tuple[int, ...], Tuple[int, ...], Optional[int], Optional[str]]


def _successors(state: CraneState, alternation: bool) -> List[Tuple[str, CraneState]]:
    """Every legal action from ``state`` — the single source of the move rule."""

    l, r, buf, last_pick = state
    moves: List[Tuple[str, CraneState]] = []
    for action in ("PL", "PR", "DL", "DR"):
        if action == "PL":
            if buf is not None or not l or (alternation and last_pick == "L"):
                continue
            moves.append((action, (l[:-1], r, l[-1], "L")))
        elif action == "PR":
            if buf is not None or not r or (alternation and last_pick == "R"):
                continue
            moves.append((action, (l, r[:-1], r[-1], "R")))
        elif action == "DL":
            if buf is None:
                continue
            # A drop never clears the side of the previous pick: alternation
            # is between successive picks, not between adjacent moves.
            moves.append((action, (l + (buf,), r, None, last_pick)))
        else:  # DR
            if buf is None:
                continue
            moves.append((action, (l, r + (buf,), None, last_pick)))
    return moves


class _CraneSweep:
    """Optimal plans from one start to every configuration reachable from it.

    One sweep answers "how long is the shortest plan to *this* target" for
    every target at once, which is what lets the sampler construct an
    instance at a chosen plan length instead of drawing targets at random
    until one happens to have the length it wanted.
    """

    def __init__(
        self,
        initial_l: Tuple[int, ...],
        initial_r: Tuple[int, ...],
        max_moves: int,
        alternation: bool,
    ) -> None:
        start: CraneState = (initial_l, initial_r, None, None)
        self._parents: Dict[CraneState, Optional[Tuple[CraneState, str]]] = {start: None}
        self._settled: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], CraneState] = {}
        self.optimum: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {
            (initial_l, initial_r): 0
        }
        queue = deque([(start, 0)])
        while queue:
            state, depth = queue.popleft()
            if depth >= max_moves:
                continue
            for action, nxt in _successors(state, alternation):
                if nxt in self._parents:
                    continue
                self._parents[nxt] = (state, action)
                if nxt[2] is None:
                    key = (nxt[0], nxt[1])
                    if key not in self.optimum:
                        self.optimum[key] = depth + 1
                        self._settled[key] = nxt
                queue.append((nxt, depth + 1))

    def plan_to(self, key: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Optional[List[str]]:
        """One optimal action list reaching ``key`` with an empty buffer."""

        state = self._settled.get(key)
        if state is None:
            return [] if self.optimum.get(key) == 0 else None
        plan: List[str] = []
        cursor: CraneState = state
        while True:
            link = self._parents.get(cursor)
            if link is None:
                break
            previous, action = link
            plan.append(action)
            cursor = previous
        plan.reverse()
        return plan


def _bfs(initial_l: Tuple[int, ...], initial_r: Tuple[int, ...], target_l: Tuple[int, ...], target_r: Tuple[int, ...], max_moves: int, alternation: bool) -> Optional[List[str]]:
    """Shortest legal plan from one configuration to another, or ``None``."""

    if (initial_l, initial_r) == (target_l, target_r):
        return []
    sweep = _CraneSweep(initial_l, initial_r, max_moves, alternation)
    return sweep.plan_to((target_l, target_r))


class LeverCraneSwapGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "lever_crane_swap"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/lever_crane_swap"
    SCENE_OUTER_MARGIN = 8
    SCENE_CHEVRON_BAND = 26
    SCENE_CHEVRON_SCALE = 0.72

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        max_optimal_moves: int = 14,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.max_optimal_moves = int(max_optimal_moves)

    MIN_OPTIMAL_MOVES = 4

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(64):
            n_blocks = self.rng.randint(3, 5)
            blocks = list(range(1, n_blocks + 1))
            self.rng.shuffle(blocks)
            split = self.rng.randint(1, n_blocks - 1)
            initial = (tuple(blocks[:split]), tuple(blocks[split:]))

            # Both sweeps are over the same start, so every reachable target
            # comes with its optimum under the crane rule and its optimum for
            # an unrestricted crane.  The load-bearing gate is then a filter
            # over known targets rather than a lottery over sampled ones.
            restricted = _CraneSweep(initial[0], initial[1], self.max_optimal_moves, True)
            unrestricted = _CraneSweep(initial[0], initial[1], self.max_optimal_moves, False)
            by_length: Dict[int, List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = defaultdict(list)
            for key, length in restricted.optimum.items():
                if length < self.MIN_OPTIMAL_MOVES:
                    continue
                free = unrestricted.optimum.get(key)
                # Mechanic load-bearing: alternation must increase plan length.
                if free is None or free >= length:
                    continue
                by_length[length].append(key)
            if not by_length:
                continue
            # Draw the plan length first so a start whose easy targets vastly
            # outnumber its hard ones cannot flatten the task to one length.
            length = self.rng.choice(sorted(by_length))
            target = self.rng.choice(by_length[length])
            plan = restricted.plan_to(target)
            if plan is None or len(plan) != length:
                continue
            return [list(initial[0]), list(initial[1])], [list(target[0]), list(target[1])], plan
        raise RuntimeError(
            "lever_crane_swap sampling found no start with an alternation-limited target; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state):
        return (tuple(state[0]), tuple(state[1]))

    def _serialize_state(self, state):
        return [list(state[0]), list(state[1])]

    def _apply_move(self, state, move):
        m = str(move).upper()
        l = list(state[0])
        r = list(state[1])
        # We need to track buffer; but apply_move is meant to transition between published states.
        # Pair PL with the next DL/DR atomically: if the move is PL we "stash" the picked block onto a hidden buffer
        # that lives in state. To stay compatible with our base-class signature, we extend state to include "buf".
        if len(state) == 2:
            buf = None
        else:
            buf = state[2]
        if m == "PL":
            if buf is not None:
                raise ValueError("buffer occupied")
            if not l:
                raise ValueError("left empty")
            buf = l.pop()
        elif m == "PR":
            if buf is not None:
                raise ValueError("buffer occupied")
            if not r:
                raise ValueError("right empty")
            buf = r.pop()
        elif m == "DL":
            if buf is None:
                raise ValueError("buffer empty")
            l.append(buf)
            buf = None
        elif m == "DR":
            if buf is None:
                raise ValueError("buffer empty")
            r.append(buf)
            buf = None
        else:
            raise ValueError(f"invalid move {move}")
        return [l, r, buf]

    def _block_color(self, k: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(k - 1) % len(ITEM_PALETTE)]

    def _layout(self, panel_w, panel_h):
        margin = max(8, int(min(panel_w, panel_h) * 0.05))
        col_w = int(panel_w * 0.22)
        col_gap = int(panel_w * 0.20)
        beam_h = int(panel_h * 0.18)
        col_y0 = margin + beam_h
        col_y1 = panel_h - margin
        left_x0 = (panel_w - 2 * col_w - col_gap) // 2
        left_x1 = left_x0 + col_w
        right_x0 = left_x1 + col_gap
        right_x1 = right_x0 + col_w
        block_h = max(20, int((col_y1 - col_y0 - 8) / 5))
        block_w = col_w - 12
        return {
            "panel_w": panel_w, "panel_h": panel_h,
            "margin": margin,
            "left_col": (left_x0, col_y0, left_x1, col_y1),
            "right_col": (right_x0, col_y0, right_x1, col_y1),
            "beam_y": margin + beam_h - 16,
            "pivot_x": (left_x1 + right_x0) // 2,
            "col_w": col_w,
            "col_y0": col_y0,
            "col_y1": col_y1,
            "block_w": block_w,
            "block_h": block_h,
            "buffer_center": ((left_x1 + right_x0) // 2, col_y0 + block_h / 2 + 4),
        }

    def _draw_column(self, draw, bbox, blocks: Sequence[int], *, block_w: float, block_h: float):
        x0, y0, x1, y1 = bbox
        draw.rectangle((x0, y0, x1, y1), fill=COLUMN_FILL, outline=COLUMN_BORDER, width=2)
        draw.rectangle((x0 - 6, y1 - 4, x1 + 6, y1), fill=(150, 150, 145))
        cx = (x0 + x1) / 2
        for i, b in enumerate(blocks):
            bx0 = cx - block_w / 2
            bx1 = cx + block_w / 2
            by1 = y1 - 4 - i * (block_h + 2)
            by0 = by1 - block_h
            self._draw_block(draw, (bx0, by0, bx1, by1), int(b))

    def _draw_buffer(self, draw, layout, holding: Optional[int] = None, hold_pix: Optional[Tuple[float, float]] = None):
        cx, cy = layout["buffer_center"]
        block_w = layout["block_w"]
        block_h = layout["block_h"]
        x0 = cx - block_w / 2 - 4
        x1 = cx + block_w / 2 + 4
        y0 = cy - block_h / 2 - 4
        y1 = cy + block_h / 2 + 4
        draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=BUFFER_FILL, outline=BUFFER_BORDER, width=2)
        if holding is not None:
            if hold_pix is not None:
                hx, hy = hold_pix
                self._draw_block(
                    draw,
                    (hx - block_w / 2, hy - block_h / 2, hx + block_w / 2, hy + block_h / 2),
                    int(holding),
                )
            else:
                self._draw_block(
                    draw,
                    (cx - block_w / 2, cy - block_h / 2, cx + block_w / 2, cy + block_h / 2),
                    int(holding),
                )

    def _draw_block(self, draw, bbox, k: int):
        x0, y0, x1, y1 = bbox
        color = self._block_color(k)
        outline = _shade(color, 0.55)
        radius = max(4, int((x1 - x0) * 0.10))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=color, outline=outline, width=2)
        font = self._draw_font(draw, max(10, int((y1 - y0) * 0.42)))
        text_color = (32, 30, 26) if (sum(color) > 380) else (250, 248, 244)
        draw.text(((x0 + x1) / 2, (y0 + y1) / 2 - 1), str(k), font=font, fill=text_color, anchor="mm")

    def _draw_crane(self, draw, layout, head_x: float, head_y: float):
        beam_y = layout["beam_y"]
        px = layout["pivot_x"]
        draw.rounded_rectangle((px - 5, layout["margin"] // 2, px + 5, beam_y + 10), radius=3, fill=CRANE_PIVOT)
        draw.line((px, beam_y, head_x, beam_y), fill=CRANE_BEAM, width=6)
        cable_end_y = head_y - layout["block_h"] / 2 - 3
        draw.line((head_x, beam_y, head_x, cable_end_y), fill=CRANE_BEAM, width=2)

    def _render_panel(self, state, panel_w, panel_h, *, moving=None):
        layout = self._layout(panel_w, panel_h)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        l_blocks = list(state[0])
        r_blocks = list(state[1])
        buf = state[2] if len(state) > 2 else None
        # During animation, hide the source-side top block (it's been picked).
        if moving is not None:
            if moving.get("hide_side") == "L" and l_blocks:
                l_blocks = l_blocks[:-1]
            elif moving.get("hide_side") == "R" and r_blocks:
                r_blocks = r_blocks[:-1]
        self._draw_column(draw, layout["left_col"], l_blocks, block_w=layout["block_w"], block_h=layout["block_h"])
        self._draw_column(draw, layout["right_col"], r_blocks, block_w=layout["block_w"], block_h=layout["block_h"])
        # Buffer pad.
        if moving is not None and "buffer_block" in moving:
            self._draw_buffer(draw, layout, holding=moving["buffer_block"], hold_pix=moving.get("hold_pix"))
        else:
            self._draw_buffer(draw, layout, holding=buf)
        # Crane.
        head_x = moving.get("head_x") if moving is not None and "head_x" in moving else layout["pivot_x"]
        head_y = moving.get("head_y") if moving is not None and "head_y" in moving else layout["buffer_center"][1]
        self._draw_crane(draw, layout, head_x, head_y)
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _animate_move(self, state, move, target_state):
        m = str(move).upper()
        # Determine source and dest.
        l_blocks = list(state[0])
        r_blocks = list(state[1])
        buf = state[2] if len(state) > 2 else None
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h)
        left_cx = (layout["left_col"][0] + layout["left_col"][2]) / 2
        right_cx = (layout["right_col"][0] + layout["right_col"][2]) / 2
        buffer_cx, buffer_cy = layout["buffer_center"]
        block_h = layout["block_h"]
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        frames: List[Image.Image] = []
        after = self._apply_move(state, move)
        if m in ("PL", "PR"):
            side = "L" if m == "PL" else "R"
            src = l_blocks if side == "L" else r_blocks
            if not src:
                return []
            picked = src[-1]
            sx = left_cx if side == "L" else right_cx
            n_src = len(src)
            sy = layout["col_y1"] - 4 - (n_src - 1) * (block_h + 2) - block_h / 2
            for step in range(1, self.move_frames + 1):
                t = step / self.move_frames
                if step == self.move_frames:
                    frames.append(self._render_static_scene(after, target_state))
                    continue
                cx, cy = self._interpolate_pixel_linear((sx, sy), (buffer_cx, buffer_cy), t)
                anim = {"head_x": cx, "head_y": cy, "buffer_block": picked, "hold_pix": (cx, cy), "hide_side": side}
                left = self._render_panel(state, panel_w, panel_h, moving=anim)
                frames.append(self._compose_scene(left, right_image))
        else:
            # DL or DR
            side = "L" if m == "DL" else "R"
            if buf is None:
                return []
            dx = left_cx if side == "L" else right_cx
            dest = l_blocks if side == "L" else r_blocks
            n_dest = len(dest) + 1
            dy = layout["col_y1"] - 4 - (n_dest - 1) * (block_h + 2) - block_h / 2
            for step in range(1, self.move_frames + 1):
                t = step / self.move_frames
                if step == self.move_frames:
                    frames.append(self._render_static_scene(after, target_state))
                    continue
                cx, cy = self._interpolate_pixel_linear((buffer_cx, buffer_cy), (dx, dy), t)
                anim = {"head_x": cx, "head_y": cy, "buffer_block": buf, "hold_pix": (cx, cy)}
                left = self._render_panel(state, panel_w, panel_h, moving=anim)
                frames.append(self._compose_scene(left, right_image))
        return frames


    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _stack_phrase(stack) -> str:
        if not stack:
            return "empty"
        if len(stack) == 1:
            return f"just block {stack[0]}"
        return ", ".join(str(b) for b in stack) + " bottom to top"

    def _sides_readout(self, state) -> str:
        left = state[0]
        right = state[1]
        buffer_value = state[2] if len(state) > 2 else None
        text = (
            f"the left stack holds {self._stack_phrase(left)}; the right stack holds "
            f"{self._stack_phrase(right)}"
        )
        if buffer_value is not None:
            text += f"; the buffer holds block {buffer_value}"
        return text

    def _author_observation(self, initial_state, target_state, moves) -> str:
        return (
            f"Under the lever crane, {self._sides_readout(initial_state)}. The target wants the left "
            f"stack to read {self._stack_phrase(target_state[0])} and the right stack "
            f"{self._stack_phrase(target_state[1])}. The buffer holds one block at a time, and each pick has "
            "to come from the opposite side to the pick before it however many drops fall in between, which "
            "is exactly what stretches this plan beyond what an unrestricted crane would need."
        )

    @staticmethod
    def _returns_where_it_came_from(move_index, moves) -> bool:
        """True when this pick/drop pair only exists to switch pick sides.

        Every drop follows the pick that filled the buffer, so a drop back
        onto the side just picked from restores that stack exactly.  Such a
        round trip buys nothing except the right to pick from the other side
        next, and the reasoning should say so rather than invent a purpose.
        """

        actions = [str(item).upper() for item in moves]
        # ``move_index`` counts the published clauses from 1.
        index = int(move_index) - 1
        if index < 0 or index >= len(actions):
            return False
        pick_index = index if actions[index].startswith("P") else index - 1
        drop_index = pick_index + 1
        if pick_index < 0 or drop_index >= len(actions):
            return False
        return actions[pick_index][1] == actions[drop_index][1]

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        action = str(move).upper()
        round_trip = self._returns_where_it_came_from(move_index, moves)
        if action in ("PL", "PR"):
            side = "left" if action == "PL" else "right"
            block = next_state[2]
            source_stack = previous_state[0] if action == "PL" else previous_state[1]
            uncovered = source_stack[:-1]
            extra = f", uncovering block {uncovered[-1]}" if uncovered else ""
            if round_trip:
                detail = (
                    f"The crane lifts block {block} off the {side} stack into the buffer just to take its turn "
                    f"on this side{extra}"
                )
            else:
                detail = f"The crane picks block {block} off the {side} stack into the buffer{extra}"
        else:
            side = "left" if action == "DL" else "right"
            block = previous_state[2]
            side_index = 0 if action == "DL" else 1
            landed = next_state[side_index]
            wanted = target_state[side_index]
            settled = (
                len(landed) <= len(wanted)
                and list(wanted[: len(landed)]) == list(landed)
            )
            if round_trip:
                purpose = "putting it straight back so the next pick may legally come from the other side"
            elif settled and wanted:
                purpose = "which is exactly where the target stacks it"
            else:
                purpose = "parking it there for now"
            detail = f"The crane drops block {block} from the buffer onto the {side} stack, {purpose}"
        return f"{detail}. Now {self._sides_readout(next_state)}."

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Both stacks now match the target with the buffer empty, after {len(moves)} alternating "
            "picks and drops."
        )

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side stacking workshops. Each has a LEFT column L and a "
            "RIGHT column R holding numbered blocks (top of stack = highest), a small blue BUFFER tray under a "
            "central LEVER CRANE shown as a simple rail and cable. The buffer holds AT MOST one block at a time. The "
            "left workshop shows the starting stacks; the right workshop shows the desired final stacks. A small "
            "blue center arrow separates start from goal. Blocks retain the same width and height in a stack, in "
            "the buffer, and while moving."
        )

    def _describe_rules(self) -> str:
        return (
            "Each move is one of four actions: PL = pick top of L into the buffer; PR = pick top of R into the "
            "buffer; DL = drop the buffer block onto L; DR = drop the buffer block onto R. A pick requires the "
            "buffer to be empty; a drop requires the buffer to be holding a block. Alternation rule: successive "
            "picks must come from DIFFERENT sides — after a PL the next pick must be PR, and after a PR the next "
            "pick must be PL, however many drops happen in between."
        )

    def _describe_output(self) -> str:
        return (
            "Return the shortest plan as `Answer: [<action>, ...].`, where each quoted action is one of `\"PL\"`, "
            "`\"PR\"`, `\"DL\"`, or `\"DR\"`."
        )

    def _format_move(self, move):
        return f"'{str(move).upper()}'"

    def _extra_payload(self, initial_state, target_state, moves):
        return {
            "optimal_moves": [str(m).upper() for m in moves],
        }


__all__ = ["LeverCraneSwapGenerator"]
