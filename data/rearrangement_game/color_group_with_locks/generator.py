"""Color-group sorting around fixed lock items.

A 1xN row of slots holds COLORED ORBS plus 1-2 LOCKED orbs (greyed out
with a tiny padlock icon on top). Each non-locked orb is colored RED,
GREEN, or BLUE. Locked orbs cannot move. The solver swaps non-locked
adjacent pairs (no lock-jump) so that each lock-delimited segment is
independently ordered RED before GREEN before BLUE, while LOCKED orbs
stay frozen in their original slots.

Move notation: each move is a single integer ``slot_idx`` of the LEFT
member of an adjacent swap pair (so the swap is between slot ``i`` and
slot ``i+1``). Both slots must be non-locked.

Mechanic load-bearing check: removing the lock-delimited segmentation changes
the required terminal colour order. Published samples reject cases whose
segment-wise target happens to equal an unlocked global colour grouping.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from PIL import Image

from data.rearrangement_game.core.answer_indexing import to_visible_plan
from data.common.pipeline.object_plans import (
    _shade,
)

PANEL_BG = (236, 232, 224)
PANEL_BORDER = (200, 192, 176)
TRACK_FILL = (218, 210, 196)
TRACK_BORDER = (172, 162, 144)
SLOT_LINE = (172, 162, 144)
LOCK_FILL = (188, 188, 196)
LOCK_OUTLINE = (108, 108, 116)
COLORS = {
    "R": (220, 88, 76),
    "G": (118, 188, 96),
    "B": (84, 124, 196),
}
COLOR_ORDER = ["R", "G", "B"]


def _is_sorted(state: Sequence[str], locks: Sequence[int]) -> bool:
    boundaries = [-1, *sorted(int(index) for index in locks), len(state)]
    for left, right in zip(boundaries, boundaries[1:]):
        segment = [state[index] for index in range(left + 1, right) if state[index] != "L"]
        if segment != sorted(segment, key=lambda color: COLOR_ORDER.index(color)):
            return False
    return True


def _bfs(initial: Tuple[str, ...], locks: Tuple[int, ...], max_moves: int) -> Optional[List[int]]:
    if _is_sorted(initial, locks):
        return []
    queue = deque([(initial, [])])
    seen = {initial}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_moves:
            continue
        for i in range(len(state) - 1):
            if i in locks or (i + 1) in locks:
                continue
            new_state = list(state)
            new_state[i], new_state[i + 1] = new_state[i + 1], new_state[i]
            new_state_t = tuple(new_state)
            if new_state_t in seen:
                continue
            seen.add(new_state_t)
            np = path + [i]
            if _is_sorted(new_state_t, locks):
                return np
            queue.append((new_state_t, np))
    return None


class ColorGroupWithLocksGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "color_group_with_locks"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/color_group_with_locks"
    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        max_optimal_moves: int = 8,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.max_optimal_moves = int(max_optimal_moves)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            n = self.rng.randint(7, 8)
            n_locks = self.rng.randint(1, 2)
            lock_positions = sorted(self.rng.sample(range(n), n_locks))
            # Assign colors to non-lock positions.
            slots: List[str] = ["L"] * n
            non_lock = [i for i in range(n) if i not in lock_positions]
            n_non_lock = len(non_lock)
            # Distribute colors: at least 1 of each color
            # Choose 2..3 of each.
            n_red = self.rng.randint(2, max(2, n_non_lock // 3 + 1))
            remaining = n_non_lock - n_red
            n_green = self.rng.randint(1, max(1, remaining - 1))
            n_blue = remaining - n_green
            if n_blue < 1:
                continue
            color_pool = ["R"] * n_red + ["G"] * n_green + ["B"] * n_blue
            self.rng.shuffle(color_pool)
            for pos, col in zip(non_lock, color_pool):
                slots[pos] = col
            initial = tuple(slots)
            locks_t = tuple(lock_positions)
            plan = _bfs(initial, locks_t, self.max_optimal_moves)
            if plan is None:
                continue
            if not (3 <= len(plan) <= self.max_optimal_moves):
                continue
            # Compute target by replaying.
            target = list(initial)
            for mv in plan:
                target[mv], target[mv + 1] = target[mv + 1], target[mv]
            unlocked_target = sorted(
                (color for color in initial if color != "L"),
                key=lambda color: COLOR_ORDER.index(color),
            )
            if [color for color in target if color != "L"] == unlocked_target:
                continue
            return list(initial), list(target), [int(m) for m in plan]
        raise RuntimeError("failed to sample a load-bearing lock-delimited colour grouping")

    def _state_signature(self, state):
        return tuple(state)

    def _serialize_state(self, state):
        return list(state)

    def _apply_move(self, state, move):
        i = int(move)
        cur = list(state)
        if cur[i] == "L" or cur[i + 1] == "L":
            raise ValueError(f"cannot swap with a locked slot at {i}/{i+1}")
        cur[i], cur[i + 1] = cur[i + 1], cur[i]
        return cur

    def _layout(self, panel_w, panel_h, n: int):
        margin = max(8, int(min(panel_w, panel_h) * 0.04))
        track_h = int(panel_h * 0.74)
        track_y0 = (panel_h - track_h) // 2
        track_y1 = track_y0 + track_h
        usable_w = panel_w - 2 * margin
        slot_w = usable_w / max(1, n)
        return {
            "panel_w": panel_w, "panel_h": panel_h,
            "margin": margin,
            "track_y0": track_y0, "track_y1": track_y1,
            "slot_w": slot_w,
        }

    def _draw_orb(self, draw, cx: float, cy: float, r: float, color_key: str):
        if color_key == "L":
            base = LOCK_FILL
            outline = LOCK_OUTLINE
        else:
            base = COLORS[color_key]
            outline = _shade(base, 0.55)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=base, outline=outline, width=2)
        if color_key == "L":
            # Padlock icon on top.
            self._draw_lock(draw, cx, cy, r * 0.85)

    def _draw_lock(self, draw, cx: float, cy: float, size: float):
        body_w = size
        body_h = size * 0.78
        x0 = cx - body_w / 2
        x1 = cx + body_w / 2
        y0 = cy - body_h / 2 + size * 0.10
        y1 = y0 + body_h
        draw.rounded_rectangle((x0, y0, x1, y1), radius=int(size * 0.18), fill=(80, 80, 88), outline=(40, 40, 48), width=1)
        sh_x0 = x0 + body_w * 0.20
        sh_x1 = x1 - body_w * 0.20
        sh_y0 = y0 - size * 0.34
        sh_y1 = y0 + size * 0.05
        draw.arc((sh_x0, sh_y0, sh_x1, sh_y1), 180, 360, fill=(160, 160, 168), width=3)
        kh_r = size * 0.10
        kh_cy = (y0 + y1) / 2 - size * 0.05
        draw.ellipse((cx - kh_r, kh_cy - kh_r, cx + kh_r, kh_cy + kh_r), fill=(20, 20, 24))

    def _render_panel(self, state, panel_w, panel_h, *, moving=None):
        n = len(state)
        layout = self._layout(panel_w, panel_h, n)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        # Track background.
        track_x0 = layout["margin"]
        track_x1 = panel_w - layout["margin"]
        track_y0 = layout["track_y0"]
        track_y1 = layout["track_y1"]
        draw.rectangle((track_x0, track_y0, track_x1, track_y1), fill=TRACK_FILL, outline=TRACK_BORDER, width=2)
        # Slot dividers.
        for i in range(1, n):
            xs = track_x0 + layout["slot_w"] * i
            draw.line((xs, track_y0 + 4, xs, track_y1 - 4), fill=SLOT_LINE, width=1)
        cy = (track_y0 + track_y1) / 2
        r = min(layout["slot_w"] * 0.36, (track_y1 - track_y0) * 0.36)
        # Slot index numerals (small, below each slot).
        slot_font = self._draw_font(draw, max(9, int(r * 0.42)))
        for i in range(n):
            cx = track_x0 + layout["slot_w"] * (i + 0.5)
            draw.text((cx, track_y1 + 12), str(i + 1), font=slot_font, fill=(96, 88, 72), anchor="mm")
        for i, col in enumerate(state):
            if moving is not None and (i in moving.get("hide_idx", set())):
                continue
            cx = track_x0 + layout["slot_w"] * (i + 0.5)
            self._draw_orb(draw, cx, cy, r, col)
        if moving is not None and "left_pix" in moving:
            self._draw_orb(draw, *moving["left_pix"], r, moving["left_color"])
            self._draw_orb(draw, *moving["right_pix"], r, moving["right_color"])
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _animate_move(self, state, move, target_state):
        i = int(move)
        n = len(state)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h, n)
        track_x0 = layout["margin"]
        cy = (layout["track_y0"] + layout["track_y1"]) / 2
        sx_left = track_x0 + layout["slot_w"] * (i + 0.5)
        sx_right = track_x0 + layout["slot_w"] * (i + 1.5)
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        frames: List[Image.Image] = []
        for step in range(1, self.move_frames + 1):
            t = step / self.move_frames
            cur_left_x, cur_left_y = self._interpolate_pixel_linear(
                (sx_left, cy), (sx_right, cy), t
            )
            cur_right_x, cur_right_y = self._interpolate_pixel_linear(
                (sx_right, cy), (sx_left, cy), t
            )
            anim = {
                "hide_idx": {i, i + 1},
                "left_pix": (cur_left_x, cur_left_y),
                "left_color": state[i],
                "right_pix": (cur_right_x, cur_right_y),
                "right_color": state[i + 1],
            }
            left = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left, right_image))
        return frames

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side horizontal tracks. Each track has 7-8 SLOTS in a row "
            "separated by faint vertical lines, with the slot numbers 1 through N printed beneath the track. Each slot "
            "holds an ORB: RED orbs are warm-red discs, GREEN orbs are leaf-green, BLUE orbs are sky-blue, "
            "and LOCKED orbs are grey discs with a small padlock icon. The LOCKED orb positions are identical in "
            "both tracks. The left track shows the starting layout; the right track shows every segment between "
            "locked orbs independently ordered RED before GREEN before BLUE. A blue chevron arrow between the tracks "
            "points from left to right."
        )

    def _describe_rules(self) -> str:
        return (
            "Each move SWAPS two adjacent orbs, the one in the named slot and its right neighbor. Both slots must contain non-locked orbs — "
            "you cannot swap a locked orb. Locked orbs stay put for the entire puzzle and divide the row into independent "
            "segments. Non-locked orbs cannot jump over them. Within each segment, arrange RED < GREEN < BLUE while "
            "keeping every locked orb in its original slot."
        )

    def _describe_output(self) -> str:
        return (
            "Return the shortest sequence of adjacent swaps that achieves the sorted layout. Each move is one printed "
            "slot number naming the left orb of the swapped pair, counted from 1 at the leftmost slot. Format the "
            "answer as a Python-style list of integers, e.g. `Answer: [<left_slot>, ...]`."
        )

    def _format_move(self, move):
        return str(int(move))

    def _format_answer(self, moves):
        """Serialize the internal plan as visible slot numbers counted from 1."""

        visible = to_visible_plan(self.TASK_NAME, [int(move) for move in moves])
        return "[" + ", ".join(self._format_move(move) for move in visible) + "]"

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    ORB_WORDS = {"R": "red", "G": "green", "B": "blue", "L": "locked"}

    def _row_reading(self, state: Sequence[str]) -> str:
        return ", ".join(
            "a locked orb" if color == "L" else f"{self.ORB_WORDS[color]}"
            for color in state
        )

    def _author_observation(self, initial_state, target_state, moves) -> str:
        locks = [i + 1 for i, c in enumerate(initial_state) if c == "L"]
        lock_fact = (
            f"the locked orb in slot {locks[0]} splits the row into segments"
            if len(locks) == 1
            else f"the locked orbs in slots {' and '.join(str(l) for l in locks)} split the row into segments"
        )
        return (
            f"Reading the track left to right, the orbs are {self._row_reading(initial_state)}. "
            f"Locked orbs never move and nothing can cross them, so {lock_fact}, and each segment "
            "must separately end up red before green before blue. Each adjacent swap should fix one "
            "pair that sits in the wrong colour order."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        i = int(move)
        left_color = self.ORB_WORDS[previous_state[i]]
        right_color = self.ORB_WORDS[previous_state[i + 1]]
        return (
            f"In slots {i + 1} and {i + 2}, the {left_color} orb sits left of the {right_color} orb, "
            f"which is the wrong way around for its segment, so I swap them to put {right_color} first. "
            f"The track now reads {self._row_reading(next_state)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Every lock-delimited segment now runs red, then green, then blue, with all locked orbs "
            f"untouched, after {len(moves)} adjacent swaps."
        )

    def _extra_payload(self, initial_state, target_state, moves):
        locked = [i for i, c in enumerate(initial_state) if c == "L"]
        return {
            "slot_count": len(initial_state),
            "locked_indices": locked,
            "optimal_moves": [int(m) for m in moves],
        }


__all__ = ["ColorGroupWithLocksGenerator"]
