"""Classical 3-peg Tower of Hanoi rearrangement generator.

Three vertical pegs sit on a wooden base. Disks of decreasing radius are
stacked from largest at the bottom to smallest at the top. The solver must
move all the disks from the starting configuration to the target
configuration following two rules: only the top disk of any peg can be moved
to another peg, and a larger disk may never be placed on top of a smaller
disk.

The generator picks N in [3, 5] and chooses an initial vs target peg setup
that requires anywhere from 4 to 14 BFS-optimal moves so the puzzle is hard
enough that random guessing fails but reachable in a few seconds of human
thought (Core Rule #4: calibrated difficulty).

A submitted answer names the pegs 1, 2, 3 from left to right; sampling, BFS,
replay and rendering below index them 0, 1, 2.  The single conversion lives in
:mod:`data.rearrangement_game.core.answer_indexing` and happens only where a
move is serialized for the model.  Disk sizes are read from the drawn widths,
so no disk number ever enters the payload.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    SlotStackMixin,
)
from data.rearrangement_game.core.answer_indexing import to_visible_move

# Spoken names for the disk palette (ITEM_PALETTE order), used only by the
# authored reasoning; the image itself stays free of printed text.
DISK_COLOR_NAMES = ("red", "orange", "yellow", "green", "sky-blue", "blue")

# A muted wood palette for the pegs and base.
PEG_COLOR = (138, 92, 52)
PEG_DARK = (90, 56, 28)
PANEL_BG = (252, 250, 246)
PANEL_BORDER = (216, 207, 188)


def _hanoi_optimal(
    initial: Tuple[Tuple[int, ...], ...],
    target: Tuple[Tuple[int, ...], ...],
    max_moves: int,
) -> Optional[List[Tuple[int, int]]]:
    """BFS to optimal Hanoi move sequence under top-only + size-constraint rules."""

    if initial == target:
        return []
    queue = deque([(initial, [])])
    seen = {initial}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_moves:
            continue
        for src in range(3):
            if not state[src]:
                continue
            top = state[src][-1]
            for dst in range(3):
                if src == dst:
                    continue
                if state[dst] and state[dst][-1] < top:
                    continue
                next_state = tuple(
                    tuple(list(stack)[:-1]) if i == src
                    else (tuple(list(stack) + [top])) if i == dst
                    else stack
                    for i, stack in enumerate(state)
                )
                if next_state in seen:
                    continue
                seen.add(next_state)
                new_path = path + [(src, dst)]
                if next_state == target:
                    return new_path
                queue.append((next_state, new_path))
    return None


class TowerOfHanoiGenerator(SlotStackMixin, RearrangementObjectPlanGenerator):
    TASK_NAME = "tower_of_hanoi"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/tower_of_hanoi"
    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        disk_count: Optional[int] = None,
        disk_count_range: Tuple[int, int] = (3, 5),
        min_moves: int = 4,
        max_moves: int = 12,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.disk_count = disk_count
        self.disk_count_range = disk_count_range
        self.min_moves = int(min_moves)
        self.max_moves = int(max_moves)
        if self.min_moves < 1 or self.max_moves < self.min_moves:
            raise ValueError("Require 1 <= min_moves <= max_moves for tower_of_hanoi.")
        self.slot_count = 3

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _choose_disk_count(self) -> int:
        if self.disk_count is not None:
            if not 3 <= int(self.disk_count) <= 7:
                raise ValueError("disk_count must lie in 3..7")
            return int(self.disk_count)
        lo, hi = self.disk_count_range
        return self.rng.randint(max(3, int(lo)), min(5, int(hi)))

    def _random_state(self, n: int) -> Tuple[Tuple[int, ...], ...]:
        # Place each disk on a random peg, but maintain size order on each peg.
        pegs: List[List[int]] = [[], [], []]
        # Disks: 1 = smallest, n = largest.
        for disk in range(n, 0, -1):
            peg = self.rng.randrange(3)
            pegs[peg].append(disk)
        return tuple(tuple(p) for p in pegs)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            n = self._choose_disk_count()
            initial = self._random_state(n)
            target = self._random_state(n)
            if initial == target:
                continue
            moves = _hanoi_optimal(initial, target, self.max_moves)
            if moves is None:
                continue
            if self.min_moves <= len(moves) <= self.max_moves:
                return initial, target, moves
        raise RuntimeError(
            "tower_of_hanoi rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state: Any) -> Any:
        return tuple(tuple(stack) for stack in state)

    def _apply_move(self, state: Any, move: Any) -> Any:
        src, dst = move
        if not (0 <= int(src) < 3 and 0 <= int(dst) < 3) or int(src) == int(dst):
            raise ValueError(f"invalid Hanoi move {move}")
        new_state: List[List[int]] = [list(stack) for stack in state]
        if not new_state[int(src)]:
            raise ValueError(f"source peg {src} is empty")
        top = new_state[int(src)][-1]
        if new_state[int(dst)] and new_state[int(dst)][-1] < top:
            raise ValueError(f"cannot place disk {top} on smaller disk {new_state[int(dst)][-1]}")
        disk = new_state[int(src)].pop()
        new_state[int(dst)].append(disk)
        return tuple(tuple(stack) for stack in new_state)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _disk_color(self, disk: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(disk - 1) % len(ITEM_PALETTE)]

    def _disk_dimensions(
        self,
        layout: Dict[str, Any],
        n: int,
    ) -> Dict[int, Tuple[float, float]]:
        """Map disk id (1=smallest..N=largest) to (width, height)."""

        slot_w = layout["slot_w"]
        max_w = slot_w * 0.88
        min_w = max(36.0, slot_w * 0.40)
        result: Dict[int, Tuple[float, float]] = {}
        for d in range(1, n + 1):
            # Linear interpolation: smallest disk is min_w, largest is max_w.
            t = (d - 1) / max(1, n - 1)
            w = min_w + (max_w - min_w) * t
            result[d] = (w, layout["item_h"])
        return result

    def _layout(self, panel_w: int, panel_h: int, n: int) -> Dict[str, Any]:
        layout = self._slot_layout(
            panel_w,
            slot_count=3,
            item_size=panel_w * 0.50,
            max_stack=n,
            panel_height=panel_h,
        )
        # Anchor the boards near the vertical centre and let the disks use a
        # larger share of the panel instead of leaving a tall empty band.
        desired_pedestal_y = int(round(panel_h * 0.80))
        vertical_shift = desired_pedestal_y - layout["pedestal_y"]
        layout["pedestal_y"] += vertical_shift
        layout["slot_floor_y"] += vertical_shift
        available = layout["slot_floor_y"] - layout["margin_top"]
        layout["item_h"] = min(30.0, max(16.0, (available * 0.62) / max(1, n)))
        return layout

    def _peg_top_y(self, layout: Dict[str, Any], n: int) -> float:
        return layout["slot_floor_y"] - n * (layout["item_h"] + 2) - 12

    @staticmethod
    def _draw_disk(draw, bbox, color, height):
        from data.common.prompt_paraphrase import active_profile, draw_crate
        if active_profile("hanoi_crates"):
            draw_crate(draw,bbox,color,PEG_DARK)
        else:
            draw.rounded_rectangle(bbox,radius=max(5,int(height*.42)),fill=color,outline=PEG_DARK,width=2)

    def _render_panel(
        self,
        state: Tuple[Tuple[int, ...], ...],
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        n = max((max(stack) if stack else 0) for stack in state) if any(state) else 3
        n = max(3, n)
        layout = self._layout(panel_w, panel_h, n)
        from data.common.prompt_paraphrase import active_profile
        winged=active_profile("hanoi_crates")
        image, draw = self._panel_canvas(panel_w,panel_h,(233,226,199) if winged else PANEL_BG)
        # Panel border for clarity.
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        # Pegs.
        peg_w = max(6, int(round(panel_w * 0.020)))
        peg_top_y = self._peg_top_y(layout, n)
        peg_bottom_y = layout["pedestal_y"]
        for cx in layout["slot_centers_x"]:
            cx = int(round(cx))
            draw.rounded_rectangle(
                (cx - peg_w // 2, peg_top_y, cx + peg_w // 2, peg_bottom_y + 2),
                radius=max(2, peg_w // 2),
                fill=PEG_COLOR,
                outline=PEG_DARK,
            )
        # Wood base under the pegs.
        self._slot_pedestal(draw, layout)
        # Disk dimensions reflect order N..1 across the puzzle.
        dims = self._disk_dimensions(layout, n)

        moving_disk: Optional[int] = None
        moving_pos: Optional[Tuple[float, float]] = None
        if moving is not None:
            moving_disk = int(moving["disk"])
            moving_pos = moving["center"]

        # Render each peg's stack.
        for peg_index, stack in enumerate(state):
            cx = layout["slot_centers_x"][peg_index]
            depth_y = layout["slot_floor_y"]
            for depth, disk in enumerate(stack):
                if moving is not None and moving.get("hide_top", False) and\
                        depth == len(stack) - 1 and peg_index == moving["src"]:
                    continue
                w, h = dims[disk]
                bbox = (cx - w / 2, depth_y - h, cx + w / 2, depth_y)
                color = self._disk_color(disk)
                self._draw_disk(draw,bbox,color,h)
                depth_y -= h + 2

        if moving_disk is not None and moving_pos is not None:
            w, h = dims[moving_disk]
            bbox = (moving_pos[0] - w / 2, moving_pos[1] - h / 2,
                    moving_pos[0] + w / 2, moving_pos[1] + h / 2)
            color = self._disk_color(moving_disk)
            self._draw_disk(draw,bbox,color,h)

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

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _animate_move(
        self,
        state: Any,
        move: Any,
        target_state: Any,
    ) -> List[Image.Image]:
        src, dst = move
        moving_disk = state[src][-1]
        # Compute pixel positions on the LEFT panel.
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        n = max((max(stack) if stack else 0) for stack in state) if any(state) else 3
        n = max(3, n)
        layout = self._layout(panel_w, panel_h, n)
        dims = self._disk_dimensions(layout, n)
        h = dims[moving_disk][1]

        src_cx = layout["slot_centers_x"][src]
        dst_cx = layout["slot_centers_x"][dst]
        src_top_depth = len(state[src]) - 1
        dst_top_depth = len(state[dst])
        src_y = layout["slot_floor_y"] - src_top_depth * (h + 2) - h / 2
        dst_y = layout["slot_floor_y"] - dst_top_depth * (h + 2) - h / 2
        minimal_clearance = self._peg_top_y(layout, n) - h / 2 - 5

        # Right panel: target state stays static.
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)

        # Three phases: lift (move_frames/3), traverse (move_frames/3), drop (move_frames/3).
        frames: List[Image.Image] = []
        n_lift = max(2, self.move_frames // 3)
        n_traverse = max(2, self.move_frames // 3)
        n_drop = max(2, self.move_frames - n_lift - n_traverse)

        # Lift phase: stay above src_cx, move y from src_y to apex_y.
        for k in range(1, n_lift + 1):
            t = k / n_lift
            cx, y = self._interpolate_pixel_linear(
                (src_cx, src_y), (src_cx, minimal_clearance), t
            )
            anim = {"disk": moving_disk, "center": (cx, y), "src": src, "hide_top": True}
            left_image = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left_image, right_image))

        # Traverse phase: the disk clears the compact peg tops by a small gap.
        for k in range(1, n_traverse + 1):
            t = k / n_traverse
            cx, y = self._interpolate_pixel_linear(
                (src_cx, minimal_clearance), (dst_cx, minimal_clearance), t
            )
            anim = {"disk": moving_disk, "center": (cx, y), "src": src, "hide_top": True}
            left_image = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left_image, right_image))

        # Drop phase.
        for k in range(1, n_drop + 1):
            t = k / n_drop
            cx, y = self._interpolate_pixel_linear(
                (dst_cx, minimal_clearance), (dst_cx, dst_y), t
            )
            anim = {"disk": moving_disk, "center": (cx, y), "src": src, "hide_top": True}
            left_image = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left_image, right_image))

        after_state = self._apply_move(state, move)
        if frames:
            frames[-1] = self._render_static_scene(after_state, target_state)
        return frames

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    def _disk_name(self, disk: int) -> str:
        return f"the {DISK_COLOR_NAMES[(disk - 1) % len(DISK_COLOR_NAMES)]} disk"

    def _stack_listing(self, stack: Sequence[int]) -> str:
        if not stack:
            return "no disks"
        names = [self._disk_name(disk) for disk in stack]
        if len(names) == 1:
            return names[0]
        return ", then ".join(names) + ", reading bottom to top"

    def _peg_listing(self, state: Sequence[Sequence[int]]) -> str:
        return "; ".join(
            f"peg {peg} holds {self._stack_listing(stack)}"
            if stack
            else f"peg {peg} is empty"
            for peg, stack in enumerate(state, start=1)
        )

    def _author_observation(self, initial_state, target_state, moves) -> str:
        misplaced = sorted(
            {
                disk
                for peg in range(3)
                for position, disk in enumerate(initial_state[peg])
                if position >= len(target_state[peg])
                or target_state[peg][position] != disk
            }
            | {
                disk
                for peg in range(3)
                for position, disk in enumerate(target_state[peg])
                if position >= len(initial_state[peg])
                or initial_state[peg][position] != disk
            }
        )
        colors = [DISK_COLOR_NAMES[(disk - 1) % len(DISK_COLOR_NAMES)] for disk in misplaced]
        if len(colors) > 1:
            misplaced_names = ", ".join(colors[:-1]) + f" and {colors[-1]} disks"
        else:
            misplaced_names = f"{colors[0]} disk"
        return (
            f"On the left board, {self._peg_listing(initial_state)}. "
            f"The right board asks for this arrangement: {self._peg_listing(target_state)}. "
            f"Only a peg's top disk can travel and nothing may rest on a narrower disk, "
            f"so the {misplaced_names} still have to relocate or be uncovered."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        src, dst = int(move[0]), int(move[1])
        visible_src, visible_dst = to_visible_move(self.TASK_NAME, (src, dst))
        disk = previous_state[src][-1]
        destination_stack = previous_state[dst]
        if destination_stack:
            landing = f"onto the wider {self._disk_name(destination_stack[-1])[4:]} on peg {visible_dst}"
        else:
            landing = f"onto the empty peg {visible_dst}"
        # Honest purpose: settling into its final place, or uncovering the disk
        # beneath it, or simply parking out of the way.
        settled = (
            len(next_state[dst]) <= len(target_state[dst])
            and tuple(target_state[dst][: len(next_state[dst])]) == tuple(next_state[dst])
        )
        uncovered = previous_state[src][:-1]
        if settled:
            purpose = "that settles it exactly where the target wants it"
        elif uncovered:
            purpose = f"that frees {self._disk_name(uncovered[-1])} underneath"
        else:
            purpose = "that parks it out of the way so the other pegs stay workable"
        return (
            f"{self._disk_name(disk).capitalize()} is on top of peg {visible_src}, so I lift it {landing}; "
            f"{purpose}. Now {self._peg_listing(next_state)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Every peg now matches the right board exactly, and each of the {len(moves)} lifts obeyed "
            "the top-disk and size rules."
        )

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side wooden Tower of Hanoi boards on a beige canvas. The left "
            "board carries the starting configuration and the right board carries the target configuration. Each board "
            "has three short vertical pegs and a stack of flat colored disks whose widths encode size. A blue chevron "
            "arrow between the two boards "
            "points from the left board to the right board."
        )

    def _describe_rules(self) -> str:
        return (
            "Each move takes the topmost disk of one peg and places it on top of another peg. A disk may never sit "
            "on top of a strictly smaller disk. Wider disks are larger; pegs are numbered 1, 2, 3 from left to right."
        )

    def _describe_output(self) -> str:
        return (
            "Return the shortest valid sequence of moves that transforms the left board into the right board. Format "
            "the answer as `Answer: [(<source_peg>, <destination_peg>), ...].`"
        )

    def _format_move(self, move: Any) -> str:
        """Publish one internal peg pair with the visible one-based numbering."""

        src, dst = to_visible_move(self.TASK_NAME, (int(move[0]), int(move[1])))
        return f"({src}, {dst})"

    def _extra_payload(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> Optional[Dict[str, Any]]:
        return {
            "peg_count": 3,
            "disk_count": sum(len(s) for s in initial_state),
            "optimal_moves": [list(m) for m in moves],
        }


__all__ = ["TowerOfHanoiGenerator"]
