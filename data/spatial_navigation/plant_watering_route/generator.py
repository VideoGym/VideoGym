"""Gardener watering route with capacity-K watering can.

A small 4x4 garden grid contains a WELL (refills the watering can to
capacity K) and 3-4 PLANT POTS (each pot needs 1 cup of water; a
watered pot is shown filled with green leaves; an un-watered pot is
brown/dry). The gardener starts at the WELL with a full can.

Each action is in {U, D, L, R, S}. The four direction actions move the
gardener one cell, while S leaves the gardener on the current cell. When
the gardener steps onto a dry plant pot AND has at least 1 cup in the can,
the can auto-pours 1 cup, watering the pot and decrementing the can level.
The gardener may enter the WELL for one refill after the initial full can.
Entering it again after that refill is illegal; staying on the well does not
spend another refill.

The puzzle goal is to water EVERY plant with any complete legal move sequence.

Move notation: a single character per move from {U, D, L, R, S}, where S stays
on the current cell.

Mechanic load-bearing check (Tier B): we generate puzzles where the
shortest plan with capacity K is strictly LONGER than the same plan with
infinite capacity (so the solver must visit the well at least once mid-
tour). Otherwise the capacity is decorative.
"""

from __future__ import annotations

from data.common.assets import render_font

from collections import deque
from pathlib import Path
from typing import Any, FrozenSet, List, Optional, Tuple

from PIL import ImageFont

from data.common.grid_coordinates import (
    GRID_PATH_ANSWER_FORMAT_INSTRUCTION,
    GRID_PATH_STAY_ENCODING_SENTENCE,
    GRID_XY_CONVENTION_SENTENCE,
    format_xy_path_payload,
)
from data.common.pipeline.object_plans import (
    ObjectPlanGenerator,
    _shade,
)
from data.common.render_primitives import DOCTRINE_SUPERSAMPLE
from data.spatial_navigation.core.prompt_contract import build_spatial_navigation_prompt_bundle

PANEL_BG = (224, 232, 220)
PANEL_BORDER = (180, 200, 172)
SOIL_FILL = (188, 168, 132)
SOIL_BORDER = (148, 124, 88)
WELL_BORDER = (96, 84, 60)
WELL_STONE = (196, 192, 184)
# Kept far enough from GARDENER_BODY that the verifier's colour-keyed token
# tracker can never confuse standing water with the moving gardener.
WELL_WATER = (150, 196, 224)
POT_DRY = (160, 102, 64)
POT_DRY_RIM = (108, 70, 36)
POT_WET_LEAF = (96, 168, 84)
POT_WET_LEAF_DARK = (60, 122, 60)
GARDENER_HAT = (208, 168, 80)
GARDENER_BODY = (96, 132, 188)
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1), "S": (0, 0)}
ROWS = 4
COLS = 4


def _bfs(
    start: Tuple[int, int],
    well: Tuple[int, int],
    plants: FrozenSet[Tuple[int, int]],
    capacity: int,
    max_moves: int,
    *,
    refill_budget: int,
    infinite_cap: bool = False,
) -> Optional[List[str]]:
    """state = (pos, can_level, watered_set, refills_used)"""
    init_state = (
        start,
        capacity if not infinite_cap else 10**6,
        frozenset(),
        0,
    )
    if not plants:
        return []
    queue = deque([(init_state, [])])
    seen = {init_state}
    while queue:
        (pos, can, watered, refills_used), path = queue.popleft()
        if len(path) >= max_moves:
            continue
        for d_name, (dr, dc) in DIRS.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            new_pos = (nr, nc)
            new_can = can
            new_watered = set(watered)
            new_refills_used = refills_used
            if new_pos == well and new_pos != pos:
                if not infinite_cap:
                    if new_refills_used >= refill_budget:
                        continue
                    new_refills_used += 1
                    new_can = capacity
                else:
                    new_can = 10**6
            # If on plant and not yet watered and have water, water it.
            if new_pos in plants and new_pos not in watered and new_can > 0:
                new_watered.add(new_pos)
                if not infinite_cap:
                    new_can -= 1
            new_state = (
                new_pos,
                new_can,
                frozenset(new_watered),
                new_refills_used,
            )
            if new_state in seen:
                continue
            seen.add(new_state)
            np = path + [d_name]
            if frozenset(new_watered) == plants:
                return np
            queue.append((new_state, np))
    return None


class PlantWateringRouteGenerator(ObjectPlanGenerator):
    TASK_NAME = "plant_watering_route"
    TASK_FAMILY = "spatial_navigation"
    SPATIAL_NAVIGATION_DISCRETE_ANIMATION = True
    DEFAULT_OUTPUT_DIR = "outputs/spatial_navigation/plant_watering_route"
    CAPACITY = 2
    REFILL_BUDGET = 1

    def _build_task_prompts(self):
        return build_spatial_navigation_prompt_bundle(
            task_name=self.TASK_NAME,
            input_description=self._describe_input(),
            rule_description=self._describe_rules(),
            ti2v_output=(
                "Replay a complete legal solution as successive complete grid states. Move the same gardener one "
                "orthogonal cell for U, D, L, or R, and leave the gardener in place for S. On arrival, commit any "
                "automatic watering or the single well refill together with the matching can level, pot color, "
                "and REFILLS badge; retain every earlier watered pot and every unchanged object."
            ),
        )

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        max_optimal_moves: int = 12,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.max_optimal_moves = int(max_optimal_moves)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            cells = [(r, c) for r in range(ROWS) for c in range(COLS)]
            self.rng.shuffle(cells)
            well = cells[0]
            n_plants = self.rng.randint(3, 4)
            plants = frozenset(cells[1:1 + n_plants])
            start = well
            plan = _bfs(
                start,
                well,
                plants,
                self.CAPACITY,
                self.max_optimal_moves,
                refill_budget=self.REFILL_BUDGET,
            )
            if plan is None:
                continue
            if not (5 <= len(plan) <= self.max_optimal_moves):
                continue
            # Mechanic load-bearing: infinite capacity must give strictly shorter plan.
            plan_inf = _bfs(
                start,
                well,
                plants,
                self.CAPACITY,
                self.max_optimal_moves,
                refill_budget=self.REFILL_BUDGET,
                infinite_cap=True,
            )
            if plan_inf is None or len(plan_inf) >= len(plan):
                continue
            initial_state = {
                "gardener": list(start),
                "well": list(well),
                "plants": [list(p) for p in sorted(plants)],
                "watered": [],
                "can": self.CAPACITY,
                "refill_budget": self.REFILL_BUDGET,
                "refills_used": 0,
            }
            target_state = initial_state
            for move in plan:
                target_state = self._apply_move(target_state, move)
            if int(target_state["refills_used"]) != self.REFILL_BUDGET:
                continue
            return initial_state, target_state, list(plan)
        # Fallback.
        plants = frozenset({(0, 1), (2, 2), (3, 0)})
        well = (1, 1)
        start = (1, 1)
        plan = _bfs(
            start,
            well,
            plants,
            self.CAPACITY,
            24,
            refill_budget=self.REFILL_BUDGET,
        ) or []
        initial_state = {
            "gardener": list(start),
            "well": list(well),
            "plants": [list(p) for p in sorted(plants)],
            "watered": [],
            "can": self.CAPACITY,
            "refill_budget": self.REFILL_BUDGET,
            "refills_used": 0,
        }
        target_state = initial_state
        for move in plan:
            target_state = self._apply_move(target_state, move)
        return initial_state, target_state, plan

    def _state_signature(self, state):
        return (
            tuple(state["gardener"]),
            tuple(sorted(tuple(w) for w in state["watered"])),
            int(state["can"]),
            int(state["refills_used"]),
        )

    def _serialize_state(self, state):
        return {
            "gardener": list(state["gardener"]),
            "well": list(state["well"]),
            "plants": [list(p) for p in state["plants"]],
            "watered": [list(p) for p in state["watered"]],
            "can": int(state["can"]),
            "refill_budget": int(state["refill_budget"]),
            "refills_used": int(state["refills_used"]),
        }

    def _apply_move(self, state, move):
        d_name = str(move).upper()
        dr, dc = DIRS[d_name]
        new_pos = [state["gardener"][0] + dr, state["gardener"][1] + dc]
        if not (0 <= new_pos[0] < ROWS and 0 <= new_pos[1] < COLS):
            raise ValueError("out of bounds")
        new_can = int(state["can"])
        new_watered = [list(w) for w in state["watered"]]
        refill_budget = int(state["refill_budget"])
        refills_used = int(state["refills_used"])
        if list(new_pos) == list(state["well"]) and list(new_pos) != list(state["gardener"]):
            if refills_used >= refill_budget:
                raise ValueError("refill budget exhausted")
            refills_used += 1
            new_can = self.CAPACITY
        plants_list = [list(p) for p in state["plants"]]
        if list(new_pos) in plants_list and list(new_pos) not in new_watered and new_can > 0:
            new_watered.append(list(new_pos))
            new_can -= 1
        return {
            "gardener": list(new_pos),
            "well": list(state["well"]),
            "plants": [list(p) for p in state["plants"]],
            "watered": new_watered,
            "can": int(new_can),
            "refill_budget": refill_budget,
            "refills_used": refills_used,
        }

    def _layout(self, panel_w, panel_h):
        margin = max(8, int(min(panel_w, panel_h) * 0.05))
        label_gutter = max(18, int(min(panel_w, panel_h) * 0.09))
        usable = min(panel_w - 2 * margin - label_gutter, panel_h - 2 * margin)
        cell = usable // max(ROWS, COLS)
        cell -= cell % 2
        bw = cell * COLS
        bh = cell * ROWS
        x0 = (panel_w - bw - label_gutter) // 2 + label_gutter
        y0 = (panel_h - bh) // 2
        return {
            "panel_w": panel_w, "panel_h": panel_h,
            "cell": cell, "board_x0": x0, "board_y0": y0,
            "board_w": bw, "board_h": bh,
            "label_gutter": label_gutter,
        }

    def _cell_center(self, layout, r, c):
        return (
            layout["board_x0"] + (c + 0.5) * layout["cell"],
            layout["board_y0"] + (r + 0.5) * layout["cell"],
        )

    def _draw_garden(self, draw, layout):
        # Soil background.
        x0 = layout["board_x0"]
        y0 = layout["board_y0"]
        x1 = x0 + layout["board_w"]
        y1 = y0 + layout["board_h"]
        draw.rectangle((x0, y0, x1, y1), fill=SOIL_FILL, outline=SOIL_BORDER, width=2)
        # Faint cell grid (path texture).
        for c in range(1, COLS):
            xs = x0 + c * layout["cell"]
            draw.line((xs, y0 + 4, xs, y1 - 4), fill=_shade(SOIL_FILL, 0.85), width=1)
        for r in range(1, ROWS):
            ys = y0 + r * layout["cell"]
            draw.line((x0 + 4, ys, x1 - 4, ys), fill=_shade(SOIL_FILL, 0.85), width=1)
        axis_font = self._panel_font(max(9, int(layout["cell"] * 0.22)))
        for col in range(COLS):
            draw.text(
                (x0 + (col + 0.5) * layout["cell"], y0 - 5),
                str(col + 1),
                fill=(30, 90, 220),
                font=axis_font,
                anchor="mb",
            )
        for row in range(ROWS):
            draw.text(
                (x0 - 6, y0 + (row + 0.5) * layout["cell"]),
                str(row + 1),
                fill=(30, 90, 220),
                font=axis_font,
                anchor="rm",
            )

    def _draw_well(self, draw, cx, cy, cell):
        size = cell * 0.74
        x0 = cx - size / 2
        y0 = cy - size / 2
        x1 = cx + size / 2
        y1 = cy + size / 2
        # Stone base ring.
        draw.ellipse((x0, y0 + size * 0.40, x1, y1), fill=WELL_STONE, outline=(120, 116, 108), width=2)
        # Water surface inside the well.
        draw.ellipse((x0 + size * 0.18, y0 + size * 0.55, x1 - size * 0.18, y1 - size * 0.10), fill=WELL_WATER, outline=(60, 110, 156), width=1)
        # Wood roof (pitched triangle).
        draw.polygon(
            [(cx, y0 + size * 0.05), (x0 + size * 0.10, y0 + size * 0.45), (x1 - size * 0.10, y0 + size * 0.45)],
            fill=(180, 132, 80), outline=(120, 84, 40),
        )
        # Two posts.
        draw.line((x0 + size * 0.20, y0 + size * 0.40, x0 + size * 0.20, y0 + size * 0.60), fill=WELL_BORDER, width=3)
        draw.line((x1 - size * 0.20, y0 + size * 0.40, x1 - size * 0.20, y0 + size * 0.60), fill=WELL_BORDER, width=3)

    def _draw_pot(self, draw, cx, cy, cell, watered: bool):
        size = cell * 0.66
        x0 = cx - size / 2
        x1 = cx + size / 2
        y0 = cy - size * 0.10
        y1 = cy + size * 0.40
        # Pot body.
        polygon = [(x0 + size * 0.12, y0), (x1 - size * 0.12, y0), (x1, y1), (x0, y1)]
        draw.polygon(polygon, fill=POT_DRY, outline=POT_DRY_RIM)
        # Rim band.
        draw.rectangle((x0 + size * 0.08, y0 - size * 0.05, x1 - size * 0.08, y0 + size * 0.10), fill=_shade(POT_DRY, 0.78), outline=POT_DRY_RIM, width=1)
        if watered:
            # Leaves popping up.
            leaf_cx = cx
            leaf_cy = y0 - size * 0.10
            r = size * 0.30
            draw.ellipse((leaf_cx - r, leaf_cy - r, leaf_cx + r, leaf_cy + r * 0.30), fill=POT_WET_LEAF, outline=POT_WET_LEAF_DARK, width=1)
            # Three leaf tufts.
            for fx in (-0.45, 0.0, 0.45):
                lx = leaf_cx + r * fx
                ly = leaf_cy - r * 0.20 + abs(fx) * r * 0.10
                draw.ellipse((lx - r * 0.30, ly - r * 0.55, lx + r * 0.30, ly + r * 0.20), fill=POT_WET_LEAF, outline=POT_WET_LEAF_DARK, width=1)
            # Stem line.
            draw.line((leaf_cx, leaf_cy + r * 0.20, leaf_cx, y0 + size * 0.05), fill=POT_WET_LEAF_DARK, width=2)
        else:
            # Dry: faint cracks lines.
            draw.line((x0 + size * 0.20, y0 + size * 0.20, x1 - size * 0.30, y1 - size * 0.10), fill=POT_DRY_RIM, width=1)

    def _draw_gardener(self, draw, cx, cy, cell, can_level: int):
        size = cell * 0.64
        # Body.
        draw.rounded_rectangle((cx - size * 0.30, cy - size * 0.05, cx + size * 0.30, cy + size * 0.40), radius=int(size * 0.10), fill=GARDENER_BODY, outline=_shade(GARDENER_BODY, 0.55), width=1)
        # Head.
        head_r = size * 0.20
        draw.ellipse((cx - head_r, cy - size * 0.40, cx + head_r, cy - size * 0.40 + 2 * head_r), fill=(252, 222, 188), outline=(118, 90, 68), width=1)
        # Hat.
        draw.rectangle((cx - size * 0.30, cy - size * 0.42, cx + size * 0.30, cy - size * 0.30), fill=GARDENER_HAT, outline=(160, 132, 60), width=1)
        draw.rectangle((cx - size * 0.16, cy - size * 0.55, cx + size * 0.16, cy - size * 0.30), fill=GARDENER_HAT, outline=(160, 132, 60), width=1)
        # Watering can (small, in the right hand).  The can's water level is
        # its only state encoding: the blue fill rises from empty to full in
        # discrete steps of the two-cup capacity.
        can_x = cx + size * 0.32
        can_y = cy + size * 0.10
        can_w = size * 0.30
        can_h = size * 0.22
        if can_level > 0:
            water_h = max(2, int(can_h * (can_level / 2.0)))
            draw.rectangle((can_x - can_w / 2, can_y + can_h / 2 - water_h, can_x + can_w / 2, can_y + can_h / 2), fill=WELL_WATER)
        draw.rectangle((can_x - can_w / 2, can_y - can_h / 2, can_x + can_w / 2, can_y + can_h / 2), outline=(80, 80, 88), width=2)
        # Spout.
        draw.line((can_x + can_w / 2, can_y - can_h * 0.10, can_x + can_w * 1.30, can_y - can_h * 0.40), fill=(80, 80, 88), width=3)
        # Handle.
        draw.arc((can_x - can_w * 0.5, can_y - can_h * 0.6, can_x + can_w * 0.4, can_y - can_h * 0.05), 180, 360, fill=(80, 80, 88), width=2)

    def _panel_font(self, size: int) -> ImageFont.ImageFont:
        return render_font(int(size) * DOCTRINE_SUPERSAMPLE, bold=False)

    def _render_panel(self, state, panel_w, panel_h, *, moving=None):
        layout = self._layout(panel_w, panel_h)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        budget_font = self._panel_font(max(9, int(min(panel_w, panel_h) * 0.035)))
        draw.text(
            (panel_w - 8, 7),
            f"REFILLS {int(state['refills_used'])}/{int(state['refill_budget'])}",
            fill=(52, 82, 112),
            font=budget_font,
            anchor="ra",
        )
        self._draw_garden(draw, layout)
        cell = layout["cell"]
        # Plants (watered/dry).
        watered = {tuple(w) for w in state["watered"]}
        for p in state["plants"]:
            r, c = int(p[0]), int(p[1])
            cx, cy = self._cell_center(layout, r, c)
            self._draw_pot(draw, cx, cy, cell, watered=tuple(p) in watered)
        # Well.
        wr, wc = int(state["well"][0]), int(state["well"][1])
        wcx, wcy = self._cell_center(layout, wr, wc)
        self._draw_well(draw, wcx, wcy, cell)
        # Gardener.
        if moving is not None and "gardener_pix" in moving:
            gx, gy = moving["gardener_pix"]
        else:
            gr, gc = int(state["gardener"][0]), int(state["gardener"][1])
            gx, gy = self._cell_center(layout, gr, gc)
        self._draw_gardener(draw, gx, gy, cell, int(state["can"]))
        # Hand the panel over still supersampled: the scene compositor pastes
        # both panels into a 4x canvas, draws the chevron divider there, and
        # takes the single LANCZOS pass for the whole artifact.  Finishing the
        # panel here instead forced the divider onto a 44px-wide band canvas of
        # its own and resampled the scene twice.
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side 4x4 garden plots on brown soil. "
            f"{GRID_XY_CONVENTION_SENTENCE} Each plot contains "
            "one wooden well (round stone ring with a tile roof and visible blue water inside), 3-4 plant pots "
            "(brown clay pots — a dry pot shows only the empty pot; a watered pot shows a green leafy plant "
            "rising from it), and a gardener carrying a small watering can whose blue water fill shows how many "
            "cups remain. The left plot shows the starting state (gardener at the well, can full, all pots dry); "
            "the right plot shows the goal state (every pot watered). A blue chevron arrow between the plots "
            "points from left to right, and a REFILLS used/1 badge sits at the top of each plot. Both plots use the "
            "same blue edge numbers and the same coordinate frame."
        )

    def _describe_rules(self) -> str:
        return (
            f"Each step uses one action from {{U, D, L, R, S}}; U, D, L, and R move one cell, while S leaves the "
            f"gardener on the current cell. The watering can holds at most "
            f"{self.CAPACITY} cups. When the gardener steps onto a plant pot AND has at least 1 cup remaining, "
            f"the pot is automatically watered (turns green) and the can level decreases by 1. When the gardener "
            f"enters the well after leaving it, the can refills to {self.CAPACITY} cups and uses the single allowed "
            f"refill. Entering the well again after refill 1/1 is illegal; staying on the well does not use another "
            f"refill. The goal is for every plant pot to end watered."
        )

    def _describe_output(self) -> str:
        return (
            "Return the complete ordered path of the gardener on the shared 4x4 grid, including the well start and "
            f"the final position. {GRID_PATH_STAY_ENCODING_SENTENCE} {GRID_PATH_ANSWER_FORMAT_INSTRUCTION}"
        )

    def _format_move(self, move):
        return f"'{str(move).upper()}'"

    def _format_answer_from_trace(self, moves, state_trace):
        del moves
        path = [tuple(state["gardener"]) for state in state_trace]
        return format_xy_path_payload(path)

    def _extra_payload(self, initial_state, target_state, moves):
        return {
            "rows": ROWS, "cols": COLS,
            "well": list(initial_state["well"]),
            "plants": [list(p) for p in initial_state["plants"]],
            "capacity": self.CAPACITY,
            "refill_budget": self.REFILL_BUDGET,
            "optimal_moves": [str(m).upper() for m in moves],
        }


__all__ = ["PlantWateringRouteGenerator"]
