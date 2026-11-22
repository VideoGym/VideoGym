"""Switch toggles a group of wall states.

The grid contains a set of "phase A" walls and a set of "phase B" walls. Initially
phase A walls are solid; stepping on the orange/blue switch toggles which set
is solid. The agent must navigate to the switch first to flip the wall phase,
then continue to the goal. Without flipping, start→goal is unreachable.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    GRID_PATH_STAY_ENCODING_SENTENCE,
    cell_to_xy,
)
from data.spatial_navigation.core.maze_style import route_run_breakdown
from data.spatial_navigation.core.mechanic_base import (
    COLOR_WALL,
    COLOR_WALL_OUTLINE,
    MechanicGridGeneratorBase,
    MechanicScenario,
)


def _bfs(size: int, start: Tuple[int, int], goal: Tuple[int, int], walls: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    if start == goal:
        return [start]
    queue = deque([start])
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {start: start}
    while queue:
        cell = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in walls or nxt in parent:
                continue
            parent[nxt] = cell
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(nxt)
    return None


class SwitchToggleWallsGenerator(MechanicGridGeneratorBase):
    TASK_NAME = "switch_toggle_walls"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Switch Toggle Walls"
    DEFAULT_SIZE = 7

    def _authored_reasoning(self, scenario: MechanicScenario) -> str:
        path = list(scenario.agent_path)
        switch = tuple(scenario.extras["switch_cell"])
        switch_step = int(scenario.extras["switch_step"])
        walls_a = [tuple(c) for c in scenario.extras["walls_a"]]
        rows_a = {cell[0] for cell in walls_a}
        cols_a = {cell[1] for cell in walls_a}
        if len(rows_a) == 1:
            barrier = f"the solid A wall filling all positions with x = {next(iter(rows_a)) + 1}"
        elif len(cols_a) == 1:
            barrier = f"the solid A wall filling all positions with y = {next(iter(cols_a)) + 1}"
        else:
            barrier = "the solid A walls"
        sentences: List[str] = [
            f"S at {cell_to_xy(path[0])} is sealed off from G at {cell_to_xy(path[-1])} by {barrier}, so no direct "
            f"walk works. The switch X at {cell_to_xy(switch)} is reachable on this side, and stepping on it swaps "
            "the phases: the A cells open into dashed outlines while the dashed B cells harden into slate.",
        ]
        first_leg = route_run_breakdown(path[: switch_step + 1])
        if first_leg:
            leg = ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in first_leg)
            sentences.append(f"The route first walks {leg} onto X, committing the one phase flip.")
        second_leg = route_run_breakdown(path[switch_step:])
        if second_leg:
            leg = ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in second_leg)
            crossed = [cell for cell in path[switch_step:] if cell in walls_a]
            crossing_note = (
                f", passing through the now-open A cell at {cell_to_xy(crossed[0])}" if crossed else ""
            )
            sentences.append(f"After the flip it walks {leg}{crossing_note} and stops on G.")
        sentences.append(
            "Before the flip the route touched no A cell, and after it the route avoids every hardened B cell, so "
            "each step lands on walkable floor."
        )
        return " ".join(sentences)

    def sample_scenario(self) -> MechanicScenario:
        for _ in range(2048):
            cells = [(r, c) for r in range(self.size) for c in range(self.size)]
            self.rng.shuffle(cells)
            start, goal, switch = cells[0], cells[1], cells[2]
            if len({start, goal, switch}) < 3:
                continue
            # Strategy: pick a barrier ROW or COL that together with a few B walls makes phase A blocked.
            # Phase A walls = a row/column band between start and switch+goal that BLOCKS the direct route.
            # Phase B walls = scattered minor obstacles that block the post-switch direct route just enough
            # to make full path interesting.
            walls_a: Set[Tuple[int, int]] = set()
            # Build a barrier on the row just past start's row, broken by 1 cell where the switch sits.
            barrier_axis = self.rng.choice(("row", "col"))
            if barrier_axis == "row":
                br = (start[0] + goal[0]) // 2
                for c in range(self.size):
                    walls_a.add((br, c))
            else:
                bc = (start[1] + goal[1]) // 2
                for r in range(self.size):
                    walls_a.add((r, bc))
            # Carve out start, goal, switch from A walls
            walls_a.discard(start)
            walls_a.discard(goal)
            walls_a.discard(switch)

            extras = [c for c in cells if c not in walls_a and c not in {start, goal, switch}]
            walls_b: Set[Tuple[int, int]] = set()
            wall_b_count = self.rng.randint(2, 4)
            for cand in extras:
                if len(walls_b) >= wall_b_count:
                    break
                walls_b.add(cand)

            phase_a = walls_a
            phase_b = walls_b

            seg_a = _bfs(self.size, start, switch, phase_a)
            if seg_a is None:
                continue
            seg_b = _bfs(self.size, switch, goal, phase_b)
            if seg_b is None:
                continue
            full_path = seg_a + seg_b[1:]
            unreachable = _bfs(self.size, start, goal, phase_a)
            if unreachable is not None:
                continue
            if len(full_path) < 6 or len(full_path) > 24:
                continue
            return self._build_scenario(
                start=start, goal=goal, switch=switch,
                walls_a=sorted(walls_a), walls_b=sorted(walls_b),
                path=full_path,
            )
        raise RuntimeError("switch_toggle_walls: failed to sample scenario.")

    def _build_scenario(
        self,
        *,
        start, goal, switch,
        walls_a, walls_b,
        path,
    ) -> MechanicScenario:
        switch_step = path.index(switch)
        return MechanicScenario(
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            title=self.TASK_TITLE,
            input_desc=(
                f"The image shows a {self.size}x{self.size} grid with row numbers down the left edge and column "
                "numbers across the top edge. The orange agent stands on the green start cell S, the blue cell G is "
                "the goal, and the grey disc labeled X is the switch. The solid dark slate cells are the A walls, "
                "currently blocking. The white cells with a slate dashed outline are the B walls, currently open. "
                f"{GRID_XY_CONVENTION_SENTENCE}"
            ),
            rules_desc=(
                "The agent moves up, right, down, or left between adjacent floor cells. "
                f"It may stay on its current cell. {GRID_PATH_STAY_ENCODING_SENTENCE} "
                "Stepping onto the switch flips the wall phase once for the rest of the run: every A wall opens and "
                "every B wall becomes solid. Solid walls and the grid border block movement; dashed-outline cells "
                "are walkable."
            ),
            ti2v_output=(
                "Generate a video that shows the agent following a complete legal route and stepping on the switch. "
                "At that single commit the switch plate turns amber, every solid A cell switches to a dashed "
                "outline, and every dashed B cell switches to solid slate; then the agent continues to the goal."
            ),
            solution_image_output=(
                "Render the solved still by drawing the complete legal route in red, the A cells as dashed "
                "outlines, the B cells as solid slate, the switch plate amber, and the agent on the goal."
            ),
            size=self.size,
            start=start,
            goal=goal,
            agent_path=path,
            walls=[],  # we render walls custom
            extras={
                "switch_cell": [switch[0], switch[1]],
                "switch_step": switch_step,
                "walls_a": [[r, c] for r, c in walls_a],
                "walls_b": [[r, c] for r, c in walls_b],
            },
            structured_answer={
                "agent_path_cell_ids": self._path_cell_ids(path),
                "switch_cell_id": self._cell_id(switch),
            },
        )

    def _draw_dashed_cell_outline(
        self,
        draw,
        cell: Tuple[int, int],
    ) -> None:
        """Clearly visible slate dashed inset outline: an open phase wall."""

        l, t, r, b = self._cell_box(cell)
        inset = max(4, self.cell_size // 12)
        l, t, r, b = l + inset, t + inset, r - inset, b - inset
        dash = max(5, self.cell_size // 8)
        gap = max(4, dash // 2)

        def dashed_line(x0: float, y0: float, x1: float, y1: float) -> None:
            length = max(abs(x1 - x0), abs(y1 - y0))
            if length <= 0:
                return
            ux = (x1 - x0) / length
            uy = (y1 - y0) / length
            position = 0.0
            while position < length:
                end = min(position + dash, length)
                draw.line(
                    [(x0 + ux * position, y0 + uy * position), (x0 + ux * end, y0 + uy * end)],
                    fill=COLOR_WALL,
                    width=3,
                )
                position = end + gap

        dashed_line(l, t, r, t)
        dashed_line(r, t, r, b)
        dashed_line(r, b, l, b)
        dashed_line(l, b, l, t)

    def _draw_phase_walls(
        self,
        draw,
        cells_a: List[Tuple[int, int]],
        cells_b: List[Tuple[int, int]],
        *,
        phase_b_active: bool,
    ) -> None:
        # One encoding carries the phase state: a blocking phase wall is a
        # solid slate cell, an open phase wall is a white cell with a slate
        # dashed outline.  The single switch commit swaps the two sets.
        for cell in cells_a:
            if not phase_b_active:
                l, t, r, b = self._cell_box(cell)
                draw.rectangle([l, t, r, b], fill=COLOR_WALL, outline=COLOR_WALL_OUTLINE, width=1)
            else:
                self._draw_dashed_cell_outline(draw, cell)
        for cell in cells_b:
            if phase_b_active:
                l, t, r, b = self._cell_box(cell)
                draw.rectangle([l, t, r, b], fill=COLOR_WALL, outline=COLOR_WALL_OUTLINE, width=1)
            else:
                self._draw_dashed_cell_outline(draw, cell)

    def _render(
        self,
        scenario: MechanicScenario,
        *,
        agent_center: Tuple[float, float],
        switch_active: bool,
        show_path: bool = False,
    ) -> Image.Image:
        image, draw = self._begin_scene()
        self._draw_grid_and_labels(draw, wall_set=())
        cells_a = [tuple(c) for c in scenario.extras["walls_a"]]
        cells_b = [tuple(c) for c in scenario.extras["walls_b"]]
        self._draw_phase_walls(draw, cells_a, cells_b, phase_b_active=switch_active)

        if show_path:
            self._draw_path_overlay(draw, scenario.agent_path)

        switch_cell = tuple(scenario.extras["switch_cell"])
        token_font = self._load_font(max(12, self.cell_size // 5))
        self._draw_pressure_plate(draw, switch_cell, active=switch_active, label="X", token_font=token_font)
        self._draw_start_goal(draw, scenario)
        self._draw_agent(draw, agent_center)
        return self._finish_scene(image)

    def create_puzzle(self) -> Dict[str, Any]:  # type: ignore[override]
        scenario = self.sample_scenario()
        switch_step = int(scenario.extras["switch_step"])

        puzzle_image = self._render(scenario, agent_center=self._cell_center(scenario.start), switch_active=False)
        solution_image = self._render(
            scenario, agent_center=self._cell_center(scenario.goal), switch_active=True, show_path=True,
        )

        def render_frame(agent_cell, agent_center, step_index):
            return self._render(
                scenario, agent_center=agent_center, switch_active=step_index >= switch_step,
            )

        video_frames = self._animate_path(scenario, render_frame)
        return self._assemble_record(
            scenario, puzzle_image=puzzle_image, solution_image=solution_image, video_frames=video_frames,
        )


__all__ = ["SwitchToggleWallsGenerator"]
