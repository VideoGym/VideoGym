"""Drop a finite bridge inventory across multiple mandatory water bands.

Default training samples use two or three complete water bands; explicit
extrapolation uses the declared bridge inventory. Start and goal are on opposite sides. Crossing each band consumes a distinct bridge
tile. The shortest route obtained by treating water as ordinary floor is
strictly shorter than every inventory-legal route, so resource-aware planning
changes the selected path rather than merely validating an ordinary BFS path.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    GRID_PATH_STAY_ENCODING_SENTENCE,
    cell_to_xy,
)
from data.spatial_navigation.core.maze_style import route_run_breakdown
from data.spatial_navigation.core.mechanic_base import (
    MechanicGridGeneratorBase,
    MechanicScenario,
)


def _shortest_floor_distance(
    size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walls: Set[Tuple[int, int]],
) -> Optional[int]:
    """Return the wall-only distance when every water cell is ordinary floor."""

    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        cell, distance = queue.popleft()
        if cell == goal:
            return distance
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if (
                0 <= nxt[0] < size
                and 0 <= nxt[1] < size
                and nxt not in walls
                and nxt not in seen
            ):
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def _bfs_with_bridges(
    size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    water: Set[Tuple[int, int]],
    tile_inventory: int,
) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """Return (full_path, dropped_water_cells_in_order)."""
    if start == goal:
        return ([start], [])
    # State = (cell, frozenset(dropped_water_cells))
    queue = deque([(start, frozenset())])
    parent: Dict[Tuple[Tuple[int, int], frozenset], Tuple[Tuple[Tuple[int, int], frozenset], Optional[Tuple[int, int]]]] = {
        (start, frozenset()): ((start, frozenset()), None)
    }
    while queue:
        cell, dropped = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in walls:
                continue
            if nxt in water and nxt not in dropped:
                # need to drop a tile
                if len(dropped) >= tile_inventory:
                    continue
                new_dropped = dropped | {nxt}
            else:
                new_dropped = dropped
            state = (nxt, new_dropped)
            if state in parent:
                continue
            parent[state] = ((cell, dropped), nxt if (nxt in water and nxt not in dropped) else None)
            if nxt == goal:
                full = [nxt]
                drops: List[Tuple[int, int]] = []
                cur = state
                while cur != (start, frozenset()):
                    prev, drop = parent[cur]
                    if drop is not None:
                        drops.append(drop)
                    full.append(prev[0])
                    cur = prev
                full.reverse()
                drops.reverse()
                return (full, drops)
            queue.append(state)
    return None


class BridgeDropTileGenerator(MechanicGridGeneratorBase):
    TASK_NAME = "bridge_drop_tile"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Bridge Drop Tile"
    DEFAULT_SIZE = 7

    def __init__(self, *args, bridge_count=None, **kwargs):
        self.bridge_count = bridge_count
        super().__init__(*args, **kwargs)

    def sample_scenario(self) -> MechanicScenario:
        for _ in range(512):
            inventory = (
                self.rng.choice((2, 3))
                if self.bridge_count is None
                else self.bridge_count
            )
            water_axis = self.rng.choice(("row", "col"))
            internal_indices = list(range(1, self.size - 1))
            if len(internal_indices) < inventory:
                continue
            water_indices = sorted(self.rng.sample(internal_indices, inventory))

            if water_axis == "row":
                water_cells = {
                    (row, col) for row in water_indices for col in range(self.size)
                }
                start_candidates = [
                    (row, col)
                    for row in range(water_indices[0])
                    for col in range(self.size)
                ]
                goal_candidates = [
                    (row, col)
                    for row in range(water_indices[-1] + 1, self.size)
                    for col in range(self.size)
                ]
            else:
                water_cells = {
                    (row, col) for col in water_indices for row in range(self.size)
                }
                start_candidates = [
                    (row, col)
                    for col in range(water_indices[0])
                    for row in range(self.size)
                ]
                goal_candidates = [
                    (row, col)
                    for col in range(water_indices[-1] + 1, self.size)
                    for row in range(self.size)
                ]
            if not start_candidates or not goal_candidates:
                continue

            start = self.rng.choice(start_candidates)
            goal = self.rng.choice(goal_candidates)
            open_cells = [
                (row, col)
                for row in range(self.size)
                for col in range(self.size)
                if (row, col) not in water_cells and (row, col) not in {start, goal}
            ]
            self.rng.shuffle(open_cells)
            wall_count = self.rng.randint(
                max(1, round(3 * (self.size / 7) ** 2)), round(6 * (self.size / 7) ** 2)
            )
            walls = set(open_cells[:wall_count])

            result = _bfs_with_bridges(
                self.size, start, goal, walls, water_cells, inventory
            )
            if result is None:
                continue
            path, drops = result
            if len(drops) != inventory or len(set(drops)) != inventory:
                continue
            relaxed_distance = _shortest_floor_distance(self.size, start, goal, walls)
            if relaxed_distance is None or len(path) - 1 <= relaxed_distance:
                # Every wall-only shortest path must fail true replay. Retain
                # only samples where conserving the finite tile inventory
                # forces a real detour.
                continue
            insufficient = _bfs_with_bridges(
                self.size, start, goal, walls, water_cells, inventory - 1
            )
            if insufficient is not None:
                continue
            if len(path) < 6 or len(path) > 3 * self.size + 1:
                continue
            return self._build_scenario(
                start=start,
                goal=goal,
                walls=sorted(walls),
                water=sorted(water_cells),
                path=path,
                drops=drops,
                inventory=inventory,
            )
        raise RuntimeError("bridge_drop_tile: failed to sample scenario.")

    def _authored_reasoning(self, scenario: MechanicScenario) -> str:
        path = list(scenario.agent_path)
        drops = [tuple(cell) for cell in scenario.extras["drops"]]
        inventory = int(scenario.extras["inventory"])
        water = {tuple(cell) for cell in scenario.extras["water"]}
        rows = sorted({cell[0] for cell in water})
        cols = sorted({cell[1] for cell in water})
        if len(rows) <= len(cols):
            band_phrase = f"{len(rows)} full water rows (x = {', '.join(str(r + 1) for r in rows)})"
        else:
            band_phrase = f"{len(cols)} full water columns (y = {', '.join(str(c + 1) for c in cols)})"
        start_xy = cell_to_xy(path[0])
        goal_xy = cell_to_xy(path[-1])
        sentences: List[str] = [
            f"S sits at {start_xy} and G at {goal_xy}, separated by {band_phrase}, and the gauge holds "
            f"{inventory} bridge tiles. Each band must be crossed once and every first step onto water costs one "
            f"tile, so the {inventory} tiles cover exactly one crossing cell per band and the route cannot afford "
            "to touch any other water cell.",
        ]
        connectors = (
            "It then runs {direction}",
            "From there it continues {direction}",
            "Next it turns {direction}",
            "The route then heads {direction}",
        )
        remaining_drops = list(drops)
        used = 0
        for run_index, (direction, cells) in enumerate(route_run_breakdown(path)):
            steps = len(cells) - 1
            exit_xy = cell_to_xy(cells[-1])
            opener = (
                f"From S the route steps {direction}"
                if run_index == 0
                else connectors[(run_index - 1) % len(connectors)].format(direction=direction)
            )
            span = f"one cell to {exit_xy}" if steps == 1 else f"{steps} cells to {exit_xy}"
            new_drops = [cell for cell in cells[1:] if cell in remaining_drops]
            if new_drops:
                for cell in new_drops:
                    remaining_drops.remove(cell)
                used += len(new_drops)
                placed = " and ".join(str(cell_to_xy(cell)) for cell in new_drops)
                plank_phrase = (
                    f"placing a plank on the water cell at {placed}"
                    if len(new_drops) == 1
                    else f"placing planks on the water cells at {placed}"
                )
                sentences.append(
                    f"{opener} {span}, {plank_phrase} and leaving "
                    f"{inventory - used} of {inventory} tiles on the gauge."
                )
            else:
                sentences.append(f"{opener} {span} without touching new water.")
        sentences.append(
            f"That spends all {inventory} tiles, one per band, and the agent stands on G with the gauge empty, "
            "having stepped only between adjacent cells and onto water only where a plank was placed."
        )
        return " ".join(sentences)

    def _build_scenario(self, *, start, goal, walls, water, path, drops, inventory):
        return MechanicScenario(
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            title=self.TASK_TITLE,
            input_desc=(
                f"The image shows a {self.size}x{self.size} grid with row numbers down the left edge and column "
                "numbers across the top edge. Green S is the start, blue G is the goal, blue cells with white wave "
                "lines are water, and dark slate cells are walls. The brown pip gauge below the grid shows the "
                f"remaining bridge tiles, one filled pip per tile; it starts with all {inventory} pips filled. "
                f"{GRID_XY_CONVENTION_SENTENCE}"
            ),
            rules_desc=(
                f"The agent moves up, right, down, or left between adjacent floor cells. "
                f"It may also stay on its current resting cell for one step. {GRID_PATH_STAY_ENCODING_SENTENCE} "
                "Staying never spends a tile, including when the agent is already standing on a placed "
                "bridge. "
                "On the first step onto an unbridged water cell, the agent must spend one carried tile and place a "
                f"wooden bridge there. The agent starts with {inventory} tiles total; it cannot enter a new water cell "
                "after the inventory reaches zero. Dropped tiles stay placed and may be crossed again. Any complete "
                "legal route to G is valid, and route length is not scored."
            ),
            ti2v_output=(
                "Generate a video that shows one complete legal route to G, placing a brown plank on first entry to "
                "each crossed water cell and emptying one pip of the tile gauge at the same placement commit."
            ),
            solution_image_output=(
                "Render the solved still by drawing one complete legal route in red, showing every used bridge tile as "
                "a brown plank on its water cell, emptying one gauge pip per placed plank, and placing the agent on G."
            ),
            size=self.size,
            start=start,
            goal=goal,
            agent_path=path,
            walls=list(walls),
            extras={
                "water": [[r, c] for r, c in water],
                "drops": [[r, c] for r, c in drops],
                "inventory": inventory,
            },
            structured_answer={
                "agent_path_cell_ids": self._path_cell_ids(path),
                "drop_cell_ids": [self._cell_id(d) for d in drops],
                "inventory": inventory,
            },
        )

    def _render(
        self,
        scenario: MechanicScenario,
        *,
        agent_center: Tuple[float, float],
        dropped_so_far: Sequence[Tuple[int, int]],
        show_path: bool = False,
    ) -> Image.Image:
        image, draw = self._begin_scene()
        wall_set = set(scenario.walls)
        self._draw_grid_and_labels(draw, wall_set=wall_set)

        # water
        for w in scenario.extras["water"]:
            cell = tuple(w)
            self._draw_water(draw, cell)
        # dropped bridges
        for d in dropped_so_far:
            self._draw_bridge_tile(draw, d)

        if show_path:
            self._draw_path_overlay(draw, scenario.agent_path)

        self._draw_start_goal(draw, scenario)
        self._draw_agent(draw, agent_center)
        # Tile inventory gauge in the bottom margin: one brown pip per carried
        # tile, spent left to right as planks are placed.
        inv = int(scenario.extras["inventory"])
        remaining = inv - len(dropped_so_far)
        self._draw_inventory_gauge(
            draw,
            filled=remaining,
            total=inv,
            fill_color=(172, 130, 78),
            outline_color=(115, 80, 40),
        )
        return self._finish_scene(image)

    def create_puzzle(self) -> Dict[str, Any]:  # type: ignore[override]
        scenario = self.sample_scenario()
        drops = [tuple(d) for d in scenario.extras["drops"]]
        path = scenario.agent_path
        # state per step: which drops have happened by step_index
        states: List[List[Tuple[int, int]]] = []
        dropped: List[Tuple[int, int]] = []
        for cell in path:
            if cell in drops and cell not in dropped:
                dropped.append(cell)
            states.append(list(dropped))

        puzzle_image = self._render(
            scenario, agent_center=self._cell_center(scenario.start), dropped_so_far=[],
        )
        solution_image = self._render(
            scenario, agent_center=self._cell_center(scenario.goal), dropped_so_far=drops, show_path=True,
        )

        def render_frame(agent_cell, agent_center, step_index):
            return self._render(scenario, agent_center=agent_center, dropped_so_far=states[min(step_index, len(states) - 1)])

        video_frames = self._animate_path(scenario, render_frame)
        return self._assemble_record(
            scenario, puzzle_image=puzzle_image, solution_image=solution_image, video_frames=video_frames,
        )


__all__ = ["BridgeDropTileGenerator"]
