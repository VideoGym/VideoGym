"""Fuel-aware route planning with positive-only refill arrivals.

Every retained sample requires a refill detour: the shortest route that sees
only walls is strictly shorter than every fuel-legal route.
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
    COLOR_PLATE_ON,
    MechanicGridGeneratorBase,
    MechanicScenario,
)


Coord = Tuple[int, int]
MIN_OIL_CAPACITY = 6
MAX_OIL_CAPACITY = 10
_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _xy(cell: Coord) -> str:
    """Render an internal 0-based cell as the model-facing 1-based ``(x, y)``."""

    return "({}, {})".format(*cell_to_xy(cell))


def _bfs_with_oil(
    size: int,
    start: Coord,
    goal: Coord,
    walls: Set[Coord],
    refills: Set[Coord],
    max_oil: int,
) -> Optional[List[Coord]]:
    if start == goal:
        return [start]
    queue = deque([(start, max_oil)])
    parent: Dict[Tuple[Coord, int], Optional[Tuple[Coord, int]]] = {
        (start, max_oil): None
    }
    while queue:
        cell, oil = queue.popleft()
        for dr, dc in _DELTAS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in walls:
                continue

            new_oil = oil - 1
            if new_oil <= 0:
                continue
            if nxt in refills:
                new_oil = max_oil
            state = (nxt, new_oil)
            if state in parent:
                continue
            parent[state] = (cell, oil)
            if nxt == goal:
                full: List[Tuple[Coord, int]] = [state]
                cursor = parent[state]
                while cursor is not None:
                    full.append(cursor)
                    cursor = parent[cursor]
                return [position for position, _fuel in reversed(full)]
            queue.append(state)
    return None


def _grid_distances(
    size: int,
    start: Coord,
    walls: Set[Coord],
) -> Dict[Coord, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for dr, dc in _DELTAS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if (
                0 <= nxt[0] < size
                and 0 <= nxt[1] < size
                and nxt not in walls
                and nxt not in distances
            ):
                distances[nxt] = distances[cell] + 1
                queue.append(nxt)
    return distances


def _oil_trace(
    path: List[Coord],
    *,
    refills: Set[Coord],
    max_oil: int,
) -> Optional[Tuple[List[int], List[Tuple[int, int]]]]:
    """Replay fuel, returning visible states and pre-refill positive arrivals."""

    if not path or max_oil <= 1:
        return None
    oil = max_oil
    oil_states = [oil]
    refill_arrivals: List[Tuple[int, int]] = []
    previous = path[0]
    for step, cell in enumerate(path[1:], start=1):
        oil -= 1
        if oil <= 0:
            return None
        if cell != previous and cell in refills:
            refill_arrivals.append((step, oil))
            oil = max_oil
        oil_states.append(oil)
        previous = cell
    return oil_states, refill_arrivals


class LanternOilRouteGenerator(MechanicGridGeneratorBase):
    TASK_NAME = "lantern_oil_route"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Lantern Oil Route"
    DEFAULT_SIZE = 7

    def sample_scenario(self) -> MechanicScenario:
        for _ in range(2048):
            cells = [(r, c) for r in range(self.size) for c in range(self.size)]
            self.rng.shuffle(cells)
            start, goal = cells[0], cells[1]
            if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 5:
                continue

            wall_count = self.rng.randint(self.size, self.size + 4)
            walls = set(cells[2 : 2 + wall_count])
            start_distances = _grid_distances(self.size, start, walls)
            if goal not in start_distances:
                continue
            goal_distances = _grid_distances(self.size, goal, walls)
            base_len = start_distances[goal]
            max_oil = self.rng.randint(MIN_OIL_CAPACITY, MAX_OIL_CAPACITY)

            # A direct route would reach F=0 or worse, so every legal route
            # must activate at least one refill.
            if base_len < max_oil:
                continue
            meaningful_refills = [
                cell
                for cell in start_distances.keys() & goal_distances.keys()
                if cell not in {start, goal}
                and 2 <= start_distances[cell] < max_oil
                and 2 <= goal_distances[cell] < max_oil
            ]
            if not meaningful_refills:
                continue
            primary_refill = self.rng.choice(meaningful_refills)
            second_candidates = [
                cell
                for cell in start_distances
                if cell not in {start, goal, primary_refill}
            ]
            if not second_candidates:
                continue
            second_meaningful = [
                cell
                for cell in second_candidates
                if cell in goal_distances
                and 2 <= start_distances[cell] < max_oil
                and 2 <= goal_distances[cell] < max_oil
            ]
            second_refill = self.rng.choice(second_meaningful or second_candidates)
            refills = [primary_refill, second_refill]

            path = _bfs_with_oil(
                self.size,
                start,
                goal,
                walls,
                set(refills),
                max_oil,
            )
            if path is None:
                continue
            if len(path) - 1 <= base_len:
                # If a fuel-legal route were geometrically shortest, an
                # ordinary wall-only BFS could solve the sample accidentally.
                continue
            if _bfs_with_oil(self.size, start, goal, walls, set(), max_oil) is not None:
                continue
            trace = _oil_trace(path, refills=set(refills), max_oil=max_oil)
            if trace is None:
                continue
            _oil_states, refill_arrivals = trace
            if not refill_arrivals or not any(
                arrival <= max_oil - 2 for _step, arrival in refill_arrivals
            ):
                continue
            if not (8 <= len(path) <= 28):
                continue

            return self._build_scenario(
                start=start,
                goal=goal,
                refills=refills,
                walls=sorted(walls),
                path=path,
                max_oil=max_oil,
            )
        raise RuntimeError("lantern_oil_route: failed to sample scenario.")

    def _build_scenario(self, *, start, goal, refills, walls, path, max_oil):
        return MechanicScenario(
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            title=self.TASK_TITLE,
            input_desc=(
                f"The image shows a {self.size}x{self.size} grid with row numbers down the left edge and column "
                "numbers across the top edge. Green S is the start, blue G is the goal, two yellow circular pads with "
                "a blue droplet are oil refill stations, and dark slate cells are walls. The amber pip gauge below "
                f"the grid shows the remaining fuel, one filled pip per unit; it starts with all {max_oil} pips "
                f"filled. {GRID_XY_CONVENTION_SENTENCE}"
            ),
            rules_desc=(
                "Each step either moves the agent up, right, down, or left to an adjacent non-wall cell, or keeps it "
                f"stationary. {GRID_PATH_STAY_ENCODING_SENTENCE} First subtract 1 fuel for every "
                "step, including a stationary one. If the result is 0, the run fails immediately and no destination "
                f"effect occurs, even when that cell is a refill pad or G. Only entering a refill pad from another "
                f"cell with fuel remaining restores the gauge to all {max_oil} pips; staying on a refill pad spends "
                "fuel and does not refill again. Refill pads remain present and may be entered again later."
            ),
            ti2v_output=(
                "Generate a video showing one complete legal route from S to G. Show every one-cell move or stationary "
                "step, empty one gauge pip at each step commit before any refill takes effect, refill the gauge to "
                "full only after a positive-fuel entry from another cell, and finish on G with at least one pip "
                "still filled."
            ),
            solution_image_output=(
                "Render the solved still by drawing one complete legal route in red, keeping both refill pads visible, "
                "showing the route's final positive fuel on the gauge, and placing the agent on G."
            ),
            size=self.size,
            start=start,
            goal=goal,
            agent_path=path,
            walls=list(walls),
            extras={
                "refills": [[r, c] for r, c in refills],
                "max_oil": max_oil,
                "fuel_failure_threshold": 0,
                "refill_requires_positive_arrival": True,
            },
            structured_answer={
                "agent_path_cell_ids": self._path_cell_ids(path),
                "refill_cell_ids": [self._cell_id(r) for r in refills],
                "max_oil": max_oil,
            },
        )

    def _refill_arrivals(self, scenario: MechanicScenario) -> List[Tuple[int, int]]:
        trace = _oil_trace(
            list(scenario.agent_path),
            refills={tuple(item) for item in scenario.extras["refills"]},
            max_oil=int(scenario.extras["max_oil"]),
        )
        if trace is None:
            raise ValueError(f"{self.TASK_NAME}: authored path reaches zero fuel")
        return trace[1]


    def _authored_reasoning_segments(self, scenario: MechanicScenario) -> List[str]:
        path = list(scenario.agent_path)
        max_oil = int(scenario.extras["max_oil"])
        refills = [tuple(item) for item in scenario.extras["refills"]]
        arrivals = self._refill_arrivals(scenario)
        trace = _oil_trace(path, refills=set(refills), max_oil=max_oil)
        assert trace is not None
        oil_states, _ = trace

        def leg_text(begin: int, end: int) -> str:
            runs = route_run_breakdown(path[begin : end + 1])
            return ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in runs)

        segments: List[str] = []
        opening = (
            f"The gauge starts with {max_oil} fuel and loses one pip per step, and S at {_xy(path[0])} is too far "
            f"from G at {_xy(path[-1])} for that to last, so the route has to run through a refill pad; the pads "
            f"sit at {' and '.join(_xy(pad) for pad in refills)}."
        )
        if arrivals:
            step, fuel = arrivals[0]
            opening += (
                f" It first walks {leg_text(0, step)}, reaching the pad at {_xy(path[step])} with {fuel} fuel "
                f"still on the gauge, and the pad refills it to {max_oil}."
            )
        segments.append(opening)
        for index in range(1, len(arrivals)):
            prev_step = arrivals[index - 1][0]
            step, fuel = arrivals[index]
            segments.append(
                f"From there it walks {leg_text(prev_step, step)} to the pad at {_xy(path[step])}, arriving with "
                f"{fuel} fuel and refilling to {max_oil} again."
            )
        last_anchor = arrivals[-1][0] if arrivals else 0
        closing = (
            f"From the refill it walks {leg_text(last_anchor, len(path) - 1)} and stops on G with "
            f"{oil_states[-1]} fuel still filled, so the gauge never reached zero at any step."
        )
        segments.append(closing)
        return segments

    def _render(
        self,
        scenario: MechanicScenario,
        *,
        agent_cell: Tuple[int, int],
        agent_center: Tuple[float, float],
        oil: int,
        show_path: bool = False,
    ) -> Image.Image:
        image, draw = self._begin_scene()
        wall_set = set(scenario.walls)
        self._draw_grid_and_labels(draw, wall_set=wall_set)

        if show_path:
            self._draw_path_overlay(draw, scenario.agent_path)

        for r in scenario.extras["refills"]:
            self._draw_oil_station(draw, tuple(r))

        self._draw_start_goal(draw, scenario)
        self._draw_agent(draw, agent_center)

        # Fuel gauge in the bottom margin: filled amber pips = remaining fuel.
        self._draw_inventory_gauge(
            draw,
            filled=int(oil),
            total=int(scenario.extras["max_oil"]),
            fill_color=COLOR_PLATE_ON,
            outline_color=(168, 116, 18),
        )
        return self._finish_scene(image)

    def create_puzzle(self) -> Dict[str, Any]:  # type: ignore[override]
        scenario = self.sample_scenario()
        max_oil = int(scenario.extras["max_oil"])
        refills = {tuple(r) for r in scenario.extras["refills"]}
        trace = _oil_trace(
            list(scenario.agent_path),
            refills=refills,
            max_oil=max_oil,
        )
        if trace is None:
            raise ValueError(f"{self.TASK_NAME}: sampled path reaches zero fuel")
        oil_states, _refill_arrivals = trace

        puzzle_image = self._render(
            scenario, agent_cell=scenario.start, agent_center=self._cell_center(scenario.start), oil=max_oil,
        )
        solution_image = self._render(
            scenario, agent_cell=scenario.goal, agent_center=self._cell_center(scenario.goal),
            oil=oil_states[-1], show_path=True,
        )

        def render_frame(agent_cell, agent_center, step_index):
            o = oil_states[min(step_index, len(oil_states) - 1)]
            return self._render(scenario, agent_cell=agent_cell, agent_center=agent_center, oil=o)

        video_frames = self._animate_path(scenario, render_frame)
        return self._assemble_record(
            scenario, puzzle_image=puzzle_image, solution_image=solution_image, video_frames=video_frames,
        )


__all__ = ["LanternOilRouteGenerator"]
