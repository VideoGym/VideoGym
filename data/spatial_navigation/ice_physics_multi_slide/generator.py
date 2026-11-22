"""Mixed-surface and all-ice route planning with compulsory multi-cell slides."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    GRID_PATH_STAY_ENCODING_SENTENCE,
    cell_to_xy,
)
from data.spatial_navigation.core.mechanic_base import (
    DIR_TO_DELTA,
    MechanicGridGeneratorBase,
    MechanicScenario,
)


Coord = Tuple[int, int]
MIXED_SURFACE = "mixed_ice_and_floor"
ALL_ICE_SURFACE = "all_traversable_cells_ice"


def _xy(cell: Coord) -> str:
    """Render an internal 0-based cell as the model-facing 1-based ``(x, y)``."""

    return "({}, {})".format(*cell_to_xy(cell))


@dataclass(frozen=True)
class _IceSolution:
    path: List[Coord]
    actions: List[List[Coord]]


def _advance(
    cell: Tuple[int, int],
    direction: str,
    size: int,
    walls: Set[Coord],
    ice: Set[Coord],
) -> Optional[List[Coord]]:
    """Return every cell committed by one deliberate direction choice."""

    dr, dc = DIR_TO_DELTA[direction]
    nxt = (cell[0] + dr, cell[1] + dc)
    if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
        return None

    visited = [nxt]
    cur = nxt
    if nxt not in ice:
        return visited

    while True:
        nxt = (cur[0] + dr, cur[1] + dc)
        if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
            break
        visited.append(nxt)
        cur = nxt
        if nxt not in ice:
            break
    return visited


def _bfs_with_ice(
    size: int,
    start: Coord,
    goal: Coord,
    walls: Set[Coord],
    ice: Set[Coord],
) -> Optional[_IceSolution]:
    if start == goal:
        return _IceSolution(path=[start], actions=[])

    queue = deque([start])
    parent: Dict[Coord, Tuple[Coord, List[Coord]]] = {}
    seen = {start}
    while queue:
        cell = queue.popleft()
        for direction in ("up", "right", "down", "left"):
            expanded = _advance(cell, direction, size, walls, ice)
            if not expanded:
                continue
            rest = expanded[-1]
            if rest in seen:
                continue
            seen.add(rest)
            parent[rest] = (cell, expanded)
            if rest != goal:
                queue.append(rest)
                continue

            actions: List[List[Coord]] = []
            node = rest
            while node != start:
                previous, action = parent[node]
                actions.append(action)
                node = previous
            actions.reverse()
            return _IceSolution(
                path=[start] + [visited for action in actions for visited in action],
                actions=actions,
            )
    return None


def _reachable_without_ice(
    size: int,
    start: Coord,
    goal: Coord,
    walls: Set[Coord],
    ice: Set[Coord],
) -> bool:
    blocked = walls | ice
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if (
                0 <= nxt[0] < size
                and 0 <= nxt[1] < size
                and nxt not in blocked
                and nxt not in seen
            ):
                seen.add(nxt)
                queue.append(nxt)
    return False


def _shortest_floor_distance(
    size: int,
    start: Coord,
    goal: Coord,
    walls: Set[Coord],
) -> Optional[int]:
    """Return the wall-only distance when ice is treated as ordinary floor."""

    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        cell, distance = queue.popleft()
        if cell == goal:
            return distance
        for dr, dc in DIR_TO_DELTA.values():
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


def _minimum_ice_path_steps(
    size: int,
    start: Coord,
    goal: Coord,
    walls: Set[Coord],
    ice: Set[Coord],
) -> Optional[int]:
    """Return the fewest entered cells among paths that obey true ice replay."""

    queue: List[Tuple[int, Coord]] = [(0, start)]
    distances = {start: 0}
    while queue:
        distance, cell = heappop(queue)
        if distance != distances[cell]:
            continue
        if cell == goal:
            return distance
        for direction in ("up", "right", "down", "left"):
            expanded = _advance(cell, direction, size, walls, ice)
            if not expanded:
                continue
            rest = expanded[-1]
            candidate = distance + len(expanded)
            previous = distances.get(rest)
            if previous is not None and candidate >= previous:
                continue
            distances[rest] = candidate
            heappush(queue, (candidate, rest))
    return None


def _is_board_edge(cell: Coord, size: int) -> bool:
    return cell[0] in {0, size - 1} or cell[1] in {0, size - 1}


class IcePhysicsMultiSlideGenerator(MechanicGridGeneratorBase):
    TASK_NAME = "ice_physics_multi_slide"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Ice Physics Multi Slide"
    DEFAULT_SIZE = 7

    def sample_scenario(self) -> MechanicScenario:
        surface_variant = self.rng.choice((MIXED_SURFACE, ALL_ICE_SURFACE))
        for _ in range(2048):
            scenario = (
                self._sample_mixed_surface()
                if surface_variant == MIXED_SURFACE
                else self._sample_all_ice_surface()
            )
            if scenario is not None:
                return scenario
        raise RuntimeError(
            f"ice_physics_multi_slide: failed to sample {surface_variant} scenario."
        )

    def _sample_mixed_surface(self) -> Optional[MechanicScenario]:
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        interior = [cell for cell in cells if not _is_board_edge(cell, self.size)]
        self.rng.shuffle(cells)
        self.rng.shuffle(interior)
        start = cells[0]
        goal = next((cell for cell in interior if cell != start), None)
        if goal is None:
            return None

        available = [cell for cell in cells if cell not in {start, goal}]
        self.rng.shuffle(available)
        ice_count = self.rng.randint(self.size + 2, self.size * 2 + 2)
        ice = set(available[:ice_count])
        wall_candidates = [cell for cell in available[ice_count:] if cell not in ice]
        wall_count = self.rng.randint(self.size - 1, self.size + 3)
        walls = set(wall_candidates[:wall_count])

        solution = _bfs_with_ice(self.size, start, goal, walls, ice)
        if solution is None:
            return None
        slide_actions = [
            action
            for action in solution.actions
            if action and action[0] in ice and len(action) >= 2
        ]
        if len(slide_actions) < 2:
            return None
        if sum(cell in ice for cell in solution.path) < 4:
            return None
        if _reachable_without_ice(self.size, start, goal, walls, ice):
            return None
        floor_distance = _shortest_floor_distance(
            self.size, start, goal, walls
        )
        ice_distance = _minimum_ice_path_steps(
            self.size, start, goal, walls, ice
        )
        if (
            floor_distance is None
            or ice_distance is None
            or ice_distance <= floor_distance
        ):
            # No wall-only shortest path may survive true slide replay.
            return None
        if not (8 <= len(solution.path) <= 28):
            return None
        return self._build_scenario(
            start=start,
            goal=goal,
            ice=sorted(ice),
            walls=sorted(walls),
            path=solution.path,
            surface_variant=MIXED_SURFACE,
        )

    def _sample_all_ice_surface(self) -> Optional[MechanicScenario]:
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        interior = [cell for cell in cells if not _is_board_edge(cell, self.size)]
        self.rng.shuffle(cells)
        self.rng.shuffle(interior)
        start = cells[0]
        goal = next((cell for cell in interior if cell != start), None)
        if goal is None:
            return None

        wall_candidates = [cell for cell in cells if cell not in {start, goal}]
        self.rng.shuffle(wall_candidates)
        wall_count = self.rng.randint(self.size + 1, self.size * 2)
        walls = set(wall_candidates[:wall_count])
        ice = set(cells) - walls

        solution = _bfs_with_ice(self.size, start, goal, walls, ice)
        if solution is None:
            return None
        slide_actions = [action for action in solution.actions if len(action) >= 2]
        if len(slide_actions) < 3 or len(solution.actions) < 3:
            return None
        if not (9 <= len(solution.path) <= 32):
            return None
        if any(cell not in ice for cell in solution.path):
            return None
        return self._build_scenario(
            start=start,
            goal=goal,
            ice=sorted(ice),
            walls=sorted(walls),
            path=solution.path,
            surface_variant=ALL_ICE_SURFACE,
        )

    def _build_scenario(self, *, start, goal, ice, walls, path, surface_variant):
        all_ice = surface_variant == ALL_ICE_SURFACE
        if all_ice:
            surface_input = (
                "This sample uses the ALL ICE surface variant: every traversable cell is pale-cyan ice, including the cells "
                "under S and G; only dark slate wall cells are not ice."
            )
            stopping_rule = (
                "Because every traversable cell is ice, a slide stops only on the cell immediately before a wall or "
                "the grid border."
            )
        else:
            surface_input = (
                "This sample uses the MIXED ICE/FLOOR surface variant: pale-cyan snowflake cells are ice and white cells are "
                "normal floor. S and G stand on normal floor."
            )
            stopping_rule = (
                "A slide stops when it enters the first white normal-floor cell, or on the ice cell immediately before "
                "a wall or the grid border."
            )

        return MechanicScenario(
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            title=self.TASK_TITLE,
            input_desc=(
                f"The image shows a {self.size}x{self.size} grid with row numbers down the left edge and column "
                "numbers across the top edge. The orange disc is the agent, resting on the green S start cell. "
                f"Blue G is the goal, and dark slate cells are walls. {surface_input} {GRID_XY_CONVENTION_SENTENCE}"
            ),
            rules_desc=(
                "From a resting cell, choose up, right, down, or left and enter the adjacent traversable cell. If the "
                "entered cell is ice, the agent must keep sliding in that same direction through every following "
                f"traversable ice cell. {stopping_rule} The direction can change only after the slide has settled. "
                f"At any settled state, the agent may instead stay for one step. {GRID_PATH_STAY_ENCODING_SENTENCE} "
                "This does not begin or retrigger a slide."
            ),
            ti2v_output=(
                "Generate a video showing one complete legal trajectory from S to G. Show every entered cell in order, "
                "including every compulsory cell of each ice slide and each settled state before the next direction "
                "choice."
            ),
            solution_image_output=(
                "Render the solved still by drawing one complete legal trajectory in red, including every ice cell "
                "slid through, and place the agent on G."
            ),
            size=self.size,
            start=start,
            goal=goal,
            agent_path=path,
            walls=list(walls),
            extras={
                "surface_variant": surface_variant,
                "all_traversable_cells_are_ice": all_ice,
                "ice": [[r, c] for r, c in ice],
            },
            structured_answer={
                "agent_path_cell_ids": self._path_cell_ids(path),
                "surface_variant": surface_variant,
                "ice_cell_ids": [self._cell_id(c) for c in ice],
            },
        )

    def _replay_action_segments(
        self,
        scenario: MechanicScenario,
    ) -> List[Tuple[int, int, List[Coord]]]:
        path = list(scenario.agent_path)
        walls = set(scenario.walls)
        ice = {tuple(cell) for cell in scenario.extras["ice"]}
        segments: List[Tuple[int, int, List[Coord]]] = []
        index = 0
        while index < len(path) - 1:
            current = path[index]
            delta = (
                path[index + 1][0] - current[0],
                path[index + 1][1] - current[1],
            )
            direction = next(
                (name for name, candidate in DIR_TO_DELTA.items() if candidate == delta),
                None,
            )
            if direction is None:
                raise ValueError(f"{self.TASK_NAME}: non-adjacent authored path step")
            expanded = _advance(current, direction, scenario.size, walls, ice)
            if not expanded or path[index + 1 : index + 1 + len(expanded)] != expanded:
                raise ValueError(f"{self.TASK_NAME}: authored path violates ice replay")
            end = index + len(expanded)
            segments.append((index, end, expanded))
            index = end
        return segments

    def _authored_reasoning(self, scenario: MechanicScenario) -> str:
        path = list(scenario.agent_path)
        ice = {tuple(cell) for cell in scenario.extras["ice"]}
        all_ice = bool(scenario.extras["all_traversable_cells_are_ice"])
        segments = self._replay_action_segments(scenario)
        if all_ice:
            surface_note = (
                "every traversable cell here is ice, so each chosen direction slides until the cell right before "
                "a wall or the border"
            )
        else:
            surface_note = (
                "the snowflake cells are ice, so a step onto ice keeps sliding straight until the first normal "
                "floor cell or the cell right before a wall or the border"
            )
        sentences: List[str] = [
            f"S is at {cell_to_xy(path[0])} and G at {cell_to_xy(path[-1])}, and {surface_note}. That makes each "
            "direction choice a commitment: what matters is where a slide stops, not the cells it rushes past, so "
            "the route is planned stop point by stop point.",
        ]
        for start_index, end_index, expanded in segments:
            origin = path[start_index]
            settle = path[end_index]
            direction = next(
                name
                for name, delta in DIR_TO_DELTA.items()
                if delta == (expanded[0][0] - origin[0], expanded[0][1] - origin[1])
            )
            if len(expanded) == 1:
                if expanded[0] in ice:
                    sentences.append(
                        f"From {cell_to_xy(origin)} it steps {direction} onto the ice at {cell_to_xy(settle)}, "
                        "which is already against the blocker, so it stops there."
                    )
                else:
                    sentences.append(
                        f"From {cell_to_xy(origin)} it steps {direction} onto normal floor at {cell_to_xy(settle)}."
                    )
            else:
                stop_reason = (
                    "the first normal floor in that line"
                    if settle not in ice
                    else "the last ice cell before the blocker"
                )
                sentences.append(
                    f"From {cell_to_xy(origin)} it pushes {direction}: the ice carries it through "
                    f"{', '.join(str(cell_to_xy(cell)) for cell in expanded[:-1])} and it settles at "
                    f"{cell_to_xy(settle)}, {stop_reason}."
                )
        sentences.append(
            "The final stop lands exactly on G, and every intermediate cell of each slide is listed in order, so "
            "the trajectory obeys the slide rule from start to finish."
        )
        return " ".join(sentences)


    def _render(
        self,
        scenario: MechanicScenario,
        *,
        agent_center: Tuple[float, float],
        show_path: bool = False,
    ) -> Image.Image:
        image, draw = self._begin_scene()
        wall_set = set(scenario.walls)
        self._draw_grid_and_labels(draw, wall_set=wall_set)

        for c in scenario.extras["ice"]:
            self._draw_ice(draw, tuple(c))

        if show_path:
            self._draw_path_overlay(draw, scenario.agent_path)

        self._draw_start_goal(draw, scenario)
        self._draw_agent(draw, agent_center)
        return self._finish_scene(image)

    def create_puzzle(self) -> Dict[str, Any]:  # type: ignore[override]
        scenario = self.sample_scenario()
        puzzle_image = self._render(scenario, agent_center=self._cell_center(scenario.start))
        solution_image = self._render(scenario, agent_center=self._cell_center(scenario.goal), show_path=True)

        def render_frame(agent_cell, agent_center, step_index):
            return self._render(scenario, agent_center=agent_center)

        video_frames = self._animate_path(scenario, render_frame)
        return self._assemble_record(
            scenario, puzzle_image=puzzle_image, solution_image=solution_image, video_frames=video_frames,
        )


__all__ = ["IcePhysicsMultiSlideGenerator"]
