"""Mandatory dogleg conveyor chains force a visible route detour.

Every retained sample has a solid wall band whose only crossing cells form a
conveyor chain. The chain first carries the agent sideways along the barrier,
then turns across it. Treating those belts as ordinary floor yields a shorter
walls-only route that is illegal under the true forced-motion replay.
"""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    GRID_PATH_STAY_ENCODING_SENTENCE,
    cell_to_xy,
)
from data.spatial_navigation.core.maze_style import route_run_breakdown
from data.spatial_navigation.core.mechanic_base import (
    DIR_TO_DELTA,
    MechanicGridGeneratorBase,
    MechanicScenario,
)


def _shove(cell: Tuple[int, int], direction: str, size: int, walls: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Apply one conveyor shove in direction; bounce back if blocked."""
    dr, dc = DIR_TO_DELTA[direction]
    nxt = (cell[0] + dr, cell[1] + dc)
    if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
        return cell
    if nxt in walls:
        return cell
    return nxt


def _bfs_with_conveyors(
    size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    conveyors: Dict[Tuple[int, int], str],
) -> Optional[List[Tuple[int, int]]]:
    """Return a shortest expanded path under the true conveyor semantics."""
    if start == goal:
        return [start]
    distance = {start: 0}
    parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], List[Tuple[int, int]]]] = {start: (start, [])}
    queue: List[Tuple[int, Tuple[int, int]]] = [(0, start)]
    while queue:
        cost, cell = heappop(queue)
        if cost != distance.get(cell):
            continue
        if cell == goal:
            trail: List[List[Tuple[int, int]]] = []
            node = goal
            while node != start:
                prev, steps = parent[node]
                trail.append(steps)
                node = prev
            trail.reverse()
            return [start] + [step for steps in trail for step in steps]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in walls:
                continue
            visited_steps: List[Tuple[int, int]] = [nxt]
            shoved = nxt
            applied = set()
            while shoved in conveyors and shoved not in applied:
                applied.add(shoved)
                d = conveyors[shoved]
                after = _shove(shoved, d, size, walls)
                if after == shoved:
                    break
                visited_steps.append(after)
                shoved = after
            rest = shoved
            next_cost = cost + len(visited_steps)
            if next_cost >= distance.get(rest, size * size * size):
                continue
            distance[rest] = next_cost
            parent[rest] = (cell, visited_steps)
            heappush(queue, (next_cost, rest))
    return None


def _bfs_treating_conveyors_as_floor(
    size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walls: Set[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """Return a walls-only shortest path that deliberately ignores belt arrows."""
    queue = deque([start])
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            cursor: Optional[Tuple[int, int]] = goal
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            return list(reversed(path))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if (
                0 <= nxt[0] < size
                and 0 <= nxt[1] < size
                and nxt not in walls
                and nxt not in parent
            ):
                parent[nxt] = cell
                queue.append(nxt)
    return None


def _path_obeys_conveyors(
    path: List[Tuple[int, int]],
    *,
    size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    conveyors: Dict[Tuple[int, int], str],
) -> bool:
    """Replay one expanded path without consulting the authored solution."""
    if not path or path[0] != start:
        return False
    cursor = 1
    cell = start
    while cursor < len(path):
        if cell == goal:
            return False
        nxt = path[cursor]
        if nxt == cell:
            cursor += 1
            continue
        if abs(nxt[0] - cell[0]) + abs(nxt[1] - cell[1]) != 1:
            return False
        if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
            return False
        expected = [nxt]
        rest = nxt
        applied: Set[Tuple[int, int]] = set()
        while rest in conveyors and rest not in applied:
            applied.add(rest)
            rest = _shove(rest, conveyors[rest], size, walls)
            if rest == expected[-1]:
                break
            expected.append(rest)
        if path[cursor: cursor + len(expected)] != expected:
            return False
        cursor += len(expected)
        cell = rest
    return cell == goal


def _reachable_avoiding_conveyors(
    size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    conveyors: Set[Tuple[int, int]],
) -> bool:
    """Return whether S can reach G without entering any conveyor cell."""
    blocked = set(walls) | set(conveyors)
    if start in blocked or goal in blocked:
        return False
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


class ConveyorBeltPathGenerator(MechanicGridGeneratorBase):
    TASK_NAME = "conveyor_belt_path"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Conveyor Belt Path"
    DEFAULT_SIZE = 7

    @staticmethod
    def _conveyor_chain_exit_steps(scenario: MechanicScenario) -> List[int]:
        conveyor_cells = {tuple(entry["cell"]) for entry in scenario.extras["conveyors"]}
        return [
            step
            for step in range(1, len(scenario.agent_path))
            if scenario.agent_path[step - 1] in conveyor_cells
            and scenario.agent_path[step] not in conveyor_cells
        ]

    def _authored_reasoning(self, scenario: MechanicScenario) -> str:
        path = list(scenario.agent_path)
        conveyors = {tuple(entry["cell"]): str(entry["direction"]) for entry in scenario.extras["conveyors"]}
        chain_cells = [cell for cell in path if cell in conveyors]
        exit_steps = self._conveyor_chain_exit_steps(scenario)
        sentences: List[str] = [
            f"S sits at {cell_to_xy(path[0])} and G at {cell_to_xy(path[-1])} on opposite sides of a solid wall "
            "band, and the only way through it is the conveyor chain at "
            f"{', '.join(str(cell_to_xy(cell)) for cell in sorted(conveyors))}. Stepping onto a belt is a "
            "commitment: each arrow shoves the agent one more cell, belt after belt, until it settles on a "
            "non-belt cell.",
        ]
        anchor = 0
        for exit_step in exit_steps:
            entry_step = exit_step - 1
            while entry_step > 0 and path[entry_step - 1] in conveyors:
                entry_step -= 1
            approach = route_run_breakdown(path[anchor : entry_step + 1])
            if approach:
                leg = ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in approach)
                sentences.append(
                    f"The route walks {leg} and enters the belt at {cell_to_xy(path[entry_step])}."
                )
            forced = path[entry_step : exit_step + 1]
            arrow_text = " to ".join(str(cell_to_xy(cell)) for cell in forced)
            sentences.append(
                f"The arrows then carry it {arrow_text}, where it settles on ordinary floor."
            )
            anchor = exit_step
        tail = route_run_breakdown(path[anchor:])
        if tail:
            leg = ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in tail)
            sentences.append(f"From the settled cell it walks {leg} and stops on G.")
        sentences.append(
            f"Every forced belt cell appears in the trajectory in shove order, so the {len(chain_cells)}-cell "
            "chain is fully accounted for and the route is legal."
        )
        return " ".join(sentences)

    def __init__(self, *args, chain_length=None, **kwargs):
        self.chain_length = chain_length
        super().__init__(*args, **kwargs)

    def sample_scenario(self) -> MechanicScenario:
        for _ in range(1024):
            max_chain_length = min(4, self.size - 2)
            chain_length = (
                self.rng.randint(2, max_chain_length)
                if self.chain_length is None
                else self.chain_length
            )
            barrier_axis = self.rng.choice(("row", "col"))
            cross_positive = self.rng.choice((True, False))
            lateral_positive = self.rng.choice((True, False))
            barrier_index = self.rng.randint(1, self.size - 2)
            if lateral_positive:
                first_offset = self.rng.randint(1, self.size - chain_length)
                chain_offsets = [first_offset + index for index in range(chain_length)]
            else:
                first_offset = self.rng.randint(chain_length - 1, self.size - 2)
                chain_offsets = [first_offset - index for index in range(chain_length)]

            if barrier_axis == "row":
                cross_direction = "down" if cross_positive else "up"
                lateral_direction = "right" if lateral_positive else "left"
                chain = [(barrier_index, offset) for offset in chain_offsets]
                conveyors = {cell: lateral_direction for cell in chain[:-1]}
                conveyors[chain[-1]] = cross_direction
                walls = {
                    (barrier_index, col)
                    for col in range(self.size)
                    if (barrier_index, col) not in conveyors
                }
                source_row = barrier_index - 1 if cross_positive else barrier_index + 1
                target_row = barrier_index + 1 if cross_positive else barrier_index - 1
                walls.update((source_row, offset) for offset in chain_offsets[1:])
                source_rows = (
                    range(0, barrier_index)
                    if cross_positive
                    else range(barrier_index + 1, self.size)
                )
                target_rows = (
                    range(barrier_index + 1, self.size)
                    if cross_positive
                    else range(0, barrier_index)
                )
                start_candidates = [
                    (row, col)
                    for row in source_rows
                    for col in range(self.size)
                    if (row, col) not in walls
                ]
                goal_candidates = [
                    (row, col)
                    for row in target_rows
                    for col in range(self.size)
                    if abs(col - chain_offsets[0]) <= 1
                ]
                entry_cell = (source_row, chain_offsets[0])
                direct_exit_cell = (target_row, chain_offsets[0])
                forced_exit_cell = (target_row, chain_offsets[-1])
            else:
                cross_direction = "right" if cross_positive else "left"
                lateral_direction = "down" if lateral_positive else "up"
                chain = [(offset, barrier_index) for offset in chain_offsets]
                conveyors = {cell: lateral_direction for cell in chain[:-1]}
                conveyors[chain[-1]] = cross_direction
                walls = {
                    (row, barrier_index)
                    for row in range(self.size)
                    if (row, barrier_index) not in conveyors
                }
                source_col = barrier_index - 1 if cross_positive else barrier_index + 1
                target_col = barrier_index + 1 if cross_positive else barrier_index - 1
                walls.update((offset, source_col) for offset in chain_offsets[1:])
                source_cols = (
                    range(0, barrier_index)
                    if cross_positive
                    else range(barrier_index + 1, self.size)
                )
                target_cols = (
                    range(barrier_index + 1, self.size)
                    if cross_positive
                    else range(0, barrier_index)
                )
                start_candidates = [
                    (row, col)
                    for col in source_cols
                    for row in range(self.size)
                    if (row, col) not in walls
                ]
                goal_candidates = [
                    (row, col)
                    for col in target_cols
                    for row in range(self.size)
                    if abs(row - chain_offsets[0]) <= 1
                ]
                entry_cell = (chain_offsets[0], source_col)
                direct_exit_cell = (chain_offsets[0], target_col)
                forced_exit_cell = (chain_offsets[-1], target_col)

            start = self.rng.choice(start_candidates)
            goal = self.rng.choice(goal_candidates)
            protected = {
                start,
                goal,
                entry_cell,
                direct_exit_cell,
                forced_exit_cell,
            } | set(conveyors)
            extra_wall_candidates = [
                (row, col)
                for row in range(self.size)
                for col in range(self.size)
                if (row, col) not in walls and (row, col) not in protected
            ]
            self.rng.shuffle(extra_wall_candidates)
            extra_wall_count = self.rng.randint(0, max(1, self.size - 3))
            walls.update(extra_wall_candidates[:extra_wall_count])

            full_path = _bfs_with_conveyors(self.size, start, goal, walls, conveyors)
            if full_path is None:
                continue
            floor_path = _bfs_treating_conveyors_as_floor(self.size, start, goal, walls)
            if floor_path is None:
                continue
            if not set(conveyors).issubset(full_path):
                continue
            if _reachable_avoiding_conveyors(
                self.size, start, goal, walls, set(conveyors)
            ):
                continue
            if len(full_path) <= len(floor_path):
                continue
            if _path_obeys_conveyors(
                floor_path,
                size=self.size,
                start=start,
                goal=goal,
                walls=walls,
                conveyors=conveyors,
            ):
                continue
            if len(full_path) < 6 or len(full_path) > 28:
                continue
            scenario = self._build_scenario(
                start=start,
                goal=goal,
                conveyors=conveyors,
                walls=sorted(walls),
                path=full_path,
            )
            if len(self._conveyor_chain_exit_steps(scenario)) > 5:
                continue
            return scenario
        raise RuntimeError("conveyor_belt_path: failed to sample scenario.")

    def _build_scenario(self, *, start, goal, conveyors, walls, path):
        return MechanicScenario(
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            title=self.TASK_TITLE,
            input_desc=(
                f"The image shows a {self.size}x{self.size} grid with row numbers down the left edge and column "
                "numbers across the top edge. Green S is the start, blue G is the goal, beige cells with orange "
                "arrows are conveyor belts pointing in their arrow direction, and dark slate cells are walls. "
                f"{GRID_XY_CONVENTION_SENTENCE}"
            ),
            rules_desc=(
                "The agent moves up, right, down, or left between adjacent floor cells. "
                f"While resting on a non-conveyor cell, it may stay for one step. {GRID_PATH_STAY_ENCODING_SENTENCE} "
                "That stay does not enter a belt or trigger a conveyor chain. "
                "Stepping onto a conveyor cell instantly shoves the agent one extra cell in the conveyor's arrow "
                "direction. If the next cell along the arrow is another conveyor, that one shoves again, chaining until "
                "the agent rests on a non-conveyor cell. A shove that would hit a wall or the grid border simply stops. "
                "Every forced conveyor cell must appear in the delivered trajectory. Any complete legal route to G is "
                "valid, and route length is not scored."
            ),
            ti2v_output=(
                "Generate a video that shows one complete legal route to G, with each conveyor "
                "shove shown as a direct switch to the next integer cell. Hold every complete semantic state for "
                "the same duration and update the agent, cumulative route, and conveyor effects atomically."
            ),
            solution_image_output=(
                "Render the solved still by drawing one complete legal trajectory, including every forced conveyor "
                "cell, in red and placing the agent on G."
            ),
            size=self.size,
            start=start,
            goal=goal,
            agent_path=path,
            walls=list(walls),
            extras={
                "conveyors": [{"cell": [r, c], "direction": d} for (r, c), d in conveyors.items()],
            },
            structured_answer={
                "agent_path_cell_ids": self._path_cell_ids(path),
                "conveyor_cell_ids": [self._cell_id(c) for c in conveyors],
            },
        )

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

        for entry in scenario.extras["conveyors"]:
            cell = tuple(entry["cell"])
            self._draw_conveyor(draw, cell, direction=entry["direction"])

        if show_path:
            self._draw_path_overlay(draw, scenario.agent_path)

        self._draw_start_goal(draw, scenario)
        self._draw_agent(draw, agent_center)
        return self._finish_scene(image)

    def create_puzzle(self) -> Dict[str, Any]:  # type: ignore[override]
        scenario = self.sample_scenario()

        puzzle_image = self._render(scenario, agent_center=self._cell_center(scenario.start))
        solution_image = self._render(
            scenario, agent_center=self._cell_center(scenario.goal), show_path=True,
        )

        def render_frame(agent_cell, agent_center, step_index):
            return self._render(scenario, agent_center=agent_center)

        video_frames = self._animate_path(scenario, render_frame)
        return self._assemble_record(
            scenario, puzzle_image=puzzle_image, solution_image=solution_image, video_frames=video_frames,
        )


__all__ = ["ConveyorBeltPathGenerator"]
