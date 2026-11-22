"""Pull levers in correct numeric order to unlock the goal door.

Levers L1, L2, L3 are scattered. The goal door D unlocks only after the agent
visits L1, then L2, then L3 in that order. Visiting them out of order breaks
the sequence — the agent cannot solve by random pickup. The mechanic is
load-bearing: a route that touches a later lever too early is immediately
illegal, and the locked door disconnects the goal until the sequence completes.
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


def _floor_components(size: int, walls: Set[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    remaining = {(r, c) for r in range(size) for c in range(size)} - walls
    components = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue = deque([first])
        component = [first]
        while queue:
            row, col = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (row + dr, col + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
                    component.append(nxt)
        components.append(component)
    return components


def _sequence_path(
    size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    levers: Sequence[Tuple[int, int]],
    door: Tuple[int, int],
    walls: Set[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    locked_walls = walls | {door}
    path = [start]
    for index, lever in enumerate(levers):
        # Later levers are illegal until their predecessor has been pulled.
        segment = _bfs(size, path[-1], lever, locked_walls | set(levers[index + 1:]))
        if segment is None:
            return None
        path.extend(segment[1:])
    tail = _bfs(size, path[-1], goal, walls)
    if tail is None:
        return None
    path.extend(tail[1:])
    if door not in path or _bfs(size, start, goal, locked_walls) is not None:
        return None
    steps = [path.index(lever) for lever in levers]
    if not (steps[0] < steps[1] < steps[2]) or not 8 <= len(path) <= 28:
        return None
    return path


class LeverUnlockSequenceGenerator(MechanicGridGeneratorBase):
    TASK_NAME = "lever_unlock_sequence"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Lever Unlock Sequence"
    DEFAULT_SIZE = 7

    def _authored_reasoning(self, scenario: MechanicScenario) -> str:
        path = list(scenario.agent_path)
        levers = [tuple(lv) for lv in scenario.extras["lever_cells"]]
        door = tuple(scenario.extras["door_cell"])
        lever_steps = [path.index(lv) for lv in levers]
        sentences: List[str] = [
            f"The door D at {cell_to_xy(door)} is the only way to reach G at {cell_to_xy(path[-1])} from S at "
            f"{cell_to_xy(path[0])}, and it stays locked until L1 at {cell_to_xy(levers[0])}, L2 at "
            f"{cell_to_xy(levers[1])}, and L3 at {cell_to_xy(levers[2])} are pulled in that exact order, so the "
            "route must visit the levers by number while steering around any later lever it has not earned yet.",
        ]
        anchors = [0] + lever_steps
        for index, lever_step in enumerate(lever_steps, start=1):
            runs = route_run_breakdown(path[anchors[index - 1] : lever_step + 1])
            leg = ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in runs)
            arrival = (
                f"pulling L{index}"
                if index < 3
                else "pulling L3, which snaps the door open"
            )
            sentences.append(
                f"It walks {leg} to L{index} at {cell_to_xy(path[lever_step])}, {arrival}."
            )
        tail = route_run_breakdown(path[lever_steps[-1] :])
        leg = ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in tail)
        door_note = f", passing through the open door at {cell_to_xy(door)}," if door in path[lever_steps[-1]:] else ""
        sentences.append(f"With all three levers pulled it walks {leg}{door_note} and stops on G.")
        sentences.append(
            "No lever was reached before its predecessor, so the ordering rule holds along the whole route."
        )
        return " ".join(sentences)

    def sample_scenario(self) -> MechanicScenario:
        if self.size > self.DEFAULT_SIZE:
            return self._sample_larger_scenario()
        for _ in range(4096):
            cells = [(r, c) for r in range(self.size) for c in range(self.size)]
            self.rng.shuffle(cells)
            start, goal = cells[0], cells[1]
            l1, l2, l3 = cells[2], cells[3], cells[4]
            door = cells[5]
            walls: Set[Tuple[int, int]] = set()
            extras = cells[6:]
            # Lower wall density: stricter ordering constraint already shrinks the
            # solvable subspace, so heavy walls can leave no valid layout.
            wall_budget = self.rng.randint(max(2, self.size // 2), self.size + 1)
            for cand in extras:
                if cand in {start, goal, l1, l2, l3, door}:
                    continue
                if len(walls) >= wall_budget:
                    break
                walls.add(cand)

            full_path = _sequence_path(self.size, start, goal, [l1, l2, l3], door, walls)
            if full_path is None:
                continue

            return self._build_scenario(
                start=start, goal=goal, levers=[l1, l2, l3], door=door,
                walls=sorted(walls), path=full_path,
            )
        raise RuntimeError("lever_unlock_sequence: failed to sample scenario.")

    def _sample_larger_scenario(self) -> MechanicScenario:
        # At larger sizes the sparse random walls rarely make a separately
        # sampled door the unique connection to a separately sampled goal.
        # Keep the same wall-budget range, but condition marker placement on
        # actual components separated by the locked door. These layouts also
        # belong to the original rejection sampler's support.
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        for _ in range(256):
            wall_budget = self.rng.randint(max(2, self.size // 2), self.size + 1)
            walls = set(self.rng.sample(cells, wall_budget))
            doors = sorted(set(cells) - walls)
            self.rng.shuffle(doors)
            for door in doors:
                neighbors = {
                    (door[0] + dr, door[1] + dc)
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                }
                components = [
                    component
                    for component in _floor_components(self.size, walls | {door})
                    if neighbors.intersection(component)
                ]
                if len(components) < 2:
                    continue
                sources = [component for component in components if len(component) >= 4]
                self.rng.shuffle(sources)
                for source in sources:
                    goals = [
                        cell for component in components if component is not source
                        for cell in component
                    ]
                    for _ in range(8):
                        start, l1, l2, l3 = self.rng.sample(source, 4)
                        goal = self.rng.choice(goals)
                        path = _sequence_path(self.size, start, goal, [l1, l2, l3], door, walls)
                        if path is not None:
                            return self._build_scenario(
                                start=start, goal=goal, levers=[l1, l2, l3], door=door,
                                walls=sorted(walls), path=path,
                            )
        raise RuntimeError("lever_unlock_sequence: failed to sample scenario.")

    def _build_scenario(
        self,
        *,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        levers: List[Tuple[int, int]],
        door: Tuple[int, int],
        walls: Sequence[Tuple[int, int]],
        path: List[Tuple[int, int]],
    ) -> MechanicScenario:
        return MechanicScenario(
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            title=self.TASK_TITLE,
            input_desc=(
                f"The image shows a {self.size}x{self.size} grid with row numbers down the left edge and column "
                "numbers across the top edge. The orange agent stands on the green start cell S, the blue cell G is "
                "the goal, three lever stations on pale tiles are labeled L1, L2, and L3 with every arm leaning "
                "right in the un-pulled state, the brown panel labeled D is a locked door, and the dark slate cells "
                f"are walls. {GRID_XY_CONVENTION_SENTENCE}"
            ),
            rules_desc=(
                "The agent moves up, right, down, or left between adjacent floor cells. "
                f"It may stay on its current resting cell. {GRID_PATH_STAY_ENCODING_SENTENCE} "
                "The door D is impassable until the agent has visited the levers in the order L1 then L2 then L3. "
                "Entering L2 before L1, or entering L3 before L2, immediately makes the route illegal. Entering the "
                "next required lever pulls it once; staying on that already reached lever is only a wait and does not "
                "count as another lever visit. Pulling L3 unlocks D for the rest of the run. "
                "Walls and the grid border block movement."
            ),
            ti2v_output=(
                "Generate a video that shows the agent following a complete legal route, flipping each reached "
                "lever's arm from right to left at the moment it is pulled, in the order L1 then L2 then L3, "
                "turning the door pale green as it opens after L3, then walking through D to the goal."
            ),
            solution_image_output=(
                "Render the solved still by drawing the complete legal route in red, every lever arm leaning left "
                "in the pulled state, the door open in pale green, and the agent on the goal."
            ),
            size=self.size,
            start=start,
            goal=goal,
            agent_path=path,
            walls=list(walls),
            extras={
                "lever_cells": [[r, c] for r, c in levers],
                "door_cell": [door[0], door[1]],
                "lever_steps": [path.index(lv) for lv in levers],
            },
            structured_answer={
                "agent_path_cell_ids": self._path_cell_ids(path),
                "lever_cell_ids": [self._cell_id(lv) for lv in levers],
                "door_cell_id": self._cell_id(door),
            },
        )

    def _render(
        self,
        scenario: MechanicScenario,
        *,
        agent_center: Tuple[float, float],
        levers_pulled: Sequence[bool],
        door_open: bool,
        show_path: bool = False,
    ) -> Image.Image:
        image, draw = self._begin_scene()
        wall_set = set(scenario.walls)
        self._draw_grid_and_labels(draw, wall_set=wall_set)
        token_font = self._load_font(max(11, self.cell_size // 6))

        levers = [tuple(lv) for lv in scenario.extras["lever_cells"]]
        door_cell = tuple(scenario.extras["door_cell"])

        if show_path:
            self._draw_path_overlay(draw, scenario.agent_path)

        for idx, lv in enumerate(levers):
            self._draw_lever(draw, lv, on=levers_pulled[idx], label=f"L{idx + 1}", token_font=token_font)
        self._draw_door(draw, door_cell, open_state=door_open, label="D", token_font=token_font)
        self._draw_start_goal(draw, scenario)
        self._draw_agent(draw, agent_center)
        return self._finish_scene(image)

    def create_puzzle(self) -> Dict[str, Any]:  # type: ignore[override]
        scenario = self.sample_scenario()
        lever_steps: List[int] = list(scenario.extras["lever_steps"])

        puzzle_image = self._render(
            scenario,
            agent_center=self._cell_center(scenario.start),
            levers_pulled=[False, False, False],
            door_open=False,
        )
        solution_image = self._render(
            scenario,
            agent_center=self._cell_center(scenario.goal),
            levers_pulled=[True, True, True],
            door_open=True,
            show_path=True,
        )

        def render_frame(agent_cell, agent_center, step_index):
            pulled = [step_index >= s for s in lever_steps]
            door_open = all(pulled)
            return self._render(scenario, agent_center=agent_center, levers_pulled=pulled, door_open=door_open)

        video_frames = self._animate_path(scenario, render_frame)
        return self._assemble_record(
            scenario,
            puzzle_image=puzzle_image,
            solution_image=solution_image,
            video_frames=video_frames,
        )


__all__ = ["LeverUnlockSequenceGenerator"]
