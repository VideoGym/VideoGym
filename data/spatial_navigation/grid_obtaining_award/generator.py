from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
    format_xy_position,
)
from data.spatial_navigation.core.base import GridReward, GridScenario, SpatialNavigationGeneratorBase, _REWARD_COLORS, _SHAPES
from data.spatial_navigation.core.maze_style import route_run_breakdown


_COLOR_NAMES = ("blue", "yellow", "pink", "purple", "green")
_DELTAS = (
    ("up", (-1, 0)),
    ("right", (0, 1)),
    ("down", (1, 0)),
    ("left", (0, -1)),
)
Cell = Tuple[int, int]


def _collection_path(
    size: int,
    start: Cell,
    goal: Cell,
    walls: Set[Cell],
    rewards: Sequence[Cell],
) -> Optional[List[Cell]]:
    reward_index = {cell: index for index, cell in enumerate(rewards)}
    full_mask = (1 << len(rewards)) - 1
    initial_mask = 0
    if start in reward_index:
        initial_mask |= 1 << reward_index[start]
    initial = (start, initial_mask)
    queue = deque([initial])
    parent: Dict[Tuple[Cell, int], Optional[Tuple[Cell, int]]] = {initial: None}
    goal_state: Optional[Tuple[Cell, int]] = None
    while queue:
        cell, collected = queue.popleft()
        if cell == goal and collected == full_mask:
            goal_state = (cell, collected)
            break
        for _, (dr, dc) in _DELTAS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
                continue
            next_collected = collected
            reward_idx = reward_index.get(nxt)
            if reward_idx is not None:
                next_collected |= 1 << reward_idx
            if nxt == goal and next_collected != full_mask:
                continue
            next_state = (nxt, next_collected)
            if next_state in parent:
                continue
            parent[next_state] = (cell, collected)
            queue.append(next_state)
    if goal_state is None:
        return None
    states = [goal_state]
    while parent[states[-1]] is not None:
        states.append(parent[states[-1]])
    return [state[0] for state in reversed(states)]


def _moves_for_path(path: Sequence[Cell]) -> List[str]:
    names = {delta: name for name, delta in _DELTAS}
    return [
        names[(current[0] - previous[0], current[1] - previous[1])]
        for previous, current in zip(path, path[1:])
    ]


class GridObtainingAwardGenerator(SpatialNavigationGeneratorBase):
    TASK_NAME = "grid_obtaining_award"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Grid Obtaining Award"

    @staticmethod
    def _pos(cell: Sequence[int]) -> str:
        return format_xy_position((cell[0] + 1, cell[1] + 1))

    def _authored_reasoning_segments(self, scenario: GridScenario) -> List[str]:
        path = list(scenario.path_cells)
        reward_lookup = {reward.cell: reward for reward in scenario.rewards}
        pickup_steps = [
            step
            for step, cell in enumerate(path[1:], start=1)
            if cell in reward_lookup
        ]
        reward_text = "; ".join(
            f"the {reward.color_name} {reward.style} at {self._pos(reward.cell)}"
            for reward in scenario.rewards
        )
        segments: List[str] = []
        opening = (
            f"The agent starts on the green cell at {self._pos(scenario.start)} and must finish on the red cell at "
            f"{self._pos(scenario.goal)}, but only after collecting every reward: {reward_text}. The slate walls "
            "block several straight connections, so the route threads the open corridors from one reward to the "
            "next before it is allowed to touch the red cell."
        )
        anchor = 0
        for index, step in enumerate(pickup_steps):
            reward = reward_lookup[path[step]]
            runs = route_run_breakdown(path[anchor : step + 1])
            leg = ", ".join(f"{direction} {len(cells) - 1}" for direction, cells in runs)
            sentence = (
                f"{'From there' if index else 'First'} it moves {leg}, reaching {self._pos(path[step])} and "
                f"collecting the {reward.color_name} {reward.style}, which disappears from the board."
            )
            if index == 0:
                segments.append(opening + " " + sentence)
            else:
                segments.append(sentence)
            anchor = step
        tail = route_run_breakdown(path[anchor:])
        if tail:
            leg = ", ".join(f"{direction} {len(cells) - 1}" for direction, cells in tail)
            closing = (
                f"With every reward collected, it moves {leg} and enters the red end cell for the first time, "
                "finishing legally."
            )
        else:
            closing = "The final reward sits on the red end cell itself, so the route is complete."
        segments.append(closing)
        return segments

    def sample_scenario(self) -> GridScenario:
        for _ in range(4096):
            reward_count = self.rng.randint(*self.reward_count_range)
            cells = [(row, col) for row in range(self.size) for col in range(self.size)]
            self.rng.shuffle(cells)
            start, goal = cells[0], cells[1]
            reward_cells = cells[2 : 2 + reward_count]
            reserved = {start, goal, *reward_cells}
            wall_count = self.rng.randint(self.size + 1, self.size * 2)
            walls = set(cells[2 + reward_count : 2 + reward_count + wall_count])
            walls -= reserved

            path_cells = _collection_path(
                self.size, start, goal, walls, reward_cells
            )
            if path_cells is None or not (10 <= len(path_cells) - 1 <= 34):
                continue
            direct_path = _collection_path(self.size, start, goal, walls, [])
            if direct_path is None or len(path_cells) < len(direct_path) + 5:
                continue
            no_wall_path = _collection_path(
                self.size, start, goal, set(), reward_cells
            )
            if no_wall_path is None or len(no_wall_path) >= len(path_cells):
                continue
            if any(
                (
                    subset_path := _collection_path(
                        self.size,
                        start,
                        goal,
                        walls,
                        reward_cells[:index] + reward_cells[index + 1 :],
                    )
                )
                is None
                or len(subset_path) >= len(path_cells)
                for index in range(len(reward_cells))
            ):
                continue

            moves = _moves_for_path(path_cells)
            rewards = [
                GridReward(
                    cell=cell,
                    color=_REWARD_COLORS[index % len(_REWARD_COLORS)],
                    color_name=_COLOR_NAMES[index % len(_COLOR_NAMES)],
                    style=_SHAPES[index % len(_SHAPES)],
                )
                for index, cell in enumerate(reward_cells)
            ]
            return GridScenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                input_desc=(
                    "The input image shows a square grid on a white canvas with 1-based row numbers down the left "
                    "edge and column numbers across the top edge. One green cell marks the start, one red cell marks "
                    "the end, an orange circular agent starts on the green cell, several colored reward shapes are "
                    "scattered on the board, and dark slate cells are impassable walls."
                ),
                rules_desc=(
                    "Each step may move the agent up, right, down, or left to an adjacent open cell, or use stay to "
                    "leave it on the current cell. Walls and the grid border block movement. A reward is collected "
                    "when the agent enters its cell, and all rewards must be collected before the agent first enters "
                    f"the red end cell. {GRID_PATH_STAY_ENCODING_SENTENCE}"
                ),
                ti2v_output=(
                    "Generate a short video that shows a complete legal route around the walls, collecting every reward "
                    "item before reaching the red end cell while keeping the board layout and camera framing unchanged."
                ),

                size=self.size,
                start=start,
                goal=goal,
                move_sequence=moves,
                path_cells=path_cells,
                blocked_cells=sorted(walls),
                rewards=rewards,
                start_color=(61, 181, 99),
                goal_color=(228, 52, 52),
                agent_color=(245, 130, 32),
            )
        raise RuntimeError("grid_obtaining_award: failed to sample load-bearing scenario")
