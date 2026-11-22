"""Route planning with visible heading state and a load-bearing turn budget.

Each sample uses a freshly sampled wall layout, endpoints, and heading pair.
The retained heading-aware route needs exactly the stated three-turn budget,
while no route exists with two turns and every geometry-only shortest route is
too short to satisfy the complete heading contract.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
)
from data.spatial_navigation.core.maze_style import route_run_breakdown
from data.spatial_navigation.core.reasoning_route_base import ReasoningRouteGeneratorBase

Coord = Tuple[int, int]
HeadingState = Tuple[Coord, int, int]


class HeadingTurnBudgetRouteGenerator(ReasoningRouteGeneratorBase):
    TASK_NAME = "heading_turn_budget_route"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Heading Turn-Budget Route"

    _HEADING_DELTA = {
        "north": (-1, 0),
        "east": (0, 1),
        "south": (1, 0),
        "west": (0, -1),
    }
    _HEADINGS = ("north", "east", "south", "west")
    _TURN_BUDGET = 3

    def _cell_xy(self, cell: Sequence[int]) -> str:
        """Name one internal 0-based cell the way the Prompt and answer name it."""

        x, y = cell_to_xy(cell)
        return f"({x}, {y})"


    def _draw_heading_badge(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        cell: Tuple[int, int],
        heading: str,
        solid: bool,
    ) -> None:
        """Draw a direction-bearing badge clear of the centered agent/token."""

        left, top, right, bottom = self._cell_box(cell)
        cx = right - self.cell_size * 0.20
        cy = top + self.cell_size * 0.20
        radius = max(8.0, self.cell_size * 0.16)
        badge_fill = (45, 93, 155) if solid else (255, 255, 255)
        badge_outline = (27, 58, 102)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=badge_fill,
            outline=badge_outline,
            width=2,
        )
        dr, dc = self._HEADING_DELTA[heading]
        arrow_color = (255, 255, 255) if solid else badge_outline
        tail = (cx - dc * radius * 0.45, cy - dr * radius * 0.45)
        tip = (cx + dc * radius * 0.55, cy + dr * radius * 0.55)
        draw.line([tail, tip], fill=arrow_color, width=3)
        perpendicular = (-dr, dc)
        wing_back = (cx + dc * radius * 0.15, cy + dr * radius * 0.15)
        wing = radius * 0.34
        draw.polygon(
            [
                tip,
                (wing_back[0] + perpendicular[0] * wing, wing_back[1] + perpendicular[1] * wing),
                (wing_back[0] - perpendicular[0] * wing, wing_back[1] - perpendicular[1] * wing),
            ],
            fill=arrow_color,
        )

    def _render(self, scenario, *, agent_cell, agent_center=None, show_path=False) -> Image.Image:
        image, draw = self._render_scene(
            scenario,
            agent_cell=agent_cell,
            agent_center=agent_center,
            show_path=show_path,
        )
        self._draw_heading_badge(
            draw,
            cell=tuple(scenario.start),
            heading=str(scenario.extras["initial_heading"]),
            solid=True,
        )
        self._draw_heading_badge(
            draw,
            cell=tuple(scenario.goal),
            heading=str(scenario.extras["final_heading"]),
            solid=False,
        )
        return self._finish_scene(image)


    _RUN_TO_HEADING = {"up": "north", "right": "east", "down": "south", "left": "west"}

    def _authored_reasoning(self, scenario) -> str:
        path = [tuple(cell) for cell in scenario.agent_path]
        initial_heading = str(scenario.extras["initial_heading"])
        final_heading = str(scenario.extras["final_heading"])
        budget = int(scenario.extras["turn_budget"])
        runs = route_run_breakdown(path)
        sentences: List[str] = [
            f"The agent starts on S at {self._cell_xy(path[0])} facing {initial_heading}, and it must stand on G at "
            f"{self._cell_xy(path[-1])} facing {final_heading} after at most {budget} turns, so every change of "
            "direction has to be spent carefully.",
        ]
        turns_used = 0
        heading = initial_heading
        for run_index, (direction, cells) in enumerate(runs):
            run_heading = self._RUN_TO_HEADING[direction]
            steps = len(cells) - 1
            exit_xy = self._cell_xy(cells[-1])
            span = f"one cell to {exit_xy}" if steps == 1 else f"{steps} cells to {exit_xy}"
            if run_heading != heading:
                turns_used += 1
                sentences.append(
                    f"{'It turns' if run_index else 'First it turns'} to face {run_heading} — turn {turns_used} of "
                    f"{budget} — and moves {span}."
                )
            else:
                sentences.append(
                    f"{'It continues' if run_index else 'It moves'} straight {run_heading} {span} without spending "
                    "a turn."
                )
            heading = run_heading
        sentences.append(
            f"That reaches G facing {heading}, matching the outlined badge, with {turns_used} of the {budget} "
            "allowed turns used and no wall touched."
        )
        return " ".join(sentences)

    def _geometric_bfs(
        self,
        start: Coord,
        goal: Coord,
        walls: Set[Coord],
    ) -> Optional[List[Coord]]:
        """Return a shortest route that ignores heading and the turn budget."""
        queue = deque([start])
        parents: Dict[Coord, Optional[Coord]] = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                break
            for delta in self._HEADING_DELTA.values():
                neighbor = (cell[0] + delta[0], cell[1] + delta[1])
                if not (0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size):
                    continue
                if neighbor in walls or neighbor in parents:
                    continue
                parents[neighbor] = cell
                queue.append(neighbor)
        if goal not in parents:
            return None
        node: Optional[Coord] = goal
        path: List[Coord] = []
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
        return path

    def _heading_bfs(
        self,
        start: Coord,
        goal: Coord,
        walls: Set[Coord],
        *,
        initial_heading: str,
        final_heading: str,
        turn_budget: int,
    ) -> Optional[Tuple[List[Coord], int]]:
        """Find a shortest route under forward/left/right heading semantics."""
        initial: HeadingState = (
            start,
            self._HEADINGS.index(initial_heading),
            0,
        )
        queue = deque([initial])
        parents: Dict[HeadingState, Optional[HeadingState]] = {initial: None}
        terminal: Optional[HeadingState] = None
        while queue:
            state = queue.popleft()
            cell, heading_index, turns = state
            if cell == goal and self._HEADINGS[heading_index] == final_heading:
                terminal = state
                break
            for turn_delta in (0, -1, 1):
                next_heading_index = (heading_index + turn_delta) % len(self._HEADINGS)
                next_turns = turns + (1 if turn_delta else 0)
                if next_turns > turn_budget:
                    continue
                direction = self._HEADINGS[next_heading_index]
                dr, dc = self._HEADING_DELTA[direction]
                neighbor = (cell[0] + dr, cell[1] + dc)
                if not (0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size):
                    continue
                if neighbor in walls:
                    continue
                next_state: HeadingState = (neighbor, next_heading_index, next_turns)
                if next_state in parents:
                    continue
                parents[next_state] = state
                queue.append(next_state)
        if terminal is None:
            return None
        state: Optional[HeadingState] = terminal
        path: List[Coord] = []
        while state is not None:
            path.append(state[0])
            state = parents[state]
        path.reverse()
        return path, terminal[2]

    def _sample_walls(self, reserved: Sequence[Coord]) -> Set[Coord]:
        candidates = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in reserved
        ]
        self.rng.shuffle(candidates)
        return set(candidates[: self.rng.randint(9, 16)])

    def sample_scenario(self):
        cells = [(row, col) for row in range(self.size) for col in range(self.size)]
        for _ in range(4000):
            start, goal = self.rng.sample(cells, 2)
            if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 5:
                continue
            initial_heading = self.rng.choice(self._HEADINGS)
            final_heading = self.rng.choice(self._HEADINGS)
            walls = self._sample_walls((start, goal))
            geometric_path = self._geometric_bfs(start, goal, walls)
            if geometric_path is None or len(geometric_path) < 6:
                continue
            solved = self._heading_bfs(
                start,
                goal,
                walls,
                initial_heading=initial_heading,
                final_heading=final_heading,
                turn_budget=self._TURN_BUDGET,
            )
            if solved is None:
                continue
            path, turn_count = solved
            if turn_count != self._TURN_BUDGET:
                continue
            if len(path) <= len(geometric_path) or len(path) > 24:
                continue
            if self._heading_bfs(
                start,
                goal,
                walls,
                initial_heading=initial_heading,
                final_heading=final_heading,
                turn_budget=self._TURN_BUDGET - 1,
            ) is not None:
                continue
            return self._build_scenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                input_desc=(
                    f"The image shows an agent on S in a {self.size}x{self.size} grid of white open cells and dark "
                    "wall cells, with row numbers down the left edge and column numbers across the top edge. A solid "
                    f"arrow badge in S points {initial_heading}; an outlined arrow badge in G points {final_heading}. "
                    f"{GRID_XY_CONVENTION_SENTENCE}"
                ),
                rules_desc=(
                    "The solid badge gives the initial heading and the outlined badge gives the required heading at G. "
                    "At each step the agent may move forward, turn left or right and then move, or stay on its current "
                    "cell without changing heading or using a turn. It can never enter a dark wall cell or leave the "
                    "grid, and it must reach G using at most three turns and match the outlined final-heading arrow."
                ),
                path=path,
                walls=sorted(walls),
                extras={
                    "turn_budget": self._TURN_BUDGET,
                    "initial_heading": initial_heading,
                    "final_heading": final_heading,
                },
                structured_answer={
                    "turn_count": turn_count,
                    "final_heading": final_heading,
                    # Reference-only mirror of the formal visible coordinate route.
                    "agent_path_xy": [list(cell_to_xy(cell)) for cell in path],
                },
            )
        raise RuntimeError(
            "heading_turn_budget_route failed to sample a load-bearing randomized layout"
        )


__all__ = ["HeadingTurnBudgetRouteGenerator"]
