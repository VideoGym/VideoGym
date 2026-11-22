"""Treasure is at one specific cell. Clue cells display arrows pointing toward the treasure direction."""

from __future__ import annotations


from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
    format_xy_path_answer,
    format_xy_position,
)
from data.spatial_navigation.core.multi_agent_base import (
    MultiAgentSpatialNavigationBase,
    MultiAgentScenario,
    bfs_path,
    cells_to_moves,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path


_DIRECTION_DELTAS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


def _ray_cells(cell, direction, size):
    dr, dc = _DIRECTION_DELTAS[direction]
    ray = []
    row, col = cell
    while True:
        row += dr
        col += dc
        if not (0 <= row < size and 0 <= col < size):
            return ray
        ray.append((row, col))


class HiddenTreasureDigGenerator(MultiAgentSpatialNavigationBase):
    TASK_NAME = "hidden_treasure_dig"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Hidden Treasure Dig"
    DEFAULT_SIZE = 8

    INPUT_DESC = (
        "The board has a blue agent and exactly two parchment clue cells at different rows and columns. Each clue "
        "shows one arrow. Dark cells are walls, and the treasure itself is hidden."
    )
    RULES_DESC = (
        "Each clue arrow emits a directed ray from its parchment cell through every grid cell in the shown direction. "
        "The treasure is the unique cell lying on both forward rays; cells behind an arrow are not on its ray. The "
        "two clue positions alone leave two possible row-column crossing cells, so the shown arrow directions decide "
        "which crossing is the treasure. Reach it with any legal sequence of up, right, down, left, or stay actions; "
        f"stay leaves the agent on its current floor cell and route length is not scored. {GRID_PATH_STAY_ENCODING_SENTENCE}"
    )
    TI2V_OUTPUT = "Generate a short video showing the agent walking from start to the treasure cell."
    SOLUTION_IMAGE_OUTPUT = "Render the solved still image with the agent's path drawn in blue and the chest icon revealed."

    @staticmethod
    def _pos(cell) -> str:
        return format_xy_position((cell[0] + 1, cell[1] + 1))

    def _authored_reasoning(self, scenario: MultiAgentScenario) -> str:
        from data.spatial_navigation.core.maze_style import route_run_breakdown

        path = [tuple(c) for c in scenario.primary_path]
        clues = [tuple(c) for c in scenario.extras["clue_cells"]]
        arrows = [direction for _axis, direction in scenario.extras["clue_arrows"]]
        treasure = tuple(scenario.extras["treasure"])
        candidates = [tuple(c) for c in scenario.extras["position_only_candidates"]]
        alternate = next(c for c in candidates if c != treasure)
        sentences = [
            f"The two parchments sit at {self._pos(clues[0])} and {self._pos(clues[1])}, so their row-column "
            f"crossings leave two possible treasure spots: {self._pos(candidates[0])} and "
            f"{self._pos(candidates[1])}. The arrows decide it: the clue at {self._pos(clues[0])} points "
            f"{arrows[0]} and the clue at {self._pos(clues[1])} points {arrows[1]}, and only "
            f"{self._pos(treasure)} lies forward of both arrows — {self._pos(alternate)} sits behind one of them, "
            "so it is ruled out.",
        ]
        runs = route_run_breakdown(path)
        leg = ", then ".join(f"{direction} {len(run) - 1}" for direction, run in runs)
        sentences.append(
            f"From {self._pos(path[0])} the agent walks {leg}, arriving at {self._pos(treasure)} and digging up "
            "the treasure exactly where the two forward rays cross."
        )
        return " ".join(sentences)

    def sample_scenario(self) -> MultiAgentScenario:
        size = self.size
        for _ in range(1200):
            cells_pool = [(r, c) for r in range(size) for c in range(size)]
            self.rng.shuffle(cells_pool)
            clue_a = cells_pool[0]
            clue_b = next(
                (
                    cell
                    for cell in cells_pool[1:]
                    if cell[0] != clue_a[0] and cell[1] != clue_a[1]
                ),
                None,
            )
            if clue_b is None:
                continue
            intersections = [
                (clue_a[0], clue_b[1]),
                (clue_b[0], clue_a[1]),
            ]
            treasure_index = self.rng.randrange(2)
            treasure = intersections[treasure_index]
            alternate = intersections[1 - treasure_index]
            reserved = {clue_a, clue_b, treasure, alternate}
            agent_start = next((cell for cell in cells_pool if cell not in reserved), None)
            if agent_start is None:
                continue
            if abs(agent_start[0] - treasure[0]) + abs(agent_start[1] - treasure[1]) < 4:
                continue

            wall_count = self.rng.randint(2, 4)
            walls = []
            for cell in cells_pool:
                if cell in reserved or cell == agent_start:
                    continue
                walls.append(cell)
                if len(walls) >= wall_count:
                    break

            clue_cells = [clue_a, clue_b]
            clue_arrows = []
            for clue in clue_cells:
                if clue[0] == treasure[0]:
                    direction = "right" if treasure[1] > clue[1] else "left"
                    axis = "horizontal"
                else:
                    direction = "down" if treasure[0] > clue[0] else "up"
                    axis = "vertical"
                clue_arrows.append((axis, direction))
            rays = [
                set(_ray_cells(clue, direction, size))
                for clue, (_, direction) in zip(clue_cells, clue_arrows)
            ]
            if rays[0] & rays[1] != {treasure}:
                continue
            path = bfs_path(agent_start, treasure, size, walls)
            alternate_path = bfs_path(agent_start, alternate, size, walls)
            if path is None or alternate_path is None:
                continue
            if not (4 <= len(path) - 1 <= 14):
                continue
            scenario = MultiAgentScenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                size=size,
                walls=walls,
                primary_path=path,
                primary_moves=cells_to_moves(path),
                extras={
                    "agent_start": agent_start,
                    "treasure": treasure,
                    "clue_cells": clue_cells,
                    "clue_arrows": clue_arrows,
                    "position_only_candidates": intersections,
                },
            )
            scenario.answer_text = format_xy_path_answer(scenario.primary_path)
            return scenario
        raise RuntimeError("hidden_treasure_dig sample failed")

    def render_static_layer(self, draw, image, scenario):
        # Two clue cells, one with a single horizontal arrow, one with a single vertical arrow.
        for cell, (kind, direction) in zip(scenario.extras["clue_cells"], scenario.extras["clue_arrows"]):
            cx, cy = self.cell_center(cell)
            l, t, r, b = self.cell_box(cell)
            draw.rectangle([l + 4, t + 4, r - 4, b - 4], fill=(252, 240, 210), outline=(180, 150, 100), width=2)
            self._draw_arrow(draw, (cx, cy), direction)
        # The treasure is HIDDEN — do not draw on the puzzle image.

    def _draw_arrow(self, draw, center, direction):
        cx, cy = center
        s = self.cell_size * 0.18
        if direction == "up":
            draw.polygon([(cx, cy - s), (cx - s * 0.6, cy + s * 0.2), (cx + s * 0.6, cy + s * 0.2)],
                         fill=(40, 40, 40))
        elif direction == "down":
            draw.polygon([(cx, cy + s), (cx - s * 0.6, cy - s * 0.2), (cx + s * 0.6, cy - s * 0.2)],
                         fill=(40, 40, 40))
        elif direction == "left":
            draw.polygon([(cx - s, cy), (cx + s * 0.2, cy - s * 0.6), (cx + s * 0.2, cy + s * 0.6)],
                         fill=(40, 40, 40))
        elif direction == "right":
            draw.polygon([(cx + s, cy), (cx - s * 0.2, cy - s * 0.6), (cx - s * 0.2, cy + s * 0.6)],
                         fill=(40, 40, 40))

    def render_solution_overlay(self, draw, image, scenario):
        if len(scenario.primary_path) >= 2:
            pts = [self.cell_center(c) for c in scenario.primary_path]
            draw_spatial_navigation_path(draw, pts, color=(56, 130, 220), width=max(5, self.cell_size // 7))
        # Reveal chest
        treasure = scenario.extras["treasure"]
        l, t, r, b = self.cell_box(treasure)
        draw.rectangle([l + 6, t + self.cell_size * 0.35, r - 6, b - 6],
                       fill=(165, 110, 50), outline=(80, 50, 20), width=2)
        # lid
        draw.arc([l + 6, t + self.cell_size * 0.18, r - 6, t + self.cell_size * 0.55], start=180, end=0,
                 fill=(80, 50, 20), width=4)
        # lock
        cx, cy = self.cell_center(treasure)
        draw.rectangle([cx - 4, cy - 2, cx + 4, cy + 6], fill=(220, 180, 60), outline=(140, 110, 30), width=1)

    def render_path_overlay(self, draw, path_cells, color=(214, 42, 42)) -> None:
        return

    def render_agents(self, draw, image, scenario, frame_state):
        cx, cy = self.cell_center(frame_state.primary_pos)
        self.draw_circle_agent(draw, (cx, cy), fill=(56, 130, 220), outline=(20, 60, 120))
