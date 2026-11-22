"""Directional arrow tiles — can be entered ONLY from the arrow's tail side.

Some cells contain a one-way arrow. The agent may step onto an arrow cell only
when entering from the cell on the arrow's tail (i.e. moving in the arrow's
direction). Stepping on it from any other side is forbidden. Once on the arrow,
the next move must continue in the arrow's direction (one forced step).

The mechanic is load-bearing: every start-to-goal route must cross every declared
full barrier line through their correctly oriented arrow gates.
"""

from __future__ import annotations

from collections import deque
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw

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


_CARDINAL_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _advance(
    size: int,
    cell: Tuple[int, int],
    delta: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    arrows: Dict[Tuple[int, int], str],
) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]], int]]:
    dr, dc = delta
    nxt = (cell[0] + dr, cell[1] + dc)
    if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
        return None
    if nxt in walls:
        return None
    if nxt not in arrows:
        return nxt, [nxt], 0

    arrow_dir = arrows[nxt]
    adr, adc = DIR_TO_DELTA[arrow_dir]
    if (dr, dc) != (adr, adc):
        return None
    forced_next = (nxt[0] + adr, nxt[1] + adc)
    if not (0 <= forced_next[0] < size and 0 <= forced_next[1] < size):
        return None
    if forced_next in walls or forced_next in arrows:
        return None
    return forced_next, [nxt, forced_next], 1


def _bfs_with_arrows(
    size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    arrows: Dict[Tuple[int, int], str],
) -> Optional[List[Tuple[int, int]]]:
    if start == goal:
        return [start]
    queue = deque([start])
    parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], List[Tuple[int, int]]]] = {start: (start, [])}
    while queue:
        cell = queue.popleft()
        for delta in _CARDINAL_DELTAS:
            transition = _advance(size, cell, delta, walls, arrows)
            if transition is None:
                continue
            rest, visited_steps, _ = transition
            if rest in parent:
                continue
            parent[rest] = (cell, visited_steps)
            if rest == goal:
                full: List[Tuple[int, int]] = [start]
                trail = []
                node = rest
                while node != start:
                    prev, steps = parent[node]
                    trail.append(steps); node = prev
                trail.reverse()
                for s in trail:
                    for c_s in s:
                        full.append(c_s)
                return full
            queue.append(rest)
    return None


def _minimum_arrow_commits(
    size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    arrows: Dict[Tuple[int, int], str],
) -> Optional[int]:
    best = {start: 0}
    queue: List[Tuple[int, Tuple[int, int]]] = [(0, start)]
    while queue:
        commits, cell = heapq.heappop(queue)
        if commits != best.get(cell):
            continue
        if cell == goal:
            return commits
        for delta in _CARDINAL_DELTAS:
            transition = _advance(size, cell, delta, walls, arrows)
            if transition is None:
                continue
            rest, _, added_commits = transition
            next_commits = commits + added_commits
            if next_commits >= best.get(rest, 10**9):
                continue
            best[rest] = next_commits
            heapq.heappush(queue, (next_commits, rest))
    return None


class DirectionalArrowEntryGenerator(MechanicGridGeneratorBase):
    TASK_NAME = "directional_arrow_entry"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Directional Arrow Entry"
    DEFAULT_SIZE = 7

    def _authored_reasoning(self, scenario: MechanicScenario) -> str:
        path = list(scenario.agent_path)
        arrows = {tuple(item["cell"]): str(item["direction"]) for item in scenario.extras["arrows"]}
        gate_count = len(arrows)
        gate_text = ", ".join(
            f"{cell_to_xy(cell)} pointing {direction}" for cell, direction in sorted(arrows.items())
        )
        sentences: List[str] = [
            f"S at {cell_to_xy(path[0])} and G at {cell_to_xy(path[-1])} are separated by "
            f"{'a full wall barrier' if gate_count == 1 else f'{gate_count} full wall barriers'}, each pierced only "
            f"by an arrow gate: {gate_text}. A gate opens only to an agent stepping in its arrow direction, and "
            "entering it forces one more step the same way, so the route must line up on each gate's tail side "
            "before crossing.",
        ]
        anchor = 0
        for step in range(1, len(path)):
            if path[step - 1] in arrows:
                gate = path[step - 1]
                direction = arrows[gate]
                approach = route_run_breakdown(path[anchor : step - 1 + 1])
                if approach and len(path[anchor : step]) > 1:
                    entry_leg = ", then ".join(
                        f"{run_direction} {len(cells) - 1}" for run_direction, cells in approach
                    )
                    sentences.append(
                        f"It walks {entry_leg}, entering the gate at {cell_to_xy(gate)} from its tail, and the "
                        f"forced {direction} step drops it at {cell_to_xy(path[step])}."
                    )
                else:
                    sentences.append(
                        f"It enters the gate at {cell_to_xy(gate)} from its tail, and the forced {direction} "
                        f"step drops it at {cell_to_xy(path[step])}."
                    )
                anchor = step
        tail = route_run_breakdown(path[anchor:])
        if tail:
            leg = ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in tail)
            sentences.append(f"From there it walks {leg} and stops on G.")
        sentences.append(
            "Each gate was entered along its arrow with its forced exit taken immediately, so every barrier "
            "crossing is legal."
        )
        return " ".join(sentences)

    def __init__(self, *args, gate_count=None, **kwargs):
        self.gate_count = gate_count
        super().__init__(*args, **kwargs)

    def sample_scenario(self) -> MechanicScenario:
        required_arrow_commits = (
            (2 if self.rng.random() < 0.5 else 1)
            if self.gate_count is None
            else self.gate_count
        )
        for _ in range(512):
            vertical = bool(self.rng.getrandbits(1))
            forward = self.rng.choice((-1, 1))
            barrier_axes = sorted(
                self.rng.sample(range(1, self.size - 1), required_arrow_commits)
            )
            if any(
                right - left < 2
                for left, right in zip(barrier_axes, barrier_axes[1:])
            ):
                continue

            if vertical:
                low_cells = [
                    (row, col)
                    for row in range(self.size)
                    for col in range(0, barrier_axes[0])
                ]
                high_cells = [
                    (row, col)
                    for row in range(self.size)
                    for col in range(barrier_axes[-1] + 1, self.size)
                ]
                start_pool, goal_pool = (
                    (low_cells, high_cells)
                    if forward > 0
                    else (high_cells, low_cells)
                )
                direction = "right" if forward > 0 else "left"
                delta = (0, forward)
                arrow_cells = [
                    (self.rng.randrange(self.size), axis)
                    for axis in barrier_axes
                ]
                barrier_cells = {
                    (row, axis)
                    for axis in barrier_axes
                    for row in range(self.size)
                }
            else:
                low_cells = [
                    (row, col)
                    for row in range(0, barrier_axes[0])
                    for col in range(self.size)
                ]
                high_cells = [
                    (row, col)
                    for row in range(barrier_axes[-1] + 1, self.size)
                    for col in range(self.size)
                ]
                start_pool, goal_pool = (
                    (low_cells, high_cells)
                    if forward > 0
                    else (high_cells, low_cells)
                )
                direction = "down" if forward > 0 else "up"
                delta = (forward, 0)
                arrow_cells = [
                    (axis, self.rng.randrange(self.size))
                    for axis in barrier_axes
                ]
                barrier_cells = {
                    (axis, col)
                    for axis in barrier_axes
                    for col in range(self.size)
                }

            start = self.rng.choice(start_pool)
            goal = self.rng.choice(goal_pool)
            arrows = {cell: direction for cell in arrow_cells}
            walls = barrier_cells - set(arrow_cells)
            gate_neighbors = {
                (cell[0] - delta[0], cell[1] - delta[1])
                for cell in arrow_cells
            } | {
                (cell[0] + delta[0], cell[1] + delta[1])
                for cell in arrow_cells
            }
            reserved = {start, goal, *arrow_cells, *gate_neighbors}
            extra_candidates = [
                (row, col)
                for row in range(self.size)
                for col in range(self.size)
                if (row, col) not in walls and (row, col) not in reserved
            ]
            self.rng.shuffle(extra_candidates)
            walls.update(
                extra_candidates[: self.rng.randint(self.size // 2, self.size)]
            )

            full_path = _bfs_with_arrows(self.size, start, goal, walls, arrows)
            if full_path is None:
                continue
            minimum_arrow_commits = _minimum_arrow_commits(
                self.size, start, goal, walls, arrows
            )
            if minimum_arrow_commits is None or minimum_arrow_commits < required_arrow_commits:
                continue
            # Every S-to-G route must use an arrow gate: treating every arrow
            # cell as blocked must disconnect the board.
            blocked_arrow_path = _bfs_with_arrows(
                self.size, start, goal, walls | set(arrows), {}
            )
            if blocked_arrow_path is not None:
                continue
            if len(full_path) < 6 or len(full_path) > 22:
                continue
            return self._build_scenario(
                start=start,
                goal=goal,
                arrows=arrows,
                walls=sorted(walls),
                path=full_path,
            )
        raise RuntimeError("directional_arrow_entry: failed to sample scenario.")

    def _build_scenario(
        self,
        *,
        start,
        goal,
        arrows,
        walls,
        path,
    ):
        return MechanicScenario(
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            title=self.TASK_TITLE,
            input_desc=(
                f"The image shows a {self.size}x{self.size} grid with row numbers down the left edge and column "
                f"numbers across the top edge. Green S is the start, blue G is the goal, {len(arrows)} pale-orange cells "
                "with chevron arrows are directional gates through full dark-slate wall barriers, and all other dark "
                f"slate cells are walls. {GRID_XY_CONVENTION_SENTENCE}"
            ),
            rules_desc=(
                "The agent moves up, right, down, or left between adjacent floor cells, and on an ordinary resting "
                f"cell it may also stay for one step. {GRID_PATH_STAY_ENCODING_SENTENCE} "
                "An arrow gate can be ENTERED only by stepping in the arrow's direction (i.e. from the tail side). "
                "Entering an arrow forces an immediate extra step in the same direction; the agent rests one cell "
                "beyond the arrow, so the arrow cell is transient and cannot be held as a stay."
            ),
            ti2v_output=(
                "Generate a video that shows the agent following a complete legal trajectory, with each entered arrow "
                "gate traversed in its forced direction."
            ),
            solution_image_output=(
                "Render the solved still by drawing the complete legal trajectory in red and placing the agent "
                "on the goal."
            ),
            size=self.size,
            start=start,
            goal=goal,
            agent_path=path,
            walls=list(walls),
            extras={
                "arrows": [{"cell": [r, c], "direction": d} for (r, c), d in arrows.items()],
            },
            structured_answer={
                # The formal answer is the 1-based (x, y) route; the linear cell
                # ids stay only for the shared spatial_navigation replay/scan fields that
                # still consume them.
                "agent_path_xy": [list(cell_to_xy(cell)) for cell in path],
                "agent_path_cell_ids": self._path_cell_ids(path),
                "arrow_xy": [list(cell_to_xy(c)) for c in arrows],
            },
        )

    def _draw_arrow_gate(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
        *,
        direction: str,
    ) -> None:
        l, t, r, b = self._cell_box(cell)
        # tile background
        draw.rectangle([l + 1, t + 1, r - 1, b - 1], fill=(255, 230, 200), outline=(220, 160, 70), width=1)
        cx, cy = self._cell_center(cell)
        # large chevron pointing direction
        ch_size = self.cell_size * 0.32
        if direction == "right":
            pts = [(cx - ch_size * 0.4, cy - ch_size * 0.4), (cx + ch_size * 0.4, cy), (cx - ch_size * 0.4, cy + ch_size * 0.4)]
            tail = [(cx - ch_size * 0.6, cy - ch_size * 0.4), (cx, cy), (cx - ch_size * 0.6, cy + ch_size * 0.4)]
        elif direction == "left":
            pts = [(cx + ch_size * 0.4, cy - ch_size * 0.4), (cx - ch_size * 0.4, cy), (cx + ch_size * 0.4, cy + ch_size * 0.4)]
            tail = [(cx + ch_size * 0.6, cy - ch_size * 0.4), (cx, cy), (cx + ch_size * 0.6, cy + ch_size * 0.4)]
        elif direction == "up":
            pts = [(cx - ch_size * 0.4, cy + ch_size * 0.4), (cx, cy - ch_size * 0.4), (cx + ch_size * 0.4, cy + ch_size * 0.4)]
            tail = [(cx - ch_size * 0.4, cy + ch_size * 0.6), (cx, cy), (cx + ch_size * 0.4, cy + ch_size * 0.6)]
        else:
            pts = [(cx - ch_size * 0.4, cy - ch_size * 0.4), (cx, cy + ch_size * 0.4), (cx + ch_size * 0.4, cy - ch_size * 0.4)]
            tail = [(cx - ch_size * 0.4, cy - ch_size * 0.6), (cx, cy), (cx + ch_size * 0.4, cy - ch_size * 0.6)]
        draw.polygon(pts, fill=(225, 130, 50), outline=(180, 90, 30))
        draw.polygon(tail, fill=(245, 170, 90), outline=(200, 130, 60))

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

        for entry in scenario.extras["arrows"]:
            self._draw_arrow_gate(draw, tuple(entry["cell"]), direction=entry["direction"])

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


__all__ = ["DirectionalArrowEntryGenerator"]
