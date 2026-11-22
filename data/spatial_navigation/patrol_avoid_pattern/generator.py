"""Scripted patrol enemy follows a fixed loop. Agent must reach goal without sharing a cell with the patrol on any step."""

from __future__ import annotations

from collections import deque
import math
from typing import List, Optional, Sequence, Tuple

from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
    format_xy_path_answer,
    format_xy_position,
)
from data.spatial_navigation.core.multi_agent_base import (
    FrameState,
    MultiAgentSpatialNavigationBase,
    MultiAgentScenario,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path

Cell = Tuple[int, int]
_MOVE_ACTIONS: Tuple[Tuple[str, Cell], ...] = (
    ("up", (-1, 0)),
    ("right", (0, 1)),
    ("down", (1, 0)),
    ("left", (0, -1)),
)


class PatrolAvoidPatternGenerator(MultiAgentSpatialNavigationBase):
    TASK_NAME = "patrol_avoid_pattern"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Patrol Avoidance"
    DEFAULT_SIZE = 8

    INPUT_DESC = (
        "The board has a blue circular agent, a green flag goal, and a red patrol guard. The patrol's looped "
        "two-cell route is shown by a faint red dashed segment. A solid red arrow starts at the guard's initial "
        "cell and points to the cell it will enter on the first tick. Dark cells are walls."
    )
    RULES_DESC = (
        "The patrol alternates between its two marked cells. On the first tick it follows the visible red arrow, "
        "then it reverses direction on every later tick. On the same tick, the agent may move up, right, down, or "
        "left, or use stay to remain on its current cell; the patrol advances even when the agent stays. Both actions "
        "commit together, then hold that state. The agent must reach the green flag without ever standing on the same "
        f"cell as the patrol after a step or swapping the same edge with it in opposite directions. "
        f"{GRID_PATH_STAY_ENCODING_SENTENCE}"
    )
    TI2V_OUTPUT = (
        "Generate a short video: move or hold the agent once per state while the patrol advances one cell along its "
        "loop in the same state, hold each committed state, and finish with the agent on the flag."
    )
    SOLUTION_IMAGE_OUTPUT = "Render the solved still image with the agent's path drawn in blue from start to goal, and the patrol's full loop drawn as red dashes."


    def _search_route(
        self,
        start: Cell,
        goal: Cell,
        walls: Sequence[Cell],
        patrol_loop: Sequence[Cell],
        *,
        allow_stay: bool,
    ) -> Optional[Tuple[List[str], List[Cell]]]:
        actions = list(_MOVE_ACTIONS)
        if allow_stay:
            actions.append(("stay", (0, 0)))
        wall_set = set(walls)
        period = len(patrol_loop)
        initial = (start, 0)
        seen = {initial}
        parent: dict[Tuple[Cell, int], Tuple[Tuple[Cell, int], str]] = {}
        queue = deque([initial])

        while queue:
            cell, phase = queue.popleft()
            if cell == goal:
                moves: List[str] = []
                cells = [cell]
                current = (cell, phase)
                while current in parent:
                    previous, direction = parent[current]
                    moves.append(direction)
                    current = previous
                    cells.append(current[0])
                moves.reverse()
                cells.reverse()
                return moves, cells

            next_phase = (phase + 1) % period
            patrol_current = patrol_loop[phase]
            patrol_next = patrol_loop[next_phase]
            for direction, (dr, dc) in actions:
                neighbor = (cell[0] + dr, cell[1] + dc)
                if not (0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size):
                    continue
                if neighbor in wall_set:
                    continue
                if neighbor == patrol_next or (cell == patrol_next and neighbor == patrol_current):
                    continue
                next_state = (neighbor, next_phase)
                if next_state in seen:
                    continue
                seen.add(next_state)
                parent[next_state] = ((cell, phase), direction)
                queue.append(next_state)
        return None

    @staticmethod
    def _pos(cell) -> str:
        return format_xy_position((cell[0] + 1, cell[1] + 1))

    def _authored_reasoning(self, scenario: MultiAgentScenario) -> str:
        from data.spatial_navigation.core.maze_style import route_run_breakdown

        path = [tuple(c) for c in scenario.primary_path]
        moves = list(scenario.primary_moves)
        gate, rest = [tuple(c) for c in scenario.extras["patrol_loop"]]
        goal = tuple(scenario.extras["goal"])
        gate_step = next((i for i, cell in enumerate(path) if cell == gate), None)
        sentences = [
            f"A wall bar seals the board apart and its only opening is {self._pos(gate)} — exactly the cell the "
            f"red guard patrols. The guard swings every tick between {self._pos(gate)} and its side cell "
            f"{self._pos(rest)}, first following its printed arrow, so the agent can occupy the opening only on a "
            "tick when the swing has carried the guard aside, and walking straight in without watching the rhythm "
            "would meet it head-on.",
        ]
        segment_anchor = 0
        described_wait = False
        for index, move in enumerate(moves):
            if move == "stay":
                before = path[segment_anchor : index + 1]
                if len(before) >= 2:
                    runs = route_run_breakdown(before)
                    leg = ", then ".join(f"{d} {len(r) - 1}" for d, r in runs)
                    sentences.append(f"The agent walks {leg} to {self._pos(path[index])}.")
                sentences.append(
                    "There it waits one tick while the guard completes its swing, letting the rhythm line up."
                )
                segment_anchor = index + 1
                described_wait = True
        tail = path[segment_anchor:]
        if len(tail) >= 2:
            runs = route_run_breakdown(tail)
            leg = ", then ".join(f"{d} {len(r) - 1}" for d, r in runs)
            if gate_step is not None and gate_step >= segment_anchor:
                sentences.append(
                    f"Then it walks {leg}, slipping through the opening at {self._pos(gate)} on the tick the guard "
                    f"stands aside at {self._pos(rest)}, and continues to the flag at {self._pos(goal)}."
                )
            else:
                sentences.append(f"Then it walks {leg} to the flag at {self._pos(goal)}.")
        if not described_wait:
            sentences.append("No wait was needed; the swing already lined up with the approach.")
        sentences.append(
            "After every one of its moves the agent stands clear of the guard's cell and never trades cells with "
            "it across an edge, so the timeline is safe from start to flag."
        )
        return " ".join(sentences)

    def sample_scenario(self) -> MultiAgentScenario:
        size = self.size
        for _ in range(300):
            vertical_barrier = bool(self.rng.getrandbits(1))
            barrier = self.rng.randint(2, size - 3)
            gate_axis = self.rng.randint(2, size - 3)
            rest_step = self.rng.choice((-1, 1))

            if vertical_barrier:
                gate = (gate_axis, barrier)
                patrol_rest = (gate_axis + rest_step, barrier)
                walls = {
                    (row, barrier)
                    for row in range(size)
                    if (row, barrier) not in {gate, patrol_rest}
                }
                walls.update(
                    {
                        (patrol_rest[0], barrier - 1),
                        (patrol_rest[0], barrier + 1),
                    }
                )
                start_pool = [
                    (row, col)
                    for row in range(size)
                    for col in range(barrier)
                    if (row, col) not in walls
                    and (row + col) % 2 == (gate[0] + gate[1]) % 2
                ]
                goal_pool = [
                    (row, col)
                    for row in range(size)
                    for col in range(barrier + 1, size)
                    if (row, col) not in walls
                ]
            else:
                gate = (barrier, gate_axis)
                patrol_rest = (barrier, gate_axis + rest_step)
                walls = {
                    (barrier, col)
                    for col in range(size)
                    if (barrier, col) not in {gate, patrol_rest}
                }
                walls.update(
                    {
                        (barrier - 1, patrol_rest[1]),
                        (barrier + 1, patrol_rest[1]),
                    }
                )
                start_pool = [
                    (row, col)
                    for row in range(barrier)
                    for col in range(size)
                    if (row, col) not in walls
                    and (row + col) % 2 == (gate[0] + gate[1]) % 2
                ]
                goal_pool = [
                    (row, col)
                    for row in range(barrier + 1, size)
                    for col in range(size)
                    if (row, col) not in walls
                ]

            if not start_pool or not goal_pool:
                continue
            agent_start = self.rng.choice(start_pool)
            goal = self.rng.choice(goal_pool)
            protected = {agent_start, goal, gate, patrol_rest}
            extra_candidates = [
                (row, col)
                for row in range(size)
                for col in range(size)
                if (row, col) not in walls and (row, col) not in protected
            ]
            self.rng.shuffle(extra_candidates)
            walls.update(extra_candidates[: self.rng.randint(1, 4)])

            patrol_loop = [gate, patrol_rest]
            route = self._search_route(
                agent_start,
                goal,
                sorted(walls),
                patrol_loop,
                allow_stay=True,
            )
            if route is None:
                continue
            moves, cells = route
            if "stay" not in moves:
                continue
            if self._search_route(
                agent_start,
                goal,
                sorted(walls),
                patrol_loop,
                allow_stay=False,
            ) is not None:
                continue

            scenario = MultiAgentScenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                size=size,
                walls=sorted(walls),
                primary_path=cells,
                primary_moves=moves,
                secondary_path=list(patrol_loop),
                extras={
                    "agent_start": agent_start,
                    "goal": goal,
                    "patrol_loop": patrol_loop,
                    "patrol_next": patrol_rest,
                    "wait_required": True,
                    "barrier_orientation": "vertical" if vertical_barrier else "horizontal",
                },
            )
            scenario.answer_text = format_xy_path_answer(scenario.primary_path)
            return scenario
        raise RuntimeError("patrol_avoid_pattern sample failed")

    def render_static_layer(self, draw, image, scenario):
        loop = scenario.extras["patrol_loop"]
        ax, ay = self.cell_center(loop[0])
        bx, by = self.cell_center(loop[1])
        length = math.hypot(bx - ax, by - ay)
        dash_count = max(4, int(length / 7))
        for index in range(dash_count):
            if index % 2:
                continue
            t0 = index / dash_count
            t1 = min(1.0, (index + 0.62) / dash_count)
            draw.line(
                [
                    (ax + (bx - ax) * t0, ay + (by - ay) * t0),
                    (ax + (bx - ax) * t1, ay + (by - ay) * t1),
                ],
                fill=(225, 130, 130),
                width=2,
            )

        unit_x = (bx - ax) / length
        unit_y = (by - ay) / length
        arrow_start = (ax + unit_x * self.cell_size * 0.24, ay + unit_y * self.cell_size * 0.24)
        arrow_tip = (ax + unit_x * self.cell_size * 0.72, ay + unit_y * self.cell_size * 0.72)
        draw.line([arrow_start, arrow_tip], fill=(190, 35, 35), width=max(3, self.cell_size // 16))
        head_length = max(8.0, self.cell_size * 0.17)
        head_width = max(6.0, self.cell_size * 0.12)
        base_x = arrow_tip[0] - unit_x * head_length
        base_y = arrow_tip[1] - unit_y * head_length
        perp_x, perp_y = -unit_y, unit_x
        draw.polygon(
            [
                arrow_tip,
                (base_x + perp_x * head_width, base_y + perp_y * head_width),
                (base_x - perp_x * head_width, base_y - perp_y * head_width),
            ],
            fill=(190, 35, 35),
        )
        self.draw_goal_flag(draw, scenario.extras["goal"], fill=(50, 160, 70))

    def render_solution_overlay(self, draw, image, scenario):
        if len(scenario.primary_path) >= 2:
            pts = [self.cell_center(c) for c in scenario.primary_path]
            draw_spatial_navigation_path(draw, pts, color=(56, 130, 220), width=max(5, self.cell_size // 7))

    def render_path_overlay(self, draw, path_cells, color=(214, 42, 42)) -> None:
        return

    def render_agents(self, draw, image, scenario, frame_state):
        cx, cy = self.cell_center(frame_state.primary_pos)
        self.draw_circle_agent(draw, (cx, cy), fill=(56, 130, 220), outline=(20, 60, 120))
        if frame_state.secondary_pos is not None:
            cx2, cy2 = self.cell_center(frame_state.secondary_pos)
            # patrol guard: red diamond with helmet
            self.draw_diamond_agent(draw, (cx2, cy2), fill=(220, 60, 60), outline=(120, 20, 20))

    def build_video_frame_states(self, scenario, substeps):
        ap = scenario.primary_path
        loop = scenario.extras["patrol_loop"]
        period = len(loop)
        states: List[FrameState] = []
        states.append(
            FrameState(primary_pos=ap[0], primary_cell=ap[0], secondary_pos=loop[0], secondary_cell=loop[0])
        )
        for i in range(len(ap) - 1):
            pb = loop[(i + 1) % period]
            states.append(
                FrameState(
                    primary_pos=ap[i + 1],
                    primary_cell=ap[i + 1],
                    secondary_pos=pb,
                    secondary_cell=pb,
                )
            )
        return states
