"""Time-aware routing around simultaneously falling blocks."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
    format_xy_path_answer,
    format_xy_position,
)
from data.spatial_navigation.core.multi_agent_base import FrameState, MultiAgentSpatialNavigationBase, MultiAgentScenario, bfs_path
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path


Cell = Tuple[int, int]
ACTIONS: Tuple[Tuple[str, Cell], ...] = (
    ("up", (-1, 0)),
    ("right", (0, 1)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("stay", (0, 0)),
)


def _block_cell(spec: Dict[str, int], time_step: int, size: int) -> Optional[Cell]:
    row = int(spec["start_row"]) + int(time_step)
    if not (0 <= row < size):
        return None
    return row, int(spec["column"])


def _falling_block_path(
    *,
    size: int,
    start: Cell,
    goal: Cell,
    walls: Sequence[Cell],
    blocks: Sequence[Dict[str, int]],
    max_steps: int,
) -> Optional[Tuple[List[Cell], List[str]]]:
    wall_set = set(walls)
    initial = (start, 0)
    queue = deque([initial])
    parent: Dict[Tuple[Cell, int], Tuple[Tuple[Cell, int], str]] = {}
    seen = {initial}

    while queue:
        cell, time_step = queue.popleft()
        if cell == goal:
            states = [(cell, time_step)]
            moves: List[str] = []
            while states[-1] != initial:
                prev, move = parent[states[-1]]
                moves.append(move)
                states.append(prev)
            states.reverse()
            moves.reverse()
            return [state[0] for state in states], moves
        if time_step >= max_steps:
            continue

        current_blocks = {_block_cell(spec, time_step, size) for spec in blocks}
        current_blocks.discard(None)
        next_time = time_step + 1
        next_blocks = {_block_cell(spec, next_time, size) for spec in blocks}
        next_blocks.discard(None)
        block_edges = {
            (current, nxt)
            for spec in blocks
            for current, nxt in [(_block_cell(spec, time_step, size), _block_cell(spec, next_time, size))]
            if current is not None and nxt is not None
        }

        for move, (dr, dc) in ACTIONS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in wall_set or nxt in current_blocks or nxt in next_blocks:
                continue
            if any(cell == block_next and nxt == block_current for block_current, block_next in block_edges):
                continue
            state = (nxt, next_time)
            if state in seen:
                continue
            seen.add(state)
            parent[state] = ((cell, time_step), move)
            queue.append(state)
    return None


def _replay_actions(
    *,
    size: int,
    start: Cell,
    goal: Cell,
    walls: Sequence[Cell],
    blocks: Sequence[Dict[str, int]],
    moves: Sequence[str],
) -> Optional[List[Cell]]:
    wall_set = set(walls)
    action_deltas = dict(ACTIONS)
    cell = start
    path = [cell]
    for time_step, move in enumerate(moves):
        delta = action_deltas.get(move)
        if delta is None or cell == goal:
            return None
        current_blocks = {_block_cell(spec, time_step, size) for spec in blocks}
        current_blocks.discard(None)
        next_blocks = {_block_cell(spec, time_step + 1, size) for spec in blocks}
        next_blocks.discard(None)
        nxt = cell[0] + delta[0], cell[1] + delta[1]
        if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
            return None
        if nxt in wall_set or nxt in current_blocks or nxt in next_blocks:
            return None
        if any(
            cell == block_next and nxt == block_current
            for spec in blocks
            for block_current, block_next in [
                (_block_cell(spec, time_step, size), _block_cell(spec, time_step + 1, size))
            ]
            if block_current is not None and block_next is not None
        ):
            return None
        cell = nxt
        path.append(cell)
    return path if cell == goal else None


def _path_uses_vacated_block_cell(path: Sequence[Cell], blocks: Sequence[Dict[str, int]], size: int) -> bool:
    initial_cells = {_block_cell(spec, 0, size) for spec in blocks}
    initial_cells.discard(None)
    return any(cell in initial_cells for cell in path[1:])


class FallingBlockDodgeGenerator(MultiAgentSpatialNavigationBase):
    TASK_NAME = "falling_block_dodge"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Falling Block Dodge"
    DEFAULT_SIZE = 8

    INPUT_DESC = (
        "The board shows a blue agent, a green goal flag, dark fixed walls, and two or three orange blocks. "
        "Every orange block is already visible at its initial cell. Its current (x, y) position fully determines its "
        "state; there are no hidden timers or staggered starts."
    )
    RULES_DESC = (
        "A step is one of up, right, down, left, or stay. First the agent moves to its adjacent destination, which "
        "must not contain a current block. After the agent settles, all visible orange blocks simultaneously drop "
        "exactly one cell on every completed tick, starting with tick 1; the agent's settled cell must not be any "
        "block's destination. A block that drops below the board disappears without refill. Stay is legal: it keeps "
        "the agent centred while still completing the tick and making every visible block fall. Walls, block columns, "
        f"and block identities never change. {GRID_PATH_STAY_ENCODING_SENTENCE}"
    )
    TI2V_OUTPUT = (
        "Hold the exact input, then serialize every step into two collision-free phases: move or hold the agent while "
        "all blocks remain fixed, hold the settled agent state, then keep the agent fixed while each active orange block "
        "drops one cell. Commit and hold the new block state before the next agent move. Finish with the agent on "
        "the goal. Do not move the agent and blocks through one another, skip a fall, or refill."
    )
    SOLUTION_IMAGE_OUTPUT = (
        "Render the terminal state with the complete legal route drawn in blue, the agent on the goal, and every "
        "still-on-board block at its terminal row; preserve walls, labels, grid, background, and camera."
    )

    @staticmethod
    def _pos(cell) -> str:
        return format_xy_position((cell[0] + 1, cell[1] + 1))

    def _authored_reasoning(self, scenario: MultiAgentScenario) -> str:
        path = [tuple(c) for c in scenario.primary_path]
        moves = list(scenario.primary_moves)
        blocks = scenario.extras["falling_blocks"]
        block_text = ", ".join(
            f"one starting at {self._pos((int(spec['start_row']), int(spec['column'])))}"
            for spec in blocks
        )
        sentences = [
            f"The agent starts at {self._pos(path[0])} and the flag waits at {self._pos(path[-1])}. "
            f"{len(blocks)} orange blocks hang over the board — {block_text} — and every one of them drops one row "
            "after each of the agent's actions, so the plan is about timing: pass under a column before its block "
            "arrives, or let it fall past and walk through the cell it has left.",
        ]
        for tick, (move, cell) in enumerate(zip(moves, path[1:]), start=1):
            rows_after = [
                _block_cell(spec, tick, scenario.size)
                for spec in blocks
            ]
            on_board = [c for c in rows_after if c is not None]
            if on_board:
                block_positions = ", ".join(self._pos(c) for c in on_board)
                block_note = f"the blocks settle at {block_positions}"
            else:
                block_note = "every block has fallen off the board"
            if move == "stay":
                sentences.append(
                    f"Tick {tick}: it waits at {self._pos(cell)} while {block_note}, keeping clear."
                )
            else:
                sentences.append(
                    f"Tick {tick}: it steps {move} to {self._pos(cell)}, and {block_note}."
                )
        sentences.append(
            "That final step lands on the flag; at no tick did the agent stand in a block's cell or cross one "
            "mid-fall, so the whole action sequence is safe."
        )
        return " ".join(sentences)

    def sample_scenario(self) -> MultiAgentScenario:
        size = self.size
        cells = [(row, col) for row in range(size) for col in range(size)]
        for _ in range(2000):
            self.rng.shuffle(cells)
            walls = cells[: self.rng.randint(4, 8)]
            free = [cell for cell in cells if cell not in walls]
            if len(free) < 8:
                continue
            start, goal = free[0], free[1]
            if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 6:
                continue

            columns = self.rng.sample(range(size), self.rng.randint(2, 3))
            blocks = [
                {
                    "column": column,
                    "start_row": self.rng.randint(0, 2),
                }
                for column in columns
            ]
            occupied_zero = {_block_cell(spec, 0, size) for spec in blocks}
            occupied_one = {_block_cell(spec, 1, size) for spec in blocks}
            if (
                len(occupied_zero) < 2
                or len(occupied_one) < 2
                or start in occupied_zero
                or goal in occupied_zero
            ):
                continue
            block_trajectory = {
                cell
                for spec in blocks
                for time_step in range(size + 1)
                for cell in [_block_cell(spec, time_step, size)]
                if cell is not None
            }
            if set(walls) & block_trajectory:
                continue

            static_path = bfs_path(start, goal, size, walls)
            if static_path is None:
                continue
            result = _falling_block_path(
                size=size,
                start=start,
                goal=goal,
                walls=walls,
                blocks=blocks,
                max_steps=16,
            )
            if result is None:
                continue
            path, moves = result
            if not (6 <= len(moves) <= 14):
                continue
            static_moves = [
                next(name for name, delta in ACTIONS if delta == (b[0] - a[0], b[1] - a[1]))
                for a, b in zip(static_path, static_path[1:])
            ]
            if _replay_actions(
                size=size,
                start=start,
                goal=goal,
                walls=walls,
                blocks=blocks,
                moves=static_moves,
            ) is not None:
                continue
            if not _path_uses_vacated_block_cell(path, blocks, size):
                continue

            scenario = MultiAgentScenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                size=size,
                walls=list(walls),
                primary_path=path,
                primary_moves=moves,
                extras={
                    "agent_start": start,
                    "goal": goal,
                    "falling_blocks": blocks,
                },
            )
            scenario.answer_text = format_xy_path_answer(scenario.primary_path)
            return scenario
        raise RuntimeError("falling_block_dodge failed to sample a load-bearing simultaneous-fall scenario")

    def _prompts(self, scenario):
        prompts = super()._prompts(scenario)
        replacements = {
            "while each on-board falling block advances when due": (
                "then every on-board falling block advances one row simultaneously"
            ),
            "each orange block and its warning label": "each orange block",
        }
        return {
            key: self._replace_prompt_phrases(value, replacements)
            for key, value in prompts.items()
        }

    @staticmethod
    def _replace_prompt_phrases(text, replacements):
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def render_static_layer(self, draw, image, scenario):
        self.draw_goal_flag(draw, scenario.extras["goal"], fill=(50, 160, 70))

    def render_solution_overlay(self, draw, image, scenario):
        if len(scenario.primary_path) >= 2:
            draw_spatial_navigation_path(
                draw,
                [self.cell_center(cell) for cell in scenario.primary_path],
                color=(56, 130, 220),
                width=max(5, self.cell_size // 7),
            )

    def render_path_overlay(self, draw, path_cells, color=(214, 42, 42)) -> None:
        return

    def render_agents(self, draw, image, scenario, frame_state):
        time_step = int(frame_state.extras.get("time_step", 0))
        for spec in scenario.extras["falling_blocks"]:
            current = _block_cell(spec, time_step, self.size)
            if current is None:
                continue
            cx, cy = self.cell_center(current)
            half = self.cell_size * 0.30
            draw.rounded_rectangle(
                [cx - half, cy - half, cx + half, cy + half],
                radius=max(4, int(self.cell_size * 0.08)),
                fill=(245, 151, 54),
                outline=(165, 82, 22),
                width=2,
            )
        cx, cy = self.cell_center(frame_state.primary_pos)
        self.draw_circle_agent(draw, (cx, cy), fill=(56, 130, 220), outline=(20, 60, 120))

    def build_video_frame_states(self, scenario, substeps):
        path = scenario.primary_path
        states: List[FrameState] = [
            FrameState(
                primary_pos=path[0],
                primary_cell=path[0],
                extras={"time_step": 0},
            )
        ]
        for step, (_start, end) in enumerate(zip(path, path[1:])):
            states.append(
                FrameState(
                    primary_pos=end,
                    primary_cell=end,
                    extras={"time_step": step},
                )
            )
            states.append(
                FrameState(
                    primary_pos=end,
                    primary_cell=end,
                    extras={"time_step": step + 1},
                )
            )
        return states


__all__ = [
    "FallingBlockDodgeGenerator",
    "_block_cell",
    "_falling_block_path",
    "_replay_actions",
]
