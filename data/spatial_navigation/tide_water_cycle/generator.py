"""Water cycle: rows fill with blue water on a fixed schedule. Cross during low-tide moments."""

from __future__ import annotations

from collections import deque
import math
from typing import List, Sequence, Tuple

from data.common.animation_policy import MAX_CLOCKED_STEPS
from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
    format_xy_path_answer,
    format_xy_position,
)
from data.spatial_navigation.core.multi_agent_base import (
    FrameState,
    MultiAgentSpatialNavigationBase,
    MultiAgentScenario,
    bfs_path,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path

Cell = Tuple[int, int]
_ACTIONS: Tuple[Tuple[str, Cell], ...] = (
    ("up", (-1, 0)),
    ("right", (0, 1)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("stay", (0, 0)),
)


class TideWaterCycleGenerator(MultiAgentSpatialNavigationBase):
    TASK_NAME = "tide_water_cycle"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Tide Water Cycle"
    DEFAULT_SIZE = 8

    INPUT_DESC = (
        "The board has a blue agent and a green flag goal. Two horizontal rows are tide rows: their cells "
        "alternate between dry sand (light tan) and high tide (blue water). A round clock indicator at the right "
        "side beside the tide rows shows their shared period. Dark cells are walls."
    )
    RULES_DESC = (
        "Tide row 1 is high on even steps and dry on odd steps. Tide row 2 is high on odd steps and dry on "
        "even steps. The agent may not stand on a high-tide cell after a step. Each logical step contains one "
        "agent action and one tide change on an autonomous clock. The action may move up, right, down, or left, or use "
        "stay to remain on the current cell; the tide advances even on stay. Each logical step has three ordered "
        "phases: the agent may act before the tide changes, the tide changes, and otherwise the agent acts afterward. "
        "The tide never waits for the agent. Move out before the current row becomes high, or enter a newly dry row "
        "after the change. Moving directly from the row becoming high into the row becoming dry is illegal because "
        f"neither phase is collision-free. {GRID_PATH_STAY_ENCODING_SENTENCE}"
    )
    TI2V_OUTPUT = (
        "Generate a short video with a steady tide rhythm. For each logical step, show the chosen agent move in its legal "
        "phase and keep the agent still in the other phase. At the tide commit, switch the complete high/low row "
        "pattern and clock directly to the next discrete phase. Do not show a fractional waterline, color lerp, "
        "partial row, or undeclared transition state."
    )
    SOLUTION_IMAGE_OUTPUT = "Render the solved still image with the agent's path drawn in blue."


    @staticmethod
    def _pos(cell) -> str:
        return format_xy_position((cell[0] + 1, cell[1] + 1))

    def _authored_reasoning_segments(self, scenario: MultiAgentScenario):
        from data.spatial_navigation.core.maze_style import route_run_breakdown

        cells = [tuple(c) for c in scenario.primary_path]
        row1 = int(scenario.extras["tide_row1"])
        row2 = int(scenario.extras["tide_row2"])
        tide_rows = {row1, row2}
        goal = tuple(scenario.extras["goal"])

        entry_steps = [
            step
            for step, (previous, current) in enumerate(zip(cells, cells[1:]), start=1)
            if current[0] in tide_rows and previous[0] != current[0]
        ]

        def leg_text(begin: int, end: int) -> str:
            span = cells[begin : end + 1]
            stays = sum(1 for a, b in zip(span, span[1:]) if a == b)
            deduped = [span[0]] + [cell for prev, cell in zip(span, span[1:]) if cell != prev]
            parts = []
            if stays:
                parts.append(f"waits {stays} tick{'s' if stays > 1 else ''} for the flip")
            if len(deduped) >= 2:
                runs = route_run_breakdown(deduped)
                parts.append("walks " + ", then ".join(f"{d} {len(r) - 1}" for d, r in runs))
            return " and ".join(parts) if parts else "stays in place"

        opening = (
            f"Two tide lanes cross the board: the lane at row {row1 + 1} carries water after even numbers of "
            f"moves and drains after odd ones, while the lane at row {row2 + 1} runs on the opposite beat, and "
            f"the flag waits at {self._pos(goal)} beyond both. Standing in water is fatal, so each lane entry has "
            "to be timed onto the beat that has just drained it — sometimes by spending a wait."
        )
        segments = []
        anchor = 0
        for index, step in enumerate(entry_steps):
            lane_row = cells[step][0]
            body = (
                f"{'The agent' if index == 0 else 'It'} {leg_text(anchor, step)}: after move {step} the lane at "
                f"row {lane_row + 1} has just drained, so it commits onto {self._pos(cells[step])} while the "
                "other lane holds the water."
            )
            if index == 0:
                body = f"{opening} {body}"
            segments.append(body)
            anchor = step
        closing = (
            f"From there it {leg_text(anchor, len(cells) - 1)} and stops on the flag at {self._pos(goal)}; after "
            "every move its cell was dry, so the crossing beats the tide legally."
        )
        segments.append(closing)
        return segments

    @staticmethod
    def _transition_is_tide_legal(
        current: Cell,
        neighbor: Cell,
        parity: int,
        tide_row1: int,
        tide_row2: int,
    ) -> bool:
        next_parity = 1 - parity
        old_high = tide_row1 if parity == 0 else tide_row2
        new_high = tide_row1 if next_parity == 0 else tide_row2
        if current[0] == new_high and neighbor[0] == old_high:
            return False
        if neighbor[0] == tide_row1 and next_parity == 0:
            return False
        if neighbor[0] == tide_row2 and next_parity == 1:
            return False
        return True

    def _path_is_tide_legal(
        self,
        path: Sequence[Cell],
        tide_row1: int,
        tide_row2: int,
    ) -> bool:
        parity = 0
        for current, neighbor in zip(path, path[1:]):
            if not self._transition_is_tide_legal(
                current,
                neighbor,
                parity,
                tide_row1,
                tide_row2,
            ):
                return False
            parity = 1 - parity
        return True

    def sample_scenario(self) -> MultiAgentScenario:
        size = self.size
        for _ in range(300):
            cells_pool = [(r, c) for r in range(size) for c in range(size)]
            self.rng.shuffle(cells_pool)
            walls = cells_pool[:self.rng.randint(2, 3)]
            tide_row1 = self.rng.randint(2, size // 2)
            tide_row2 = self.rng.randint(size // 2 + 1, size - 3)
            agent_start = (0, self.rng.randint(0, size - 1))
            goal = (size - 1, self.rng.randint(0, size - 1))
            if agent_start in walls or goal in walls:
                continue
            if any((tide_row1, c) in walls for c in range(size)) or any((tide_row2, c) in walls for c in range(size)):
                continue
            plain_path = bfs_path(agent_start, goal, size, walls)
            if plain_path is None or self._path_is_tide_legal(plain_path, tide_row1, tide_row2):
                continue

            init = (agent_start, 0)
            seen = {init}
            parent = {}
            q = deque([init])
            found = None
            while q:
                cell, parity = q.popleft()
                if cell == goal:
                    found = (cell, parity)
                    break
                for d, (dr, dc) in _ACTIONS:
                    nb = (cell[0] + dr, cell[1] + dc)
                    if not (0 <= nb[0] < size and 0 <= nb[1] < size):
                        continue
                    if nb in walls:
                        continue
                    if not self._transition_is_tide_legal(
                        cell,
                        nb,
                        parity,
                        tide_row1,
                        tide_row2,
                    ):
                        continue
                    new_parity = 1 - parity
                    nxt = (nb, new_parity)
                    if nxt in seen:
                        continue
                    if len(seen) > 5000:
                        break
                    seen.add(nxt)
                    parent[nxt] = ((cell, parity), d)
                    q.append(nxt)
            if found is None:
                continue
            cur = found
            cells = [cur[0]]
            moves = []
            while cur in parent:
                prev, d = parent[cur]
                moves.append(d)
                cur = prev
                cells.append(cur[0])
            moves.reverse()
            cells.reverse()
            if not (5 <= len(moves) <= MAX_CLOCKED_STEPS):
                continue
            if "stay" not in moves:
                continue
            # Bound the interleaved key states: each tide-lane entry is one
            # reasoning image, so keep the reference within the 7-image
            tide_rows = {tide_row1, tide_row2}
            entries = sum(
                1
                for previous, current in zip(cells, cells[1:])
                if current[0] in tide_rows and previous[0] != current[0]
            )
            if not (1 <= entries <= 4):
                continue
            scenario = MultiAgentScenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                size=size,
                walls=walls,
                primary_path=cells,
                primary_moves=moves,
                extras={
                    "agent_start": agent_start,
                    "goal": goal,
                    "tide_row1": tide_row1,
                    "tide_row2": tide_row2,
                    "reference_uses_stay": True,
                    "static_route_tide_legal": False,
                },
            )
            scenario.answer_text = format_xy_path_answer(scenario.primary_path)
            return scenario
        raise RuntimeError("tide_water_cycle sample failed")

    def render_static_layer(self, draw, image, scenario):
        # Default static rendering shows tide phase 0 (row1=high, row2=dry).
        for r in (scenario.extras["tide_row1"], scenario.extras["tide_row2"]):
            for c in range(self.size):
                if (r, c) in scenario.walls:
                    continue
                l, t, rt, b = self.cell_box((r, c))
                # tan sand background
                draw.rectangle([l + 1, t + 1, rt - 1, b - 1], fill=(252, 240, 215), outline=(220, 195, 150), width=1)
        # mark each tide row with a "tide" badge on the right
        for ri, r in enumerate((scenario.extras["tide_row1"], scenario.extras["tide_row2"])):
            cy = self.margin + r * self.cell_size + self.cell_size // 2
            # phase indicator: row1 = "H/D" on left
            label = "T1" if ri == 0 else "T2"
            draw.text((self.margin - self.cell_size * 0.85, cy), label, fill=(20, 80, 130), font=self._font_id, anchor="mm")
        self.draw_goal_flag(draw, scenario.extras["goal"], fill=(50, 160, 70))

    def render_solution_overlay(self, draw, image, scenario):
        if len(scenario.primary_path) >= 2:
            pts = [self.cell_center(c) for c in scenario.primary_path]
            draw_spatial_navigation_path(draw, pts, color=(56, 130, 220), width=max(5, self.cell_size // 7))

    def render_path_overlay(self, draw, path_cells, color=(214, 42, 42)) -> None:
        return

    def render_agents(self, draw, image, scenario, frame_state):
        parity = int(frame_state.extras.get("parity", 0))
        row1 = int(scenario.extras["tide_row1"])
        row2 = int(scenario.extras["tide_row2"])
        high_row = row1 if parity == 0 else row2
        levels = {row1: 0.0, row2: 0.0}
        levels[high_row] = 1.0
        clock_position = int(round(float(frame_state.extras.get("clock_position", parity))))

        for row, level in levels.items():
            if level <= 0.0:
                continue
            for c in range(self.size):
                if (row, c) in scenario.walls:
                    continue
                l, t, rt, b = self.cell_box((row, c))
                water_top = (b - 1) - (self.cell_size - 2) * level
                draw.rectangle(
                    [l + 1, water_top, rt - 1, b - 1],
                    fill=(140, 200, 240),
                    outline=(40, 110, 170),
                    width=1,
                )
                if level >= 0.28:
                    cx = (l + rt) // 2
                    wave_y = water_top + min(8.0, max(3.0, (b - water_top) * 0.25))
                    draw.arc(
                        [cx - 8, wave_y - 3, cx + 8, wave_y + 3],
                        start=0,
                        end=180,
                        fill=(60, 130, 200),
                        width=1,
                    )

        # A visible clock hand advances by half a turn on every equal-length tick.
        clock_x = self.margin + self.size * self.cell_size + min(14, self.margin * 0.45)
        clock_y = self.margin + ((row1 + row2 + 1) * self.cell_size) / 2
        radius = max(9, int(self.cell_size * 0.18))
        draw.ellipse(
            [clock_x - radius, clock_y - radius, clock_x + radius, clock_y + radius],
            fill=(250, 250, 248),
            outline=(40, 90, 130),
            width=2,
        )
        angle = -math.pi / 2 + math.pi * clock_position
        draw.line(
            [
                (clock_x, clock_y),
                (clock_x + math.cos(angle) * radius * 0.72, clock_y + math.sin(angle) * radius * 0.72),
            ],
            fill=(30, 75, 120),
            width=2,
        )
        cx, cy = self.cell_center(frame_state.primary_pos)
        self.draw_circle_agent(draw, (cx, cy), fill=(56, 130, 220), outline=(20, 60, 120))

    def build_video_frame_states(self, scenario, substeps):
        ap = scenario.primary_path
        states: List[FrameState] = [
            FrameState(
                primary_pos=ap[0],
                primary_cell=ap[0],
                extras={"parity": 0, "clock_position": 0.0},
            )
        ]
        parity = 0
        tide_row1 = int(scenario.extras["tide_row1"])
        tide_row2 = int(scenario.extras["tide_row2"])
        for i in range(len(ap) - 1):
            next_parity = 1 - parity
            old_high = tide_row1 if parity == 0 else tide_row2
            new_high = tide_row1 if next_parity == 0 else tide_row2
            must_move_before = ap[i][0] == new_high
            must_move_after = ap[i + 1][0] == old_high
            if must_move_before and must_move_after:
                raise ValueError("tide phase has no collision-free fixed-clock slot")
            move_before = must_move_before or not must_move_after
            before_cell = ap[i + 1] if move_before else ap[i]
            states.append(
                FrameState(
                    primary_pos=before_cell,
                    primary_cell=before_cell,
                    extras={
                        "parity": parity,
                        "clock_phase": "agent_before",
                        "clock_position": float(parity),
                    },
                )
            )
            states.append(
                FrameState(
                    primary_pos=before_cell,
                    primary_cell=before_cell,
                    extras={
                        "parity": next_parity,
                        "clock_phase": "world_transition",
                        "clock_position": float(next_parity),
                        "from_parity": parity,
                        "to_parity": next_parity,
                    },
                )
            )
            states.append(
                FrameState(
                    primary_pos=ap[i + 1],
                    primary_cell=ap[i + 1],
                    extras={
                        "parity": next_parity,
                        "clock_phase": "agent_after",
                        "clock_position": float(next_parity),
                    },
                )
            )
            parity = next_parity
        return states
