"""Agent A picks up parcel, walks to a fixed meeting cell, hands off to agent B which delivers to dropoff.

Solution = single concatenated path: A's pickup→meet, then B's meet→dropoff.
"""

from __future__ import annotations

from typing import List

from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
    format_named_xy_paths_answer,
    format_xy_position,
)
from data.spatial_navigation.core.multi_agent_base import (
    FrameState,
    MultiAgentSpatialNavigationBase,
    MultiAgentScenario,
    bfs_path,
    cells_to_moves,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path


class AgentCourierRelayGenerator(MultiAgentSpatialNavigationBase):
    TASK_NAME = "agent_courier_relay"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Agent Courier Relay"
    DEFAULT_SIZE = 8

    INPUT_DESC = (
        "The board has a blue agent A holding a brown parcel and an orange agent B waiting on a fixed "
        "stationary cell. A green diamond marks the dropoff. The orange agent never moves until A reaches "
        "its cell. Dark cells are walls."
    )
    RULES_DESC = (
        "Agent A walks to agent B's cell, hands the parcel over, and then agent B walks to the green dropoff "
        "while A stays put. During its own leg, the active courier may move up, right, down, or left, or use stay "
        "to remain on its current floor cell. Any complete legal pair of route legs is accepted; route length is "
        f"not scored. {GRID_PATH_STAY_ENCODING_SENTENCE}"
    )
    TI2V_OUTPUT = (
        "Generate a short video showing agent A walking to B with the parcel riding on A. While both agents share "
        "the meeting cell they appear side by side as two smaller discs, A on the left and B on the right, and the "
        "handoff commit moves the parcel from A's disc to B's disc in one switch. Then B carries the parcel to the "
        "green dropoff while A remains on the meeting cell."
    )
    SOLUTION_IMAGE_OUTPUT = (
        "Render the solved still image with agent A's blue path drawn from start to the meeting cell, agent B's "
        "orange path drawn from the meeting cell to the dropoff, A resting on the meeting cell, and B on the "
        "dropoff with the parcel."
    )

    @staticmethod
    def _pos(cell) -> str:
        return format_xy_position((cell[0] + 1, cell[1] + 1))

    def _authored_reasoning(self, scenario: MultiAgentScenario) -> str:
        from data.spatial_navigation.core.maze_style import route_run_breakdown

        leg1 = [tuple(c) for c in scenario.extras["leg1"]]
        leg2 = [tuple(c) for c in scenario.extras["leg2"]]

        def leg_text(cells) -> str:
            runs = route_run_breakdown(cells)
            return ", then ".join(f"{direction} {len(run) - 1}" for direction, run in runs)

        sentences = [
            f"Blue A starts at {self._pos(leg1[0])} with the parcel, orange B waits at {self._pos(leg1[-1])}, and "
            f"the green dropoff diamond sits at {self._pos(leg2[-1])}. The parcel can only change hands on B's "
            "cell, so the job splits into A's leg to the meeting cell and B's leg from there to the dropoff.",
            f"A walks {leg_text(leg1)}, arriving beside B on the meeting cell, and hands the parcel over; A stays "
            "put from now on.",
            f"Carrying the parcel, B walks {leg_text(leg2)} and stops on the dropoff diamond, so the parcel is "
            "delivered with each courier having moved only during its own leg.",
        ]
        return " ".join(sentences)

    def sample_scenario(self) -> MultiAgentScenario:
        size = self.size
        for _ in range(300):
            cells_pool = [(r, c) for r in range(size) for c in range(size)]
            self.rng.shuffle(cells_pool)
            walls = cells_pool[:self.rng.randint(3, 6)]
            free = [c for c in cells_pool if c not in walls]
            if len(free) < 6:
                continue
            a_start = free[0]
            b_meet = free[1]
            dropoff = free[2]
            if a_start in (b_meet, dropoff) or b_meet == dropoff:
                continue
            if abs(a_start[0] - b_meet[0]) + abs(a_start[1] - b_meet[1]) < 3:
                continue
            if abs(b_meet[0] - dropoff[0]) + abs(b_meet[1] - dropoff[1]) < 3:
                continue
            leg1 = bfs_path(a_start, b_meet, size, walls)
            leg2 = bfs_path(b_meet, dropoff, size, walls)
            if leg1 is None or leg2 is None:
                continue
            if not (3 <= len(leg1) + len(leg2) - 1 <= 14):
                continue
            full_path = leg1 + leg2[1:]
            scenario = MultiAgentScenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                size=size,
                walls=walls,
                primary_path=full_path,
                primary_moves=cells_to_moves(full_path),
                extras={
                    "a_start": a_start, "b_meet": b_meet, "dropoff": dropoff,
                    "leg1_len": len(leg1), "leg1": leg1, "leg2": leg2,
                },
            )
            a_moves = cells_to_moves(leg1)
            b_moves = cells_to_moves(leg2)
            scenario.primary_moves = a_moves + b_moves
            scenario.answer_text = format_named_xy_paths_answer(
                {"A": leg1, "B": leg2}
            )
            return scenario
        raise RuntimeError("agent_courier_relay sample failed")

    def render_static_layer(self, draw, image, scenario):
        dropoff = scenario.extras["dropoff"]
        # dropoff: green diamond
        self.draw_filled_cell(draw, dropoff, fill=(180, 230, 175), outline=(40, 130, 60))
        cx, cy = self.cell_center(dropoff)
        # green diamond marker
        r = self.cell_size * 0.22
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=(40, 160, 70), outline=(20, 80, 40))

    def render_solution_overlay(self, draw, image, scenario):
        leg1 = scenario.extras["leg1"]
        leg2 = scenario.extras["leg2"]
        if len(leg1) >= 2:
            pts = [self.cell_center(c) for c in leg1]
            draw_spatial_navigation_path(draw, pts, color=(56, 130, 220), width=max(5, self.cell_size // 7))
        if len(leg2) >= 2:
            pts = [self.cell_center(c) for c in leg2]
            draw_spatial_navigation_path(draw, pts, color=(238, 132, 36), width=max(5, self.cell_size // 7))

    def render_path_overlay(self, draw, path_cells, color=(214, 42, 42)) -> None:
        return

    def render_agents(self, draw, image, scenario, frame_state):
        """Draw both couriers with a joint-occupancy layout at the meet.

        While A travels, B waits centred on its cell.  Whenever both agents
        occupy the meeting cell they are drawn side by side as two smaller
        discs (A left, B right) so neither identity is ever hidden; the
        handoff commit is the parcel switching from A's disc to B's disc in
        that shared layout.  After B departs, A stays centred at the meeting
        cell.
        """

        ph = frame_state.extras.get("phase", "leg1")
        b_meet = tuple(scenario.extras["b_meet"])
        carrier_pos = tuple(frame_state.primary_pos)
        shared = (
            (ph == "leg1" and tuple(frame_state.primary_cell) == b_meet)
            or (ph == "leg2" and tuple(frame_state.primary_cell) == b_meet)
        )
        blue = ((56, 130, 220), (20, 60, 120))
        orange = ((238, 132, 36), (150, 70, 10))
        if shared:
            mx, my = self.cell_center(b_meet)
            offset = self.cell_size * 0.22
            a_center = (mx - offset, my)
            b_center = (mx + offset, my)
            self.draw_circle_agent(draw, a_center, fill=blue[0], outline=blue[1], radius_factor=0.20, ring_color=None)
            self.draw_circle_agent(draw, b_center, fill=orange[0], outline=orange[1], radius_factor=0.20, ring_color=None)
            holder = a_center if ph == "leg1" else b_center
            self._draw_parcel(draw, (holder[0], holder[1] - self.cell_size * 0.26))
            return
        if ph == "leg1":
            # A moving with the parcel; B waits centred at the meeting cell.
            cx, cy = self.cell_center(carrier_pos)
            self.draw_circle_agent(draw, (cx, cy), fill=blue[0], outline=blue[1])
            self._draw_parcel(draw, (cx + self.cell_size * 0.18, cy - self.cell_size * 0.18))
            cx2, cy2 = self.cell_center(b_meet)
            self.draw_circle_agent(draw, (cx2, cy2), fill=orange[0], outline=orange[1])
        else:
            # B moving with the parcel; A stays centred at the meeting cell.
            cx, cy = self.cell_center(b_meet)
            self.draw_circle_agent(draw, (cx, cy), fill=blue[0], outline=blue[1])
            cx2, cy2 = self.cell_center(carrier_pos)
            self.draw_circle_agent(draw, (cx2, cy2), fill=orange[0], outline=orange[1])
            self._draw_parcel(draw, (cx2 + self.cell_size * 0.18, cy2 - self.cell_size * 0.18))

    def _draw_parcel(self, draw, xy):
        cx, cy = xy
        s = self.cell_size * 0.16
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=(165, 110, 50), outline=(80, 50, 20), width=2)
        draw.line([(cx - s, cy), (cx + s, cy)], fill=(80, 50, 20), width=2)
        draw.line([(cx, cy - s), (cx, cy + s)], fill=(80, 50, 20), width=2)

    def build_video_frame_states(self, scenario, substeps):
        leg1 = scenario.extras["leg1"]
        leg2 = scenario.extras["leg2"]
        states: List[FrameState] = []
        for cell in leg1:
            states.append(FrameState(primary_pos=cell, primary_cell=cell, extras={"phase": "leg1"}))
        # Handoff is a discrete contact update between two stationary holds.
        states.append(FrameState(primary_pos=leg2[0], primary_cell=leg2[0], extras={"phase": "leg2"}))
        for cell in leg2[1:]:
            states.append(FrameState(primary_pos=cell, primary_cell=cell, extras={"phase": "leg2"}))
        return states
