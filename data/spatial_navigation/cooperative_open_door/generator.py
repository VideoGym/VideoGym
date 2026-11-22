"""Two agents must stand on two plates simultaneously to open the door, then one agent passes through.

Plates are red/blue circles. Door is a horizontal beam. Goal is past the door.
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


class CooperativeOpenDoorGenerator(MultiAgentSpatialNavigationBase):
    TASK_NAME = "cooperative_open_door"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Cooperative Open Door"
    DEFAULT_SIZE = 8

    INPUT_DESC = (
        "The board has two agents (blue and orange), two plates (one blue ring, one orange ring), a row of "
        "wooden door cells separating two halves of the board, and a green flag goal in the lower half. "
        "Dark cells are walls."
    )
    RULES_DESC = (
        "Each synchronized step contains one action for each agent, chosen from up, right, down, left, or stay. "
        "The blue and orange action sequences must have the same length, and actions at the same index commit on "
        f"the same tick. {GRID_PATH_STAY_ENCODING_SENTENCE} "
        "Both agents must step on their same-coloured plate at the same step, which opens the door permanently. "
        "Once opened, each brown door cell turns into a normal white passable floor cell for the rest of the run. "
        "The orange agent then remains on its plate while the blue agent passes straight down through one opened "
        "door cell and reaches the green flag. The agents may neither share a cell nor exchange cells in one step."
    )
    TI2V_OUTPUT = (
        "Generate a short video showing the synchronised plate-press, the brown door cells changing into normal "
        "white floor cells, and the blue agent crossing straight downward through the opened door to the goal."
    )
    SOLUTION_IMAGE_OUTPUT = (
        "Render the solved still image with the opened door cells shown as normal white floor cells, the blue "
        "agent's full path drawn from start to goal in blue, and the orange agent's path to its plate in orange."
    )

    @staticmethod
    def _pos(cell) -> str:
        return format_xy_position((cell[0] + 1, cell[1] + 1))

    @staticmethod
    def _stay_prefix(moves) -> int:
        count = 0
        for move in moves:
            if move != "stay":
                break
            count += 1
        return count

    def _leg_text(self, cells) -> str:
        from data.spatial_navigation.core.maze_style import route_run_breakdown

        deduped = [cells[0]] + [cell for prev, cell in zip(cells, cells[1:]) if cell != prev]
        if len(deduped) < 2:
            return "stays in place"
        runs = route_run_breakdown(deduped)
        return "walks " + ", then ".join(f"{d} {len(r) - 1}" for d, r in runs)

    def _authored_reasoning_segments(self, scenario: MultiAgentScenario):
        blue = [tuple(c) for c in scenario.primary_path]
        orange = [tuple(c) for c in scenario.secondary_path]
        press_step = int(scenario.extras["press_step"])
        blue_plate = tuple(scenario.extras["blue_plate"])
        orange_plate = tuple(scenario.extras["orange_plate"])
        door_cells = [tuple(c) for c in scenario.extras["door_cells"]]
        goal = tuple(scenario.extras["goal"])
        cross_cell = tuple(scenario.extras["cross_cell"])
        blue_pad = self._stay_prefix(scenario.primary_moves)
        orange_pad = self._stay_prefix(scenario.secondary_moves)
        pad_note = ""
        if blue_pad:
            pad_note = f" Blue first waits {blue_pad} tick{'s' if blue_pad > 1 else ''} so both arrive together."
        elif orange_pad:
            pad_note = f" Orange first waits {orange_pad} tick{'s' if orange_pad > 1 else ''} so both arrive together."
        door_span = " and ".join(self._pos(cell) for cell in door_cells)
        seg1 = (
            f"The brown door cells at {door_span} seal the lower half, and they open only when blue stands on its "
            f"ring at {self._pos(blue_plate)} and orange stands on its ring at {self._pos(orange_plate)} on the "
            f"same tick — so the two approach legs must be paced together.{pad_note} Blue "
            f"{self._leg_text(blue[: press_step + 1])} to its plate while orange "
            f"{self._leg_text(orange[: press_step + 1])} to its own, and the synchronized press turns every door "
            "cell into plain floor."
        )
        seg2 = (
            f"With the door open, orange holds its plate while blue {self._leg_text(blue[press_step:blue.index(cross_cell) + 1])} "
            f"and passes into the opened doorway at {self._pos(cross_cell)}."
        )
        seg3 = (
            f"Blue then {self._leg_text(blue[blue.index(cross_cell):])} and stops on the flag at {self._pos(goal)}; "
            "the two never shared a cell or swapped across an edge, and orange is still holding its plate, so the "
            "synchronized plan is complete."
        )
        return [seg1, seg2, seg3]

    def sample_scenario(self) -> MultiAgentScenario:
        size = self.size
        for _ in range(300):
            # Door row in the middle
            door_row = size // 2
            # door span: 2 cells in the middle
            door_cells = [(door_row, c) for c in range(size // 2 - 1, size // 2 + 1)]
            wall_row_cells = [(door_row, c) for c in range(size) if (door_row, c) not in door_cells]
            cross_cell = self.rng.choice(door_cells)
            door_entry = (door_row - 1, cross_cell[1])
            door_exit = (door_row + 1, cross_cell[1])
            # blue start: top half, orange start: top half
            all_top_free = [(r, c) for r in range(door_row) for c in range(size)]
            all_bot_free = [(r, c) for r in range(door_row + 1, size) for c in range(size)]
            top_free = [c for c in all_top_free if c != door_entry]
            bot_free = [c for c in all_bot_free if c != door_exit]
            goal_candidates = [(r, cross_cell[1]) for r in range(door_exit[0] + 1, size)]
            self.rng.shuffle(top_free)
            self.rng.shuffle(bot_free)
            self.rng.shuffle(goal_candidates)
            if len(top_free) < 4 or not goal_candidates:
                continue
            blue_start = top_free[0]
            orange_start = top_free[1]
            blue_plate = top_free[2]
            orange_plate = top_free[3]
            goal = goal_candidates[0]
            goal_corridor = {(r, cross_cell[1]) for r in range(door_exit[0], goal[0] + 1)}
            # Wall set
            walls = set(wall_row_cells)
            # extra walls in top/bot half (avoid critical cells)
            critical = {blue_start, orange_start, blue_plate, orange_plate, goal, cross_cell, door_entry, door_exit}
            critical.update(goal_corridor)
            extras = []
            for c in all_top_free + all_bot_free:
                if c in critical:
                    continue
                if self.rng.random() < 0.10:
                    extras.append(c)
            walls.update(extras)
            walls_closed = set(walls) | set(door_cells)
            # blue path: start → blue_plate → door_entry → cross_cell → door_exit → goal
            l1 = bfs_path(blue_start, blue_plate, size, walls_closed)
            # Orange occupies its plate from the synchronized press onward, so
            # blue's continuation must route around that cell.
            l2 = bfs_path(blue_plate, door_entry, size, walls_closed | {orange_plate})
            l3 = bfs_path(door_exit, goal, size, walls)
            # orange path: start → orange_plate (then stay)
            o1 = bfs_path(orange_start, orange_plate, size, walls_closed)
            if any(x is None for x in (l1, l2, l3, o1)):
                continue
            # Synchronise: at the step blue enters cross_cell, orange must be on plate.
            # Easiest: pad blue's first leg + plate-press so blue at plate at step k, orange at plate at step k.
            # We simply require len(l1)-1 == len(o1)-1; if not, pad shorter with stays at the start.
            blue_to_plate = list(l1)
            orange_to_plate = list(o1)
            # pad whichever is shorter at the start with stays
            db = len(blue_to_plate)
            do = len(orange_to_plate)
            if db < do:
                pad = do - db
                blue_to_plate = [blue_to_plate[0]] * pad + blue_to_plate
            elif do < db:
                pad = db - do
                orange_to_plate = [orange_to_plate[0]] * pad + orange_to_plate
            # full blue path: blue_to_plate + l2[1:] + cross_cell + door_exit + l3[1:]
            blue_full = blue_to_plate + l2[1:] + [cross_cell, door_exit] + l3[1:]
            # orange stays after plate
            tail_steps = len(blue_full) - len(orange_to_plate)
            orange_full = orange_to_plate + [orange_to_plate[-1]] * max(0, tail_steps)
            if any(blue == orange for blue, orange in zip(blue_full, orange_full)):
                continue
            if any(
                blue_next == orange_prev and orange_next == blue_prev
                for blue_prev, blue_next, orange_prev, orange_next in zip(
                    blue_full, blue_full[1:], orange_full, orange_full[1:]
                )
            ):
                continue
            if not (5 <= len(blue_full) - 1 <= 16):
                continue
            if self._has_immediate_reverse(cells_to_moves(blue_full)):
                continue
            cross_step = blue_full.index(cross_cell)
            if blue_full[cross_step - 1:cross_step + 2] != [door_entry, cross_cell, door_exit]:
                continue
            scenario = MultiAgentScenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                size=size,
                walls=list(walls),
                primary_path=blue_full,
                secondary_path=orange_full,
                primary_moves=cells_to_moves(blue_full),
                secondary_moves=cells_to_moves(orange_full),
                extras={
                    "blue_start": blue_start, "orange_start": orange_start,
                    "blue_plate": blue_plate, "orange_plate": orange_plate,
                    "goal": goal, "door_cells": door_cells, "cross_cell": cross_cell,
                    "door_entry": door_entry, "door_exit": door_exit,
                    "press_step": len(blue_to_plate) - 1,
                    "cross_step": cross_step,
                },
            )
            scenario.answer_text = format_named_xy_paths_answer(
                {
                    "blue": scenario.primary_path,
                    "orange": scenario.secondary_path,
                }
            )
            return scenario
        raise RuntimeError("cooperative_open_door sample failed")

    @staticmethod
    def _has_immediate_reverse(moves: List[str]) -> bool:
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
        previous = None
        for move in moves:
            if move == "stay":
                continue
            if previous is not None and opposite[previous] == move:
                return True
            previous = move
        return False

    def render_static_layer(self, draw, image, scenario):
        # door cells: light wood
        for cell in scenario.extras["door_cells"]:
            l, t, r, b = self.cell_box(cell)
            draw.rectangle([l, t, r, b], fill=(170, 130, 70), outline=(120, 80, 30), width=2)
            # plank stripe
            draw.line([(l, (t + b) // 2), (r, (t + b) // 2)], fill=(120, 80, 30), width=1)
        # blue plate
        bp = scenario.extras["blue_plate"]
        cx, cy = self.cell_center(bp)
        rd = self.cell_size * 0.36
        draw.ellipse([cx - rd, cy - rd, cx + rd, cy + rd], outline=(40, 90, 200), width=4)
        draw.ellipse([cx - rd * 0.4, cy - rd * 0.4, cx + rd * 0.4, cy + rd * 0.4], fill=(180, 210, 240))
        # orange plate
        op = scenario.extras["orange_plate"]
        cx, cy = self.cell_center(op)
        draw.ellipse([cx - rd, cy - rd, cx + rd, cy + rd], outline=(200, 110, 30), width=4)
        draw.ellipse([cx - rd * 0.4, cy - rd * 0.4, cx + rd * 0.4, cy + rd * 0.4], fill=(245, 215, 170))
        # goal flag
        self.draw_goal_flag(draw, scenario.extras["goal"], fill=(50, 160, 70))

    def render_dynamic_layer(self, draw, image, scenario, frame_state):
        if not frame_state.extras.get("door_open"):
            return
        for cell in scenario.extras["door_cells"]:
            l, t, r, b = self.cell_box(cell)
            draw.rectangle([l, t, r, b], fill=(252, 252, 252), outline=(210, 210, 210), width=1)

    def render_solution_overlay(self, draw, image, scenario):
        if len(scenario.primary_path) >= 2:
            pts = [self.cell_center(c) for c in scenario.primary_path]
            draw_spatial_navigation_path(draw, pts, color=(56, 130, 220), width=max(5, self.cell_size // 7))
        if len(scenario.secondary_path) >= 2:
            pts = [self.cell_center(c) for c in scenario.secondary_path]
            draw_spatial_navigation_path(draw, pts, color=(238, 132, 36), width=max(5, self.cell_size // 7))

    def render_path_overlay(self, draw, path_cells, color=(214, 42, 42)) -> None:
        return

    def render_agents(self, draw, image, scenario, frame_state):
        cx, cy = self.cell_center(frame_state.primary_pos)
        self.draw_circle_agent(draw, (cx, cy), fill=(56, 130, 220), outline=(20, 60, 120))
        if frame_state.secondary_pos is not None:
            cx2, cy2 = self.cell_center(frame_state.secondary_pos)
            self.draw_circle_agent(draw, (cx2, cy2), fill=(238, 132, 36), outline=(150, 70, 10))

    def build_video_frame_states(self, scenario, substeps):
        bp = scenario.primary_path
        op = scenario.secondary_path
        states: List[FrameState] = []
        press_step = int(scenario.extras["press_step"])
        for index, (primary, secondary) in enumerate(zip(bp, op)):
            states.append(
                FrameState(
                    primary_pos=primary,
                    primary_cell=primary,
                    secondary_pos=secondary,
                    secondary_cell=secondary,
                    extras={"door_open": index >= press_step},
                )
            )
        return states
