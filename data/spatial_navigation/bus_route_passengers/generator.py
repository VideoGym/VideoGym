"""Bus picks up passengers at stops in numbered order then drops them at the depot."""

from __future__ import annotations

from collections import deque

from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
    format_xy_path_answer,
    format_xy_position,
)
from data.spatial_navigation.core.multi_agent_base import (
    DISCRETE_STATE_HOLD_FRAMES,
    FrameState,
    MultiAgentSpatialNavigationBase,
    MultiAgentScenario,
    cells_to_moves,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path


def _ordered_bus_path(start, stops, depot, size, walls):
    """Find a route that cannot cross a later passenger stop prematurely."""

    targets = tuple(stops) + (depot,)
    stop_index = {cell: index for index, cell in enumerate(stops)}
    initial = (start, 0)
    queue = deque([initial])
    parent = {initial: None}
    terminal = None
    while queue:
        cell, progress = queue.popleft()
        if cell == depot and progress == len(targets):
            terminal = (cell, progress)
            break
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nxt = cell[0] + dr, cell[1] + dc
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
                continue
            next_progress = int(progress)
            visited_stop = stop_index.get(nxt)
            if visited_stop is not None and visited_stop > next_progress:
                continue
            if next_progress < len(targets) and nxt == targets[next_progress]:
                next_progress += 1
            state = (nxt, next_progress)
            if state in parent:
                continue
            parent[state] = (cell, progress)
            queue.append(state)
    if terminal is None:
        return None
    states = [terminal]
    while parent[states[-1]] is not None:
        states.append(parent[states[-1]])
    return [state[0] for state in reversed(states)]


class BusRoutePassengersGenerator(MultiAgentSpatialNavigationBase):
    TASK_NAME = "bus_route_passengers"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Bus Route Passengers"
    DEFAULT_SIZE = 8

    INPUT_DESC = (
        "The board has a yellow bus, three numbered stops 1, 2, 3, one waiting passenger marker at each stop, "
        "and a depot square marked DEP. Dark cells are walls. A stop's number remains visible after pickup."
    )
    RULES_DESC = (
        "Move one cell up, right, down, or left through non-wall cells; staying in place is also legal. "
        f"{GRID_PATH_STAY_ENCODING_SENTENCE} The bus "
        "must pick up the waiting passengers at stops 1, 2, and 3 in that order, then end at DEP. On pickup, that "
        "stop's passenger marker disappears permanently and the same passenger rides visibly on the bus. "
        "Entering stop 2 before stop 1, or stop 3 before stops 1 and 2, is illegal because it would pick up a "
        "passenger out of order. "
        "When the bus reaches DEP with all three aboard, the riding passengers leave the bus and disappear into "
        "the depot, and no passenger remains waiting at any stop. Any complete legal route is valid."
    )
    TI2V_OUTPUT = (
        "Generate a short video showing every legal bus move and pickup commit. Each reached stop must lose its "
        "waiting passenger marker while keeping its numbered stop badge, boarded passengers must ride visibly on "
        "the bus, and at the DEP arrival commit the riding passenger dots leave the bus into the depot, ending "
        "with a terminal hold of the empty bus at DEP and no passenger left at any stop."
    )
    SOLUTION_IMAGE_OUTPUT = (
        "Render the completed still with the full legal bus route drawn in blue, the empty bus at DEP after its "
        "passengers have entered the depot, and no waiting passenger marker remaining at stops 1, 2, or 3."
    )

    PASSENGER_COLORS = (
        (210, 70, 82),
        (225, 146, 38),
        (64, 154, 92),
    )

    def build_video_frame_states(self, scenario: MultiAgentScenario, substeps: int) -> list[FrameState]:
        del substeps
        states: list[FrameState] = []
        picked_count = 0
        stops = [tuple(scenario.extras[f"stop{i}"]) for i in (1, 2, 3)]
        depot = tuple(scenario.extras["depot"])
        for cell in scenario.primary_path:
            if picked_count < 3 and tuple(cell) == stops[picked_count]:
                picked_count += 1
            delivered = picked_count == 3 and tuple(cell) == depot
            self._append_state_hold(
                states,
                FrameState(
                    primary_pos=tuple(cell),
                    primary_cell=tuple(cell),
                    extras={
                        "picked_count": picked_count,
                        "onboard_count": 0 if delivered else picked_count,
                        "delivered_count": 3 if delivered else 0,
                    },
                ),
                DISCRETE_STATE_HOLD_FRAMES,
            )
        return states


    @staticmethod
    def _pos(cell) -> str:
        return format_xy_position((cell[0] + 1, cell[1] + 1))

    def _authored_reasoning(self, scenario: MultiAgentScenario) -> str:
        from data.spatial_navigation.core.maze_style import route_run_breakdown

        path = [tuple(c) for c in scenario.primary_path]
        stops = [tuple(scenario.extras[f"stop{i}"]) for i in (1, 2, 3)]
        depot = tuple(scenario.extras["depot"])
        sentences = [
            f"The bus starts at {self._pos(path[0])} and must board the waiting passengers strictly in stop order: "
            f"stop 1 at {self._pos(stops[0])}, stop 2 at {self._pos(stops[1])}, stop 3 at {self._pos(stops[2])}, "
            f"and finish at the depot at {self._pos(depot)}. Because a later stop may not be entered early, the "
            "route has to steer around any later stop that sits in the way.",
        ]
        anchors = [0]
        targets = stops + [depot]
        remaining = list(targets)
        for step, cell in enumerate(path):
            if remaining and cell == remaining[0]:
                anchors.append(step)
                remaining.pop(0)
        for index in range(1, len(anchors)):
            leg_cells = path[anchors[index - 1] : anchors[index] + 1]
            runs = route_run_breakdown(leg_cells)
            leg = ", then ".join(f"{direction} {len(run) - 1}" for direction, run in runs) if runs else "nowhere"
            target = targets[index - 1]
            if target == depot:
                sentences.append(
                    f"With all three aboard it drives {leg} to the depot, where the passengers step off into the "
                    "building and the bus rests empty."
                )
            else:
                stop_number = index
                sentences.append(
                    f"It drives {leg} to stop {stop_number} at {self._pos(target)}, boarding passenger "
                    f"{stop_number} ({stop_number} of 3 aboard)."
                )
        sentences.append(
            "Each stop was first entered exactly in its numbered turn, so every pickup was legal and the depot "
            "arrival completed the run."
        )
        return " ".join(sentences)

    def sample_scenario(self) -> MultiAgentScenario:
        size = self.size
        for _ in range(300):
            cells_pool = [(r, c) for r in range(size) for c in range(size)]
            self.rng.shuffle(cells_pool)
            walls = cells_pool[:self.rng.randint(2, 4)]
            free = [c for c in cells_pool if c not in walls]
            if len(free) < 8:
                continue
            bus_start = free[0]
            stop1 = free[1]
            stop2 = free[2]
            stop3 = free[3]
            depot = free[4]
            checkpoints = [bus_start, stop1, stop2, stop3, depot]
            if len(set(checkpoints)) != 5:
                continue
            cells = _ordered_bus_path(
                bus_start,
                (stop1, stop2, stop3),
                depot,
                size,
                set(walls),
            )
            if cells is None:
                continue
            if not (8 <= len(cells) - 1 <= 22):
                continue
            scenario = MultiAgentScenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                size=size,
                walls=walls,
                primary_path=cells,
                primary_moves=cells_to_moves(cells),
                extras={"bus_start": bus_start, "stop1": stop1, "stop2": stop2, "stop3": stop3, "depot": depot},
            )
            scenario.answer_text = format_xy_path_answer(scenario.primary_path)
            return scenario
        raise RuntimeError("bus_route_passengers sample failed")

    def render_static_layer(self, draw, image, scenario):
        # Static stop platforms and number badges remain after passenger pickup.
        for label, key, color in (("1", "stop1", (220, 80, 80)), ("2", "stop2", (220, 160, 50)), ("3", "stop3", (90, 170, 90))):
            cell = scenario.extras[key]
            l, t, r, b = self.cell_box(cell)
            inset = self.cell_size * 0.08
            draw.rounded_rectangle(
                [l + inset, t + inset, r - inset, b - inset],
                radius=5,
                fill=tuple(min(255, c + 115) for c in color),
                outline=color,
                width=3,
            )
            badge_r = self.cell_size * 0.16
            badge_x = l + self.cell_size * 0.25
            badge_y = t + self.cell_size * 0.25
            draw.ellipse(
                [badge_x - badge_r, badge_y - badge_r, badge_x + badge_r, badge_y + badge_r],
                fill=(255, 255, 255),
                outline=color,
                width=2,
            )
            draw.text(
                (badge_x, badge_y),
                label,
                fill=tuple(max(0, c - 70) for c in color),
                font=self._font_id,
                anchor="mm",
            )
        # Depot
        depot = scenario.extras["depot"]
        l, t, r, b = self.cell_box(depot)
        draw.rectangle([l + 4, t + 4, r - 4, b - 4], fill=(180, 200, 230), outline=(60, 100, 170), width=3)
        cx, cy = self.cell_center(depot)
        draw.text((cx, t + self.cell_size * 0.23), "DEP", fill=(30, 60, 130), font=self._font_id, anchor="mm")

    def _draw_passenger(self, draw, center, color, *, scale=1.0):
        cx, cy = center
        head = self.cell_size * 0.055 * scale
        body = self.cell_size * 0.09 * scale
        outline = tuple(max(0, channel - 65) for channel in color)
        draw.ellipse([cx - head, cy - body - 2 * head, cx + head, cy - body], fill=color, outline=outline, width=1)
        draw.rounded_rectangle(
            [cx - body * 0.58, cy - body, cx + body * 0.58, cy + body],
            radius=max(1, int(2 * scale)),
            fill=color,
            outline=outline,
            width=1,
        )

    def render_dynamic_layer(self, draw, image, scenario, frame_state):
        picked_count = int(frame_state.extras.get("picked_count", 0))
        for index, key in enumerate(("stop1", "stop2", "stop3")):
            if index < picked_count:
                continue
            l, t, r, b = self.cell_box(scenario.extras[key])
            self._draw_passenger(
                draw,
                (l + self.cell_size * 0.70, t + self.cell_size * 0.63),
                self.PASSENGER_COLORS[index],
            )

    def render_solution_overlay(self, draw, image, scenario):
        if len(scenario.primary_path) >= 2:
            pts = [self.cell_center(c) for c in scenario.primary_path]
            draw_spatial_navigation_path(draw, pts, color=(56, 130, 220), width=max(5, self.cell_size // 7))

    def render_path_overlay(self, draw, path_cells, color=(214, 42, 42)) -> None:
        return

    def render_agents(self, draw, image, scenario, frame_state):
        cx, cy = self.cell_center(frame_state.primary_pos)
        # bus
        s = self.cell_size * 0.31
        draw.rounded_rectangle([cx - s, cy - s * 0.6, cx + s, cy + s * 0.5], radius=5,
                               fill=(245, 200, 60), outline=(160, 120, 30), width=2)
        # windows
        draw.rectangle([cx - s + 4, cy - s * 0.45, cx + s - 4, cy - s * 0.10], fill=(170, 220, 240), outline=(60, 100, 130), width=1)
        # divider
        draw.line([(cx, cy - s * 0.45), (cx, cy - s * 0.10)], fill=(60, 100, 130), width=1)
        onboard_count = int(frame_state.extras.get("onboard_count", 0))
        for index in range(onboard_count):
            px = cx - s * 0.62 + index * s * 0.62
            py = cy - s * 0.28
            radius = max(2, self.cell_size * 0.035)
            color = self.PASSENGER_COLORS[index]
            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=color,
                outline=tuple(max(0, channel - 65) for channel in color),
                width=1,
            )
        # wheels
        wr = self.cell_size * 0.08
        draw.ellipse([cx - s + 4, cy + s * 0.4, cx - s + 4 + 2 * wr, cy + s * 0.4 + 2 * wr],
                     fill=(40, 40, 40), outline=(20, 20, 20), width=1)
        draw.ellipse([cx + s - 4 - 2 * wr, cy + s * 0.4, cx + s - 4, cy + s * 0.4 + 2 * wr],
                     fill=(40, 40, 40), outline=(20, 20, 20), width=1)
        # Delivery has one encoding: at the depot commit the riding passenger
        # dots leave the bus (the passengers walk into the depot building), so
        # the terminal state is the empty bus at DEP with every stop cleared.
        # No extra standing-figure cluster is drawn under the bus.
