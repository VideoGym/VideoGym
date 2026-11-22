"""Pickup K parcels at K stations, deliver each parcel to its labelled destination — open TSP-like sequencing."""

from __future__ import annotations

from collections import deque
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
    cells_to_moves,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path


Coord = Tuple[int, int]
DeliveryState = Tuple[Coord, int, int, int]


def _shortest_delivery_path(
    start: Coord,
    pickups: Sequence[Coord],
    destinations: Sequence[Coord],
    size: int,
    walls: Sequence[Coord],
) -> Optional[List[Coord]]:
    """Solve the actual automatic-pickup/capacity-one state space.

    A shortest geometric leg may cross another still-present pickup.  Treating
    a requested permutation as if that crossing had no effect can author an
    impossible delivery trace, so the search state explicitly carries parcel
    presence, the held parcel, and completed deliveries.
    """

    if len(pickups) != len(destinations) or not pickups:
        return None
    pickup_at = {tuple(cell): index for index, cell in enumerate(pickups)}
    wall_set = {tuple(cell) for cell in walls}
    all_mask = (1 << len(pickups)) - 1
    initial: DeliveryState = (tuple(start), all_mask, -1, 0)
    queue = deque([initial])
    parent: dict[DeliveryState, Optional[DeliveryState]] = {initial: None}
    goal_state: Optional[DeliveryState] = None

    while queue:
        state = queue.popleft()
        cell, remaining, held, delivered = state
        if delivered == all_mask and held == -1:
            goal_state = state
            break
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            neighbor = cell[0] + dr, cell[1] + dc
            if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
                continue
            if neighbor in wall_set:
                continue

            next_remaining = remaining
            next_held = held
            next_delivered = delivered
            if next_held == -1:
                pickup_index = pickup_at.get(neighbor)
                if pickup_index is not None and next_remaining & (1 << pickup_index):
                    next_remaining &= ~(1 << pickup_index)
                    next_held = pickup_index
            if next_held >= 0 and neighbor == tuple(destinations[next_held]):
                next_delivered |= 1 << next_held
                next_held = -1

            next_state: DeliveryState = (
                neighbor,
                next_remaining,
                next_held,
                next_delivered,
            )
            if next_state in parent:
                continue
            parent[next_state] = state
            queue.append(next_state)

    if goal_state is None:
        return None
    path: List[Coord] = []
    cursor: Optional[DeliveryState] = goal_state
    while cursor is not None:
        path.append(cursor[0])
        cursor = parent[cursor]
    return list(reversed(path))


class ParcelDeliveryRouteGenerator(MultiAgentSpatialNavigationBase):
    TASK_NAME = "parcel_delivery_route"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Parcel Delivery Route"
    DEFAULT_SIZE = 8

    INPUT_DESC = (
        "The board has a blue agent and several pickup-destination pairs. Each pickup is a brown parcel box "
        "labelled with a letter, and each destination is a coloured target square with the matching letter. "
        "Dark cells are walls."
    )
    RULES_DESC = (
        "Pick up each parcel by stepping on its box, then deliver it by stepping on the matching destination "
        "before picking up the next parcel. The agent may carry only one parcel at a time. Each step may move up, "
        "right, down, or left, or use stay to remain on the current cell. Complete all deliveries in any legal "
        f"pickup-then-deliver order; route length is not scored. {GRID_PATH_STAY_ENCODING_SENTENCE}"
    )
    TI2V_OUTPUT = "Generate a short video showing the agent picking up and delivering each parcel in any legal order."
    SOLUTION_IMAGE_OUTPUT = "Render the solved still image with the full delivery route drawn in blue."

    @staticmethod
    def _pos(cell) -> str:
        return format_xy_position((cell[0] + 1, cell[1] + 1))

    def _authored_reasoning(self, scenario: MultiAgentScenario) -> str:
        from data.spatial_navigation.core.maze_style import route_run_breakdown

        path = [tuple(c) for c in scenario.primary_path]
        pickups = [tuple(c) for c in scenario.extras["pickups"]]
        dests = [tuple(c) for c in scenario.extras["dests"]]
        labels = "ABCD"
        pair_text = "; ".join(
            f"parcel {labels[i]} at {self._pos(p)} goes to target {labels[i]} at {self._pos(d)}"
            for i, (p, d) in enumerate(zip(pickups, dests))
        )
        sentences = [
            f"The agent starts at {self._pos(path[0])} with empty hands, and each labelled box must reach its "
            f"matching target: {pair_text}. Only one parcel can be carried at a time, so each box must be walked "
            "straight to its own target before touching the next one.",
        ]
        present = [True] * len(pickups)
        carrying = None
        anchor = 0
        for step, cell in enumerate(path):
            if step == 0:
                continue
            if carrying is None and cell in pickups:
                index = pickups.index(cell)
                if present[index]:
                    present[index] = False
                    carrying = index
                    runs = route_run_breakdown(path[anchor : step + 1])
                    leg = ", then ".join(f"{d} {len(r) - 1}" for d, r in runs)
                    sentences.append(f"It walks {leg} and picks up parcel {labels[index]}.")
                    anchor = step
                    continue
            if carrying is not None and cell == dests[carrying]:
                runs = route_run_breakdown(path[anchor : step + 1])
                leg = ", then ".join(f"{d} {len(r) - 1}" for d, r in runs)
                sentences.append(
                    f"Carrying it, the agent walks {leg} and sets parcel {labels[carrying]} down on its matching "
                    "target."
                )
                carrying = None
                anchor = step
        sentences.append(
            "Every labelled parcel now sits on its own target and the agent finished empty-handed, so the "
            "delivery order was legal throughout."
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
            agent_start = free[0]
            num_pairs = 2
            pickups = []
            dests = []
            used = {agent_start}
            for k in range(num_pairs):
                p = next((c for c in free if c not in used), None)
                if p is None:
                    break
                used.add(p)
                d = next((c for c in free if c not in used), None)
                if d is None:
                    break
                used.add(d)
                pickups.append(p)
                dests.append(d)
            if len(pickups) != num_pairs:
                continue
            best_cells = _shortest_delivery_path(
                agent_start,
                pickups,
                dests,
                size,
                walls,
            )
            if best_cells is None:
                continue
            if not (6 <= len(best_cells) - 1 <= 18):
                continue
            scenario = MultiAgentScenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                size=size,
                walls=walls,
                primary_path=best_cells,
                primary_moves=cells_to_moves(best_cells),
                extras={"agent_start": agent_start, "pickups": pickups, "dests": dests},
            )
            scenario.answer_text = format_xy_path_answer(scenario.primary_path)
            return scenario
        raise RuntimeError("parcel_delivery_route sample failed")

    def render_static_layer(self, draw, image, scenario):
        labels = ["A", "B", "C", "D"]
        colors = [(220, 90, 90), (70, 150, 220), (200, 160, 50), (130, 90, 200)]
        for i, (pickup, dest) in enumerate(zip(scenario.extras["pickups"], scenario.extras["dests"])):
            # destination square
            l2, t2, r2, b2 = self.cell_box(dest)
            color = colors[i]
            draw.rectangle([l2 + 5, t2 + 5, r2 - 5, b2 - 5], fill=tuple(min(255, c + 80) for c in color), outline=color, width=3)
            cx2, cy2 = self.cell_center(dest)
            draw.text((cx2, cy2), labels[i], fill=tuple(max(0, c - 60) for c in color), font=self._font_big, anchor="mm")

    def _draw_parcel(self, draw, center, index, *, scale=1.0, show_label=True):
        labels = ["A", "B", "C", "D"]
        cx, cy = center
        half = self.cell_size * 0.30 * scale
        draw.rounded_rectangle(
            [cx - half, cy - half, cx + half, cy + half],
            radius=max(3, int(4 * scale)),
            fill=(165, 110, 50),
            outline=(80, 50, 20),
            width=2,
        )
        draw.line([(cx - half, cy), (cx + half, cy)], fill=(80, 50, 20), width=2)
        draw.line([(cx, cy - half), (cx, cy + half)], fill=(80, 50, 20), width=2)
        if show_label:
            draw.text((cx, cy), labels[index], fill=(255, 255, 255), font=self._font_id, anchor="mm")

    def render_dynamic_layer(self, draw, image, scenario, frame_state):
        present = list(frame_state.extras.get("parcel_present", [True] * len(scenario.extras["pickups"])))
        delivered = set(int(index) for index in frame_state.extras.get("delivered", []))
        for index, pickup in enumerate(scenario.extras["pickups"]):
            if index < len(present) and present[index]:
                self._draw_parcel(draw, self.cell_center(pickup), index)
        for index in delivered:
            if 0 <= index < len(scenario.extras["dests"]):
                cx, cy = self.cell_center(scenario.extras["dests"][index])
                # The target's centered A/B label already identifies this
                # parcel. Show a small persistent box in the corner instead
                # of drawing the same letter twice on one pixel region.
                self._draw_parcel(
                    draw,
                    (cx + self.cell_size * 0.23, cy - self.cell_size * 0.23),
                    index,
                    scale=0.42,
                    show_label=False,
                )

    def render_solution_overlay(self, draw, image, scenario):
        if len(scenario.primary_path) >= 2:
            pts = [self.cell_center(c) for c in scenario.primary_path]
            draw_spatial_navigation_path(draw, pts, color=(56, 130, 220), width=max(5, self.cell_size // 7))

    def render_path_overlay(self, draw, path_cells, color=(214, 42, 42)) -> None:
        return

    def render_agents(self, draw, image, scenario, frame_state):
        cx, cy = self.cell_center(frame_state.primary_pos)
        self.draw_circle_agent(draw, (cx, cy), fill=(56, 130, 220), outline=(20, 60, 120))
        carrying = frame_state.extras.get("carrying")
        if isinstance(carrying, int):
            self._draw_parcel(
                draw,
                (cx + self.cell_size * 0.23, cy - self.cell_size * 0.23),
                carrying,
                scale=0.42,
            )

    def _discrete_delivery_states(self, scenario):
        pickups = [tuple(cell) for cell in scenario.extras["pickups"]]
        destinations = [tuple(cell) for cell in scenario.extras["dests"]]
        present = [True] * len(pickups)
        delivered = set()
        carrying = None
        states = []
        for cell in scenario.primary_path:
            if carrying is None and cell in pickups:
                index = pickups.index(cell)
                if present[index] and index not in delivered:
                    present[index] = False
                    carrying = index
            if isinstance(carrying, int) and cell == destinations[carrying]:
                delivered.add(carrying)
                carrying = None
            states.append(
                {
                    "parcel_present": list(present),
                    "delivered": sorted(delivered),
                    "carrying": carrying,
                }
            )
        return states

    def build_video_frame_states(self, scenario, substeps):
        path = list(scenario.primary_path)
        if not path:
            return []
        discrete = self._discrete_delivery_states(scenario)
        return [
            FrameState(primary_pos=cell, primary_cell=cell, extras=dict(extra))
            for cell, extra in zip(path, discrete)
        ]
