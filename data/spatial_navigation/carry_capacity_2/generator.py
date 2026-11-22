"""Carry capacity = 2: agent picks up items + drops them at delivery cell.

There are 4 colored cargo cells C1-C4 and one delivery zone Z. The agent
can hold at most 2 items at once. The puzzle: pick up all 4 items and drop
them at Z. The agent must make a round-trip (pick 2 → deliver → pick 2 →
deliver → goal). Mechanic load-bearing: ignoring capacity gives a much
shorter (incorrect) path; respecting capacity forces 2 round trips.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    GRID_PATH_STAY_ENCODING_SENTENCE,
    cell_to_xy,
)
from data.spatial_navigation.core.maze_style import route_run_breakdown
from data.spatial_navigation.core.mechanic_base import (
    MechanicGridGeneratorBase,
    MechanicScenario,
)

CRATE_FILL = (170, 120, 75)
CRATE_EDGE = (110, 70, 30)
CRATE_PLANK = (85, 55, 20)
CAPACITY = 2


def _bfs_path(size: int, start: Tuple[int, int], goal: Tuple[int, int], walls: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    if start == goal:
        return [start]
    queue = deque([start])
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {start: start}
    while queue:
        cell = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in walls or nxt in parent:
                continue
            parent[nxt] = cell
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(nxt)
    return None


def _capacity_route(
    size: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    zone: Tuple[int, int],
    cargo: Sequence[Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    *,
    capacity: int,
    delivery_budget: int,
) -> Optional[List[Tuple[int, int]]]:
    """Shortest route in the actual pickup/capacity/drop state space."""

    cargo_index = {cell: index for index, cell in enumerate(cargo)}
    full_mask = (1 << len(cargo)) - 1
    initial = (start, full_mask, 0, 0, 0)
    queue = deque([initial])
    parent: Dict[tuple, Optional[tuple]] = {initial: None}
    while queue:
        state = queue.popleft()
        cell, remaining, held, delivered, delivery_commits = state
        if cell == goal and delivered == full_mask:
            states = [state]
            while parent[states[-1]] is not None:
                states.append(parent[states[-1]])
            return [item[0] for item in reversed(states)]
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
                continue
            next_remaining = int(remaining)
            next_held = int(held)
            next_delivered = int(delivered)
            next_delivery_commits = int(delivery_commits)
            cargo_idx = cargo_index.get(nxt)
            if (
                cargo_idx is not None
                and next_remaining & (1 << cargo_idx)
                and bin(next_held).count("1") < capacity
            ):
                next_remaining &= ~(1 << cargo_idx)
                next_held |= 1 << cargo_idx
            if nxt == zone and next_held:
                if next_delivery_commits >= delivery_budget:
                    continue
                next_delivered |= next_held
                next_held = 0
                next_delivery_commits += 1
            next_state = (
                nxt,
                next_remaining,
                next_held,
                next_delivered,
                next_delivery_commits,
            )
            if next_state in parent:
                continue
            parent[next_state] = state
            queue.append(next_state)
    return None


def _carry_trace(
    path: Sequence[Tuple[int, int]],
    cargo: Sequence[Tuple[int, int]],
    zone: Tuple[int, int],
    *,
    capacity: int = CAPACITY,
) -> List[Dict[str, Any]]:
    """Replay automatic pickup/drop along ``path`` under the carry capacity. An open-coded copy in the video path once omitted the capacity guard and made a crate vanish from its cell mid-route."""

    present = [True] * len(cargo)
    held = 0
    delivered = 0
    commits = 0
    trace: List[Dict[str, Any]] = []
    for step, cell in enumerate(path):
        picked: Optional[int] = None
        if held < capacity:
            for index, crate in enumerate(cargo):
                if present[index] and tuple(crate) == tuple(cell):
                    present[index] = False
                    held += 1
                    picked = index
                    break
        dropped = 0
        if tuple(cell) == tuple(zone) and held:
            dropped = held
            delivered += held
            held = 0
            commits += 1
        trace.append(
            {
                "step": step,
                "cell": cell,
                "held": held,
                "delivered": delivered,
                "delivery_commits": commits,
                "cargo_present": list(present),
                "picked_index": picked,
                "dropped": dropped,
            }
        )
    return trace


def _delivery_batches(
    path: Sequence[Tuple[int, int]],
    cargo: Sequence[Tuple[int, int]],
    zone: Tuple[int, int],
    *,
    capacity: int,
) -> List[int]:
    """Each non-empty Z delivery, in order."""

    return [
        int(state["dropped"])
        for state in _carry_trace(path, cargo, zone, capacity=capacity)
        if state["dropped"]
    ]


class CarryCapacity2Generator(MechanicGridGeneratorBase):
    TASK_NAME = "carry_capacity_2"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Carry Capacity 2"
    DEFAULT_SIZE = 7
    DELIVERY_BUDGET = 2

    def __init__(self, *args, route_steps=(10, 32), **kwargs):
        self.route_steps = tuple(route_steps)
        super().__init__(*args, **kwargs)

    def sample_scenario(self) -> MechanicScenario:
        for _ in range(512):
            cells = [(r, c) for r in range(self.size) for c in range(self.size)]
            self.rng.shuffle(cells)
            start, goal = cells[0], cells[1]
            zone = cells[2]
            cargo = cells[3:7]
            named = {start, goal, zone} | set(cargo)
            if len(named) < 7:
                continue
            extras = [c for c in cells[7:] if c not in named]
            wall_count = self.rng.randint(self.size - 1, self.size + 1)
            walls: Set[Tuple[int, int]] = set()
            for cand in extras:
                if len(walls) >= wall_count:
                    break
                walls.add(cand)

            best_path = _capacity_route(
                self.size,
                start,
                goal,
                zone,
                cargo,
                walls,
                capacity=2,
                delivery_budget=self.DELIVERY_BUDGET,
            )
            relaxed_path = _capacity_route(
                self.size,
                start,
                goal,
                zone,
                cargo,
                walls,
                capacity=4,
                delivery_budget=self.DELIVERY_BUDGET,
            )
            if best_path is None or relaxed_path is None:
                continue
            capacity_one_path = _capacity_route(
                self.size,
                start,
                goal,
                zone,
                cargo,
                walls,
                capacity=1,
                delivery_budget=self.DELIVERY_BUDGET,
            )
            if capacity_one_path is not None:
                continue
            # The named mechanic is two full capacity-2 trips, not an
            # incidental capacity upper bound that a shortest route may use
            # as 1+1+2 deliveries.  Make that semantic promise true.
            if _delivery_batches(best_path, cargo, zone, capacity=2) != [2, 2]:
                continue
            if not self.route_steps[0] <= len(best_path) - 1 <= self.route_steps[1]:
                continue
            if len(relaxed_path) >= len(best_path):
                continue
            return self._build_scenario(
                start=start,
                goal=goal,
                zone=zone,
                cargo=list(cargo),
                walls=sorted(walls),
                path=best_path,
            )
        raise RuntimeError("carry_capacity_2: failed to sample scenario.")

    def _authored_reasoning(self, scenario: MechanicScenario) -> str:
        path = list(scenario.agent_path)
        cargo = [tuple(c) for c in scenario.extras["cargo_cells"]]
        zone = tuple(scenario.extras["zone_cell"])
        events: List[Tuple[int, str]] = []
        for state in _carry_trace(path, cargo, zone):
            step = int(state["step"])
            if state["picked_index"] is not None:
                events.append((
                    step,
                    f"pick up the crate at {cell_to_xy(state['cell'])}, "
                    f"now holding {state['held']} of {CAPACITY}",
                ))
            if state["dropped"] and step > 0:
                events.append(
                    (step, f"drop both crates at Z, bringing the delivered stack to {state['delivered']}")
                    if state["dropped"] == 2
                    else (step, f"drop the held crate at Z, bringing the delivered stack to {state['delivered']}")
                )
        sentences: List[str] = [
            f"S is at {cell_to_xy(path[0])}, G at {cell_to_xy(path[-1])}, the delivery zone Z at "
            f"{cell_to_xy(zone)}, and the four crates sit at "
            f"{', '.join(str(cell_to_xy(c)) for c in cargo)}. The agent carries at most two crates and Z accepts "
            "only two drops, so the crates must go over in two full pairs: collect two, unload at Z, collect the "
            "other two, unload again, then finish on G.",
        ]
        anchor = 0
        for step, event in events:
            runs = route_run_breakdown(path[anchor : step + 1])
            if runs:
                leg = ", then ".join(
                    f"{direction} {len(cells) - 1}" for direction, cells in runs
                )
                sentences.append(f"It walks {leg} to {cell_to_xy(path[step])} to {event}.")
            else:
                sentences.append(f"Standing at {cell_to_xy(path[step])}, it can {event}.")
            anchor = step
        tail_runs = route_run_breakdown(path[anchor:])
        if tail_runs:
            leg = ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in tail_runs)
            sentences.append(f"With all four crates stacked at Z, it walks {leg} and stops on G.")
        else:
            sentences.append("With all four crates stacked at Z, it is already standing on G.")
        sentences.append(
            "Both drops were full pairs, so the two-drop budget and the two-crate capacity are both respected."
        )
        return " ".join(sentences)

    def _build_scenario(self, *, start, goal, zone, cargo, walls, path):
        return MechanicScenario(
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            title=self.TASK_TITLE,
            input_desc=(
                f"The image shows a {self.size}x{self.size} grid with row numbers down the left edge and column "
                "numbers across the top edge. Green S is the start, blue G is the goal, the orange-bordered tile "
                "marked Z in its corner is the delivery zone, four identical brown crates sit on their own cells, "
                f"and dark slate cells are walls. {GRID_XY_CONVENTION_SENTENCE}"
            ),
            rules_desc=(
                "The agent moves up, right, down, or left between adjacent floor cells, and it may also stay on its "
                f"current resting cell for one step. {GRID_PATH_STAY_ENCODING_SENTENCE} "
                "The agent can hold at most two crates at a time. Stepping onto a crate's cell picks it up; "
                "stepping onto Z while holding crates automatically drops every held crate there. "
                "At most two nonempty drops may be committed at Z, so each legal solution must deliver two crates "
                "on the first drop and the other two on the second. All four crates must be delivered before "
                "the agent ends on G."
            ),
            ti2v_output=(
                "Generate a video that shows the agent following a complete legal route. When it picks a crate up, "
                "the crate leaves its cell and rides at the bottom of the agent's cell, moving with every step; when "
                "the agent enters Z, every riding crate transfers into Z's stack in the same commit. End with all "
                "four crates stacked in Z and the agent on G."
            ),
            solution_image_output=(
                "Render the solved still by drawing the complete legal route in red, all four crates "
                "stacked inside Z, and the agent on G."
            ),
            size=self.size,
            start=start,
            goal=goal,
            agent_path=path,
            walls=list(walls),
            extras={
                "zone_cell": [zone[0], zone[1]],
                "cargo_cells": [[r, c] for r, c in cargo],
                "delivery_budget": self.DELIVERY_BUDGET,
            },
            structured_answer={
                # The formal answer is the 1-based (x, y) route; the linear cell
                # ids stay only for the shared spatial_navigation replay/scan fields that
                # still consume them.
                "agent_path_xy": [list(cell_to_xy(cell)) for cell in path],
                "agent_path_cell_ids": self._path_cell_ids(path),
                "zone_xy": list(cell_to_xy(zone)),
                "cargo_xy": [list(cell_to_xy(c)) for c in cargo],
            },
        )

    def _draw_crate(
        self,
        draw,
        center: Tuple[float, float],
        side: float,
    ) -> None:
        """One anonymous brown crate glyph, reused at every scale.

        The crate has a single stable identity encoding (brown box with cross
        planks); no per-crate label exists because the cargo items are
        interchangeable and the answer never references an individual crate.
        """

        cx, cy = center
        half = side / 2.0
        draw.rounded_rectangle(
            [cx - half, cy - half, cx + half, cy + half],
            radius=max(2, int(side * 0.12)),
            fill=CRATE_FILL,
            outline=CRATE_EDGE,
            width=2,
        )
        pad = max(2, side * 0.16)
        draw.line([(cx - half + pad, cy), (cx + half - pad, cy)], fill=CRATE_PLANK, width=2)
        draw.line([(cx, cy - half + pad), (cx, cy + half - pad)], fill=CRATE_PLANK, width=2)

    def _render(
        self,
        scenario: MechanicScenario,
        *,
        agent_center: Tuple[float, float],
        held: int,
        delivered: int,
        delivery_commits: int,
        cargo_present: Sequence[bool],
        show_path: bool = False,
    ) -> Image.Image:
        image, draw = self._begin_scene()
        wall_set = set(scenario.walls)
        self._draw_grid_and_labels(draw, wall_set=wall_set)
        token_font = self._load_font(max(11, self.cell_size // 6))

        if show_path:
            self._draw_path_overlay(draw, scenario.agent_path)

        # Delivery zone: corner-seated Z letter, delivered crates stack into
        # four fixed 2x2 slots in delivery order — the crate count in the zone
        # is the only delivered-state encoding.
        zone = tuple(scenario.extras["zone_cell"])
        zl, zt, zr, zb = self._cell_box(zone)
        draw.rounded_rectangle([zl + 4, zt + 4, zr - 4, zb - 4], radius=8, fill=(255, 230, 180), outline=(220, 130, 50), width=3)
        draw.text(
            (zl + 4 + max(6, self.cell_size // 9), zt + 4 + max(6, self.cell_size // 9)),
            "Z",
            fill=(60, 30, 0),
            font=token_font,
            anchor="mm",
        )
        zcx, zcy = self._cell_center(zone)
        slot = self.cell_size * 0.24
        offset = self.cell_size * 0.16
        slot_centers = (
            (zcx - offset, zcy - offset + slot * 0.2),
            (zcx + offset, zcy - offset + slot * 0.2),
            (zcx - offset, zcy + offset + slot * 0.2),
            (zcx + offset, zcy + offset + slot * 0.2),
        )
        for index in range(max(0, min(int(delivered), 4))):
            self._draw_crate(draw, slot_centers[index], slot)

        # cargo still waiting on its source cells
        for i, c in enumerate(scenario.extras["cargo_cells"]):
            if not cargo_present[i]:
                continue
            cell = tuple(c)
            self._draw_crate(draw, self._cell_center(cell), self.cell_size - 2 * max(8, self.cell_size // 6))

        self._draw_start_goal(draw, scenario)
        self._draw_agent(draw, agent_center)
        # Held crates ride with the agent: up to two fixed slots at the bottom
        # of the agent's cell, moving with each discrete step.
        if held > 0:
            cx, cy = agent_center
            ride = self.cell_size * 0.22
            ride_y = cy + self.cell_size * 0.30
            ride_offsets = (-self.cell_size * 0.16, self.cell_size * 0.16)
            if int(held) > CAPACITY:
                # Clamping here is what let an over-capacity state render as a
                # legal-looking frame while the extra crate vanished from the
                # board.  Refuse the frame instead of hiding the state.
                raise AssertionError(
                    f"carry_capacity_2: {held} held crates exceeds capacity {CAPACITY}"
                )
            for index in range(int(held)):
                self._draw_crate(draw, (cx + ride_offsets[index], ride_y), ride)
        return self._finish_scene(image)

    def create_puzzle(self) -> Dict[str, Any]:  # type: ignore[override]
        scenario = self.sample_scenario()
        cargo_cells = [tuple(c) for c in scenario.extras["cargo_cells"]]
        zone = tuple(scenario.extras["zone_cell"])

        states = _carry_trace(scenario.agent_path, cargo_cells, zone)

        puzzle_image = self._render(
            scenario, agent_center=self._cell_center(scenario.start),
            held=0, delivered=0, delivery_commits=0, cargo_present=[True] * 4,
        )
        solution_image = self._render(
            scenario, agent_center=self._cell_center(scenario.goal),
            held=0,
            delivered=4,
            delivery_commits=self.DELIVERY_BUDGET,
            cargo_present=[False] * 4,
            show_path=True,
        )

        def render_frame(agent_cell, agent_center, step_index):
            state = states[min(step_index, len(states) - 1)]
            return self._render(
                scenario,
                agent_center=agent_center,
                held=int(state["held"]),
                delivered=int(state["delivered"]),
                delivery_commits=int(state["delivery_commits"]),
                cargo_present=list(state["cargo_present"]),
            )

        video_frames = self._animate_path(scenario, render_frame)
        return self._assemble_record(
            scenario, puzzle_image=puzzle_image, solution_image=solution_image, video_frames=video_frames,
        )


__all__ = ["CarryCapacity2Generator"]
