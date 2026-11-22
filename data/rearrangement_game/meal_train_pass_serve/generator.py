"""Meal train pass serve — shortest mixed-direction service walk.

A round table has six to nine seats numbered clockwise from the top. A
platter starts on one visible seat and may move one adjacent seat clockwise
(CW) or counter-clockwise (CCW). Gold-marked seats are served automatically
the first time the platter reaches them. The task asks for the unique
shortest walk that serves every marked seat.

The seat count is part of the visible input and is sampled per instance:
a six-seat table admits only four-pass routes, while seven, eight and nine
seats open five-, six- and seven-pass ones, so the route length is read off
this table rather than memorised.

Generated instances are deliberately non-degenerate: the unique shortest
walk must use both directions and must be shorter than either pure-CW or
pure-CCW traversal. Thus the demand pattern changes the route instead of
merely decorating an always-one-direction answer.

The solver uses clockwise/counter-clockwise moves internally. The formal
answer instead lists every visited 1-based seat, including the starting and
final seats.

Mechanic load-bearing: both one-direction baselines are strictly longer
than the generated unique mixed-direction optimum.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

import math
from collections import deque
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

PANEL_BG = (244, 240, 232)
PANEL_BORDER = (212, 198, 168)
TABLE_FILL = (236, 196, 132)
TABLE_OUTLINE = (160, 116, 60)
PLATE_FILL = (252, 252, 248)
PLATE_OUTLINE = (180, 168, 132)
SERVED_CHECK = (90, 168, 96)

# Seat counts admitted around the table.  Six seats can only ever produce a
# four-pass route; the larger tables are what make the shortest route vary.
MIN_SEATS = 6
MAX_SEATS = 9
#: Kept as the smallest admitted table so callers that only need a default
#: table size stay meaningful; every solver argument is explicit.
NUM_SEATS = MIN_SEATS
#: At most this many seats may want service.  One reasoning image is authored
#: per serving pass, so the cap keeps the worst branch inside the seven-image
MAX_DEMANDS = 5


def _solve_min_passes_with_count(
    start: int,
    demands: Dict[int, str],
    num_seats: int = NUM_SEATS,
) -> Tuple[Optional[List[str]], int]:
    """Return one shortest plan and the number of shortest plans.

    Each step is "CW" (advance +1 around the table) or "CCW" (-1). When the
    dish lands on a seat in demands not yet served, it's served (free
    serve action, no extra cost).
    """
    target_set = frozenset(demands.keys())
    served_initial = frozenset({start} & target_set)
    initial_state = (start, served_initial)
    if served_initial == target_set:
        return [], 1

    queue = deque([initial_state])
    distance = {initial_state: 0}
    ways = {initial_state: 1}
    parent: Dict[Tuple[int, frozenset[int]], Tuple[Tuple[int, frozenset[int]], str]] = {}
    goal_distance: Optional[int] = None
    goal_states: set[Tuple[int, frozenset[int]]] = set()

    while queue:
        state = queue.popleft()
        seat, served = state
        state_distance = distance[state]
        if goal_distance is not None and state_distance >= goal_distance:
            continue
        for direction, delta in (("CW", 1), ("CCW", -1)):
            new_seat = (seat + delta) % num_seats
            new_served = served
            if new_seat in target_set:
                new_served = served | {new_seat}
            new_state = (new_seat, new_served)
            new_distance = state_distance + 1
            if goal_distance is not None and new_distance > goal_distance:
                continue
            if new_state not in distance:
                distance[new_state] = new_distance
                ways[new_state] = ways[state]
                parent[new_state] = (state, direction)
                queue.append(new_state)
            elif distance[new_state] == new_distance:
                ways[new_state] += ways[state]
            if new_served == target_set:
                if goal_distance is None:
                    goal_distance = new_distance
                if new_distance == goal_distance:
                    goal_states.add(new_state)

    if goal_distance is None or not goal_states:
        return None, 0

    shortest_count = sum(ways[state] for state in goal_states)
    goal = min(goal_states, key=lambda state: state[0])
    plan: List[str] = []
    cursor = goal
    while cursor != initial_state:
        cursor, direction = parent[cursor]
        plan.append(direction)
    plan.reverse()
    return plan, shortest_count




def _pure_direction(start: int, demands: Dict[int, str], delta: int, num_seats: int = NUM_SEATS) -> int:
    """Count one-direction steps needed to serve all demanded seats."""
    served = set({start} & set(demands.keys()))
    seat = start
    moves = 0
    while served != set(demands.keys()):
        seat = (seat + delta) % num_seats
        moves += 1
        if seat in demands and seat not in served:
            served.add(seat)
        if moves > num_seats * 3:
            break
    return moves


def _format_seat_path(seats: Sequence[int], num_seats: int) -> str:
    if (
        not seats
        or any(type(seat) is not int or not 1 <= seat <= int(num_seats) for seat in seats)
    ):
        raise ValueError("meal-train path must contain visible 1-based seats")
    return f"Answer: [{', '.join(str(seat) for seat in seats)}]."


def _moves_from_seat_path(
    start: int,
    seats: Sequence[int],
    num_seats: int,
) -> Optional[List[str]]:
    """Recover internal pass directions from a complete submitted seat path."""

    if (
        not isinstance(seats, (list, tuple))
        or len(seats) < 2
        or not 0 <= int(start) < int(num_seats)
        or any(type(seat) is not int or not 1 <= seat <= int(num_seats) for seat in seats)
        or int(seats[0]) != int(start) + 1
    ):
        return None
    moves: List[str] = []
    for previous, following in zip(seats, seats[1:]):
        current = int(previous) - 1
        destination = int(following) - 1
        if destination == (current + 1) % int(num_seats):
            moves.append("CW")
        elif destination == (current - 1) % int(num_seats):
            moves.append("CCW")
        else:
            return None
    return moves




@lru_cache(maxsize=None)
def _eligible_scenarios(num_seats: int = NUM_SEATS) -> Tuple[Tuple[int, Tuple[int, ...], Tuple[str, ...]], ...]:
    """Every admissible table of this size, enumerated rather than sampled.

    The whole scenario space for one seat count is small, so the generator
    filters it once and then draws from what is actually eligible instead of
    proposing random demand patterns until one satisfies the gates.
    """

    scenarios: List[Tuple[int, Tuple[int, ...], Tuple[str, ...]]] = []
    for start in range(num_seats):
        for demand_count in range(3, min(num_seats, MAX_DEMANDS) + 1):
            for demand_seats in combinations(range(num_seats), demand_count):
                demands = {seat: "service" for seat in demand_seats}
                plan, shortest_count = _solve_min_passes_with_count(start, demands, num_seats)
                if plan is None or shortest_count != 1 or set(plan) != {"CW", "CCW"}:
                    continue
                if min(
                    _pure_direction(start, demands, 1, num_seats),
                    _pure_direction(start, demands, -1, num_seats),
                ) <= len(plan):
                    continue
                scenarios.append((start, demand_seats, tuple(plan)))
    return tuple(scenarios)


class MealTrainPassServeGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "meal_train_pass_serve"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/meal_train_pass_serve"
    SCENE_OUTER_MARGIN = 8
    SCENE_CHEVRON_BAND = 26
    SCENE_CHEVRON_SCALE = 0.72

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        num_seats = self.rng.randint(MIN_SEATS, MAX_SEATS)
        scenarios = _eligible_scenarios(num_seats)
        if not scenarios:
            raise RuntimeError("No unique mixed-direction service scenarios are available.")
        # Draw the route length first: the eligible space of a larger table is
        # dominated by its longest routes, and sampling scenarios directly
        # would hide the shorter ones the same table can also pose.
        by_length: Dict[int, List[Tuple[int, Tuple[int, ...], Tuple[str, ...]]]] = {}
        for scenario in scenarios:
            by_length.setdefault(len(scenario[2]), []).append(scenario)
        length = self.rng.choice(sorted(by_length))
        start, demand_seats, optimal_tuple = self.rng.choice(by_length[length])
        demands = {seat: "service" for seat in demand_seats}
        optimal = list(optimal_tuple)
        end_seat = start
        served_order: List[int] = [start] if start in demands else []
        for move in optimal:
            end_seat = (end_seat + (1 if move == "CW" else -1)) % num_seats
            if end_seat in demands and end_seat not in served_order:
                served_order.append(end_seat)
        no_service_seats = sorted(set(range(num_seats)) - set(demand_seats))
        initial = {
            "current_seat": start,
            "served": [start] if start in demands else [],
            "demands": demands,
            "empty_seats": no_service_seats,
        }
        target = {
            "current_seat": end_seat,
            "served": served_order,
            "demands": demands,
            "empty_seats": no_service_seats,
        }
        return initial, target, optimal

    @staticmethod
    def _seat_count(state: Any) -> int:
        """Seats around this table, read from the state itself.

        The verifier replays moves and re-renders source templates from a
        generator it constructs itself, so the table size has to travel with
        the state rather than live on the instance that sampled it.
        """

        demands = state.get("demands") or {}
        empty_seats = state.get("empty_seats") or []
        return len(demands) + len(empty_seats)

    def _state_signature(self, state: Any) -> Any:
        return (state["current_seat"], tuple(sorted(state["served"])))

    def _serialize_state(self, state: Any) -> Any:
        return {
            "current_seat": int(state["current_seat"]),
            "served": list(state["served"]),
            "demands": {int(k): v for k, v in state["demands"].items()},
            "empty_seats": [int(seat) for seat in state.get("empty_seats", [])],
        }

    def _apply_move(self, state: Any, move: Any) -> Any:
        direction = str(move).upper()
        delta = 1 if direction == "CW" else -1 if direction == "CCW" else 0
        if delta == 0:
            raise ValueError(f"Bad direction {direction}")
        # JSON object keys are strings after a canonical-manifest round trip.
        # Normalize them here so independent verifier replay has exactly the
        # same semantics as direct in-memory generation.
        demands = {int(seat): value for seat, value in state["demands"].items()}
        new_seat = (state["current_seat"] + delta) % self._seat_count(state)
        new_served = list(state["served"])
        if new_seat in demands and new_seat not in new_served:
            new_served.append(new_seat)
        return {"current_seat": new_seat, "served": new_served,
                "demands": demands,
                "empty_seats": list(state.get("empty_seats", []))}

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    def _author_observation(self, initial_state, target_state, moves) -> str:
        start = int(initial_state["current_seat"]) + 1
        waiting = sorted(
            int(seat) + 1
            for seat in initial_state["demands"]
            if int(seat) not in {int(s) for s in initial_state["served"]}
        )
        seats = ", ".join(str(seat) for seat in waiting)
        served_note = (
            f" Seat {start} wanted a dish and is served the moment the platter starts there."
            if int(initial_state["current_seat"]) in {int(k) for k in initial_state["demands"]}
            else ""
        )
        seat_count = self._seat_count(initial_state)
        demands = {int(seat): str(value) for seat, value in initial_state["demands"].items()}
        clockwise_only = _pure_direction(int(initial_state["current_seat"]), demands, 1, seat_count)
        counter_only = _pure_direction(int(initial_state["current_seat"]), demands, -1, seat_count)
        return (
            f"The platter starts on seat {start} of this {seat_count}-seat table, and the gold dots mark seats "
            f"{seats} as still waiting for service.{served_note} Each pass moves the platter one seat clockwise "
            "or counter-clockwise and serves a waiting seat on arrival. Sweeping clockwise the whole way round "
            f"would take {clockwise_only} passes and counter-clockwise {counter_only}, so the shortest route "
            f"doubles back instead and finishes in {len(moves)}."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        direction = str(move).upper()
        start = int(previous_state["current_seat"]) + 1
        destination = int(next_state["current_seat"]) + 1
        newly_served = [
            int(seat) + 1
            for seat in next_state["served"]
            if seat not in previous_state["served"]
        ]
        effect = (
            f"seat {newly_served[0]} is waiting and is served on arrival"
            if newly_served
            else "that seat needs no service, so the platter just passes through"
        )
        return (
            f"From seat {start}, the shortest remaining route passes the platter {direction} to seat "
            f"{destination}; {effect}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Every gold-dot seat now shows the green served check, and the {len(moves)} passes form the "
            "unique shortest service route."
        )

    def _layout(self, panel_w: int, panel_h: int) -> Dict[str, Any]:
        cx = panel_w / 2
        cy = panel_h / 2
        table_r = min(panel_w, panel_h) * 0.32
        plate_r = table_r * 0.28
        return {
            "panel_w": panel_w,
            "panel_h": panel_h,
            "cx": cx,
            "cy": cy,
            "table_r": table_r,
            "plate_r": plate_r,
        }

    def _seat_position(self, layout: Dict[str, Any], seat: int, num_seats: int) -> Tuple[float, float]:
        # Seats arranged clockwise from top
        theta = -math.pi / 2 + seat * (2 * math.pi / num_seats)
        seat_r = layout["table_r"] + layout["plate_r"] * 0.65
        return (layout["cx"] + seat_r * math.cos(theta),
                layout["cy"] + seat_r * math.sin(theta))


    def _draw_seat(
        self,
        draw: ImageDraw.ImageDraw,
        cx: float,
        cy: float,
        plate_r: float,
        wants_dish: Optional[str],
        served: bool,
        is_empty: bool,
        has_platter: bool = False,
    ) -> None:
        plate_outline = (48, 146, 78) if served else PLATE_OUTLINE
        plate_w = 3 if served else 2
        draw.ellipse((cx - plate_r, cy - plate_r, cx + plate_r, cy + plate_r),
                     fill=(224, 244, 228) if served else PLATE_FILL, outline=plate_outline, width=plate_w)
        if wants_dish is not None and not served:
            marker_r = max(5, plate_r * 0.22)
            marker_y = cy - plate_r - marker_r - 4
            draw.ellipse((cx - marker_r, marker_y - marker_r, cx + marker_r, marker_y + marker_r), fill=(224, 170, 70))
        # Served check on plate
        if served:
            cx_check = cx - plate_r * 0.30
            cy_check = cy
            draw.line((cx_check, cy_check, cx_check + plate_r * 0.20,
                       cy_check + plate_r * 0.30), fill=SERVED_CHECK, width=4)
            draw.line((cx_check + plate_r * 0.20, cy_check + plate_r * 0.30,
                       cx_check + plate_r * 0.50, cy_check - plate_r * 0.30),
                      fill=SERVED_CHECK, width=4)
        if is_empty:
            draw.line((cx - plate_r * 0.50, cy - plate_r * 0.50,
                       cx + plate_r * 0.50, cy + plate_r * 0.50),
                      fill=(180, 180, 180), width=2)
            draw.line((cx + plate_r * 0.50, cy - plate_r * 0.50,
                       cx - plate_r * 0.50, cy + plate_r * 0.50),
                      fill=(180, 180, 180), width=2)
        if has_platter:
            draw.ellipse((cx - plate_r * 0.35, cy - plate_r * 0.35,
                          cx + plate_r * 0.35, cy + plate_r * 0.35),
                         fill=(200, 152, 80), outline=(140, 100, 50), width=2)

    def _render_panel(
        self,
        state: Dict[str, Any],
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        layout = self._layout(panel_w, panel_h)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        # Round table
        cx, cy = layout["cx"], layout["cy"]
        r = layout["table_r"]
        draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                     fill=TABLE_FILL, outline=TABLE_OUTLINE, width=3)
        # Seats
        platter_seat = state["current_seat"]
        animate_pos = None
        if moving is not None and "platter_pos" in moving:
            animate_pos = moving["platter_pos"]
            platter_seat = -1  # don't draw the static platter on a seat
        demands = {int(seat): value for seat, value in state["demands"].items()}
        num_seats = self._seat_count(state)
        for seat in range(num_seats):
            sx, sy = self._seat_position(layout, seat, num_seats)
            wants = demands.get(seat)
            served = seat in state["served"]
            is_empty = seat in set(state.get("empty_seats", []))
            has_platter = (platter_seat == seat)
            self._draw_seat(draw, sx, sy, layout["plate_r"], wants,
                            served, is_empty, has_platter=has_platter)
        # Animated platter
        if animate_pos is not None:
            ax, ay = animate_pos
            pr = layout["plate_r"] * 0.35
            draw.ellipse((ax - pr, ay - pr, ax + pr, ay + pr),
                         fill=(200, 152, 80), outline=(140, 100, 50), width=2)
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _animate_move(self, state, move, target_state):
        direction = str(move).upper()
        delta = 1 if direction == "CW" else -1
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h)
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        # Interpolate platter along the arc between current_seat and next_seat
        cur_seat = state["current_seat"]
        num_seats = self._seat_count(state)
        next_seat = (cur_seat + delta) % num_seats
        cur_pos = self._seat_position(layout, cur_seat, num_seats)
        nxt_pos = self._seat_position(layout, next_seat, num_seats)
        frames = []
        after = self._apply_move(state, move)
        for step in range(1, self.move_frames + 1):
            t = step / self.move_frames
            if step == self.move_frames:
                frames.append(self._render_static_scene(after, target_state))
                continue
            ax, ay = self._interpolate_pixel_linear(cur_pos, nxt_pos, t)
            anim = {"platter_pos": (ax, ay)}
            left = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left, right_image))
        return frames

    def _describe_input(self) -> str:
        # The prompt is rebuilt from a fresh generator at the family
        # publication boundary, so it cannot quote this sample's seat count;
        # the table size is visible in the image and varies between samples.
        return (
            "The input puzzle image shows two side-by-side dining scenes on a soft-cream canvas. Each scene "
            "has a flat circular TABLE in the middle with the same evenly spaced unlabeled seat plates around "
            "it, seat 1 at the top and the rest following clockwise. A small gold dot above a plate means that seat "
            "still needs service; a pale X means that seat needs no service; a green plate/check means served. One bronze "
            "PLATTER starts on the visibly occupied seat. The left scene shows the starting state; the right scene shows "
            "the desired final state with every wanting guest served. A blue chevron arrow between the "
            "scenes points from left to right. Guest labels and decorative salad, pizza, soup, fish, and cake "
            "illustrations are omitted because dish type does not change the routing rule."
        )

    def _describe_rules(self) -> str:
        return (
            "The platter starts on the seat where it is visibly drawn. Each move passes it one seat clockwise (`CW`) or "
            "one seat counter-clockwise (`CCW`); after the move, if the platter lands on a guest who "
            "wants a dish and has not yet been served, that guest is served automatically (no extra move "
            "needed). A pale-X seat is treated normally — the platter still passes through it but never "
            "serves anything there. This state has one unique shortest route, and that route uses both "
            "directions. Find the shortest sequence of `CW`/`CCW` passes that serves every "
            "guest who wants a dish."
        )

    def _describe_output(self) -> str:
        return (
            "Return the complete visited-seat sequence for the unique shortest route, including the platter's starting "
            "seat and every landing through the final seat. Use 1-based seat numbers and one space after each comma. "
            "As a syntax-only example: `Answer: [3, 4, 5, 4, ...].`"
        )

    def _format_move(self, move):
        return f'"{str(move).upper()}"'

    def _format_answer(self, moves):
        items = ", ".join(self._format_move(m) for m in moves)
        return f"[{items}]"

    def _format_answer_from_trace(self, moves, state_trace):
        del moves
        seats = [int(state["current_seat"]) + 1 for state in state_trace]
        return "[" + ", ".join(str(seat) for seat in seats) + "]"

    def _extra_payload(self, initial_state, target_state, moves):
        demands = {int(k): v for k, v in initial_state["demands"].items()}
        return {
            "num_seats": self._seat_count(initial_state),
            "start_seat": int(initial_state["current_seat"]),
            "demands": {str(k): v for k, v in demands.items()},
            "empty_seats": [int(seat) for seat in initial_state.get("empty_seats", [])],
            "moves_canonical": [str(m).upper() for m in moves],
            "optimal_count": len(moves),
            "shortest_plan_count": 1,
        }


__all__ = ["MealTrainPassServeGenerator"]
