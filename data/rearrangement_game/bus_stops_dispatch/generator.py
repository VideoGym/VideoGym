"""Bus pickup/dropoff route with capacity C.

A linear route with 5 STOPS (S1, S2, S3, S4, S5) is laid out left-to-right.
A BUS with capacity C=2 starts at S1 and must complete its route. Each
stop S_i has 0-2 RIDERS waiting; each rider has a destination stop label
visible on their badge. The bus serves passengers in this order:

* Each move drives the bus EXACTLY ONE STOP in {L, R} (left or right).
* Stop S1 is serviced once before the exposed input state.
* Upon each later arrival at a stop:
  - Every rider on board whose destination is THIS stop disembarks and stays
    visibly delivered below that stop.
  - Waiting riders board left-to-right until capacity 2 is reached; riders
    who cannot board remain in their original waiting slots without reflow.

The puzzle goal: deliver every rider to their destination stop. The
solver uses left/right moves internally, while the formal answer lists
every visited 1-based stop, including the initial and final stops.

Mechanic load-bearing check (Tier B): we generate puzzles where capacity
C=2 strictly increases the optimal trip length compared to C=infinity
(unconstrained capacity).
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    _shade,
)

PANEL_BG = (244, 246, 248)
PANEL_BORDER = (202, 207, 214)
ROAD_FILL = (188, 194, 200)
ROAD_LINE = (250, 250, 246)
BUS_BODY = (236, 184, 64)
BUS_OUTLINE = (164, 118, 36)

N_STOPS = 5
CAPACITY = 2


def _copy_rider(rider: Sequence[int]) -> List[int]:
    if len(rider) != 2:
        raise ValueError("Each rider must be [rider_id, destination_index]")
    rider_id, destination = int(rider[0]), int(rider[1])
    if rider_id <= 0 or not (0 <= destination < N_STOPS):
        raise ValueError("Invalid rider identity or destination")
    return [rider_id, destination]


def _copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    waiting = state.get("waiting", [])
    delivered = state.get("delivered", [[] for _ in range(N_STOPS)])
    if len(waiting) != N_STOPS or len(delivered) != N_STOPS:
        raise ValueError("A bus state must contain exactly five stops")
    return {
        "bus_pos": int(state.get("bus_pos", 0)),
        "on_board": [_copy_rider(rider) for rider in state.get("on_board", [])],
        "waiting": [
            [None if rider is None else _copy_rider(rider) for rider in stop]
            for stop in waiting
        ],
        "delivered": [[_copy_rider(rider) for rider in stop] for stop in delivered],
    }


def _state_key(state: Dict[str, Any]) -> Tuple[Any, ...]:
    copied = _copy_state(state)
    return (
        copied["bus_pos"],
        tuple(tuple(rider) for rider in copied["on_board"]),
        tuple(
            tuple(None if rider is None else tuple(rider) for rider in stop)
            for stop in copied["waiting"]
        ),
        tuple(tuple(tuple(rider) for rider in stop) for stop in copied["delivered"]),
    )


def _service_stop(state: Dict[str, Any], capacity: int) -> Dict[str, Any]:
    result = _copy_state(state)
    bus_pos = result["bus_pos"]
    if not (0 <= bus_pos < N_STOPS):
        raise ValueError("Bus position out of range")
    if capacity < 1:
        raise ValueError("Capacity must be positive")

    staying: List[List[int]] = []
    for rider in result["on_board"]:
        if rider[1] == bus_pos:
            result["delivered"][bus_pos].append(rider)
        else:
            staying.append(rider)
    result["on_board"] = staying

    available = capacity - len(result["on_board"])
    if available < 0:
        raise ValueError("Bus is already over capacity")
    boarding: List[List[int]] = []
    for slot, rider in enumerate(result["waiting"][bus_pos]):
        if rider is None or len(boarding) >= available:
            continue
        boarding.append(rider)
        result["waiting"][bus_pos][slot] = None
    result["on_board"].extend(boarding)
    result["on_board"].sort(key=lambda rider: rider[0])
    result["delivered"][bus_pos].sort(key=lambda rider: rider[0])
    return result


def _apply_direction_state(
    state: Dict[str, Any], move: Any, capacity: int = CAPACITY
) -> Dict[str, Any]:
    direction = str(move).upper()
    if direction not in {"L", "R"}:
        raise ValueError(f"Unknown direction {move!r}")
    result = _copy_state(state)
    result["bus_pos"] += 1 if direction == "R" else -1
    if not (0 <= result["bus_pos"] < N_STOPS):
        raise ValueError("Bus move leaves the five-stop route")
    return _service_stop(result, capacity)


def _is_complete(state: Dict[str, Any]) -> bool:
    copied = _copy_state(state)
    return not copied["on_board"] and all(
        all(rider is None for rider in stop) for stop in copied["waiting"]
    )


def _shortest_plan(
    initial_state: Dict[str, Any], capacity: int, max_moves: int
) -> Optional[List[str]]:
    start = _copy_state(initial_state)
    if _is_complete(start):
        return []
    queue = deque([(start, [])])
    seen = {_state_key(start)}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_moves:
            continue
        for direction in ("L", "R"):
            try:
                next_state = _apply_direction_state(state, direction, capacity)
            except ValueError:
                continue
            key = _state_key(next_state)
            if key in seen:
                continue
            seen.add(key)
            next_path = path + [direction]
            if _is_complete(next_state):
                return next_path
            queue.append((next_state, next_path))
    return None


def _format_stop_path(stops: Sequence[int]) -> str:
    if not stops or any(type(stop) is not int or not 1 <= stop <= N_STOPS for stop in stops):
        raise ValueError("bus stop path must contain 1-based stops 1 through 5")
    return f"Answer: [{', '.join(str(stop) for stop in stops)}]."


def _moves_from_stop_path(
    initial_state: Dict[str, Any],
    stops: Sequence[int],
) -> Optional[List[str]]:
    """Recover internal left/right moves from a complete submitted stop path."""

    if (
        not isinstance(stops, (list, tuple))
        or len(stops) < 2
        or any(type(stop) is not int or not 1 <= stop <= N_STOPS for stop in stops)
    ):
        return None
    state = _copy_state(initial_state)
    if int(stops[0]) != int(state["bus_pos"]) + 1:
        return None
    moves: List[str] = []
    for previous, following in zip(stops, stops[1:]):
        delta = int(following) - int(previous)
        if delta not in {-1, 1}:
            return None
        move = "R" if delta == 1 else "L"
        try:
            state = _apply_direction_state(state, move, CAPACITY)
        except ValueError:
            return None
        if int(state["bus_pos"]) + 1 != int(following):
            return None
        moves.append(move)
    return moves


class BusStopsDispatchGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "bus_stops_dispatch"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/bus_stops_dispatch"
    SCENE_OUTER_MARGIN = 8
    SCENE_CHEVRON_BAND = 26
    SCENE_CHEVRON_SCALE = 0.72

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        max_optimal_moves: int = 9,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.max_optimal_moves = int(max_optimal_moves)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            n_riders = self.rng.randint(3, 4)
            riders: List[Tuple[int, int, int]] = []  # (id, start, destination)
            for rider_id in range(1, n_riders + 1):
                s = self.rng.randint(0, N_STOPS - 1)
                d = self.rng.randint(0, N_STOPS - 1)
                if s == d:
                    break
                riders.append((rider_id, s, d))
            if len(riders) != n_riders:
                continue
            waiting: List[List[List[int]]] = [[] for _ in range(N_STOPS)]
            for rider_id, start, destination in riders:
                waiting[start].append([rider_id, destination])
            # Cap waiting at 2 per stop.
            if any(len(w) > 2 for w in waiting):
                continue
            unresolved = {
                "bus_pos": 0,
                "on_board": [],
                "waiting": waiting,
                "delivered": [[] for _ in range(N_STOPS)],
            }
            initial_state = _service_stop(unresolved, CAPACITY)
            infinite_initial = _service_stop(unresolved, 10**6)
            plan = _shortest_plan(initial_state, CAPACITY, self.max_optimal_moves)
            if plan is None:
                continue
            if not (5 <= len(plan) <= self.max_optimal_moves):
                continue
            # Mechanic load-bearing.
            plan_inf = _shortest_plan(
                infinite_initial, 10**6, self.max_optimal_moves
            )
            if plan_inf is None or len(plan_inf) >= len(plan):
                continue
            target_state = _copy_state(initial_state)
            for move in plan:
                target_state = _apply_direction_state(target_state, move, CAPACITY)
            return initial_state, target_state, list(plan)
        raise RuntimeError(
            "bus_stops_dispatch rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state):
        return _state_key(state)

    def _serialize_state(self, state):
        return {
            "bus_pos": int(state["bus_pos"]),
            "on_board": [_copy_rider(rider) for rider in state["on_board"]],
            "waiting": [
                [None if rider is None else _copy_rider(rider) for rider in stop]
                for stop in state["waiting"]
            ],
            "delivered": [[_copy_rider(rider) for rider in stop] for stop in state["delivered"]],
        }

    def _apply_move(self, state, move):
        return _apply_direction_state(state, move, CAPACITY)

    def _layout(self, panel_w, panel_h):
        # Keep the 72 px bus fully inside the panel at both endpoint stops.
        margin = max(40, int(min(panel_w, panel_h) * 0.05))
        usable_w = panel_w - 2 * margin
        stop_gap = usable_w / max(1, N_STOPS - 1)
        road_half = max(30, int(panel_h * 0.075))
        road_y = int(panel_h * 0.62)
        return {
            "panel_w": panel_w, "panel_h": panel_h,
            "margin": margin,
            "stop_gap": stop_gap,
            "road_y": road_y,
            "road_y0": road_y - road_half,
            "road_y1": road_y + road_half,
        }

    def _stop_x(self, layout, idx: int) -> float:
        return layout["margin"] + layout["stop_gap"] * idx

    def _draw_road(self, draw, layout):
        x0 = layout["margin"] - 12
        x1 = layout["panel_w"] - layout["margin"] + 12
        draw.rectangle((x0, layout["road_y0"], x1, layout["road_y1"]), fill=ROAD_FILL)
        cy = layout["road_y"]
        draw.line((x0 + 4, cy, x1 - 4, cy), fill=ROAD_LINE, width=2)

    def _draw_stop(self, draw, layout, idx: int):
        sx = self._stop_x(layout, idx)
        marker_color = ITEM_PALETTE[idx % N_STOPS]
        cy = layout["road_y0"] - 82
        draw.line((sx, cy + 18, sx, layout["road_y0"] - 4), fill=marker_color, width=4)
        draw.rounded_rectangle((sx - 16, cy - 14, sx + 16, cy + 14), radius=7, fill=marker_color, outline=_shade(marker_color, 0.55), width=2)
        font = self._draw_font(draw, 14)
        text_color = (32, 30, 26) if sum(marker_color) > 380 else (250, 248, 244)
        draw.text((sx, cy - 1), str(idx + 1), font=font, fill=text_color, anchor="mm")

    def _waiting_rider_pos(self, layout, stop: int, index: int) -> Tuple[float, float]:
        return (
            self._stop_x(layout, stop) - 9 + index * 18,
            layout["road_y0"] - 32,
        )

    def _delivered_rider_pos(self, layout, stop: int, index: int) -> Tuple[float, float]:
        # Four fixed 2x2 landing slots keep delivered riders inside both edge
        # panels without making earlier arrivals reflow when another rider is
        # delivered to the same stop.
        return (
            self._stop_x(layout, stop) - 9 + (index % 2) * 18,
            layout["road_y1"] + 22 + (index // 2) * 18,
        )

    def _bus_rider_pos(self, bus_x: float, bus_y: float, index: int) -> Tuple[float, float]:
        return bus_x - 10 + index * 20, bus_y

    def _draw_rider(self, draw, cx: float, cy: float, rider: Sequence[int]):
        _, dest = _copy_rider(rider)
        radius = 7
        color = ITEM_PALETTE[dest % len(ITEM_PALETTE)]
        outline = _shade(color, 0.55)
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=color,
            outline=outline,
            width=2,
        )

    def _draw_bus(self, draw, cx: float, cy: float, on_board: Sequence[Sequence[int]]):
        body_w = 72
        body_h = 34
        x0 = cx - body_w / 2
        x1 = cx + body_w / 2
        y0 = cy - body_h / 2
        y1 = cy + body_h / 2
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=BUS_BODY, outline=BUS_OUTLINE, width=2)
        for index, rider in enumerate(on_board):
            tx, ty = self._bus_rider_pos(cx, cy, index)
            self._draw_rider(draw, tx, ty, rider)

    def _render_panel(self, state, panel_w, panel_h, *, moving=None):
        layout = self._layout(panel_w, panel_h)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        self._draw_road(draw, layout)
        for i in range(N_STOPS):
            self._draw_stop(draw, layout, i)
        moving = moving or {}
        hidden_waiting = set(int(rider_id) for rider_id in moving.get("hide_waiting", []))
        hidden_on_board = set(int(rider_id) for rider_id in moving.get("hide_on_board", []))
        hidden_delivered = set(int(rider_id) for rider_id in moving.get("hide_delivered", []))
        for stop, riders in enumerate(state["waiting"]):
            for slot_index, rider in enumerate(riders):
                if rider is None:
                    continue
                rider_id, _ = _copy_rider(rider)
                if rider_id in hidden_waiting:
                    continue
                rx, ry = self._waiting_rider_pos(layout, stop, slot_index)
                self._draw_rider(draw, rx, ry, rider)
        for stop, riders in enumerate(state["delivered"]):
            for slot_index, rider in enumerate(riders):
                rider_id, _ = _copy_rider(rider)
                if rider_id in hidden_delivered:
                    continue
                rx, ry = self._delivered_rider_pos(layout, stop, slot_index)
                self._draw_rider(draw, rx, ry, rider)
        # Bus.
        if "bus_x" in moving:
            bx = moving["bus_x"]
            bus_on_board = [
                _copy_rider(rider)
                for rider in moving.get("on_board", state["on_board"])
                if int(rider[0]) not in hidden_on_board
            ]
        else:
            bus_pos = int(state["bus_pos"])
            bx = self._stop_x(layout, max(0, bus_pos))
            bus_on_board = [
                _copy_rider(rider)
                for rider in state["on_board"]
                if int(rider[0]) not in hidden_on_board
            ]
        by = layout["road_y"]
        if not moving.get("hide_bus", False) and int(state["bus_pos"]) >= 0:
            self._draw_bus(draw, bx, by, bus_on_board)
        for floating in moving.get("floating_riders", []):
            self._draw_rider(draw, floating["center"][0], floating["center"][1], floating["rider"])
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _render_static_scene(self, initial_state, target_state):
        geom = self._panel_geometry()
        left = self._render_panel(initial_state, geom["panel_w"], geom["panel_h"])
        right = self._render_panel(
            target_state,
            geom["panel_w"],
            geom["panel_h"],
            moving={"hide_bus": True},
        )
        return self._compose_scene(left, right)

    def _animate_move(self, state, move, target_state):
        direction = str(move).upper()
        if direction not in {"L", "R"}:
            raise ValueError(f"Unknown direction {move!r}")
        delta = 1 if direction == "R" else -1
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h)
        sx = self._stop_x(layout, state["bus_pos"])
        destination_stop = state["bus_pos"] + delta
        dx = self._stop_x(layout, destination_stop)
        right_image = self._render_panel(
            target_state, panel_w, panel_h, moving={"hide_bus": True}
        )
        frames: List[Image.Image] = []
        after = self._apply_move(state, move)
        for step in range(1, self.move_frames + 1):
            t = step / self.move_frames
            cx, _ = self._interpolate_pixel_linear((sx, 0.0), (dx, 0.0), t)
            anim = {"bus_x": cx, "on_board": list(state["on_board"])}
            left = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left, right_image))

        pre_service = _copy_state(state)
        pre_service["bus_pos"] = destination_stop
        dropped = [rider for rider in state["on_board"] if int(rider[1]) == destination_stop]
        after_on_ids = {int(rider[0]) for rider in after["on_board"]}
        before_on_ids = {int(rider[0]) for rider in state["on_board"]}
        boarded = [
            rider for rider in state["waiting"][destination_stop]
            if rider is not None and int(rider[0]) in after_on_ids - before_on_ids
        ]
        moving_ids = [int(rider[0]) for rider in dropped + boarded]
        if moving_ids:
            for step in range(1, self.move_frames + 1):
                t = step / self.move_frames
                floating = []
                for rider in dropped:
                    source_index = next(
                        index for index, current in enumerate(state["on_board"])
                        if int(current[0]) == int(rider[0])
                    )
                    destination_index = next(
                        index for index, current in enumerate(after["delivered"][destination_stop])
                        if int(current[0]) == int(rider[0])
                    )
                    source = self._bus_rider_pos(dx, layout["road_y"], source_index)
                    destination = self._delivered_rider_pos(
                        layout, destination_stop, destination_index
                    )
                    floating.append({
                        "rider": rider,
                        "center": self._interpolate_pixel_linear(source, destination, t),
                    })
                for rider in boarded:
                    source_index = next(
                        index for index, current in enumerate(state["waiting"][destination_stop])
                        if current is not None and int(current[0]) == int(rider[0])
                    )
                    destination_index = next(
                        index for index, current in enumerate(after["on_board"])
                        if int(current[0]) == int(rider[0])
                    )
                    source = self._waiting_rider_pos(layout, destination_stop, source_index)
                    destination = self._bus_rider_pos(dx, layout["road_y"], destination_index)
                    floating.append({
                        "rider": rider,
                        "center": self._interpolate_pixel_linear(source, destination, t),
                    })
                anim = {
                    "bus_x": dx,
                    "hide_waiting": [int(rider[0]) for rider in boarded],
                    "hide_on_board": [int(rider[0]) for rider in dropped],
                    "floating_riders": floating,
                }
                left = self._render_panel(pre_service, panel_w, panel_h, moving=anim)
                frames.append(self._compose_scene(left, right_image))
        if frames:
            frames[-1] = self._render_static_scene(after, target_state)
        return frames


    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _rider_phrase(rider) -> str:
        # Riders carry no printed numerals; their visible identity is the
        # destination-coloured token, so the narration speaks that colour link.
        return f"a rider bound for stop {int(rider[1]) + 1}"

    def _bus_readout(self, state) -> str:
        on_board = [self._rider_phrase(r) for r in state["on_board"] if r is not None]
        return ", ".join(on_board) if on_board else "no one"

    def _author_observation(self, initial_state, target_state, moves) -> str:
        waiting_facts = []
        for stop, riders in enumerate(initial_state["waiting"], start=1):
            real = [r for r in riders if r is not None]
            if real:
                waiting_facts.append(
                    f"stop {stop} has {' and '.join(self._rider_phrase(r) for r in real)} waiting"
                )
        waiting_text = "; ".join(waiting_facts) if waiting_facts else "no one is left waiting"
        return (
            f"The bus stands at stop {int(initial_state['bus_pos']) + 1} carrying "
            f"{self._bus_readout(initial_state)}, and {waiting_text}. Each move drives exactly one stop "
            "left or right; on arrival, riders bound for that stop step off before waiting riders board "
            "left to right up to the two-seat cap. That cap is why the shortest route here has to "
            "revisit ground a bigger bus could skip."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        direction = "right" if str(move).upper() == "R" else "left"
        arrive = int(next_state["bus_pos"]) + 1
        def rider_multiset(riders):
            counts = {}
            for r in riders:
                if r is not None:
                    counts[int(r[1])] = counts.get(int(r[1]), 0) + 1
            return counts
        before = rider_multiset(previous_state["on_board"])
        after = rider_multiset(next_state["on_board"])
        events = []
        dropped = sum(max(0, before.get(d, 0) - after.get(d, 0)) for d in before)
        if dropped:
            noun = "rider" if dropped == 1 else "riders"
            events.append(f"the {noun} bound for stop {arrive} step{'s' if dropped == 1 else ''} off at home")
        boarded = [d for d in after for _ in range(max(0, after.get(d, 0) - before.get(d, 0)))]
        if boarded:
            names = " and ".join(f"a rider bound for stop {d + 1}" for d in sorted(boarded))
            events.append(f"{names} board{'s' if len(boarded) == 1 else ''}")
        still_waiting = [r for r in next_state["waiting"][arrive - 1] if r is not None]
        if still_waiting and boarded:
            names = " and ".join(self._rider_phrase(r) for r in still_waiting)
            events.append(f"{names} must keep waiting at the full bus")
        event_text = "; ".join(events) if events else "no rider needs this stop right now"
        return (
            f"The bus drives one stop {direction} to stop {arrive}: {event_text}. "
            f"On board now: {self._bus_readout(next_state)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Every rider now stands delivered below their destination stop and the bus is empty, "
            f"after {len(moves)} one-stop drives."
        )

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side bus routes. Each route is a horizontal road with FIVE "
            "numbered stop markers 1 through 5. Each marker has a distinct color. A flat yellow BUS token rides on "
            "the road. Waiting riders are small flat tokens whose color matches their destination stop marker. The "
            "left scene shows the starting layout after stop 1 has already been serviced: riders selected there are "
            "already visible inside the bus and any unboarded riders remain at their stop. The right scene shows the "
            "goal with every rider resting below the matching destination marker; the bus is omitted there because "
            "its final stop is unconstrained. Decorative flags, windows, wheels, people, and repeated `S` labels are omitted."
        )

    def _describe_rules(self) -> str:
        return (
            f"Each move drives the bus EXACTLY ONE STOP in {{L, R}}. Upon arrival at a stop, every rider on board "
            f"whose destination color equals that stop disembarks and remains visibly delivered below the marker; "
            f"then waiting riders board from left to right until the bus's capacity of {CAPACITY} riders is reached "
            f"(extra waiting riders stay fixed at the stop). Stop 1 has already performed this service in the input, "
            f"so do not service it a second time before the first move. The goal is to deliver every rider."
        )

    def _describe_output(self) -> str:
        return (
            "Return the complete visited-stop sequence for a shortest plan, including the bus's initial stop and "
            "every arrival through the final stop. Use 1-based stop numbers and one space after each comma. "
            "As a syntax-only example: `Answer: [1, 2, 3, 2, ...].`"
        )

    def _format_move(self, move):
        return f"'{str(move).upper()}'"

    def _format_answer_from_trace(self, moves, state_trace):
        del moves
        stops = [int(state["bus_pos"]) + 1 for state in state_trace]
        return "[" + ", ".join(str(stop) for stop in stops) + "]"

    def _extra_payload(self, initial_state, target_state, moves):
        return {
            "n_stops": N_STOPS,
            "capacity": CAPACITY,
            "optimal_moves": [str(m).upper() for m in moves],
        }


__all__ = ["BusStopsDispatchGenerator"]
