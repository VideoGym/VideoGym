"""Train coupling reorder via a single-position siding.

A horizontal main track holds 4-6 numbered railcars. Below it sits a single
siding slot that can store at most one railcar at a time. Each move either
detaches a chosen car from the main track and parks it on the siding, OR
re-attaches the parked car at any slot in the main track. The solver
returns the shortest sequence of moves that reorders the train into the
target order.

A move is encoded as ``("park", car_id)`` for moving the named car onto the
siding, or ``("place", slot)`` to re-attach the parked car at the given
main-track boundary slot. The move list is rendered as a list of
``(action, value)`` tuples.

``car_id`` is the identity printed on the railcar and is the same number in
every representation. The insertion slot is an ordinal: solving, replay and
rendering count boundaries ``0..len(track)``, while the formal answer counts
them ``1..len(track) + 1``. Only that value crosses
:mod:`data.rearrangement_game.core.answer_indexing`, once in each direction.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    _shade,
)
from data.rearrangement_game.core.answer_indexing import to_visible_move

PANEL_BG = (244, 240, 230)
PANEL_BORDER = (212, 200, 170)
TRACK_COLOR = (116, 108, 96)
SIDING_COLOR = (148, 136, 118)
CAR_COUNT_RANGE = (4, 6)
MAX_OPTIMAL_MOVES = 6
MOVE_FRAMES = 9

Move = Tuple[str, int]
State = Dict[str, Any]
StateKey = Tuple[Tuple[int, ...], Optional[int]]


@dataclass(frozen=True)
class SolveResult:
    distance: int
    path_count: int
    path: Tuple[Move, ...]


def _state_signature(track: Sequence[int], parked: Optional[int]) -> StateKey:
    return (tuple(track), parked)


def _validate_state(raw: Any) -> State:
    if not isinstance(raw, dict) or set(raw) != {"track", "parked"}:
        raise ValueError("state must contain exactly track and parked")
    track_raw = raw.get("track")
    parked_raw = raw.get("parked")
    if not isinstance(track_raw, list):
        raise ValueError("state.track must be a list")
    track: List[int] = []
    for value in track_raw:
        if type(value) is not int:
            raise ValueError("car ids must be non-boolean integers")
        track.append(int(value))
    if parked_raw is not None and type(parked_raw) is not int:
        raise ValueError("state.parked must be null or a non-boolean integer")
    parked = None if parked_raw is None else int(parked_raw)
    cars = track + ([] if parked is None else [parked])
    if not CAR_COUNT_RANGE[0] <= len(cars) <= CAR_COUNT_RANGE[1]:
        raise ValueError("state must contain four through six cars")
    if sorted(cars) != list(range(1, len(cars) + 1)):
        raise ValueError("state must contain each car id 1..N exactly once")
    return {"track": track, "parked": parked}


def _validate_move(raw: Any, *, car_count: int, track_length: int, parked: Optional[int]) -> Move:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("a move must be a two-entry tuple")
    action, value = raw
    if action not in ("park", "place"):
        raise ValueError("move action must be exactly 'park' or 'place'")
    if type(value) is not int:
        raise ValueError("move value must be a non-boolean integer")
    value = int(value)
    if action == "park":
        if parked is not None:
            raise ValueError("park requires an empty siding")
        if not 1 <= value <= car_count:
            raise ValueError("park car id must be in 1..N")
    else:
        if parked is None:
            raise ValueError("place requires one parked car")
        if not 0 <= value <= track_length:
            raise ValueError("place slot must be an internal boundary index in 0..len(track)")
    return str(action), value


def _legal_successors(state: StateKey) -> Tuple[Tuple[StateKey, Move], ...]:
    track, parked = state
    successors: List[Tuple[StateKey, Move]] = []
    if parked is None:
        for index, car in enumerate(track):
            next_track = track[:index] + track[index + 1 :]
            successors.append(((next_track, car), ("park", car)))
    else:
        for slot in range(len(track) + 1):
            next_track = track[:slot] + (parked,) + track[slot:]
            successors.append(((next_track, None), ("place", slot)))
    return tuple(successors)


def _shortest_solution(
    initial_track: Sequence[int],
    target_track: Sequence[int],
    *,
    max_moves: Optional[int] = None,
) -> Optional[SolveResult]:
    initial = tuple(initial_track)
    target = tuple(target_track)
    if not CAR_COUNT_RANGE[0] <= len(initial) <= CAR_COUNT_RANGE[1]:
        raise ValueError("initial track must contain four through six cars")
    if len(target) != len(initial):
        raise ValueError("initial and target tracks must contain the same number of cars")
    expected = list(range(1, len(initial) + 1))
    if any(type(value) is not int for value in initial + target):
        raise ValueError("track car ids must be non-boolean integers")
    if sorted(initial) != expected or sorted(target) != expected:
        raise ValueError("initial and target tracks must each be a permutation of 1..N")
    if max_moves is not None and (type(max_moves) is not int or max_moves < 0):
        raise ValueError("max_moves must be a non-negative integer or None")

    start: StateKey = (initial, None)
    goal: StateKey = (target, None)
    queue = deque([start])
    distance = {start: 0}
    path_count = {start: 1}
    parent: Dict[StateKey, Tuple[StateKey, Move]] = {}
    goal_distance: Optional[int] = 0 if start == goal else None
    while queue:
        current = queue.popleft()
        depth = distance[current]
        if goal_distance is not None and depth >= goal_distance:
            continue
        if max_moves is not None and depth >= max_moves:
            continue
        for next_state, move in _legal_successors(current):
            next_depth = depth + 1
            if max_moves is not None and next_depth > max_moves:
                continue
            if next_state not in distance:
                distance[next_state] = next_depth
                path_count[next_state] = path_count[current]
                parent[next_state] = current, move
                queue.append(next_state)
            elif distance[next_state] == next_depth:
                path_count[next_state] += path_count[current]
            if next_state == goal:
                goal_distance = next_depth if goal_distance is None else min(goal_distance, next_depth)
    if goal not in distance:
        return None
    path: List[Move] = []
    cursor = goal
    while cursor != start:
        previous, move = parent[cursor]
        path.append(move)
        cursor = previous
    path.reverse()
    return SolveResult(distance[goal], path_count[goal], tuple(path))


class TrainCoupleSwapGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "train_couple_swap"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/train_couple_swap"
    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        car_count_range: Tuple[int, int] = CAR_COUNT_RANGE,
        max_optimal_moves: int = MAX_OPTIMAL_MOVES,
        move_frames: int = MOVE_FRAMES,
    ) -> None:
        if (
            not isinstance(car_count_range, tuple)
            or len(car_count_range) != 2
            or any(type(value) is not int for value in car_count_range)
            or not CAR_COUNT_RANGE[0] <= car_count_range[0] <= car_count_range[1] <= CAR_COUNT_RANGE[1]
        ):
            raise ValueError("car_count_range must be an integer subrange of (4, 6)")
        if type(max_optimal_moves) is not int or max_optimal_moves not in (4, 6):
            raise ValueError("max_optimal_moves must be exactly 4 or 6")
        if type(move_frames) is not int or move_frames < 5 or move_frames % 2 == 0:
            raise ValueError("move_frames must be an odd integer of at least five")
        super().__init__(
            output_dir=output_dir,
            seed=seed,
            video=video,
            move_frames=move_frames,
        )
        self.car_count_range = car_count_range
        self.max_optimal_moves = max_optimal_moves

    def _make_problem(self, initial_track: Sequence[int], target_track: Sequence[int]) -> Tuple[State, State, List[Move]]:
        solution = _shortest_solution(initial_track, target_track, max_moves=self.max_optimal_moves)
        if solution is None or not 4 <= solution.distance <= self.max_optimal_moves or solution.distance % 2:
            raise ValueError("published train order must require two or three complete relocations")
        initial: State = {"track": list(initial_track), "parked": None}
        target: State = {"track": list(target_track), "parked": None}
        replay = initial
        for move in solution.path:
            replay = self._apply_move(replay, move)
        if replay != target:
            raise RuntimeError("independently solved path does not reach the derived target")
        self._train_prompt_moves = tuple(solution.path)
        return initial, target, list(solution.path)

    def _sample_problem(self) -> Tuple[State, State, List[Move]]:
        for _ in range(4000):
            n = self.rng.randint(*self.car_count_range)
            cars = list(range(1, n + 1))
            initial_track = list(cars)
            self.rng.shuffle(initial_track)
            target_track = list(cars)
            self.rng.shuffle(target_track)
            solution = _shortest_solution(initial_track, target_track, max_moves=self.max_optimal_moves)
            if solution is not None and 4 <= solution.distance <= self.max_optimal_moves and solution.distance % 2 == 0:
                return self._make_problem(initial_track, target_track)
        raise RuntimeError("no train pair matches the configured shortest-move range")

    def _state_signature(self, state: Any) -> Any:
        parsed = _validate_state(state)
        return _state_signature(parsed["track"], parsed["parked"])

    def _serialize_state(self, state: Any) -> Any:
        return _validate_state(state)

    def _apply_move(self, state: Any, move: Any) -> State:
        parsed = _validate_state(state)
        track = list(parsed["track"])
        parked = parsed["parked"]
        car_count = len(track) + (1 if parked is not None else 0)
        action, value = _validate_move(
            move,
            car_count=car_count,
            track_length=len(track),
            parked=parked,
        )
        if action == "park":
            if value not in track:
                raise ValueError("park may name only a car currently on the main track")
            track.remove(value)
            return _validate_state({"track": track, "parked": value})
        assert parked is not None
        new_track = track[:value] + [parked] + track[value:]
        return _validate_state({"track": new_track, "parked": None})

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _car_color(self, car_id: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(car_id - 1) % len(ITEM_PALETTE)]

    def _layout(self, panel_w: int, panel_h: int, n: int) -> Dict[str, Any]:
        margin_x = max(8, int(round(panel_w * 0.05)))
        # Push the main track up to the upper third and the siding to the
        # lower third; both sit on the canvas with breathing room between
        # them so the composition uses the full vertical span.
        track_y = int(panel_h * 0.30)
        siding_y = int(panel_h * 0.72)
        margin_top = max(12, int(round(panel_h * 0.08)))
        usable_w = panel_w - 2 * margin_x
        car_w = min((usable_w / n), panel_w * 0.24)
        car_h = car_w * 0.78
        car_w = max(32.0, car_w)
        car_h = max(24.0, car_h)
        return {
            "margin_x": margin_x,
            "margin_top": margin_top,
            "panel_w": panel_w,
            "panel_h": panel_h,
            "track_y": track_y,
            "siding_y": siding_y,
            "car_w": car_w,
            "car_h": car_h,
            "siding_cx": panel_w / 2,
        }

    def _draw_train_track(
        self,
        draw: ImageDraw.ImageDraw,
        layout: Dict[str, Any],
        y: int,
        is_main: bool,
    ) -> None:
        margin_x = layout["margin_x"]
        panel_w = layout["panel_w"]
        # Two simple parallel rails. Repeated sleepers created distracting
        # flicker while cars moved, so the track stays intentionally flat.
        rail_color = TRACK_COLOR if is_main else SIDING_COLOR
        x0 = margin_x
        x1 = panel_w - margin_x
        if is_main:
            draw.rounded_rectangle((x0, y - 3, x1, y), radius=2, fill=rail_color)
            draw.rounded_rectangle((x0, y + 11, x1, y + 14), radius=2, fill=rail_color)
        else:
            # Siding: a single short rail, only behind a single car.
            siding_w = layout["car_w"] * 1.4
            cx = layout["siding_cx"]
            sx0 = cx - siding_w / 2
            sx1 = cx + siding_w / 2
            draw.rounded_rectangle((sx0, y - 3, sx1, y), radius=2, fill=rail_color)
            draw.rounded_rectangle((sx0, y + 11, sx1, y + 14), radius=2, fill=rail_color)

    def _draw_railcar(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        car_w: float,
        car_h: float,
        car_id: int,
    ) -> None:
        cx, cy = center
        color = self._car_color(car_id)
        outline_color = _shade(color, 0.72)
        x0 = cx - car_w / 2
        x1 = cx + car_w / 2
        y0 = cy - car_h / 2
        y1 = cy + car_h / 2
        # Flat body and two restrained wheels. The car has no shadow,
        # highlight strip, or changing decorative detail.
        draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=color, outline=outline_color, width=2)
        wheel_r = max(2, int(car_h * 0.13))
        wheel_y = y1 + wheel_r * 0.35
        for wx_frac in (0.22, 0.78):
            wx = x0 + car_w * wx_frac
            draw.ellipse(
                (wx - wheel_r, wheel_y - wheel_r, wx + wheel_r, wheel_y + wheel_r),
                fill=(70, 66, 60),
            )
        # Label.
        text_color = (32, 30, 26) if (color[0] + color[1] + color[2]) > 380 else (250, 248, 244)
        font_size = max(11, int(car_h * 0.50))
        font = self._draw_font(draw, font_size)
        draw.text((cx, cy - 1), str(car_id), font=font, fill=text_color, anchor="mm")

    def _render_panel(
        self,
        state: Dict[str, Any],
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        track = list(state["track"])
        parked = state.get("parked")
        n = max(1, len(track) + (1 if parked is not None else 0))
        layout = self._layout(panel_w, panel_h, n)
        fixed_car_w = layout["car_w"] * 0.82
        fixed_car_h = layout["car_h"]
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        # A quiet funnel-shaped switch zone makes the direct main-to-siding
        # motion read as rail-guided rather than as a car floating in space.
        siding_half = layout["car_w"] * 0.70
        draw.line(
            (
                layout["margin_x"],
                layout["track_y"] + 14,
                layout["siding_cx"] - siding_half,
                layout["siding_y"] - 3,
            ),
            fill=(202, 194, 180),
            width=2,
        )
        draw.line(
            (
                panel_w - layout["margin_x"],
                layout["track_y"] + 14,
                layout["siding_cx"] + siding_half,
                layout["siding_y"] - 3,
            ),
            fill=(202, 194, 180),
            width=2,
        )
        self._draw_train_track(draw, layout, layout["track_y"], is_main=True)
        self._draw_train_track(draw, layout, layout["siding_y"], is_main=False)

        # During a transition every car receives an explicit position. This
        # lets the selected car and all cars whose slots shift slide together,
        # avoiding a landing-frame reflow jump.
        if moving is not None and "positions" in moving:
            for car_id, center in moving["positions"]:
                self._draw_railcar(
                    draw, center, fixed_car_w, fixed_car_h, int(car_id)
                )
            return image

        # Position cars on the main track.
        margin_x = layout["margin_x"]
        track_w = panel_w - 2 * margin_x
        slots = max(len(track), 1)
        slot_w = track_w / max(1, slots)
        car_w = fixed_car_w
        car_h = fixed_car_h
        track_y = layout["track_y"]
        for idx, car_id in enumerate(track):
            if moving is not None and moving.get("hide_main_idx") == idx:
                continue
            cx = margin_x + slot_w * (idx + 0.5)
            self._draw_railcar(draw, (cx, track_y + 4), car_w, car_h, int(car_id))

        # Parked car.
        if parked is not None and (moving is None or not moving.get("hide_parked", False)):
            cx = layout["siding_cx"]
            self._draw_railcar(draw, (cx, layout["siding_y"] + 4), car_w, car_h, int(parked))

        # Floating moving car.
        if moving is not None and "car_id" in moving and "center" in moving:
            cx, cy = moving["center"]
            self._draw_railcar(draw, (cx, cy), car_w, car_h, int(moving["car_id"]))

        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(
        self,
        state: Any,
        panel_width: int,
        *,
        panel_height: Optional[int] = None,
        animation: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _track_pos(self, layout: Dict[str, Any], track_len: int, idx: int) -> Tuple[float, float]:
        margin_x = layout["margin_x"]
        track_w = layout["panel_w"] - 2 * margin_x
        slot_w = track_w / max(1, track_len)
        cx = margin_x + slot_w * (idx + 0.5)
        return (cx, layout["track_y"] + 4)

    def _siding_pos(self, layout: Dict[str, Any]) -> Tuple[float, float]:
        return (layout["siding_cx"], layout["siding_y"] + 4)

    def _state_positions(
        self,
        layout: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[int, Tuple[float, float]]:
        track = list(state["track"])
        positions = {
            int(car_id): self._track_pos(layout, len(track), idx)
            for idx, car_id in enumerate(track)
        }
        parked = state.get("parked")
        if parked is not None:
            positions[int(parked)] = self._siding_pos(layout)
        return positions

    def _animate_move(
        self,
        state: Any,
        move: Any,
        target_state: Any,
    ) -> List[Image.Image]:
        parsed = _validate_state(state)
        target = _validate_state(target_state)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        track = list(parsed["track"])
        n = max(1, len(track) + (1 if parsed["parked"] is not None else 0))
        layout = self._layout(panel_w, panel_h, n)
        right_image = self._render_panel(target, panel_w, panel_h, moving=None)
        next_state = self._apply_move(parsed, move)
        source_positions = self._state_positions(layout, parsed)
        target_positions = self._state_positions(layout, next_state)
        car_order = [int(car_id) for car_id in parsed["track"]]
        if parsed["parked"] is not None:
            car_order.append(int(parsed["parked"]))
        frames: List[Image.Image] = []
        for step in range(1, self.move_frames + 1):
            t = step / self.move_frames
            positions = [
                (
                    car_id,
                    self._interpolate_pixel_linear(
                        source_positions[car_id], target_positions[car_id], t
                    ),
                )
                for car_id in car_order
            ]
            left = self._render_panel(
                parsed, panel_w, panel_h, moving={"positions": positions}
            )
            frames.append(self._compose_scene(left, right_image))
        return frames

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _describe_input(self) -> str:
        return (
            "The input image contains two fixed train-yard panels on one beige canvas. The left panel is "
            "the current state and the right panel is the target state, as indicated only by the central blue left-to-right "
            "chevron. Each panel has one long upper main track and one short lower siding centered beneath it. Four through "
            "six railcars are identified by stable body colors and the necessary integer labels from 1 through N. The left siding and "
            "the target siding are empty. Cars have one flat rounded body and two restrained wheels; the image contains no "
            "panel title, track label, slot number, rule text, move list, legend, status badge, shadow, highlight, sleeper "
            "pattern, coupling detail, scenery, or decorative texture."
        )

    def _describe_rules(self) -> str:
        return (
            "Every car id from 1 through N appears exactly once. A legal `('park', car_id)` move requires an empty siding and names "
            "one car currently on the main track; that car leaves the main track for the single siding, while all main-track "
            "survivors keep their relative left-to-right order. A legal `('place', slot)` move requires one parked car and "
            "inserts it into the current main track at a boundary slot from 1 through len(track) + 1, inclusive; slot 1 "
            "is before the leftmost car and the last slot is after the rightmost car. Cars at or right of the insertion "
            "shift right without changing relative order. Because the siding capacity is one, moves alternate a legal park "
            "of a currently visible main-track car and a legal place at one of the current track boundaries; parking while "
            "the siding is occupied or placing while it is empty is illegal. "
            "The goal is the exact right-panel order with the siding empty. Return a globally shortest legal sequence; every "
            "shortest sequence is acceptable. No "
            "car is deleted, duplicated, recolored, rotated, reversed, resized, or transferred between the fixed target panel "
            "and the changing left panel."
        )

    def _describe_output(self) -> str:
        return (
            "Return only `Answer: [(\"action\", value), ...].` using lowercase `park` or `place`, double quotes around "
            "every action, decimal integer values, one space after each comma, parentheses around each move, square brackets "
            "around the sequence, exactly one final period, and no prefix, suffix, explanation, omitted move, or extra move."
        )

    def _build_prompts(self) -> Dict[str, str]:
        """Use the family-owned natural three-section prompt contract."""
        return super()._build_prompts()

    def _format_move(self, move: Any) -> str:
        """Serialize one internal move; only a `place` slot becomes one-based."""

        action, value = to_visible_move(self.TASK_NAME, (str(move[0]), int(move[1])))
        return f"(\"{action}\", {value})"

    def _format_answer(self, moves: Sequence[Any]) -> str:
        return "[" + ", ".join(self._format_move(move) for move in moves) + "]"

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _track_phrase(track) -> str:
        return ", ".join(str(car) for car in track) if track else "empty"

    def _author_observation(self, initial_state, target_state, moves) -> str:
        initial = _validate_state(initial_state)
        target = _validate_state(target_state)
        return (
            f"The main track couples cars {self._track_phrase(initial['track'])} left to right, and the "
            f"target order is {self._track_phrase(target['track'])}. The siding below holds one car at a "
            "time, so the train is reordered in park-and-place pairs: uncouple one car into the siding, "
            "let the rest close up, then splice it back at a chosen boundary."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        before = _validate_state(previous_state)
        after = _validate_state(next_state)
        target = _validate_state(target_state)
        action, value = str(move[0]), int(move[1])
        if action == "park":
            position = before["track"].index(value) + 1
            detail = (
                f"Car {value} at main-track position {position} uncouples into the siding, and the "
                f"remaining cars close up to {self._track_phrase(after['track'])}"
            )
        else:
            visible_slot = to_visible_move(self.TASK_NAME, (action, value))[1]
            car = before["parked"]
            landed = after["track"].index(car)
            settled = (
                landed < len(target["track"]) and target["track"][landed] == car
                and after["track"][: landed + 1] == target["track"][: landed + 1]
            )
            extra = (
                ", exactly where the target order wants it"
                if settled
                else ""
            )
            detail = (
                f"The parked car {car} splices back in at boundary slot {visible_slot}{extra}, and the "
                f"main track now reads {self._track_phrase(after['track'])}"
            )
        return f"{detail}."

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"The main track now matches the target exactly with the siding empty again, after "
            f"{len(moves) // 2} park-and-place relocations ({len(moves)} moves, the recomputed minimum)."
        )

    def _extra_payload(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> Optional[Dict[str, Any]]:
        initial = _validate_state(initial_state)
        target = _validate_state(target_state)
        solution = _shortest_solution(initial["track"], target["track"], max_moves=self.max_optimal_moves)
        if solution is None or tuple(moves) != solution.path:
            raise RuntimeError("record does not contain the independently recomputed shortest plan")
        return {
            "car_count": len(initial["track"]),
            "optimal_distance": solution.distance,
            "shortest_path_count": solution.path_count,
            "relocation_count": solution.distance // 2,
            "optimal_moves": [[m[0], int(m[1])] for m in solution.path],
        }


__all__ = [
    "CAR_COUNT_RANGE",
    "MAX_OPTIMAL_MOVES",
    "MOVE_FRAMES",
    "SolveResult",
    "TrainCoupleSwapGenerator",
    "_legal_successors",
    "_shortest_solution",
    "_state_signature",
    "_validate_move",
    "_validate_state",
]
