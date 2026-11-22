"""Shortest forward-only frog rearrangement with one visible empty pad.

Pads are printed and answered from 1 at the left; the solver, the replay and
the renderer geometry below index them from 0.  The two systems meet only at
:meth:`JumpingFrogSwapGenerator._format_answer`, at the drawn pad label, and in
the authored reasoning that quotes those same visible pad numbers.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from data.rearrangement_game.core.answer_indexing import to_visible_plan

PANEL_BG = (220, 232, 220)
PANEL_BORDER = (180, 200, 172)
WATER_FILL = (132, 188, 220)
WATER_DARK = (88, 152, 192)
LILYPAD_FILL = (96, 168, 96)
LILYPAD_DARK = (60, 132, 60)
GREEN_FROG_COLOR = (96, 188, 96)
GREEN_FROG_DARK = (52, 138, 60)
ORANGE_FROG_COLOR = (240, 156, 80)
ORANGE_FROG_DARK = (160, 100, 36)

GROUP_COUNT_PAIRS: Tuple[Tuple[int, int], ...] = ((2, 2), (2, 3), (3, 2))
MIN_OPTIMAL_MOVES = 5
MAX_OPTIMAL_MOVES = 10
MOVE_FRAMES = 9

State = Tuple[str, ...]


@dataclass(frozen=True)
class SolveResult:
    distance: int
    path_count: int
    path: Tuple[int, ...]


def _validate_state(raw: Any) -> State:
    if not isinstance(raw, (list, tuple)):
        raise ValueError("frog state must be a list or tuple")
    state = tuple(raw)
    if not 5 <= len(state) <= 6:
        raise ValueError("frog state must contain five or six pads")
    if any(type(value) is not str or value not in {"G", "O", "."} for value in state):
        raise ValueError("each pad must be exactly G, O, or .")
    if state.count(".") != 1:
        raise ValueError("frog state must contain exactly one empty pad")
    counts = (state.count("G"), state.count("O"))
    if counts not in GROUP_COUNT_PAIRS:
        raise ValueError("frog counts must be one of (2, 2), (2, 3), or (3, 2)")
    return state


def _validate_pair(initial: Any, target: Any) -> Tuple[State, State]:
    source = _validate_state(initial)
    destination = _validate_state(target)
    if (source.count("G"), source.count("O")) != (destination.count("G"), destination.count("O")):
        raise ValueError("initial and target must preserve both frog counts")
    return source, destination


def _legal_successors(state: State) -> Tuple[Tuple[State, int], ...]:
    state = _validate_state(state)
    empty = state.index(".")
    successors: List[Tuple[State, int]] = []

    green_step = empty - 1
    if green_step >= 0 and state[green_step] == "G":
        successors.append((_apply_legal_move(state, green_step), green_step))
    green_leap = empty - 2
    if green_leap >= 0 and state[green_leap] == "G" and state[empty - 1] == "O":
        successors.append((_apply_legal_move(state, green_leap), green_leap))

    orange_step = empty + 1
    if orange_step < len(state) and state[orange_step] == "O":
        successors.append((_apply_legal_move(state, orange_step), orange_step))
    orange_leap = empty + 2
    if orange_leap < len(state) and state[orange_leap] == "O" and state[empty + 1] == "G":
        successors.append((_apply_legal_move(state, orange_leap), orange_leap))
    return tuple(successors)


def _validate_move(state: Any, raw_move: Any) -> Tuple[State, int, int, str]:
    parsed = _validate_state(state)
    if type(raw_move) is not int:
        raise ValueError("move must be a non-boolean integer source index")
    source = int(raw_move)
    if not 0 <= source < len(parsed):
        raise ValueError("move source index is out of range")
    species = parsed[source]
    if species not in {"G", "O"}:
        raise ValueError("move source must contain a frog")
    empty = parsed.index(".")
    delta = empty - source
    if species == "G" and delta not in (1, 2):
        raise ValueError("green frogs move only one or two pads right")
    if species == "O" and delta not in (-1, -2):
        raise ValueError("orange frogs move only one or two pads left")
    if abs(delta) == 2:
        middle = parsed[(source + empty) // 2]
        required = "O" if species == "G" else "G"
        if middle != required:
            raise ValueError("a two-pad leap must cross exactly one opposite-color frog")
    return parsed, source, empty, species


def _apply_legal_move(state: State, source: int) -> State:
    current = list(state)
    empty = current.index(".")
    current[empty] = current[source]
    current[source] = "."
    return tuple(current)


def _apply_move(state: Any, raw_move: Any) -> State:
    parsed, source, _empty, _species = _validate_move(state, raw_move)
    return _apply_legal_move(parsed, source)


def _shortest_solution(
    initial: Sequence[str],
    target: Sequence[str],
    *,
    max_moves: Optional[int] = None,
) -> Optional[SolveResult]:
    source, destination = _validate_pair(initial, target)
    if max_moves is not None and (type(max_moves) is not int or max_moves < 0):
        raise ValueError("max_moves must be a non-negative integer or None")
    if source == destination:
        return SolveResult(0, 1, tuple())

    queue = deque([source])
    distance = {source: 0}
    path_count = {source: 1}
    parent: Dict[State, Tuple[State, int]] = {}
    goal_distance: Optional[int] = None
    while queue:
        state = queue.popleft()
        depth = distance[state]
        if goal_distance is not None and depth >= goal_distance:
            continue
        if max_moves is not None and depth >= max_moves:
            continue
        for next_state, move in _legal_successors(state):
            next_depth = depth + 1
            if max_moves is not None and next_depth > max_moves:
                continue
            if next_state not in distance:
                distance[next_state] = next_depth
                path_count[next_state] = path_count[state]
                parent[next_state] = state, move
                queue.append(next_state)
            elif distance[next_state] == next_depth:
                path_count[next_state] += path_count[state]
            if next_state == destination:
                goal_distance = next_depth if goal_distance is None else min(goal_distance, next_depth)
    if destination not in distance:
        return None

    path: List[int] = []
    cursor = destination
    while cursor != source:
        previous, move = parent[cursor]
        path.append(move)
        cursor = previous
    path.reverse()
    return SolveResult(distance[destination], path_count[destination], tuple(path))


def _leap_count(initial: Sequence[str], moves: Sequence[int]) -> int:
    state = _validate_state(initial)
    count = 0
    for move in moves:
        empty = state.index(".")
        count += int(abs(empty - int(move)) == 2)
        state = _apply_move(state, move)
    return count


@lru_cache(maxsize=None)
def _states_for_counts(green_count: int, orange_count: int) -> Tuple[State, ...]:
    if (green_count, orange_count) not in GROUP_COUNT_PAIRS:
        raise ValueError("unsupported frog-count pair")
    return tuple(sorted(set(permutations("G" * green_count + "O" * orange_count + "."))))


@lru_cache(maxsize=None)
def _problem_catalog(
    green_count: int,
    orange_count: int,
    max_moves: int,
) -> Dict[int, Tuple[Tuple[State, State, SolveResult], ...]]:
    buckets: Dict[int, List[Tuple[State, State, SolveResult]]] = {}
    for source in _states_for_counts(green_count, orange_count):
        for destination in _states_for_counts(green_count, orange_count):
            solution = _shortest_solution(source, destination, max_moves=max_moves)
            if solution is None or not MIN_OPTIMAL_MOVES <= solution.distance <= max_moves:
                continue
            if _leap_count(source, solution.path) < 2:
                continue
            buckets.setdefault(solution.distance, []).append((source, destination, solution))
    return {distance: tuple(entries) for distance, entries in sorted(buckets.items())}


@lru_cache(maxsize=None)
def _problem_schedule(max_moves: int) -> Tuple[Tuple[State, State, SolveResult], ...]:
    buckets: List[Tuple[Tuple[State, State, SolveResult], ...]] = []
    for counts in GROUP_COUNT_PAIRS:
        buckets.extend(_problem_catalog(counts[0], counts[1], max_moves).values())
    schedule: List[Tuple[State, State, SolveResult]] = []
    for row in range(max((len(entries) for entries in buckets), default=0)):
        for entries in buckets:
            if row < len(entries):
                schedule.append(entries[row])
    return tuple(schedule)


class JumpingFrogSwapGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "jumping_frog_swap"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/jumping_frog_swap"
    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        max_optimal_moves: int = MAX_OPTIMAL_MOVES,
        move_frames: int = MOVE_FRAMES,
    ) -> None:
        if type(max_optimal_moves) is not int or not 8 <= max_optimal_moves <= MAX_OPTIMAL_MOVES:
            raise ValueError("max_optimal_moves must be an integer from 8 through 10")
        if type(move_frames) is not int or move_frames < 5 or move_frames % 2 == 0:
            raise ValueError("move_frames must be an odd integer of at least five")
        super().__init__(output_dir=output_dir, seed=seed, video=video, move_frames=move_frames)
        self.max_optimal_moves = max_optimal_moves
        self._sample_serial = 0

    def _make_problem(self, initial: Sequence[str], target: Sequence[str]) -> Tuple[List[str], List[str], List[int]]:
        source, destination = _validate_pair(initial, target)
        solution = _shortest_solution(source, destination, max_moves=self.max_optimal_moves)
        if solution is None or not MIN_OPTIMAL_MOVES <= solution.distance <= self.max_optimal_moves:
            raise ValueError("published frog pair must require five through the configured maximum moves")
        if _leap_count(source, solution.path) < 2:
            raise ValueError("published frog pair must require at least two legal opposite-color leaps")
        replay = source
        for move in solution.path:
            replay = _apply_move(replay, move)
        if replay != destination:
            raise RuntimeError("shortest frog path does not reach the visible target")
        return list(source), list(destination), list(solution.path)

    def _sample_problem(self) -> Tuple[List[str], List[str], List[int]]:
        seed_selector = int(self.seed) if self.seed is not None else self.rng.randrange(10_000_000)
        selector = seed_selector + self._sample_serial * 1_000_003
        self._sample_serial += 1
        schedule = _problem_schedule(self.max_optimal_moves)
        if not schedule:
            raise RuntimeError("no frog state pair satisfies the configured shortest-path contract")
        source, target, _solution = schedule[selector % len(schedule)]
        return self._make_problem(source, target)

    def _state_signature(self, state: Any) -> State:
        return _validate_state(state)

    def _serialize_state(self, state: Any) -> List[str]:
        return list(_validate_state(state))

    def _apply_move(self, state: Any, move: Any) -> List[str]:
        return list(_apply_move(state, move))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _layout(self, panel_w: int, panel_h: int, n: int) -> Dict[str, Any]:
        margin = max(8, int(min(panel_w, panel_h) * 0.04))
        usable = panel_w - 2 * margin
        slot_w = usable / n
        lane_h = min(panel_h * 0.34, panel_h - 2 * margin - 20)
        return {
            "panel_w": panel_w,
            "panel_h": panel_h,
            "margin": margin,
            "slot_w": slot_w,
            "center_y": int(panel_h * 0.48),
            "lane_h": lane_h,
        }

    def _slot_x(self, layout: Dict[str, Any], idx: int) -> float:
        return layout["margin"] + layout["slot_w"] * (idx + 0.5)

    def _draw_water(self, draw: ImageDraw.ImageDraw, layout: Dict[str, Any], n: int) -> None:
        del n
        x0 = layout["margin"] - 8
        x1 = layout["panel_w"] - layout["margin"] + 8
        cy = layout["center_y"]
        h = layout["lane_h"]
        draw.rounded_rectangle(
            (x0, cy - h / 2, x1, cy + h / 2),
            radius=int(h * 0.16),
            fill=WATER_FILL,
            outline=WATER_DARK,
            width=2,
        )

    def _draw_lilypad(self, draw: ImageDraw.ImageDraw, cx: float, cy: float, rx: float, ry: float) -> None:
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=LILYPAD_FILL, outline=LILYPAD_DARK, width=2)

    def _draw_frog(self, draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, kind: str) -> None:
        base, outline, facing = (
            (GREEN_FROG_COLOR, GREEN_FROG_DARK, 1)
            if kind == "G"
            else (ORANGE_FROG_COLOR, ORANGE_FROG_DARK, -1)
        )
        body_r = size * 0.34
        draw.ellipse(
            (cx - body_r, cy - body_r * 0.68, cx + body_r, cy + body_r * 0.68),
            fill=base,
            outline=outline,
            width=2,
        )
        if facing == 1:
            points = [
                (cx - body_r * 0.25, cy - body_r * 0.18),
                (cx + body_r * 0.48, cy),
                (cx - body_r * 0.25, cy + body_r * 0.18),
            ]
        else:
            points = [
                (cx + body_r * 0.25, cy - body_r * 0.18),
                (cx - body_r * 0.48, cy),
                (cx + body_r * 0.25, cy + body_r * 0.18),
            ]
        draw.polygon(points, fill=outline)

    def _render_panel(
        self,
        state: Any,
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        parsed = _validate_state(state)
        layout = self._layout(panel_w, panel_h, len(parsed))
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        self._draw_water(draw, layout, len(parsed))
        slot_w = layout["slot_w"]
        center_y = layout["center_y"]
        pad_rx = slot_w * 0.40
        pad_ry = min(slot_w * 0.30, layout["lane_h"] * 0.34)
        for index in range(len(parsed)):
            self._draw_lilypad(draw, self._slot_x(layout, index), center_y, pad_rx, pad_ry)
        font = self._draw_font(draw, max(12, int(13 * panel_w / 232)))
        label_y = center_y + layout["lane_h"] / 2 + max(14, int(panel_w * 0.06))
        for index in range(len(parsed)):
            # The lane is drawn from a 0-based position but labelled with the
            # visible pad number the answer uses.
            draw.text(
                (self._slot_x(layout, index), label_y),
                str(index + 1),
                font=font,
                fill=(60, 80, 60),
                anchor="mm",
            )
        frog_size = pad_rx * 2.45
        hidden = moving.get("hide_idx") if moving is not None else None
        for index, kind in enumerate(parsed):
            if index == hidden or kind == ".":
                continue
            self._draw_frog(draw, self._slot_x(layout, index), center_y, frog_size, kind)
        if moving is not None and "frog_pix" in moving:
            cx, cy = moving["frog_pix"]
            self._draw_frog(draw, cx, cy, frog_size, str(moving["kind"]))
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(
        self,
        state: Any,
        panel_width: int,
        *,
        panel_height: Optional[int] = None,
        animation: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        panel_h = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, panel_h, moving=animation)

    def _animate_move(self, state: Any, move: Any, target_state: Any) -> List[Image.Image]:
        parsed, source, empty, species = _validate_move(state, move)
        _source_state, target = _validate_pair(parsed, target_state)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h, len(parsed))
        src = (self._slot_x(layout, source), layout["center_y"])
        dst = (self._slot_x(layout, empty), layout["center_y"])
        right_image = self._render_panel(target, panel_w, panel_h)
        frames: List[Image.Image] = []
        for step in range(1, self.move_frames + 1):
            progress = step / self.move_frames
            center = self._interpolate_pixel_linear(src, dst, progress)
            left = self._render_panel(
                parsed,
                panel_w,
                panel_h,
                moving={"hide_idx": source, "frog_pix": center, "kind": species},
            )
            frames.append(self._compose_scene(left, right_image))
        after = _apply_move(parsed, source)
        if frames:
            frames[-1] = self._render_static_scene(after, target)
        return frames

    # ------------------------------------------------------------------
    # Prompts and answers
    # ------------------------------------------------------------------

    def _describe_input(self) -> str:
        return (
            "The input image has two fixed side-by-side blue lanes. The left lane is the current state "
            "and the right lane is the exact target, indicated only by the central blue chevron. Each lane has five or "
            "six green oval pads numbered from 1 at the left below them and exactly one empty pad. A green flat frog "
            "token has a right-pointing triangle and an orange flat frog token has a left-pointing triangle. The exact "
            "frog counts are directly visible in both lanes. The image has no title, task name, "
            "rule text, move list, legend, status badge, face, eye, mouth, water wave, leaf vein, shadow, highlight, or texture."
        )

    def _describe_rules(self) -> str:
        return (
            "At every state exactly one pad is empty. Pads are numbered from 1 at the left, and a legal move is "
            "identified by the current pad number of "
            "one frog. A green frog may move only right: it may step one pad into the empty pad, or move two pads into the "
            "empty pad only when the single middle pad contains an orange frog. An orange frog may move only left: it may "
            "step one pad into the empty pad, or move two pads only when the middle pad contains a green frog. A two-pad "
            "move over the same color or an empty middle pad is illegal. A frog cannot move backward, wrap, leave the lane, "
            "start from an empty pad, or land on an occupied pad. After a legal move, the source alone becomes empty and the former empty destination contains the "
            "selected frog. The crossed frog in a two-pad move and every other frog, pad, index, panel, color, marker, "
            "background element, and camera remain fixed; nothing falls, compacts, refills, reorders automatically, changes "
            "species, disappears, or duplicates. The goal is the exact right-panel state. Return a globally shortest legal "
            "sequence; every shortest sequence is acceptable."
        )

    def _describe_output(self) -> str:
        return (
            "Return a globally shortest list of current frog source indices as `Answer: [<source_index>, ...].`, "
            "counting the leftmost pad as 1."
        )

    def _format_move(self, move: Any) -> str:
        return str(int(move))

    def _format_answer(self, moves: Sequence[Any]) -> str:
        """Serialize the internal plan as visible pad numbers counted from 1."""

        visible = to_visible_plan(self.TASK_NAME, [int(move) for move in moves])
        return "[" + ", ".join(self._format_move(move) for move in visible) + "]"

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _lane_reading(state: Any) -> str:
        words = {"G": "a green frog", "O": "an orange frog", ".": "the empty pad"}
        return ", ".join(words[cell] for cell in _validate_state(state))

    def _author_observation(self, initial_state, target_state, moves) -> str:
        source = _validate_state(initial_state)
        return (
            f"Reading the pads from 1 on the left, the lane holds {self._lane_reading(source)}; the target "
            f"lane wants {self._lane_reading(_validate_state(target_state))}. Green frogs only ever move "
            "right and orange frogs only left, either one pad into the empty pad or a two-pad leap over a "
            "single frog of the opposite colour, so the empty pad has to shuttle back and forth to let the "
            "two colours pass each other."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        parsed, source, empty, species = _validate_move(previous_state, int(move))
        colour = "green" if species == "G" else "orange"
        direction = "right" if species == "G" else "left"
        if abs(empty - source) == 2:
            middle_index = (empty + source) // 2
            middle_colour = "orange" if parsed[middle_index] == "O" else "green"
            action = (
                f"leaps {direction} over the {middle_colour} frog on pad {middle_index + 1} into the empty "
                f"pad {empty + 1}"
            )
        else:
            action = f"steps one pad {direction} into the empty pad {empty + 1}"
        return (
            f"The {colour} frog on pad {source + 1} {action}, and the vacancy moves back to pad "
            f"{source + 1}. The lane now reads {self._lane_reading(next_state)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        source, target = _validate_pair(initial_state, target_state)
        solution = _shortest_solution(source, target, max_moves=self.max_optimal_moves)
        if solution is None or tuple(int(move) for move in moves) != solution.path:
            raise RuntimeError("frog confirmation requires the recomputed canonical shortest path")
        return (
            f"The lane now matches the target exactly, and the {len(moves)} moves match the independently "
            "recomputed minimum, with every leap crossing a single opposite-colour frog."
        )

    def _extra_payload(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> Dict[str, Any]:
        source, target = _validate_pair(initial_state, target_state)
        solution = _shortest_solution(source, target, max_moves=self.max_optimal_moves)
        if solution is None or tuple(int(move) for move in moves) != solution.path:
            raise RuntimeError("record does not contain the recomputed shortest frog path")
        return {
            "green_count": source.count("G"),
            "orange_count": source.count("O"),
            "optimal_distance": solution.distance,
            "shortest_path_count": solution.path_count,
            "leap_count": _leap_count(source, solution.path),
            "optimal_moves": list(solution.path),
        }


__all__ = [
    "GROUP_COUNT_PAIRS",
    "MAX_OPTIMAL_MOVES",
    "MIN_OPTIMAL_MOVES",
    "MOVE_FRAMES",
    "JumpingFrogSwapGenerator",
    "SolveResult",
    "_apply_move",
    "_leap_count",
    "_legal_successors",
    "_problem_catalog",
    "_problem_schedule",
    "_shortest_solution",
    "_states_for_counts",
    "_validate_move",
    "_validate_pair",
    "_validate_state",
]
