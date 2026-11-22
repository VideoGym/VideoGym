"""Route a visible arrival order through three load-bearing LIFO hold cells."""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.pipeline.object_plans import ITEM_PALETTE
from data.common.video_encoding import pad_frames_to_wan_compatible_length

MIN_PRODUCTS = 5
MAX_PRODUCTS = 6
NUM_CELLS = 3
MOVE_FRAMES = 9

PANEL_BG = (236, 241, 247)
PANEL_BORDER = (176, 190, 206)
CONVEYOR_FILL = (96, 108, 126)
CONVEYOR_OUTLINE = (62, 72, 88)
CELL_BG = (248, 246, 236)
CELL_BORDER = (154, 136, 92)

Move = Tuple[str, int]
CellState = Tuple[Tuple[int, ...], ...]
SearchState = Tuple[int, CellState, int]
State = Dict[str, Any]


@dataclass(frozen=True)
class SolveResult:
    distance: int
    path_count: int
    path: Tuple[Move, ...]


def _validate_permutation(values: Any, *, name: str) -> Tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not MIN_PRODUCTS <= len(values) <= MAX_PRODUCTS:
        raise ValueError(f"{name} must contain five or six products")
    parsed: List[int] = []
    for value in values:
        if type(value) is not int:
            raise ValueError(f"{name} product ids must be non-boolean integers")
        parsed.append(int(value))
    expected = tuple(range(1, len(parsed) + 1))
    if tuple(sorted(parsed)) != expected:
        raise ValueError(f"{name} must be a permutation of 1 through N")
    return tuple(parsed)


def _validate_state(raw: Any) -> State:
    if not isinstance(raw, Mapping) or set(raw) != {"arrival", "cells", "outfeed", "target"}:
        raise ValueError("pipeline state must contain exactly arrival, cells, outfeed, and target")
    target = _validate_permutation(raw.get("target"), name="target")
    n = len(target)
    arrival_raw = raw.get("arrival")
    cells_raw = raw.get("cells")
    outfeed_raw = raw.get("outfeed")
    if not isinstance(arrival_raw, (list, tuple)) or len(arrival_raw) != n:
        raise ValueError("arrival must have exactly N fixed slots")
    if not isinstance(cells_raw, (list, tuple)) or len(cells_raw) != NUM_CELLS:
        raise ValueError("cells must contain exactly three LIFO stacks")
    if not isinstance(outfeed_raw, (list, tuple)) or len(outfeed_raw) > n:
        raise ValueError("outfeed must contain at most N products")

    arrival: List[Optional[int]] = []
    seen_remaining = False
    for value in arrival_raw:
        if value is None:
            if seen_remaining:
                raise ValueError("consumed arrival slots must form one left prefix")
            arrival.append(None)
            continue
        if type(value) is not int or not 1 <= value <= n:
            raise ValueError("arrival ids must be None or non-boolean integers from 1 through N")
        seen_remaining = True
        arrival.append(int(value))

    cells: List[List[int]] = []
    for raw_cell in cells_raw:
        if not isinstance(raw_cell, (list, tuple)):
            raise ValueError("each hold cell must be an ordered bottom-to-top sequence")
        cell: List[int] = []
        for value in raw_cell:
            if type(value) is not int or not 1 <= value <= n:
                raise ValueError("cell ids must be non-boolean integers from 1 through N")
            cell.append(int(value))
        cells.append(cell)

    outfeed: List[int] = []
    for value in outfeed_raw:
        if type(value) is not int or not 1 <= value <= n:
            raise ValueError("outfeed ids must be non-boolean integers from 1 through N")
        outfeed.append(int(value))
    combined = [value for value in arrival if value is not None]
    combined.extend(value for cell in cells for value in cell)
    combined.extend(outfeed)
    if len(combined) != n or tuple(sorted(combined)) != tuple(range(1, n + 1)):
        raise ValueError("every product must appear exactly once across arrival, cells, and outfeed")
    if tuple(outfeed) != target[: len(outfeed)]:
        raise ValueError("published pipeline states must keep outfeed equal to a target prefix")
    return {
        "arrival": arrival,
        "cells": cells,
        "outfeed": outfeed,
        "target": list(target),
    }


def _is_solvable(arrival: Sequence[int], target: Sequence[int], num_cells: int) -> bool:
    """Existence test with unlabeled-stack symmetry reduction."""

    source = _validate_permutation(arrival, name="arrival")
    destination = _validate_permutation(target, name="target")
    if set(source) != set(destination) or type(num_cells) is not int or not 1 <= num_cells <= NUM_CELLS:
        return False

    @lru_cache(maxsize=None)
    def visit(arrival_index: int, cells: CellState, output_index: int) -> bool:
        if output_index == len(destination):
            return arrival_index == len(source) and all(not cell for cell in cells)
        needed = destination[output_index]
        for index, cell in enumerate(cells):
            if cell and cell[-1] == needed:
                updated = list(cells)
                updated[index] = cell[:-1]
                canonical = tuple(sorted(updated))
                if visit(arrival_index, canonical, output_index + 1):
                    return True
        if arrival_index < len(source):
            product = source[arrival_index]
            distinct_cells: set[Tuple[int, ...]] = set()
            for index, cell in enumerate(cells):
                if cell in distinct_cells:
                    continue
                distinct_cells.add(cell)
                updated = list(cells)
                updated[index] = cell + (product,)
                canonical = tuple(sorted(updated))
                if visit(arrival_index + 1, canonical, output_index):
                    return True
        return False

    return visit(0, tuple(() for _ in range(num_cells)), 0)


def _pipeline_solution(
    arrival: Sequence[int],
    target: Sequence[int],
    num_cells: int = NUM_CELLS,
) -> Optional[SolveResult]:
    """Labeled-cell BFS returning one canonical path and all shortest-path multiplicity."""

    source = _validate_permutation(arrival, name="arrival")
    destination = _validate_permutation(target, name="target")
    if set(source) != set(destination):
        raise ValueError("arrival and target must contain the same products")
    if type(num_cells) is not int or not 1 <= num_cells <= NUM_CELLS:
        raise ValueError("num_cells must be one, two, or three")
    start: SearchState = (0, tuple(() for _ in range(num_cells)), 0)
    queue = deque([start])
    distance = {start: 0}
    path_count = {start: 1}
    parent: Dict[SearchState, Tuple[SearchState, Move]] = {}
    goal: Optional[SearchState] = None
    goal_distance: Optional[int] = None
    while queue:
        state = queue.popleft()
        depth = distance[state]
        arrival_index, cells, output_index = state
        if goal_distance is not None and depth >= goal_distance:
            continue
        successors: List[Tuple[SearchState, Move]] = []
        if arrival_index < len(source):
            product = source[arrival_index]
            for cell_index in range(num_cells):
                updated = list(cells)
                updated[cell_index] = cells[cell_index] + (product,)
                successors.append(
                    ((arrival_index + 1, tuple(updated), output_index), ("push", cell_index + 1))
                )
        if output_index < len(destination):
            needed = destination[output_index]
            for cell_index in range(num_cells):
                if cells[cell_index] and cells[cell_index][-1] == needed:
                    updated = list(cells)
                    updated[cell_index] = cells[cell_index][:-1]
                    successors.append(
                        ((arrival_index, tuple(updated), output_index + 1), ("pop", cell_index + 1))
                    )
        for next_state, move in successors:
            next_depth = depth + 1
            if next_state not in distance:
                distance[next_state] = next_depth
                path_count[next_state] = path_count[state]
                parent[next_state] = state, move
                queue.append(next_state)
            elif distance[next_state] == next_depth:
                path_count[next_state] += path_count[state]
            if (
                next_state[0] == len(source)
                and next_state[2] == len(destination)
                and all(not cell for cell in next_state[1])
            ):
                if goal is None:
                    goal = next_state
                    goal_distance = next_depth
    if goal is None:
        return None
    path: List[Move] = []
    cursor = goal
    while cursor != start:
        previous, move = parent[cursor]
        path.append(move)
        cursor = previous
    path.reverse()
    return SolveResult(distance[goal], path_count[goal], tuple(path))


def _minimum_cells_required(arrival: Sequence[int], target: Sequence[int]) -> Optional[int]:
    for num_cells in range(1, NUM_CELLS + 1):
        if _is_solvable(arrival, target, num_cells):
            return num_cells
    return None


@lru_cache(maxsize=None)
def _all_orders(n: int) -> Tuple[Tuple[int, ...], ...]:
    if type(n) is not int or n not in (MIN_PRODUCTS, MAX_PRODUCTS):
        raise ValueError("published product count must be five or six")
    return tuple(permutations(range(1, n + 1)))


@lru_cache(maxsize=None)
def _three_cell_relative_targets(n: int) -> Tuple[Tuple[int, ...], ...]:
    identity = tuple(range(1, n + 1))
    targets = tuple(
        target
        for target in _all_orders(n)
        if not _is_solvable(identity, target, 2) and _is_solvable(identity, target, 3)
    )
    if not targets:
        raise RuntimeError("no load-bearing three-cell target permutations found")
    return targets


def _validate_pair(initial: Any, target: Any) -> Tuple[State, State]:
    source = _validate_state(initial)
    destination = _validate_state(target)
    n = len(source["target"])
    if destination["target"] != source["target"] or len(destination["target"]) != n:
        raise ValueError("source and target panels must use the same target permutation")
    if any(value is None for value in source["arrival"]):
        raise ValueError("published source infeed must begin full")
    if source["cells"] != [[], [], []] or source["outfeed"]:
        raise ValueError("published source must begin with empty cells and outfeed")
    if destination["arrival"] != [None] * n:
        raise ValueError("published target infeed must be empty without compaction")
    if destination["cells"] != [[], [], []] or destination["outfeed"] != source["target"]:
        raise ValueError("published target must empty all cells into the exact outfeed order")
    arrival = tuple(int(value) for value in source["arrival"] if value is not None)
    target_order = tuple(source["target"])
    if _minimum_cells_required(arrival, target_order) != NUM_CELLS:
        raise ValueError("published target must require all three hold cells")
    return source, destination


def _validate_move(state: Any, raw_move: Any) -> Tuple[State, str, int]:
    parsed = _validate_state(state)
    if not isinstance(raw_move, (list, tuple)) or len(raw_move) != 2:
        raise ValueError("move must be a two-field action,cell pair")
    action, cell_id = raw_move
    if type(action) is not str or action not in ("push", "pop"):
        raise ValueError("action must be exactly push or pop")
    if type(cell_id) is not int or not 1 <= cell_id <= NUM_CELLS:
        raise ValueError("cell id must be a non-boolean integer from one through three")
    if action == "push" and all(value is None for value in parsed["arrival"]):
        raise ValueError("cannot push from an empty infeed")
    if action == "pop" and not parsed["cells"][cell_id - 1]:
        raise ValueError("cannot pop an empty cell")
    return parsed, action, int(cell_id)


def _apply_move(state: Any, raw_move: Any) -> State:
    parsed, action, cell_id = _validate_move(state, raw_move)
    arrival = list(parsed["arrival"])
    cells = [list(cell) for cell in parsed["cells"]]
    outfeed = list(parsed["outfeed"])
    if action == "push":
        front_index = next(index for index, value in enumerate(arrival) if value is not None)
        product = arrival[front_index]
        arrival[front_index] = None
        cells[cell_id - 1].append(int(product))
    else:
        outfeed.append(cells[cell_id - 1].pop())
    result = {
        "arrival": arrival,
        "cells": cells,
        "outfeed": outfeed,
        "target": list(parsed["target"]),
    }
    return _validate_state(result)


class PipelineHoldCells3Generator(RearrangementObjectPlanGenerator):
    TASK_NAME = "pipeline_hold_cells_3"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/pipeline_hold_cells_3"
    # Keep the native two-panel scene independent of mutable base defaults.
    # The narrow divider gives both five- and six-product layouts enough room
    # without adding labels or decorative machinery.
    SCENE_OUTER_MARGIN = 8
    SCENE_CHEVRON_BAND = 26
    SCENE_CHEVRON_SCALE = 0.72

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        product_count_range: Tuple[int, int] = (MIN_PRODUCTS, MAX_PRODUCTS),
        move_frames: int = MOVE_FRAMES,
    ) -> None:
        if (
            not isinstance(product_count_range, tuple)
            or len(product_count_range) != 2
            or any(type(value) is not int for value in product_count_range)
            or product_count_range[0] not in (MIN_PRODUCTS, MAX_PRODUCTS)
            or product_count_range[1] not in (MIN_PRODUCTS, MAX_PRODUCTS)
            or product_count_range[0] > product_count_range[1]
        ):
            raise ValueError("product_count_range endpoints must be ordered integers in {5,6}")
        if type(move_frames) is not int or move_frames < 7 or move_frames % 2 == 0:
            raise ValueError("move_frames must be an odd integer of at least seven")
        super().__init__(
            output_dir=output_dir,
            seed=seed,
            video=video,
            move_frames=move_frames,
            pause_frames=2,
            hold_frames=4,
        )
        self.product_count_range = product_count_range
        self._seed_value = int(seed or 0)
        self._sample_serial = 0

    def _sample_problem(self) -> Tuple[State, State, List[Move]]:
        selector = self._seed_value + self._sample_serial * 1_000_003
        allowed_counts = tuple(
            n for n in (MIN_PRODUCTS, MAX_PRODUCTS)
            if self.product_count_range[0] <= n <= self.product_count_range[1]
        )
        n = allowed_counts[selector % len(allowed_counts)]
        arrivals = _all_orders(n)
        relative_targets = _three_cell_relative_targets(n)
        arrival = arrivals[(selector * 97_409 + 31_337) % len(arrivals)]
        relative_target = relative_targets[((selector // len(allowed_counts)) * 65_537 + 17) % len(relative_targets)]
        target = tuple(arrival[index - 1] for index in relative_target)
        solution = _pipeline_solution(arrival, target, NUM_CELLS)
        if solution is None or solution.distance != 2 * n or _minimum_cells_required(arrival, target) != NUM_CELLS:
            raise RuntimeError("pipeline sampler failed the three-cell shortest-plan contract")
        source: State = {
            "arrival": list(arrival),
            "cells": [[], [], []],
            "outfeed": [],
            "target": list(target),
        }
        destination: State = {
            "arrival": [None] * n,
            "cells": [[], [], []],
            "outfeed": list(target),
            "target": list(target),
        }
        _validate_pair(source, destination)
        self._sample_serial += 1
        return source, destination, list(solution.path)

    def _state_signature(self, state: Any) -> Any:
        parsed = _validate_state(state)
        return (
            tuple(parsed["arrival"]),
            tuple(tuple(cell) for cell in parsed["cells"]),
            tuple(parsed["outfeed"]),
            tuple(parsed["target"]),
        )

    def _serialize_state(self, state: Any) -> State:
        parsed = _validate_state(state)
        return {
            "arrival": list(parsed["arrival"]),
            "cells": [list(cell) for cell in parsed["cells"]],
            "outfeed": list(parsed["outfeed"]),
            "target": list(parsed["target"]),
        }

    def _apply_move(self, state: Any, move: Any) -> State:
        return _apply_move(state, move)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _layout(self, panel_w: int, panel_h: int, n: int) -> Dict[str, float]:
        margin = max(12, int(panel_w * 0.055))
        infeed_y = margin + int(panel_h * 0.105)
        outfeed_y = panel_h - margin - int(panel_h * 0.105)
        conveyor_h = max(28, int(panel_h * 0.075))
        cell_y0 = infeed_y + conveyor_h / 2 + int(panel_h * 0.075)
        cell_y1 = outfeed_y - conveyor_h / 2 - int(panel_h * 0.055)
        usable_w = panel_w - 2 * margin
        slot_w = usable_w / n
        item_w = min(slot_w * 0.78, panel_w * 0.14)
        item_h = max(24, int(panel_h * 0.062))
        cell_gap = max(10, int(panel_w * 0.028))
        cell_w = (usable_w - 2 * cell_gap) / NUM_CELLS
        return {
            "margin": float(margin),
            "infeed_y": float(infeed_y),
            "outfeed_y": float(outfeed_y),
            "conveyor_h": float(conveyor_h),
            "cell_y0": float(cell_y0),
            "cell_y1": float(cell_y1),
            "slot_w": float(slot_w),
            "item_w": float(item_w),
            "item_h": float(item_h),
            "cell_gap": float(cell_gap),
            "cell_w": float(cell_w),
        }

    def _draw_product(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        size: Tuple[float, float],
        product_id: int,
    ) -> None:
        cx, cy = center
        item_w, item_h = size
        color = ITEM_PALETTE[(product_id - 1) % len(ITEM_PALETTE)]
        bbox = (cx - item_w / 2, cy - item_h / 2, cx + item_w / 2, cy + item_h / 2)
        draw.rounded_rectangle(
            bbox,
            radius=max(5, int(item_h * 0.16)),
            fill=color,
            outline=(54, 62, 72),
            width=max(2, int(item_h * 0.055)),
        )
        font = self._draw_font(draw, max(16, int(item_h * 0.48)))
        draw.text((cx, cy - 1), str(product_id), font=font, fill=(28, 28, 26), anchor="mm")

    def _draw_conveyor(
        self,
        draw: ImageDraw.ImageDraw,
        x0: float,
        x1: float,
        center_y: float,
        height: float,
    ) -> None:
        draw.rounded_rectangle(
            (x0, center_y - height / 2, x1, center_y + height / 2),
            radius=max(5, int(height * 0.18)),
            fill=CONVEYOR_FILL,
            outline=CONVEYOR_OUTLINE,
            width=max(2, int(height * 0.055)),
        )

    def _cell_bounds(self, layout: Mapping[str, float], cell_index: int) -> Tuple[float, float, float, float]:
        x0 = layout["margin"] + cell_index * (layout["cell_w"] + layout["cell_gap"])
        return (x0, layout["cell_y0"], x0 + layout["cell_w"], layout["cell_y1"])

    def _arrival_center(self, layout: Mapping[str, float], slot_index: int) -> Tuple[float, float]:
        return (
            layout["margin"] + layout["slot_w"] * (slot_index + 0.5),
            layout["infeed_y"],
        )

    def _cell_product_center(
        self,
        layout: Mapping[str, float],
        cell_index: int,
        stack_index: int,
    ) -> Tuple[float, float]:
        x0, _y0, x1, y1 = self._cell_bounds(layout, cell_index)
        return (
            (x0 + x1) / 2,
            y1 - layout["item_h"] / 2 - 6 - stack_index * layout["item_h"] * 1.05,
        )

    def _outfeed_center(self, layout: Mapping[str, float], slot_index: int) -> Tuple[float, float]:
        return (
            layout["margin"] + layout["slot_w"] * (slot_index + 0.5),
            layout["outfeed_y"],
        )

    def _render_panel(
        self,
        state: Any,
        panel_w: int,
        panel_h: int,
        *,
        moving: Optional[Mapping[str, Any]] = None,
    ) -> Image.Image:
        parsed = _validate_state(state)
        n = len(parsed["target"])
        layout = self._layout(panel_w, panel_h, n)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle(
            (0, 0, panel_w - 1, panel_h - 1),
            outline=PANEL_BORDER,
            width=max(2, int(panel_w * 0.007)),
        )
        self._draw_conveyor(
            draw,
            layout["margin"],
            panel_w - layout["margin"],
            layout["infeed_y"],
            layout["conveyor_h"],
        )
        self._draw_conveyor(
            draw,
            layout["margin"],
            panel_w - layout["margin"],
            layout["outfeed_y"],
            layout["conveyor_h"],
        )
        hidden_arrival = moving.get("hide_arrival") if moving else None
        hidden_cell = moving.get("hide_cell") if moving else None
        for slot_index, product in enumerate(parsed["arrival"]):
            if product is None or hidden_arrival == slot_index:
                continue
            self._draw_product(
                draw,
                self._arrival_center(layout, slot_index),
                (layout["item_w"], layout["item_h"]),
                product,
            )
        for cell_index in range(NUM_CELLS):
            bounds = self._cell_bounds(layout, cell_index)
            draw.rounded_rectangle(
                bounds,
                radius=max(6, int(layout["cell_w"] * 0.07)),
                fill=CELL_BG,
                outline=CELL_BORDER,
                width=max(2, int(panel_w * 0.006)),
            )
            for stack_index, product in enumerate(parsed["cells"][cell_index]):
                if hidden_cell == (cell_index, stack_index):
                    continue
                self._draw_product(
                    draw,
                    self._cell_product_center(layout, cell_index, stack_index),
                    (layout["item_w"], layout["item_h"]),
                    product,
                )
        for slot_index, product in enumerate(parsed["outfeed"]):
            self._draw_product(
                draw,
                self._outfeed_center(layout, slot_index),
                (layout["item_w"], layout["item_h"]),
                product,
            )
        if moving and "product" in moving and "center" in moving:
            self._draw_product(
                draw,
                tuple(moving["center"]),
                (layout["item_w"], layout["item_h"]),
                int(moving["product"]),
            )
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

    def _render_static_scene(self, initial_state: Any, target_state: Any) -> Image.Image:
        source = _validate_state(initial_state)
        target = _validate_state(target_state)
        if source["target"] != target["target"]:
            raise ValueError("rendered panels must share one visible target permutation")
        return super()._render_static_scene(source, target)

    def _build_video_frames(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> List[np.ndarray]:
        source, target = _validate_pair(initial_state, target_state)
        initial_image = self._render_static_scene(source, target)
        frames: List[np.ndarray] = [
            np.array(initial_image, dtype=np.uint8) for _ in range(self.hold_frames)
        ]
        state = source
        for move in moves:
            for frame in self._animate_move(state, move, target):
                frames.append(np.array(frame, dtype=np.uint8))
            state = _apply_move(state, move)
            exact_state = self._render_static_scene(state, target)
            frames.extend(
                np.array(exact_state, dtype=np.uint8) for _ in range(self.pause_frames)
            )
        final_image = self._render_static_scene(state, target)
        frames.extend(
            np.array(final_image, dtype=np.uint8) for _ in range(self.hold_frames)
        )
        padded = pad_frames_to_wan_compatible_length(frames)
        if len(padded) > 157:
            raise RuntimeError("pipeline video exceeds the 10-second 4N+1 media contract")
        return [np.asarray(frame, dtype=np.uint8) for frame in padded]

    def _animate_move(self, state: Any, move: Any, target_state: Any) -> List[Image.Image]:
        parsed, action, cell_id = _validate_move(state, move)
        target = _validate_state(target_state)
        if target["target"] != parsed["target"]:
            raise ValueError("animation target must preserve the visible target permutation")
        after = _apply_move(parsed, (action, cell_id))
        geometry = self._panel_geometry()
        panel_w = geometry["panel_w"]
        panel_h = geometry["panel_h"]
        layout = self._layout(panel_w, panel_h, len(parsed["target"]))
        right = self._render_panel(target, panel_w, panel_h)
        if action == "push":
            source_index = next(index for index, value in enumerate(parsed["arrival"]) if value is not None)
            product = int(parsed["arrival"][source_index])
            source_center = self._arrival_center(layout, source_index)
            destination_center = self._cell_product_center(
                layout,
                cell_id - 1,
                len(parsed["cells"][cell_id - 1]),
            )
            hidden = {"hide_arrival": source_index}
        else:
            stack_index = len(parsed["cells"][cell_id - 1]) - 1
            product = parsed["cells"][cell_id - 1][-1]
            source_center = self._cell_product_center(layout, cell_id - 1, stack_index)
            destination_center = self._outfeed_center(layout, len(parsed["outfeed"]))
            hidden = {"hide_cell": (cell_id - 1, stack_index)}
        frames: List[Image.Image] = []
        for step in range(1, self.move_frames + 1):
            center = self._interpolate_pixel_linear(source_center, destination_center, step / self.move_frames)
            animation = {**hidden, "product": product, "center": center}
            left = self._render_panel(parsed, panel_w, panel_h, moving=animation)
            frames.append(self._compose_scene(left, right))
        if frames:
            frames[-1] = self._render_static_scene(after, target)
        return frames

    # ------------------------------------------------------------------
    # Prompt and answer contracts
    # ------------------------------------------------------------------

    def _describe_input(self) -> str:
        return (
            "Two fixed production panels sit side by side, separated by a blue chevron. Each has a top infeed lane, three "
            "hold cells numbered 1 through 3 by left-to-right position, and a bottom outfeed lane. Five or six product tokens "
            "carry stable visible integer identities. The left panel is the current state: products occupy fixed infeed "
            "slots, while cells and outfeed are empty. The right panel is the exact target: infeed and cells are empty, "
            "and outfeed shows the required left-to-right product order."
        )

    def _describe_rules(self) -> str:
        return (
            "Only the leftmost remaining infeed product is the front; PUSH removes it from its fixed slot and places it "
            "on top of cell 1, 2, or 3. Vacated infeed slots stay empty, without sliding or compaction. Each cell is LIFO, "
            "and POP moves only its top product to the next empty outfeed slot; earlier outfeed products stay fixed. Empty "
            "sources, invalid cells, non-top moves, cell-to-cell transfers, direct infeed-to-outfeed moves, and rewriting "
            "outfeed are illegal. All unselected products, lower stack items, fixed slots, lanes, target panel, background, "
            "and camera stay unchanged. The goal consumes infeed, empties all cells, and exactly matches target outfeed. "
            "Every product is pushed and popped once, so every complete solution has the minimum 2N actions; accept any "
            "legal one reaching the target."
        )

    def _describe_output(self) -> str:
        return (
            "Return a minimum 2N-action plan. The complete move vocabulary is `('push', 1)`, `('push', 2)`, "
            "`('push', 3)`, `('pop', 1)`, `('pop', 2)`, and `('pop', 3)`, where the integer names a hold-cell "
            "slot rather than a grid position. As a syntax-only example: "
            "`Answer: [('push', 2), ('push', 1), ('pop', 1), ...].`"
        )

    def _build_prompts(self) -> Dict[str, str]:
        """Use the family-owned natural three-section prompt contract."""
        return super()._build_prompts()

    def _format_move(self, move: Any) -> str:
        action, cell_id = move
        return f"('{action}', {int(cell_id)})"

    def _format_answer(self, moves: Sequence[Any]) -> str:
        return "[" + ", ".join(self._format_move(move) for move in moves) + "]"

    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _row_phrase(values) -> str:
        real = [str(v) for v in values if v is not None]
        return ", ".join(real) if real else "empty"

    @classmethod
    def _cells_phrase(cls, cells) -> str:
        return "; ".join(
            f"cell {index} holds {', '.join(str(v) for v in cell) + ' bottom to top' if len(cell) > 1 else ('just product ' + str(cell[0]) if cell else '')}"
            if cell
            else f"cell {index} is empty"
            for index, cell in enumerate(cells, start=1)
        )

    def _author_observation(self, initial_state, target_state, moves) -> str:
        parsed = _validate_state(initial_state)
        return (
            f"The infeed queues products {self._row_phrase(parsed['arrival'])} front to back, and the "
            f"outfeed must finish as {', '.join(str(v) for v in parsed['target'])} left to right. Only the "
            "infeed front can be pushed, each hold cell is last-in-first-out, and only a cell top can pop "
            "to the next outfeed slot, so a product stacked above another must leave for the outfeed "
            "first. Every product is pushed once and popped once, which makes the whole plan exactly "
            "2N actions."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        parsed, action, cell_id = _validate_move(previous_state, move)
        target_order = list(parsed["target"])
        if action == "push":
            product = next(v for v in parsed["arrival"] if v is not None)
            below = parsed["cells"][cell_id - 1]
            if below:
                under = below[-1]
                relation = (
                    f", stacking above product {under}, which the outfeed needs later"
                    if target_order.index(product) < target_order.index(under)
                    else f", stacking above product {under}"
                )
            else:
                relation = ", starting a fresh stack there"
            detail = f"The infeed front, product {product}, pushes onto cell {cell_id}{relation}"
        else:
            product = parsed["cells"][cell_id - 1][-1]
            detail = (
                f"Product {product} on top of cell {cell_id} is exactly the next outfeed need, so it pops "
                "to the outfeed"
            )
        after = _validate_state(next_state)
        return (
            f"{detail}. Infeed: {self._row_phrase(after['arrival'])}; {self._cells_phrase(after['cells'])}; "
            f"outfeed: {self._row_phrase(after['outfeed'])}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"The outfeed now reads exactly the target order with the infeed consumed and all three cells "
            f"empty, after the minimum {len(moves)} push and pop actions."
        )

    def _extra_payload(
        self,
        initial_state: Any,
        target_state: Any,
        moves: Sequence[Any],
    ) -> Dict[str, Any]:
        source, _target = _validate_pair(initial_state, target_state)
        arrival = tuple(int(value) for value in source["arrival"] if value is not None)
        target_order = tuple(source["target"])
        solution = _pipeline_solution(arrival, target_order, NUM_CELLS)
        parsed_moves = tuple((str(move[0]), int(move[1])) for move in moves)
        if solution is None or parsed_moves != solution.path:
            raise RuntimeError("record does not contain the recomputed shortest pipeline plan")
        return {
            "task_contract": {
                "schema_version": "native_rearrangement_v1",
                "input_count": 1,
                "canvas_size": [self.canvas_size, self.canvas_size],
                "active_modalities": ["ti2v"],
                "render_profile": "native_pipeline_lifo_direct_trace_v1",
                "native_media": True,
                "video": {
                    "fps": self.fps,
                    "move_frames": self.move_frames,
                    "pause_frames": self.pause_frames,
                    "hold_frames": self.hold_frames,
                    "maximum_frames": 157,
                    "wan_compatible_4n_plus_1": True,
                    "normalization_policy": "preserve_native_pixels",
                },
                "answer": {
                    "kind": "typed_text",
                    "format_template": "Answer: [('push', 2), ('push', 1), ('pop', 1), ...].",
                    "move_vocabulary": [
                        ["push", 1],
                        ["push", 2],
                        ["push", 3],
                        ["pop", 1],
                        ["pop", 2],
                        ["pop", 3],
                    ],
                    "parser_profile": "strict_pipeline_move_sequence_with_typed_replay",
                    "verifier_profile": "independent_three_cell_lifo_shortest_replay",
                    "accepts_multiple_valid_answers": True,
                },
            },
            "product_count": len(arrival),
            "arrival_order": list(arrival),
            "target_permutation": list(target_order),
            "optimal_distance": solution.distance,
            "shortest_path_count": solution.path_count,
            "minimum_cells_required": _minimum_cells_required(arrival, target_order),
            "all_three_cells_load_bearing": True,
            "retained_image_count": 1,
            "moves_canonical": [list(move) for move in solution.path],
        }


__all__ = [
    "MAX_PRODUCTS",
    "MIN_PRODUCTS",
    "MOVE_FRAMES",
    "NUM_CELLS",
    "PipelineHoldCells3Generator",
    "SolveResult",
    "_all_orders",
    "_apply_move",
    "_is_solvable",
    "_minimum_cells_required",
    "_pipeline_solution",
    "_three_cell_relative_targets",
    "_validate_move",
    "_validate_pair",
    "_validate_state",
]
