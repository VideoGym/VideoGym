"""Family-owned source-only replay semantics for Spatial Navigation final deliveries.

This module contains parsers and deterministic state-transition models used by
``data.spatial_navigation.core.verifier``.  It deliberately has no candidate-artifact
IO, reference-media comparison, authored trajectory lookup, or reward routing.
Every public consumer supplies the scrubbed visible source state explicitly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from data.common.grid_coordinates import (
    parse_named_xy_paths_answer,
    parse_xy_path_answer,
)
SEMANTIC_AGENT_PATH_VIDEO_TASKS = {
    "bridge_drop_tile",
    "conveyor_belt_path",
    "carry_capacity_2",
    "directional_arrow_entry",
    "heading_turn_budget_route",
    "ice_physics_multi_slide",
    "lantern_oil_route",
    "lever_unlock_sequence",
    "one_way_portal_chain",
    "switch_toggle_walls",
}

SEMANTIC_PRIMARY_PATH_VIDEO_TASKS = {
    "agent_courier_relay",
    "bus_route_passengers",
    "color_matched_collection",
    "cooperative_open_door",
    "falling_block_dodge",
    "hidden_treasure_dig",
    "parcel_delivery_route",
    "patrol_avoid_pattern",
    "tide_water_cycle",
}

_DIR_TO_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "stay": (0, 0),
}
_CARDINAL_DIR_TO_DELTA = {
    direction: delta
    for direction, delta in _DIR_TO_DELTA.items()
    if direction != "stay"
}

def _parse_bounded_grid_path(
    prediction: str,
    *,
    rows: int,
    cols: int,
) -> Optional[List[Tuple[int, int]]]:
    """Parse only the canonical coordinate-path grammar and enforce bounds."""

    path = parse_xy_path_answer(prediction)
    if (
        path is None
        or rows <= 0
        or cols <= 0
        or any(not (0 <= row < rows and 0 <= col < cols) for row, col in path)
    ):
        return None
    return path

def _replay_basic_grid_path(
    structured: Dict[str, Any],
    predicted_path: Optional[Sequence[Tuple[int, int]]],
    *,
    task_name: str,
) -> tuple[dict[str, Any], bool]:
    try:
        size = int(structured.get("grid_size"))
    except Exception:
        size = 0
    start = _to_int_pair(structured.get("start"))
    goal = _to_int_pair(structured.get("goal"))
    blocked = set(_to_int_tuple_path(structured.get("blocked_cells")) or [])
    required = _to_int_tuple_path(structured.get("required_cells")) or []
    if size <= 0 or start is None or goal is None:
        return {"text_replay_reason": "missing basic-grid source state"}, False
    start_cell = (int(start[0]), int(start[1]))
    goal_cell = (int(goal[0]), int(goal[1]))
    required_index = {cell: index for index, cell in enumerate(required)}
    full_mask = (1 << len(required)) - 1

    def transition(
        cell: Tuple[int, int],
        mask: int,
        nxt: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int], int]]:
        delta = (nxt[0] - cell[0], nxt[1] - cell[1])
        if delta not in _DIR_TO_DELTA.values():
            return None
        if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in blocked:
            return None
        next_mask = mask
        if nxt in required_index:
            next_mask |= 1 << required_index[nxt]
        if nxt == goal_cell and next_mask != full_mask:
            return None
        return nxt, next_mask

    initial_mask = 0
    if start_cell in required_index:
        initial_mask |= 1 << required_index[start_cell]
    cell = start_cell
    mask = initial_mask
    legal = bool(predicted_path and tuple(predicted_path[0]) == start_cell)
    for raw_next in list(predicted_path or [])[1:]:
        if cell == goal_cell and mask == full_mask:
            legal = False
            break
        result = transition(cell, mask, tuple(raw_next))
        if result is None:
            legal = False
            break
        cell, mask = result

    solved = bool(legal and cell == goal_cell and mask == full_mask)
    metrics = {
        "text_replay_legal": legal,
        "text_replay_final_cell": list(cell),
        "text_replay_goal_cell": list(goal_cell),
        "text_replay_required_mask": mask,
        "text_replay_required_full_mask": full_mask,
        "text_predicted_path_length": max(0, len(predicted_path or []) - 1),
        "text_replay_check_pass": solved,
    }
    return metrics, solved

def _valid_grid_path(
    path: Optional[Sequence[Tuple[int, int]]],
    *,
    size: int,
    walls: set[Tuple[int, int]],
) -> bool:
    return bool(
        path
        and all(0 <= row < size and 0 <= col < size and (row, col) not in walls for row, col in path)
        and all(
            abs(cell[0] - prev[0]) + abs(cell[1] - prev[1]) in {0, 1}
            for prev, cell in zip(path, path[1:])
        )
    )

def _evaluate_plant_watering_text(
    structured: Dict[str, Any],
    prediction: str,
) -> tuple[dict[str, Any], bool]:
    initial = structured.get("initial_state") or {}
    gardener_raw = _to_int_pair(initial.get("gardener")) if isinstance(initial, dict) else None
    well_raw = _to_int_pair(initial.get("well")) if isinstance(initial, dict) else None
    plants = _to_int_tuple_path(initial.get("plants")) if isinstance(initial, dict) else None
    try:
        capacity = int(structured.get("capacity"))
        refill_budget = int(
            structured.get("refill_budget", initial.get("refill_budget"))
        )
        rows = int(structured.get("rows"))
        cols = int(structured.get("cols"))
    except Exception:
        capacity = refill_budget = rows = cols = 0
    path = _parse_bounded_grid_path(prediction, rows=rows, cols=cols)
    plant_index = {cell: index for index, cell in enumerate(plants or [])}
    full_mask = (1 << len(plant_index)) - 1

    def advance(
        state: Tuple[Tuple[int, int], int, int, int],
        nxt: Tuple[int, int],
    ):
        cell, mask, can, refills_used = state
        delta = (nxt[0] - cell[0], nxt[1] - cell[1])
        if delta not in _DIR_TO_DELTA.values():
            return None
        next_mask = mask
        next_can = can
        next_refills_used = refills_used
        if well_raw is not None and nxt == tuple(well_raw) and nxt != cell:
            if next_refills_used >= refill_budget:
                return None
            next_refills_used += 1
            next_can = capacity
        if nxt in plant_index and not (next_mask & (1 << plant_index[nxt])) and next_can > 0:
            next_mask |= 1 << plant_index[nxt]
            next_can -= 1
        return nxt, next_mask, next_can, next_refills_used

    source_ok = (
        gardener_raw is not None
        and well_raw is not None
        and plants is not None
        and rows > 0
        and cols > 0
        and capacity > 0
        and refill_budget >= 0
    )
    state = (
        (tuple(gardener_raw), 0, capacity, 0)
        if source_ok
        else ((-1, -1), 0, 0, 0)
    )
    legal = bool(
        path
        and source_ok
        and gardener_raw is not None
        and tuple(path[0]) == tuple(gardener_raw)
    )
    for nxt_cell in list(path or [])[1:]:
        if state[1] == full_mask:
            legal = False
            break
        nxt = advance(state, tuple(nxt_cell))
        if nxt is None:
            legal = False
            break
        state = nxt

    solved = bool(legal and state[1] == full_mask)
    return {
        "canonical_answer_kind": "path",
        "text_parse_ok": path is not None,
        "text_replay_legal": legal,
        "text_replay_watered_mask": state[1],
        "text_replay_all_plants_watered": state[1] == full_mask,
        "text_replay_refills_used": state[3],
        "text_replay_refill_budget": refill_budget,
        "text_predicted_path_length": max(0, len(path or []) - 1),
        "text_replay_check_pass": solved,
    }, solved

def _parse_named_grid_paths(
    prediction: str,
    first_name: str,
    second_name: str,
) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    parsed = parse_named_xy_paths_answer(
        prediction,
        (first_name, second_name),
    )
    if parsed is None:
        return None
    return parsed[first_name], parsed[second_name]

def _evaluate_courier_relay_text(
    structured: Dict[str, Any],
    prediction: str,
) -> tuple[dict[str, Any], bool]:
    parsed = _parse_named_grid_paths(prediction, "A", "B")
    extras = structured.get("extras") or {}
    start = _to_int_pair(extras.get("a_start")) if isinstance(extras, dict) else None
    meet = _to_int_pair(extras.get("b_meet")) if isinstance(extras, dict) else None
    dropoff = _to_int_pair(extras.get("dropoff")) if isinstance(extras, dict) else None
    walls = set(_to_int_tuple_path(structured.get("walls")) or [])
    try:
        size = int(structured.get("grid_size"))
    except Exception:
        size = 0
    a_path = parsed[0] if parsed else None
    b_path = parsed[1] if parsed else None
    legal = _valid_grid_path(a_path, size=size, walls=walls) and _valid_grid_path(b_path, size=size, walls=walls)
    endpoints = bool(
        a_path
        and b_path
        and start is not None
        and meet is not None
        and dropoff is not None
        and a_path[0] == tuple(start)
        and a_path[-1] == tuple(meet)
        and b_path[0] == tuple(meet)
        and b_path[-1] == tuple(dropoff)
    )
    solved = bool(parsed and legal and endpoints)
    return {
        "text_joint_parse_ok": parsed is not None,
        "text_first_agent_path_legal": bool(a_path and _valid_grid_path(a_path, size=size, walls=walls)),
        "text_second_agent_path_legal": bool(b_path and _valid_grid_path(b_path, size=size, walls=walls)),
        "text_handoff_and_dropoff_ok": endpoints,
        "text_replay_check_pass": solved,
    }, solved


def _evaluate_dual_agent_text(
    sample: Any,
    structured: Dict[str, Any],
    prediction: str,
) -> tuple[dict[str, Any], bool]:
    """Replay the cooperative door contract from visible source state."""

    task_name = str(getattr(sample, "task_name", ""))
    extras = structured.get("extras") or {}
    try:
        size = int(structured.get("grid_size"))
    except Exception:
        size = 0
    walls = set(_to_int_tuple_path(structured.get("walls")) or [])
    if task_name != "cooperative_open_door" or size <= 0 or not isinstance(extras, dict):
        return {
            "text_joint_parse_ok": False,
            "text_joint_state_replay_ok": False,
            "text_replay_check_pass": False,
        }, False

    blue_start = _to_int_pair(extras.get("blue_start"))
    orange_start = _to_int_pair(extras.get("orange_start"))
    blue_path: Optional[List[Tuple[int, int]]] = None
    orange_path: Optional[List[Tuple[int, int]]] = None
    parsed = False
    pair = _parse_named_grid_paths(prediction, "blue", "orange")
    if pair is not None:
        blue_path, orange_path = list(pair[0]), list(pair[1])
        parsed = (
            bool(blue_path)
            and len(blue_path) == len(orange_path)
            and blue_start is not None
            and orange_start is not None
            and blue_path[0] == tuple(blue_start)
            and orange_path[0] == tuple(orange_start)
        )
    paths_legal = bool(
        blue_path
        and orange_path
        and len(blue_path) == len(orange_path)
        and _valid_grid_path(blue_path, size=size, walls=walls)
        and _valid_grid_path(orange_path, size=size, walls=walls)
    )

    collision_free = bool(paths_legal)
    if paths_legal and blue_path is not None and orange_path is not None:
        for index, (blue, orange) in enumerate(zip(blue_path, orange_path)):
            if blue == orange:
                collision_free = False
                break
            if index > 0 and blue == orange_path[index - 1] and orange == blue_path[index - 1]:
                collision_free = False
                break

    task_semantics_ok = False
    terminal_ok = False
    if paths_legal and collision_free and blue_path is not None and orange_path is not None:
        blue_plate = _to_int_pair(extras.get("blue_plate"))
        orange_plate = _to_int_pair(extras.get("orange_plate"))
        blue_goal = _to_int_pair(extras.get("goal"))
        door_cells = set(_to_int_tuple_path(extras.get("door_cells")) or [])
        press_step = next(
            (
                index
                for index, (blue, orange) in enumerate(zip(blue_path[1:], orange_path[1:]), 1)
                if blue_plate is not None
                and orange_plate is not None
                and blue == tuple(blue_plate)
                and orange == tuple(orange_plate)
                and blue_path[index - 1] != tuple(blue_plate)
                and orange_path[index - 1] != tuple(orange_plate)
            ),
            -1,
        )
        cross_step = next(
            (
                index
                for index in range(1, len(blue_path) - 1)
                if blue_path[index] in door_cells
                and blue_path[index - 1]
                == (blue_path[index][0] - 1, blue_path[index][1])
                and blue_path[index + 1]
                == (blue_path[index][0] + 1, blue_path[index][1])
            ),
            -1,
        )
        task_semantics_ok = bool(
            press_step >= 0
            and cross_step > press_step
            and cross_step < len(blue_path) - 1
            and door_cells
            and all(cell not in door_cells for cell in blue_path[: press_step + 1])
            and all(cell not in door_cells for cell in orange_path[: press_step + 1])
            and orange_plate is not None
            and all(cell == tuple(orange_plate) for cell in orange_path[press_step:])
        )
        terminal_ok = bool(
            blue_goal is not None
            and orange_plate is not None
            and blue_path[-1] == tuple(blue_goal)
            and orange_path[-1] == tuple(orange_plate)
        )

    solved = bool(parsed and paths_legal and collision_free and task_semantics_ok and terminal_ok)
    return {
        "text_joint_parse_ok": parsed,
        "text_joint_state_replay_ok": paths_legal and task_semantics_ok,
        "text_joint_collision_and_swap_free": collision_free,
        "text_joint_terminal_ok": terminal_ok,
        "text_replay_check_pass": solved,
    }, solved

def _to_int_grid(payload: Any) -> Optional[List[List[int]]]:
    if not isinstance(payload, list):
        return None
    parsed: List[List[int]] = []
    for row in payload:
        if not isinstance(row, list):
            return None
        try:
            parsed.append([int(cell) for cell in row])
        except Exception:
            return None
    return parsed

def _to_int_pair(payload: Any) -> Optional[List[int]]:
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        return None
    try:
        return [int(payload[0]), int(payload[1])]
    except Exception:
        return None

def _to_int_pair_list(payload: Any) -> Optional[List[List[int]]]:
    if not isinstance(payload, list):
        return None
    parsed: List[List[int]] = []
    for item in payload:
        pair = _to_int_pair(item)
        if pair is None:
            return None
        parsed.append(pair)
    return parsed

def _to_int_tuple_path(payload: Any) -> Optional[List[Tuple[int, int]]]:
    pairs = _to_int_pair_list(payload)
    if pairs is None:
        return None
    return [(int(row), int(col)) for row, col in pairs]

def _derive_hidden_treasure(
    extras: Mapping[str, Any],
    *,
    size: int,
) -> Optional[Tuple[int, int]]:
    clue_cells = _to_int_tuple_path(extras.get("clue_cells"))
    raw_arrows = extras.get("clue_arrows")
    if (
        clue_cells is None
        or len(clue_cells) != 2
        or not isinstance(raw_arrows, list)
        or len(raw_arrows) != 2
    ):
        return None
    directions: List[str] = []
    for raw_arrow in raw_arrows:
        if isinstance(raw_arrow, Mapping):
            direction = str(raw_arrow.get("direction") or "").lower()
        elif isinstance(raw_arrow, (list, tuple)) and len(raw_arrow) == 2:
            direction = str(raw_arrow[1]).lower()
        else:
            return None
        if direction not in _CARDINAL_DIR_TO_DELTA:
            return None
        directions.append(direction)

    first, second = clue_cells
    if first[0] == second[0] or first[1] == second[1]:
        return None
    position_only_candidates = {
        (first[0], second[1]),
        (second[0], first[1]),
    }
    if len(position_only_candidates) != 2:
        return None

    rays: List[set[Tuple[int, int]]] = []
    for cell, direction in zip(clue_cells, directions):
        delta = _CARDINAL_DIR_TO_DELTA[direction]
        ray: set[Tuple[int, int]] = set()
        current = cell
        while True:
            current = (current[0] + delta[0], current[1] + delta[1])
            if not (0 <= current[0] < size and 0 <= current[1] < size):
                break
            ray.add(current)
        rays.append(ray)
    intersection = rays[0] & rays[1]
    if len(intersection) != 1:
        return None
    treasure = next(iter(intersection))
    return treasure if treasure in position_only_candidates else None


def _parse_single_predicted_path(
    structured: Dict[str, Any],
    prediction: str,
    *,
    start: Optional[Tuple[int, int]],
) -> tuple[Optional[List[Tuple[int, int]]], List[str]]:
    try:
        size = int(structured.get("grid_size"))
    except Exception:
        size = 0
    path = _parse_bounded_grid_path(prediction, rows=size, cols=size)
    if path is None or start is None or path[0] != tuple(start):
        return None, []
    return path, []

def _build_agent_mechanic_model(
    structured: Dict[str, Any],
    *,
    task_name: str,
) -> Optional[tuple[Any, Any, Any]]:
    """Build an independent state-transition model for path-answer mechanics.

    The returned successor function emits ``(next_state, expanded_cells)`` for
    one deliberate action. ``expanded_cells`` includes automatic conveyor,
    slide, wind, or teleport substeps, so a submitted trajectory can be
    replayed without trusting the saved canonical path or event indices.
    """

    try:
        size = int(structured.get("grid_size"))
    except Exception:
        return None
    start_raw = _to_int_pair(structured.get("start"))
    goal_raw = _to_int_pair(structured.get("goal"))
    if size <= 0 or start_raw is None or goal_raw is None:
        return None
    start = (int(start_raw[0]), int(start_raw[1]))
    goal = (int(goal_raw[0]), int(goal_raw[1]))
    walls = set(_to_int_tuple_path(structured.get("walls")) or [])
    extras = structured.get("extras") or {}
    if not isinstance(extras, dict):
        return None

    def adjacent(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        for delta_row, delta_col in _DIR_TO_DELTA.values():
            nxt = (cell[0] + delta_row, cell[1] + delta_col)
            if 0 <= nxt[0] < size and 0 <= nxt[1] < size:
                result.append(nxt)
        return result

    if task_name == "bridge_drop_tile":
        water_list = _to_int_tuple_path(extras.get("water"))
        try:
            inventory = int(extras.get("inventory"))
        except Exception:
            return None
        if water_list is None or inventory < 0:
            return None
        water = tuple(water_list)
        water_index = {cell: index for index, cell in enumerate(water)}
        initial = (start, 0)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, dropped_mask = state
            result = []
            for nxt in adjacent(cell):
                if nxt in walls:
                    continue
                new_mask = int(dropped_mask)
                if nxt in water_index and not (new_mask & (1 << water_index[nxt])):
                    if bin(new_mask).count("1") >= inventory:
                        continue
                    new_mask |= 1 << water_index[nxt]
                result.append(((nxt, new_mask), [nxt]))
            return result

        return initial, successors, lambda state: state[0] == goal

    if task_name == "carry_capacity_2":
        cargo_list = _to_int_tuple_path(extras.get("cargo_cells"))
        zone_raw = _to_int_pair(extras.get("zone_cell"))
        try:
            delivery_budget = int(extras.get("delivery_budget"))
        except Exception:
            return None
        if (
            cargo_list is None
            or zone_raw is None
            or not cargo_list
            or delivery_budget <= 0
        ):
            return None
        cargo = tuple(cargo_list)
        cargo_index = {cell: index for index, cell in enumerate(cargo)}
        zone = (int(zone_raw[0]), int(zone_raw[1]))
        all_mask = (1 << len(cargo)) - 1
        initial = (start, all_mask, 0, 0, 0)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, remaining_mask, held_mask, delivered_mask, delivery_commits = state
            result = []
            for nxt in adjacent(cell):
                if nxt in walls:
                    continue
                new_remaining = int(remaining_mask)
                new_held = int(held_mask)
                new_delivered = int(delivered_mask)
                new_delivery_commits = int(delivery_commits)
                cargo_idx = cargo_index.get(nxt)
                if cargo_idx is not None and (new_remaining & (1 << cargo_idx)) and bin(new_held).count("1") < 2:
                    new_remaining &= ~(1 << cargo_idx)
                    new_held |= 1 << cargo_idx
                if nxt == zone and new_held:
                    if new_delivery_commits >= delivery_budget:
                        continue
                    new_delivered |= new_held
                    new_held = 0
                    new_delivery_commits += 1
                result.append(
                    (
                        (
                            nxt,
                            new_remaining,
                            new_held,
                            new_delivered,
                            new_delivery_commits,
                        ),
                        [nxt],
                    )
                )
            return result

        return initial, successors, lambda state: state[0] == goal and state[3] == all_mask

    if task_name == "conveyor_belt_path":
        raw_conveyors = extras.get("conveyors")
        if not isinstance(raw_conveyors, list):
            return None
        conveyors: Dict[Tuple[int, int], str] = {}
        for entry in raw_conveyors:
            if not isinstance(entry, dict):
                return None
            cell_raw = _to_int_pair(entry.get("cell"))
            direction = str(entry.get("direction") or "").lower()
            if cell_raw is None or direction not in _CARDINAL_DIR_TO_DELTA:
                return None
            conveyors[(cell_raw[0], cell_raw[1])] = direction
        initial = (start,)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell = state[0]
            result = [((cell,), [cell])]
            for delta in _CARDINAL_DIR_TO_DELTA.values():
                nxt = (cell[0] + delta[0], cell[1] + delta[1])
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                    continue
                if nxt in walls:
                    continue
                expanded = [nxt]
                rest = nxt
                applied: set[Tuple[int, int]] = set()
                while rest in conveyors and rest not in applied:
                    applied.add(rest)
                    delta = _DIR_TO_DELTA[conveyors[rest]]
                    shoved = (rest[0] + delta[0], rest[1] + delta[1])
                    if not (0 <= shoved[0] < size and 0 <= shoved[1] < size) or shoved in walls:
                        break
                    expanded.append(shoved)
                    rest = shoved
                result.append(((rest,), expanded))
            return result

        return initial, successors, lambda state: state[0] == goal

    if task_name == "directional_arrow_entry":
        raw_arrows = extras.get("arrows")
        if not isinstance(raw_arrows, list):
            return None
        arrows: Dict[Tuple[int, int], str] = {}
        for entry in raw_arrows:
            if not isinstance(entry, dict):
                return None
            cell_raw = _to_int_pair(entry.get("cell"))
            direction = str(entry.get("direction") or "").lower()
            if cell_raw is None or direction not in _CARDINAL_DIR_TO_DELTA:
                return None
            arrows[(cell_raw[0], cell_raw[1])] = direction
        initial = (start,)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell = state[0]
            result = [((cell,), [cell])]
            for direction, delta in _CARDINAL_DIR_TO_DELTA.items():
                nxt = (cell[0] + delta[0], cell[1] + delta[1])
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
                    continue
                if nxt not in arrows:
                    result.append(((nxt,), [nxt]))
                    continue
                if arrows[nxt] != direction:
                    continue
                forced = (nxt[0] + delta[0], nxt[1] + delta[1])
                if not (0 <= forced[0] < size and 0 <= forced[1] < size) or forced in walls:
                    continue
                if forced in arrows and arrows[forced] != direction:
                    continue
                result.append(((forced,), [nxt, forced]))
            return result

        reachable_without_arrows = {start}
        queue_without_arrows = [start]
        for cell in queue_without_arrows:
            for delta in _CARDINAL_DIR_TO_DELTA.values():
                nxt = (cell[0] + delta[0], cell[1] + delta[1])
                if (
                    not (0 <= nxt[0] < size and 0 <= nxt[1] < size)
                    or nxt in walls
                    or nxt in arrows
                    or nxt in reachable_without_arrows
                ):
                    continue
                reachable_without_arrows.add(nxt)
                queue_without_arrows.append(nxt)
        if goal in reachable_without_arrows:
            return None

        best_commits: Dict[Any, int] = {initial: 0}
        pending = [initial]
        while pending:
            state = min(pending, key=lambda item: best_commits[item])
            pending.remove(state)
            current_cost = best_commits[state]
            for next_state, expanded in successors(state):
                if next_state == state:
                    continue
                next_cost = current_cost + sum(cell in arrows for cell in expanded)
                if next_cost >= best_commits.get(next_state, size * size + 1):
                    continue
                best_commits[next_state] = next_cost
                pending.append(next_state)
        recomputed_minimum = min(
            (
                commits
                for state, commits in best_commits.items()
                if state[0] == goal
            ),
            default=None,
        )
        if recomputed_minimum is None or recomputed_minimum < 1:
            return None

        return initial, successors, lambda state: state[0] == goal

    if task_name == "ice_physics_multi_slide":
        ice_list = _to_int_tuple_path(extras.get("ice"))
        if ice_list is None:
            return None
        ice = set(ice_list)
        initial = (start,)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell = state[0]
            result = [((cell,), [cell])]
            for delta in _CARDINAL_DIR_TO_DELTA.values():
                nxt = (cell[0] + delta[0], cell[1] + delta[1])
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
                    continue
                expanded = [nxt]
                rest = nxt
                if nxt in ice:
                    while True:
                        after = (rest[0] + delta[0], rest[1] + delta[1])
                        if not (0 <= after[0] < size and 0 <= after[1] < size) or after in walls:
                            break
                        expanded.append(after)
                        rest = after
                        if rest not in ice:
                            break
                result.append(((rest,), expanded))
            return result

        return initial, successors, lambda state: state[0] == goal

    if task_name == "lantern_oil_route":
        refill_list = _to_int_tuple_path(extras.get("refills"))
        try:
            max_oil = int(extras.get("max_oil"))
        except Exception:
            return None
        if refill_list is None or max_oil <= 0:
            return None
        refills = set(refill_list)
        initial = (start, max_oil)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, oil = state
            result = []
            stayed_oil = int(oil) - 1
            if stayed_oil > 0:
                result.append(((cell, stayed_oil), [cell]))
            for nxt in adjacent(cell):
                if nxt == cell:
                    continue
                if nxt in walls:
                    continue
                new_oil = int(oil) - 1
                if new_oil <= 0:
                    continue
                if nxt in refills:
                    new_oil = max_oil
                result.append(((nxt, new_oil), [nxt]))
            return result

        return initial, successors, lambda state: state[0] == goal

    if task_name == "lever_unlock_sequence":
        lever_list = _to_int_tuple_path(extras.get("lever_cells"))
        door_raw = _to_int_pair(extras.get("door_cell"))
        if lever_list is None or door_raw is None or not lever_list:
            return None
        levers = tuple(lever_list)
        lever_index = {cell: index for index, cell in enumerate(levers)}
        door = (int(door_raw[0]), int(door_raw[1]))
        initial = (start, 0)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, progress = state
            result = []
            for nxt in adjacent(cell):
                if nxt in walls or (nxt == door and progress < len(levers)):
                    continue
                new_progress = int(progress)
                lever_idx = lever_index.get(nxt)
                # The visible rule says that visiting a later lever before its
                # turn breaks the sequence; it is not an ignorable visit.
                if lever_idx is not None and lever_idx > new_progress:
                    continue
                if lever_idx == new_progress:
                    new_progress += 1
                result.append(((nxt, new_progress), [nxt]))
            return result

        return initial, successors, lambda state: state[0] == goal and state[1] == len(levers)

    if task_name == "one_way_portal_chain":
        raw_pairs = extras.get("portal_pairs")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            return None
        portal_map: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]] = {}
        for index, entry in enumerate(raw_pairs):
            if not isinstance(entry, dict):
                return None
            source_raw = _to_int_pair(entry.get("source"))
            destination_raw = _to_int_pair(entry.get("destination"))
            if source_raw is None or destination_raw is None:
                return None
            source = (source_raw[0], source_raw[1])
            destination = (destination_raw[0], destination_raw[1])
            portal_map[source] = (destination, index)
        initial = (start, 0)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, progress = state
            result = [((cell, int(progress)), [cell])]
            for direction, delta in _DIR_TO_DELTA.items():
                if direction == "stay":
                    continue
                nxt = (cell[0] + delta[0], cell[1] + delta[1])
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                    continue
                if nxt in walls:
                    continue
                expanded = [nxt]
                rest = nxt
                new_progress = int(progress)
                portal = portal_map.get(nxt)
                if portal is not None:
                    rest, portal_index = portal
                    if portal_index != new_progress:
                        continue
                    expanded.append(rest)
                    new_progress += 1
                result.append(((rest, new_progress), expanded))
            return result

        return initial, successors, lambda state: state[0] == goal and state[1] == len(raw_pairs)

    if task_name == "switch_toggle_walls":
        switch_raw = _to_int_pair(extras.get("switch_cell"))
        walls_a_list = _to_int_tuple_path(extras.get("walls_a"))
        walls_b_list = _to_int_tuple_path(extras.get("walls_b"))
        if switch_raw is None or walls_a_list is None or walls_b_list is None:
            return None
        switch = (switch_raw[0], switch_raw[1])
        walls_a = set(walls_a_list)
        walls_b = set(walls_b_list)
        initial = (start, False)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, switched = state
            blocked = walls_b if switched else walls_a
            result = []
            for nxt in adjacent(cell):
                if nxt in blocked:
                    continue
                new_switched = bool(switched or nxt == switch)
                result.append(((nxt, new_switched), [nxt]))
            return result

        return initial, successors, lambda state: state[0] == goal and bool(state[1])

    if task_name == "heading_turn_budget_route":
        initial_heading = str(extras.get("initial_heading") or "").lower()
        final_heading = str(extras.get("final_heading") or "").lower()
        try:
            turn_budget = int(extras.get("turn_budget"))
        except Exception:
            return None
        compass = ("north", "east", "south", "west")
        heading_for_direction = {"up": "north", "right": "east", "down": "south", "left": "west"}
        if initial_heading not in compass or final_heading not in compass or turn_budget < 0:
            return None
        initial = (start, compass.index(initial_heading), 0)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, heading_index, turns = state
            result = []
            result.append(((cell, int(heading_index), int(turns)), [cell]))
            for direction, delta in _CARDINAL_DIR_TO_DELTA.items():
                nxt = (cell[0] + delta[0], cell[1] + delta[1])
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
                    continue
                next_heading = compass.index(heading_for_direction[direction])
                difference = (next_heading - int(heading_index)) % 4
                if difference == 2:
                    continue
                next_turns = int(turns) + (1 if difference in {1, 3} else 0)
                if next_turns > turn_budget:
                    continue
                result.append(((nxt, next_heading, next_turns), [nxt]))
            return result

        return initial, successors, lambda state: state[0] == goal and compass[state[1]] == final_heading

    return None

def _build_primary_mechanic_model(
    structured: Dict[str, Any],
    *,
    task_name: str,
) -> Optional[tuple[Any, Any, Any]]:
    """Independent transition models for single-primary-path spatial_navigation tasks."""

    try:
        size = int(structured.get("grid_size"))
    except Exception:
        return None
    extras = structured.get("extras") or {}
    if size <= 0 or not isinstance(extras, dict):
        return None
    start_raw = _to_int_pair(extras.get("agent_start"))
    if start_raw is None:
        start_raw = _to_int_pair(extras.get("bus_start"))
    if start_raw is None:
        return None
    start = (start_raw[0], start_raw[1])
    walls = set(_to_int_tuple_path(structured.get("walls")) or [])

    def adjacent(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        return [
            (cell[0] + delta_row, cell[1] + delta_col)
            for delta_row, delta_col in _DIR_TO_DELTA.values()
            if 0 <= cell[0] + delta_row < size and 0 <= cell[1] + delta_col < size
        ]

    if task_name == "bus_route_passengers":
        targets_raw = [
            _to_int_pair(extras.get("stop1")),
            _to_int_pair(extras.get("stop2")),
            _to_int_pair(extras.get("stop3")),
            _to_int_pair(extras.get("depot")),
        ]
        if any(target is None for target in targets_raw):
            return None
        targets = tuple((target[0], target[1]) for target in targets_raw if target is not None)
        stop_index = {cell: index for index, cell in enumerate(targets[:-1])}
        initial = (start, 0)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, progress = state
            result = []
            for nxt in adjacent(cell):
                if nxt in walls:
                    continue
                new_progress = int(progress)
                visited_stop = stop_index.get(nxt)
                if visited_stop is not None and visited_stop > new_progress:
                    continue
                if new_progress < len(targets) and nxt == targets[new_progress]:
                    new_progress += 1
                result.append(((nxt, new_progress), [nxt]))
            return result

        return initial, successors, lambda state: state[0] == targets[-1] and state[1] == len(targets)

    if task_name in {"color_matched_collection", "parcel_delivery_route"}:
        pickup_key = "balls" if task_name == "color_matched_collection" else "pickups"
        destination_key = "slots" if task_name == "color_matched_collection" else "dests"
        pickup_list = _to_int_tuple_path(extras.get(pickup_key))
        destination_list = _to_int_tuple_path(extras.get(destination_key))
        if pickup_list is None or destination_list is None or len(pickup_list) != len(destination_list) or not pickup_list:
            return None
        pickups = tuple(pickup_list)
        destinations = tuple(destination_list)
        pickup_index = {cell: index for index, cell in enumerate(pickups)}
        all_mask = (1 << len(pickups)) - 1
        initial = (start, all_mask, -1, 0)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, remaining_mask, held_index, delivered_mask = state
            result = []
            for nxt in adjacent(cell):
                if nxt in walls:
                    continue
                new_remaining = int(remaining_mask)
                new_held = int(held_index)
                new_delivered = int(delivered_mask)
                pickup_idx = pickup_index.get(nxt)
                if new_held < 0 and pickup_idx is not None and (new_remaining & (1 << pickup_idx)):
                    new_remaining &= ~(1 << pickup_idx)
                    new_held = pickup_idx
                if new_held >= 0 and nxt == destinations[new_held]:
                    new_delivered |= 1 << new_held
                    new_held = -1
                result.append(((nxt, new_remaining, new_held, new_delivered), [nxt]))
            return result

        return initial, successors, lambda state: state[3] == all_mask and state[2] < 0

    if task_name == "hidden_treasure_dig":
        goal = _derive_hidden_treasure(extras, size=size)
        if goal is None or goal in walls:
            return None
        blocked = set(walls)
        initial = (start,)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            result = []
            for nxt in adjacent(state[0]):
                if nxt in blocked:
                    continue
                result.append(((nxt,), [nxt]))
            return result

        return initial, successors, lambda state: state[0] == goal

    if task_name == "falling_block_dodge":
        goal_raw = _to_int_pair(extras.get("goal"))
        raw_blocks = extras.get("falling_blocks")
        if goal_raw is None or not isinstance(raw_blocks, list) or not raw_blocks:
            return None
        goal = (goal_raw[0], goal_raw[1])
        blocks: List[Dict[str, int]] = []
        clear_step = 0
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                return None
            try:
                spec = {
                    "column": int(raw["column"]),
                    "start_row": int(raw["start_row"]),
                }
            except Exception:
                return None
            blocks.append(spec)
            clear_step = max(clear_step, size - spec["start_row"])

        def block_cell(spec: Dict[str, int], step: int) -> Optional[Tuple[int, int]]:
            row = spec["start_row"] + step
            return (row, spec["column"]) if 0 <= row < size else None

        initial = (start, 0)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, step = state
            next_step = int(step) + 1
            current_blocks = [block_cell(spec, int(step)) for spec in blocks]
            next_blocks = [block_cell(spec, next_step) for spec in blocks]
            result = []
            for nxt in adjacent(cell):
                if nxt in walls or nxt in current_blocks or nxt in next_blocks:
                    continue
                if any(
                    current is not None and after is not None and cell == after and nxt == current
                    for current, after in zip(current_blocks, next_blocks)
                ):
                    continue
                result.append(((nxt, min(next_step, clear_step)), [nxt]))
            return result

        return initial, successors, lambda state: state[0] == goal

    if task_name == "patrol_avoid_pattern":
        goal_raw = _to_int_pair(extras.get("goal"))
        patrol_list = _to_int_tuple_path(extras.get("patrol_loop"))
        if goal_raw is None or patrol_list is None or len(patrol_list) < 2:
            return None
        goal = (goal_raw[0], goal_raw[1])
        patrol = tuple(patrol_list)
        if any(not (0 <= cell[0] < size and 0 <= cell[1] < size) for cell in patrol):
            return None
        initial = (start, 0)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, phase = state
            next_phase = (int(phase) + 1) % len(patrol)
            current_patrol = patrol[int(phase)]
            result = []
            for nxt in adjacent(cell):
                if nxt in walls or nxt == patrol[next_phase]:
                    continue
                if cell == patrol[next_phase] and nxt == current_patrol:
                    continue
                result.append(((nxt, next_phase), [nxt]))
            return result

        return initial, successors, lambda state: state[0] == goal

    if task_name == "tide_water_cycle":
        goal_raw = _to_int_pair(extras.get("goal"))
        if goal_raw is None:
            return None
        goal = (goal_raw[0], goal_raw[1])
        try:
            row_one = int(extras.get("tide_row1"))
            row_two = int(extras.get("tide_row2"))
        except Exception:
            return None
        row_one_blocked_parity, row_two_blocked_parity = 0, 1
        initial = (start, 0)

        def successors(state: Any) -> List[tuple[Any, List[Tuple[int, int]]]]:
            cell, parity = state
            next_parity = 1 - int(parity)
            result = []
            for nxt in adjacent(cell):
                if nxt in walls:
                    continue
                old_blocked = row_one if int(parity) == row_one_blocked_parity else row_two
                new_blocked = row_one if next_parity == row_one_blocked_parity else row_two
                if cell[0] == new_blocked and nxt[0] == old_blocked:
                    continue
                if nxt[0] == row_one and next_parity == row_one_blocked_parity:
                    continue
                if nxt[0] == row_two and next_parity == row_two_blocked_parity:
                    continue
                result.append(((nxt, next_parity), [nxt]))
            return result

        return initial, successors, lambda state: state[0] == goal

    return None

def _path_possible_with_hidden_stays(
    model: tuple[Any, Any, Any], path: Sequence[Tuple[int, int]],
) -> bool:
    """Test reachability with stationary actions inserted between observations.

    This is a finite state/index search in the task's transition model, not a
    search over every placement of a guessed number of waits. If it fails, no
    insertion of stationary actions can make the observed route legal.
    """

    if not path:
        return False
    initial, successors, is_goal = model
    if path[0] != initial[0]:
        return False
    frontier = [(initial, 1)]
    visited = set(frontier)
    for state, index in frontier:
        if index == len(path) and is_goal(state):
            return True
        if is_goal(state):
            continue
        for nxt, expanded in successors(state):
            indices = []
            if list(expanded) == [state[0]]:
                indices.append(index)
            end = index + len(expanded)
            if end <= len(path) and list(path[index:end]) == list(expanded):
                indices.append(end)
            for next_index in indices:
                key = (nxt, next_index)
                if key not in visited:
                    visited.add(key)
                    frontier.append(key)
    return False


def _replay_transition_model_path(model: tuple[Any, Any, Any], path: Sequence[Tuple[int, int]]) -> tuple[Optional[int], bool]:
    if not path:
        return None, False
    initial, successors, is_goal = model
    if path[0] != initial[0]:
        return None, False
    frontier: Dict[tuple[Any, int], int] = {(initial, 1): 0}
    while frontier:
        next_frontier: Dict[tuple[Any, int], int] = {}
        progressed = False
        for (state, path_index), action_count in frontier.items():
            if path_index == len(path):
                next_frontier[(state, path_index)] = action_count
                continue
            if is_goal(state):
                continue
            for next_state, expanded in successors(state):
                end_index = path_index + len(expanded)
                if end_index > len(path) or list(path[path_index:end_index]) != list(expanded):
                    continue
                progressed = True
                key = (next_state, end_index)
                previous = next_frontier.get(key)
                if previous is None or action_count + 1 < previous:
                    next_frontier[key] = action_count + 1
        frontier = next_frontier
        if not progressed:
            break
    completed = [
        count for (state, index), count in frontier.items() if index == len(path) and is_goal(state)
    ]
    return (min(completed), True) if completed else (None, False)

def _replay_agent_mechanic_path(
    structured: Dict[str, Any],
    *,
    task_name: str,
    path: Sequence[Tuple[int, int]],
) -> tuple[Optional[int], bool]:
    model = _build_agent_mechanic_model(structured, task_name=task_name)
    if model is None or not path:
        return None, False
    initial, successors, is_goal = model
    if path[0] != initial[0]:
        return None, False
    frontier: Dict[tuple[Any, int], int] = {(initial, 1): 0}
    while frontier:
        next_frontier: Dict[tuple[Any, int], int] = {}
        progressed = False
        for (state, path_index), action_count in frontier.items():
            if path_index == len(path):
                next_frontier[(state, path_index)] = min(
                    action_count,
                    next_frontier.get((state, path_index), action_count),
                )
                continue
            if is_goal(state):
                continue
            for next_state, expanded in successors(state):
                end_index = path_index + len(expanded)
                if end_index > len(path) or list(path[path_index:end_index]) != list(expanded):
                    continue
                progressed = True
                key = (next_state, end_index)
                previous = next_frontier.get(key)
                if previous is None or action_count + 1 < previous:
                    next_frontier[key] = action_count + 1
        frontier = next_frontier
        if not progressed:
            break
    completed = [
        action_count
        for (state, path_index), action_count in frontier.items()
        if path_index == len(path) and is_goal(state)
    ]
    return (min(completed), True) if completed else (None, False)
