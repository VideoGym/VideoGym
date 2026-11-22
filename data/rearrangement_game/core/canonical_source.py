"""Source-only canonical inputs for Rearrangement Game.

The native generators retain their authored plan, state trace, selected
indices, schedule, shortest-path diagnostics, and reference-render metadata.
Those values are useful for reference production but are not verifier input.
This module therefore uses an explicit task allowlist.  A genuinely visible
goal panel is serialized as ``visible_goal_state``; a blank, masked, or merely
"all done" panel is not.  For the latter tasks the verifier must reconstruct
the terminal state from the visible initial state and the public rule.  A
newly added solver/output field cannot silently cross the canonical boundary.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


OBJECT_PLAN_TASKS = frozenset({
    "bookshelf_sort_by_height",
    "bus_stops_dispatch",
    "chess_pawn_swap",
    "color_group_with_locks",
    "container_ship_loading_order",
    "crane_block_stack",
    "dock_yard_crane_swap",
    "filing_cabinet_alphabet_lock",
    "jumping_frog_swap",
    "lever_crane_swap",
    "meal_train_pass_serve",
    "office_inbox_priority_pull",
    "pancake_flip",
    "parking_2d_n_by_n",
    "parking_lot_swap_minimal_moves",
    "pipeline_hold_cells_3",
    "stack_reversal_minimal_moves",
    "token_trade_matrix",
    "tower_of_hanoi",
    "train_couple_swap",
    "vending_dispense_fifo",
})


# These generators either render a single current-state scene, replace the
# answer-bearing goal with a neutral mask, or render only a coarse all-complete
# condition whose list order does not appear in the pixels.  Their saved
# ``target_state`` is therefore a derived oracle, not an input fact.
DERIVED_GOAL_TASKS = frozenset({
    "meal_train_pass_serve",
    "office_inbox_priority_pull",
})


# Only visible rule parameters which are not already unambiguously encoded in
# the two state panels belong here.  In particular this list contains no
# canonical move, selected subset, target schedule, shortest path, trace,
# reference keyframe, final answer, or solver diagnostic.
VISIBLE_PARAMETER_FIELDS: dict[str, tuple[str, ...]] = {
    "bookshelf_sort_by_height": ("book_count",),
    "bus_stops_dispatch": ("n_stops", "capacity"),
    "chess_pawn_swap": ("rows", "cols", "pawn_count_per_color"),
    "color_group_with_locks": ("slot_count", "locked_indices"),
    "container_ship_loading_order": ("container_count",),
    "crane_block_stack": ("column_count", "block_count"),
    "dock_yard_crane_swap": ("lane_count", "box_count"),
    "filing_cabinet_alphabet_lock": ("drawers", "locked"),
    "jumping_frog_swap": ("green_count", "orange_count"),
    "lever_crane_swap": (),
    "meal_train_pass_serve": ("num_seats", "start_seat", "demands", "empty_seats"),
    "office_inbox_priority_pull": ("paper_count", "initial_inbox_bottom_to_top"),
    "pancake_flip": ("pancake_count",),
    "parking_2d_n_by_n": ("rows", "cols", "empty_count"),
    "parking_lot_swap_minimal_moves": ("slot_count", "car_count"),
    "pipeline_hold_cells_3": ("product_count",),
    "stack_reversal_minimal_moves": ("block_count", "buffer_count", "primary_column"),
    "token_trade_matrix": ("n_positions", "edges"),
    "tower_of_hanoi": ("peg_count", "disk_count"),
    "train_couple_swap": ("car_count",),
    "vending_dispense_fifo": ("can_count",),
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Rearrangement canonical source lacks {label}")
    return value


def _object_plan_source(task_name: str, record: Mapping[str, Any]) -> dict[str, Any]:
    if "initial_state" not in record or "target_state" not in record:
        raise ValueError(f"{task_name} must expose both visible state panels")
    fields = VISIBLE_PARAMETER_FIELDS[task_name]
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"{task_name} lacks visible source fields: {missing}")
    source = {
        "visible_initial_state": deepcopy(record["initial_state"]),
        "visible_parameters": {field: deepcopy(record[field]) for field in fields},
    }
    if task_name not in DERIVED_GOAL_TASKS:
        source["visible_goal_state"] = deepcopy(record["target_state"])
    return source


def _sliding_tile_source(record: Mapping[str, Any]) -> dict[str, Any]:
    initial = deepcopy(record.get("initial_grid"))
    size = int(record.get("board_size") or 0)
    if size != 4 or not isinstance(initial, list) or len(initial) != size:
        raise ValueError("sliding_tile_15 source must be one visible 4x4 board")
    return {
        "visible_initial_grid": initial,
        "board_size": size,
    }


def _rush_hour_source(record: Mapping[str, Any]) -> dict[str, Any]:
    cars = record.get("cars")
    if not isinstance(cars, list) or not cars:
        raise ValueError("maze_rush_hour lacks visible vehicles")
    visible_cars = []
    for raw in cars:
        car = _mapping(raw, "rush-hour vehicle")
        fields = ("id", "row", "col", "orientation", "length", "color")
        if any(field not in car for field in fields):
            raise ValueError("rush-hour vehicle lacks a visible identity or geometry field")
        visible_cars.append({field: deepcopy(car[field]) for field in fields})
    grid_size = deepcopy(record.get("grid_size"))
    exit_row = record.get("exit_row")
    if grid_size != [6, 6] or type(exit_row) is not int:
        raise ValueError("maze_rush_hour source requires the visible 6x6 lot and exit row")
    return {
        "grid_size": grid_size,
        "exit_row": int(exit_row),
        "cars": visible_cars,
    }


def canonical_rearrangement_semantic_input(
    task_name: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the minimal executable source for one Rearrangement sample."""

    source = _mapping(record, f"{task_name} record")
    if task_name in OBJECT_PLAN_TASKS:
        return _object_plan_source(task_name, source)
    if task_name == "sliding_tile_15":
        return _sliding_tile_source(source)
    if task_name == "maze_rush_hour":
        return _rush_hour_source(source)
    raise ValueError(f"No Rearrangement canonical allowlist for {task_name!r}")


__all__ = [
    "DERIVED_GOAL_TASKS",
    "OBJECT_PLAN_TASKS",
    "VISIBLE_PARAMETER_FIELDS",
    "canonical_rearrangement_semantic_input",
]
