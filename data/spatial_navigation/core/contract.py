"""Shared Spatial Navigation task-contract constants."""

from __future__ import annotations


# Tasks whose formal route answer is one strict list of model-facing 1-based
# ``(x, y)`` matrix positions.  The path includes both its visible start and
# final position; a repeated position is a legal stay where the task permits
# one.  Candidate parsers fail closed on the removed direction, action,
# linear-cell-ID, and pipe-separated representations.
SINGLE_XY_ROUTE_TASKS = frozenset(
    {
        "bridge_drop_tile",
        "bus_route_passengers",
        "carry_capacity_2",
        "color_matched_collection",
        "conveyor_belt_path",
        "directional_arrow_entry",
        "falling_block_dodge",
        "grid_highest_cost",
        "grid_obtaining_award",
        "heading_turn_budget_route",
        "hidden_treasure_dig",
        "ice_physics_multi_slide",
        "lantern_oil_route",
        "lever_unlock_sequence",
        "one_way_portal_chain",
        "parcel_delivery_route",
        "patrol_avoid_pattern",
        "plant_watering_route",
        "switch_toggle_walls",
        "tide_water_cycle",
    }
)

# These tasks submit two independently named coordinate paths.  Each named
# list follows the same complete-path contract as a single route.
NAMED_XY_ROUTE_TASKS = frozenset(
    {
        "agent_courier_relay",
        "cooperative_open_door",
    }
)

XY_ROUTE_TASKS = SINGLE_XY_ROUTE_TASKS | NAMED_XY_ROUTE_TASKS


__all__ = [
    "NAMED_XY_ROUTE_TASKS",
    "SINGLE_XY_ROUTE_TASKS",
    "XY_ROUTE_TASKS",
]
