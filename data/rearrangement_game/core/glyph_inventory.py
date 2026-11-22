"""Bounded rendered-glyph contracts for Rearrangement Game."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


_TEXT_FREE_TASKS = frozenset({
    "bookshelf_sort_by_height",
    "chess_pawn_swap",
    "meal_train_pass_serve",
    "pancake_flip",
    "tower_of_hanoi",
})


def _strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=lambda value: (len(value), value))


def _bounded_scalars(value: Any, output: set[str]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        if -1 <= value <= 99:
            output.add(str(value))
        return
    if isinstance(value, float):
        if -99.0 <= value <= 99.0:
            output.add(f"{value:g}")
        return
    if isinstance(value, str):
        text = value.strip()
        if text and len(text) <= 3 and text.replace("_", "").isalnum():
            output.add(text)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _bounded_scalars(item, output)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _bounded_scalars(item, output)


def _range_labels(count: Any, *, start: int = 1, prefix: str = "") -> set[str]:
    """Declare the ordinal labels a renderer actually draws.

    Every model-facing ordinal in this repository is 1-based,
    so the default start is 1.  Declaring 0-based labels here would over-report
    glyphs that are never drawn, which would describe invisible state.
    """
    if type(count) is not int or count < 0 or count > 99:
        return set()
    return {f"{prefix}{index}" for index in range(start, int(count) + start)}


def _rendered_labels(record: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Mirror the actual task renderers instead of scraping arbitrary state.

    This explicit switch is intentionally fail-closed: adding a task or a new
    rendered label requires updating its bounded inventory here.
    """

    task = str(record.get("task_name") or "")
    fixed: set[str] = set()
    dynamic: set[str] = set()
    if task in _TEXT_FREE_TASKS:
        return fixed, dynamic
    if task == "bus_stops_dispatch":
        dynamic |= _range_labels(record.get("n_stops"), start=1)
    elif task == "color_group_with_locks":
        dynamic |= _range_labels(record.get("slot_count"))
    elif task == "container_ship_loading_order":
        dynamic |= _range_labels(record.get("container_count"), start=1)
    elif task == "crane_block_stack":
        dynamic |= _range_labels(record.get("block_count"), start=1)
    elif task == "dock_yard_crane_swap":
        fixed |= {"L", "R"}
        dynamic |= _range_labels(record.get("box_count"), start=1)
    elif task == "filing_cabinet_alphabet_lock":
        fixed |= {str(value) for value in record.get("drawers", [])}
        state = record.get("initial_state") or {}
        _bounded_scalars(
            list((state.get("drawers") or {}).values()) if isinstance(state, Mapping) else [],
            dynamic,
        )
    elif task == "jumping_frog_swap":
        state = record.get("initial_state") or []
        dynamic |= _range_labels(len(state) if isinstance(state, (list, tuple)) else None)
    elif task == "lever_crane_swap":
        state = record.get("initial_state") or []
        count = sum(len(stack) for stack in state) if isinstance(state, (list, tuple)) else 0
        dynamic |= _range_labels(count, start=1)
    elif task == "maze_rush_hour":
        fixed |= {f"C{int(car['id'])}" for car in record.get("cars", []) if isinstance(car, Mapping) and type(car.get("id")) is int}
    elif task == "office_inbox_priority_pull":
        dynamic |= _range_labels(record.get("paper_count"), start=1)
    elif task == "parking_2d_n_by_n":
        _bounded_scalars(record.get("initial_state", []), dynamic)
    elif task == "parking_lot_swap_minimal_moves":
        dynamic |= _range_labels(record.get("car_count"), start=1)
    elif task == "pipeline_hold_cells_3":
        dynamic |= _range_labels(record.get("product_count"), start=1)
    elif task == "sliding_tile_15":
        fixed |= {str(value) for value in range(1, 16)}
    elif task == "stack_reversal_minimal_moves":
        dynamic |= _range_labels(record.get("block_count"), start=1)
    elif task == "token_trade_matrix":
        dynamic |= _range_labels(record.get("n_positions"))
        _bounded_scalars((record.get("initial_state") or {}).get("holdings", []), dynamic)
    elif task in {"train_couple_swap", "vending_dispense_fifo"}:
        count = record.get("car_count") if task == "train_couple_swap" else record.get("can_count")
        dynamic |= _range_labels(count, start=1)
    else:
        raise ValueError(f"missing rendered-glyph contract for Rearrangement task {task!r}")
    dynamic -= fixed
    return fixed, dynamic


def rearrangement_rendered_glyph_contract(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    fixed, dynamic = _rendered_labels(record)
    inventory = {
        "schema_version": "rearrangement_rendered_glyph_inventory_v1",
        "fixed_labels": _strings(fixed),
        "dynamic_labels": _strings(dynamic),
        "all_rendered_text": _strings(fixed | dynamic),
    }
    schema = {
        "fixed_labels": "task-bounded lane, drawer, vehicle, or direction identities",
        "dynamic_labels": "sample-bounded object IDs, sizes, priorities, or tile numbers",
        "exclusions": "no task title, rule prose, solver step, changing subtitle, status sentence, or answer banner may be rendered",
    }
    return inventory, schema


def annotate_rearrangement_glyphs(record: Mapping[str, Any]) -> dict[str, Any]:
    annotated = dict(record)
    inventory, schema = rearrangement_rendered_glyph_contract(annotated)
    annotated["rendered_glyph_inventory"] = inventory
    annotated["rendered_glyph_schema"] = schema
    return annotated


__all__ = [
    "annotate_rearrangement_glyphs",
    "rearrangement_rendered_glyph_contract",
]
