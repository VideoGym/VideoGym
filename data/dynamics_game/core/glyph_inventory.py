"""Per-sample rendered-text inventories for the Dynamics family.

The inventory is attached after media normalization.  It records only glyphs
that the renderer can actually place on the input, cumulative states, terminal
image, or video; geometric arrows, coloured pieces, dots, and icons are not
font glyphs and therefore are intentionally absent.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


_CHOICE_TASKS = {
    "block_blast_clear_predict",
    "breakout_brick_clear_predict",
    "bubble_shooter_chain_predict",
    "candy_crush_special_predict",
    "jewel2_future_board",
    "lane_tap_step",
    "lemmings_path_predict",
    "mr_driller_drop_predict",
    "pegboard_marble_drop_predict",
}


def _semantic_state(record: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("structured_answer", "answer_payload"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values}, key=lambda value: (len(value), value))


def _grid_values(state: Mapping[str, Any]) -> list[str]:
    values: set[int] = set()
    candidates = [state.get("initial_grid"), *(state.get("state_trace") or [])]
    for grid in candidates:
        if not isinstance(grid, list):
            continue
        for row in grid:
            if not isinstance(row, list):
                continue
            for cell in row:
                try:
                    number = int(cell)
                except (TypeError, ValueError):
                    continue
                if number > 0:
                    values.add(number)
    return _strings(values)


def dynamics_rendered_glyph_contract(record: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    task = str(record.get("task_name") or "")
    state = _semantic_state(record)
    fixed: set[str] = set()
    dynamic: set[str] = set()

    if task in _CHOICE_TASKS:
        fixed.update(("A", "B", "C", "D"))
    if task == "block_blast_clear_predict":
        fixed.update(str(index) for index in range(1, 9))
    elif task == "breakout_brick_clear_predict":
        fixed.update(str(index) for index in range(1, 11))
        fixed.update(str(index) for index in range(1, 8))
    elif task == "bubble_shooter_chain_predict":
        fixed.update(str(index) for index in range(1, 9))
    elif task == "candy_crush_special_predict":
        fixed.update(str(index) for index in range(1, 8))
    elif task == "jewel2_future_board":
        initial = state.get("initial_grid")
        rows = len(initial) if isinstance(initial, list) else 0
        cols = len(initial[0]) if rows and isinstance(initial[0], list) else 0
        fixed.update(str(index) for index in range(1, max(rows, cols) + 1))
    elif task == "lane_tap_step":
        fixed.update(str(index) for index in range(1, 8))
        dynamic.update(
            f"↓{int(speed)}"
            for speed in state.get("lane_speeds", [])
            if isinstance(speed, (int, float))
        )
    elif task == "lemmings_path_predict":
        fixed.update(str(index) for index in range(1, 11))
    elif task == "merge_swipe_score":
        fixed.update(("1", "2", "3", "4"))
        dynamic.update(_grid_values(state))
    elif task == "pegboard_marble_drop_predict":
        fixed.update(str(index) for index in range(1, 8))
        fixed.update(("A", "B", "C", "D"))
    elif task == "sokoban":
        fixed.update(str(index) for index in range(1, 8))
    elif task == "tetris_future_board":
        fixed.update(str(index) for index in range(1, 11))
    elif task == "track_match_shot":
        fixed.update(str(index) for index in range(1, 9))

    inventory: Dict[str, Any] = {
        "schema_version": "dynamics_rendered_glyph_inventory_v2",
        "fixed_labels": _strings(fixed),
        "dynamic_labels": _strings(dynamic),
        "all_rendered_text": _strings(fixed | dynamic),
    }
    schema = {
        "fixed_labels": "task-bounded coordinate, round, candidate, or status labels listed exactly in the inventory",
        "dynamic_labels": "sample-derived tile values, lane speeds, or policy letter listed exactly in the inventory",
        "exclusions": "no task title, rule prose, answer banner, changing subtitle, or step sentence may be rendered",
    }
    return inventory, schema


def annotate_dynamics_glyphs(record: Mapping[str, Any]) -> Dict[str, Any]:
    annotated = dict(record)
    inventory, schema = dynamics_rendered_glyph_contract(annotated)
    annotated["rendered_glyph_inventory"] = inventory
    annotated["rendered_glyph_schema"] = schema
    return annotated


__all__ = ["annotate_dynamics_glyphs", "dynamics_rendered_glyph_contract"]
