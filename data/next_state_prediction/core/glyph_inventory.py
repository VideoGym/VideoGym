"""Dynamic rendered-glyph inventories for all next-state tasks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from data.next_state_prediction.core.catalog import EXPANSION_TASK_NAMES
from data.next_state_prediction.core.rendering import rendered_glyph_inventory


def _state(record: Mapping[str, Any]) -> Dict[str, Any]:
    value = record.get("structured_answer") or record.get("answer_payload")
    return dict(value) if isinstance(value, Mapping) else {}


def _sort(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values}, key=lambda value: (len(value), value))


def _matrix_values(*matrices: Any) -> set[str]:
    values: set[str] = set()
    for matrix in matrices:
        if not isinstance(matrix, list):
            continue
        for row in matrix:
            if isinstance(row, list):
                values.update(str(value) for value in row if value is not None)
    return values


def _shared_inventory(task_name: str, state: Mapping[str, Any]) -> Dict[str, Any]:
    params = state.get("params")
    trace = state.get("replay_trace")
    if not isinstance(params, Mapping) or not isinstance(trace, list):
        return {}
    merged: Dict[str, set] = {}
    for frame_state in trace:
        if not isinstance(frame_state, Mapping):
            continue
        inventory = rendered_glyph_inventory(task_name, frame_state, params)
        for key, values in inventory.items():
            merged.setdefault(key, set()).update(str(item) for item in values)
    return {key: _sort(values) for key, values in merged.items()}


def _legacy_inventory(task_name: str, state: Mapping[str, Any]) -> Dict[str, Any]:
    fixed: set[str] = set()
    dynamic: set[str] = set()
    if task_name == "arc1d_compact_nonzero":
        row = state.get("input_row") or []
        fixed.update(str(index + 1) for index in range(len(row)))
        # Empty slots render as pale colour only; only token digits are drawn.
        dynamic.update(str(value) for value in row if int(value) != 0)
    elif task_name == "binary_matrix":
        # Input bits live in cell colour alone; the wave layers print only the
        # committed distance values.
        dynamic.update(_matrix_values(state.get("distance_matrix")))
    elif task_name == "connected_components":
        grid = state.get("input_grid") or []
        fixed.update(str(index + 1) for index in range(len(grid)))
    elif task_name == "grid_bfs":
        # Input categories live in cell colour alone; only distance values
        # (including -1 on walls and unreachable cells) are printed.
        dynamic.update(_matrix_values(state.get("distance_grid")))
    elif task_name == "grid_component":
        # Both cell states live in colour alone; the scene prints no text.
        pass
    elif task_name == "grid_shift":
        matrix = state.get("base_matrix") or []
        size = len(matrix)
        fixed.update(str(index + 1) for index in range(size))
        shift_label = str(state.get("shift_label") or "").upper()
        try:
            shift = tuple(int(value) for value in state.get("shift", ()))
        except (TypeError, ValueError):
            shift = ()
        arrow = {(-1, 0): "↑", (1, 0): "↓", (0, 1): "→", (0, -1): "←"}.get(shift)
        if shift_label:
            dynamic.add(shift_label)
        if arrow:
            dynamic.add(arrow)
        dynamic.add("1")
        dynamic.update(
            str(value)
            for value in {
                cell
                for row in (matrix, state.get("target_matrix") or [])
                for line in row
                for cell in (line if isinstance(line, list) else [])
                if cell is not None
            }
        )
    elif task_name == "rotten_oranges":
        # All three cell states live in colour alone; the scene prints no text.
        pass
    elif task_name == "traffic_light_countdown":
        fixed.update(("N", "S", "E", "W", "R", "Y", "G"))
        entries = [*(state.get("initial_states") or []), *(state.get("final_states") or [])]
        digits: set[str] = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            for duration in dict(entry.get("durations") or {}).values():
                digits.update(str(int(duration)))
        dynamic.update(digits)
    return {"fixed_labels": _sort(fixed), "dynamic_labels": _sort(dynamic)}


def annotate_next_state_glyphs(record: Mapping[str, Any], task_name: str) -> Dict[str, Any]:
    annotated = dict(record)
    state = _state(annotated)
    if task_name in EXPANSION_TASK_NAMES:
        inventory = _shared_inventory(task_name, state)
    else:
        inventory = _legacy_inventory(task_name, state)
    inventory = {
        "schema_version": "next_state_rendered_glyph_inventory_v1",
        **inventory,
    }
    annotated["rendered_glyph_inventory"] = inventory
    annotated["rendered_glyph_schema"] = {
        "scope": "exact font-rendered strings across input, cumulative states, terminal image, and video",
        "exclusions": "no title, rule prose, answer banner, status sentence, or changing subtitle",
    }
    return annotated


__all__ = ["annotate_next_state_glyphs"]
