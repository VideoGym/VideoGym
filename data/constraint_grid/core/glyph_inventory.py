"""Per-sample rendered-text contracts for Constraint Grid tasks.

The inventory is audit metadata, not model input and not verifier evidence.  It
is attached at the family boundary after a task has authored its record.  Pure
geometry (cell fills, paths, stars, tents, bulbs, dots, pearls, arrows drawn as
shapes, and region colours) is intentionally absent.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence


def _strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=lambda value: (len(value), value))


def _state(record: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("structured_answer", "answer_payload"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _scalar_tokens(value: Any, *, include_zero: bool = True) -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            tokens.update(_scalar_tokens(item, include_zero=include_zero))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            tokens.update(_scalar_tokens(item, include_zero=include_zero))
    elif isinstance(value, bool) or value is None:
        pass
    elif isinstance(value, (int, float)):
        number = int(value)
        if number > 0 or (include_zero and number == 0):
            tokens.add(str(number))
    elif isinstance(value, str) and value.strip():
        tokens.add(value.strip())
    return tokens


def _grid_tokens(value: Any) -> set[str]:
    """Return cell glyphs; zero/negative/None are blank or geometric states."""

    return _scalar_tokens(value, include_zero=False)


_GRID_FIELDS: Dict[str, tuple[str, str]] = {
    "fillomino_completion": ("puzzle", "solution"),
}


def constraint_grid_rendered_glyph_contract(
    record: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, str]]:
    task = str(record.get("task_name") or "")
    state = _state(record)
    fixed: set[str] = set()
    visible: set[str] = set()
    terminal: set[str] = set()

    field_pair = _GRID_FIELDS.get(task)
    if field_pair:
        visible.update(_grid_tokens(state.get(field_pair[0])))
        terminal.update(_grid_tokens(state.get(field_pair[1])))

    # Row/column and edge clues are visible and remain fixed in every state.
    for key in (
        "row_counts", "col_counts", "row_sums", "col_sums",
        "left_clues", "right_clues", "top_clues", "bottom_clues",
    ):
        visible.update(_scalar_tokens(state.get(key), include_zero=True))

    if task == "aquarium_completion":
        pass
    elif task in {"light_up_akari_completion", "nurikabe_completion", "shikaku_completion"}:
        for item in state.get("clues") or []:
            if isinstance(item, Sequence) and len(item) >= 3:
                visible.add(str(int(item[2])))
    elif task == "nqueens_completion":
        size = int(state.get("board_size") or 0)
        fixed.add("Q")
        fixed.update(str(index + 1) for index in range(size))
    elif task == "slitherlink_completion":
        visible.update(_scalar_tokens(state.get("clues"), include_zero=True))
        # One-based lattice-corner labels drawn along the top and left edges.
        fixed.update(str(index) for index in range(1, int(state.get("rows") or 0) + 2))
        fixed.update(str(index) for index in range(1, int(state.get("cols") or 0) + 2))
    elif task == "yajilin_completion":
        arrows = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
        for item in state.get("clues") or []:
            if isinstance(item, Sequence) and len(item) >= 4:
                visible.add(str(int(item[3])))
                visible.add(arrows.get(str(item[2]), str(item[2])))

    # One-based row/column labels drawn along the board edges for tasks whose
    # typed answers or authored reasoning name cells by (row, column) ordinals

    if task in {
        "aquarium_completion",
        "campsite_completion",
        "fillomino_completion",
        "light_up_akari_completion",
        "masyu_completion",
        "maze_flow_pairs",
        "nurikabe_completion",
        "shikaku_completion",
        "star_battle_completion",
        "yajilin_completion",
        "yin_yang_completion",
    }:
        if task in {"aquarium_completion", "star_battle_completion", "yin_yang_completion"}:
            axis_rows = axis_cols = int(state.get("size") or 0)
        else:
            axis_rows = int(state.get("rows") or 0)
            axis_cols = int(state.get("cols") or 0)
        fixed.update(str(index) for index in range(1, axis_rows + 1))
        fixed.update(str(index) for index in range(1, axis_cols + 1))

    # Generation annotates worker records and then annotates the merged record
    # again.  Preserve the exact first-pass inventory so this transform is
    # idempotent even when answer-only raw fields have already been removed.
    existing = record.get("rendered_glyph_inventory")
    if isinstance(existing, Mapping):
        fixed.update(str(value) for value in existing.get("fixed_labels") or [])
        visible.update(str(value) for value in existing.get("dynamic_labels") or [])
        visible.update(str(value) for value in existing.get("cell_ids") or [])
        terminal.update(str(value) for value in existing.get("terminal_only_labels") or [])

    terminal_only = terminal - visible - fixed
    all_text = fixed | visible | terminal
    inventory: Dict[str, Any] = {
        "schema_version": "constraint_grid_rendered_glyph_inventory_v1",
        "fixed_labels": _strings(fixed),
        "dynamic_labels": _strings(visible),
        "terminal_only_labels": _strings(terminal_only),
        "all_rendered_text": _strings(all_text),
    }
    schema = {
        "fixed_labels": "task-bounded coordinate, candidate, blank, or operator labels listed exactly in the inventory",
        "dynamic_labels": "sample-derived text already visible in the input and fixed across applicable output states",
        "terminal_only_labels": "sample-derived answer glyphs allowed only after the corresponding visual state is committed",
        "exclusions": "no task title, traditional puzzle name, rule prose, answer banner, status sentence, changing subtitle, or step narration may be rendered",
        "verifier_boundary": "inventory metadata is excluded from semantic_input and must never be used as candidate-answer evidence",
    }
    return inventory, schema


def annotate_constraint_grid_glyphs(record: Mapping[str, Any]) -> Dict[str, Any]:
    annotated = dict(record)
    inventory, schema = constraint_grid_rendered_glyph_contract(annotated)
    annotated["rendered_glyph_inventory"] = inventory
    annotated["rendered_glyph_schema"] = schema
    return annotated


__all__ = ["annotate_constraint_grid_glyphs", "constraint_grid_rendered_glyph_contract"]
