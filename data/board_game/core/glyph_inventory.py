"""Exact per-sample rendered-glyph contracts for Board Game tasks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


def _strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=lambda value: (len(value), value))


def _state(record: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("structured_answer", "answer_payload"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def board_rendered_glyph_contract(record: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    task = str(record.get("task_name") or "")
    state = _state(record)
    fixed: set[str] = set()
    dynamic: set[str] = set()
    terminal: set[str] = set()

    semantic = state.get("semantic_state") if isinstance(state.get("semantic_state"), Mapping) else state

    if task == "chess_l_waypoint_search":
        # A visited waypoint is encoded only by its pale-green cell fill;
        # no check glyph or other terminal-only text is ever rendered.
        fixed.update(str(index) for index in range(1, 6))
        dynamic.add("L")
    elif task in {"go_rule_simulator", "go_max_capture_search"}:
        fixed.update(str(index) for index in range(1, 8))
    elif task == "gomoku_rule_simulator":
        fixed.update(str(index) for index in range(1, 10))
    elif task == "hex_chain_count":
        for value in state.get("reference_component_sizes") or []:
            if isinstance(value, int) and not isinstance(value, bool):
                dynamic.add(str(value))
    elif task == "lightsout_single_press":
        source = semantic if isinstance(semantic, Mapping) else {}
        size = int(source.get("board_size") or len(source.get("initial_grid") or []))
        fixed.update(str(index) for index in range(1, size + 1))
        dynamic.update(("P", "0", "1"))
    elif task == "mancala_distribute_predict":
        # Pit names are 1-based ordinals; the dynamic digits below are counts.
        fixed.update([*(f"P{index}" for index in range(1, 7)), "B"])
        for key in ("pits", "state_grid", "state_trace", "process_state_trace"):
            values = state.get(key)
            stack = [values]
            while stack:
                value = stack.pop()
                if isinstance(value, (list, tuple)):
                    stack.extend(value)
                elif isinstance(value, int) and not isinstance(value, bool):
                    dynamic.add(str(value))
        if isinstance(state.get("bank"), int):
            dynamic.add(str(state["bank"]))
    elif task == "tic_tac_toe_rule_simulator":
        # X and O marks are stroke-drawn geometry, not text draw calls, so
        # only the coordinate labels are rendered glyphs.
        fixed.update(("1", "2", "3"))

    # Generation annotates each worker record and the merged record.  Union
    # the first pass so answer-only raw fields may be scrubbed without making
    # the transform non-idempotent.
    existing = record.get("rendered_glyph_inventory")
    if isinstance(existing, Mapping):
        fixed.update(str(value) for value in existing.get("fixed_labels") or [])
        dynamic.update(str(value) for value in existing.get("dynamic_labels") or [])
        terminal.update(str(value) for value in existing.get("terminal_only_labels") or [])

    terminal_only = terminal - fixed - dynamic
    inventory: Dict[str, Any] = {
        "schema_version": "board_game_rendered_glyph_inventory_v1",
        "fixed_labels": _strings(fixed),
        "dynamic_labels": _strings(dynamic),
        "terminal_only_labels": _strings(terminal_only),
        "all_rendered_text": _strings(fixed | dynamic | terminal),
    }
    schema = {
        "fixed_labels": "task-bounded coordinates, slot labels, legends, or stable token letters rendered exactly as listed",
        "dynamic_labels": "sample-derived token letters or counts that can appear in the input and cumulative states",
        "terminal_only_labels": "sample-derived verdict, count, visit mark, or cumulative numbering rendered only after its semantic commit",
        "exclusions": "no task title, traditional game name, rule prose, answer banner, step narration, changing subtitle, or decorative text may be rendered",
        "verifier_boundary": "glyph inventory is audit metadata excluded from semantic_input and never candidate-answer evidence",
    }
    return inventory, schema


def annotate_board_glyphs(record: Mapping[str, Any]) -> Dict[str, Any]:
    annotated = dict(record)
    inventory, schema = board_rendered_glyph_contract(annotated)
    annotated["rendered_glyph_inventory"] = inventory
    annotated["rendered_glyph_schema"] = schema
    return annotated


__all__ = ["annotate_board_glyphs", "board_rendered_glyph_contract"]
