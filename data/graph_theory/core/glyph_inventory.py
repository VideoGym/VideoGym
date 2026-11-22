"""Per-sample rendered-text contracts for Graph Theory tasks.

The inventory is attached at the family boundary, after each task has produced
its raw semantic record but before canonical-manifest serialization.  It lists
every string that the task renderer may place on an input, cumulative state,
terminal image, or video.  Purely geometric marks (tracks, colored halos,
route strokes, and endpoint rings) are intentionally excluded.  All 17 unified
tasks render numeric node labels only; no letters, titles, or rule prose.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from data.graph_theory.core.task_specs import NEW_GRAPH_TASK_NAMES


def _strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=lambda value: (len(value), value))


def _state(record: Mapping[str, Any]) -> Dict[str, Any]:
    value = record.get("structured_answer")
    return dict(value) if isinstance(value, Mapping) else {}


def graph_rendered_glyph_contract(record: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    task = str(record.get("task_name") or "")
    state = _state(record)
    fixed: set[str] = set()
    dynamic: set[str] = set()
    terminal_only: set[str] = set()

    if task in NEW_GRAPH_TASK_NAMES:
        for node in state.get("nodes", []):
            if not isinstance(node, Mapping):
                continue
            if node.get("label") is not None:
                dynamic.add(str(node["label"]))
            if node.get("degree_badge") is not None:
                fixed.add(str(node["degree_badge"]))
        if task == "degree_missing_value":
            fixed.add("?")
            raw_answer = str(record.get("answer") or "").strip()
            if raw_answer:
                terminal_only.add(raw_answer)
    else:
        raise ValueError(f"unknown Graph Theory task for glyph inventory: {task!r}")

    inventory: Dict[str, Any] = {
        "schema_version": "graph_theory_rendered_glyph_inventory_v2",
        "fixed_labels": _strings(fixed),
        "dynamic_labels": _strings(dynamic),
        "terminal_only_labels": _strings(terminal_only),
        "all_rendered_text": _strings(fixed | dynamic | terminal_only),
    }
    schema = {
        "fixed_labels": "task-bounded labels that may appear exactly as inventoried in every applicable state",
        "dynamic_labels": "sample-derived numeric node labels listed exactly in the inventory",
        "terminal_only_labels": "sample-derived count labels allowed only after their visual state is committed",
        "exclusions": "no task title, rule prose, answer banner, changing subtitle, status sentence, or step narration may be rendered",
    }
    return inventory, schema


def annotate_graph_glyphs(record: Mapping[str, Any]) -> Dict[str, Any]:
    annotated = dict(record)
    inventory, schema = graph_rendered_glyph_contract(annotated)
    existing_inventory = annotated.get("rendered_glyph_inventory")
    if isinstance(existing_inventory, Mapping):
        merged_inventory = dict(existing_inventory)
        fixed = set(merged_inventory.get("fixed_labels") or []) | set(inventory["fixed_labels"])
        dynamic = (
            set(merged_inventory.get("dynamic_labels") or [])
            | set(merged_inventory.get("cell_ids") or [])
            | set(inventory["dynamic_labels"])
        )
        terminal_only = set(merged_inventory.get("terminal_only_labels") or []) | set(
            inventory["terminal_only_labels"]
        )
        merged_inventory.update(
            {
                "schema_version": "graph_theory_rendered_glyph_inventory_v2",
                "fixed_labels": _strings(fixed),
                "dynamic_labels": _strings(dynamic),
                "terminal_only_labels": _strings(terminal_only),
                "all_rendered_text": _strings(fixed | dynamic | terminal_only),
            }
        )
        inventory = merged_inventory
    existing_schema = annotated.get("rendered_glyph_schema")
    if isinstance(existing_schema, Mapping):
        schema = {**schema, **dict(existing_schema)}
        schema.setdefault(
            "exclusions",
            "no task title, rule prose, answer banner, changing subtitle, status sentence, or step narration may be rendered",
        )
    annotated["rendered_glyph_inventory"] = inventory
    annotated["rendered_glyph_schema"] = schema
    return annotated


__all__ = ["annotate_graph_glyphs", "graph_rendered_glyph_contract"]
