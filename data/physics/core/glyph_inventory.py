"""Per-sample font-glyph contracts for Physics media."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def _strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=lambda value: (len(value), value))


def _state(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("structured_answer", "answer_payload"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _state_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def physics_rendered_glyph_contract(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    task = str(record.get("task_name") or "")
    state = _state(record)
    fixed: set[str] = set()
    dynamic: set[str] = set()

    if task == "ball_bounces_given_time":
        fixed.update("ABCD")
    elif task == "domino_reach_cascade":
        semantic = _state_mapping(state.get("semantic_state"))
        nodes = semantic.get("nodes") or ()
        dynamic.update(str(int(node.get("id"))) for node in nodes)
        dynamic.update(f"H {int(node.get('height'))}" for node in nodes)
        dynamic.update(f"G {int(edge.get('gap'))}" for edge in semantic.get("edges") or ())
    elif task == "mirror_aim_detector":
        fixed.update("ABCD")
    elif task == "gear_train_direction":
        semantic = _state_mapping(state.get("semantic_state"))
        dynamic.update(str(int(gear.get("id"))) for gear in semantic.get("gears") or ())
    elif task == "chamber_flow_fill":
        semantic = _state_mapping(state.get("semantic_state"))
        count = len(semantic.get("widths") or ())
        dynamic.update(str(index) for index in range(1, count + 1))
    elif task == "elastic_collision_track":
        semantic = _state_mapping(state.get("semantic_state"))
        count = len(semantic.get("starts") or ())
        fixed.update("ABCD"[:count])
        fixed.update(str(tick) for tick in range(1, 13))
        dynamic.update(
            str(abs(int(value))) for value in semantic.get("velocities") or () if int(value) != 0
        )
    elif task == "water_pour_levels":
        semantic = _state_mapping(state.get("semantic_state"))
        capacities = [int(value) for value in semantic.get("capacities") or ()]
        dynamic.update(str(index) for index in range(1, len(capacities) + 1))
        for capacity in capacities:
            dynamic.update(str(unit) for unit in range(1, capacity + 1))

    all_text = fixed | dynamic
    inventory = {
        "schema_version": "physics_rendered_glyph_inventory_v1",
        "fixed_labels": _strings(fixed),
        "dynamic_labels": _strings(dynamic),
        "all_rendered_text": _strings(all_text),
    }
    schema = {
        "fixed_labels": "task-bounded identity, candidate, axis, unit, or scale labels listed exactly",
        "dynamic_labels": "sample-derived visible measurements and bounded intermediate numeric badges listed exactly",
        "exclusions": "no task title, rule prose, step sentence, changing subtitle, or answer banner may be rendered",
    }
    return inventory, schema


def annotate_physics_glyphs(record: Mapping[str, Any]) -> dict[str, Any]:
    annotated = dict(record)
    inventory, schema = physics_rendered_glyph_contract(annotated)
    annotated["rendered_glyph_inventory"] = inventory
    annotated["rendered_glyph_schema"] = schema
    return annotated


__all__ = ["annotate_physics_glyphs", "physics_rendered_glyph_contract"]
