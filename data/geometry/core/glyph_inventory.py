"""Exact per-sample rendered-glyph contracts for Geometry media."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from data.common.choice_cards import CHOICE_CARD_INPUT_POLICIES
from data.geometry.core.canonical_source import _POINT_SOURCE_FIELDS


def _strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=lambda value: (len(value), value))


def _state(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("structured_answer") or record.get("answer_payload")
    return value if isinstance(value, Mapping) else {}


def _integer_strings(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            result.update(_integer_strings(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            result.update(_integer_strings(item))
    elif isinstance(value, int) and not isinstance(value, bool):
        result.add(str(value))
    return result


def geometry_rendered_glyph_contract(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    task = str(record.get("task_name") or "")
    state = _state(record)
    fixed: set[str] = set()
    dynamic: set[str] = set()
    terminal: set[str] = set()
    input_media_policy = str(record.get("input_media_policy") or "")
    choice_card_delivery = input_media_policy in CHOICE_CARD_INPUT_POLICIES

    if task in _POINT_SOURCE_FIELDS:
        fixed.update(
            str(item.get("label"))
            for item in record.get("candidates") or ()
            if isinstance(item, Mapping) and item.get("label")
        )
    elif not choice_card_delivery:
        fixed.update("ABCD")

    if task == "lattice_points_strict_inside":
        # Native full-canvas delivery: the only rendered text is the fixed
        # coordinate labels along the lattice edges.  The A-D option values
        # are Prompt text and the terminal adds unlettered cyan markers only.
        fixed.update(str(value) for value in range(-4, 5))
    elif task == "symmetry_completion":
        # The board and each candidate patch carry 1-based row/column edge
        # indices; plates additionally carry their bottom-centered A-D label,
        # which stays an input-plate label outside this scene inventory.
        grid_size = state.get("grid_size")
        if isinstance(grid_size, int) and grid_size > 0:
            fixed.update(str(value) for value in range(1, grid_size + 1))
    elif task == "right_triangle_selection":
        # Candidate label sets are Prompt text only.  The rendered media show
        # exactly the per-triangle numeric badges; the direct terminal adds
        # unlettered red evidence circles and no derived text.
        for triangle in state.get("triangles") or ():
            if isinstance(triangle, Mapping) and triangle.get("label"):
                dynamic.add(str(triangle["label"]))
    elif task == "rotation":
        fixed.update(("?", "→"))
    elif task == "triangle_inequality_check":
        # Native full-scene delivery: every printed side length of all four
        # rows is visible in the input, and the solved comparison zones add
        # each row's shorter-side sum and longest value.
        triples = state.get("triples")
        dynamic.update(_integer_strings(triples))
        if isinstance(triples, (list, tuple)):
            for triple in triples:
                if isinstance(triple, (list, tuple)) and len(triple) == 3:
                    ordered = sorted(int(value) for value in triple)
                    terminal.update((str(ordered[0] + ordered[1]), str(ordered[2])))

    # Some renderers carry candidate labels in structured state rather than
    # the common A-D layout.  Include them without importing any answer flag.
    for candidate in state.get("candidates") or ():
        if isinstance(candidate, Mapping) and candidate.get("label"):
            fixed.add(str(candidate["label"]))

    existing = record.get("rendered_glyph_inventory")
    if isinstance(existing, Mapping):
        fixed.update(str(value) for value in existing.get("fixed_labels") or ())
        dynamic.update(str(value) for value in existing.get("dynamic_labels") or ())
        terminal.update(str(value) for value in existing.get("terminal_only_labels") or ())

    if choice_card_delivery:
        fixed.difference_update("ABCD")

    terminal_only = terminal - fixed - dynamic
    inventory = {
        "schema_version": "geometry_rendered_glyph_inventory_v1",
        "fixed_labels": _strings(fixed),
        "dynamic_labels": _strings(dynamic),
        "terminal_only_labels": _strings(terminal_only),
        "all_rendered_text": _strings(fixed | dynamic | terminal),
    }
    fixed_labels_scope = (
        "coordinate, axis, endpoint, source-object, or other stable non-choice "
        "labels rendered exactly as listed"
        if choice_card_delivery
        else "bounded candidate, coordinate, axis, endpoint, panel, or stable "
        "symbol labels rendered exactly as listed"
    )
    dynamic_labels_scope = (
        "sample-derived source facts or direct-result values rendered exactly "
        "as listed; Prompt-only values and input-plate labels are excluded"
        if choice_card_delivery
        else "sample-derived visible option values, side values, landmark "
        "initials, or vector labels rendered exactly as listed"
    )
    exclusions = (
        "no task name, rule prose, step narration, answer banner, changing "
        "subtitle, renderer/debug metadata, or A-D choice label may appear in "
        "terminal images or TI2V"
        if choice_card_delivery
        else "no task name, rule prose, step narration, answer banner, changing "
        "subtitle, or renderer/debug metadata may be rendered"
    )
    schema = {
        "fixed_labels": fixed_labels_scope,
        "dynamic_labels": dynamic_labels_scope,
        "terminal_only_labels": "derived rank, count, category, or comparison values that appear only after their semantic commit",
        "exclusions": exclusions,
        "verifier_boundary": "glyph inventory is audit metadata outside semantic_input and never candidate-answer evidence",
    }
    return inventory, schema


def annotate_geometry_glyphs(record: Mapping[str, Any]) -> dict[str, Any]:
    annotated = dict(record)
    inventory, schema = geometry_rendered_glyph_contract(annotated)
    annotated["rendered_glyph_inventory"] = inventory
    annotated["rendered_glyph_schema"] = schema
    return annotated


__all__ = ["annotate_geometry_glyphs", "geometry_rendered_glyph_contract"]
