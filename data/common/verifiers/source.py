"""Parse structured public source states for semantic replay."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from data.common.schemas import CanonicalSample


def _to_int_grid(payload: Any) -> Optional[List[List[int]]]:
    if not isinstance(payload, list):
        return None
    grid: List[List[int]] = []
    for raw_row in payload:
        if not isinstance(raw_row, list):
            return None
        try:
            grid.append([int(cell) for cell in raw_row])
        except Exception:
            return None
    return grid


def _extract_expected_structured(sample: CanonicalSample) -> dict[str, Any]:
    structured = _extract_canonical_semantic_state(sample)
    if structured:
        return structured
    raw_record = sample.semantic_input or {}
    if isinstance(raw_record, dict):
        raw_structured = (
            raw_record.get("semantic_state")
            or raw_record.get("structured_answer")
            or raw_record.get("answer_payload")
            or {}
        )
        if isinstance(raw_structured, dict):
            return dict(raw_structured)
    return {}


def _extract_canonical_semantic_state(sample: CanonicalSample) -> dict[str, Any]:
    """Return only the source-side semantic state carried by the manifest."""

    raw = sample.semantic_input or {}
    if not isinstance(raw, dict):
        return {}
    state = raw.get("semantic_state")
    return dict(state) if isinstance(state, dict) else {}


def _semantic_metric(
    passed: bool,
    ok_reason: str = "ok",
    fail_reason: str = "semantic mismatch",
    **extra: Any,
) -> tuple[dict[str, Any], bool]:
    return {
        "video_semantic_process_check_pass": passed,
        "video_semantic_process_reason": ok_reason if passed else fail_reason,
        **extra,
    }, passed


def _grid_dimensions(grid: Sequence[Sequence[Any]]) -> tuple[int, int]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    return rows, cols


def _rectangular_grid(grid: Optional[Sequence[Sequence[Any]]]) -> bool:
    if grid is None or not grid:
        return False
    width = len(grid[0])
    return width > 0 and all(len(row) == width for row in grid)


def _to_rgb_color(raw_color: Any) -> Optional[tuple[int, int, int]]:
    if not isinstance(raw_color, (list, tuple)) or len(raw_color) != 3:
        return None
    try:
        color = tuple(int(channel) for channel in raw_color)
    except Exception:
        return None
    if any(channel < 0 or channel > 255 for channel in color):
        return None
    return color  # type: ignore[return-value]


def _to_rgb_grid(raw_grid: Any) -> Optional[List[List[tuple[int, int, int]]]]:
    if not isinstance(raw_grid, list):
        return None
    grid: List[List[tuple[int, int, int]]] = []
    for raw_row in raw_grid:
        if not isinstance(raw_row, list):
            return None
        row: List[tuple[int, int, int]] = []
        for raw_cell in raw_row:
            color = _to_rgb_color(raw_cell)
            if color is None:
                return None
            row.append(color)
        grid.append(row)
    return grid if _rectangular_grid(grid) else None
