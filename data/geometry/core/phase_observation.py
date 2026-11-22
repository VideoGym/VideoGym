"""Read operation boundaries from semantic observations on every decoded frame."""

from __future__ import annotations

from typing import Sequence


def phase_onset(values: Sequence[float], threshold: float) -> int:
    """First visible operation, with an absent operation anchored at the tail.

    Callers independently require completed geometry and legal phase order.
    The fallback only keeps an absent operation's diagnostic slice in bounds.
    """
    return next((index for index, value in enumerate(values) if value > threshold), max(0, len(values) - 3))


def selection_onset(labels: Sequence[str | None]) -> int:
    return next((index for index, value in enumerate(labels) if value is not None), max(0, len(labels) - 3))


def serial_construction_observation(frames, source, record, *, task_name: str,
                                    expected: str, end: int):
    """Read declared directed strokes without prescribing their frame budget.

    Resolve the shared observer at call time: the route imports task-owned
    endpoint parsers, while this observation consumes their replayed geometry.
    """
    from data.geometry.core.verifier import _verify_serial_coordinate_strokes

    return _verify_serial_coordinate_strokes(
        frames, source, record, task_name=task_name, expected=expected,
        construction_start=0, construction_end=end,
    )
