"""Build verifier diagnostics without changing hard-gate decisions."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence


def _verifier_contract_metrics(
    *,
    verifier_profile: str,
    artifact_kind: str,
    component_scores: dict[str, float],
    binary_correct: bool,
    failure_reason: str,
    started_at: float,
    hard_gate_failures: Sequence[str] = (),
) -> dict[str, Any]:
    normalized_scores = {
        str(name): max(0.0, min(1.0, float(value)))
        for name, value in component_scores.items()
    }
    failures = [str(value) for value in hard_gate_failures if str(value)]
    latency_ms = max(0.0, (perf_counter() - started_at) * 1000.0)
    return {
        "verifier_profile": str(verifier_profile),
        "artifact_kind": str(artifact_kind),
        "component_scores": normalized_scores,
        "hard_gate_failures": failures,
        "binary_correct": bool(binary_correct),
        "failure_reason": "ok" if binary_correct else str(failure_reason or "verification failed"),
        "latency_ms": latency_ms,
        "performance_diagnostics": {
            "latency_ms": latency_ms,
            "automatic_failure_thresholds": False,
        },
    }
