"""Proportional task-quota allocation across In-Domain task families."""

from __future__ import annotations


def family_quotas(total: int, counts: dict[str, int]) -> dict[str, int]:
    """Largest remainder with one seat per family; small suites round up to coverage."""
    if total < len(counts):
        raise ValueError("task budget cannot cover every ID family")
    denominator = sum(counts.values())
    quotas = {f: max(1, int(total * n / denominator)) for f, n in counts.items()}
    while sum(quotas.values()) > total:
        f = min(
            (f for f in quotas if quotas[f] > 1),
            key=lambda f: (total * counts[f] / denominator - quotas[f], f),
        )
        quotas[f] -= 1
    while sum(quotas.values()) < total:
        f = max(quotas, key=lambda f: (total * counts[f] / denominator - quotas[f], f))
        quotas[f] += 1
    return quotas
