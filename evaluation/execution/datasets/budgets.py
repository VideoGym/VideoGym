def allocate_task_budget(case, count):
    """Balance a task total across declared tiers, favoring later harder tiers."""
    tiers = [tier["name"] for tier in case.difficulty_tiers()]
    if type(count) is not int or count < len(tiers):
        raise ValueError(
            "task budget must include at least one sample per difficulty tier"
        )
    base, remainder = divmod(count, len(tiers))
    return {
        tier: base + int(index >= len(tiers) - remainder)
        for index, tier in enumerate(tiers)
    }


def frozen_tier_budgets(metadata, definitions):
    if "tier_budgets" in metadata:
        return metadata["tier_budgets"]
    return {
        name: {
            tier["name"]: metadata["budgets"][name] for tier in case.difficulty_tiers()
        }
        for name, case in definitions.items()
    }
