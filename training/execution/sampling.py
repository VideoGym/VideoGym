"""Global sample batches with deterministic, resumable exposure."""

from __future__ import annotations

import math
import random


def epoch_order(size: int, seed: int, epoch: int) -> list[int]:
    indices = list(range(size))
    random.Random(f"{seed}:{epoch}").shuffle(indices)
    return indices


def rank_slots(indices: list[int], rank: int, world_size: int):
    """Padding slots participate in collectives but have zero loss/exposure."""
    if not indices or not 0 <= rank < world_size:
        raise ValueError("nonempty batch and valid distributed rank required")
    for slot in range(math.ceil(len(indices) / world_size)):
        position = slot * world_size + rank
        yield (
            (indices[position], True)
            if position < len(indices)
            else (indices[0], False)
        )


def learning_rate(config, update: int, total_updates: int) -> float:
    if config.warmup_updates and update < config.warmup_updates:
        return config.learning_rate * (update + 1) / config.warmup_updates
    span = max(1, total_updates - config.warmup_updates - 1)
    progress = min(1.0, max(0.0, (update - config.warmup_updates) / span))
    decay = (
        1.0 - progress
        if getattr(config, "lr_scheduler", "linear") == "linear"
        else (1.0 + math.cos(math.pi * progress)) / 2.0
    )
    return config.learning_rate * (
        config.min_lr_ratio + (1.0 - config.min_lr_ratio) * decay
    )
