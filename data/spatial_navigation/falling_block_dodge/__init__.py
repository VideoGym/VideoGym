TASK_NAME = 'falling_block_dodge'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Each step the agent moves first (the destination must not be a current block position), then all orange blocks fall one cell simultaneously (the agent's landing must not be the next-cell destination of any block); a block that falls off the boundary disappears and is not replenished. A block's row/column fully determines its state, with no hidden timer. Sampling guarantees that the block-static shortest path is unsafe under real simultaneous-fall replay (timing is load-bearing) and that the solution passes through a cell initially vacated by some block. Any safe action sequence is accepted.",
    "design_intent": "Synchronized falling obstacles test time-indexed look-ahead planning: encode the continuous fall into one's own action timing, since the static shortest path is necessarily crushed.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
