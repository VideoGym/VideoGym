TASK_NAME = 'symmetry_completion'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Write each cell as `(x, y)`, where `x` is its 1-based top-to-bottom row and `y` is its 1-based left-to-right column. Reflect every occupied source cell perpendicularly across the red between-cells mirror axis onto the empty half: the coordinate parallel to the axis is unchanged and the perpendicular grid distance is equal on the other side. The correct patch contains every reflected cell and no extra cell. Generation uses an even 6x6 or 8x8 grid, an edge-connected 5-10 cell source pattern spanning at least two rows and two columns and touching the axis edge, and three distinct near-miss distractors built by drop/add/swap, parallel flips, and depth shifts; a validator proves exactly one candidate matches the reflection.',
    "design_intent": 'Symmetry Completion makes exact between-cells reflection carry correctness: near-miss patches (shifted, flipped, off-by-one) are the distractors, the labeled grid grounds cell references in the authored reasoning, and the single synchronized fill is the honest process evidence for a commutative cell set.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
