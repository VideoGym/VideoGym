TASK_NAME = 'paper_fold_hole_matrix'
TASK_FAMILY = 'spatial_3d'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A fixed 8x8 sheet is folded in half twice; each fold lays one half exactly onto the other, and its direction is drawn as a bold arrow (tail dot on the moving half) on the panel it acts on. 1-3 distinct cells of the final folded packet are punched through every layer. The answer is recovered by reversing the folds strictly last fold first: one reverse fold keeps every existing hole and adds the mirror image of every existing hole across that fold crease, so the hole count grows exactly P -> 2P -> 4P. Because every fold splits an even side in half, no hole ever lands on its own mirror, and exactly one complete binary matrix follows from the visible program (no multi-solution case).',
    "design_intent": 'Train discrete fold inversion, mirror symmetry, and cumulative state tracking; the visual trace preserves the one-fold-at-a-time causal structure that pure text tends to collapse.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
