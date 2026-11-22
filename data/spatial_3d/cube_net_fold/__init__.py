TASK_NAME = 'cube_net_fold'
TASK_FAMILY = 'spatial_3d'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A flat hexomino net of six squares (sampled from the eleven free cube nets under all eight planar orientations, any of the six cells as base) carries a dice-pip permutation 1-6, one count per square, printed identically on both sides. The blue-shaded base square stays flat; every crease folds upward by a quarter turn, so the five flaps assemble the unit box above the base. A folding flap rigidly carries every square attached beyond it, and the terminal pose depends only on the final hinge angles, never on the temporal fold order. Exactly one square ends level above the base as the lid; its pip count is the unique typed answer.',
    "design_intent": 'Train 2D-to-3D assembly: tracking where net cells land on a solid, the core skill behind net/opposite-face benchmark items, with the whole assembly process visible as supervision.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
