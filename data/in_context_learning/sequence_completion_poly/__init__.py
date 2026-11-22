TASK_NAME = 'sequence_completion_poly'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Four per-position cards in one row. Cards 1-3 hold one irregular 6-8 vertex polygon with one gold corner dot, in successive orientations produced by repeating a single D4 operation; card 4 holds only a question mark over the same backdrop and must show the operation applied once more to card 3. The generator proves the polygon's eight D4 images are pairwise distinct (free group action), so orientation determines the operation and the typed operation name is a faithful encoding of state 4 given state 3. The gold corner moves rigidly with the shape; flip axes are the card's fixed center lines and center diagonals, anchored by the backdrop grid.",
    "design_intent": 'Rigid-transform rule induction on an asymmetric marked shape where naming the operation and completing the fourth orientation are provably the same answer.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
