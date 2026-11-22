TASK_NAME = 'block_blast_clear_predict'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'The translucent wells fix one legal anchor for the shown multi-cell piece. All piece cells land together, become occupied, and then every completely filled row and column is cleared in one update. Cells in row/column intersections clear once; all surviving blocks remain in their original wells. The source excludes pre-existing full lines and guarantees at least one line becomes full.',
    "design_intent": 'Isolate rigid placement followed by multi-axis simultaneous clearing, distinct from gravity-driven falling pieces and from color-component removal.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
