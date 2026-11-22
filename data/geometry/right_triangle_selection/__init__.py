TASK_NAME = 'right_triangle_selection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Judge each non-degenerate triangle from its visible sides: it is a right triangle exactly when one corner is perpendicular. Every non-right triangle keeps every corner at least 15 degrees away from 90 degrees, so the task never depends on a borderline visual judgment. Choose the unique A-D Options entry containing every right triangle and no non-right triangle; each entry is a set of the visible numeric labels, so its printed order does not affect membership.',
    "design_intent": 'The task tests robust visual recognition of perpendicular incidence and exact set matching. The 15-degree negative margin prevents a renderer-resolution trick, while source-side integer replay keeps correctness deterministic and independent.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
