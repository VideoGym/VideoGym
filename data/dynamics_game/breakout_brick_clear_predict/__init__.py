TASK_NAME = 'breakout_brick_clear_predict'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A single ball follows the complete dotted route to its marked left, right, or top wall contact, changes direction there once, and destroys only the first brick touched on the second segment. The route, paddle, wall marker, and every unstruck brick remain unchanged. Logical grid intersection and rendered-pixel first contact must identify the same unique brick.',
    "design_intent": 'Measure visible trajectory following and first contact after exactly one boundary turn, not generic brick counting or a one-dimensional insertion clear.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
