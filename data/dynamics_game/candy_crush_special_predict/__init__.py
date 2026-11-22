TASK_NAME = 'candy_crush_special_predict'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Activate the unique double-rimmed token once. Horizontal bars clear its whole row; vertical bars its whole column; the gold square its in-bounds 3x3 neighborhood; the silver star every token with the marked token's visible fill color. The marked cell is always included. Cleared wells become pale empty wells; survivors never move, recolor, fall, refill, or trigger another clear.",
    "design_intent": 'Test interpretation of a visible action-scope glyph and exact set deletion, separate from match formation, gravity, or connected-component search.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
