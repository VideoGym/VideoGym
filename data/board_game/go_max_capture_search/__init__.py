TASK_NAME = 'go_max_capture_search'
TASK_FAMILY = 'board_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Over the 7x7 initial state, enumerate the one-move Black placement on every empty intersection; following the orthogonal-group and liberty rules, first simultaneously remove all adjacent zero-liberty opponent groups, then exclude suicidal placements, maximizing the number of captured White stones. The capture target varies per sample: a single 1-3 stone connected group, or two independent groups that share the target as their common last liberty (a simultaneous two-group capture); there is also a strictly smaller alternative motive (a capturable small group when max>1, and a two-liberty near-miss group when max=1). The solver keeps every tied maximum, and the generator accepts only a unique maximum greater than zero.',
    "design_intent": "The positive-capture branches and their capture counts are a stable comparison entailed by the position rather than an arbitrary scanning trace; the varying captured-group shapes (including the simultaneous two-group capture) keep 'find the maximum' from degenerating into the visual template of 'find the point wedged between two white stones'.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
