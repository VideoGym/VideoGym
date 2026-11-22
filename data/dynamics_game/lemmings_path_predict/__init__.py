TASK_NAME = 'lemmings_path_predict'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'For exactly eight ticks, inspect the cell immediately ahead. If it is out of bounds or solid, stay in place, reverse facing, and consume the tick. Otherwise move horizontally into it, then fall vertically in that column until the nearest supporting surface stops the walker. Terrain never changes, and a turn tick cannot also move. Instances guarantee at least one turn and one fall within five turn-or-fall events.',
    "design_intent": 'Measure a fixed-length hybrid state machine combining obstacle decisions with continuous-looking gravity, distinct from distance-policy tracking and rigid formation motion.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
