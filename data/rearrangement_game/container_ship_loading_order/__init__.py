TASK_NAME = 'container_ship_loading_order'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A dock stack holds 4-6 numbered containers (top accessible only); the ship must be loaded in descending number order, and a single buffer pad holds at most one container. Moves are (ship, id), (buffer, id), (unbuffer, id). Sampling rejects pre-sorted docks and requires at least one buffer detour with plans of N+1..12 actions. Container numbers are identities; no ordinal shifting exists.',
    "design_intent": 'Stack-to-sequence reordering through a unit buffer: the descending target turns the dock order into forced detours that exercise LIFO lookahead.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
