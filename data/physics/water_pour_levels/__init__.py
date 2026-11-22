TASK_NAME = 'water_pour_levels'
TASK_FAMILY = 'physics'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Three or four numbered open containers carry unit tick marks (capacity = highest tick, 4-7) and whole-unit starting levels. The prompt states an ordered schedule of 2-4 pours by container number. One pour transfers water until the source empties or the target reaches its top tick, whichever comes first; the excess stays in the source; water is conserved. Sampling guarantees every pour moves at least one unit, the capacity clip binds at least once AND the source-empty stop binds at least once (both halves of the stop rule load-bearing), and the final vector differs from the initial one. The answer lists final levels in container order.',
    "design_intent": 'Train stateful bounded-transfer tracking where each amount is a min() of two visible quantities and the process is genuinely continuous with discrete stops.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
