TASK_NAME = 'light_up_akari_completion'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Bulbs light their row and column up to the nearest wall; every white cell must end lit, no bulb may see another, and a digit on a wall counts its orthogonal bulb neighbours exactly; given bulbs are fixed. Forced rules: a wall digit whose remaining count equals its remaining open neighbours forces bulbs on all of them; a satisfied wall closes its other neighbours; an unlit cell whose sight line holds a single open cell forces a bulb there. Construction samples walls, a full lighting, and shown clues, then un-gives bulbs monotonically; forcedness implies uniqueness. Difficulty gates: at least 4 open bulbs, at least 2 waves, givens at most half the bulbs and at least one.',
    "design_intent": 'Isolates sight-line coverage with local counting constraints where every deduction is a visible light source whose effect region is itself rendered.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
