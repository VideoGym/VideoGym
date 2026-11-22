TASK_NAME = 'polycube_rotation'
TASK_FAMILY = 'spatial_3d'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A connected polycube of 4-5 identical cubes, sampled to be fully rotation-asymmetric (all 24 orientations distinct, which guarantees every wrong-rotation distractor is a genuinely different pose), floats beside a labeled X/Y/Z axis triad. The Rule states 1-3 quarter turns, each about one named axis through the structure center, with direction read from the axis arrow tip (8.5 single convention). Applying the turns in order yields one deterministic final pose; the answer is the letter of the unique option plate matching that pose. Distractors target real error modes: inverted last turn, wrong axis, missing last turn, one extra turn, or the mirror image - never a second correct pose.',
    "design_intent": 'Train mental rotation with explicit axis grounding - the 3D counterpart of 2D rotation tasks - with genuine pose hypotheses as input evidence and the full rigid motion as video supervision.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
