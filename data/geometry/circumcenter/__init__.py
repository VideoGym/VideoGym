TASK_NAME = 'circumcenter'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Select the unique candidate at the common intersection of the three side perpendicular bisectors, equivalently the point equally distant from all three vertices.',
    "design_intent": 'The frame composition the rest of the family uses is off here: the three bisectors run until the frame cuts them, and the whole-video gate rebuilds those cut chords from the record, so sliding the triangle inside a fixed frame re-cuts them and changes how far each stroke travels per frame. Circumcenter makes this exact relation carry correctness: Select the unique candidate at the common intersection of the three side perpendicular bisectors, equivalently the point equally distant from all three vertices. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
