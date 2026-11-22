TASK_NAME = 'reflection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows one black line, one small hollow white source marker outlined in black, and five labeled candidate circles A-E on the other side or nearby. The required relation is: Determine which candidate is the mirror reflection of the source point across the black line.',
    "design_intent": 'Reflection makes this exact relation carry correctness: A white canvas shows one black line, one small hollow white source marker outlined in black, and five labeled candidate circles A-E on the other side or nearby. The required relation is: Determine which candidate is the mirror reflection of the source point across the black line. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
