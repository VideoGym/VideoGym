TASK_NAME = 'midpoint'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A straight segment joins two hollow endpoint markers, with five labeled candidate markers nearby. The required relation is: Select the candidate on the segment that divides it into two equal halves. Presentation parity: all five markers sit on the drawn segment, so the answer-free reading "which marker is on the segment?" cannot narrow the field. A distractor is wrong because it divides the segment unevenly, and its centre stays at least 32 px along the segment from the midpoint and clear of both endpoint anchors.',
    "design_intent": 'Midpoint makes this exact relation carry correctness: A straight segment joins two hollow endpoint markers, with five labeled candidate markers nearby. The required relation is: Select the candidate on the segment that divides it into two equal halves. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
