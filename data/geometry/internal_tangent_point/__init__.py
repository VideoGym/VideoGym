TASK_NAME = 'internal_tangent_point'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Identify the point on the larger circle where the common internal tangent line (crossing between the circles) touches it. The two circles are sampled constructively across arbitrary headings with a positive separation gap (0.10-0.24 of the canvas beyond r1 + r2) so both circles, tangency points, and option markers stay inside the canvas. All five candidate markers lie on the larger circle's rim, and the neighbourhood of the second internal tangent contact is excluded from distractor placement so the answer is unique.",
    "design_intent": 'Internal Tangent Point makes this exact relation carry correctness: Identify the point on the larger circle where the common internal tangent line (crossing between the circles) touches it. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
