TASK_NAME = 'ray_reflect'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Determine which candidate lies on the outgoing reflected ray after mentally reflecting the incoming ray off the mirror. Exactly one candidate does. The outgoing ray is unbounded, so the generator clears every distractor of the whole forward half-line and not merely of the drawn contact-to-target segment, which would have let a marker sit on the ray just past the answer and give the sample a second valid answer.',
    "design_intent": 'Ray Reflect makes this exact relation carry correctness: Determine which candidate lies on the outgoing reflected ray after mentally reflecting the incoming ray off the mirror. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
