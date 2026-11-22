TASK_NAME = 'ball_bounces_given_time'
TASK_FAMILY = 'physics'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'The ball center moves in straight lines at constant speed inside the rectangular wall. A vertical wall reverses only the horizontal velocity component, a horizontal wall only the vertical one, and an exact corner reverses both while counting as one collision (ordinary samples exclude corners so every reflection is visually unambiguous). The process stops exactly at the prompt-stated Nth collision. A-D are four equally styled wall markers: one is the true Nth collision center, one is another real collision of the same trajectory, and two are wall contacts that are never reached, so candidates target mechanism-level errors. Candidates stay >=52px apart and off-trace markers stay >=38px from every real collision (perceptual margin).',
    "design_intent": 'Train discrete contact counting and state accumulation inside continuous geometric motion, rather than candidate-appearance shortcuts or saved-trace lookups.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
