TASK_NAME = 'traffic_light_countdown'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Four independent lights each follow green->yellow->red->green with per-light duration digits; each simulated second decrements a countdown above 1 or, at 1, switches to the next colour with its full duration. The horizon (6-10 seconds) is stated in the Rule; sampling requires at least 4 colour switches and at least two distinct final colours.',
    "design_intent": 'Independent per-card durations force per-light simulation instead of one shared clock.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
