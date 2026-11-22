TASK_NAME = 'meal_train_pass_serve'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Six to nine seats ring a table; the seat count is sampled per instance. Gold dots mark 3-5 seats wanting service, and the platter starts on a visible seat (served immediately if marked). Each move passes the platter one seat CW or CCW, serving a still-waiting seat on arrival. Instances are drawn from the enumerated catalog for that seat count, keeping only scenarios whose shortest route is unique, uses both directions, and beats both pure-direction loops, with balanced route lengths (4-7 passes depending on table size). Seats are numbered 1-based clockwise from the top, and the answer lists every visited seat from start to finish.",
    "design_intent": 'Bidirectional ring service where dot counts force direction reversals and table size determines how far doubling-back reaches across ordered service steps.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
