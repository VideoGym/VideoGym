TASK_NAME = 'pegboard_marble_drop_predict'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'The marble contacts exactly one peg in each of six triangular rows. A red left-arrow peg sends it down-left; a blue right-arrow peg sends it down-right. After the sixth contact, the same rule chooses one of numbered slots 1-7 in the visible coordinate system. Four distinct slots carry A-D badges and exactly one badge corresponds to the replayed landing slot, including boundary slots.',
    "design_intent": 'Represent a fixed-depth visual conditional program with six real contacts, unlike a single bounce or a simultaneous motion field.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
