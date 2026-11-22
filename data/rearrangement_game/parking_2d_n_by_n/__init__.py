TASK_NAME = 'parking_2d_n_by_n'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A 3x3 lot holds 7 numbered cars and 2 empty bays. A move drives one car into an orthogonally adjacent empty bay, leaving its old cell empty, so every step chooses both a car and a bay. Sampling scrambles the target 4..7 moves, requires the BFS optimum to match, and keeps only instances where every shortest plan uses both original bays, ensuring the two-bay mechanic is load-bearing. Each formal destination uses `(x, y)`, where `x` is the 1-based top-to-bottom row and `y` the 1-based left-to-right column.',
    "design_intent": 'Extends the 1-D bay walk to a 2-D lot with two interacting bays: each step couples car choice with bay choice, and the both-bays gate guarantees single-bay reasoning cannot solve the instance.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
