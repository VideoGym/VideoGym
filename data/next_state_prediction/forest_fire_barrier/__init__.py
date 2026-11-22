TASK_NAME = 'forest_fire_barrier'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Trees ignite from orthogonally adjacent old flames while every old flame turns to ash each synchronous tick (2 or 3 ticks); empty, ash, water, and rock never change. Sampling draws a random terrain, requires the front to stay alive through every tick with at least 4 total ignitions, and proves both barrier types load-bearing: for each of water and rock there is a barrier cell whose replacement by a tree would burn a currently surviving tree.',
    "design_intent": 'Expiring sources and two immutable barriers distinguish it from rotten_oranges.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
