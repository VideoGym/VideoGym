TASK_NAME = 'stack_reversal_minimal_moves'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'One primary column holds 3-5 numbered blocks; two or three auxiliary columns start empty. A move pops exactly the top block of one non-empty column and pushes it onto another column. The target reverses the primary stack in place with every buffer empty again. The buffer count is load-bearing: within the admitted block range a third buffer shortens the optimum by exactly one move. Optimal distances stay in the enforced 6..13 window. Column positions are 0-based internally and published 1-based; block numerals are identities and never shift. The exact-distance/path-count BFS also reports how many shortest plans exist, and any of them is accepted.',
    "design_intent": 'Trains shortest LIFO full-reversal planning under variable stack depth and variable buffer resources, including the resource-plan relation that a third buffer saves exactly one move.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
