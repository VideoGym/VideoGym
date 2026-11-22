TASK_NAME = 'fermat_point'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A black triangle outline is shown on a white canvas with five labeled candidate circles A-E inside it. The Fermat point minimizes the total distance to all three vertices; equivalently, it is the interior point from which each pair of vertices subtends exactly 120 degrees. The required relation is: Identify the Fermat point. The distance-sum objective is second-order flat at the optimum, so candidates are never meant to be separated by visually comparing distance sums; the load-bearing discriminator is the 120-degree condition, which only the correct marker satisfies and every distractor visibly violates.',
    "design_intent": "A black triangle outline is shown on a white canvas with five labeled candidate circles A-E inside it. The Fermat point minimizes the total distance to all three vertices; equivalently, it is the interior point from which each pair of vertices subtends 120 degrees. Distractors stay strictly inside the triangle with clear marker separation and visible angular deviation from 120 degrees.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
