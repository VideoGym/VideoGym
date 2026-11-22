TASK_NAME = 'ice_physics_multi_slide'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Two explicit prompt-declared variants (declared in the Prompt): MIXED (snowflake cells are ice, white cells are ordinary floor, S/G on floor) and ALL ICE (all walkable cells are ice). Stepping onto an ice cell slides continuously in the same direction; in MIXED it stops on the first floor cell or the last ice cell before a wall/boundary, and in ALL ICE it stops only before a wall/boundary; only after stopping can it change direction or wait in place. Sampling guarantees: in MIXED, going around the ice is unreachable and the shortest step count treating ice as floor is strictly smaller than the real sliding minimum (sliding semantics are load-bearing), with >=2 multi-cell slides; ALL ICE has >=3 multi-cell slides. Any legal trajectory is accepted.',
    "design_intent": 'The sliding commitment makes the reachability graph determined by stop points rather than cell adjacency; the intuitive solution treating ice as floor is excluded in sampling, testing look-ahead planning under forced dynamics.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
