TASK_NAME = 'bridge_drop_tile'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'S and G are separated by 2-3 complete water bands; the first step onto an un-bridged water cell spends one bridge plank and leaves a permanent, re-walkable wooden plank. Sampling guarantees: all bridge planks must be used up (one fewer makes it unsolvable), the shortest path that treats water as floor is strictly shorter than any legal route (the resource constraint necessarily changes route choice), and any complete legal route is accepted regardless of length; waiting in place is legal and spends no plank.',
    "design_intent": 'A finite irreversible resource changes the feasible route set: the shortest path taken as if the water were plain floor is entirely illegal under real plank replay, so the model must plan the unique crossing point for each water band and conserve planks.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
