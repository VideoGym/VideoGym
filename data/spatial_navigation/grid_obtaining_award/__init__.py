TASK_NAME = 'grid_obtaining_award'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'On a 10x10 grid, 3-4 colored rewards and slate walls; entering a reward cell collects it, and first entry into the red goal cell is forbidden before all are collected. Sampling guarantees: the direct route (no rewards) is strictly shorter by >=5 cells (collection genuinely reroutes), the wall-free collection route is strictly shorter (walls are load-bearing), and every reward forces an extra detour (removing each one-by-one is shorter). Any legal move sequence that arrives after collecting everything is accepted.',
    "design_intent": 'Multi-target collection overlaid with free visit order: rewards are load-bearing one-by-one and the wall shape forces detours, testing multi-subgoal route organization without dictating a unique order.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
