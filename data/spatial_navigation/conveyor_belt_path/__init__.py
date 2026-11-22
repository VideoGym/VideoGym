TASK_NAME = 'conveyor_belt_path'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A solid wall band with a single opening is a 2-4 cell conveyor-belt polyline chain: the chain first moves sideways along the wall band, and its last cell turns to cross the wall. Stepping onto a conveyor cell immediately chain-pushes the agent along the arrow direction until it stops on a non-conveyor cell; hitting a wall/boundary stops it. Sampling guarantees: going around the conveyor is unreachable (the chain is the only passage), the shortest path treating the belt as floor is illegal and shorter under real forced replay (forced motion genuinely changes the trajectory), and waiting in place is allowed on non-conveyor cells. Any legal complete trajectory is accepted.',
    "design_intent": "Forced-motion semantics change the feasible trajectory: the model must encode the conveyor chain's passive displacement into its own trajectory rather than going around it or treating it as floor.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
