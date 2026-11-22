TASK_NAME = 'bus_route_passengers'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'The bus must board passengers at the three numbered stops in first-arrival order 1->2->3, and finally park at DEP. Entering a later stop early is illegal (it would board out of order). Sampling uses an ordered-progress state search to guarantee out-of-order is infeasible; any legal route is accepted, and length is not scored.',
    "design_intent": 'Ordered boarding overlaid with a detour forces the model to plan the visit order by number, with on-board/alighting honestly expressing the passenger-load state via a discrete prefix encoding.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
