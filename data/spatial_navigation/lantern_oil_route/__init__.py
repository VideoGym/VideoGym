TASK_NAME = 'lantern_oil_route'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Each step (including in place) first deducts 1 fuel; deducting to 0 fails and the destination effect does not take effect; only entering a supply pad from another cell with positive fuel refills to full, and staying on the pad does not refill. Sampling guarantees: the straight shortest path necessarily runs out of fuel (base_len >= max_oil), no-supply is unsolvable, the legal reference route is strictly longer than the geometric shortest, and at least one supply-arrival margin is <= max_oil-2. Any complete route with positive fuel throughout is accepted.',
    "design_intent": 'The fuel budget forces the route to detour through supply points: the geometric shortest path necessarily fails, testing detour planning and step-by-step budget accounting under a resource constraint.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
