TASK_NAME = 'maze_keys_and_locks'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Along the trunk of a tree-shaped maze, 2-3 gates are placed in order, and each key is hidden on a side branch of the trunk segment before the gate of its own colour; stepping on a key collects it, and a gate is impassable until the key of the same colour has been collected; revisiting and waiting are allowed. Generation hard gates: with all gates treated as walls an ordinary BFS has no path (gate_bypass=0), every key and every gate lies on the route, and the dependency order is exactly yellow key, yellow gate, cyan key, cyan gate (, magenta key, magenta gate).',
    "design_intent": 'Event-driven state navigation where collecting keys unlocks matching gates along a causal multi-stage route.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
