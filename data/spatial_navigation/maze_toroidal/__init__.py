TASK_NAME = 'maze_toroidal'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A wraparound grid: crossing one edge re-enters from the opposite side of the same row/column; the boundary wall ring is kept but the injected wrap bridges open several gaps. Generation guarantees that the unique injected route must use a wrap at least once and meets the length bar; an ordinary BFS without wrap has no path in most samples.',
    "design_intent": "Represents a non-local topological-edge (wraparound) mechanic: tests the model's spatial reasoning about coordinate-system re-entry rather than memorized planar connectivity.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
