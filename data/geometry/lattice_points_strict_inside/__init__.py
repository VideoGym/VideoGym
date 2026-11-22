TASK_NAME = 'lattice_points_strict_inside'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A lattice point counts only when it lies strictly inside the integer-vertex polygon and on no polygon edge; every vertex and every other lattice point exactly on an edge is boundary and excluded. The generator proves each instance by exact enumeration (integer on-segment and winding tests) and cross-checks the count with Pick's theorem. Sampling gates enforce readability: polygon twice-area at least half of `span_x * span_y` (fatness), minimum vertex angle 20 degrees, and every off-edge lattice point at least 0.15 cells clear of the outline.",
    "design_intent": 'Lattice Points Strict Inside makes the boundary-exclusive interior definition carry correctness: the solver must separate on-edge lattice points (including non-vertex edge points) from strictly interior ones on a labeled full-canvas coordinate lattice, with generation-time fatness, angle, and clearance gates guaranteeing that every inside/outside judgement has a visible margin.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
