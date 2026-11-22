TASK_NAME = 'voxel_three_views'
TASK_FAMILY = 'spatial_3d'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Unit blocks stand in stacks on a 3x3 or 4x4 floor (heights 0-3, at least two distinct heights, at least two empty floor cells). A bold blue 3D arrow names the asked orthographic view: top (floor plan), or one of two side directions whose exact grid mapping the Rule states (view column j = floor column j or floor row j, view row 1 = highest layer - both un-mirrored ascending mappings). The answer is the exact binary view grid: a cell is filled iff any block lies on its viewing ray. Exactly one grid follows from the visible structure; the top view is sampled with reduced weight because it reads only the footprint.',
    "design_intent": 'Train 3D-to-2D orthographic projection - the canonical engineering-drawing skill - on structures certified readable from one picture.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
