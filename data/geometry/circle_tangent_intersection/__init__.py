TASK_NAME = 'circle_tangent_intersection'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a black circle with two hollow markers on the circumference and five labeled candidate circles A-E. At each marked point, the tangent line is perpendicular to the radius. The required relation is: Find where the two tangent lines intersect outside the circle.',
    "design_intent": 'Circle Tangent Intersection makes this exact relation carry correctness: A white canvas shows a black circle with two hollow markers on the circumference and five labeled candidate circles A-E. At each marked point, the tangent line is perpendicular to the radius. The required relation is: Find where the two tangent lines intersect outside the circle. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
