TASK_NAME = 'circle_inversion'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a large black inversion circle with its center marked by a small hollow marker, one hollow source point, and five labeled candidate circles A-E. By default the source is outside and the candidates are inside the circle. Under circle inversion the image point lies on the ray from center through the source, at distance r-squared divided by the center-to-source distance, where r is the circle radius. The required relation is: Identify the inverted image.',
    "design_intent": 'Circle Inversion makes this exact relation carry correctness: A white canvas shows a large black inversion circle with its center marked by a small hollow marker, one hollow source point, and five labeled candidate circles A-E. Under circle inversion the image point lies on the ray from center through the source, at distance r-squared divided by the center-to-source distance, where r is the circle radius. The required relation is: Identify the inverted image. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
