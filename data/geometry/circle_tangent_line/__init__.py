TASK_NAME = 'circle_tangent_line'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Infer the circle center and radius through the contact. Select the unique candidate center on the forward tangent ray perpendicular to that radius; marker outlines and the opposite ray do not count.',
    "design_intent": 'Circle Tangent Line makes this exact relation carry correctness: Infer the circle center and radius through the contact. Select the unique candidate center on the forward tangent ray perpendicular to that radius; marker outlines and the opposite ray do not count. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
