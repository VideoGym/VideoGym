TASK_NAME = 'isosceles_trapezoid'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Call the open polyline's bend J, its farther endpoint U, and its nearer endpoint V. Select the unique X for which J-U-X-V is strictly convex, VX is a positive shorter base parallel to JU in the same direction, and UX equals JV. Reject crossings, concavity, and merely near-equal cases.",
    "design_intent": 'Isosceles Trapezoid makes this exact relation carry correctness: Select the unique X for which J-U-X-V is strictly convex, VX is a positive shorter base parallel to JU in the same direction, and UX equals JV. Reject crossings, concavity, and merely near-equal cases. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
