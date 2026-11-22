TASK_NAME = 'perpendicular'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Determine which candidate lies on the line through the small circle that is perpendicular to the reference line. Presentation parity: the answer lies on none of the drawn ink, and the candidate row is kept clear of the reference segment by a marker's diameter so the correct marker can never become the only one touching the drawing. The row therefore sits beyond the reference segment along the normal rather than straddling it.",
    "design_intent": 'Perpendicular makes this exact relation carry correctness: Determine which candidate lies on the line through the small circle that is perpendicular to the reference line. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
