TASK_NAME = 'external_angle_bisector'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Two black rays determine a smaller non-reflex angle. Its external angle-bisector locus is the full line through their common vertex perpendicular to the smaller angle's internal bisector. Either opposite ray of that line is valid; the generated candidates contain one unique point on the locus.",
    "design_intent": 'The task tests the external-bisector line, not a mislabeled reflex-angle midpoint and not hidden ray order. The internal and external bisector overlays expose only the stated construction.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
