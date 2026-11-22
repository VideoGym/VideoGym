TASK_NAME = 'dilation_to_circle'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows two black circle outlines of different radii and separated centers. The external center of similitude of two circles is the unique point on the line through their centers from which a positive dilation maps one circle onto the other; equivalently, it is the intersection of the two common external tangents. Five labeled candidate markers A-E stand clear of both discs, the line through the two centres included. The required relation is: Identify the candidate at the external center of similitude.',
    "design_intent": 'Dilation To Circle makes this exact relation carry correctness: A white canvas shows two black circle outlines of different radii and separated centers. The external center of similitude of two circles is the unique point on the line through their centers from which a positive dilation maps one circle onto the other; equivalently, it is the intersection of the two common external tangents. Five labeled candidate markers A-E are scattered. The required relation is: Identify the candidate at the external center of similitude. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
