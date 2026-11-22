TASK_NAME = 'rotation'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Each sample requests exactly one of three rigid transforms: 90 degrees clockwise, 180 degrees, or 90 degrees counterclockwise. Apply the displayed transform to the entire filled asymmetric polygon about the center of its tile. Clockwise and counterclockwise are read from the viewer's perspective. The polygon outline is one rigid object. Translation, reflection, scaling, stretching, cropping, recoloring, or partial deformation is forbidden. Exactly one candidate must have the complete final pose.",
    "design_intent": "Rotation makes the sample-requested 90-degree clockwise, 180-degree, or 90-degree counterclockwise relation carry correctness. Apply it to the entire filled asymmetric polygon about the center of its tile. Clockwise and counterclockwise are read from the viewer's perspective. Translation, reflection, scaling, stretching, cropping, recoloring, or partial deformation is forbidden. Exactly one candidate must have the complete final pose. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
