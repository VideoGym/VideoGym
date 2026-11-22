TASK_NAME = 'point_rotation'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a hollow pivot marker, a solid black source dot, a prominent indicator arc centered on the pivot, and five labeled candidate circles A-E. The arc span defines the rotation amount, and the arrowhead defines the rotation direction. All candidates lie on the same invisible circle centered on the pivot at the source-to-pivot radius. Rigidly rotate the source through the complete indicated sweep while preserving that radius. Exactly one candidate center coincides with the rotated source position.',
    "design_intent": "The arc's visible signed sweep, rigid rotation about the pivot, and preservation of the source radius carry correctness; the distractors sit on that same circle to either side of the answer, at a rank along it drawn before any of them is placed, so each is wrong only by where it stops on the sweep and no reading of the marker row names the answer. Visual overlays are permitted only when they expose that rotation construction or the terminal commit.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
