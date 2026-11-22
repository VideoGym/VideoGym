TASK_NAME = 'inverse_rotation'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows a solid black filled-disc pivot point, a hollow image point (the result of rotation), a prominent indicator arc centered on the pivot, and five labeled candidate circles A-E. The arc span defines the rotation amount, and the arrowhead defines the reverse rotation direction from the hollow image point back to the hidden original point. All candidates lie on the same invisible circle centered on the pivot at the image-to-pivot radius. The required relation is: Find the original point by following the arrowed reverse rotation from the hollow image point.',
    "design_intent": 'Inverse Rotation makes this exact relation carry correctness: A white canvas shows a solid black filled-disc pivot point, a hollow image point (the result of rotation), a prominent indicator arc centered on the pivot, and five labeled candidate circles A-E. The arc span defines the rotation amount, and the arrowhead defines the reverse rotation direction from the hollow image point back to the hidden original point. All candidates lie on the same invisible circle centered on the pivot at the image-to-pivot radius. The required relation is: Find the original point by following the arrowed reverse rotation from the hollow image point. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
