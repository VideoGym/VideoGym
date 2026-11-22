TASK_NAME = 'incircle_tangent_point'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'The candidate cluster identifies one target side of the triangle. The three internal angle bisectors meet at the incenter. Using the distance from that center to the target side as the incircle radius determines the unique point where the incircle touches the side. Presentation parity: all five markers lie on the marked side itself, so incidence with the triangle is answer-free. A distractor is wrong because the incircle does not reach the side there; it stays at least one marker clearance along the side from the contact and keeps a marker\'s clearance from both of that side\'s corners. The triangle is built from its three tangent lengths rather than from three sampled vertices: the contact divides the marked side into exactly the two tangent runs of that side\'s vertices, so bounding those runs is the same statement as "the row has room on either side of the answer", and the split between them is drawn uniformly across every share the angle bounds admit. A contact parked near the middle of its own side would let "the marker nearest that side\'s midpoint" answer the sample without ever constructing the incircle.',
    "design_intent": "The answer is carried by the incenter and the incircle's center-to-side radius relation, not by candidate proximity. Visual overlays are limited to the three bisectors, the center marker, the incircle, and the terminal tangent-point commit.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
