TASK_NAME = 'tangent_circle_touch'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows two black circles of different sizes touching each other externally at exactly one point, plus five labeled candidate circles A-E spaced out along one of the two circles. The required relation is: Identify the exact point where the two circles touch. Presentation parity: the kissing point belongs to both rims, so seating the whole row on either one keeps every option on the figure. The row rides a single rim because the two rims curve to opposite sides of their common tangent: a row dealt across both straddles the kiss at every rank and hands "the marker nearest the cloud centroid" the answer. One rim is a long open run -- half its circumference each way -- so all five ranks are reachable and the answer\'s place in the row is drawn uniformly. A distractor is wrong because it sits where the rims have already parted, and it is kept far enough from the other rim to read as parted.',
    "design_intent": 'Tangent Circle Touch makes this exact relation carry correctness: the one point the two externally tangent rims share. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
