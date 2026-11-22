TASK_NAME = 'diameter_opposite'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A white canvas shows a black circle, one hollow source marker on the circle boundary, and five labeled candidate circles A-E. The required relation is: Determine which candidate is diametrically opposite the source point across the circle center. Presentation parity: the antipode is a point of the drawn rim, so all five markers are points of that rim and incidence with the circle is answer-free. A distractor is wrong because the chord from the source marker misses the centre there; the option scatter also keeps a marker's clearance from the source marker itself.",
    "design_intent": 'Diameter Opposite makes this exact relation carry correctness: A white canvas shows a black circle, one hollow source marker on the circle boundary, and five labeled candidate circles A-E. The required relation is: Determine which candidate is diametrically opposite the source point across the circle center. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
