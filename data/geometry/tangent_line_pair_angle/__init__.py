TASK_NAME = 'tangent_line_pair_angle'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A white canvas shows a black circle and one filled black anchor above it, outside the circle, plus five candidate wedges A-E in a row in the lower half showing different openings. The two tangents from the anchor are never drawn: the solver has to derive their opening from the tangency condition, that a line from the anchor touching the circle meets the radius to its contact at a right angle. That right angle makes contact, centre and anchor a right triangle whose hypotenuse is the centre-to-anchor distance and whose opposite leg is the radius, so the half-opening has sine R / |PC| and the full opening is 2 * arcsin(R / |PC|). The required relation is: identify the candidate whose wedge equals that opening. The answer is unique because the five openings are separated by at least 13 degrees, which the wedge rays resolve and which no other candidate's radius-to-distance ratio can reach. Removing the circle leaves no radius and no answer; moving the anchor changes the answer, so both carry the relation.",
    "design_intent": "Tangent Line Pair Angle makes the tangency condition carry correctness. The solver reads the circle's radius and the centre-to-anchor distance from the public input and must convert their ratio into an angle. Visual overlays are permitted only when they expose a necessary construction or the terminal commit, so the tangents and the opening arc appear in the solution and video alone.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
