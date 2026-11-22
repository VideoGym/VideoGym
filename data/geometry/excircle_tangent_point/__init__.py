TASK_NAME = 'excircle_tangent_point'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A white canvas shows a black triangle outline with one vertex marked by a filled dot, and five labeled candidate markers A-E spread along the side opposite that marked vertex. The excircle opposite the marked vertex is never drawn: the solver has to place its contact from the stated rule, that this circle lies outside the triangle on the far side of the opposite side and touches that side plus the extensions of the two sides meeting at the marked vertex. Equal tangent lengths from the marked vertex make each of them the triangle's half-perimeter, so the contact sits at s - c from one end of the opposite side and s - b from the other, where b and c are the two sides at the marked vertex and s is the half-perimeter. The required relation is: identify the candidate at that contact. The answer is unique because a circle meets a line it is tangent to at exactly one point, and the markers are separated along the side by more than a marker diameter. Removing the marked dot leaves three different contacts and no answer; removing the triangle leaves no contact at all, so both carry the relation.",
    "design_intent": "Excircle Tangent Point makes the excircle's defining tangency carry correctness. Because the excircle is absent from the input, the solver must use the marked vertex, the two sides that meet there, and the equal-tangent-length identity; locating the contact requires the tangency relation rather than a visible circle-side intersection. Visual overlays are permitted only when they expose a necessary construction or the terminal commit, so the excircle arc appears in the solution and video alone.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
