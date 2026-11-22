TASK_NAME = 'inscribed_square_vertex'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'One triangle side is drawn in blue and is the square\'s base. The unique inscribed square has one full side on it and both raised corners on the two black sides. The option row lies along one of those black sides, and the target is the raised corner on that side. Presentation parity: all five markers lie on the triangle side that hosts the answer corner, so neither "outside the triangle" nor "not on the triangle" can eliminate an option. A distractor is wrong because the square built on the blue side cannot reach that side there; it keeps a marker\'s clearance from both of that side\'s corners. The raised corner sits at base/(height+base) of the host side, so the base is the marked side rather than the longest one: with the base forced to be longest that share could never fall below 0.54 and the host side could never be long enough for four markers past the answer on either side, which left the answer in the middle of its own row. The share and the host side length are now drawn across every shape the declared angle band admits.',
    "design_intent": "The blue base side and the option row together make the requested square corner invariant to canvas rotation and remove any dependence on hidden vertex order or the ambiguous phrase 'top-left'. Visual overlays are permitted only when they expose the square construction or terminal commit.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
