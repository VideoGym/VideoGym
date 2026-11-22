TASK_NAME = 'convex_hull_count'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'The convex hull is the tightest convex polygon whose boundary and interior contain every point center; the answer is the number of point centers that are corners of that closed boundary. Generation guarantees an unambiguous classification: 4-6 hull corners, at least 4 strictly interior points, minimum pairwise point distance 30 px, interior points at least 28 px clear of every hull edge, and near-corner points at least 24 px from corner status, so no point reads as on-edge or borderline.',
    "design_intent": 'Convex Hull Count makes boundary-versus-interior classification carry correctness: every point is clearly a hull corner or clearly inside by generation-time margins, and the serial edge construction is the visible process evidence on a clean full-canvas scatter.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
