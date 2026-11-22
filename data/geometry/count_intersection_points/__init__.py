TASK_NAME = 'count_intersection_points'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Treat every blue object as one finite straight segment, not as an infinite line, and count only a strict interior crossing shared by two segments. A segment endpoint never counts; parallel or non-crossing pairs add no point, including pairs whose infinite extensions meet outside either segment. No two visible segments overlap collinearly or form an ambiguous near-parallel crossing; a collinear overlap would not define one finite crossing point. If several segment pairs meet at exactly the same coordinate, that coordinate counts once; all crossing coordinates in the supplied image are distinct.',
    "design_intent": 'Count Intersection Points makes the strict finite-segment crossing definition carry correctness: endpoints never count, parallel and outside-meeting pairs contribute nothing, and every distinct interior crossing is counted exactly once on a clean full-canvas scene whose only output effect is the single cyan marker set.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
