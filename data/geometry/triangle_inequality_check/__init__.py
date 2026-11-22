TASK_NAME = 'triangle_inequality_check'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "One 512x512 scene shows four labeled rows A-D. Each row displays one unordered triple of exact positive-integer side lengths as same-scale colored segments with their printed values. A triple forms a non-degenerate triangle exactly when the sum of its two shorter values is strictly greater than its longest value; equality and smaller sums fail. Valid rows are sampled with margin >= 2 so no valid row is a near-miss, and exactly one row fails. Sampling is also constructed so the shorter of every row's two comparison bars is at least 9 units: below that a bar is no longer than the pair of endpoint dots it grows out of, so neither a reader nor the stroke verifier can see it advance. The task is to name the failing row.",
    "design_intent": 'Tests exact integer comparison alongside same-scale segment bars that make the sum-versus-longest-side inequality visually explicit across ordered rows.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
