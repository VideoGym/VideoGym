TASK_NAME = 'maze_flow_pairs'
TASK_FAMILY = 'constraint_grid'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Numberlink with full cover: join each color's two endpoint dots with one route over side-adjacent cells; routes never share a cell and together cover every cell exactly once; pre-drawn route prefixes anchored at endpoint dots are fixed. Instances space-fill one path per color (two colors on 5x5, three on 6x6 or 7x7) and trim prefixes while keeping the solution uniquely forced by deterministic endpoint-extension and capacity-and-cycle propagation, requiring at least 3 waves, at least one pre-drawn stroke, given stroke edges at most half the route edges, and at least max(8, total/3) open cells.",
    "design_intent": "The family's only multi-object routing task: several colored routes negotiate one shared surface, so every wave visibly reallocates the remaining space and the drawn state is the working memory of the deduction.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
