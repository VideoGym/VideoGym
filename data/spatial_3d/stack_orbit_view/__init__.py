TASK_NAME = 'stack_orbit_view'
TASK_FAMILY = 'spatial_3d'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Colored unit cubes (five-colour palette, each cube one solid colour) stand in stacks on a 3x3 floor. A bold blue arrow looks back at the scene from the far side (behind, or the far-left side). The answer is the letter grid of what that viewer sees: each view cell shows the colour of the nearest cube along its straight ray (nearer stacks hide cubes behind them) or a dot when the ray is empty, with view row 1 at the top and view column i mirrored across the far-side perspective (floor index size + 1 - i). Sampling requires at least one occluded cube and at least two colours in the visible answer.',
    "design_intent": 'Train perspective-taking (SenseNova-SI coordinate): predicting what a differently-placed viewer sees, including the left-right swap and which objects their view hides.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
