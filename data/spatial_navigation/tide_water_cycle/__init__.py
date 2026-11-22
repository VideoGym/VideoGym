TASK_NAME = 'tide_water_cycle'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Two tidal rows in anti-phase: row 1 is high tide on even steps, row 2 on odd steps; each logical step contains one agent action and one autonomous-clock tide change, and after a step one may not stop on a high-tide cell. Sampling guarantees that the tide-static shortest path is illegal (timing is load-bearing) and that the reference solution includes stay -- that is, one must enter a row at the low-tide instant by timing (sometimes waiting). Any safe route is accepted, and length is not scored.',
    "design_intent": 'The anti-phase tide schedule tests timing planning around an autonomous clock: one must enter a row in the ebb window, waiting when necessary, since the static shortest path is necessarily flooded.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
