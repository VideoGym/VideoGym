TASK_NAME = 'tetris_future_board'
TASK_FAMILY = 'dynamics_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Generate the fixed board as a compact but irregular mid-game stack with uneven column surfaces and a few covered holes/overhangs consistent with legal prior play, keeping the active piece's columns clear to the floor so its drop completes the bottom row; the displayed initial state has no more than one wholly empty row and no empty settled row trapped below occupied fixed cells. Move the four active cells straight downward as one rigid shape without rotation or horizontal motion until one further row would be out of bounds or overlap a fixed block. Lock those cells as fixed blocks, remove every fully occupied 10-cell row simultaneously, then shift each surviving row above removed rows downward by the number of cleared rows beneath it and insert empty rows at the top. Samples guarantee at least one clear.",
    "design_intent": 'Couple continuous-looking rigid fall/contact with a global simultaneous deletion and stable row reindexing; preserve the exact-grid answer kind instead of converting the task to multiple choice.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
