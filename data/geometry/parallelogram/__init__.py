TASK_NAME = 'parallelogram'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows three black vertices connected as two adjacent sides of a parallelogram, plus five labeled candidate circles A-E near the missing corner. The required relation is: Determine which candidate is the fourth vertex that closes the parallelogram.',
    "design_intent": 'Parallelogram makes this exact relation carry correctness: A white canvas shows three black vertices connected as two adjacent sides of a parallelogram, plus five labeled candidate circles A-E near the missing corner. The required relation is: Determine which candidate is the fourth vertex that closes the parallelogram. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
