TASK_NAME = 'translation_composition'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows two black arrows representing two translation vectors and a hollow source point, plus five labeled candidate circles A-E. Apply both translations to the source point sequentially. The required relation is: Determine the final position. Presentation parity: the composed image point lies on neither drawn arrow, and a layout in which the correct marker is the only one touching an arrow shaft or head is redrawn, so that coincidence cannot single out the answer.',
    "design_intent": 'Translation Composition makes this exact relation carry correctness: A white canvas shows two black arrows representing two translation vectors and a hollow source point, plus five labeled candidate circles A-E. Apply both translations to the source point sequentially. The required relation is: Determine the final position. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
