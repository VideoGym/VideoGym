TASK_NAME = 'triangle_from_midpoints'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows three hollow circular markers outlined in black and connected by line segments forming the medial triangle (each marker is the midpoint of one side of a hidden triangle), plus five labeled candidate circles A-E. Each original vertex equals the vector sum of its two adjacent side midpoints minus the opposite side midpoint. Reconstruct all three possible original vertices; exactly one candidate center coincides with one of them.',
    "design_intent": 'The medial-triangle identity and its three possible vertex reconstructions carry correctness; distractors target sign mistakes and incorrect midpoint roles. Visual overlays are permitted only when they expose the original-triangle construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
