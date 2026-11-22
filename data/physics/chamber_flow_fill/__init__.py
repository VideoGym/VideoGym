TASK_NAME = 'chamber_flow_fill'
TASK_FAMILY = 'physics'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A side-view tank has 3-6 numbered chambers over one common floor, separated by interior walls whose crest heights differ pairwise by at least 24px. A spout pours at a steady rate into an interior chamber. The receiving chamber rises; on reaching its lowest side wall it spills into that neighbor and holds while the neighbor fills; when a fed chamber catches up to its feeding crest the two connect and rise as one pool, spilling onward at its lowest outer wall; the pour stops when the last dry chamber reaches its feeding crest. The cascade is deterministic and defines the exact level trajectory and first-wet chamber order.',
    "design_intent": 'Make fill ordering a genuinely continuous process with discrete cross-weir events, where the answer emerges from repeated visual height comparisons of a growing pool.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
