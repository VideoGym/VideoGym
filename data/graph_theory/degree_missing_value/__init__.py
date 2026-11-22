TASK_NAME = 'degree_missing_value'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Tree-plus-chords graph whose purple-ringed query node has degree >= 2 (>= 3 at max difficulty). The query carries a yellow `?` badge whose anchor is a pure function of the visible geometry (the clearest of sixteen directions), shared with the verifier ROI.',
    "design_intent": "Local-count member of the unified engine; its badge is the family's only rendered non-label glyph and doubles as the terminal answer slot.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
