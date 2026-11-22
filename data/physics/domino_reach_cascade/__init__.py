TASK_NAME = 'domino_reach_cascade'
TASK_FAMILY = 'physics'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Top-down domino network. Nodes are numbered upright dominoes with printed heights H; dashed links carry printed gaps G, and the drawn base-to-base distance of each link equals G at the sample's uniform pixel scale. Domino 1 is pushed along the visible arrow. A toppling domino pivots at its base and its tip lands exactly H away along its topple direction; it knocks over a linked neighbor iff H >= G on that link, otherwise the tip settles short and the whole branch behind that link never falls. Struck dominoes topple away from their striker and apply the same rule along their own outgoing links; two children of one faller start simultaneously. The network is a tree, so every domino is struck at most once and the cascade is deterministic. Sampling guarantees: 5-8 dominoes, at least one branch node with two children inside the fallen region, at least one failed link from a fallen domino, 3..N-1 fallen, |H-G| >= 8 units on every link, and H-G <= 26 so landings stay near their targets. The answer is the ascending list of fallen domino numbers.",
    "design_intent": 'Make a numeric threshold rule control spatial propagation through a branching layout, so the model must bind printed values to drawn geometry instead of reading a linear left-to-right chain.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
