TASK_NAME = 'shortest_path_choice'
TASK_FAMILY = 'graph_theory'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'The source is a connected undirected numeric-labelled graph (6-10 nodes, tree plus 1-3 chords) with green start and red goal roles at drawn distance >= 3 (4 at max difficulty). Generation proves the fewest-edge simple route unique before publishing anything.',
    "design_intent": 'Baseline optimization member of the unified labelled-graph engine; shares source schema, renderer, replay, and verifier with the other 16 tasks while isolating fewest-edge route search.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
