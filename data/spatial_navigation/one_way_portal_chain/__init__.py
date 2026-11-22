TASK_NAME = 'one_way_portal_chain'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Slate barriers cut the ground into four sealed chambers, and the only cross-chamber means are three pairs of one-way portals S1->D1, S2->D2, S3->D3 (stepping onto a source teleports immediately to the matching destination, which is ordinary floor); the chamber-chain structure forces the use order S1, S2, S3. Sampling guarantees: without portals S-G is disconnected, removing any one source also disconnects it (each portal is indispensable), and the reference path uses each of the three portals exactly once. A source cell cannot be a stopping cell because it teleports immediately.',
    "design_intent": 'One-way portals and sealed chambers together define a topological order: tests reading an asymmetric connectivity structure and planning by chain, with teleports presented as two-frame snapshot pairs rather than continuous passage.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
