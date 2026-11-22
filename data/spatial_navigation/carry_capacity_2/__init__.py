TASK_NAME = 'carry_capacity_2'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Four equivalent wooden crates, one delivery zone Z, capacity 2, at most two non-empty deliveries. Stepping onto a crate cell picks it up automatically (within capacity), and stepping into Z while carrying crates drops them all automatically; sampling guarantees: capacity 1 is unsolvable, capacity 4 is strictly shorter (capacity is genuinely load-bearing), the legal solution is necessarily a 2+2 two-trip structure, and one may stop at G only after everything is delivered. Any complete route satisfying the constraints is accepted, and route length is not scored.',
    "design_intent": 'Capacity and the delivery budget together force a two-round-trip structure, testing multi-subgoal ordering and resource-cycle planning; the capacity-ignoring shortcut is excluded by sampling.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
