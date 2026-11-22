TASK_NAME = 'packet_router_tick'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Three input queues hold packets with IDs, destinations O1-O3, and TTLs; one tick removes leading TTL-0 packets, lets surviving heads contend per output with cyclic arbitration from the pointer, transfers winners (TTL-1), ages losing heads in place, closes queues toward the head, and advances the pointer. Sampling constructs guaranteed contention, an expiry, and an idle output each sample.',
    "design_intent": 'It predicts a network successor rather than searching a route.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
