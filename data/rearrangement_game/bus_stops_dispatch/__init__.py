TASK_NAME = 'bus_stops_dispatch'
TASK_FAMILY = 'rearrangement_game'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A five-stop line holds 3-4 badge-labelled riders; the two-seat bus starts at stop 1 (already serviced). Each move drives one stop left or right; on arrival, riders bound for that stop alight, then waiting riders board left to right up to capacity. Sampling requires the capacity-2 optimum to be strictly longer than the infinite- capacity optimum, so the seat cap is load-bearing. Stops are spoken 1-based; internal indices stay 0-based. The formal payload is the complete visited-stop sequence, including the initial stop and every arrival.',
    "design_intent": 'Capacity-coupled pickup-and-delivery on a line: the two-seat cap forces boarding-order planning and revisits that an uncapacitated bus would skip.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
