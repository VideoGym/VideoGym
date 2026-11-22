TASK_NAME = 'protocol_handshake_tick'
TASK_FAMILY = 'next_state_prediction'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A client/server pair communicates through an oldest-first message lane for a stated 2-6 ticks across normal, timeout-delay, and duplicate-HELLO variants: deliveries drive the endpoint machines (START->WAIT+HELLO, HELLO->READY+ACK, ACK->OPEN+DATA), and an undelivered-ACK WAIT client decrements its timeout, resending HELLO at zero. Every response joins the tail one tick later.',
    "design_intent": 'Two machines communicate through persistent channel state.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
