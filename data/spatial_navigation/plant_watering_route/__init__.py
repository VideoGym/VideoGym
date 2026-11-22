TASK_NAME = 'plant_watering_route'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A 4x4 garden: the gardener starts from the well holding a watering can filled with 2 cups; stepping onto a dry flowerpot with water in the can waters it automatically (can -1), and re-entering the well after leaving it uses the single refill and refills to full, while re-entering the well beyond the budget is illegal; the goal is to water all 3-4 pots. Sampling guarantees: the unlimited-capacity plan is strictly shorter (capacity is genuinely load-bearing), and the reference plan uses exactly 1 refill. Any legal action sequence that waters all pots is accepted.',
    "design_intent": 'The capacity-supply loop forces the watering order to detour through the well: the unlimited-capacity shortcut is excluded in sampling, testing small-scale resource-tour planning.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
