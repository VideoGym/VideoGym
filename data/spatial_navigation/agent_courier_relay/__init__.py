TASK_NAME = 'agent_courier_relay'
TASK_FAMILY = 'spatial_navigation'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "Blue A carries the parcel to the fixed meeting cell and hands off to orange B, after which B carries the parcel to the green delivery point while A stays put. The parcel can only be handed off on B's cell. Each leg is walked by the current courier itself (up/right/down/left/stay), any legal two-leg route is accepted, and route length is not scored.",
    "design_intent": 'A two-courier relay tests handoff coordination and split-leg routing, with the joint-occupancy rendering keeping the shared-cell identities always distinguishable.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
