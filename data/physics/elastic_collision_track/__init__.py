TASK_NAME = 'elastic_collision_track'
TASK_FAMILY = 'physics'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "A rail carries 12 numbered ticks and 3-4 labeled balls of exactly one tick diameter resting on ticks (pairwise start gaps >= 2). Each ball's integer speed (0-2 ticks per second) and direction are stated by its arrow and numeral; no arrow means at rest. Rules: constant speed between touches; touching balls instantly exchange velocities (equal masses) and never pass through each other; the layout freezes after exactly T whole seconds (T=3-6, stated in the prompt per section 7.4). With integer starts, speeds, and T, every terminal center is exactly on a tick. Sampling guarantees 1-4 collisions with pairwise temporal margins >= 0.3s, no event within 0.35s of the start or of T, all centers within ticks 1-12 throughout, and a terminal layout that differs from the collisionless (pass-through) extrapolation, so the exchange rule is always load-bearing. The answer lists each ball's terminal tick in label order.",
    "design_intent": 'Train event-ordered continuous kinematics where a stated symmetric rule (velocity exchange) must be applied at visually exact contacts and integrated to a stated time.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
