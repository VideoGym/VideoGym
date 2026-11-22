TASK_NAME = 'mirror_aim_detector'
TASK_FAMILY = 'physics'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A beam enters a walled box through a wall port along a visible entry arrow. Two to four interior plane mirror bars are reflective on both faces but only over their drawn extent. The beam travels straight, reflects specularly at each bar it meets (angle to the surface preserved; r = i - 2(i.n)n), and ends at the first wall point it reaches because walls absorb. Exactly one of four labeled wall detector disks contains that endpoint. Sampling guarantees: the intended contact sequence equals the full re-trace against all bars, contacts stay >=16px inside bar ends, non-target bars keep >=22px clearance from every segment, consecutive contacts stay >=70px apart, detectors stay >=78px from each other, >=44px from the beam path, and away from corners and the port. Decoys are mechanism-targeted: skip-final-reflection, skip-all-mirrors, and a flipped final reflection, backfilled with clean wall positions.',
    "design_intent": "Train chained specular reflection off arbitrarily oriented finite mirrors, where each segment's direction must be constructed from the previous contact instead of pattern-matching one fixed wall orientation.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
