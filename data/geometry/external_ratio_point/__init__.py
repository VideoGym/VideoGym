TASK_NAME = 'external_ratio_point'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'A white canvas shows one filled black start anchor, one hollow through anchor, and five labeled candidate circles A-E, all of them on the ray beyond the hollow anchor. The two anchors define a ray that starts at the filled anchor and passes through the hollow anchor. For the sample-specific integer ratio r shown in the Rule, the target point lies on that ray beyond the hollow anchor and satisfies distance(start anchor, target) = r * distance(through anchor, target). Equivalently, the through-to-target distance is the start-to-through distance divided by r-1. The required relation is: Determine which candidate marks the target point.',
    "design_intent": 'External Ratio Point makes this exact relation carry correctness: A white canvas shows one filled black start anchor, one hollow through anchor, and five labeled candidate circles A-E. The two anchors define a ray that starts at the filled anchor and passes through the hollow anchor. The target point lies on that ray beyond the hollow anchor. For the sample-specific integer ratio r, it satisfies distance(start anchor, target) = r * distance(through anchor, target); equivalently, its through-to-target distance is the start-to-through distance divided by r-1. The required relation is: Determine which candidate marks the target point. Visual overlays are permitted only when they expose a necessary construction or the terminal commit.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
