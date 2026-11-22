TASK_NAME = 'odd_one_out_shape'
TASK_FAMILY = 'in_context_learning'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": "The prompt names one property: side count, convexity, horizontal symmetry, or closure. Exactly three shapes share one value and one differs; non-target qualitative properties are held fixed. The four shapes are sampled in two independent layers. Every candidate, outlier included, draws its answer-irrelevant presentation - rotation, anisotropy, vertex-radius spread, angular clustering, vertex jitter, reflection, size and slot offset - independently from one shared distribution, so irregularity carries no signal about which shape is the odd one. Only the outlier then receives the property-bearing deviation, exaggerated well past the majority: a reflex notch whose perpendicular depth is at least a tenth of the shape's diameter against majority corners capped at 170 degrees, a side-count gap of exactly two between four and six sides, a silhouette-level mirror-mismatch margin, or a break of fixed visible length in an otherwise closed outline. The three majority shapes are always sampled independently, never cloned.",
    "design_intent": "Test one named geometric property (side count, convexity, horizontal symmetry, or closure) without allowing shortcut ranking on answer-irrelevant appearance statistics. All four candidates are matched across compactness, edge-length spread, vertex-radius spread, elongation, diameter, area, and painted ink while maintaining clear corner angles and a visible margin on the target property.",
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
