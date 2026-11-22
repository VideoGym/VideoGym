TASK_NAME = 'mirror_image_select'
TASK_FAMILY = 'geometry'

TASK_CONTRACT = {
    "task_name": TASK_NAME,
    "task_family": TASK_FAMILY,
    "logic": 'Reflect the asymmetric six-vertex ordered polyline across the vertical axis into the empty right workspace; the answer is the unique candidate pose equal to that complete left-right reflection. Candidates realize four distinct transforms drawn from mirror_lr, identity, flip_ud, rotate_180, rotate_90_cw, rotate_90_ccw; the glyph sampler rejects any polyline whose six transform images are not pairwise distinct, and also rejects reflex turns and near-touching segments, so every candidate comparison has a visible margin. The red start and green end identities must reflect with the path.',
    "design_intent": 'Mirror Image Select makes exact rigid reflection carry correctness: near-miss transforms of the same glyph are the distractors, endpoint identities force orientation awareness, and the guide-then-trace construction is the visible process evidence.',
}

__all__ = ["TASK_NAME", "TASK_FAMILY", "TASK_CONTRACT"]
