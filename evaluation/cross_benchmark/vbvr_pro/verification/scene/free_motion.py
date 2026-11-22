from evaluation.cross_benchmark.vbvr_pro.verification.scene.attention import _attention
from evaluation.cross_benchmark.vbvr_pro.verification.scene.cards import _cards

SUPPORTED_IDS = {"G-39", "O-46"}


def verify(task, frames, first, final, metadata, reference):
    if task == "G-39":
        return _attention(frames, first, metadata, reference)
    if task == "O-46":
        # Native projected poses include the renderer's occlusion boundaries.
        # They are one complete valid-motion proof; arbitrary curved routes
        # have a separate rigid-card compositing proof below.
        from evaluation.cross_benchmark.vbvr_pro.verification.scene.motion import _motion

        native = _motion(task, frames, final, metadata, reference)
        if native["passed"]:
            native["evidence"]["trajectory_reader"] = "ordered_object_pose_dictionary"
            return native
        return _cards(frames, first, final, metadata, reference)
    return None
