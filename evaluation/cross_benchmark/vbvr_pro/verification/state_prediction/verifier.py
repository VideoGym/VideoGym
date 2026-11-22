from evaluation.cross_benchmark.vbvr_pro.verification.physics.vessels import _vessels
from evaluation.cross_benchmark.vbvr_pro.verification.state_prediction.clock import _clock
from evaluation.cross_benchmark.vbvr_pro.verification.state_prediction.graphs import _graph
from evaluation.cross_benchmark.vbvr_pro.verification.state_prediction.grids import _lights, _shift
from evaluation.cross_benchmark.vbvr_pro.verification.state_prediction.panel import _panel

SUPPORTED_IDS = {"G-31", "O-36", "O-37", "O-53", "O-54", "O-75"}


def verify(task, frames, first, final, metadata, reference):
    task_id = task.split("_", 1)[0]
    if task_id not in SUPPORTED_IDS:
        return None
    if not metadata.get("parameters"):
        raise ValueError("Task parameters unavailable")
    if task_id == "G-31":
        return _graph(frames, first, metadata["parameters"], final)
    if task_id == "O-36":
        return _shift(frames, first, metadata["parameters"], final, metadata)
    if task_id == "O-37":
        return _lights(frames, first, metadata)
    if task_id == "O-53":
        return _clock(frames, first, metadata["parameters"], reference)
    if task_id == "O-54":
        return _panel(frames, first, metadata)
    if task_id == "O-75":
        return _vessels(frames, first, metadata)
    return None
