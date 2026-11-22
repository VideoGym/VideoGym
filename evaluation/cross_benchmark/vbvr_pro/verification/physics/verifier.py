from __future__ import annotations
from evaluation.cross_benchmark.vbvr_pro.verification.physics.buoyancy import _buoyancy
from evaluation.cross_benchmark.vbvr_pro.verification.physics.colour_mixing import _additive, _subtractive
from evaluation.cross_benchmark.vbvr_pro.verification.physics.dominoes import _branch_dominos, _gap_dominos
from evaluation.cross_benchmark.vbvr_pro.verification.physics.motion import _bounces, _eating, _gravity, _rolling
from evaluation.cross_benchmark.vbvr_pro.verification.physics.optics import _optics
import cv2

PREFIXES = frozenset({"O-2", "O-15", "O-16", "O-18", "O-19", "O-23", "O-24", "O-31", "O-32", "O-62", "G-273"})


def verify(task, frames, first, final, metadata, reference):
    prefix = task.split("_")[0]
    if prefix not in PREFIXES:
        return None
    # Keep all decoded frames; only resize the observation canvas for bounded CV cost.
    declared = metadata.get("generic_declarative_render", {}).get("canvas", {})
    width = declared.get("width", first.shape[1])
    scale = min(1.0, 512 / first.shape[1])

    def resize(f):
        return cv2.resize(f, (round(f.shape[1] * scale), round(f.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    small_first = resize(first)
    small_frames = [resize(f) for f in frames]
    small_final = resize(final)
    geometry_scale = small_first.shape[1] / width
    if prefix == "O-15":
        return _bounces(small_frames, small_first, metadata, geometry_scale, small_final)
    if prefix == "O-2":
        return _subtractive(small_frames, small_first, metadata, geometry_scale)
    if prefix == "O-16":
        return _additive(small_frames, small_first, metadata, geometry_scale)
    if prefix in {"O-18", "O-19"}:
        return _optics(prefix, small_frames, small_first, metadata, geometry_scale)
    if prefix == "G-273":
        return _buoyancy(small_frames, small_first, metadata, geometry_scale, small_final)
    if prefix == "O-62":
        return _gravity(small_frames, small_first, metadata, geometry_scale, small_final)
    if prefix == "O-32":
        return _rolling(small_frames, small_first, metadata, geometry_scale, small_final)
    if prefix == "O-23":
        return _branch_dominos(small_frames, small_first, metadata, geometry_scale)
    if prefix == "O-24":
        return _gap_dominos(small_frames, small_first, metadata, geometry_scale)
    return _eating(small_frames, small_first, metadata, geometry_scale)
