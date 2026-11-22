from evaluation.cross_benchmark.vbvr_pro.verification.scene.pipes import _pipes
from evaluation.cross_benchmark.vbvr_pro.verification.scene.sequences import _complete_sequence, _raven, _sequence
from evaluation.cross_benchmark.vbvr_pro.verification.scene.subtraction import _subtraction
from evaluation.cross_benchmark.vbvr_pro.verification.scene.animals import _animal_sort
from evaluation.cross_benchmark.vbvr_pro.verification.scene.books import _books
from evaluation.cross_benchmark.vbvr_pro.verification.scene.merging import _merge_balls
from evaluation.cross_benchmark.vbvr_pro.verification.scene.occlusion import verify as _occlusion

from evaluation.cross_benchmark.vbvr_pro.verification.spatial_3d.voxel import SUPPORTED_IDS as VOXEL_IDS
from evaluation.cross_benchmark.vbvr_pro.verification.assembly.blueprint import SUPPORTED_IDS as BLUEPRINT_IDS
from evaluation.cross_benchmark.vbvr_pro.verification.scene.free_motion import SUPPORTED_IDS as FREE_MOTION_IDS

INVENTORY_IDS = {"G-21", "O-29", "O-30", "O-65"}

MOTION_IDS = {"G-5", "G-8", "O-27"}

SUPPORTED_IDS = (
    BLUEPRINT_IDS
    | VOXEL_IDS
    | FREE_MOTION_IDS
    | INVENTORY_IDS
    | MOTION_IDS
    | {"G-51", "G-193", "O-43", "O-44", "O-45", "O-56", "O-64"}
)


def verify(task, frames, first, final, metadata, reference):
    if task not in SUPPORTED_IDS:
        return None
    if task == "O-64":
        from evaluation.cross_benchmark.vbvr_pro.verification.scene.animal_matching import verify as verify_animal_matching

        return verify_animal_matching(task, frames, first, final, metadata, reference)
    if task in FREE_MOTION_IDS:
        from evaluation.cross_benchmark.vbvr_pro.verification.scene.free_motion import verify as verify_free_motion

        return verify_free_motion(task, frames, first, final, metadata, reference)
    if task in BLUEPRINT_IDS:
        from evaluation.cross_benchmark.vbvr_pro.verification.assembly.blueprint import verify as verify_blueprint

        return verify_blueprint(task, frames, first, final, metadata, reference)
    if task in VOXEL_IDS:
        from evaluation.cross_benchmark.vbvr_pro.verification.spatial_3d.voxel import verify as verify_voxels

        return verify_voxels(task, frames, first, final, metadata, reference)
    if task == "G-21":
        return _occlusion(frames, first, final, metadata)
    if task == "O-30":
        return _books(frames, first, final, metadata)
    if task == "O-65":
        return _animal_sort(frames, first, final, metadata)
    if task == "O-29":
        return _merge_balls(frames, first, metadata)
    if task in MOTION_IDS:
        from evaluation.cross_benchmark.vbvr_pro.verification.scene.motion import _motion

        return _motion(task, frames, final, metadata, reference)
    if task in {"G-51", "G-193"}:
        return _sequence(task, frames, first, final, metadata)
    if task == "O-56":
        return _raven(frames, reference[0] if reference else first, final, metadata)
    if task == "O-43":
        return _subtraction(frames, first, final, metadata)
    if task == "O-45":
        return _complete_sequence(frames, first, final, metadata, reference)
    return _pipes(frames, first, final, metadata)
