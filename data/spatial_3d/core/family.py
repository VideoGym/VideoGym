from __future__ import annotations

from data.task_spec import TaskFamily, TaskRoute
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.common.verifiers.routes import VerifierRoute

FAMILY = TaskFamily(
    name='spatial_3d',
    description='Reason about three-dimensional structure: paper folding and unfolding, cube-net assembly, polycube rotation, orthographic views, and solid cross-sections.',
    tasks=(
        TaskRoute('cube_net_fold', 'data.spatial_3d.cube_net_fold.generator', 'CubeNetFoldPattern', verifier=VerifierRoute('data.spatial_3d.core.verifier', 'verify_spatial_3d'), answer_kind='nonnegative_integer'),
        TaskRoute('paper_fold_hole_matrix', 'data.spatial_3d.paper_fold_hole_matrix.generator', 'PaperFoldHoleMatrixPattern', verifier=VerifierRoute('data.spatial_3d.core.verifier', 'verify_spatial_3d'), answer_kind='grid'),
        TaskRoute('slice_cross_section', 'data.spatial_3d.slice_cross_section.generator', 'SliceCrossSectionPattern', verifier=VerifierRoute('data.spatial_3d.core.verifier', 'verify_spatial_3d'), answer_kind='typed_text'),
        TaskRoute('polycube_rotation', 'data.spatial_3d.polycube_rotation.generator', 'PolycubeRotationPattern', verifier=VerifierRoute('data.spatial_3d.core.verifier', 'verify_spatial_3d'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('stack_orbit_view', 'data.spatial_3d.stack_orbit_view.generator', 'StackOrbitViewPattern', verifier=VerifierRoute('data.spatial_3d.core.verifier', 'verify_spatial_3d'), answer_kind='grid'),
        TaskRoute('voxel_three_views', 'data.spatial_3d.voxel_three_views.generator', 'VoxelThreeViewsPattern', verifier=VerifierRoute('data.spatial_3d.core.verifier', 'verify_spatial_3d'), answer_kind='grid'),
    ),
)

__all__ = ["FAMILY"]
