from __future__ import annotations

from data.task_spec import TaskFamily, TaskRoute
from data.common.verifiers.routes import VerifierRoute

FAMILY = TaskFamily(
    name='constraint_grid',
    description='Complete constraint grids by region, adjacency, connectivity, loop, flow, or placement rules.',
    tasks=(
        TaskRoute('aquarium_completion', 'data.constraint_grid.aquarium_completion.generator', 'AquariumCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='grid'),
        TaskRoute('campsite_completion', 'data.constraint_grid.campsite_completion.generator', 'CampsiteCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='positions', position_collection_order='unordered'),
        TaskRoute('fillomino_completion', 'data.constraint_grid.fillomino_completion.generator', 'FillominoCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='grid'),
        TaskRoute('light_up_akari_completion', 'data.constraint_grid.light_up_akari_completion.generator', 'LightUpAkariCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='positions', position_collection_order='unordered'),
        TaskRoute('masyu_completion', 'data.constraint_grid.masyu_completion.generator', 'MasyuCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='path'),
        TaskRoute('maze_flow_pairs', 'data.constraint_grid.maze_flow_pairs.generator', 'MazeFlowPairsGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='named_paths'),
        TaskRoute('nqueens_completion', 'data.constraint_grid.nqueens_completion.generator', 'NQueensCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='positions', position_collection_order='unordered'),
        TaskRoute('nurikabe_completion', 'data.constraint_grid.nurikabe_completion.generator', 'NurikabeCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='grid'),
        TaskRoute('shikaku_completion', 'data.constraint_grid.shikaku_completion.generator', 'ShikakuCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='typed_text'),
        TaskRoute('slitherlink_completion', 'data.constraint_grid.slitherlink_completion.generator', 'SlitherlinkCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='typed_text'),
        TaskRoute('star_battle_completion', 'data.constraint_grid.star_battle_completion.generator', 'StarBattleCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='positions', position_collection_order='unordered'),
        TaskRoute('yajilin_completion', 'data.constraint_grid.yajilin_completion.generator', 'YajilinCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='path'),
        TaskRoute('yin_yang_completion', 'data.constraint_grid.yin_yang_completion.generator', 'YinYangCompletionGenerator', verifier=VerifierRoute('data.constraint_grid.core.verifier', 'verify_grid_completion'), answer_kind='grid'),
    ),
)

__all__ = ["FAMILY"]
