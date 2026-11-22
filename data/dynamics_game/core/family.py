from __future__ import annotations

from data.task_spec import TaskFamily, TaskRoute
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.common.verifiers.routes import VerifierRoute

FAMILY = TaskFamily(
    name='dynamics_game',
    description='Apply shooting, dropping, falling-block, merging, or clearing updates in arcade-style discrete scenes.',
    tasks=(
        TaskRoute('block_blast_clear_predict', 'data.dynamics_game.block_blast_clear_predict.generator', 'BlockBlastClearPredictGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('breakout_brick_clear_predict', 'data.dynamics_game.breakout_brick_clear_predict.generator', 'BreakoutBrickClearPredictGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('bubble_shooter_chain_predict', 'data.dynamics_game.bubble_shooter_chain_predict.generator', 'BubbleShooterChainPredictGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('candy_crush_special_predict', 'data.dynamics_game.candy_crush_special_predict.generator', 'CandyCrushSpecialPredictGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('jewel2_future_board', 'data.dynamics_game.jewel2_future_board.generator', 'Jewel2FutureBoardGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('lane_tap_step', 'data.dynamics_game.lane_tap_step.generator', 'RhythmGameTapStepGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('lemmings_path_predict', 'data.dynamics_game.lemmings_path_predict.generator', 'LemmingsPathPredictGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('merge_swipe_score', 'data.dynamics_game.merge_swipe_score.generator', 'Game2048BestSwipeGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='nonnegative_integer'),
        TaskRoute('mr_driller_drop_predict', 'data.dynamics_game.mr_driller_drop_predict.generator', 'MrDrillerDropPredictGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('pegboard_marble_drop_predict', 'data.dynamics_game.pegboard_marble_drop_predict.generator', 'PegboardMarbleDropPredictGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('snake_path', 'data.dynamics_game.snake_path.generator', 'SnakePathGenerator', verifier=VerifierRoute('data.dynamics_game.snake_path.evaluator', 'verify_snake_path'), answer_kind='path'),
        TaskRoute('sokoban', 'data.dynamics_game.sokoban.generator', 'SokobanPlanGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='path'),
        TaskRoute('tetris_future_board', 'data.dynamics_game.tetris_future_board.generator', 'TetrisFutureBoardGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='grid'),
        TaskRoute('track_match_shot', 'data.dynamics_game.track_match_shot.generator', 'ZumaTrackShotGenerator', verifier=VerifierRoute('data.dynamics_game.core.verifier', 'verify_dynamics_game'), answer_kind='grid'),
    ),
)

__all__ = ["FAMILY"]
