from __future__ import annotations

from data.task_spec import TaskFamily, TaskRoute
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.common.verifiers.routes import VerifierRoute

FAMILY = TaskFamily(
    name='physics',
    description='Track real-world system states such as motion, collision, reflection, and domino propagation.',
    tasks=(
        TaskRoute('ball_bounces_given_time', 'data.physics.ball_bounces_given_time.generator', 'BallBouncesGivenTimePattern', verifier=VerifierRoute('data.physics.core.verifier', 'verify_physics'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('chamber_flow_fill', 'data.physics.chamber_flow_fill.generator', 'ChamberFlowFillPattern', verifier=VerifierRoute('data.physics.core.verifier', 'verify_physics'), answer_kind='nonnegative_integer_list'),
        TaskRoute('domino_reach_cascade', 'data.physics.domino_reach_cascade.generator', 'DominoReachCascadePattern', verifier=VerifierRoute('data.physics.core.verifier', 'verify_physics'), answer_kind='nonnegative_integer_list'),
        TaskRoute('elastic_collision_track', 'data.physics.elastic_collision_track.generator', 'ElasticCollisionTrackPattern', verifier=VerifierRoute('data.physics.core.verifier', 'verify_physics'), answer_kind='nonnegative_integer_list'),
        TaskRoute('gear_train_direction', 'data.physics.gear_train_direction.generator', 'GearTrainDirectionPattern', verifier=VerifierRoute('data.physics.core.verifier', 'verify_physics'), answer_kind='typed_text'),
        TaskRoute('mirror_aim_detector', 'data.physics.mirror_aim_detector.generator', 'MirrorAimDetectorPattern', verifier=VerifierRoute('data.physics.core.verifier', 'verify_physics'), answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('water_pour_levels', 'data.physics.water_pour_levels.generator', 'WaterPourLevelsPattern', verifier=VerifierRoute('data.physics.core.verifier', 'verify_physics'), answer_kind='nonnegative_integer_list'),
    ),
)

__all__ = ["FAMILY"]
