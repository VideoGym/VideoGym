from __future__ import annotations

from data.task_spec import TaskFamily, TaskRoute
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.common.verifiers.routes import VerifierRoute

_DIRECT_CHOICE_VERIFIER = VerifierRoute(
    'data.in_context_learning.core.direct_choice_verifier',
    'verify_icl_direct_choice_delivery',
)

_SEQUENCE_COMPLETION_VERIFIER = VerifierRoute(
    'data.in_context_learning.core.sequence_completion_verifier',
    'verify_icl_sequence_completion',
)

FAMILY = TaskFamily(
    name='in_context_learning',
    description='Induce and apply deterministic rules from multi-example sequences, matrices, set regions, legends, or multi-panel visual evidence.',
    tasks=(
        TaskRoute('alternating_size_sequence', 'data.in_context_learning.alternating_size_sequence.generator', 'AlternatingSizeSequencePattern', verifier=_SEQUENCE_COMPLETION_VERIFIER, answer_kind='nonnegative_integer'),
        TaskRoute('analogy_completion', 'data.in_context_learning.analogy_completion.generator', 'AnalogyCompletionPattern', verifier=_DIRECT_CHOICE_VERIFIER, answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('dot_count_progression', 'data.in_context_learning.dot_count_progression.generator', 'DotCountProgressionPattern', verifier=_SEQUENCE_COMPLETION_VERIFIER, answer_kind='nonnegative_integer'),
        TaskRoute('growing_polygon_step', 'data.in_context_learning.growing_polygon_step.generator', 'GrowingPolygonStepPattern', verifier=_SEQUENCE_COMPLETION_VERIFIER, answer_kind='nonnegative_integer'),
        TaskRoute('hue', 'data.in_context_learning.hue.generator', 'HuePattern', verifier=VerifierRoute('data.in_context_learning.core.rendered_verifier', 'verify_rendered_answer'), answer_kind='typed_text'),
        TaskRoute('increasing_size_sequence', 'data.in_context_learning.increasing_size_sequence.generator', 'IncreasingSizeSequencePattern', verifier=_SEQUENCE_COMPLETION_VERIFIER, answer_kind='nonnegative_integer'),
        TaskRoute('legend_operator_program', 'data.in_context_learning.legend_operator_program.generator', 'LegendOperatorProgramPattern', verifier=VerifierRoute('data.in_context_learning.core.rendered_verifier', 'verify_rendered_answer'), answer_kind='grid'),
        TaskRoute('linear_sequence_next', 'data.in_context_learning.linear_sequence_next.generator', 'LinearSequenceNextPattern', verifier=_SEQUENCE_COMPLETION_VERIFIER, answer_kind='positions', position_collection_order='x_then_y'),
        TaskRoute('object_subtraction', 'data.in_context_learning.object_subtraction.generator', 'ObjectSubtractionPattern', verifier=_DIRECT_CHOICE_VERIFIER, answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('odd_one_out_shape', 'data.in_context_learning.odd_one_out_shape.generator', 'OddOneOutShapePattern', verifier=_DIRECT_CHOICE_VERIFIER, answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('rotate_matrix', 'data.in_context_learning.rotate_matrix.generator', 'RotateMatrixPattern', verifier=VerifierRoute('data.in_context_learning.core.rendered_verifier', 'verify_rendered_answer'), answer_kind='grid'),
        TaskRoute('sequence_completion_poly', 'data.in_context_learning.sequence_completion_poly.generator', 'SequenceCompletionPolyPattern', verifier=_SEQUENCE_COMPLETION_VERIFIER, answer_kind='typed_text'),
        TaskRoute('set_inclusion_diagram', 'data.in_context_learning.set_inclusion_diagram.generator', 'SetInclusionDiagramPattern', verifier=_DIRECT_CHOICE_VERIFIER, answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('shape_outline_fill', 'data.in_context_learning.shape_outline_fill.generator', 'ShapeOutlineFillPattern', verifier=VerifierRoute('data.in_context_learning.core.rendered_verifier', 'verify_rendered_answer'), answer_kind='typed_text'),
        TaskRoute('spiral_matrix', 'data.in_context_learning.spiral_matrix.generator', 'SpiralMatrixPattern', verifier=VerifierRoute('data.in_context_learning.core.rendered_verifier', 'verify_rendered_answer'), answer_kind='nonnegative_integer_list'),
        TaskRoute('tile_color_sequence', 'data.in_context_learning.tile_color_sequence.generator', 'TileColorSequencePattern', verifier=_SEQUENCE_COMPLETION_VERIFIER, answer_kind='typed_text'),
        TaskRoute('triple_intersection_red', 'data.in_context_learning.triple_intersection_red.generator', 'TripleIntersectionRedPattern', verifier=VerifierRoute('data.in_context_learning.core.rendered_verifier', 'verify_rendered_answer'), answer_kind='cartesian_point'),
        TaskRoute('venn_intersection_count', 'data.in_context_learning.venn_intersection_count.generator', 'VennIntersectionCountPattern', verifier=_DIRECT_CHOICE_VERIFIER, answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
        TaskRoute('venn_symmetric_difference', 'data.in_context_learning.venn_symmetric_difference.generator', 'VennSymmetricDifferencePattern', verifier=_DIRECT_CHOICE_VERIFIER, answer_kind='option', answer_codec=FOUR_CHOICE_CODEC),
    ),
)

__all__ = ["FAMILY"]
