"""High-score contamination and same-question ranking have distinct denominators."""

from evaluation.cross_benchmark.vbvr_pro.comparison import parse_pairs
from evaluation.cross_benchmark.vbvr_pro.discrimination import (
    discrimination,
    question_pairs,
)


def test_high_score_failures_and_tie_aware_auc():
    rows = [{'strict_binary_score': label, 'continuous_score': score}
            for label, score in [(1, 1), (1, .9), (0, 1), (0, .2), (None, .99)]]
    result = discrimination(rows)
    assert result['samples'] == 5 and result['resolved'] == 4
    assert result['auroc'] == .625
    assert result['thresholds'][0]['incorrect_share'] == 1 / 3
    assert result['thresholds'][-1]['incorrect_share'] == .5
    assert result['thresholds'][-1]['correct_recall'] == .5
    assert sum(b['correct'] for b in result['histogram']) == 2
    assert sum(b['incorrect'] for b in result['histogram']) == 2


def test_pairing_never_compares_different_questions():
    rows = [dict(sample_id=q, model=m, strict_binary_score=y, continuous_score=r)
            for q, m, y, r in [('q1', 'a', 1, 1), ('q1', 'b', 0, 1),
                               ('q2', 'a', 1, .9), ('q2', 'b', 0, 1)]]
    result = question_pairs(rows)
    assert result['pairs'] == 2 and result['questions'] == 2
    assert result['ties'] == result['inversions'] == result['both_perfect'] == 1
    assert result['pairwise_auc'] == .25


def test_question_pairs_handles_single_model_without_pairs():
    rows = [
        dict(sample_id='q1', model='custom', strict_binary_score=1, continuous_score=0.9),
        dict(sample_id='q2', model='custom', strict_binary_score=0, continuous_score=0.8),
    ]
    result = question_pairs(rows)
    assert result['pairs'] == 0
    assert result['pairwise_auc'] is None


def test_parse_pairs_accepts_arbitrary_models():
    parsed = parse_pairs([["my-model", "continuous.json", "strict.jsonl"]])
    assert parsed[0][0] == "my-model"
    assert parsed[0][1].name == "continuous.json"
    assert parsed[0][2].name == "strict.jsonl"
