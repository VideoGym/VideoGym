"""Shared answer and verifier contracts for option-letter geometry counting tasks."""

from __future__ import annotations

from typing import Any, Dict

from data.common.typed_answers import FOUR_CHOICE_CODEC


COUNTING_COMPONENTS = {
    "text": ["parse_score", "source_replay_score", "candidate_uniqueness_score", "final_state_score"],
    "image": ["parse_score", "source_replay_score", "preservation_score", "final_state_score"],
    "video": [
        "parse_score",
        "source_replay_score",
        "state_order_score",
        "temporal_continuity_score",
        "preservation_score",
        "final_state_score",
    ],
}

COUNTING_PERFORMANCE_BUDGET = {
    "text_warm_p95_ms": 5.0,
    "image_warm_p95_ms": 250.0,
    "video_warm_p95_ms": 2500.0,
    "video_min_samples_per_second": 0.4,
    "peak_python_memory_mb": 64.0,
    "regression_tolerance_fraction": 0.15,
}


def augment_counting_contract(
    sample: Dict[str, Any],
    *,
    verifier_profile: str,
    semantic_source: str,
) -> Dict[str, Any]:
    structured = sample["structured_answer"]
    structured["answer_schema"] = {
        "schema_version": "uppercase_option_letter_v1",
        "kind": "option_letter",
        "parser_profile": "strict_uppercase_option_letter_a_to_d",
        "format_template": FOUR_CHOICE_CODEC.schema,
        "format_example": FOUR_CHOICE_CODEC.example,
        "judged_artifacts": {
            "ti2v": ["full_video"],
        },
        "accepts_multiple_valid_answers": False,
        "extra_text_allowed": False,
    }
    structured["verifier_contract"] = {
        "profile": verifier_profile,
        "semantic_source": semantic_source,
        "component_scores": {mode: list(names) for mode, names in COUNTING_COMPONENTS.items()},
        "binary_predicate": (
            "strict option parse, independent visible-structure count replay, unique numeric candidate, exact final "
            "selection, fixed-content preservation, and monotonic whole-video tally order"
        ),
        "performance_budget": dict(COUNTING_PERFORMANCE_BUDGET),
    }
    task_contract = dict(sample["task_contract"])
    task_contract["answer"] = {
        **dict(task_contract["answer"]),
        "parser_profile": "strict_uppercase_option_letter_a_to_d",
        "verifier_profile": verifier_profile,
    }
    sample["task_contract"] = task_contract
    sample["verifier_contract"] = structured["verifier_contract"]
    return sample


__all__ = ["COUNTING_COMPONENTS", "COUNTING_PERFORMANCE_BUDGET", "augment_counting_contract"]
