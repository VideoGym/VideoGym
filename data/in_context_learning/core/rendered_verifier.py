"""Full-video semantic verifier for rendered-answer in-context learning tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.verifiers.source import _extract_expected_structured
from data.in_context_learning.hue.evaluator import (
    _validate_hue,
    _verify_hue_candidate_state_video,
)
from data.in_context_learning.legend_operator_program.evaluator import (
    _verify_legend_operator_candidate_state_video,
)
from data.in_context_learning.rotate_matrix.evaluator import (
    _validate_rotate_matrix,
    _verify_rotate_matrix_candidate_state_video,
)
from data.in_context_learning.shape_outline_fill.evaluator import (
    _verify_shape_outline_candidate_state_video,
)
from data.in_context_learning.spiral_matrix.evaluator import (
    _verify_spiral_matrix_candidate_state_video,
)
from data.in_context_learning.triple_intersection_red.evaluator import (
    _verify_triple_intersection_candidate_state_video,
)

_VERIFIER_PROFILES = {
    "hue": "hue_visible_clue_cycle_multimodal_hard_gate_v3",
    "legend_operator_program": "legend_operator_program_multimodal_hard_gate_v1",
    "rotate_matrix": "rotate_matrix_multimodal_hard_gate_v2",
    "shape_outline_fill": "shape_outline_fill_multimodal_hard_gate_v3",
    "spiral_matrix": "spiral_matrix_multimodal_hard_gate_v3",
    "triple_intersection_red": "triple_intersection_multimodal_hard_gate_v3",
}


def verify_rendered_answer(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
) -> List[EvalRecord]:
    records: List[EvalRecord] = []

    for sample in samples:
        record = EvalRecord(
            sample_id=sample.id,
            task_name=sample.task_name,
            input_asset=sample.primary_input_image,
            task_family=sample.task_family,
            split=sample.split,
        )
        try:
            if sample.task_name not in _VERIFIER_PROFILES:
                raise ValueError(
                    "The rendered-answer verifier has no registered source-semantic "
                    f"profile for {sample.task_name!r}."
                )
            puzzle_image = Path(sample.primary_input_image).expanduser().resolve()
            if not puzzle_image.exists():
                raise FileNotFoundError(f"Puzzle image not found: {puzzle_image}")

            candidate_video = resolve_candidate_video(pred_root, sample.id, sample.split)
            if candidate_video is None:
                raise FileNotFoundError(f"No prediction video found for sample_id={sample.id}")

            structured = _extract_expected_structured(sample)
            if sample.task_name == "hue":
                semantic_metrics, semantic_pass = _validate_hue(structured)
                candidate_metrics, candidate_pass = _verify_hue_candidate_state_video(
                    candidate_video,
                    structured,
                    puzzle_image,
                )
                video_metrics: dict[str, Any] = {
                    "video_process_verifier": "hue_fixed_palette_semantic_single_decode",
                    **semantic_metrics,
                    **candidate_metrics,
                }
                video_pass = bool(semantic_pass and candidate_pass)
            elif sample.task_name == "legend_operator_program":
                candidate_metrics, candidate_pass = _verify_legend_operator_candidate_state_video(
                    candidate_video,
                    structured,
                    puzzle_image,
                )
                video_metrics = {
                    "video_process_verifier": "legend_operator_program_semantic_single_decode",
                    **candidate_metrics,
                }
                video_pass = bool(candidate_pass)
            elif sample.task_name == "rotate_matrix":
                semantic_metrics, semantic_pass = _validate_rotate_matrix(structured)
                candidate_metrics, candidate_pass = _verify_rotate_matrix_candidate_state_video(
                    candidate_video,
                    structured,
                    puzzle_image,
                )
                video_metrics = {
                    "video_process_verifier": "rotate_matrix_semantic_single_decode",
                    **semantic_metrics,
                    **candidate_metrics,
                }
                video_pass = bool(semantic_pass and candidate_pass)
            elif sample.task_name == "shape_outline_fill":
                candidate_metrics, candidate_pass = _verify_shape_outline_candidate_state_video(
                    candidate_video,
                    structured,
                    puzzle_image,
                )
                video_metrics = {
                    "video_process_verifier": "shape_outline_fill_semantic_single_decode",
                    **candidate_metrics,
                }
                video_pass = bool(candidate_pass)
            elif sample.task_name == "spiral_matrix":
                candidate_metrics, candidate_pass = _verify_spiral_matrix_candidate_state_video(
                    candidate_video,
                    structured,
                )
                video_metrics = {
                    "video_process_verifier": "spiral_matrix_semantic_single_decode",
                    **candidate_metrics,
                }
                video_pass = bool(candidate_pass)
            else:
                candidate_metrics, candidate_pass = _verify_triple_intersection_candidate_state_video(
                    candidate_video,
                    structured,
                    puzzle_image,
                )
                video_metrics = {
                    "video_process_verifier": "triple_intersection_semantic_single_decode",
                    **candidate_metrics,
                }
                video_pass = bool(candidate_pass)

            record.prediction_asset = candidate_video.as_posix()
            record.offline_pass = video_pass
            component_scores = dict(video_metrics.get("component_scores") or {})
            failure_reason = str(video_metrics.get("failure_reason") or "video verification failed")
            record.offline_metrics = {
                **video_metrics,
                "video_keyframe_process_check_pass": None,
                "video_keyframe_scope": "semantic_full_sequence",
                "video_keyframe_required_for_pass": False,
                "candidate_video": candidate_video.as_posix(),
                "verifier_profile": _VERIFIER_PROFILES[sample.task_name],
                "artifact_results": {"video": video_pass},
                "component_scores_by_artifact": {"video": component_scores},
                "component_scores": {"video": component_scores},
                "binary_correct": video_pass,
                "failure_reason": "ok" if video_pass else failure_reason,
                "latency_ms": float(video_metrics.get("latency_ms") or 0.0),
            }
        except Exception as exc:  # pragma: no cover - guarded by the integration audit
            record.error = str(exc)
        records.append(record)

    return records


__all__ = ["verify_rendered_answer"]
