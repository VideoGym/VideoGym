"""Score full video rollouts through the same registered verifier as evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from data.registry import TASK_SPECS
from data.common.reward_contract import RAW_REWARD_DIMENSION_NAMES
from data.common.video_encoding import encode_rgb_frames_to_webm
from training.data.samples import DecodedVideo, Sample, VideoMapping
from training.data.video import load_image, load_video, restore_video


class RewardEvaluationError(RuntimeError):
    """A scorer failed; this must not become a negative training example."""


@dataclass(frozen=True)
class RewardResult:
    sample_id: str
    reward: float
    components: dict[str, float]
    metrics: dict[str, Any]


def _scoring_failure(metrics: dict[str, Any]) -> str | None:
    """Read explicit scorer/decode errors without treating rule failures as errors."""
    for name in (
        "whole_video_error",
        "video_error",
        "source_replay_error",
        "video_source_terminal_render_error",
    ):
        if metrics.get(name):
            return f"{name}: {metrics[name]}"
    if metrics.get("whole_video_loaded") is False:
        return "whole_video_loaded=False: scorer did not load the supplied full video"
    if metrics.get("video_process_reason") == "no decodable frames":
        return "scorer decoded no video frames"
    for name in ("video_frame_count", "video_candidate_frame_count"):
        if name in metrics and metrics[name] == 0:
            return f"{name}=0: scorer decoded no video frames"
    if "source_replay" in metrics.get("hard_gate_failures", []):
        return str(metrics.get("failure_reason") or "independent source replay failed")
    contract = metrics.get("video_contract")
    if isinstance(contract, dict) and contract.get("error"):
        return f"video_contract: {contract['error']}"
    # Some task routes scope decode/replay diagnostics under artifact sections.
    for value in metrics.values():
        if isinstance(value, dict):
            error = _scoring_failure(value)
            if error:
                return error
    return None


def _semantic_rejection(error: str, metrics: dict[str, Any]) -> bool:
    # These routes also put ordinary candidate failures in EvalRecord.error.
    # A loaded, valid video with illegal marks/process is a negative example.
    if error == "formal delivery failed source-replay verification":
        return metrics.get("whole_video_loaded") is True
    failures = set(metrics.get("hard_gate_failures", []))
    return bool(
        metrics.get("verifier_profile")
        == "constraint_grid_progressive_commit_full_video_v1"
        and metrics.get("video_candidate_frame_count", 0) > 0
        and error == metrics.get("failure_reason")
        and failures
        and failures
        <= {"semantic_final_state", "temporal_process", "fixed_content_preservation"}
    )


class VerifierReward:
    """The scalar reward is the product of the three canonical hard gates.

    Incorrect rule execution returns the verifier's ordinary zero components.
    Source replay, media, or evaluator errors abort instead of injecting a
    fabricated zero into a rollout group.
    """

    def score(
        self,
        sample: Sample,
        video: np.ndarray | DecodedVideo | str | Path,
        mapping: VideoMapping | None = None,
    ) -> RewardResult:
        canonical = sample.canonical()
        if isinstance(video, (str, Path)):
            video = load_video(video)
        if mapping is not None:
            restored = restore_video(video, mapping)
        elif isinstance(video, DecodedVideo):
            restored = video
        else:
            restored = DecodedVideo(video, float(canonical.video_fps))
        canvas = load_image(sample.input_image).shape[:2]
        if (
            restored.frames.shape[1:3] != canvas
            or len(restored.frames) != canonical.video_num_frames
            or abs(restored.fps - float(canonical.video_fps)) > 0.01
        ):
            raise ValueError(
                f"{sample.id}: reward video must be restored to the original canvas, fps, and frame count"
            )

        spec = TASK_SPECS[sample.task_name]
        with tempfile.TemporaryDirectory(prefix="videogym-training-reward-") as temporary:
            root = Path(temporary)
            candidate = (
                root / canonical.split / canonical.id / "prediction.webm"
            )
            encode_rgb_frames_to_webm(
                list(restored.frames), candidate, fps=restored.fps
            )
            records = spec.verifier([canonical], root)
        if len(records) != 1 or records[0].sample_id != sample.id:
            raise RewardEvaluationError(
                "task evaluate must return exactly one result for the supplied sample"
            )
        record = records[0]
        metrics = dict(record.offline_metrics)
        failure = _scoring_failure(metrics)
        if failure or (record.error and not _semantic_rejection(record.error, metrics)):
            raise RewardEvaluationError(
                f"{sample.id}: verifier failed: {failure or record.error}"
            )
        raw = metrics.get("raw_reward_vector")
        if not isinstance(raw, dict):
            raise RewardEvaluationError(
                f"{sample.id}: verifier omitted canonical raw_reward_vector"
            )
        components = {}
        for name in RAW_REWARD_DIMENSION_NAMES:
            value = raw.get(name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise RewardEvaluationError(
                    f"{sample.id}: invalid or missing reward component {name}"
                )
            components[name] = float(value)
        if components["media_contract"] != 1:
            raise RewardEvaluationError(
                f"{sample.id}: verifier rejected the restored, encoded media contract"
            )
        if mapping is not None:
            metrics["training_video_mapping"] = mapping.to_dict()
        return RewardResult(
            sample.id, math.prod(components.values()), components, metrics
        )
