"""Source-replay verifier for ICL per-position-card sequence completion. Every converged sequence task delivers the completed final card directly:
The expected payload and the
expected scene pixels are recomputed from the canonical semantic state through
the task's own renderer; no saved answer field, reference image, or reference
video is consulted. The final card content is decoded semantically
(counts, cells, colors, heights, orientations) rather than compared as raw
pixels, and everything outside the final card must stay preserved."""

from __future__ import annotations

from data.common.verifiers.keyframes import keyframe_indices, visual_delta
from data.common.verifiers.transitions import uniform_fade_metrics

import importlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from data.common.schemas import CanonicalSample, EvalRecord
from data.common.typed_answers import parse_answer_line
from data.common.verifiers.region_gates import (
    STATIC_REGION_FLOOR,
    static_region_errors,
)

_MODALITIES = frozenset({"ti2v"})
_VIDEO_NAMES = ("generated.webm", "output.webm", "prediction.webm", "result.webm")

SEQUENCE_COMPLETION_TASKS = frozenset(
    {
        "alternating_size_sequence",
        "dot_count_progression",
        "growing_polygon_step",
        "increasing_size_sequence",
        "linear_sequence_next",
        "sequence_completion_poly",
        "tile_color_sequence",
    }
)


@dataclass
class SequenceReplay:
    """One task's independently recomputed expectations for a sample."""

    expected_payload: str
    payload_pattern: str
    problem: np.ndarray
    terminal: np.ndarray
    decode_final_card: Callable[[np.ndarray], tuple[Optional[str], dict]]
    payload_equal: Optional[Callable[[str, str], bool]] = None
    diagnostics: dict = field(default_factory=dict)

    def equal(self, candidate: Optional[str]) -> bool:
        if candidate is None:
            return False
        if self.payload_equal is not None:
            return bool(self.payload_equal(candidate, self.expected_payload))
        return candidate.strip() == self.expected_payload.strip()


def make_template_decoder(
    templates: Mapping[str, np.ndarray],
    box: Sequence[float],
    *,
    max_best_rmse: float = 0.10,
    min_margin: float = 0.010,
) -> Callable[[np.ndarray], tuple[Optional[str], dict]]:
    """Semantic decode of one card region by nearest source-rendered template.

    ``templates`` maps every plausible payload to a full replayed scene whose
    final card holds that payload's content; the decoder compares only the
    final card's content region.  The decoded payload must be the strict
    nearest template within an absolute error budget and with a real margin
    over the runner-up, so between-payload ambiguity fails closed.
    """

    left, top, right, bottom = (int(round(float(value))) for value in box)

    def _crop(image: np.ndarray) -> np.ndarray:
        return image[top:bottom, left:right].astype(np.float32)

    crops = {payload: _crop(np.asarray(image)) for payload, image in templates.items()}

    def _decode(candidate: np.ndarray) -> tuple[Optional[str], dict]:
        region = _crop(candidate)
        errors: dict[str, float] = {}
        for payload, crop in crops.items():
            if crop.shape != region.shape:
                return None, {"template_shape_mismatch": True}
            errors[payload] = float(np.sqrt(np.mean((region - crop) ** 2)) / 255.0)
        ranked = sorted(errors.items(), key=lambda item: item[1])
        best_payload, best_error = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 1.0
        decoded = (
            best_payload
            if best_error <= max_best_rmse and (runner_up - best_error) >= min_margin
            else None
        )
        return decoded, {
            "template_errors": {key: round(value, 4) for key, value in errors.items()},
            "template_best": best_payload,
            "template_best_rmse": round(best_error, 4),
            "template_margin": round(runner_up - best_error, 4),
        }

    return _decode


def _contract_metrics(
    *,
    profile: str,
    artifact_kind: str,
    scores: Mapping[str, float],
    passed: bool,
    reason: str,
    failures: Sequence[str],
    started_at: float,
) -> dict[str, Any]:
    latency_ms = max(0.0, (perf_counter() - started_at) * 1000.0)
    return {
        "verifier_profile": profile,
        "artifact_kind": artifact_kind,
        "component_scores": {
            str(name): max(0.0, min(1.0, float(value))) for name, value in scores.items()
        },
        "hard_gate_failures": [str(value) for value in failures],
        "binary_correct": bool(passed),
        "failure_reason": "ok" if passed else str(reason),
        "latency_ms": latency_ms,
        "performance_diagnostics": {
            "latency_ms": latency_ms,
            "automatic_failure_thresholds": False,
        },
    }


def _semantic_state(sample: CanonicalSample) -> dict[str, Any]:
    raw = sample.semantic_input if isinstance(sample.semantic_input, dict) else {}
    state = raw.get("semantic_state")
    if not isinstance(state, dict):
        raise ValueError("canonical ICL sample has no source semantic_state")
    return dict(state)


def _render_replay(task_name: str, state: dict[str, Any]) -> SequenceReplay:
    if task_name not in SEQUENCE_COMPLETION_TASKS:
        raise ValueError(f"unsupported ICL sequence-completion task: {task_name}")
    module = importlib.import_module(f"data.in_context_learning.{task_name}.generator")
    replay = module.build_sequence_replay(state)
    if not isinstance(replay, SequenceReplay):
        raise TypeError(f"{task_name} build_sequence_replay returned {type(replay)!r}")
    changed = np.max(
        np.abs(replay.terminal.astype(np.int16) - replay.problem.astype(np.int16)), axis=2
    ) > 8
    if int(np.count_nonzero(changed)) < 24:
        raise ValueError("source replay produced no load-bearing final-card change")
    return replay


def _dynamic_mask(source: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    changed = np.max(np.abs(terminal.astype(np.int16) - source.astype(np.int16)), axis=2) > 8
    return cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)


def _static_mutation_fraction(candidate: np.ndarray, replay: SequenceReplay) -> float:
    if candidate.shape != replay.problem.shape:
        return 1.0
    static = ~_dynamic_mask(replay.problem, replay.terminal)
    if not np.any(static):
        return 0.0
    delta = np.max(np.abs(candidate.astype(np.int16) - replay.problem.astype(np.int16)), axis=2)
    return float(np.mean(delta[static] > 38))


def _text_metrics(sample: CanonicalSample, line: str) -> tuple[dict[str, Any], bool]:
    started = perf_counter()
    replay = _render_replay(sample.task_name, _semantic_state(sample))
    payload = parse_answer_line(line)
    parse_ok = bool(
        payload is not None
        and re.fullmatch(replay.payload_pattern, payload) is not None
    )
    predicted = payload if parse_ok else None
    semantic_ok = replay.equal(predicted)
    passed = bool(parse_ok and semantic_ok)
    failures = [] if passed else ["text_parse" if not parse_ok else "text_semantic"]
    return {
        "text_parse_ok": parse_ok,
        "text_parse_score": float(parse_ok),
        "artifact_results": {"text": bool(passed)},
        "text_predicted_payload": predicted,
        "text_source_replay_expected_payload": replay.expected_payload,
        **replay.diagnostics,
        **_contract_metrics(
            profile="icl_sequence_completion_source_replay_text_v1",
            artifact_kind="typed_text",
            scores={"format_score": float(parse_ok), "answer_score": float(semantic_ok)},
            passed=passed,
            reason="typed payload disagrees with visible-sequence replay",
            failures=failures,
            started_at=started,
        ),
    }, passed


def _image_metrics(sample: CanonicalSample, path: Optional[Path]) -> tuple[dict[str, Any], bool]:
    started = perf_counter()
    profile = "icl_sequence_completion_source_replay_terminal_v1"
    if path is None:
        return _contract_metrics(
            profile=profile,
            artifact_kind="terminal_image",
            scores={"media_validity_score": 0.0, "final_state_score": 0.0, "preservation_score": 0.0},
            passed=False,
            reason="candidate terminal image is missing",
            failures=("missing_image",),
            started_at=started,
        ), False
    try:
        replay = _render_replay(sample.task_name, _semantic_state(sample))
        with Image.open(path) as opened:
            size = opened.size
            candidate = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        media_ok = size == (512, 512)
        decoded, decode_diag = (None, {})
        if media_ok:
            decoded, decode_diag = replay.decode_final_card(candidate)
        final_ok = media_ok and replay.equal(decoded)
        mutation = _static_mutation_fraction(candidate, replay) if media_ok else 1.0
        preservation_ok = mutation <= 0.012
        passed = bool(media_ok and final_ok and preservation_ok)
        failures: list[str] = []
        if not media_ok:
            failures.append("media_contract")
        if not final_ok:
            failures.append("final_card_semantic")
        if not preservation_ok:
            failures.append("source_preservation")
        return {
            "image_size": list(size),
            "image_decoded_payload": decoded,
            "image_expected_payload": replay.expected_payload,
            "image_decode_diagnostics": decode_diag,
            "image_static_mutation_fraction": mutation,
            "reference_artifact_used": False,
            **replay.diagnostics,
            **_contract_metrics(
                profile=profile,
                artifact_kind="terminal_image",
                scores={
                    "media_validity_score": float(media_ok),
                    "final_state_score": float(final_ok),
                    "preservation_score": float(preservation_ok),
                },
                passed=passed,
                reason="; ".join(failures),
                failures=failures,
                started_at=started,
            ),
        }, passed
    except Exception as exc:
        return _contract_metrics(
            profile=profile,
            artifact_kind="terminal_image",
            scores={"media_validity_score": 0.0, "final_state_score": 0.0, "preservation_score": 0.0},
            passed=False,
            reason=str(exc),
            failures=("image_decode_or_source_replay",),
            started_at=started,
        ), False


def _resolve_video(pred_root: Path, sample: CanonicalSample) -> Optional[Path]:
    directories = []
    if sample.split:
        directories.append(pred_root / sample.split / sample.id)
    directories.append(pred_root / sample.id)
    candidates: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        candidates.extend(directory / name for name in _VIDEO_NAMES)
        candidates.extend(directory.glob("*.webm"))
        candidates.extend(directory.glob("*.mov"))
    candidates.extend(pred_root / f"{sample.id}{suffix}" for suffix in (".webm", ".mov"))
    return next((path for path in candidates if path.is_file()), None)


def _decode_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise ValueError("candidate video cannot be decoded")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError("candidate video contains no frames")
    return frames, fps




def _video_metrics(sample: CanonicalSample, path: Optional[Path]) -> tuple[dict[str, Any], bool]:
    started = perf_counter()
    profile = "icl_answer_keyframes_v1"
    try:
        if path is None:
            raise ValueError("candidate video is missing")
        replay = _render_replay(sample.task_name, _semantic_state(sample))
        frames, fps = _decode_video(path)
        source = replay.problem
        media_ok = bool(all(f.shape == source.shape for f in frames) and abs(fps - 16) <= .25
                        and 5 <= len(frames) <= 157 and len(frames) % 4 == 1)
        if not media_ok:
            raise ValueError("invalid video media contract")
        indices = keyframe_indices(frames)
        static = ~_dynamic_mask(source, replay.terminal)
        static_error = max(static_region_errors(frame, source, static)[0] for frame in frames)
        first_error = float(visual_delta(frames[0], source).mean())
        preserved = static_error <= STATIC_REGION_FLOOR and first_error <= 10
        decoded, decode_diag = replay.decode_final_card(frames[-1])
        final_ok = replay.equal(decoded)
        final_diag = {"video_final_decoded_payload": decoded, "video_final_decode_diagnostics": decode_diag}
        # The public instruction requires one whole-footprint fade.
        direction = replay.terminal.astype(float) - source.astype(float)
        energy = max(1., float(np.square(direction).sum()))
        progress = [float(np.sum((frames[i].astype(float) - source) * direction) / energy) for i in indices]
        fade_metrics, fade_ok = uniform_fade_metrics(frames, source, replay.terminal)
        # The fade gate registers the independently decoded final observation
        # and checks its complete footprint, monotonicity and endpoint holds.
        # A second projection onto the ideal raster would reject a constant
        # harmless displacement of an otherwise correctly decoded symbol.
        process = bool(preserved and fade_ok)
        passed = bool(process and final_ok)
        failures = ([] if preserved else ["source_preservation"]) + ([] if final_ok else ["video_final_state"])
        if not fade_ok:
            failures.append("complete_footprint_fade")
        return {
            "video_frame_count": len(frames), "video_fps": fps, "video_keyframe_indices": indices,
            "video_keyframe_answer_progress": progress, "video_static_region_max_error": static_error,
            "video_initial_source_mae": first_error, "reference_artifact_used": False,
            **fade_metrics,
            **final_diag,
            **_contract_metrics(profile=profile, artifact_kind="full_video",
                                scores={"media_validity_score": 1., "temporal_process_score": float(process),
                                        "preservation_score": float(preserved), "final_state_score": float(final_ok)},
                                passed=passed, reason="; ".join(failures), failures=failures, started_at=started),
        }, passed
    except Exception as exc:
        return _contract_metrics(profile=profile, artifact_kind="full_video",
                                 scores={"media_validity_score": 0., "temporal_process_score": 0., "final_state_score": 0.},
                                 passed=False, reason=str(exc), failures=("video_decode_or_source_replay",), started_at=started), False


def _eval_record(sample: CanonicalSample) -> EvalRecord:
    return EvalRecord(
        sample_id=sample.id,
        task_name=sample.task_name,
        task_family=sample.task_family,
        input_asset=sample.primary_input_image,
        split=sample.split,
    )


def verify_icl_sequence_completion(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: str = "ti2v",
) -> list[EvalRecord]:
    """Verify only the modality's formal final delivery."""

    selected = str(modality or "").strip().lower()
    if selected not in _MODALITIES:
        raise ValueError(f"unsupported ICL sequence-completion modality: {modality!r}")
    records: list[EvalRecord] = []
    for sample in samples:
        record = _eval_record(sample)
        try:
            if sample.task_name not in SEQUENCE_COMPLETION_TASKS:
                raise ValueError(f"Unsupported task: {sample.task_name}")
            video_path = _resolve_video(pred_root, sample)
            metrics, passed = _video_metrics(sample, video_path)
            record.prediction_asset = str(video_path) if video_path is not None else None
            record.offline_metrics = {**metrics, "ground_truth_used": False,
                                      "reference_artifact_used": False, "candidate_cot_inspected": False}
            record.offline_pass = bool(passed)
            record.error = None if passed else str(metrics.get("failure_reason") or "video verification failed")
        except Exception as error:
            record.offline_pass = False
            record.error = str(error)
        records.append(record)
    return records


__all__ = [
    "SEQUENCE_COMPLETION_TASKS",
    "SequenceReplay",
    "make_template_decoder",
    "verify_icl_sequence_completion",
]
