"""Source-replay verifier for ICL tasks with real option cards.

The option label is recomputed from the canonical source state.  The direct
terminal is then re-rendered from those same visible semantics; neither the
saved label nor any reference terminal/video is consulted.  TI2V accepts only
a task-local fade of the complete terminal footprint from the problem scene
and explicitly rejects spatial scans, static/hard-commit videos, and changes
outside the source-derived dynamic region.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Optional, Sequence

from data.common.verifiers.keyframes import keyframe_indices, visual_delta
from data.common.verifiers.transitions import uniform_fade_metrics

import cv2
import numpy as np
from PIL import Image

from data.common.schemas import CanonicalSample, EvalRecord
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.common.verifiers.region_gates import (
    STATIC_REGION_FLOOR,
    static_region_errors,
)
from data.in_context_learning.core.direct_delivery import DIRECT_CHOICE_TASKS, _odd_collection, _task_direct_pair
from data.in_context_learning.core.option_delivery import option_panel_boxes


_MODALITIES = frozenset({"ti2v"})
_VIDEO_NAMES = ("generated.webm", "output.webm", "prediction.webm", "result.webm")


@dataclass(frozen=True)
class _Replay:
    expected_label: str
    source: np.ndarray
    terminal: np.ndarray
    diagnostics: Mapping[str, Any]


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


def _labels(count: int) -> list[str]:
    if count != len(FOUR_CHOICE_CODEC.labels):
        raise ValueError("direct-choice source must expose exactly four candidates")
    return list(FOUR_CHOICE_CODEC.labels)


def _unique_match(values: Sequence[bool], *, task: str) -> int:
    indices = [index for index, value in enumerate(values) if bool(value)]
    if len(indices) != 1:
        raise ValueError(f"{task} source replay did not identify one unique candidate")
    return indices[0]


def _as_rgb(pair: tuple[Image.Image, Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    return tuple(np.asarray(image.convert("RGB"), dtype=np.uint8) for image in pair)  # type: ignore[return-value]


def _replay_analogy(state: dict[str, Any]) -> tuple[int, dict[str, Any], Image.Image, Image.Image]:
    from data.in_context_learning.analogy_completion.generator import (
        AnalogyCompletionPattern,
        _TRANSFORMS,
        _apply,
        _normalise,
    )

    shape_a = list(state.get("shape_a") or [])
    shape_b = list(state.get("shape_b") or [])
    shape_c = list(state.get("shape_c") or [])
    candidates = list(state.get("candidates") or [])
    transforms = [name for name in _TRANSFORMS if _normalise(_apply(shape_a, name)) == _normalise(shape_b)]
    if len(transforms) != 1:
        raise ValueError("analogy source does not identify one rigid transform")
    transform = transforms[0]
    answer_cells = [list(cell) for cell in _normalise(_apply(shape_c, transform))]
    candidate_cells = [list(item.get("cells") or []) for item in candidates]
    index = _unique_match(
        [_normalise(cells) == _normalise(answer_cells) for cells in candidate_cells],
        task="analogy_completion",
    )
    task = {
        "shape_a": shape_a,
        "shape_b": shape_b,
        "shape_c": shape_c,
        "color_a": list(state.get("shape_a_rgb") or [47, 92, 168]),
        "color_c": list(state.get("shape_c_rgb") or [224, 120, 60]),
        "transform": transform,
        "answer": answer_cells,
        "distractors": [cells for position, cells in enumerate(candidate_cells) if position != index],
    }
    renderer = object.__new__(AnalogyCompletionPattern)
    source_state = renderer._state_trace()[0]
    final_state = dict(renderer._clean_solution_state(), select=False)
    return (
        index,
        task,
        renderer._render_scene(task, candidate_cells, _labels(4)[index], source_state),
        renderer._render_scene(task, candidate_cells, _labels(4)[index], final_state),
    )


def _replay_object(state: dict[str, Any]) -> tuple[int, dict[str, Any], Image.Image, Image.Image]:
    from data.in_context_learning.object_subtraction.generator import (
        ObjectSubtractionPattern,
        _largest_index,
        _without_slot,
    )

    examples: list[dict[str, Any]] = []
    for raw in list(state.get("examples") or []):
        before, after = list(raw.get("before") or []), list(raw.get("after") or [])
        removed = sorted(set(int(item["slot"]) for item in before) - set(int(item["slot"]) for item in after))
        if len(removed) != 1 or _largest_index(before) != removed[0]:
            raise ValueError("object-subtraction demonstration does not replay")
        examples.append({"before": before, "after": after, "removed_slot": removed[0]})
    target_before = list(state.get("target_before") or [])
    target_removed = _largest_index(target_before)
    target_after = _without_slot(target_before, target_removed)
    candidates = list(state.get("candidates") or [])
    canonical = lambda items: sorted((int(item["slot"]), str(item["shape"]), tuple(item["color"]), int(item["size"])) for item in items)
    index = _unique_match([canonical(items) == canonical(target_after) for items in candidates], task="object_subtraction")
    task = {
        "examples": examples,
        "target_before": target_before,
        "target_after": target_after,
        "target_removed_slot": target_removed,
        "candidates": candidates,
        "answer_index": index,
    }
    renderer = object.__new__(ObjectSubtractionPattern)
    return index, task, renderer._render_state(task), renderer._render_state(task, solved=True)


def _replay_odd(state: dict[str, Any]) -> tuple[int, dict[str, Any], Image.Image, Image.Image]:
    from data.in_context_learning.odd_one_out_shape.generator import replay_odd_one_out_semantics

    replay = replay_odd_one_out_semantics(state)
    if not replay.get("rule_replay_valid"):
        raise ValueError(str(replay.get("failure_reason") or "odd-one-out source replay failed"))
    index = int(replay["odd_index"])
    return index, state, _odd_collection(state), _odd_collection(state, selected_index=index)


def _replay_set_inclusion(state: dict[str, Any]) -> tuple[int, dict[str, Any], Image.Image, Image.Image]:
    from data.in_context_learning.set_inclusion_diagram.generator import (
        _DIAGRAM_KINDS,
        SetInclusionDiagramPattern,
        _REL_TO_DIAGRAM,
        _classify_relation,
    )

    meta = state.get("task_meta")
    if not isinstance(meta, dict):
        raise ValueError("set-inclusion source metadata is missing")
    x_values = [int(value) for value in meta.get("x_elements") or []]
    y_values = [int(value) for value in meta.get("y_elements") or []]
    candidate_kinds = [str(value) for value in meta.get("candidate_kinds") or []]
    relation = _classify_relation(x_values, y_values)
    correct_kind = _REL_TO_DIAGRAM[relation]
    index = _unique_match([kind == correct_kind for kind in candidate_kinds], task="set_inclusion_diagram")
    # Render a deterministic authoring shell for the visible membership rows;
    # the direct terminal is reconstructed from X/Y alone by _task_direct_pair.
    render_kinds = list(_DIAGRAM_KINDS)
    task = {
        "x_elements": x_values,
        "y_elements": y_values,
        "shared_elements": sorted(set(x_values) & set(y_values)),
        "x_only": sorted(set(x_values) - set(y_values)),
        "y_only": sorted(set(y_values) - set(x_values)),
        "candidate_kinds": render_kinds,
        "answer_index": render_kinds.index(correct_kind),
    }
    renderer = object.__new__(SetInclusionDiagramPattern)
    puzzle = renderer._render_state(task)
    return index, task, puzzle, puzzle.copy()


def _replay_venn_count(state: dict[str, Any]) -> tuple[int, dict[str, Any], Image.Image, Image.Image]:
    meta = state.get("task_meta")
    if not isinstance(meta, dict):
        raise ValueError("Venn-count source metadata is missing")
    mapping = meta.get("candidate_values_by_label")
    positions = meta.get("dot_positions")
    if not isinstance(mapping, dict) or not isinstance(positions, dict):
        raise ValueError("Venn-count source choices or dots are missing")
    count = len(list(positions.get("ab_both") or []))
    labels = _labels(4)
    index = _unique_match([int(mapping[label]) == count for label in labels], task="venn_intersection_count")
    blank = Image.new("RGB", (512, 512), (255, 255, 255))
    problem, terminal = _task_direct_pair(
        blank,
        blank,
        option_panel_boxes(state),
        state,
        task="venn_intersection_count",
        selected_index=index,
    )
    return index, state, problem, terminal


def _replay_symdiff(state: dict[str, Any]) -> tuple[int, dict[str, Any], Image.Image, Image.Image]:
    meta = state.get("task_meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("candidate_masks_by_label"), dict):
        raise ValueError("symmetric-difference source candidates are missing")
    mapping = meta["candidate_masks_by_label"]
    labels = _labels(4)
    target = [0, 1, 0, 1]
    index = _unique_match([list(mapping[label]) == target for label in labels], task="venn_symmetric_difference")
    blank = Image.new("RGB", (512, 512), (255, 255, 255))
    problem, terminal = _task_direct_pair(
        blank,
        blank,
        option_panel_boxes(state),
        state,
        task="venn_symmetric_difference",
        selected_index=index,
    )
    return index, state, problem, terminal


def _render_replay(task_name: str, state: dict[str, Any]) -> _Replay:
    branches = {
        "analogy_completion": _replay_analogy,
        "object_subtraction": _replay_object,
        "odd_one_out_shape": _replay_odd,
        "set_inclusion_diagram": _replay_set_inclusion,
        "venn_intersection_count": _replay_venn_count,
        "venn_symmetric_difference": _replay_symdiff,
    }
    if task_name not in branches:
        raise ValueError(f"unsupported ICL direct-choice task: {task_name}")
    index, _, puzzle, solution = branches[task_name](state)
    if task_name not in {"odd_one_out_shape", "venn_intersection_count", "venn_symmetric_difference"}:
        boxes = option_panel_boxes(state)
        problem, terminal = _task_direct_pair(
            puzzle,
            solution,
            boxes,
            dict(state),
            task=task_name,
            selected_index=index,
        )
    else:
        problem, terminal = puzzle, solution
    source, target = _as_rgb((problem, terminal))
    changed = np.max(np.abs(target.astype(np.int16) - source.astype(np.int16)), axis=2) > 8
    if int(np.count_nonzero(changed)) < 24:
        raise ValueError("source replay produced no load-bearing direct terminal change")
    return _Replay(
        expected_label=_labels(4)[index],
        source=source,
        terminal=target,
        diagnostics={"source_replay_changed_pixels": int(np.count_nonzero(changed))},
    )


def _dynamic_mask(source: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    changed = np.max(np.abs(terminal.astype(np.int16) - source.astype(np.int16)), axis=2) > 8
    return cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)


def _terminal_scores(candidate: np.ndarray, replay: _Replay) -> dict[str, float]:
    if candidate.shape != replay.terminal.shape:
        return {"rmse": 1.0, "mismatch": 1.0, "progress": -1.0, "static": 1.0}
    dynamic = _dynamic_mask(replay.source, replay.terminal)
    error = candidate.astype(np.float32) - replay.terminal.astype(np.float32)
    rmse = float(np.sqrt(np.mean(error[dynamic] ** 2)) / 255.0)
    mismatch = float(np.mean(np.max(np.abs(error), axis=2)[dynamic] > 56.0))
    baseline = replay.source.astype(np.float32) - replay.terminal.astype(np.float32)
    baseline_rmse = float(np.sqrt(np.mean(baseline[dynamic] ** 2)) / 255.0)
    progress = float((baseline_rmse - rmse) / baseline_rmse) if baseline_rmse > 1e-9 else 0.0
    static = ~dynamic
    static_error = np.max(np.abs(candidate.astype(np.int16) - replay.source.astype(np.int16)), axis=2)
    mutation = float(np.mean(static_error[static] > 38.0)) if np.any(static) else 0.0
    return {"rmse": rmse, "mismatch": mismatch, "progress": progress, "static": mutation}


def _text_metrics(sample: CanonicalSample, line: str) -> tuple[dict[str, Any], bool]:
    started = perf_counter()
    replay = _render_replay(sample.task_name, _semantic_state(sample))
    predicted = FOUR_CHOICE_CODEC.parse(line)
    parse_ok = predicted is not None
    semantic_ok = predicted == replay.expected_label
    passed = bool(parse_ok and semantic_ok)
    failures = [] if passed else ["text_parse" if not parse_ok else "text_semantic"]
    return {
        "text_parse_ok": parse_ok,
        "text_parse_score": float(parse_ok),
        # VerifierRoute's reward normalizer intentionally does not infer text
        # semantics from a generic nested ``answer_score``.  Expose the
        "artifact_results": {"text": bool(passed)},
        "text_predicted_label": predicted,
        "text_source_replay_expected_label": replay.expected_label,
        **replay.diagnostics,
        **_contract_metrics(
            profile="icl_direct_choice_source_replay_text_v1",
            artifact_kind="typed_text",
            scores={"format_score": float(parse_ok), "answer_score": float(semantic_ok)},
            passed=passed,
            reason="typed label disagrees with visible-source replay",
            failures=failures,
            started_at=started,
        ),
    }, passed


def _image_metrics(sample: CanonicalSample, path: Optional[Path]) -> tuple[dict[str, Any], bool]:
    started = perf_counter()
    profile = "icl_direct_choice_source_replay_terminal_v1"
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
        scores = _terminal_scores(candidate, replay)
        final_ok = scores["rmse"] <= 0.11 and scores["mismatch"] <= 0.24 and scores["progress"] >= 0.68
        preservation_ok = scores["static"] <= 0.012
        passed = bool(media_ok and final_ok and preservation_ok)
        failures: list[str] = []
        if not media_ok:
            failures.append("media_contract")
        if not final_ok:
            failures.append("direct_terminal_semantic")
        if not preservation_ok:
            failures.append("source_preservation")
        return {
            "image_size": list(size),
            "image_dynamic_rmse": scores["rmse"],
            "image_dynamic_mismatch_fraction": scores["mismatch"],
            "image_terminal_progress": scores["progress"],
            "image_static_mutation_fraction": scores["static"],
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
        source = replay.source
        media_ok = bool(all(f.shape == source.shape for f in frames) and abs(fps - 16) <= .25
                        and 5 <= len(frames) <= 157 and len(frames) % 4 == 1)
        if not media_ok:
            raise ValueError("invalid video media contract")
        indices = keyframe_indices(frames)
        static = ~_dynamic_mask(source, replay.terminal)
        static_error = max(static_region_errors(frame, source, static)[0] for frame in frames)
        first_error = float(visual_delta(frames[0], source).mean())
        preserved = static_error <= STATIC_REGION_FLOOR and first_error <= 10
        final_scores = _terminal_scores(frames[-1], replay)
        final_ok = final_scores["rmse"] <= .12 and final_scores["progress"] >= .66
        final_diag = {"video_final_scores": final_scores}
        # The public instruction requires one whole-footprint fade.
        direction = replay.terminal.astype(float) - source.astype(float)
        energy = max(1., float(np.square(direction).sum()))
        progress = [float(np.sum((frames[i].astype(float) - source) * direction) / energy) for i in indices]
        committed = next((i for i, value in enumerate(progress) if value >= .80), None)
        retained = committed is not None and all(value >= .55 for value in progress[committed:])
        fade_metrics, fade_ok = uniform_fade_metrics(
            frames, source, replay.terminal, erasure=sample.task_name == "object_subtraction"
        )
        process = bool(preserved and retained and fade_ok)
        passed = bool(process and final_ok)
        failures = ([] if preserved else ["source_preservation"]) + ([] if retained else ["answer_commit_erased_or_missing"]) + ([] if final_ok else ["video_final_state"])
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


def verify_icl_direct_choice_delivery(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: str = "ti2v",
) -> list[EvalRecord]:
    """Verify only the modality's formal final delivery."""

    selected = str(modality or "").strip().lower()
    if selected not in _MODALITIES:
        raise ValueError(f"unsupported ICL direct-choice modality: {modality!r}")
    records: list[EvalRecord] = []
    for sample in samples:
        record = _eval_record(sample)
        try:
            if sample.task_name not in DIRECT_CHOICE_TASKS:
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


__all__ = ["verify_icl_direct_choice_delivery"]
