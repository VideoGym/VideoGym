"""Native video records for evaluation-only tasks, using the canonical media contract."""

import numpy as np
from data.common.schemas import EvalRecord
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.verifiers.keyframes import visual_delta
from data.common.video_encoding import (
    encode_rgb_frames_to_webm,
    validate_video_contract,
)


def write_record(generator, source, states, render, answer, prompt):
    from evaluation.catalog.registry import EVAL_TASK_SPECS

    spec = EVAL_TASK_SPECS[generator.task_name]
    image = render(source, states[0])
    target = render(source, states[-1])
    image_path = generator.output_dir / "input.png"
    target_path = generator.output_dir / "terminal.png"
    image.save(image_path)
    target.save(target_path)
    video_path = generator.output_dir / "reference.webm"
    frames = [np.asarray(image).copy() for _ in range(5)]
    for state in states[1:]:
        frames.extend(np.asarray(render(source, state)).copy() for _ in range(4))
    frames.extend(np.asarray(target).copy() for _ in range(8))
    while len(frames) % 4 != 1:
        frames.append(frames[-1].copy())
    if len(frames) > 157:
        raise ValueError("evaluation trajectory exceeds video capacity")
    video = bool(generator.video)
    if video and not encode_rgb_frames_to_webm(frames, video_path, fps=16):
        raise RuntimeError("evaluation reference encoding failed")
    from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

    prompt += "\n\n" + build_ti2v_continuity_paragraph(
        timing="discrete",
        moving_nouns="the agent, answer objects and selected marks",
        static_nouns="all unchanged scene content, labels, background and camera",
        hold_noun="the solved final state",
        allow_rule_teleports=source.get("allow_rule_teleports", False),
    )
    return {
        "id": generator.task_name + "-source",
        "task_name": generator.task_name,
        "task_family": spec.task_family,
        "split": "eval",
        "ti2v_prompt": prompt,
        "answer": answer,
        "structured_answer": source,
        "image": str(image_path),
        "solution_image_path": str(target_path),
        "solution_video_path": str(video_path) if video else None,
        "video_fps": 16 if video else None,
        "video_num_frames": len(frames) if video else None,
        "rendered_glyph_inventory": {"fixed_labels": []},
        "rendered_glyph_schema": {"fixed_labels": "task-native labels only"},
    }


def read_video(path):
    import cv2

    validate_video_contract(path)
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def matches(frame, expected):
    delta = visual_delta(frame, expected)
    return all(
        float(delta[y : y + 32, x : x + 32].mean()) <= 12
        for y in range(0, 512, 32)
        for x in range(0, 512, 32)
    )


def eval_record(sample, pred_root, grader):
    record = EvalRecord(
        sample_id=sample.id,
        task_name=sample.task_name,
        task_family=sample.task_family,
        input_asset=sample.primary_input_image,
        split=sample.split,
    )
    path = resolve_candidate_video(pred_root, sample.id, sample.split)
    try:
        if path is None:
            raise FileNotFoundError("Missing complete video")
        frames = read_video(path)
        source = sample.semantic_input.get("semantic_state", sample.semantic_input)
        process, final, diagnostics = grader(source, frames)
        record.prediction_asset = str(path)
        record.offline_pass = bool(process and final)
        record.offline_metrics = {
            **diagnostics,
            "artifact_results": {"video": bool(process and final)},
            "whole_video_process_correct": float(process),
            "video_final_state_correct": float(final),
            "whole_video_loaded": True,
            "ground_truth_used": False,
            "reference_artifact_used": False,
        }
    except Exception as error:
        record.offline_pass = False
        record.error = str(error)
        record.offline_metrics.update(
            whole_video_loaded=False, whole_video_error=str(error)
        )
    return record
