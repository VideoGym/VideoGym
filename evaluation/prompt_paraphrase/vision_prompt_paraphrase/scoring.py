"""Verifier controls and candidate scoring for Vision Prompt Paraphrase."""

from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from data.common.io import matches_schema
from data.common.paths import to_absolute
from data.common.video_encoding import encode_rgb_frames_to_webm
from evaluation.execution.datasets.records import read_rows, write_json
from evaluation.execution.scoring.statistics import summarize
from evaluation.prompt_paraphrase.text_paraphrase.scoring import _write_results
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.dataset import _source_rows, load_visual
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.plans import (
    PROTOCOL,
    mapped_sample,
    sample_from_row,
)
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.prompts import prompt_paraphrase_prompt
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.rendering import (
    observation_chart,
    prompt_paraphrase_context,
    render_presented,
    render_reference,
)


def _score_case(payload):
    import cv2

    cv2.setNumThreads(1)
    row, pair, predictions, dataset_root, source_root = payload
    dataset_root, source_root = Path(dataset_root), Path(source_root)
    sample = sample_from_row(row, base_dir=source_root)
    mapped = mapped_sample(sample, pair["plan"])
    chart = observation_chart(sample, pair["plan"])
    charted = chart is not None
    mapped.image = to_absolute(pair["observation_image"], dataset_root).as_posix()
    mapped.prompt = prompt_paraphrase_prompt(sample, pair["plan"])
    from data.common.verifiers.candidates import resolve_candidate_video
    from data.registry import get_task_spec
    from evaluation.execution.scoring.statistics import DIMENSIONS

    candidate = resolve_candidate_video(Path(predictions), sample.id, sample.split)
    observation = nullcontext()
    if charted and candidate is not None:
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase.charts import observe_candidate

        observation = observe_candidate(candidate, chart)
    with prompt_paraphrase_context(mapped, pair["plan"]), observation:
        records = get_task_spec(sample.task_name).verifier([mapped], Path(predictions))
    if len(records) != 1 or records[0].sample_id != sample.id:
        raise RuntimeError("verifier did not return the requested sample")
    record = records[0]
    record.input_asset = to_absolute(pair["variant_image"], dataset_root).as_posix()
    vector = record.offline_metrics["raw_reward_vector"]
    available = {k: bool(record.offline_metrics.get("reward_availability_mask", {}).get(k, True)) for k in DIMENSIONS}
    record.prediction_asset = str(candidate.resolve()) if candidate else None
    result = {k: row[k] for k in ("evaluation_task", "difficulty_tier", "track", "role", "pair_id", "features", "scene_group")}
    result.update(
        sample_id=sample.id,
        task=sample.task_name,
        rewards={k: float(vector[k]) for k in DIMENSIONS},
        reward_available=available,
        success=int(all(available.values()) and all(vector[k] == 1 for k in DIMENSIONS)),
        verifier=record.to_dict(),
    )
    result["prompt_paraphrase_plan"] = pair["plan"]
    metrics = result["verifier"]["offline_metrics"]
    metrics.update(
        prompt_paraphrase_protocol=PROTOCOL,
        prompt_paraphrase_candidate_file_rewritten=False,
        prompt_paraphrase_observation_reindexed=charted,
        prompt_paraphrase_used_reference_pixels=False,
    )
    return result


def _calibrate_case(payload):
    import cv2

    cv2.setNumThreads(1)
    from data.common.verifiers.routes import VerifierRoute
    from evaluation.shared.task_io import read_video

    row, pair, dataset_root, source_root, *persistent_reference_roots = payload
    dataset_root, source_root = Path(dataset_root), Path(source_root)
    sample = sample_from_row(row, base_dir=source_root)
    variant_image_path = to_absolute(pair["variant_image"], dataset_root)
    observation_image_path = to_absolute(pair["observation_image"], dataset_root)
    frozen_reference_path = to_absolute(pair["reference"], dataset_root)
    with tempfile.TemporaryDirectory(prefix="videogym-visual-qualification-") as tmp:
        root = Path(tmp)
        image = render_presented(sample, pair["plan"], root / "replay")
        with Image.open(variant_image_path) as im:
            if not np.array_equal(np.array(image), np.array(im.convert("RGB"))):
                raise ValueError("variant image differs from its renderer")
        positive_reference = frozen_reference_path
        rebuilt = False
        if sample.task_name == "block_blast_clear_predict":
            positive_reference = render_reference(sample, pair["plan"], root / "single-piece-reference")
            rebuilt = True
            if persistent_reference_roots:
                reference_root = Path(persistent_reference_roots[0])
                reference_root.mkdir(parents=True, exist_ok=True)
                persistent = reference_root / f"{sample.id}.webm"
                shutil.copyfile(positive_reference, persistent)
                positive_reference = persistent
        VerifierRoute._link_candidate(positive_reference, root / "positive" / sample.id / "prediction.webm")
        positive = _score_case((row, pair, root / "positive", dataset_root, source_root))
        frames = read_video(positive_reference)
        input_pixels = np.array(image)
        negatives = {
            "unchanged_input": [input_pixels] * len(frames),
            "erased_scene": [frames[0]] + [np.full_like(frames[0], 255)] * (len(frames) - 1),
        }
        if pair["plan"]["profile"] == "geometry_workbench":
            from evaluation.prompt_paraphrase.vision_prompt_paraphrase.charts import GeometryChart

            chart = GeometryChart(sample, pair["plan"]["quarter_turns"])
            with Image.open(observation_image_path) as obs_image:
                original = np.array(obs_image.convert("RGB"))
            native = [chart.undo(frame) for frame in frames]

            def red_mask(frame):
                a = frame.astype(np.int16)
                return (a[..., 0] > a[..., 1] + 25) & (a[..., 0] > a[..., 2] + 25)

            yy, xx = np.where(red_mask(native[-1]))
            if not len(xx):
                raise ValueError("geometry control has no final selection mark")
            centre = np.array([xx.mean(), yy.mean()])
            candidates = sample.semantic_input["candidates"]
            wrong = max(candidates, key=lambda c: np.linalg.norm(np.array([c["x"], c["y"]]) - centre))
            shift = np.rint(np.array([wrong["x"], wrong["y"]]) - centre).astype(int)
            shifted = []
            for frame in native:
                bad = frame.copy()
                mask = red_mask(frame)
                y, x = np.where(mask)
                bad[mask] = original[mask]
                tx, ty = x + shift[0], y + shift[1]
                valid = (tx >= 0) & (tx < 512) & (ty >= 0) & (ty < 512)
                bad[ty[valid], tx[valid]] = frame[y[valid], x[valid]]
                shifted.append(chart.apply(bad))
            negatives["misplaced_selection"] = shifted
        if sample.task_name in {"maze_square", "maze_keys_and_locks"}:
            negatives["reversed_route_process"] = frames[:4] + frames[4:-4][::-1] + frames[-4:]
        rejected = {}
        if sample.task_name == "block_blast_clear_predict":
            from data.dynamics_game.core.source_replay import replay_source

            with prompt_paraphrase_context(sample, pair["plan"]):
                replay = replay_source(sample.task_name, sample.semantic_input["semantic_state"])
            left, top, right, bottom = replay.metrics["source_piece_card_box"]
            duplicated = [frame.copy() for frame in frames]
            for frame in duplicated[20:]:
                frame[top:bottom, left:right] = input_pixels[top:bottom, left:right]
            negatives["duplicated_supply_piece"] = duplicated
        for name, images in negatives.items():
            destination = root / name / sample.id / "prediction.webm"
            if not encode_rgb_frames_to_webm(images, destination, fps=sample.video_fps):
                raise RuntimeError("negative control encoding failed")
            rejected[name] = _score_case((row, pair, root / name, dataset_root, source_root))["success"] == 0
        return {
            "sample_id": sample.id,
            "evaluation_task": row["evaluation_task"],
            "positive_success": positive["success"],
            "negative_controls_rejected": rejected,
            "positive": positive,
            "rebuilt_reference": rebuilt,
            "positive_reference": str(positive_reference),
            "positive_reference_frames": len(frames),
            "positive_reference_fps": sample.video_fps,
            "frozen_reference": str(frozen_reference_path),
        }


def _reference_overrides(controls):
    return [
        {
            "sample_id": r["sample_id"],
            "reference": r["positive_reference"],
            "frames": r["positive_reference_frames"],
            "fps": r["positive_reference_fps"],
            "frozen_reference": r["frozen_reference"],
        }
        for r in controls
        if r.get("rebuilt_reference")
    ]


def calibrate(dataset, output, workers=1):
    dataset_root = Path(dataset).expanduser().resolve()
    metadata, rows, pairs = load_visual(dataset_root)
    source_root = to_absolute(metadata["dataset"], dataset_root)
    reference_root = str((Path(output) / "positive-references").resolve())
    controls = _write_results(
        output,
        "controls.jsonl",
        [(r, p, str(dataset_root), str(source_root), reference_root) for r, p in zip(rows, pairs)],
        _calibrate_case,
        workers,
    )
    overrides = _reference_overrides(controls)
    overrides_path = Path(output) / "reference-overrides.json"
    write_json(overrides_path, {"references": overrides})
    passed = all(r["positive_success"] == 1 and all(r["negative_controls_rejected"].values()) for r in controls)
    report = {
        "protocol": PROTOCOL,
        "dataset": str(dataset_root),
        "samples": len(rows),
        "complete": passed,
        "positive_passes": sum(r["positive_success"] for r in controls),
        "all_negative_rejections": sum(all(r["negative_controls_rejected"].values()) for r in controls),
        "reference_overrides": str(overrides_path.resolve()),
        "complete_reviewed_selection": metadata["complete_reviewed_selection"],
    }
    write_json(Path(output) / "qualification.json", report)
    if not passed:
        raise RuntimeError("visual verifier controls failed; inspect controls.jsonl")
    return report


def score(dataset, output, predictions, model, workers=1, shard_index=0, shards=1):
    from data.common.verifiers.candidates import resolve_candidate_video

    dataset_root = Path(dataset).expanduser().resolve()
    metadata, rows, pairs = load_visual(dataset_root)
    source_root = to_absolute(metadata["dataset"], dataset_root)
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("invalid scoring shard")
    selected = rows[shard_index::shards]
    selected_pairs = pairs[shard_index::shards]
    if not selected:
        raise ValueError("empty scoring shard")
    predictions = Path(predictions).resolve()
    for row in selected:
        sample = row["sample"]
        if not resolve_candidate_video(predictions, sample["id"], sample["split"]):
            raise ValueError(f"missing candidate: {sample['id']}")
    results = _write_results(
        output,
        "scores.jsonl",
        [(r, p, str(predictions), str(dataset_root), str(source_root)) for r, p in zip(selected, selected_pairs)],
        _score_case,
        workers,
    )
    report = {
        "protocol": PROTOCOL,
        "evaluation_protocol": "reference_length",
        "model": model,
        "dataset": str(dataset_root),
        "predictions": str(predictions),
        "samples": len(results),
        "frozen_samples": metadata["samples"],
        "shard_index": shard_index,
        "shards": shards,
        "complete": True,
        **summarize(results),
    }
    write_json(Path(output) / "report.json", report)
    return report


def rescore_baseline(
    dataset,
    output,
    model,
    workers=1,
    *,
    predictions_root=None,
    baseline_scores=None,
):
    from evaluation.prompt_paraphrase.text_paraphrase.scoring import (
        _resolve_baseline_candidates,
        _score_baseline,
    )

    dataset_root = Path(dataset).expanduser().resolve()
    metadata, _, _ = load_visual(dataset_root)
    source_root = to_absolute(metadata["dataset"], dataset_root)
    rows = _source_rows(source_root)
    work, source = _resolve_baseline_candidates(
        rows,
        dataset_root=source_root,
        predictions_root=predictions_root,
        baseline_scores=baseline_scores,
    )
    results = _write_results(output, "scores.jsonl", work, _score_baseline, workers)
    report = {
        "protocol": PROTOCOL,
        "evaluation_protocol": "reference_length",
        "model": model,
        "dataset": str(dataset_root),
        "samples": len(rows),
        "scored_samples": len(results),
        "complete": True,
        "kind": "id_baseline_rescore",
        "candidate_source": source,
        **summarize(results),
    }
    write_json(Path(output) / "report.json", report)
    return report


def merge(dataset, shard_roots, output, id_scores):
    dataset_root = Path(dataset).expanduser().resolve()
    metadata, rows, _ = load_visual(dataset_root)
    source_root = to_absolute(metadata["dataset"], dataset_root)
    reports = [json.loads((Path(root) / "report.json").read_text()) for root in shard_roots]
    if not reports:
        raise ValueError("no scoring shards")
    first = reports[0]
    shared = ("protocol", "evaluation_protocol", "model", "dataset", "shards")
    if any(r.get("complete") is not True or any(r[k] != first[k] for k in shared) for r in reports):
        raise ValueError("incompatible or incomplete shards")
    if (
        not matches_schema(first["protocol"], PROTOCOL)
        or first["dataset"] != str(dataset_root)
        or first["evaluation_protocol"] != "reference_length"
    ):
        raise ValueError("wrong package or protocol")
    if len(reports) != first["shards"] or {r["shard_index"] for r in reports} != set(range(first["shards"])):
        raise ValueError("missing or duplicate shard")
    baseline = json.loads(Path(id_scores).with_name("report.json").read_text())
    if (
        baseline.get("kind") != "id_baseline_rescore"
        or baseline.get("complete") is not True
        or any(baseline.get(k) != first[k] for k in ("model", "dataset", "evaluation_protocol"))
    ):
        raise ValueError("baseline must match model, dataset, and protocol")
    scores = [r for root in shard_roots for r in read_rows(Path(root) / "scores.jsonl")]
    by_id = {r["sample_id"]: r for r in scores}
    if len(by_id) != len(scores) or set(by_id) != {r["sample"]["id"] for r in rows}:
        raise ValueError("scoring does not cover the paired source set exactly once")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("merge output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "scores.jsonl").write_text("".join(json.dumps(by_id[r["sample"]["id"]]) + "\n" for r in rows))
    comparison = compare_subset(id_scores, scores, rows, _source_rows(source_root))
    write_json(output / "comparison.json", comparison)
    report = {
        **first,
        "samples": metadata["samples"],
        "scored_samples": metadata["samples"],
        **summarize(scores),
        "baseline_scores": str(Path(id_scores).resolve()),
    }
    report.pop("shard_index", None)
    write_json(output / "report.json", report)
    return report


def compare_subset(id_scores, variants, selected_rows, full_rows):
    """Keep the full ID denominator separate from the paired visual subset."""
    originals = read_rows(id_scores)

    def index(values):
        result = {r["sample_id"]: r for r in values}
        if len(result) != len(values) or any(r["success"] not in (0, 1) for r in values):
            raise ValueError("duplicate sample or nonbinary score")
        return result

    base, indexed = index(originals), index(variants)
    if set(base) != {r["sample"]["id"] for r in full_rows} or set(indexed) != {r["sample"]["id"] for r in selected_rows}:
        raise ValueError("comparison needs the complete ID baseline and visual subset")
    tasks, families = defaultdict(list), defaultdict(list)
    transitions = {f"{a}_to_{b}": 0 for a in (0, 1) for b in (0, 1)}
    for row in selected_rows:
        sid = row["sample"]["id"]
        a, b = base[sid], indexed[sid]
        if a["evaluation_task"] != row["evaluation_task"] or b["evaluation_task"] != row["evaluation_task"]:
            raise ValueError("paired task identity differs")
        values = (a["success"], b["success"])
        tasks[row["evaluation_task"]].append(values)
        families[row["sample"]["task_family"]].append(values)
        transitions[f"{values[0]}_to_{values[1]}"] += 1

    def summary(values):
        n = len(values)
        a, b = map(sum, zip(*values))
        return dict(
            samples=n,
            id_correct=a,
            variant_correct=b,
            id_success_rate=a / n,
            variant_success_rate=b / n,
            success_rate_difference=(b - a) / n,
        )

    return dict(
        full_id_baseline=dict(
            samples=len(base),
            correct=sum(r["success"] for r in base.values()),
            success_rate=sum(r["success"] for r in base.values()) / len(base),
        ),
        paired_subset=summary([v for group in tasks.values() for v in group]),
        tasks={k: summary(v) for k, v in sorted(tasks.items())},
        families={k: summary(v) for k, v in sorted(families.items())},
        paired_transitions=dict(transitions),
    )
