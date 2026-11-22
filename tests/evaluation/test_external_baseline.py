"""Baseline rescoring consumes explicit rollout roots or In-Domain scores."""

import json
from importlib import import_module

import pytest


@pytest.mark.parametrize("variant", ["text_paraphrase", "vision_prompt_paraphrase"])
def test_rescore_accepts_predictions_root_or_baseline_scores(tmp_path, monkeypatch, variant):
    scoring = import_module(f"evaluation.prompt_paraphrase.{variant}.scoring")
    predictions = tmp_path / "rollouts"
    candidate = predictions / "question" / "prediction.webm"
    candidate.parent.mkdir(parents=True)
    candidate.touch()
    rows = [
        {
            "sample": {"id": "question", "split": "eval"},
            "evaluation_task": "id_midpoint",
        }
    ]
    metadata = {"samples": 1, "dataset": str(tmp_path / "source")}
    expected_base = (
        tmp_path.resolve()
        if variant == "text_paraphrase"
        else (tmp_path / "source").resolve()
    )
    if variant == "text_paraphrase":
        monkeypatch.setattr(scoring, "load_package", lambda *args: (metadata, rows))
    else:
        monkeypatch.setattr(scoring, "load_visual", lambda *args: (metadata, rows, []))
        monkeypatch.setattr(scoring, "_source_rows", lambda path: rows)
        monkeypatch.setattr(scoring, "summarize", lambda results: {"correct": 1})

    def score_work(output, filename, work, score_one, workers):
        assert work == [(rows[0], str(candidate.resolve()), expected_base)]
        return [{"success": 1}]

    monkeypatch.setattr(scoring, "_write_results", score_work)
    monkeypatch.setattr(scoring, "write_json", lambda *args: None)

    report = scoring.rescore_baseline(
        tmp_path,
        tmp_path / "output-dir",
        "custom-model",
        predictions_root=predictions,
    )
    assert report["model"] == "custom-model"
    assert report["candidate_source"] == str(predictions.resolve())
    assert report["correct"] == 1

    scores = tmp_path / "baseline.jsonl"
    scores.write_text(
        json.dumps(
            {
                "sample_id": "question",
                "track": "id",
                "evaluation_task": "id_midpoint",
                "verifier": {"prediction_asset": str(candidate)},
            }
        )
        + "\n"
    )
    report_from_scores = scoring.rescore_baseline(
        tmp_path,
        tmp_path / "output-scores",
        "custom-model",
        baseline_scores=scores,
    )
    assert report_from_scores["candidate_source"] == str(scores.resolve())
    assert report_from_scores["correct"] == 1

    with pytest.raises(ValueError, match="exactly one"):
        scoring.rescore_baseline(tmp_path, tmp_path / "output-none", "custom-model")

    scores.write_text("")
    with pytest.raises(ValueError, match="cover"):
        scoring.rescore_baseline(
            tmp_path,
            tmp_path / "output-empty",
            "custom-model",
            baseline_scores=scores,
        )
