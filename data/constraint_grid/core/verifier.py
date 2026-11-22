from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from data.constraint_grid.core.final_delivery import (
    verify_constraint_progressive_video,
)
from data.common.schemas import CanonicalSample, EvalRecord
from data.common.verifiers.candidates import resolve_candidate_video


def verify_grid_completion(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
) -> List[EvalRecord]:
    """Verify the formal TI2V final delivery.

    TI2V is a whole-video temporal program; no reasoning text or intermediate
    reasoning image is ever inspected.
    """

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
            candidate_video = resolve_candidate_video(pred_root, sample.id, sample.split)
            if candidate_video is None:
                raise FileNotFoundError(f"No prediction video found for sample_id={sample.id}")
            metrics, passed = verify_constraint_progressive_video(sample, candidate_video)
            record.prediction_asset = candidate_video.as_posix()
            record.offline_metrics.update(metrics)
            record.offline_pass = passed
            record.error = None if passed else str(metrics.get("failure_reason") or "whole-video replay failed")
        except Exception as exc:  # pragma: no cover - converted to an EvalRecord failure
            record.error = str(exc)
            record.offline_pass = False
        records.append(record)
    return records


__all__ = ["verify_grid_completion"]
