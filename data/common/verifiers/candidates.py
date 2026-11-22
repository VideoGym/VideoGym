from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from data.common.paths import first_existing_path
from data.common.schemas import CanonicalSample


def resolve_source_record_id(sample: CanonicalSample) -> str:
    raw_record = resolve_source_record(sample)
    raw_id = str(raw_record.get("id") or "").strip()
    if raw_id:
        return raw_id
    return sample.id


def resolve_source_record(sample: CanonicalSample) -> dict[str, Any]:
    """Return an evaluator-ready copy of the canonical semantic source.

    Canonical manifests intentionally scrub record ids and answer-derived
    fields from ``semantic_input``.  Evaluators still need a stable lookup id
    and the task-owned structured *input* geometry, so reconstruct only those
    two envelopes here.  No value from ``ground_truth`` is copied and no
    ``correct_option``/saved target is reintroduced.
    """

    raw_record = sample.semantic_input
    if not isinstance(raw_record, dict):
        raise ValueError(
            f"Canonical sample {sample.id!r} does not contain semantic_input"
        )
    record = dict(raw_record)
    record.setdefault("id", sample.id)
    semantic_state = record.get("semantic_state")
    if isinstance(semantic_state, dict):
        record["structured_answer"] = dict(semantic_state)
    return record


def resolve_source_task_dir(sample: CanonicalSample) -> Path:
    """Return the asset base directory for the canonical sample."""

    if not sample.image:
        raise ValueError(
            f"Canonical sample {sample.id!r} does not contain an input image"
        )
    return Path(sample.image).expanduser().resolve().parent


DEFAULT_CANDIDATE_VIDEOS = [
    "generated.webm",
    "output.webm",
    "prediction.webm",
    "result.webm",
    "generated.mp4",
    "output.mp4",
    "prediction.mp4",
    "result.mp4",
]


def resolve_candidate_video(
    pred_root: Path, sample_id: str, split: Optional[str] = None
) -> Optional[Path]:
    candidate_dirs = []
    if split:
        candidate_dirs.append(pred_root / split / sample_id)
    candidate_dirs.append(pred_root / sample_id)
    candidates = []
    for sample_dir in candidate_dirs:
        if not sample_dir.exists() or not sample_dir.is_dir():
            continue
        candidates.extend(sample_dir / name for name in DEFAULT_CANDIDATE_VIDEOS)
        candidates.extend(sample_dir.glob("*.webm"))
        candidates.extend(sample_dir.glob("*.mp4"))
        candidates.extend(sample_dir.glob("*.mov"))
    return first_existing_path(candidates)
