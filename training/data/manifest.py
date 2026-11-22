from __future__ import annotations
from training.data.samples import Sample
from training.data.video import load_image, load_video
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path
import tempfile
from typing import Any
from data.common.paths import to_absolute
from data.registry import TASK_SPECS
from data.common.schemas import CanonicalSample
from data.common.video_encoding import validate_video_contract


def _is_training_row(row: dict[str, Any]) -> bool:
    # Check raw fields before CanonicalSample normalizers.
    # Evaluation-only wrappers and novel task routes can never become training.
    return (
        not any(key in row for key in ("evaluation_task", "sample", "track", "role"))
        and row.get("task_name") in TASK_SPECS
        and row.get("split") == "train"
    )


def _read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield number, row


def _select_tasks(tasks: Sequence[str] | None) -> set[str] | None:
    if not tasks or list(tasks) == ["all"]:
        return None
    selected = set(tasks)
    unknown = selected - set(TASK_SPECS)
    if unknown:
        raise ValueError(f"unknown training tasks: {sorted(unknown)}")
    return selected


def _sample_from_row(row: dict[str, Any], base: Path, *, verify_media: bool) -> Sample:
    exported = "canonical_sample" in row
    payload = row["canonical_sample"] if exported else row
    if (
        any(key in row for key in ("evaluation_task", "sample", "track", "role"))
        or not isinstance(payload, dict)
        or not _is_training_row(payload)
    ):
        raise ValueError(
            "training requires explicit split='train' rows; evaluation and OOD are excluded"
        )
    canonical = CanonicalSample.from_dict(payload, base_dir=base)
    if not canonical.prompt or not canonical.video:
        raise ValueError(f"{canonical.id}: missing prompt or complete reference video")
    sample = Sample(
        canonical.id,
        canonical.task_name,
        canonical.prompt,
        Path(canonical.image),
        Path(canonical.video),
        canonical.to_dict(),
    )
    if exported:
        for name in ("id", "task_name", "prompt"):
            if row.get(name) != getattr(sample, name):
                raise ValueError(
                    f"{sample.id}: exported {name} disagrees with canonical_sample"
                )
        for name in ("input_image", "reference_video"):
            if not row.get(name) or to_absolute(row[name], base).as_posix() != str(
                getattr(sample, name)
            ):
                raise ValueError(
                    f"{sample.id}: exported {name} disagrees with canonical_sample"
                )
    if verify_media:
        image = load_image(sample.input_image)
        video = load_video(sample.reference_video)
        contract = validate_video_contract(sample.reference_video)
        if (
            image.shape[:2] != video.frames.shape[1:3]
            or canonical.video_num_frames != len(video.frames)
            or abs(float(canonical.video_fps) - float(video.fps)) > 0.01
            or contract["frame_count"] != len(video.frames)
        ):
            raise ValueError(
                f"{sample.id}: canonical media metadata or canvas dimensions disagree"
            )
    return sample


def load_training_manifest(
    path: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    samples_per_task: int | None = None,
    verify_media: bool = False,
) -> list[Sample]:
    """Read canonical or exported rows; reject evaluation contamination.

    ``samples_per_task`` takes the first at most N rows per task in persisted
    manifest order. Decoding is lazy unless ``verify_media`` is requested.
    """
    path = Path(path).expanduser().resolve()
    selected = _select_tasks(tasks)
    if samples_per_task is not None and (
        type(samples_per_task) is not int or samples_per_task < 1
    ):
        raise ValueError("samples_per_task must be a positive integer")
    samples, seen, counts = [], set(), Counter()
    for number, row in _read_rows(path):
        payload = row.get("canonical_sample", row)
        if (
            selected is not None
            and isinstance(payload, dict)
            and payload.get("task_name") not in selected
        ):
            continue
        try:
            sample = _sample_from_row(row, path.parent, verify_media=verify_media)
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        if sample.id in seen:
            raise ValueError(
                f"{path}:{number}: duplicate training sample id {sample.id}"
            )
        seen.add(sample.id)
        if (
            samples_per_task is not None
            and counts[sample.task_name] >= samples_per_task
        ):
            continue
        samples.append(sample)
        counts[sample.task_name] += 1
    if not samples:
        raise ValueError("training manifest contains no selected samples")
    if selected is not None and selected - set(counts):
        raise ValueError(
            f"training manifest is missing requested tasks: {sorted(selected - set(counts))}"
        )
    return samples


def export_training_manifest(
    source: str | Path,
    output: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    samples_per_task: int | None = None,
) -> dict[str, Any]:
    """Export only In-Domain training examples and preserve full canonical verifier state."""
    source, output = (
        Path(source).expanduser().resolve(),
        Path(output).expanduser().resolve(),
    )
    selected = _select_tasks(tasks)
    if samples_per_task is not None and (
        type(samples_per_task) is not int or samples_per_task < 1
    ):
        raise ValueError("samples_per_task must be a positive integer")
    samples, seen, counts = [], set(), Counter()
    for number, row in _read_rows(source):
        payload = row.get("canonical_sample", row)
        if (
            any(key in row for key in ("evaluation_task", "sample", "track", "role"))
            or not isinstance(payload, dict)
            or not _is_training_row(payload)
        ):
            continue
        if selected is not None and payload["task_name"] not in selected:
            continue
        if payload.get("id") in seen:
            raise ValueError(
                f"{source}:{number}: duplicate training sample id {payload.get('id')}"
            )
        seen.add(payload.get("id"))
        if (
            samples_per_task is not None
            and counts[payload["task_name"]] >= samples_per_task
        ):
            continue
        sample = _sample_from_row(row, source.parent, verify_media=True)
        samples.append(sample)
        counts[sample.task_name] += 1
    if not samples:
        raise ValueError("source contains no selected training samples")
    if selected is not None and selected - set(counts):
        raise ValueError(
            f"source is missing requested training tasks: {sorted(selected - set(counts))}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            for sample in samples:
                stream.write(
                    json.dumps(
                        sample.to_dict(base_dir=output.parent),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "manifest": str(output),
        "samples": len(samples),
        "tasks": dict(sorted(counts.items())),
    }
