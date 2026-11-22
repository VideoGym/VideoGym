from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, TypeVar

from PIL import Image


RECORD_NAMESPACE = "videogym"
_SAMPLE_DIGEST_DOMAIN = "videogym.sample_identity.v1"
SAMPLE_ID_DIGEST_LENGTH = 24
_SampleT = TypeVar("_SampleT")


def _sha256_image_pixels(path: str) -> str:
    """Hash canonical decoded RGB pixels, not PNG container metadata."""

    digest = hashlib.sha256()
    with Image.open(Path(path)) as image:
        canonical = image.convert("RGB")
        digest.update(f"{canonical.width}x{canonical.height}:RGB".encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def build_canonical_sample_id(
    *,
    task_name: str,
    seed: int,
    semantic_input: Mapping[str, object],
    ground_truth: Mapping[str, object],
    prompt: str,
    image: str,
) -> str:
    """Return a worker- and output-root-independent semantic sample identity."""

    clean_task_name = str(task_name).strip()
    if not clean_task_name:
        raise ValueError("canonical sample identity requires a task name")
    if type(seed) is not int:
        raise ValueError("canonical sample identity requires an exact integer seed")
    clean_prompt = str(prompt or "").strip()
    clean_image = str(image or "").strip()
    if not clean_image:
        raise ValueError("canonical sample identity requires an input image")

    identity_payload = {
        "version": _SAMPLE_DIGEST_DOMAIN,
        "task_name": clean_task_name,
        "seed": seed,
        "semantic_input": dict(semantic_input),
        "ground_truth": {
            str(key): value
            for key, value in ground_truth.items()
            if value is not None
        },
        "prompt": clean_prompt,
        "image_sha256": _sha256_image_pixels(clean_image),
    }
    serialized = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{clean_task_name}-{digest[:SAMPLE_ID_DIGEST_LENGTH]}"


def index_unique_samples(
    samples: Iterable[_SampleT],
    *,
    context: str,
) -> dict[str, _SampleT]:
    """Index opaque canonical rows by id, rejecting every duplicate."""

    indexed: dict[str, _SampleT] = {}
    for sample in samples:
        sample_id = str(getattr(sample, "id", "") or "").strip()
        if not sample_id:
            raise ValueError(f"{context}: canonical sample has no id")
        if sample_id in indexed:
            raise ValueError(
                f"{context}: duplicate canonical sample id {sample_id!r}"
            )
        indexed[sample_id] = sample
    return indexed


def validate_sample_identity_witness(
    *,
    sample_id: str,
    task_name: str,
    seed: object,
) -> None:
    """Validate the persisted identity witnesses without reopening media."""

    if type(seed) is not int:
        raise ValueError(f"{task_name}: seed must be an exact integer")
    expected = re.compile(
        rf"{re.escape(task_name)}-[0-9a-f]{{{SAMPLE_ID_DIGEST_LENGTH}}}"
    )
    if expected.fullmatch(str(sample_id)) is None:
        raise ValueError(
            f"{task_name}: canonical id must be the task name followed by a "
            f"{SAMPLE_ID_DIGEST_LENGTH}-hex identity digest"
        )


__all__ = [
    "RECORD_NAMESPACE",
    "SAMPLE_ID_DIGEST_LENGTH",
    "build_canonical_sample_id",
    "index_unique_samples",
    "validate_sample_identity_witness",
]
