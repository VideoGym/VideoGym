"""Verification that a canonical manifest row never declares absent media.

A manifest row is the contract future ``training/`` and ``evaluation/``
pipelines consume.  A row that promises a terminal image or a solution video
which is not on disk is worse than a missing row: a replay reads it as a
verifier defect, and any check that treats "nothing to compare" as "nothing
wrong" scores the absent artifact as a pass.

Every declared asset must exist, be non-zero, and be decodable. The helpers
here enforce that contract and raise :class:`DeclaredArtifactError`, naming
the task, sample id, artifact role, and path.

The module holds no mutable module-level state and imports no optional heavy
runtime, so it is safe to call from any generation worker process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from PIL import Image

CONTRACT_IMAGE_SIZE: Tuple[int, int] = (512, 512)


class DeclaredArtifactError(RuntimeError):
    """A declared canonical artifact is missing, empty, or undecodable."""


def _fail(
    *,
    task_name: str,
    sample_id: Optional[str],
    role: str,
    path: Path,
    problem: str,
) -> "DeclaredArtifactError":
    sample = f" sample {sample_id}" if sample_id else ""
    return DeclaredArtifactError(
        f"{task_name}{sample}: declared {role} {problem}: {path}"
    )


def _require_readable_file(
    path: Path,
    *,
    task_name: str,
    sample_id: Optional[str],
    role: str,
) -> int:
    """Return the artifact's byte size, or raise naming the missing asset."""

    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise _fail(
            task_name=task_name,
            sample_id=sample_id,
            role=role,
            path=path,
            problem="does not exist",
        ) from exc
    except OSError as exc:
        raise _fail(
            task_name=task_name,
            sample_id=sample_id,
            role=role,
            path=path,
            problem=f"cannot be read ({exc.strerror or exc})",
        ) from exc
    if not path.is_file():
        raise _fail(
            task_name=task_name,
            sample_id=sample_id,
            role=role,
            path=path,
            problem="is not a regular file",
        )
    if stat.st_size <= 0:
        raise _fail(
            task_name=task_name,
            sample_id=sample_id,
            role=role,
            path=path,
            problem="is zero bytes",
        )
    return int(stat.st_size)


def verify_image_artifact(
    path: str | Path,
    *,
    task_name: str,
    sample_id: Optional[str] = None,
    role: str = "image",
    expected_size: Optional[Tuple[int, int]] = CONTRACT_IMAGE_SIZE,
) -> Tuple[int, int]:
    """Confirm a declared image exists, is non-zero, and fully decodes.

    ``Image.open`` alone only parses the header, so a write that was truncated
    part-way through still opens and still reports the right dimensions.  The
    pixels are therefore decoded here; that is what distinguishes a usable
    artifact from a plausible-looking stub.
    """

    asset = Path(path)
    _require_readable_file(asset, task_name=task_name, sample_id=sample_id, role=role)
    try:
        with Image.open(asset) as handle:
            handle.load()
            size = handle.size
    except DeclaredArtifactError:
        raise
    except Exception as exc:  # PIL raises a wide, version-dependent family here
        raise _fail(
            task_name=task_name,
            sample_id=sample_id,
            role=role,
            path=asset,
            problem=f"cannot be decoded ({type(exc).__name__}: {exc})",
        ) from exc
    if expected_size is not None and size != tuple(expected_size):
        raise _fail(
            task_name=task_name,
            sample_id=sample_id,
            role=role,
            path=asset,
            problem=f"must be {expected_size[0]}x{expected_size[1]}, got {size[0]}x{size[1]}",
        )
    return size


def verify_video_artifact(
    path: str | Path,
    *,
    task_name: str,
    sample_id: Optional[str] = None,
    role: str = "solution video",
    expected_frame_count: Optional[int] = None,
) -> int:
    """Confirm a declared video exists, is non-zero, and decodes end to end.

    The container's frame-count header is metadata, not evidence: a truncated
    encode keeps the header it was given while the frames behind it are gone.
    Every frame is therefore decoded and counted.  When
    ``expected_frame_count`` is supplied the decoded total must match it, so a
    partially written video cannot pass as a complete visual program.
    """

    import cv2

    asset = Path(path)
    _require_readable_file(asset, task_name=task_name, sample_id=sample_id, role=role)

    capture = cv2.VideoCapture(asset.as_posix())
    try:
        if not capture.isOpened():
            raise _fail(
                task_name=task_name,
                sample_id=sample_id,
                role=role,
                path=asset,
                problem="cannot be opened by the video decoder",
            )
        decoded = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            decoded += 1
    finally:
        capture.release()

    if decoded <= 0:
        raise _fail(
            task_name=task_name,
            sample_id=sample_id,
            role=role,
            path=asset,
            problem="decodes to zero frames",
        )
    if expected_frame_count is not None and decoded != int(expected_frame_count):
        raise _fail(
            task_name=task_name,
            sample_id=sample_id,
            role=role,
            path=asset,
            problem=(
                f"decodes to {decoded} frames but the row declares "
                f"{int(expected_frame_count)}"
            ),
        )
    return decoded


def verify_declared_images(
    paths: Sequence[Tuple[str, str]],
    *,
    task_name: str,
    sample_id: Optional[str] = None,
    already_verified: Optional[set[str]] = None,
) -> None:
    """Verify ``(path, role)`` pairs, decoding each distinct asset once.

    ``already_verified`` lets one caller share a per-sample set across several
    groups of artifacts.  It is deliberately a caller-owned argument rather
    than module state so that concurrent generation workers never share it.
    """

    seen = already_verified if already_verified is not None else set()
    for raw_path, role in paths:
        if not raw_path:
            continue
        key = str(raw_path)
        if key in seen:
            continue
        seen.add(key)
        verify_image_artifact(
            key,
            task_name=task_name,
            sample_id=sample_id,
            role=role,
        )


__all__ = [
    "CONTRACT_IMAGE_SIZE",
    "DeclaredArtifactError",
    "verify_declared_images",
    "verify_image_artifact",
    "verify_video_artifact",
]
