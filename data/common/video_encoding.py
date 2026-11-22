from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

VIDEO_FPS: int = 16
# One ffmpeg invocation that dies under machine load is a transient, not a
# defect: every load-induced failure observed so far encoded cleanly on a
# serial retry.  Because the CLI aborts an entire multi-task run on the first
# exception, failing on the first hiccup would turn a transient into a dead
# run.  A deterministic failure is not masked by this: it fails all attempts
# and then raises, so retrying can only ever rescue a genuine transient.
VIDEO_ENCODE_ATTEMPTS: int = 3
# The retry exists to outlast a load spike, so the gaps have to be long enough
# for one to pass; retrying immediately just fails again under the same
# conditions.  Five seconds total is nothing against a multi-task run, and it
# is only ever spent on an encode that is already failing.
_VIDEO_ENCODE_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 4.0)


class VideoEncodeError(RuntimeError):
    """ffmpeg could not produce a declared video after every allowed attempt."""
SIMPLE_VIDEO_FRAMES: int = 49
MOTION_VIDEO_FRAMES: int = 65
MULTISTEP_VIDEO_FRAMES: int = 97
# Historical default retiming budget, not a semantic acceptance ceiling.
MAX_VIDEO_FRAMES: int = 157


def align_wan_video_num_frames(num_frames: int) -> int:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    remainder = (num_frames - 1) % 4
    if remainder == 0:
        return num_frames
    return num_frames + (4 - remainder)


def largest_wan_video_num_frames_at_most(num_frames: int) -> int:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    return num_frames - ((num_frames - 1) % 4)


def _clone_frame(frame: Image.Image | np.ndarray) -> Image.Image | np.ndarray:
    if isinstance(frame, Image.Image):
        return frame.copy()
    return np.array(frame, copy=True)


def pad_frames_to_wan_compatible_length(
    frames: Sequence[Image.Image | np.ndarray],
) -> list[Image.Image | np.ndarray]:
    frame_list = list(frames)
    if not frame_list:
        return []
    target_len = align_wan_video_num_frames(len(frame_list))
    if target_len == len(frame_list):
        return frame_list
    last_frame = frame_list[-1]
    frame_list.extend(_clone_frame(last_frame) for _ in range(target_len - len(frame_list)))
    return frame_list


def fit_frames_to_video_budget(
    frames: Sequence[Image.Image | np.ndarray],
    *,
    max_frames: int = MAX_VIDEO_FRAMES,
    final_hold_frames: int = 1,
    protected_indices: Sequence[int] | None = None,
) -> list[Image.Image | np.ndarray]:
    """Return 4N+1 frames without silently dropping semantic boundaries.

    Short sequences are padded only at the solved endpoint. An overlong
    sequence must name every contact/commit/state boundary in
    ``protected_indices``; otherwise generation fails and the task must reduce
    its semantic workload or provide phase metadata. Retiming may remove only
    unprotected interpolation frames.
    """

    frame_list = list(frames)
    if not frame_list:
        return []

    target_len = largest_wan_video_num_frames_at_most(max_frames)
    if len(frame_list) <= target_len:
        padded = pad_frames_to_wan_compatible_length(frame_list)
        if len(padded) <= target_len:
            return padded
        frame_list = padded

    if target_len <= 1:
        return [_clone_frame(frame_list[-1])]

    if protected_indices is None:
        raise ValueError(
            f"Video has {len(frame_list)} frames but the budget is {target_len}; "
            "provide protected semantic boundaries or reduce the task length"
        )

    protected = {
        0,
        len(frame_list) - 1,
        *(int(index) for index in protected_indices),
    }
    if any(index < 0 or index >= len(frame_list) for index in protected):
        raise ValueError("protected frame index is outside the source sequence")
    tail_len = max(1, min(int(final_hold_frames), target_len, len(frame_list)))
    protected.update(range(len(frame_list) - tail_len, len(frame_list)))
    if len(protected) > target_len:
        raise ValueError(
            f"{len(protected)} protected frames exceed the {target_len}-frame budget"
        )

    selected = set(protected)
    candidates = [index for index in range(len(frame_list)) if index not in selected]
    remaining = target_len - len(selected)
    if remaining and candidates:
        if remaining >= len(candidates):
            selected.update(candidates)
        else:
            positions = np.linspace(0, len(candidates) - 1, num=remaining)
            selected.update(candidates[int(round(position))] for position in positions)
    ordered = sorted(selected)
    return [_clone_frame(frame_list[index]) for index in ordered]


def inspect_video_contract(path: Path) -> dict[str, object]:
    """Read the canonical media properties used by generator-side QA."""

    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open generated video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps if fps > 0 else None,
    }


def validate_video_contract(path: Path) -> dict[str, object]:
    metrics = inspect_video_contract(path)
    failures: list[str] = []
    if abs(float(metrics["fps"]) - VIDEO_FPS) > 0.01:
        failures.append(f"fps={metrics['fps']}")
    if (int(metrics["width"]), int(metrics["height"])) != (512, 512):
        failures.append(f"size={metrics['width']}x{metrics['height']}")
    frame_count = int(metrics["frame_count"])
    if frame_count <= 0 or (frame_count - 1) % 4 != 0:
        failures.append(f"frame_count={frame_count} is not 4N+1")
    duration = metrics.get("duration_seconds")
    if duration is None or float(duration) <= 0:
        failures.append(f"duration={duration}")
    if failures:
        raise ValueError(f"Video media contract failed for {path}: {', '.join(failures)}")
    return metrics


def _frame_to_rgb_array(frame: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(frame, Image.Image):
        array = np.array(frame.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Video frames must have shape HxWx3")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            array = array.copy()
    return np.ascontiguousarray(array)


def _prepare_frames(frames: Sequence[Image.Image | np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    if not frames:
        return [], 0, 0

    arrays = [_frame_to_rgb_array(frame) for frame in frames]
    height, width = arrays[0].shape[:2]
    for array in arrays[1:]:
        if array.shape[:2] != (height, width):
            raise ValueError("All video frames must share the same size")

    target_width = width + (width % 2)
    target_height = height + (height % 2)
    if target_width != width or target_height != height:
        padded_arrays: list[np.ndarray] = []
        for array in arrays:
            padded = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
            padded[:height, :width] = array
            padded_arrays.append(padded)
        arrays = padded_arrays
        width = target_width
        height = target_height

    return arrays, width, height


def _run_ffmpeg_encode(
    cmd: Sequence[str],
    frame_arrays: Sequence[np.ndarray],
    path: Path,
) -> str | None:
    """Run one encode attempt; return ``None`` on success or a failure reason."""

    try:
        proc = subprocess.Popen(list(cmd), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        return f"could not launch ffmpeg: {exc}"

    stderr_output = b""
    try:
        for frame in frame_arrays:
            if proc.stdin is None:
                raise BrokenPipeError("ffmpeg stdin pipe is unavailable")
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stderr is not None:
            stderr_output = proc.stderr.read()
        proc.wait()
    else:
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stderr is not None:
            stderr_output = proc.stderr.read()
        proc.wait()
    finally:
        if proc.stderr is not None:
            proc.stderr.close()

    if proc.returncode != 0:
        detail = stderr_output.decode("utf-8", errors="replace").strip()
        return detail or f"ffmpeg exited with return code {proc.returncode}"
    if not path.exists():
        return "ffmpeg reported success but wrote no file"
    if path.stat().st_size <= 0:
        return "ffmpeg reported success but wrote a zero-byte file"
    return None


def encode_rgb_frames_to_webm(
    frames: Sequence[Image.Image | np.ndarray],
    path: Path,
    *,
    fps: int = 16,
    extra_args: Sequence[str] | None = None,
) -> bool:
    """Encode RGB frames as the lossless VP9 WebM used by canonical data.

    Returns ``True`` once the file is on disk and non-empty.  ``False`` means
    only that the caller supplied no frames, which is a caller decision rather
    than a failure.  Anything else -- a missing ffmpeg, a launch error, a
    non-zero exit, an empty output -- raises :class:`VideoEncodeError` after
    ``VIDEO_ENCODE_ATTEMPTS`` attempts.  An encode failure must never be
    reported as a quiet falsy return, because callers historically turned that
    into ``None``/``0`` and still wrote a manifest row declaring the video.
    """

    frame_arrays, width, height = _prepare_frames(frames)
    if not frame_arrays:
        return False
    if fps <= 0:
        raise ValueError("fps must be positive")

    path = Path(path)
    if path.suffix.lower() != ".webm":
        raise ValueError(f"Lossless source videos must use a .webm path: {path}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        # Deterministic: no number of retries conjures an encoder.
        raise VideoEncodeError(f"ffmpeg is unavailable, cannot encode {path}")

    cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "gbrp",
        "-lossless",
        "1",
        "-row-mt",
        "1",
        # Matroska/WebM otherwise receives a random SegmentUID even when
        # every decoded frame is identical.  Deterministic containers are
        # required for worker-count-independent canonical asset hashes.
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-map_metadata",
        "-1",
    ]
    if extra_args:
        cmd.extend(list(extra_args))
    cmd.append(str(path))

    attempts = max(1, int(VIDEO_ENCODE_ATTEMPTS))
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        # Re-made every attempt, not once up front: one of the two confirmed
        # interference modes is an output tree removed out from under a
        # writer, and a retry that does not restore the directory just
        # reproduces "No such file or directory" three times.
        path.parent.mkdir(parents=True, exist_ok=True)
        reason = _run_ffmpeg_encode(cmd, frame_arrays, path)
        if reason is None:
            if attempt > 1:
                # Never absorb a retry silently: a run that only survives on
                # the second try is still evidence of a sick machine.
                print(
                    f"Warning: ffmpeg encoded {path} only on attempt {attempt} of "
                    f"{attempts}; earlier attempts failed: {' | '.join(failures)}",
                    flush=True,
                )
            return True

        failures.append(f"attempt {attempt}: {reason}")
        # ffmpeg's -y truncates the target before it fails, so a stale partial
        # file must not be left behind to look like a usable artifact.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        if attempt < attempts:
            backoff_index = min(attempt - 1, len(_VIDEO_ENCODE_RETRY_BACKOFF_SECONDS) - 1)
            time.sleep(_VIDEO_ENCODE_RETRY_BACKOFF_SECONDS[backoff_index])

    raise VideoEncodeError(
        f"ffmpeg failed to encode {path} after {attempts} attempts: "
        + " ; ".join(failures)
    )
