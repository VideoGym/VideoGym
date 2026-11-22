from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps

from data.common.video_encoding import (
    MAX_VIDEO_FRAMES,
    VIDEO_FPS,
    align_wan_video_num_frames,
    encode_rgb_frames_to_webm,
)

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency in some environments
    cv2 = None  # type: ignore[assignment]


TARGET_IMAGE_SIZE = (512, 512)
TARGET_VIDEO_FPS = VIDEO_FPS
TARGET_VIDEO_MAX_FRAMES = MAX_VIDEO_FRAMES
_IMAGE_PATH_KEYS = ("image", "reasoning_image", "solution_image_path")
_VIDEO_PATH_KEYS = ("solution_video_path", "video")
_IMAGE_LIST_KEYS = (
    "option_image_paths",
    "option_images",
    "ti2v_input_images",
    "video_keyframe_paths",
)


def _normalize_public_prompt_fields(record: Dict[str, Any]) -> None:
    for key in list(record):
        if key == "prompt" or (key.endswith("_prompt") and key != "ti2v_prompt"):
            record.pop(key, None)


def _resolve_asset_path(base_dir: Path, raw_value: Any) -> Optional[Path]:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def _iter_additional_image_paths(base_dir: Path, record: Dict[str, Any]) -> Iterable[Path]:
    containers = [record]
    for nested_key in ("answer_payload", "structured_answer"):
        nested = record.get(nested_key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        for key in _IMAGE_LIST_KEYS:
            raw_values = container.get(key)
            if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
                continue
            for raw_value in raw_values:
                asset_path = _resolve_asset_path(base_dir, raw_value)
                if asset_path is not None:
                    yield asset_path


def _parse_positive_int(raw_value: Any) -> Optional[int]:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def estimate_normalized_video_frame_count(
    raw_num_frames: Any,
    raw_fps: Any,
) -> Optional[int]:
    source_num_frames = _parse_positive_int(raw_num_frames)
    if source_num_frames is None:
        return None
    source_fps = _parse_positive_int(raw_fps) or TARGET_VIDEO_FPS
    estimated_duration = source_num_frames / float(source_fps)
    target_frames = max(1, int(round(estimated_duration * TARGET_VIDEO_FPS)))
    target_frames = align_wan_video_num_frames(target_frames)
    return min(target_frames, TARGET_VIDEO_MAX_FRAMES)


def _estimate_background_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    samples = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((width - 1, 0)),
        rgb.getpixel((0, height - 1)),
        rgb.getpixel((width - 1, height - 1)),
    ]
    channels = np.array(samples, dtype=np.uint8)
    medians = np.median(channels, axis=0)
    return tuple(int(round(value)) for value in medians.tolist())


def _compute_uniform_border_crop_box(
    image: Image.Image,
    *,
    border_tolerance: int = 8,
    extra_margin: int = 8,
) -> tuple[int, int, int, int]:
    rgb = image.convert("RGB")
    bg = np.array(_estimate_background_color(rgb), dtype=np.int16)
    array = np.asarray(rgb, dtype=np.int16)
    diff = np.max(np.abs(array - bg), axis=2)
    mask = diff > border_tolerance
    if not mask.any():
        return (0, 0, rgb.width, rgb.height)
    ys, xs = np.nonzero(mask)
    left = max(0, int(xs.min()) - extra_margin)
    top = max(0, int(ys.min()) - extra_margin)
    right = min(rgb.width, int(xs.max()) + extra_margin + 1)
    bottom = min(rgb.height, int(ys.max()) + extra_margin + 1)
    return (left, top, right, bottom)


def _crop_uniform_border(
    image: Image.Image,
    *,
    border_tolerance: int = 8,
    extra_margin: int = 8,
) -> Image.Image:
    rgb = image.convert("RGB")
    crop_box = _compute_uniform_border_crop_box(
        rgb,
        border_tolerance=border_tolerance,
        extra_margin=extra_margin,
    )
    return rgb.crop(crop_box)


def _normalize_image_asset_with_transform(
    path: Path,
    *,
    crop_uniform_border: bool = True,
    content_size: tuple[int, int] = TARGET_IMAGE_SIZE,
) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    # Normalization rewrites the canonical asset in place.  Saving straight
    # over it means an interrupted or failing write leaves a truncated image
    # behind that still opens and still reports the right dimensions, so the
    # staging file below is what keeps a half-written PNG from ever reaching
    # the path a manifest row declares.
    staging_path = path.with_name(f"{path.stem}__normalizing{path.suffix}")
    try:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            # A native contract-sized render is already a complete canvas, not
            # an object crop.  Re-estimating a border independently for the
            # input, each reasoning state, and the terminal artifact changes
            # their coordinate systems whenever newly drawn content expands
            # the bounding box.  Preserve compliant assets byte-for-pixel and
            # require generators to own their intentional margins.
            if rgb.size == TARGET_IMAGE_SIZE:
                return None
            crop_box = (
                _compute_uniform_border_crop_box(rgb)
                if crop_uniform_border
                else (0, 0, rgb.width, rgb.height)
            )
            cropped = rgb.crop(crop_box)
            background = _estimate_background_color(rgb)
            resized = ImageOps.contain(cropped, content_size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", TARGET_IMAGE_SIZE, background)
            offset_x = (TARGET_IMAGE_SIZE[0] - resized.width) // 2
            offset_y = (TARGET_IMAGE_SIZE[1] - resized.height) // 2
            canvas.paste(resized, (offset_x, offset_y))
            canvas.save(staging_path)
        staging_path.replace(path)
        crop_width = max(1, crop_box[2] - crop_box[0])
        crop_height = max(1, crop_box[3] - crop_box[1])
        return {
            "source_size": (int(rgb.width), int(rgb.height)),
            "crop_box": crop_box,
            "scale_x": resized.width / float(crop_width),
            "scale_y": resized.height / float(crop_height),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "target_size": TARGET_IMAGE_SIZE,
        }
    except Exception:
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _normalize_video_frame(
    frame: np.ndarray,
    *,
    crop_uniform_border: bool = True,
    content_size: tuple[int, int] = TARGET_IMAGE_SIZE,
) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGB")
    cropped = _crop_uniform_border(image) if crop_uniform_border else image
    background = _estimate_background_color(image)
    resized = ImageOps.contain(cropped, content_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", TARGET_IMAGE_SIZE, background)
    offset_x = (TARGET_IMAGE_SIZE[0] - resized.width) // 2
    offset_y = (TARGET_IMAGE_SIZE[1] - resized.height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


def _transform_point_with_image_transform(
    point: Sequence[Any],
    transform: Dict[str, Any],
) -> list[float]:
    x = float(point[0])
    y = float(point[1])
    left, top, _, _ = transform["crop_box"]
    mapped_x = (x - float(left)) * float(transform["scale_x"]) + float(transform["offset_x"])
    mapped_y = (y - float(top)) * float(transform["scale_y"]) + float(transform["offset_y"])
    return [mapped_x, mapped_y]


def _transform_bbox_with_image_transform(
    bbox: Sequence[Any],
    transform: Dict[str, Any],
) -> list[float]:
    left_top = _transform_point_with_image_transform(bbox[:2], transform)
    right_bottom = _transform_point_with_image_transform(bbox[2:4], transform)
    target_width, target_height = transform["target_size"]
    return [
        max(0.0, min(left_top[0], right_bottom[0])),
        max(0.0, min(left_top[1], right_bottom[1])),
        min(float(target_width), max(left_top[0], right_bottom[0])),
        min(float(target_height), max(left_top[1], right_bottom[1])),
    ]


def _normalize_record_pixel_geometry(record: Dict[str, Any], transform: Optional[Dict[str, Any]]) -> None:
    if not transform:
        return
    for key in ("start_point", "goal_point", "start_pixel", "goal_pixel"):
        value = record.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            record[key] = _transform_point_with_image_transform(value, transform)
    cell_bboxes = record.get("cell_bboxes")
    if isinstance(cell_bboxes, list):
        normalized_rows: list[list[list[float]]] = []
        for row in cell_bboxes:
            if not isinstance(row, list):
                normalized_rows.append(row)
                continue
            normalized_row: list[list[float]] = []
            for bbox in row:
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    normalized_row.append(_transform_bbox_with_image_transform(bbox, transform))
                else:
                    normalized_row.append(bbox)
            normalized_rows.append(normalized_row)
        record["cell_bboxes"] = normalized_rows
    record["canvas_dimensions"] = [int(TARGET_IMAGE_SIZE[0]), int(TARGET_IMAGE_SIZE[1])]
    canvas_size = record.get("canvas_size")
    if isinstance(canvas_size, (int, float)) and canvas_size > 0 and TARGET_IMAGE_SIZE[0] == TARGET_IMAGE_SIZE[1]:
        record["canvas_size"] = int(TARGET_IMAGE_SIZE[0])


def _load_video_frames(path: Path) -> tuple[list[np.ndarray], Optional[float]]:
    if cv2 is None:
        return [], None
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        capture.release()
        return [], None
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, source_fps if source_fps and source_fps > 0 else None


def normalize_video_asset(
    path: Path,
    *,
    raw_num_frames: Any = None,
    raw_fps: Any = None,
    crop_uniform_border: bool = True,
    content_size: tuple[int, int] = TARGET_IMAGE_SIZE,
) -> Optional[int]:
    if not path.exists() or not path.is_file():
        return None

    frames, detected_fps = _load_video_frames(path)
    if not frames:
        return estimate_normalized_video_frame_count(raw_num_frames, raw_fps)

    source_num_frames = len(frames)
    source_fps = detected_fps or float(_parse_positive_int(raw_fps) or TARGET_VIDEO_FPS)
    # Do not crop/re-encode an already compliant visual program.  Normalizing
    # each frame around its own foreground causes camera zoom/pan as moving or
    # cumulative answer marks change the foreground extent, which destroys the
    # very process semantics that TI2V must verify.
    native_media_contract = bool(
        all(frame.shape == (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0], 3) for frame in frames)
        and abs(source_fps - TARGET_VIDEO_FPS) <= 0.25
        and source_num_frames % 4 == 1
    )
    if native_media_contract:
        return source_num_frames
    target_num_frames = estimate_normalized_video_frame_count(source_num_frames, source_fps)
    if target_num_frames is None:
        target_num_frames = min(align_wan_video_num_frames(source_num_frames), TARGET_VIDEO_MAX_FRAMES)
    if target_num_frames <= 0:
        return None

    if source_num_frames == 1:
        sampled_frames = [frames[0].copy() for _ in range(target_num_frames)]
    else:
        positions = np.linspace(0, source_num_frames - 1, num=target_num_frames)
        sampled_frames = [frames[int(round(position))].copy() for position in positions]

    sampled_frames = [
        _normalize_video_frame(
            frame,
            crop_uniform_border=crop_uniform_border,
            content_size=content_size,
        )
        for frame in sampled_frames
    ]

    temp_path = path.with_name(f"{path.stem}__normalized{path.suffix}")
    if not encode_rgb_frames_to_webm(sampled_frames, temp_path, fps=TARGET_VIDEO_FPS):
        if temp_path.exists():
            temp_path.unlink()
        return estimate_normalized_video_frame_count(raw_num_frames, raw_fps)

    temp_path.replace(path)
    return len(sampled_frames)


def normalize_task_output_records(
    base_dir: Path,
    records: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    normalized_records: list[Dict[str, Any]] = []
    normalized_images: dict[Path, Optional[Dict[str, Any]]] = {}
    normalized_videos: dict[Path, Optional[int]] = {}

    for raw_record in records:
        record = dict(raw_record)
        _normalize_public_prompt_fields(record)
        # A typed VisualReasoningContract means the generator owns its full-frame
        # 512x512 render geometry and compliant native video. Cropping or re-encoding
        # those assets here shifts structured coordinates and destroys process fidelity.
        has_native_render_contract = isinstance(record.get("task_contract"), dict)
        primary_image_transform: Optional[Dict[str, Any]] = None
        for key in _IMAGE_PATH_KEYS:
            asset_path = _resolve_asset_path(base_dir, record.get(key))
            if asset_path is None:
                continue
            if asset_path not in normalized_images:
                normalized_images[asset_path] = (
                    None
                    if has_native_render_contract
                    else _normalize_image_asset_with_transform(asset_path)
                )
            if key == "image":
                primary_image_transform = normalized_images.get(asset_path)

        for asset_path in _iter_additional_image_paths(base_dir, record):
            if asset_path not in normalized_images:
                normalized_images[asset_path] = (
                    None
                    if has_native_render_contract
                    else _normalize_image_asset_with_transform(asset_path)
                )

        for key in ("reasoning_images", "images"):
            raw_paths = record.get(key)
            if not isinstance(raw_paths, list):
                continue
            for raw_path in raw_paths:
                asset_path = _resolve_asset_path(base_dir, raw_path)
                if asset_path is None:
                    continue
                if asset_path not in normalized_images:
                    normalized_images[asset_path] = (
                        None
                        if has_native_render_contract
                        else _normalize_image_asset_with_transform(asset_path)
                    )


        _normalize_record_pixel_geometry(record, primary_image_transform)

        video_path: Optional[Path] = None
        for key in _VIDEO_PATH_KEYS:
            video_path = _resolve_asset_path(base_dir, record.get(key))
            if video_path is not None:
                break
        if video_path is not None:
            if video_path not in normalized_videos:
                normalized_videos[video_path] = (
                    _parse_positive_int(record.get("video_num_frames"))
                    if has_native_render_contract
                    else normalize_video_asset(
                        video_path,
                        raw_num_frames=record.get("video_num_frames"),
                        raw_fps=record.get("video_fps"),
                    )
                )
            normalized_frame_count = normalized_videos[video_path]
            record["video_fps"] = TARGET_VIDEO_FPS
            if normalized_frame_count is not None:
                record["video_num_frames"] = normalized_frame_count

        normalized_records.append(record)

    return normalized_records
