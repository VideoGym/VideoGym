from evaluation.cross_benchmark.vbvr_pro.verification.scene.analogy import _analogy
from evaluation.cross_benchmark.vbvr_pro.verification.scene.motion import _motion
from evaluation.cross_benchmark.vbvr_pro.verification.scene.sorting import _sorting
from evaluation.cross_benchmark.vbvr_pro.verification.scene.symbols import _symbols
from evaluation.cross_benchmark.vbvr_pro.verification.scene.symmetry import _concentric, _symmetry
from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result, _rgb_values
import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.verification.scene.task_dispatch import SUPPORTED_IDS as EXTENSION_IDS
from evaluation.cross_benchmark.vbvr_pro.vision.bandwidth import bind as bind_observations, original_image
from evaluation.cross_benchmark.vbvr_pro.vision.masks import anchored_foreground, canvas_background, color_mask, ink, no_added_components

SORT_TASKS = {"G-3", "G-174"}

ANALOGY_TASKS = {"O-9", "O-10", "O-11", "O-12", "O-13", "O-14"}

SYMBOL_TASKS = {"O-5", "O-58", "O-59", "O-60", "O-61"}

MOTION_TASKS = {"G-24", "G-25", "O-6", "O-85"}

SUPPORTED_IDS = EXTENSION_IDS | SORT_TASKS | ANALOGY_TASKS | SYMBOL_TASKS | MOTION_TASKS | {"G-194", "O-49"}


def _verify(
    task: str,
    frames: list[np.ndarray],
    first: np.ndarray,
    final: np.ndarray,
    metadata: dict,
    reference: list[np.ndarray],
) -> dict | None:
    """Return None for an unsupported family, otherwise a conjunctive verdict."""
    task_id = task.split("_", 1)[0]
    if task_id not in SUPPORTED_IDS:
        return None
    if not frames:
        return _result({"video_frames_available": False})
    # Use the independently decoded reference endpoint as the contour template
    # when available: thin strokes can lose chroma in the benchmark's own MP4.
    # The lossless final image remains the scene-content and metadata anchor.
    template = reference[-1] if reference and reference[-1].shape == final.shape else final
    source = reference[0] if reference and reference[0].shape == first.shape else first
    if task_id in EXTENSION_IDS:
        from evaluation.cross_benchmark.vbvr_pro.verification.scene.task_dispatch import verify as verify_extended

        return verify_extended(task_id, frames, source, template, metadata, reference)
    if task_id in SORT_TASKS:
        return _sorting(task_id, frames, source, template, metadata)
    if task_id in ANALOGY_TASKS:
        return _analogy(frames, source, template, metadata, reference, color_then_move=task_id == "O-11")
    if task_id in SYMBOL_TASKS:
        return _symbols(frames, source, template, metadata)
    if task_id == "O-49":
        return _symmetry(frames, final, metadata)
    if task_id == "G-194":
        return _concentric(frames, final, metadata)
    return _motion(task_id, frames, template, metadata, reference)


def _verify_observations(task, frames, first, final, metadata, reference):
    """Check the semantic answer and every frame's legal foreground support."""
    result = _verify(task, frames, first, final, metadata, reference)
    diagrams = ANALOGY_TASKS | SYMBOL_TASKS | {"O-49", "G-51", "G-193", "O-45", "O-56", "O-43"}
    if result is None or not frames or task.split("_", 1)[0] not in diagrams:
        return result
    source = reference[0] if reference and reference[0].shape == first.shape else first
    goal = reference[-1] if reference and reference[-1].shape == final.shape else final
    # Track glyph transformations from their given/answer locations. Components
    # can move between frames; their route need not match the reference video.
    scale = min(1, 512 / source.shape[1])
    size = (round(source.shape[1] * scale), round(source.shape[0] * scale))
    if scale < 1:
        source, goal = (cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in (source, goal))
    fixed = ink(source) | ink(goal)
    params = metadata.get("parameters", {})
    drop = round(params.get("drop_height_px", 0) * scale)
    target_color = params.get("color_map", {}).get(params.get("target_symbol"))
    if task.split("_", 1)[0] in SYMBOL_TASKS and drop > 0 and target_color is not None:
        # Inserted symbols can first become visible at the declared drop
        # origin, disconnected from existing ink. Register the glyph's birth
        # footprint, then track its observed motion instead of a GT path.
        # A one-pixel insertion stroke is still part of the declared glyph.
        # Complete only the observed outline attached to its paint seeds.
        lossless_goal = original_image(final)
        birth = color_mask(lossless_goal, target_color)
        birth[: round(lossless_goal.shape[0] * 0.3)] = 0
        birth |= anchored_foreground(ink(lossless_goal), birth, radius=3)
        if scale < 1:
            birth = (cv2.resize(birth.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        fixed[:-drop] |= birth[drop:]
    previous = fixed
    failures = []
    for index, frame in enumerate(frames):
        if scale < 1:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        foreground = ink(frame)
        allowed = anchored_foreground(foreground, fixed | previous)
        passed, detail = no_added_components(foreground, allowed)
        previous = allowed
        result["checks"][f"frame_{index}.no_isolated_added_content"] = passed
        if not passed:
            failures.append({"frame": index, **detail})
    if failures:
        result["evidence"]["isolated_added_content"] = failures
    return _result(result["checks"], result["evidence"])


def verify(task, frames, first, final, metadata, reference):
    declared = metadata.get("generic_declarative_render", {}).get("canvas", {}).get("background_rgb")
    source = reference[0] if reference else first
    samples = np.concatenate(
        [
            source[:8].reshape(-1, 3),
            source[-8:].reshape(-1, 3),
            source[:, :8].reshape(-1, 3),
            source[:, -8:].reshape(-1, 3),
        ]
    ).astype(float)
    if declared is None:
        background = np.median(samples, axis=0)
    else:
        background = np.array(declared[::-1])
        distance = np.sum((samples - background) ** 2, axis=1)
        canvas = distance <= 24**2
        for color in dict.fromkeys(_rgb_values(metadata)):
            bgr = np.array(color[::-1])
            if np.linalg.norm(bgr - background) > 1:
                canvas &= distance < np.sum((samples - bgr) ** 2, axis=1)
        # A fixed observation of the declared canvas includes the reference
        # decoder's colour transfer. Moving objects never redefine it later.
        if np.count_nonzero(canvas) >= 16:
            background = np.median(samples[canvas], axis=0)
    # The trusted decoder already shifts nominal RGB values. Calibrate paint
    # identities from opaque interior pixels of the trusted first/final images;
    # nearby paints must remain distinct after another H.264 colour transfer.
    palette = list(dict.fromkeys(_rgb_values(metadata)))
    paints = {}
    observed = {color: [] for color in palette}
    for image in (source, final):
        small = cv2.resize(image, (256, round(image.shape[0] * 256 / image.shape[1])), interpolation=cv2.INTER_AREA)
        pixels = small.reshape(-1, 3).astype(float)
        colors = np.array([color[::-1] for color in palette]) if palette else np.empty((0, 3))
        if not len(colors):
            continue
        distances = np.sum((pixels[:, None, :] - colors[None, :, :]) ** 2, axis=2)
        labels = distances.argmin(axis=1)
        nearest = distances.min(axis=1)
        foreground = nearest < np.sum((pixels - background) ** 2, axis=1)
        for index, color in enumerate(palette):
            # The metadata palette need not list the canvas colour. Without
            # this background competitor, an almost-white actor can inherit
            # millions of white canvas pixels as its supposedly opaque paint.
            seeds = ((labels == index) & foreground & (nearest <= 24**2)).reshape(small.shape[:2]).astype(np.uint8)
            seeds = cv2.erode(seeds, np.ones((3, 3), np.uint8)).astype(bool)
            if np.count_nonzero(seeds) >= 8:
                observed[color].append(small[seeds])
    for color, values in observed.items():
        if values:
            paints[color] = tuple(np.median(np.concatenate(values), axis=0)[::-1])
    with canvas_background(background, paints), bind_observations(first, final, reference):
        return _verify_observations(task, frames, first, final, metadata, reference)
