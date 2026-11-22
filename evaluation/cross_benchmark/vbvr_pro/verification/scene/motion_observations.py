import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.vision.masks import _paint, color_mask, mask_coverage, no_added_components


def _mask(frame, color):
    mask = color_mask(frame, color)
    mask[np.linalg.norm(frame.astype(float) - 255, axis=2) < 8] = 0
    return mask


def _coverage(frame, expected, colors, regions=None, observed=None):
    checks = {}
    if regions is None:
        for index, color in enumerate(colors):
            a = observed[0][index] if observed is not None else _mask(frame, color)
            e = _mask(expected, color)
            checks[f"object_{index}.visible_rigid_shape_and_position"] = min(mask_coverage(a, e, radius=3)) >= 0.97
    af = (
        observed[1]
        if observed is not None
        else (np.linalg.norm(frame.astype(float) - 255, axis=2) > 35).astype(np.uint8)
    )
    ef = (np.linalg.norm(expected.astype(float) - 255, axis=2) > 35).astype(np.uint8)
    checks["other_scene_content_preserved"] = min(mask_coverage(af, ef, radius=3)) >= 0.98
    checks["no_isolated_added_content"] = no_added_components(af, ef)[0]
    return checks


def _overlay_paint(frame, underlay, color):
    """Read a painted line against its trusted local underlay, including overlap."""
    observed = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    behind = cv2.cvtColor(underlay, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    paint = cv2.cvtColor(np.uint8([[_paint(color)]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(np.float32)
    direction = paint[1:] - behind[:, :, 1:]
    denominator = np.sum(direction**2, axis=2)
    alpha_c = np.sum((observed[:, :, 1:] - behind[:, :, 1:]) * direction, axis=2) / np.maximum(1, denominator)
    residual = np.sum((observed[:, :, 1:] - behind[:, :, 1:] - alpha_c[..., None] * direction) ** 2, axis=2)
    contrast = paint[0] - behind[:, :, 0]
    alpha_y = np.ones(contrast.shape, np.float32)
    visible_y = np.abs(contrast) >= 20
    alpha_y[visible_y] = (observed[:, :, 0] - behind[:, :, 0])[visible_y] / contrast[visible_y]
    delta_c = np.sum((observed[:, :, 1:] - behind[:, :, 1:]) ** 2, axis=2)
    return (
        (denominator >= 36)
        & (alpha_c >= 0.4)
        & (alpha_c <= 1.35)
        & (alpha_y >= 0.4)
        & (alpha_y <= 1.35)
        & (residual <= np.maximum(16, delta_c * 0.15))
    ).astype(np.uint8)


def _stationary_support(frame, paint, outline):
    """Separate the declared actor paints from white, including pale fills.

    A fixed distance from white also labels low-opacity chroma ringing from
    the moving green box as actor geometry. Assign each pixel to its nearest
    actual paint or the background instead; the RGB identity check remains
    independent of this foreground support.
    """
    pixels = frame.astype(np.float32)
    background = np.sum((pixels - 255) ** 2, axis=2)
    foreground = np.minimum(
        np.sum((pixels - _paint(paint)) ** 2, axis=2),
        np.sum((pixels - _paint(outline)) ** 2, axis=2),
    )
    return (foreground < background).astype(np.uint8)
