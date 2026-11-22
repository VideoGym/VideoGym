from contextvars import ContextVar
from contextlib import contextmanager
import cv2
import numpy as np


@contextmanager
def canvas_background(bgr, paint_observations=None):
    """Bind fixed trusted canvas and paint observations for one verification."""
    token = _canvas_background.set(np.asarray(bgr, dtype=np.float32))
    paint_token = _paint_observations.set(paint_observations or {})
    try:
        yield
    finally:
        _canvas_background.reset(token)
        _paint_observations.reset(paint_token)


def _paint(rgb):
    return np.asarray(_paint_observations.get().get(tuple(rgb), rgb)[::-1], dtype=np.float32)


def _background(frame):
    declared = _canvas_background.get()
    if declared is not None:
        return declared
    return np.median(np.concatenate([frame[:8].reshape(-1, 3), frame[-8:].reshape(-1, 3)]), axis=0).astype(np.float32)


def _chroma_score(ycc, bgr, background):
    """Allow independent luma and chroma coverage, as in 4:2:0 sampling.

    RGB alpha blending assumes both have the same pixel support. Subsampled
    chroma violates that assumption at thin strokes and object boundaries.
    Chroma direction still identifies the paint; luma rejects coloured halos
    on unchanged background and neutral annotation fringes.
    """
    color = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(np.float32)
    bg = cv2.cvtColor(np.uint8([[np.round(background)]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(np.float32)
    vector = color[1:] - bg[1:]
    alpha_c = np.sum((ycc[:, :, 1:] - bg[1:]) * vector, axis=2) / max(1, float(vector @ vector))
    residual = np.sum((ycc[:, :, 1:] - (bg[1:] + alpha_c[..., None] * vector)) ** 2, axis=2)
    chroma = np.sum((ycc[:, :, 1:] - bg[1:]) ** 2, axis=2)
    delta = float(color[0] - bg[0])
    alpha_y = (ycc[:, :, 0] - bg[0]) / delta if abs(delta) >= 8 else np.ones(ycc.shape[:2], np.float32)
    score = 4 * residual + 500 * (1 - alpha_c) ** 2 + 500 * (1 - alpha_y) ** 2
    score[
        (alpha_c < 0.30)
        | (alpha_c > 1.3)
        | (alpha_y < 0.12)
        | (alpha_y > 1.3)
        | (chroma < 16)
        | (residual > np.maximum(4, chroma * 0.10))
    ] = 1e8
    return score


def color_mask(frame: np.ndarray, rgb, tolerance: float = 36, *, opaque: bool = False) -> np.ndarray:
    bgr = _paint(rgb)
    score = np.sum((frame.astype(np.float32) - bgr) ** 2, axis=2)
    if not opaque and max(rgb) - min(rgb) > 25:
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        chroma_score = _chroma_score(ycc, bgr, _background(frame))
        score[(chroma_score >= 1e8) & (score > 12**2)] = 1e8
        score = np.minimum(score, chroma_score)
    return (score <= tolerance**2).astype(np.uint8)


def mask_coverage(a: np.ndarray, b: np.ndarray, radius: int = 2) -> tuple[float, float]:
    """Bidirectional foreground coverage, with a small boundary tolerance."""
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    aa, bb = a.astype(np.uint8), b.astype(np.uint8)
    if not np.any(aa) and not np.any(bb):
        return 1.0, 1.0
    ad = cv2.dilate(aa, kernel)
    bd = cv2.dilate(bb, kernel)
    precision = float(np.count_nonzero(aa & bd) / max(1, np.count_nonzero(aa)))
    recall = float(np.count_nonzero(bb & ad) / max(1, np.count_nonzero(bb)))
    return precision, recall


def ink(frame: np.ndarray, background=None) -> np.ndarray:
    if background is None:
        background = _background(frame)
    return (np.linalg.norm(frame.astype(float) - np.asarray(background), axis=2) > 35).astype(np.uint8)


def no_unexpected_ink(actual: np.ndarray, expected: np.ndarray) -> tuple[bool, dict]:
    """A veto on added / erased scene content, not an answer predicate.

    Foreground coverage detects erasure. A separate local connected-component
    veto prevents large legitimate foreground from hiding a small added patch.
    The independent object checks decide the answer.
    """
    a, e = ink(actual), ink(expected)
    p, r = mask_coverage(a, e, radius=3)
    local, detail = no_added_components(a, e)
    return bool(min(p, r) >= 0.965 and local), {"foreground_precision": p, "foreground_recall": r, **detail}


def no_added_components(actual: np.ndarray, allowed: np.ndarray, radius: int = 3) -> tuple[bool, dict]:
    """Veto local added ink without dividing by the amount of scene content.

    ``allowed`` is the current semantic footprint, including fixed foreground
    and the objects at their observed legal poses. Callers must not substitute
    a swept path or the whole canvas. Boundary tolerance handles antialiasing
    and codec fringes; a separate connected patch of 16 pixels is observable
    even in a scene containing hundreds of thousands of legitimate ink pixels.
    """
    nearby = cv2.dilate(allowed.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    added = actual.astype(np.uint8) & (1 - nearby)
    _, _, stats, _ = cv2.connectedComponentsWithStats(added, 8)
    extras = [list(map(int, stat)) for stat in stats[1:] if stat[cv2.CC_STAT_AREA] >= 16]
    return not extras, {"added_component_boxes_xywh_area": extras}


def anchored_foreground(actual: np.ndarray, anchors: np.ndarray, radius: int = 3) -> np.ndarray:
    """Read complete visible components attached to recognized object seeds.

    An occluder can split one palette object into several fragments. Connecting
    through the observed foreground keeps those fragments with their object,
    without admitting a separate new component elsewhere in the background.
    """
    foreground = cv2.morphologyEx(actual.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(foreground, 8)
    near = cv2.dilate(anchors.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    selected = np.unique(labels[near.astype(bool)])
    selected = selected[selected != 0]
    return np.isin(labels, selected).astype(np.uint8)


_canvas_background = ContextVar("scene_canvas_background", default=None)


_paint_observations = ContextVar("scene_paint_observations", default={})
