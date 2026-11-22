import cv2
import numpy as np


def normalized(mask: np.ndarray, side: int = 96) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST)


def fit_ring(mask: np.ndarray, radius: float):
    """Fit a known-radius ring in clutter, including clipping and other rings."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    rng = np.random.default_rng(0)
    ids = rng.choice(len(xs), min(1800, len(xs)), replace=False)
    points = np.c_[xs[ids], ys[ids]].astype(float)
    best = None
    best_count = 0
    for _ in range(120):
        a, b = points[rng.choice(len(points), 2, replace=False)]
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < radius * 0.25 or length > radius * 2:
            continue
        middle = (a + b) / 2
        offset = np.sqrt(max(0, radius * radius - length * length / 4))
        normal = np.array([-delta[1], delta[0]]) / length
        for center in (middle + normal * offset, middle - normal * offset):
            residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
            count = int(np.count_nonzero(residual <= 5))
            if count > best_count:
                best_count = count
                best = center
    if best is None or best_count < 50:
        return None
    selected = points[np.abs(np.linalg.norm(points - best, axis=1) - radius) <= 6]
    x, y = selected.T
    cx, cy, c = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones(len(x))], x * x + y * y, rcond=None)[0]
    fitted = float(np.sqrt(max(0, c + cx * cx + cy * cy)))
    return (float(cx), float(cy)), fitted, best_count / len(points)
