from __future__ import annotations
import cv2
import numpy as np


def _decision(family, checks, **evidence):
    if any(type(v) not in (bool, np.bool_) for v in checks.values()):
        raise TypeError("Physics predicates must be explicit booleans")
    checks = {k: bool(v) for k, v in checks.items()}
    return {"passed": all(checks.values()), "checks": checks, "evidence": {"family": family, **evidence}}


def _mask(frame, rgb, tolerance=35):
    return (np.max(np.abs(frame.astype(np.float32) - np.asarray(rgb[::-1])), axis=2) <= tolerance).astype(np.uint8)


def _pigment(frame, rgb, backgrounds=((255, 255, 255),), minimum_alpha=0.5):
    """Observe palette coverage, including antialiased and YUV-mixed edges.

    A pixel must lie near the foreground/background colour segment. Merely
    being dark, saturated or close to the object's screen position is not
    sufficient to become part of its geometry.
    """
    pixels = frame.astype(np.float32)
    foreground = np.asarray(rgb[::-1], np.float32)
    result = np.zeros(frame.shape[:2], bool)
    for rgb_background in backgrounds:
        background = np.asarray(rgb_background[::-1], np.float32)
        vector = foreground - background
        if float(vector @ vector) < 1:
            continue
        alpha = np.clip(np.sum((pixels - background) * vector, axis=2) / (vector @ vector), 0, 1)
        residual = np.max(np.abs(pixels - (background + alpha[..., None] * vector)), axis=2)
        chroma = vector - vector.mean()
        colour_present = np.ones(frame.shape[:2], bool)
        if float(chroma @ chroma) >= 64:
            # A pale grey pixel must not count as a faint coloured line just
            # because its small absolute RGB residual fits a soft boundary.
            chroma_alpha = np.sum((pixels - background) * chroma, axis=2) / (chroma @ chroma)
            colour_present = chroma_alpha >= minimum_alpha * 0.5
        result |= (alpha >= minimum_alpha) & (residual <= 28) & colour_present
    return result.astype(np.uint8)


def _chroma_ink(frame, rgb, minimum_alpha=0.12):
    """Read thin coloured ink when the codec separates luma from chroma.

    A dark shaft or glyph can keep its local luma while its colour is shared
    with neighbouring white pixels. Its hue contribution survives, although
    the resulting RGB value is not a white/foreground alpha mixture.
    """
    pixels = frame.astype(np.float32)
    chroma = pixels - pixels.mean(axis=2, keepdims=True)
    target = np.asarray(rgb[::-1], np.float32)
    target = target - target.mean()
    energy = float(target @ target)
    if energy < 64:
        raise ValueError("Chroma ink requires a coloured foreground")
    alpha = np.sum(chroma * target, axis=2) / energy
    residual = np.max(np.abs(chroma - alpha[..., None] * target), axis=2)
    return ((alpha >= minimum_alpha) & (residual <= 20)).astype(np.uint8)


def _ink_on_background(frame, rgb, background_luma):
    """Locate visible ink without extending its extent into chroma ringing."""
    observed_luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    ink_luma = float(np.asarray(rgb) @ np.array([0.299, 0.587, 0.114]))
    contrast = ink_luma - background_luma
    coverage = np.divide(
        observed_luma - background_luma, contrast, out=np.ones_like(observed_luma), where=abs(contrast) >= 8
    )
    return _chroma_ink(frame, rgb) & (coverage >= 0.25)


def _circles(frame, rgb, minimum=16):
    mask = _mask(frame, rgb)
    # A close may bridge a tiny near-colour trail fragment to a filled ball.
    # Remove components already too small to be a ball before repairing its
    # own narrow observation gaps.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= minimum
    keep[0] = False
    mask = keep[labels].astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < minimum:
            continue
        # A filled disk must occupy its own extent; prey outlines and text do not.
        fill = area / max(1, w * h)
        if 0.65 <= w / max(1, h) <= 1.55 and 0.61 <= fill <= 0.91:
            result.append(
                {
                    "center": centers[k],
                    "radius": np.sqrt(area / np.pi),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": int(area),
                }
            )
    return result


def _preserved(frame, first, allowed, vacated=None):
    """Local veto on corrupted stationary content, with no correctness credit."""
    changed = (np.max(np.abs(frame.astype(np.int16) - first.astype(np.int16)), axis=2) > 65) & ~allowed
    if vacated is not None:
        # A source-position ghost can straddle four fixed tiles while being
        # one clear foreign feature. Inspect that reconstructed region as a
        # connected feature, without changing the global scene tolerance.
        localized = (changed & vacated).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(localized, 8)
        if len(stats) > 1 and int(np.max(stats[1:, cv2.CC_STAT_AREA])) > 30:
            return False
    height, width = first.shape[:2]
    if height % 16 == 0 and width % 16 == 0:
        shape = (height // 16, 16, width // 16, 16)
        valid_counts = np.count_nonzero((~allowed).reshape(shape), axis=(1, 3))
        changed_counts = np.count_nonzero(changed.reshape(shape), axis=(1, 3))
        return bool(not np.any((valid_counts >= 20) & (changed_counts > 0.20 * valid_counts)))
    for y in range(0, first.shape[0], 16):
        for x in range(0, first.shape[1], 16):
            valid = ~allowed[y : y + 16, x : x + 16]
            if np.count_nonzero(valid) >= 20 and np.count_nonzero(
                changed[y : y + 16, x : x + 16]
            ) > 0.20 * np.count_nonzero(valid):
                return False
    return True


def _disk_allowed(shape, centers, radius):
    mask = np.zeros(shape[:2], np.uint8)
    for center in centers:
        cv2.circle(mask, tuple(np.rint(center).astype(int)), int(np.ceil(radius)), 1, -1)
    return mask.astype(bool)


def _vacated(first, region, metadata, final=None, final_body=None):
    """Reconstruct exposed canvas without permanently exempting source bodies."""
    empty = first.copy()
    rgb = metadata.get("generic_declarative_render", {}).get("canvas", {}).get("background_rgb", [255, 255, 255])
    empty[region] = rgb[::-1]
    if final is not None:
        known = region if final_body is None else region & ~final_body
        empty[known] = final[known]
    return empty


def _polyline(initial, velocity, bounds, radius, bounces):
    point = np.asarray(initial, float)
    speed = np.asarray(velocity, float)
    xmin, xmax, ymin, ymax = np.asarray(bounds, float) + np.array([radius, -radius, radius, -radius])
    if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax) or not np.linalg.norm(speed):
        raise ValueError("Invalid initial billiard state")
    points = [point.copy()]
    for _ in range(bounces):
        tx = ((xmax if speed[0] > 0 else xmin) - point[0]) / speed[0] if abs(speed[0]) > 1e-10 else np.inf
        ty = ((ymax if speed[1] > 0 else ymin) - point[1]) / speed[1] if abs(speed[1]) > 1e-10 else np.inf
        t = min(tx, ty)
        if t <= 0:
            raise ValueError("Nonpositive billiard collision time")
        point = point + speed * t
        points.append(point.copy())
        if np.isclose(tx, t):
            speed[0] *= -1
        if np.isclose(ty, t):
            speed[1] *= -1
    return np.asarray(points)
