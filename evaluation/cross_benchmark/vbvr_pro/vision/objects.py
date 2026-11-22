from evaluation.cross_benchmark.vbvr_pro.vision.masks import _background, _chroma_score, _paint, mask_coverage
from evaluation.cross_benchmark.vbvr_pro.vision.shapes import normalized
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Object:
    color: tuple[int, int, int]  # RGB identity from the trusted task palette
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    mask: np.ndarray  # local, including holes and outline thickness
    silhouette: np.ndarray  # local exterior contour filled
    pixels: int
    observed_rgb: tuple[float, float, float]


def palette_objects(frame: np.ndarray, palette, minimum: int = 24) -> list[Object]:
    """Segment by nearest palette entry, retaining holes in each component.

    Nearest assignment prevents a pixel from being counted twice for similar
    colours. The colour radius is deliberately smaller than slot-frame versus
    grey-symbol separation in the benchmark.
    """
    colors = list(dict.fromkeys(tuple(int(v) for v in c) for c in palette))
    if not colors:
        return []
    pixels = frame.astype(np.float32)
    background = _background(frame)
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    scores = []
    direct_scores = []
    for color in colors:
        bgr = _paint(color)
        direct = np.sum((pixels - bgr) ** 2, axis=2)
        direct_scores.append(direct)
        if max(color) - min(color) > 25:
            chroma_score = _chroma_score(ycc, bgr, background)
            # A loose RGB radius around a pale colour also contains neutral
            # outline fringes. Require hue evidence unless the pixel is an
            # actual close match to the trusted opaque paint observation.
            direct = np.where((chroma_score >= 1e8) & (direct > 12**2), 1e8, direct)
            direct = np.minimum(direct, chroma_score)
        scores.append(direct)
    distance = np.stack(scores)
    direct_distance = np.stack(direct_scores)
    # Opaque paint cores disambiguate nearby palette entries. A weaker paint
    # must not steal the compressed core of a similar, lighter object merely
    # because its chroma ray happens to pass through that colour.
    opaque = direct_distance.min(axis=0) <= 12**2
    distance[:, opaque] = direct_distance[:, opaque]
    label = distance.argmin(axis=0)
    closest = distance.min(axis=0)
    # Chroma from a nearby, similar paint can form a false-colour shell and
    # connect two otherwise distinct objects. Attribute only the narrow edge
    # band to its nearest opaque paint core; unseeded thin strokes retain their
    # independent hue observation.
    groups = []
    unseen = set(range(len(colors)))
    while unseen:
        group = {unseen.pop()}
        changed = True
        while changed:
            changed = False
            for index in list(unseen):
                if any(np.linalg.norm(np.subtract(colors[index], colors[j])) < 90 for j in group):
                    unseen.remove(index)
                    group.add(index)
                    changed = True
        if len(group) > 1:
            groups.append(sorted(group))
    direct_label = direct_distance.argmin(axis=0)
    for group in groups:
        fields = []
        for index in group:
            core = opaque & (direct_label == index)
            fields.append(cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3))
        fields = np.stack(fields)
        nearest = fields.argmin(axis=0)
        band = fields.min(axis=0) <= 3 * max(1, frame.shape[1] / 512)
        eligible = band & np.isin(label, group) & ~opaque
        label[eligible] = np.asarray(group)[nearest[eligible]]
    closest[np.linalg.norm(pixels - background, axis=2) < 8] = 1e8
    result = []
    for index, rgb in enumerate(colors):
        binary = ((label == index) & (closest <= 32**2)).astype(np.uint8)
        # Luma has full spatial bandwidth in 4:2:0. Read the physical contour
        # at half of each component's observed paint coverage, independently
        # of the broader chroma footprint used to establish its identity.
        nominal = cv2.cvtColor(np.uint8([[_paint(rgb)]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(float)
        bg_ycc = cv2.cvtColor(np.uint8([[np.round(background)]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(float)
        contrast_y = nominal[0] - bg_ycc[0]
        contrast_c = nominal[1:] - bg_ycc[1:]
        coverage = None
        if abs(contrast_y) >= 8:
            coverage = (ycc[:, :, 0] - bg_ycc[0]) / contrast_y
        elif float(contrast_c @ contrast_c) >= 64:
            # Pale yellow can be isoluminant with white; its half-coverage
            # chroma contour remains observable at the encoded resolution.
            coverage = np.sum((ycc[:, :, 1:] - bg_ycc[1:]) * contrast_c, axis=2) / float(contrast_c @ contrast_c)
        if coverage is not None:
            count, parts, boxes, _ = cv2.connectedComponentsWithStats(binary, 8)
            for part in range(1, count):
                x, y, w, h, area = boxes[part]
                if area < minimum:
                    continue
                local = parts[y : y + h, x : x + w] == part
                luma = coverage[y : y + h, x : x + w]
                peak = float(np.quantile(luma[local], 0.90))
                remove = luma < max(0.08, peak / 2)
                if float(contrast_c @ contrast_c) >= 64:
                    chroma_coverage = np.sum((ycc[y : y + h, x : x + w, 1:] - bg_ycc[1:]) * contrast_c, axis=2) / float(
                        contrast_c @ contrast_c
                    )
                    paint_peak = float(np.quantile(chroma_coverage[local], 0.90))
                    remove |= chroma_coverage < max(0.08, paint_peak / 2)
                binary[y : y + h, x : x + w][local & remove] = 0
        # A close repairs video-codec pinholes without filling intentional holes.
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for k in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[k])
            if area < minimum or min(w, h) < 3:
                continue
            local = (labels[y : y + h, x : x + w] == k).astype(np.uint8)
            contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            silhouette = np.zeros_like(local)
            cv2.drawContours(silhouette, contours, -1, 1, cv2.FILLED)
            # Bounding-box centres are invariant to asymmetric fill / outlines.
            painted = local.astype(bool)
            if coverage is not None:
                strength = coverage[y : y + h, x : x + w]
                painted &= strength >= np.quantile(strength[painted], 0.75)
            observed = tuple(float(v) for v in np.median(frame[y : y + h, x : x + w][painted], axis=0)[::-1])
            result.append(
                Object(
                    rgb,
                    (x + (w - 1) / 2, y + (h - 1) / 2),
                    (x, y, w, h),
                    local,
                    silhouette,
                    area,
                    observed,
                )
            )
    return result


def merge_objects(objects: list[Object]) -> Object | None:
    """One semantic glyph may consist of many disconnected dashes or dots."""
    if not objects:
        return None
    x = min(o.bbox[0] for o in objects)
    y = min(o.bbox[1] for o in objects)
    right = max(o.bbox[0] + o.bbox[2] for o in objects)
    bottom = max(o.bbox[1] + o.bbox[3] for o in objects)
    w, h = right - x, bottom - y
    mask = np.zeros((h, w), np.uint8)
    for o in objects:
        ox, oy, ow, oh = o.bbox
        mask[oy - y : oy - y + oh, ox - x : ox - x + ow] |= o.mask
    ys, xs = np.nonzero(mask)
    silhouette = np.zeros_like(mask)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    cv2.drawContours(silhouette, [hull], -1, 1, cv2.FILLED)
    weights = np.array([o.pixels for o in objects], dtype=float)
    observed = tuple(np.average([o.observed_rgb for o in objects], axis=0, weights=weights))
    color = max(objects, key=lambda o: o.pixels).color
    return Object(
        color,
        (x + (w - 1) / 2, y + (h - 1) / 2),
        (x, y, w, h),
        mask,
        silhouette,
        int(mask.sum()),
        observed,
    )


def cell_objects(frame: np.ndarray, palette, cells) -> list[Object]:
    parts = palette_objects(frame, palette, minimum=4)
    result = []
    for x0, y0, x1, y1 in cells:
        merged = merge_objects([o for o in parts if x0 <= o.center[0] < x1 and y0 <= o.center[1] < y1])
        if merged is not None:
            result.append(merged)
    return result


def object_checks(actual: Object, expected: Object, position: bool = True) -> dict[str, bool]:
    aw, ah = actual.bbox[2:]
    ew, eh = expected.bbox[2:]
    shape_p, shape_r = mask_coverage(normalized(actual.silhouette), normalized(expected.silhouette))
    stroke_p, stroke_r = mask_coverage(normalized(actual.mask), normalized(expected.mask))
    fill_a = actual.pixels / max(1, np.count_nonzero(actual.silhouette))
    fill_e = expected.pixels / max(1, np.count_nonzero(expected.silhouette))
    color_matches = bool(np.linalg.norm(np.subtract(actual.observed_rgb, expected.observed_rgb)) <= 65)
    if not color_matches and fill_e < 0.25 and max(expected.color) - min(expected.color) > 25:
        # A thin hollow stroke has no opaque 512px core. VAE reconstruction
        # mixes its paint with the canvas, so RGB magnitude is not its color
        # identity. Its labelled pigment, hue and substantial chroma remain
        # observable; shape and hollow topology are checked independently.
        observed, nominal = cv2.cvtColor(
            np.uint8([[actual.observed_rgb[::-1], expected.observed_rgb[::-1]]]), cv2.COLOR_BGR2HSV
        )[0].astype(float)
        hue = abs(observed[0] - nominal[0])
        color_matches = bool(min(hue, 180 - hue) <= 6 and observed[1] >= max(25, nominal[1] * 0.5))
    checks = {
        "color": actual.color == expected.color and color_matches,
        "shape": min(shape_p, shape_r) >= 0.96,
        "size": max(abs(aw / ew - 1), abs(ah / eh - 1)) <= 0.09,
        "fill_and_stroke": min(stroke_p, stroke_r) >= 0.92 and abs(fill_a - fill_e) <= max(0.035, fill_e * 0.22),
    }
    if position:
        checks["position"] = bool(np.linalg.norm(np.subtract(actual.center, expected.center)) <= 12)
    return checks


def assign_objects(actual: list[Object], expected: list[Object], position: bool = True):
    if len(actual) != len(expected) or not expected:
        return []
    cost = np.empty((len(expected), len(actual)), dtype=float)
    for i, e in enumerate(expected):
        for j, a in enumerate(actual):
            delta = np.linalg.norm(np.subtract(a.center, e.center)) / 100 if position else 0
            size = abs(np.log(max(1, a.pixels) / max(1, e.pixels)))
            color = 20 * (a.color != e.color)
            sp, sr = mask_coverage(normalized(a.silhouette), normalized(e.silhouette))
            cost[i, j] = delta + size + color + 5 * (2 - sp - sr)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols)]
