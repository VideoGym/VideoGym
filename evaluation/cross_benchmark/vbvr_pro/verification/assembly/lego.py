"""Rigid LEGO part transfer with explicit occlusion and local stud/face checks.

The reference supplies the part's projected silhouette and visible stud templates.
It does not prescribe a trajectory: every independently observed pose may follow a
legal different route from the callout to the declared target. Static callout text
and existing bricks may occlude a pose only where their actual pixels remain.
"""

import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _blobs, _layers, _result
from evaluation.cross_benchmark.vbvr_pro.vision.masks import mask_coverage

# A coloured stud rim may have little grayscale contrast, especially after
# 4:2:0 sampling. Use the strongest colour-channel gradient for its geometry.
EDGE_THRESHOLDS = (12, 24)
FACE_EDGE_THRESHOLDS = (25, 50)


def _preserved(frame, background, allowed):
    a = cv2.GaussianBlur(frame, (3, 3), 0).astype(np.int16)
    b = cv2.GaussianBlur(background, (3, 3), 0).astype(np.int16)
    # Pale old LEGO faces need lower contrast tolerance than saturated shapes;
    # a large grey face must not be erased just because it is close to white.
    actual_ycc = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2YCrCb).astype(np.int16)
    expected_ycc = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2YCrCb).astype(np.int16)
    gradient = cv2.morphologyEx(background, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    old_edge = cv2.dilate((np.max(gradient, axis=2) > 25).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    # 4:2:0 can move colour across an existing edge while retaining luminance.
    # Interior faces and empty background still require their original colour.
    changed = (
        (np.abs(actual_ycc[:, :, 0] - expected_ycc[:, :, 0]) > 25)
        | ((np.max(np.abs(actual_ycc[:, :, 1:] - expected_ycc[:, :, 1:]), axis=2) > 25) & ~old_edge)
    ) & ~allowed
    areas = [blob[1] for blob in _blobs(changed, 1)]
    worst = max(areas, default=0)
    return worst <= 40, worst


def _hue_mask(frame, hue):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    delta = np.abs(hsv[:, :, 0].astype(int) - hue)
    return cv2.morphologyEx(
        ((np.minimum(delta, 180 - delta) <= 9) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 30)).astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )


def _polygon_mask(shape, layers):
    out = np.zeros(shape[:2], np.uint8)
    for layer in layers:
        if layer.get("points_px"):
            cv2.fillPoly(out, [np.round(layer["points_px"]).astype(np.int32)], 1)
    return out


def _definition(first, final, metadata):
    p = metadata["parameters"]
    new = next(obj for obj in p["objects"] if obj["is_new"])
    definition = p["new_brick"]
    if (
        new["brick_type"] != definition["brick_type"]
        or new["color"] != definition["color"]
        or new["position"] != [definition[k] for k in ("x", "y", "z")]
    ):
        raise ValueError("New part labels contradict assembly instruction")
    if any(obj["color"] == new["color"] for obj in p["objects"] if not obj["is_new"]):
        raise ValueError("New part color is not independently identifiable among existing bricks")
    layers = _layers(metadata)
    source_layers = [
        v
        for v in layers
        if v.get("kind") == "polygon"
        and "fill_rgb" in v
        and v.get("points_px")
        and max(q[0] for q in v["points_px"]) < first.shape[1] * 0.4
        and max(v["fill_rgb"]) - min(v["fill_rgb"]) > 40
    ]
    if not source_layers:
        raise ValueError("Callout part polygon geometry unavailable")
    hues = cv2.cvtColor(np.array([[v["fill_rgb"][::-1] for v in source_layers]], np.uint8), cv2.COLOR_BGR2HSV)[0, :, 0]
    hue = int(np.median(hues))
    target_mask = _hue_mask(final, hue)
    blobs = _blobs(target_mask, 50)
    if not blobs:
        raise ValueError("No visible final part observation")
    _, _, box = max(blobs, key=lambda v: v[1])
    x, y, w, h = map(int, box)
    template = target_mask[y : y + h, x : x + w].copy()
    target = np.array([x + w / 2, y + h / 2])
    if np.linalg.norm(target - np.array(new["center_2d"])) > max(w, h) * 0.25:
        raise ValueError("Reference target contradicts declared part projection")
    source_geom = _polygon_mask(first.shape, source_layers)
    source_mask = _hue_mask(first, hue) & cv2.dilate(source_geom, np.ones((5, 5), np.uint8))
    yy, xx = np.nonzero(source_mask)
    if len(xx) < 100:
        raise ValueError("Callout part is not observable")
    source_box = (int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1))
    # The clean callout is visible in the final reference, including portions of
    # a large part which initially protruded outside its white instruction box.
    background = first.copy()
    sx, sy, sw, sh = source_box
    background[max(0, sy - 4) : sy + sh + 4, max(0, sx - 4) : sx + sw + 4] = final[
        max(0, sy - 4) : sy + sh + 4, max(0, sx - 4) : sx + sw + 4
    ]
    annotation = np.zeros(first.shape[:2], np.uint8)
    for layer in layers:
        if layer.get("type") == "line" and layer.get("stroke_rgb") in ([255, 0, 0], [255, 255, 255]):
            points = np.round(layer["points_px"]).astype(np.int32)
            cv2.polylines(annotation, [points], False, 1, round(layer.get("stroke_width_px", 3)) + 6)
        elif layer.get("kind") == "polygon" and layer.get("fill_rgb") == [255, 0, 0]:
            cv2.fillPoly(annotation, [np.round(layer["points_px"]).astype(np.int32)], 1)
    # Only explicitly rendered fixed foreground can hide a piece of the moving
    # brick; an arbitrary missing region on the canvas cannot be excused.
    occluders = np.zeros(first.shape[:2], np.uint8)
    for layer in layers:
        if (
            layer.get("kind") == "rect"
            and layer.get("fill_rgb") == [255, 255, 255]
            and layer.get("x_px", first.shape[1]) < first.shape[1] * 0.4
        ):
            a, b, c, d = (round(layer[k]) for k in ("x_px", "y_px", "w_px", "h_px"))
            occluders[b : b + d, a : a + c] = 1
    old_layers = [
        v
        for v in layers
        if v.get("kind") == "polygon"
        and "fill_rgb" in v
        and v not in source_layers
        and v.get("fill_rgb") != [255, 0, 0]
    ]
    occluders |= _polygon_mask(first.shape, old_layers)
    return new, hue, template, (x, y, w, h), target, source_box, background, annotation, occluders


def _observe(frame, hue, template, background, annotation, occluders):
    mask = _hue_mask(frame, hue)
    annotation_color = (np.max(np.abs(frame.astype(np.int16) - np.array([0, 0, 255])), axis=2) < 25) | (
        np.min(frame, axis=2) > 245
    )
    background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    red_reference = ((background_hsv[:, :, 0] < 10) | (background_hsv[:, :, 0] > 170)) & (background_hsv[:, :, 1] > 80)
    annotation_color |= red_reference & (
        np.max(np.abs(frame.astype(np.int16) - background.astype(np.int16)), axis=2) < 45
    )
    actual_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    actual_red = ((actual_hsv[:, :, 0] < 10) | (actual_hsv[:, :, 0] > 170)) & (actual_hsv[:, :, 1] > 40)
    actual_luma = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.int16)
    expected_luma = cv2.cvtColor(background, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.int16)
    annotation_color |= red_reference & actual_red & (np.abs(actual_luma - expected_luma) < 25)
    # A thin white callout stroke over a coloured brick becomes pale pink/blue
    # after 4:2:0. Its high luminance and low saturation remain observable.
    annotation_color |= (actual_luma > 200) & (actual_hsv[:, :, 1] < 80)
    visible_annotation = (annotation > 0) & annotation_color
    mask[visible_annotation] = 0
    # Ignore only isolated codec specks, never a second sizeable part.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    minimum = max(12, int(np.count_nonzero(template) * 0.001))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum:
            clean[labels == label] = 1
    actual = np.count_nonzero(clean)
    h, w = template.shape
    if actual < max(40, np.count_nonzero(template) * 0.20):
        return None
    fixed_pixels = np.max(np.abs(frame.astype(np.int16) - background.astype(np.int16)), axis=2) < 45
    hidden = ((occluders > 0) & fixed_pixels) | visible_annotation
    intersections = cv2.matchTemplate(clean.astype(np.float32), template.astype(np.float32), cv2.TM_CCORR)
    hidden_counts = cv2.matchTemplate(hidden.astype(np.float32), template.astype(np.float32), cv2.TM_CCORR)
    visible = np.count_nonzero(template) - hidden_counts
    cost = 2 * (actual - intersections) + np.maximum(0, visible - intersections)
    y, x = np.unravel_index(int(np.argmin(cost)), cost.shape)
    expected = np.zeros_like(clean)
    expected[y : y + h, x : x + w] = template
    full_hull = np.zeros_like(clean)
    contours, _ = cv2.findContours(expected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(full_hull, contours, -1, 1, -1)
    # Visible mask agreement is measured in the part, not in the white canvas.
    tolerance = np.ones((5, 5), np.uint8)
    actual_near = cv2.dilate(clean, tolerance) > 0
    expected_near = cv2.dilate(expected, tolerance) > 0
    required = (expected > 0) & ~hidden
    recall = np.count_nonzero(required & actual_near) / max(np.count_nonzero(required), 1)
    precision = np.count_nonzero((clean > 0) & expected_near) / max(actual, 1)
    return {
        "center": np.array([x + w / 2, y + h / 2]),
        "box": [int(x), int(y), w, h],
        "recall": float(recall),
        "precision": float(precision),
        "mask": clean,
        "hull": full_hull,
        "visible_fraction": float(np.count_nonzero(required) / max(np.count_nonzero(template), 1)),
        "occluded": hidden,
    }


def _rim_profile(chroma, center, axes, phase, allowed, frame):
    angles = np.arange(64) * (2 * np.pi / 64)
    directions = np.column_stack([np.sin(angles), np.cos(angles)])
    points = directions * [axes[1], axes[0]] + [center[1], center[0]]
    rim = np.round(points[:, None, :] + directions[:, None, :] * np.arange(-phase, phase + 1)[None, :, None]).astype(
        int
    )
    rim = np.clip(rim, [0, 0], np.subtract(chroma.shape, 1))
    rim_values = chroma[rim[:, :, 0], rim[:, :, 1]]
    # Neutral ink reduces every readable opponent-colour component on both
    # sides. A B-only codec ripple can reduce max-minus-min chroma without
    # drawing a grey rim: its G-minus-R component remains flat or increases.
    # Select a colour-consistent valley, rather than first picking the lowest
    # chroma ripple and discarding a real rim at another sampling phase.
    side_points = np.round(
        rim[:, :, None, :] + directions[:, None, None, :] * np.array([-phase - 1, phase + 1])[None, None, :, None]
    ).astype(int)
    side_points = np.clip(side_points, [0, 0], np.subtract(chroma.shape, 1))
    rim_bgr = frame[rim[:, :, 0], rim[:, :, 1]].astype(float)
    body_bgr = frame[side_points[:, :, :, 0], side_points[:, :, :, 1]].astype(float)
    pairs_a, pairs_b = [0, 0, 1], [1, 2, 2]
    body_opponents = body_bgr[:, :, :, pairs_a] - body_bgr[:, :, :, pairs_b]
    rim_opponents = rim_bgr[:, :, pairs_a] - rim_bgr[:, :, pairs_b]
    readable = np.abs(body_opponents) >= EDGE_THRESHOLDS[0]
    toward_neutral = np.sign(body_opponents) * (body_opponents - rim_opponents[:, :, None, :]) > 0
    neutral = np.all(toward_neutral | ~readable, axis=(2, 3))
    rim_values = np.where(allowed[rim[:, :, 0], rim[:, :, 1]] & neutral, rim_values, 255)
    valleys = rim[np.arange(64), np.argmin(rim_values, axis=1)]
    rim_paint = rim_values.min(axis=1)
    # Check a local valley on both sides of this actual rim pixel. Looking
    # farther away can cross a neighbouring stud's real rim, incorrectly
    # treating its monotone codec fringe as a second neutral outline.
    body = np.round(
        valleys[:, None, :] + directions[:, None, :] * np.array([-phase - 1, phase + 1])[None, :, None]
    ).astype(int)
    body = np.clip(body, [0, 0], np.subtract(chroma.shape, 1))
    face_paint = chroma[body[:, :, 0], body[:, :, 1]].min(axis=1)
    return np.maximum(0, face_paint - rim_paint)


def _extra_rim(near, unknown, hull, box, axes, declared, phase, chroma, frame, face_paint, cap_delta, novel_paint):
    """Prove a new complete stud around a chromatic arc, at any position."""
    x, y, w, h = box
    rx, ry = axes
    angles = np.arange(64) * (2 * np.pi / 64)
    offsets = np.round(np.column_stack([ry * np.sin(angles), rx * np.cos(angles)])).astype(int)
    step = max(1, phase // 2)
    for cy in range(round(y + h - ry - phase), round(y + ry + phase) + 1, step):
        for cx in range(round(x + w - rx - phase), round(x + rx + phase) + 1, step):
            # Several aliased arcs around one stud must retain that stud's
            # identity. A separate stud has its own non-overlapping ellipse;
            # an edge-centre jitter inside an existing footprint is not a
            # newly counted part feature.
            if any(np.hypot((cx - a) / (2 * rx), (cy - b) / (2 * ry)) < 1 for a, b in declared):
                continue
            points = offsets + [cy, cx]
            if np.any(points < 0) or np.any(points >= near.shape):
                continue
            yy, xx = points.T
            if np.mean(hull[yy, xx] > 0) < 0.95:
                continue
            support = near[yy, xx].reshape(8, 8).mean(axis=1)
            novel = unknown[yy, xx].reshape(8, 8).mean(axis=1)
            # Every octant must have a rim; unrelated face lines and codec
            # blocks cannot substitute for the other side of an ellipse.
            # Opposing unexplained arcs prevent reusing a declared stud.
            unexplained = novel >= 0.5
            if np.all(support >= 0.75) and support.mean() >= 0.90 and np.any(unexplained[:4] & unexplained[4:]):
                # The original labelled part supplies the cap-vs-face paint
                # direction. The candidate calibrates only their shared
                # lighting offset through its visible flat face. A patch of
                # flat face cannot become a stud just by borrowing the rims
                # of neighbouring studs to complete a geometric stencil.
                inner = np.round(offsets[::8] / 3 + [cy, cx]).astype(int)
                core = frame[inner[:, 0], inner[:, 1]].astype(float)
                cap = False
                if face_paint is not None:
                    expected_cap = np.clip(face_paint + cap_delta, 0, 255)
                    observed_delta = expected_cap - face_paint
                    cap_power = float(np.dot(observed_delta, observed_delta))
                    # Require actual cap-coloured samples. A componentwise
                    # RGB median can invent that paint between differently
                    # coloured codec pixels around neighbouring real studs.
                    cap_samples = ((core - face_paint) @ observed_delta >= cap_power / 2) & (
                        np.max(np.abs(core - expected_cap), axis=1) <= EDGE_THRESHOLDS[0] / 2
                    )
                    cap = cap_power > 0 and np.count_nonzero(cap_samples) >= len(core) / 2
                # Also retain genuinely outlined same-paint caps. Opposing
                # arcs must have their own two-sided neutral paint evidence;
                # other declared rims cannot supply that evidence for them.
                profile = _rim_profile(chroma, (cx, cy), axes, phase, novel_paint, frame)
                own_sectors = np.median(profile.reshape(8, 8), axis=1) >= EDGE_THRESHOLDS[0] / 2
                annulus = np.count_nonzero(own_sectors) >= 4 and np.any(own_sectors[:4] & own_sectors[4:])
                if cap or annulus:
                    return True
    return False


def _lego(frames, first, final, metadata):
    phase_radius = max(2, round(min(first.shape[:2]) / 256))
    new, hue, template, target_box, target, source_box, background, annotation, occluders = _definition(
        first, final, metadata
    )
    observations = []
    bad_identity = []
    static = True
    worst = 0
    for index, frame in enumerate(frames):
        observation = _observe(frame, hue, template, background, annotation, occluders)
        if observation is None:
            return _result({"one_readable_new_part_every_frame": False}, {"unreadable_part_frame": index})
        observations.append(observation)
        if observation["recall"] < 0.96 or observation["precision"] < 0.96:
            bad_identity.append(index)
        allowed = cv2.dilate(observation["hull"], np.ones((9, 9), np.uint8)).astype(bool) | (annotation > 0)
        preserved, error = _preserved(frame, background, allowed)
        static &= preserved
        worst = max(worst, error)
    points = np.asarray([v["center"] for v in observations])
    sx, sy, sw, sh = source_box
    start = np.array([sx + sw / 2, sy + sh / 2])
    vector = target - start
    progress = (points - start) @ vector / max(np.dot(vector, vector), 1)
    x, y, w, h = target_box
    aligned = cv2.warpAffine(
        frames[-1],
        np.float32([[1, 0, target[0] - points[-1, 0]], [0, 1, target[1] - points[-1, 1]]]),
        (first.shape[1], first.shape[0]),
    )
    expected_edges = cv2.Canny(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY), *EDGE_THRESHOLDS)
    actual_edges = cv2.Canny(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), *EDGE_THRESHOLDS)
    # The endpoint is another observation of the same required rim. Match
    # the per-frame colour/luminance union: 4:2:0 may retain the red/blue
    # channel boundary while the grayscale contrast falls below Canny.
    expected_colour_edges = expected_edges | cv2.Canny(final, *EDGE_THRESHOLDS)
    actual_colour_edges = actual_edges | cv2.Canny(aligned, *EDGE_THRESHOLDS)
    studs = []
    stud_templates = []
    stud_axes = []
    stud_centers = []
    stud_paints = []
    nominal_rims = np.zeros(first.shape[:2], np.uint8)
    stud_regions = np.zeros(first.shape[:2], np.uint8)
    for layer in _layers(metadata, "final_frame"):
        if layer.get("kind") != "ellipse" or "fill_rgb" not in layer:
            continue
        hsv = cv2.cvtColor(np.uint8([[layer["fill_rgb"][::-1]]]), cv2.COLOR_BGR2HSV)[0, 0]
        dh = abs(int(hsv[0]) - hue)
        if min(dh, 180 - dh) > 9 or hsv[1] < 80:
            continue
        cx, cy = round(layer["cx_px"]), round(layer["cy_px"])
        rx, ry = round(layer["rx_px"] + 4), round(layer["ry_px"] + 4)
        crop = (
            slice(max(0, cy - ry), min(first.shape[0], cy + ry + 1)),
            slice(max(0, cx - rx), min(first.shape[1], cx + rx + 1)),
        )
        # Either complete observation may establish the same labelled rim.
        # Keep the grayscale proof independent: a codec-only colour fringe
        # must not reduce its precision by being merged into that proof.
        coverage = [
            mask_coverage(
                (actual[crop] > 0).astype(np.uint8),
                (expected[crop] > 0).astype(np.uint8),
                radius=phase_radius,
            )
            for actual, expected in ((actual_edges, expected_edges), (actual_colour_edges, expected_colour_edges))
        ]
        studs.append(bool(any(min(observation) >= 0.90 for observation in coverage)))
        stud_axes.append((layer["rx_px"], layer["ry_px"]))
        stud_centers.append((cx, cy))
        stud_paints.append(layer["fill_rgb"][::-1])
        cv2.ellipse(nominal_rims, (cx, cy), (round(layer["rx_px"]), round(layer["ry_px"])), 0, 0, 360, 1, 1)
        # One 4:2:0 sample spans four canonical pixels at the registered 512px
        # observation. Rim phase can move within that footprint, not vanish.
        cv2.ellipse(stud_regions, (cx, cy), (rx + 2, ry + 2), 0, 0, 360, 1, -1)
        ring = np.zeros(first.shape[:2], np.uint8)
        cv2.ellipse(ring, (cx, cy), (max(1, round(layer["rx_px"])), max(1, round(layer["ry_px"]))), 0, 0, 360, 1, 7)
        stud_templates.append(np.argwhere((ring > 0) & (expected_edges > 0)))
    declared_studs = int(new["brick_type"].split("x")[0]) * int(new["brick_type"].split("x")[1])
    if len(studs) != declared_studs:
        raise ValueError("Rendered stud count contradicts instructed brick type")
    known_features = (expected_edges > 0) | (cv2.Canny(final, *EDGE_THRESHOLDS) > 0) | (nominal_rims > 0)
    # A blurred outer boundary's Canny maximum can fall on either side of its
    # declared outline. Keep that silhouette boundary as another observation
    # of the same edge, rather than treating the other maximum as a new stud.
    # Individual face polygons also contain hidden edges: only the boundary
    # of their union is observable, never those interior polygon seams.
    face_outlines = np.zeros(first.shape[:2], np.uint8)
    part_faces = []
    for layer in _layers(metadata, "final_frame"):
        if layer.get("kind") != "polygon" or "fill_rgb" not in layer or not layer.get("points_px"):
            continue
        hsv = cv2.cvtColor(np.uint8([[layer["fill_rgb"][::-1]]]), cv2.COLOR_BGR2HSV)[0, 0]
        dh = abs(int(hsv[0]) - hue)
        face_points = np.round(layer["points_px"]).astype(np.int32)
        if (
            min(dh, 180 - dh) <= 9
            and hsv[1] >= 80
            and len(face_points) == 4
            and np.all(
                (face_points >= [x - phase_radius, y - phase_radius])
                & (face_points <= [x + w + phase_radius, y + h + phase_radius])
            )
        ):
            cv2.fillPoly(face_outlines, [face_points], 1)
            part_faces.append((int(hsv[2]), face_points, np.array(layer["fill_rgb"][::-1], float)))
    top_face = np.zeros_like(face_outlines)
    if not part_faces:
        raise ValueError("Declared new-part face geometry is unavailable")
    # The renderer gives the horizontal studded face its brightest paint.
    # A closed edge pattern on a vertical wall is not another stud.
    top = max(part_faces, key=lambda item: item[0])
    cv2.fillPoly(top_face, [top[1]], 1)
    cap_delta = np.median(stud_paints, axis=0) - top[2]
    # The outer stroke has an inner and an outer transition. Retain their
    # one-pixel raster phase before applying the registered sampling radius.
    face_outlines = cv2.morphologyEx(face_outlines, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    known_features |= face_outlines > 0
    phase_kernel = np.ones((2 * phase_radius + 1, 2 * phase_radius + 1), np.uint8)
    known_near = cv2.dilate(known_features.astype(np.uint8), phase_kernel)
    minimum_extra = max(6, round(min(map(len, stud_templates)) * 0.10))
    stud_rx = min(axes[0] for axes in stud_axes)
    stud_ry = min(axes[1] for axes in stud_axes)
    extra_feature_frames = []
    largest_extra_feature = 0
    damaged_stud_frames = []
    for index, (frame, observation) in enumerate(zip(frames, observations)):
        # A translated, subsampled blue/red rim may lose grayscale contrast
        # at one phase while retaining its colour edge. Either visible gradient
        # supports the same required rim; the geometric coverage is unchanged.
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), *EDGE_THRESHOLDS) | cv2.Canny(
            frame, *EDGE_THRESHOLDS
        )
        near = cv2.dilate((edges > 0).astype(np.uint8), phase_kernel) > 0
        dx, dy = np.round(observation["center"] - target).astype(int)
        shifted_near = cv2.warpAffine(known_near, np.float32([[1, 0, dx], [0, 1, dy]]), first.shape[1::-1]) > 0
        occluded_near = cv2.dilate(observation["occluded"].astype(np.uint8), phase_kernel) > 0
        # Canny's derivative can turn a low-contrast codec block into a
        # connected face feature. Suppress that subpixel ringing before the
        # high-threshold face test; required stud rims still use the original
        # colour/luminance edge union above and keep their own coverage test.
        face_kernel = 2 * max(1, phase_radius - 1) + 1
        face_luma = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (face_kernel, face_kernel), 0)
        unexpected = (
            (cv2.Canny(face_luma, *FACE_EDGE_THRESHOLDS) > 0)
            & (observation["hull"] > 0)
            & ~shifted_near
            & ~occluded_near
            & (annotation == 0)
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(unexpected.astype(np.uint8), 8)
        largest = int(np.max(stats[1:count, cv2.CC_STAT_AREA])) if count > 1 else 0
        largest_extra_feature = max(largest_extra_feature, largest)
        if largest >= minimum_extra:
            extra_feature_frames.append(index)
        # A blue rim can retain its chromatic arc while losing luminance
        # contrast in 4:2:0. It is still an extra stud. Small codec colour
        # fragments can propose a cap, but cannot by themselves establish a
        # complete cap's geometry and labelled appearance.
        unknown_colour = (
            (edges > 0) & (observation["hull"] > 0) & ~shifted_near & ~occluded_near & (annotation == 0)
        ).astype(np.uint8)
        visible_top = cv2.warpAffine(top_face, np.float32([[1, 0, dx], [0, 1, dy]]), first.shape[1::-1])
        visible_top &= cv2.dilate(observation["mask"], phase_kernel)
        unknown_colour &= visible_top
        # A short arc may propose a stud; it need not itself span the full
        # stud. Only the complete geometric/paint proof below decides whether
        # those fragments really belong to an additional rim.
        count, _, stats, _ = cv2.connectedComponentsWithStats(unknown_colour, 8)
        declared_centers = np.asarray(stud_centers) + [dx, dy]
        chroma = frame.max(axis=2).astype(float) - frame.min(axis=2).astype(float)
        shifted_studs = cv2.warpAffine(stud_regions, np.float32([[1, 0, dx], [0, 1, dy]]), first.shape[1::-1])
        flat = (visible_top > 0) & (shifted_studs == 0) & (observation["mask"] > 0) & ~observation["occluded"]
        face_paint = np.median(frame[flat].astype(float), axis=0) if np.count_nonzero(flat) >= 25 else None
        unknown_near = cv2.dilate(unknown_colour, phase_kernel) > 0
        novel_paint = (
            (cv2.dilate(unknown_colour, np.ones((3, 3), np.uint8)) > 0)
            & ~shifted_near
            & ~occluded_near
            & (visible_top > 0)
        )
        for label in range(1, count):
            _, _, cw, ch, area = stats[label]
            if area < 6 or cw <= phase_radius or ch <= phase_radius:
                continue
            if _extra_rim(
                near,
                unknown_near,
                visible_top,
                stats[label, :4],
                (stud_rx, stud_ry),
                declared_centers,
                phase_radius,
                chroma,
                frame,
                face_paint,
                cap_delta,
                novel_paint,
            ):
                if index not in extra_feature_frames:
                    extra_feature_frames.append(index)
                largest_extra_feature = max(largest_extra_feature, int(area))
                break
        for stud_points in stud_templates:
            shifted = stud_points + np.array([dy, dx])
            valid = (
                (shifted[:, 0] >= 0)
                & (shifted[:, 0] < first.shape[0])
                & (shifted[:, 1] >= 0)
                & (shifted[:, 1] < first.shape[1])
            )
            shifted = shifted[valid]
            shifted = shifted[~observation["occluded"][shifted[:, 0], shifted[:, 1]]]
            if len(shifted) >= max(8, len(stud_points) * 0.25) and np.mean(near[shifted[:, 0], shifted[:, 1]]) < 0.90:
                damaged_stud_frames.append(index)
                break
    margin = 4
    crop = (
        slice(max(0, y - margin), min(first.shape[0], y + h + margin)),
        slice(max(0, x - margin), min(first.shape[1], x + w + margin)),
    )
    # Every instructed stud has an independent rim check above. Its thin rim
    # is not also a required high-threshold face edge. Unknown added features
    # outside those specific stud regions remain in the face comparison.
    face_actual = (cv2.Canny(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), *FACE_EDGE_THRESHOLDS) > 0) & (
        stud_regions == 0
    )
    face_expected = (cv2.Canny(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY), *FACE_EDGE_THRESHOLDS) > 0) & (
        stud_regions == 0
    )
    faces = (
        min(
            mask_coverage(
                face_actual[crop].astype(np.uint8),
                face_expected[crop].astype(np.uint8),
                radius=max(2, round(min(first.shape[:2]) * 3 / 1024)),
            )
        )
        >= 0.90
    )
    remaining = _hue_mask(frames[-1], hue)
    remaining[annotation > 0] = 0
    checks = {
        "one_new_rigid_part_throughout": not bad_identity,
        "starts_in_callout": np.linalg.norm(points[0] - start) <= 6,
        "correct_part_snapped_to_target": np.linalg.norm(points[-1] - target) <= 6,
        "actual_transfer_is_shown": bool(np.any((progress > 0.15) & (progress < 0.85))),
        "callout_is_empty_at_end": np.count_nonzero(remaining[sy : sy + sh, sx : sx + sw]) < 50,
        "existing_structure_and_scene_preserved": static,
        "all_instructed_studs_preserved": all(studs),
        "visible_studs_preserved_throughout": not damaged_stud_frames,
        "no_extra_part_features": not extra_feature_frames,
        "part_face_edges_match": bool(faces),
    }
    return _result(
        checks,
        {
            "part_type": new["brick_type"],
            "target_center": target.tolist(),
            "observed_part_centers": points.tolist(),
            "part_visibility": [v["visible_fraction"] for v in observations],
            "minimum_part_recall": min(v["recall"] for v in observations),
            "minimum_part_precision": min(v["precision"] for v in observations),
            "invalid_part_frames": bad_identity,
            "stud_checks": studs,
            "damaged_stud_frames": damaged_stud_frames,
            "unexpected_part_feature_frames": extra_feature_frames,
            "largest_unexpected_part_feature": largest_extra_feature,
            "unexpected_static_component": worst,
        },
    )


def verify(task, frames, first, final, metadata, reference):
    if task.split("_", 1)[0] == "O-25":
        return _lego(frames, first, final, metadata)
    return None
