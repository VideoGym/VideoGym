"""Binary checks for static, spatially labelled annotations.

Selection contracts independently solve source geometry, values and symbols,
then require closed answer rings to enclose exactly the required objects or
points. Remaining drawing contracts match the reference's answer regions.
Scene preservation and required temporal order are separate necessary checks.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import Counter

# G-39 moves an existing box and is deliberately excluded from this static family.
PREFIXES = frozenset(
    {
        "G-9",
        "G-29",
        "G-43",
        "G-131",
        "G-134",
        "G-135",
        "G-136",
        "G-138",
        "G-140",
        "G-147",
        "G-158",
        "G-160",
        "G-161",
        "G-167",
        "G-168",
        "G-169",
        "G-189",
        "G-202",
        "G-206",
        "G-212",
        "G-217",
        "G-218",
        "G-219",
        "G-221",
        "G-222",
        "G-223",
        "G-240",
        "G-247",
        "G-248",
        "G-250",
    }
)


def _layer_center(layer: dict) -> list[float] | None:
    if "cx_px" in layer:
        return [float(layer["cx_px"]), float(layer["cy_px"])]
    if layer.get("points_px"):
        points = np.asarray(layer["points_px"])
        return ((points.min(axis=0) + points.max(axis=0)) / 2).tolist()
    if "x_px" in layer and "w_px" in layer:
        return [layer["x_px"] + layer["w_px"] / 2, layer["y_px"] + layer["h_px"] / 2]
    return None


def _symbolic_reference(prefix: str, metadata: dict, targets: list[dict], shape: tuple) -> dict:
    """Recompute available answer labels independently of recorded answer indices."""
    s = metadata.get("semantic_ground_truth") or {}
    if not s:
        return {"checked": False, "reason": "no_symbolic_metadata"}
    layers = (metadata.get("generic_declarative_render") or {}).get("first_frame", {}).get("layers", [])
    centers = []
    rule = None
    if prefix in {"G-131", "G-134", "G-135"}:
        seq, candidates = s["sequence"], s["candidates"]
        sizes = [v["size"] for v in seq]
        if prefix == "G-131":
            steps = np.diff(sizes)
            if len(steps) < 1 or not np.allclose(steps, steps[0]) or steps[0] <= 0:
                raise ValueError("Reference sequence is not arithmetic and increasing")
            size = sizes[-1] + steps[0]
        else:
            if len(set(sizes)) != 2 or any(sizes[i] != sizes[i - 2] for i in range(2, len(sizes))):
                raise ValueError("Reference sequence does not alternate two sizes")
            size = sizes[-2]
        answer = [
            i
            for i, c in enumerate(candidates)
            if c["shape"] == seq[0]["shape"] and c["color"] == seq[0]["color"] and np.isclose(c["size"], size)
        ]
        if len(answer) != 1 or answer[0] != s["correct_option_index"]:
            raise ValueError("Sequence answer label contradicts recomputed shape/color/size rule")
        canvas_h = (metadata.get("generic_declarative_render") or {}).get("canvas", {}).get("height", shape[0])
        option_layers = []
        for layer in layers:
            center = _layer_center(layer)
            fill = layer.get("fill_rgb")
            if (
                layer.get("type") == "filled_shape"
                and center
                and center[1] > 0.74 * canvas_h
                and fill
                and min(fill) < 200
            ):
                option_layers.append(center)
        option_layers.sort(key=lambda c: c[0])
        if len(option_layers) != len(candidates):
            raise ValueError("Cannot associate symbolic candidates with rendered option positions")
        centers = [option_layers[answer[0]]]
        rule = "next_size_and_same_shape_color"
    elif prefix == "G-160":
        numbers = s["numbers"]
        idx = int(np.argmax([n["value"] for n in numbers]))
        if idx != s["largest_value_index"]:
            raise ValueError("Largest-number label contradicts numeric values")
        centers = [numbers[idx]["position"]]
        rule = "numeric_argmax"
    elif prefix == "G-161":
        shapes = s["shapes"]
        if len({q["type"] for q in shapes}) != 1:
            raise ValueError("Second-largest comparison needs common shape geometry")
        idx = sorted(range(len(shapes)), key=lambda i: shapes[i]["size"], reverse=True)[1]
        if idx != s["second_largest_index"]:
            raise ValueError("Second-largest label contradicts sizes")
        centers = [shapes[idx]["center"]]
        rule = "second_largest_same_geometry"
    elif prefix == "G-219":
        idx = min(range(len(s["shapes"])), key=lambda i: s["shapes"][i]["position"][0])
        if idx != s["leftmost_index"]:
            raise ValueError("Leftmost label contradicts x positions")
        centers = [s["shapes"][idx]["position"]]
        rule = "minimum_center_x"
    elif prefix == "G-206":
        idxs = [i for i, sides in enumerate(s["polygon_sides"]) if sides == 5]
        if idxs != [s["target_index"]]:
            raise ValueError("Pentagon label contradicts polygon side counts")
        polygons = [q for q in layers if q.get("type") == "filled_shape" and q.get("points_px")]
        if len(polygons) != len(s["polygon_sides"]) or len(polygons[idxs[0]]["points_px"]) != 5:
            raise ValueError("Pentagon declaration contradicts rendered polygon vertices")
        centers = [s["target_center_px"]]
        rule = "five_sides"
    elif prefix == "G-217":
        dots = sorted(
            [_layer_center(q) for q in layers if q.get("type") == "filled_shape" and q.get("kind") == "circle"],
            key=lambda c: c[0],
        )
        if len(dots) != s["dot_count"] or len(dots) % 2 != 1 or s["middle_index"] != len(dots) // 2:
            raise ValueError("Central-dot declaration contradicts rendered dots")
        centers = [dots[len(dots) // 2]]
        rule = "middle_by_count"
    elif prefix == "G-218":
        vertices = np.asarray(s["triangle"]["vertices_xy"], dtype=float)
        angles = []
        for i in range(3):
            a, b = vertices[(i + 1) % 3] - vertices[i], vertices[(i + 2) % 3] - vertices[i]
            angles.append(float(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
        idx = int(np.argmax(angles))
        if idx != s["target_vertex_index"]:
            raise ValueError("Largest-angle label contradicts vertex geometry")
        centers = [vertices[idx].tolist()]
        rule = "maximum_triangle_angle"
    elif prefix == "G-168":
        rectangles = s["rectangles"]
        idx = min(
            range(len(rectangles)),
            key=lambda i: abs(
                rectangles[i]["axis_aligned_bbox_xywh"][2] / rectangles[i]["axis_aligned_bbox_xywh"][3] - 1
            ),
        )
        if rectangles[idx]["index"] != s["answer_index"]:
            raise ValueError("Nearest-square label contradicts rectangle dimensions")
        x, y, w, h = rectangles[idx]["axis_aligned_bbox_xywh"]
        centers = [[x + w / 2, y + h / 2]]
        rule = "minimum_aspect_ratio_distance_from_one"
    elif prefix == "G-138":
        counts = Counter(tuple(q["color_rgb"]) for q in s["shapes"])
        unique = [q["index"] for q in s["shapes"] if counts[tuple(q["color_rgb"])] == 1]
        if unique != [s["unique_shape_index"]]:
            raise ValueError("Unique-color label contradicts shape color frequencies")
        geometry = metadata["parameters"]["shapes"][unique[0]]
        answer = next(q for q in s["shapes"] if q["index"] == unique[0])
        if geometry["fill_rgb"] != answer["color_rgb"]:
            raise ValueError("Unique-color shape geometry contradicts the symbolic palette")
        vertices = np.asarray(geometry["exterior_xy"], float)
        centers = [((vertices.min(axis=0) + vertices.max(axis=0)) / 2).tolist()]
        rule = "unique_color_frequency"
    if centers:
        canvas = (metadata.get("generic_declarative_render") or {}).get("canvas", {})
        scale = np.array([shape[1] / canvas.get("width", shape[1]), shape[0] / canvas.get("height", shape[0])])
        if len(centers) != len(targets):
            raise ValueError("Reference annotation count contradicts symbolic answer count")
        for center in centers:
            expected = np.asarray(center) * scale
            if not any(
                np.linalg.norm(expected - t["center"]) <= max(4, 0.2 * np.hypot(*t["bbox"][2:])) for t in targets
            ):
                raise ValueError("Reference annotation position contradicts recomputed symbolic answer")
    return {"checked": rule is not None, "rule": rule, "expected_centers": centers}


def _palette(first: np.ndarray, final: np.ndarray) -> np.ndarray:
    changed = np.max(np.abs(final.astype(np.int16) - first.astype(np.int16)), axis=2) > 45
    pixels = final[changed]
    if len(pixels) < 8:
        raise ValueError("Reference contains no readable annotation")
    # Quantisation excludes antialias pixels without requiring a task-specific hue.
    codes = (pixels.astype(np.int32) // 32) @ np.array([64, 8, 1])
    vals, counts = np.unique(codes, return_counts=True)
    return np.median(pixels[codes == vals[np.argmax(counts)]], axis=0)


def _ink_mixture(image: np.ndarray, first: np.ndarray, color: np.ndarray) -> tuple:
    original = first.astype(np.float32)
    direction = color - original
    observed = image.astype(np.float32) - original
    norm = np.sum(direction * direction, axis=2)
    alpha = np.sum(observed * direction, axis=2) / np.maximum(norm, 1)
    residual = np.max(np.abs(observed - alpha[:, :, None] * direction), axis=2)
    return norm, alpha, residual


def _certified_fade_mask(image: np.ndarray, first: np.ndarray, color: np.ndarray) -> np.ndarray:
    norm, alpha, residual = _ink_mixture(image, first, color)
    changed = np.max(np.abs(image.astype(np.int16) - first.astype(np.int16)), axis=2) > 25
    return (changed & (norm > 900) & (alpha > 0.10) & (alpha <= 1.15) & (residual <= 8)).astype(np.uint8) * 255


def _mark_mask(image: np.ndarray, first: np.ndarray, color: np.ndarray, *, allow_fade: bool = False) -> np.ndarray:
    delta = np.max(np.abs(image.astype(np.int16) - first.astype(np.int16)), axis=2)
    if not np.any(delta > 25):
        return np.zeros(first.shape[:2], np.uint8)
    color_distance = np.max(np.abs(image.astype(np.float32) - color), axis=2)
    matched = color_distance < 65
    norm, alpha, residual = _ink_mixture(image, first, color)
    # A thin opaque stroke becomes a partial color contribution at antialiased
    # pixels. Read that contribution instead of breaking a visible ring merely
    # because its edge is lighter than an opaque RGB sample. This does not join
    # different components or relax target/cardinality/topology predicates.
    minimum_alpha = 0.10 if allow_fade else 0.25
    contribution = (norm > 900) & (alpha > minimum_alpha) & (alpha <= 1.15) & (residual < 30)
    ink_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]
    if not allow_fade and ink_hsv[1] > 50:
        # Chromatic codec bleed at an existing chart edge can point toward the
        # marker color too. Partial pixels only continue a nearby observed ink
        # core; they cannot grow a new mark along an unrelated existing edge.
        core = ((delta > 25) & matched).astype(np.uint8)
        contribution &= cv2.dilate(core, np.ones((7, 7), np.uint8)) > 0
    matched |= contribution
    mask = ((delta > 25) & matched).astype(np.uint8) * 255
    if ink_hsv[1] > 50:
        # 4:2:0 averages chroma over a wider footprint than luminance. Thin red
        # ink can therefore retain a dark, continuous shaft while its RGB no
        # longer lies on the white-to-red alpha line. Observe these two signals
        # separately; enlarging RGB distance would also admit unrelated hues.
        source_ycc = cv2.cvtColor(first, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        actual_ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ink_ycc = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2YCrCb)[0, 0].astype(np.float32)
        chroma_direction = ink_ycc[1:] - source_ycc[:, :, 1:]
        observed_chroma = actual_ycc[:, :, 1:] - source_ycc[:, :, 1:]
        chroma_norm = np.sum(chroma_direction**2, axis=2)
        chroma_alpha = np.sum(observed_chroma * chroma_direction, axis=2) / np.maximum(chroma_norm, 1)
        chroma_residual = np.max(np.abs(observed_chroma - chroma_alpha[:, :, None] * chroma_direction), axis=2)
        chroma_length = np.linalg.norm(observed_chroma, axis=2)
        luma_change = actual_ycc[:, :, 0] - source_ycc[:, :, 0]
        luma_direction = ink_ycc[0] - source_ycc[:, :, 0]
        luma_compatible = (np.abs(luma_direction) < 20) | (luma_change * luma_direction > 0)
        source_hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
        actual_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        source_hue = np.abs(source_hsv[:, :, 0].astype(np.int16) - int(ink_hsv[0]))
        actual_hue = np.abs(actual_hsv[:, :, 0].astype(np.int16) - int(ink_hsv[0]))
        source_hue = np.minimum(source_hue, 180 - source_hue)
        actual_hue = np.minimum(actual_hue, 180 - actual_hue)
        # A broad RGB box around muted green also contains neutral gray. Wall
        # and furniture edge reconstruction is not green annotation evidence.
        # Require an observed chromatic shaft; alpha and chroma observations
        # below still recover its lighter codec edges.
        chromatic = (actual_hue <= 12) & (actual_hsv[:, :, 1] >= 25)
        # A fade over a colored source need not yet have the final ink's hue
        # (yellow over green is one example). A tightly matched source-to-ink
        # RGB contribution is direct evidence of that intermediate color.
        fade_contribution = contribution & (residual <= 8) & allow_fade
        chromatic |= fade_contribution
        mask = ((mask > 0) & chromatic).astype(np.uint8) * 255
        hue_change = (actual_hue < 10) & ((source_hsv[:, :, 1] < 40) | (source_hue > actual_hue + 4))
        # Red ink over a darker red shape can preserve hue and luminance while
        # visibly increasing both chroma and value. It is a legitimate drawing
        # phase, unlike a small chroma leak along an unchanged orange edge.
        saturation_change = (
            (actual_hue <= 8)
            & (actual_hsv[:, :, 1].astype(np.int16) > source_hsv[:, :, 1].astype(np.int16) + 25)
            & (actual_hsv[:, :, 2].astype(np.int16) > source_hsv[:, :, 2].astype(np.int16) + 25)
        )
        # Chroma averaging at a static orange edge can enter the red RGB box
        # while preserving both its original hue and luminance. It is not ink.
        mask = (
            (mask > 0) & ((np.abs(luma_change) > 18) | hue_change | saturation_change | fade_contribution)
        ).astype(np.uint8) * 255
        # Chroma is source-relative evidence: a blue stroke mixed with an
        # orange background need not yet have the final pure-blue hue.
        candidate = (
            (delta > 25)
            & (chroma_norm > 400)
            & (chroma_alpha > (0.08 if allow_fade else 0.12))
            & (chroma_alpha < 1.5)
            & (chroma_residual < 18)
            & (chroma_residual < np.maximum(3, chroma_length * 0.35))
            & luma_compatible
            & (np.abs(luma_change) > 18)
            & (actual_hsv[:, :, 1] >= 25)
        )
        # A small hue leak on an existing black edge has no independent shaft.
        # It needs either substantial luma support or connected evidence away
        # from the original edge. This preserves existing same-colour objects.
        gradient = cv2.morphologyEx(first, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        old_edge = cv2.dilate((np.max(gradient, axis=2) > 25).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        seed = candidate & ((np.abs(luma_change) > 45) | ~old_edge | (mask > 0))
        count, labels = cv2.connectedComponents((candidate | (mask > 0)).astype(np.uint8), 8)
        supported = np.bincount(labels[seed | (mask > 0)], minlength=count) > 0
        supported[0] = False
        mask = supported[labels].astype(np.uint8) * 255
    # Compression can cut a one-pixel hole in an otherwise closed stroke. Close
    # each observed component separately: a global closing would join two real
    # answer rings separated by a narrow, still-visible background gap.
    if ink_hsv[1] <= 50:
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    closed = np.zeros_like(mask)
    for label in range(1, count):
        x, y, w, h = stats[label, :4]
        crop = (
            slice(max(0, y - 2), min(mask.shape[0], y + h + 2)),
            slice(max(0, x - 2), min(mask.shape[1], x + w + 2)),
        )
        component = (labels[crop] == label).astype(np.uint8) * 255
        closed[crop] |= cv2.morphologyEx(component, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return closed


def _components(mask: np.ndarray, first: np.ndarray | None = None, frame: np.ndarray | None = None) -> list[dict]:
    min_pixels = max(6, int(mask.size * 0.000006))
    result = []
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    for label in range(1, count):
        x, y, w, h, pixels = stats[label]
        if pixels < min_pixels:
            continue
        component = (labels[y : y + h, x : x + w] == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) + (x, y)
        local = np.zeros((h, w), np.uint8)
        cv2.drawContours(local, [contour - (x, y)], -1, 255, -1)
        actual = (mask[y : y + h, x : x + w] > 0) & (local > 0)
        n = int(np.count_nonzero(actual))
        if n < min_pixels:
            continue
        area = int(np.count_nonzero(local))
        result.append(
            {
                "bbox": (x, y, w, h),
                "center": (x + w / 2, y + h / 2),
                "contour": contour,
                "pixels": n,
                "filled_area": area,
                "fill_fraction": n / max(area, 1),
                "mask": component,
                "source_frame": first,
                "observed_frame": frame,
            }
        )
    return sorted(result, key=lambda c: (c["center"][0], c["center"][1]))


def _reflection_symmetric(vertices: list) -> bool:
    points = np.asarray(vertices, float)
    points -= np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1], points[:, 0])
    # Any reflection maps the first vertex to another vertex. Its axis is
    # their angular bisector through the vertex-set centroid.
    for angle in (angles + angles[0]) / 2:
        axis = np.asarray([np.cos(angle), np.sin(angle)])
        reflected = 2 * np.outer(points @ axis, axis) - points
        distances = np.linalg.norm(reflected[:, None] - points[None, :], axis=2)
        if np.max(np.min(distances, axis=1)) < 0.01:
            return True
    return False


def _point_contract(prefix: str, metadata: dict, first: np.ndarray) -> dict | None:
    """Public point-selection semantics do not prescribe a reference radius."""
    s = metadata.get("semantic_ground_truth") or {}
    if prefix == "G-158":
        markers = metadata["parameters"]["points"]
        points = np.asarray([marker["position"] for marker in markers], float)
        targets = [index for index, marker in enumerate(markers) if marker["type"] == "hollow"]
        target = targets[0]
    elif prefix == "G-160" and s:
        points = np.asarray([v["position"] for v in s["numbers"]], float)
        target = int(np.argmax([v["value"] for v in s["numbers"]]))
    elif prefix == "G-161" and s:
        points = np.asarray([v["center"] for v in s["shapes"]], float)
        target = sorted(range(len(points)), key=lambda i: s["shapes"][i]["size"], reverse=True)[1]
    elif prefix == "G-168" and s:
        boxes = np.asarray([v["axis_aligned_bbox_xywh"] for v in s["rectangles"]], float)
        points = boxes[:, :2] + boxes[:, 2:] / 2
        target = int(np.argmin(np.abs(boxes[:, 2] / boxes[:, 3] - 1)))
    elif prefix == "G-169" and s:
        a, b = np.asarray(s["segments"][0]["endpoints"], float)
        c, d = np.asarray(s["segments"][1]["endpoints"], float)
        t, u = np.linalg.solve(np.column_stack([b - a, c - d]), c - a)
        if not (0 <= t <= 1 and 0 <= u <= 1):
            raise ValueError("The source segments do not intersect")
        points = np.asarray([a + t * (b - a), a, b, c, d])
        target = 0
    elif prefix == "G-206" and s:
        layers = metadata["generic_declarative_render"]["first_frame"]["layers"]
        polygons = [v for v in layers if v.get("type") == "filled_shape" and v.get("points_px")]
        points = np.asarray([_layer_center(v) for v in polygons], float)
        indices = [i for i, polygon in enumerate(polygons) if len(polygon["points_px"]) == 5]
        if len(indices) != 1:
            raise ValueError("The source does not contain exactly one pentagon")
        target = indices[0]
    elif prefix == "G-212" and s:
        arrows = [v for v in s["objects"] if v.get("shape") == "arrow"]
        points = np.asarray([v["position_px"] for v in arrows], float)
        center = np.asarray(s["ring_center_px"], float)
        directions = np.asarray([v["direction_rad"] for v in arrows])
        offsets = points - center
        sense = np.sign(offsets[:, 0] * np.sin(directions) - offsets[:, 1] * np.cos(directions))
        majority = np.sign(np.sum(sense))
        minority = np.flatnonzero(sense != majority)
        if not majority or len(minority) != 1:
            raise ValueError("The source arrows do not have exactly one opposite direction")
        target = int(minority[0])
    elif prefix == "G-217" and s:
        layers = metadata["generic_declarative_render"]["first_frame"]["layers"]
        points = np.asarray(sorted(
            [_layer_center(v) for v in layers if v.get("type") == "filled_shape" and v.get("kind") == "circle"],
            key=lambda point: point[0],
        ), float)
        target = len(points) // 2
    elif prefix == "G-218" and s:
        points = np.asarray(s["triangle"]["vertices_xy"], float)
        lengths = [np.linalg.norm(points[(i + 1) % 3] - points[(i + 2) % 3]) for i in range(3)]
        target = int(np.argmax(lengths))
    elif prefix == "G-219" and s:
        points = np.asarray([v["position"] for v in s["shapes"]], float)
        target = int(np.argmin(points[:, 0]))
    elif prefix == "G-223" and s:
        segments = np.asarray([[v["start"], v["end"]] for v in s["lines"]], float)
        points = np.mean(segments, axis=1)
        horizontal = np.flatnonzero(np.abs(segments[:, 0, 1] - segments[:, 1, 1]) < 0.01).tolist()
        if not horizontal:
            raise ValueError("The source has no horizontal segments")
        target = horizontal[0]
    elif prefix == "G-247" and s:
        characters = s["all_characters"]
        points = np.asarray([v["position"] for v in characters], float)
        indices = [i for i, v in enumerate(characters) if any(0x3400 <= ord(c) <= 0x9FFF for c in v["char"])]
        if len(indices) != 1:
            raise ValueError("The source does not contain exactly one Chinese character")
        target = indices[0]
    elif prefix == "G-248" and s:
        shapes = metadata["parameters"]["shapes"]
        points = np.asarray([v["center"] for v in shapes], float)
        indices = [i for i, v in enumerate(shapes) if not _reflection_symmetric(v["points"])]
        if len(indices) != 1:
            raise ValueError("The source does not contain exactly one reflection-asymmetric shape")
        target = indices[0]
    else:
        return None
    canvas = (metadata.get("generic_declarative_render") or {}).get("canvas", {})
    if prefix == "G-158":
        canvas = metadata["parameters"]["canvas"]
    scale = np.asarray([first.shape[1] / canvas.get("width", first.shape[1]),
                        first.shape[0] / canvas.get("height", first.shape[0])])
    points *= scale
    contract = {"points": points, "target": target, "targets": [target],
                "whole_object": prefix not in {"G-169", "G-218"},
                "geometric_object": prefix in {"G-158", "G-161", "G-168", "G-206", "G-219", "G-248"}}
    if prefix == "G-158":
        contract["targets"] = targets
    elif prefix == "G-223":
        contract["targets"] = horizontal
        contract["segments"] = segments * scale
    return contract


def _elliptical_points(outline: np.ndarray) -> bool:
    if len(outline) < 5:
        return False
    outline = np.asarray(outline, np.float32)
    center, axes, angle = cv2.fitEllipse(outline)
    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    local = (outline - center) @ rotation.T
    radii = np.maximum(np.asarray(axes) / 2, 1)
    residual = np.abs(np.linalg.norm(local / radii, axis=1) - 1)
    return float(np.quantile(residual, 0.95)) <= 0.12


def _elliptical_outline(contour: np.ndarray) -> bool:
    if len(contour) < 5:
        return False
    vertices = contour[:, 0]
    # Simple contour approximation keeps corners but discards straight-edge
    # interiors. Include both when testing ellipse geometry.
    outline = np.concatenate([
        start + (end - start) * np.linspace(0, 1, max(2, round(np.linalg.norm(end - start) / 2)), endpoint=False)[:, None]
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0))
    ]).astype(np.float32)
    return _elliptical_points(outline)


def _source_pixel_support(first: np.ndarray, actual: np.ndarray, points: np.ndarray,
                          eligible: np.ndarray | None = None) -> np.ndarray:
    supported = np.zeros(len(points), bool)
    observed = actual[points[:, 1], points[:, 0]].astype(np.int16)
    phase = max(1, round(min(first.shape[:2]) / 512))
    for dy in range(-phase, phase + 1):
        for dx in range(-phase, phase + 1):
            xx = np.clip(points[:, 0] + dx, 0, first.shape[1] - 1)
            yy = np.clip(points[:, 1] + dy, 0, first.shape[0] - 1)
            compatible = np.max(np.abs(observed - first[yy, xx].astype(np.int16)), axis=1) <= 55
            supported |= compatible if eligible is None else compatible & eligible[yy, xx]
    return supported


def _source_supported_boundary(pred: dict, boundary: np.ndarray) -> np.ndarray:
    first, actual = pred.get("source_frame"), pred.get("observed_frame")
    if first is None or actual is None:
        return np.zeros(len(boundary), bool)
    # A colored source line can become slightly redder at a stroke crossing
    # without acquiring a new branch. Use the same local RGB fidelity budget
    # as scene preservation, only on thin existing colored structures.
    colored = (cv2.cvtColor(first, cv2.COLOR_BGR2HSV)[:, :, 1] > 50).astype(np.uint8)
    thickness = cv2.distanceTransform(np.pad(colored, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    thin = (colored > 0) & (thickness <= max(2, min(first.shape[:2]) * 6 / 512))
    x, y = pred["bbox"][:2]
    return _source_pixel_support(first, actual, boundary[:, 0] + (x, y), thin)


def _single_outline(pred: dict, *, elliptical: bool = False) -> bool:
    contours, hierarchy = cv2.findContours(pred["mask"], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    interior = [boundary for boundary, relation in zip(contours, hierarchy[0]) if relation[3] >= 0]
    largest_hole = max((cv2.contourArea(boundary) for boundary in interior), default=0)
    holes = [boundary for boundary in interior if cv2.contourArea(boundary) >= max(4, largest_hole * 0.01)]
    if len(holes) != 1:
        return False
    # A uniform outline's inner boundary follows the outer boundary at its
    # observed stroke width. An inward branch makes a deep excursion into the
    # enclosed region even when it leaves the number of holes unchanged.
    x, y, w, h = pred["bbox"]
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [pred["contour"] - (x, y)], -1, 1, -1)
    distance = cv2.distanceTransform(np.pad(filled, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    boundary = holes[0][:, 0]
    widths = distance[boundary[:, 1], boundary[:, 0]]
    typical = float(np.median(widths))
    uniform = float(np.quantile(widths, 0.98)) <= typical + max(3, typical * 0.75)
    if uniform and (not elliptical or _elliptical_outline(holes[0])):
        return True
    supported = _source_supported_boundary(pred, holes[0])
    remaining = holes[0][~supported]
    pred["source_supported_inner_boundary_pixels"] = int(np.count_nonzero(supported))
    # These are dense observed boundary pixels with source-supported intervals
    # omitted. Do not fabricate straight chords across those intervals.
    if len(remaining) < 5 or (elliptical and not _elliptical_points(remaining[:, 0])):
        return False
    points = remaining[:, 0]
    widths = distance[points[:, 1], points[:, 0]]
    typical = float(np.median(widths))
    return float(np.quantile(widths, 0.98)) <= typical + max(3, typical * 0.75)


def _point_pair(pred: dict, contract: dict, first: np.ndarray) -> dict:
    points, target = contract["points"], contract["target"]
    contour = pred["contour"]
    center, radius = cv2.minEnclosingCircle(contour)
    radius = max(float(radius), 1)
    circularity = float(cv2.contourArea(contour) / (np.pi * radius**2))
    ellipse_ok = _elliptical_outline(contour)
    # A true annulus has one elliptical interior boundary. Looking only at its
    # outer contour would silently accept attached spokes, crosses and dots.
    annulus = _single_outline(pred, elliptical=True)
    distances = [float(cv2.pointPolygonTest(contour, tuple(point), True)) for point in points]
    selected = [i for i, distance in enumerate(distances) if distance >= 0]
    center_error = float(np.linalg.norm(points[target] - center))
    hollow = pred["fill_fraction"] < 0.65
    target_coverage = 1.0
    target_envelope = True
    coverage_ok = True
    if contract["whole_object"]:
        # Observe the source object's silhouette or glyph ink, rather than
        # accepting a small ring around its center.
        if "segments" in contract:
            start, end = contract["segments"][target]
            target_pixels = np.rint(start + (end - start) * np.linspace(0, 1, max(4, round(np.linalg.norm(end - start))))[:, None]).astype(int)
        elif contract["geometric_object"]:
            foreground = (np.max(255 - first.astype(np.int16), axis=2) > 55).astype(np.uint8)
            count, labels, stats, centers = cv2.connectedComponentsWithStats(foreground, 8)
            components = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 6]
            label = min(components, key=lambda i: np.linalg.norm(centers[i] - points[target]))
            # Source objects are separate connected shapes. Nearest-center
            # pixel ownership would wrongly steal the bottom of a nearby
            # large disk and assign it to a smaller disk below it.
            silhouette = (labels == label).astype(np.uint8)
            boundaries, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(silhouette, boundaries, -1, 1, -1)
            yy, xx = np.nonzero(silhouette)
            target_pixels = np.column_stack([xx, yy])
        else:
            ys, xs = np.nonzero(np.max(255 - first.astype(np.int16), axis=2) > 55)
            glyph_points = np.column_stack([xs, ys])
            owners = np.argmin(np.sum((glyph_points[:, None] - points[None, :])**2, axis=2), axis=1)
            target_pixels = glyph_points[owners == target]
        enclosed = np.zeros(first.shape[:2], np.uint8)
        cv2.drawContours(enclosed, [contour], -1, 1, -1)
        target_coverage = float(np.mean(enclosed[target_pixels[:, 1], target_pixels[:, 0]]))
        if contract["geometric_object"]:
            # Circling a polygon may cut its corners while clearly enclosing
            # its complete horizontal and vertical extent. Require that
            # envelope and a majority of its body, rather than fitting the
            # renderer's circle or demanding all corner pixels lie inside.
            low, high = target_pixels.min(axis=0), target_pixels.max(axis=0)
            x, y, w, h = pred["bbox"]
            phase = max(2, round(min(first.shape[:2]) * 2 / 512))
            target_envelope = bool(np.all(np.asarray([x, y]) <= low + phase)
                                   and np.all(np.asarray([x + w - 1, y + h - 1]) >= high - phase))
            coverage_ok = target_envelope and target_coverage > 0.5
        else:
            coverage_ok = target_coverage >= 0.95
    matched = (selected == [target] and ellipse_ok and annulus
               and hollow and coverage_ok)
    return {
        "matched": bool(matched), "selection_semantics": "independent_point_enclosure",
        "region_iou": 0.0, "selected_point_indices": selected, "target_point_index": target,
        "circle_center_error_px": center_error, "circle_radius_px": radius,
        "circle_area_ratio": circularity, "hollow_preserved": bool(hollow),
        "elliptical_outline": bool(ellipse_ok), "single_annulus": bool(annulus),
        "source_supported_inner_boundary_pixels": pred.get("source_supported_inner_boundary_pixels", 0),
        "target_object_coverage": target_coverage,
        "target_envelope_covered": target_envelope,
    }


def _pair(ref: dict, pred: dict, shape: tuple[int, ...]) -> dict:
    a = np.zeros(shape[:2], np.uint8)
    b = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(a, [ref["contour"]], -1, 1, -1)
    cv2.drawContours(b, [pred["contour"]], -1, 1, -1)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    iou = inter / max(union, 1)
    r = ref["bbox"]
    p = pred["bbox"]
    center_error = float(np.linalg.norm(np.subtract(ref["center"], pred["center"])))
    # Relative to the answer region, not the whole mostly empty frame.
    center_limit = max(3.0, 0.16 * np.hypot(r[2], r[3]))
    ratios = (p[2] / r[2], p[3] / r[3])
    size_ok = all(0.70 <= q <= 1.40 for q in ratios)
    # A thin open line has no enclosed area; use stroke overlap instead.
    closed_reference = ref["fill_fraction"] < 0.70 and min(r[2:]) >= 6
    topology_ok = not closed_reference or (
        pred["fill_fraction"] < min(0.75, ref["fill_fraction"] * 2.5 + 0.08) and _single_outline(pred)
    )
    filled_reference = ref["fill_fraction"] > 0.80
    fill_ok = not filled_reference or pred["fill_fraction"] >= ref["fill_fraction"] * 0.80
    return {
        "matched": bool(iou >= 0.55 and center_error <= center_limit and size_ok and topology_ok and fill_ok),
        "region_iou": round(float(iou), 4),
        "center_error_px": round(center_error, 3),
        "center_limit_px": round(float(center_limit), 3),
        "size_ratios": list(ratios),
        "hollow_required": bool(closed_reference),
        "hollow_preserved": bool(topology_ok),
        "filled_region_required": bool(filled_reference),
        "filled_region_preserved": bool(fill_ok),
        "reference_fill_fraction": round(ref["fill_fraction"], 4),
        "predicted_fill_fraction": round(pred["fill_fraction"], 4),
    }


def _preservation(first: np.ndarray, last: np.ndarray, mark: np.ndarray) -> dict:
    # Tolerance is local to objects/tiles: deleting a tiny answer object cannot be
    # hidden by a million unchanged white pixels.
    # Registered 512px chroma edges occupy a wider footprint on the canonical
    # 1024px canvas. Exclude that immediate ink halo as well as the ink shaft;
    # otherwise its tinted fringe is counted as damage to a nearby thin edge.
    halo = max(2, round(min(first.shape[:2]) * 1.5 / 512))
    excluded = cv2.dilate(mark, np.ones((2 * halo + 1, 2 * halo + 1), np.uint8)) > 0
    # Compare at the codec's small edge scale, so a shifted antialias pixel is
    # not mistaken for a deleted thin object. Object-local checks remain strict.
    smooth_first = cv2.GaussianBlur(first, (3, 3), 0)
    smooth_last = cv2.GaussianBlur(last, (3, 3), 0)
    diff = np.max(np.abs(smooth_last.astype(np.int16) - smooth_first.astype(np.int16)), axis=2) > 55
    # Generated 512px observations can move a thin original edge within one
    # native pixel. Compare both RGB directions in a tiny neighborhood instead
    # of treating that edge phase as destruction. A missing object or stroke
    # still has no nearby counterpart in the candidate.
    yy, xx = np.nonzero(diff & ~excluded)
    if len(xx):
        original = smooth_first.astype(np.int16)
        actual = smooth_last.astype(np.int16)
        forward = np.full(len(xx), 32767, np.int16)
        backward = forward.copy()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ys = np.clip(yy + dy, 0, first.shape[0] - 1)
                xs = np.clip(xx + dx, 0, first.shape[1] - 1)
                forward = np.minimum(forward, np.max(np.abs(actual[yy, xx] - original[ys, xs]), axis=1))
                backward = np.minimum(backward, np.max(np.abs(original[yy, xx] - actual[ys, xs]), axis=1))
        diff[yy, xx] = (forward > 55) | (backward > 55)
    changed = diff & ~excluded
    h, w = first.shape[:2]
    border = np.concatenate([first[0], first[-1], first[:, 0], first[:, -1]])
    bg = np.median(border, axis=0)
    fg = np.max(np.abs(first.astype(np.float32) - bg), axis=2) > 35
    tested = fg & ~excluded
    max_object_error = 0.0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    min_object = max(10, int(h * w * 0.000015))
    valid_counts = np.bincount(labels[tested], minlength=n)
    changed_counts = np.bincount(labels[changed & tested], minlength=n)
    valid_objects = (stats[:, cv2.CC_STAT_AREA] >= min_object) & (valid_counts >= min_object)
    valid_objects[0] = False
    if np.any(valid_objects):
        max_object_error = float(np.max(changed_counts[valid_objects] / valid_counts[valid_objects]))
    max_tile_error = 0.0
    step = max(16, min(h, w) // 16)
    for y in range(0, h, step):
        for x in range(0, w, step):
            region = tested[y : y + step, x : x + step]
            size = np.count_nonzero(region)
            if size >= max(10, step * step * 0.02):
                bad = changed[y : y + step, x : x + step] & region
                max_tile_error = max(max_tile_error, float(np.count_nonzero(bad) / size))
    global_error = float(np.count_nonzero(changed) / changed.size)
    return {
        "passed": bool(global_error <= 0.01 and max_object_error <= 0.12 and max_tile_error <= 0.25),
        "changed_scene_ratio": global_error,
        "max_object_error": max_object_error,
        "max_local_foreground_error": max_tile_error,
    }


def _background_reference(first: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Immutable flat regions, separated from existing local feature edges.

    Both white maze corridors and dark walls can be unchanged background. A
    single global background color cannot describe that contract. Guard actual
    color edges instead, including equal-luminance boundaries that Canny misses.
    """
    gradient = cv2.morphologyEx(first, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edges = (np.max(gradient, axis=2) > 25).astype(np.uint8)
    guard = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
    smooth = cv2.GaussianBlur(first, (3, 3), 0).astype(np.int16)
    minimum = max(6, round(first.shape[0] * first.shape[1] * 0.00002))
    return smooth, ~guard, minimum, first


def _foreign_background_component(
    frame: np.ndarray,
    reference: tuple[np.ndarray, np.ndarray, int, np.ndarray],
    legitimate_edit: np.ndarray,
    *,
    chromatic_ink: bool = False,
) -> tuple[bool, int]:
    """Veto an independent new component; never average it over the canvas."""
    original, flat, minimum, raw_first = reference
    allowed = cv2.dilate((legitimate_edit > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    actual = cv2.GaussianBlur(frame, (3, 3), 0).astype(np.int16)
    changed = (np.max(np.abs(actual - original), axis=2) > 45) & flat & ~allowed
    # The blurred observation can spread a low-contrast color halo one working
    # pixel past the registered ink footprint. It is not an independent object
    # (for example the mild inner color cast of a growing ring over cyan).
    # Keep strong dark/color marks observable even in this immediate halo.
    halo = (cv2.dilate((legitimate_edit > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~allowed
    if np.any(changed & halo):
        expected_lab = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.int16)
        actual_lab = cv2.cvtColor(actual.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.int16)
        mild = np.max(np.abs(actual_lab - expected_lab), axis=2) <= 20
        changed &= ~(halo & mild)
    if chromatic_ink:
        # An achromatic black object cannot be supplied by a colored pen.
        # Test its observed component before ink-halo dilation can swallow its
        # center, keeping the same independent-object area floor.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        source_y = cv2.cvtColor(raw_first, cv2.COLOR_BGR2GRAY).astype(np.int16)
        actual_y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        dark = (hsv[:, :, 1] < 25) & (hsv[:, :, 2] <= 80) & (source_y - actual_y > 80) & flat
        count, _, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
        largest_dark = int(np.max(stats[1:count, cv2.CC_STAT_AREA])) if count > 1 else 0
        if largest_dark >= minimum:
            return False, largest_dark
    count, _, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), 8)
    largest = int(np.max(stats[1:count, cv2.CC_STAT_AREA])) if count > 1 else 0
    return largest < minimum, largest


def verify(
    task: str,
    frames: list[np.ndarray],
    first: np.ndarray,
    final: np.ndarray,
    metadata: dict,
    reference: list[np.ndarray],
    *,
    observation_resolution: tuple[int, int] | None = None,
    native_observation: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict | None:
    """Return explicit predicates; no continuous legacy score is consulted."""
    prefix = task.split("_", 1)[0]
    if prefix not in PREFIXES:
        return None
    if not frames:
        return {"passed": False, "checks": {"frames_present": False}, "evidence": {}}
    if prefix == "G-250":
        from evaluation.cross_benchmark.vbvr_pro.verification.annotation.regions import verify_triple_intersection

        return verify_triple_intersection(frames, first, metadata, observation_resolution)
    semantics = metadata.get("semantic_ground_truth") or {}
    expected_mark = semantics.get("expected_mark") or {}
    annotation = semantics.get("annotation_expectation") or {}
    explicit_black = (
        expected_mark.get("style") == "stroked_circle_black" or annotation.get("style") == "black_closed_contour"
    )
    # At a tangent point, low-pass black ink is blended with two colored disks;
    # its dominant observed RGB is not the ink color stated by the task. Dark
    # codec tints also cannot override a declared black contour.
    color = np.zeros(3) if explicit_black else _palette(first, final)
    chromatic_ink = bool(cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0, 1] > 50)
    gt_mask = _mark_mask(final, first, color)
    pred_mask = _mark_mask(frames[-1], first, color)
    targets = _components(gt_mask)
    predictions = _components(pred_mask, first, frames[-1])
    if not targets:
        raise ValueError("Reference annotation extraction failed")
    # Codec ringing near an existing similarly coloured edge can create a few
    # isolated pixels. Keep marks down to a small fraction of the smallest true
    # annotation; cap the floor so extra small, deliberate circles are retained.
    readable_pixels = max(6, min(64, min(t["pixels"] for t in targets) * 0.15))
    predictions = [p for p in predictions if p["pixels"] >= readable_pixels]
    source_edge_fragments = []
    if prefix == "G-221" and native_observation is not None:
        native_first, native_last = native_observation
        native_ink = _mark_mask(native_last, native_first, color)
        gradient = cv2.morphologyEx(native_first, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        source_edges = np.max(gradient, axis=2) > 25
        retained = []
        for pred in predictions:
            # Use native pigment only to refute an interpolated source-edge
            # ghost. Geometry, topology and cardinality retain their original
            # grid and thresholds, including small genuine extra blue dots.
            coverage = np.zeros(first.shape[:2], np.uint8)
            x, y, width, height = pred["bbox"]
            coverage[y : y + height, x : x + width] = pred["mask"]
            coverage = cv2.resize(
                coverage, (native_first.shape[1], native_first.shape[0]), interpolation=cv2.INTER_AREA
            ) > 0
            if np.any(native_ink[coverage]):
                retained.append(pred)
                continue
            yy, xx = np.nonzero(coverage)
            supported = _source_pixel_support(native_first, native_last, np.column_stack([xx, yy]), source_edges)
            if np.all(supported):
                source_edge_fragments.append({"bbox": list(pred["bbox"]), "pixels": pred["pixels"]})
            else:
                retained.append(pred)
        predictions = retained
    symbolic = _symbolic_reference(prefix, metadata, targets, first.shape)
    point_contract = _point_contract(prefix, metadata, first)
    # Exact cardinality plus injective matching: duplicates cannot replace a
    # missing target; a huge circle cannot count as several selected objects.
    used: set[int] = set()
    pairs = []
    answer_targets = [{**point_contract, "target": index} for index in point_contract["targets"]] if point_contract else targets
    for target_index, target in enumerate(answer_targets):
        candidates = [
            (j, _point_pair(p, target, first) if point_contract else _pair(target, p, first.shape))
            for j, p in enumerate(predictions) if j not in used
        ]
        good = [(j, d) for j, d in candidates if d["matched"]]
        if good:
            j, detail = max(good, key=lambda x: x[1]["region_iou"])
            used.add(j)
        else:
            j, detail = max(candidates, key=lambda x: x[1]["region_iou"]) if candidates else (-1, {"matched": False})
        reference_bbox = list(targets[target_index]["bbox"]) if target_index < len(targets) else None
        pairs.append({"reference_bbox": reference_bbox, "prediction_index": j, **detail})
    preservation = _preservation(first, frames[-1], pred_mask)
    # Check every decoded frame at a fixed working resolution. A correct last
    # frame cannot erase intervening scene destruction or wrong annotations.
    h, w = first.shape[:2]
    scale = min(1.0, 320 / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small_first = cv2.resize(first, size, interpolation=cv2.INTER_AREA)
    if prefix == "G-221":
        # The answer is a hollow square outline. Its enclosed fill is not an
        # editable region: a small extra blue dot must remain an invalid mark
        # even when it is below the terminal component-readability floor.
        allowed = gt_mask.copy()
        for selected_index in used:
            target = predictions[selected_index]
            x, y, width, height = target["bbox"]
            allowed[y : y + height, x : x + width] |= target["mask"]
    else:
        allowed = np.zeros(first.shape[:2], np.uint8)
        for target in targets:
            cv2.drawContours(allowed, [target["contour"]], -1, 255, -1)
        for selected_index in used:
            cv2.drawContours(allowed, [predictions[selected_index]["contour"]], -1, 255, -1)
    allowed = cv2.resize(allowed, size, interpolation=cv2.INTER_NEAREST)
    reference_stroke = cv2.resize(gt_mask, size, interpolation=cv2.INTER_NEAREST)
    allowance = max(2, int(round(min(size) * 0.01)))
    allowed = cv2.dilate(allowed, np.ones((2 * allowance + 1, 2 * allowance + 1), np.uint8)) > 0
    background_reference = _background_reference(small_first)
    original_ink = (np.max(np.abs(small_first.astype(np.float32) - color), axis=2) < 65).astype(np.uint8)
    # A codec may spread an already-present red/black edge by a pixel. That is
    # an observation halo around existing ink, not a newly drawn annotation.
    original_ink_halo = cv2.dilate(original_ink, np.ones((5, 5), np.uint8)) > 0
    small_point_contract = _point_contract(prefix, metadata, small_first) if point_contract else None
    bad_frames, off_target_frames = [], []
    largest_off_target = 0
    foreign_frames = []
    largest_foreign = 0
    largest_source_edge_phase = 0
    peak_onsets = []
    peak_centers = None
    if prefix == "G-202":
        wave = np.asarray(metadata["parameters"]["wave_points"], float)
        peaks = wave[1:-1][(wave[1:-1, 1] < wave[:-2, 1]) & (wave[1:-1, 1] < wave[2:, 1])]
        canvas = metadata["parameters"]["canvas"]
        peak_centers = peaks * [size[0] / canvas[0], size[1] / canvas[1]]
        peak_onsets = [None] * len(peaks)
    for index, frame in enumerate(frames):
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        marks = _mark_mask(small, small_first, color, allow_fade=True)
        if peak_centers is not None:
            for peak_index, center in enumerate(peak_centers):
                zone = np.zeros(marks.shape, np.uint8)
                radius = max(3, round(float(semantics["marker_ring_radius_px"]) * scale * 1.5))
                cv2.circle(zone, tuple(np.rint(center).astype(int)), radius, 1, -1)
                if peak_onsets[peak_index] is None and np.count_nonzero((marks > 0) & (zone > 0)) >= 6:
                    peak_onsets[peak_index] = index
        # Reference stroke pixels may fade in from the underlying source color,
        # so they need not yet be close to the final annotation's opaque color.
        scene_exclusion = cv2.bitwise_or(marks, reference_stroke)
        if not _preservation(small_first, small, scene_exclusion)["passed"]:
            bad_frames.append(index)
        opaque_marks = _mark_mask(small, small_first, color)
        certified_fade = _certified_fade_mask(small, small_first, color)
        # Codec phase at an original object edge can accidentally lie on the
        # source-to-ink color ray, even in an unchanged conditioning frame.
        # Exclude only pixels independently explained by that nearby source
        # edge; a real contrasting stroke has no such RGB witness.
        yy, xx = np.nonzero((certified_fade > 0) & ~background_reference[1])
        if len(xx):
            supported = _source_pixel_support(small_first, small, np.column_stack([xx, yy]))
            certified_fade[yy[supported], xx[supported]] = 0
            largest_source_edge_phase = max(largest_source_edge_phase, int(np.count_nonzero(supported)))
        certified_marks = cv2.bitwise_or(opaque_marks, certified_fade)
        frame_allowed = allowed
        if small_point_contract and np.any((certified_marks > 0) & ~allowed & ~original_ink_halo):
            # The public selection may use a different readable circle radius
            # midway through the video. A complete intermediate ring has to
            # select the same independently solved target before its region
            # can be admitted; a translucent wrong-target ring stays wrong.
            frame_allowed = allowed.copy().astype(np.uint8)
            for component in _components(certified_marks, small_first, small):
                if any(_point_pair(component, {**small_point_contract, "target": target}, small_first)["matched"]
                       for target in small_point_contract["targets"]):
                    cv2.drawContours(frame_allowed, [component["contour"]], -1, 1, -1)
            frame_allowed = frame_allowed > 0
        off_target = ((certified_marks > 0) & ~frame_allowed & ~original_ink_halo).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(off_target, 8)
        extra = int(np.max(stats[1:count, cv2.CC_STAT_AREA])) if count > 1 else 0
        largest_off_target = max(largest_off_target, extra)
        # One wrong annotation remains wrong when many correct annotations are
        # visible. Ignore only isolated codec pixels, never a proportion of ink.
        if extra >= max(6, round(off_target.size * 0.00002)):
            off_target_frames.append(index)
        clean, area = _foreign_background_component(small, background_reference, scene_exclusion, chromatic_ink=chromatic_ink)
        largest_foreign = max(largest_foreign, area)
        if not clean:
            foreign_frames.append(index)
    checks = {
        "annotation_present": bool(predictions),
        "exact_answer_count": len(answer_targets) == len(predictions),
        "all_targets_matched_once": len(used) == len(answer_targets),
        "no_extra_targets": len(used) == len(predictions),
        "source_objects_preserved": preservation["passed"],
        "source_objects_preserved_throughout": not bad_frames,
        "annotations_stay_on_answer_regions": not off_target_frames,
        "no_new_background_objects": not foreign_frames,
    }
    if peak_centers is not None:
        checks["peaks_marked_left_to_right_one_at_a_time"] = bool(peak_onsets) and all(
            value is not None for value in peak_onsets
        ) and all(a < b for a, b in zip(peak_onsets, peak_onsets[1:]))
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "family": "static_annotation_set",
            "target_source": "independent_point_enclosure" if point_contract else "reference_annotation",
            "marker_bgr": color.tolist(),
            "marker_color_source": "explicit_metadata" if explicit_black else "reference_edit",
            **({"native_source_edge_fragments": source_edge_fragments} if prefix == "G-221" else {}),
            "expected_count": len(answer_targets),
            "predicted_count": len(predictions),
            "pairs": pairs,
            "preservation": preservation,
            "symbolic_reference": symbolic,
            "corrupted_frame_indices": bad_frames,
            "off_target_frame_indices": off_target_frames,
            "largest_off_target_component": largest_off_target,
            "foreign_background_frame_indices": foreign_frames,
            "largest_foreign_background_component": largest_foreign,
            "source_edge_phase_pixels_max": largest_source_edge_phase,
            **({"peak_mark_onset_frames": peak_onsets} if peak_centers is not None else {}),
        },
    }
