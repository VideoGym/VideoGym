"""O-64: rigid face sprites, independent visible features and a bijective move.

The trusted source supplies each animal's appearance, while the declared source
and target geometry supplies its identity.  No animal classifier or OCR is used.
Templates are local objects, never whole reference-video poses.  Occlusion order
and speed are observed; neither is copied from the renderer's animation.
"""

from dataclasses import dataclass
from itertools import permutations

import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _result

SUPPORTED_IDS = {"O-64"}


def _shift_layer(layer, delta):
    shifted = dict(layer)
    if "points_px" in layer:
        shifted["points_px"] = (np.asarray(layer["points_px"]) + delta).tolist()
    for key, axis in (("cx_px", 0), ("cy_px", 1), ("x_px", 0), ("y_px", 1)):
        if key in layer:
            shifted[key] += float(delta[axis])
    return shifted


def _same_layer(a, b):
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        if isinstance(value, (list, int, float)):
            if np.shape(value) != np.shape(b[key]) or not np.allclose(
                value, b[key], atol=1.01 if key.endswith("_px") else 0
            ):
                return False
        elif value != b[key]:
            return False
    return True


def _layer_mask(layer, shape):
    mask = np.zeros(shape, np.uint8)
    filled = layer["type"] == "filled_shape"
    width = -1 if filled else int(layer.get("stroke_width_px", 1))
    if layer.get("kind") == "ellipse":
        cv2.ellipse(
            mask,
            (round(layer["cx_px"]), round(layer["cy_px"])),
            (round(layer["rx_px"]), round(layer["ry_px"])),
            0,
            0,
            360,
            255,
            width,
        )
    elif layer.get("kind") == "rect":
        x, y = round(layer["x_px"]), round(layer["y_px"])
        cv2.rectangle(mask, (x, y), (x + round(layer["w_px"]), y + round(layer["h_px"])), 255, width)
    elif "points_px" in layer:
        points = np.asarray(layer["points_px"], np.int32)
        if filled:
            cv2.fillPoly(mask, [points], 255)
        else:
            cv2.polylines(mask, [points], layer.get("kind") == "polygon", 255, width)
    else:
        raise ValueError(f"Unsupported animal primitive: {layer}")
    return mask


@dataclass
class Sprite:
    index: int
    animal: str
    start: np.ndarray
    goal: np.ndarray
    offset: np.ndarray
    image: np.ndarray
    mask: np.ndarray
    features: list[np.ndarray]


def _dictionary(first, metadata):
    params = metadata["parameters"]
    objects = params["objects"]
    if len(objects) != params["num_animals"] or len({o["index"] for o in objects}) != len(objects):
        raise ValueError("Animal count/index declarations disagree")
    semantic = {
        o["index"]: o
        for o in metadata.get("semantic_ground_truth", {}).get("objects", [])
        if o.get("role") == "animal_face"
    }
    render = metadata["generic_declarative_render"]
    canvas = render["canvas"]
    raw_shape = (canvas["height"], canvas["width"])
    scale = np.array([first.shape[1] / raw_shape[1], first.shape[0] / raw_shape[0]])
    source_layers = render["first_frame"]["layers"]
    target_layers = render["final_frame"]["layers"]
    assigned = [[] for _ in objects]
    for layer in source_layers:
        matches = []
        for i, obj in enumerate(objects):
            delta = np.subtract(obj["target_center"], obj["initial_center"])
            if any(_same_layer(_shift_layer(layer, delta), target) for target in target_layers):
                matches.append(i)
        if len(matches) > 1:
            if "points_px" in layer:
                center = np.mean(layer["points_px"], axis=0)
            else:
                center = np.array([layer.get("cx_px", layer.get("x_px")), layer.get("cy_px", layer.get("y_px"))], float)
            matches = [
                min(matches, key=lambda i: np.linalg.norm(center - objects[i]["initial_center"]) / objects[i]["size"])
            ]
        if matches:
            assigned[matches[0]].append(layer)
    sprites = []
    for obj, layers in zip(objects, assigned):
        if semantic:
            s = semantic.get(obj["index"])
            if s is None or s["animal_type"] != obj["type"] or s["color_rgb"] != obj["color"]:
                raise ValueError("Animal identity declarations disagree")
            for key, declaration in (("initial_center", "initial_position"), ("target_center", "target_position")):
                if not np.allclose(obj[key], [s[declaration]["x"], s[declaration]["y"]], atol=0.01):
                    raise ValueError("Animal endpoint declarations disagree")
        if len(layers) < 3:
            raise ValueError("Animal source lacks a complete local feature dictionary")
        labels = np.zeros(raw_shape, np.uint16)
        for i, layer in enumerate(layers, 1):
            labels[_layer_mask(layer, raw_shape) > 0] = i
        labels = cv2.resize(labels, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = labels > 0
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = (
            max(0, xs.min() - 2),
            max(0, ys.min() - 2),
            min(first.shape[1], xs.max() + 3),
            min(first.shape[0], ys.max() + 3),
        )
        labels = labels[y0:y1, x0:x1]
        local = first[y0:y1, x0:x1].copy()
        # Vector masks anchor every primitive; actual source pixels retain the
        # renderer's antialiasing and the registered observation bandwidth.
        mask = labels > 0
        features = []
        for i in range(1, len(layers) + 1):
            region = (labels == i).astype(np.uint8)
            eroded = cv2.erode(region, np.ones((3, 3), np.uint8)) > 0
            if eroded.sum() >= max(3, region.sum() * 0.28):
                region = eroded
            else:
                region = region > 0
            if region.sum() >= 3:
                features.append(region)
        start = np.asarray(obj["initial_center"], float) * scale
        sprites.append(
            Sprite(
                obj["index"],
                obj["type"],
                start,
                np.asarray(obj["target_center"], float) * scale,
                np.array([x0, y0]) - start,
                local,
                mask,
                features,
            )
        )
    return sprites, np.asarray(canvas["background_rgb"][::-1], np.uint8)


def _place(array, top_left, shape, fill=0):
    result = np.full((*shape, *array.shape[2:]), fill, array.dtype)
    x, y = map(int, top_left)
    h, w = array.shape[:2]
    left, top, right, bottom = max(0, x), max(0, y), min(shape[1], x + w), min(shape[0], y + h)
    if right > left and bottom > top:
        result[top:bottom, left:right] = array[top - y : bottom - y, left - x : right - x]
    return result


def _pose(frame, sprite):
    # Sample each independently declared feature equally. A large occluded
    # cheek must not overpower still-visible ears, eyes or mane pieces.
    route = sprite.goal - sprite.start
    distance = float(np.linalg.norm(route))
    direction = np.array([-route[1], route[0]]) / distance
    fractions = np.linspace(0, 1, max(2, round(distance * 1.6)))
    candidates = np.unique(
        np.rint(
            np.concatenate(
                [
                    sprite.start + fractions[:, None] * route + sprite.offset + offset * direction
                    for offset in (-1.5, 0, 1.5)
                ]
            )
        ),
        axis=0,
    ).astype(int)
    h, w = sprite.image.shape[:2]
    valid = (
        (candidates[:, 0] >= 0)
        & (candidates[:, 1] >= 0)
        & (candidates[:, 0] + w <= frame.shape[1])
        & (candidates[:, 1] + h <= frame.shape[0])
    )
    candidates = candidates[valid]
    template = cv2.GaussianBlur(sprite.image, (3, 3), 0).astype(float)
    observed = cv2.GaussianBlur(frame, (3, 3), 0).astype(float)
    background = np.median(template[~sprite.mask], axis=0)
    costs = []
    for feature in sprite.features:
        ys, xs = np.nonzero(feature)
        take = np.linspace(0, len(xs) - 1, min(32, len(xs))).astype(int)
        xs, ys = xs[take], ys[take]
        expected = template[ys, xs]
        actual = observed[candidates[:, 1, None] + ys, candidates[:, 0, None] + xs]
        contrast = np.maximum(15, np.max(np.abs(expected - background), axis=1))
        error = np.max(np.abs(actual - expected), axis=2) / contrast
        costs.append(np.mean(np.minimum(error, 3), axis=1))
    costs = np.asarray(costs)
    witnesses = max(2, int(np.ceil(len(costs) * 0.45)))
    robust = np.mean(np.sort(costs, axis=0)[:witnesses], axis=0) + 0.015 * np.mean(costs, axis=0)
    position = candidates[int(np.argmin(robust))]
    center = position - sprite.offset
    progress = float((center - sprite.start) @ route / (route @ route))
    return position, progress


def _order(frame, images, masks):
    """Choose an observed opaque order, allowing either animal to pass in front."""
    count = len(masks)
    graph = [{j for j in range(count) if j != i and np.any(masks[i] & masks[j])} for i in range(count)]
    remaining = set(range(count))
    result = []
    while remaining:
        group = {min(remaining)}
        frontier = list(group)
        while frontier:
            for j in graph[frontier.pop()] - group:
                group.add(j)
                frontier.append(j)
        remaining -= group
        active = sorted(group)
        if len(active) == 1:
            result.extend(active)
            continue
        overlapping = np.sum([masks[i] for i in active], axis=0) > 1
        support = np.asarray([m[overlapping] for m in masks])
        values = np.asarray([im[overlapping].astype(np.float32) for im in images])
        observed = frame[overlapping].astype(np.float32)
        costs = np.mean(np.minimum(np.abs(values - observed), 70), axis=2)
        best, chosen = float("inf"), active
        columns = np.arange(support.shape[1])
        for order in permutations(active):
            winner = np.zeros(support.shape[1], int)
            for i in order:
                winner[support[i]] = i
            error = float(np.mean(costs[winner, columns]))
            if error < best:
                best, chosen = error, order
        result.extend(chosen)
    return result


def _refine_occluded(frame, sprites, poses, base):
    """Fit exposed caps of partly hidden faces against the complete local scene.

    Independent feature votes locate visible faces. An almost hidden face may
    share dark features with its occluder; the exposed outline, including empty
    background outside it, then provides the remaining geometric evidence.
    """
    shape = frame.shape[:2]
    for _ in range(2):
        masks = [_place(s.mask, p, shape) for s, p in zip(sprites, poses)]
        images = [_place(s.image, p, shape) for s, p in zip(sprites, poses)]
        order = _order(frame, images, masks)
        for index, sprite in enumerate(sprites):
            others = (
                np.any([m for j, m in enumerate(masks) if j != index], axis=0)
                if len(masks) > 1
                else np.zeros(shape, bool)
            )
            if np.count_nonzero(masks[index] & others) < np.count_nonzero(masks[index]) * 0.3:
                continue
            origin = poses[index]
            h, w = sprite.mask.shape
            x0, y0 = np.maximum(0, origin - 11)
            x1, y1 = np.minimum([shape[1], shape[0]], origin + [w + 11, h + 11])
            local_shape = (int(y1 - y0), int(x1 - x0))
            local_base = base[y0:y1, x0:x1]
            actual = cv2.GaussianBlur(frame[y0:y1, x0:x1], (3, 3), 0).astype(float)
            others_images = {j: images[j][y0:y1, x0:x1] for j in order if j != index}
            others_masks = {j: masks[j][y0:y1, x0:x1] for j in order if j != index}
            route = sprite.goal - sprite.start
            direction = route / np.linalg.norm(route)
            perpendicular = np.array([-direction[1], direction[0]])
            trials = np.unique(
                np.rint(
                    [origin + step * direction + side * perpendicular for step in range(-8, 9) for side in (-1, 0, 1)]
                ),
                axis=0,
            ).astype(int)
            best, chosen = float("inf"), origin
            other_order = [j for j in order if j != index]
            original_depth = order.index(index)
            depths = set((0, len(other_order), original_depth))
            for trial in trials:
                anchor = trial - sprite.offset
                progress = float((anchor - sprite.start) @ route / (route @ route))
                perpendicular_distance = float(np.linalg.norm(anchor - sprite.start - progress * route))
                if progress < -0.007 or progress > 1.007 or perpendicular_distance > 2:
                    continue
                own_image = _place(sprite.image, trial - [x0, y0], local_shape)
                own_mask = _place(sprite.mask, trial - [x0, y0], local_shape)
                for depth in depths:
                    composite = local_base.copy()
                    trial_order = other_order[:depth] + [index] + other_order[depth:]
                    for j in trial_order:
                        image, mask = (own_image, own_mask) if j == index else (others_images[j], others_masks[j])
                        composite[mask] = image[mask]
                    expected = cv2.GaussianBlur(composite, (3, 3), 0).astype(float)
                    error = float(np.mean(np.minimum(np.abs(actual - expected), 80)))
                    if error < best:
                        best, chosen = error, trial
            poses[index] = chosen
            images[index] = _place(sprite.image, chosen, shape)
            masks[index] = _place(sprite.mask, chosen, shape)
    return poses


def _near_error(frame, expected):
    # One observation pixel of phase freedom, without moving a feature to a
    # different eye/ear or borrowing ink from an unrelated face.
    a = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2YCrCb).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(expected, (3, 3), 0), cv2.COLOR_BGR2YCrCb).astype(np.float32)
    errors = np.full(a.shape, 255, np.float32)
    best = np.full(a.shape[:2], np.inf, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(a, (dy, dx), axis=(0, 1))
            error = np.abs(shifted - b)
            # Keep a single observed pixel as the witness for all channels.
            score = np.max(error, axis=2)
            better = score < best
            errors[better] = error[better]
            best[better] = score[better]
    return errors


def _large_components(mask, minimum=5):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [list(map(int, row[:5])) for row in stats[1:n] if row[cv2.CC_STAT_AREA] >= minimum]


def _pale_erasure_components(mask):
    """Require a resolved two-dimensional loss of nearly-background paint.

    The registered H264 channel can weaken a one-pixel chain of pale pixels
    inside an otherwise intact face. A 2x2 interior witness distinguishes a
    resolved patch from that sparse quantisation residue; a compact 2x3 hole
    still fails the existing six-pixel component rule.
    """
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    interior = cv2.erode(binary, np.ones((2, 2), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    supported = set(np.unique(labels[interior]).tolist())
    return [
        list(map(int, stats[index, :5]))
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 6 and index in supported
    ]


def _pale_erasure_witnesses(erased, replacement, raw_y, contrast, current, background_y):
    """Read enclosed holes or coherent background replacement at an edge.

    A codec-softened silhouette connects directly to the visible canvas. It
    does not establish an interior hole, and chroma leakage is not replacement
    by the canvas paint. A real boundary notch can be canvas-connected too;
    its complete RGB replacement remains an independent erasure witness.
    """
    witnessed = replacement.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(erased.astype(np.uint8), 8)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 6:
            continue
        region = labels == index
        near_canvas = np.abs(raw_y - background_y) < float(np.median(contrast[region])) * 0.45
        # Pair 8-connected foreground with 4-connected canvas: a one-pixel
        # diagonal contact cannot open an otherwise enclosed missing patch.
        _, canvas_labels = cv2.connectedComponents(near_canvas.astype(np.uint8), connectivity=4)
        exterior = np.unique(canvas_labels[~current])
        enclosed = near_canvas & ~np.isin(canvas_labels, exterior)
        witnessed |= region & enclosed
    return _pale_erasure_components(witnessed)


def _source_chroma_residue(actual, base, expected, current, candidates):
    """A weak luma fringe needs a compatible nearby source-paint direction."""
    from evaluation.cross_benchmark.vbvr_pro.media.video import ENCODING

    radius = ENCODING["macro_block_size"]
    supported = np.zeros(current.shape, bool)
    for y, x in zip(*np.nonzero(candidates)):
        region = (
            slice(max(0, y - radius), min(current.shape[0], y + radius + 1)),
            slice(max(0, x - radius), min(current.shape[1], x + radius + 1)),
        )
        paint = expected[region][current[region]].astype(float)
        direction = paint - base[y, x]
        observed = actual[y, x] - base[y, x]
        norm = np.sum(direction * direction, axis=1)
        alpha = (direction @ observed) / np.maximum(norm, 1)
        residual = np.max(np.abs(observed - alpha[:, None] * direction), axis=1)
        supported[y, x] = bool(np.any((alpha > 0) & (alpha <= 1) & (residual <= 8)))
    return supported


def verify(task, frames, first, final, metadata, reference):
    if task.split("_", 1)[0] not in SUPPORTED_IDS:
        return None
    if not frames:
        return _result({"animal_frames_available": False})
    width = min(512, first.shape[1])
    size = (width, round(first.shape[0] * width / first.shape[1]))

    def resize(image):
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA) if image.shape[1::-1] != size else image

    source = resize(first)
    observed = [resize(frame) for frame in frames]
    sprites, background = _dictionary(source, metadata)
    shape = source.shape[:2]
    base = source.copy()
    for sprite in sprites:
        source_mask = _place(sprite.mask, np.rint(sprite.start + sprite.offset), shape)
        base[cv2.dilate(source_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = background
    static_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    static_edges = cv2.Canny(static_gray, 15, 45) > 0
    static_halo = cv2.dilate(static_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    checks = {}
    trajectories = [[] for _ in sprites]
    failures = []
    for frame_index, frame in enumerate(observed):
        if frame_index and np.array_equal(frame, observed[frame_index - 1]):
            previous_prefix = f"frame_{frame_index - 1}."
            reused = {
                f"frame_{frame_index}." + name[len(previous_prefix) :]: value
                for name, value in checks.items()
                if name.startswith(previous_prefix)
            }
            checks.update(reused)
            for trajectory in trajectories:
                trajectory.append(trajectory[-1])
            continue
        poses, images, masks = [], [], []
        for i, sprite in enumerate(sprites):
            position, progress = _pose(frame, sprite)
            poses.append(position)
        poses = _refine_occluded(frame, sprites, poses, base)
        images = [_place(sprite.image, pose, shape) for sprite, pose in zip(sprites, poses)]
        masks = [_place(sprite.mask, pose, shape) for sprite, pose in zip(sprites, poses)]
        for i, sprite in enumerate(sprites):
            route = sprite.goal - sprite.start
            center = poses[i] - sprite.offset
            trajectories[i].append(float((center - sprite.start) @ route / (route @ route)))
        order = _order(frame, images, masks)
        expected = base.copy()
        visible = [mask.copy() for mask in masks]
        covering = np.zeros(shape, bool)
        for i in reversed(order):
            visible[i] &= ~covering
            covering |= masks[i]
        for i in order:
            expected[masks[i]] = images[i][masks[i]]
        error = _near_error(frame, expected)
        expected_y = cv2.cvtColor(expected, cv2.COLOR_BGR2YCrCb)[..., 0]
        local_contrast = cv2.dilate(expected_y, np.ones((11, 11), np.uint8)).astype(float) - cv2.erode(
            expected_y, np.ones((11, 11), np.uint8)
        )
        background_y = float(cv2.cvtColor(background[None, None], cv2.COLOR_BGR2YCrCb)[0, 0, 0])
        for i, sprite in enumerate(sprites):
            safe_visible = cv2.erode(visible[i].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            for feature_index, feature in enumerate(sprite.features):
                area = _place(feature, poses[i], shape) & safe_visible
                if area.sum() < max(3, feature.sum() * 0.15):
                    continue  # Actually occluded by another independently read face.
                values = error[area]
                trusted_y = expected_y[area].astype(float)
                tolerance_y = np.clip(np.maximum(np.abs(trusted_y - background_y), local_contrast[area]) * 0.22, 5, 32)
                matched = (values[:, 0] <= tolerance_y) & (values[:, 1:].max(axis=1) <= 36)
                passed = float(np.mean(matched)) >= 0.78
                key = f"frame_{frame_index}.animal_{sprite.index}.feature_{feature_index}_preserved"
                checks[key] = passed
                if not passed:
                    failures.append(
                        {
                            "frame": frame_index,
                            "animal": sprite.index,
                            "feature": feature_index,
                            "visible_pixels": int(area.sum()),
                            "matched_fraction": float(np.mean(matched)),
                            "position": poses[i].tolist(),
                        }
                    )
        # Flat background has a local veto, so one small foreign object cannot
        # disappear into a whole-frame average. Only current faces are exempt.
        current = np.any(masks, axis=0)
        halo = static_halo | (cv2.dilate(current.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        smooth_a = cv2.GaussianBlur(frame, (3, 3), 0).astype(np.int16)
        smooth_b = cv2.GaussianBlur(base, (3, 3), 0).astype(np.int16)
        rgb_delta = np.max(np.abs(smooth_a - smooth_b), axis=2)
        observed_y = cv2.cvtColor(smooth_a.astype(np.uint8), cv2.COLOR_BGR2YCrCb)[..., 0].astype(float)
        base_y = cv2.cvtColor(smooth_b.astype(np.uint8), cv2.COLOR_BGR2YCrCb)[..., 0].astype(float)
        # Low-contrast paint is readable in flat canvas, but weak ringing
        # beside a high-contrast moving edge has a wider codec footprint.
        low_contrast_halo = static_halo | (cv2.dilate(current.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0)
        # 4:2:0 gives chroma the wider registered footprint. An RGB tint needs
        # actual nearby source paint; an unrelated colour cannot borrow that
        # footprint. A dark extra feature retains the smaller luma guard.
        luma_delta = np.abs(observed_y - base_y)
        added = ((rgb_delta > 20) & ~halo) | ((rgb_delta > 8) & (luma_delta > 6) & ~low_contrast_halo)
        candidate_chroma = added & low_contrast_halo & (luma_delta <= 20)
        added &= ~_source_chroma_residue(
            smooth_a.astype(float), smooth_b.astype(float), expected, current, candidate_chroma
        )
        components = _large_components(added)
        checks[f"frame_{frame_index}.no_foreign_or_vacated_source_content"] = not components
        # Static divider/target strokes remain obligations when visible. Codec
        # edge halos excuse phase, not deletion of the original target outline.
        required = static_edges & ~cv2.dilate(current.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        observed_edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 15, 45)
        near = cv2.dilate(observed_edges, np.ones((5, 5), np.uint8)) > 0
        missing_static = _large_components(required & ~near, minimum=8)
        checks[f"frame_{frame_index}.stationary_outlines_preserved"] = not missing_static
        # Extra dark marks inside a body are likewise checked locally, rather
        # than averaged with thousands of correctly retained fill pixels.
        expected_edges = cv2.Canny(cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY), 15, 45)
        edge_halo = cv2.dilate(expected_edges, np.ones((5, 5), np.uint8)) > 0
        expected_smooth = cv2.GaussianBlur(expected, (3, 3), 0)
        wrong = (np.max(np.abs(smooth_a - expected_smooth.astype(np.int16)), axis=2) > 45) & current & ~edge_halo
        extras = _large_components(wrong, minimum=5)
        checks[f"frame_{frame_index}.no_extra_face_features"] = not extras
        expected_smooth_y = cv2.cvtColor(expected_smooth, cv2.COLOR_BGR2YCrCb)[..., 0].astype(float)
        pale_contrast = np.abs(expected_smooth_y - background_y)
        pale_core = (pale_contrast > 5) & (pale_contrast < 25) & (local_contrast < 30)
        pale_core &= cv2.erode(current.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        # Codec DC rounding affects even untouched flat canvas. Measure it on
        # visible declared background before comparing a faint fill to canvas;
        # a one-level offset must not hide an actual interior erasure.
        flat_background = ~low_contrast_halo & np.all(base == background, axis=2)
        background_offset = (
            background_y - float(np.median(observed_y[flat_background])) if np.any(flat_background) else 0.0
        )
        registered_y = observed_y + background_offset
        erased_pale = (
            pale_core
            & (np.abs(registered_y - background_y) < pale_contrast * 0.45)
            & (np.abs(registered_y - expected_smooth_y) > 5)
        )
        observed_background = (
            np.median(smooth_a[flat_background], axis=0) if np.any(flat_background) else background.astype(float)
        )
        paint_rgb_contrast = np.abs(expected_smooth.astype(float) - background)
        replaced_rgb = np.abs(smooth_a.astype(float) - observed_background)
        # A canvas-connected edge additionally needs the original measured
        # luma loss: a one-level DC registration cannot manufacture evidence
        # of a cut into a low-contrast silhouette. Enclosed holes have their
        # separate raw-pixel topology witness below.
        measured_erasure = (np.abs(observed_y - background_y) < pale_contrast * 0.45) & (
            np.abs(observed_y - expected_smooth_y) > 5
        )
        coherent_replacement = (
            erased_pale
            & measured_erasure
            & np.all((paint_rgb_contrast < 5) | (replaced_rgb <= paint_rgb_contrast * 0.45), axis=2)
        )
        raw_y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[..., 0].astype(float) + background_offset
        pale_holes = _pale_erasure_witnesses(
            erased_pale, coherent_replacement, raw_y, pale_contrast, current, background_y
        )
        checks[f"frame_{frame_index}.pale_face_fill_not_erased"] = not pale_holes
        if components or missing_static or extras or pale_holes:
            failures.append(
                {
                    "frame": frame_index,
                    "foreign": components,
                    "missing_static": missing_static,
                    "extra_features": extras,
                    "pale_holes": pale_holes,
                    "background_luma_offset": background_offset,
                }
            )
    departures, arrivals = [], []
    for sprite, trajectory in zip(sprites, trajectories):
        checks[f"animal_{sprite.index}.starts_at_its_source"] = abs(trajectory[0]) <= 0.015
        checks[f"animal_{sprite.index}.ends_at_its_matching_outline"] = abs(trajectory[-1] - 1) <= 0.015
        checks[f"animal_{sprite.index}.shortest_segment_monotonic"] = all(
            b >= a - 0.015 for a, b in zip(trajectory, trajectory[1:])
        )
        departures.append(next((i for i, value in enumerate(trajectory) if value > 0.025), len(trajectory)))
        arrivals.append(next((i for i, value in enumerate(trajectory) if value >= 0.985), len(trajectory)))
    checks["animals_depart_together"] = max(departures) - min(departures) <= 1
    checks["animals_arrive_together"] = max(arrivals) - min(arrivals) <= 1
    return _result(
        checks,
        {
            "family": "animal_face_bijection",
            "animal_count": len(sprites),
            "local_features_per_animal": [len(sprite.features) for sprite in sprites],
            "progress": trajectories,
            "departures": departures,
            "arrivals": arrivals,
            "observation_scale": list(size),
            "failures": failures,
            "ocr_required": False,
        },
    )
