from evaluation.cross_benchmark.vbvr_pro.verification.scene.motion_observations import _coverage, _mask, _overlay_paint, _stationary_support
import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.vision.masks import mask_coverage, no_added_components


def _attention(frames, first, metadata, reference):
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result

    p = metadata["parameters"]
    factor = 512 / first.shape[1]
    size = (512, round(first.shape[0] * factor))
    source = cv2.resize(reference[0] if reference else first, size, interpolation=cv2.INTER_AREA)
    frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
    color = p["box_color_rgb"]
    thickness = max(1, round(p["box_width"] * factor))
    start = np.array(p["start_center"]) * factor
    goal = np.array(p["end_center"]) * factor
    static = source.copy()
    initial_half = round(p["start_box_half"] * factor)
    old_border = np.zeros(static.shape[:2], np.uint8)
    a, b = np.round(start - initial_half).astype(int), np.round(start + initial_half).astype(int)
    cv2.rectangle(old_border, tuple(a), tuple(b), 1, thickness + 4)
    static[old_border.astype(bool)] = 255
    colors = [obj["color"] for obj in p["objects"]] + [color]
    regions = []
    for obj in p["objects"]:
        cx, cy = np.array(obj["center"]) * factor
        half = obj["size"] * factor + 6
        regions.append(
            (
                max(0, round(cx - half)),
                max(0, round(cy - half)),
                min(size[0], round(cx + half)),
                min(size[1], round(cy + half)),
            )
        )
    separation = round(min(p["start_box_half"], p["end_box_half"]) * factor)

    def border_pair(profile):
        first = int(np.argmax(profile))
        remaining = profile.copy()
        remaining[max(0, first - separation) : first + separation + 1] = 0
        second = int(np.argmax(remaining))
        centers = []
        for peak in (first, second):
            lo, hi = max(0, peak - thickness - 1), min(len(profile), peak + thickness + 2)
            weights = profile[lo:hi].copy()
            weights[weights < profile[peak] * 0.8] = 0
            centers.append(round(float(np.arange(lo, hi) @ weights / max(1, weights.sum()))))
        return sorted(centers)

    checks = {}
    centers, halves, rectangles, border_support = [], [], [], []
    source_luma = cv2.cvtColor(static, cv2.COLOR_BGR2GRAY).astype(float)
    actor_support = (np.linalg.norm(static.astype(float) - 255, axis=2) > 15).astype(np.uint8)
    actor_support = cv2.dilate(actor_support, np.ones((3, 3), np.uint8))
    for index, frame in enumerate(frames):
        pixels = _mask(frame, color) | _overlay_paint(frame, static, color)
        # The prompt identifies a green box, rather than prescribing an RGB
        # triplet. Generated green strokes can shift towards teal/olive while
        # keeping that identity; a strict reference-paint ray loses the whole
        # otherwise complete side. Actor paint identity is checked separately.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pixels |= ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] >= 40)).astype(np.uint8)
        # A thin border crossing a coloured actor can retain its luminance
        # edge while 4:2:0/VAE mixing changes its chroma. Read that edge only
        # on the independently fixed actor support. Exposed background sides
        # still require green paint, and off-border actor edits still fail.
        dark_edge = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float) < source_luma - 25
        pixels |= dark_edge.astype(np.uint8) & actor_support
        # Read the moving border's own footprint. A dictionary restricted to
        # integer square half-sizes misreads small raster drift as a broken
        # box, and does not represent a legitimate rectangular box.
        moving = pixels.copy()
        moving[np.linalg.norm(frame.astype(float) - static.astype(float), axis=2) < 36] = 0
        if not np.any(moving):
            checks[f"frame_{index}.visible_attention_rectangle"] = False
            return _result(checks)
        x0, x1 = border_pair(moving.sum(axis=0).astype(float))
        y0, y1 = border_pair(moving.sum(axis=1).astype(float))
        if x1 <= x0 or y1 <= y0:
            checks[f"frame_{index}.visible_attention_rectangle"] = False
            return _result(checks)
        near = cv2.dilate(pixels, np.ones((7, 7), np.uint8))
        sides = [near[y0, x0 : x1 + 1], near[y1, x0 : x1 + 1], near[y0 : y1 + 1, x0], near[y0 : y1 + 1, x1]]
        # Quantise the 3% support budget to actual observed pixels. Applying
        # a fractional threshold directly makes a one-pixel registration
        # phase flip a 131px side with four missing samples (3.93px budget).
        missing = [int(np.count_nonzero(side == 0)) for side in sides]
        budgets = [round(side.size * 0.03) for side in sides]
        complete = all(count <= budget for count, budget in zip(missing, budgets))
        border_support.append({"missing_pixels": missing, "pixel_budgets": budgets})
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        centers.append(center)
        halves.append(((x1 - x0) + (y1 - y0)) / 4)
        rectangles.append([x0, y0, x1, y1])
        checks[f"frame_{index}.one_complete_attention_rectangle"] = complete
        expected = static.copy()
        cv2.rectangle(expected, (x0, y0), (x1, y1), tuple(color[::-1]), thickness)
        local = _coverage(frame, expected, colors, regions)
        border = np.zeros(frame.shape[:2], np.uint8)
        cv2.rectangle(border, (x0, y0), (x1, y1), 1, thickness + 8)
        for object_index, (x0, y0, x1, y1) in enumerate(regions):
            a, e = frame[y0:y1, x0:x1], expected[y0:y1, x0:x1]
            visible = 1 - border[y0:y1, x0:x1]
            outline = p.get("shape_outline_rgb", [70, 70, 70])
            am = _stationary_support(a, colors[object_index], outline) * visible
            em = _stationary_support(e, colors[object_index], outline) * visible
            source_region = static[y0:y1, x0:x1]
            body = (np.linalg.norm(source_region.astype(float) - 255, axis=2) > 15).astype(np.uint8)
            interior = cv2.erode(body, np.ones((7, 7), np.uint8)) * (1 - border[y0:y1, x0:x1])
            errors = np.linalg.norm(a.astype(float) - source_region.astype(float), axis=2)
            unchanged = not np.any(interior) or float(np.mean(errors[interior.astype(bool)] > 40)) <= 0.02
            local[f"object_{object_index}.visible_rigid_shape_and_position"] = (
                min(mask_coverage(am, em, radius=3)) >= 0.97
                and no_added_components(am, em, radius=3)[0]
                and no_added_components(em, am, radius=3)[0]
                and unchanged
            )
        local[f"object_{len(colors) - 1}.visible_rigid_shape_and_position"] = complete
        checks.update({f"frame_{index}.{k}": v for k, v in local.items()})
    centers = np.array(centers)
    travel = float(np.linalg.norm(goal - start))
    checks["attention_starts_on_source_object"] = np.linalg.norm(centers[0] - start) <= 7
    checks["attention_finishes_on_target_object"] = np.linalg.norm(centers[-1] - goal) <= 7
    checks["correct_final_padding"] = abs(halves[-1] - p["end_box_half"] * factor) <= max(
        2, p["end_box_half"] * factor * 0.09
    )
    checks["continuous_box_motion"] = np.max(np.linalg.norm(np.diff(centers, axis=0), axis=1), initial=0) <= max(
        20, travel * 0.35
    )
    checks["intermediate_box_positions_observed"] = bool(
        np.any(
            (np.linalg.norm(centers - start, axis=1) > travel * 0.15)
            & (np.linalg.norm(centers - goal, axis=1) > travel * 0.15)
        )
    )
    return _result(
        checks,
        {
            "attention_centers": centers.tolist(),
            "attention_half_sizes": halves,
            "attention_rectangles": rectangles,
            "attention_border_support": border_support,
        },
    )
