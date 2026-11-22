import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from evaluation.cross_benchmark.vbvr_pro.vision.masks import anchored_foreground, ink, mask_coverage, no_added_components, no_unexpected_ink
from evaluation.cross_benchmark.vbvr_pro.vision.shapes import normalized


def _faces(frame, baseline):
    mask = (np.linalg.norm(frame.astype(float) - 255, axis=2) > 18).astype(np.uint8)
    mask[max(0, baseline - 3) : baseline + 4] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = []
    for k in range(1, n):
        x, y, w, h, area = map(int, stats[k])
        if area < 200 or min(w, h) < 10:
            continue
        local = (labels[y : y + h, x : x + w] == k).astype(np.uint8)
        result.append(
            {
                "bbox": (x, y, w, h),
                "mask": local,
                "rgb": cv2.resize(frame[y : y + h, x : x + w], (96, 96)),
                "pixels": area,
            }
        )
    # A face can have detached eyes or a mouth inside the outer face contour.
    # They belong to that animal, rather than creating extra animal identities.
    merged = []
    for obj in sorted(result, key=lambda item: item["pixels"], reverse=True):
        x, y, w, h = obj["bbox"]
        container = next(
            (
                other
                for other in merged
                if other["bbox"][0] <= x
                and other["bbox"][1] <= y
                and x + w <= other["bbox"][0] + other["bbox"][2]
                and y + h <= other["bbox"][1] + other["bbox"][3]
            ),
            None,
        )
        if container is None:
            merged.append(obj)
        else:
            ox, oy, _, _ = container["bbox"]
            container["mask"][y - oy : y - oy + h, x - ox : x - ox + w] |= obj["mask"]
            container["pixels"] += obj["pixels"]
    return merged


def _face_color_error(actual, expected, foreground, *, details=False, phase=2):
    # Normalizing independently observed bounding boxes introduces a small
    # subpixel shift at a high-contrast facial contour. Match those contours
    # locally in both directions while preserving colour in flat interiors.
    errors, regions = [], []
    for direction, (source, target) in enumerate(((actual, expected), (expected, actual))):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        direct = np.sum((source - target) ** 2, axis=2)
        gradient = np.zeros(source.shape[:2], np.uint8)
        gradient[1:] |= (np.linalg.norm(np.diff(target, axis=0), axis=2) > 20).astype(np.uint8)
        gradient[:, 1:] |= (np.linalg.norm(np.diff(target, axis=1), axis=2) > 20).astype(np.uint8)
        contour = cv2.dilate(gradient, np.ones((2 * phase + 1, 2 * phase + 1), np.uint8)).astype(bool)
        padded = np.pad(target, ((phase, phase), (phase, phase), (0, 0)), constant_values=255)
        nearest = direct.copy()
        for dy in range(2 * phase + 1):
            for dx in range(2 * phase + 1):
                shifted = padded[dy : dy + source.shape[0], dx : dx + source.shape[1]]
                nearest = np.minimum(nearest, np.sum((source - shifted) ** 2, axis=2))
        direct[contour] = nearest[contour]
        wrong = ((direct > 65**2) & foreground).astype(np.uint8)
        errors.append(float(wrong[foreground].mean()))
        if details:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(wrong, 8)
            # A resolved paint patch has an interior. Thin contour/codec fringes
            # are assessed through their independently retained dark feature below.
            core = cv2.erode(wrong, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            supported = set(np.unique(labels[core]).tolist())
            for index, box in enumerate(stats[1:count], 1):
                if box[cv2.CC_STAT_AREA] >= 16 and index in supported:
                    regions.append({"direction": direction, "bbox_xywh_area": list(map(int, box))})
    error = max(errors)
    return (error, regions) if details else error


def _face_identity_views(actual_frame, expected_frame, actual, expected):
    """Align complete actors in one physical canvas before normalising once.

    Independently stretching two codec-dependent bounding boxes moves a small
    face's ears and eyes. A common scale preserves their geometry and includes
    all observed foreground, including additions beyond the original outline.
    """
    width = max(actual["bbox"][2], expected["bbox"][2]) + 4
    height = max(actual["bbox"][3], expected["bbox"][3]) + 4
    images, masks = [], []
    for frame, obj in ((actual_frame, actual), (expected_frame, expected)):
        x, y, w, h = obj["bbox"]
        left, top = (width - w) // 2, (height - h) // 2
        silhouette = np.zeros_like(obj["mask"])
        contours, _ = cv2.findContours(obj["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(silhouette, contours, -1, 1, cv2.FILLED)
        canvas = np.full((height, width, 3), 255, np.uint8)
        patch = canvas[top : top + h, left : left + w]
        patch[silhouette.astype(bool)] = frame[y : y + h, x : x + w][silhouette.astype(bool)]
        mask = np.zeros((height, width), np.uint8)
        mask[top : top + h, left : left + w] = silhouette
        images.append(cv2.resize(canvas, (96, 96), interpolation=cv2.INTER_AREA))
        masks.append(cv2.resize(mask, (96, 96), interpolation=cv2.INTER_NEAREST))
    return images[0], images[1], (masks[0] | masks[1]).astype(bool)


def _face_feature_support(actual, expected, phase):
    """Keep each readable interior dark feature, including a thin mouth line."""
    expected_shape = (np.linalg.norm(expected.astype(float) - 255, axis=2) > 18).astype(np.uint8)
    contours, _ = cv2.findContours(expected_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    silhouette = np.zeros_like(expected_shape)
    cv2.drawContours(silhouette, contours, -1, 1, cv2.FILLED)
    interior = cv2.erode(silhouette, np.ones((7, 7), np.uint8)).astype(bool)
    expected_gray = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)
    body_pixels = interior & (expected_shape > 0)
    if not np.any(body_pixels):
        return []
    body_y = float(np.median(expected_gray[body_pixels]))
    expected_dark = (expected_gray < min(120, body_y - 30)) & (np.ptp(expected.astype(float), axis=2) < 65)
    required = expected_dark & interior
    # Chroma subsampling tints a neutral black stroke with its surrounding
    # skin. Require its local RGB witness, rather than a newly segmented dark
    # class from the encoded image.
    ys, xs = np.nonzero(required)
    if not len(xs):
        return []
    target = np.pad(actual.astype(np.float32), ((phase, phase), (phase, phase), (0, 0)), constant_values=255)
    pixels = expected[ys, xs].astype(np.float32)
    difference = np.full(len(xs), np.inf, np.float32)
    for dy in range(2 * phase + 1):
        for dx in range(2 * phase + 1):
            shifted = target[ys + dy, xs + dx]
            difference = np.minimum(difference, np.sum((pixels - shifted) ** 2, axis=1))
    near = np.zeros(required.shape, bool)
    near[ys, xs] = difference <= 65**2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(required.astype(np.uint8), 8)
    missing = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 8:
            continue
        covered = float(np.mean(near[labels == index]))
        if covered < 0.70:
            missing.append({"bbox_xywh_area": list(map(int, stats[index])), "retained_fraction": covered})
    return missing


def _face_identity_measurements(actual_frame, expected_frame, actual, expected):
    images = _face_identity_views(actual_frame, expected_frame, actual, expected)
    # Registration/contour phase is local to the common 96px view. Flat
    # interiors still require direct colour agreement, and an absent feature
    # has no corresponding dark support anywhere in this small neighbourhood.
    phase = 4
    error, regions = _face_color_error(*images, details=True, phase=phase)
    missing = _face_feature_support(images[0], images[1], phase)
    return error <= 0.10 and not regions and not missing, {
        "colour_error_fraction": error,
        "unmatched_feature_regions": regions,
        "missing_dark_features": missing,
        "contour_phase": phase,
    }


def _separate_face_boxes(parts, radius):
    for index, a in enumerate(parts):
        ax, ay, aw, ah = a["bbox"]
        for b in parts[index + 1 :]:
            bx, by, bw, bh = b["bbox"]
            dx = max(0, ax - (bx + bw), bx - (ax + aw))
            dy = max(0, ay - (by + bh), by - (ay + ah))
            if np.hypot(dx, dy) <= radius:
                return False
    return True


def _animal_sort(frames, first, final, metadata):
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result

    p = metadata["parameters"]
    baseline = int(p["baseline_y"])
    actual, expected = _faces(frames[-1], baseline), _faces(final, baseline)
    checks = {"every_animal_present_exactly_once": len(actual) == len(expected) == p["num_animals"]}
    if not checks["every_animal_present_exactly_once"]:
        return _result(checks, {"observed_count": len(actual), "expected_count": len(expected)})
    cost = np.zeros((len(expected), len(actual)))
    for i, e in enumerate(expected):
        for j, a in enumerate(actual):
            precision, recall = mask_coverage(normalized(a["mask"]), normalized(e["mask"]))
            foreground = (normalized(a["mask"]) | normalized(e["mask"])).astype(bool)
            color = _face_color_error(a["rgb"], e["rgb"], foreground)
            cost[i, j] = 5 * (2 - precision - recall) + color + abs(np.log(a["pixels"] / e["pixels"]))
    rows, cols = linear_sum_assignment(cost)
    identity, identity_evidence = [], {}
    for i, j in zip(rows, cols):
        a, e = actual[j], expected[i]
        aw, ah = a["bbox"][2:]
        ew, eh = e["bbox"][2:]
        preserved, detail = _face_identity_measurements(frames[-1], final, a, e)
        checks[f"animal_{i}.identity_and_color"] = preserved
        identity_evidence[f"animal_{i}"] = detail
        checks[f"animal_{i}.absolute_size"] = max(abs(aw / ew - 1), abs(ah / eh - 1)) <= 0.09
        checks[f"animal_{i}.shape"] = min(mask_coverage(normalized(a["mask"]), normalized(e["mask"]))) >= 0.96
        # Animal drawings have type-dependent margins inside their nominal
        # square. Preserve the reference's per-animal baseline offset while
        # allowing any horizontal spacing.
        checks[f"animal_{i}.on_baseline"] = abs(a["bbox"][1] + ah - (e["bbox"][1] + eh)) <= 10
        ec = np.array([e["bbox"][0] + ew / 2, e["bbox"][1] + eh / 2])
        index = min(
            range(len(p["animals"])), key=lambda k: np.linalg.norm(ec - np.array(p["animals"][k]["output_pos"]))
        )
        identity.append((a["bbox"][0], p["animals"][index]["size"]))
    sizes = [size for _, size in sorted(identity)]
    checks["requested_size_order"] = sizes == sorted(sizes, reverse=p["sort_order"] == "descending")
    ordered = sorted(actual, key=lambda obj: obj["bbox"][0])
    checks["animals_do_not_overlap"] = all(
        a["bbox"][0] + a["bbox"][2] <= b["bbox"][0] + 2 for a, b in zip(ordered, ordered[1:])
    )
    checks["baseline_preserved_throughout"] = all(
        no_unexpected_ink(f[max(0, baseline - 3) : baseline + 4], first[max(0, baseline - 3) : baseline + 4])[0]
        for f in frames
    )
    fixed = np.zeros(first.shape[:2], np.uint8)
    fixed[max(0, baseline - 3) : baseline + 4] = ink(first)[max(0, baseline - 3) : baseline + 4]
    smallest = min(obj["pixels"] for obj in expected)
    for frame_index, frame in enumerate(frames):
        # Read visible animal components at their actual locations. Overlapping
        # animals may form one component, and the route is not prescribed.
        parts = [obj for obj in _faces(frame, baseline) if obj["pixels"] >= smallest * 0.5]
        parts = sorted(parts, key=lambda obj: obj["pixels"], reverse=True)[: p["num_animals"]]
        # Clearly separated actors expose complete faces. Their identities
        # are invariant during free translation, regardless of route or speed.
        # Contact/merged components are not evidence of a missing hidden eye.
        if len(parts) == p["num_animals"] and _separate_face_boxes(parts, max(3, frame.shape[1] / 512 * 3)):
            local_cost = np.array(
                [
                    [
                        _face_color_error(
                            a["rgb"], e["rgb"], (normalized(a["mask"]) | normalized(e["mask"])).astype(bool)
                        )
                        + abs(np.log(a["pixels"] / e["pixels"]))
                        for a in parts
                    ]
                    for e in expected
                ]
            )
            expected_ids, actual_ids = linear_sum_assignment(local_cost)
            for ei, ai in zip(expected_ids, actual_ids):
                preserved, detail = _face_identity_measurements(frame, final, parts[ai], expected[ei])
                checks[f"frame_{frame_index}.animal_{ei}.visible_identity_preserved"] = preserved
                if not preserved:
                    identity_evidence[f"frame_{frame_index}.animal_{ei}"] = detail
        footprint = fixed.copy()
        for obj in parts:
            x, y, w, h = obj["bbox"]
            silhouette = np.zeros_like(obj["mask"])
            contours, _ = cv2.findContours(obj["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(silhouette, contours, -1, 1, cv2.FILLED)
            footprint[y : y + h, x : x + w] |= silhouette
        foreground = ink(frame)
        footprint = anchored_foreground(foreground, footprint, radius=4)
        checks[f"frame_{frame_index}.no_isolated_added_content"] = no_added_components(foreground, footprint, radius=4)[
            0
        ]
    return _result(checks, {"left_to_right_sizes": sizes, "face_identity": identity_evidence})
