from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _match_scene, _palette, _result
import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.vision.bandwidth import (
    register_equivalent,
    observation_frame,
)
from evaluation.cross_benchmark.vbvr_pro.vision.masks import ink, no_unexpected_ink
from evaluation.cross_benchmark.vbvr_pro.vision.objects import palette_objects


def _color_then_move_process(frames, first, final, metadata):
    """Observe O-11's ordered states, without replaying a reference route.

    E must reach the target paint before F is displaced from E's baseline.
    A stationary F preview is legal, as is a single discrete move after E.
    Once displaced, F must retain the target paint rather than recolor again.
    """
    roles = {obj.get("label"): obj for obj in metadata.get("semantic_ground_truth", {}).get("objects", [])}
    missing = [label for label in "ABCDEF" if label not in roles or "center" not in roles[label]]
    if missing:
        raise ValueError(f"O-11 reference roles lack centers: {', '.join(missing)}")
    scale = min(1, 512 / first.shape[1])
    size = (round(first.shape[1] * scale), round(first.shape[0] * scale))

    def small(image):
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image

    source, goal = small(first), small(final)
    centers = {label: np.array(obj["center"], dtype=float) * scale for label, obj in roles.items()}
    palette = _palette(metadata)
    given = palette_objects(source, palette)
    answers = palette_objects(goal, palette)
    if len(given) < 4 or len(answers) < 6:
        raise ValueError("O-11 reference must expose four given objects and six completed objects")
    fixed = {
        label: min(given, key=lambda obj: np.linalg.norm(np.subtract(obj.center, centers[label]))) for label in "ABCD"
    }
    target = min(answers, key=lambda obj: np.linalg.norm(np.subtract(obj.center, centers["E"])))
    background = np.median(np.concatenate([source[:8].reshape(-1, 3), source[-8:].reshape(-1, 3)]), axis=0)

    def core(mask):
        # A tight object's bbox is not foreground beyond its edges. Padding
        # makes that boundary explicit; erode's default high border otherwise
        # leaves a filled rectangle's entire edge in the supposed interior.
        padded = np.pad(mask.astype(np.uint8), 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        # Read paint beyond the outline, 4:2:0 mixing and resize-filter edge
        # support on the 512px grid. Thin symbols use their deepest observed
        # support, never a fallback to all boundary pixels.
        interior = distance > 4
        return interior if np.any(interior) else distance == distance.max()

    def pixels(image, obj):
        x, y, w, h = obj.bbox
        return image[y : y + h, x : x + w][core(obj.mask)]

    target_paint = np.median(pixels(goal, target), axis=0)

    def paint_matches(values, paint, fade=False):
        values = values.astype(float)
        if fade:
            # An opaque interior and that same paint fading from the canvas
            # lie on one RGB ray. Source-to-target recoloring generally does
            # not; nearest-palette hue masks can conflate different paints.
            ray = paint - background
            alpha = ((values - background) @ ray) / max(1, float(ray @ ray))
            error = np.linalg.norm(values - (background + alpha[:, None] * ray), axis=1)
            return bool(np.mean((error <= 18) & (alpha >= 0) & (alpha <= 1.15)) >= 0.9)
        return bool(np.mean(np.linalg.norm(values - paint, axis=1) <= 18) >= 0.9)

    fixed_paints = {label: np.median(pixels(source, obj), axis=0) for label, obj in fixed.items()}
    boundaries = [0, (centers["D"][0] + centers["E"][0]) / 2, (centers["E"][0] + centers["F"][0]) / 2, size[0]]

    def actor(image, column):
        x0, x1 = round(boundaries[column]), round(boundaries[column + 1])
        y0 = size[1] // 2
        cut = image[y0:, x0:x1]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink(cut))
        # Ignore the small neutral question mark and arrow, not unfamiliar
        # intermediate paints. Geometry and color are independent readings.
        candidates = [
            index
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= target.pixels * 0.4
            and stats[index, cv2.CC_STAT_WIDTH] >= target.bbox[2] * 0.5
            and stats[index, cv2.CC_STAT_HEIGHT] >= target.bbox[3] * 0.5
        ]
        if not candidates:
            return None
        index = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        x, y, w, h = stats[index, :4]
        return np.array([x0 + x + (w - 1) / 2, y0 + y + (h - 1) / 2]), cut[core(labels == index)]

    completed, moved, wrong_paint, fixed_changes = [], [], [], []
    e_retained = True
    for index, frame in enumerate(frames):
        image = small(frame)
        for label, obj in fixed.items():
            if not paint_matches(pixels(image, obj), fixed_paints[label]):
                fixed_changes.append({"frame": index, "label": label})
        e, f = actor(image, 1), actor(image, 2)
        e_ready = e is not None and np.linalg.norm(e[0] - centers["E"]) <= 2 and paint_matches(e[1], target_paint)
        if e_ready:
            completed.append(index)
        # Local raster/codec tolerance on the 512px observation grid is
        # independent of the required displacement and reference timing.
        if f is not None and abs(f[0][1] - centers["E"][1]) > 1.5:
            moved.append(index)
            e_retained &= e_ready
            if not paint_matches(f[1], target_paint, fade=True):
                wrong_paint.append(index)
    return {
        "color_then_move.given_paints_preserved_throughout": not fixed_changes,
        "color_then_move.recolor_completed_before_displacement": bool(completed and moved and completed[0] < moved[0]),
        "color_then_move.completed_e_retained_during_displacement": bool(moved and e_retained),
        "color_then_move.displaced_f_keeps_target_paint": bool(moved and not wrong_paint),
    }, {
        "color_then_move": {
            "first_completed_e_frame": completed[0] if completed else None,
            "first_displaced_f_frame": moved[0] if moved else None,
            "displaced_f_wrong_paint_frames": wrong_paint,
            "given_paint_changes": fixed_changes,
        }
    }


def _scale_answer_template(actual, final, checks, actual_objects, expected_objects, matches, roles, columns):
    """Register a correctly sized answer glyph to its observed footprint."""
    resized = final.copy()
    background = tuple(float(value) for value in np.median(final[:8].reshape(-1, 3), axis=0))
    for i, j in matches:
        # Only the answer cells whose role calls for a scale change use the
        # size category. Givens and unchanged-size answers keep their geometry.
        if i <= columns or roles[i]["scale"] == roles[columns]["scale"]:
            continue
        prefix = f"object_{i}."
        if all(checks[prefix + key] for key in ("size", "complete_foreground_support")):
            continue
        if not all(checks[prefix + key] for key in ("color", "shape", "fill_and_stroke", "position")):
            continue
        observed, expected = actual_objects[j], expected_objects[i]
        scales = np.asarray(observed.bbox[2:]) / expected.bbox[2:]
        scale = float(np.sqrt(np.prod(scales)))
        requested, source = roles[i]["scale"], roles[columns]["scale"]
        midpoint = (requested + source) / (2 * requested)
        # The selected size must remain nearer to the requested role than to
        # the unscaled operand. Also bound undersizing and aspect distortion.
        if not 0.75 <= scale <= 1.25 or (scale - midpoint) * (source - requested) >= 0:
            continue
        if max(scales) > 1.25 * min(scales):
            continue
        ax, ay, aw, ah = observed.bbox
        ex, ey, ew, eh = expected.bbox
        left, top = max(0, min(ax, ex) - 6), max(0, min(ay, ey) - 6)
        right, bottom = max(ax + aw, ex + ew) + 6, max(ay + ah, ey + eh) + 6
        reference = final[top:bottom, left:right]
        # Keep the antialiased outline with the glyph. The new footprint is
        # derived from the reference, so attached additions stay observable.
        offset = (np.subtract(observed.center, (left, top))
                  - scales * np.subtract(expected.center, (left, top)))
        affine = np.column_stack((np.diag(scales), offset))
        adjusted = cv2.warpAffine(reference, affine, reference.shape[1::-1], borderValue=background)
        candidate = actual[top:bottom, left:right]
        if no_unexpected_ink(candidate, adjusted)[0]:
            resized[top:bottom, left:right] = adjusted
            checks[prefix + "size"] = True
            checks[prefix + "complete_foreground_support"] = True
    if checks.get("exact_object_count") and all(
        value for key, value in checks.items() if key != "original_labels_have_distinct_observed_support"
    ):
        checks.pop("original_labels_have_distinct_observed_support", None)
    return resized


def _analogy(frames, first, final, metadata, reference, *, color_then_move=False):
    palette = _palette(metadata)
    objects = metadata.get("semantic_ground_truth", {}).get("objects", [])
    count = len(objects)
    columns = 2 if count == 4 else 3
    height, width = final.shape[:2]
    cells = [
        (c * width / columns, r * height / 2, (c + 1) * width / columns, (r + 1) * height / 2)
        for r in range(2)
        for c in range(columns)
    ]
    checks, evidence, observed, expected, matches = _match_scene(frames[-1], final, palette, cells=cells)
    goal = final
    if (
        count == len(expected) == 6
        and metadata.get("parameters", {}).get("transformation_type") in {"color_then_scale", "scale_then_outline"}
    ):
        goal = _scale_answer_template(frames[-1], final, checks, observed, expected, matches, objects, columns)
    if objects:
        checks["declared_object_inventory"] = (
            evidence["expected_count"] == len(objects) and checks["exact_object_count"]
        )
    guard, detail = no_unexpected_ink(frames[-1], goal)
    checks["no_unrequested_scene_edits"] = guard
    evidence.update(detail)
    # The top example row is part of the problem and must be preserved for
    # the entire solution, not just restored at the final frame.
    h = first.shape[0] // 2
    checks["example_row_preserved_throughout"] = all(no_unexpected_ink(f[:h], first[:h])[0] for f in frames)
    # The bottom-left shape is a given operand, just like the example row.
    # Completing an analogy adds answers in the remaining bottom cells; it
    # does not license erasing or repainting the operand and restoring it.
    given_cells = cells[:columns] + [cells[columns]]
    # Native analogy references change only the answer cells. Their fixed
    # givens retain their original PNG labels, while temporal codec prediction
    # can change a thin stroke's decoded appearance after the first frame.
    # Register those appearances only within the predetermined given cells;
    # the reader still checks full actor support on the complete candidate.
    for x0, y0, x1, y1 in given_cells:
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        source_cell = first[y0:y1, x0:x1]
        for variant in reference[1:]:
            register_equivalent(source_cell, variant[y0:y1, x0:x1])
    changed_givens = []
    for index, frame in enumerate(frames):
        with observation_frame(index):
            observed, _, _, _, _ = _match_scene(frame, first, palette, cells=given_cells)
        if not all(observed.values()):
            changed_givens.append(
                {"frame": index, "failed_checks": [name for name, value in observed.items() if not value]}
            )
            # Later restoration cannot repair a violated given-object contract.
            break
    checks["given_operands_preserved_throughout"] = not changed_givens
    if changed_givens:
        evidence["changed_given_operands"] = changed_givens
    if color_then_move:
        process, detail = _color_then_move_process(frames, first, final, metadata)
        checks.update(process)
        evidence.update(detail)
    return _result(checks, evidence)
