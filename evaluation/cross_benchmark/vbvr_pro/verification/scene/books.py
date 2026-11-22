import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.vision.masks import (
    anchored_foreground,
    color_mask,
    ink,
    mask_coverage,
    no_added_components,
    no_unexpected_ink,
)
from evaluation.cross_benchmark.vbvr_pro.vision.objects import assign_objects, merge_objects, object_checks, palette_objects


def _books(frames, first, final, metadata):
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _palette, _result

    palette = _palette(metadata)
    objects = metadata["parameters"]["objects"]
    width = metadata["parameters"]["visual_props"]["book_width"]
    layout = metadata["parameters"]["layout_snapshot"]
    shelf = layout["shelf_y"]

    def books(frame):
        output = []
        for index in range(len(layout["positions"])):
            left = round(layout["x_start"] + index * (width + layout["spacing"]))
            patch = frame[: int(shelf) - 3, left : left + width]
            merged = merge_objects(palette_objects(patch, palette, minimum=24))
            if merged is not None:
                merged.center = (merged.center[0] + left, merged.center[1])
                x, y, w, h = merged.bbox
                merged.bbox = (x + left, y, w, h)
                output.append(merged)
        return output

    actual, expected = books(frames[-1]), books(final)
    checks = {"exact_book_inventory": len(actual) == len(expected) == len(objects)}
    evidence = {"observed_books": len(actual), "declared_books": len(objects)}
    for i, j in assign_objects(actual, expected):
        from evaluation.cross_benchmark.vbvr_pro.vision.bandwidth import match_local

        local = object_checks(actual[j], expected[i])
        local["height_unchanged"] = abs(actual[j].bbox[3] - expected[i].bbox[3]) <= max(3, expected[i].bbox[3] * 0.02)
        if not all(local.values()) and match_local(frames[-1], final, expected[i], palette, actual_object=actual[j]):
            local = {name: True for name in local}
        checks.update({f"book_{i}.{key}": value for key, value in local.items()})
    old_color = next(o["color"] for o in objects if o["type"] == "existing")
    new_color = next(o["color"] for o in objects if o["type"] == "new")
    source = color_mask(first, old_color)
    initial_new = color_mask(first, new_color)
    final_new = color_mask(final, new_color)
    checks["insertion_process_observed"] = any(
        min(mask_coverage(color_mask(frame, new_color), initial_new, radius=3)) < 0.90
        and min(mask_coverage(color_mask(frame, new_color), final_new, radius=3)) < 0.90
        for frame in frames
    )
    checks["existing_books_remain_at_original_positions"] = True
    for frame_index, frame in enumerate(frames):
        covered = cv2.dilate(color_mask(frame, new_color), np.ones((7, 7), np.uint8))
        a = color_mask(frame, old_color) * (1 - covered)
        e = source * (1 - covered)
        checks["existing_books_remain_at_original_positions"] &= min(mask_coverage(a, e, radius=3)) >= 0.99
        footprint = ink(first) * (1 - cv2.dilate(initial_new, np.ones((7, 7), np.uint8)))
        parts = palette_objects(frame, [new_color], minimum=24)
        moving = sorted(parts, key=lambda obj: obj.pixels, reverse=True)[: sum(o["type"] == "new" for o in objects)]
        for obj in moving:
            x, y, w, h = obj.bbox
            footprint[y : y + h, x : x + w] |= obj.silhouette
        checks[f"frame_{frame_index}.no_isolated_added_content"] = no_added_components(
            ink(frame), anchored_foreground(ink(frame), footprint, radius=5), radius=5
        )[0]
    row = sorted(actual, key=lambda obj: obj.center[0])
    silhouette = ink(frames[-1])
    top = min(obj.bbox[1] for obj in row) - 6 if row else 0
    baseline = round(np.median([obj.bbox[1] + obj.bbox[3] for obj in row])) if row else 0
    heights = []
    for obj in row:
        cx = round(obj.center[0])
        occupied = np.nonzero(np.mean(silhouette[max(0, top) : baseline - 3, cx - 1 : cx + 2], axis=1) > 0.6)[0]
        heights.append(baseline - (max(0, top) + int(occupied[0])) if len(occupied) else 0)
    checks["books_sorted_by_height"] = all(a <= b + 2 for a, b in zip(heights, heights[1:]))
    evidence["visible_book_heights"] = heights
    bottoms = [o.bbox[1] + o.bbox[3] for o in row]
    checks["all_books_stand_on_shelf"] = bool(bottoms and np.ptp(bottoms) <= 5)
    checks["no_unrequested_edits"] = no_unexpected_ink(frames[-1], final)[0]
    return _result(checks, evidence)
