"""A moving occluder preserves every stationary object's geometry and paint."""

import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.vision.bandwidth import original_image
from evaluation.cross_benchmark.vbvr_pro.vision.masks import color_mask, ink, no_added_components


def _visible_ink_matches(actual, expected, visible, observation_pixel_area):
    # Apply the visibility boundary after looking for contour support. A
    # matching contour can straddle that boundary by a few raster pixels.
    # A one-pixel discrepancy cannot resolve a change to an exposed tip.
    neighbourhood = np.ones((7, 7), np.uint8)
    for observed, other in ((actual, expected), (expected, actual)):
        exposed = observed & visible
        supported = exposed & cv2.dilate(other, neighbourhood)
        area = cv2.countNonZero(exposed)
        if area - cv2.countNonZero(supported) > max(observation_pixel_area, 0.03 * area):
            return False
    return True


def verify(frames, first, final, metadata):
    """Read the actual mask pose, geometry and visible original paint per frame."""
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result

    p = metadata["parameters"]
    spec = p["mask_rect"]
    color = spec["fill_rgb"]
    x, w, h = int(spec["x"]), int(spec["width"]), int(spec["height"])
    checks = {}
    positions = []
    expected = ink(final)
    gold = original_image(final)
    # The trusted lossless role supplies each original paint pixel. The
    # eroded interior excludes antialiasing and 4:2:0 colour boundaries; it
    # never learns a replacement colour from the candidate being judged.
    neighbourhood = np.ones((7, 7), np.uint8)
    local_range = cv2.dilate(gold, neighbourhood).astype(np.int16) - cv2.erode(gold, neighbourhood).astype(np.int16)
    # Adjacent differently coloured actors can touch. Eroding their union
    # would retain the chroma boundary between them as an opaque colour core.
    paint_core = ink(gold) & (np.max(local_range, axis=2) <= 12).astype(np.uint8)
    phase = max(1, round(first.shape[1] / 512))
    # The opaque interior ends before the moving boundary's compositing ramp.
    # Check exposed geometry outside that narrow transition.
    edge_margin = 6 * phase
    colour_changes = []
    mixed_occlusion = []
    for index, frame in enumerate(frames):
        mask = color_mask(frame, color, tolerance=24)
        row_coverage = mask[:, x + 6 : x + w - 6].mean(axis=1)
        rows = np.nonzero(row_coverage > 0.85)[0]
        occluder = np.zeros(mask.shape, np.uint8)
        if len(rows) > 0:
            by, bottom = int(rows[0]), int(rows[-1] + 1)
            inset = 2 if bottom - by > 4 else 0
            columns = np.nonzero(mask[by + inset : bottom - inset].mean(axis=0) > 0.85)[0]
            positions.append(by)
            # A few matching horizontal edges do not prove a filled rectangle.
            # With no supported column, retain the failed predicates and leave
            # the scene visible to all independent object/content checks.
            checks[f"frame_{index}.occluder_geometry"] = False
            checks[f"frame_{index}.occluder_in_front"] = False
            if len(columns) > 0:
                bx, right = int(columns[0]), int(columns[-1] + 1)
                bw, bh = right - bx, bottom - by
                checks[f"frame_{index}.occluder_geometry"] = (
                    abs(bx - x) <= 4 and abs(bw - w) <= 5 and abs(bh - min(h, frame.shape[0] - by)) <= 5
                )
                interior = mask[by + inset : bottom - inset, bx + 2 : right - 2]
                checks[f"frame_{index}.occluder_in_front"] = bool(interior.size and interior.mean() >= 0.97)
                occluder[max(0, by - edge_margin) : bottom + edge_margin, max(0, bx - 6) : right + 6] = 1
        else:
            positions.append(frame.shape[0])
        visible = 1 - occluder
        # Moving mask edges acquire a short compositing ramp in generated
        # video. Its exposed pixels mix the original paint with the declared
        # occluder paint; that is partial occlusion, not an object recolour.
        # Require both a local edge band and this independently fixed RGB ray.
        edge_band = cv2.dilate(occluder, np.ones((17, 17), np.uint8)).astype(bool)
        for obj in p["objects"]:
            cx, cy = obj["position"]
            # size is the declared side/diameter, not a radius. A 2.4-size
            # window includes neighbouring objects: their remaining codec
            # fringe can otherwise fail the identity of a fully hidden actor.
            # Keep this actor's complete declared bounds plus the observation
            # edge; the independent whole-scene checks still cover neighbours.
            half = int(np.ceil(obj["size"] / 2)) + 4
            x0, x1 = max(0, cx - half), min(frame.shape[1], cx + half)
            y0, y1 = max(0, cy - half), min(frame.shape[0], cy + half)
            roi = np.s_[y0:y1, x0:x1]
            checks[f"frame_{index}.object_{obj['id']}_unchanged_when_visible"] = _visible_ink_matches(
                ink(frame[roi], background=[255, 255, 255]), expected[roi], visible[roi], phase**2
            )
            stable_visible = paint_core[y0:y1, x0:x1] & visible[y0:y1, x0:x1]
            observed = frame[y0:y1, x0:x1].astype(np.float32)
            original = gold[y0:y1, x0:x1].astype(np.float32)
            direction = np.asarray(color[::-1], np.float32) - original
            alpha = np.sum((observed - original) * direction, axis=2) / np.maximum(1, np.sum(direction**2, axis=2))
            residual = np.linalg.norm(observed - (original + alpha[..., None] * direction), axis=2)
            blend = edge_band[y0:y1, x0:x1] & (alpha >= 0) & (alpha <= 1) & (residual <= 36)
            difference = np.linalg.norm(observed - original, axis=2)
            wrong_colour = ((difference > 36) & stable_visible.astype(bool) & ~blend).astype(np.uint8)
            if np.any((difference > 36) & stable_visible.astype(bool) & blend):
                mixed_occlusion.append({"frame": index, "object": obj["id"]})
            # A connected changed patch decides this local obligation even
            # when it occupies a small fraction of a large legitimate actor.
            # A codec-wide fringe has no resolved paint interior. Require a
            # changed component to retain a core beyond the one-pixel contour
            # phase on the 512px observation grid, as well as its local area.
            core = cv2.erode(wrong_colour, np.ones((2 * phase + 1, 2 * phase + 1), np.uint8))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(wrong_colour, 8)
            supported = set(np.unique(labels[core.astype(bool)]).tolist()) - {0}
            boxes = [
                list(map(int, stats[label]))
                for label in range(1, count)
                if label in supported and stats[label, cv2.CC_STAT_AREA] >= 16
            ]
            same_colour, detail = not boxes, {"added_component_boxes_xywh_area": boxes}
            checks[f"frame_{index}.object_{obj['id']}_colour_preserved_when_visible"] = same_colour
            if not same_colour:
                colour_changes.append({"frame": index, "object": obj["id"], "crop_origin": [x0, y0], **detail})
        foreground = ink(frame)
        checks[f"frame_{index}.no_other_moving_content"] = _visible_ink_matches(foreground, expected, visible, phase**2)
        checks[f"frame_{index}.no_isolated_added_content"] = no_added_components(foreground, expected | occluder)[0]
    checks["mask_starts_above_objects"] = abs(positions[0] - p["mask_y_initial"]) <= 4
    checks["mask_exits_canvas"] = positions[-1] >= frame.shape[0] - 3
    checks["vertical_downward_continuity"] = all(
        -3 <= b - a <= max(60, frame.shape[0] * 0.25) for a, b in zip(positions, positions[1:])
    )
    checks["occlusion_actually_occurs"] = any(frame.shape[0] * 0.30 < y < frame.shape[0] * 0.65 for y in positions)
    return _result(
        checks,
        {
            "mask_top_y": positions,
            "visible_colour_changes": colour_changes,
            "partial_occlusion_observations": mixed_occlusion,
        },
    )
