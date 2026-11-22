from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _blobs, _block_mask, _layers, _mask, _result, _static
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from evaluation.cross_benchmark.vbvr_pro.vision.masks import color_mask


def _shift(frames, first, p, final=None, metadata=None):
    n = p["grid_size"]
    dr, dc = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}[p["direction"]]
    initial = np.array(p["positions"], float)
    expected = initial + np.array([dr, dc]) * p["steps"]
    if sorted(map(tuple, expected)) != sorted(map(tuple, p["shifted_positions"])):
        raise ValueError("Invalid benchmark: shifted cells disagree with requested translation")
    h, w = first.shape[:2]
    base = np.array([((col + 0.5) * w / n, (row + 0.5) * h / n) for row, col in initial])
    final_offset = np.array([dc * w / n, dr * h / n]) * p["steps"]
    unit = final_offset / np.dot(final_offset, final_offset)
    progress, shape_areas, max_error = [], [], 0.0
    original = cv2.dilate(_block_mask(first, p["color_rgb"]).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    empty = first.copy()
    background = first[(np.ptp(first.astype(np.int16), axis=2) < 15) & (np.min(first, axis=2) > 235)]
    if not len(background):
        raise ValueError("No visible grid background to reconstruct vacated cells")
    empty[original] = np.median(background, axis=0).astype(np.uint8)
    if final is not None:
        occupied_final = cv2.dilate(
            _block_mask(final, p["color_rgb"]).astype(np.uint8), np.ones((17, 17), np.uint8)
        ).astype(bool)
        known = original & ~occupied_final
        empty[known] = final[known]
    if metadata is not None:
        grid_guard = np.zeros(first.shape[:2], np.uint8)
        for layer in _layers(metadata):
            if layer.get("type") == "line":
                points = np.rint(layer["points_px"]).astype(np.int32)
                cv2.polylines(grid_guard, [points], False, 1, round(layer.get("stroke_width_px", 1)) + 2)
        empty[grid_guard.astype(bool)] = first[grid_guard.astype(bool)]
    static, worst = True, 0
    rectangles = True
    for frame in frames:
        mask = _block_mask(frame, p["color_rgb"])
        soft_paint = None
        blobs = _blobs(mask, 30)
        if len(blobs) != len(initial):
            return _result({"exact_block_count_every_frame": False}, {"observed_blocks": len(blobs)})
        centers = np.array([blob[0] for blob in blobs])
        offset = centers.mean(axis=0) - base.mean(axis=0)
        alpha = float(np.dot(offset, unit))
        wanted = base + alpha * final_offset
        ids, js = linear_sum_assignment(np.linalg.norm(wanted[:, None] - centers[None], axis=2))
        max_error = max(
            max_error,
            float(np.linalg.norm(wanted[ids] - centers[js], axis=1).max()),
            float(np.linalg.norm(offset - alpha * final_offset)),
        )
        progress.append(alpha)
        shape_areas.extend(blob[1] for blob in blobs)
        for _, area, box in blobs:
            x, y, bw, bh = map(int, box)
            local = mask[y : y + bh, x : x + bw].astype(np.uint8)
            # A few codec pixels can extend an otherwise solid square's
            # bounding box into a mostly empty row or column. Measure its
            # body after discarding at most 0.5% colour mass at each edge;
            # retain the complete mask for count, trajectory and scene veto.
            rows, columns = local.sum(axis=1), local.sum(axis=0)
            budget = area * 0.005
            top = np.searchsorted(np.cumsum(rows), budget, side="right")
            bottom = bh - np.searchsorted(np.cumsum(rows[::-1]), budget, side="right")
            left = np.searchsorted(np.cumsum(columns), budget, side="right")
            right = bw - np.searchsorted(np.cumsum(columns[::-1]), budget, side="right")
            local = local[top:bottom, left:right]
            bh, bw = local.shape
            # A colour corner may lose a few edge pixels under 4:2:0. Require
            # a filled square at that registered spatial tolerance, rather
            # than treating every codec notch as another polygon vertex.
            filled = cv2.dilate(local, np.ones((5, 5), np.uint8))
            ph, pw = max(1, round(bh * 0.15)), max(1, round(bw * 0.15))
            corners = [
                local[r, c].mean() for r in (slice(0, ph), slice(-ph, None)) for c in (slice(0, pw), slice(-pw, None))
            ]
            if min(corners) < 0.25:
                # The opaque RGB mask can lose a compressed colour corner
                # although its lower-coverage paint remains visible. Read
                # that actual chroma/luma support; white missing corners and
                # round blocks cannot acquire invented fill by dilation.
                if soft_paint is None:
                    soft_paint = color_mask(frame, p["color_rgb"])
                material = soft_paint[y + top : y + bottom, x + left : x + right]
                corners = [
                    material[r, c].mean()
                    for r in (slice(0, ph), slice(-ph, None))
                    for c in (slice(0, pw), slice(-pw, None))
                ]
            rectangles &= (
                local.mean() >= 0.86 and abs(bw - bh) <= 5 and np.mean(filled) >= 0.98 and min(corners) >= 0.25
            )
        allowed = cv2.dilate(mask.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        preserved, error = _static([frame], empty, allowed)
        static &= preserved
        worst = max(worst, error)
    checks = {
        "all_blocks_translate_together": max_error <= 5,
        "exact_final_displacement": abs(progress[-1] - 1) <= 0.035,
        "no_reverse_or_extra_steps": min(progress) >= -0.035
        and max(progress) <= 1.035
        and min(np.diff(progress), default=0) >= -0.035,
        "block_shapes_preserved": min(shape_areas) >= max(shape_areas) * 0.86,
        "all_blocks_remain_filled_squares": rectangles,
        "grid_and_background_preserved": static,
    }
    return _result(
        checks,
        {
            "maximum_translation_error_px": max_error,
            "final_progress": progress[-1],
            "unexpected_static_component": worst,
        },
    )


def _light_target(p):
    n, kind, rule = p["num_lights"], p["task_type"], p["task_type_params"]
    if kind in (1, 2):
        indices = set(i - 1 for i in rule["positions"])
    elif "pattern" in rule:
        indices = set(range(0 if rule["pattern"] == "odd" else 1, n, 2))
    elif "start" in rule:
        indices = set(range(rule["start"] - 1, rule["end"]))
    elif "count" in rule:
        indices = set(range(rule["count"])) if rule["side"] == "left" else set(range(n - rule["count"], n))
    else:
        raise ValueError("Unrecognized light sequence rule")
    return [int(i in indices) for i in range(n)]


def _lights(frames, first, metadata):
    p = metadata["parameters"]
    target = _light_target(p)
    if target != p["final_states"]:
        raise ValueError("Invalid benchmark: light final state contradicts positional rule")
    circles = sorted(
        (layer for layer in _layers(metadata) if layer.get("kind") == "circle"), key=lambda layer: layer["cx_px"]
    )
    if len(circles) != p["num_lights"]:
        raise ValueError("Reference light observation geometry unavailable")
    allowed = np.zeros(first.shape[:2], np.uint8)
    observations = []
    geometry, observable = True, True
    for circle in circles:
        cv2.circle(allowed, (round(circle["cx_px"]), round(circle["cy_px"])), round(circle["radius_px"] * 1.7), 1, -1)
    for frame in frames:
        state = []
        for light_index, circle in enumerate(circles):
            cx, cy, r = circle["cx_px"], circle["cy_px"], circle["radius_px"] * 0.55
            patch = frame[round(cy - r) : round(cy + r), round(cx - r) : round(cx + r)]
            observed = np.median(patch.reshape(-1, 3), axis=0)
            on = np.max(np.abs(observed - np.array(p["color"][::-1]))) <= 40
            off = np.max(np.abs(observed - 128)) <= 40
            state.append(1 if on else 0 if off else -1)
            start_color = np.array(p["color"][::-1] if p["initial_states"][light_index] else [128] * 3, float)
            end_color = np.array(p["color"][::-1] if target[light_index] else [128] * 3, float)
            vector = end_color - start_color
            alpha = (
                np.clip(np.dot(observed - start_color, vector) / np.dot(vector, vector), 0, 1)
                if np.any(vector)
                else 0.0
            )
            observable &= np.max(np.abs(observed - (start_color + alpha * vector))) <= 35
            radius = circle["radius_px"]
            xa, xb = max(0, round(cx - radius * 1.5)), min(frame.shape[1], round(cx + radius * 1.5) + 1)
            ya, yb = max(0, round(cy - radius * 1.5)), min(frame.shape[0], round(cy + radius * 1.5) + 1)
            yy, xx = np.indices((yb - ya, xb - xa))
            distance = np.sqrt((xx + xa - cx) ** 2 + (yy + ya - cy) ** 2)
            body = _mask(frame[ya:yb, xa:xb], observed[::-1], 30)
            inside = distance < radius * 0.80
            geometry &= np.mean(body[inside]) >= 0.97 and np.count_nonzero(body & (distance > radius + 3)) <= 15
        observations.append(state)
    static, worst = _static(frames, first, allowed.astype(bool))
    checks = {
        "exact_final_light_vector": observations[-1] == target,
        "all_lights_observable": observable,
        "complete_circular_light_bodies": geometry,
        "light_positions_and_scene_preserved": static,
    }
    # A state change may happen at any time; a light already at its target must
    # not change to a third state during the video.
    checks["unchanged_lights_remain_correct"] = all(
        all(state[i] == target[i] for state in observations)
        for i in range(len(target))
        if p["initial_states"][i] == target[i]
    )
    return _result(
        checks,
        {"expected_lights": target, "observed_final_lights": observations[-1], "unexpected_static_component": worst},
    )
