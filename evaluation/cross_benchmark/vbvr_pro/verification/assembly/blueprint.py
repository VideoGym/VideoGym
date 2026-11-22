"""O-21: candidate inspection, fit previews, judgments, and final gap fill."""

import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.vision.masks import (
    anchored_foreground,
    color_mask as _color_mask,
    ink,
    mask_coverage,
    no_added_components,
)
from evaluation.cross_benchmark.vbvr_pro.vision.shapes import normalized


SUPPORTED_IDS = {"O-21"}


def _opaque_mask(frame, rgb, tolerance=36):
    # Fit previews are translucent; the carried piece is opaque. Keep these
    # states distinct instead of treating a same-hue ghost as a moving piece.
    return _color_mask(frame, rgb, tolerance, opaque=True)


def _status(frame, card):
    x, y, w, h = card
    rgb = np.median(frame[y + 20 : y + 40, x + 20 : x + 40].reshape(-1, 3), axis=0)[::-1]
    if rgb[1] > rgb[0] + 25 and rgb[1] > rgb[2] + 25:
        return "accepted"
    if rgb[0] > rgb[1] + 20 and rgb[0] > rgb[2] + 20:
        return "rejected"
    return "pending"


def _active(frame, card):
    x, y, w, h = card
    mask = _opaque_mask(frame[y : y + h, x : x + w], [255, 200, 100], tolerance=50)
    interior = mask.copy()
    interior[5:-5, 5:-5] = 0
    return int(interior.sum()) > 30


def _preview(frame, source, split):
    delta = np.linalg.norm(frame[:split].astype(float) - source[:split].astype(float), axis=2)
    mask = (delta > 15).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def verify(task, frames, first, final, metadata, reference):
    if task.split("_", 1)[0] != "O-21":
        return None
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result

    p = metadata["parameters"]
    semantic = metadata["semantic_ground_truth"]
    color = semantic["structure_color_rgb"]
    correct = p["correct_index"]
    layers = metadata["generic_declarative_render"]["first_frame"]["layers"]
    cards = sorted(
        [
            (int(layer["x_px"]), int(layer["y_px"]), int(layer["w_px"]), int(layer["h_px"]))
            for layer in layers
            if layer.get("kind") == "rect"
            and layer.get("type") == "filled_shape"
            and layer.get("y_px", 0) > first.shape[0] * 0.7
            and layer.get("w_px", 0) > 100
        ]
    )
    if len(cards) != len(p["candidates"]) or not reference:
        return _result({"candidate_geometry_and_reference_available": False})
    split = min(y for x, y, w, h in cards) - 10
    source = reference[0]
    blue_source = _opaque_mask(source, color)
    blue_goal = _opaque_mask(final, color)
    initial_blocks = [
        layer
        for layer in layers
        if layer.get("kind") == "rect" and layer.get("fill_rgb") == color and layer.get("y_px", 0) < split
    ]
    cells = np.array(semantic["structure_cells_grid"])
    px = sorted({layer["x_px"] for layer in initial_blocks})
    py = sorted({layer["y_px"] for layer in initial_blocks})
    gx = sorted(set(cells[:, 0]))
    gy = sorted(set(cells[:, 1]))
    scale_x = float(np.median(np.diff(px) / np.diff(gx)))
    scale_y = float(np.median(np.diff(py) / np.diff(gy)))
    offset = np.array([px[0] - gx[0] * scale_x, py[0] - gy[0] * scale_y])
    block_w, block_h = initial_blocks[0]["w_px"], initial_blocks[0]["h_px"]
    target = np.zeros(first.shape[:2], np.uint8)
    for x, y in semantic["structure_cells_grid"] + semantic["gap_cells_grid"]:
        x, y = np.round(np.array([x * scale_x, y * scale_y]) + offset).astype(int)
        target[y : y + block_h, x : x + block_w] = 1
    checks = {
        "complete_structure_exactly_fills_declared_cells": min(
            mask_coverage(_opaque_mask(frames[-1][:split], color), target[:split], radius=3)
        )
        >= 0.985
    }
    background = np.array(metadata["generic_declarative_render"]["canvas"]["background_rgb"])[::-1]
    for block_index, block in enumerate(initial_blocks):
        x, y, w, h = (int(block[k]) for k in ("x_px", "y_px", "w_px", "h_px"))
        checks[f"structure_block_{block_index}.present_throughout"] = all(
            float(
                np.mean(np.linalg.norm(f[y + 4 : y + h - 4, x + 4 : x + w - 4].astype(float) - background, axis=2) > 30)
            )
            >= 0.95
            for f in frames
        )
    gap = np.array(semantic["gap_cells_grid"])
    normalized_gap = {tuple(q) for q in gap - gap.min(axis=0)}
    candidate = np.array(p["candidates"][correct]["grid_positions"])
    checks["selected_candidate_is_the_geometric_match"] = {
        tuple(q) for q in candidate - candidate.min(axis=0)
    } == normalized_gap
    active_times = []
    verdict_times = []
    statuses = []
    for index, card in enumerate(cards):
        x, y, w, h = card
        active = [i for i, f in enumerate(frames) if _active(f, card)]
        status = [_status(f, card) for f in frames]
        statuses.append(status)
        expected = "accepted" if index == correct else "rejected" if index < correct else "pending"
        checks[f"candidate_{index}.final_judgment"] = status[-1] == expected
        checks[f"candidate_{index}.no_incorrect_green_judgment"] = index == correct or "accepted" not in status
        if index <= correct:
            active_times.append(min(active) if active else len(frames))
            judged = [i for i, s in enumerate(status) if s == expected]
            verdict_times.append(min(judged) if judged else -1)
            checks[f"candidate_{index}.highlight_before_judgment"] = bool(
                active and judged and min(active) < min(judged)
            )
            # Reference fit previews are local renderings of each declared
            # polyomino, with timing and opacity allowed to vary.
            ref_active = [f for f in reference if _active(f, card)]
            previews = [_preview(f, source, split) for f in ref_active]
            preview = max(previews, key=lambda m: int(m.sum())) if previews else np.zeros_like(target[:split])
            # The candidate's validated first frame supplies its own codec
            # observation of the fixed structure. Comparing later frames with
            # a once-coded reference injects static chroma fringes into the
            # preview geometry after a second encode.
            comparisons = [min(mask_coverage(_preview(frames[i], frames[0], split), preview, radius=4)) for i in active]
            checks[f"candidate_{index}.fit_preview_has_correct_geometry"] = bool(
                preview.sum() > 1000 and comparisons and max(comparisons) >= 0.92
            )
            # Cross / check glyph identity is read independently of card colour.
            roi = (slice(y + 3, y + 32), slice(x + w - 32, x + w - 3))
            judgment_color = [100, 200, 100] if index == correct else [200, 100, 100]
            glyph = _opaque_mask(frames[-1][roi], judgment_color, tolerance=50)
            expected_glyph = _opaque_mask(final[roi], judgment_color, tolerance=50)
            checks[f"candidate_{index}.judgment_symbol"] = bool(
                expected_glyph.sum() > 10 and min(mask_coverage(glyph, expected_glyph, radius=2)) >= 0.95
            )
        # Candidate cards are visual dictionaries and remain visible after the
        # selected copy is animated. Additional moving copies cannot erase them.
        e = blue_source[y : y + h, x : x + w]
        checks[f"candidate_{index}.piece_preserved"] = all(
            mask_coverage(_opaque_mask(f[y : y + h, x : x + w], color), e, radius=3)[1] >= 0.98 for f in frames
        )
    checks["candidates_checked_left_to_right"] = all(a < b for a, b in zip(active_times, active_times[1:]))
    checks["previous_candidate_judged_before_next"] = all(
        verdict_times[i] <= active_times[i + 1] for i in range(len(active_times) - 1)
    )
    # A solved endpoint alone does not satisfy the requested demonstration.
    stationary = cv2.dilate((blue_source | blue_goal).astype(np.uint8), np.ones((7, 7), np.uint8))
    centers = []
    shape_fits = []
    cx, cy, cw, ch = cards[correct]
    piece = blue_source[cy : cy + ch, cx : cx + cw]
    piece_region = np.zeros_like(piece)
    for layer in layers:
        if (
            layer.get("kind") == "rect"
            and layer.get("fill_rgb") == color
            and cx <= layer.get("x_px", -1) < cx + cw
            and cy <= layer.get("y_px", -1) < cy + ch
            and layer.get("w_px", 0) < cw
        ):
            px, py, pw, ph = (int(layer[k]) for k in ("x_px", "y_px", "w_px", "h_px"))
            piece_region[py - cy : py - cy + ph, px - cx : px - cx + pw] = 1
    piece = piece * cv2.dilate(piece_region, np.ones((3, 3), np.uint8))
    sy, sx = np.nonzero(piece)
    source_pattern = piece[sy.min() : sy.max() + 1, sx.min() : sx.max() + 1]
    density = float(source_pattern.mean())
    aspect = source_pattern.shape[1] / source_pattern.shape[0]
    times = []
    for i, frame in enumerate(frames):
        moving = _opaque_mask(frame, color) * (1 - stationary)
        moving[split:] = 0
        yy, xx = np.nonzero(moving)
        if len(xx) > 500:
            centers.append([float(np.mean(xx)), float(np.mean(yy))])
            times.append(i)
            patch = moving[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
            _, _, cell_stats, _ = cv2.connectedComponentsWithStats(patch)
            areas = [int(stat[cv2.CC_STAT_AREA]) for stat in cell_stats[1:] if stat[cv2.CC_STAT_AREA] > 50]
            complete_cells = (
                len(areas) == len(p["candidates"][correct]["grid_positions"]) and max(areas) / min(areas) < 1.3
            )
            if (
                complete_cells
                and abs(patch.shape[1] / patch.shape[0] / aspect - 1) < 0.15
                and abs(patch.mean() / density - 1) < 0.20
            ):
                shape_fits.append(min(mask_coverage(normalized(patch), normalized(source_pattern), radius=3)))
        footprint = ink(source) | target
        # Card backgrounds, highlights and judgment glyphs change only inside
        # their declared cards. Preview locations are fixed around the gap.
        for x, y, w, h in cards:
            footprint[y - 3 : y + h + 3, x - 3 : x + w + 3] = 1
        gap_pixels = np.array([gap[:, 0] * scale_x + offset[0], gap[:, 1] * scale_y + offset[1]]).T
        gx0, gy0 = np.floor(gap_pixels.min(axis=0) - max(block_w, block_h) * 2).astype(int)
        gx1, gy1 = np.ceil(gap_pixels.max(axis=0) + max(block_w, block_h) * 3).astype(int)
        footprint[max(0, gy0) : min(split, gy1), max(0, gx0) : min(frame.shape[1], gx1)] = 1
        # A moving copy is read from its rectangular block components at their
        # observed positions; this does not constrain its route to the GT path.
        mask = _opaque_mask(frame, color)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
            local = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
            contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), max(1, min(w, h) * 0.02), True)[:, 0]
            edges = np.roll(outline, -1, axis=0) - outline
            rectilinear = len(outline) >= 4 and np.all(np.min(np.abs(edges), axis=1) <= 2)
            if area >= 20 and (area / (w * h) >= 0.8 or rectilinear):
                footprint[y : y + h, x : x + w] |= local
        footprint = anchored_foreground(ink(frame), footprint, radius=4)
        checks[f"frame_{i}.no_isolated_added_content"] = no_added_components(ink(frame), footprint, radius=4)[0]
    checks["selected_piece_moves_from_candidates_into_gap"] = (
        len(centers) >= 3 and np.ptp(np.array(centers)[:, 1]) > 100
    )
    checks["selected_piece_geometry_preserved_during_placement"] = bool(shape_fits and min(shape_fits) >= 0.94)
    if centers:
        checks["placement_follows_correct_judgment"] = times[0] >= verdict_times[correct]
    return _result(
        checks,
        {
            "candidate_states": statuses,
            "highlight_times": active_times,
            "judgment_times": verdict_times,
            "moving_piece_centers": centers,
        },
    )
