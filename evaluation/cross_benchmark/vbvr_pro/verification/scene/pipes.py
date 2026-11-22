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


def _pipes(frames, first, final, metadata):
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result

    p = metadata["parameters"]
    tiles = p["objects"]
    side = float(p["square_size"])
    color = p["pipe_color"]
    mask = color_mask(frames[-1], color)
    directions = {"top": (0, -1), "right": (1, 0), "bottom": (0, 1), "left": (-1, 0)}
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    ports = []
    checks = {}
    evidence = {}
    allowed = np.zeros(mask.shape, np.uint8)
    static = ink(first)
    for tile in tiles:
        tx, ty = tile["position"]
        static[ty + 2 : round(ty + side) - 2, tx + 2 : round(tx + side) - 2] = 0
    footprints = [static.copy() for _ in frames]
    for index, tile in enumerate(tiles):
        cx, cy = tile["center"]
        half = side / 2
        x0, y0, x1, y1 = map(round, (cx - half, cy - half, cx + half, cy + half))
        allowed[y0 : y1 + 1, x0 : x1 + 1] = 1
        local = mask[y0 : y1 + 1, x0 : x1 + 1]
        declared = {}
        for name, (dx, dy) in directions.items():
            x, y = round(cx + dx * side * 0.42), round(cy + dy * side * 0.42)
            radius = max(3, round(side * 0.035))
            coverage = float(mask[y - radius : y + radius + 1, x - radius : x + radius + 1].mean())
            declared[name] = coverage > 0.40
        ports.append(declared)
        required = sum(bool(v) for v in tile["pipe_pattern"].values())
        checks[f"tile_{index}.correct_port_count"] = sum(declared.values()) == required
        n, _, stats, _ = cv2.connectedComponentsWithStats(
            cv2.morphologyEx(local, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        )
        checks[f"tile_{index}.pipe_is_one_connected_piece"] = sum(st[cv2.CC_STAT_AREA] > 40 for st in stats[1:]) == 1
        reference = color_mask(final[y0 : y1 + 1, x0 : x1 + 1], color)
        checks[f"tile_{index}.pipe_size_preserved"] = abs(float(local.sum()) / max(1, int(reference.sum())) - 1) <= 0.15
        checks[f"tile_{index}.pipe_geometry_preserved"] = any(
            min(mask_coverage(local, np.rot90(reference, k), radius=3)) >= 0.97 for k in range(4)
        )
        ry, rx = np.nonzero(reference)
        pivot = np.array([(reference.shape[1] - 1) / 2, (reference.shape[0] - 1) / 2])
        ref_vector = np.array([rx.mean(), ry.mean()]) - pivot
        for frame_index, frame in enumerate(frames):
            current = color_mask(frame[y0 : y1 + 1, x0 : x1 + 1], color)
            yy, xx = np.nonzero(current)
            if not len(xx):
                checks[f"tile_{index}.frame_{frame_index}.rigid_pipe_shape"] = False
                continue
            vector = np.array([xx.mean(), yy.mean()]) - pivot
            delta = np.arctan2(vector[1], vector[0]) - np.arctan2(ref_vector[1], ref_vector[0])
            if np.linalg.norm(ref_vector) < 3:
                _, eigenvectors = np.linalg.eigh(np.cov(np.c_[rx, ry].T))
                _, current_eigenvectors = np.linalg.eigh(np.cov(np.c_[xx, yy].T))
                v, u = eigenvectors[:, -1], current_eigenvectors[:, -1]
                delta = np.arctan2(u[1], u[0]) - np.arctan2(v[1], v[0])
            fits = []
            projections = []
            for adjustment in (-4, -2, 0, 2, 4):
                rotation = cv2.getRotationMatrix2D(tuple(pivot), -float(np.rad2deg(delta)) + adjustment, 1)
                projected = cv2.warpAffine(
                    reference, rotation, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_NEAREST
                )
                fits.append(min(mask_coverage(current, projected, radius=3)))
                projections.append(projected)
            checks[f"tile_{index}.frame_{frame_index}.rigid_pipe_shape"] = max(fits) >= 0.97
            footprints[frame_index][y0 : y1 + 1, x0 : x1 + 1] |= projections[int(np.argmax(fits))]
        if abs(tile.get("rotation_angle", 0)) > 2:
            source = color_mask(first[y0 : y1 + 1, x0 : x1 + 1], color)
            middle = []
            for frame_index, frame in enumerate(frames):
                current = color_mask(frame[y0 : y1 + 1, x0 : x1 + 1], color)
                if (
                    min(mask_coverage(current, source, radius=1)) < 0.97
                    and min(mask_coverage(current, reference, radius=1)) < 0.97
                ):
                    middle.append(frame_index)
            checks[f"tile_{index}.intermediate_rotation_observed"] = bool(middle)
    # Tile borders and positions are immutable throughout the clip.
    checks["tiles_and_background_stationary"] = True
    for frame_index, frame in enumerate(frames):
        a, b = frame.copy(), first.copy()
        for tile in tiles:
            tx, ty = tile["position"]
            a[ty + 2 : round(ty + side) - 2, tx + 2 : round(tx + side) - 2] = 255
            b[ty + 2 : round(ty + side) - 2, tx + 2 : round(tx + side) - 2] = 255
        checks["tiles_and_background_stationary"] &= no_unexpected_ink(a, b)[0]
        foreground = ink(frame)
        current = anchored_foreground(foreground, footprints[frame_index], radius=3)
        checks[f"frame_{frame_index}.no_isolated_added_content"] = no_added_components(foreground, current, radius=3)[0]
    graph = {i: set() for i in range(len(tiles))}
    for index, tile in enumerate(tiles):
        center = np.array(tile["center"])
        for name, (dx, dy) in directions.items():
            if not ports[index][name]:
                continue
            candidates = []
            for j, neighbor in enumerate(tiles):
                delta = np.array(neighbor["center"]) - center
                along = float(delta @ np.array([dx, dy]))
                perp = abs(float(delta @ np.array([-dy, dx])))
                if j != index and along > 0 and perp < side * 0.1:
                    candidates.append((along, j))
            neighbor = min(candidates)[1] if candidates else None
            matched = neighbor is not None and ports[neighbor][opposite[name]]
            checks[f"tile_{index}.{name}_port_connected"] = matched
            if matched:
                graph[index].add(neighbor)
    reached = {0}
    frontier = [0]
    while frontier:
        for neighbor in graph[frontier.pop()] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    checks["all_pipes_form_one_component"] = len(reached) == len(tiles)
    checks["no_pipes_outside_tiles"] = np.count_nonzero(mask * (1 - allowed)) <= 20
    evidence["observed_ports"] = ports
    evidence["connectivity"] = {str(i): sorted(js) for i, js in graph.items()}
    return _result(checks, evidence)
