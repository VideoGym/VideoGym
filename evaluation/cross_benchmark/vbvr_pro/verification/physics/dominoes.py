from __future__ import annotations
from evaluation.cross_benchmark.vbvr_pro.verification.physics.observations import _decision, _mask, _pigment, _preserved, _vacated
import cv2
import numpy as np


def _branch_dominos(frames, first, metadata, scale):
    """Schematic branch-path rule, not an invented rigid-contact requirement.

    The public instruction specifies trunk/branch order and an explicit gap.
    It does not require the renderer's illustrative centre rotation to be a
    physical hinge simulation.
    """
    from scipy.optimize import linear_sum_assignment

    p = metadata["parameters"]
    objects = p["dominos"]
    width, height = p["domino_width"] * scale, p["domino_height"] * scale
    identities = [q["id"] for q in objects]
    expected_centers = np.asarray([[q["x"], q["y"]] for q in objects]) * scale
    expected_colors = [tuple(q["color"]) for q in objects]
    lettered_colors = {tuple(q["color"]) for q in objects if q["id"] == "START"}
    orders = {
        branch: sorted([i for i, q in enumerate(objects) if q["branch"] == branch], key=lambda i: objects[i]["x"])
        for branch in ["trunk", "A", "B"]
    }
    will_fall = set(orders["trunk"])
    for branch in ["A", "B"]:
        limit = p["block_after_index"] + 1 if p["blocked_branch"] == branch else len(orders[branch])
        will_fall.update(orders[branch][:limit])
    angles, invalid, scene = [], [], []
    source_masks = np.zeros(first.shape[:2], np.uint8)
    for q in objects:
        x, y = np.array([q["x"], q["y"]]) * scale
        cv2.rectangle(
            source_masks,
            (int(x - width / 2 - 3), int(y - height / 2 - 3)),
            (int(x + width / 2 + 3), int(y + height / 2 + 3)),
            1,
            -1,
        )
    empty = _vacated(first, source_masks.astype(bool), metadata)
    for frame_index, frame in enumerate(frames):
        observations = []
        for color in dict.fromkeys(expected_colors):
            binary = _pigment(frame, color) if color in lettered_colors else _mask(frame, color, 30)
            # Narrow START cards contain black lettering across their width.
            # A bandwidth round-trip can erase the last one-pixel colour
            # connection around a glyph; close that observation gap before
            # measuring the known rigid card's exterior geometry.
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            for k in range(1, n):
                if stats[k, 4] < width * height * 0.45:
                    continue
                yy, xx = np.where(labels == k)
                rectangle = cv2.minAreaRect(np.c_[xx, yy].astype(np.float32))
                corners = cv2.boxPoints(rectangle)
                edges = np.roll(corners, -1, axis=0) - corners
                lengths = np.linalg.norm(edges, axis=1)
                vector = edges[int(np.argmax(lengths))]
                if vector[1] > 0:
                    vector = -vector
                tilt = float(np.degrees(np.arctan2(vector[0], -vector[1])))
                observations.append(
                    {
                        "color": color,
                        "center": np.array(rectangle[0]),
                        "corners": corners,
                        "angle": tilt,
                        "width": float(lengths.min()),
                        "height": float(lengths.max()),
                    }
                )
        if len(observations) != len(objects):
            invalid.append(frame_index)
            angles.append([float("nan")] * len(objects))
            continue
        costs = np.array(
            [
                [np.linalg.norm(o["center"] - c) + 1000 * (o["color"] != color) for o in observations]
                for c, color in zip(expected_centers, expected_colors)
            ]
        )
        rows, cols = linear_sum_assignment(costs)
        measured = np.zeros(len(objects))
        allowed = np.zeros(first.shape[:2], np.uint8)
        for i, j in zip(rows, cols):
            o = observations[j]
            measured[i] = o["angle"]
            if (
                o["color"] != expected_colors[i]
                or costs[i, j] > height * 0.75
                or abs(o["width"] / width - 1) > 0.15
                or abs(o["height"] / height - 1) > 0.12
            ):
                invalid.append(frame_index)
            cv2.fillPoly(allowed, [np.rint(o["corners"]).astype(np.int32)], 1)
        allowed = cv2.dilate(allowed, np.ones((7, 7), np.uint8)).astype(bool)
        if not _preserved(frame, empty, allowed, source_masks.astype(bool)):
            scene.append(frame_index)
        angles.append(measured.tolist())
    values = np.asarray(angles)
    onset = [next((i for i, a in enumerate(values[:, k]) if a > 10), None) for k in range(len(objects))]
    completed = [next((i for i, a in enumerate(values[:, k]) if a >= 68), None) for k in range(len(objects))]
    order_ok = True
    for branch, order in orders.items():
        active = [k for k in order if k in will_fall]
        for a, b in zip(active, active[1:]):
            order_ok &= onset[a] is not None and onset[b] is not None and onset[a] < onset[b]
            if onset[b] is not None:
                order_ok &= values[onset[b], a] >= 30
        if branch != "trunk" and active:
            trunk_end = completed[orders["trunk"][-1]]
            order_ok &= trunk_end is not None and onset[active[0]] is not None and onset[active[0]] >= trunk_end - 1
    checks = {
        "every_domino_identity_shape_and_color_retained": not invalid,
        "all_dominos_initially_upright": bool(np.all(abs(values[0]) <= 8)),
        "exact_fallen_set_respects_branch_gap": bool(
            all(values[-1, k] >= 68 if k in will_fall else abs(values[-1, k]) <= 8 for k in range(len(objects)))
        ),
        "trunk_and_branches_follow_causal_diagram_order": bool(order_ok),
        "every_fall_contains_intermediate_pose": bool(
            all(np.any((values[:, k] > 15) & (values[:, k] < 65)) for k in will_fall)
        ),
        "falls_do_not_reverse_or_flip_left": bool(
            np.all(np.diff(values, axis=0) >= -8) and np.all(values >= -8) and np.all(values <= 95)
        ),
        "gap_marker_and_legend_preserved": not scene,
    }
    readable_angles = [[float(a) if np.isfinite(a) else None for a in row] for row in angles]
    return _decision(
        "schematic_domino_branch_propagation",
        checks,
        identities=identities,
        required_fallen_ids=[identities[k] for k in sorted(will_fall)],
        observed_angles_deg=readable_angles,
        fall_onset_frames=onset,
        invalid_frames=sorted(set(invalid)),
        scene_errors=scene,
    )


def _domino_polygon(x, ground, width, height, angle):
    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    vertices = np.array([[-width / 2, -height], [width / 2, -height], [width / 2, 0], [-width / 2, 0]]) @ rotation.T
    vertices += [x, ground - width / 2 * np.sin(theta)]
    return vertices


def _gap_dominos(frames, first, metadata, scale):
    p = metadata["parameters"]
    width, height = p["domino_width"] * scale, p["domino_height"] * scale
    xs = np.asarray(p["positions"], float) * scale
    ground = p["ground_y"] * scale
    reach = np.hypot(height, width / 2) + width / 2
    gaps = np.flatnonzero(np.diff(xs) > reach)
    if len(gaps) != 1:
        raise ValueError("Domino input does not establish the unique unreachable spacing")
    count = int(gaps[0]) + 1
    candidate_angles = np.arange(0, 91, 2, dtype=float)
    # Samples describe one rigid rectangle in local coordinates, not a GT image.
    xx, yy = np.meshgrid(np.linspace(-width / 2 + 2, width / 2 - 2, 10), np.linspace(-height + 2, -2, 24))
    local = np.c_[xx.ravel(), yy.ravel()]
    vertical = np.linspace(-height, 0, 32)
    horizontal = np.linspace(-width / 2, width / 2, 12)
    boundary = np.vstack(
        [
            np.c_[np.full(32, -width / 2), vertical],
            np.c_[np.full(32, width / 2), vertical],
            np.c_[horizontal, np.full(12, -height)],
            np.c_[horizontal, np.zeros(12)],
        ]
    )
    samples = []
    for x in xs:
        bank, edge_bank = [], []
        for angle in candidate_angles:
            theta = np.radians(angle)
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points = local @ rotation.T + [x, ground - width / 2 * np.sin(theta)]
            bank.append(np.rint(points).astype(int))
            edge_bank.append(np.rint(boundary @ rotation.T + [x, ground - width / 2 * np.sin(theta)]).astype(int))
        samples.append((np.asarray(bank), np.asarray(edge_bank)))
    angles, polygons, invalid, scene = [], [], [], []
    source_bodies = np.zeros(first.shape[:2], np.uint8)
    for x in xs:
        cv2.fillPoly(source_bodies, [np.rint(_domino_polygon(x, ground, width, height, 0)).astype(np.int32)], 1)
    source_bodies = cv2.dilate(source_bodies, np.ones((7, 7), np.uint8)).astype(bool)
    empty = _vacated(first, source_bodies, metadata)
    empty[max(0, int(ground) - 1) :] = first[max(0, int(ground) - 1) :]
    annotation_allowed = np.zeros(first.shape[:2], np.uint8)
    for layer in metadata["generic_declarative_render"]["first_frame"]["layers"]:
        if layer.get("type") == "text" and layer.get("content") == "PUSH":
            top = max(0, int(layer["y_px"] * scale - 4))
            bottom = int((layer["y_px"] + layer["font_size_px"] * 2) * scale)
            annotation_allowed[top:bottom, : int(xs[0] - width / 2)] = 1
    for frame_index, frame in enumerate(frames):
        pigment = _mask(frame, p["domino_color_rgb"], 40)
        tolerant = cv2.dilate(pigment, np.ones((3, 3), np.uint8))
        outline = cv2.dilate((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) < 95).astype(np.uint8), np.ones((3, 3), np.uint8))
        measured, shapes, scores = [], [], []
        fitted = np.zeros(first.shape[:2], np.uint8)
        for x, (bank, edge_bank) in zip(xs, samples):
            visible = (
                (bank[:, :, 0] >= 0)
                & (bank[:, :, 0] < frame.shape[1])
                & (bank[:, :, 1] >= 0)
                & (bank[:, :, 1] < frame.shape[0])
            )
            xx = np.clip(bank[:, :, 0], 0, frame.shape[1] - 1)
            yy = np.clip(bank[:, :, 1], 0, frame.shape[0] - 1)
            edge_visible = (
                (edge_bank[:, :, 0] >= 0)
                & (edge_bank[:, :, 0] < frame.shape[1])
                & (edge_bank[:, :, 1] >= 0)
                & (edge_bank[:, :, 1] < frame.shape[0])
            )
            ex = np.clip(edge_bank[:, :, 0], 0, frame.shape[1] - 1)
            ey = np.clip(edge_bank[:, :, 1], 0, frame.shape[0] - 1)
            fill_score = (tolerant[yy, xx] * visible).mean(axis=1)
            score = 0.72 * fill_score + 0.28 * (outline[ey, ex] * edge_visible).mean(axis=1)
            k = int(np.argmax(score))
            angle = candidate_angles[k]
            shape = _domino_polygon(x, ground, width, height, angle)
            measured.append(float(angle))
            shapes.append(shape)
            scores.append(float(fill_score[k]))
            cv2.fillPoly(fitted, [np.rint(shape).astype(np.int32)], 1)
        if min(scores) < 0.82:
            invalid.append(frame_index)
        # Every coloured body must be explained by the complete fitted set.
        covered = cv2.dilate(fitted, np.ones((7, 7), np.uint8))
        if np.count_nonzero(pigment & (1 - covered)) > max(12, pigment.sum() * 0.03):
            invalid.append(frame_index)
        allowed = cv2.dilate(fitted | annotation_allowed, np.ones((7, 7), np.uint8)).astype(bool)
        if not _preserved(frame, empty, allowed, source_bodies):
            scene.append(frame_index)
        angles.append(measured)
        polygons.append(shapes)
    values = np.asarray(angles)
    onset = [next((i for i, angle in enumerate(values[:, k]) if angle > 10), None) for k in range(len(xs))]
    contacts = []
    for k in range(1, count):
        time = onset[k]
        contact = False
        if time is not None and onset[k - 1] is not None and onset[k - 1] < time:
            previous = np.vstack([polygons[max(0, time - 1)][k - 1], polygons[time][k - 1]])
            sweep = cv2.convexHull(previous.astype(np.float32))
            target = polygons[max(0, time - 1)][k].astype(np.float32)
            area, _ = cv2.intersectConvexConvex(sweep, target)
            distance = min(abs(cv2.pointPolygonTest(sweep, tuple(q), True)) for q in target)
            contact = area > 0 or distance <= 4
        contacts.append(bool(contact))
    checks = {
        "all_rigid_domino_bodies_observed": not invalid,
        "starts_with_upright_chain": bool(np.all(values[0] <= 8)),
        "gap_stops_exact_fallen_prefix": bool(np.all(values[-1, :count] >= 20) and np.all(values[:, count:] <= 8)),
        "last_reachable_domino_falls_completely": bool(values[-1, count - 1] >= 74),
        "each_fall_triggered_by_predecessor_contact": all(contacts),
        "each_fall_shows_intermediate_angles": bool(
            all(np.any((values[:, k] > 12) & (values[:, k] < 65)) for k in range(count))
        ),
        "no_backward_or_leftward_falls": bool(np.all(np.diff(values, axis=0) >= -10)),
        "ground_and_numbered_scene_preserved": not scene,
    }
    return _decision(
        "rigid_domino_gap_propagation",
        checks,
        independently_reachable_count=count,
        maximum_center_trigger_spacing_px=float(reach),
        observed_angles_deg=angles,
        fall_onset_frames=onset,
        predecessor_contacts=contacts,
        invalid_frames=sorted(set(invalid)),
        scene_errors=scene,
    )
