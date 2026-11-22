from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _result, _static
import cv2
import numpy as np


def _clock(frames, first, p, reference):
    center = np.array([first.shape[1] / 2, first.shape[0] / 2])
    hours = p["hours_to_add"]
    if (p["initial_hour"] + hours) % 12 != p["final_hour"] % 12 or p["initial_minute"] != p["final_minute"]:
        raise ValueError("Invalid benchmark: clock arithmetic contradicts requested hours")
    angles, lengths, widths, shaft_gaps, gap_radii = {}, {}, {}, {}, {}
    occlusions = []
    background = np.array(p.get("background_color", first[0, 0, ::-1].tolist())[::-1], np.float32)
    # The trusted moving demonstration reveals the fixed face hidden by the
    # initial hands. It supplies appearance only, never a required trajectory.
    face_background = np.median(np.stack(reference), axis=0).astype(np.uint8)
    # The black hub's half-coverage edge is wider than its opaque RGB core
    # after 512px sampling. Keep geometry and the retained-black test separate.
    dark_hub = (np.max(first, axis=2) < 128) & (np.ptp(first.astype(np.int16), axis=2) < 35)
    _, hub_labels = cv2.connectedComponents(dark_hub.astype(np.uint8))
    hub_id = hub_labels[round(center[1]), round(center[0])]
    if not hub_id:
        raise ValueError("Clock's fixed central hub is unreadable")
    hub = (hub_labels == hub_id).astype(np.uint8)
    opaque_hub = hub * (np.max(first, axis=2) < 40)
    hub_core = cv2.erode(opaque_hub, np.ones((3, 3), np.uint8)).astype(bool)
    if not np.any(hub_core):
        hub_core = hub.astype(bool)
    fringe = max(1, round(min(first.shape[:2]) * 0.004))
    hub_occlusion = cv2.dilate(hub, np.ones((2 * fringe + 1, 2 * fringe + 1), np.uint8)).astype(bool)
    # A thin hand leaving the black hub can inherit the other hand's chroma
    # over one 8px luma coding neighbourhood at the 512px observation scale.
    # This region may borrow only visible hand-colour support; unlike the
    # opaque hub itself, it cannot conceal a missing shaft.
    chroma_fringe = 4 * fringe
    hub_chroma_overlap = cv2.dilate(hub, np.ones((2 * chroma_fringe + 1, 2 * chroma_fringe + 1), np.uint8)).astype(bool)

    def hand_ink(patch, hand, minimum_alpha=0.4, substrate=None, preserve_fixed=False):
        underlay = background if substrate is None else substrate.astype(np.float32)
        own = np.array(p[f"{hand}_hand_color"][::-1], np.float32) - underlay
        other_name = "hour" if hand == "minute" else "minute"
        other = np.array(p[f"{other_name}_hand_color"][::-1], np.float32) - underlay
        pixels = patch.astype(np.float32) - underlay
        alpha = np.clip(np.sum(pixels * own, axis=2) / np.maximum(1, np.sum(own * own, axis=-1)), 0, 1)
        beta = np.clip(np.sum(pixels * other, axis=2) / np.maximum(1, np.sum(other * other, axis=-1)), 0, 1)
        residual = np.max(np.abs(pixels - alpha[:, :, None] * own), axis=2)
        competing = np.max(np.abs(pixels - beta[:, :, None] * other), axis=2)
        matched = (alpha >= minimum_alpha) & (residual <= 35) & (residual <= competing + 1)
        if preserve_fixed:
            fixed = np.array(p.get("border_color", [0, 0, 0])[::-1], np.float32) - underlay
            fixed_alpha = np.clip(np.sum(pixels * fixed, axis=2) / np.maximum(1, np.sum(fixed * fixed, axis=-1)), 0, 1)
            fixed_residual = np.max(np.abs(pixels - fixed_alpha[:, :, None] * fixed), axis=2)
            matched &= residual < fixed_residual
        return matched

    persistent_hand = hand_ink(face_background, "hour", 0.06, preserve_fixed=True) | hand_ink(
        face_background, "minute", 0.06, preserve_fixed=True
    )
    yy, xx = np.indices(first.shape[:2])
    reach = min(first.shape[:2]) * p["clock_radius_fraction"] / 2 * 0.7 + 8
    envelope = np.hypot(xx - center[0], yy - center[1]) <= reach
    face_background[persistent_hand & envelope & ~hub.astype(bool)] = background

    initial_regions = np.zeros(first.shape[:2], np.uint8)
    for name, fraction in (("hour", 0.5), ("minute", 0.7)):
        turns = (p["initial_hour"] + p["initial_minute"] / 60) / 12 if name == "hour" else p["initial_minute"] / 60
        angle = turns * 2 * np.pi - np.pi / 2
        radius = min(first.shape[:2]) * p["clock_radius_fraction"] / 2 * fraction
        tip = center + radius * np.array([np.cos(angle), np.sin(angle)])
        cv2.line(
            initial_regions,
            tuple(center.astype(int)),
            tuple(np.round(tip).astype(int)),
            1,
            max(3, round(min(first.shape[:2]) * 0.012)),
        )

    for hand in ("minute", "hour"):
        aa, ll, ww = [], [], []
        expected_length = min(first.shape[:2]) * p["clock_radius_fraction"] / 2 * (0.5 if hand == "hour" else 0.7)
        extent = round(expected_length + 10)
        xa, xb = max(0, round(center[0]) - extent), min(first.shape[1], round(center[0]) + extent + 1)
        ya, yb = max(0, round(center[1]) - extent), min(first.shape[0], round(center[1]) + extent + 1)
        original = cv2.GaussianBlur(first[ya:yb, xa:xb], (3, 3), 0).astype(np.int16)
        face = face_background[ya:yb, xa:xb]
        for frame_index, frame in enumerate(frames):
            patch = frame[ya:yb, xa:xb]
            changed = np.max(np.abs(cv2.GaussianBlur(patch, (3, 3), 0).astype(np.int16) - original), axis=2) > 25
            # Static numerals can acquire a hand-like chroma fringe after
            # YUV420 compression; only a changed hand or the original hands
            # can supply the observation. The static-face veto remains active.
            ink = hand_ink(patch, hand, substrate=face) & (changed | initial_regions[ya:yb, xa:xb].astype(bool))
            yy, xx = np.nonzero(ink)
            positions = np.column_stack([xx + xa, yy + ya]) - center
            positions = positions[np.linalg.norm(positions, axis=1) <= expected_length + 8]
            if hand == "hour":
                minute_direction = np.array(
                    [np.cos(angles["minute"][frame_index]), np.sin(angles["minute"][frame_index])]
                )
                transverse = np.abs(positions[:, 0] * minute_direction[1] - positions[:, 1] * minute_direction[0])
                projection = positions @ minute_direction
                positions = positions[
                    ~((transverse <= 4) & (projection >= 0) & (projection <= lengths["minute"][frame_index] + 4))
                ]
            if len(positions) < 15:
                if hand == "hour":
                    phase = (angles["minute"][frame_index] + np.pi / 2) % (2 * np.pi)
                    coincidence = phase * 11 / (2 * np.pi)
                    if abs(coincidence - round(coincidence)) <= 0.06 * 11 / (2 * np.pi):
                        aa.append(float(angles["minute"][frame_index]))
                        ll.append(expected_length)
                        ww.append(1.0)
                        occlusions.append(frame_index)
                        continue
                return _result({"hands_visible_or_geometrically_occluded": False})
            radii = np.linalg.norm(positions, axis=1)
            outer = positions[radii > radii.max() * 0.7]
            directions = np.arctan2(outer[:, 1], outer[:, 0])
            histogram, bins = np.histogram(directions, bins=180, range=(-np.pi, np.pi))
            peak = (bins[int(histogram.argmax())] + bins[int(histogram.argmax()) + 1]) / 2
            difference = np.angle(np.exp(1j * (directions - peak)))
            outer = outer[np.abs(difference) < 0.12]
            direction = outer.mean(axis=0)
            aa.append(float(np.arctan2(direction[1], direction[0])))
            ll.append(float(np.quantile(radii, 0.98)))
            vector = direction / np.linalg.norm(direction)
            ww.append(float(np.quantile(np.abs(positions[:, 0] * vector[1] - positions[:, 1] * vector[0]), 0.95)))
        angles[hand] = np.unwrap(aa)
        lengths[hand], widths[hand] = ll, ww
    for hand in ("minute", "hour"):
        # H.264 mixes two nearby hand colours. Continuous support may use the
        # other hand only inside that independently observed hand's footprint.
        # Correct endpoint dots still cannot stand in for a missing shaft.
        other = "hour" if hand == "minute" else "minute"
        expected_length = min(first.shape[:2]) * p["clock_radius_fraction"] / 2 * (0.5 if hand == "hour" else 0.7)
        extent = round(expected_length + 10)
        xa, xb = max(0, round(center[0]) - extent), min(first.shape[1], round(center[0]) + extent + 1)
        ya, yb = max(0, round(center[1]) - extent), min(first.shape[0], round(center[1]) + extent + 1)
        yy, xx = np.indices((yb - ya, xb - xa))
        face = face_background[ya:yb, xa:xb]
        dx, dy = xx + xa - center[0], yy + ya - center[1]
        radial = np.linspace(max(6, expected_length * 0.06), expected_length * 0.94, max(2, round(expected_length)))
        gaps, locations = [], []
        for index, frame in enumerate(frames):
            patch = frame[ya:yb, xa:xb]
            vector = np.array([np.cos(angles[hand][index]), np.sin(angles[hand][index])])
            other_vector = np.array([np.cos(angles[other][index]), np.sin(angles[other][index])])
            transverse = np.abs(dx * other_vector[1] - dy * other_vector[0])
            projection = dx * other_vector[0] + dy * other_vector[1]
            footprint = (
                (transverse <= max(2, min(first.shape[:2]) * 0.006))
                & (projection >= 0)
                & (projection <= lengths[other][index] + 3)
            )
            footprint |= hub_chroma_overlap[ya:yb, xa:xb]
            support = hand_ink(patch, hand, 0.06, face) | (hand_ink(patch, other, 0.06, face) & footprint)
            support |= hub_occlusion[ya:yb, xa:xb]
            distance = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, 3)
            points = center + radial[:, None] * vector
            px = np.clip(np.round(points[:, 0] - xa).astype(int), 0, xb - xa - 1)
            py = np.clip(np.round(points[:, 1] - ya).astype(int), 0, yb - ya - 1)
            gaps.append(float(np.max(distance[py, px])))
            locations.append(float(radial[int(np.argmax(distance[py, px]))]))
        shaft_gaps[hand] = gaps
        gap_radii[hand] = locations
    checks = {}
    for hand in ("hour", "minute"):
        declared_length = min(first.shape[:2]) * p["clock_radius_fraction"] / 2 * (0.5 if hand == "hour" else 0.7)
        checks[f"{hand}_length_preserved"] = (
            min(lengths[hand]) >= declared_length * 0.9 and max(lengths[hand]) <= declared_length * 1.07
        )
        checks[f"{hand}_thin_single_hand"] = max(widths[hand]) <= 10
        checks[f"{hand}_continuous_shaft"] = max(shaft_gaps[hand]) <= max(2, min(first.shape[:2]) * 0.005)
        final_angle = (
            (p["final_hour"] + p["final_minute"] / 60) * 2 * np.pi / 12
            if hand == "hour"
            else p["final_minute"] * 2 * np.pi / 60
        ) - np.pi / 2
        checks[f"{hand}_absolute_goal_angle"] = abs(np.angle(np.exp(1j * (angles[hand][-1] - final_angle)))) <= 0.035
        initial_angle = (
            (p["initial_hour"] + p["initial_minute"] / 60) * 2 * np.pi / 12
            if hand == "hour"
            else p["initial_minute"] * 2 * np.pi / 60
        ) - np.pi / 2
        checks[f"{hand}_absolute_initial_angle"] = (
            abs(np.angle(np.exp(1j * (angles[hand][0] - initial_angle)))) <= 0.035
        )
    checks["hour_hand_shorter_than_minute"] = max(lengths["hour"]) < min(lengths["minute"]) * 0.85
    checks["central_hub_preserved"] = all(np.mean(np.max(frame[hub_core], axis=1) < 60) >= 0.9 for frame in frames)
    static, worst = True, 0
    for index, frame in enumerate(frames):
        allowed = np.zeros(first.shape[:2], np.uint8)
        for hand in ("hour", "minute"):
            angle = angles[hand][index]
            tip = center + max(lengths[hand]) * np.array([np.cos(angle), np.sin(angle)])
            cv2.line(allowed, tuple(center.astype(int)), tuple(np.round(tip).astype(int)), 1, 9)
        allowed = cv2.dilate(allowed.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        preserved, error = _static([frame], face_background, allowed)
        static &= preserved
        worst = max(worst, error)
    checks["clock_face_preserved"] = static
    return _result(
        checks,
        {
            "hour_turns": float((angles["hour"][-1] - angles["hour"][0]) / (2 * np.pi)),
            "minute_turns": float((angles["minute"][-1] - angles["minute"][0]) / (2 * np.pi)),
            "unexpected_static_component": worst,
            "occluded_hour_hand_frames": occlusions,
            "observed_length_ranges_px": {hand: [min(lengths[hand]), max(lengths[hand])] for hand in lengths},
            "maximum_transverse_width_px": {hand: max(widths[hand]) for hand in widths},
            "maximum_shaft_gap_px": {hand: max(shaft_gaps[hand]) for hand in shaft_gaps},
            "maximum_shaft_gap_frame": {hand: int(np.argmax(shaft_gaps[hand])) for hand in shaft_gaps},
            "maximum_shaft_gap_radius_px": {
                hand: gap_radii[hand][int(np.argmax(shaft_gaps[hand]))] for hand in shaft_gaps
            },
        },
    )
