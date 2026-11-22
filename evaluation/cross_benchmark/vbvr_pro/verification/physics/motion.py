from __future__ import annotations
from evaluation.cross_benchmark.vbvr_pro.verification.physics.observations import (
    _chroma_ink,
    _circles,
    _decision,
    _disk_allowed,
    _ink_on_background,
    _mask,
    _pigment,
    _polyline,
    _preserved,
    _vacated,
)
import cv2
import numpy as np


def _bounces(frames, first, metadata, scale, final=None):
    p = metadata["parameters"]
    radius = float(p["ball_radius"]) * scale
    points = (
        _polyline(p["initial_position"], p["initial_velocity"], p["bounds"], p["ball_radius"], int(p["num_bounces"]))
        * scale
    )
    # A wrong annotation is a reference error, not a reason to change the law.
    recorded = np.asarray([q[:2] for q in p["bounce_points"]], float) * scale
    if recorded.shape != points[1:].shape or not np.allclose(recorded, points[1:], atol=0.03):
        raise ValueError("Billiard reference bounce labels contradict elastic ray-wall intersections")
    starts, vectors = points[:-1], np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    positions, progress, bad, outside = [], [], [], []
    corridor = np.zeros(first.shape[:2], np.uint8)
    cv2.polylines(corridor, [np.rint(points).astype(np.int32)], False, 1, int(radius * 2.7))
    # The initial direction arrow may disappear, and the orange trace may grow.
    arrow = _mask(first, p["trajectory_rgb"])
    corridor |= cv2.dilate(arrow, np.ones((7, 7), np.uint8))
    source = _disk_allowed(first.shape, [points[0]], radius + 4) | cv2.dilate(arrow, np.ones((7, 7), np.uint8)).astype(
        bool
    )
    final_body = (
        None
        if final is None
        else (
            _disk_allowed(first.shape, [points[-1]], radius + 4)
            | cv2.dilate(_chroma_ink(final, p["trajectory_rgb"]), np.ones((7, 7), np.uint8)).astype(bool)
        )
    )
    empty = _vacated(first, source, metadata, final, final_body)
    for i, frame in enumerate(frames):
        balls = _circles(frame, p["ball_rgb"], minimum=int(radius * radius))
        if len(balls) != 1 or not 0.78 * radius <= balls[0]["radius"] <= 1.15 * radius:
            bad.append(i)
            continue
        point = balls[0]["center"]
        positions.append(point)
        t = np.clip(np.sum((point - starts) * vectors, axis=1) / lengths**2, 0, 1)
        distances = np.linalg.norm(point - (starts + t[:, None] * vectors), axis=1)
        k = int(np.argmin(distances))
        if distances[k] > max(3.0, radius * 0.27):
            bad.append(i)
        progress.append(float(cumulative[k] + t[k] * lengths[k]))
        trace = cv2.dilate(_chroma_ink(frame, p["trajectory_rgb"]), np.ones((7, 7), np.uint8)).astype(
            bool
        ) & corridor.astype(bool)
        allowed = _disk_allowed(first.shape, [point], radius + 4) | trace
        if not _preserved(frame, empty, allowed, source):
            outside.append(i)
    observed = np.asarray(positions) if positions else np.zeros((0, 2))
    gaps = np.diff(progress) if progress else np.array([])
    segment_visits = [
        int(sum(lo + length * 0.20 <= v <= lo + length * 0.80 for v in progress))
        for lo, length in zip(cumulative, lengths)
    ]
    checks = {
        "one_same_size_ball_every_frame": not bad and len(progress) == len(frames),
        "initial_state": bool(len(observed)) and np.linalg.norm(observed[0] - points[0]) <= radius * 0.35,
        "stops_at_requested_final_wall": bool(len(observed))
        and np.linalg.norm(observed[-1] - points[-1]) <= radius * 0.35,
        "elastic_segments_in_required_order": bool(len(gaps)) and np.all(gaps >= -max(3, radius * 0.2)),
        "no_missing_motion_or_teleport": bool(len(gaps)) and np.all(gaps <= max(radius * 3, cumulative[-1] * 0.10)),
        "all_bounce_segments_observed": all(
            v >= (3 if length >= cumulative[-1] * 0.12 else 1) for v, length in zip(segment_visits, lengths)
        ),
        "boundary_and_other_scene_preserved": not outside,
    }
    return _decision(
        "elastic_billiard",
        checks,
        recomputed_collision_points=points.tolist(),
        requested_bounces=int(p["num_bounces"]),
        progress=progress,
        segment_interior_observations=segment_visits,
        invalid_frames=bad,
        scene_errors=outside,
    )


def _eating(frames, first, metadata, scale):
    s = metadata["semantic_ground_truth"]
    predator, prey = s["black_ball"], s["prey_balls"]
    growth = float(s["growth_factor"])
    initial_radius = predator["size"] * scale / 2
    regions = [
        _disk_allowed(first.shape, [np.asarray([q["x"], q["y"]]) * scale], q["size"] * scale / 2 + 3) for q in prey
    ]
    old_counts = [int(np.count_nonzero(_mask(first, q["color"]) & region)) for q, region in zip(prey, regions)]
    eaten, contacts, events, bad, positions, scene, radii = set(), set(), [], [], [], [], []
    source_center = np.asarray([predator["x"], predator["y"]]) * scale
    source_allowed = _disk_allowed(first.shape, [source_center], initial_radius * 1.3)
    empty = _vacated(first, source_allowed, metadata)
    vacated = source_allowed.copy()
    canvas_rgb = metadata.get("generic_declarative_render", {}).get("canvas", {}).get("background_rgb", [255, 255, 255])
    radius = initial_radius
    for index, frame in enumerate(frames):
        candidates = _circles(frame, [0, 0, 0], minimum=int(initial_radius**2))
        if len(candidates) != 1:
            bad.append(index)
            continue
        black = candidates[0]
        center, observed_radius = black["center"], black["radius"]
        positions.append(center.tolist())
        radii.append(float(observed_radius))
        for k, q in enumerate(prey):
            pigment = _mask(frame, q["color"]) & regions[k]
            remaining = int(np.count_nonzero(pigment))
            if k in eaten:
                if remaining > old_counts[k] * 0.15:
                    bad.append(index)
                continue
            target = np.asarray([q["x"], q["y"]]) * scale
            prey_radius = q["size"] * scale / 2
            distance = np.linalg.norm(center - target)
            if distance <= observed_radius + prey_radius + 3:
                contacts.add(k)
            if remaining <= old_counts[k] * 0.12:
                legal = k in contacts and prey_radius < radius and distance <= radius * growth + prey_radius + 5
                events.append({"frame": index, "prey": k, "legal_contact_and_size": bool(legal)})
                if not legal:
                    bad.append(index)
                eaten.add(k)
                empty[regions[k]] = canvas_rgb[::-1]
                vacated |= regions[k]
                radius *= growth
            elif k not in contacts and remaining < old_counts[k] * 0.75:
                bad.append(index)
        # Growth may interpolate during a contact; unrelated size jumps may not.
        touching = any(k in contacts and k not in eaten for k in range(len(prey)))
        upper = radius * growth if touching else radius
        if not radius / growth * 0.85 <= observed_radius <= upper * 1.10:
            bad.append(index)
        allowed = _disk_allowed(first.shape, [center], observed_radius * 1.25 + 3)
        for k in eaten | contacts:
            if k not in eaten:
                allowed |= regions[k]
            else:
                remnant = np.zeros(frame.shape[:2], np.uint8)
                pixels = frame[regions[k]].reshape(-1, 1, 3)
                remnant[regions[k]] = _pigment(pixels, prey[k]["color"], minimum_alpha=0.12).ravel()
                allowed |= cv2.dilate(remnant, np.ones((3, 3), np.uint8)).astype(bool)
        if not _preserved(frame, empty, allowed, vacated):
            scene.append(index)
    movement = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(positions) > 1 else np.array([])
    final_radius = _circles(frames[-1], [0, 0, 0], minimum=int(initial_radius**2))
    expected_radius = initial_radius * growth ** len(prey)
    return _decision(
        "size_legal_contact_growth",
        {
            "one_readable_predator_and_legal_growth_every_frame": not bad and len(positions) == len(frames),
            "every_prey_consumed_once_after_legal_contact": len(eaten) == len(prey)
            and len(events) == len(prey)
            and all(e["legal_contact_and_size"] for e in events),
            "complete_movement_between_consumptions": len(movement) >= 3 * len(prey)
            and np.count_nonzero(movement > 1) >= 3 * len(prey),
            "no_predator_teleport": len(movement) > 0 and np.all(movement <= first.shape[1] * 0.16),
            "correct_final_size_after_all_consumptions": len(final_radius) == 1
            and abs(final_radius[0]["radius"] / expected_radius - 1) <= 0.10,
            "predator_never_shrinks": len(radii) > 1 and np.all(np.diff(radii) >= -2),
            "uneaten_prey_and_other_scene_preserved": not scene,
        },
        events=events,
        invalid_frames=sorted(set(bad)),
        positions=positions,
        expected_final_radius=float(expected_radius),
        consumed_prey=sorted(eaten),
        scene_errors=scene,
        observed_radii=radii,
    )


def _freefall(time, height, velocity, gravity, elasticity):
    """Analytic flight and exact ground impacts; no recorded GT trajectory."""
    impact_speed = np.sqrt(velocity * velocity + 2 * gravity * height)
    first_impact = (velocity + impact_speed) / gravity
    if time <= first_impact:
        return max(0.0, height + velocity * time - 0.5 * gravity * time * time), velocity - gravity * time
    remaining = time - first_impact
    upward = elasticity * impact_speed
    for _ in range(100):
        duration = 2 * upward / gravity
        if remaining <= duration:
            return max(0.0, upward * remaining - 0.5 * gravity * remaining * remaining), upward - gravity * remaining
        remaining -= duration
        upward *= elasticity
        if upward * upward / (2 * gravity) < 0.005:
            return 0.0, 0.0
    return 0.0, 0.0


def _gravity(frames, first, metadata, scale, final=None):
    from scipy.signal import find_peaks

    p, s = metadata["parameters"], metadata["semantic_ground_truth"]
    body = p["objects"][0]
    radius = body["radius"] * scale
    layers = metadata["generic_declarative_render"]["first_frame"]["layers"]
    circle = next(q for q in layers if q.get("kind") == "circle" and q.get("fill_rgb") == body["color"])
    x = circle["cx_px"] * scale
    ground = next(q["y_px"] for q in layers if q.get("type") == "filled_shape" and q.get("kind") == "rect") * scale
    ticks = [
        (float(q["content"][:-1]), float(q["y_px"]))
        for q in layers
        if q.get("type") == "text" and q.get("content", "").endswith("m") and q["content"][:-1].isdigit()
    ]
    if len(ticks) < 2:
        raise ValueError("Gravity input has no readable metric ruler declaration")
    tick_slope = np.polyfit([q[0] for q in ticks], [q[1] for q in ticks], 1)[0]
    unit = -tick_slope * scale
    height, velocity, g = p["initial_height"], p["initial_velocity"], p["gravity"]
    elasticity = s["physics_model"]["elasticity"]
    positions, invalid = [], []
    for i, frame in enumerate(frames):
        circles = _circles(frame, body["color"], minimum=int(radius * radius))
        if (
            len(circles) != 1
            or not 0.80 * radius <= circles[0]["radius"] <= 1.12 * radius
            or abs(circles[0]["center"][0] - x) > 3
        ):
            invalid.append(i)
            positions.append(None)
        else:
            positions.append(float(circles[0]["center"][1]))
    if invalid:
        return _decision(
            "analytic_gravity", {"one_constant_ball_on_fixed_vertical_axis": False}, invalid_frames=invalid
        )
    heights = (ground - radius - np.asarray(positions)) / unit
    troughs, _ = find_peaks(-heights, prominence=0.25)
    first_impact = int(troughs[0]) if len(troughs) else len(heights) - 1
    first_motion = next((i for i, h in enumerate(heights) if abs(h - heights[0]) > 0.10), 0)
    fits = []
    for start in range(min(first_motion + 1, max(1, first_impact - 5))):
        stop = first_impact
        if stop - start < 5:
            continue
        xx = np.arange(stop - start, dtype=float)
        a, b, c = np.polyfit(xx, heights[start:stop], 2)
        if a >= 0:
            continue
        dt = np.sqrt(-2 * a / g)
        residual = float(np.max(np.abs(np.polyval([a, b, c], xx) - heights[start:stop])))
        cost = residual + abs(c - height) + abs(b / dt - velocity) * 0.15
        fits.append((cost, start, dt, residual, b / dt, c))
    if not fits:
        return _decision(
            "analytic_gravity", {"observable_initial_gravity_parabola": False}, observed_heights_m=heights.tolist()
        )
    _, start, dt, residual, observed_v0, observed_h0 = min(fits)
    physical_times = np.maximum(0, np.arange(len(heights)) - start) * dt
    expected = np.asarray([_freefall(t, height, velocity, g, elasticity) for t in physical_times])
    errors = np.abs(heights - expected[:, 0])
    arrows, bad_arrows, bad_scene, gravity_retention = [], [], [], []
    velocity_color = [60, 180, 60]
    gravity_label = next(q for q in layers if q.get("type") == "text" and q.get("content", "").startswith("g="))
    gravity_color = gravity_label["color_rgb"]
    fixed_gravity_ink = _mask(first, gravity_color, 45).astype(bool)
    if not np.any(fixed_gravity_ink):
        raise ValueError("Gravity indicator is unreadable at the observation bandwidth")

    def velocity_ink(frame):
        # The allowed edit layer includes the anti-aliased velocity label.
        # Thin glyphs can contain no exact palette pixel after a bandwidth
        # round-trip; their green chroma still identifies the same layer.
        b, green, red = cv2.split(frame.astype(np.int16))
        return ((green - red > 12) & (green - b > 12)).astype(np.uint8)

    initial_allowed = _disk_allowed(first.shape, [[x, positions[0]]], radius + 4)
    initial_allowed |= cv2.dilate(velocity_ink(first), np.ones((5, 5), np.uint8)).astype(bool)
    final_body = (
        None
        if final is None
        else cv2.dilate((_mask(final, body["color"]) | velocity_ink(final)), np.ones((9, 9), np.uint8)).astype(bool)
    )
    empty = _vacated(first, initial_allowed, metadata, final, final_body)
    # The stationary canvas is white above the ground and grey below it.
    # A row median excludes the sparse ball, arrow and label while retaining
    # the actual antialiased ground boundary. Chroma bleed beyond an arrow
    # does not alter that background's luma, so it cannot extend its length.
    source_luma = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    background_luma = np.median(source_luma[:, max(0, int(x - 80)) : int(x + 81)], axis=1)[:, None]
    for i, (frame, cy, (_, speed)) in enumerate(zip(frames, positions, expected)):
        allowed = _disk_allowed(first.shape, [[x, cy]], radius + 4)
        crop = frame[:, max(0, int(x - 5)) : int(x + 6)]
        strip = _ink_on_background(crop, velocity_color, background_luma)
        # A green ball can tint its dark outline and the adjacent ground.
        # Reuse the registered ball footprint for both scene and arrow
        # observations; ball-edge paint is not a reverse velocity shaft.
        strip[allowed[:, max(0, int(x - 5)) : int(x + 6)]] = 0
        yy, _ = np.nonzero(strip)
        displacement = yy - cy
        expected_sign = -np.sign(speed)  # image y increases downward
        visibly_settled = abs(heights[i]) <= 0.12 and expected[i, 0] <= 0.12
        if abs(speed) >= 0.65 and not visibly_settled:
            correct = displacement[displacement * expected_sign > 0]
            wrong = displacement[displacement * expected_sign < 0]
            length = float(np.max(np.abs(correct)) - radius - 2.5) if len(correct) else 0.0
            # Public drawing uses 6 px/(m/s), with a 15 px visibility floor;
            # orientation and length are observed, not rendered text/OCR.
            expected_length = max(15.0, abs(speed) * 6) * scale
            okay = (
                len(correct) > 2
                and len(wrong) <= 4
                and abs(length - expected_length) <= max(4.0, expected_length * 0.15)
            )
            if not okay:
                bad_arrows.append(i)
            arrows.append({"frame": i, "observed_length": length, "expected_length": float(expected_length)})
        green_allow = cv2.dilate(velocity_ink(frame), np.ones((5, 5), np.uint8)).astype(bool)
        green_allow[:, : max(0, int(x - 260 * scale))] = False
        green_allow[:, int(x + 260 * scale) :] = False
        allowed |= green_allow
        fixed_pixels = frame[fixed_gravity_ink].reshape(-1, 1, 3)
        retained = float(np.mean(_chroma_ink(fixed_pixels, gravity_color)))
        gravity_retention.append(retained)
        # The changing green label cannot authorize recolouring the fixed
        # blue gravity indicator, even when their screen regions overlap.
        if retained < 0.92 or not _preserved(frame, empty, allowed, initial_allowed):
            bad_scene.append(i)
    stopped = len(heights) >= 4 and max(abs(heights[-4:])) <= 0.12 and np.ptp(heights[-4:]) <= 0.07
    checks = {
        "one_constant_ball_on_fixed_vertical_axis": True,
        "initial_height_and_velocity": abs(observed_h0 - height) <= 0.12 and abs(observed_v0 - velocity) <= 0.55,
        "uniform_time_resampling_of_gravity_and_elastic_impacts": residual <= 0.12 and np.max(errors) <= 0.20,
        "ball_never_penetrates_ground": np.min(heights) >= -0.08,
        "velocity_arrow_direction_and_magnitude_update": not bad_arrows,
        "trajectory_reaches_visible_rest": stopped,
        "ruler_ground_and_gravity_indicator_preserved": not bad_scene,
    }
    return _decision(
        "analytic_gravity",
        checks,
        gravity_m_s2=g,
        restitution=elasticity,
        observed_initial_velocity_m_s=float(observed_v0),
        physical_seconds_per_frame=float(dt),
        leading_hold_frames=int(start),
        observed_heights_m=heights.tolist(),
        expected_heights_m=expected[:, 0].tolist(),
        maximum_height_error_m=float(np.max(errors)),
        terminal_heights_m=heights[-4:].tolist(),
        velocity_arrows=arrows,
        invalid_arrow_frames=bad_arrows,
        scene_errors=bad_scene,
        fixed_gravity_ink_retention=gravity_retention,
    )


def _rolling(frames, first, metadata, scale, final=None):
    p = metadata["parameters"]
    layers = metadata["generic_declarative_render"]["first_frame"]["layers"]
    platforms = [
        q
        for q in layers
        if q.get("type") == "filled_shape" and q.get("kind") == "polygon" and q.get("fill_rgb") == p["platform_color"]
    ]
    balls = [
        q
        for q in layers
        if q.get("type") == "filled_shape" and q.get("kind") == "circle" and q.get("fill_rgb") == p["ball_color"]
    ]
    if len(platforms) != p["num_platforms"] or len(balls) != 1:
        raise ValueError("Rolling input does not declare the ordered platform chain and one ball")
    polygons = [np.asarray(q["points_px"], float) * scale for q in platforms]
    centers = np.asarray([q.mean(axis=0) for q in polygons])
    radius = balls[0]["radius_px"] * scale
    offset = np.asarray([balls[0]["cx_px"], balls[0]["cy_px"]]) * scale - centers[0]
    anchors = centers + offset
    vectors = np.diff(anchors, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    invalid, scene, positions, progress, landings = [], [], [], [], []
    source_allowed = _disk_allowed(first.shape, [anchors[0]], radius + 4)
    final_body = _disk_allowed(first.shape, [anchors[-1]], radius + 4)
    empty = _vacated(first, source_allowed, metadata, final, final_body)
    platform_masks = []
    platform_fringe = np.zeros(first.shape[:2], np.uint8)
    for polygon in polygons:
        mask = np.zeros(first.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        platform_masks.append(cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool))
        platform_fringe |= cv2.dilate(mask, np.ones((5, 5), np.uint8))
    # These small supports contain only a few chroma samples at 512px.
    # VAE/4:2:0 colour shifts preserve their material while moving it away from
    # the exact RGB ray. Observe hue and visible saturation locally; geometry
    # still comes from each declared support, never from the candidate.
    palette = cv2.cvtColor(np.asarray([[p["platform_color"][::-1]]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    for index, frame in enumerate(frames):
        found = _circles(frame, p["ball_color"], minimum=int(radius * radius))
        if len(found) != 1 or abs(found[0]["radius"] / radius - 1) > 0.15:
            invalid.append(index)
            continue
        point = found[0]["center"]
        positions.append(point)
        t = np.clip(np.sum((point - anchors[:-1]) * vectors, axis=1) / lengths**2, 0, 1)
        distances = np.linalg.norm(point - (anchors[:-1] + t[:, None] * vectors), axis=1)
        k = int(np.argmin(distances))
        if distances[k] > max(2.0, radius * 0.22):
            invalid.append(index)
        progress.append(float(cumulative[k] + t[k] * lengths[k]))
        ball_allowed = _disk_allowed(first.shape, [point], radius + 4)
        if not _preserved(frame, empty, ball_allowed | platform_fringe.astype(bool), source_allowed):
            scene.append(index)
        # Some declared support polygons are only a few pixels thick. Read
        # their colour contribution, rather than requiring an opaque centre
        # pixel after the same downsampling and chroma codec as the candidate.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hue_delta = np.abs(hsv[:, :, 0] - int(palette[0]))
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        color = (hue_delta <= 15) & (hsv[:, :, 1] >= max(40, int(palette[1]) * 0.4)) & (hsv[:, :, 2] >= 35)
        # A 4:2:0 sample next to the ball mixes both materials. The additional
        # two-pixel chroma footprint cannot establish a missing support.
        support_occlusion = _disk_allowed(first.shape, [point], radius + 6)
        for mask in platform_masks:
            visible = mask & ~support_occlusion
            visible_pixels = np.count_nonzero(visible)
            # Three mixed edge samples must not erase the identity of a
            # support whose entire exposed interior is eleven pixels.
            missing_pixels = np.count_nonzero(visible & ~color)
            if visible_pixels >= 5 and missing_pixels > max(3, visible_pixels * 0.15):
                scene.append(index)
    for anchor in anchors:
        matches = [i for i, point in enumerate(positions) if np.linalg.norm(point - anchor) <= max(3.0, radius * 0.22)]
        # A small support may be crossed between two decoded observations.
        # Only the observed straight sweep supplies that contact; no unseen
        # turn or reference trajectory is inserted to complete the route.
        for i, (start, end) in enumerate(zip(positions, positions[1:])):
            vector = end - start
            length2 = float(vector @ vector)
            if not length2:
                continue
            alpha = float(np.clip(np.dot(anchor - start, vector) / length2, 0, 1))
            if np.linalg.norm(anchor - (start + alpha * vector)) <= max(3.0, radius * 0.22):
                matches.append(i + alpha)
        matches.sort()
        landings.append(matches)
    ordered = []
    for matches in landings:
        after = [i for i in matches if not ordered or i >= ordered[-1]]
        if not after:
            break
        ordered.append(after[0])
    delta = np.diff(progress)
    return _decision(
        "ordered_platform_support",
        {
            "one_constant_ball_on_platform_path": not invalid and len(positions) == len(frames),
            "each_platform_landed_in_input_order": len(ordered) == len(anchors),
            "starts_first_and_ends_last_platform": bool(positions)
            and np.linalg.norm(positions[0] - anchors[0]) <= 3
            and np.linalg.norm(positions[-1] - anchors[-1]) <= 3,
            "forward_continuous_transition_between_platforms": len(delta) > 0
            and np.all(delta >= -2)
            and np.max(delta) <= max(lengths) * 1.4,
            "platforms_preserved_throughout": not scene,
        },
        platform_anchor_centers=anchors.tolist(),
        observed_positions=[q.tolist() for q in positions],
        landing_frame_indices=ordered,
        invalid_frames=invalid,
        scene_errors=sorted(set(scene)),
    )
