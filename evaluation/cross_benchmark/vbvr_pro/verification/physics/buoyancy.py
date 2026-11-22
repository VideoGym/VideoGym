from __future__ import annotations
from evaluation.cross_benchmark.vbvr_pro.verification.physics.observations import _decision, _mask, _pigment, _preserved, _vacated
import cv2
import numpy as np


def _buoyancy(frames, first, metadata, scale, final=None):
    s = metadata["semantic_ground_truth"]
    cups, objects = s["cups"], s["objects"]
    special = int(s["special_cup_index"])
    densities = np.asarray([q["liquid_density"] for q in cups])
    high = bool(s["special_has_high_density"])
    differing = np.flatnonzero(densities != np.median(densities))
    if len(differing) != 1 or int(differing[0]) != special or bool(densities[special] > np.median(densities)) != high:
        raise ValueError("Buoyancy density labels contradict the uniquely different cup")
    layers = metadata["generic_declarative_render"]["first_frame"]["layers"]
    color = metadata["parameters"]["objects"][0]["fill_rgb"]
    shapes = [q for q in layers if q.get("type") == "filled_shape" and q.get("fill_rgb") == color]
    if len(shapes) != len(objects):
        raise ValueError("Cannot identify one buoyancy object per cup")
    backgrounds = [
        (255, 255, 255),
        *dict.fromkeys(tuple(q["water_fill_rgb"]) for q in metadata["parameters"]["containers"]),
    ]
    outlines = {tuple(q["outline_rgb"]) for q in metadata["parameters"]["objects"]}

    def object_ink(frame):
        # The opaque core locates candidates; coverage measures their actual
        # boundary within a two-pixel codec halo. This also avoids treating a
        # distant background colour as a soft foreground mixture.
        nearby = cv2.dilate(_mask(frame, color), np.ones((5, 5), np.uint8)).astype(bool)
        pixels = frame[nearby].reshape(-1, 1, 3)
        coverage = _pigment(pixels, color, backgrounds)
        for outline in outlines:
            coverage |= _pigment(pixels, outline, backgrounds)
        ink = np.zeros(frame.shape[:2], np.uint8)
        ink[nearby] = coverage.ravel()
        return ink

    original = object_ink(first)
    n, _, stats, centers = cv2.connectedComponentsWithStats(original, 8)
    initial = sorted(
        [(centers[k], stats[k, 4], stats[k, 2:4]) for k in range(1, n) if stats[k, 4] >= 40], key=lambda q: q[0][0]
    )
    if len(initial) != len(cups):
        raise ValueError("Initial buoyancy objects are unreadable")
    tracks = [[] for _ in cups]
    invalid, scene = [], []
    initial_allowed = np.zeros(first.shape[:2], np.uint8)
    for center, _, wh in initial:
        cv2.ellipse(
            initial_allowed, tuple(np.rint(center).astype(int)), tuple(np.ceil(wh * 0.75).astype(int)), 0, 0, 360, 1, -1
        )
    empty = _vacated(first, initial_allowed.astype(bool), metadata, final)
    for index, frame in enumerate(frames):
        binary = object_ink(frame)
        n, _, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
        observed = sorted(
            [(centers[k], stats[k, 4], stats[k, 2:4]) for k in range(1, n) if stats[k, 4] >= 40], key=lambda q: q[0][0]
        )
        if len(observed) != len(cups):
            invalid.append(index)
            continue
        allowed = np.zeros(first.shape[:2], np.uint8)
        for k, ((center, area, wh), (old, old_area, old_wh)) in enumerate(zip(observed, initial)):
            tracks[k].append(center.tolist())
            if (
                abs(center[0] - old[0]) > 5
                or not 0.85 <= area / old_area <= 1.15
                or np.max(abs(wh / old_wh - 1)) > 0.12
            ):
                invalid.append(index)
            cv2.ellipse(
                allowed, tuple(np.rint(center).astype(int)), tuple(np.ceil(wh * 0.75).astype(int)), 0, 0, 360, 1, -1
            )
        if not _preserved(frame, empty, allowed.astype(bool), initial_allowed.astype(bool)):
            scene.append(index)
    endpoints, intermediate, continuity = [], [], []
    for k, (track, cup, source) in enumerate(zip(tracks, cups, initial)):
        if not track:
            endpoints.append(False)
            intermediate.append(False)
            continuity.append(False)
            continue
        yy = np.asarray(track)[:, 1]
        cy = yy[-1]
        ry = source[2][1] / 2
        water = cup["water_level"] * scale
        bottom = (cup["position"][1] + cup["size"][1]) * scale
        floating = (k == special) == high
        endpoints.append(
            bool(abs(cy - water) <= ry * 0.7 if floating else bottom - 2.5 * ry <= cy <= bottom - 0.6 * ry)
        )
        distance = max(1.0, cy - yy[0])
        progress = (yy - yy[0]) / distance
        intermediate.append(
            all(
                np.any((progress >= low) & (progress <= high))
                for low, high in ((0.1, 0.35), (0.35, 0.65), (0.65, 0.95))
            )
        )
        continuity.append(np.max(np.abs(np.diff(yy)), initial=0) <= distance * 0.30)
    return _decision(
        "density_order_buoyancy",
        {
            "one_same_shape_object_per_cup_throughout": not invalid,
            "denser_liquid_floats_others_sink": all(endpoints),
            "complete_fall_stages_observed": all(intermediate),
            "no_missing_motion_or_teleport": all(continuity),
            "liquids_containers_and_other_scene_preserved": not scene,
        },
        special_cup=special,
        density_order=densities.tolist(),
        tracks=tracks,
        endpoint_checks=endpoints,
        invalid_frames=sorted(set(invalid)),
        scene_errors=scene,
    )
