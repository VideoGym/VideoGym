from __future__ import annotations
from evaluation.cross_benchmark.vbvr_pro.verification.physics.observations import _chroma_ink, _decision, _preserved
import cv2
import numpy as np


def _optics(prefix, frames, first, metadata, scale):
    p = metadata["parameters"]
    rays = p["objects"]
    incident = next(q for q in rays if q["type"] == "incident")
    start = np.asarray(incident["end"], float) * scale
    incoming = np.subtract(incident["end"], incident["start"]).astype(float)
    if prefix == "O-18":
        theta = np.arcsin(p["n_air"] / p["n_glass"] * abs(incoming[0]) / np.linalg.norm(incoming))
        direction = np.array([np.sign(incoming[0]) * np.sin(theta), np.cos(theta)])
        label = next(q for q in rays if q["type"] == "refracted")
        if abs(np.degrees(theta) - p["theta_refracted_degrees"]) > 0.005:
            raise ValueError("Refraction angle label contradicts Snell law")
    else:
        direction = incoming * np.array([1, -1])
        direction /= np.linalg.norm(direction)
        label = next(q for q in rays if q["type"] == "reflected")
    label_vector = np.subtract(label["end"], label["start"]).astype(float)
    if np.linalg.norm(label_vector / np.linalg.norm(label_vector) - direction) > 0.001:
        raise ValueError("Reference optical ray direction contradicts recomputed physical law")
    length = np.linalg.norm(label_vector) * scale
    angular_slack = np.tan(np.deg2rad(1.0))
    # Only the labelled angle text and the new ray may change.
    allowed = np.zeros(first.shape[:2], np.uint8)
    edge = start + direction * max(first.shape) * 2
    cv2.line(allowed, tuple(np.rint(start).astype(int)), tuple(np.rint(edge).astype(int)), 1, 14)
    text_masks = []
    layers = metadata.get("generic_declarative_render", {}).get("first_frame", {}).get("layers", [])
    for q in layers:
        if q.get("type") == "text" and "θ" in q.get("content", ""):
            x, y = int(q["x_px"] * scale), int(q["y_px"] * scale)
            size = q["font_size_px"] * scale
            rect = (slice(max(0, y - 3), int(y + size * 1.8)), slice(max(0, x - 3), int(x + size * 6)))
            allowed[rect] = 1
            text_masks.append(rect)
    red_observations, bad, scene = [], [], []
    source_hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    incident_blue = (source_hsv[:, :, 0] >= 100) & (source_hsv[:, :, 0] <= 140) & (source_hsv[:, :, 1] >= 50)
    incident_blue = cv2.dilate(incident_blue.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for i, frame in enumerate(frames):
        # Thin red shafts lose saturation under 4:2:0 chroma subsampling.
        # Their red contribution remains observable even where an opaque-red
        # threshold would puncture a continuous, correctly directed shaft.
        red = _chroma_ink(frame, [255, 0, 0]).astype(bool)
        # Weak warm codec halos around the incident arrow are not red ink.
        # Chroma projection alone can classify pale yellow pixels as red.
        hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0]
        # At the intersection, red and the fixed blue incident arrow share a
        # chroma sample. Linux H.264 at the reference's 16 FPS can observe its
        # dark red pixel as BGR [39,8,84] (hue 168). Retain that red contribution
        # only beside the original blue ink, without admitting warm halos or
        # changing any angular, origin or continuity tolerance.
        red &= (hue <= 10) | (hue >= 170) | (incident_blue & (hue >= 165))
        ys, xs = np.nonzero(red)
        detail = {"pixels": len(xs), "extent": 0.0, "valid": True, "continuity": False, "touches_interface": False}
        if len(xs) > 6:
            delta = np.c_[xs, ys] - start
            along = delta @ direction
            across = delta @ np.array([-direction[1], direction[0]])
            detail["extent"] = float(along.max())
            shaft = (along > 4) & (along < along.max() - 16)
            # Arrowheads are wider than the shaft and are not angular errors.
            shaft_error = np.abs(across) - np.maximum(along, 0) * angular_slack
            shaft_ok = not np.any(shaft) or np.quantile(shaft_error[shaft], 0.95) <= 3
            detail["valid"] = bool(shaft_ok and np.max(shaft_error) <= 10 and along.min() >= -8)
            bins = np.floor(along[along >= 0] / 4).astype(int)
            occupied = len(np.unique(bins))
            detail["continuity"] = bool(occupied >= 0.93 * max(1, int(along.max() / 4)))
            detail["touches_interface"] = bool(along.min() <= 8)
            if not detail["valid"] or not detail["continuity"] or not detail["touches_interface"]:
                bad.append(i)
        red_observations.append(detail)
        ray_paint = cv2.dilate(red.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if not _preserved(frame, first, allowed.astype(bool) | ray_paint):
            scene.append(i)
    final = red_observations[-1]
    retained_text = []
    for rect in text_masks:
        # The only black ink at this declared text position is the annotation.
        original = np.max(first[rect], axis=2) < 90
        remaining = np.max(frames[-1][rect], axis=2) < 90
        retained_text.append(int(np.count_nonzero(original & remaining)))
    return _decision(
        "snell_refraction" if prefix == "O-18" else "specular_reflection",
        {
            "physically_correct_ray_direction_throughout": not bad,
            "ray_reaches_declared_scene_boundary": final["pixels"] >= 10 and final["extent"] >= length - 8,
            "continuous_ray_starts_at_interface": final["continuity"] and final["touches_interface"],
            "angle_annotation_removed": all(v <= 5 for v in retained_text),
            "incident_ray_interface_and_background_preserved": not scene,
        },
        direction=direction.tolist(),
        required_ray_length=float(length),
        observations=red_observations,
        invalid_ray_frames=bad,
        corrupted_scene_frames=scene,
        retained_annotation_pixels=retained_text,
    )
