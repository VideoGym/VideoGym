from __future__ import annotations
from evaluation.cross_benchmark.vbvr_pro.verification.physics.observations import _decision, _preserved
import numpy as np


def _subtractive(frames, first, metadata, scale):
    p = metadata["parameters"]
    expected = np.rint(np.asarray(p["color1"]) * np.asarray(p["color2"]) / 255).astype(int)
    if not np.array_equal(expected, p["mixed_color"]):
        raise ValueError("Subtractive reference label contradicts per-channel rounded pigment product")
    x, y, w, h = np.rint(np.asarray(p["mixing_zone_bbox_xywh"]) * scale).astype(int)
    border = max(2, int(np.ceil(p["mixing_zone_border_width_px"] * scale)))
    allowed = np.zeros(first.shape[:2], bool)
    allowed[y : y + h + 1, x : x + w + 1] = True
    target = expected[::-1]
    interior = frames[-1][y + border + 2 : y + h - border - 2, x + border + 2 : x + w - border - 2]
    correct = np.max(np.abs(interior.astype(float) - target), axis=2) <= 20
    invalid_fill = []
    vector = target.astype(float) - 255
    for i, frame in enumerate(frames):
        pixels = frame[y + border + 2 : y + h - border - 2, x + border + 2 : x + w - border - 2].astype(float)
        alpha = np.clip(np.sum((pixels - 255) * vector, axis=2) / (vector @ vector), 0, 1)
        residual = np.max(np.abs(pixels - (255 + alpha[..., None] * vector)), axis=2)
        if np.mean(residual > 22) > 0.03:
            invalid_fill.append(i)
    bad_scene = [i for i, f in enumerate(frames) if not _preserved(f, first, allowed)]
    black_border = (np.max(first, axis=2) < 60) & allowed
    border_ok = all(np.mean(np.max(f[black_border], axis=1) < 100) >= 0.98 for f in frames)
    return _decision(
        "subtractive_rgb_product",
        {
            "computed_pigment_fills_complete_zone": correct.size > 0 and np.mean(correct) >= 0.98,
            "source_pigments_and_border_preserved_throughout": not bad_scene,
            "black_mixing_boundary_retained": border_ok,
            "no_wrong_pigment_during_fill": not invalid_fill,
        },
        expected_rgb=expected.tolist(),
        zone_xywh=[int(v) for v in (x, y, w, h)],
        scene_errors=bad_scene,
        wrong_fill_frames=invalid_fill,
    )


def _additive(frames, first, metadata, scale):
    p = metadata["parameters"]
    colors = np.asarray([p["color1"], p["color2"]], float)
    mixed = np.minimum(255, colors.sum(axis=0))
    if not np.array_equal(mixed, p["mixed_color"]):
        raise ValueError("Additive reference label contradicts channelwise light addition")
    initial = np.asarray([p["ball1_start_xy"], p["ball2_start_xy"]], float) * scale
    middle = initial.mean(axis=0)
    direction = initial[1] - initial[0]
    distance = np.linalg.norm(direction)
    unit = direction / distance
    radius = p["ball_radius_px"] * scale
    layers = metadata.get("generic_declarative_render", {}).get("first_frame", {}).get("layers", [])
    outline = max(
        (
            layer.get("stroke_width_px", 0) * scale / 2
            for layer in layers
            if layer.get("kind") == "circle"
            and "stroke_rgb" in layer
            and np.isclose(layer.get("radius_px", 0) * scale, radius)
        ),
        default=0.0,
    )
    outer_radius = radius + outline
    yy, xx = np.indices(first.shape[:2])
    xy = np.stack((xx, yy), axis=2)
    progress, bad, seen = [], [], set()
    for i, frame in enumerate(frames):
        foreground = np.any(frame < 220, axis=2)
        ys, xs = np.nonzero(foreground)
        if len(xs) < radius * radius:
            bad.append(i)
            continue
        projection = (np.c_[xs, ys] - middle) @ unit
        # Foreground extents include the declared centred outline. Its outer
        # half-width is not additional separation between the disk centres.
        current_distance = max(0.0, projection.max() - projection.min() - 2 * outer_radius)
        t = float(np.clip(1 - current_distance / distance, 0, 1))
        progress.append(t)
        centers = initial * (1 - t) + middle * t
        disks = [np.sum((xy - c) ** 2, axis=2) < (radius - 3) ** 2 for c in centers]
        wide = [np.sum((xy - c) ** 2, axis=2) <= (radius + 3) ** 2 for c in centers]
        regions = [disks[0] & ~wide[1], disks[1] & ~wide[0], disks[0] & disks[1]]
        expected = [colors[0], colors[1], mixed]
        for region, rgb in zip(regions, expected):
            if np.count_nonzero(region) >= 30:
                match = np.max(np.abs(frame[region].astype(float) - rgb[::-1]), axis=1) <= 32
                if np.mean(match) < 0.95:
                    bad.append(i)
        union = wide[0] | wide[1]
        if np.count_nonzero(foreground & ~union) > max(12, np.count_nonzero(union) * 0.01):
            bad.append(i)
        if 0.12 < t < 0.85:
            seen.add(int(t * 4))
    deltas = np.diff(progress)
    return _decision(
        "two_light_disks_merge",
        {
            "two_equal_disks_and_correct_overlap_color_every_frame": not bad and len(progress) == len(frames),
            "starts_separated": bool(progress) and progress[0] < 0.035,
            "ends_one_additive_disk_at_midpoint": bool(progress) and progress[-1] > 0.97,
            "equal_speed_straight_motion_to_midpoint": len(deltas) > 0 and np.all(deltas >= -0.025),
            "overlap_and_approach_stages_observed": len(seen) >= 3,
            "no_missing_motion_or_teleport": len(deltas) > 0 and np.all(deltas <= 0.18),
        },
        expected_rgb=mixed.tolist(),
        midpoint=middle.tolist(),
        progress=progress,
        fill_radius_px=float(radius),
        outer_radius_px=float(outer_radius),
        invalid_frames=sorted(set(bad)),
    )
