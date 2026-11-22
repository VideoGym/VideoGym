"""O-55: verify visible geometry by projecting the declared voxel sculpture.

The latent 3D scene is not recovered from video. The reader verifies that every
observed view is a valid projection of the *given* voxel structure, at the given
elevation, and that camera azimuth follows one continuous requested sweep.
"""

import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.vision.masks import ink, mask_coverage, no_added_components


SUPPORTED_IDS = {"O-55"}


def _faces(positions, azimuth, elevation):
    voxels = {tuple(p) for p in positions}
    center = (np.min(positions, axis=0) + np.max(positions, axis=0) + 1) / 2
    az, el = np.deg2rad([azimuth, elevation])
    # The generator's canvas convention makes the declared positive x axis
    # project leftward at the initial azimuths; y projects with cos(azimuth).
    u = np.array([-np.sin(az), np.cos(az), 0])
    v = np.array([np.cos(az) * np.sin(el), np.sin(az) * np.sin(el), -np.cos(el)])
    camera = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    result = []
    for voxel in positions:
        voxel = np.asarray(voxel)
        for axis in range(3):
            for direction in (-1, 1):
                normal = np.zeros(3)
                normal[axis] = direction
                if normal @ camera <= 1e-8 or tuple(voxel + normal) in voxels:
                    continue
                axes = [i for i in range(3) if i != axis]
                corners = []
                for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                    point = voxel.astype(float).copy()
                    point[axis] += 1 if direction > 0 else 0
                    point[axes[0]] += a
                    point[axes[1]] += b
                    corners.append(point - center)
                points = np.array(corners)
                xy = np.c_[points @ u, points @ v]
                result.append((float(np.mean(points @ camera)), xy))
    return sorted(result, key=lambda item: item[0])


def _render(faces, scale, offset, size=512):
    image = np.full((size, size), 255, np.uint8)
    for _, xy in faces:
        points = np.round(xy * scale + offset).astype(np.int32)
        cv2.fillConvexPoly(image, points, 180)
        cv2.polylines(image, [points], True, 0, 1)
    return (image < 240).astype(np.uint8), (image < 80).astype(np.uint8)


def verify(task, frames, first, final, metadata, reference):
    if task.split("_", 1)[0] != "O-55":
        return None
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result

    p = metadata["parameters"]
    positions = np.array([o["position_3d"] for o in p["objects"]], float)
    azimuth = float(p["initial_view"]["azimuth"])
    elevation = float(p["initial_view"]["elevation"])
    rotation = float(p["rotation_angle"])
    size = 512
    initial = cv2.resize(first, (size, size), interpolation=cv2.INTER_AREA)
    source = ink(initial, background=[255, 255, 255])
    yy, xx = np.nonzero(source)
    if not len(xx):
        return _result({"sculpture_visible": False})
    faces = _faces(positions, azimuth, elevation)
    all_points = np.concatenate([xy for _, xy in faces])
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    raw_size = hi - lo
    observed_size = np.array([xx.max() - xx.min(), yy.max() - yy.min()])
    scale = float(observed_size @ raw_size / (raw_size @ raw_size))
    offset = np.array([xx.min(), yy.min()]) - lo * scale
    # Both 180-degree directions are valid when the prompt does not prescribe a
    # direction. Every intermediate pose is rendered from voxel coordinates.
    angles = np.linspace(-abs(rotation), abs(rotation), round(abs(rotation) / 2.5) * 2 + 1)
    dictionary = [_render(_faces(positions, azimuth + angle, elevation), scale, offset, size) for angle in angles]
    small = np.stack([cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA) > 0.4 for mask, _ in dictionary])
    checks = {"declared_voxel_count": len(positions) == p["num_voxels"]}
    observed_angles = []
    geometry = []
    expected_hsv = cv2.cvtColor(np.array([[np.array(p["voxel_color"])[::-1] * 255]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    for index, frame in enumerate(frames):
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        mask = ink(frame, background=[255, 255, 255])
        edges = (cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 25, 60) > 0).astype(np.uint8)
        target = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA) > 0.4
        mismatch = np.count_nonzero(small ^ target, axis=(1, 2)) / np.maximum(
            1, np.count_nonzero(small | target, axis=(1, 2))
        )
        stage = int(np.argmin(mismatch))
        predicted, predicted_edges = dictionary[stage]
        silhouette = mask_coverage(mask, predicted, radius=3)
        topology = mask_coverage(edges, predicted_edges, radius=3)
        checks[f"frame_{index}.declared_voxel_projection"] = min(silhouette) >= 0.97
        checks[f"frame_{index}.no_isolated_added_content"] = no_added_components(mask, predicted, radius=4)[0]
        checks[f"frame_{index}.visible_face_edge_topology"] = min(topology) >= 0.88
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colored = (mask > 0) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 100)
        if np.any(colored):
            hue = float(np.median(hsv[:, :, 0][colored]))
            saturation = float(np.median(hsv[:, :, 1][colored]))
            delta = abs(hue - float(expected_hsv[0]))
            checks[f"frame_{index}.voxel_color"] = (
                min(delta, 180 - delta) <= 10 and abs(saturation - float(expected_hsv[1])) <= 30
            )
        else:
            checks[f"frame_{index}.voxel_color"] = False
        observed_angles.append(float(angles[stage]))
        geometry.append({"silhouette_coverage": silhouette, "edge_coverage": topology})
    sweep = np.rad2deg(np.unwrap(np.deg2rad(observed_angles)))
    signed = float(sweep[-1] - sweep[0])
    steps = np.diff(sweep)
    checks["correct_initial_view"] = abs(sweep[0]) <= 5
    checks["requested_final_azimuth"] = abs(abs(signed) - abs(rotation)) <= 6
    checks["one_continuous_azimuth_direction"] = bool(np.all(steps * np.sign(signed) >= -3))
    checks["no_view_teleport"] = bool(np.max(np.abs(steps), initial=0) <= 45)
    checks["intermediate_camera_views_observed"] = len(set(observed_angles)) >= 5
    return _result(
        checks,
        {
            "azimuth_sweep_degrees": sweep.tolist(),
            "projected_geometry": geometry,
            "projection_scale": scale,
            "projection_offset": offset.tolist(),
        },
    )
