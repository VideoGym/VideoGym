"""Observe a fixed cyan marker set and its joint reveal at codec resolution."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from data.common.verifiers.keyframes import visual_features


def marker_set_fade_metrics(
    frames: Sequence[np.ndarray], source: np.ndarray, points: Sequence[Sequence[float]],
    marker_radius: float,
) -> tuple[dict[str, Any], bool]:
    """Read each source-derived marker, without requiring ideal pixel alpha.

    The caller separately verifies the complete terminal marker set and scene.
    Register that legal endpoint's paint footprint, then inspect its centroid,
    radial second moment and four angular sectors during the reveal. Fixed
    3px observation smoothing suppresses codec fringes; it cannot fill a
    missing half-marker or turn a growing small disk into a full footprint.
    """
    markers = []
    for point in points:
        x, y = map(float, point)
        reach = int(math.ceil(marker_radius + 5))
        left, top = max(0, int(x) - reach), max(0, int(y) - reach)
        right, bottom = min(512, int(x) + reach + 2), min(512, int(y) + reach + 2)
        base = visual_features(source[top:bottom, left:right])
        observed = [visual_features(frame[top:bottom, left:right]) for frame in frames]
        terminal = observed[-1]
        yy, xx = np.indices(base.shape[:2], dtype=float)
        xx += left
        yy += top
        allowed = (xx - x) ** 2 + (yy - y) ** 2 <= (marker_radius + 2) ** 2
        # Cyan body, excluding the unchanged carrier/grid ink and the
        # optional dark contour. The semantic terminal gate owns its validity.
        cyan = ((terminal[:, :, 1] - terminal[:, :, 0] >= 45)
                & (terminal[:, :, 2] - terminal[:, :, 0] >= 45))
        direction = terminal - base
        pixel_energy = np.sum(direction * direction, axis=2)
        footprint = allowed & cyan & (pixel_energy >= 144.)
        if int(footprint.sum()) < 24:
            markers.append({"point": [x, y], "pass": False,
                            "reason": "source point has no readable cyan marker footprint"})
            continue
        energy = float(pixel_energy[footprint].sum())
        positions = np.column_stack((xx[footprint], yy[footprint]))
        endpoint_center = positions.mean(axis=0)
        endpoint_radius = float(np.sqrt(np.mean(np.sum((positions - endpoint_center) ** 2, axis=1))))
        sectors = [(xx < x) & (yy < y), (xx >= x) & (yy < y),
                   (xx < x) & (yy >= y), (xx >= x) & (yy >= y)]
        sector_masks = [sector[footprint] for sector in sectors]
        progress, geometry, residuals = [], [], []
        for index, image in enumerate(observed):
            departure = image - base
            alpha = (np.sum(departure * direction, axis=2)[footprint]
                     / pixel_energy[footprint])
            depth = float(np.sum(departure[footprint] * direction[footprint]) / energy)
            progress.append(depth)
            residuals.append(float(np.sqrt(np.sum((departure[footprint] - depth * direction[footprint]) ** 2) / energy)))
            if not .10 < depth < .90:
                continue
            # A small color deadband prevents weak quantisation residue from
            # pretending that an otherwise absent portion of a marker exists.
            weights = np.maximum(np.clip(alpha, 0., 1.2) - .05, 0.)
            if float(weights.sum()) <= 1e-6:
                geometry.append({"frame": index, "pass": False})
                continue
            center = np.sum(positions * weights[:, None], axis=0) / weights.sum()
            radius = float(np.sqrt(np.sum(weights * np.sum((positions - center) ** 2, axis=1)) / weights.sum()))
            center_error = float(np.linalg.norm(center - endpoint_center))
            radius_error = abs(radius - endpoint_radius)
            sector_depth = [float(np.mean(np.maximum(alpha[mask], 0.))) if mask.any() else 0.
                            for mask in sector_masks]
            support = min(sector_depth) / max(max(sector_depth), 1e-6)
            geometry.append({"frame": index, "centroid_error_pixels": center_error,
                             "radial_moment_error_pixels": radius_error,
                             "min_sector_relative_opacity": support,
                             "pass": center_error <= 2. and radius_error <= 1.5 and support >= .30})
        first = next((i for i, value in enumerate(progress) if value > .10), None)
        last = next((i for i, value in enumerate(progress) if value >= .85), None)
        monotone = all(right >= left - .08 for left, right in zip(progress, progress[1:]))
        holds = max(progress[:2], default=1.) <= .08 and min(progress[-2:], default=0.) >= .85
        passed = bool(geometry and all(row["pass"] for row in geometry) and monotone and holds
                      and max(residuals, default=1.) <= .40)
        markers.append({"point": [x, y], "pixels": int(footprint.sum()), "progress": progress,
                        "geometry": geometry, "opacity_monotone": monotone,
                        "endpoint_holds": holds, "max_color_fit_error": max(residuals, default=1.),
                        "onset_frame": first, "complete_frame": last, "pass": passed})
    # Compare shared paint progress, not threshold-crossing frame numbers:
    # the same small color error can shift a slow fade's crossing by many
    # frames. A one-pixel rim occupies roughly a quarter of these small markers; .25 allows
    # that spatial reading uncertainty while separating staged marker reveals.
    progress_rows = [row["progress"] for row in markers if "progress" in row]
    spread = max((max(values) - min(values) for values in zip(*progress_rows)), default=1.)
    synchronized = bool(len(progress_rows) == len(points) and spread <= .25)
    passed = bool(markers and all(row["pass"] for row in markers) and synchronized)
    return {"video_uniform_fade_pass": passed,
            "video_marker_geometry_fade": markers,
            "video_marker_reveal_synchronized": synchronized,
            "video_marker_max_shared_opacity_spread": spread,
            "video_marker_shared_opacity_tolerance": .25,
            "video_marker_observer": "source_point_set_and_registered_cyan_body_geometry_v1"}, passed
