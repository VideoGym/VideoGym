"""Source-derived strict-interior lattice points for the cyan-marker observer."""
from data.geometry.core.marker_fade_observation import marker_set_fade_metrics
from data.geometry.lattice_points_strict_inside.generator import (
    _MARKER_RADIUS, _lattice_point_sets, _screen_point,
)


def marker_fade_metrics(frames, source, state):
    _, interior = _lattice_point_sets([tuple(point) for point in state["verts"]])
    points = [_screen_point(point) for point in interior]
    metrics, passed = marker_set_fade_metrics(frames, source, points, _MARKER_RADIUS)
    return {key.replace("video_marker_", "video_interior_marker_"): value
            for key, value in metrics.items()}, passed
