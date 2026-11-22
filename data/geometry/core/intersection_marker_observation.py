"""Source-derived finite-segment crossings for the shared cyan-marker observer."""
from data.geometry.core.marker_fade_observation import marker_set_fade_metrics
from data.geometry.count_intersection_points.generator import (
    _MARKER_RADIUS, _cluster_intersections, _raw_pair_intersections,
)


def marker_fade_metrics(frames, source, state):
    points = [record["point"] for record in
              _cluster_intersections(_raw_pair_intersections(state["lines"]))]
    metrics, passed = marker_set_fade_metrics(frames, source, points, _MARKER_RADIUS)
    return {key.replace("video_marker_", "video_crossing_marker_"): value
            for key, value in metrics.items()}, passed
