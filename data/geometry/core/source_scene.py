"""Source-owned preservation observations for additive geometry task parsers."""
from __future__ import annotations

from data.common.verifiers.region_gates import STATIC_REGION_FLOOR, static_region_errors
from data.geometry.core.terminal_scene import point_target_delivery_ink_state, final_marker_state_gate


def observe_additive_source_scene(frame, source, record, *, allowed, discs, prefix):
    """Keep each original scene member and each static region readable.

    Task-owned construction/highlight parsers decide the additions and their
    order. This layer tolerates reconstructed source edges while rejecting a
    missing scene object, compact paint or a changed unselected answer marker.
    """
    scene_metrics, scene_pass = point_target_delivery_ink_state(
        frame, source, record, discs=discs,
        construction_declared=False, prefix=prefix,
    )
    static_error, static_violations = static_region_errors(frame, source, allowed == 0)
    marker_metrics, marker_pass = final_marker_state_gate(frame, source, record, prefix=prefix)
    return {
        "source_preservation_pass": bool(scene_pass and static_violations == 0),
        "source_scene_metrics": scene_metrics,
        "source_static_region_max_error": static_error,
        "source_static_region_violations": static_violations,
        "source_static_region_bound": STATIC_REGION_FLOOR,
        "marker_state_pass": marker_pass,
        "marker_state_metrics": marker_metrics,
    }
