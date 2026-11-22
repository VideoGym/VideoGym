"""Shared temporal rendering policy for grid-based tasks.

Spatial Navigation reference media is serialized as exact semantic snapshots:

* hold the complete current state for a fixed cadence;
* switch directly to the next complete state;
* commit attached objects, traces, counters, and world updates atomically;
* finish with a stable terminal hold.

Periodic hazards use the same fixed-clock slots, but each slot is itself an
exact state.  There are no sub-cell positions, interpolation frames, fades, or
other undeclared transition states.
"""

from __future__ import annotations

from data.common.video_encoding import MAX_VIDEO_FRAMES, MOTION_VIDEO_FRAMES, MULTISTEP_VIDEO_FRAMES


STATE_HOLD_FRAMES = 2
DISCRETE_STATE_HOLD_FRAMES = STATE_HOLD_FRAMES
MAX_CLOCKED_STEPS = 12


def start_hold_frames(fps: int) -> int:
    return max(4, int(fps) // 4)


def end_hold_frames(fps: int) -> int:
    return max(6, int(fps) // 2)


def frame_budget_for_discrete_state_count(state_count: int) -> int:
    """Choose a standard media budget for an exact-state timeline."""

    states = max(1, int(state_count))
    preferred = start_hold_frames(16) + states * DISCRETE_STATE_HOLD_FRAMES + end_hold_frames(16)
    if preferred <= MOTION_VIDEO_FRAMES:
        return MOTION_VIDEO_FRAMES
    if preferred <= MULTISTEP_VIDEO_FRAMES:
        return MULTISTEP_VIDEO_FRAMES
    return MAX_VIDEO_FRAMES


def discrete_state_timeline(state_count: int, *, fps: int = 16) -> list[int]:
    """Per-state frame counts for an exact-state timeline.

    Every state is held for the same number of frames and then switched
    directly to the next one, which is the transition model a task with a
    discrete state space must use.  Only the opening and closing holds are
    longer, so the input and the terminal state stay comfortably readable
    without making any intermediate state privileged.
    """

    states = max(1, int(state_count))
    hold = max(DISCRETE_STATE_HOLD_FRAMES, int(fps) // 4)
    holds = [hold] * states
    holds[0] += start_hold_frames(fps)
    holds[-1] += end_hold_frames(fps)
    return holds


def frame_budget_for_frame_count(frame_count: int) -> int:
    """Choose a standard budget from an already assembled semantic timeline."""

    count = max(1, int(frame_count))
    if count <= MOTION_VIDEO_FRAMES:
        return MOTION_VIDEO_FRAMES
    if count <= MULTISTEP_VIDEO_FRAMES:
        return MULTISTEP_VIDEO_FRAMES
    return MAX_VIDEO_FRAMES


__all__ = [
    "DISCRETE_STATE_HOLD_FRAMES",
    "MAX_CLOCKED_STEPS",
    "STATE_HOLD_FRAMES",
    "discrete_state_timeline",
    "end_hold_frames",
    "frame_budget_for_discrete_state_count",
    "frame_budget_for_frame_count",
    "start_hold_frames",
]
