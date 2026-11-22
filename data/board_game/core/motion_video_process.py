"""Source-derived process verification for the L-waypoint board video.

The final-state verifier is intentionally insufficient: a correct last frame
can hide a teleport, skipped landing, early target entry, or forbidden yellow
trail.  This module reconstructs a finite set of *semantic motion probes*
directly from ``BoardReplay.source_payload``.  Candidate frames must visit
those probes in order.  No authored/reference video is read.

The probes are not an FPS contract.  Extra holds and extra genuine in-between
positions are allowed.  They merely require several ordered positions on each
source-to-destination segment plus the source-derived commit states.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from data.common.verifiers.region_gates import (
    STATIC_REGION_FLOOR,
    static_region_errors,
)


_SUPPORTED = frozenset({"chess_l_waypoint_search"})
_CANVAS = 512


@dataclass(frozen=True)
class _Probe:
    image: Image.Image
    kind: str
    action: int
    progress: float


def _rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(frame)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError("motion video frames must be RGB-like images")
        array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape != (_CANVAS, _CANVAS, 3):
        raise ValueError(f"motion video frame must be {_CANVAS}x{_CANVAS} RGB")
    return np.ascontiguousarray(array)


def _chess_route_probes(payload: Mapping[str, Any]) -> List[_Probe]:
    from data.board_game.core.chess_engine import (
        ChessBoardRenderer,
        l_search_board,
        visited_waypoints_for_prefix,
    )

    try:
        start = tuple(payload["start"])
        waypoints = tuple(tuple(cell) for cell in payload["waypoints"])
        target = tuple(payload["target"])
        route = tuple(tuple(cell) for cell in payload["route"])
    except (KeyError, TypeError) as exc:
        raise ValueError("L-route source replay is incomplete") from exc
    if not route:
        raise ValueError("L-route source replay has no hops")
    renderer = ChessBoardRenderer()

    def commit_image(landing_count: int) -> Image.Image:
        token = start if landing_count == 0 else route[landing_count - 1]
        landed = route[:landing_count]
        return renderer.render(
            l_search_board(token),
            waypoints=waypoints,
            visited_waypoints=visited_waypoints_for_prefix(waypoints, landed),
            target=target,
        )

    probes = [_Probe(commit_image(0), "initial", -1, 0.0)]
    for action, destination in enumerate(route):
        source = start if action == 0 else route[action - 1]
        landed = route[:action]
        before_board = l_search_board(source)
        for index in range(1, 9):
            probes.append(
                _Probe(
                    renderer.render(
                        before_board,
                        waypoints=waypoints,
                        visited_waypoints=visited_waypoints_for_prefix(waypoints, landed),
                        target=target,
                        moving_piece=("N", True),
                        moving_from=source,
                        moving_to=destination,
                        progress=index / 9.0,
                    ),
                    "motion",
                    action,
                    index / 9.0,
                )
            )
        probes.append(_Probe(commit_image(action + 1), "commit", action, 1.0))
    return probes


def _build_probes(replay: Any) -> List[_Probe]:
    task_name = str(getattr(replay, "task_name", ""))
    if task_name not in _SUPPORTED:
        raise ValueError(f"motion process verifier does not support {task_name!r}")
    payload = getattr(replay, "source_payload", None)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{task_name}: source_payload must be a mapping")
    return _chess_route_probes(payload)


def _dilate(mask: np.ndarray, size: int = 9) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(size=size))) > 0


def _mae(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None = None) -> float:
    delta = np.abs(left.astype(np.float32) - right.astype(np.float32))
    if mask is not None:
        if not np.any(mask):
            return 0.0
        delta = delta[mask]
    return float(np.mean(delta) / 255.0)


def _gold_trail_pixels(frame: np.ndarray) -> int:
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    mask = (
        (red >= 145)
        & (green >= 100)
        & (blue <= 140)
        & (red - green >= 15)
        & (green - blue >= 25)
    )
    return int(np.sum(mask))


def _probe_masks(expected: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
    local: List[np.ndarray] = []
    for index, current in enumerate(expected):
        neighbours: List[np.ndarray] = []
        if index:
            neighbours.append(expected[index - 1])
        if index + 1 < len(expected):
            neighbours.append(expected[index + 1])
        changed = np.zeros(current.shape[:2], dtype=bool)
        for neighbour in neighbours:
            changed |= np.max(np.abs(current.astype(np.int16) - neighbour.astype(np.int16)), axis=2) >= 5
        local.append(_dilate(changed, 9))
    dynamic = np.zeros(expected[0].shape[:2], dtype=bool)
    baseline = expected[0].astype(np.int16)
    for current in expected[1:]:
        dynamic |= np.max(np.abs(current.astype(np.int16) - baseline), axis=2) >= 5
    return local, _dilate(dynamic, 11)


def _ordered_alignment(
    frames: Sequence[np.ndarray],
    expected: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> Tuple[List[int], List[float], List[float], float]:
    """Minimum-cost strictly ordered probe-to-frame matching."""

    probe_count, frame_count = len(expected), len(frames)
    if frame_count < probe_count:
        return [], [], [], float("inf")
    local = np.empty((probe_count, frame_count), dtype=np.float64)
    global_distance = np.empty_like(local)
    for probe_index, template in enumerate(expected):
        mask = masks[probe_index]
        for frame_index, frame in enumerate(frames):
            local[probe_index, frame_index] = _mae(frame, template, mask)
            global_distance[probe_index, frame_index] = _mae(frame, template)
    # Local structure carries token position; global distance prevents a
    # candidate from altering the rest of the board while matching that crop.
    cost = local + 2.0 * global_distance
    total = np.full((probe_count, frame_count), np.inf, dtype=np.float64)
    parent = np.full((probe_count, frame_count), -1, dtype=np.int32)
    total[0] = cost[0]
    for probe_index in range(1, probe_count):
        best_value = np.inf
        best_position = -1
        # Prefix minimum over the preceding row enforces strict time order.
        for frame_index in range(1, frame_count):
            previous = total[probe_index - 1, frame_index - 1]
            if previous < best_value:
                best_value = previous
                best_position = frame_index - 1
            total[probe_index, frame_index] = best_value + cost[probe_index, frame_index]
            parent[probe_index, frame_index] = best_position
    end = int(np.argmin(total[-1]))
    if not np.isfinite(total[-1, end]):
        return [], [], [], float("inf")
    positions = [end]
    for probe_index in range(probe_count - 1, 0, -1):
        end = int(parent[probe_index, end])
        if end < 0:
            return [], [], [], float("inf")
        positions.append(end)
    positions.reverse()
    local_hits = [float(local[index, position]) for index, position in enumerate(positions)]
    global_hits = [float(global_distance[index, position]) for index, position in enumerate(positions)]
    return positions, local_hits, global_hits, float(total[-1, positions[-1]] / probe_count)


def verify_motion_process(frames: Sequence[Any], replay: Any) -> Tuple[Dict[str, Any], bool]:
    """Verify ordered moving-token process from source semantics alone.

    ``frames`` may contain RGB numpy arrays or PIL images.  ``replay`` is the
    BoardReplay returned by ``source_replay.replay_board_sample`` (duck typing
    is deliberate to avoid importing the verifier/replay layer).
    """

    task_name = str(getattr(replay, "task_name", ""))
    metrics: Dict[str, Any] = {
        "video_motion_process_verifier": "source_motion_probe_v1",
        "video_motion_reference_media_used": False,
        "video_motion_task_supported": float(task_name in _SUPPORTED),
    }
    try:
        candidate = [_rgb_array(frame) for frame in frames]
        probes = _build_probes(replay)
        expected = [_rgb_array(probe.image) for probe in probes]
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        metrics["video_motion_error"] = str(exc)
        metrics["video_motion_process_score"] = 0.0
        return metrics, False
    if not candidate:
        metrics["video_motion_error"] = "candidate contains no frames"
        metrics["video_motion_process_score"] = 0.0
        return metrics, False

    masks, dynamic_mask = _probe_masks(expected)
    positions, local_hits, global_hits, alignment_cost = _ordered_alignment(
        candidate, expected, masks
    )
    aligned = bool(positions)
    max_local = max(local_hits, default=1.0)
    max_global = max(global_hits, default=1.0)

    # WebM intra-frame compression remains far below these limits, while a
    # dissolve, hard cut, early removal, or early overlay misses multiple
    # source-rendered probes by a wide margin.
    template_pass = bool(aligned and max_local <= 0.055 and max_global <= 0.018)

    motion_indices = [index for index, probe in enumerate(probes) if probe.kind == "motion"]
    actions = sorted({probes[index].action for index in motion_indices})
    monotonic_actions = True
    action_probe_counts: List[int] = []
    action_spans: List[int] = []
    for action in actions:
        indices = [index for index in motion_indices if probes[index].action == action]
        action_probe_counts.append(len(indices))
        if not positions:
            monotonic_actions = False
            action_spans.append(0)
            continue
        times = [positions[index] for index in indices]
        action_spans.append(times[-1] - times[0] + 1)
        monotonic_actions &= len(indices) >= 5 and all(
            left < right for left, right in zip(times, times[1:])
        )

    # Every mapped frame should retain source-rendered pixels outside all
    # legal dynamic regions.  This catches mutation of an unrelated checker,
    # marker, label, grid edge, background, or camera framing.
    static_mask = ~dynamic_mask
    static_maes: List[float] = []
    static_changed_fractions: List[float] = []
    static_region_maxima: List[float] = []
    if np.any(static_mask):
        # Pixels outside the union of every legal source-derived mutation are
        # identical in all probes, including during a legitimate in-between
        # position.  Inspect *every* candidate frame so a one-frame camera
        # shift or unrelated-token flicker cannot hide between mapped probes.
        for frame in candidate:
            delta = np.max(
                np.abs(frame.astype(np.int16) - expected[0].astype(np.int16)),
                axis=2,
            )
            static_maes.append(float(np.mean(delta[static_mask]) / 255.0))
            static_changed_fractions.append(float(np.mean(delta[static_mask] > 28)))
            # Both statistics above are means over the whole static mask, so a
            # patch painted over one untouched checker is 0.2% of them.  The
            # same statistic per 32x32 region is not diluted (section 11.8).
            static_region_maxima.append(
                static_region_errors(
                    frame.astype(np.float32), expected[0].astype(np.float32), static_mask
                )[0]
            )
    max_static_mae = max(static_maes, default=1.0)
    max_static_changed = max(static_changed_fractions, default=1.0)
    max_static_region = max(static_region_maxima, default=0.0)
    static_pass = bool(
        max_static_mae <= 0.012
        and max_static_changed <= 0.004
        and max_static_region <= STATIC_REGION_FLOOR
    )

    # Map every frame to its nearest position on the complete source-derived
    # trajectory.  Genuine extra interpolation states fall between adjacent
    # probes and therefore retain a nondecreasing nearest label.  A rollback,
    # premature terminal, or reordered action forces this timeline backward.
    timeline_cost = np.empty((len(candidate), len(expected)), dtype=np.float64)
    for frame_index, frame in enumerate(candidate):
        for probe_index, template in enumerate(expected):
            timeline_cost[frame_index, probe_index] = (
                _mae(frame, template, dynamic_mask) + 2.0 * _mae(frame, template)
            )
    timeline_labels = np.argmin(timeline_cost, axis=1).astype(int).tolist()
    timeline_monotonic = all(
        left <= right for left, right in zip(timeline_labels, timeline_labels[1:])
    )
    initial_hold = sum(
        _mae(frame, expected[0], dynamic_mask) <= 0.045
        and _mae(frame, expected[0]) <= 0.018
        for frame in candidate[: max(3, min(6, len(candidate)))]
    )
    terminal_hold = sum(
        _mae(frame, expected[-1], dynamic_mask) <= 0.045
        and _mae(frame, expected[-1]) <= 0.018
        for frame in candidate[-max(4, min(8, len(candidate))) :]
    )
    endpoint_holds_pass = initial_hold >= 3 and terminal_hold >= 4

    kinds = [probe.kind for probe in probes]
    commit_indices = [index for index, kind in enumerate(kinds) if kind == "commit"]
    commit_pass = bool(
        positions
        and len(commit_indices) == len(actions)
        and all(
            positions[max(index for index in motion_indices if probes[index].action == action)]
            < positions[commit_indices[action]]
            for action in actions
        )
    )

    # These named scores expose the task-specific consequences of the same
    # source probe contract.  L-route probes retain committed waypoint state,
    # cannot put L on the target before the last source-derived hop, and never
    # authorize a yellow movement trail.
    gold_pixel_counts = [_gold_trail_pixels(frame) for frame in candidate]
    no_trail_pass = max(gold_pixel_counts, default=0) <= 32
    target_order_pass = bool(actions) and actions[-1] == max(actions) and template_pass and commit_pass

    passed = bool(
        template_pass
        and monotonic_actions
        and timeline_monotonic
        and endpoint_holds_pass
        and static_pass
        and commit_pass
        and no_trail_pass
        and target_order_pass
    )
    metrics.update(
        {
            "video_motion_frame_count": len(candidate),
            "video_motion_probe_count": len(probes),
            "video_motion_action_count": len(actions),
            "video_motion_alignment_cost": alignment_cost,
            "video_motion_max_local_probe_error": max_local,
            "video_motion_max_global_probe_error": max_global,
            "video_motion_template_sequence_score": float(template_pass),
            "video_motion_probe_frame_indices": positions,
            "video_motion_intermediate_probe_counts": action_probe_counts,
            "video_motion_action_frame_spans": action_spans,
            "video_motion_monotonic_positions_score": float(monotonic_actions),
            "video_motion_timeline_labels": timeline_labels,
            "video_motion_no_rollback_score": float(timeline_monotonic),
            "video_motion_initial_hold_count": initial_hold,
            "video_motion_terminal_hold_count": terminal_hold,
            "video_motion_endpoint_holds_score": float(endpoint_holds_pass),
            "video_motion_max_static_mae": max_static_mae,
            "video_motion_max_static_changed_fraction": max_static_changed,
            "video_motion_max_static_region_error": max_static_region,
            "video_motion_static_region_bound": STATIC_REGION_FLOOR,
            "video_motion_unrelated_board_static_score": float(static_pass),
            "video_motion_gold_pixel_counts": gold_pixel_counts,
            "video_motion_no_movement_trail_score": float(no_trail_pass),
            "video_motion_target_order_score": float(target_order_pass),
            "video_motion_process_score": float(passed),
        }
    )
    return metrics, passed


def run_self_check() -> Dict[str, bool]:
    """Exercise ideal sequences and direct process corruptions in memory."""

    from data.board_game.core.chess_engine import build_l_waypoint_scenario

    rng = random.Random(9137)
    l_scenario = build_l_waypoint_scenario(rng, difficulty="minimum")
    stub = SimpleNamespace(
        task_name="chess_l_waypoint_search",
        source_payload={
            "start": l_scenario.start,
            "waypoints": l_scenario.waypoints,
            "target": l_scenario.target,
            "route": l_scenario.unique_route,
        },
    )
    results: Dict[str, bool] = {}
    ideal = [_rgb_array(probe.image) for probe in _build_probes(stub)]
    good = [ideal[0]] * 4 + ideal[1:-1] + [ideal[-1]] * 5
    _, accepted = verify_motion_process(good, stub)
    assert accepted
    results["chess_l_waypoint_search_ideal"] = True

    hard_cut = [ideal[0]] * 8 + [ideal[-1]] * 8
    _, accepted = verify_motion_process(hard_cut, stub)
    assert not accepted
    results["hard_cut_rejected"] = True

    crossfade = [
        np.clip(
            ideal[0].astype(np.float32) * (1.0 - depth)
            + ideal[-1].astype(np.float32) * depth,
            0,
            255,
        ).astype(np.uint8)
        for depth in np.linspace(0.0, 1.0, len(ideal))
    ]
    _, accepted = verify_motion_process(crossfade, stub)
    assert not accepted
    results["endpoint_crossfade_rejected"] = True

    skipped = ideal[::2]
    _, accepted = verify_motion_process(skipped, stub)
    assert not accepted
    results["skipped_intermediates_rejected"] = True

    trailed: List[np.ndarray] = []
    for frame in good:
        image = Image.fromarray(frame.copy(), mode="RGB")
        ImageDraw.Draw(image).line((115, 115, 395, 395), fill=(224, 166, 54), width=6)
        trailed.append(np.asarray(image, dtype=np.uint8))
    trail_metrics, accepted = verify_motion_process(trailed, stub)
    assert not accepted
    assert trail_metrics["video_motion_no_movement_trail_score"] == 0.0
    results["yellow_trail_rejected"] = True
    return results


__all__ = ["verify_motion_process", "run_self_check"]
