"""Source-only semantic replay and rendering for the Physics family. Only fields that describe the visible initial state or an explicitly stated
rule are read here."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from data.common.schemas import CanonicalSample


RenderProgress = Callable[[float, float], Image.Image]


@dataclass(frozen=True)
class RotationCarrier:
    """A rigid body whose turning phase the source draws into the picture.

    ``center`` and the radius band bound an annulus in 512-space in which
    everything static is rotationally symmetric and everything else rotates with
    the body, so the band's angular profile carries the phase and nothing else.
    ``degrees_per_progress`` is the signed angle the source rule turns the body
    through over the whole process, positive clockwise as the viewer reads it;
    the rotation is linear in progress, so the declared angle between two
    progress values is that constant times their difference.  The Verifier reads
    the body's turning back out of the candidate and requires it never to run
    backwards and never to outrun one legal progress advance of this declared
    sweep; the constant also calibrates the reader against the source's own
    renders.  Because both the reading and the bound are in this same declared
    angle, the gate does not change if the scene is paced faster or slower.
    """

    name: str
    center: tuple[float, float]
    inner_radius: float
    outer_radius: float
    degrees_per_progress: float


@dataclass(frozen=True)
class ProgressCarrier:
    """A quantity the source rule forbids the picture from taking backwards.

    ``samples`` are points in 512-space the process advances *through* and
    ``rest_rgb`` the colour they carry until it reaches them, so the number of
    reached points is the extent the process has reached: trace already drawn,
    water already standing, ground a domino has already covered.  ``direction``
    is ``+1`` when the rule lets that extent only grow and ``0`` when it may move
    both ways but only ever in the direction the current step declares - a
    container that receives water and later gives it away.  A ``+1`` carrier is
    additionally gated against travelling backwards at all; every carrier is
    gated on the distance it travels in total and on how far one frame may move
    it, both measured against the source's own renders.  Each carrier is a
    member in its own right: one container, one domino, one chamber.

    ``fill_rgb`` is the colour the process leaves at a point it has *covered*,
    and it says how the extent is legible at a single pixel.  A carrier whose
    ink is a solid body - a domino's rectangle - covers each of its own sample
    points completely, so a reached point is one that has taken the body's own
    colour and a point showing anything else is showing something else.  A
    carrier whose ink is a thin stroke - a drawn beam, a ball's trace - only
    partly covers the pixels its centreline rounds to, so its reached points are
    the ones that have departed from rest at all and it declares ``None``.  The
    distinction matters because the scenes are downsampled with a filter that
    lifts a neighbourhood above its own content before an edge arrives: a
    solid-body carrier read by bare departure counts foreign slivers - the tail
    of a dashed link, the antialiased boundary of an already-settled body - and
    those slivers drop back under a bare threshold for a frame as the body's
    edge approaches, which is motion the process never made.
    """

    name: str
    samples: tuple[tuple[float, float], ...]
    rest_rgb: tuple[int, int, int]
    fill_rgb: tuple[int, int, int] | None
    direction: int


def carrier_polyline_samples(
    points: Sequence[Sequence[float]],
    *,
    step: float = 1.0,
    self_clearance: float = 0.0,
) -> list[tuple[float, float]]:
    """Sample points one pixel apart along a declared polyline.

    A traced path, a beam, or the ground a domino covers all advance along a
    line the source knows exactly, so the natural sample set is that line itself
    at the resolution the picture is drawn in.

    ``self_clearance`` drops the points a *distant* stretch of the same path
    comes closer to than that.  A scene whose moving object is wider than the
    line it draws reaches such a point before the line does and leaves it again
    without having drawn there, which is a real dip in the reading and not
    motion; the source's own renders show the dip, so leaving those points in
    would only raise the floor for the whole path.  Two points count as distant
    when they are more than ``2 * self_clearance`` apart *along* the path, which
    keeps every point's own neighbourhood - including the fold at a reflection -
    where the object does pass in order.
    """

    path = [(float(point[0]), float(point[1])) for point in points]
    if len(path) < 2:
        raise ValueError("a polyline carrier needs at least two points")
    lengths = [math.dist(first, second) for first, second in zip(path, path[1:])]
    total = sum(lengths)
    if total <= 0.0:
        raise ValueError("a polyline carrier needs positive length")
    samples: list[tuple[float, float]] = []
    for index in range(int(total / max(step, 1e-6)) + 1):
        remaining = index * step
        for segment, length in enumerate(lengths):
            if remaining <= length or segment == len(lengths) - 1:
                fraction = min(1.0, remaining / length) if length > 0.0 else 0.0
                first, second = path[segment], path[segment + 1]
                samples.append(
                    (
                        first[0] + (second[0] - first[0]) * fraction,
                        first[1] + (second[1] - first[1]) * fraction,
                    )
                )
                break
            remaining -= length
    if self_clearance <= 0.0:
        return samples
    grid = np.asarray(samples, dtype=np.float64)
    span = int(round(2.0 * self_clearance / max(step, 1e-6)))
    near = np.hypot(
        grid[:, None, 0] - grid[None, :, 0], grid[:, None, 1] - grid[None, :, 1]
    ) < float(self_clearance)
    order = np.arange(grid.shape[0])
    near &= np.abs(order[:, None] - order[None, :]) > span
    return [point for point, reachable in zip(samples, ~near.any(axis=1)) if reachable]


def carrier_column_samples(
    columns: Sequence[float], top: float, bottom: float, *, step: float = 1.0
) -> list[tuple[float, float]]:
    """Sample points one pixel apart down one or more vertical lines.

    A water level is read by how much of a column stands wet, so the columns are
    placed inside the vessel and clear of anything that is not its own water -
    the wall, the tick marks, and the pour stream that falls down its centre.
    """

    rows = int(max(0.0, float(bottom) - float(top)) / max(step, 1e-6))
    if rows < 1 or not len(columns):
        raise ValueError("a column carrier needs a non-empty span and at least one column")
    return [
        (float(column), float(top) + index * step)
        for column in columns
        for index in range(rows + 1)
    ]


@dataclass(frozen=True)
class PhysicsReplay:
    answer: Any
    render_progress: RenderProgress
    semantic_checkpoints: tuple[float, ...]
    source_variant: str
    # ``local_opacity`` identifies post-process answer emphasis that must pass
    # the complete-footprint pixel gate. ``semantic_terminal`` is a physical
    # terminal commit and ``none`` means there is no mark-only delta.
    answer_overlay_mode: str = "none"
    # Source-derived per-member evidence boxes ``(name, (x0, y0, x1, y1))`` in
    # 512-space.  Each named member (a domino, container, gear, ball, or
    # detector) is hard-gated individually against the source-rendered
    # terminal, so one wrong member can never be averaged away by the others.
    member_regions: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()
    # ``False`` marks a physically static process (for example a jammed gear
    # train): the whole video is the input plus one element-local answer
    # reveal, so motion-progress gates do not apply.
    motion: bool = True
    # Bodies whose turning direction the source declares and draws.  Each is
    # hard-gated on its own: a candidate in which any one of them is ever
    # seen turning the wrong way is rejected, however short the reversal.
    rotation_carriers: tuple[RotationCarrier, ...] = ()
    # Extents the source rule only ever lets move one way - a lengthening trace,
    # a rising level, a reaching domino.  Each is hard-gated on its own.
    progress_carriers: tuple[ProgressCarrier, ...] = ()
    # Small numeric answers need glyph readings as well as region appearance.
    numeric_regions: tuple[tuple[str, tuple[int, int, int, int], int], ...] = ()
    # Read the source font independently of which values happen to be answers.
    numeric_templates: tuple[tuple[int, Image.Image], ...] = ()
    # Fixed overlays and blocked bodies remain protected even next to a
    # moving silhouette's raster guard band.
    preserved_regions: tuple[tuple[float, float, float, float], ...] = ()


def source_state(sample: CanonicalSample) -> Mapping[str, Any]:
    semantic = sample.semantic_input
    if not isinstance(semantic, Mapping):
        return {}
    for key in ("semantic_state", "source_contract", "structured_input"):
        value = semantic.get(key)
        if isinstance(value, Mapping):
            return value
    return semantic


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Physics source lacks {name}")
    return value


def _float_pair(value: Any, name: str) -> tuple[float, float]:
    try:
        result = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Physics source has malformed {name}") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"Physics source has non-finite {name}")
    return result


def _declared_progress_carriers(renderer: Any, task: Mapping[str, Any]) -> tuple[ProgressCarrier, ...]:
    """Collect a renderer's own declaration of its one-way extents.

    A renderer that declares none simply has none; ``elastic_collision_track`` is
    the family's honest case, since equal-mass exchange sends balls back the way
    they came and no visible quantity of that scene grows in one direction.
    """

    declare = getattr(renderer, "progress_carriers", None)
    if not callable(declare):
        return ()
    return tuple(
        ProgressCarrier(
            str(name),
            tuple((float(point[0]), float(point[1])) for point in samples),
            (int(rest[0]), int(rest[1]), int(rest[2])),
            None if fill is None else (int(fill[0]), int(fill[1]), int(fill[2])),
            int(direction),
        )
        for name, samples, rest, fill, direction in declare(task)
    )


def _ball_replay(state: Mapping[str, Any]) -> PhysicsReplay:
    from data.physics.ball_bounces_given_time.generator import BallBouncesGivenTimePattern

    evidence = _mapping(state.get("source_evidence"), "ball source_evidence")
    start = _float_pair(evidence.get("start"), "ball start")
    velocity = _float_pair(evidence.get("initial_velocity"), "ball initial_velocity")
    bounds = tuple(float(value) for value in evidence.get("center_bounds", ()))
    count = int(evidence.get("bounce_count"))
    candidates_raw = evidence.get("candidates")
    if len(bounds) != 4 or not isinstance(candidates_raw, list) or len(candidates_raw) != 4 or not 3 <= count <= 9:
        raise ValueError("ball source geometry or collision count is invalid")
    candidates: list[tuple[float, float]] = []
    labels: list[str] = []
    for candidate in candidates_raw:
        item = _mapping(candidate, "ball candidate")
        labels.append(str(item.get("label")))
        candidates.append((float(item.get("x")), float(item.get("y"))))
    if sorted(labels) != list("ABCD"):
        raise ValueError("ball candidates must contain A-D exactly once")

    renderer = BallBouncesGivenTimePattern(Path(tempfile.gettempdir()), seed=0, video=False)
    radius = 14
    margin = int(round(bounds[0] - radius))
    expected_bounds = renderer._center_bounds(512, 512, margin, radius)
    if any(abs(left - right) > 1e-6 for left, right in zip(bounds, expected_bounds)):
        raise ValueError("ball center bounds disagree with the visible arena")
    trace, terminal = renderer._simulate(512, 512, margin, radius, start, velocity, count)
    distances = [math.hypot(point[0] - terminal[0], point[1] - terminal[1]) for point in candidates]
    best = min(range(4), key=distances.__getitem__)
    if distances[best] > 1.0 or sorted(distances)[1] - distances[best] <= 1e-6:
        raise ValueError("ball candidates do not identify one replayed terminal collision")
    answer = labels[best]
    task = {
        "margin": margin,
        "radius": radius,
        "center_bounds": list(bounds),
        "start": start,
        "velocity": velocity,
        "bounce_count": count,
        "trace": trace,
        "candidates": candidates,
    }
    carriers = _declared_progress_carriers(renderer, task)
    lengths = [math.dist(first, second) for first, second in zip(trace, trace[1:])]
    total_length = sum(lengths)

    def render(progress: float, mark_answer: bool) -> Image.Image:
        del mark_answer
        distance = max(0.0, min(1.0, float(progress))) * total_length
        prefix = [trace[0]]
        current = trace[0]
        for index, segment_length in enumerate(lengths):
            if distance >= segment_length - 1e-9:
                current = trace[index + 1]
                prefix.append(current)
                distance -= segment_length
                continue
            fraction = 0.0 if segment_length <= 1e-9 else distance / segment_length
            first, second = trace[index], trace[index + 1]
            current = (
                first[0] + (second[0] - first[0]) * fraction,
                first[1] + (second[1] - first[1]) * fraction,
            )
            if fraction > 0.0:
                prefix.append(current)
            break
        return renderer._draw_frame(task, path_points=prefix, current_position=current)

    cumulative = 0.0
    checkpoints = [0.0]
    for length in lengths:
        cumulative += length
        checkpoints.append(cumulative / total_length)
    # the settled ball and every candidate marker carry their own
    # evidence box, so a ball on the wrong marker cannot be averaged away.
    member_regions = [
        ("final_ball", (terminal[0] - 24.0, terminal[1] - 24.0, terminal[0] + 24.0, terminal[1] + 24.0)),
    ]
    for label, point in zip(labels, candidates):
        member_regions.append(
            (f"marker_{label}", (point[0] - 24.0, point[1] - 24.0, point[0] + 24.0, point[1] + 24.0))
        )
    return PhysicsReplay(
        answer,
        render,
        tuple(checkpoints),
        "elastic_collision_source_v1",
        member_regions=tuple(member_regions),
        progress_carriers=carriers,
    )


def _copy_records(value: Any, name: str, *keys: str) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Physics source lacks {name}")
    records = []
    for item in value:
        record = _mapping(item, name)
        missing = [key for key in keys if key not in record]
        if missing:
            raise ValueError(f"Physics source {name} lacks fields: {missing}")
        records.append({key: record[key] for key in keys})
    return records


def _physical_scene_replay(task_name: str, state: Mapping[str, Any]) -> PhysicsReplay:
    motion = True
    if task_name == "domino_reach_cascade":
        from data.physics.core.domino_physics import (
            DominoReachCascadePattern as Pattern,
            simulate_topple_waves,
        )

        nodes = _copy_records(state.get("nodes"), "domino nodes", "id", "base", "dir", "height")
        edges = _copy_records(state.get("edges"), "domino edges", "from", "to", "gap")
        scale = float(state.get("px_per_unit") or 0.0)
        if not 0.2 <= scale <= 2.0:
            raise ValueError("domino px_per_unit is out of range")
        task: dict[str, Any] = {"nodes": nodes, "edges": edges, "px_per_unit": scale}
        waves, _failed, _fallen = simulate_topple_waves(nodes, edges)
        total = len(waves)
        checkpoints = tuple(index / total for index in range(total + 1))
    elif task_name == "gear_train_direction":
        from data.physics.core.gear_physics import (
            GearTrainDirectionPattern as Pattern,
            solve_directions,
        )

        gears = _copy_records(state.get("gears"), "gear disks", "id", "center", "radius")
        belts = [
            [int(pair[0]), int(pair[1])]
            for pair in (state.get("belts") if isinstance(state.get("belts"), (list, tuple)) else [])
        ]
        driver_direction = str(state.get("driver_direction") or "")
        if driver_direction not in {"CW", "CCW"}:
            raise ValueError("gear driver direction must be CW or CCW")
        target = int(state.get("target") or 0)
        if target != len(gears) or not 3 <= len(gears) <= 7:
            raise ValueError("gear target must be the highest-numbered gear")
        task = {
            "gears": gears,
            "belts": belts,
            "driver_direction": driver_direction,
            "target": target,
        }
        _signs, conflict = solve_directions(gears, belts, driver_direction)
        motion = not bool(conflict)
        checkpoints = (0.0, 1.0)
    elif task_name == "chamber_flow_fill":
        from data.physics.core.chamber_physics import (
            ChamberFlowFillPattern as Pattern,
            simulate_fill,
        )

        widths = [float(value) for value in (state.get("widths") or ())]
        crests = [float(value) for value in (state.get("crests") or ())]
        source_chamber = int(state.get("source") or 0)
        if not 3 <= len(widths) <= 6 or len(crests) != len(widths) - 1:
            raise ValueError("chamber state shape is invalid")
        task = {"widths": widths, "crests": crests, "source": source_chamber}
        fill = simulate_fill(widths, crests, source_chamber)
        total = fill["total_time"]
        checkpoints = tuple(
            sorted({0.0, 1.0, *(fill["first_wet_times"][c] / total for c in fill["wet_order"])})
        )
    elif task_name == "elastic_collision_track":
        from data.physics.core.collision_physics import (
            ElasticCollisionTrackPattern as Pattern,
            simulate_collisions,
        )

        starts = [int(value) for value in (state.get("starts") or ())]
        velocities = [int(value) for value in (state.get("velocities") or ())]
        horizon = int(state.get("horizon") or 0)
        if not 3 <= len(starts) <= 4 or len(velocities) != len(starts) or not 1 <= horizon <= 8:
            raise ValueError("collision state shape is invalid")
        task = {"starts": starts, "velocities": velocities, "horizon": horizon}
        sim = simulate_collisions(starts, velocities, horizon)
        checkpoints = tuple(
            sorted({0.0, 1.0, *(float(event["time"]) / horizon for event in sim["events"])})
        )
    elif task_name == "water_pour_levels":
        from data.physics.core.liquid_physics import WaterPourLevelsPattern as Pattern

        capacities = [int(value) for value in (state.get("capacities") or ())]
        levels = [int(value) for value in (state.get("levels") or ())]
        pours = [
            [int(pair[0]), int(pair[1])]
            for pair in (state.get("pours") or ())
        ]
        if not 3 <= len(capacities) <= 4 or len(levels) != len(capacities) or not 1 <= len(pours) <= 5:
            raise ValueError("pour state shape is invalid")
        if any(not 0 <= level <= cap for level, cap in zip(levels, capacities)):
            raise ValueError("pour levels exceed visible capacities")
        task = {"capacities": capacities, "levels": levels, "pours": pours}
        probe = Pattern(Path(tempfile.gettempdir()), seed=0, video=False)
        checkpoints = tuple(probe._time_boundaries(task))
    elif task_name == "mirror_aim_detector":
        from data.physics.core.optics_physics import MirrorAimDetectorPattern as Pattern

        evidence = _mapping(state.get("source_evidence"), "mirror source_evidence")
        source = _float_pair(evidence.get("source"), "mirror source")
        entry = _float_pair(evidence.get("entry_dir"), "mirror entry direction")
        mirrors = _copy_records(evidence.get("mirrors"), "mirror bars", "center", "tangent", "half_length")
        detectors = _copy_records(evidence.get("detectors"), "mirror detectors", "label", "center")
        if len(detectors) != 4 or sorted(str(d["label"]) for d in detectors) != list("ABCD"):
            raise ValueError("mirror detectors must contain A-D exactly once")
        if not 2 <= len(mirrors) <= 4:
            raise ValueError("mirror box requires two to four mirrors")
        task = {
            "source": list(source),
            "entry_dir": list(entry),
            "mirrors": mirrors,
            "detectors": detectors,
        }
        probe = Pattern(Path(tempfile.gettempdir()), seed=0, video=False)
        path = probe._path(task)
        lengths = [math.dist(first, second) for first, second in zip(path, path[1:])]
        total_length = sum(lengths)
        cumulative, marks = 0.0, [0.0]
        for length in lengths:
            cumulative += length
            marks.append(cumulative / total_length)
        checkpoints = tuple(marks)
    else:
        raise ValueError(f"unsupported Physics source replay task: {task_name}")

    renderer = Pattern(Path(tempfile.gettempdir()), seed=0, video=False)
    preserved_regions = []
    if task_name == "domino_reach_cascade":
        from data.physics.core.domino_physics import _rect_corners
        geometry = renderer._scene(task)
        def protect(center, half_width, half_height):
            x, y = center
            preserved_regions.append((x-half_width, y-half_height, x+half_width, y+half_height))
        for node in geometry["nodes"].values():
            protect(node["disc_center"], 12, 12)
            protect(node["height_label"], 21, 8)
            if not node["fallen"]:
                corners = _rect_corners(node["base"], node["dir"], node["half_thick_px"],
                                        node["half_thick_px"], node["half_width_px"])
                xs, ys = zip(*corners)
                preserved_regions.append((min(xs)-2, min(ys)-2, max(xs)+2, max(ys)+2))
        for edge in geometry["edges"]:
            protect(edge["gap_label"], 20, 8)
    answer = renderer._answer_payload(task)
    mode = str(renderer.answer_overlay_mode or "")
    if mode not in {"local_opacity", "semantic_terminal", "none"}:
        raise ValueError(f"{task_name} has no valid answer-overlay mode")
    member_regions = tuple(
        (str(name), tuple(float(value) for value in box))
        for name, box in renderer.terminal_member_boxes(task)
    )
    declare_rotation = getattr(renderer, "rotation_carriers", None)
    rotation_carriers = (
        tuple(
            RotationCarrier(
                str(name),
                (float(center[0]), float(center[1])),
                float(inner),
                float(outer),
                float(degrees),
            )
            for name, center, inner, outer, degrees in declare_rotation(task)
        )
        if motion and callable(declare_rotation)
        else ()
    )
    progress_carriers = _declared_progress_carriers(renderer, task) if motion else ()
    numeric_regions = ()
    numeric_templates = ()
    if task_name == "chamber_flow_fill":
        geometry = renderer._scene(task)
        order = renderer._replay(task)["wet_order"]
        fields = []
        for index, chamber in enumerate(geometry["chambers"], start=1):
            x = round((chamber["left"] + chamber["right"]) / 2)
            y = round(geometry["badge_y"])
            fields.append((f"chamber_{index}_rank", (x - 11, y - 11, x + 12, y + 12), order.index(index) + 1))
        numeric_regions = tuple(fields)
        numeric_templates = renderer.numeric_glyph_templates(max(9, len(fields)))

    def render(progress: float, answer_opacity: float) -> Image.Image:
        logical_amount = max(0.0, min(1.0, float(progress)))
        return renderer._render_with_answer_opacity(
            task,
            logical_amount,
            answer_opacity=(
                max(0.0, min(1.0, float(answer_opacity)))
                if mode != "none" and logical_amount >= 1.0 - 1e-9
                else 0.0
            ),
        )

    return PhysicsReplay(
        answer,
        render,
        tuple(checkpoints),
        f"{task_name}_visible_source_v2",
        mode,
        member_regions,
        motion,
        rotation_carriers,
        progress_carriers,
        numeric_regions,
        numeric_templates,
        tuple(preserved_regions),
    )


def replay_physics(sample: CanonicalSample, *, modality: str | None = None) -> PhysicsReplay:
    del modality
    state = source_state(sample)
    if sample.task_name == "ball_bounces_given_time":
        return _ball_replay(state)
    return _physical_scene_replay(sample.task_name, state)


__all__ = [
    "PhysicsReplay",
    "ProgressCarrier",
    "RotationCarrier",
    "carrier_column_samples",
    "carrier_polyline_samples",
    "replay_physics",
    "source_state",
]
