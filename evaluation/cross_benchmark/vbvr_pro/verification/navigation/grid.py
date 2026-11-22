"""Symbolic grid route verification with no continuous reward terms.

Coordinates are normalised once to (row, column). A prediction must show one
agent, legal swept cell transitions, all required visits and the correct terminal cell.
Only prompts requesting optimality require an independently recomputed shortest
route. Equal-length alternatives and permitted legal detours are accepted.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.verification.annotation.selection import (
    _background_reference,
    _foreign_background_component,
)

PREFIXES = frozenset({"G-13", "G-15", "G-16", "G-18", "G-45", "G-47", "O-39"})
Cell = tuple[int, int]


@dataclass
class Spec:
    rows: int
    cols: int
    origin: tuple[float, float]
    cell_size: float
    start: Cell
    goal: Cell
    walls: set[Cell]
    required: list[Cell]
    ordered: bool
    color: np.ndarray
    radius: float
    key_items: list[dict]

    def cell(self, point: tuple[float, float]) -> Cell:
        x, y = point
        return (
            int(np.floor((y - self.origin[1]) / self.cell_size)),
            int(np.floor((x - self.origin[0]) / self.cell_size)),
        )

    def center(self, cell: Cell) -> tuple[float, float]:
        return (self.origin[0] + (cell[1] + 0.5) * self.cell_size, self.origin[1] + (cell[0] + 0.5) * self.cell_size)


def _spec(prefix: str, metadata: dict, image: np.ndarray) -> Spec | None:
    s = metadata.get("semantic_ground_truth") or {}
    p = metadata.get("parameters") or {}
    layers = (metadata.get("generic_declarative_render") or {}).get("first_frame", {}).get("layers", [])
    grid = next((q for q in layers if q.get("type") in {"grid_lines", "cell_grid"}), None)
    if prefix == "G-18" and grid is None:
        vertical = sorted(
            {
                q["points_px"][0][0]
                for q in layers
                if q.get("type") == "line"
                and len(q.get("points_px", [])) == 2
                and q["points_px"][0][0] == q["points_px"][1][0]
            }
        )
        horizontal = sorted(
            {
                q["points_px"][0][1]
                for q in layers
                if q.get("type") == "line"
                and len(q.get("points_px", [])) == 2
                and q["points_px"][0][1] == q["points_px"][1][1]
            }
        )
        if len(vertical) >= 3 and len(horizontal) == len(vertical):
            dx, dy = np.diff(vertical), np.diff(horizontal)
            if not np.allclose(dx, dx[0]) or not np.allclose(dy, dx[0]):
                raise ValueError("Rendered grid lines are not a uniform square lattice")
            grid = {
                "rows": len(horizontal) - 1,
                "cols": len(vertical) - 1,
                "cell_size_px": float(dx[0]),
                "origin_x_px": vertical[0],
                "origin_y_px": horizontal[0],
            }
    if not s or grid is None:
        return None
    rows, cols = int(grid["rows"]), int(grid["cols"])
    origin = (float(grid.get("origin_x_px", 0)), float(grid.get("origin_y_px", 0)))
    size = float(grid["cell_size_px"])
    row_col = prefix in {"G-45", "G-47", "O-39"}

    def rc(v):
        return tuple(int(x) for x in (v if row_col else v[::-1]))

    walls = set()
    if "state" in grid:
        state = np.asarray(grid["state"])
        walls = set(zip(*np.where(state == 0)))
    required, ordered, items = [], False, []
    if prefix == "G-13":
        start, goal = rc(s["start_position"]), rc(s["end_position"])
        required = [rc(q["position"]) for q in sorted(s["number_positions"], key=lambda q: q["number"])]
        ordered = True
    elif prefix in {"G-15", "G-16"}:
        start, goal = rc(s["start"]), rc(s["end"])
        walls = {rc(v) for v in s.get("obstacles", [])}
        if prefix == "G-16":
            required = [rc(v["position"]) for v in s["blocks"]]
            ordered = True
    elif prefix == "G-18":
        start, goal = rc(s["marks_and_agent"]["start_cell"]), rc(s["marks_and_agent"]["goal_cell"])
        walls = {rc(v) for v in s["grid"].get("obstacles", [])}
    elif prefix == "G-45":
        index = s["target_pair_index"]
        if s["keys"][index]["color"] != s["doors"][index]["color"]:
            raise ValueError("Target key and door colors contradict one another")
        start, goal = rc(s["agent_start_position"]), rc(s["doors"][index]["pos"])
        required = [rc(s["keys"][index]["pos"])]
        items = [dict(s["keys"][index], cell=required[0])]
        ordered = True
    elif prefix == "G-47":
        start, goal = rc(s["agent_start_position"]), rc(s["door"]["pos"])
        required = [rc(v["pos"]) for v in s["keys"]]
        items = [dict(v, cell=rc(v["pos"])) for v in s["keys"]]
    else:
        start, goal = rc(p["start"]), rc(p["end"])
    if prefix == "O-39":
        color = next(v["color"] for v in p["objects"] if v["symbol"] == "path")
        radius = size * 0.12
    else:
        circles = [q for q in layers if q.get("type") == "filled_shape" and q.get("kind") == "circle"]
        if len(circles) != 1:
            raise ValueError("Reference must define exactly one circular grid agent")
        color, radius = circles[0]["fill_rgb"], float(circles[0]["radius_px"])
    # Geometry comes from the render declaration; labels from task semantics.
    canvas = (metadata.get("generic_declarative_render") or {}).get("canvas", {})
    sx, sy = image.shape[1] / canvas.get("width", image.shape[1]), image.shape[0] / canvas.get("height", image.shape[0])
    if not np.isclose(sx, sy):
        raise ValueError("Grid geometry requires a uniform reference scale")
    spec = Spec(
        rows,
        cols,
        (origin[0] * sx, origin[1] * sy),
        size * sx,
        start,
        goal,
        walls,
        required,
        ordered,
        np.asarray(color[::-1], float),
        radius * sx,
        items,
    )
    for cell in [start, goal, *required]:
        if not (0 <= cell[0] < rows and 0 <= cell[1] < cols) or cell in walls:
            raise ValueError("Symbolic reference endpoint/waypoint is outside traversable grid")
    return spec


def _shortest(spec: Spec) -> int:
    """BFS in (cell, progress/mask) accepts every optimal waypoint ordering."""

    def advance(cell, progress):
        if spec.ordered:
            if progress < len(spec.required) and cell == spec.required[progress]:
                return progress + 1
            return progress
        for i, target in enumerate(spec.required):
            if cell == target:
                progress |= 1 << i
        return progress

    complete = len(spec.required) if spec.ordered else (1 << len(spec.required)) - 1
    progress = advance(spec.start, 0)
    q = deque([(spec.start, progress, 0)])
    seen = {(spec.start, progress)}
    while q:
        cell, progress, distance = q.popleft()
        if cell == spec.goal and progress == complete:
            return distance
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < spec.rows and 0 <= nxt[1] < spec.cols) or nxt in spec.walls:
                continue
            if spec.ordered and nxt in spec.required[progress + 1 :]:
                continue
            if nxt == spec.goal and advance(nxt, progress) != complete:
                continue
            state = (nxt, advance(nxt, progress))
            if state not in seen:
                seen.add(state)
                q.append((*state, distance + 1))
    raise ValueError("Reference task has no legal route through required targets")


def _color_mask(image: np.ndarray, color: np.ndarray) -> np.ndarray:
    return (np.max(np.abs(image.astype(np.float32) - color), axis=2) < 65).astype(np.uint8)


def _segment_cells(spec: Spec, start: tuple[float, float], end: tuple[float, float]) -> list[Cell] | None:
    """Read the complete straight corridor segment between two observations.

    Within-cell raster jitter is allowed. Endpoints in different rows AND
    columns do not establish an orthogonal turn: no shortest-path completion
    or hidden corner is invented. A same-row/column segment stays inside that
    cell strip, so its ordered cells include every crossed grid edge.
    """
    a, b = spec.cell(start), spec.cell(end)
    if a == b:
        return [a]
    if a[0] == b[0]:
        direction = 1 if b[1] > a[1] else -1
        return [(a[0], column) for column in range(a[1], b[1] + direction, direction)]
    if a[1] == b[1]:
        direction = 1 if b[0] > a[0] else -1
        return [(row, a[1]) for row in range(a[0], b[0] + direction, direction)]
    return None


def _agent(image: np.ndarray, spec: Spec) -> tuple[list[tuple[float, float]], np.ndarray]:
    mask = _color_mask(image, spec.color)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    expected = np.pi * spec.radius**2
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= max(6, 0.10 * expected)]
    out = []
    for i in candidates:
        area = stats[i, cv2.CC_STAT_AREA]
        if 0.45 * expected <= area <= 1.8 * expected:
            out.append(tuple(centers[i]))
        else:
            # A malformed agent is an explicit extra/invalid observation.
            out.append((float("nan"), float("nan")))
    return out, mask


def _scene_preserved(
    frames: list[np.ndarray],
    first: np.ndarray,
    spec: Spec,
    positions: list[tuple[float, float] | None],
    final: np.ndarray,
    *,
    unobserved_source: bool = False,
) -> tuple[bool, list[int]]:
    h, w = first.shape[:2]
    scale = min(1.0, 320 / max(h, w))
    size = (round(w * scale), round(h * scale))
    empty = first.copy()
    vacated = np.zeros(first.shape[:2], np.uint8)
    cv2.circle(vacated, tuple(round(v) for v in spec.center(spec.start)), round(spec.radius * 1.35), 255, -1)
    unknown = np.zeros(first.shape[:2], np.uint8)
    if unobserved_source:
        # A value completely covered by the conditioning agent was never
        # published to the model. The final reference reveals a private
        # label, so it cannot define preservation inside that opaque footprint.
        # Visible digits and grid lines outside it remain fixed observations.
        cv2.circle(unknown, tuple(round(v) for v in spec.center(spec.start)), round(spec.radius), 1, -1)
        unknown *= _color_mask(first, spec.color)
    keys = []
    for item in spec.key_items:
        region = np.zeros(first.shape[:2], np.uint8)
        cv2.circle(region, tuple(round(v) for v in spec.center(item["cell"])), round(spec.cell_size * 0.45), 255, -1)
        color = np.asarray(item["color"][::-1], float)
        count = np.count_nonzero((_color_mask(first, color) > 0) & (region > 0))
        keys.append((region, color, count))
        vacated |= region
    # The final reference supplies the exposed terrain. Where its label was
    # hidden in the public input, unknown explicitly excludes that private ink.
    empty[vacated > 0] = final[vacated > 0]
    base = cv2.resize(empty, size, interpolation=cv2.INTER_AREA)
    background_reference = _background_reference(base)
    smooth_base = cv2.GaussianBlur(base, (3, 3), 0).astype(np.int16)
    vacated_small = cv2.resize(vacated, size, interpolation=cv2.INTER_NEAREST) > 0
    unknown_small = cv2.resize(unknown, size, interpolation=cv2.INTER_NEAREST) > 0
    fixed_ink = np.max(base, axis=2) < 128
    fixed_ink_distance = cv2.distanceTransform((~fixed_ink).astype(np.uint8), cv2.DIST_L2, 3)
    bad = []
    for i, (frame, point) in enumerate(zip(frames, positions)):
        allowed = unknown.copy()
        if point is not None:
            cv2.circle(allowed, tuple(round(v) for v in point), round(spec.radius * 1.35), 255, -1)
        for region, color, initial_count in keys:
            visible = np.count_nonzero((_color_mask(frame, color) > 0) & (region > 0))
            if visible > max(8, initial_count * 0.20):
                allowed |= region
        allowed = cv2.resize(allowed, size, interpolation=cv2.INTER_NEAREST) > 0
        actual = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        wrong = (np.max(np.abs(actual.astype(np.int16) - base.astype(np.int16)), axis=2) > 60) & ~allowed
        local_failure = False
        for y in range(0, size[1], 20):
            for x in range(0, size[0], 20):
                valid = ~allowed[y : y + 20, x : x + 20]
                area = np.count_nonzero(valid)
                if area >= 100 and np.count_nonzero(wrong[y : y + 20, x : x + 20]) / area > 0.20:
                    local_failure = True
        clean, _ = _foreign_background_component(actual, background_reference, allowed)
        if unobserved_source:
            # Numeric labels are necessary input, even when a changed glyph
            # affects too few pixels to trip a whole-tile preservation ratio.
            # Match their visible ink at codec-scale spatial tolerance.
            actual_ink = np.max(actual, axis=2) < 128
            actual_distance = cv2.distanceTransform((~actual_ink).astype(np.uint8), cv2.DIST_L2, 3)
            changed_ink = (
                (fixed_ink & (actual_distance > 1.5)) | (actual_ink & (fixed_ink_distance > 1.5))
            ) & ~allowed
            _, _, ink_stats, _ = cv2.connectedComponentsWithStats(changed_ink.astype(np.uint8), 8)
            if len(ink_stats) > 1 and np.max(ink_stats[1:, cv2.CC_STAT_AREA]) >= 6:
                clean = False
            for row in range(spec.rows):
                for col in range(spec.cols):
                    cx, cy = spec.center((row, col))
                    half = spec.cell_size * scale * 0.35
                    roi = (
                        slice(max(0, round(cy * scale - half)), round(cy * scale + half)),
                        slice(max(0, round(cx * scale - half)), round(cx * scale + half)),
                    )
                    known_ink = fixed_ink[roi] & ~allowed[roi]
                    if np.count_nonzero(changed_ink[roi]) > max(6, np.count_nonzero(known_ink) * 0.15):
                        clean = False
        # The now-visible cost/start label is part of the reconstructed scene.
        # Its edge halo must not become a permanent hole at the vacated actor.
        current_near = cv2.dilate(allowed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        source_change = (
            (np.max(np.abs(cv2.GaussianBlur(actual, (3, 3), 0).astype(np.int16) - smooth_base), axis=2) > 55)
            & vacated_small
            & ~unknown_small
            & ~current_near
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(source_change.astype(np.uint8), 8)
        source_extra = int(np.max(stats[1:count, cv2.CC_STAT_AREA])) if count > 1 else 0
        if (
            source_extra >= 6
            or not clean
            or local_failure
            or np.count_nonzero(wrong) / max(np.count_nonzero(~allowed), 1) > 0.008
        ):
            bad.append(i)
    return not bad, bad


def _drawn_maze(frames: list[np.ndarray], first: np.ndarray, spec: Spec, shortest: int) -> dict:
    scale = min(1.0, 320 / max(first.shape[:2]))
    size = (round(first.shape[1] * scale), round(first.shape[0] * scale))
    base = cv2.resize(first, size, interpolation=cv2.INTER_AREA)
    background = _background_reference(base)
    icon_pixels = (np.ptp(base, axis=2) > 35).astype(np.uint8)
    icon_edges = cv2.dilate(icon_pixels, np.ones((3, 3), np.uint8)) > 0
    walkable = np.zeros(base.shape[:2], np.uint8)
    patches = {}
    for row in range(spec.rows):
        for col in range(spec.cols):
            cell = row, col
            x, y = spec.center(cell)
            half = 0.35 * spec.cell_size
            patches[cell] = (
                slice(max(0, round((y - half) * scale)), round((y + half) * scale)),
                slice(max(0, round((x - half) * scale)), round((x + half) * scale)),
            )
            if cell not in spec.walls:
                x1, y1 = (round(v * scale) for v in (x - spec.cell_size / 2, y - spec.cell_size / 2))
                x2, y2 = (round(v * scale) for v in (x + spec.cell_size / 2, y + spec.cell_size / 2))
                walkable[max(0, y1) : y2, max(0, x1) : x2] = 1
    drawn, bad, foreign_frames = [], [], []
    largest_foreign = 0
    minimum = max(3, spec.cell_size**2 * scale**2 * 0.008)
    for index, frame in enumerate(frames):
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        delta = np.max(cv2.absdiff(small, base), axis=2)
        # The prompt specifies a journey through the white corridors. New
        # coloured ink is observable against the neutral maze in any hue.
        mark = (delta > 30) & (np.ptp(small, axis=2) > 40)
        new_ink = mark & ~icon_edges
        cells = {cell for cell, patch in patches.items() if np.count_nonzero(new_ink[patch]) >= minimum}
        drawn.append(cells)
        legal_ink = mark & (walkable > 0)
        support = np.zeros_like(walkable)
        for cell in cells | {spec.start}:
            support[patches[cell]] = 1
        # Associate the anti-aliased trail and its moving tip with an observed
        # cell. The close bridges only a raster gap, not a missing grid step.
        joined = cv2.morphologyEx(legal_ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        _, labels = cv2.connectedComponents(joined, connectivity=8)
        observed = np.unique(labels[legal_ink & (support > 0)])
        allowed_ink = (np.isin(labels, observed[observed > 0]) & legal_ink).astype(np.uint8)
        allowed = cv2.dilate(allowed_ink, np.ones((3, 3), np.uint8)) > 0
        if np.mean((delta > 60) & ~allowed) > 0.008:
            bad.append(index)
        clean, area = _foreign_background_component(small, background, allowed_ink)
        largest_foreign = max(largest_foreign, area)
        if not clean:
            foreign_frames.append(index)
    final = drawn[-1]
    # The prompt requires a complete legal journey, not a shortest route. Read
    # its drawing order: a legal longer route may touch an earlier cell and
    # create chords in the final cell-adjacency graph without branching in time.
    route, seen, invalid_order = [spec.start], {spec.start}, []
    for index, cells in enumerate(drawn):
        pending = cells - seen
        while pending:
            r, c = route[-1]
            following = pending & {(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)}
            if len(following) != 1:
                invalid_order.append(index)
                break
            point = following.pop()
            route.append(point)
            pending.remove(point)
        seen |= cells
    # The fixed end icon can occlude its path dot. Only bridge that one final
    # adjacent step, never an absent section of the journey.
    if route[-1] != spec.goal and spec.goal not in route:
        if sum(abs(a - b) for a, b in zip(route[-1], spec.goal)) == 1:
            route.append(spec.goal)
    visible = final | {spec.start, spec.goal}
    accounted = not invalid_order and set(route) == visible
    wall_cells = set().union(*drawn) & spec.walls
    monotonic = all(a <= b for a, b in zip(drawn, drawn[1:]))
    checks = {
        "path_present": bool(final),
        "zero_wall_crossings": not bool(wall_cells),
        "one_connected_path": accounted,
        "adjacent_drawing_steps": not invalid_order,
        "all_path_cells_accounted_for": accounted,
        "goal_reached": route[-1] == spec.goal,
        "path_grows_without_erasure": monotonic,
        "progressive_path_evidence": len({frozenset(v) for v in drawn}) >= min(4, len(route)),
    }
    checks["maze_preserved_throughout"] = not bad
    checks["no_new_background_objects"] = not foreign_frames
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "family": "drawn_maze_route",
            "shortest_edges": shortest,
            "observed_edges": len(route) - 1,
            "route": [list(cell) for cell in route],
            "invalid_drawing_order_frames": invalid_order,
            "drawn_cells": [list(v) for v in sorted(final)],
            "wall_cells_drawn": [list(v) for v in sorted(wall_cells)],
            "corrupted_frame_indices": bad,
            "foreign_background_frame_indices": foreign_frames,
            "largest_foreign_background_component": largest_foreign,
        },
    }


def verify(
    task: str,
    frames: list[np.ndarray],
    first: np.ndarray,
    final: np.ndarray,
    metadata: dict,
    reference: list[np.ndarray],
) -> dict | None:
    prefix = task.split("_", 1)[0]
    if prefix not in PREFIXES:
        return None
    spec = _spec(prefix, metadata, first)
    if spec is None:
        return None
    shortest = _shortest(spec)
    if not frames:
        return {"passed": False, "checks": {"frames_present": False}, "evidence": {}}
    if prefix == "O-39":
        return _drawn_maze(frames, first, spec, shortest)
    positions = []
    invalid = []
    for i, frame in enumerate(frames):
        agents, _ = _agent(frame, spec)
        if len(agents) != 1 or not np.all(np.isfinite(agents[0])):
            invalid.append(i)
            positions.append(None)
            continue
        point = agents[0]
        positions.append(point)
    route, segments, unsupported_segments = [], [], []
    for i, point in enumerate(positions):
        if point is None:
            segments.append(None)
            continue
        segment = (
            _segment_cells(spec, positions[i - 1], point) if i and positions[i - 1] is not None else [spec.cell(point)]
        )
        segments.append(segment)
        if segment is None:
            unsupported_segments.append(i)
        for cell in segment if segment is not None else [spec.cell(point)]:
            if not route or route[-1] != cell:
                route.append(cell)
    steps = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(route, route[1:])]
    visited = set(route)
    waypoint_progress = 0
    ordered_visits_ok = True
    for cell in route:
        if waypoint_progress < len(spec.required) and cell == spec.required[waypoint_progress]:
            waypoint_progress += 1
        elif spec.ordered and cell in spec.required[waypoint_progress:]:
            ordered_visits_ok = False
    first_goal = route.index(spec.goal) if spec.goal in route else len(route)
    keys_before_goal = all(cell in route[:first_goal] for cell in spec.required)
    scene_ok, bad_scene = _scene_preserved(frames, first, spec, positions, final)
    keys_ok = True
    key_evidence = []
    for item in spec.key_items:
        color = np.asarray(item["color"][::-1], float)
        x, y = spec.center(item["cell"])
        r = spec.cell_size * 0.43
        roi = (slice(max(0, int(y - r)), int(y + r)), slice(max(0, int(x - r)), int(x + r)))
        old = np.count_nonzero(_color_mask(first[roi], color))
        new = np.count_nonzero(_color_mask(frames[-1][roi], color))
        gone = new <= max(8, old * 0.12)
        contacted = False
        contact_frame = None
        early_missing = []
        retained = []
        for i, (frame, point) in enumerate(zip(frames, positions)):
            distance = float("inf") if point is None else np.linalg.norm(np.subtract(point, (x, y))) / spec.cell_size
            crossed = segments[i] is not None and item["cell"] in segments[i]
            if not contacted and (distance < 0.75 or crossed):
                contacted = True
                contact_frame = i
            count = np.count_nonzero(_color_mask(frame[roi], color))
            if not contacted and count < old * 0.55:
                early_missing.append(i)
            if contacted and distance > 1 and count > max(8, old * 0.20):
                retained.append(i)
        keys_ok = keys_ok and gone and not early_missing and not retained
        key_evidence.append(
            {
                "cell": list(item["cell"]),
                "reference_pixels": old,
                "final_pixels": new,
                "collected_key_absent": bool(gone),
                "contact_frame": contact_frame,
                "disappeared_before_contact": early_missing,
                "retained_after_collection": retained,
            }
        )
    checks = {
        "one_readable_agent_every_frame": not invalid,
        "starts_at_source": bool(route) and route[0] == spec.start,
        "ends_at_goal": bool(route) and route[-1] == spec.goal,
        "all_cells_in_grid": all(0 <= r < spec.rows and 0 <= c < spec.cols for r, c in route),
        "zero_wall_crossings": not bool(visited & spec.walls),
        "observed_segments_have_supported_geometry": not unsupported_segments,
        "adjacent_orthogonal_steps": not unsupported_segments and all(v == 1 for v in steps),
        "all_required_cells_visited": all(c in visited for c in spec.required),
        "required_visit_order": not spec.ordered or (waypoint_progress == len(spec.required) and ordered_visits_ok),
        "required_targets_before_goal": keys_before_goal,
        "collected_keys_absent": keys_ok,
        "scene_preserved_throughout": scene_ok,
    }
    if prefix != "G-45":
        checks["shortest_legal_route"] = len(route) - 1 == shortest
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evidence": {
            "family": "symbolic_grid_route",
            "route": [list(c) for c in route],
            "shortest_edges": shortest,
            "shortest_route_required": prefix != "G-45",
            "observed_edges": max(0, len(route) - 1),
            "unsupported_observed_segments": unsupported_segments,
            "missing_or_multiple_agent_frames": invalid,
            "corrupted_scene_frames": bad_scene,
            "keys": key_evidence,
            "grid": {"rows": spec.rows, "cols": spec.cols, "origin": list(spec.origin), "cell_size": spec.cell_size},
            "start": list(spec.start),
            "goal": list(spec.goal),
            "required": [list(c) for c in spec.required],
        },
    }
