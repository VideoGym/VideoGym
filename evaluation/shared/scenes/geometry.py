"""Composed metric goals using native point-target drawing and marker methods."""

import math, random, tempfile
import json
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image
from data.geometry.core.point_target_base import (
    Point,
    PointCandidate,
)
from data.geometry.point_rotation.generator import PointRotationGenerator
from data.common.verifiers.transitions import uniform_fade_metrics
from evaluation.shared.scenes.catalog import TASKS


def point(v):
    return Point(float(v[0]), float(v[1]))


def rotate(p, o, angle=90):
    a = math.radians(angle)
    x, y = (p[0] - o[0], o[1] - p[1])
    return [
        o[0] + math.cos(a) * x - math.sin(a) * y,
        o[1] - math.sin(a) * x - math.cos(a) * y,
    ]


def midpoint(a, b):
    return [(a[i] + b[i]) / 2 for i in (0, 1)]


def unit(v):
    d = math.hypot(*v)
    return [v[0] / d, v[1] / d]


def bisector(a, v, b, length=105):
    u = unit([a[i] - v[i] for i in (0, 1)])
    w = unit([b[i] - v[i] for i in (0, 1)])
    d = unit([u[i] + w[i] for i in (0, 1)])
    return [v[i] + length * d[i] for i in (0, 1)]


def target(s, primitive=False):
    k = s["key"]
    a = s["anchors"]
    turn = s.get("turn", 90)
    if k == "rotate_midpoint":
        r = rotate(a["S"], a["O"], turn)
        return r if primitive else midpoint(r, a["A"])
    if k == "midpoint_rotate":
        m = midpoint(a["S"], a["A"])
        return m if primitive else rotate(m, a["O"], turn)
    raise ValueError(k)


def solve(s):
    pts = s["candidates"]
    key = s["key"]
    if s["variant"] == "second_goal":
        target_point = second_target(s)
        choices = [
            c["label"] for c in pts if math.dist(c["point"], target_point) < 1e-05
        ]
        if len(choices) != 1:
            raise ValueError("second rule removal has no unique visible candidate")
        return choices[0]
    primitive = s["variant"] == "source_goal"
    if key in ("midpoint_bisector", "midpoint_parallel", "rotate_parallel"):
        a = s["anchors"]
        if key == "midpoint_bisector":
            origin = a["B"] if primitive else midpoint(a["A"], a["B"])
            end = bisector(a["C"], origin, a["A"])
            direction = unit([end[i] - origin[i] for i in (0, 1)])
            ray = True
        elif key == "midpoint_parallel":
            origin = a["A"] if primitive else midpoint(a["A"], a["B"])
            direction = unit([a["E"][i] - a["D"][i] for i in (0, 1)])
            ray = False
        else:
            origin = a["A"]
            end = a["S"] if primitive else rotate(a["S"], a["O"], s.get("turn", 90))
            direction = unit([end[i] - a["O"][i] for i in (0, 1)])
            ray = False
        candidates = []
        for c in pts:
            delta = [c["point"][i] - origin[i] for i in (0, 1)]
            if math.hypot(*delta) < 20:
                continue
            along = sum((x * y for x, y in zip(delta, direction)))
            offset = abs(delta[0] * direction[1] - delta[1] * direction[0])
            if (
                offset <= 1e-05
                and (not ray or along > 20)
                and (math.hypot(*delta) > 20)
            ):
                candidates.append(c["label"])
        if len(candidates) != 1:
            raise ValueError("geometric predicate is not uniquely satisfied")
        return candidates[0]
    t = target(s, primitive)
    nearest = sorted(((math.dist(t, c["point"]), c["label"]) for c in pts))
    if nearest[0][0] > 1e-05 or nearest[1][0] < 24:
        raise ValueError("candidate does not uniquely encode composed target")
    return nearest[0][1]


def _native_scene(key, rng):
    from data.geometry.midpoint.generator import MidpointGenerator
    from data.geometry.angle_bisector.generator import AngleBisectorGenerator
    from data.geometry.parallel.generator import ParallelGenerator

    classes = {
        "rotate_midpoint": PointRotationGenerator,
        "midpoint_rotate": MidpointGenerator,
        "midpoint_bisector": AngleBisectorGenerator,
        "midpoint_parallel": ParallelGenerator,
        "rotate_parallel": ParallelGenerator,
    }
    with tempfile.TemporaryDirectory(prefix="videogym-native-geometry-") as tmp:
        g = classes[key](tmp, seed=rng.randrange(2**31))

        def finish_scene():
            if key == "rotate_midpoint" and (
                g.rotation_deg != 90 or g.rotation_sign != 1
            ):
                return
            g._compose_off_centre()

        g.save_puzzle = finish_scene
        g.create_puzzle()
        candidates = [
            {"label": c.label, "point": [float(c.x), float(c.y)]} for c in g.candidates
        ]
        if key == "rotate_midpoint":
            if g.rotation_deg != 90 or g.rotation_sign != 1:
                return None
            o, s = ([g.pivot.x, g.pivot.y], [g.source_point.x, g.source_point.y])
            rotated = rotate(s, o)
            choices = [c for c in candidates if math.dist(c["point"], rotated) > 35]
            rng.shuffle(choices)
            for c in choices:
                anchor = [2 * c["point"][i] - rotated[i] for i in (0, 1)]
                if (
                    all((45 < x < 467 for x in anchor))
                    and all((math.dist(anchor, c["point"]) > 35 for c in candidates))
                    and (min(math.dist(anchor, o), math.dist(anchor, s)) > 45)
                ):
                    return ({"O": o, "S": s, "A": anchor}, candidates)
            return None
        if key == "midpoint_rotate":
            left, right = g._ordered_endpoints(g._segment)
            m = midpoint(left, right)
            choices = [c for c in candidates if math.dist(c["point"], m) > 55]
            rng.shuffle(choices)
            for c in choices:
                t = c["point"]
                o = [(t[0] + t[1] + m[0] - m[1]) / 2, (t[1] - t[0] + m[0] + m[1]) / 2]
                if (
                    all((45 < x < 467 for x in o))
                    and all((math.dist(o, c["point"]) > 28 for c in candidates))
                    and (min(math.dist(o, left), math.dist(o, right)) > 35)
                ):
                    return ({"S": list(left), "A": list(right), "O": o}, candidates)
            return None
        if key == "midpoint_bisector":
            a, b, c = ([p.x, p.y] for p in g.points)
            if not (a[0] < b[0] - 50 and c[1] < b[1] - 50 and (a[1] > c[1] + 50)):
                return None
            m = midpoint(a, b)
            u = [g.target_point.x, g.target_point.y]
            direction = unit(
                [candidates[1]["point"][i] - candidates[0]["point"][i] for i in (0, 1)]
            )
            v = unit([bisector(c, m, a)[i] - m[i] for i in (0, 1)])
            matrix = np.array([[v[0], -direction[0]], [v[1], -direction[1]]])
            rhs = np.subtract(u, m)
            if abs(np.linalg.det(matrix)) < 0.05:
                return None
            along, _ = np.linalg.solve(matrix, rhs)
            t = [m[i] + float(along) * v[i] for i in (0, 1)]
            if along < 40 or not all((65 < x < 447 for x in t)) or math.dist(t, u) < 40:
                return None
            replace = next(
                (
                    i
                    for i, p in enumerate(candidates)
                    if p["label"] != g.correct_label
                    and all(
                        (
                            math.dist(t, q["point"]) > 40
                            for j, q in enumerate(candidates)
                            if i != j
                        )
                    )
                ),
                None,
            )
            if replace is None:
                return None
            candidates[replace]["point"] = t
            return ({"A": a, "B": b, "C": c}, candidates)
        if key == "midpoint_parallel":
            a = [g.points[0].x, g.points[0].y]
            d, e = ([p.x, p.y] for p in g._ordered_reference_endpoints())
            direction = unit([e[i] - d[i] for i in (0, 1)])
            choices = [c for c in candidates if c["label"] != g.correct_label]
            rng.shuffle(choices)
            for c in choices:
                for extent in (60, 90, 120, 150, -60, -90):
                    m = [c["point"][i] - extent * direction[i] for i in (0, 1)]
                    b = [2 * m[i] - a[i] for i in (0, 1)]
                    if b[0] < a[0] + 60 or not all((40 < x < 472 for x in (*a, *b))):
                        continue
                    if max(a[1], b[1]) + 40 > min(d[1], e[1]):
                        continue
                    if any(
                        (
                            min(math.dist(p["point"], a), math.dist(p["point"], b)) < 35
                            for p in candidates
                        )
                    ):
                        continue
                    return ({"A": a, "B": b, "D": d, "E": e}, candidates)
            return None
        if key == "rotate_parallel":
            a = [g.points[0].x, g.points[0].y]
            d, e = ([p.x, p.y] for p in g._ordered_reference_endpoints())
            direction = unit([e[i] - d[i] for i in (0, 1)])
            o = [e[i] - 100 * direction[i] for i in (0, 1)]
            along = sum(((o[i] - a[i]) * direction[i] for i in (0, 1)))
            if abs(along) < 18:
                return None
            normal = [direction[1], -direction[0]]
            choices = [
                i for i, c in enumerate(candidates) if c["label"] != g.correct_label
            ]
            rng.shuffle(choices)
            for index in choices:
                for extent in (80, 110, 140, 170, 200, -80, -110, -140, -170):
                    t = [a[i] + extent * normal[i] for i in (0, 1)]
                    if not all((50 < x < 462 for x in t)):
                        continue
                    if (
                        abs((t[0] - d[0]) * direction[1] - (t[1] - d[1]) * direction[0])
                        < 25
                    ):
                        continue
                    if any(
                        (
                            math.dist(t, p["point"]) < 40
                            for j, p in enumerate(candidates)
                            if j != index
                        )
                    ):
                        continue
                    cloud = [
                        dict(p, point=t) if j == index else p
                        for j, p in enumerate(candidates)
                    ]
                    test = {
                        "key": key,
                        "anchors": {"A": a, "D": d, "E": e, "O": o, "S": e},
                        "candidates": cloud,
                        "turn": 90,
                    }
                    try:
                        if not clear_candidates(test):
                            continue
                    except ValueError:
                        continue
                    return (test["anchors"], cloud)
            return None
        raise ValueError(key)


def clear_candidates(s):
    if s["key"] not in ("midpoint_bisector", "midpoint_parallel", "rotate_parallel"):
        return True
    for primitive in (False, True):
        test = dict(s, variant="source_goal" if primitive else "main")
        chosen = solve(test)
        a = s["anchors"]
        origin = (
            (a["B"] if primitive else midpoint(a["A"], a["B"]))
            if s["key"] == "midpoint_bisector"
            else a["A"]
            if primitive or s["key"] == "rotate_parallel"
            else midpoint(a["A"], a["B"])
        )
        p = next((c["point"] for c in s["candidates"] if c["label"] == chosen))
        direction = unit([p[i] - origin[i] for i in (0, 1)])
        for c in s["candidates"]:
            if c["label"] == chosen:
                continue
            delta = [c["point"][i] - origin[i] for i in (0, 1)]
            if math.hypot(*delta) < 20:
                continue
            if (
                s["key"] == "midpoint_bisector"
                and sum((x * y for x, y in zip(delta, direction))) <= 0
            ):
                continue
            if abs(delta[0] * direction[1] - delta[1] * direction[0]) < 26:
                return False
    return True


def sample(key, seed, variant="main"):
    rng = random.Random(seed)
    for attempt in range(1000):
        s = dict(key=key, host=TASKS[key].host, variant=variant, turn=90)
        if key in (
            "rotate_midpoint",
            "midpoint_rotate",
            "midpoint_bisector",
            "midpoint_parallel",
            "rotate_parallel",
        ):
            native = _native_scene(key, rng)
            if native is None:
                continue
            s["anchors"], s["candidates"] = native
            if not admit_second_candidate(s):
                continue
            try:
                main = solve(dict(s, variant="main"))
                control = solve(dict(s, variant="source_goal"))
                if main == control or not clear_candidates(s):
                    continue
            except ValueError:
                continue
            from evaluation.shared.scenes.ablation import evidence

            if not evidence(dict(s, variant="main"))["passed"]:
                continue
            return s
    raise RuntimeError(key + ": native geometry sampling exhausted")


def second_target(s):
    """The other rule removal, in the same candidate-selection answer space."""
    a, key = (s["anchors"], s["key"])
    if key == "rotate_midpoint":
        return midpoint(a["S"], a["A"])
    if key == "midpoint_rotate":
        return rotate(a["S"], a["O"], s["turn"])
    if key in ("midpoint_bisector", "midpoint_parallel"):
        return midpoint(a["A"], a["B"])
    if key == "rotate_parallel":
        return rotate(a["S"], a["O"], s["turn"])
    raise ValueError(key)


def admit_second_candidate(s):
    """Keep five native markers and make BOTH counterfactual questions soluble.

    A point outside the selectable set is not a legitimate single-rule answer.
    Replace one distractor, retaining the host answer and joint answer, then
    recheck geometry/readability. No marker style, count or action is added.
    """
    try:
        main = solve(dict(s, variant="main"))
        source = solve(dict(s, variant="source_goal"))
        t = second_target(s)
        protected = [c for c in s["candidates"] if c["label"] in (main, source)]
        if not all((45 < x < 467 for x in t)):
            return False
        if any((math.dist(t, c["point"]) < 30 for c in protected)):
            return False
        if any((math.dist(t, p) < 26 for p in s["anchors"].values())):
            return False
        old = [dict(c) for c in s["candidates"]]
        for index, c in enumerate(old):
            if c["label"] in (main, source):
                continue
            if any(
                (
                    math.dist(t, other["point"]) < 30
                    for j, other in enumerate(old)
                    if j != index
                )
            ):
                continue
            s["candidates"] = [
                dict(other, point=t) if j == index else other
                for j, other in enumerate(old)
            ]
            if (
                solve(dict(s, variant="main")) == main
                and solve(dict(s, variant="source_goal")) == source
                and clear_candidates(s)
            ):
                return True
        s["candidates"] = old
    except ValueError:
        return False
    return False


def renderer(s):
    g = object.__new__(PointRotationGenerator)
    g.canvas_dimensions = (512, 512)
    g._recording_active = False
    g._recorder = None
    g.point_radius = 11
    g.line_width = 5
    g._candidate_font = None
    g.candidates = [PointCandidate(*c["point"], c["label"]) for c in s["candidates"]]
    return g


def parallel_points(origin, selected):
    from data.geometry.parallel.generator import ParallelGenerator

    native = object.__new__(ParallelGenerator)
    native.points = [point(origin)]
    native.target_point = point(selected)
    native.point_radius = 11
    tail, _, end = native._construction_points()
    return [[tail.x, tail.y], [end.x, end.y]]


def constructions(s):
    k = s["key"]
    a = s["anchors"]
    primitive = s["variant"] == "source_goal"
    chosen = solve(s)
    positions = {c["label"]: c["point"] for c in s["candidates"]}
    if s["variant"] == "second_goal":
        if k in ("midpoint_rotate", "rotate_parallel"):
            return [("arc", [a["O"], a["S"], s["turn"]], "dark")]
        return []
    if primitive:
        if k == "rotate_midpoint":
            return [("arc", [a["O"], a["S"], s["turn"]], "dark")]
        if k == "midpoint_rotate":
            return []
        if k == "midpoint_bisector":
            return [("line", [a["B"], positions[chosen]], "cyan")]
        origin = a["A"]
        return [("line", parallel_points(origin, positions[chosen]), "cyan")]
    if k == "rotate_midpoint":
        return [("arc", [a["O"], a["S"], s["turn"]], "dark")]
    if k == "midpoint_rotate":
        m = midpoint(a["S"], a["A"])
        return (
            []
            if math.dist(m, a["O"]) < 1
            else [("arc", [a["O"], m, s["turn"]], "dark")]
        )
    if k == "midpoint_bisector":
        return [("line", [midpoint(a["A"], a["B"]), positions[chosen]], "cyan")]
    if k == "midpoint_parallel":
        return [
            (
                "line",
                parallel_points(midpoint(a["A"], a["B"]), positions[chosen]),
                "cyan",
            )
        ]
    return ([("arc", [a["O"], a["S"], s["turn"]], "dark")] if s["turn"] else []) + [
        ("line", parallel_points(a["A"], positions[chosen]), "cyan")
    ]


def render(s, progress=0.0, answer=None):
    g = renderer(s)
    draw, base = g.get_draw_base()
    k = s["key"]
    a = s["anchors"]
    if k == "midpoint_rotate":
        g.draw_line(draw, [point(a["S"]), point(a["A"])])
    elif k == "midpoint_bisector":
        g.draw_line(draw, [point(a["A"]), point(a["B"]), point(a["C"])])
    elif k == "midpoint_parallel":
        g.draw_line(draw, [point(a["A"]), point(a["B"])])
        g.draw_line(draw, [point(a["D"]), point(a["E"])])
    elif k == "rotate_parallel":
        g.draw_line(draw, [point(a["D"]), point(a["E"])])
    if "O" in a and math.dist(a["O"], a["S"]) > 1 and s["turn"]:
        g.pivot = point(a["O"])
        g.source_point = point(a["S"])
        g.radius = math.dist(a["O"], a["S"])
        g.source_angle = math.atan2(a["O"][1] - a["S"][1], a["S"][0] - a["O"][0])
        g.rotation_amount = math.radians(s["turn"])
        g.rotation_sign = 1
        g._draw_indicator_arc(draw, base)
    stages = constructions(s)
    amount = float(np.clip(progress, 0, 1)) * (len(stages) + 1)
    for index, (kind, data, color) in enumerate(stages):
        part = float(np.clip(amount - index, 0, 1))
        if part <= 0:
            continue
        if kind == "arc":
            o, initial, turn = data
            g.pivot = point(o)
            g.radius = math.dist(o, initial)
            g.source_angle = math.atan2(o[1] - initial[1], initial[0] - o[0])
            g.rotation_amount = math.radians(turn)
            g.rotation_sign = 1
            g._draw_arc_with_recording_direction(
                draw,
                base,
                radius=g.radius,
                width_factor=1.0,
                step_degrees=0.5,
                sweep=math.radians(turn) * part,
            )
        else:
            lengths = [math.dist(x, y) for x, y in zip(data, data[1:])]
            remaining = sum(lengths) * part
            out = [point(data[0])]
            for start, end, length in zip(data, data[1:], lengths):
                f = min(1, remaining / max(length, 1e-09))
                out.append(point([start[i] + f * (end[i] - start[i]) for i in (0, 1)]))
                remaining -= length
                if f < 1:
                    break
            g.draw_line(
                draw,
                out,
                width_factor=0.72
                if color == "cyan" and k in ("midpoint_parallel", "rotate_parallel")
                else 1.0,
                fill=g.CONSTRUCTION_COLOR
                if color == "cyan"
                else g.CANDIDATE_OUTLINE_COLOR,
            )
    for name, p in a.items():
        if name == "S" and k in ("rotate_midpoint", "rotate_parallel"):
            g.source_point = point(p)
            g._draw_source(draw)
        elif (
            k == "midpoint_rotate"
            and name in ("S", "A")
            or (k == "rotate_midpoint" and name == "A")
            or (k in ("midpoint_bisector", "midpoint_parallel") and name in ("A", "B"))
        ):
            from data.geometry.midpoint.generator import MidpointGenerator

            MidpointGenerator._draw_anchor(g, draw, tuple(p))
        elif name == "O" or (k == "rotate_parallel" and name == "A"):
            g.draw_anchor_marker(draw, point(p), 9 if name == "O" else 7)
    g.draw_candidates(draw)
    fraction = float(np.clip(amount - len(stages), 0, 1))
    before = g._downscale(base)
    if fraction > 0:
        chosen = solve(s) if answer is None else answer
        g.draw_candidates(draw, highlight_label=chosen)
        return Image.blend(before, g._downscale(base), fraction)
    return before


def prompt(s):
    k = s["key"]
    rule = TASKS[k].rule
    descriptions = {
        "rotate_midpoint": "A hollow pivot carries a small turn-indicator arc; a solid source and a separate hollow anchor are fixed. Five candidates carry letters A through E.",
        "midpoint_rotate": "A black segment has two hollow endpoints, and a separate hollow pivot carries the small turn-indicator arc. Five candidates carry letters A through E.",
        "midpoint_bisector": "Two black segments meet at a right-hand shared vertex. Their outer endpoints lie to its left and above it. Five candidates carry letters A through E.",
        "midpoint_parallel": "Two black segments lie in an upper and a lower row; their left and right endpoints are fixed. Five candidates carry letters A through E.",
        "rotate_parallel": "A black reference segment has a solid endpoint and a hollow pivot marked on the segment. A separate hollow through-point and five labeled candidates are fixed; the small arc specifies the turn.",
    }
    rules = {
        "rotate_midpoint": "Rotate the solid source counterclockwise by the indicated quarter-turn about the hollow pivot. Select the candidate halfway between that rotated point and the separate hollow anchor.",
        "midpoint_rotate": "Find the midpoint of the black segment, then rotate that midpoint counterclockwise by the indicated quarter-turn about the separate hollow pivot. Select the candidate at the rotated point.",
        "midpoint_bisector": "Find the midpoint of the black segment from the left outer endpoint to the shared right-hand vertex. At that midpoint, consider rays toward the upper outer endpoint and the left outer endpoint. Select the candidate on their internal, non-reflex angle-bisector ray.",
        "midpoint_parallel": "Find the midpoint of the upper black segment. Through that midpoint, draw the parallel to the lower reference segment and select the candidate on it.",
        "rotate_parallel": "Rotate the reference segment counterclockwise a quarter-turn about its hollow pivot. Through the separate hollow through-point, draw a parallel to that rotated reference and select the candidate on it.",
    }
    rule = rules.get(k, rule)
    if s["variant"] == "source_goal":
        rule = {
            "rotate_midpoint": "Select the candidate obtained by the indicated quarter-turn of the solid source about the hollow pivot. Ignore the extra hollow anchor.",
            "midpoint_rotate": "Select the candidate at the midpoint of the given black segment. Ignore the extra pivot and turn indicator.",
            "midpoint_bisector": "Select the candidate on the internal bisector of the original angle, at the shared right-hand vertex of the two black segments.",
            "midpoint_parallel": "Through the left endpoint of the upper segment, draw a parallel to the lower reference segment and select the candidate on it. The other endpoint of the upper segment is irrelevant.",
            "rotate_parallel": "Through the separate hollow through-point, draw a parallel to the unrotated reference segment and select its candidate. Ignore the turn indicator.",
        }[k]
    if s["variant"] == "second_goal":
        rule = {
            "rotate_midpoint": "Ignore the turn. Select the candidate halfway between the solid source and the separate hollow anchor.",
            "midpoint_rotate": "Ignore midpoint construction. Rotate the left endpoint of the black segment by the indicated quarter-turn about the separate pivot and select its candidate.",
            "midpoint_bisector": "Ignore the angle-bisector rule. Select the midpoint of the segment joining the left outer endpoint and the shared right-hand vertex.",
            "midpoint_parallel": "Ignore the parallel rule and lower segment. Select the midpoint of the upper black segment.",
            "rotate_parallel": "Ignore the parallel rule and separate through-point. Rotate the solid endpoint about the hollow pivot and select the candidate at its image.",
        }[k]
    action = "Draw the required dark rotation arc or segment, and any cyan bisector/parallel construction, in rule order; then smoothly fade the selected native candidate marker(s) to pale-red fill and dark-red outline."
    if k in ("midpoint_parallel", "rotate_parallel"):
        action += " Grow the cyan parallel with one advancing endpoint, starting at its short tail on the opposite side of the through-point and ending just before the selected marker, using the native thin-line convention."
    return (
        "Input Image: "
        + descriptions[k]
        + " All unlabeled hollow/solid markers are fixed input geometry; each printed candidate letter is unique.\n\nRule: "
        + rule
        + "\n\nRequired Output: "
        + action
        + " Begin unchanged and end in a stable solved hold. Keep all source geometry and candidate letters fixed; preserve every completed construction. No counter, extra symbol, cut, erasure, retracing or camera motion."
    )


def construction_paths(s, scene):
    """Complete source geometry, independent of animation cadence."""
    paths = []
    for kind, data, _ in scene.constructions(s):
        if kind == "line":
            paths.append(np.asarray(data, dtype=np.float32))
        else:
            centre, initial, turn = data
            angles = np.linspace(0, turn, max(25, int(abs(turn)) + 1))
            paths.append(np.asarray([rotate(initial, centre, angle) for angle in angles], dtype=np.float32))
    return paths


def _construction_masks(frames, initial):
    source = initial.astype(np.int16)
    masks = []
    for frame in frames:
        rgb = frame.astype(np.int16)
        changed = np.max(np.abs(rgb - source), axis=2) >= 24
        dark = np.max(rgb, axis=2) < 150
        cyan = (rgb[:, :, 0] < 110) & (rgb[:, :, 1] > 88) & (rgb[:, :, 2] > 112) & (rgb[:, :, 1] - rgb[:, :, 0] > 30)
        masks.append(changed & (dark | cyan))
    return masks


@lru_cache(8)
def _source_growth_masks(serialized):
    from evaluation.shared.scenes.catalog import module
    from evaluation.shared.scenes.media import frames

    source = json.loads(serialized)
    initial = np.asarray(module(source["key"]).render(source, 0))
    return _construction_masks(frames(source), initial)


def grade(s, path, *, scene=None):
    from evaluation.shared.task_io import read_video, matches
    from data.geometry.core.verifier import (
        _construction_presence, _verify_serial_coordinate_variant,
        _verify_cumulative_ink_order, _verify_source_static_preservation,
        _sweep, overlay_occluders,
    )
    from data.geometry.core.terminal_scene import _marker_gate, point_target_delivery_ink_state
    from data.geometry.circumcenter.evaluator import foreign_ink_clearance
    from data.geometry.core.phase_observation import phase_onset

    if scene is None:
        import sys
        scene = sys.modules[__name__]
    frames = read_video(path)
    answer = scene.solve(s)
    first = np.asarray(scene.render(s, 0))
    last = np.asarray(scene.render(s, 1))
    paths = construction_paths(s, scene)
    record = dict(point_radius=11, line_width=5, candidates=[
        dict(label=c["label"], x=c["point"][0], y=c["point"][1]) for c in s["candidates"]
    ])
    discs = overlay_occluders(record, paths)
    clearance = foreign_ink_clearance(5)
    marker, marker_ok = _marker_gate(frames[-1], first, record["candidates"], answer, point_radius=11)
    ink, ink_ok = point_target_delivery_ink_state(
        frames[-1], first, record, paths=paths, discs=discs,
        clearance=clearance, construction_declared=True,
    )
    if paths:
        terminal, terminal_ok = _construction_presence(
            frames[-1], first, record, task_name=s["key"], expected=answer,
            paths_override=paths,
        )
    else:
        terminal, terminal_ok = {}, True
    final = bool(marker_ok and ink_ok and terminal_ok is not False)
    # Observe native dark connectors/arcs and cyan bisectors, while the shared
    # stroke sampler excludes source ink, preceding strokes and marker discs.
    masks = _construction_masks(frames, first)
    n = len(paths) + 1
    before_commit = np.asarray(scene.render(s, (n - 1) / n))
    fade, fade_ok = uniform_fade_metrics(frames, before_commit, last)
    commit = phase_onset(fade.get("video_fade_progress", []), .08)
    if paths:
        serial, serial_ok = _verify_serial_coordinate_variant(
            frames, first, record, task_name=s["key"], expected=answer,
            construction_start=0, construction_end=commit + 1, paths=paths,
            construction_masks=masks,
            source_growth_masks=_source_growth_masks(json.dumps(s, sort_keys=True)),
        )
    else:
        serial, serial_ok = {}, True
    # Corridors come from the problem's complete geometry, not from a sparse
    # rendering or from whatever content the candidate leaves in its last frame.
    dynamic = np.zeros(first.shape[:2], dtype=np.uint8)
    for curve in paths:
        cv2.polylines(dynamic, [np.rint(curve).astype(np.int32)], False, 1, 1)
    dynamic = _sweep(dynamic, 11).astype(np.uint8)
    for c in s["candidates"]:
        cv2.circle(dynamic, tuple(round(v) for v in c["point"]), 14, 1, -1)
    preserved, preserved_ok = _verify_source_static_preservation(frames, first, dynamic)
    order, order_ok = _verify_cumulative_ink_order(frames, first)
    scene_ok = all(point_target_delivery_ink_state(
        frame, first, record, paths=paths, discs=discs,
        clearance=clearance, construction_declared=True,
    )[1] for frame in frames)
    holds = bool(len(frames) >= 5 and all(matches(f, first) for f in frames[:2])
                 and all(_marker_gate(f, first, record["candidates"], answer, point_radius=11)[1] for f in frames[-2:]))
    process = bool(final and scene_ok and preserved_ok and order_ok and holds
                   and serial_ok is not False and fade_ok)
    return dict(
        whole_video_loaded=True,
        whole_video_process_correct=float(process),
        video_final_state_correct=float(final),
        expected_candidate=answer,
        static_preserved=bool(scene_ok and preserved_ok),
        video_semantic_process_parser="source_geometry_serial_strokes_then_marker_fade",
        video_endpoint_holds=holds,
        **marker, **ink, **terminal, **serial, **preserved, **order, **fade,
    )
