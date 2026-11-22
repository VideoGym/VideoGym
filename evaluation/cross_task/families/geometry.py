"""Ten distinct geometric problems: topology, metric optimization and construction."""

import math
from functools import lru_cache
from itertools import combinations, pairwise, permutations

import numpy as np

from evaluation.cross_task.drawing import (
    BLUE,
    INK,
    RED,
    TEAL,
    Canvas,
    Slot,
    answer_vector,
    cell_slots,
)


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def convex(points):
    return all(
        cross(points[i - 1], points[i], points[(i + 1) % len(points)]) > 0
        for i in range(len(points))
    )


def sample(rng, key):
    s = {"goal": key}
    if key in ("polygon_triangulation", "minimum_width_strip"):
        while True:
            angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(8))
            p = [
                [round(256 + 182 * math.cos(a)), round(278 + 170 * math.sin(a))]
                for a in angles
            ]
            if min(math.dist(a, b) for a, b in combinations(p, 2)) > 48 and convex(p):
                break
        s["points"] = p
    elif key == "taut_obstacle_path":
        s.update(
            start=[48, rng.randint(155, 210)],
            goal_point=[464, rng.randint(330, 390)],
            obstacles=[
                [135, 100 + rng.randrange(20), 195, 315 + rng.randrange(20)],
                [290, 235 + rng.randrange(20), 350, 440],
            ],
        )
    elif key == "bichromatic_matching":
        while True:
            points = [[rng.randint(62, 450), rng.randint(115, 446)] for _ in range(10)]
            if min(math.dist(a, b) for a, b in combinations(points, 2)) < 48:
                continue
            s["red"], s["blue"] = points[:5], points[5:]
            nearest = [
                min(range(5), key=lambda j: math.dist(a, s["blue"][j]))
                for a in s["red"]
            ]
            if len(set(nearest)) < 5:
                break
    elif key == "rectangle_partition":
        # A skyline with holes forces a global, non-overlapping rectangulation.
        while True:
            cells = sorted(rng.sample(list(range(25)), rng.randrange(17, 22)))
            s["cells"] = cells
            if len(rectangle_solutions(s)[0]) >= 5:
                break
    elif key == "halfplane_intersection":
        # Keep a visible segment between the two antialiased corner joins;
        # 2–3 px slivers cannot be distinguished from a joined corner.
        for _ in range(2000):
            s["halfplanes"] = [
                [1, 0, rng.randrange(345, 410)],
                [-1, 0, -rng.randrange(95, 120)],
                [0, 1, rng.randrange(380, 435)],
                [0, -1, -rng.randrange(105, 155)],
                [1, 1, rng.randrange(590, 635)],
                [-1, 1, rng.randrange(205, 245)],
                [1, -1, rng.randrange(135, 170)],
            ]
            polygon = clipped(s)
            if (
                len(polygon) >= 4
                and min(
                    math.dist(a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1])
                )
                >= 24
            ):
                break
        else:
            raise ValueError("no readable half-plane intersection")
    elif key == "power_cells":
        s["sites"] = [
            [rng.randrange(3), rng.randrange(3), rng.choice([0, 2, 4, 6])],
            [rng.randrange(4, 7), rng.randrange(3), rng.choice([0, 2, 4, 6])],
            [rng.randrange(2, 6), rng.randrange(4, 7), rng.choice([0, 2, 4, 6])],
        ]
        # Only queries with a visible integer power margin are used.
        s["queries"] = rng.sample(
            [
                [x, y]
                for x in range(7)
                for y in range(7)
                if [x, y] not in [p[:2] for p in s["sites"]]
                and len({(x - a) ** 2 + (y - b) ** 2 - w for a, b, w in s["sites"]})
                == 3
            ],
            10,
        )
    elif key == "projective_landmarks":
        s["corners"] = [
            [rng.randrange(40, 80), rng.randrange(320, 350)],
            [rng.randrange(375, 425), rng.randrange(300, 330)],
            [rng.randrange(405, 450), rng.randrange(445, 475)],
            [rng.randrange(90, 130), rng.randrange(450, 480)],
        ]
        s["queries"] = rng.sample(
            [[x / 4, y / 4] for x in (1, 2, 3) for y in (1, 2, 3)], 4
        )
    elif key == "disk_union_boundary":
        s["circles"] = [
            [180 + rng.randrange(20), 230 + rng.randrange(20), 93],
            [305 + rng.randrange(20), 235 + rng.randrange(20), 100],
            [250 + rng.randrange(20), 345 + rng.randrange(20), 90],
        ]
    elif key == "minkowski_sum":
        s["a"] = rng.sample([[x, y] for x in range(3) for y in range(3)], 5)
        s["b"] = rng.sample([[x, y] for x in range(3) for y in range(3)], 4)
    else:
        raise KeyError(key)
    return s


def diagonals(n):
    return [
        (a, b)
        for a, b in combinations(range(n), 2)
        if b != a + 1 and (a, b) != (0, n - 1)
    ]


def triangulations(n):
    @lru_cache(None)
    def recur(i, j):
        if j - i < 2:
            return (frozenset(),)
        out = []
        for k in range(i + 1, j):
            edges = {(i, k), (k, j)} & set(diagonals(n))
            out.extend(a | b | edges for a in recur(i, k) for b in recur(k, j))
        return tuple(set(out))

    return recur(0, n - 1)


def rectangles(s):
    cells = set(s["cells"])
    out = []
    for y0, y1 in combinations(range(6), 2):
        for x0, x1 in combinations(range(6), 2):
            a = frozenset(y * 5 + x for y in range(y0, y1) for x in range(x0, x1))
            if a <= cells:
                out.append((a, (x0, y0, x1, y1)))
    return out


def rectangle_solutions(s):
    rs = rectangles(s)
    by_cell = {i: [j for j, (a, _) in enumerate(rs) if i in a] for i in s["cells"]}

    @lru_cache(None)
    def cover(rem):
        if not rem:
            return ((),)
        chosen = min(rem, key=lambda i: sum(rs[j][0] <= rem for j in by_cell[i]))
        found = []
        best = 99
        for j in by_cell[chosen]:
            cells = rs[j][0]
            if not cells <= rem:
                continue
            for rest in cover(rem - cells):
                if len(rest) + 1 < best:
                    best = len(rest) + 1
                    found = []
                if len(rest) + 1 == best:
                    found.append(tuple(sorted((j,) + rest)))
        return tuple(set(found))

    return cover(frozenset(s["cells"]))


def clipped(s):
    poly = [[30.0, 80.0], [482.0, 80.0], [482.0, 482.0], [30.0, 482.0]]
    for a, b, c in s["halfplanes"]:
        out = []
        for p, q in zip(poly, poly[1:] + poly[:1]):
            u = a * p[0] + b * p[1] - c
            v = a * q[0] + b * q[1] - c
            if u <= 0:
                out.append(p)
            if (u < 0 < v) or (v < 0 < u):
                out.append([p[k] + u / (u - v) * (q[k] - p[k]) for k in range(2)])
        poly = out
    return poly


def projected(s):
    rows = []
    rhs = []
    for (x, y), (u, v) in zip(((0, 0), (1, 0), (1, 1), (0, 1)), s["corners"]):
        rows.extend(
            [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        )
        rhs.extend([u, v])
    h = np.append(np.linalg.solve(rows, rhs), 1).reshape(3, 3)
    pts = []
    for p in s["queries"]:
        v = h @ np.array([*p, 1])
        pts.append((v[:2] / v[2]).tolist())
    return pts


def visible(s, a, b):
    # Sample the open segment against axis-aligned obstacle interiors; vertices
    # are exact and all scene margins exceed the sampling interval.
    for x0, y0, x1, y1 in s["obstacles"]:
        lo, hi = 0.0, 1.0
        for av, bv, mn, mx in ((a[0], b[0], x0, x1), (a[1], b[1], y0, y1)):
            if abs(bv - av) < 1e-9:
                if not mn + 1e-8 < av < mx - 1e-8:
                    lo, hi = 1, 0
                    break
            else:
                p, q = sorted(((mn - av) / (bv - av), (mx - av) / (bv - av)))
                lo = max(lo, p + 1e-9)
                hi = min(hi, q - 1e-9)
        if lo < hi:
            return False
    return True


def path_graph(s):
    p = [s["start"], s["goal_point"]]
    for x0, y0, x1, y1 in s["obstacles"]:
        p.extend([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    edges = [
        (a, b) for a, b in combinations(range(len(p)), 2) if visible(s, p[a], p[b])
    ]
    return p, edges


def union_arcs(s):
    arcs = []
    circles = s["circles"]
    for i, (x, y, r) in enumerate(circles):
        angles = [0.0, 2 * math.pi]
        for j, (u, v, t) in enumerate(circles):
            d = math.hypot(u - x, v - y)
            if i == j or not abs(r - t) < d < r + t:
                continue
            theta = math.atan2(v - y, u - x)
            alpha = math.acos((r * r + d * d - t * t) / (2 * r * d))
            angles.extend(
                [(theta - alpha) % (2 * math.pi), (theta + alpha) % (2 * math.pi)]
            )
        angles = sorted(angles)
        for a, b in pairwise(angles):
            mid = (a + b) / 2
            px = x + r * math.cos(mid)
            py = y + r * math.sin(mid)
            if any(
                j != i and math.hypot(px - u, py - v) < t - 1e-6
                for j, (u, v, t) in enumerate(circles)
            ):
                continue
            arcs.append(
                ((x - r, y - r, x + r, y + r), math.degrees(a), math.degrees(b))
            )

    return arcs


def solve(s):
    k = s["goal"]
    if k == "polygon_triangulation":
        p = s["points"]
        edges = diagonals(len(p))
        opts = triangulations(len(p))
        costs = [sum(math.dist(p[a], p[b]) for a, b in o) for o in opts]
        best = min(costs)
        return [
            answer_vector(len(edges), [edges.index(e) for e in o])
            for o, c in zip(opts, costs)
            if abs(c - best) < 1e-6
        ]
    if k == "bichromatic_matching":
        r, b = s["red"], s["blue"]
        valid = []
        for perm in permutations(range(5)):
            if any(
                cross(r[i], b[perm[i]], r[j]) * cross(r[i], b[perm[i]], b[perm[j]]) < 0
                and cross(r[j], b[perm[j]], r[i]) * cross(r[j], b[perm[j]], b[perm[i]])
                < 0
                for i, j in combinations(range(5), 2)
            ):
                continue
            valid.append((sum(math.dist(r[i], b[v]) for i, v in enumerate(perm)), perm))
        best = min(c for c, _ in valid)
        return [
            answer_vector(25, [i * 5 + j for i, j in enumerate(p)])
            for c, p in valid
            if c <= best + 1e-6
        ]
    if k == "taut_obstacle_path":
        import heapq

        p, edges = path_graph(s)
        queue = [(0.0, 0, ())]
        seen = {}
        while queue:
            d, u, path = heapq.heappop(queue)
            if u == 1:
                return [answer_vector(len(edges), path)]
            if d > seen.get(u, float("inf")):
                continue
            for i, (a, b) in enumerate(edges):
                if u not in (a, b):
                    continue
                v = b if a == u else a
                nd = d + math.dist(p[u], p[v])
                if nd < seen.get(v, float("inf")) - 1e-8:
                    seen[v] = nd
                    heapq.heappush(queue, (nd, v, path + (i,)))
    if k == "rectangle_partition":
        # Encode internal cuts, rather than filling a common menu of rectangles.
        edges = partition_edges(s)
        out = []
        for rects in rectangle_solutions(s):
            region = {
                cell: i for i, r in enumerate(rects) for cell in rectangles(s)[r][0]
            }
            out.append(
                answer_vector(
                    len(edges),
                    [j for j, (a, b) in enumerate(edges) if region[a] != region[b]],
                )
            )
        return list(set(out))
    if k == "halfplane_intersection":
        return [tuple([1] * len(clipped(s)))]
    if k == "power_cells":
        return [
            tuple(
                min(
                    range(3),
                    key=lambda i: (
                        (x - s["sites"][i][0]) ** 2
                        + (y - s["sites"][i][1]) ** 2
                        - s["sites"][i][2]
                    ),
                )
                for x, y in s["queries"]
            )
        ]
    if k == "projective_landmarks":
        return [(1, 2, 3, 4)]
    if k == "disk_union_boundary":
        return [tuple([1] * len(union_arcs(s)))]
    if k == "minkowski_sum":
        cells = {(x + u, y + v) for x, y in s["a"] for u, v in s["b"]}
        return [answer_vector(25, [y * 5 + x for x, y in cells])]
    if k == "minimum_width_strip":
        p = s["points"]
        widths = [
            max(abs(cross(a, b, c)) / math.dist(a, b) for c in p)
            for a, b in zip(p, p[1:] + p[:1])
        ]
        best = min(widths)
        return [
            answer_vector(2 * len(p), [2 * i, 2 * i + 1])
            for i, w in enumerate(widths)
            if abs(w - best) < 1e-6
        ]
    raise KeyError(k)


def partition_edges(s):
    cells = set(s["cells"])
    return [
        (a, b)
        for a in sorted(cells)
        for b in (a + 1, a + 5)
        if b in cells and (b == a + 5 or a // 5 == b // 5)
    ]


def scene(s):
    k = s["goal"]
    c = Canvas()
    slots = []
    if k in ("polygon_triangulation", "minimum_width_strip"):
        p = s["points"]
        c.polygon([tuple(v) for v in p], fill="#e4f1ef", outline=TEAL, width=3)
        for i, v in enumerate(p):
            c.dot(v, 5, TEAL, chr(65 + i))
        if k == "polygon_triangulation":
            slots = [Slot("line", [p[a], p[b]]) for a, b in diagonals(len(p))]
        else:
            for i, a in enumerate(p):
                b = p[(i + 1) % len(p)]
                v = np.subtract(b, a).astype(float)
                v /= np.linalg.norm(v)
                n = np.array([-v[1], v[0]])
                d = max(np.dot(np.subtract(q, a), n) for q in p)
                # The two support segments are drawn together as one geometric strip.
                pts = [np.subtract(a, 20 * v), np.add(b, 20 * v)]
                slots.extend(
                    [
                        Slot("line", [pts[0], pts[1]]),
                        Slot("line", [pts[0] + d * n, pts[1] + d * n]),
                    ]
                )
    elif k == "taut_obstacle_path":
        for x0, y0, x1, y1 in s["obstacles"]:
            c.d.rectangle((x0, y0, x1, y1), fill="#b5c2ce", outline=INK, width=2)
            for y in range(y0 + 8, y1, 12):
                c.line([(x0 + 3, y), (x1 - 3, y)], "#d9e1e7", 1)
        p, edges = path_graph(s)
        for i, v in enumerate(p):
            c.dot(
                v, 5, TEAL if i < 2 else INK, "S" if i == 0 else "G" if i == 1 else None
            )
        slots = [Slot("line", [p[a], p[b]]) for a, b in edges]
    elif k == "bichromatic_matching":
        for i, p in enumerate(s["red"]):
            c.dot(p, 11, RED, chr(65 + i))
        for i, p in enumerate(s["blue"]):
            c.dot(p, 11, BLUE, str(i + 1))
        slots = [Slot("line", [a, b]) for a in s["red"] for b in s["blue"]]
    elif k == "rectangle_partition":
        cells = set(s["cells"])
        xy = c.grid(5, 5, 66, 110, 76)
        for i in range(25):
            x, y = xy[i]
            c.d.rectangle(
                (x - 37, y - 37, x + 37, y + 37), fill="#dbeee8" if i in cells else INK
            )
        for a, b in partition_edges(s):
            x, y = xy[a]
            slots.append(
                Slot(
                    "line",
                    [(x + 38, y - 38), (x + 38, y + 38)]
                    if b == a + 1
                    else [(x - 38, y + 38), (x + 38, y + 38)],
                )
            )
    elif k == "halfplane_intersection":
        for i, (a, b, d) in enumerate(s["halfplanes"]):
            pts = []
            for x in (30, 482):
                if b and 80 <= (d - a * x) / b <= 482:
                    pts.append((x, (d - a * x) / b))
            for y in (80, 482):
                if a and 30 <= (d - b * y) / a <= 482:
                    pts.append(((d - b * y) / a, y))
            if len(pts) >= 2:
                c.line(pts[:2], BLUE, 2)
                mid = np.mean(pts[:2], axis=0)
                v = np.array([a, b], float)
                v /= np.linalg.norm(v)
                c.arrow(mid, mid - 22 * v, TEAL, 2)
                c.text(mid + 16 * v, chr(65 + i), 12)
        p = clipped(s)
        slots = [Slot("line", [a, b]) for a, b in zip(p, p[1:] + p[:1])]
    elif k == "power_cells":
        xy = c.grid(7, 7, 62, 102, 55)
        for i, (x, y, w) in enumerate(s["sites"]):
            p = xy[y * 7 + x]
            c.dot(p, 16, (TEAL, BLUE, RED)[i])
            c.text(p, chr(65 + i), 16, "white")
            c.text((p[0], p[1] + 24), f"w={w}", 13)
        for i, (x, y) in enumerate(s["queries"]):
            px, py = xy[y * 7 + x]
            c.d.ellipse((px - 8, py - 8, px + 8, py + 8), outline=INK, width=1)
            slots.append(
                Slot(
                    "ring",
                    (px - 19, py - 19, px + 19, py + 19),
                    (0, 1, 2),
                    (TEAL, BLUE, RED),
                )
            )
    elif k == "projective_landmarks":
        c.panel((26, 78, 486, 272))
        p = [[188, 115], [323, 115], [323, 250], [188, 250]]
        c.polygon(p, outline=BLUE, width=3)
        for i, v in enumerate(p):
            c.dot(v, 4, BLUE, chr(65 + i))
        for i, (x, y) in enumerate(s["queries"]):
            c.dot((188 + 135 * x, 115 + 135 * y), 5, RED, str(i + 1))
        q = s["corners"]
        c.polygon([tuple(v) for v in q], fill="#e6f1f7", outline=BLUE, width=3)
        for i, v in enumerate(q):
            c.dot(v, 4, BLUE, chr(65 + i))
        for i, (x, y) in enumerate(projected(s)):
            slots.append(Slot("number", (x, y), (i + 1,), size=16, guide=False))
    elif k == "disk_union_boundary":
        for i, (x, y, r) in enumerate(s["circles"]):
            c.d.ellipse(
                (x - r, y - r, x + r, y + r), outline=(TEAL, BLUE, RED)[i], width=2
            )
            c.dot((x, y), 3, INK, chr(65 + i))
        slots = [Slot("arc", arc) for arc in union_arcs(s)]
    elif k == "minkowski_sum":
        for j, key in enumerate(("a", "b")):
            xy = c.grid(3, 3, 75 + j * 230, 105, 35)
            c.text((125 + j * 230, 88), key.upper(), 16)
            for x, y in s[key]:
                px, py = xy[y * 3 + x]
                c.d.rectangle(
                    (px - 14, py - 14, px + 14, py + 14), fill=TEAL if j == 0 else BLUE
                )
        xy = c.grid(5, 5, 156, 270, 40)
        slots = cell_slots(xy, 40)
    return c, slots
