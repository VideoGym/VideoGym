"""Capacitated flow conservation and globally rooted directed trees."""

from itertools import product

from evaluation.cross_task.drawing import (
    BLUE,
    INK,
    RED,
    TEAL,
    Canvas,
    Slot,
    answer_vector,
)


def sample(rng, key):
    s = {"goal": key}
    if key == "capacitated_max_flow":
        s["edges"] = [
            [a, b, rng.randint(2, 5)]
            for a, b in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5))
        ]
        if len(solve(s)) > 30:
            return sample(rng, key)
    else:
        edges = {(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)}
        edges.update(
            rng.sample([(a, b) for a in range(6) for b in range(1, 6) if a != b], 7)
        )
        s["edges"] = [[a, b, rng.randint(1, 9)] for a, b in sorted(edges)]
        if len(solve(s)) > 4 or not has_greedy_cycle(s):
            return sample(rng, key)
    return s


def has_greedy_cycle(s):
    chosen = [
        min((e for e in s["edges"] if e[1] == v), key=lambda e: e[2])
        for v in range(1, 6)
    ]
    seen = {0}
    for _ in range(6):
        seen |= {v for u, v, _ in chosen if u in seen}
    return len(seen) < 6


def solve(s):
    edges = s["edges"]
    out = []
    if s["goal"] == "capacitated_max_flow":
        caps = [e[2] for e in edges]
        best = -1
        for a, b, c, d in product(*(range(caps[i] + 1) for i in range(2, 6))):
            flows = (a + b, c + d, a, b, c, d, a + c, b + d)
            if any(f > cap for f, cap in zip(flows, caps)):
                continue
            value = a + b + c + d
            if value > best:
                best = value
                out = []
            if value == best:
                out.append(flows)
    else:
        groups = [
            [i for i, (_, v, _) in enumerate(edges) if v == j] for j in range(1, 6)
        ]
        best = 999
        for chosen in product(*groups):
            seen = {0}
            for _ in range(6):
                seen |= {edges[i][1] for i in chosen if edges[i][0] in seen}
            if len(seen) != 6:
                continue
            cost = sum(edges[i][2] for i in chosen)
            if cost < best:
                best = cost
                out = []
            if cost == best:
                out.append(answer_vector(len(edges), chosen))
    return out


def scene(s):
    k = s["goal"]
    c = Canvas()
    slots = []
    pos = [(53, 270), (180, 135), (180, 400), (331, 135), (331, 400), (459, 270)]
    for j, (u, v, w) in enumerate(s["edges"]):
        a, b = pos[u], pos[v]
        # Opposite directions receive opposite bends, with a fixed public arc.
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = (dx * dx + dy * dy) ** 0.5
        shift = 17 if [v, u] in [e[:2] for e in s["edges"]] else 0
        mid = (
            (a[0] + b[0]) / 2 - shift * dy / length,
            (a[1] + b[1]) / 2 + shift * dx / length,
        )
        c.line([a, mid, b], "#9bb0c0", 2)
        c.arrow(mid, b, BLUE, 2, end_inset=24)
        t = 0.34 if j % 2 == 0 else 0.66
        x = a[0] + t * dx - shift * dy / length
        y = a[1] + t * dy + shift * dx / length
        if k == "capacitated_max_flow":
            c.d.rounded_rectangle(
                (x - 22, y - 19, x + 22, y + 19),
                radius=4,
                fill="white",
            )
            c.text((x, y + 30), str(w), 11)
            slots.append(Slot("number", (x, y), tuple(range(w + 1)), size=18))
        else:
            c.text((x, y - 15), w, 15, INK)
            # Finite edge strokes leave terminal nodes visible.
            aa = (a[0] + dx / length * 24, a[1] + dy / length * 24)
            bb = (b[0] - dx / length * 24, b[1] - dy / length * 24)
            slots.append(Slot("line", [aa, mid, bb]))
    for i, p in enumerate(pos):
        c.dot(
            p,
            23,
            TEAL if i == 0 else RED if i == 5 and k == "capacitated_max_flow" else BLUE,
        )
        c.text(
            p,
            "S"
            if i == 0
            else "T"
            if i == 5 and k == "capacitated_max_flow"
            else str(i),
            19,
            "white",
        )
    return c, slots
