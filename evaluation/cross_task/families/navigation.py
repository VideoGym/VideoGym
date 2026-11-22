"""Planning in joint configuration, reactive, probabilistic and homotopy spaces."""

import math
from collections import deque
from functools import lru_cache
from itertools import pairwise, product

from evaluation.cross_task.drawing import (
    BLUE,
    INK,
    RED,
    TEAL,
    Canvas,
    Slot,
    answer_vector,
)

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def neighbours(cell, blocked, n=5):
    r, c = divmod(cell, n)
    return [
        rr * n + cc
        for dr, dc in DIRECTIONS
        if 0 <= (rr := r + dr) < n
        and 0 <= (cc := c + dc) < n
        and rr * n + cc not in blocked
    ]


@lru_cache(2048)
def distances(walls, target):
    q = deque([target])
    result = {target: 0}
    while q:
        u = q.popleft()
        for v in neighbours(u, walls):
            if v not in result:
                result[v] = result[u] + 1
                q.append(v)
    return result


def arrived(s, state):
    return (
        state[0] == s["target"][0]
        if s["goal"] == "reactive_pursuer_escape"
        else state == tuple(s["target"])
    )


def moves(s, state):
    blocked = set(s["walls"])
    k = s["goal"]
    if k == "reactive_pursuer_escape":
        runner, hunter = state
        for dest in [runner] + neighbours(runner, blocked):
            if dest == hunter:
                continue
            d = distances(tuple(sorted(blocked)), dest)
            options = neighbours(hunter, blocked)
            nxt = min(options, key=lambda cell: d.get(cell, 99)) if options else hunter
            if nxt == dest:
                continue
            yield (dest, nxt), (dest, nxt)
    elif k == "joint_robot_exchange":
        a, b = state
        for x, y in product([a] + neighbours(a, blocked), [b] + neighbours(b, blocked)):
            if x == y or (x, y) == (b, a) or (x, y) == state:
                continue
            yield (x, y), (x, y)
    else:
        cell, w1, w2 = state
        for v in neighbours(cell, blocked):
            r, c = divmod(cell, 5)
            rr, _cc = divmod(v, 5)
            wind = [w1, w2]
            # Oriented crossings of eastward rays from obstacle centres.
            for j, ob in enumerate(s["islands"]):
                y, x = divmod(ob, 5)
                if c > x and ((r == y and rr == y + 1) or (rr == y and r == y + 1)):
                    wind[j] += 1 if rr > r else -1
            if all(abs(w) <= 1 for w in wind):
                yield v, (v, *wind)


def shortest(s):
    start = tuple(s["initial"])
    q = deque([start])
    parents = {start: None}
    while q:
        state = q.popleft()
        if arrived(s, state):
            path = []
            while parents[state] is not None:
                prev, action = parents[state]
                path.append(action)
                state = prev
            return path[::-1]
        for action, nxt in moves(s, state):
            if nxt not in parents:
                parents[nxt] = (state, action)
                q.append(nxt)
    return None


def sample(rng, key):
    s = {"goal": key}
    if key == "reliable_bridge_route":
        layers = [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        s["edges"] = []
        for left, right in pairwise(layers):
            for u in left:
                for v in rng.sample(right, min(2, len(right))):
                    s["edges"].append([u, v, rng.choice([45, 55, 65, 75, 85, 95])])
        s.update(initial=[0], target=[10])
        opts = reliable_paths(s)
        u = 0
        greedy = 1
        while u != 10:
            outgoing = [e for e in s["edges"] if e[0] == u]
            _, u, w = max(outgoing, key=lambda e: e[2])
            greedy *= w
        if len(opts) < 4 or greedy >= opts[0][0]:
            return sample(rng, key)
    else:
        for _ in range(2000):
            if key == "winding_route":
                islands = rng.sample([6, 7, 8, 11, 12, 13, 16, 17, 18], 2)
                free = sorted(set(range(25)) - set(islands))
                s.update(
                    walls=islands, islands=islands, initial=[rng.choice(free), 0, 0]
                )
                s["target"] = [s["initial"][0], 1, 0]
            else:
                walls = rng.sample(range(25), 6)
                free = sorted(set(range(25)) - set(walls))
                a, b = rng.sample(free, 2)
                s["walls"] = walls
                if key == "reactive_pursuer_escape":
                    target = rng.choice([v for v in free if v not in (a, b)])
                    s.update(initial=[a, b], target=[target])
                else:
                    s.update(initial=[a, b], target=[b, a])
            path = shortest(s)
            if (
                key == "reactive_pursuer_escape"
                and path
                and len(path)
                <= distances(tuple(sorted(s["walls"])), s["target"][0]).get(a, 99)
            ):
                continue
            if path and 6 <= len(path) <= 12:
                s["horizon"] = 12
                break
        else:
            raise ValueError("could not sample a nontrivial planning instance")
    return s


def reliable_paths(s):
    out = []

    def visit(u, path, p):
        if u == s["target"][0]:
            out.append((p, path))
            return
        for i, (a, b, v) in enumerate(s["edges"]):
            if a == u:
                visit(b, path + [i], p * v)

    visit(0, [], 1)
    return sorted(out, reverse=True)


def solve(s):
    if s["goal"] == "reliable_bridge_route":
        opts = reliable_paths(s)
        best = opts[0][0]
        return [answer_vector(len(s["edges"]), p) for v, p in opts if v == best]
    path = shortest(s)
    if s["goal"] in ("joint_robot_exchange", "reactive_pursuer_escape"):
        return [
            tuple(x + 1 for pair in path for x in pair) + (0,) * (2 * (12 - len(path)))
        ]
    return [tuple(v + 1 for v in path) + (0,) * (12 - len(path))]


def accepts(s, answer):
    """Accept any shortest valid plan, not just the BFS witness."""
    if s["goal"] == "reliable_bridge_route":
        return tuple(answer) in solve(s)
    if -1 in answer:
        return False
    state = tuple(s["initial"])
    k = s["goal"]
    if k in ("joint_robot_exchange", "reactive_pursuer_escape"):
        actions = [(answer[i] - 1, answer[i + 1] - 1) for i in range(0, len(answer), 2)]
        stop = (-1, -1)
    else:
        actions = [v - 1 for v in answer]
        stop = -1
    if stop in actions:
        end = actions.index(stop)
        if any(a != stop for a in actions[end:]):
            return False
        actions = actions[:end]
    if len(actions) != len(shortest(s)):
        return False
    for a in actions:
        possible = dict(moves(s, state))
        if a not in possible:
            return False
        state = possible[a]
    return arrived(s, state)


def scene(s):
    k = s["goal"]
    c = Canvas()
    slots = []
    if k == "reliable_bridge_route":
        p = (
            [(52, 270)]
            + [
                (154 + 100 * stage, 130 + row * 140)
                for stage in range(3)
                for row in range(3)
            ]
            + [(460, 270)]
        )
        for j, (u, v, w) in enumerate(s["edges"]):
            c.arrow(p[u], p[v], "#8da5b8", 2, 19)
            t = 0.35 if j % 2 else 0.65
            x = p[u][0] + t * (p[v][0] - p[u][0])
            y = p[u][1] + t * (p[v][1] - p[u][1])
            c.d.rounded_rectangle(
                (x - 16, y - 11, x + 16, y + 11), radius=3, fill="white"
            )
            c.text((x, y), f"{w}%", 10)
        for i, xy in enumerate(p):
            c.dot(xy, 18, TEAL if i == 0 else RED if i == 10 else BLUE)
            c.text(xy, "S" if i == 0 else "G" if i == 10 else str(i), 15, "white")
        for u, v, _ in s["edges"]:
            a, b = p[u], p[v]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            d = math.hypot(dx, dy)
            # Offset answer strokes preserve the probability labels on source arcs.
            n = (-dy / d * 8, dx / d * 8)
            slots.append(
                Slot(
                    "line",
                    [
                        (a[0] + dx / d * 22 + n[0], a[1] + dy / d * 22 + n[1]),
                        (b[0] - dx / d * 22 + n[0], b[1] - dy / d * 22 + n[1]),
                    ],
                )
            )
        return c, slots
    xy = c.grid(5, 5, 106, 82, 60)
    for i, (x, y) in enumerate(xy):
        if i in s["walls"]:
            c.d.rounded_rectangle((x - 28, y - 28, x + 28, y + 28), radius=5, fill=INK)
            for yy in range(int(y - 20), int(y + 20), 10):
                c.line([(x - 20, yy), (x + 20, yy)], "#4d6074")
        else:
            c.text((x - 21, y - 20), i + 1, 10, "#8193a4")
    if k == "reactive_pursuer_escape":
        for cell, label, color in [
            (s["initial"][0], "R", TEAL),
            (s["initial"][1], "H", RED),
            (s["target"][0], "G", BLUE),
        ]:
            c.dot(xy[cell], 18, color)
            c.text(xy[cell], label, 19, "white")
    elif k == "joint_robot_exchange":
        for j, cell in enumerate(s["initial"]):
            c.dot(xy[cell], 18, (TEAL, BLUE)[j])
            c.text(xy[cell], "AB"[j], 19, "white")
    else:
        c.dot(xy[s["initial"][0]], 16, TEAL)
        c.text(xy[s["initial"][0]], "S", 17, "white")
        for j, cell in enumerate(s["islands"]):
            c.text(xy[cell], str(j + 1), 23, "white")
    options = tuple(range(26))
    h = s["horizon"]
    for i in range(h):
        x = 30 + i * (452 / max(h - 1, 1))
        for row in range(
            2 if k in ("joint_robot_exchange", "reactive_pursuer_escape") else 1
        ):
            y = 432 + row * 43
            slots.append(
                Slot(
                    "number",
                    (x, y),
                    options,
                    tuple(([TEAL] if row == 0 else [BLUE]) * len(options)),
                    size=14,
                )
            )
    return c, slots
