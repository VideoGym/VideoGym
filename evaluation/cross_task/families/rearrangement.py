"""Preference stability, disjunctive scheduling and exact spatial packing."""

from itertools import pairwise, permutations, product

from evaluation.cross_task.drawing import (
    BLUE,
    INK,
    PALETTE,
    TEAL,
    Canvas,
    Slot,
    cell_slots,
)


def sample(rng, key):
    s = {"goal": key}
    if key == "stable_partner_assignment":
        s.update(
            left=[rng.sample(range(5), 5) for _ in range(5)],
            right=[rng.sample(range(5), 5) for _ in range(5)],
        )
        if (
            len(solve(s)) > 4
            or sum(v - 1 == s["left"][i][0] for i, v in enumerate(solve(s)[0])) > 2
        ):
            return sample(rng, key)
    elif key == "jobshop_schedule":
        s["jobs"] = [
            [[m, rng.randint(1, 4)] for m in rng.sample(range(3), 3)] for _ in range(3)
        ]
        if len(solve(s)) > 60:
            return sample(rng, key)
    else:
        # Sample a complete partition into connected four-cell pieces, then
        # discard all placements. The public input contains shapes only.
        for _ in range(2000):
            remaining = set(range(16))
            pieces = []
            for j in range(4):
                chosen = {rng.choice(sorted(remaining))}
                while len(chosen) < 4:
                    adjacent = [
                        i
                        for i in sorted(remaining - chosen)
                        if any(
                            abs(i // 4 - v // 4) + abs(i % 4 - v % 4) == 1
                            for v in chosen
                        )
                    ]
                    if not adjacent:
                        break
                    chosen.add(rng.choice(adjacent))
                if len(chosen) != 4:
                    break
                pieces.append([[i % 4, i // 4] for i in sorted(chosen)])
                remaining -= chosen
            if len(pieces) == 4:
                normalized = []
                for p in pieces:
                    mx = min(x for x, y in p)
                    my = min(y for x, y in p)
                    normalized.append([[x - mx, y - my] for x, y in p])
                if len({min(rotations(p)) for p in normalized}) >= 3:
                    s["pieces"] = normalized
                    return s
        raise ValueError("no varied connected packing")
    return s


def schedules(s):
    jobs = s["jobs"]
    machine = {
        m: [j * 3 + o for j in range(3) for o in range(3) if jobs[j][o][0] == m]
        for m in range(3)
    }
    found = []
    for orders in product(*(list(permutations(machine[m])) for m in range(3))):
        edges = [(j * 3 + o, j * 3 + o + 1) for j in range(3) for o in range(2)]
        edges += [(a, b) for order in orders for a, b in pairwise(order)]
        remaining = set(range(9))
        start = [0] * 9
        while remaining:
            ready = [
                i
                for i in remaining
                if not any(v == i and u in remaining for u, v in edges)
            ]
            if not ready:
                break
            for i in ready:
                start[i] = max(
                    [start[u] + jobs[u // 3][u % 3][1] for u, v in edges if v == i]
                    or [0]
                )
                remaining.remove(i)
        if remaining:
            continue
        makespan = max(start[i] + jobs[i // 3][i % 3][1] for i in range(9))
        found.append((makespan, tuple(start)))
    best = min(t for t, _ in found)
    return best, list({p for t, p in found if t == best})


def rotations(piece):
    out = set()
    for flip in (1, -1):
        p = [(x * flip, y) for x, y in piece]
        for _ in range(4):
            p = [(-y, x) for x, y in p]
            mx = min(x for x, y in p)
            my = min(y for x, y in p)
            out.add(tuple(sorted((x - mx, y - my) for x, y in p)))
    return out


def packings(s):
    placements = []
    for piece in s["pieces"]:
        choices = []
        for shape in rotations(piece):
            for dx in range(4 - max(x for x, y in shape)):
                for dy in range(4 - max(y for x, y in shape)):
                    choices.append(frozenset((y + dy) * 4 + x + dx for x, y in shape))
        placements.append(set(choices))
    out = []

    def place(j, used, board):
        if j == 4:
            out.append(tuple(board))
            return
        for p in placements[j]:
            if p & used:
                continue
            b = board.copy()
            for cell in p:
                b[cell] = j
            place(j + 1, used | p, b)

    place(0, set(), [-1] * 16)
    return sorted(set(out))


def solve(s):
    k = s["goal"]
    if k == "stable_partner_assignment":
        out = []
        for p in permutations(range(5)):
            if not any(
                s["left"][a].index(b) < s["left"][a].index(p[a])
                and s["right"][b].index(a) < s["right"][b].index(p.index(b))
                for a in range(5)
                for b in range(5)
            ):
                out.append(tuple(v + 1 for v in p))
        return out
    if k == "jobshop_schedule":
        return schedules(s)[1]
    return packings(s)


def accepts(s, a):
    if s["goal"] != "jobshop_schedule":
        return tuple(a) in solve(s)
    if -1 in a:
        return False
    jobs = s["jobs"]
    if any(
        a[j * 3 + o] + jobs[j][o][1] > a[j * 3 + o + 1]
        for j in range(3)
        for o in range(2)
    ):
        return False
    for i in range(9):
        for j in range(i + 1, 9):
            if jobs[i // 3][i % 3][0] == jobs[j // 3][j % 3][0] and max(
                a[i], a[j]
            ) < min(a[i] + jobs[i // 3][i % 3][1], a[j] + jobs[j // 3][j % 3][1]):
                return False
    return max(a[i] + jobs[i // 3][i % 3][1] for i in range(9)) == schedules(s)[0]


def scene(s):
    k = s["goal"]
    c = Canvas()
    slots = []
    if k == "stable_partner_assignment":
        for side, key in enumerate(("left", "right")):
            x = 35 + side * 245
            c.panel((x, 85, x + 220, 356))
            c.text((x + 110, 105), "AB"[side], 17, TEAL if side == 0 else BLUE)
            for i, row in enumerate(s[key]):
                y = 137 + i * 47
                c.dot((x + 19, y), 13, TEAL if side == 0 else BLUE)
                c.text((x + 19, y), i + 1, 14, "white")
                for j, v in enumerate(row):
                    c.text((x + 58 + j * 33, y), v + 1, 16)
        for i in range(5):
            x = 78 + i * 90
            c.text((x, 412), f"A{i + 1} → B", 13)
            slots.append(Slot("number", (x, 450), tuple(range(1, 6))))
    elif k == "jobshop_schedule":
        for j, job in enumerate(s["jobs"]):
            y = 135 + j * 77
            c.text((47, y), chr(65 + j), 22)
            for o, (m, d) in enumerate(job):
                x = 125 + o * 147
                c.d.rounded_rectangle(
                    (x - 42, y - 23, x + 42, y + 23), radius=7, fill=PALETTE[m]
                )
                c.text((x, y), f"M{m + 1} / {d}", 17, "white")
                if o < 2:
                    c.arrow((x + 48, y), (x + 98, y), INK, 2)
        for j in range(3):
            for o in range(3):
                x = 145 + o * 115
                y = 391 + j * 39
                c.text((78, y), chr(65 + j), 16)
                slots.append(Slot("number", (x, y), tuple(range(25)), size=16))
    else:
        for j, piece in enumerate(s["pieces"]):
            for x, y in piece:
                xx = 37 + j * 121 + x * 22
                yy = 108 + y * 22
                c.d.rectangle((xx, yy, xx + 20, yy + 20), fill=PALETTE[j])
        xy = c.grid(4, 4, 144, 254, 56)
        slots = cell_slots(xy, 56, tuple(range(4)), tuple(PALETTE[:4]))
    return c, slots
