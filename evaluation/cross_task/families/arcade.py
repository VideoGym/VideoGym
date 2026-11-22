"""Destructive timed chain reactions and recurrent rotor-router dynamics."""

from evaluation.cross_task.drawing import GOLD, INK, TEAL, Canvas, Slot


def bomb_times(s):
    bombs = s["bombs"]
    times = {cell: t for cell, t in bombs}
    walls = set(s["walls"])
    radius = s["radius"]
    changed = True
    while changed:
        changed = False
        for cell, _ in bombs:
            r, c = divmod(cell, 7)
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                for d in range(1, radius + 1):
                    rr, cc = r + dr * d, c + dc * d
                    if not 0 <= rr < 7 or not 0 <= cc < 7 or rr * 7 + cc in walls:
                        break
                    v = rr * 7 + cc
                    if v in times and times[v] > times[cell]:
                        times[v] = times[cell]
                        changed = True
    return [times[cell] for cell, _ in bombs]


def rotor(s):
    dirs = s["rotors"][:]
    visits = [0] * 9
    u = s["start"]
    for _ in range(s["steps"]):
        visits[u] += 1
        dirs[u] = (dirs[u] + 1) % len(s["edges"][u])
        u = s["edges"][u][dirs[u]]
    return visits, dirs, u


def sample(rng, key):
    s = {"goal": key}
    if key == "bomb_chain_timing":
        s.update(walls=rng.sample(list(range(49)), 8), radius=3)
        free = sorted(set(range(49)) - set(s["walls"]))
        s["bombs"] = [[cell, rng.randint(2, 9)] for cell in rng.sample(free, 10)]
        result = bomb_times(s)
        if (
            sum(t != b[1] for t, b in zip(result, s["bombs"])) < 5
            or len(set(result)) < 2
        ):
            return sample(rng, key)
    else:
        s.update(
            edges=[
                [j for j in range(9) if abs(i // 3 - j // 3) + abs(i % 3 - j % 3) == 1]
                for i in range(9)
            ],
            start=rng.randrange(9),
            steps=rng.randint(19, 25),
        )
        s["rotors"] = [rng.randrange(len(e)) for e in s["edges"]]
        visits, _, _ = rotor(s)
        if min(visits) == 0 or max(visits) < 4:
            return sample(rng, key)
    return s


def solve(s):
    if s["goal"] == "bomb_chain_timing":
        return [tuple(bomb_times(s))]
    visits, _dirs, u = rotor(s)
    return [tuple(visits) + (u + 1,)]


def scene(s):
    k = s["goal"]
    c = Canvas()
    slots = []
    if k == "bomb_chain_timing":
        xy = c.grid(7, 7, 60, 83, 56)
        for cell in s["walls"]:
            x, y = xy[cell]
            c.d.rectangle((x - 27, y - 27, x + 27, y + 27), fill=INK)
            c.line([(x - 25, y), (x + 25, y)], "#5e7283", 2)
        for cell, t in s["bombs"]:
            x, y = xy[cell]
            c.d.ellipse((x - 19, y - 22, x + 19, y + 16), fill=INK)
            c.line([(x + 2, y - 23), (x + 10, y - 30)], GOLD, 3)
            c.text((x, y - 14), t, 12, "white")
            slots.append(Slot("number", (x, y + 7), tuple(range(2, 10)), size=16))
    else:
        p = [(111 + i % 3 * 145, 130 + i // 3 * 125) for i in range(9)]
        for i, edges in enumerate(s["edges"]):
            for j in edges:
                if i < j:
                    c.line([p[i], p[j]], "#afc4d2", 3)
        for i, edges in enumerate(s["edges"]):
            x, y = p[i]
            c.d.ellipse(
                (x - 31, y - 31, x + 31, y + 31), fill="#e9f2f4", outline=TEAL, width=2
            )
            target = p[edges[s["rotors"][i]]]
            dx = target[0] - x
            dy = target[1] - y
            norm = (dx * dx + dy * dy) ** 0.5
            c.arrow((x, y), (x + 24 * dx / norm, y + 24 * dy / norm), TEAL, 3)
            c.text((x - 40, y - 25), i + 1, 14)
            for j, v in enumerate(edges):
                tx, ty = p[v]
                dx, dy = tx - x, ty - y
                norm = (dx * dx + dy * dy) ** 0.5
                c.text((x + 43 * dx / norm, y + 43 * dy / norm), j + 1, 11)
            slots.append(
                Slot("number", (x + 48, y + 29), tuple(range(s["steps"] + 1)), size=16)
            )
        slots.append(Slot("number", (462, 473), tuple(range(1, 10)), size=16))
    return c, slots
