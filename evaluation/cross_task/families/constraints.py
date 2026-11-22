"""Global run-length and sum/distinctness constraints, absent from the ID family."""

from itertools import permutations, product

from evaluation.cross_task.drawing import BLUE, INK, TEAL, Canvas, Slot, cell_slots


def runs(row):
    out = []
    n = 0
    for v in list(row) + [0]:
        if v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    return out or [0]


def groups(cells):
    out = []
    for vertical in (False, True):
        for a in range(4):
            run = []
            for b in range(5):
                i = b * 4 + a if vertical else a * 4 + b
                if b < 4 and i in cells:
                    run.append(i)
                elif run:
                    out.append(run)
                    run = []
    return out


def kakuro(s, limit=None):
    cells = s["cells"]
    gs = groups(cells)
    givens = {int(k): v for k, v in s["givens"].items()}
    domains = [
        [
            p
            for p in permutations(range(1, 7), len(g))
            if sum(p) == total
            and all(i not in givens or givens[i] == v for i, v in zip(g, p))
        ]
        for g, total in zip(gs, s["sums"])
    ]
    out = []

    def search(assigned, ds):
        if limit and len(out) >= limit:
            return
        remaining = [i for i in cells if i not in assigned]
        if not remaining:
            out.append(tuple(assigned[i] for i in cells if i not in givens))
            return
        options = {
            i: set.intersection(
                *({p[g.index(i)] for p in d} for g, d in zip(gs, ds) if i in g)
            )
            for i in remaining
        }
        i = min(remaining, key=lambda j: len(options[j]))
        for v in sorted(options[i]):
            nd = [
                ([p for p in d if p[g.index(i)] == v] if i in g else d)
                for g, d in zip(gs, ds)
            ]
            if all(nd):
                search({**assigned, i: v}, nd)

    search(givens, domains)
    return out


def solve(s):
    if s["goal"] == "kakuro_cross_sums":
        return kakuro(s)
    rows = [
        [p for p in product((0, 1), repeat=5) if runs(p) == clue] for clue in s["rows"]
    ]
    out = []
    for grid in product(*rows):
        if all(runs([grid[r][c] for r in range(5)]) == s["cols"][c] for c in range(5)):
            out.append(tuple(v for row in grid for v in row))
    return out


def sample(rng, key):
    s = {"goal": key}
    if key == "nonogram_runs":
        for _ in range(1000):
            grid = [[rng.randrange(2) for _ in range(5)] for _ in range(5)]
            s.update(
                rows=[runs(r) for r in grid],
                cols=[runs([grid[r][c] for r in range(5)]) for c in range(5)],
            )
            if (
                all(0 not in c and c != [5] for c in s["rows"] + s["cols"])
                and len(solve(s)) == 1
                and sum(map(sum, grid)) in range(11, 17)
            ):
                return s
    else:
        cells = [i for i in range(16) if i not in (0, 15)]
        digits = rng.sample(range(1, 7), 6)
        values = {i: digits[(i // 4 + i % 4) % 6] for i in cells}
        gs = groups(cells)
        s.update(cells=cells, sums=[sum(values[i] for i in g) for g in gs], givens={})
        order = rng.sample(cells, len(cells))
        while len(kakuro(s, 2)) != 1:
            i = order.pop()
            s["givens"][str(i)] = values[i]
        if len(s["givens"]) > 5:
            return sample(rng, key)
        return s
    raise ValueError("no unique run-length puzzle")


def scene(s):
    k = s["goal"]
    c = Canvas()
    slots = []
    if k == "nonogram_runs":
        xy = c.grid(5, 5, 157, 173, 60)
        for i, row in enumerate(s["rows"]):
            c.text((128, 203 + i * 60), " ".join(map(str, row)), 19, INK, "rm")
        for i, col in enumerate(s["cols"]):
            for j, v in enumerate(reversed(col)):
                c.text((187 + i * 60, 148 - j * 24), v, 19)
        slots = cell_slots(xy, 60, (0, 1), ("#d3dce5", TEAL))
    else:
        cells = set(s["cells"])
        gs = groups(cells)
        values = {int(k): v for k, v in s["givens"].items()}
        xy = c.grid(5, 5, 70, 155, 72)
        for r in range(5):
            for col in range(5):
                x, y = xy[r * 5 + col]
                i = (r - 1) * 4 + col - 1
                if r == 0 or col == 0 or i not in cells:
                    c.d.rectangle((x - 35, y - 35, x + 35, y + 35), fill=INK)
        for g, total in zip(gs, s["sums"]):
            r, col = divmod(g[0], 4)
            horizontal = g[1] - g[0] == 1
            x, y = xy[
                (r + 1 if horizontal else r) * 5 + (col if horizontal else col + 1)
            ]
            c.line([(x - 34, y - 34), (x + 34, y + 34)], "#a2b8cb", 1)
            c.text(
                (x + 14, y - 13) if horizontal else (x - 14, y + 14), total, 17, "white"
            )
        for i in sorted(cells):
            r, col = divmod(i, 4)
            x, y = xy[(r + 1) * 5 + col + 1]
            if i in values:
                c.text((x, y), values[i], 24, BLUE)
            else:
                slots.append(Slot("number", (x, y), tuple(range(1, 7)), size=24))
    return c, slots
