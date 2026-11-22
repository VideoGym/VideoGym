"""Unseen state mechanisms: token concurrency, causal intervention, uncertainty and delay."""

from collections import deque
from itertools import product

from evaluation.cross_task.drawing import (
    BLUE,
    GOLD,
    INK,
    RED,
    TEAL,
    Canvas,
    Slot,
)


def petri_paths(s):
    start = tuple(s["initial"])
    target = tuple(s["target"])
    q = deque([(start, ())])
    depth = {start: 0}
    out = []
    best = None
    while q:
        state, path = q.popleft()
        if best is not None and len(path) > best:
            break
        if state == target:
            out.append(path)
            best = len(path)
            continue
        if len(path) >= 9:
            continue
        for j, (ins, outs) in enumerate(s["transitions"]):
            if any(state[i] < n for i, n in enumerate(ins)):
                continue
            nxt = tuple(state[i] - ins[i] + outs[i] for i in range(4))
            if max(nxt) > 4:
                continue
            if depth.get(nxt, 99) >= len(path) + 1:
                depth[nxt] = len(path) + 1
                q.append((nxt, path + (j + 1,)))
    return list(dict.fromkeys(out))


def sample(rng, key):
    s = {"goal": key}
    if key == "petri_target_marking":
        # Conservation of six tokens makes the full reachability graph finite.
        while True:
            transitions = []
            for _ in range(5):
                ins = [0] * 4
                outs = [0] * 4
                for i in rng.sample(range(4), 2):
                    ins[i] += 1
                for i in rng.sample(range(4), 2):
                    outs[i] += 1
                transitions.append([ins, outs])
            initial = rng.choice([[3, 2, 1, 0], [0, 3, 1, 2], [2, 0, 3, 1]])
            target = [0] * 4
            for _ in range(6):
                target[rng.randrange(4)] += 1
            s.update(initial=initial, target=target, transitions=transitions)
            paths = petri_paths(s)
            if paths and 5 <= len(paths[0]) <= 8 and len(paths) < 300:
                break
    elif key == "causal_counterfactual":
        # Factual node values identify independent exogenous disturbances.
        s["parents"] = [[], [], [0, 1], [0, 2], [1, 3], [2, 4]]
        s["ops"] = ["ROOT", "ROOT"] + [
            rng.choice(["AND", "OR", "XOR"]) for _ in range(4)
        ]
        s["observed"] = [rng.randrange(2) for _ in range(6)]
        s["interventions"] = [
            [0, 1 - s["observed"][0]],
            [1, 1 - s["observed"][1]],
            [2, 1 - s["observed"][2]],
        ]
        if (
            sum(a != b for row in counterfactual(s) for a, b in zip(row, s["observed"]))
            < 7
        ):
            return sample(rng, key)
    elif key == "hidden_markov_path":
        s.update(
            transition=[[rng.choice([6, 7, 8]), 0], [0, rng.choice([6, 7, 8])]],
            emission=[[8, 2], [2, 8]],
            prior=[5, 5],
            observations=[rng.randrange(2) for _ in range(7)],
        )
        s["transition"][0][1] = 10 - s["transition"][0][0]
        s["transition"][1][0] = 10 - s["transition"][1][1]
        best = solve(s)[0]
        if len(set(best)) < 2 or all(a == b for a, b in zip(best, s["observations"])):
            return sample(rng, key)
    elif key == "delayed_wave_superposition":
        s["edges"] = [
            [0, 2, rng.randint(1, 2)],
            [1, 2, rng.randint(1, 2)],
            [0, 3, rng.randint(1, 3)],
            [2, 3, rng.randint(1, 2)],
            [2, 4, rng.randint(1, 3)],
            [3, 4, rng.randint(1, 2)],
        ]
        s["pulses"] = [[rng.choice([-1, 0, 1]) for _ in range(3)] for _ in range(2)]
        result = waves(s)
        if (
            sum(v != 0 for v in result[4]) < 4
            or not any(v < 0 for v in result[4])
            or not any(v > 0 for v in result[4])
        ):
            return sample(rng, key)
    return s


def op(name, a, b):
    return {"AND": a & b, "OR": a | b, "XOR": a ^ b}[name]


def counterfactual(s):
    obs = s["observed"]
    noise = obs[:2] + [
        obs[i] ^ op(s["ops"][i], *(obs[p] for p in s["parents"][i]))
        for i in range(2, 6)
    ]
    out = []
    for node, value in s["interventions"]:
        row = []
        for i in range(6):
            row.append(
                value
                if i == node
                else noise[i]
                if i < 2
                else noise[i] ^ op(s["ops"][i], *(row[p] for p in s["parents"][i]))
            )
        out.append(row)
    return out


def waves(s):
    v = [[0] * 10 for _ in range(5)]
    v[0][:3] = s["pulses"][0]
    v[1][:3] = s["pulses"][1]
    for node in range(2, 5):
        for u, w, d in s["edges"]:
            if w == node:
                for t in range(d, 10):
                    v[w][t] += v[u][t - d]
    return v


def solve(s):
    k = s["goal"]
    if k == "petri_target_marking":
        paths = petri_paths(s)
        return [tuple(p) + (0,) * (9 - len(p)) for p in paths]
    if k == "causal_counterfactual":
        return [tuple(v for row in counterfactual(s) for v in row)]
    if k == "hidden_markov_path":
        scored = []
        for path in product(range(2), repeat=7):
            cost = s["prior"][path[0]] * s["emission"][path[0]][s["observations"][0]]
            for i in range(1, 7):
                cost *= (
                    s["transition"][path[i - 1]][path[i]]
                    * s["emission"][path[i]][s["observations"][i]]
                )
            scored.append((cost, path))
        best = max(c for c, _ in scored)
        return [p for c, p in scored if c == best]
    if k == "delayed_wave_superposition":
        return [tuple(str(v) for v in waves(s)[4])]
    raise KeyError(k)


def scene(s):
    k = s["goal"]
    c = Canvas()
    slots = []
    if k == "petri_target_marking":
        left = [(65, 117 + i * 72) for i in range(4)]
        right = [(447, 117 + i * 72) for i in range(4)]
        transitions = [(256, 112 + i * 51) for i in range(5)]
        for j, (ins, outs) in enumerate(s["transitions"]):
            for i in range(4):
                if ins[i]:
                    c.arrow(
                        left[i], transitions[j], BLUE, 2, start_inset=24, end_inset=18
                    )
                if outs[i]:
                    c.arrow(
                        transitions[j], right[i], TEAL, 2, start_inset=18, end_inset=24
                    )
        for j, p in enumerate(transitions):
            x, y = p
            c.d.rectangle((x - 15, y - 17, x + 15, y + 17), fill=INK)
            c.text(p, j + 1, 16, "white")
        for i in range(4):
            for p, value, color in (
                (left[i], s["initial"][i], BLUE),
                (right[i], s["target"][i], TEAL),
            ):
                x, y = p
                c.d.ellipse(
                    (x - 23, y - 23, x + 23, y + 23),
                    fill="white",
                    outline=color,
                    width=3,
                )
                c.text(p, value, 21)
                c.text((x, y + 35), chr(65 + i), 14, color)
        for i in range(9):
            p = (56 + i * 50, 439)
            c.text((p[0], 480), i + 1, 11)
            slots.append(Slot("number", p, tuple(range(6)), size=18))
    elif k == "causal_counterfactual":
        pos = [(80, 130), (80, 240), (205, 130), (205, 240), (340, 240), (432, 130)]
        for i in range(2, 6):
            for p in s["parents"][i]:
                a, b = pos[p], pos[i]
                dx, dy = b[0] - a[0], b[1] - a[1]
                length = (dx * dx + dy * dy) ** 0.5
                inset = (
                    min(
                        32 * length / abs(dx) if dx else float("inf"),
                        26 * length / abs(dy) if dy else float("inf"),
                    )
                    + 2
                )
                c.arrow(a, b, BLUE, 2, inset)
        for i, (x, y) in enumerate(pos):
            c.d.rounded_rectangle(
                (x - 32, y - 26, x + 32, y + 26), radius=8, fill="#e4eef6", outline=BLUE
            )
            c.text((x - 13, y - 8), chr(65 + i), 16)
            c.text((x + 16, y - 8), s["observed"][i], 16, BLUE)
            if i >= 2:
                c.text(
                    (x, y + 11), {"AND": "∧", "OR": "∨", "XOR": "⊕"}[s["ops"][i]], 16
                )
        for r, (i, v) in enumerate(s["interventions"]):
            y = 343 + r * 56
            c.text((80, y), f"do({chr(65 + i)}={v})", 14)
            for j in range(6):
                p = (172 + j * 53, y)
                if r == 0:
                    c.text((p[0], 315), chr(65 + j), 12)
                slots.append(Slot("number", p, (0, 1)))
    elif k == "hidden_markov_path":
        for i, x in enumerate((145, 367)):
            c.dot((x, 152), 27, (TEAL, BLUE)[i])
            c.text((x, 152), i, 24, "white")
            for bit in range(2):
                bx = x - 34 + 68 * bit
                c.dot((bx, 206), 11, RED if bit else GOLD)
                c.text((bx, 206), bit, 13, "white")
                c.text((bx, 232), f"{s['emission'][i][bit]}/10", 12)
        c.arrow((178, 140), (332, 140), BLUE, 2)
        c.text((256, 124), f"{s['transition'][0][1]}/10", 13)
        c.arrow((332, 169), (178, 169), TEAL, 2)
        c.text((256, 184), f"{s['transition'][1][0]}/10", 13)
        for i, v in enumerate(s["observations"]):
            x = 61 + i * 65
            c.dot((x, 325), 20, RED if v else GOLD)
            c.text((x, 325), v, 20, "white")
            c.arrow((x, 349), (x, 382), "#b4c3d0", 1)
            slots.append(Slot("number", (x, 415), (0, 1), (TEAL, BLUE)))
    else:
        pos = [(80, 125), (80, 245), (225, 190), (365, 125), (430, 265)]
        for u, v, d in s["edges"]:
            c.arrow(pos[u], pos[v], BLUE, 2, 23)
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            c.text((x, y - 12), f"+{d}", 13)
        for i, p in enumerate(pos):
            c.dot(p, 20, TEAL if i < 2 else BLUE)
            c.text(p, chr(65 + i), 20, "white")
        for r, pulse in enumerate(s["pulses"]):
            c.text((93, 310 + r * 33), chr(65 + r), 16, TEAL)
            for t, value in enumerate(pulse):
                c.text((150 + t * 58, 310 + r * 33), value, 18)
        for i in range(10):
            p = (40 + i * 48, 423)
            c.text((p[0], 462), str(i), 11)
            slots.append(
                Slot("number", p, tuple(str(v) for v in range(-8, 9)), size=16)
            )
    return c, slots
