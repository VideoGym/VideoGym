"""Snell refraction across media, a physical law absent from reflection-only ID tasks."""

import itertools
import math

from evaluation.cross_task.drawing import RED, Canvas, Slot


def sample(rng, key):
    return {
        "goal": key,
        "indices": [
            1.0,
            rng.choice([1.4, 1.5, 1.6]),
            rng.choice([1.1, 1.2]),
            rng.choice([1.7, 1.8]),
            rng.choice([1.2, 1.3]),
        ],
        "sine": rng.choice([0.5, 0.55, 0.6, 0.65]),
        "start": [70, 125],
        "interfaces": [185, 255, 325, 395],
        "end_y": 485,
    }


def ray(s):
    p = s["indices"][0] * s["sine"]
    points = [s["start"]]
    for n, y in zip(s["indices"], s["interfaces"] + [s["end_y"]]):
        x0, y0 = points[-1]
        points.append([x0 + (y - y0) * p / math.sqrt(n * n - p * p), y])
    return points


def solve(s):
    return [tuple([1] * 5)]


def scene(s):
    c = Canvas()
    ys = [80] + s["interfaces"] + [485]
    for j, (a, b) in enumerate(itertools.pairwise(ys)):
        c.d.rectangle(
            (28, a, 484, b),
            fill=["#eef4fa", "#d1e6ed", "#e7f0eb", "#c4dce8", "#e0eee5"][j],
        )
        c.text((451, (a + b) / 2), f"n={s['indices'][j]:.1f}", 14)
        if j:
            c.line([(28, a), (484, a)], "#85a7b5", 2)
    x, y = s["start"]
    p = s["sine"]
    dx = 40 * p
    dy = 40 * math.sqrt(1 - p * p)
    c.arrow((x - dx, y - dy), (x, y), RED, 4)
    c.dot((x, y), 4, RED)
    points = ray(s)
    slots = [Slot("line", [a, b]) for a, b in itertools.pairwise(points)]
    return c, slots
