"""Three-dimensional polycube assembly under proper rigid rotations."""

from itertools import permutations, product

import numpy as np
from PIL import ImageColor

from data.common.render_primitives import DOCTRINE_SUPERSAMPLE
from data.spatial_3d.core.iso_geometry import fit_projection, render_faces, voxel_faces
from evaluation.cross_task.drawing import (
    PALETTE,
    Canvas,
    cell_slots,
)


def orientations(piece):
    out = set()
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            mat = np.zeros((3, 3), int)
            for i in range(3):
                mat[i, perm[i]] = signs[i]
            if round(np.linalg.det(mat)) != 1:
                continue
            p = np.array(piece) @ mat.T
            p -= p.min(axis=0)
            out.add(tuple(sorted(map(tuple, p.tolist()))))
    return out


def solve(s):
    choices = []
    for piece in s["pieces"]:
        placements = set()
        for p in orientations(piece):
            maximum = [max(v[i] for v in p) for i in range(3)]
            for off in product(*(range(n - m) for n, m in zip((3, 2, 2), maximum))):
                placements.add(
                    frozenset(
                        (z + off[2]) * 6 + (y + off[1]) * 3 + x + off[0]
                        for x, y, z in p
                    )
                )
        choices.append(placements)
    out = []
    for a, b in product(choices[0], choices[1]):
        if a & b:
            continue
        missing = frozenset(range(12)) - a - b
        if missing in choices[2]:
            out.append(tuple(0 if i in a else 1 if i in b else 2 for i in range(12)))
    return sorted(set(out))


def sample(rng, key):
    for _ in range(1000):
        remaining = set(range(12))
        pieces = []
        for j in range(3):
            if not remaining:
                break
            chosen = {rng.choice(sorted(remaining))}
            while len(chosen) < 4:
                adjacent = [
                    i
                    for i in sorted(remaining - chosen)
                    if any(
                        abs(i % 3 - v % 3)
                        + abs(i // 3 % 2 - v // 3 % 2)
                        + abs(i // 6 - v // 6)
                        == 1
                        for v in chosen
                    )
                ]
                if not adjacent:
                    break
                chosen.add(rng.choice(adjacent))
            if len(chosen) < 4:
                break
            pieces.append([[i % 3, i // 3 % 2, i // 6] for i in sorted(chosen)])
            remaining -= chosen
        if len(pieces) == 3 and all(
            len({p[2] for p in piece}) == 2 for piece in pieces
        ):
            s = {
                "goal": key,
                "pieces": [(np.array(p) - np.min(p, axis=0)).tolist() for p in pieces],
            }
            if len(solve(s)) >= 2:
                return s
    raise ValueError("no spatial assembly")


def scene(s):
    c = Canvas()
    for j, piece in enumerate(s["pieces"]):
        p = np.array(piece)
        p -= p.min(axis=0)
        cells = list(map(tuple, p.tolist()))
        faces = voxel_faces(
            cells,
            default_color=ImageColor.getrgb(PALETTE[j]),
            edge_color=(255, 255, 255),
            edge_width=1.3,
        )
        projection = fit_projection(
            [v for face in faces for v in face.points],
            box=(22 + j * 166, 72, 153 + j * 166, 205),
            max_scale=42,
        )
        render_faces(c.surface, faces, projection, DOCTRINE_SUPERSAMPLE)
        for z in range(2):
            for y in range(2):
                for x in range(3):
                    xx = 33 + j * 166 + z * 65 + x * 15
                    yy = 229 + y * 15
                    c.d.rectangle(
                        (xx, yy, xx + 13, yy + 13),
                        fill=PALETTE[j] if [x, y, z] in p.tolist() else "#edf2f7",
                        outline="#b5c5d0",
                    )
            c.text((52 + j * 166 + z * 65, 267), f"z={z}", 10)
    slots = []
    for z in range(2):
        c.text((133 + z * 245, 315), f"z={z}", 14)
        xy = c.grid(2, 3, 43 + z * 245, 343, 59)
        slots += cell_slots(xy, 59, (0, 1, 2), tuple(PALETTE[:3]))
    return c, slots
