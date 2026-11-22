from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import random
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from data.common.grid_coordinates import cell_to_xy, format_xy_position


Grid = List[List[int]]
State = Dict[str, Any]
Params = Dict[str, Any]
Coord = Tuple[int, int]

DIRS: Dict[str, Coord] = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
DIR_ORDER = ("N", "E", "S", "W")
LEFT = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT = {value: key for key, value in LEFT.items()}
REVERSE = {"N": "S", "S": "N", "E": "W", "W": "E"}
VELOCITY_BITS = {"N": 1, "E": 2, "S": 4, "W": 8}


@dataclass(frozen=True)
class BuiltCase:
    task: str
    params: Params
    initial_state: State
    trace: List[State]
    answer: Dict[str, Any]
    evidence: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "params": deepcopy(self.params),
            "initial_state": deepcopy(self.initial_state),
            "trace": deepcopy(self.trace),
            "answer": deepcopy(self.answer),
            "evidence": deepcopy(self.evidence),
        }


def _grid(rows: int, cols: int, value: int = 0) -> Grid:
    return [[value for _ in range(cols)] for _ in range(rows)]


def _copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _shape(grid: Grid) -> Tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def _count_differences(a: Sequence[Any], b: Sequence[Any]) -> int:
    if a and isinstance(a[0], (list, tuple)):
        return sum(
            left != right
            for row_a, row_b in zip(a, b)
            for left, right in zip(row_a, row_b)
        )
    return sum(left != right for left, right in zip(a, b))


def _bit_count(value: int) -> int:
    return bin(int(value)).count("1")


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _inside(row: int, col: int, rows: int, cols: int) -> bool:
    return 0 <= row < rows and 0 <= col < cols


def _lookup(grid: Grid, row: int, col: int, boundary: str, outside: int = 0) -> int:
    rows, cols = _shape(grid)
    if boundary == "wrap":
        return grid[row % rows][col % cols]
    if _inside(row, col, rows, cols):
        return grid[row][col]
    return outside


def _neighbour_offsets(kind: str) -> Tuple[Coord, ...]:
    if kind == "orthogonal":
        return tuple(DIRS.values())
    if kind == "moore":
        return tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
    raise ValueError(f"unknown neighbourhood: {kind}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# 1. Sample-specific one-dimensional rule table


def _step_rule_table(state: State, params: Params) -> State:
    row = state["row"]
    width = len(row)
    rule = params["rule"]
    boundary = params["boundary"]
    successor = []
    for index in range(width):
        left = row[(index - 1) % width] if boundary == "wrap" else (row[index - 1] if index else 0)
        centre = row[index]
        right = row[(index + 1) % width] if boundary == "wrap" else (row[index + 1] if index + 1 < width else 0)
        successor.append(rule[(left << 2) | (centre << 1) | right])
    return {"row": successor, "generation": state["generation"] + 1}


def _replay_rule_table(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_rule_table(trace[-1], params))
    return trace


def _sample_rule_table(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(800):
        width = rng.randint(9, 12)
        ones = rng.randint(2, 6)
        one_indices = set(rng.sample(range(8), ones))
        rule = [int(index in one_indices) for index in range(8)]
        row = [int(rng.random() < rng.uniform(0.32, 0.68)) for _ in range(width)]
        if not (0 < sum(row) < width):
            continue
        params: Params = {
            "width": width,
            "rule": rule,
            "boundary": rng.choice(("fixed_zero", "wrap")),
            "steps": rng.randint(2, 3),
        }
        initial = {"row": row, "generation": 0}
        trace = _replay_rule_table(params, initial)
        alternate = dict(params)
        alternate["boundary"] = "wrap" if params["boundary"] == "fixed_zero" else "fixed_zero"
        alt_trace = _replay_rule_table(alternate, initial)
        if trace[-1]["row"] != alt_trace[-1]["row"] and _count_differences(row, trace[-1]["row"]) >= 2:
            return params, initial
    raise RuntimeError("could not sample a load-bearing one-dimensional rule table")


# ---------------------------------------------------------------------------
# 2. Sample-specific Life-like rule


def _step_lifelike(state: State, params: Params) -> State:
    old = state["grid"]
    rows, cols = _shape(old)
    new = _grid(rows, cols)
    offsets = _neighbour_offsets("moore")
    birth = set(params["birth"])
    survive = set(params["survive"])
    for row in range(rows):
        for col in range(cols):
            count = sum(
                _lookup(old, row + dr, col + dc, params["boundary"], 0)
                for dr, dc in offsets
            )
            new[row][col] = int(count in (survive if old[row][col] else birth))
    return {"grid": new, "generation": state["generation"] + 1}


def _replay_lifelike(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_lifelike(trace[-1], params))
    return trace


def _sample_lifelike(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(1200):
        size = rng.randint(5, 7)
        birth = sorted(rng.sample(range(1, 9), rng.randint(1, 3)))
        survive = sorted(rng.sample(range(9), rng.randint(2, 4)))
        density = rng.uniform(0.3, 0.62)
        old = [[int(rng.random() < density) for _ in range(size)] for _ in range(size)]
        params: Params = {
            "birth": birth,
            "survive": survive,
            "boundary": rng.choice(("fixed_zero", "wrap")),
            "steps": rng.randint(2, 3),
        }
        initial = {"grid": old, "generation": 0}
        trace = _replay_lifelike(params, initial)
        alternate = dict(params)
        alternate["boundary"] = "wrap" if params["boundary"] == "fixed_zero" else "fixed_zero"
        alt_trace = _replay_lifelike(alternate, initial)
        changes = sum(_count_differences(a["grid"], b["grid"]) for a, b in zip(trace, trace[1:]))
        if changes >= 6 and trace[-1]["grid"] != alt_trace[-1]["grid"]:
            return params, initial
    raise RuntimeError("could not sample a load-bearing Life-like rule")


# ---------------------------------------------------------------------------
# 3. Cyclic categorical threshold wave


def _step_cyclic(state: State, params: Params) -> State:
    old = state["grid"]
    rows, cols = _shape(old)
    new = _copy_grid(old)
    colours = params["colours"]
    offsets = _neighbour_offsets(params["neighbourhood"])
    for row in range(rows):
        for col in range(cols):
            successor = (old[row][col] + 1) % colours
            count = sum(
                _lookup(old, row + dr, col + dc, params["boundary"], -1) == successor
                for dr, dc in offsets
            )
            if count >= params["threshold"]:
                new[row][col] = successor
    return {"grid": new, "generation": state["generation"] + 1}


def _replay_cyclic(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_cyclic(trace[-1], params))
    return trace


def _sample_cyclic(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(1500):
        size = rng.randint(5, 7)
        colours = rng.randint(3, 5)
        neighbourhood = rng.choice(("orthogonal", "moore"))
        max_threshold = 2 if neighbourhood == "orthogonal" else 3
        threshold = rng.randint(1, max_threshold)
        old = [[rng.randrange(colours) for _ in range(size)] for _ in range(size)]
        params: Params = {
            "colours": colours,
            "neighbourhood": neighbourhood,
            "threshold": threshold,
            "boundary": rng.choice(("fixed", "wrap")),
            "steps": rng.randint(2, 3),
        }
        initial = {"grid": old, "generation": 0}
        trace = _replay_cyclic(params, initial)
        alt_threshold = dict(params)
        alt_threshold["threshold"] = 1 + (threshold % max_threshold)
        alt_neighbourhood = dict(params)
        alt_neighbourhood["neighbourhood"] = "moore" if neighbourhood == "orthogonal" else "orthogonal"
        changes = sum(_count_differences(a["grid"], b["grid"]) for a, b in zip(trace, trace[1:]))
        if (
            changes >= 6
            and trace[-1]["grid"] != _replay_cyclic(alt_threshold, initial)[-1]["grid"]
            and trace[-1]["grid"] != _replay_cyclic(alt_neighbourhood, initial)[-1]["grid"]
        ):
            return params, initial
    raise RuntimeError("could not sample a load-bearing cyclic threshold lattice")


# ---------------------------------------------------------------------------
# 4. Excitable signal network


def _step_excitable(state: State, params: Params) -> State:
    old = state["grid"]
    rows, cols = _shape(old)
    new = _copy_grid(old)
    for row in range(rows):
        for col in range(cols):
            value = old[row][col]
            if value == 2:  # head
                new[row][col] = 3
            elif value == 3:  # refractory tail
                new[row][col] = 1
            elif value == 1:  # conductor
                heads = sum(
                    _lookup(old, row + dr, col + dc, "fixed", 0) == 2
                    for dr, dc in DIRS.values()
                )
                new[row][col] = 2 if heads == 1 else 1
            # empty and blocker remain unchanged
    return {"grid": new, "generation": state["generation"] + 1}


def _replay_excitable(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_excitable(trace[-1], params))
    return trace


def _excitable_step_events(old: Grid, new: Grid) -> Tuple[int, int, int]:
    """Count ignitions, single-head branch fanouts, and collision retentions."""
    rows, cols = _shape(old)
    ignitions = collision_retentions = 0
    fanout: Dict[Coord, int] = {}
    for row in range(rows):
        for col in range(cols):
            if old[row][col] != 1:
                continue
            heads = [
                (row + dr, col + dc)
                for dr, dc in DIRS.values()
                if _lookup(old, row + dr, col + dc, "fixed", 0) == 2
            ]
            if len(heads) == 1 and new[row][col] == 2:
                ignitions += 1
                fanout[heads[0]] = fanout.get(heads[0], 0) + 1
            elif len(heads) >= 2 and new[row][col] == 1:
                collision_retentions += 1
    branches = sum(count >= 2 for count in fanout.values())
    return ignitions, branches, collision_retentions


def _sample_excitable(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(6000):
        size = rng.choice((9, 10))
        grid = _grid(size, size)
        conductor_density = rng.uniform(0.30, 0.46)
        for row in range(size):
            for col in range(size):
                roll = rng.random()
                if roll < conductor_density:
                    grid[row][col] = 1
                elif roll < conductor_density + 0.055:
                    grid[row][col] = 4
        # Independently placed travelling signals: each head leaves a
        # refractory tail behind it so its direction of travel is visible.
        head_count = rng.randint(2, 3)
        placed_heads: List[Coord] = []
        for _head in range(head_count):
            for _attempt in range(60):
                row, col = rng.randrange(1, size - 1), rng.randrange(1, size - 1)
                if grid[row][col] != 1:
                    continue
                direction = rng.choice(DIR_ORDER)
                dr, dc = DIRS[direction]
                tail_row, tail_col = row - dr, col - dc
                if not _inside(tail_row, tail_col, size, size):
                    continue
                if any(abs(row - hr) + abs(col - hc) <= 2 for hr, hc in placed_heads):
                    continue
                grid[row][col] = 2
                grid[tail_row][tail_col] = 3
                placed_heads.append((row, col))
                break
        if len(placed_heads) < 2:
            continue
        params: Params = {"steps": rng.choice((2, 3)), "head_rule": "exactly_one_orthogonal"}
        initial: State = {"grid": grid, "generation": 0}
        trace = _replay_excitable(params, initial)
        branch_total = collision_total = ignition_total = 0
        for old, new in zip(trace, trace[1:]):
            ignitions, branches, collisions = _excitable_step_events(old["grid"], new["grid"])
            ignition_total += ignitions
            branch_total += branches
            collision_total += collisions
        if ignition_total < 3 or branch_total < 1 or collision_total < 1:
            continue
        # Every blocker must be inert scenery only in appearance: at least one
        # blocker must actually stop the wave, i.e. replacing it with a
        # conductor changes the terminal grid.
        blockers = [
            (row, col)
            for row in range(size)
            for col in range(size)
            if grid[row][col] == 4
        ]
        if not blockers:
            continue
        load_bearing_blocker = False
        for row, col in blockers:
            variant = _copy_grid(grid)
            variant[row][col] = 1
            alt = _replay_excitable(params, {"grid": variant, "generation": 0})
            if alt[-1]["grid"] != trace[-1]["grid"]:
                load_bearing_blocker = True
                break
        if not load_bearing_blocker:
            continue
        return params, initial
    raise RuntimeError("could not sample an excitable network with branch, collision, and blocker load")


# ---------------------------------------------------------------------------
# 5. Alternating disjoint-block rotation


def _active_block_starts(size: int, phase: int) -> List[Coord]:
    offset = phase % 2
    return [(row, col) for row in range(offset, size, 2) for col in range(offset, size, 2)]


def _step_block_rotation(state: State, params: Params) -> State:
    old = state["grid"]
    size = len(old)
    new = _copy_grid(old)
    clockwise = params["direction"] == "clockwise"
    for row, col in _active_block_starts(size, state["phase"]):
        r0, r1 = row % size, (row + 1) % size
        c0, c1 = col % size, (col + 1) % size
        a, b, c, d = old[r0][c0], old[r0][c1], old[r1][c0], old[r1][c1]
        if clockwise:
            new[r0][c0], new[r0][c1], new[r1][c0], new[r1][c1] = c, a, d, b
        else:
            new[r0][c0], new[r0][c1], new[r1][c0], new[r1][c1] = b, d, a, c
    return {"grid": new, "phase": 1 - state["phase"], "generation": state["generation"] + 1}


def _replay_block_rotation(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_block_rotation(trace[-1], params))
    return trace


def _sample_block_rotation(rng: random.Random) -> Tuple[Params, State]:
    size = rng.choice((6, 8))
    phase = rng.randrange(2)
    positions = {(rng.randrange(size), rng.randrange(size)) for _ in range(size * 2)}
    while len(positions) < rng.randint(9, min(16, size * size - 1)):
        positions.add((rng.randrange(size), rng.randrange(size)))
    # Phase-one blocks wrap. Ensure a token actually occupies that partition's edge block.
    edge_candidates = [(size - 1, rng.randrange(size)), (rng.randrange(size), size - 1)]
    positions.add(rng.choice(edge_candidates))
    positions.add((0, 0))
    grid = _grid(size, size, -1)
    for token, (row, col) in enumerate(sorted(positions)):
        grid[row][col] = token
    params: Params = {"direction": rng.choice(("clockwise", "counterclockwise")), "steps": 2}
    return params, {"grid": grid, "phase": phase, "generation": 0}


# ---------------------------------------------------------------------------
# 6. Wrapped XOR parity lattice


def _step_xor(state: State, params: Params) -> State:
    old = state["grid"]
    rows, cols = _shape(old)
    new = _grid(rows, cols)
    for row in range(rows):
        for col in range(cols):
            bits = [_lookup(old, row + dr, col + dc, "wrap", 0) for dr, dc in DIRS.values()]
            if params["include_self"]:
                bits.append(old[row][col])
            new[row][col] = sum(bits) % 2
    return {"grid": new, "generation": state["generation"] + 1}


def _replay_xor(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_xor(trace[-1], params))
    return trace


def _sample_xor(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(500):
        size = rng.randint(5, 7)
        density = rng.uniform(0.25, 0.65)
        grid = [[int(rng.random() < density) for _ in range(size)] for _ in range(size)]
        params: Params = {"include_self": bool(rng.randrange(2)), "steps": rng.randint(2, 3)}
        initial = {"grid": grid, "generation": 0}
        trace = _replay_xor(params, initial)
        fixed_first = _grid(size, size)
        for row in range(size):
            for col in range(size):
                bits = [_lookup(grid, row + dr, col + dc, "fixed", 0) for dr, dc in DIRS.values()]
                if params["include_self"]:
                    bits.append(grid[row][col])
                fixed_first[row][col] = sum(bits) % 2
        edge_diff = any(
            trace[1]["grid"][row][col] != fixed_first[row][col]
            for row in range(size)
            for col in range(size)
            if row in (0, size - 1) or col in (0, size - 1)
        )
        if edge_diff and _count_differences(grid, trace[-1]["grid"]) >= 4:
            return params, initial
    raise RuntimeError("could not sample a wrapped XOR lattice with edge load")


# ---------------------------------------------------------------------------
# 7. Open-boundary sandpile avalanche


def _topple_wave(grid: Grid) -> Tuple[Grid, int, int]:
    rows, cols = _shape(grid)
    unstable = [(row, col) for row in range(rows) for col in range(cols) if grid[row][col] >= 4]
    new = _copy_grid(grid)
    lost = 0
    for row, col in unstable:
        new[row][col] -= 4
        for dr, dc in DIRS.values():
            nr, nc = row + dr, col + dc
            if _inside(nr, nc, rows, cols):
                new[nr][nc] += 1
            else:
                lost += 1
    return new, lost, len(unstable)


def _replay_sandpile(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    grid = _copy_grid(initial["grid"])
    add_row, add_col = params["add"]
    grid[add_row][add_col] += 1
    trace.append({"grid": grid, "lost": 0, "phase": "added", "wave": 0})
    cumulative_lost = 0
    for wave in range(1, 101):
        if all(value < 4 for row in grid for value in row):
            break
        grid, lost, toppled = _topple_wave(grid)
        cumulative_lost += lost
        trace.append({
            "grid": grid,
            "lost": cumulative_lost,
            "phase": "toppled",
            "wave": wave,
            "toppled": toppled,
        })
    else:
        raise RuntimeError("sandpile failed to stabilize")
    return trace


def _sample_sandpile(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(1000):
        size = rng.choice((5, 6))
        grid = [[rng.randrange(4) for _ in range(size)] for _ in range(size)]
        side = rng.randrange(4)
        offset = rng.randrange(1, size - 1)
        if side == 0:
            add = (0, offset)
        elif side == 1:
            add = (offset, size - 1)
        elif side == 2:
            add = (size - 1, offset)
        else:
            add = (offset, 0)
        row, col = add
        grid[row][col] = 3
        # Bias a short connected ridge toward the interior without fixing the answer.
        inward = DIRS[("S", "W", "N", "E")[side]]
        for distance in range(1, rng.randint(2, 4)):
            rr, cc = row + inward[0] * distance, col + inward[1] * distance
            if _inside(rr, cc, size, size):
                grid[rr][cc] = 3
                if rng.random() < 0.65:
                    sr, sc = rr + inward[1], cc + inward[0]
                    if _inside(sr, sc, size, size):
                        grid[sr][sc] = 3
        initial: State = {"grid": grid, "lost": 0, "phase": "stable", "wave": 0}
        params: Params = {"add": list(add), "boundary": "open", "toppling": "synchronous_until_stable"}
        trace = _replay_sandpile(params, initial)
        waves = len(trace) - 2
        initial_mass = sum(map(sum, grid)) + 1
        final_mass = sum(map(sum, trace[-1]["grid"])) + trace[-1]["lost"]
        if 2 <= waves <= 8 and trace[-1]["lost"] > 0 and initial_mass == final_mass:
            return params, initial
    raise RuntimeError("could not sample a bounded multi-wave sandpile avalanche")


# ---------------------------------------------------------------------------
# 8. Directional lattice-gas collision and reflection


def _step_lattice_gas(state: State, params: Params) -> State:
    old = state["grid"]
    walls = params["walls"]
    rows, cols = _shape(old)
    collided = _copy_grid(old)
    for row in range(rows):
        for col in range(cols):
            if collided[row][col] == (VELOCITY_BITS["E"] | VELOCITY_BITS["W"]):
                collided[row][col] = VELOCITY_BITS["N"] | VELOCITY_BITS["S"]
            elif collided[row][col] == (VELOCITY_BITS["N"] | VELOCITY_BITS["S"]):
                collided[row][col] = VELOCITY_BITS["E"] | VELOCITY_BITS["W"]
    streamed = _grid(rows, cols)
    reflections = 0
    for row in range(rows):
        for col in range(cols):
            if walls[row][col]:
                continue
            for direction, bit in VELOCITY_BITS.items():
                if not collided[row][col] & bit:
                    continue
                dr, dc = DIRS[direction]
                nr, nc = row + dr, col + dc
                if not _inside(nr, nc, rows, cols) or walls[nr][nc]:
                    streamed[row][col] |= VELOCITY_BITS[REVERSE[direction]]
                    reflections += 1
                else:
                    streamed[nr][nc] |= bit
    return {
        "grid": streamed,
        "generation": state["generation"] + 1,
        "last_reflections": reflections,
    }


def _replay_lattice_gas(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_lattice_gas(trace[-1], params))
    return trace


def _sample_lattice_gas(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(500):
        size = rng.choice((7, 8, 9))
        grid = _grid(size, size)
        walls = _grid(size, size)
        centre = (rng.randint(2, size - 3), rng.randint(2, size - 3))
        collision_axis = rng.choice(("horizontal", "vertical"))
        grid[centre[0]][centre[1]] = (
            VELOCITY_BITS["E"] | VELOCITY_BITS["W"]
            if collision_axis == "horizontal"
            else VELOCITY_BITS["N"] | VELOCITY_BITS["S"]
        )
        # A independently sampled particle immediately faces a wall.
        direction = rng.choice(DIR_ORDER)
        dr, dc = DIRS[direction]
        candidates = [
            (row, col)
            for row in range(1, size - 1)
            for col in range(1, size - 1)
            if (row, col) != centre and (row + dr, col + dc) != centre
        ]
        source = rng.choice(candidates)
        wall = (source[0] + dr, source[1] + dc)
        grid[source[0]][source[1]] = VELOCITY_BITS[direction]
        walls[wall[0]][wall[1]] = 1
        # Sparse, randomly placed channels make position and direction load-bearing.
        for _extra in range(rng.randint(1, 4)):
            for _attempt in range(30):
                row, col = rng.randrange(size), rng.randrange(size)
                extra_direction = rng.choice(DIR_ORDER)
                if walls[row][col] or grid[row][col]:
                    continue
                grid[row][col] = VELOCITY_BITS[extra_direction]
                break
        params: Params = {"walls": walls, "steps": rng.choice((2, 3)), "boundary": "reflect"}
        initial: State = {"grid": grid, "generation": 0, "last_reflections": 0}
        trace = _replay_lattice_gas(params, initial)
        counts = [sum(_bit_count(value) for row in item["grid"] for value in row) for item in trace]
        if len(set(counts)) == 1 and trace[1]["last_reflections"] >= 1:
            return params, initial
    raise RuntimeError("could not sample a collision/reflection lattice gas without channel aliasing")


# ---------------------------------------------------------------------------
# 9. Strict local majority with immutable vacancies


def _step_majority(state: State, params: Params) -> State:
    old = state["grid"]
    rows, cols = _shape(old)
    new = _copy_grid(old)
    offsets = _neighbour_offsets(params["neighbourhood"])
    for row in range(rows):
        for col in range(cols):
            if old[row][col] == 0:
                continue
            counts = {1: 0, 2: 0}
            for dr, dc in offsets:
                value = _lookup(old, row + dr, col + dc, params["boundary"], 0)
                if value in counts:
                    counts[value] += 1
            if counts[1] > counts[2]:
                new[row][col] = 1
            elif counts[2] > counts[1]:
                new[row][col] = 2
            # A tie, including zero versus zero, retains the old opinion.
    return {"grid": new, "generation": state["generation"] + 1}


def _replay_majority(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_majority(trace[-1], params))
    return trace


def _majority_transition_counts(old: Grid, params: Params) -> Tuple[int, int]:
    new = _step_majority({"grid": old, "generation": 0}, params)["grid"]
    flips = 0
    ties = 0
    rows, cols = _shape(old)
    offsets = _neighbour_offsets(params["neighbourhood"])
    for row in range(rows):
        for col in range(cols):
            if old[row][col] == 0:
                continue
            one = two = 0
            for dr, dc in offsets:
                value = _lookup(old, row + dr, col + dc, params["boundary"], 0)
                one += value == 1
                two += value == 2
            flips += new[row][col] != old[row][col]
            ties += one == two
    return flips, ties


def _sample_majority(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(1500):
        size = rng.randint(5, 7)
        grid = [
            [rng.choices((0, 1, 2), weights=(0.12, 0.44, 0.44), k=1)[0] for _ in range(size)]
            for _ in range(size)
        ]
        if not any(value == 0 for row in grid for value in row):
            grid[rng.randrange(size)][rng.randrange(size)] = 0
        params: Params = {
            "neighbourhood": rng.choice(("orthogonal", "moore")),
            "boundary": rng.choice(("fixed", "wrap")),
            "steps": rng.choice((2, 3)),
        }
        flips, ties = _majority_transition_counts(grid, params)
        alternate = dict(params)
        alternate["boundary"] = "wrap" if params["boundary"] == "fixed" else "fixed"
        trace = _replay_majority(params, {"grid": grid, "generation": 0})
        alt_trace = _replay_majority(alternate, {"grid": grid, "generation": 0})
        if flips >= 2 and ties >= 1 and trace[-1]["grid"] != alt_trace[-1]["grid"]:
            return params, {"grid": grid, "generation": 0}
    raise RuntimeError("could not sample majority dynamics with flip, tie, vacancy, and edge load")


# ---------------------------------------------------------------------------
# 10. Expiring fire front with water and rock barriers


def _step_forest(state: State, params: Params) -> State:
    old = state["grid"]
    rows, cols = _shape(old)
    new = _copy_grid(old)
    for row in range(rows):
        for col in range(cols):
            if old[row][col] == 2:  # flame expires
                new[row][col] = 3
            elif old[row][col] == 1:  # tree ignites from old flame only
                if any(
                    _lookup(old, row + dr, col + dc, "fixed", 0) == 2
                    for dr, dc in DIRS.values()
                ):
                    new[row][col] = 2
            # empty, ash, water, and rock remain unchanged
    return {"grid": new, "generation": state["generation"] + 1}


def _replay_forest(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_forest(trace[-1], params))
    return trace


def _forest_barrier_route(
    grid: Grid,
    trace: List[State],
    params: Params,
    code: int,
) -> List[List[int]] | None:
    """Find one barrier of ``code`` that provably blocks the front.

    Returns ``[barrier, behind]`` where the behind tree survives the actual
    rollout but burns when the barrier is replaced with a tree.
    """
    rows, cols = _shape(grid)
    ever_flame = {
        (row, col)
        for item in trace
        for row in range(rows)
        for col in range(cols)
        if item["grid"][row][col] == 2
    }
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] != code:
                continue
            touches_front = any(
                (row + dr, col + dc) in ever_flame for dr, dc in DIRS.values()
            )
            if not touches_front:
                continue
            variant = _copy_grid(grid)
            variant[row][col] = 1
            alt = _replay_forest(params, {"grid": variant, "generation": 0})
            for dr, dc in DIRS.values():
                tr, tc = row + dr, col + dc
                if not _inside(tr, tc, rows, cols) or grid[tr][tc] != 1:
                    continue
                survives = all(item["grid"][tr][tc] == 1 for item in trace)
                burns_without = alt[-1]["grid"][tr][tc] != 1
                if survives and burns_without:
                    return [[row, col], [tr, tc]]
    return None


def _sample_forest(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(6000):
        size = rng.choice((8, 9))
        grid = _grid(size, size)
        for row in range(size):
            for col in range(size):
                roll = rng.random()
                if roll < 0.46:
                    grid[row][col] = 1
                elif roll < 0.53:
                    grid[row][col] = 4
                elif roll < 0.60:
                    grid[row][col] = 5
        flame_count = rng.randint(1, 2)
        seeds: List[Coord] = []
        for _flame in range(flame_count):
            for _attempt in range(40):
                row, col = rng.randrange(size), rng.randrange(size)
                if grid[row][col] == 1 and all(
                    abs(row - sr) + abs(col - sc) > 3 for sr, sc in seeds
                ):
                    grid[row][col] = 2
                    seeds.append((row, col))
                    break
        if not seeds:
            continue
        params: Params = {"steps": rng.choice((2, 3))}
        initial: State = {"grid": grid, "generation": 0}
        trace = _replay_forest(params, initial)
        ignitions_per_step = [
            sum(
                a == 1 and b == 2
                for row_a, row_b in zip(old["grid"], new["grid"])
                for a, b in zip(row_a, row_b)
            )
            for old, new in zip(trace, trace[1:])
        ]
        # The front must stay alive through the whole horizon.
        if not all(count >= 1 for count in ignitions_per_step):
            continue
        if sum(ignitions_per_step) < 4:
            continue
        water_route = _forest_barrier_route(grid, trace, params, 4)
        rock_route = _forest_barrier_route(grid, trace, params, 5)
        if water_route is None or rock_route is None:
            continue
        params["barrier_routes"] = {"water": water_route, "rock": rock_route}
        return params, initial
    raise RuntimeError("could not sample a forest fire with a live front and both blocking barriers")


# ---------------------------------------------------------------------------
# 11. Coupled predator/prey transition cycle


def _step_predator_prey(state: State, params: Params) -> State:
    old = state["grid"]
    rows, cols = _shape(old)
    new = _copy_grid(old)
    offsets = _neighbour_offsets("moore")
    for row in range(rows):
        for col in range(cols):
            value = old[row][col]
            neighbours = [_lookup(old, row + dr, col + dc, "fixed", 0) for dr, dc in offsets]
            prey = sum(item in (1, 3) for item in neighbours)
            predators = sum(item == 2 for item in neighbours)
            if value == 0 and prey >= 2 and predators == 0:
                new[row][col] = 1
            elif value == 1 and predators >= 2:
                new[row][col] = 2
            elif value == 2 and prey == 0:
                new[row][col] = 0
            # Sheltered prey and rock are immutable; all other states retain.
    return {"grid": new, "generation": state["generation"] + 1}


def _replay_predator_prey(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_predator_prey(trace[-1], params))
    return trace


def _sample_predator_prey(rng: random.Random) -> Tuple[Params, State]:
    offsets = _neighbour_offsets("moore")
    for _ in range(6000):
        size = rng.choice((9, 10))
        grid = _grid(size, size)
        for row in range(size):
            for col in range(size):
                roll = rng.random()
                if roll < 0.26:
                    grid[row][col] = 1
                elif roll < 0.37:
                    grid[row][col] = 2
                elif roll < 0.42:
                    grid[row][col] = 3
                elif roll < 0.47:
                    grid[row][col] = 4
        # Two ticks make the intermediate population a genuine causal state.
        params: Params = {"steps": 2, "neighbourhood": "moore", "boundary": "fixed"}
        initial: State = {"grid": grid, "generation": 0}
        trace = _replay_predator_prey(params, initial)
        pairs = [
            (a, b)
            for row_a, row_b in zip(trace[0]["grid"], trace[1]["grid"])
            for a, b in zip(row_a, row_b)
        ]
        if pairs.count((0, 1)) < 1 or pairs.count((1, 2)) < 1 or pairs.count((2, 0)) < 1:
            continue
        second = [
            (a, b)
            for row_a, row_b in zip(trace[1]["grid"], trace[2]["grid"])
            for a, b in zip(row_a, row_b)
        ]
        if sum(a != b for a, b in second) < 1:
            continue
        # Shelter must be load-bearing: at least one sheltered prey has enough
        # predator neighbours that an ordinary prey there would be predated.
        shelter_matters = False
        rock_matters = False
        for row in range(size):
            for col in range(size):
                neighbours = [
                    _lookup(grid, row + dr, col + dc, "fixed", 0) for dr, dc in offsets
                ]
                predators = neighbours.count(2)
                if grid[row][col] == 3 and predators >= 2:
                    shelter_matters = True
                if grid[row][col] == 4 and predators >= 1:
                    rock_matters = True
        if shelter_matters and rock_matters:
            return params, initial
    raise RuntimeError("could not sample predator/prey dynamics with all four load-bearing transitions")


# ---------------------------------------------------------------------------
# 12. Oriented read-turn-write-move agent


def _step_turn_write(state: State, params: Params) -> State:
    board = _copy_grid(state["grid"])
    row, col = state["head"]
    direction = state["direction"]
    read = board[row][col]
    action = params["table"][str(read)]
    direction = LEFT[direction] if action["turn"] == "L" else RIGHT[direction]
    board[row][col] = action["write"]
    dr, dc = DIRS[direction]
    raw_row, raw_col = row + dr, col + dc
    rows, cols = _shape(board)
    new_row, new_col = raw_row % rows, raw_col % cols
    return {
        "grid": board,
        "head": [new_row, new_col],
        "direction": direction,
        "tick": state["tick"] + 1,
        "last": {
            "from": [row, col], "read": read, "turn": action["turn"],
            "write": action["write"], "wrapped": not _inside(raw_row, raw_col, rows, cols),
        },
    }


def _replay_turn_write(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_turn_write(trace[-1], params))
    return trace


def _sample_turn_write(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(2500):
        size = rng.choice((5, 6))
        board = [[int(rng.random() < rng.uniform(0.28, 0.62)) for _ in range(size)] for _ in range(size)]
        table = {
            "0": {"turn": rng.choice(("L", "R")), "write": rng.randrange(2)},
            "1": {"turn": rng.choice(("L", "R")), "write": rng.randrange(2)},
        }
        # Reject a table that cannot create useful rewriting supervision.
        if table["0"]["write"] == 0 and table["1"]["write"] == 1:
            continue
        edge = rng.randrange(size)
        start = rng.choice(([0, edge], [size - 1, edge], [edge, 0], [edge, size - 1]))
        params: Params = {"table": table, "steps": rng.randint(5, 8), "boundary": "wrap"}
        initial: State = {"grid": board, "head": start, "direction": rng.choice(DIR_ORDER), "tick": 0, "last": None}
        trace = _replay_turn_write(params, initial)
        reads = {item["last"]["read"] for item in trace[1:]}
        rewrites = sum(item["last"]["read"] != item["last"]["write"] for item in trace[1:])
        wraps = sum(item["last"]["wrapped"] for item in trace[1:])
        if reads == {0, 1} and rewrites >= 1 and wraps >= 1:
            return params, initial
    raise RuntimeError("could not sample a turn/write trace that reads both bits, rewrites, and wraps")


# ---------------------------------------------------------------------------
# 13. Finite-control circular tape machine


def _step_tape_machine(state: State, params: Params) -> State:
    tape = state["tape"][:]
    head = state["head"]
    control = state["control"]
    read = tape[head]
    action = params["table"][f"{control}:{read}"]
    tape[head] = action["write"]
    raw_head = head + (-1 if action["move"] == "L" else 1)
    next_head = raw_head % len(tape)
    return {
        "tape": tape,
        "head": next_head,
        "control": action["next"],
        "tick": state["tick"] + 1,
        "last": {
            "from": head, "read": read, "write": action["write"],
            "old_control": control, "new_control": action["next"],
            "move": action["move"], "wrapped": not 0 <= raw_head < len(tape),
        },
    }


def _replay_tape_machine(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_tape_machine(trace[-1], params))
    return trace


def _sample_tape_machine(rng: random.Random) -> Tuple[Params, State]:
    controls = ("A", "B", "C")
    for _ in range(3500):
        length = rng.randint(8, 11)
        tape = [rng.randrange(2) for _ in range(length)]
        table: Dict[str, Any] = {}
        for control in controls:
            for bit in (0, 1):
                table[f"{control}:{bit}"] = {
                    "write": rng.randrange(2),
                    "next": rng.choice(controls),
                    "move": rng.choice(("L", "R")),
                }
        params: Params = {"table": table, "steps": rng.randint(6, 9), "boundary": "circular"}
        initial: State = {
            "tape": tape,
            "head": rng.choice((0, length - 1)),
            "control": rng.choice(controls),
            "tick": 0,
            "last": None,
        }
        trace = _replay_tape_machine(params, initial)
        rewrites = sum(item["last"]["read"] != item["last"]["write"] for item in trace[1:])
        changes = sum(item["last"]["old_control"] != item["last"]["new_control"] for item in trace[1:])
        wraps = sum(item["last"]["wrapped"] for item in trace[1:])
        reads = {item["last"]["read"] for item in trace[1:]}
        if rewrites >= 1 and changes >= 2 and wraps >= 1 and reads == {0, 1}:
            return params, initial
    raise RuntimeError("could not sample a load-bearing finite tape-machine trace")


# ---------------------------------------------------------------------------
# 14. Simultaneous agent proposal arbitration


def _agent_proposals(agents: Sequence[Mapping[str, Any]], size: int) -> Dict[str, Coord]:
    proposals: Dict[str, Coord] = {}
    for agent in agents:
        dr, dc = DIRS[agent["direction"]]
        proposals[agent["id"]] = ((agent["row"] + dr) % size, (agent["col"] + dc) % size)
    return proposals


def _resolve_agent_moves(agents: Sequence[Mapping[str, Any]], size: int, policy: str) -> Tuple[set[str], Dict[str, Coord], List[List[str]]]:
    proposals = _agent_proposals(agents, size)
    by_id = {agent["id"]: agent for agent in agents}
    old_occupants = {(agent["row"], agent["col"]): agent["id"] for agent in agents}
    blocked: set[str] = set()
    conflict_groups: List[List[str]] = []
    ids = sorted(by_id)
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            left_old = (by_id[left]["row"], by_id[left]["col"])
            right_old = (by_id[right]["row"], by_id[right]["col"])
            if proposals[left] == right_old and proposals[right] == left_old:
                blocked.update((left, right))
                conflict_groups.append([left, right])
    target_groups: Dict[Coord, List[str]] = {}
    for agent_id, target in proposals.items():
        if agent_id not in blocked:
            target_groups.setdefault(target, []).append(agent_id)
    movers: set[str] = set()
    for contenders in target_groups.values():
        contenders.sort()
        if len(contenders) == 1:
            movers.add(contenders[0])
        else:
            conflict_groups.append(contenders[:])
            if policy == "lowest_id_moves":
                movers.add(contenders[0])
    # A provisional mover cannot enter a cell whose old occupant ultimately waits.
    changed = True
    while changed:
        changed = False
        for agent_id in tuple(movers):
            occupant = old_occupants.get(proposals[agent_id])
            if occupant is not None and occupant not in movers:
                movers.remove(agent_id)
                changed = True
    return movers, proposals, conflict_groups


def _step_agent_conflict(state: State, params: Params) -> State:
    size = params["size"]
    movers, proposals, groups = _resolve_agent_moves(state["agents"], size, params["policy"])
    agents = []
    for old in state["agents"]:
        agent = dict(old)
        if agent["id"] in movers:
            agent["row"], agent["col"] = proposals[agent["id"]]
        agents.append(agent)
    return {
        "agents": agents,
        "tick": state["tick"] + 1,
        "last": {"moved": sorted(movers), "conflicts": groups, "proposals": {key: list(value) for key, value in proposals.items()}},
    }


def _replay_agent_conflict(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_agent_conflict(trace[-1], params))
    return trace


def _sample_agent_conflict(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(6000):
        size = rng.choice((6, 7))
        agent_count = rng.randint(3, 5)
        cells = [(row, col) for row in range(size) for col in range(size)]
        positions = rng.sample(cells, agent_count)
        agents = [
            {
                "id": chr(ord("A") + index),
                "row": position[0],
                "col": position[1],
                "direction": rng.choice(DIR_ORDER),
            }
            for index, position in enumerate(positions)
        ]
        params: Params = {
            "size": size,
            "policy": rng.choice(("all_wait", "lowest_id_moves")),
            "swap_rule": "both_wait",
            "steps": 1,
        }
        initial: State = {"agents": agents, "tick": 0, "last": None}
        event = _step_agent_conflict(initial, params)["last"]
        participants = {agent for group in event["conflicts"] for agent in group}
        unaffected = set(event["moved"]) - participants
        if event["conflicts"] and unaffected:
            return params, initial
    raise RuntimeError("could not sample an agent tick with a conflict and an unaffected mover")


# ---------------------------------------------------------------------------
# 15. Ordered resource pickup/drop and movement


def _step_resource_swarm(state: State, params: Params) -> State:
    piles = _copy_grid(state["piles"])
    size = len(piles)
    agents = [dict(agent) for agent in state["agents"]]
    actions: List[Dict[str, Any]] = []
    for agent in sorted(agents, key=lambda item: item["id"]):
        row, col = agent["row"], agent["col"]
        if agent["carrying"]:
            piles[row][col] += 1
            agent["carrying"] = False
            action = "drop"
        elif piles[row][col] > 0:
            piles[row][col] -= 1
            agent["carrying"] = True
            action = "pick"
        else:
            action = "none"
        dr, dc = DIRS[agent["direction"]]
        raw_row, raw_col = row + dr, col + dc
        agent["row"], agent["col"] = raw_row % size, raw_col % size
        actions.append({
            "id": agent["id"], "action": action, "at": [row, col],
            "to": [agent["row"], agent["col"]],
            "wrapped": not _inside(raw_row, raw_col, size, size),
        })
    return {"piles": piles, "agents": agents, "tick": state["tick"] + 1, "last": actions}


def _replay_resource_swarm(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_resource_swarm(trace[-1], params))
    return trace


def _sample_resource_swarm(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(6000):
        size = rng.choice((6, 7))
        piles = _grid(size, size)
        for _pile in range(rng.randint(3, 6)):
            row, col = rng.randrange(size), rng.randrange(size)
            piles[row][col] += rng.randint(1, 3)
        cells = [(row, col) for row in range(size) for col in range(size)]
        pick_start = rng.choice([cell for cell in cells if piles[cell[0]][cell[1]] > 0])
        drop_start = rng.choice([cell for cell in cells if cell != pick_start])
        agents = [
            {"id": "A", "row": pick_start[0], "col": pick_start[1], "direction": rng.choice(DIR_ORDER), "carrying": False},
            {"id": "B", "row": drop_start[0], "col": drop_start[1], "direction": rng.choice(DIR_ORDER), "carrying": True},
        ]
        params: Params = {"size": size, "steps": 2, "order": "ascending_id", "boundary": "wrap"}
        initial: State = {"piles": piles, "agents": agents, "tick": 0, "last": None}
        trace = _replay_resource_swarm(params, initial)
        events = [event for item in trace[1:] for event in item["last"]]
        picks = sum(event["action"] == "pick" for event in events)
        drops = sum(event["action"] == "drop" for event in events)
        # The two badges must stay on distinct cells in every committed state
        # so both identities remain readable without joint-occupancy glyphs.
        distinct = all(
            len({(agent["row"], agent["col"]) for agent in item["agents"]}) == len(item["agents"])
            for item in trace
        )
        if picks >= 1 and drops >= 1 and distinct:
            return params, initial
    raise RuntimeError("could not sample a swarm trace with pick, drop, and distinct badges")


# ---------------------------------------------------------------------------
# 16. Aggregate growth plus deterministic walker conflicts


def _adjacent_to_aggregate(row: int, col: int, aggregate: set[Coord], size: int) -> bool:
    return any(((row + dr) % size, (col + dc) % size) in aggregate for dr, dc in DIRS.values())


def _step_dla(state: State, params: Params) -> State:
    size = params["size"]
    aggregate = {tuple(coord) for coord in state["aggregate"]}
    walkers = [dict(item) for item in state["walkers"]]
    frozen = [item for item in walkers if _adjacent_to_aggregate(item["row"], item["col"], aggregate, size)]
    frozen_ids = {item["id"] for item in frozen}
    for item in frozen:
        aggregate.add((item["row"], item["col"]))
    mobile = [item for item in walkers if item["id"] not in frozen_ids]
    proposals: Dict[str, Coord] = {}
    for item in mobile:
        dr, dc = DIRS[item["direction"]]
        proposals[item["id"]] = ((item["row"] + dr) % size, (item["col"] + dc) % size)
    groups: Dict[Coord, List[str]] = {}
    for item_id, target in proposals.items():
        groups.setdefault(target, []).append(item_id)
    movers: set[str] = set()
    conflict_groups: List[List[str]] = []
    for target, contenders in groups.items():
        contenders.sort()
        if target in aggregate:
            conflict_groups.append(contenders[:])
        elif len(contenders) == 1:
            movers.add(contenders[0])
        else:
            movers.add(contenders[0])
            conflict_groups.append(contenders[:])
    next_walkers = []
    for item in mobile:
        if item["id"] in movers:
            item["row"], item["col"] = proposals[item["id"]]
        next_walkers.append(item)
    return {
        "aggregate": [list(coord) for coord in sorted(aggregate)],
        "walkers": next_walkers,
        "tick": state["tick"] + 1,
        "last": {"frozen": sorted(frozen_ids), "moved": sorted(movers), "conflicts": conflict_groups},
    }


def _replay_dla(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_dla(trace[-1], params))
    return trace


def _sample_dla(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(6000):
        size = rng.choice((8, 9, 10))
        seed_row, seed_col = rng.randrange(size), rng.randrange(size)
        aggregate: List[Coord] = [(seed_row, seed_col)]
        for _growth in range(rng.randint(1, 2)):
            row, col = rng.choice(aggregate)
            dr, dc = DIRS[rng.choice(DIR_ORDER)]
            candidate = ((row + dr) % size, (col + dc) % size)
            if candidate not in aggregate:
                aggregate.append(candidate)
        walker_count = rng.randint(4, 6)
        cells = [
            (row, col)
            for row in range(size)
            for col in range(size)
            if (row, col) not in aggregate
        ]
        positions = rng.sample(cells, walker_count)
        walkers = [
            {
                "id": chr(ord("A") + index),
                "row": position[0],
                "col": position[1],
                "direction": rng.choice(DIR_ORDER),
            }
            for index, position in enumerate(positions)
        ]
        params: Params = {"size": size, "steps": 1, "boundary": "wrap", "conflict_policy": "lowest_id_moves"}
        initial: State = {
            "aggregate": [list(coord) for coord in sorted(aggregate)],
            "walkers": walkers,
            "tick": 0,
            "last": None,
        }
        first = _step_dla(initial, params)
        event = first["last"]
        participants = {walker for group in event["conflicts"] for walker in group}
        unaffected = set(event["moved"]) - participants
        multi_conflict = any(len(group) >= 2 for group in event["conflicts"])
        if len(event["frozen"]) >= 1 and multi_conflict and unaffected:
            return params, initial
    raise RuntimeError("could not sample a DLA tick with freeze, conflict, and unaffected mover")


# ---------------------------------------------------------------------------
# 17. Hierarchical interface navigation


def _step_gui(state: State, params: Params) -> State:
    index = state["action_index"]
    if index >= len(params["actions"]):
        return deepcopy(state)
    action = params["actions"][index]
    stack = state["stack"][:]
    old_page = stack[-1]
    if action["kind"] == "BACK":
        if len(stack) > 1:
            stack.pop()
        outcome = "popped" if len(stack) < len(state["stack"]) else "root_retained"
    else:
        tiles = {tile["label"]: tile["target"] for tile in params["pages"][old_page]}
        label = action["label"]
        _require(label in tiles, f"tile {label} is not available on page {old_page}")
        stack.append(tiles[label])
        outcome = f"opened:{tiles[label]}"
    return {
        "page": stack[-1],
        "stack": stack,
        "action_index": index + 1,
        "cursor": action.get("label", "BACK"),
        "tick": state["tick"] + 1,
        "last": {"action": deepcopy(action), "from": old_page, "to": stack[-1], "outcome": outcome},
    }


def _replay_gui(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in params["actions"]:
        trace.append(_step_gui(trace[-1], params))
    return trace


def _sample_gui(rng: random.Random) -> Tuple[Params, State]:
    top = rng.sample(("ALPHA", "BETA", "GAMMA"), 3)
    pages: Dict[str, List[Dict[str, str]]] = {"ROOT": []}
    children_by_top: Dict[str, List[str]] = {}
    for top_name in top:
        children = [f"{top_name}-{suffix}" for suffix in rng.sample(("ONE", "TWO", "THREE"), 3)]
        children_by_top[top_name] = children
        pages["ROOT"].append({"label": top_name, "target": top_name})
        pages[top_name] = [{"label": child.split("-")[-1], "target": child} for child in children]
        for child in children:
            pages[child] = []
    first, sibling = top[0], top[1]
    child = children_by_top[first][rng.randrange(3)]
    child_label = child.split("-")[-1]
    actions = [
        {"kind": "CLICK", "label": first},
        {"kind": "CLICK", "label": child_label},
        {"kind": "BACK"},
        {"kind": "BACK"},
        {"kind": "CLICK", "label": sibling},
    ]
    params: Params = {"pages": pages, "actions": actions, "root": "ROOT"}
    initial: State = {"page": "ROOT", "stack": ["ROOT"], "action_index": 0, "cursor": None, "tick": 0, "last": None}
    return params, initial


# ---------------------------------------------------------------------------
# 18. Ordered heterogeneous function pipeline


def _apply_pipeline_op(value: int, operation: Mapping[str, Any]) -> int:
    kind = operation["kind"]
    if kind == "ADD":
        return value + operation["value"]
    if kind == "MUL":
        return value * operation["value"]
    if kind == "XOR":
        return value ^ operation["value"]
    if kind == "MOD":
        return value % operation["value"]
    if kind == "NEG":
        return -value
    if kind == "CLAMP":
        return min(operation["high"], max(operation["low"], value))
    raise ValueError(f"unknown pipeline operation: {kind}")


def _step_pipeline(state: State, params: Params) -> State:
    index = state["index"]
    if index >= len(params["operations"]):
        return deepcopy(state)
    operation = params["operations"][index]
    old_value = state["value"]
    new_value = _apply_pipeline_op(old_value, operation)
    history = state["history"] + [new_value]
    return {
        "value": new_value,
        "index": index + 1,
        "history": history,
        "last": {"operation": deepcopy(operation), "input": old_value, "output": new_value},
    }


def _replay_pipeline(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in params["operations"]:
        trace.append(_step_pipeline(trace[-1], params))
    return trace


def _sample_pipeline(rng: random.Random) -> Tuple[Params, State]:
    for _ in range(800):
        initial_value = rng.randint(-12, 20)
        ordinary = [
            {"kind": "ADD", "value": rng.choice((-7, -4, 3, 5, 8))},
            {"kind": "MUL", "value": rng.choice((-2, 2, 3))},
            {"kind": "XOR", "value": rng.randint(1, 15)},
            {"kind": "NEG"},
        ]
        boundary = rng.choice((
            {"kind": "MOD", "value": rng.randint(3, 9)},
            {"kind": "CLAMP", "low": rng.randint(-8, -2), "high": rng.randint(4, 11)},
        ))
        operations = rng.sample(ordinary, 3) + [boundary]
        rng.shuffle(operations)
        if rng.random() < 0.5:
            operations.append(rng.choice(ordinary))
        params: Params = {"operations": deepcopy(operations), "order": "left_to_right"}
        initial: State = {"value": initial_value, "index": 0, "history": [initial_value], "last": None}
        trace = _replay_pipeline(params, initial)
        boundary_event = next(
            item["last"] for item in trace[1:] if item["last"]["operation"]["kind"] in ("MOD", "CLAMP")
        )
        kind = boundary_event["operation"]["kind"]
        if kind == "MOD":
            triggered = not 0 <= boundary_event["input"] < boundary_event["operation"]["value"]
        else:
            triggered = not boundary_event["operation"]["low"] <= boundary_event["input"] <= boundary_event["operation"]["high"]
        if triggered and len(set(trace[-1]["history"])) >= 4:
            return params, initial
    raise RuntimeError("could not sample a pipeline with a load-bearing boundary operation")


# ---------------------------------------------------------------------------
# 19. Queue-head packet routing tick


def _step_packet_router(state: State, params: Params) -> State:
    queues = [[dict(packet) for packet in queue] for queue in state["inputs"]]
    expired: List[str] = []
    for queue in queues:
        while queue and queue[0]["ttl"] == 0:
            expired.append(queue.pop(0)["id"])
    heads = {index: queue[0] for index, queue in enumerate(queues) if queue}
    output_count = params["outputs"]
    order = [(state["pointer"] + offset) % len(queues) for offset in range(len(queues))]
    winners: Dict[int, int] = {}
    contenders_by_output: Dict[int, List[int]] = {}
    for output in range(output_count):
        contenders = [index for index in order if index in heads and heads[index]["dest"] == output]
        contenders_by_output[output] = contenders
        if contenders:
            winners[output] = contenders[0]
    winner_inputs = set(winners.values())
    transfers: List[Dict[str, Any]] = []
    for output, input_index in sorted(winners.items()):
        packet = queues[input_index].pop(0)
        packet["ttl"] -= 1
        transfers.append({"output": output, "from": input_index, "packet": packet})
    losers: List[str] = []
    for input_index, packet in heads.items():
        if input_index not in winner_inputs:
            # The same packet is still at the queue head after arbitration.
            queues[input_index][0]["ttl"] = max(0, queues[input_index][0]["ttl"] - 1)
            losers.append(packet["id"])
    return {
        "inputs": queues,
        "pointer": (state["pointer"] + 1) % len(queues),
        "tick": state["tick"] + 1,
        "last": {
            "expired": expired,
            "transfers": transfers,
            "losers": losers,
            "contenders": {str(key): value for key, value in contenders_by_output.items()},
        },
    }


def _replay_packet_router(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_packet_router(trace[-1], params))
    return trace


def _sample_packet_router(rng: random.Random) -> Tuple[Params, State]:
    contested, service, idle = rng.sample(range(3), 3)
    input_order = rng.sample(range(3), 3)
    packet_ids = iter(f"P{value}" for value in rng.sample(range(100, 1000), 5))
    queues: List[List[Dict[str, Any]]] = [[], [], []]
    queues[input_order[0]] = [{"id": next(packet_ids), "dest": contested, "ttl": rng.randint(1, 4)}]
    queues[input_order[1]] = [
        {"id": next(packet_ids), "dest": contested, "ttl": rng.randint(1, 4)},
        {"id": next(packet_ids), "dest": idle, "ttl": rng.randint(2, 5)},
    ]
    queues[input_order[2]] = [
        {"id": next(packet_ids), "dest": rng.choice((contested, service)), "ttl": 0},
        {"id": next(packet_ids), "dest": service, "ttl": rng.randint(1, 4)},
    ]
    params: Params = {"outputs": 3, "steps": 1, "arbitration": "cyclic_from_pointer"}
    initial: State = {"inputs": queues, "pointer": rng.randrange(3), "tick": 0, "last": None}
    return params, initial


# ---------------------------------------------------------------------------
# 20. Safety-constrained rail signal interlocking


def _step_rail(state: State, params: Params) -> State:
    occupancy = set(state["occupied"])
    accepted = None
    blocked: Dict[str, str] = {}
    for route_id in state["priority"]:
        if route_id not in state["requests"]:
            continue
        route = params["routes"][route_id]
        occupied_blocks = sorted(occupancy.intersection(route["blocks"]))
        if occupied_blocks:
            blocked[route_id] = "occupied: " + ", ".join(map(str, occupied_blocks))
        elif accepted is None:
            accepted = route_id
        else:
            blocked[route_id] = f"priority_conflict:{accepted}"
    signals = {route_id: ("GREEN" if route_id == accepted else "RED") for route_id in params["routes"]}
    switch = params["routes"][accepted]["switch"] if accepted is not None else state["switch"]
    return {
        "occupied": sorted(occupancy),
        "requests": state["requests"][:],
        "priority": state["priority"][:],
        "switch": switch,
        "signals": signals,
        "tick": state["tick"] + 1,
        "last": {"accepted": accepted, "blocked": blocked},
    }


def _replay_rail(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_rail(trace[-1], params))
    return trace


def _sample_rail(rng: random.Random) -> Tuple[Params, State]:
    trunk, branch_a, branch_b, branch_c, branch_d = rng.sample(range(8), 5)
    routes = {
        "A": {"blocks": sorted([trunk, branch_a]), "switch": "LEFT"},
        "B": {"blocks": sorted([trunk, branch_b]), "switch": "RIGHT"},
        "C": {"blocks": sorted(rng.choice(([branch_c], [branch_c, branch_d]))), "switch": rng.choice(("LEFT", "RIGHT"))},
    }
    blocked_route = rng.choice(tuple(routes))
    occupied = [rng.choice(routes[blocked_route]["blocks"])]
    safe_candidates = [route for route in routes if not set(routes[route]["blocks"]).intersection(occupied)]
    safe_route = rng.choice(safe_candidates)
    requests = [blocked_route, safe_route]
    if rng.random() < 0.5:
        third = next(route for route in routes if route not in requests)
        requests.append(third)
    priority = rng.sample(tuple(routes), 3)
    params: Params = {
        "routes": routes,
        "priority": priority[:],
        "steps": 1,
        "maximum_accepts": 1,
    }
    initial: State = {
        "occupied": occupied,
        "requests": requests,
        "priority": priority,
        "switch": rng.choice(("LEFT", "RIGHT")),
        "signals": {route: "RED" for route in routes},
        "tick": 0,
        "last": None,
    }
    final = _step_rail(initial, params)
    _require(blocked_route in final["last"]["blocked"], "constructed occupied route must remain blocked")
    return params, initial


# ---------------------------------------------------------------------------
# 21. Coupled elevator dispatcher tick


def _step_elevator(state: State, params: Params) -> State:
    cars = [dict(car) for car in state["cars"]]
    available = {car["id"] for car in cars}
    assignments: Dict[str, int] = {}
    tie_calls: List[int] = []
    for call in state["calls"]:
        candidates = [car for car in cars if car["id"] in available]
        if not candidates:
            break
        distances = {car["id"]: abs(car["floor"] - call) for car in candidates}
        minimum = min(distances.values())
        tied = sorted(car_id for car_id, distance in distances.items() if distance == minimum)
        if len(tied) > 1:
            tie_calls.append(call)
        winner = tied[0]
        assignments[winner] = call
        available.remove(winner)
    blocked_full: List[str] = []
    for car in cars:
        car["door"] = "CLOSED"
        target = assignments.get(car["id"])
        if target is None:
            continue
        if car["floor"] == target:
            if car["load"] < car["capacity"]:
                car["door"] = "OPEN"
            else:
                blocked_full.append(car["id"])
        elif target > car["floor"]:
            car["floor"] += 1
        else:
            car["floor"] -= 1
    return {
        "cars": cars,
        "calls": state["calls"][:],
        "tick": state["tick"] + 1,
        "last": {"assignments": assignments, "tie_calls": tie_calls, "blocked_full": blocked_full},
    }


def _replay_elevator(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_elevator(trace[-1], params))
    return trace


def _sample_elevator(rng: random.Random) -> Tuple[Params, State]:
    floor_count = rng.randint(7, 12)
    pairs = [
        (low, high)
        for low in range(floor_count)
        for high in range(low + 2, floor_count)
        if (high - low) % 2 == 0
    ]
    low, high = rng.choice(pairs)
    midpoint = (low + high) // 2
    capacity_a = rng.randint(3, 8)
    capacity_b = rng.randint(3, 8)
    cars = [
        {"id": "A", "floor": low, "door": "CLOSED", "load": rng.randrange(capacity_a), "capacity": capacity_a},
        {"id": "B", "floor": high, "door": "CLOSED", "load": capacity_b, "capacity": capacity_b},
    ]
    params: Params = {"steps": 1, "floors": floor_count, "tie_break": "lowest_id", "one_call_per_car": True}
    initial: State = {"cars": cars, "calls": [midpoint, high], "tick": 0, "last": None}
    final = _step_elevator(initial, params)
    _require(final["last"]["tie_calls"] == [midpoint], "midpoint call must be a real tie")
    _require(final["last"]["blocked_full"] == ["B"], "full B car must be blocked at its floor")
    return params, initial


# ---------------------------------------------------------------------------
# 22. Coupled finite-state message handshake


def _step_protocol(state: State, params: Params) -> State:
    queue = [dict(message) for message in state["queue"]]
    client = state["client"]
    server = state["server"]
    timeout = state["timeout"]
    delivered = queue.pop(0) if queue else None
    enqueued: List[Dict[str, str]] = []
    event = "idle"
    if delivered is not None:
        kind = delivered["kind"]
        event = f"delivered:{kind}"
        if kind == "START" and delivered["to"] == "client" and client == "IDLE":
            client = "WAIT"
            timeout = params["timeout_reset"]
            enqueued.append({"id": f"S{params['session']}H{state['tick']}", "kind": "HELLO", "to": "server"})
        elif kind == "HELLO" and delivered["to"] == "server":
            server = "READY"
            enqueued.append({"id": f"S{params['session']}A{state['tick']}", "kind": "ACK", "to": "client"})
        elif kind == "ACK" and delivered["to"] == "client" and client == "WAIT":
            client = "OPEN"
            timeout = 0
            enqueued.append({"id": f"S{params['session']}D{state['tick']}", "kind": "DATA", "to": "client"})
    queue.extend(enqueued)
    timeout_resend = False
    if client == "WAIT" and not (delivered and delivered["kind"] == "ACK" and delivered["to"] == "client"):
        timeout = max(0, timeout - 1)
        if timeout == 0:
            queue.append({"id": f"S{params['session']}R{state['tick']}", "kind": "HELLO", "to": "server"})
            timeout = params["timeout_reset"]
            timeout_resend = True
    return {
        "client": client,
        "server": server,
        "queue": queue,
        "timeout": timeout,
        "tick": state["tick"] + 1,
        "last": {
            "event": event,
            "delivered": delivered,
            "enqueued": enqueued,
            "timeout_resend": timeout_resend,
        },
    }


def _replay_protocol(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_protocol(trace[-1], params))
    return trace


def _sample_protocol(rng: random.Random) -> Tuple[Params, State]:
    reset = rng.randint(2, 6)
    # Keep every identity short enough to remain legible inside a message badge.
    # The two-digit session still provides meaningful per-sample identity variation.
    session = rng.randrange(10, 100)
    variant = rng.choice(("normal", "timeout_delay", "duplicate_hello"))
    if variant == "normal":
        noise = [
            {"id": f"S{session}N{index}", "kind": "DATA", "to": "client"}
            for index in range(rng.randint(0, 3))
        ]
        initial = {
            "client": "IDLE", "server": "LISTEN",
            "queue": [{"id": f"S{session}S", "kind": "START", "to": "client"}] + noise,
            "timeout": 0, "tick": 0, "last": None,
        }
        steps = rng.randint(3, 5)
    elif variant == "timeout_delay":
        noise_count = rng.randint(1, 4)
        initial = {
            "client": "WAIT", "server": "LISTEN",
            "queue": [
                {"id": f"S{session}N{index}", "kind": "DATA", "to": "client"}
                for index in range(noise_count)
            ],
            "timeout": 1, "tick": 0, "last": None,
        }
        steps = rng.randint(3, 6)
    else:
        duplicate_count = rng.randint(1, 3)
        initial = {
            "client": "WAIT", "server": "READY",
            "queue": [
                {"id": f"S{session}H{index}", "kind": "HELLO", "to": "server"}
                for index in range(duplicate_count)
            ],
            "timeout": rng.randint(1, reset), "tick": 0, "last": None,
        }
        steps = rng.randint(2, 5)
    params: Params = {
        "timeout_reset": reset, "steps": steps, "variant": variant,
        "delivery": "oldest_first", "session": session,
    }
    return params, initial


# ---------------------------------------------------------------------------
# 23. Instruction-addressed register machine


def _step_register_machine(state: State, params: Params) -> State:
    registers = dict(state["registers"])
    pc = state["pc"]
    program = params["program"]
    instruction = program[pc]
    kind = instruction["kind"]
    next_pc = (pc + 1) % len(program)
    zero = state["zero"]
    detail: Dict[str, Any] = {"pc": pc, "instruction": deepcopy(instruction)}
    if kind == "ADD":
        value = (registers[instruction["dst"]] + registers[instruction["src"]]) % 16
        registers[instruction["dst"]] = value
        zero = value == 0
        detail["result"] = value
    elif kind == "XOR":
        value = registers[instruction["dst"]] ^ registers[instruction["src"]]
        registers[instruction["dst"]] = value
        zero = value == 0
        detail["result"] = value
    elif kind == "MOV":
        value = registers[instruction["src"]]
        registers[instruction["dst"]] = value
        zero = value == 0
        detail["result"] = value
    elif kind == "JZ":
        if zero:
            next_pc = instruction["target"]
        detail["taken"] = bool(zero)
    elif kind == "DECJNZ":
        register = instruction["reg"]
        registers[register] = (registers[register] - 1) % 16
        zero = registers[register] == 0
        if not zero:
            next_pc = instruction["target"]
        detail.update({"result": registers[register], "taken": not zero})
    else:
        raise ValueError(f"unknown register instruction: {kind}")
    return {"registers": registers, "pc": next_pc, "zero": zero, "tick": state["tick"] + 1, "last": detail}


def _replay_register_machine(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_register_machine(trace[-1], params))
    return trace


def _sample_register_machine(rng: random.Random) -> Tuple[Params, State]:
    variant = rng.choice(("overflow_jump", "nonzero_fallthrough", "decrement_loop"))
    registers = {"A": rng.randrange(16), "B": rng.randrange(16), "C": rng.randrange(16)}
    if variant == "overflow_jump":
        registers["A"] = rng.randint(1, 15)
        registers["B"] = 16 - registers["A"]
        program = [
            {"kind": "ADD", "dst": "A", "src": "B"},
            {"kind": "JZ", "target": 3},
            {"kind": "XOR", "dst": "C", "src": "A"},
            {"kind": "MOV", "dst": "C", "src": "B"},
        ]
        steps = 2
    elif variant == "nonzero_fallthrough":
        registers["A"] = rng.randint(1, 7)
        registers["B"] = rng.randint(1, 7)
        program = [
            {"kind": "ADD", "dst": "A", "src": "B"},
            {"kind": "JZ", "target": 3},
            {"kind": "XOR", "dst": "C", "src": "A"},
            {"kind": "MOV", "dst": "B", "src": "C"},
        ]
        steps = 3
    else:
        registers["A"] = rng.randint(2, 4)
        program = [
            {"kind": "DECJNZ", "reg": "A", "target": 0},
            {"kind": "MOV", "dst": "C", "src": "B"},
            {"kind": "XOR", "dst": "B", "src": "C"},
            {"kind": "JZ", "target": 0},
        ]
        steps = registers["A"]
    params: Params = {"program": program, "steps": steps, "modulus": 16, "variant": variant}
    initial: State = {"registers": registers, "pc": 0, "zero": False, "tick": 0, "last": None}
    return params, initial


# ---------------------------------------------------------------------------
# 24. Simultaneous clocked logic-register capture


def _eval_logic(expression: Mapping[str, Any], q: Sequence[int], inputs: Mapping[str, int]) -> int:
    kind = expression["kind"]
    if kind == "Q":
        return q[expression["index"]]
    if kind == "IN":
        return inputs[expression["name"]]
    if kind == "NOT":
        return 1 - _eval_logic(expression["arg"], q, inputs)
    left = _eval_logic(expression["left"], q, inputs)
    right = _eval_logic(expression["right"], q, inputs)
    if kind == "XOR":
        return left ^ right
    if kind == "AND":
        return left & right
    if kind == "OR":
        return left | right
    raise ValueError(f"unknown logic expression: {kind}")


def _step_clocked_logic(state: State, params: Params) -> State:
    old_q = state["q"][:]
    d_values = [_eval_logic(expression, old_q, state["inputs"]) for expression in params["expressions"]]
    return {"q": d_values, "inputs": dict(state["inputs"]), "tick": state["tick"] + 1, "last_d": d_values}


def _replay_clocked_logic(params: Params, initial: State) -> List[State]:
    trace = [deepcopy(initial)]
    for _ in range(params["steps"]):
        trace.append(_step_clocked_logic(trace[-1], params))
    return trace


def _sequential_clock_result(state: State, params: Params) -> List[int]:
    q = state["q"][:]
    for index, expression in enumerate(params["expressions"]):
        q[index] = _eval_logic(expression, q, state["inputs"])
    return q


def _sample_clocked_logic(rng: random.Random) -> Tuple[Params, State]:
    atoms = [
        {"kind": "Q", "index": index} for index in range(3)
    ] + [
        {"kind": "IN", "name": name} for name in ("X", "Y")
    ]

    def random_expression() -> Dict[str, Any]:
        if rng.random() < 0.22:
            return {"kind": "NOT", "arg": deepcopy(rng.choice(atoms))}
        return {
            "kind": rng.choice(("XOR", "AND", "OR")),
            "left": deepcopy(rng.choice(atoms)),
            "right": deepcopy(rng.choice(atoms)),
        }

    for _ in range(1500):
        expressions = [random_expression() for _ in range(3)]
        # At least one later D expression must read a register captured earlier
        # under an incorrect in-place interpretation.
        if not any(
            expression.get("kind") == "Q" and expression.get("index", 3) < index
            or expression.get("arg", {}).get("kind") == "Q" and expression["arg"].get("index", 3) < index
            or expression.get("left", {}).get("kind") == "Q" and expression["left"].get("index", 3) < index
            or expression.get("right", {}).get("kind") == "Q" and expression["right"].get("index", 3) < index
            for index, expression in enumerate(expressions)
        ):
            continue
        initial: State = {
            "q": [rng.randrange(2) for _ in range(3)],
            "inputs": {"X": rng.randrange(2), "Y": rng.randrange(2)},
            "tick": 0,
            "last_d": None,
        }
        params: Params = {"expressions": expressions, "steps": rng.choice((1, 2)), "capture": "simultaneous_from_old_q"}
        simultaneous = _step_clocked_logic(initial, params)["q"]
        sequential = _sequential_clock_result(initial, params)
        if simultaneous != sequential and simultaneous != initial["q"]:
            return params, initial
    raise RuntimeError("could not sample feedback logic with load-bearing simultaneous capture")


# ---------------------------------------------------------------------------
# Shared replay, evidence, answer, and rendering-facing semantic adapters


_SAMPLERS: Dict[str, Callable[[random.Random], Tuple[Params, State]]] = {
    "rule_table_1d_rollout": _sample_rule_table,
    "lifelike_rule_rollout": _sample_lifelike,
    "cyclic_threshold_wave": _sample_cyclic,
    "excitable_signal_network": _sample_excitable,
    "alternating_block_rotation": _sample_block_rotation,
    "xor_parity_lattice": _sample_xor,
    "sandpile_avalanche": _sample_sandpile,
    "lattice_gas_collision": _sample_lattice_gas,
    "majority_consensus": _sample_majority,
    "forest_fire_barrier": _sample_forest,
    "predator_prey_cycle": _sample_predator_prey,
    "turn_write_agent_rollout": _sample_turn_write,
    "finite_tape_machine_rollout": _sample_tape_machine,
    "simultaneous_agent_conflict": _sample_agent_conflict,
    "resource_pick_drop_swarm": _sample_resource_swarm,
    "diffusion_limited_aggregation": _sample_dla,
    "gui_hierarchical_navigation": _sample_gui,
    "chained_function_pipeline": _sample_pipeline,
    "packet_router_tick": _sample_packet_router,
    "rail_signal_interlocking": _sample_rail,
    "elevator_dispatch_tick": _sample_elevator,
    "protocol_handshake_tick": _sample_protocol,
    "register_machine_tick": _sample_register_machine,
    "clocked_logic_register_tick": _sample_clocked_logic,
}


_REPLAYERS: Dict[str, Callable[[Params, State], List[State]]] = {
    "rule_table_1d_rollout": _replay_rule_table,
    "lifelike_rule_rollout": _replay_lifelike,
    "cyclic_threshold_wave": _replay_cyclic,
    "excitable_signal_network": _replay_excitable,
    "alternating_block_rotation": _replay_block_rotation,
    "xor_parity_lattice": _replay_xor,
    "sandpile_avalanche": _replay_sandpile,
    "lattice_gas_collision": _replay_lattice_gas,
    "majority_consensus": _replay_majority,
    "forest_fire_barrier": _replay_forest,
    "predator_prey_cycle": _replay_predator_prey,
    "turn_write_agent_rollout": _replay_turn_write,
    "finite_tape_machine_rollout": _replay_tape_machine,
    "simultaneous_agent_conflict": _replay_agent_conflict,
    "resource_pick_drop_swarm": _replay_resource_swarm,
    "diffusion_limited_aggregation": _replay_dla,
    "gui_hierarchical_navigation": _replay_gui,
    "chained_function_pipeline": _replay_pipeline,
    "packet_router_tick": _replay_packet_router,
    "rail_signal_interlocking": _replay_rail,
    "elevator_dispatch_tick": _replay_elevator,
    "protocol_handshake_tick": _replay_protocol,
    "register_machine_tick": _replay_register_machine,
    "clocked_logic_register_tick": _replay_clocked_logic,
}


def replay_case(task: str, params: Mapping[str, Any], initial_state: Mapping[str, Any]) -> List[State]:
    """Independently recompute the complete finite transition trace."""
    if task not in _REPLAYERS:
        raise KeyError(f"unknown next-state task: {task}")
    return _REPLAYERS[task](deepcopy(dict(params)), deepcopy(dict(initial_state)))


def _task_evidence(task: str, params: Params, trace: List[State]) -> Dict[str, Any]:
    initial, final = trace[0], trace[-1]
    if task == "rule_table_1d_rollout":
        alternate = dict(params)
        alternate["boundary"] = "wrap" if params["boundary"] == "fixed_zero" else "fixed_zero"
        alternate_final = _replay_rule_table(alternate, initial)[-1]["row"]
        return {
            "rule_ones": sum(params["rule"]),
            "changed_cells": _count_differences(initial["row"], final["row"]),
            "boundary_terminal_differences": sum(a != b for a, b in zip(final["row"], alternate_final)),
        }
    if task == "lifelike_rule_rollout":
        alternate = dict(params)
        alternate["boundary"] = "wrap" if params["boundary"] == "fixed_zero" else "fixed_zero"
        alternate_final = _replay_lifelike(alternate, initial)[-1]["grid"]
        return {
            "total_cell_updates": sum(_count_differences(a["grid"], b["grid"]) for a, b in zip(trace, trace[1:])),
            "boundary_terminal_differences": _count_differences(final["grid"], alternate_final),
            "birth_counts": params["birth"],
            "survival_counts": params["survive"],
        }
    if task == "cyclic_threshold_wave":
        max_threshold = 2 if params["neighbourhood"] == "orthogonal" else 3
        alt_threshold = dict(params)
        alt_threshold["threshold"] = 1 + (params["threshold"] % max_threshold)
        alt_neighbourhood = dict(params)
        alt_neighbourhood["neighbourhood"] = "moore" if params["neighbourhood"] == "orthogonal" else "orthogonal"
        return {
            "total_advances": sum(_count_differences(a["grid"], b["grid"]) for a, b in zip(trace, trace[1:])),
            "threshold_terminal_differences": _count_differences(final["grid"], _replay_cyclic(alt_threshold, initial)[-1]["grid"]),
            "neighbourhood_terminal_differences": _count_differences(final["grid"], _replay_cyclic(alt_neighbourhood, initial)[-1]["grid"]),
        }
    if task == "excitable_signal_network":
        rows, cols = _shape(trace[0]["grid"])
        branch_targets = collision_targets = 0
        for old_state, new_state in zip(trace, trace[1:]):
            old, new = old_state["grid"], new_state["grid"]
            for row in range(rows):
                for col in range(cols):
                    if old[row][col] != 1:
                        continue
                    heads = sum(_lookup(old, row + dr, col + dc, "fixed", 0) == 2 for dr, dc in DIRS.values())
                    branch_targets += heads == 1 and new[row][col] == 2
                    collision_targets += heads >= 2 and new[row][col] == 1
        blockers = [(r, c) for r in range(rows) for c in range(cols) if trace[0]["grid"][r][c] == 4]
        return {
            "branch_targets": branch_targets,
            "collision_targets_retained": collision_targets,
            "blockers_preserved_all_steps": all(all(item["grid"][r][c] == 4 for r, c in blockers) for item in trace),
        }
    if task == "alternating_block_rotation":
        token_count = lambda state: sum(value >= 0 for row in state["grid"] for value in row)
        alternate = deepcopy(initial)
        alternate["phase"] = 1 - alternate["phase"]
        alternate_first = _step_block_rotation(alternate, params)
        size = len(initial["grid"])
        wrap_tokens = 0
        for item in trace[:-1]:
            if item["phase"] == 1:
                edge_coords = (
                    {(edge, col) for edge in (0, size - 1) for col in range(size)}
                    | {(row, edge) for row in range(size) for edge in (0, size - 1)}
                )
                wrap_tokens += sum(item["grid"][row][col] >= 0 for row, col in edge_coords)
        return {
            "tokens_preserved": len({token_count(item) for item in trace}) == 1,
            "phase_first_step_differences": _count_differences(trace[1]["grid"], alternate_first["grid"]),
            "tokens_in_wrapping_partition_edges": wrap_tokens,
        }
    if task == "xor_parity_lattice":
        grid = initial["grid"]
        size = len(grid)
        fixed = _grid(size, size)
        for row in range(size):
            for col in range(size):
                bits = [_lookup(grid, row + dr, col + dc, "fixed", 0) for dr, dc in DIRS.values()]
                if params["include_self"]:
                    bits.append(grid[row][col])
                fixed[row][col] = sum(bits) % 2
        edge_diff = sum(
            trace[1]["grid"][row][col] != fixed[row][col]
            for row in range(size) for col in range(size)
            if row in (0, size - 1) or col in (0, size - 1)
        )
        return {"wrapped_edge_contributions": edge_diff, "terminal_changes": _count_differences(grid, final["grid"])}
    if task == "sandpile_avalanche":
        initial_mass = sum(map(sum, initial["grid"])) + 1
        terminal_accounted = sum(map(sum, final["grid"])) + final["lost"]
        return {
            "waves": len(trace) - 2,
            "open_boundary_loss": final["lost"],
            "mass_conserved_with_loss": initial_mass == terminal_accounted,
            "terminal_stable": all(value < 4 for row in final["grid"] for value in row),
        }
    if task == "lattice_gas_collision":
        ew, ns = VELOCITY_BITS["E"] | VELOCITY_BITS["W"], VELOCITY_BITS["N"] | VELOCITY_BITS["S"]
        collision_cells = sum(value in (ew, ns) for row in initial["grid"] for value in row)
        counts = [sum(_bit_count(value) for row in item["grid"] for value in row) for item in trace]
        return {
            "head_on_collision_cells": collision_cells,
            "reflections": sum(item.get("last_reflections", 0) for item in trace[1:]),
            "particle_count_conserved": len(set(counts)) == 1,
            "particle_count": counts[0],
        }
    if task == "majority_consensus":
        flips, ties = _majority_transition_counts(initial["grid"], params)
        alternate = dict(params)
        alternate["boundary"] = "wrap" if params["boundary"] == "fixed" else "fixed"
        return {
            "first_step_flips": flips,
            "first_step_ties_retained": ties,
            "vacancies": sum(value == 0 for row in initial["grid"] for value in row),
            "boundary_terminal_differences": _count_differences(final["grid"], _replay_majority(alternate, initial)[-1]["grid"]),
        }
    if task == "forest_fire_barrier":
        first_ignitions = sum(a == 1 and b == 2 for row_a, row_b in zip(initial["grid"], trace[1]["grid"]) for a, b in zip(row_a, row_b))
        routes = params["barrier_routes"]
        barrier_preserved = True
        behind_survives = True
        for name, (barrier, behind) in routes.items():
            code = 4 if name == "water" else 5
            barrier_preserved &= all(item["grid"][barrier[0]][barrier[1]] == code for item in trace)
            behind_survives &= all(item["grid"][behind[0]][behind[1]] == 1 for item in trace)
        return {
            "first_step_ignitions": first_ignitions,
            "both_barriers_preserved": barrier_preserved,
            "trees_behind_barriers_survive": behind_survives,
            "flames_expired": sum(a == 2 and b == 3 for row_a, row_b in zip(initial["grid"], trace[1]["grid"]) for a, b in zip(row_a, row_b)),
        }
    if task == "predator_prey_cycle":
        old, new = initial["grid"], trace[1]["grid"]
        pairs = [(a, b) for row_a, row_b in zip(old, new) for a, b in zip(row_a, row_b)]
        sheltered = [(r, c) for r, row in enumerate(old) for c, value in enumerate(row) if value == 3]
        return {
            "births": pairs.count((0, 1)),
            "predations": pairs.count((1, 2)),
            "starvations": pairs.count((2, 0)),
            "sheltered_preserved": all(trace[1]["grid"][r][c] == 3 for r, c in sheltered),
        }
    if task == "turn_write_agent_rollout":
        events = [item["last"] for item in trace[1:]]
        return {
            "bits_read": sorted({event["read"] for event in events}),
            "rewrites": sum(event["read"] != event["write"] for event in events),
            "wrap_crossings": sum(event["wrapped"] for event in events),
            "ticks": len(events),
        }
    if task == "finite_tape_machine_rollout":
        events = [item["last"] for item in trace[1:]]
        return {
            "bits_read": sorted({event["read"] for event in events}),
            "rewrites": sum(event["read"] != event["write"] for event in events),
            "control_changes": sum(event["old_control"] != event["new_control"] for event in events),
            "wrap_crossings": sum(event["wrapped"] for event in events),
        }
    if task == "simultaneous_agent_conflict":
        event = trace[1]["last"]
        participants = {agent for group in event["conflicts"] for agent in group}
        return {
            "conflict_groups": event["conflicts"],
            "unaffected_movers": sorted(set(event["moved"]) - participants),
            "stable_identity_count": len(initial["agents"]),
        }
    if task == "resource_pick_drop_swarm":
        events = [event for item in trace[1:] for event in item["last"]]
        total = lambda item: sum(map(sum, item["piles"])) + sum(agent["carrying"] for agent in item["agents"])
        return {
            "pickups": sum(event["action"] == "pick" for event in events),
            "drops": sum(event["action"] == "drop" for event in events),
            "wrap_crossings": sum(event["wrapped"] for event in events),
            "resource_total_conserved": len({total(item) for item in trace}) == 1,
            "resource_total": total(initial),
        }
    if task == "diffusion_limited_aggregation":
        event = trace[1]["last"]
        participants = {agent for group in event["conflicts"] for agent in group}
        return {
            "frozen_walkers": event["frozen"],
            "conflict_groups": event["conflicts"],
            "unaffected_movers": sorted(set(event["moved"]) - participants),
            "aggregate_growth": len(final["aggregate"]) - len(initial["aggregate"]),
        }
    if task == "gui_hierarchical_navigation":
        events = [item["last"] for item in trace[1:]]
        return {
            "clicks": sum(event["action"]["kind"] == "CLICK" for event in events),
            "back_pops": sum(event["outcome"] == "popped" for event in events),
            "visited_pages": list(dict.fromkeys(item["page"] for item in trace)),
            "terminal_depth": len(final["stack"]),
        }
    if task == "chained_function_pipeline":
        events = [item["last"] for item in trace[1:]]
        boundary_events = [event for event in events if event["operation"]["kind"] in ("MOD", "CLAMP")]
        return {
            "operations_executed": len(events),
            "distinct_values": len(set(final["history"])),
            "boundary_events": boundary_events,
            "ordered_history": final["history"],
        }
    if task == "packet_router_tick":
        event = trace[1]["last"]
        contention = {output: values for output, values in event["contenders"].items() if len(values) > 1}
        idle = [output for output, values in event["contenders"].items() if not values]
        return {
            "expired_packets": event["expired"],
            "contended_outputs": contention,
            "idle_outputs": idle,
            "transfers": event["transfers"],
        }
    if task == "rail_signal_interlocking":
        event = trace[1]["last"]
        return {
            "accepted_route": event["accepted"],
            "blocked_requests": event["blocked"],
            "green_signal_count": sum(value == "GREEN" for value in final["signals"].values()),
            "occupied_blocks_preserved": final["occupied"] == initial["occupied"],
        }
    if task == "elevator_dispatch_tick":
        event = trace[1]["last"]
        return {
            "assignments": event["assignments"],
            "distance_tie_calls": event["tie_calls"],
            "full_car_blocks": event["blocked_full"],
            "cars_updated": len(final["cars"]),
        }
    if task == "protocol_handshake_tick":
        events = [item["last"] for item in trace[1:]]
        return {
            "variant": params["variant"],
            "delivered_events": [event["event"] for event in events],
            "timeout_resends": sum(event["timeout_resend"] for event in events),
            "terminal_client": final["client"],
            "terminal_server": final["server"],
        }
    if task == "register_machine_tick":
        events = [item["last"] for item in trace[1:]]
        return {
            "variant": params["variant"],
            "executed_pcs": [event["pc"] for event in events],
            "branch_decisions": [event["taken"] for event in events if "taken" in event],
            "modulo_zero_results": sum(event.get("result") == 0 for event in events),
        }
    if task == "clocked_logic_register_tick":
        simultaneous = trace[1]["q"]
        sequential = _sequential_clock_result(initial, params)
        return {
            "simultaneous_first_capture": simultaneous,
            "incorrect_in_place_capture": sequential,
            "simultaneity_load_bearing": simultaneous != sequential,
            "captures": len(trace) - 1,
        }
    raise KeyError(task)


def answer_payload(task: str, state: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the unique typed terminal payload; no cached oracle field is read."""
    state = deepcopy(dict(state))
    if task == "rule_table_1d_rollout":
        return {"generation": state["generation"], "row": "".join(map(str, state["row"]))}
    if task in {
        "lifelike_rule_rollout", "cyclic_threshold_wave", "excitable_signal_network",
        "xor_parity_lattice", "majority_consensus", "forest_fire_barrier", "predator_prey_cycle",
    }:
        return {"generation": state["generation"], "grid": state["grid"]}
    if task == "alternating_block_rotation":
        return {"generation": state["generation"], "phase": state["phase"], "token_grid": state["grid"]}
    if task == "sandpile_avalanche":
        return {"stable_grid": state["grid"], "lost": state["lost"], "waves": state["wave"]}
    if task == "lattice_gas_collision":
        return {"generation": state["generation"], "velocity_masks": state["grid"]}
    if task == "turn_write_agent_rollout":
        return {
            "tick": state["tick"], "grid": state["grid"],
            "head": format_xy_position(cell_to_xy(state["head"])),
            "direction": state["direction"],
        }
    if task == "finite_tape_machine_rollout":
        # Model-facing slot ordinals start at 1 at the left end of the tape; the
        # replayed state keeps internal 0-based indices.  ``tape`` bits and
        # ``tick`` are values, not ordinals, so they are exported unchanged.
        return {
            "tick": state["tick"], "tape": state["tape"],
            "head": int(state["head"]) + 1, "control": state["control"],
        }
    if task == "simultaneous_agent_conflict":
        return {
            "tick": state["tick"],
            "agents": [
                {
                    "id": agent["id"],
                    "position": format_xy_position(cell_to_xy((agent["row"], agent["col"]))),
                    "direction": agent["direction"],
                }
                for agent in state["agents"]
            ],
        }
    if task == "resource_pick_drop_swarm":
        return {
            "tick": state["tick"],
            "piles": state["piles"],
            "agents": [
                {
                    "id": agent["id"],
                    "position": format_xy_position(cell_to_xy((agent["row"], agent["col"]))),
                    "direction": agent["direction"],
                    "carrying": bool(agent["carrying"]),
                }
                for agent in state["agents"]
            ],
        }
    if task == "diffusion_limited_aggregation":
        return {
            "tick": state["tick"],
            "aggregate": [
                format_xy_position(cell_to_xy((row, col)))
                for row, col in state["aggregate"]
            ],
            "walkers": [
                {
                    "id": walker["id"],
                    "position": format_xy_position(cell_to_xy((walker["row"], walker["col"]))),
                    "direction": walker["direction"],
                }
                for walker in state["walkers"]
            ],
        }
    if task == "gui_hierarchical_navigation":
        return {"tick": state["tick"], "page": state["page"], "stack": state["stack"], "actions_consumed": state["action_index"]}
    if task == "chained_function_pipeline":
        return {"value": state["value"], "history": state["history"]}
    if task == "packet_router_tick":
        # Input queues I1-I3 and output ports O1-O3 are model-facing ordinals
        # numbered from 1, so the exported payload uses the same numbers the
        # scene draws; the replayed state keeps internal 0-based indices.
        # ``ttl`` and ``tick`` are counts, not ordinals, and stay unchanged.
        def _visible_packet(packet: Mapping[str, Any]) -> Dict[str, Any]:
            return {**dict(packet), "dest": int(packet["dest"]) + 1}

        return {
            "tick": state["tick"],
            "inputs": [[_visible_packet(packet) for packet in queue] for queue in state["inputs"]],
            "pointer": int(state["pointer"]) + 1,
            "transfers": [
                {
                    **dict(transfer),
                    "output": int(transfer["output"]) + 1,
                    "from": int(transfer["from"]) + 1,
                    "packet": _visible_packet(transfer["packet"]),
                }
                for transfer in state["last"]["transfers"]
            ],
        }
    if task == "rail_signal_interlocking":
        return {"tick": state["tick"], "switch": state["switch"], "signals": state["signals"], "accepted": state["last"]["accepted"]}
    if task == "elevator_dispatch_tick":
        return {"tick": state["tick"], "cars": state["cars"], "assignments": state["last"]["assignments"]}
    if task == "protocol_handshake_tick":
        return {
            "tick": state["tick"], "client": state["client"], "server": state["server"],
            "queue": state["queue"], "timeout": state["timeout"],
        }
    if task == "register_machine_tick":
        # ``pc`` names a program line, so the exported typed answer uses the
        # 1-based line numbers the Prompt and the state card declare.  ``tick``
        # counts completed instructions and stays a count, not an ordinal.
        return {
            "tick": state["tick"],
            "registers": state["registers"],
            "pc": int(state["pc"]) + 1,
            "zero": state["zero"],
        }
    if task == "clocked_logic_register_tick":
        return {"tick": state["tick"], "q": state["q"], "inputs": state["inputs"]}
    raise KeyError(task)


_ARROWS = {"N": "↑", "E": "→", "S": "↓", "W": "←"}


def visual_matrix(task: str, state: Mapping[str, Any], params: Mapping[str, Any]) -> List[List[str]]:
    """Normalize heterogeneous semantic state into a stable rectangular visual matrix."""
    state = dict(state)
    if task == "rule_table_1d_rollout":
        return [[str(value) for value in state["row"]]]
    if task in {
        "lifelike_rule_rollout", "cyclic_threshold_wave", "excitable_signal_network",
        "alternating_block_rotation", "xor_parity_lattice", "sandpile_avalanche",
        "lattice_gas_collision", "majority_consensus", "forest_fire_barrier", "predator_prey_cycle",
    }:
        return [[str(value) for value in row] for row in state["grid"]]
    if task == "turn_write_agent_rollout":
        matrix = [[str(value) for value in row] for row in state["grid"]]
        row, col = state["head"]
        matrix[row][col] = f"H{_ARROWS[state['direction']]}/{matrix[row][col]}"
        return matrix
    if task == "finite_tape_machine_rollout":
        matrix = [[str(value) for value in state["tape"]]]
        matrix[0][state["head"]] = f"{state['control']}{_ARROWS['E']}/{matrix[0][state['head']]}"
        return matrix
    if task == "simultaneous_agent_conflict":
        size = params["size"]
        matrix = [["" for _ in range(size)] for _ in range(size)]
        for agent in state["agents"]:
            matrix[agent["row"]][agent["col"]] = f"{agent['id']}{_ARROWS[agent['direction']]}"
        return matrix
    if task == "resource_pick_drop_swarm":
        matrix = [[str(value) if value else "" for value in row] for row in state["piles"]]
        for agent in state["agents"]:
            base = matrix[agent["row"]][agent["col"]] or "0"
            carry = "●" if agent["carrying"] else "○"
            matrix[agent["row"]][agent["col"]] = f"{agent['id']}{_ARROWS[agent['direction']]}{carry}/{base}"
        return matrix
    if task == "diffusion_limited_aggregation":
        size = params["size"]
        matrix = [["" for _ in range(size)] for _ in range(size)]
        for row, col in state["aggregate"]:
            matrix[row][col] = "#"
        for walker in state["walkers"]:
            matrix[walker["row"]][walker["col"]] = f"{walker['id']}{_ARROWS[walker['direction']]}"
        return matrix
    if task == "gui_hierarchical_navigation":
        tiles = params["pages"][state["page"]]
        cells = [f"{tile['label']}→{tile['target']}" for tile in tiles][:3]
        cells.extend(["—"] * (3 - len(cells)))
        consumed = int(state["action_index"])
        total = len(params["actions"])
        progress = "●" * consumed + "○" * (total - consumed)
        # BACK is a stable interface control, not a changing status caption.
        # Keeping it visible lets a BACK transition show the actual load-bearing
        # action before the page commits, just as a CLICK can press a visible
        # tile.  Empty slots in this row are ordinary interface spacing.
        return [
            ["← BACK", "", ""],
            cells,
            [f"PAGE:{state['page']}", f"DEPTH:{len(state['stack'])}", progress],
        ]
    if task == "chained_function_pipeline":
        cells = [f"IN:{state['history'][0]}"]
        for index, operation in enumerate(params["operations"]):
            marker = "✓" if index < state["index"] else "·"
            cells.append(f"{marker}{_format_operation(operation)}")
        cells.append(f"OUT:{state['value']}")
        return [cells]
    if task == "packet_router_tick":
        # Queue and port names are 1-based ordinals; ``ttl`` stays a count.
        width = max(3, max((len(queue) for queue in state["inputs"]), default=0))
        matrix: List[List[str]] = []
        for index, queue in enumerate(state["inputs"]):
            cells = [f"I{index + 1}{'*' if index == state['pointer'] else ''}"]
            cells.extend(f"{packet['id']}→O{packet['dest'] + 1}/t{packet['ttl']}" for packet in queue)
            cells.extend(["—"] * (width - len(queue)))
            matrix.append(cells)
        return matrix
    if task == "rail_signal_interlocking":
        matrix = [["ROUTE", "REQ", "SIGNAL", "BLOCKS"]]
        for route_id in sorted(params["routes"]):
            route = params["routes"][route_id]
            matrix.append([
                route_id,
                "YES" if route_id in state["requests"] else "NO",
                state["signals"][route_id],
                ", ".join(map(str, route["blocks"])),
            ])
        matrix.append(["OCCUPIED", ", ".join(map(str, state["occupied"])), "SW", state["switch"]])
        return matrix
    if task == "elevator_dispatch_tick":
        matrix = [["CAR", "FLOOR", "DOOR", "LOAD"]]
        for car in sorted(state["cars"], key=lambda item: item["id"]):
            matrix.append([car["id"], str(car["floor"]), car["door"], f"{car['load']}/{car['capacity']}"])
        matrix.append(["CALLS", ", ".join(map(str, state["calls"])), "TICK", str(state["tick"])])
        return matrix
    if task == "protocol_handshake_tick":
        queue = [f"{item['id']}:{item['kind']}→{item['to'][0].upper()}" for item in state["queue"]]
        queue.extend(["—"] * (5 - len(queue)))
        return [
            ["CLIENT", state["client"], "", "SERVER", state["server"]],
            ["TIMEOUT", str(state["timeout"]), "", "TICK", str(state["tick"])],
            queue[:5],
        ]
    if task == "register_machine_tick":
        registers = state["registers"]
        # Program lines are numbered from 1 for the model; ``state["pc"]`` stays
        # a 0-based list index internally.  ZERO and the register contents are
        # values, not ordinals, and are drawn unchanged.
        return [
            ["PC", str(int(state["pc"]) + 1), "ZERO", str(int(state["zero"]))],
            ["A", str(registers["A"]), "B", str(registers["B"])],
            ["C", str(registers["C"]), "", ""],
        ]
    if task == "clocked_logic_register_tick":
        # Register slots are numbered Q1-Q3 for the model; the stored bits are
        # values and keep their 0/1 encoding.
        return [
            ["Q1", str(state["q"][0]), "Q2", str(state["q"][1]), "Q3", str(state["q"][2])],
            ["X", str(state["inputs"]["X"]), "Y", str(state["inputs"]["Y"]), "", ""],
        ]
    raise KeyError(task)


def _format_operation(operation: Mapping[str, Any]) -> str:
    kind = operation["kind"]
    if kind in ("ADD", "MUL", "XOR", "MOD"):
        return f"{kind}:{operation['value']}"
    if kind == "CLAMP":
        return f"CLAMP:{operation['low']}..{operation['high']}"
    return kind


def extrapolate_regime(task, params, initial, regime, rng):
    """Extend public rule parameters or starting phases; replay/render stay native."""
    params, initial = deepcopy(params), deepcopy(initial)
    if (task, regime) == ("rule_table_1d_rollout", "singleton_rule"):
        params["rule"] = [0] * 8
        params["rule"][rng.choice((1, 2, 4))] = 1
    elif (task, regime) == ("lifelike_rule_rollout", "birth_zero"):
        params["birth"] = [0, 3]
        # Make the newly enabled zero-neighbour branch observable.
        grid = initial["grid"]
        for row in range(3):
            for col in range(3):
                grid[row][col] = 0
    elif (task, regime) == ("cyclic_threshold_wave", "threshold_four"):
        params.update(neighbourhood="moore", threshold=4)
        grid = initial["grid"]
        row = col = len(grid) // 2
        successor = (grid[row][col] + 1) % params["colours"]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            grid[row + dr][col + dc] = successor
    elif (task, regime) == ("sandpile_avalanche", "interior_addition"):
        size = len(initial["grid"])
        row, col = rng.randrange(1, size - 1), rng.randrange(1, size - 1)
        params["add"] = [row, col]
        initial["grid"][row][col] = 3
    elif (task, regime) == ("elevator_dispatch_tick", "full_first_car"):
        a, b = initial["cars"]
        a["load"], b["load"] = a["capacity"], max(0, b["capacity"] - 1)
        initial["calls"] = [sum(c["floor"] for c in initial["cars"]) // 2, a["floor"]]
    elif (task, regime) == ("protocol_handshake_tick", "ready_server_restart"):
        initial.update(client="IDLE", server="READY", timeout=0)
        initial["queue"] = [
            {"id": f"S{params['session']}S", "kind": "START", "to": "client"}
        ]
        params["steps"] = 4
        params["variant"] = regime
    else:
        raise ValueError(f"unsupported mechanism regime: {task}/{regime}")
    return params, initial


MAX_ROLLOUT_STEPS = 7


def build_case(
    task: str, rng: random.Random | int | None = None, *, regime: str | None = None
) -> Dict[str, Any]:
    """Sample and close a complete semantic case under independent replay."""
    if task not in _SAMPLERS:
        raise KeyError(f"unknown next-state task: {task}")
    if isinstance(rng, int):
        rng = random.Random(rng)
    elif rng is None:
        rng = random.Random()
    # Keep every successor visible and distinct within the calibrated rollout range.
    for _attempt in range(128):
        params, initial = _SAMPLERS[task](rng)
        if regime is not None:
            params, initial = extrapolate_regime(task, params, initial, regime, rng)
        trace = replay_case(task, params, initial)
        visible_state_keys = [
            _json_key(visual_matrix(task, state, params)) for state in trace
        ]
        if len(trace) - 1 <= MAX_ROLLOUT_STEPS and len(set(visible_state_keys)) == len(
            visible_state_keys
        ):
            break
    else:
        raise RuntimeError(
            f"{task}: could not sample a rollout within the "
            f"{MAX_ROLLOUT_STEPS}-successor video budget"
        )
    replayed = replay_case(task, params, initial)
    _require(_json_key(trace) == _json_key(replayed), f"{task}: replay is not deterministic")
    _require(len(trace) >= 2, f"{task}: trace has no state transition")
    _require(
        len(trace) - 1 <= MAX_ROLLOUT_STEPS,
        f"{task}: rollout exceeds video successor-state budget",
    )
    initial_matrix = visual_matrix(task, trace[0], params)
    final_matrix = visual_matrix(task, trace[-1], params)
    _require(initial_matrix and all(initial_matrix), f"{task}: empty input visual state")
    _require(final_matrix and all(final_matrix), f"{task}: empty terminal visual state")
    _require(_json_key(answer_payload(task, trace[-1], params)) != "null", f"{task}: missing terminal payload")
    evidence = _task_evidence(task, params, trace)
    return BuiltCase(
        task=task,
        params=params,
        initial_state=initial,
        trace=trace,
        answer=answer_payload(task, trace[-1], params),
        evidence=evidence,
    ).as_dict()


__all__ = [
    "BuiltCase",
    "DIRS",
    "EXPANSION_TASK_NAMES",
    "answer_payload",
    "build_case",
    "replay_case",
    "visual_matrix",
]


# Kept local to avoid a catalog import cycle at module load time.
from .catalog import EXPANSION_TASK_NAMES  # noqa: E402  (import after semantic definitions)
