"""Native grid mechanics composed in a source-only discrete transition system."""

from copy import deepcopy
from functools import lru_cache
import heapq, random, tempfile
from types import SimpleNamespace
import numpy as np
from data.spatial_navigation.core.mechanic_base import (
    COLOR_PORTAL_A,
    COLOR_PORTAL_B,
    COLOR_PORTAL_C,
)
from data.spatial_navigation.bridge_drop_tile.generator import BridgeDropTileGenerator
from data.spatial_navigation.conveyor_belt_path.generator import (
    ConveyorBeltPathGenerator,
)
from data.spatial_navigation.one_way_portal_chain.generator import (
    OneWayPortalChainGenerator,
)
from data.spatial_navigation.directional_arrow_entry.generator import (
    DirectionalArrowEntryGenerator,
)
from data.spatial_navigation.core.verifier import _track_tokens
from evaluation.shared.scenes.catalog import TASKS

DIRECTIONS = ("up", "right", "down", "left")
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CLASSES = {
    "bridge_drop_tile": BridgeDropTileGenerator,
    "conveyor_belt_path": ConveyorBeltPathGenerator,
    "one_way_portal_chain": OneWayPortalChainGenerator,
    "directional_arrow_entry": DirectionalArrowEntryGenerator,
}


@lru_cache(maxsize=8)
def board(size):
    return BridgeDropTileGenerator(
        output_dir=tempfile.gettempdir() + "/videogym-native-grid-renderer",
        size=size,
        video=False,
    )


def parts(s):
    e = s["extras"]
    return (
        set(map(tuple, s["walls"])),
        {tuple(v): i for i, v in enumerate(e.get("water", ()))},
        {
            tuple(v["cell"]): DIRECTIONS.index(v["direction"])
            for v in e.get("conveyors", ())
        },
        {
            tuple(v["cell"]): DIRECTIONS.index(v["direction"])
            for v in e.get("arrows", ())
        },
        {
            tuple(v["source"]): (i, tuple(v["destination"]))
            for i, v in enumerate(e.get("portal_pairs", ()))
        },
    )


def active(s, kind):
    if s["variant"] == "main":
        return True
    host = s["host"] if s["variant"] == "source_goal" else TASKS[s["key"]].primitives[1]
    return (
        kind
        == {
            "bridge_drop_tile": "water",
            "one_way_portal_chain": "portals",
            "conveyor_belt_path": "belts",
            "directional_arrow_entry": "arrows",
        }[host]
    )


def initial(s):
    return (*s["start"], 0, 0, 0, -1)


def successors(s, state):
    walls, water, belts, arrows, portals = parts(s)
    r, c, mask, stage, pending, last = state
    size = s["size"]

    def enter(dest, direction):
        if not (0 <= dest[0] < size and 0 <= dest[1] < size) or dest in walls:
            return None
        if active(s, "arrows") and dest in arrows and (direction != arrows[dest]):
            return None
        newmask = mask
        newstage = stage
        next_pending = 0
        cost = 0
        if active(s, "water") and dest in water and (not mask >> water[dest] & 1):
            if mask.bit_count() >= s["extras"]["inventory"]:
                return None
            newmask |= 1 << water[dest]
        if active(s, "portals") and dest in portals:
            i, _ = portals[dest]
            if i != stage:
                return None
            next_pending = 20 + i
        elif active(s, "arrows") and dest in arrows:
            next_pending = 11 + arrows[dest]
        elif active(s, "belts") and dest in belts:
            next_pending = 1 + belts[dest]
        newlast = last
        return ((*dest, newmask, newstage, next_pending, newlast), cost)

    if pending >= 20:
        entry = next(
            ((src, i, dst) for src, (i, dst) in portals.items() if i == pending - 20),
            None,
        )
        if entry is None:
            return []
        _, i, dst = entry
        return [((*dst, mask, stage + 1, 0, last), 0)]
    if pending:
        arrow = pending >= 11
        direction = pending - 11 if arrow else pending - 1
        dr, dc = DELTAS[direction]
        dest = (r + dr, c + dc)
        if arrow and dest in arrows:
            return []
        result = enter(dest, direction)
        if result:
            return [result]
        if arrow:
            return []
    out = []
    for direction, (dr, dc) in enumerate(DELTAS):
        result = enter((r + dr, c + dc), direction)
        if result:
            out.append(result)
    return out


def goal(s, state):
    return (
        tuple(state[:2]) == tuple(s["goal"])
        and state[4] == 0
        and (
            not active(s, "portals")
            or state[3] == len(s["extras"].get("portal_pairs", ()))
        )
    )


def solve(s, required_mask=None):
    start = initial(s)
    best = {start: (0, 0)}
    parents = {start: None}
    queue = [(0, 0, start)]
    expanded = 0
    while queue:
        primary, length, state = heapq.heappop(queue)
        if best.get(state) != (primary, length):
            continue
        if goal(s, state) and (required_mask is None or state[2] == required_mask):
            path = []
            node = state
            while node is not None:
                path.append(node)
                node = parents[node]
            return {
                "states": path[::-1],
                "cost": primary,
                "steps": length,
                "expanded": expanded,
            }
        for nxt, added in successors(s, state):
            if required_mask is not None and nxt[2] & ~required_mask:
                continue
            cost = (primary + added, length + 1)
            if cost < best.get(nxt, (10**9, 10**9)):
                best[nxt] = cost
                parents[nxt] = state
                heapq.heappush(queue, (*cost, nxt))
        expanded += 1
        if expanded > 160000:
            return None
    return None


def _public_scenario(scenario):
    keep = {
        "water",
        "inventory",
        "conveyors",
        "arrows",
        "portal_pairs",
        "required_portal_order",
    }
    return {
        "size": scenario.size,
        "start": list(scenario.start),
        "goal": list(scenario.goal),
        "walls": [list(p) for p in sorted(scenario.walls)],
        "extras": {k: deepcopy(v) for k, v in scenario.extras.items() if k in keep},
    }


def path_legal(s, states):
    previous = initial(s)
    for item in states[1:]:
        choices = [
            n for n, _ in successors(s, previous) if tuple(n[:2]) == tuple(item[:2])
        ]
        if not choices:
            return False
        previous = choices[0]
    return goal(s, previous)


def sample(key, seed, variant="main"):
    task = TASKS[key]
    rng = random.Random(seed)
    for attempt in range(200):
        host = CLASSES[task.host](
            output_dir=tempfile.gettempdir() + "/videogym-native-grid-sampling",
            seed=rng.randrange(2**31),
            video=False,
        )
        original = host.sample_scenario()
        s = _public_scenario(original)
        s.update(key=key, host=task.host, variant="main")
        if task.host == "one_way_portal_chain":
            sr, sc = s["start"]
            gr, gc = s["goal"]
            corridor = {(sr, c) for c in range(min(sc, gc), max(sc, gc) + 1)}
            corridor.update(((r, gc) for r in range(min(sr, gr), max(sr, gr) + 1)))
            s["walls"] = [p for p in s["walls"] if tuple(p) not in corridor]
        if key in (
            "bridge_conveyor",
            "portal_conveyor",
            "portal_arrow",
            "bridge_arrow",
        ):
            mechanic = "conveyors" if key.endswith("conveyor") else "arrows"
            reserved = {tuple(s["start"]), tuple(s["goal"])} | set(
                map(tuple, s["extras"].get("water", ()))
            )
            for portal in s["extras"].get("portal_pairs", ()):
                reserved.update((tuple(portal["source"]), tuple(portal["destination"])))
            candidates = [
                (r, c)
                for r in range(s["size"])
                for c in range(s["size"])
                if [r, c] not in s["walls"] and (r, c) not in reserved
            ]
            rng.shuffle(candidates)
            on_path = set(map(tuple, original.agent_path))
            candidates.sort(key=lambda p: p not in on_path)
            found = False
            for cell in candidates[:25]:
                for direction in rng.sample(list(DIRECTIONS), 4):
                    s["extras"][mechanic] = [
                        {"cell": list(cell), "direction": direction}
                    ]
                    full = solve(s)
                    if not full or len(full["states"]) > 35:
                        continue
                    without = dict(s, variant="source_goal")
                    primitive = solve(without)
                    if not primitive:
                        continue
                    changed = not path_legal(s, primitive["states"])
                    if not changed:
                        continue
                    found = True
                    break
                if found:
                    break
            if not found:
                continue
        full = solve(s)
        if not full or not 6 <= len(full["states"]) <= 35:
            continue
        from evaluation.shared.scenes.ablation import evidence

        if not evidence(s)["passed"]:
            continue
        s["variant"] = variant
        return s
    raise RuntimeError(key + ": non-vacuous native navigation sampling exhausted")


def render(s, state=None, **_):
    g = board(s["size"])
    state = initial(s) if state is None else state
    image, draw = g._begin_scene()
    walls, water, belts, arrows, portals = parts(s)
    g._draw_grid_and_labels(draw, wall_set=walls)
    for p in water:
        g._draw_water(draw, p)
    for p, index in water.items():
        if state[2] >> index & 1:
            g._draw_bridge_tile(draw, p)
    for p, d in belts.items():
        g._draw_conveyor(draw, p, direction=DIRECTIONS[d])
    for p, d in arrows.items():
        DirectionalArrowEntryGenerator._draw_arrow_gate(
            g, draw, p, direction=DIRECTIONS[d]
        )
    font = g._load_font(max(11, g.cell_size // 6))
    colors = (COLOR_PORTAL_A, COLOR_PORTAL_B, COLOR_PORTAL_C)
    for p, (index, dst) in portals.items():
        g._draw_portal(
            draw, p, color=colors[index], label="S" + str(index + 1), token_font=font
        )
        OneWayPortalChainGenerator._draw_portal_dest(
            g,
            draw,
            dst,
            color=colors[index],
            label="D" + str(index + 1),
            token_font=font,
        )
    g._draw_start_goal(
        draw, SimpleNamespace(start=tuple(s["start"]), goal=tuple(s["goal"]))
    )
    g._draw_agent(draw, g._cell_center(tuple(state[:2])))
    if water:
        inventory = s["extras"]["inventory"]
        g._draw_inventory_gauge(
            draw,
            filled=inventory - state[2].bit_count(),
            total=inventory,
            fill_color=(172, 130, 78),
            outline_color=(115, 80, 40),
        )
    return g._finish_scene(image)


def prompt(s):
    task = TASKS[s["key"]]
    rule = task.rule
    if s["variant"] == "source_goal":
        rule = (
            "Reach G by any complete legal route under the "
            + s["host"].replace("_", " ")
            + " rule. Additional mechanic icons are inert in this control."
        )
    mechanics = []
    if s["extras"].get("water"):
        mechanics.append(
            "Each first entry to an unbridged blue water cell places a permanent brown plank and spends one inventory pip. No new water entry is possible with zero pips; a placed bridge can be reused."
        )
    if s["extras"].get("conveyors"):
        mechanics.append(
            "Entering a beige conveyor arrow immediately forces one step along its arrow, continuing through a chain. A shove blocked by a wall or border stops."
        )
    if s["extras"].get("arrows"):
        mechanics.append(
            "Enter an orange arrow gate only while moving in its arrow direction, then immediately leave one further cell in that direction."
        )
    if s["extras"].get("portal_pairs"):
        mechanics.append(
            "Entering S1, S2 or S3 immediately jumps to the paired D1, D2 or D3. Destinations trigger no return jump. Use the three sources in numbered order."
        )
    if s["variant"] == "source_goal":
        mechanics = [
            text
            for text in mechanics
            if s["host"] == "bridge_drop_tile"
            and text.startswith("Each first")
            or (
                s["host"] == "conveyor_belt_path"
                and text.startswith("Entering a beige")
            )
            or (
                s["host"] == "directional_arrow_entry"
                and text.startswith("Enter an orange")
            )
            or (s["host"] == "one_way_portal_chain" and text.startswith("Entering S1"))
        ]
    return (
        "Input Image: Native numbered "
        + str(s["size"])
        + " by "
        + str(s["size"])
        + " grid. Green S is start, blue G is goal, dark slate cells are walls; the shaded orange disc is the agent.\n\nRule: "
        + rule
        + " "
        + " ".join(mechanics)
        + " Moves use side-adjacent grid cells. Never enter walls or leave the board.\n\nRequired Output: Show the native agent moving through exact integer-cell states, including every forced move and both ends of a portal jump. Commit each bridge, its inventory pip and agent entry together. Keep all static labels and icons fixed. Hold the initial and solved states. No connecting line across portals, interpolated movement, extra agent, hard-cut solution, skipped update or camera motion."
    )


def grade(s, path):
    from evaluation.shared.task_io import read_video, matches
    from data.common.verifiers.keyframes import visual_delta

    frames = read_video(path)
    tracks, metrics = _track_tokens(
        s["host"],
        {"grid_size": s["size"], "start": s["start"], "extras": s["extras"]},
        frames,
    )
    aligned = metrics.pop("_video_frame_token_tracks")["primary"]
    detections = metrics.pop("_video_frame_token_detections")["primary"]
    metrics.pop("_video_frame_token_centers", None)
    current = initial(s)
    cost = 0
    valid = True
    first_error = None
    cache = {}
    g = board(s["size"])
    for i, (frame, cell, detected) in enumerate(zip(frames, aligned, detections)):
        if not detected or cell is None:
            valid = False
            first_error = i
            break
        if tuple(cell) != tuple(current[:2]):
            options = [
                (n, c) for n, c in successors(s, current) if tuple(n[:2]) == tuple(cell)
            ]
            if not options:
                valid = False
                first_error = i
                break
            current, added = options[0]
            cost += added
        if current not in cache:
            cache[current] = np.asarray(render(s, current))
        if not matches(frame, cache[current]):
            valid = False
            first_error = i
            break
        if s["extras"].get("water"):
            for box in g._gauge_pip_boxes(s["extras"]["inventory"]):
                x0, y0, x1, y1 = map(lambda x: int(round(x)), box)
                if (
                    float(
                        visual_delta(
                            frame[y0 : y1 + 1, x0 : x1 + 1],
                            cache[current][y0 : y1 + 1, x0 : x1 + 1],
                        ).mean()
                    )
                    > 12
                ):
                    valid = False
                    first_error = i
                    break
        if not valid:
            break
    final = False
    if (
        frames
        and detections[-1]
        and (aligned[-1] is not None)
        and (tuple(aligned[-1]) == tuple(s["goal"]))
    ):
        _, water, _, _, _ = parts(s)
        mask = 0
        stage = len(s["extras"].get("portal_pairs", ())) if active(s, "portals") else 0
        dry = np.asarray(render(s, (*s["goal"], 0, stage, 0, -1)))
        clear = True
        for cell, index in water.items():
            x0, y0, x1, y1 = g._cell_box(cell)
            tile = (slice(y0, y1 + 1), slice(x0, x1 + 1))
            wet = np.asarray(render(s, (*s["goal"], 1 << index, stage, 0, -1)))
            de = float(visual_delta(frames[-1][tile], dry[tile]).mean())
            we = float(visual_delta(frames[-1][tile], wet[tile]).mean())
            if min(de, we) > 12:
                clear = False
                break
            if we < de:
                mask |= 1 << index
        if clear and mask.bit_count() <= s["extras"].get("inventory", 0):
            witness = solve(s, required_mask=mask)
            if witness and matches(
                frames[-1], np.asarray(render(s, witness["states"][-1]))
            ):
                final = True
    optimal = solve(s)
    optimality = bool(optimal)
    process = (
        valid
        and final
        and goal(s, current)
        and optimality
        and (aligned[0] == tuple(s["start"]))
    )
    return dict(
        whole_video_loaded=True,
        whole_video_process_correct=float(process),
        video_final_state_correct=float(final),
        first_invalid_frame=first_error,
        observed_objective=cost,
        optimal_objective=optimal["cost"] if optimal else None,
        objective_optimal=optimality,
        **metrics,
    )
