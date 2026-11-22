"""Directed one-way portal chain with three mandatory transitions.

Answer space: the formal route is delivered as ordered model-facing 1-based
``(x, y)`` grid positions, where ``x`` is the row counted from the top and ``y``
is the column counted from the left, so the top-left cell is ``(1, 1)`` -- the
same numbering the board draws down its left gutter and across its top gutter.
The sampler, the BFS replay, the portal map and every pixel computation keep
working in internal 0-based ``(row, col)`` cells; the conversion happens only at
the answer-export and candidate-parsing boundaries.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    GRID_PATH_STAY_ENCODING_SENTENCE,
    cell_to_xy,
)
from data.spatial_navigation.core.maze_style import route_run_breakdown
from data.spatial_navigation.core.mechanic_base import (
    COLOR_PORTAL_A,
    COLOR_PORTAL_B,
    COLOR_PORTAL_C,
    MechanicGridGeneratorBase,
    MechanicScenario,
)


Coord = Tuple[int, int]
CARDINAL_DELTAS: Tuple[Coord, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _bfs_with_one_way_portals(
    size: int,
    start: Coord,
    goal: Coord,
    walls: Set[Coord],
    portal_map: Dict[Coord, Coord],
) -> Optional[Tuple[List[Coord], List[int]]]:
    if start == goal:
        return ([start], [])
    queue = deque([start])
    parent: Dict[Coord, Tuple[Coord, List[Coord], List[int]]] = {start: (start, [], [])}
    while queue:
        cell = queue.popleft()
        for dr, dc in CARDINAL_DELTAS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in walls:
                continue
            visited_steps: List[Coord] = [nxt]
            tps_local: List[int] = []
            rest = nxt
            if rest in portal_map:
                dest = portal_map[rest]
                if dest in walls:
                    continue
                tps_local.append(0)
                visited_steps.append(dest)
                rest = dest
            if rest in parent:
                continue
            parent[rest] = (cell, visited_steps, tps_local)
            if rest == goal:
                full: List[Coord] = [start]
                tp_offsets: List[int] = []
                trail: List[Tuple[List[Coord], List[int]]] = []
                node = rest
                while node != start:
                    prev, steps, tps = parent[node]
                    trail.append((steps, tps))
                    node = prev
                trail.reverse()
                for steps, tps in trail:
                    base = len(full) - 1
                    for off in tps:
                        tp_offsets.append(base + 1 + off)
                    for s in steps:
                        full.append(s)
                return (full, tp_offsets)
            queue.append(rest)
    return None


def _four_chambers(size: int, rng) -> Tuple[List[List[Coord]], Set[Coord]]:
    """Return four disconnected floor regions and their separating walls."""
    if size == 4:
        components = [
            [(0, 0), (0, 1)],
            [(0, 3), (1, 3)],
            [(3, 3), (3, 2)],
            [(3, 0), (2, 0)],
        ]
        transform = rng.randrange(8)

        def apply_symmetry(cell: Coord) -> Coord:
            row, col = cell
            if transform >= 4:
                col = size - 1 - col
            for _ in range(transform % 4):
                row, col = col, size - 1 - row
            return row, col

        transformed = [[apply_symmetry(cell) for cell in component] for component in components]
        floor = {cell for component in transformed for cell in component}
        walls = {(row, col) for row in range(size) for col in range(size)} - floor
        return transformed, walls

    barrier_row = rng.randint(2, size - 3)
    barrier_col = rng.randint(2, size - 3)
    walls = (
        {(barrier_row, col) for col in range(size)}
        | {(row, barrier_col) for row in range(size)}
    )
    chambers = [
        [(row, col) for row in range(0, barrier_row) for col in range(0, barrier_col)],
        [(row, col) for row in range(0, barrier_row) for col in range(barrier_col + 1, size)],
        [(row, col) for row in range(barrier_row + 1, size) for col in range(barrier_col + 1, size)],
        [(row, col) for row in range(barrier_row + 1, size) for col in range(0, barrier_col)],
    ]
    return chambers, walls


def _far_endpoint_pair(cells: Sequence[Coord], rng) -> Tuple[Coord, Coord]:
    """Pick a well-separated (but not always maximal) chamber endpoint pair.

    Always taking the maximal-distance pair made every reference route exactly
    the same total length, a template constancy the sweep flagged; sampling
    within two steps of the maximum keeps legs long while varying lengths.
    """

    pairs = [
        (first, second)
        for first in cells
        for second in cells
        if first != second
    ]
    max_distance = max(
        abs(first[0] - second[0]) + abs(first[1] - second[1])
        for first, second in pairs
    )
    floor = max(1, max_distance - 2)
    candidates = [
        (first, second)
        for first, second in pairs
        if abs(first[0] - second[0]) + abs(first[1] - second[1]) >= floor
    ]
    return rng.choice(candidates)


def _portal_sources_used(path: Sequence[Coord], teleport_steps: Iterable[int]) -> List[Coord]:
    return [path[int(step)] for step in teleport_steps]


class OneWayPortalChainGenerator(MechanicGridGeneratorBase):
    TASK_NAME = "one_way_portal_chain"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "One-Way Portal Chain"
    DEFAULT_SIZE = 7

    def _authored_reasoning(self, scenario: MechanicScenario) -> str:
        path = list(scenario.agent_path)
        pairs = [
            (tuple(entry["source"]), tuple(entry["destination"]))
            for entry in scenario.extras["portal_pairs"]
        ]
        teleports = [int(step) for step in scenario.extras["teleport_steps"]]
        sentences: List[str] = [
            "The slate barriers cut the floor into four sealed chambers, so walking alone can never leave the "
            f"chamber holding S at {cell_to_xy(path[0])}. The only exits are the one-way portals: "
            + "; ".join(
                f"S{index} at {cell_to_xy(source)} jumps to D{index} at {cell_to_xy(dest)}"
                for index, (source, dest) in enumerate(pairs, start=1)
            )
            + ". Chamber by chamber, that forces the order S1, then S2, then S3.",
        ]
        anchor = 0
        for index, teleport_step in enumerate(teleports, start=1):
            runs = route_run_breakdown(path[anchor : teleport_step + 1])
            leg = ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in runs)
            sentences.append(
                f"Inside its current chamber the route walks {leg} into S{index} at "
                f"{cell_to_xy(path[teleport_step])}, and the portal instantly places the same agent at D{index}, "
                f"{cell_to_xy(path[teleport_step + 1])}."
            )
            anchor = teleport_step + 1
        tail = route_run_breakdown(path[anchor:])
        if tail:
            leg = ", then ".join(f"{direction} {len(cells) - 1}" for direction, cells in tail)
            sentences.append(f"From D3 it walks {leg} and stops on G at {cell_to_xy(path[-1])}.")
        else:
            sentences.append(f"D3 is already G at {cell_to_xy(path[-1])}.")
        sentences.append(
            "Each source appears immediately before its destination in the trajectory and the destinations "
            "trigger nothing, so all three one-way jumps are legal and the chain order is respected."
        )
        return " ".join(sentences)

    def sample_scenario(self) -> MechanicScenario:
        for _ in range(128):
            chambers, walls = _four_chambers(self.size, self.rng)
            self.rng.shuffle(chambers)
            endpoints = [_far_endpoint_pair(chamber, self.rng) for chamber in chambers]
            start, src1 = endpoints[0]
            dst1, src2 = endpoints[1]
            dst2, src3 = endpoints[2]
            dst3, goal = endpoints[3]
            srcs = [src1, src2, src3]
            dsts = [dst1, dst2, dst3]
            portal_map = dict(zip(srcs, dsts))
            result = _bfs_with_one_way_portals(self.size, start, goal, walls, portal_map)
            if result is None:
                continue
            full_path, teleports = result
            if _portal_sources_used(full_path, teleports) != srcs:
                continue
            if len(teleports) != len(portal_map):
                continue

            # Portal-free movement cannot cross the chamber barriers. Removing
            # any single source transition must likewise destroy reachability.
            if _bfs_with_one_way_portals(self.size, start, goal, walls, {}) is not None:
                continue
            indispensable = True
            for source in srcs:
                reduced_map = dict(portal_map)
                reduced_map.pop(source)
                if _bfs_with_one_way_portals(self.size, start, goal, walls, reduced_map) is not None:
                    indispensable = False
                    break
            if not indispensable:
                continue

            return self._build_scenario(
                start=start, goal=goal,
                portal_pairs=[(srcs[i], dsts[i]) for i in range(3)],
                walls=sorted(walls), path=full_path, teleports=teleports,
            )
        raise RuntimeError("one_way_portal_chain: failed to sample scenario.")

    def _build_scenario(self, *, start, goal, portal_pairs, walls, path, teleports):
        return MechanicScenario(
            task_name=self.TASK_NAME,
            task_family=self.TASK_FAMILY,
            title=self.TASK_TITLE,
            input_desc=(
                f"The image shows a {self.size}x{self.size} grid with row numbers down the left edge and column "
                "numbers across the top edge. "
                "Green S is the start, blue G is the goal, three filled-disc cells with thicker rings are portal "
                "sources labeled S1, S2, S3, three lighter-ring cells labeled D1, D2, D3 are their matching "
                "destinations, and dark slate barriers split the floor into four chambers. "
                "Each colored pair is one-way: stepping into Sk teleports to Dk; stepping into Dk does nothing. "
                f"{GRID_XY_CONVENTION_SENTENCE}"
            ),
            rules_desc=(
                "The agent may move up, right, down, or left to an adjacent floor cell, or stay on its current "
                "resting cell for one time step. The chamber chain can be crossed only by entering S1, then S2, "
                "then S3. Entering source Sk instantly places the same agent at matching destination Dk, so the "
                "source and destination must appear consecutively in the route. Destination cells are ordinary "
                f"floor with no further teleportation. Walls and the grid border block movement. "
                f"{GRID_PATH_STAY_ENCODING_SENTENCE} A source position cannot be repeated, because "
                "its teleport happens immediately."
            ),
            ti2v_output=(
                "Generate a video that shows one complete legal route through S1 to D1, S2 to D2, and S3 to D3 "
                "before reaching G. Show each teleport as a source-contact snapshot immediately followed by the "
                "matching destination-arrival snapshot, with the same agent identity and both complete portal glyphs "
                "remaining visible. Then resume walking. If the route waits, hold the unchanged resting state for "
                "one additional time step. Hold every semantic state for the same duration; do not use sub-cell "
                "interpolation, fades, opacity changes, or undeclared transition states. Entering Dk never activates "
                "a teleport."
            ),
            solution_image_output=(
                "Render the solved still by drawing one complete legal route in red through all three source portals "
                "in numbered order, with the path on each side of every teleport drawn but no line connecting the "
                "two portal endpoints across the teleport."
            ),
            size=self.size,
            start=start,
            goal=goal,
            agent_path=path,
            walls=list(walls),
            extras={
                "portal_pairs": [
                    {"source": [s[0], s[1]], "destination": [d[0], d[1]], "label": f"P{i + 1}"}
                    for i, (s, d) in enumerate(portal_pairs)
                ],
                "required_portal_order": ["P1", "P2", "P3"],
                "teleport_steps": list(teleports),
            },
            structured_answer={
                # Reference-only mirror of the formal visible coordinate route.
                "agent_path_xy": [list(cell_to_xy(cell)) for cell in path],
                "required_portal_order": ["P1", "P2", "P3"],
                "teleport_steps": list(teleports),
            },
        )

    def _render(
        self,
        scenario: MechanicScenario,
        *,
        agent_center: Tuple[float, float],
        show_path: bool = False,
    ) -> Image.Image:
        image, draw = self._begin_scene()
        wall_set = set(scenario.walls)
        self._draw_grid_and_labels(draw, wall_set=wall_set)
        token_font = self._load_font(max(11, self.cell_size // 6))

        colors = [COLOR_PORTAL_A, COLOR_PORTAL_B, COLOR_PORTAL_C]
        for i, entry in enumerate(scenario.extras["portal_pairs"]):
            src = tuple(entry["source"])
            dst = tuple(entry["destination"])
            self._draw_portal(draw, src, color=colors[i], label=f"S{i + 1}", token_font=token_font)
            self._draw_portal_dest(draw, dst, color=colors[i], label=f"D{i + 1}", token_font=token_font)

        if show_path:
            teleports = scenario.extras["teleport_steps"]
            self._draw_path_overlay(draw, scenario.agent_path, teleport_breaks=teleports)

        self._draw_start_goal(draw, scenario)
        self._draw_agent(draw, agent_center)
        return self._finish_scene(image)

    def _draw_portal_dest(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
        *,
        color: Tuple[int, int, int],
        label: str,
        token_font,
    ) -> None:
        cx, cy = self._cell_center(cell)
        outer = self.cell_size * 0.34
        from data.common.prompt_paraphrase import active_profile
        if active_profile("portal_gates"):
            shade=tuple(min(255,c+25) for c in color)
            draw.polygon([(cx-outer,cy),(cx-outer*.3,cy-outer*.7),(cx+outer,cy-outer*.7),(cx+outer,cy+outer*.7),(cx-outer*.3,cy+outer*.7)],fill=(248,245,220),outline=color,width=3)
            draw.line((cx-outer*.4,cy,cx+outer*.5,cy),fill=color,width=3)
            draw.line((cx+outer*.2,cy-outer*.3,cx+outer*.5,cy,cx+outer*.2,cy+outer*.3),fill=color,width=3)
        else:
            # 2 light rings + dashed core
            for k, factor in enumerate((1.0, 0.78)):
                r = outer * factor
                shade = tuple(min(255, c + 30 * k) for c in color)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=shade, width=2)
            # dashed cross hatch (destination markings)
            s = outer * 0.45
            draw.line([(cx - s, cy), (cx + s, cy)], fill=color, width=2)
            draw.line([(cx, cy - s), (cx, cy + s)], fill=color, width=2)
        draw.text((cx, cy - outer - 6), label, fill=(30, 30, 30), font=token_font, anchor="mm")

    def create_puzzle(self) -> Dict[str, Any]:  # type: ignore[override]
        scenario = self.sample_scenario()
        puzzle_image = self._render(scenario, agent_center=self._cell_center(scenario.start))
        solution_image = self._render(scenario, agent_center=self._cell_center(scenario.goal), show_path=True)

        def render_frame(agent_cell, agent_center, step_index):
            return self._render(scenario, agent_center=agent_center)

        teleport_steps = list(scenario.extras["teleport_steps"])
        video_frames = self._animate_path(scenario, render_frame, teleport_steps=teleport_steps)
        return self._assemble_record(
            scenario, puzzle_image=puzzle_image, solution_image=solution_image, video_frames=video_frames,
        )


__all__ = ["OneWayPortalChainGenerator"]
