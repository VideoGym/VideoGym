"""Slitherlink completion: draw a single closed loop on the grid lines so each numbered cell has
exactly that many of its four sides on the loop."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

from data.constraint_grid.core.forced_deduction import (
    ForcedChain,
    ForcedDeductionError,
    ForcedMove,
    encode_commit_video,
    group_paragraph,
    merge_rounds,
    prove_complete_chain,
)
from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.constraint_grid.core.coordinate_format import display_cell_text, display_edge_text
from data.constraint_grid.core.scene import (
    AXIS_LABEL,
    TEXT,
    board_font,
    finish_board_scene,
    new_board_scene,
)
from data.common.contracts.grid_completion import answer_xy_composites
from data.common.pipeline.grid_rendering import fit_canvas

_INPUT_DESCRIPTION = (
    "The image shows a lattice of dots with small gray numbers along the top and left edges labeling "
    "the corner columns and corner rows from 1. Some cells hold a digit, and red segments already "
    "drawn are part of the final loop."
)
_RULE_DESCRIPTION = (
    "Draw one closed non-crossing loop along the dotted lattice lines. A digit in a cell counts "
    "exactly how many of that cell's four sides the loop uses; cells without a digit are free. The "
    "loop is a single piece that never branches, and the segments already drawn are fixed."
)
_TERMINAL_VISUAL = (
    "the finished board on which the complete loop is drawn in red while the dots, digits, and edge "
    "numbers stay unchanged"
)
_BG = (250, 251, 254)
_GRID_LIGHT = (210, 215, 225)
_GRID_BORDER = (40, 44, 52)
_TEXT = (20, 24, 32)
_LOOP = (210, 50, 60)
_DOT = (90, 96, 108)


@dataclass
class SlitherlinkCompletionRecord:
    id: str
    task_name: str = "slitherlink_completion"
    task_family: str = "constraint_grid"
    image: str = ""
    reasoning_image: str = ""
    solution_image_path: str = ""
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None
    ti2v_prompt: str = ""
    solution_image_prompt: str = ""
    structured_answer: Dict[str, Any] = field(default_factory=dict)

    answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "answer": self.answer,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "image": self.image,
            "reasoning_image": self.reasoning_image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "ti2v_prompt": self.ti2v_prompt,
            "structured_answer": dict(self.structured_answer),
        }


class SlitherlinkCompletionGenerator(AbstractPuzzleGenerator[SlitherlinkCompletionRecord]):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        rows: int = 5,
        cols: int = 5,
        video: bool = False,
        video_fps: int = 16,
    ) -> None:
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.seed = seed
        self.rows = rows
        self.cols = cols
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self._counter = 0
        self.puzzle_dir = self.output_dir / "puzzles"
        self.reasoning_dir = self.output_dir / "reasoning"
        self.solution_dir = self.output_dir / "solutions"
        for directory in (self.puzzle_dir, self.reasoning_dir, self.solution_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _next_id(self) -> str:
        puzzle_id = f"slitherlink_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    def create_puzzle(self, *args, **kwargs) -> SlitherlinkCompletionRecord:
        return self.create_random_puzzle()

    # ------------------------------------------------------------------
    # Construct loop first, derive clue numbers from inside/outside cells
    # ------------------------------------------------------------------

    def _build_loop_via_inside_set(self) -> Tuple[List[List[int]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Pick a connected, simply connected set of cells (the inside of the loop). The loop is
        the boundary between inside and outside. Each clue cell counts how many of its four sides
        cross from inside to outside (or to grid edge). We choose the inside set to be a connected
        region whose complement is also connected (so the loop is simple)."""
        rows, cols = self.rows, self.cols
        for _ in range(800):
            inside = [[0 for _ in range(cols)] for _ in range(rows)]
            # Random walk-style fill: start with a random rectangle, then add or remove single cells.
            sr, sc = self.rng.randint(1, rows - 2), self.rng.randint(1, cols - 2)
            inside[sr][sc] = 1
            target = self.rng.randint(rows * cols // 4, rows * cols // 2)
            for _ in range(rows * cols * 4):
                # Add an adjacent cell.
                cur = sum(sum(r) for r in inside)
                if cur >= target:
                    break
                expansions = []
                for r in range(rows):
                    for c in range(cols):
                        if inside[r][c] == 1:
                            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols and inside[nr][nc] == 0:
                                    expansions.append((nr, nc))
                if not expansions:
                    break
                pick = self.rng.choice(expansions)
                inside[pick[0]][pick[1]] = 1
                # Maintain inside-connected; complement-connected; no holes.
                if not self._is_simply_connected(inside):
                    inside[pick[0]][pick[1]] = 0
            inside_count = sum(sum(r) for r in inside)
            if inside_count < 3:
                continue
            if not self._is_simply_connected(inside):
                continue
            # Build edge segments along the boundary.
            edges = self._boundary_edges(inside)
            # Verify it's a single loop.
            if not self._is_single_loop(edges):
                continue
            return inside, edges
        raise RuntimeError("Failed to build slitherlink loop")

    def _is_simply_connected(self, inside: List[List[int]]) -> bool:
        """The inside is connected and its complement (with the outside being everything outside the rectangle)
        treated as one connected region — i.e., no inside cell encloses an outside hole."""
        rows, cols = self.rows, self.cols
        # Inside connectivity.
        cells = [(r, c) for r in range(rows) for c in range(cols) if inside[r][c] == 1]
        if not cells:
            return False
        seen = {cells[0]}
        stack = [cells[0]]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and inside[nr][nc] == 1 and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        if len(seen) != len(cells):
            return False
        # Outside connectivity: include the implicit "outer" frame as one component.
        # Imagine a (rows+2)x(cols+2) padding. All outside cells (incl padding) must be one component.
        out_seen: Set[Tuple[int, int]] = set()
        # Start from (-1, -1) padding cell.
        start = (-1, -1)
        out_seen.add(start)
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if -1 <= nr <= rows and -1 <= nc <= cols and (nr, nc) not in out_seen:
                    if nr in (-1, rows) or nc in (-1, cols):
                        # padding outside grid
                        out_seen.add((nr, nc))
                        stack.append((nr, nc))
                    elif inside[nr][nc] == 0:
                        out_seen.add((nr, nc))
                        stack.append((nr, nc))
        # Check that every grid cell with inside==0 is in out_seen.
        for r in range(rows):
            for c in range(cols):
                if inside[r][c] == 0 and (r, c) not in out_seen:
                    return False
        return True

    def _boundary_edges(self, inside: List[List[int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return edge segments between inside and outside cells. Each segment is ((r1,c1),(r2,c2))
        in lattice coordinates."""
        rows, cols = self.rows, self.cols
        edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for r in range(rows):
            for c in range(cols):
                if inside[r][c] == 1:
                    # Top edge: between (r,c) and (r-1,c).
                    upper_outside = (r == 0) or inside[r - 1][c] == 0
                    if upper_outside:
                        edges.append(((r, c), (r, c + 1)))
                    lower_outside = (r == rows - 1) or inside[r + 1][c] == 0
                    if lower_outside:
                        edges.append(((r + 1, c), (r + 1, c + 1)))
                    left_outside = (c == 0) or inside[r][c - 1] == 0
                    if left_outside:
                        edges.append(((r, c), (r + 1, c)))
                    right_outside = (c == cols - 1) or inside[r][c + 1] == 0
                    if right_outside:
                        edges.append(((r, c + 1), (r + 1, c + 1)))
        # Deduplicate (each edge is unique by construction since each side is visited exactly once
        # per inside cell pair).
        seen = set()
        unique = []
        for a, b in edges:
            key = tuple(sorted([a, b]))
            if key not in seen:
                seen.add(key)
                unique.append((a, b))
        return unique

    def _is_single_loop(self, edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> bool:
        # Each vertex must have degree 2; the graph must be one connected component.
        adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for a, b in edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        for v, neighbors in adj.items():
            if len(neighbors) != 2:
                return False
        # Check connectivity.
        if not adj:
            return False
        start = next(iter(adj))
        seen = {start}
        stack = [start]
        while stack:
            v = stack.pop()
            for u in adj[v]:
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        return len(seen) == len(adj)

    def _cell_clue(self, inside: List[List[int]], r: int, c: int) -> int:
        """Number of edges around cell (r,c) that are on the loop.

        A loop edge is the boundary between an inside cell and an outside cell
        (or between an inside cell and the area outside the grid). Treat the
        area outside the grid as "outside" (value 0) and count edges where the
        two sides differ. If cell (r, c) is itself outside, an edge to the grid
        border is between two outside areas and is NOT a loop edge.
        """
        rows, cols = self.rows, self.cols
        cur = inside[r][c]
        count = 0
        # Top.
        other = inside[r - 1][c] if r > 0 else 0
        count += int(cur != other)
        # Bottom.
        other = inside[r + 1][c] if r < rows - 1 else 0
        count += int(cur != other)
        # Left.
        other = inside[r][c - 1] if c > 0 else 0
        count += int(cur != other)
        # Right.
        other = inside[r][c + 1] if c < cols - 1 else 0
        count += int(cur != other)
        return count

    def _build_instance(self) -> Tuple[List[List[int]], List[List[Optional[int]]],
                                       List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        rows, cols = self.rows, self.cols
        inside, edges = self._build_loop_via_inside_set()
        # Compute clue numbers and randomly hide most.
        full_clues = [[self._cell_clue(inside, r, c) for c in range(cols)] for r in range(rows)]
        # Hide ~50% to make a meaningful puzzle.
        clues: List[List[Optional[int]]] = [[None for _ in range(cols)] for _ in range(rows)]
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        self.rng.shuffle(cells)
        keep = max(rows * cols // 2, rows + cols)
        for r, c in cells[:keep]:
            clues[r][c] = full_clues[r][c]
        return inside, clues, edges

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    def _all_edges(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        rows, cols = self.rows, self.cols
        edges = []
        for r in range(rows + 1):
            for c in range(cols):
                edges.append(((r, c), (r, c + 1)))
        for r in range(rows):
            for c in range(cols + 1):
                edges.append(((r, c), (r + 1, c)))
        return edges

    @staticmethod
    def _edge_phrase(edge: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
        return f"the corner segment {display_edge_text(edge[0], edge[1])}"

    def _forced_chain(
        self,
        solution_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
        clues: Sequence[Sequence[Optional[int]]],
        given: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    ) -> Optional[ForcedChain]:
        rows, cols = self.rows, self.cols
        edges = self._all_edges()
        state: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Optional[bool]] = {
            edge: None for edge in edges
        }
        for edge in given:
            state[edge] = True
        cell_sides = {
            (r, c): [
                tuple(sorted(((r, c), (r, c + 1)))),
                tuple(sorted(((r + 1, c), (r + 1, c + 1)))),
                tuple(sorted(((r, c), (r + 1, c)))),
                tuple(sorted(((r, c + 1), (r + 1, c + 1)))),
            ]
            for r in range(rows)
            for c in range(cols)
        }
        corner_edges: Dict[Tuple[int, int], List] = {}
        for edge in edges:
            for corner in edge:
                corner_edges.setdefault(corner, []).append(edge)
        pending: List[str] = []
        rounds: List[List[ForcedMove]] = []

        def components() -> Tuple[Dict[Tuple[int, int], int], Dict[int, Set[Tuple[int, int]]]]:
            parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for edge, value in state.items():
                if value is True:
                    for corner in edge:
                        parent.setdefault(corner, corner)
            for edge, value in state.items():
                if value is True:
                    a, b = find(edge[0]), find(edge[1])
                    if a != b:
                        parent[a] = b
            groups: Dict[Tuple[int, int], int] = {}
            ids: Dict[Tuple[int, int], int] = {}
            for corner in parent:
                root = find(corner)
                ids.setdefault(root, len(ids))
                groups[corner] = ids[root]
            return groups, ids

        while True:
            on_count = sum(1 for value in state.values() if value is True)
            if on_count == len(solution_edges) and all(
                (state[edge] is True) == (edge in solution_edges) or state[edge] is not True
                for edge in edges
            ) and {edge for edge, value in state.items() if value is True} == solution_edges:
                return ForcedChain(rounds=rounds)

            forced_on: Dict[Tuple, str] = {}
            new_off: List[Tuple[Tuple, str]] = []

            for (r, c), sides in cell_sides.items():
                clue = clues[r][c]
                if clue is None or clue < 0:
                    continue
                on = sum(1 for side in sides if state[side] is True)
                unknown = [side for side in sides if state[side] is None]
                if not unknown:
                    continue
                if on == clue:
                    if clue == 0:
                        fact = f"the 0 at {display_cell_text((r, c))} keeps all four sides empty"
                    else:
                        fact = (
                            f"the {clue} at {display_cell_text((r, c))} already has its "
                            f"{clue} side{'s' if clue != 1 else ''}"
                        )
                    for side in unknown:
                        new_off.append((side, fact))
                elif on + len(unknown) == clue:
                    for side in unknown:
                        forced_on.setdefault(
                            side,
                            f"The {clue} at {display_cell_text((r, c))} has exactly "
                            f"{len(unknown)} open side{'s' if len(unknown) != 1 else ''} left "
                            f"for its remaining count, so {self._edge_phrase(side)} is drawn",
                        )

            for corner, incident in corner_edges.items():
                on = sum(1 for edge in incident if state[edge] is True)
                unknown = [edge for edge in incident if state[edge] is None]
                if on == 2 and unknown:
                    for edge in unknown:
                        new_off.append(
                            (
                                edge,
                                f"corner {display_cell_text(corner)} already joins two "
                                "segments",
                            )
                        )
                elif on == 1 and len(unknown) == 1:
                    edge = unknown[0]
                    forced_on.setdefault(
                        edge,
                        f"The line entering corner {display_cell_text(corner)} has a single "
                        f"open continuation, so {self._edge_phrase(edge)} is drawn",
                    )
                elif on == 0 and len(unknown) == 1:
                    edge = unknown[0]
                    new_off.append(
                        (
                            edge,
                            f"corner {display_cell_text(corner)} cannot host a dead end",
                        )
                    )

            # Premature-closure analysis.
            groups, ids = components()
            component_count = len(ids)
            for edge in edges:
                if state[edge] is not None or edge in forced_on:
                    continue
                a, b = edge
                if a in groups and b in groups and groups[a] == groups[b]:
                    would_close_all = component_count == 1
                    if would_close_all:
                        closed = {e for e, value in state.items() if value is True} | {edge}
                        ok = True
                        for (r, c), sides in cell_sides.items():
                            clue = clues[r][c]
                            if clue is None or clue < 0:
                                continue
                            if sum(1 for side in sides if side in closed) != clue:
                                ok = False
                                break
                        if ok and closed == solution_edges:
                            forced_on.setdefault(
                                edge,
                                f"Only {self._edge_phrase(edge)} remains to close the single "
                                "line into one loop that satisfies every digit, so it is drawn",
                            )
                            continue
                    new_off.append(
                        (
                            edge,
                            f"joining the open line back onto itself at "
                            f"{self._edge_phrase(edge)} would close a loop too early",
                        )
                    )

            if forced_on:
                round_moves: List[ForcedMove] = []
                for edge in sorted(forced_on):
                    if edge not in solution_edges:
                        raise ForcedDeductionError(
                            "slitherlink_completion: forced segment deviates from the loop"
                        )
                    reason = forced_on[edge]
                    if pending:
                        reason = "; ".join(
                            ["note that " + fact for fact in pending[:2]] + [reason]
                        )
                        reason = reason[0].upper() + reason[1:]
                        pending = []
                    state[edge] = True
                    round_moves.append(
                        ForcedMove(kind="segment", cells=edge, value="on", reason=reason)
                    )
                rounds.append(round_moves)
                continue

            progressed = False
            for edge, fact in new_off:
                if state[edge] is None:
                    if edge in solution_edges:
                        raise ForcedDeductionError(
                            "slitherlink_completion: eliminated a true loop segment"
                        )
                    state[edge] = False
                    progressed = True
                    if fact not in pending and len(pending) < 4:
                        pending.append(fact)
            if not progressed:
                return None

    def _construct_instance(self):
        for _ in range(60):
            inside, clues, edge_list = None, None, None
            try:
                inside, clues, edge_list = self._build_instance()
            except Exception:
                continue
            solution_edges = {tuple(sorted((a, b))) for a, b in edge_list}
            given: Set = set(solution_edges)
            order = sorted(given)
            self.rng.shuffle(order)
            for edge in order:
                given.discard(edge)
                if self._forced_chain(solution_edges, clues, given) is None:
                    given.add(edge)
            chain = self._forced_chain(solution_edges, clues, given)
            open_count = len(solution_edges) - len(given)
            if chain is None or open_count < 6 or len(chain.rounds) < 2:
                continue
            if len(given) > len(solution_edges) // 2 or not given:
                continue
            prove_complete_chain(chain, expected_cells=open_count * 2, task_name="slitherlink_completion")
            return inside, clues, edge_list, sorted(given), chain
        raise ForcedDeductionError(
            "slitherlink_completion: failed to construct a forced-hand-solvable loop"
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        clues: List[List[Optional[int]]],
        committed_edges: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    ) -> Image.Image:
        rows, cols = self.rows, self.cols
        canvas = 512
        margin = 40
        usable = canvas - 2 * margin
        cell = (usable // max(rows, cols)) & ~1
        board_w = cols * cell
        board_h = rows * cell
        left = (canvas - board_w) // 2
        top = (canvas - board_h) // 2
        scene, draw = new_board_scene()
        font = board_font(max(20, cell * 2 // 5))
        label_font = board_font(max(12, min(20, cell // 4)))

        for r in range(rows + 1):
            for c in range(cols + 1):
                x = left + c * cell
                y = top + r * cell
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=_DOT)

        for r in range(rows + 1):
            draw.text((left - 16, top + r * cell), str(r + 1), fill=AXIS_LABEL, font=label_font, anchor="mm")
        for c in range(cols + 1):
            draw.text((left + c * cell, top - 16), str(c + 1), fill=AXIS_LABEL, font=label_font, anchor="mm")

        for r in range(rows):
            for c in range(cols):
                value = clues[r][c]
                if value is None:
                    continue
                cx = left + c * cell + cell // 2
                cy = top + r * cell + cell // 2
                draw.text((cx, cy), str(value), fill=TEXT, font=font, anchor="mm")

        for (r1, c1), (r2, c2) in committed_edges:
            x1 = left + c1 * cell
            y1 = top + r1 * cell
            x2 = left + c2 * cell
            y2 = top + r2 * cell
            draw.line([(x1, y1), (x2, y2)], fill=_LOOP, width=6)
        return finish_board_scene(scene)

    def _build_prompts(self) -> Dict[str, str]:
        solution_image = build_sectioned_prompt(
            _INPUT_DESCRIPTION,
            f"Render {_TERMINAL_VISUAL}.",
            rule_description=_RULE_DESCRIPTION,
            mode="solution_image",
        )
        ti2v = build_sectioned_prompt(
            _INPUT_DESCRIPTION,
            (
                "Generate a solution video that starts from the unchanged puzzle. Red segments then "
                "appear in deduction order: each wave of segments the digits and dot connections "
                "force fades in together, earlier segments stay exactly in place, and the video ends "
                f"holding {_TERMINAL_VISUAL}."
            ),
            rule_description=_RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    def _narrate(
        self,
        groups: Sequence[Sequence[ForcedMove]],
        given_count: int,
        clue_count: int,
    ) -> List[str]:
        opener = (
            f"The lattice frames a {self.rows} by {self.cols} board with {clue_count} digit "
            f"clue{'s' if clue_count != 1 else ''}, and {given_count} loop segment"
            f"{'s are' if given_count != 1 else ' is'} already drawn. Every digit counts its "
            "drawn sides exactly and every corner joins zero or two segments, so I will extend "
            "the line where those facts leave a single choice."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " The drawn segments now form one closed loop, every digit counts its sides exactly, "
            "and no corner holds a branch or dead end, so the loop is complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    def create_random_puzzle(self) -> SlitherlinkCompletionRecord:
        inside, clues, edge_list, given_edges, chain = self._construct_instance()
        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        committed: List[Tuple[Tuple[int, int], Tuple[int, int]]] = list(given_edges)
        states = [fit_canvas(self._render(clues, committed))]
        for group in groups:
            committed = committed + [tuple(move.cells) for move in group]
            states.append(fit_canvas(self._render(clues, committed)))
        puzzle_image, solution_image = states[0], states[-1]

        puzzle_path = self.puzzle_dir / f"{puzzle_id}_puzzle.png"
        reasoning_path = self.reasoning_dir / f"{puzzle_id}_reasoning.png"
        solution_path = self.solution_dir / f"{puzzle_id}_solution.png"
        puzzle_image.save(puzzle_path)
        puzzle_image.save(reasoning_path)
        solution_image.save(solution_path)

        nonterminal_paths: List[str] = []
        for state_index, state in enumerate(states[1:-1], start=1):
            state_path = self.reasoning_dir / f"{puzzle_id}_state_{state_index:02d}.png"
            state.save(state_path)
            nonterminal_paths.append(self.relativize_path(state_path))

        solution_video_path: Optional[str] = None
        video_num_frames: Optional[int] = None
        if self.video:
            video_path = self.solution_dir / f"{puzzle_id}_solution.webm"
            frames = encode_commit_video(states, video_path, fps=self.video_fps)
            if frames > 0 and video_path.exists():
                solution_video_path = self.relativize_path(video_path)
                video_num_frames = frames

        edge_payload = [[a[0] + 1, a[1] + 1, b[0] + 1, b[1] + 1] for a, b in edge_list]
        edge_composites = [
            ((x1, y1), (x2, y2))
            for x1, y1, x2, y2 in edge_payload
        ]

        return SlitherlinkCompletionRecord(
            answer=answer_xy_composites(edge_composites),
            id=puzzle_id,
            image=self.relativize_path(puzzle_path),
            reasoning_image=self.relativize_path(reasoning_path),
            solution_image_path=self.relativize_path(solution_path),
            solution_video_path=solution_video_path,
            video_fps=self.video_fps if solution_video_path else None,
            video_num_frames=video_num_frames,
            ti2v_prompt=prompts["ti2v"],
            solution_image_prompt=prompts["solution_image"],
            structured_answer={
                "rows": self.rows,
                "cols": self.cols,
                "clues": [[(-1 if v is None else v) for v in row] for row in clues],
                "given_edges": [
                    [a[0] + 1, a[1] + 1, b[0] + 1, b[1] + 1] for a, b in given_edges
                ],
                "edges": edge_payload,
                "inside_mask": inside,
            },
        )
