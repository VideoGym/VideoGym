"""Flow pairs: join every color's two endpoint dots with disjoint full-cover routes.

Numberlink-style task rebuilt on the family forced-deduction contract.  The
generator first space-fills the grid with one path per color, then presents
givens as pre-drawn route prefixes anchored at endpoint dots: starting from the
maximally given board it monotonically trims prefix cells while the corridor
forcing engine still completes every route with zero guessing.  The proven
chain is generator-side offline QA; it never enters ``semantic_input`` or any
reward surface.

The engine is sound edge-domain propagation over the visible state: every cell
needs exactly two route sides (one at an endpoint dot), so a cell whose
remaining side count equals its remaining candidate sides makes those sides
certain, and a side that would close a drawn-or-certain chain into a cycle is
impossible.  Certain free-free edges stay internal knowledge — they have no
color to draw — and a visible commitment happens only at a live route end:
an end whose sole remaining side is certain extends (or closes onto its
partner end), then keeps walking while the entered cells' own squeezes keep
the continuation certain (capped at two cells per wave so long corridors
unfold across waves).  Boards whose deduction stalls simply keep more given
prefix instead.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from PIL import Image

from data.constraint_grid.core.forced_deduction import (
    ForcedChain,
    ForcedDeductionError,
    ForcedMove,
    encode_commit_video,
    merge_rounds,
    prove_complete_chain,
)
from data.constraint_grid.core.coordinate_format import display_cell_text
from data.constraint_grid.core.scene import (
    board_font,
    draw_axis_labels,
    draw_board_grid,
    finish_board_scene,
    new_board_scene,
    standard_board_geometry,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.contracts.grid_completion import (
    answer_named_xy_paths,
)
from data.common.pipeline.grid_rendering import fit_canvas
from data.common.prompts import build_sectioned_prompt

Cell = Tuple[int, int]

# Flow color swatches, in deal order.  Saturated and mutually distant so the
# terminal decoder separates them from each other and from the pale scene.
FLOW_SWATCHES: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("red", (214, 69, 69)),
    ("blue", (58, 110, 165)),
    ("green", (46, 139, 87)),
)

_TERMINAL_VISUAL = (
    "the finished board on which every color's two dots are joined by one thick route in that "
    "color, the routes together filling every cell while the dots, grid, and edge numbers stay "
    "unchanged"
)


@dataclass
class MazeFlowPairsRecord:
    id: str
    task_name: str = "maze_flow_pairs"
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


def _pos(cell: Cell) -> str:
    return display_cell_text(cell)


class MazeFlowPairsGenerator(AbstractPuzzleGenerator[MazeFlowPairsRecord]):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        rows: int = 7,
        cols: int = 7,
        flow_pairs: Optional[int] = None,
        video: bool = False,
        video_fps: int = 16,
    ) -> None:
        super().__init__(output_dir)
        if rows < 5 or cols < 5 or rows > 9 or cols > 9:
            raise ValueError("rows and cols must be between 5 and 9")
        if flow_pairs is not None and not 2 <= int(flow_pairs) <= len(FLOW_SWATCHES):
            raise ValueError("flow_pairs must be 2 or 3")
        self.rng = random.Random(seed)
        self.seed = seed
        self.rows = int(rows)
        self.cols = int(cols)
        self.flow_pairs = int(flow_pairs) if flow_pairs is not None else None
        # An explicit shape or color count pins one board profile (the
        # independent render replay reconstructs with explicit rows/cols);
        # the default constructor samples a profile per record.
        if self.flow_pairs is not None:
            self._fixed_profile: Optional[Tuple[int, int, int]] = (
                self.flow_pairs,
                self.rows,
                self.cols,
            )
        elif (int(rows), int(cols)) != (7, 7):
            self._fixed_profile = (3 if max(self.rows, self.cols) >= 6 else 2, self.rows, self.cols)
        else:
            self._fixed_profile = None
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self._counter = 0
        self.puzzle_dir = self.output_dir / "puzzles"
        self.reasoning_dir = self.output_dir / "reasoning"
        self.solution_dir = self.output_dir / "solutions"
        for directory in (self.puzzle_dir, self.reasoning_dir, self.solution_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def create_puzzle(self, *args, **kwargs) -> MazeFlowPairsRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"maze_flow_pairs-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    # ------------------------------------------------------------------
    # Full-cover solution construction (space-filling backtracking)
    # ------------------------------------------------------------------

    def _neighbors(self, cell: Cell) -> List[Cell]:
        row, col = cell
        out: List[Cell] = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                out.append((nr, nc))
        return out

    def _sample_sizes(self, total: int, count: int) -> List[int]:
        minimum = max(3, total // (count + 3))
        sizes = [minimum] * count
        remaining = total - minimum * count
        while remaining > 0:
            sizes[self.rng.randrange(count)] += 1
            remaining -= 1
        self.rng.shuffle(sizes)
        return sizes

    def _free_components(self, occupied: Mapping[Cell, int]) -> List[List[Cell]]:
        seen: Set[Cell] = set()
        components: List[List[Cell]] = []
        for row in range(self.rows):
            for col in range(self.cols):
                cell = (row, col)
                if cell in occupied or cell in seen:
                    continue
                component: List[Cell] = []
                queue: deque[Cell] = deque([cell])
                seen.add(cell)
                while queue:
                    current = queue.popleft()
                    component.append(current)
                    for neighbor in self._neighbors(current):
                        if neighbor not in occupied and neighbor not in seen:
                            seen.add(neighbor)
                            queue.append(neighbor)
                components.append(component)
        return components

    def _can_partition(self, component_sizes: List[int], remaining: List[int]) -> bool:
        if sum(component_sizes) != sum(remaining):
            return False
        if not component_sizes:
            return not remaining
        target = component_sizes[0]
        count = len(remaining)
        for mask in range(1, 1 << count):
            subset = [remaining[i] for i in range(count) if (mask >> i) & 1]
            if sum(subset) != target:
                continue
            rest = [remaining[i] for i in range(count) if not ((mask >> i) & 1)]
            if self._can_partition(component_sizes[1:], rest):
                return True
        return False

    def _extend_path(
        self,
        path: List[Cell],
        target_length: int,
        occupied: Dict[Cell, int],
        color_index: int,
        sizes: Sequence[int],
    ) -> bool:
        if len(path) == target_length:
            used = set(occupied.values())
            remaining = sorted(
                (sizes[i] for i in range(len(sizes)) if i not in used), reverse=True
            )
            component_sizes = sorted(
                (len(component) for component in self._free_components(occupied)),
                reverse=True,
            )
            if not remaining:
                return not component_sizes
            return self._can_partition(component_sizes, remaining)
        current = path[-1]
        candidates = [n for n in self._neighbors(current) if n not in occupied]

        def open_degree(cell: Cell) -> int:
            return sum(
                1 for n in self._neighbors(cell) if n not in occupied and n != current
            )

        candidates.sort(key=open_degree)
        for nxt in candidates:
            path.append(nxt)
            occupied[nxt] = color_index
            if self._extend_path(path, target_length, occupied, color_index, sizes):
                return True
            path.pop()
            del occupied[nxt]
        return False

    def _fill_with_sizes(self, sizes: Sequence[int]) -> Optional[Dict[int, List[Cell]]]:
        occupied: Dict[Cell, int] = {}
        paths: Dict[int, List[Cell]] = {}

        def corner_score(cell: Cell) -> int:
            row, col = cell
            return (row in (0, self.rows - 1)) + (col in (0, self.cols - 1))

        for color_index in sorted(range(len(sizes)), key=lambda i: -sizes[i]):
            starts = [
                (r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if (r, c) not in occupied
            ]
            self.rng.shuffle(starts)
            starts.sort(key=corner_score, reverse=True)
            placed = False
            for start in starts[:12]:
                path = [start]
                occupied[start] = color_index
                if self._extend_path(path, sizes[color_index], occupied, color_index, sizes):
                    paths[color_index] = list(path)
                    placed = True
                    break
                del occupied[start]
            if not placed:
                return None
        if len(occupied) != self.rows * self.cols:
            return None
        return {i: paths[i] for i in range(len(sizes))}

    def _build_solution(self, color_count: int) -> Dict[int, List[Cell]]:
        total = self.rows * self.cols
        for _ in range(400):
            paths = self._fill_with_sizes(self._sample_sizes(total, color_count))
            if paths is None:
                continue
            manhattan = [
                abs(path[0][0] - path[-1][0]) + abs(path[0][1] - path[-1][1]) + 1
                for path in paths.values()
            ]
            if any(len(path) > distance for path, distance in zip(paths.values(), manhattan)):
                return paths
        raise ForcedDeductionError("maze_flow_pairs: failed to space-fill a route partition")

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    @staticmethod
    def _edge(a: Cell, b: Cell) -> Tuple[Cell, Cell]:
        return (a, b) if a <= b else (b, a)

    def _forced_edges(
        self,
        endpoints: Set[Cell],
        occupied: Mapping[Cell, int],
        stroke_edges: Set[Tuple[Cell, Cell]],
        heads_by_cell: Mapping[Cell, Tuple[int, str]],
    ) -> Tuple[Set[Tuple[Cell, Cell]], Dict[Tuple[Cell, Cell], Tuple[Cell, Tuple[Cell, ...]]]]:
        """Color-agnostic sound edge propagation over the visible state.

        Every cell needs exactly two route sides (one at an endpoint dot).
        Whenever a cell's remaining side count equals its remaining candidate
        sides, those sides are certain in every completion; a side that would
        close a drawn-or-certain chain into a cycle is impossible.  The closure
        of these two eliminations is internal knowledge: only edges that touch
        a live end become visible commitments.  Returns the certain edges plus
        per-edge provenance ``(driver cell, that cell's sides at firing)`` so
        the narration can state the true squeeze.
        """

        forced: Set[Tuple[Cell, Cell]] = set()
        provenance: Dict[Tuple[Cell, Cell], Tuple[Cell, Tuple[Cell, ...]]] = {}
        parent: Dict[Cell, Cell] = {}

        def find(cell: Cell) -> Cell:
            parent.setdefault(cell, cell)
            while parent[cell] != cell:
                parent[cell] = parent[parent[cell]]
                cell = parent[cell]
            return cell

        def union(a: Cell, b: Cell) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in stroke_edges:
            union(a, b)

        def required(cell: Cell) -> int:
            return 1 if cell in endpoints else 2

        def known_degree(cell: Cell) -> int:
            return sum(
                1
                for n in self._neighbors(cell)
                if self._edge(cell, n) in stroke_edges or self._edge(cell, n) in forced
            )

        def candidate_sides(cell: Cell) -> List[Cell]:
            sides: List[Cell] = []
            for n in self._neighbors(cell):
                edge = self._edge(cell, n)
                if edge in stroke_edges or edge in forced:
                    continue
                if cell in occupied and n in occupied:
                    # Both committed: only the two live ends of one color may
                    # still link (the closure side).
                    if not (
                        cell in heads_by_cell
                        and n in heads_by_cell
                        and occupied[cell] == occupied[n]
                    ):
                        continue
                elif n in occupied and n not in heads_by_cell:
                    continue
                if required(n) - known_degree(n) <= 0:
                    continue
                if find(cell) == find(n):
                    continue
                sides.append(n)
            return sides

        changed = True
        while changed:
            changed = False
            for row in range(self.rows):
                for col in range(self.cols):
                    cell = (row, col)
                    if cell in occupied and cell not in heads_by_cell:
                        continue
                    remaining = required(cell) - known_degree(cell)
                    if remaining <= 0:
                        continue
                    sides = candidate_sides(cell)
                    if len(sides) < remaining:
                        raise ForcedDeductionError(
                            "maze_flow_pairs: a cell has fewer open sides than it needs"
                        )
                    if len(sides) == remaining:
                        for n in sides:
                            edge = self._edge(cell, n)
                            if edge not in forced:
                                forced.add(edge)
                                provenance[edge] = (cell, tuple(sides))
                                union(cell, n)
                                changed = True
        return forced, provenance

    def _forced_chain(
        self,
        paths: Dict[int, List[Cell]],
        givens: Mapping[int, Tuple[int, int]],
    ) -> Optional[ForcedChain]:
        """Run the forced rules from the given prefixes; ``None`` on stall.

        Visible commitments happen only at live route ends: an end whose only
        remaining side is certain extends there, and it keeps running while
        the entered cells' own two-side squeezes chain onward.  Every
        commitment is asserted against the constructed route — the rules are
        sound, so a deviation can only be an engine bug or a non-unique
        instance, and either fails loudly.
        """

        names = {index: FLOW_SWATCHES[index][0] for index in paths}
        endpoints = {path[0] for path in paths.values()} | {
            path[-1] for path in paths.values()
        }
        sub_a: Dict[int, List[Cell]] = {}
        sub_b: Dict[int, List[Cell]] = {}
        closed: Dict[int, bool] = {}
        occupied: Dict[Cell, int] = {}
        for index, path in paths.items():
            a_len, b_len = givens[index]
            sub_a[index] = list(path[:a_len])
            sub_b[index] = list(reversed(path[len(path) - b_len :]))
            closed[index] = False
            for cell in sub_a[index] + sub_b[index]:
                occupied[cell] = index

        def head(index: int, end: str) -> Cell:
            return (sub_a if end == "a" else sub_b)[index][-1]

        def expected_next(index: int, end: str) -> Optional[Cell]:
            path = paths[index]
            if end == "a":
                position = len(sub_a[index])
                return path[position] if position < len(path) else None
            position = len(path) - 1 - len(sub_b[index])
            return path[position] if position >= 0 else None

        def stroke_edges() -> Set[Tuple[Cell, Cell]]:
            edges: Set[Tuple[Cell, Cell]] = set()
            for index in paths:
                for sub in (sub_a[index], sub_b[index]):
                    edges.update(self._edge(a, b) for a, b in zip(sub, sub[1:]))
                if closed[index]:
                    edges.add(self._edge(sub_a[index][-1], sub_b[index][-1]))
            return edges

        def commit(index: int, end: str, cells: Sequence[Cell], closes: bool) -> None:
            sub = sub_a if end == "a" else sub_b
            for cell in cells:
                if cell != expected_next(index, end):
                    raise ForcedDeductionError(
                        "maze_flow_pairs: forced segment deviates from the constructed route"
                    )
                sub[index].append(cell)
                occupied[cell] = index
            if closes:
                if len(sub_a[index]) + len(sub_b[index]) != len(paths[index]) or (
                    head(index, "a") not in self._neighbors(head(index, "b"))
                ):
                    raise ForcedDeductionError(
                        "maze_flow_pairs: forced closure deviates from the constructed route"
                    )
                closed[index] = True

        rounds: List[List[ForcedMove]] = []
        while True:
            if len(occupied) == self.rows * self.cols and all(closed.values()):
                return ForcedChain(rounds=rounds) if rounds else None

            heads_by_cell = {
                head(index, end): (index, end)
                for index in paths
                if not closed[index]
                for end in ("a", "b")
            }
            forced, provenance = self._forced_edges(
                endpoints, occupied, stroke_edges(), heads_by_cell
            )

            committed: List[ForcedMove] = []
            for index in paths:
                for end in ("a", "b"):
                    if closed[index]:
                        continue
                    h = head(index, end)
                    partner = head(index, "b" if end == "a" else "a")
                    partner_adjacent = partner in self._neighbors(h)
                    free_sides = [n for n in self._neighbors(h) if n not in occupied]
                    forced_at_h = [
                        n
                        for n in self._neighbors(h)
                        if self._edge(h, n) in forced
                        and (n not in occupied or n == partner)
                    ]
                    move_target: Optional[Cell] = None
                    squeeze_start = False
                    if len(forced_at_h) == 1:
                        move_target = forced_at_h[0]
                        squeeze_start = move_target != partner and (
                            len(free_sides) + (1 if partner_adjacent else 0) > 1
                        )
                    elif len(forced_at_h) > 1:
                        raise ForcedDeductionError(
                            "maze_flow_pairs: a route end carries two certain sides"
                        )
                    elif len(free_sides) + (1 if partner_adjacent else 0) == 1:
                        move_target = free_sides[0] if free_sides else partner
                    if move_target is None:
                        continue

                    if move_target == partner:
                        commit(index, end, (), closes=True)
                        committed.append(
                            ForcedMove(
                                kind="close",
                                cells=(),
                                value={"color": index, "end": end, "closes": True, "head": h},
                                reason=(
                                    f"The two {names[index]} ends at {_pos(h)} and "
                                    f"{_pos(partner)} now sit side by side and no other side "
                                    f"of either end remains open, so {names[index]} links "
                                    "them and its route is complete"
                                ),
                            )
                        )
                        continue

                    # Walk forward while the entered cells' own squeezes keep
                    # the continuation certain.  The walk is capped so a long
                    # certain corridor unfolds over successive waves instead
                    # of disappearing into one commitment.
                    chain: List[Cell] = [move_target]
                    closes = False
                    while len(chain) < 2:
                        current = chain[-1]
                        partner_now = head(index, "b" if end == "a" else "a")
                        onward = [
                            n
                            for n in self._neighbors(current)
                            if self._edge(current, n) in forced
                            and n != (chain[-2] if len(chain) > 1 else h)
                            and (n not in occupied or n == partner_now)
                        ]
                        if len(onward) != 1:
                            break
                        if onward[0] == partner_now:
                            closes = True
                            break
                        chain.append(onward[0])
                    commit(index, end, chain, closes=closes)
                    committed.append(
                        ForcedMove(
                            kind="squeeze" if squeeze_start else "run",
                            cells=tuple(chain),
                            value={
                                "color": index,
                                "end": end,
                                "closes": closes,
                                "head": h,
                            },
                            reason=self._move_reason(
                                names[index],
                                h,
                                chain,
                                closes,
                                squeeze_start,
                                provenance.get(self._edge(h, chain[0])),
                            ),
                        )
                    )
            if not committed:
                return None
            rounds.append(committed)

    def _move_reason(
        self,
        color: str,
        h: Cell,
        chain: Sequence[Cell],
        closes: bool,
        squeeze_start: bool,
        first_edge_provenance: Optional[Tuple[Cell, Tuple[Cell, ...]]],
    ) -> str:
        into_other = False
        if not squeeze_start:
            base = (
                f"The {color} end at {_pos(h)} has a single open side, so {color} continues "
                f"to {_pos(chain[0])}"
            )
        elif first_edge_provenance is not None and first_edge_provenance[0] == chain[0]:
            _driver, sides = first_edge_provenance
            others = [side for side in sides if side != h]
            if others:
                base = (
                    f"The cell at {_pos(chain[0])} has only two ways left to be covered, "
                    f"through the {color} end at {_pos(h)} and through {_pos(others[0])}, so "
                    f"{color} must run straight through it"
                )
                into_other = len(chain) == 2 and chain[1] == others[0]
            else:
                base = (
                    f"The cell at {_pos(chain[0])} has just one way left to be covered, "
                    f"through the {color} end at {_pos(h)}, so {color} must take it"
                )
        else:
            base = (
                f"Every other side of the {color} end at {_pos(h)} is already claimed by the "
                f"cells around it, so {color} must continue to {_pos(chain[0])}"
            )
        if into_other:
            base += " and on into that second cell"
        elif len(chain) > 1:
            middle = ", then ".join(_pos(cell) for cell in chain[1:-1])
            tail = " and the squeeze carries it on"
            if middle:
                tail += f" through {middle}"
            tail += f" to {_pos(chain[-1])}"
            base += "," + tail
        if closes:
            base += ", where it meets its other end and the route is complete"
        return base

    # ------------------------------------------------------------------
    # Given-prefix trimming
    # ------------------------------------------------------------------

    # Sampled per record unless the constructor pinned an explicit shape: two
    # colors deduce on a 5x5 board, three colors on 6x6 or 7x7, so the
    # per-route deduction load stays comparable across the difficulty ladder.
    _BOARD_PROFILES: Tuple[Tuple[int, int, int], ...] = (
        (2, 5, 5),
        (3, 6, 6),
        (3, 7, 7),
        (3, 7, 7),
    )

    def _sample_profile(self) -> Tuple[int, int, int]:
        if self._fixed_profile is not None:
            return self._fixed_profile
        return self.rng.choice(self._BOARD_PROFILES)

    def _construct_instance(
        self,
    ) -> Tuple[Dict[int, List[Cell]], Dict[int, Tuple[int, int]], ForcedChain]:
        for _ in range(60):
            color_count, self.rows, self.cols = self._sample_profile()
            paths = self._build_solution(color_count)
            givens: Dict[int, Tuple[int, int]] = {
                index: (len(path) - 1, 1) for index, path in paths.items()
            }

            def drawn_stroke_count(state: Mapping[int, Tuple[int, int]]) -> int:
                return sum((a > 1) + (b > 1) for a, b in state.values())

            improved = True
            while improved:
                improved = False
                candidates = [
                    (index, end) for index in paths for end in ("a", "b")
                ]
                self.rng.shuffle(candidates)
                for index, end in candidates:
                    a_len, b_len = givens[index]
                    if (a_len if end == "a" else b_len) <= 1:
                        continue
                    trial = dict(givens)
                    trial[index] = (a_len - 1, b_len) if end == "a" else (a_len, b_len - 1)
                    if drawn_stroke_count(trial) < 1:
                        continue
                    if self._forced_chain(paths, trial) is not None:
                        givens = trial
                        improved = True
            chain = self._forced_chain(paths, givens)
            total = self.rows * self.cols
            given_cells = sum(a + b for a, b in givens.values())
            given_edges = given_cells - 2 * color_count
            total_edges = total - color_count
            if chain is None or len(chain.rounds) < 3:
                continue
            if given_edges > total_edges // 2 or total - given_cells < max(8, total // 3):
                continue
            prove_complete_chain(
                chain, expected_cells=total - given_cells, task_name="maze_flow_pairs"
            )
            return paths, givens, chain
        raise ForcedDeductionError(
            "maze_flow_pairs: failed to construct a forced-hand-solvable board"
        )

    # ------------------------------------------------------------------
    # Rendering (shared 4x supersampled scene primitives)
    # ------------------------------------------------------------------

    def _render(
        self,
        endpoints: Sequence[Mapping[str, Any]],
        strokes: Mapping[str, Sequence[Sequence[Cell]]],
    ) -> Image.Image:
        """Render the board with the committed route polylines.

        ``endpoints`` lists ``{color, color_rgb, cells}`` with internal 0-based
        cells; ``strokes`` maps a color name to its committed polylines (each a
        cell sequence; single-cell polylines draw nothing).  The independent
        verifier replays candidate terminals through this same function.
        """

        rows, cols = self.rows, self.cols
        cell, left, top = standard_board_geometry(rows, cols)
        scene, draw = new_board_scene()
        label_font = board_font(max(12, min(20, cell // 4)))
        draw_board_grid(draw, rows, cols, cell=cell, left=left, top=top)
        draw_axis_labels(
            draw, rows, cols, cell=cell, left=left, top=top, font=label_font, offset=16
        )

        def center(target: Cell) -> Tuple[int, int]:
            return (
                left + int(target[1]) * cell + cell // 2,
                top + int(target[0]) * cell + cell // 2,
            )

        stroke = max(6, (cell // 3) & ~1)
        half = stroke // 2
        color_rgb = {
            str(entry["color"]).strip().lower(): tuple(int(v) for v in entry["color_rgb"])
            for entry in endpoints
        }
        for color, polylines in strokes.items():
            rgb = color_rgb[str(color).strip().lower()]
            for polyline in polylines:
                points = [center(target) for target in polyline]
                if len(points) < 2:
                    continue
                # Axis-aligned segments render as endpoint-order-invariant
                # rectangles with built-in square caps, so a route drawn in
                # waves is byte-identical to the same route drawn at once.
                for (x1, y1), (x2, y2) in zip(points, points[1:]):
                    draw.rectangle(
                        [
                            min(x1, x2) - half,
                            min(y1, y2) - half,
                            max(x1, x2) + half,
                            max(y1, y2) + half,
                        ],
                        fill=rgb,
                    )

        dot_radius = max(8, int(cell * 0.30))
        for entry in endpoints:
            rgb = tuple(int(v) for v in entry["color_rgb"])
            for raw_cell in entry["cells"]:
                x, y = center((int(raw_cell[0]), int(raw_cell[1])))
                draw.ellipse(
                    [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
                    fill=rgb,
                )
        return finish_board_scene(scene)

    @staticmethod
    def _stroke_state(
        names: Mapping[int, str],
        sub_a: Mapping[int, Sequence[Cell]],
        sub_b: Mapping[int, Sequence[Cell]],
        closed: Mapping[int, bool],
    ) -> Dict[str, List[List[Cell]]]:
        strokes: Dict[str, List[List[Cell]]] = {}
        for index, name in names.items():
            if closed[index]:
                strokes[name] = [list(sub_a[index]) + list(reversed(list(sub_b[index])))]
            else:
                strokes[name] = [list(sub_a[index]), list(sub_b[index])]
        return strokes

    # ------------------------------------------------------------------
    # Prompts and narration
    # ------------------------------------------------------------------

    def _color_names(self, count: int) -> List[str]:
        return [FLOW_SWATCHES[index][0] for index in range(count)]

    def _input_description(self, count: int) -> str:
        names = self._color_names(count)
        listed = ", ".join(names[:-1]) + f" and {names[-1]}"
        return (
            f"The image shows a {self.rows} by {self.cols} grid with small gray 1-based row and "
            f"column numbers along the left and top edges, and {count} pairs of colored endpoint "
            f"dots ({listed}); each color appears as exactly two dots. Thick colored route "
            "segments already drawn out of some endpoints are fixed parts of the final routes."
        )

    _RULE_DESCRIPTION = (
        "Join the two dots of every color with one route that steps only between side-adjacent "
        "cells. Routes of different colors never share a cell, every cell of the grid ends up on "
        "exactly one route, and each dot is part of its own color's route. The route segments "
        "already drawn are fixed and must stay part of the final routes."
    )

    def _build_prompts(self, count: int) -> Dict[str, str]:
        input_description = self._input_description(count)
        solution_image = build_sectioned_prompt(
            input_description,
            f"Render {_TERMINAL_VISUAL}.",
            rule_description=self._RULE_DESCRIPTION,
            mode="solution_image",
        )
        ti2v = build_sectioned_prompt(
            input_description,
            (
                "Generate a solution video that starts from the unchanged puzzle. Colored route "
                "segments then appear in deduction order: each forced wave of segments fades in "
                "together, earlier segments and the endpoint dots stay exactly in place, and the "
                f"video ends holding {_TERMINAL_VISUAL}."
            ),
            rule_description=self._RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    def _narrate(
        self,
        groups: Sequence[Sequence[ForcedMove]],
        names: Mapping[int, str],
        color_count: int,
        given_segments: int,
    ) -> List[str]:
        listed = ", ".join(self._color_names(color_count))
        opener = (
            f"The board is {self.rows} by {self.cols} with {color_count} colored endpoint pairs "
            f"({listed}), and {given_segments} route segment"
            f"{'s are' if given_segments != 1 else ' is'} already drawn. Every cell must end up "
            "on exactly one color's route and routes never share a cell, so wherever a route end "
            "has a single open side, or a free cell has only two open sides, the next segment is "
            "decided."
        )
        paragraphs: List[str] = []
        for group_index, group in enumerate(groups):
            sentences: List[str] = [opener] if group_index == 0 else []
            position = 0
            moves = list(group)
            while position < len(moves):
                move = moves[position]
                value = dict(move.value)  # type: ignore[arg-type]
                if move.kind == "run":
                    run = [move]
                    while (
                        position + len(run) < len(moves)
                        and moves[position + len(run)].kind == "run"
                        and dict(moves[position + len(run)].value)["color"] == value["color"]  # type: ignore[arg-type]
                        and dict(moves[position + len(run)].value)["end"] == value["end"]  # type: ignore[arg-type]
                    ):
                        run.append(moves[position + len(run)])
                    if len(run) == 1:
                        sentences.append(move.reason)
                    else:
                        cells = [cell for step in run for cell in step.cells]
                        head = _pos(tuple(value["head"]))
                        color = names[int(value["color"])]
                        body = ", then ".join(_pos(cell) for cell in cells[:-1])
                        sentences.append(
                            f"The {color} end at {head} keeps a single open side at each step, "
                            f"so {color} runs through {body} to {_pos(cells[-1])}"
                        )
                    position += len(run)
                else:
                    sentences.append(move.reason)
                    position += 1
            paragraphs.append(
                " ".join(
                    sentence if sentence.endswith((".", "!", "?")) else f"{sentence}."
                    for sentence in sentences
                )
            )
        paragraphs[-1] += (
            " Every cell is now covered and each color's two dots are joined, so the routes are "
            "complete."
        )
        return paragraphs

    # ------------------------------------------------------------------
    # Record assembly
    # ------------------------------------------------------------------

    def create_random_puzzle(self) -> MazeFlowPairsRecord:
        paths, givens, chain = self._construct_instance()
        names = {index: FLOW_SWATCHES[index][0] for index in paths}
        color_count = len(paths)
        prompts = self._build_prompts(color_count)
        puzzle_id = self._next_id()

        endpoints = [
            {
                "color": names[index],
                "color_rgb": [int(v) for v in FLOW_SWATCHES[index][1]],
                "cells": [list(path[0]), list(path[-1])],
            }
            for index, path in paths.items()
        ]
        sub_a = {index: list(path[: givens[index][0]]) for index, path in paths.items()}
        sub_b = {
            index: list(reversed(path[len(path) - givens[index][1] :]))
            for index, path in paths.items()
        }
        closed = {index: False for index in paths}
        given_segments = sum(len(sub) - 1 for subs in (sub_a, sub_b) for sub in subs.values())

        groups = merge_rounds(chain.rounds)
        states = [
            fit_canvas(self._render(endpoints, self._stroke_state(names, sub_a, sub_b, closed)))
        ]
        for group in groups:
            for move in group:
                value = dict(move.value)  # type: ignore[arg-type]
                index, end = int(value["color"]), str(value["end"])
                sub = sub_a if end == "a" else sub_b
                sub[index].extend(move.cells)
                if bool(value["closes"]):
                    closed[index] = True
            states.append(
                fit_canvas(
                    self._render(endpoints, self._stroke_state(names, sub_a, sub_b, closed))
                )
            )
        if not all(closed.values()):
            raise ForcedDeductionError("maze_flow_pairs: replayed chain leaves an open route")
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

        payload = {
            names[index]: [[row + 1, col + 1] for row, col in path]
            for index, path in paths.items()
        }
        answer_named_xy_paths(payload)
        self._narrate(groups, names, color_count, given_segments)

        given_paths = {
            names[index]: [
                [list(cell) for cell in paths[index][: givens[index][0]]],
                [
                    list(cell)
                    for cell in reversed(paths[index][len(paths[index]) - givens[index][1] :])
                ],
            ]
            for index in paths
        }
        return MazeFlowPairsRecord(
            answer=answer_named_xy_paths(payload),
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
                "endpoints": endpoints,
                "given_paths": given_paths,
                "solution_paths": {
                    names[index]: [list(cell) for cell in path]
                    for index, path in paths.items()
                },
            },
        )


__all__ = ["MazeFlowPairsGenerator"]
