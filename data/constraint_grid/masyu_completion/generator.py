"""Masyu completion: draw a closed loop through cell centers honoring black/white pearl rules."""

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
from data.constraint_grid.core.coordinate_format import display_cell_text, display_edge_text
from data.constraint_grid.core.scene import (
    AXIS_LABEL,
    GRID_BORDER,
    GRID_LIGHT,
    board_font,
    finish_board_scene,
    new_board_scene,
    standard_board_geometry,
)
from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.contracts.grid_completion import answer_xy_positions
from data.common.pipeline.grid_rendering import fit_canvas

Cell = Tuple[int, int]
Edge = Tuple[Cell, Cell]

_INPUT_DESCRIPTION = (
    "The image shows a square grid with small gray 1-based row and column numbers along the top and "
    "left edges, black and white circles at some cell centers, and red loop segments that are already "
    "drawn as part of the final loop."
)
_RULE_DESCRIPTION = (
    "Draw one closed non-crossing loop through cell centers, moving only horizontally or vertically. "
    "The loop passes through every circle: at a white circle it runs straight through the cell and "
    "turns in at least one of the two neighbouring cells along the loop; at a black circle it turns "
    "and runs straight through both neighbouring cells along the loop. The segments already drawn are "
    "fixed."
)
_TERMINAL_VISUAL = (
    "the finished board on which the complete loop is drawn in red through every circle while the "
    "circles, grid, and edge numbers stay unchanged"
)
_BG = (250, 251, 254)
_GRID_LIGHT = (210, 215, 225)
_GRID_BORDER = (40, 44, 52)
_TEXT = (20, 24, 32)
_LOOP = (210, 50, 60)
_BLACK_PEARL = (24, 26, 32)
_WHITE_PEARL = (255, 255, 255)
_PEARL_OUTLINE = (24, 26, 32)


@dataclass
class MasyuCompletionRecord:
    id: str
    task_name: str = "masyu_completion"
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


class MasyuCompletionGenerator(AbstractPuzzleGenerator[MasyuCompletionRecord]):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        rows: int = 6,
        cols: int = 6,
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

    def create_puzzle(self, *args, **kwargs) -> MasyuCompletionRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"masyu_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    # Build a Hamiltonian cycle on a subset of cells, then derive black/white pearls from corner pattern.
    def _build_loop(self) -> Optional[List[Tuple[int, int]]]:
        rows, cols = self.rows, self.cols
        # Strategy: start with a rectangular loop perimeter that covers the entire grid; then perturb.
        # Initialize: outer rectangle.
        path: List[Tuple[int, int]] = []
        for c in range(cols):
            path.append((0, c))
        for r in range(1, rows):
            path.append((r, cols - 1))
        for c in range(cols - 2, -1, -1):
            path.append((rows - 1, c))
        for r in range(rows - 2, 0, -1):
            path.append((r, 0))
        # Verify closed loop.
        # Try to perturb by inserting "bumps" that go inward.
        for _ in range(40):
            self._try_perturb(path)
        return path

    def _try_perturb(self, path: List[Tuple[int, int]]) -> None:
        rows, cols = self.rows, self.cols
        n = len(path)
        # Try to take a horizontal segment (r, c)->(r, c+1) and replace it with (r, c)->(r+1, c)->(r+1, c+1)->(r, c+1).
        # Or vertical analog. The new cells must not already be on the path.
        on_path = set(path)
        candidates = []
        for i in range(n):
            a = path[i]
            b = path[(i + 1) % n]
            dr = b[0] - a[0]
            dc = b[1] - a[1]
            for offset_dr, offset_dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                # The perturbation goes in a direction perpendicular to (dr,dc).
                if dr * offset_dr + dc * offset_dc != 0:
                    continue
                a2 = (a[0] + offset_dr, a[1] + offset_dc)
                b2 = (b[0] + offset_dr, b[1] + offset_dc)
                if not (0 <= a2[0] < rows and 0 <= a2[1] < cols):
                    continue
                if not (0 <= b2[0] < rows and 0 <= b2[1] < cols):
                    continue
                if a2 in on_path or b2 in on_path:
                    continue
                candidates.append((i, a2, b2))
        if not candidates:
            return
        i, a2, b2 = self.rng.choice(candidates)
        # Insert a2 and b2 between i and i+1.
        path.insert(i + 1, a2)
        path.insert(i + 2, b2)

    def _classify_loop_cells(self, path: List[Tuple[int, int]]) -> Dict[Tuple[int, int], str]:
        """Return mapping cell -> 'straight' | 'turn' for every cell on the loop, where turn means
        the loop changes direction at this cell."""
        n = len(path)
        result: Dict[Tuple[int, int], str] = {}
        for i in range(n):
            prev = path[(i - 1) % n]
            cur = path[i]
            nxt = path[(i + 1) % n]
            d1 = (cur[0] - prev[0], cur[1] - prev[1])
            d2 = (nxt[0] - cur[0], nxt[1] - cur[1])
            if d1 == d2:
                result[cur] = "straight"
            else:
                result[cur] = "turn"
        return result

    def _classify_for_pearls(self, path: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Produce (black_pearls, white_pearls)."""
        n = len(path)
        kind = self._classify_loop_cells(path)
        white_candidates: List[Tuple[int, int]] = []
        black_candidates: List[Tuple[int, int]] = []
        for i in range(n):
            cur = path[i]
            prev = path[(i - 1) % n]
            nxt = path[(i + 1) % n]
            if kind[cur] == "turn":
                # Black pearl candidate: entering and leaving cells must both be straight (loop must go
                # straight in both adjacent cells along the loop).
                # Adjacent loop cells = prev and nxt.
                if kind[prev] == "straight" and kind[nxt] == "straight":
                    black_candidates.append(cur)
            else:
                # White pearl candidate: at least one of prev or nxt must be a turn.
                if kind[prev] == "turn" or kind[nxt] == "turn":
                    white_candidates.append(cur)
        return black_candidates, white_candidates

    def _build_instance(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        for _ in range(20):
            path = self._build_loop()
            if path is None:
                continue
            black_cands, white_cands = self._classify_for_pearls(path)
            if len(black_cands) < 2 or len(white_cands) < 2:
                continue
            self.rng.shuffle(black_cands)
            self.rng.shuffle(white_cands)
            black = black_cands[:max(2, len(black_cands) // 3)]
            white = white_cands[:max(2, len(white_cands) // 3)]
            return path, black, white
        raise RuntimeError("Failed to generate Masyu instance")

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_edge(a: Cell, b: Cell) -> Edge:
        return tuple(sorted((a, b)))

    def _cell_edges(self, cell: Cell) -> List[Edge]:
        r, c = cell
        result = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                result.append(self._norm_edge((r, c), (nr, nc)))
        return result

    @staticmethod
    def _edge_phrase(edge: Edge) -> str:
        return f"the segment {display_edge_text(edge[0], edge[1])}"

    def _forced_chain(
        self,
        solution_edges: Set[Edge],
        blacks: Sequence[Cell],
        whites: Sequence[Cell],
        given: Set[Edge],
    ) -> Optional[ForcedChain]:
        rows, cols = self.rows, self.cols
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        state: Dict[Edge, Optional[bool]] = {}
        for cell in cells:
            for edge in self._cell_edges(cell):
                state.setdefault(edge, None)
        for edge in given:
            state[edge] = True
        loop_cells: Set[Cell] = set(blacks) | set(whites)
        for a, b in given:
            loop_cells.add(a)
            loop_cells.add(b)
        nonloop: Set[Cell] = set()
        pending: List[str] = []
        rounds: List[List[ForcedMove]] = []

        def cell_status(cell: Cell):
            edges = self._cell_edges(cell)
            on = [e for e in edges if state[e] is True]
            unknown = [e for e in edges if state[e] is None]
            return edges, on, unknown

        def other_end(edge: Edge, cell: Cell) -> Cell:
            return edge[0] if edge[1] == cell else edge[1]

        def straight_pairs(cell: Cell) -> List[Tuple[Edge, Edge]]:
            r, c = cell
            pairs = []
            horizontal = [
                self._norm_edge((r, c), (r, c - 1)) if c - 1 >= 0 else None,
                self._norm_edge((r, c), (r, c + 1)) if c + 1 < cols else None,
            ]
            vertical = [
                self._norm_edge((r, c), (r - 1, c)) if r - 1 >= 0 else None,
                self._norm_edge((r, c), (r + 1, c)) if r + 1 < rows else None,
            ]
            if all(horizontal):
                pairs.append(tuple(horizontal))
            if all(vertical):
                pairs.append(tuple(vertical))
            return pairs

        while {edge for edge, value in state.items() if value is True} != solution_edges:
            forced_on: Dict[Edge, str] = {}
            new_off: List[Tuple[Edge, str]] = []

            # Degree bookkeeping.
            for cell in cells:
                edges, on, unknown = cell_status(cell)
                if len(on) > 0:
                    loop_cells.add(cell)
                if len(on) == 2 and unknown:
                    for edge in unknown:
                        new_off.append(
                            (edge, f"{display_cell_text(cell)} already carries "
                                   "its two loop segments")
                        )
                elif cell in loop_cells and len(on) + len(unknown) == 2 and unknown:
                    for edge in unknown:
                        other_end(edge, cell)
                        forced_on.setdefault(
                            edge,
                            f"The loop through {display_cell_text(cell)} has a "
                            f"single open continuation, so {self._edge_phrase(edge)} is drawn",
                        )
                if cell in nonloop:
                    for edge in unknown:
                        new_off.append(
                            (edge, f"{display_cell_text(cell)} is off the loop")
                        )
                if not on and not unknown:
                    nonloop.add(cell)

            # White circles: run straight through, so one of the two straight
            # pairs must be fully drawn.
            for cell in whites:
                pairs = straight_pairs(cell)
                viable = [
                    pair for pair in pairs
                    if state[pair[0]] is not False and state[pair[1]] is not False
                    and other_end(pair[0], cell) not in nonloop
                    and other_end(pair[1], cell) not in nonloop
                ]
                if not viable:
                    return None
                if len(viable) == 1:
                    for edge in viable[0]:
                        if state[edge] is None:
                            forced_on.setdefault(
                                edge,
                                f"The white circle at {display_cell_text(cell)} "
                                f"can only run straight one way, so {self._edge_phrase(edge)} "
                                "is drawn",
                            )

            # Black circles: turn here with two straight arms of length two.
            for cell in blacks:
                r, c = cell
                for axis, options in (
                    ("horizontal", ((0, 1), (0, -1))),
                    ("vertical", ((1, 0), (-1, 0))),
                ):
                    viable_arms = []
                    for dr, dc in options:
                        n1 = (r + dr, c + dc)
                        n2 = (r + 2 * dr, c + 2 * dc)
                        if not (0 <= n2[0] < rows and 0 <= n2[1] < cols):
                            continue
                        e1 = self._norm_edge(cell, n1)
                        e2 = self._norm_edge(n1, n2)
                        if state[e1] is False or state[e2] is False:
                            continue
                        if n1 in nonloop or n2 in nonloop:
                            continue
                        viable_arms.append((e1, e2))
                    if not viable_arms:
                        return None
                    if len(viable_arms) == 1:
                        for edge in viable_arms[0]:
                            if state[edge] is None:
                                forced_on.setdefault(
                                    edge,
                                    f"The black circle at {display_cell_text((r, c))} has a "
                                    f"single way to point its {axis} arm, so "
                                    f"{self._edge_phrase(edge)} is drawn",
                                )

            # Premature closure.
            parent: Dict[Cell, Cell] = {}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for edge, value in state.items():
                if value is True:
                    for endpoint in edge:
                        parent.setdefault(endpoint, endpoint)
            for edge, value in state.items():
                if value is True:
                    a, b = find(edge[0]), find(edge[1])
                    if a != b:
                        parent[a] = b
            roots = {find(x) for x in parent}
            for edge in sorted(state):
                if state[edge] is not None or edge in forced_on:
                    continue
                a, b = edge
                if a in parent and b in parent and find(a) == find(b):
                    on_set = {e for e, value in state.items() if value is True}
                    closed = on_set | {edge}
                    if len(roots) == 1 and closed == solution_edges:
                        forced_on.setdefault(
                            edge,
                            f"Only {self._edge_phrase(edge)} remains to close the single line "
                            "into one loop through every circle, so it is drawn",
                        )
                    else:
                        new_off.append(
                            (edge, f"closing the line early at {self._edge_phrase(edge)} would "
                                   "leave a second loop")
                        )

            if forced_on:
                round_moves: List[ForcedMove] = []
                for edge in sorted(forced_on):
                    if edge not in solution_edges:
                        raise ForcedDeductionError(
                            "masyu_completion: forced segment deviates from the loop"
                        )
                    reason = forced_on[edge]
                    if pending:
                        reason = "; ".join(
                            ["note that " + fact for fact in pending[:2]] + [reason]
                        )
                        reason = reason[0].upper() + reason[1:]
                        pending = []
                    state[edge] = True
                    for endpoint in edge:
                        loop_cells.add(endpoint)
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
                            "masyu_completion: eliminated a true loop segment"
                        )
                    state[edge] = False
                    progressed = True
                    if fact not in pending and len(pending) < 4:
                        pending.append(fact)
            # Mark fully-dead cells as off the loop.
            for cell in cells:
                if cell in loop_cells or cell in nonloop:
                    continue
                edges, on, unknown = cell_status(cell)
                if not on and len(unknown) <= 1:
                    nonloop.add(cell)
                    progressed = True
            if not progressed:
                return None
        return ForcedChain(rounds=rounds)

    def _construct_instance(self):
        for _ in range(60):
            try:
                path, blacks, whites = self._build_instance()
            except Exception:
                continue
            n = len(path)
            solution_edges = {
                self._norm_edge(path[i], path[(i + 1) % n]) for i in range(n)
            }
            given: Set[Edge] = set(solution_edges)
            order = sorted(given)
            self.rng.shuffle(order)
            for edge in order:
                given.discard(edge)
                if self._forced_chain(solution_edges, blacks, whites, given) is None:
                    given.add(edge)
            chain = self._forced_chain(solution_edges, blacks, whites, given)
            open_count = len(solution_edges) - len(given)
            if chain is None or open_count < 6 or len(chain.rounds) < 2:
                continue
            if len(given) > len(solution_edges) // 2 or not given:
                continue
            prove_complete_chain(
                chain, expected_cells=open_count * 2, task_name="masyu_completion"
            )
            return path, blacks, whites, sorted(given), chain
        raise ForcedDeductionError(
            "masyu_completion: failed to construct a forced-hand-solvable loop"
        )

    def _render(
        self,
        committed_edges: Sequence[Edge],
        blacks: Sequence[Cell],
        whites: Sequence[Cell],
    ) -> Image.Image:
        rows, cols = self.rows, self.cols
        cell, left, top = standard_board_geometry(rows, cols)
        scene, draw = new_board_scene()
        label_font = board_font(max(12, min(20, cell // 4)))

        for r in range(rows + 1):
            y = top + r * cell
            draw.line([(left, y), (left + cols * cell, y)], fill=GRID_LIGHT, width=1)
        for c in range(cols + 1):
            x = left + c * cell
            draw.line([(x, top), (x, top + rows * cell)], fill=GRID_LIGHT, width=1)
        draw.rectangle([left, top, left + cols * cell, top + rows * cell], outline=GRID_BORDER, width=3)

        for r in range(rows):
            draw.text((left - 16, top + r * cell + cell // 2), str(r + 1),
                      fill=AXIS_LABEL, font=label_font, anchor="mm")
        for c in range(cols):
            draw.text((left + c * cell + cell // 2, top - 16), str(c + 1),
                      fill=AXIS_LABEL, font=label_font, anchor="mm")

        for (a, b) in committed_edges:
            ax = left + a[1] * cell + cell // 2
            ay = top + a[0] * cell + cell // 2
            bx = left + b[1] * cell + cell // 2
            by = top + b[0] * cell + cell // 2
            draw.line([(ax, ay), (bx, by)], fill=_LOOP, width=6)

        radius = max(8, cell // 3)
        for r, c in blacks:
            cx = left + c * cell + cell // 2
            cy = top + r * cell + cell // 2
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                         fill=_BLACK_PEARL, outline=_PEARL_OUTLINE, width=2)
        for r, c in whites:
            cx = left + c * cell + cell // 2
            cy = top + r * cell + cell // 2
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                         fill=_WHITE_PEARL, outline=_PEARL_OUTLINE, width=2)
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
                "appear in deduction order: each wave of segments the circle rules and line ends "
                "force fades in together, earlier segments stay exactly in place, and the video "
                f"ends holding {_TERMINAL_VISUAL}."
            ),
            rule_description=_RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    def _narrate(
        self,
        groups: Sequence[Sequence[ForcedMove]],
        blacks: Sequence[Cell],
        whites: Sequence[Cell],
        given_count: int,
    ) -> List[str]:
        opener = (
            f"The board is {self.rows} by {self.cols} with {len(blacks)} black and {len(whites)} "
            f"white circles, and {given_count} loop segment"
            f"{'s are' if given_count != 1 else ' is'} already drawn. White circles run straight "
            "with a turn beside them, black circles turn between two straight arms, and every "
            "visited cell carries exactly two segments, so I will extend the line where those "
            "facts leave a single choice."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " The segments now form one closed loop that honors every circle, so the loop is "
            "complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    def create_random_puzzle(self) -> MasyuCompletionRecord:
        path, black, white, given_edges, chain = self._construct_instance()
        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        committed: List[Edge] = list(given_edges)
        states = [fit_canvas(self._render(committed, black, white))]
        for group in groups:
            committed = committed + [tuple(move.cells) for move in group]
            states.append(fit_canvas(self._render(committed, black, white)))
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

        path_payload = [list(p) for p in path] + [list(path[0])]
        visible_path = [[row + 1, col + 1] for row, col in path_payload]

        return MasyuCompletionRecord(
            answer=answer_xy_positions(visible_path),
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
                "loop_path": path_payload,
                "given_edges": [
                    [a[0] + 1, a[1] + 1, b[0] + 1, b[1] + 1] for a, b in given_edges
                ],
                "black_pearls": [list(p) for p in black],
                "white_pearls": [list(p) for p in white],
            },
        )
