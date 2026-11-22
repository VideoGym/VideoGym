"""Yajilin completion: place black cells (no two adjacent) and draw a closed loop through every white cell;
arrow clues count black cells in their indicated direction."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
    board_font,
    finish_board_scene,
    new_board_scene,
    standard_board_geometry,
)

from PIL import Image

from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.contracts.grid_completion import answer_xy_positions
from data.common.pipeline.grid_rendering import fit_canvas

_INPUT_DESCRIPTION = (
    "The image shows a square grid with small gray 1-based row and column numbers along the top and "
    "left edges. Boxed clue cells carry an arrow and a number. Cells already shaded dark and red loop "
    "segments already drawn are part of the final answer."
)
_RULE_DESCRIPTION = (
    "Shade some empty cells dark and draw one closed non-crossing loop through the centers of all "
    "remaining empty cells, moving only horizontally or vertically. A clue's number counts the dark "
    "cells lying beyond it in its arrow's direction, no two dark cells may share an edge, and the "
    "loop never enters clue or dark cells. The shades and segments already shown are fixed."
)
_TERMINAL_VISUAL = (
    "the finished board on which every dark cell is shaded and the complete loop is drawn in red "
    "while the clue boxes, grid, and edge numbers stay unchanged"
)
_BG = (250, 251, 254)
_GRID_LIGHT = (210, 215, 225)
_GRID_BORDER = (40, 44, 52)
_TEXT = (24, 28, 36)
_LOOP = (210, 50, 60)
_BLACK = (32, 36, 44)
_CLUE_FILL = (235, 235, 240)


@dataclass
class YajilinCompletionRecord:
    id: str
    task_name: str = "yajilin_completion"
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


class YajilinCompletionGenerator(AbstractPuzzleGenerator[YajilinCompletionRecord]):
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

    def create_puzzle(self, *args, **kwargs) -> YajilinCompletionRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"yajilin_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    def _build_loop(self) -> List[Tuple[int, int]]:
        rows, cols = self.rows, self.cols
        path: List[Tuple[int, int]] = []
        for c in range(cols):
            path.append((0, c))
        for r in range(1, rows):
            path.append((r, cols - 1))
        for c in range(cols - 2, -1, -1):
            path.append((rows - 1, c))
        for r in range(rows - 2, 0, -1):
            path.append((r, 0))
        # Keep a bounded number of off-loop cells for the black/clue layer.
        # Repeatedly perturbing until no move remained often produced a
        # Hamiltonian cycle, after which no legal Yajilin instance could be
        # authored and otherwise healthy seeds exhausted the rejection loop.
        target_perturbations = self.rng.randint(2, 5)
        successful = 0
        attempts = 0
        while successful < target_perturbations and attempts < 40:
            successful += int(self._perturb(path))
            attempts += 1
        return path

    def _perturb(self, path: List[Tuple[int, int]]) -> bool:
        rows, cols = self.rows, self.cols
        n = len(path)
        on_path = set(path)
        candidates = []
        for i in range(n):
            a = path[i]
            b = path[(i + 1) % n]
            dr = b[0] - a[0]
            dc = b[1] - a[1]
            for offset_dr, offset_dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
            return False
        i, a2, b2 = self.rng.choice(candidates)
        path.insert(i + 1, a2)
        path.insert(i + 2, b2)
        return True

    def _build_instance(self):
        rows, cols = self.rows, self.cols
        for _ in range(200):
            path = self._build_loop()
            on_path = set(path)
            free_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in on_path]
            if len(free_cells) < 4:
                continue
            self.rng.shuffle(free_cells)
            black: List[Tuple[int, int]] = []
            for cell in free_cells:
                black_set = set(black)
                if not any(
                    (cell[0] + dr, cell[1] + dc) in black_set
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ) and self.rng.random() < 0.7:
                    black.append(cell)
            if len(black) < 1:
                black.append(free_cells[0])
            # Standard Yajilin requires every non-clue cell to be either on
            # the loop or black.  Therefore all remaining off-loop cells are
            # clue cells; leaving decorative unused white cells would make the
            # generated reference itself illegal.
            clue_cells = [cell for cell in free_cells if cell not in set(black)]
            while len(clue_cells) < 2 and len(black) > 1:
                clue_cells.append(black.pop())
            if len(black) < 1 or len(clue_cells) < 2:
                continue
            black_set = set(black)
            clues = []
            directions = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
            for cell in clue_cells:
                dnames = list(directions.keys())
                self.rng.shuffle(dnames)
                # Allow zero-count clues too.
                best_dname = None
                best_count = -1
                for dname in dnames:
                    dr, dc = directions[dname]
                    count = 0
                    nr, nc = cell[0] + dr, cell[1] + dc
                    while 0 <= nr < rows and 0 <= nc < cols:
                        if (nr, nc) in black_set:
                            count += 1
                        nr += dr
                        nc += dc
                    if count >= 1:
                        clues.append((cell[0], cell[1], dname, count))
                        best_dname = dname
                        break
                    elif count > best_count:
                        best_count = count
                        best_dname = dname
                if best_dname and not any(cl[0] == cell[0] and cl[1] == cell[1] for cl in clues):
                    # Fallback to a count-0 clue.
                    clues.append((cell[0], cell[1], best_dname, 0))
            if len(clues) < 2:
                continue
            return path, list(black), clues
        raise RuntimeError("Failed to generate yajilin instance")

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_edge(a, b):
        return tuple(sorted((a, b)))

    @staticmethod
    def _edge_phrase(edge) -> str:
        return f"the segment {display_edge_text(edge[0], edge[1])}"

    def _forced_chain(
        self,
        solution_edges: Set,
        solution_black: Set,
        clues: Sequence[Tuple[int, int, str, int]],
        given_black: Set,
        given_edges: Set,
    ) -> Optional[ForcedChain]:
        rows, cols = self.rows, self.cols
        directions = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        clue_cells = {(r, c) for r, c, _, _ in clues}
        cells = [
            (r, c) for r in range(rows) for c in range(cols) if (r, c) not in clue_cells
        ]
        cell_state: Dict[Tuple[int, int], Optional[str]] = {cell: None for cell in cells}
        for cell in given_black:
            cell_state[cell] = "black"
        edge_state: Dict[Tuple, Optional[bool]] = {}
        for r, c in cells:
            for dr, dc in ((1, 0), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in cell_state:
                    edge_state[self._norm_edge((r, c), nb)] = None
        for edge in given_edges:
            edge_state[edge] = True
            for endpoint in edge:
                cell_state[endpoint] = "loop"
        pending: List[str] = []
        rounds: List[List[ForcedMove]] = []

        def neighbors(cell):
            r, c = cell
            return [
                (r + dr, c + dc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (r + dr, c + dc) in cell_state
            ]

        def viable_edges(cell):
            result = []
            for nb in neighbors(cell):
                edge = self._norm_edge(cell, nb)
                if edge_state.get(edge) is False:
                    continue
                if cell_state.get(nb) == "black":
                    continue
                result.append(edge)
            return result

        while not (
            {e for e, v in edge_state.items() if v is True} == solution_edges
            and {c for c, v in cell_state.items() if v == "black"} == solution_black
        ):
            forced_black: Dict[Tuple[int, int], str] = {}
            forced_loop: Dict[Tuple[int, int], str] = {}
            forced_on: Dict[Tuple, str] = {}
            new_off: List[Tuple] = []

            # Clue arithmetic along each arrow ray.
            for r, c, dname, count in clues:
                dr, dc = directions[dname]
                ray = []
                nr, nc = r + dr, c + dc
                while 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) in cell_state:
                        ray.append((nr, nc))
                    nr += dr
                    nc += dc
                have = sum(1 for cell in ray if cell_state[cell] == "black")
                open_cells = [
                    cell for cell in ray
                    if cell_state[cell] is None
                    and not any(cell_state.get(nb) == "black" for nb in neighbors(cell))
                ]
                remaining = count - have
                if remaining == 0:
                    for cell in [cell for cell in ray if cell_state[cell] is None]:
                        forced_loop.setdefault(
                            cell,
                            f"The {count} clue at {display_cell_text((r, c))} already sees all "
                            f"its dark cells, so {display_cell_text(cell)} joins "
                            "the loop",
                        )
                elif remaining == len(open_cells) and open_cells:
                    adjacent_pair = any(
                        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                        for i, a in enumerate(open_cells)
                        for b in open_cells[i + 1:]
                    )
                    if not adjacent_pair:
                        for cell in open_cells:
                            forced_black.setdefault(
                                cell,
                                f"The {count} clue at {display_cell_text((r, c))} still needs "
                                f"{remaining} dark cell{'s' if remaining != 1 else ''} and has "
                                f"exactly that many spots left, so {display_cell_text(cell)} is shaded",
                            )

            # Neighbours of dark cells stay on the loop.
            for cell in cells:
                if cell_state[cell] is not None or cell in forced_black:
                    continue
                dark_nb = [nb for nb in neighbors(cell) if cell_state.get(nb) == "black"]
                if dark_nb:
                    nb = dark_nb[0]
                    forced_loop.setdefault(
                        cell,
                        f"{display_cell_text(cell)} touches the dark cell at "
                        f"{display_cell_text(nb)}, so it must carry the loop",
                    )

            # A cell that cannot host two loop segments must be dark.
            for cell in cells:
                if cell_state[cell] is not None or cell in forced_black or cell in forced_loop:
                    continue
                if len(viable_edges(cell)) < 2:
                    forced_black.setdefault(
                        cell,
                        f"{display_cell_text(cell)} has fewer than two open "
                        "sides for the loop, so it must be shaded dark",
                    )

            # Loop degree propagation.
            for cell in cells:
                if cell_state[cell] == "black":
                    continue
                on = [
                    e for e in viable_edges(cell) if edge_state.get(e) is True
                ]
                if cell_state[cell] is None and on:
                    forced_loop.setdefault(
                        cell,
                        f"A drawn segment already reaches {display_cell_text(cell)}, so it lies on the loop",
                    )
                if cell_state[cell] != "loop":
                    continue
                viable = viable_edges(cell)
                on = [e for e in viable if edge_state.get(e) is True]
                unknown = [e for e in viable if edge_state.get(e) is None]
                if len(on) == 2:
                    for edge in unknown:
                        new_off.append(
                            (edge, f"{display_cell_text(cell)} already carries "
                                   "its two loop segments")
                        )
                elif len(on) + len(unknown) == 2 and unknown:
                    for edge in unknown:
                        forced_on.setdefault(
                            edge,
                            f"The loop through {display_cell_text(cell)} has a "
                            f"single open continuation, so {self._edge_phrase(edge)} is drawn",
                        )

            # Premature closure.
            parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for edge, value in edge_state.items():
                if value is True:
                    for endpoint in edge:
                        parent.setdefault(endpoint, endpoint)
            for edge, value in edge_state.items():
                if value is True:
                    a, b = find(edge[0]), find(edge[1])
                    if a != b:
                        parent[a] = b
            roots = {find(x) for x in parent}
            for edge in sorted(edge_state):
                if edge_state[edge] is not None or edge in forced_on:
                    continue
                a, b = edge
                if a in parent and b in parent and find(a) == find(b):
                    closed = {e for e, v in edge_state.items() if v is True} | {edge}
                    if len(roots) == 1 and closed == solution_edges:
                        forced_on.setdefault(
                            edge,
                            f"Only {self._edge_phrase(edge)} remains to close the single line "
                            "into one loop through every open cell, so it is drawn",
                        )
                    else:
                        new_off.append(
                            (edge, f"closing the line early at {self._edge_phrase(edge)} "
                                   "would leave a second loop")
                        )

            if forced_black or forced_on:
                round_moves: List[ForcedMove] = []
                for cell in sorted(forced_black):
                    if cell not in solution_black:
                        raise ForcedDeductionError(
                            "yajilin_completion: forced dark cell deviates from the solution"
                        )
                    reason = forced_black[cell]
                    if pending:
                        reason = "; ".join(
                            ["note that " + fact for fact in pending[:2]] + [reason]
                        )
                        reason = reason[0].upper() + reason[1:]
                        pending = []
                    cell_state[cell] = "black"
                    round_moves.append(
                        ForcedMove(kind="dark", cells=(cell,), value="black", reason=reason)
                    )
                for edge in sorted(forced_on):
                    if edge not in solution_edges:
                        raise ForcedDeductionError(
                            "yajilin_completion: forced segment deviates from the loop"
                        )
                    reason = forced_on[edge]
                    if pending:
                        reason = "; ".join(
                            ["note that " + fact for fact in pending[:2]] + [reason]
                        )
                        reason = reason[0].upper() + reason[1:]
                        pending = []
                    edge_state[edge] = True
                    for endpoint in edge:
                        cell_state[endpoint] = "loop"
                    round_moves.append(
                        ForcedMove(kind="segment", cells=edge, value="on", reason=reason)
                    )
                rounds.append(round_moves)
                continue

            progressed = False
            for cell, reason in forced_loop.items():
                if cell_state[cell] is None:
                    if cell in solution_black:
                        raise ForcedDeductionError(
                            "yajilin_completion: marked a true dark cell as loop"
                        )
                    cell_state[cell] = "loop"
                    progressed = True
                    fact = reason[0].lower() + reason[1:]
                    if fact not in pending and len(pending) < 4:
                        pending.append(fact)
            for edge, fact in new_off:
                if edge_state.get(edge) is None:
                    if edge in solution_edges:
                        raise ForcedDeductionError(
                            "yajilin_completion: eliminated a true loop segment"
                        )
                    edge_state[edge] = False
                    progressed = True
                    if fact not in pending and len(pending) < 4:
                        pending.append(fact)
            if not progressed:
                return None
        return ForcedChain(rounds=rounds)

    def _construct_instance(self):
        for _ in range(80):
            try:
                path, black, clues = self._build_instance()
            except Exception:
                continue
            n = len(path)
            solution_edges = {
                self._norm_edge(path[i], path[(i + 1) % n]) for i in range(n)
            }
            solution_black = set(black)
            marks = [("black", cell) for cell in sorted(solution_black)] + [
                ("edge", edge) for edge in sorted(solution_edges)
            ]
            given_black: Set = set(solution_black)
            given_edges: Set = set(solution_edges)
            self.rng.shuffle(marks)
            for kind, mark in marks:
                if kind == "black":
                    given_black.discard(mark)
                else:
                    given_edges.discard(mark)
                if self._forced_chain(
                    solution_edges, solution_black, clues, given_black, given_edges
                ) is None:
                    if kind == "black":
                        given_black.add(mark)
                    else:
                        given_edges.add(mark)
            chain = self._forced_chain(
                solution_edges, solution_black, clues, given_black, given_edges
            )
            open_marks = (len(solution_black) - len(given_black)) + (
                len(solution_edges) - len(given_edges)
            )
            if chain is None or open_marks < 6 or len(chain.rounds) < 2:
                continue
            if len(given_edges) > len(solution_edges) // 2:
                continue
            expected = (len(solution_black) - len(given_black)) + 2 * (
                len(solution_edges) - len(given_edges)
            )
            prove_complete_chain(chain, expected_cells=expected, task_name="yajilin_completion")
            return path, black, clues, sorted(given_black), sorted(given_edges), chain
        raise ForcedDeductionError(
            "yajilin_completion: failed to construct a forced-hand-solvable board"
        )

    def _render(self, committed_edges, committed_black, clues) -> Image.Image:
        rows, cols = self.rows, self.cols
        cell, left, top = standard_board_geometry(rows, cols)
        scene, draw = new_board_scene()
        font = board_font(max(16, cell * 2 // 5))
        small_font = board_font(max(12, cell // 4))
        label_font = board_font(max(12, min(20, cell // 4)))

        for r in range(rows + 1):
            y = top + r * cell
            draw.line([(left, y), (left + cols * cell, y)], fill=_GRID_LIGHT, width=1)
        for c in range(cols + 1):
            x = left + c * cell
            draw.line([(x, top), (x, top + rows * cell)], fill=_GRID_LIGHT, width=1)
        draw.rectangle([left, top, left + cols * cell, top + rows * cell], outline=_GRID_BORDER, width=3)

        for r in range(rows):
            draw.text((left - 16, top + r * cell + cell // 2), str(r + 1),
                      fill=AXIS_LABEL, font=label_font, anchor="mm")
        for c in range(cols):
            draw.text((left + c * cell + cell // 2, top - 16), str(c + 1),
                      fill=AXIS_LABEL, font=label_font, anchor="mm")

        for r, c in committed_black:
            x0 = left + c * cell
            y0 = top + r * cell
            inset = 4
            draw.rectangle([x0 + inset, y0 + inset, x0 + cell - inset, y0 + cell - inset],
                           fill=_BLACK)

        for (a, b) in committed_edges:
            ax = left + a[1] * cell + cell // 2
            ay = top + a[0] * cell + cell // 2
            bx = left + b[1] * cell + cell // 2
            by = top + b[0] * cell + cell // 2
            draw.line([(ax, ay), (bx, by)], fill=_LOOP, width=6)

        arrow_glyphs = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
        for r, c, dname, count in clues:
            x0 = left + c * cell
            y0 = top + r * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=_CLUE_FILL,
                           outline=_GRID_BORDER, width=1)
            draw.text((x0 + cell // 4, y0 + cell // 4), str(count),
                      fill=_TEXT, font=small_font, anchor="mm")
            draw.text((x0 + cell * 3 // 4, y0 + cell * 3 // 4), arrow_glyphs[dname],
                      fill=_TEXT, font=font, anchor="mm")
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
                "Generate a solution video that starts from the unchanged puzzle. Dark shades and "
                "red segments then appear in deduction order: each wave the arrow counts and loop "
                "rules force fades in together, earlier commitments stay exactly in place, and the "
                f"video ends holding {_TERMINAL_VISUAL}."
            ),
            rule_description=_RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    def _narrate(
        self,
        groups,
        clues,
        given_black_count: int,
        given_edge_count: int,
    ) -> List[str]:
        opener = (
            f"The board is {self.rows} by {self.cols} with {len(clues)} arrow clues; "
            f"{given_black_count} dark cell{'s are' if given_black_count != 1 else ' is'} "
            f"already shaded and {given_edge_count} loop segment"
            f"{'s are' if given_edge_count != 1 else ' is'} already drawn. Arrows count the dark "
            "cells beyond them, dark cells never touch, and the loop must pass through every "
            "other empty cell, so I will commit shades and segments where those facts leave a "
            "single choice."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " Every empty cell is now dark or on the loop, the loop closes as a single line, no "
            "two dark cells touch, and every arrow count is exact, so the board is complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    def create_random_puzzle(self) -> YajilinCompletionRecord:
        path, black, clues, given_black, given_edges, chain = self._construct_instance()
        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        edges = list(given_edges)
        darks = list(given_black)
        states = [fit_canvas(self._render(list(edges), list(darks), clues))]
        for group in groups:
            for move in group:
                if move.kind == "dark":
                    darks.append(move.cells[0])
                else:
                    edges.append(tuple(move.cells))
            states.append(fit_canvas(self._render(list(edges), list(darks), clues)))
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

        return YajilinCompletionRecord(
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
                "given_black": [list(cell) for cell in given_black],
                "given_edges": [
                    [a[0] + 1, a[1] + 1, b[0] + 1, b[1] + 1] for a, b in given_edges
                ],
                "black_cells": [list(b) for b in black],
                "clues": [[r, c, d, n] for r, c, d, n in clues],
            },
        )
