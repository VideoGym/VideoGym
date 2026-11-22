"""Nurikabe completion: shade the sea around numbered islands by forced deduction. Every emitted puzzle is forced-hand solvable: propagation over the visible
rules (a finished island seals its border; a cell out of every island's reach
is sea; an unfinished island with one growth cell grows there; three sea
cells in a 2x2 square keep the fourth island; a cell touching two different
islands is sea; a cut cell that alone keeps the committed sea connected is
sea) decides every cell from the clues and the given cells with zero
guessing. The generator proves the chain per sample, revealing
settling-consistent given cells only when the engine stalls; the chain drives
the authored CoT and the TI2V commit order."""

from __future__ import annotations

import random
from collections import deque
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
    reveal_budget_guard,
)
from data.constraint_grid.core.coordinate_format import display_cell_text
from data.constraint_grid.core.scene import (
    GRID_BORDER,
    GRID_LIGHT,
    TEXT,
    board_font,
    draw_axis_labels,
    finish_board_scene,
    new_board_scene,
    standard_board_geometry,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.contracts.grid_completion import answer_json
from data.common.pipeline.grid_rendering import fit_canvas
from data.common.prompts import build_sectioned_prompt

Cell = Tuple[int, int]

_BLACK = (32, 36, 44)
_WHITE_FILL = (255, 255, 255)
_DOT = (110, 125, 150)

_INPUT_DESCRIPTION = (
    "The image shows a square grid with small gray 1-based row and column numbers along the top and "
    "left edges. Numbered cells seed the white islands. Cells already shaded dark are confirmed sea, "
    "and cells carrying a small gray dot are confirmed island ground."
)
_RULE_DESCRIPTION = (
    "Shade cells dark so that every numbered cell sits in a white island whose cell count equals its "
    "number, different islands never share an edge, all dark sea cells form one edge-connected body, "
    "and no 2 by 2 square is entirely dark. The sea and island cells already shown are fixed."
)
_TERMINAL_VISUAL = (
    "the finished board on which every sea cell is shaded dark and every island cell without a number "
    "carries the small gray dot, while the numbers and edge numbers stay unchanged"
)
@dataclass
class NurikabeCompletionRecord:
    id: str
    task_name: str = "nurikabe_completion"
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


class NurikabeCompletionGenerator(AbstractPuzzleGenerator[NurikabeCompletionRecord]):
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

    def create_puzzle(self, *args, **kwargs) -> NurikabeCompletionRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"nurikabe_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    def _neighbors4(self, r: int, c: int) -> List[Cell]:
        return [
            (r + dr, c + dc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols
        ]

    # ------------------------------------------------------------------
    # Solution-first construction
    # ------------------------------------------------------------------

    def _build_solution(self) -> Tuple[List[List[int]], List[Tuple[int, int, int]]]:
        rows, cols = self.rows, self.cols
        for _ in range(800):
            grid = [[1 for _ in range(cols)] for _ in range(rows)]
            island_cells: List[List[Cell]] = []
            clues: List[Tuple[int, int, int]] = []
            target_white = (rows * cols) // 2
            white_count = 0
            attempts = 0
            while white_count < target_white and attempts < 60:
                attempts += 1
                candidates = [
                    (r, c)
                    for r in range(rows)
                    for c in range(cols)
                    if grid[r][c] == 1 and not any(grid[nr][nc] == 0 for nr, nc in self._neighbors4(r, c))
                ]
                if not candidates:
                    break
                seed = self.rng.choice(candidates)
                size = self.rng.choices([1, 2, 3, 4], weights=[2, 3, 3, 2])[0]
                cells = [seed]
                grid[seed[0]][seed[1]] = 0
                ok = True
                for _ in range(size - 1):
                    expansions: Set[Cell] = set()
                    for r, c in cells:
                        for nr, nc in self._neighbors4(r, c):
                            if grid[nr][nc] != 1:
                                continue
                            if any(
                                grid[ar][ac] == 0 and (ar, ac) not in cells
                                for ar, ac in self._neighbors4(nr, nc)
                            ):
                                continue
                            expansions.add((nr, nc))
                    if not expansions:
                        ok = False
                        break
                    pick = self.rng.choice(sorted(expansions))
                    grid[pick[0]][pick[1]] = 0
                    cells.append(pick)
                if not ok or len(cells) != size:
                    for r, c in cells:
                        grid[r][c] = 1
                    continue
                island_cells.append(cells)
                clue_cell = self.rng.choice(cells)
                clues.append((clue_cell[0], clue_cell[1], size))
                white_count += size
            if not island_cells or not any(size >= 2 for _, _, size in clues):
                continue
            if not self._has_no_2x2_black(grid) or not self._black_connected(grid):
                continue
            return grid, sorted(clues)
        raise ForcedDeductionError("nurikabe_completion: failed to build a solution")

    def _has_no_2x2_black(self, grid: List[List[int]]) -> bool:
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                if grid[r][c] == grid[r + 1][c] == grid[r][c + 1] == grid[r + 1][c + 1] == 1:
                    return False
        return True

    def _black_connected(self, grid: List[List[int]]) -> bool:
        black = {(r, c) for r in range(self.rows) for c in range(self.cols) if grid[r][c] == 1}
        if not black:
            return False
        seen = {next(iter(sorted(black)))}
        queue = deque(seen)
        while queue:
            r, c = queue.popleft()
            for cell in self._neighbors4(r, c):
                if cell in black and cell not in seen:
                    seen.add(cell)
                    queue.append(cell)
        return seen == black

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    def _forced_chain(
        self,
        solution: Sequence[Sequence[int]],
        clues: Sequence[Tuple[int, int, int]],
        given_black: Set[Cell],
        given_white: Set[Cell],
    ) -> Optional[ForcedChain]:
        rows, cols = self.rows, self.cols
        state: Dict[Cell, Optional[int]] = {
            (r, c): None for r in range(rows) for c in range(cols)
        }
        for cell in given_black:
            state[cell] = 1
        for cell in given_white:
            state[cell] = 0
        clue_at = {(r, c): size for r, c, size in clues}
        for cell in clue_at:
            state[cell] = 0
        rounds: List[List[ForcedMove]] = []

        def white_components() -> List[Tuple[Optional[Tuple[Cell, int]], Set[Cell]]]:
            seen: Set[Cell] = set()
            found = []
            for cell, value in sorted(state.items()):
                if value != 0 or cell in seen:
                    continue
                component = {cell}
                queue = deque([cell])
                seen.add(cell)
                while queue:
                    current = queue.popleft()
                    for nb in self._neighbors4(*current):
                        if state[nb] == 0 and nb not in seen:
                            seen.add(nb)
                            component.add(nb)
                            queue.append(nb)
                clue = None
                for member in sorted(component):
                    if member in clue_at:
                        clue = (member, clue_at[member])
                        break
                found.append((clue, component))
            return found

        def island_phrase(clue: Tuple[Cell, int]) -> str:
            (r, c), size = clue
            return f"the island numbered {size} at {display_cell_text((r, c))}"

        while any(value is None for value in state.values()):
            forced: Dict[Cell, Tuple[int, str]] = {}
            components = white_components()

            # Reach analysis: which blanks can any island still absorb?  The
            # walk may pass through blanks and through dotted island cells that
            # no clue owns yet (those may merge into the island), each costing
            # one unit of the island's remaining size budget, which keeps the
            # bound optimistic and the sea deduction sound.
            unclued_white: Set[Cell] = set()
            for clue, component in components:
                if clue is None:
                    unclued_white.update(component)
            reachable: Dict[Cell, int] = {}
            for clue, component in components:
                if clue is None:
                    continue
                budget = clue[1] - len(component)
                if budget <= 0:
                    continue
                distances = {cell: 0 for cell in component}
                queue = deque(sorted(component))
                while queue:
                    current = queue.popleft()
                    for nb in self._neighbors4(*current):
                        if nb in distances:
                            continue
                        if state[nb] is not None and nb not in unclued_white:
                            continue
                        distance = distances[current] + 1
                        if distance > budget:
                            continue
                        distances[nb] = distance
                        queue.append(nb)
                for cell, distance in distances.items():
                    if state[cell] is None:
                        reachable[cell] = reachable.get(cell, 0) + 1

            for clue, component in components:
                if clue is None:
                    continue
                growth = sorted(
                    {
                        nb
                        for cell in component
                        for nb in self._neighbors4(*cell)
                        if state[nb] is None
                    }
                )
                if len(component) == clue[1]:
                    for cell in growth:
                        forced.setdefault(
                            cell,
                            (
                                1,
                                f"{island_phrase(clue).capitalize()} is complete, so its "
                                f"neighbour at {display_cell_text(cell)} is sea",
                            ),
                        )
                elif len(growth) == 1:
                    cell = growth[0]
                    missing = clue[1] - len(component)
                    forced.setdefault(
                        cell,
                        (
                            0,
                            f"{island_phrase(clue).capitalize()} still needs {missing} more "
                            f"cell{'s' if missing != 1 else ''} and can only grow into "
                            f"{display_cell_text(cell)}, so that cell joins it",
                        ),
                    )

            for cell in sorted(state):
                if state[cell] is not None or cell in forced:
                    continue
                if cell not in reachable:
                    forced.setdefault(
                        cell,
                        (
                            1,
                            f"{display_cell_text(cell)} lies beyond the reach of "
                            "every numbered island, so it is sea",
                        ),
                    )

            # A blank next to two different clued islands separates them: sea.
            for cell in sorted(state):
                if state[cell] is not None or cell in forced:
                    continue
                touching = set()
                for clue, component in components:
                    if clue is None:
                        continue
                    if any(nb in component for nb in self._neighbors4(*cell)):
                        touching.add(clue)
                if len(touching) >= 2:
                    names = sorted(touching, key=lambda item: item[0])
                    forced.setdefault(
                        cell,
                        (
                            1,
                            f"{display_cell_text(cell)} touches "
                            f"{island_phrase(names[0])} and {island_phrase(names[1])} at once, "
                            "so it must be sea",
                        ),
                    )

            # Three sea cells in a 2x2 square force the fourth white.
            for r in range(rows - 1):
                for c in range(cols - 1):
                    square = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
                    values = [state[cell] for cell in square]
                    if values.count(1) == 3 and values.count(None) == 1:
                        cell = square[values.index(None)]
                        if cell not in forced:
                            forced[cell] = (
                                0,
                                f"Three cells of the square at rows {r + 1}-{r + 2}, columns "
                                f"{c + 1}-{c + 2} are already sea, so "
                                f"{display_cell_text(cell)} stays island ground",
                            )

            # A blank whose removal separates the committed sea must be sea.
            committed_black = sorted(cell for cell, value in state.items() if value == 1)
            if len(committed_black) >= 2:
                for cell in sorted(state):
                    if state[cell] is not None or cell in forced:
                        continue
                    allowed = {
                        other
                        for other, value in state.items()
                        if (value == 1 or value is None) and other != cell
                    }
                    start = committed_black[0]
                    seen = {start}
                    queue = deque([start])
                    while queue:
                        current = queue.popleft()
                        for nb in self._neighbors4(*current):
                            if nb in allowed and nb not in seen:
                                seen.add(nb)
                                queue.append(nb)
                    detached = [black for black in committed_black if black not in seen]
                    if detached:
                        other = detached[0]
                        forced[cell] = (
                            1,
                            f"Every route joining the sea at {display_cell_text(start)} to the sea at "
                            f"{display_cell_text(other)} runs through {display_cell_text(cell)}, "
                            "so that cell is sea",
                        )

            if not forced:
                return None
            round_moves: List[ForcedMove] = []
            for cell in sorted(forced):
                value, reason = forced[cell]
                if int(solution[cell[0]][cell[1]]) != value:
                    raise ForcedDeductionError(
                        "nurikabe_completion: forced cell deviates from the constructed board"
                    )
                state[cell] = value
                round_moves.append(
                    ForcedMove(
                        kind="sea" if value == 1 else "island",
                        cells=(cell,),
                        value=value,
                        reason=reason,
                    )
                )
            rounds.append(round_moves)
        return ForcedChain(rounds=rounds)

    def _build_instance(self):
        rows, cols = self.rows, self.cols
        for _ in range(200):
            try:
                grid, clues = self._build_solution()
            except ForcedDeductionError:
                continue
            clue_cells = {(r, c) for r, c, _ in clues}
            blacks = sorted(
                (r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1
            )
            whites = sorted(
                (r, c)
                for r in range(rows)
                for c in range(cols)
                if grid[r][c] == 0 and (r, c) not in clue_cells
            )
            if not blacks or not whites:
                continue
            given_black: Set[Cell] = {self.rng.choice(blacks)}
            given_white: Set[Cell] = {self.rng.choice(whites)}
            chain = self._forced_chain(grid, clues, given_black, given_white)
            stalled = 0
            while chain is None and stalled < rows * cols:
                stalled += 1
                open_blacks = [cell for cell in blacks if cell not in given_black]
                open_whites = [cell for cell in whites if cell not in given_white]
                if stalled % 2 == 1 and open_blacks:
                    given_black.add(self.rng.choice(open_blacks))
                elif open_whites:
                    given_white.add(self.rng.choice(open_whites))
                elif open_blacks:
                    given_black.add(self.rng.choice(open_blacks))
                else:
                    break
                chain = self._forced_chain(grid, clues, given_black, given_white)
            if chain is None or len(chain.rounds) < 2:
                continue
            open_cells = rows * cols - len(clue_cells) - len(given_black) - len(given_white)
            if open_cells < (rows * cols) // 2:
                continue
            try:
                reveal_budget_guard(
                    revealed=len(given_black) + len(given_white),
                    open_before_reveals=rows * cols - len(clue_cells),
                    max_fraction=0.4,
                    task_name="nurikabe_completion",
                )
                prove_complete_chain(chain, expected_cells=open_cells, task_name="nurikabe_completion")
            except ForcedDeductionError:
                continue
            return grid, clues, sorted(given_black), sorted(given_white), chain
        raise ForcedDeductionError(
            "nurikabe_completion: failed to construct a forced-hand-solvable board"
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        committed_black: Set[Cell],
        committed_white: Set[Cell],
        clues: Sequence[Tuple[int, int, int]],
    ) -> Image.Image:
        rows, cols = self.rows, self.cols
        cell, left, top = standard_board_geometry(rows, cols)
        scene, draw = new_board_scene()
        clue_font = board_font(max(20, cell * 3 // 5))
        axis_font = board_font(max(11, cell // 4))

        dot_radius = max(4, int(cell * 0.11))
        for r in range(rows):
            for c in range(cols):
                x0 = left + c * cell
                y0 = top + r * cell
                fill = _BLACK if (r, c) in committed_black else _WHITE_FILL
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fill, outline=GRID_LIGHT, width=1)
                if (r, c) in committed_white:
                    cx = x0 + cell // 2
                    cy = y0 + cell // 2
                    draw.ellipse(
                        [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
                        fill=_DOT,
                    )
        draw.rectangle([left, top, left + cols * cell, top + rows * cell], outline=GRID_BORDER, width=3)
        for r, c, size in clues:
            draw.text(
                (left + c * cell + cell // 2, top + r * cell + cell // 2),
                str(size),
                fill=TEXT,
                font=clue_font,
                anchor="mm",
            )
        draw_axis_labels(draw, rows, cols, cell=cell, left=left, top=top, font=axis_font)
        return finish_board_scene(scene)

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

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
                "Generate a solution video that starts from the unchanged puzzle. Cells are then "
                "decided in deduction order: each wave of newly determined cells fades in together, "
                "dark shades for sea and small gray dots for island ground, while earlier "
                f"commitments stay unchanged, and the video ends holding {_TERMINAL_VISUAL}."
            ),
            rule_description=_RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    # ------------------------------------------------------------------
    # Authored CoT
    # ------------------------------------------------------------------

    def _narrate(
        self,
        groups: Sequence[Sequence[ForcedMove]],
        clues: Sequence[Tuple[int, int, int]],
        given_black: Sequence[Cell],
        given_white: Sequence[Cell],
    ) -> List[str]:
        opener = (
            f"The board is {self.rows} by {self.cols} with {len(clues)} numbered islands; "
            f"{len(given_black)} sea cell{'s are' if len(given_black) != 1 else ' is'} already "
            f"shaded and {len(given_white)} island cell{'s' if len(given_white) != 1 else ''} "
            "already dotted. I will decide the remaining cells as the island sizes and the sea "
            "rules pin them down."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " Every cell is now decided: each numbered island has exactly its counted size, no "
            "two islands touch, the sea is one connected body, and no 2 by 2 square is all sea."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    # ------------------------------------------------------------------
    # Record assembly
    # ------------------------------------------------------------------

    def create_random_puzzle(self) -> NurikabeCompletionRecord:
        grid, clues, given_black, given_white, chain = self._build_instance()
        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        black: Set[Cell] = set(given_black)
        white: Set[Cell] = set(given_white)
        states: List[Image.Image] = [fit_canvas(self._render(set(black), set(white), clues))]
        for group in groups:
            for move in group:
                if move.value == 1:
                    black.update(move.cells)
                else:
                    white.update(move.cells)
            states.append(fit_canvas(self._render(set(black), set(white), clues)))
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

        answer_grid = [[int(v) for v in row] for row in grid]

        return NurikabeCompletionRecord(
            answer=answer_json(answer_grid),
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
                "clues": [list(c) for c in clues],
                "given_black": [list(cell) for cell in given_black],
                "given_white": [list(cell) for cell in given_white],
                "solution": answer_grid,
            },
        )
