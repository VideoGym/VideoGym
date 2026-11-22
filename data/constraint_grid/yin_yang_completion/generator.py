"""Yin-yang completion: place black and white stones by forced deduction. Every emitted puzzle is forced-hand solvable: propagation over the two
visible rules (three same-colored stones in a 2x2 square force the fourth
cell to the other color; a blank cell that alone keeps one color's stones
connectable must take that color) completes the board from the given stones
with zero guessing. Both colors of the finished board form single
edge-connected groups and no 2x2 square is monochrome. The generator samples
a fresh legal board per sample (a comb-shaped base randomized by
legality-preserving flips), trims the given stones monotonically, and proves
the chain; the chain drives the authored CoT, the TI2V key states, and the
TI2V commit order."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from data.constraint_grid.core.coordinate_format import display_cell_text
from data.constraint_grid.core.scene import (
    GRID_BORDER,
    GRID_LIGHT,
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

_BG_CELL = (247, 244, 236)
_BLACK = (32, 36, 44)
_WHITE_FILL = (255, 255, 255)
_STONE_EDGE = (70, 76, 88)
_GIVEN_OUT = (220, 90, 70)

_INPUT_DESCRIPTION = (
    "The image shows a square grid of parchment cells with small gray 1-based row and column numbers "
    "along the top and left edges. Stones already placed are black or white discs, each wearing a thin "
    "coral ring to mark it as given; the remaining cells are empty."
)
_RULE_DESCRIPTION = (
    "Fill every cell with a black or a white stone so that all black stones form one edge-connected "
    "group, all white stones form one edge-connected group, and no 2 by 2 square holds four stones of "
    "one color. The ringed stones are fixed."
)
_TERMINAL_VISUAL = (
    "the finished board on which every cell holds its stone, the ringed given stones stay exactly as "
    "they are, and the grid and edge numbers stay unchanged"
)
@dataclass
class YinYangCompletionRecord:
    id: str
    task_name: str = "yin_yang_completion"
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


class YinYangCompletionGenerator(AbstractPuzzleGenerator[YinYangCompletionRecord]):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        size: int = 6,
        video: bool = False,
        video_fps: int = 16,
    ) -> None:
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.seed = seed
        self.size = size
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self._counter = 0
        self.puzzle_dir = self.output_dir / "puzzles"
        self.reasoning_dir = self.output_dir / "reasoning"
        self.solution_dir = self.output_dir / "solutions"
        for directory in (self.puzzle_dir, self.reasoning_dir, self.solution_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def create_puzzle(self, *args, **kwargs) -> YinYangCompletionRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"yin_yang_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    def _neighbors4(self, r: int, c: int) -> List[Cell]:
        return [
            (r + dr, c + dc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= r + dr < self.size and 0 <= c + dc < self.size
        ]

    # ------------------------------------------------------------------
    # Legal-board sampling: comb base plus legality-preserving flips
    # ------------------------------------------------------------------

    def _legal(self, grid: List[List[int]]) -> bool:
        n = self.size
        for r in range(n - 1):
            for c in range(n - 1):
                if grid[r][c] == grid[r + 1][c] == grid[r][c + 1] == grid[r + 1][c + 1]:
                    return False
        for value in (0, 1):
            cells = {(r, c) for r in range(n) for c in range(n) if grid[r][c] == value}
            if not cells:
                return False
            seen = {next(iter(sorted(cells)))}
            queue = deque(seen)
            while queue:
                current = queue.popleft()
                for nb in self._neighbors4(*current):
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            if seen != cells:
                return False
        return True

    def _build_solution(self) -> List[List[int]]:
        n = self.size
        # Comb base: a black spine down column 1 with teeth along the even
        # rows, interlocking a white spine down the last column with teeth
        # along the odd rows.  Legal by construction for every n >= 4.
        grid = [[0 for _ in range(n)] for _ in range(n)]
        for r in range(n):
            grid[r][0] = 1
            if r % 2 == 0:
                for c in range(n - 1):
                    grid[r][c] = 1
        if not self._legal(grid):
            raise ForcedDeductionError("yin_yang_completion: comb base is not legal")
        # Randomize with legality-preserving single-cell flips.
        cells = [(r, c) for r in range(n) for c in range(n)]
        accepted = 0
        for _ in range(n * n * 14):
            r, c = self.rng.choice(cells)
            grid[r][c] = 1 - grid[r][c]
            total = sum(sum(row) for row in grid)
            if 3 <= total <= n * n - 3 and self._legal(grid):
                accepted += 1
            else:
                grid[r][c] = 1 - grid[r][c]
        if accepted < n:
            raise ForcedDeductionError("yin_yang_completion: board randomization stalled")
        return grid

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    def _forced_chain(
        self,
        solution: Sequence[Sequence[int]],
        given: Dict[Cell, int],
    ) -> Optional[ForcedChain]:
        n = self.size
        state: Dict[Cell, Optional[int]] = {
            (r, c): None for r in range(n) for c in range(n)
        }
        for cell, value in given.items():
            state[cell] = value
        rounds: List[List[ForcedMove]] = []
        color_name = {1: "black", 0: "white"}

        while any(value is None for value in state.values()):
            forced: Dict[Cell, Tuple[int, str]] = {}
            for r in range(n - 1):
                for c in range(n - 1):
                    square = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
                    values = [state[cell] for cell in square]
                    for value in (0, 1):
                        if values.count(value) == 3 and values.count(None) == 1:
                            cell = square[values.index(None)]
                            if cell not in forced:
                                forced[cell] = (
                                    1 - value,
                                    f"The square at rows {r + 1}-{r + 2}, columns {c + 1}-"
                                    f"{c + 2} already shows three {color_name[value]} stones, "
                                    f"so {display_cell_text(cell)} must be "
                                    f"{color_name[1 - value]}",
                                )
            for value in (0, 1):
                committed = sorted(cell for cell, v in state.items() if v == value)
                if len(committed) < 2:
                    continue
                for cell in sorted(state):
                    if state[cell] is not None or cell in forced:
                        continue
                    allowed = {
                        other
                        for other, v in state.items()
                        if (v == value or v is None) and other != cell
                    }
                    start = committed[0]
                    seen = {start}
                    queue = deque([start])
                    while queue:
                        current = queue.popleft()
                        for nb in self._neighbors4(*current):
                            if nb in allowed and nb not in seen:
                                seen.add(nb)
                                queue.append(nb)
                    detached = [other for other in committed if other not in seen]
                    if detached:
                        other = detached[0]
                        forced[cell] = (
                            value,
                            f"Every route joining the {color_name[value]} stones at "
                            f"{display_cell_text(start)} and {display_cell_text(other)} runs through "
                            f"{display_cell_text(cell)}, so that cell must be {color_name[value]}",
                        )
            if not forced:
                return None
            round_moves: List[ForcedMove] = []
            for cell in sorted(forced):
                value, reason = forced[cell]
                if int(solution[cell[0]][cell[1]]) != value:
                    raise ForcedDeductionError(
                        "yin_yang_completion: forced stone deviates from the constructed board"
                    )
                state[cell] = value
                round_moves.append(
                    ForcedMove(kind="stone", cells=(cell,), value=value, reason=reason)
                )
            rounds.append(round_moves)
        return ForcedChain(rounds=rounds)

    def _build_instance(self):
        n = self.size
        for _ in range(200):
            grid = self._build_solution()
            cells = [(r, c) for r in range(n) for c in range(n)]
            given: Dict[Cell, int] = {cell: grid[cell[0]][cell[1]] for cell in cells}
            order = list(cells)
            self.rng.shuffle(order)
            for cell in order:
                value = given.pop(cell)
                if self._forced_chain(grid, given) is None:
                    given[cell] = value
            chain = self._forced_chain(grid, given)
            open_count = n * n - len(given)
            if chain is None or open_count < (n * n) // 2 or len(chain.rounds) < 3:
                continue
            if not any(v == 1 for v in given.values()) or not any(v == 0 for v in given.values()):
                continue
            prove_complete_chain(chain, expected_cells=open_count, task_name="yin_yang_completion")
            return grid, given, chain
        raise ForcedDeductionError(
            "yin_yang_completion: failed to construct a forced-hand-solvable board"
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        committed: Dict[Cell, int],
        given: Dict[Cell, int],
    ) -> Image.Image:
        n = self.size
        cell, left, top = standard_board_geometry(n, n)
        scene, draw = new_board_scene()
        axis_font = board_font(max(11, cell // 4))

        for r in range(n):
            for c in range(n):
                x0 = left + c * cell
                y0 = top + r * cell
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=_BG_CELL, outline=GRID_LIGHT, width=1)
        draw.rectangle([left, top, left + n * cell, top + n * cell], outline=GRID_BORDER, width=3)
        draw_axis_labels(draw, n, n, cell=cell, left=left, top=top, font=axis_font)

        radius = max(8, int(cell * 0.34))
        for (r, c), value in sorted(committed.items()):
            cx = left + c * cell + cell // 2
            cy = top + r * cell + cell // 2
            fill = _BLACK if value == 1 else _WHITE_FILL
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=fill,
                outline=_STONE_EDGE,
                width=2,
            )
            if (r, c) in given:
                ring = radius + 4
                draw.ellipse(
                    [cx - ring, cy - ring, cx + ring, cy + ring],
                    outline=_GIVEN_OUT,
                    width=3,
                )
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
                "Generate a solution video that starts from the unchanged puzzle. Stones then appear "
                "in deduction order: each wave of cells the rules force fades in together as black "
                "and white discs, earlier stones stay exactly in place, and the video ends holding "
                f"{_TERMINAL_VISUAL}."
            ),
            rule_description=_RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    # ------------------------------------------------------------------
    # Authored CoT
    # ------------------------------------------------------------------

    def _narrate(self, groups: Sequence[Sequence[ForcedMove]], given: Dict[Cell, int]) -> List[str]:
        blacks = sum(1 for v in given.values() if v == 1)
        whites = len(given) - blacks
        opener = (
            f"The board is {self.size} by {self.size} with {blacks} ringed black and {whites} ringed "
            "white stones given. Both colors must stay connected and no 2 by 2 square may be one "
            "color, so I will place stones where those two facts leave a single choice."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " Every cell now holds a stone, both color groups are connected, and no 2 by 2 square is "
            "monochrome, so the board is complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    # ------------------------------------------------------------------
    # Record assembly
    # ------------------------------------------------------------------

    def create_random_puzzle(self) -> YinYangCompletionRecord:
        grid, given, chain = self._build_instance()
        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        committed: Dict[Cell, int] = dict(given)
        states: List[Image.Image] = [fit_canvas(self._render(dict(committed), given))]
        for group in groups:
            for move in group:
                committed[move.cells[0]] = int(move.value)
            states.append(fit_canvas(self._render(dict(committed), given)))
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

        return YinYangCompletionRecord(
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
                "size": self.size,
                "given": [[r, c, v] for (r, c), v in sorted(given.items())],
                "solution": answer_grid,
            },
        )
