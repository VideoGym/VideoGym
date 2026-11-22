"""Fillomino completion: restore the erased digits of a finished polyomino tiling. Every emitted puzzle is forced-hand solvable. The generator first builds a
complete tiling, then erases digits one at a time and keeps an erasure only if
its growth engine still restores the whole board: a visible region whose digit
demands more cells than it currently shows, and which has exactly one open
neighbouring blank, forces that blank to take its digit. The proven chain
drives the authored CoT and the TI2V commit order."""

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
from data.constraint_grid.core.coordinate_format import display_cell_text
from data.constraint_grid.core.scene import (
    TEXT,
    board_font,
    draw_axis_labels,
    draw_board_grid,
    finish_board_scene,
    new_board_scene,
    standard_board_geometry,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.contracts.grid_completion import answer_json
from data.common.pipeline.grid_rendering import fit_canvas
from data.common.prompts import build_sectioned_prompt

Cell = Tuple[int, int]

_GIVEN = (40, 89, 168)

_INPUT_DESCRIPTION = (
    "The image shows a square grid with 1-based row and column numbers in small gray along the top and "
    "left edges. Navy digits are the given clues; the remaining cells are blank."
)
_RULE_DESCRIPTION = (
    "The finished grid is divided into groups of orthogonally connected cells, every cell of a group "
    "holds the group's size as its digit, and two different groups of the same size never share a cell "
    "edge. Fill every blank cell with the digit of the group it belongs to; the navy digits are fixed."
)
_TERMINAL_VISUAL = (
    "the finished grid in which every formerly blank cell shows its digit in dark ink, the given navy "
    "digits stay exactly as they are, and the grid lines and edge numbers stay unchanged"
)
@dataclass
class FillominoCompletionRecord:
    id: str
    task_name: str = "fillomino_completion"
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


class FillominoCompletionGenerator(AbstractPuzzleGenerator[FillominoCompletionRecord]):
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

    def create_puzzle(self, *args, **kwargs) -> FillominoCompletionRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"fillomino_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    # ------------------------------------------------------------------
    # Solution construction
    # ------------------------------------------------------------------

    def _build_solution(self) -> List[List[int]]:
        rows, cols = self.rows, self.cols
        for _ in range(800):
            region_id = [[-1 for _ in range(cols)] for _ in range(rows)]
            cell_size = [[0 for _ in range(cols)] for _ in range(rows)]
            next_id = 0
            success = True
            for r0 in range(rows):
                for c0 in range(cols):
                    if region_id[r0][c0] != -1:
                        continue
                    forbidden = set()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r0 + dr, c0 + dc
                        if 0 <= nr < rows and 0 <= nc < cols and region_id[nr][nc] >= 0:
                            forbidden.add(cell_size[nr][nc])
                    candidates = [s for s in (1, 2, 3, 4, 5) if s not in forbidden]
                    if not candidates:
                        success = False
                        break
                    target_size = self.rng.choice(candidates)
                    grown = self._grow_polyomino(r0, c0, target_size, region_id, cell_size, next_id)
                    if grown is None:
                        if 1 not in forbidden:
                            region_id[r0][c0] = next_id
                            cell_size[r0][c0] = 1
                            next_id += 1
                            continue
                        success = False
                        break
                    next_id += 1
                if not success:
                    break
            if not success:
                continue
            if all(region_id[r][c] >= 0 for r in range(rows) for c in range(cols)):
                if self._check_no_adjacent_equal_size(cell_size, region_id):
                    return cell_size
        raise ForcedDeductionError("fillomino_completion: failed to construct a tiling")

    def _grow_polyomino(
        self,
        r0: int,
        c0: int,
        target_size: int,
        region_id: List[List[int]],
        cell_size: List[List[int]],
        new_rid: int,
    ) -> Optional[List[Cell]]:
        rows, cols = self.rows, self.cols
        region_id[r0][c0] = new_rid
        cell_size[r0][c0] = target_size
        cells = [(r0, c0)]
        for _ in range(target_size - 1):
            boundary: List[Cell] = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and region_id[nr][nc] == -1:
                        boundary.append((nr, nc))
            valid: List[Cell] = []
            for nr, nc in boundary:
                bad = False
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ar, ac = nr + dr, nc + dc
                    if 0 <= ar < rows and 0 <= ac < cols and region_id[ar][ac] >= 0\
                            and region_id[ar][ac] != new_rid\
                            and cell_size[ar][ac] == target_size:
                        bad = True
                        break
                if not bad:
                    valid.append((nr, nc))
            if not valid:
                for r, c in cells:
                    region_id[r][c] = -1
                    cell_size[r][c] = 0
                return None
            pick = self.rng.choice(sorted(set(valid)))
            region_id[pick[0]][pick[1]] = new_rid
            cell_size[pick[0]][pick[1]] = target_size
            cells.append(pick)
        return cells

    def _check_no_adjacent_equal_size(
        self, cell_size: List[List[int]], region_id: List[List[int]]
    ) -> bool:
        for r in range(self.rows):
            for c in range(self.cols):
                for dr, dc in ((1, 0), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if nr < self.rows and nc < self.cols:
                        if region_id[r][c] != region_id[nr][nc]\
                                and cell_size[r][c] == cell_size[nr][nc]:
                            return False
        return True

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    def _forced_chain(
        self,
        puzzle: Sequence[Sequence[int]],
        solution: Sequence[Sequence[int]],
    ) -> Optional[ForcedChain]:
        """Replay region-growth deduction from the visible clues; None on stall."""

        rows, cols = self.rows, self.cols
        committed = [[int(puzzle[r][c]) for c in range(cols)] for r in range(rows)]
        rounds: List[List[ForcedMove]] = []

        def components() -> List[Tuple[int, List[Cell]]]:
            seen = [[False] * cols for _ in range(rows)]
            found: List[Tuple[int, List[Cell]]] = []
            for r in range(rows):
                for c in range(cols):
                    if committed[r][c] == 0 or seen[r][c]:
                        continue
                    value = committed[r][c]
                    stack = [(r, c)]
                    seen[r][c] = True
                    cells: List[Cell] = []
                    while stack:
                        cr, cc = stack.pop()
                        cells.append((cr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr][nc]\
                                    and committed[nr][nc] == value:
                                seen[nr][nc] = True
                                stack.append((nr, nc))
                    found.append((value, sorted(cells)))
            return found

        while any(committed[r][c] == 0 for r in range(rows) for c in range(cols)):
            forced: Dict[Cell, Tuple[int, str]] = {}
            for value, cells in components():
                if len(cells) > value:
                    raise ForcedDeductionError(
                        "fillomino_completion: committed group exceeds its digit"
                    )
                if len(cells) == value:
                    continue
                open_neighbours: Set[Cell] = set()
                for r, c in cells:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and committed[nr][nc] == 0:
                            open_neighbours.add((nr, nc))
                if len(open_neighbours) != 1:
                    continue
                target = next(iter(open_neighbours))
                anchor = cells[0]
                missing = value - len(cells)
                reason = (
                    f"The {value}-group containing {display_cell_text(anchor)} still "
                    f"needs {missing} more cell{'s' if missing != 1 else ''} and its only open "
                    f"neighbour is {display_cell_text(target)}, so that cell takes "
                    f"a {value}"
                )
                if target in forced and forced[target][0] != value:
                    raise ForcedDeductionError(
                        "fillomino_completion: conflicting forced digits for one blank"
                    )
                forced.setdefault(target, (value, reason))
            if not forced:
                return None
            round_moves: List[ForcedMove] = []
            for cell in sorted(forced):
                value, reason = forced[cell]
                if int(solution[cell[0]][cell[1]]) != value:
                    raise ForcedDeductionError(
                        "fillomino_completion: forced digit deviates from the constructed tiling"
                    )
                committed[cell[0]][cell[1]] = value
                round_moves.append(
                    ForcedMove(kind="digit", cells=(cell,), value=value, reason=reason)
                )
            rounds.append(round_moves)
        return ForcedChain(rounds=rounds)

    def _construct_instance(self) -> Tuple[List[List[int]], List[List[int]], List[Cell], ForcedChain]:
        for _ in range(60):
            solution = self._build_solution()
            puzzle = [[int(v) for v in row] for row in solution]
            candidates = [(r, c) for r in range(self.rows) for c in range(self.cols)]
            self.rng.shuffle(candidates)
            removed: List[Cell] = []
            for r, c in candidates:
                digit = puzzle[r][c]
                puzzle[r][c] = 0
                if self._forced_chain(puzzle, solution) is None:
                    puzzle[r][c] = digit
                else:
                    removed.append((r, c))
            if len(removed) < 8:
                continue
            chain = self._forced_chain(puzzle, solution)
            if chain is None or len(chain.rounds) < 2:
                continue
            prove_complete_chain(chain, expected_cells=len(removed), task_name="fillomino_completion")
            return solution, puzzle, sorted(removed), chain
        raise ForcedDeductionError(
            "fillomino_completion: failed to construct a forced-hand-solvable puzzle"
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        grid: Sequence[Sequence[int]],
        puzzle: Sequence[Sequence[int]],
    ) -> Image.Image:
        """Render the board with all non-zero digits of ``grid`` committed."""

        cell, left, top = standard_board_geometry(self.rows, self.cols)
        scene, draw = new_board_scene()
        digit_font = board_font(max(20, cell * 3 // 5))
        axis_font = board_font(max(11, cell // 4))

        draw_board_grid(draw, self.rows, self.cols, cell=cell, left=left, top=top)
        draw_axis_labels(
            draw, self.rows, self.cols, cell=cell, left=left, top=top, font=axis_font
        )

        for r in range(self.rows):
            for c in range(self.cols):
                value = int(grid[r][c])
                if value == 0:
                    continue
                cx = left + c * cell + cell // 2
                cy = top + r * cell + cell // 2
                color = _GIVEN if int(puzzle[r][c]) != 0 else TEXT
                draw.text((cx, cy), str(value), fill=color, font=digit_font, anchor="mm")
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
                "Generate a solution video that starts from the unchanged puzzle. Digits then appear in "
                "deduction order: each wave of blanks forced by the incomplete groups fades in together "
                "as dark digits, earlier digits stay exactly in place, and the video ends holding "
                f"{_TERMINAL_VISUAL}."
            ),
            rule_description=_RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    # ------------------------------------------------------------------
    # Authored CoT
    # ------------------------------------------------------------------

    def _narrate(self, groups: Sequence[Sequence[ForcedMove]], blanks: int) -> List[str]:
        opener = (
            f"The grid is {self.rows} by {self.cols} with {blanks} blank cells. Each navy clue names "
            "the size of its group, so any group still smaller than its digit has to grow; I will fill "
            "the blank that each incomplete group is forced into."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " Every blank now holds a digit, each connected group's size equals its digit, and no two "
            "equal-sized groups touch, so the grid is complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    # ------------------------------------------------------------------
    # Record assembly
    # ------------------------------------------------------------------

    def create_random_puzzle(self) -> FillominoCompletionRecord:
        solution, puzzle, removed, chain = self._construct_instance()
        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        committed = [[int(v) for v in row] for row in puzzle]
        states: List[Image.Image] = [fit_canvas(self._render(committed, puzzle))]
        for group in groups:
            for move in group:
                (r, c) = move.cells[0]
                committed[r][c] = int(move.value)
            states.append(
                fit_canvas(self._render([[int(v) for v in row] for row in committed], puzzle))
            )
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

        answer_grid = [[int(v) for v in row] for row in solution]

        return FillominoCompletionRecord(
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
                "puzzle": [[int(v) for v in row] for row in puzzle],
                "solution": answer_grid,
                "removed_cells": [list(cell) for cell in removed],
            },
        )
