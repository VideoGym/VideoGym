"""N-Queens completion: extend fixed queens to a full non-attacking placement.

Every emitted puzzle is forced-hand solvable: unit propagation over the
visible attack rules (cells attacked by placed queens are dead; a row or
column whose last free square remains forces a queen there) completes the
board from the fixed queens with zero guessing.  The generator proves the
chain per sample by replaying its own engine after monotone trimming of the
fixed set; the chain drives the step-by-step deduction order and the TI2V commit order.
"""

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
    board_font,
    draw_axis_labels,
    finish_board_scene,
    new_board_scene,
    standard_board_geometry,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.contracts.grid_completion import answer_xy_positions
from data.common.pipeline.grid_rendering import fit_canvas
from data.common.prompts import build_sectioned_prompt

Cell = Tuple[int, int]

_LIGHT = (240, 217, 181)
_DARK = (181, 136, 99)
_QUEEN = (32, 32, 32)
_GIVEN = (40, 89, 168)

_INPUT_DESCRIPTION = (
    "The image shows a chessboard with alternating light and dark squares, small gray 1-based row and "
    "column numbers along the top and left edges, and several fixed queens drawn as navy Q markers."
)
_RULE_DESCRIPTION = (
    "A queen attacks every square in its own row, its own column, and both of its diagonals. Add one "
    "queen to every row that lacks one so that the finished board holds exactly one queen per row and "
    "per column and no two queens attack each other. The navy queens are fixed and never move."
)
_TERMINAL_VISUAL = (
    "the completed board on which every added queen appears as a dark Q marker while the navy fixed "
    "queens, the squares, and the edge numbers stay unchanged"
)


@dataclass
class NQueensCompletionRecord:
    id: str
    task_name: str = "nqueens_completion"
    task_family: str = "constraint_grid"
    image: str = ""
    reasoning_image: str = ""
    solution_image_path: str = ""
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None
    ti2v_prompt: str = ""
    solution_image_prompt: str = ""
    answer: str = ""
    question: str = ""
    question_description: str = "nqueens_completion"
    structured_answer: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "image": self.image,
            "reasoning_image": self.reasoning_image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "ti2v_prompt": self.ti2v_prompt,
            "answer": self.answer,
            "question": self.question,
            "question_description": self.question_description,
            "structured_answer": self.structured_answer,
        }


def _solve_full(n: int, rng: random.Random) -> List[int]:
    cols: set[int] = set()
    diag1: set[int] = set()
    diag2: set[int] = set()
    solution: List[int] = [-1] * n

    def backtrack(row: int) -> bool:
        if row == n:
            return True
        candidates = list(range(n))
        rng.shuffle(candidates)
        for col in candidates:
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            solution[row] = col
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            if backtrack(row + 1):
                return True
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
            solution[row] = -1
        return False

    if not backtrack(0):
        raise ForcedDeductionError("nqueens_completion: failed to construct a solution")
    return solution


class NQueensCompletionGenerator(AbstractPuzzleGenerator[NQueensCompletionRecord]):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        video_fps: int = 16,
        board_size: int = 8,
    ) -> None:
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.seed = seed
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self.board_size = max(4, int(board_size))
        self._counter = 0

    def _next_id(self) -> str:
        puzzle_id = f"{NQueensCompletionRecord.task_name}-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    def _forced_chain(
        self,
        solution: Sequence[int],
        fixed_rows: Set[int],
    ) -> Optional[ForcedChain]:
        n = self.board_size
        placed: Dict[int, int] = {row: solution[row] for row in fixed_rows}
        rounds: List[List[ForcedMove]] = []

        def attacked(row: int, col: int) -> bool:
            return any(
                col == pcol or row == prow or abs(row - prow) == abs(col - pcol)
                for prow, pcol in placed.items()
            )

        while len(placed) < n:
            open_rows = [row for row in range(n) if row not in placed]
            free = {
                row: [col for col in range(n) if not attacked(row, col)]
                for row in open_rows
            }
            forced: List[Tuple[int, int, str]] = []
            claimed_rows: Set[int] = set()
            for row in open_rows:
                if len(free[row]) == 1:
                    col = free[row][0]
                    forced.append(
                        (
                            row,
                            col,
                            f"Row {row + 1} has a single unattacked square left at "
                            f"{display_cell_text((row, col))}, so its queen goes there",
                        )
                    )
                    claimed_rows.add(row)
            open_cols = {col for col in range(n) if col not in placed.values()}
            for col in sorted(open_cols):
                homes = [row for row in open_rows if col in free[row]]
                if len(homes) == 1 and homes[0] not in claimed_rows:
                    row = homes[0]
                    forced.append(
                        (
                            row,
                            col,
                            f"Column {col + 1} can only be covered at "
                            f"{display_cell_text((row, col))}, so that queen is pinned to it",
                        )
                    )
                    claimed_rows.add(row)
            if not forced:
                return None
            round_moves: List[ForcedMove] = []
            for row, col, reason in forced:
                if solution[row] != col:
                    raise ForcedDeductionError(
                        "nqueens_completion: forced square deviates from the constructed solution"
                    )
                placed[row] = col
                round_moves.append(
                    ForcedMove(kind="queen", cells=((row, col),), value="Q", reason=reason)
                )
            rounds.append(round_moves)
        return ForcedChain(rounds=rounds)

    def _construct_instance(self) -> Tuple[List[int], Dict[int, int], ForcedChain]:
        n = self.board_size
        for _ in range(80):
            solution = _solve_full(n, self.rng)
            fixed_rows: Set[int] = set(range(n))
            order = list(range(n))
            self.rng.shuffle(order)
            for row in order:
                fixed_rows.discard(row)
                if self._forced_chain(solution, fixed_rows) is None:
                    fixed_rows.add(row)
            chain = self._forced_chain(solution, fixed_rows)
            open_count = n - len(fixed_rows)
            if chain is None or open_count < 4 or len(chain.rounds) < 2:
                continue
            if len(fixed_rows) > n // 2 + 1:
                continue
            prove_complete_chain(chain, expected_cells=open_count, task_name="nqueens_completion")
            fixed = {row: solution[row] for row in sorted(fixed_rows)}
            return solution, fixed, chain
        raise ForcedDeductionError(
            "nqueens_completion: failed to construct a forced-hand-solvable board"
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_board(
        self,
        board: Sequence[Sequence[str]],
        *,
        fixed_positions: Sequence[Sequence[int]] | set[tuple[int, int]],
    ) -> Image.Image:
        n = len(board)
        fixed_cells = {(int(row), int(col)) for row, col in fixed_positions}
        cell, left, top = standard_board_geometry(n, n)
        scene, draw = new_board_scene()
        from data.common.prompt_paraphrase import active_profile, draw_crown
        crowns = active_profile("crown_queens")
        queen_font = board_font(max(18, cell * 3 // 5))
        axis_font = board_font(max(11, cell // 4))

        for row in range(n):
            for col in range(n):
                x0 = left + col * cell
                y0 = top + row * cell
                fill = ((239,245,224) if (row+col)%2==0 else (154,188,174)) if crowns else (_LIGHT if (row+col)%2==0 else _DARK)
                if crowns:
                    draw.rectangle([x0,y0,x0+cell,y0+cell],fill=(247,232,190))
                    cx,cy=x0+cell/2,y0+cell/2;rad=cell*.43
                    draw.polygon([(cx,cy-rad),(cx+rad,cy),(cx,cy+rad),(cx-rad,cy)],fill=fill,outline=(100,119,102),width=2)
                else:
                    draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fill)
                cell_value = board[row][col]
                if cell_value == "Q" and crowns:
                    pad=cell*.22
                    draw_crown(draw,(x0+pad,y0+pad,x0+cell-pad,y0+cell-pad),_GIVEN if (row,col) in fixed_cells else _QUEEN)
                elif cell_value == "Q":
                    draw.text(
                        (x0 + cell / 2, y0 + cell / 2),
                        "Q",
                        fill=_GIVEN if (row, col) in fixed_cells else _QUEEN,
                        font=queen_font,
                        anchor="mm",
                    )
        draw_axis_labels(draw, n, n, cell=cell, left=left, top=top, font=axis_font)
        return finish_board_scene(scene)

    def _board_for(self, placed: Dict[int, int]) -> List[List[str]]:
        n = self.board_size
        return [
            ["Q" if placed.get(row) == col else "_" for col in range(n)]
            for row in range(n)
        ]

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
                "Generate a solution video that starts from the unchanged puzzle. Dark Q markers then "
                "appear in deduction order: each wave of squares the attack lines force fades in "
                "together, earlier queens stay exactly in place, and the video ends holding "
                f"{_TERMINAL_VISUAL}."
            ),
            rule_description=_RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    # ------------------------------------------------------------------
    # Authored CoT
    # ------------------------------------------------------------------

    def _narrate(self, groups: Sequence[Sequence[ForcedMove]], fixed: Dict[int, int]) -> List[str]:
        n = self.board_size
        fixed_phrase = "; ".join(
            display_cell_text((row, col)) for row, col in sorted(fixed.items())
        )
        opener = (
            f"The board is {n} by {n} and the navy queens sit at {fixed_phrase}. Every row and column "
            "still missing a queen must gain one that no existing queen attacks, so I will place "
            "queens as the attack lines force them."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " All rows and columns now hold exactly one queen each and no two queens share a row, "
            "column, or diagonal, so the placement is complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    # ------------------------------------------------------------------
    # Record assembly
    # ------------------------------------------------------------------

    def create_puzzle(self) -> NQueensCompletionRecord:
        sample_id = self._next_id()
        solution_rows, fixed, chain = self._construct_instance()
        prompts = self._build_prompts()

        n = len(solution_rows)
        solution_positions = [[row, col] for row, col in enumerate(solution_rows)]
        fixed_positions = [[row, solution_rows[row]] for row in sorted(fixed)]
        fixed_cells = {(int(row), int(col)) for row, col in fixed_positions}

        puzzle_dir = self.output_dir / "puzzles"
        reasoning_dir = self.output_dir / "reasoning"
        solution_dir = self.output_dir / "solutions"
        for directory in (puzzle_dir, reasoning_dir, solution_dir):
            directory.mkdir(parents=True, exist_ok=True)

        groups = merge_rounds(chain.rounds)
        placed = dict(fixed)
        states: List[Image.Image] = [
            fit_canvas(self._render_board(self._board_for(placed), fixed_positions=fixed_cells))
        ]
        for group in groups:
            for move in group:
                (row, col) = move.cells[0]
                placed[row] = col
            states.append(
                fit_canvas(
                    self._render_board(self._board_for(placed), fixed_positions=fixed_cells)
                )
            )
        puzzle_image, solution_image = states[0], states[-1]

        puzzle_path = puzzle_dir / f"{sample_id}.png"
        reasoning_path = reasoning_dir / f"{sample_id}_reasoning.png"
        solution_path = solution_dir / f"{sample_id}.png"
        puzzle_image.save(puzzle_path)
        puzzle_image.save(reasoning_path)
        solution_image.save(solution_path)

        nonterminal_paths: List[str] = []
        for state_index, state in enumerate(states[1:-1], start=1):
            state_path = reasoning_dir / f"{sample_id}_state_{state_index:02d}.png"
            state.save(state_path)
            nonterminal_paths.append(self.relativize_path(state_path))

        video_num_frames = 0
        solution_video_path: Optional[str] = None
        if self.video:
            video_path = solution_dir / f"{sample_id}.webm"
            frames = encode_commit_video(states, video_path, fps=self.video_fps)
            if frames > 0 and video_path.exists():
                video_num_frames = frames
                solution_video_path = self.relativize_path(video_path)

        visible_positions = [[row + 1, col + 1] for row, col in solution_positions]
        answer_text = answer_xy_positions(visible_positions)

        record = NQueensCompletionRecord(
            id=sample_id,
            image=self.relativize_path(puzzle_path),
            reasoning_image=self.relativize_path(reasoning_path),
            solution_image_path=self.relativize_path(solution_path),
            solution_video_path=solution_video_path,
            video_fps=self.video_fps if solution_video_path else None,
            video_num_frames=video_num_frames if solution_video_path else None,
            ti2v_prompt=prompts["ti2v"],
            solution_image_prompt=prompts["solution_image"],
            answer=answer_text,
            question=(
                "Rule: A queen attacks along its own row, its own column, and both diagonals; no two "
                "queens may share any of those lines. Place the remaining queens so no two queens "
                "attack each other."
            ),
            structured_answer={
                "board_size": n,
                "fixed_positions": fixed_positions,
                "solution_positions": solution_positions,
                "solution_board": self._board_for({row: col for row, col in enumerate(solution_rows)}),
            },
        )
        return record
