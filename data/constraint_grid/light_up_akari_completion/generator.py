"""Light Up (Akari) completion: extend given bulbs until every white cell is lit. Every emitted puzzle is forced-hand solvable: propagation over the visible
rules (a satisfied numbered wall closes its remaining neighbours; a numbered
wall whose open neighbours all fit its remaining count forces bulbs; a cell
lit by a bulb can hold no bulb; an unlit cell whose entire sight line offers
one last open cell forces a bulb there) completes the board from the given
bulbs with zero guessing. The generator proves the chain per sample after
monotone trimming of the given set; the chain drives the authored CoT, the
TI2V key states, and the TI2V commit order."""

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
    GRID_BORDER,
    GRID_LIGHT,
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

_BLACK = (32, 36, 44)
_LIGHT = (250, 200, 70)
_LIT = (255, 247, 224)
_WHITE_FILL = (255, 255, 255)

_INPUT_DESCRIPTION = (
    "The image shows a grid of white cells and dark wall cells, with small gray 1-based row and column "
    "numbers along the top and left edges. Some walls carry a white digit from 0 to 4. Golden bulbs "
    "that are already placed light their rows and columns, and every cell they light is tinted warm."
)
_RULE_DESCRIPTION = (
    "Place bulbs in white cells so that every white cell is lit. A bulb lights every white cell along "
    "its row and column until a wall blocks the line, and no two bulbs may light each other. A digit "
    "on a wall counts the bulbs in its up, down, left, and right neighbour cells exactly. The bulbs "
    "already shown are fixed."
)
_TERMINAL_VISUAL = (
    "the completed board on which every bulb of the placement appears as the same golden bulb, every "
    "lit white cell carries the warm tint, and the walls, digits, and edge numbers stay unchanged"
)
@dataclass
class LightUpAkariCompletionRecord:
    id: str
    task_name: str = "light_up_akari_completion"
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


class LightUpAkariCompletionGenerator(AbstractPuzzleGenerator[LightUpAkariCompletionRecord]):
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

    def create_puzzle(self, *args, **kwargs) -> LightUpAkariCompletionRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"light_up_akari_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    # ------------------------------------------------------------------
    # Solution sampling
    # ------------------------------------------------------------------

    def _segment(self, cell: Cell, black_set: Set[Cell]) -> List[Cell]:
        rows, cols = self.rows, self.cols
        r, c = cell
        result = [(r, c)]
        for nr in range(r - 1, -1, -1):
            if (nr, c) in black_set:
                break
            result.append((nr, c))
        for nr in range(r + 1, rows):
            if (nr, c) in black_set:
                break
            result.append((nr, c))
        for nc in range(c - 1, -1, -1):
            if (r, nc) in black_set:
                break
            result.append((r, nc))
        for nc in range(c + 1, cols):
            if (r, nc) in black_set:
                break
            result.append((r, nc))
        return result

    def _build_solution(self) -> Tuple[Set[Cell], Dict[Cell, int], Set[Cell]]:
        rows, cols = self.rows, self.cols
        for _ in range(80):
            black_count = self.rng.randint(rows * cols // 8, rows * cols // 5)
            cells = [(r, c) for r in range(rows) for c in range(cols)]
            self.rng.shuffle(cells)
            black_set = set(cells[:black_count])
            lights: Set[Cell] = set()
            unlit = {(r, c) for r in range(rows) for c in range(cols) if (r, c) not in black_set}
            ok = True
            iterations = 0
            while unlit and iterations < 80:
                iterations += 1
                cell = self.rng.choice(sorted(unlit))
                candidates = self._segment(cell, black_set)
                self.rng.shuffle(candidates)
                placed = False
                for cand in candidates:
                    if cand in black_set or cand in lights:
                        continue
                    if any(s != cand and s in lights for s in self._segment(cand, black_set)):
                        continue
                    lights.add(cand)
                    for lit_cell in self._segment(cand, black_set):
                        unlit.discard(lit_cell)
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if not ok or unlit:
                continue
            clue_numbers: Dict[Cell, int] = {}
            for r, c in black_set:
                count = sum(
                    1
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if (r + dr, c + dc) in lights
                )
                clue_numbers[(r, c)] = count
            clue_keys = sorted(clue_numbers)
            self.rng.shuffle(clue_keys)
            keep = max(2, len(clue_keys) // 2)
            shown = {key: clue_numbers[key] for key in clue_keys[:keep]}
            return black_set, shown, lights
        raise ForcedDeductionError("light_up_akari_completion: failed to build a solution")

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    def _forced_chain(
        self,
        black_set: Set[Cell],
        clues: Dict[Cell, int],
        lights: Set[Cell],
        given: Set[Cell],
    ) -> Optional[ForcedChain]:
        rows, cols = self.rows, self.cols
        placed: Set[Cell] = set(given)
        rounds: List[List[ForcedMove]] = []
        whites = {(r, c) for r in range(rows) for c in range(cols) if (r, c) not in black_set}

        def neighbours(cell: Cell) -> List[Cell]:
            r, c = cell
            return [
                (r + dr, c + dc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (r + dr, c + dc) in whites
            ]

        while placed != lights:
            lit = set()
            for bulb in placed:
                lit.update(self._segment(bulb, black_set))
            # A cell is open for a bulb if it is white, unlit, and not banned
            # by a satisfied wall digit.
            banned: Set[Cell] = set()
            for wall, count in clues.items():
                have = sum(1 for cell in neighbours(wall) if cell in placed)
                if have == count:
                    banned.update(cell for cell in neighbours(wall) if cell not in placed)
            open_set = {
                cell for cell in whites
                if cell not in placed and cell not in lit and cell not in banned
            }
            forced: Dict[Cell, str] = {}
            for wall in sorted(clues):
                count = clues[wall]
                have = sum(1 for cell in neighbours(wall) if cell in placed)
                open_neighbours = [cell for cell in neighbours(wall) if cell in open_set]
                remaining = count - have
                if remaining > 0 and len(open_neighbours) == remaining:
                    for cell in open_neighbours:
                        forced.setdefault(
                            cell,
                            f"The {count} on the wall at {display_cell_text(wall)} "
                            f"still needs {remaining} bulb{'s' if remaining != 1 else ''} and has "
                            f"exactly {remaining} open neighbour{'s' if remaining != 1 else ''} "
                            "left, so a bulb goes there",
                        )
            for cell in sorted(whites - lit):
                if cell in placed:
                    continue
                sources = [s for s in self._segment(cell, black_set) if s in open_set]
                if len(sources) == 1:
                    source = sources[0]
                    if cell == source:
                        reason = (
                            f"The white cell at {display_cell_text(cell)} can only "
                            "be lit by a bulb on that very cell, so one goes there"
                        )
                    else:
                        reason = (
                            f"The white cell at {display_cell_text(cell)} can only "
                            f"be lit from {display_cell_text(source)}, so a bulb "
                            "goes there"
                        )
                    forced.setdefault(source, reason)
            if not forced:
                return None
            for cell in forced:
                if cell not in lights:
                    raise ForcedDeductionError(
                        "light_up_akari_completion: forced bulb deviates from the constructed solution"
                    )
            self_lit = sorted(
                cell for cell, reason in forced.items() if "on that very cell" in reason
            )
            round_moves: List[ForcedMove] = []
            if len(self_lit) > 1:
                cells_phrase = ", ".join(
                    display_cell_text((r, c)) for r, c in self_lit
                )
                round_moves.append(
                    ForcedMove(
                        kind="bulb",
                        cells=tuple(self_lit),
                        value="B",
                        reason=(
                            f"The white cells at {cells_phrase} each sit where no other cell's "
                            "light can reach, so each holds its own bulb"
                        ),
                    )
                )
            elif len(self_lit) == 1:
                cell = self_lit[0]
                round_moves.append(
                    ForcedMove(kind="bulb", cells=(cell,), value="B", reason=forced[cell])
                )
            for cell in sorted(forced):
                if cell in self_lit:
                    continue
                round_moves.append(
                    ForcedMove(kind="bulb", cells=(cell,), value="B", reason=forced[cell])
                )
            for move in round_moves:
                placed.update(move.cells)
            rounds.append(round_moves)
        return ForcedChain(rounds=rounds)

    def _construct_instance(self):
        for _ in range(120):
            black_set, clues, lights = self._build_solution()
            given: Set[Cell] = set(lights)
            order = sorted(lights)
            self.rng.shuffle(order)
            for bulb in order:
                given.discard(bulb)
                if self._forced_chain(black_set, clues, lights, given) is None:
                    given.add(bulb)
            if not given:
                # The input legend promises visible given bulbs; re-pin one.
                given.add(self.rng.choice(sorted(lights)))
            chain = self._forced_chain(black_set, clues, lights, given)
            open_count = len(lights) - len(given)
            if chain is None or open_count < 4 or len(chain.rounds) < 2:
                continue
            if len(given) > len(lights) // 2:
                continue
            prove_complete_chain(
                chain, expected_cells=open_count, task_name="light_up_akari_completion"
            )
            return black_set, clues, lights, sorted(given), chain
        raise ForcedDeductionError(
            "light_up_akari_completion: failed to construct a forced-hand-solvable board"
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        black_set: Set[Cell],
        clues: Dict[Cell, int],
        bulbs: Set[Cell],
        *,
        solved: bool = True,
    ) -> Image.Image:
        rows, cols = self.rows, self.cols
        cell, left, top = standard_board_geometry(rows, cols)
        scene, draw = new_board_scene()
        digit_font = board_font(max(18, cell * 3 // 5))
        axis_font = board_font(max(11, cell // 4))

        lit_cells: Set[Cell] = set()
        for bulb in bulbs:
            lit_cells.update(self._segment(bulb, black_set))
        for r in range(rows):
            for c in range(cols):
                x0 = left + c * cell
                y0 = top + r * cell
                if (r, c) in black_set:
                    draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=_BLACK)
                else:
                    fill = _LIT if (r, c) in lit_cells else _WHITE_FILL
                    draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fill, outline=GRID_LIGHT, width=1)
        for (r, c), value in clues.items():
            draw.text(
                (left + c * cell + cell // 2, top + r * cell + cell // 2),
                str(value),
                fill=_WHITE_FILL,
                font=digit_font,
                anchor="mm",
            )
        draw.rectangle([left, top, left + cols * cell, top + rows * cell], outline=GRID_BORDER, width=3)
        draw_axis_labels(draw, rows, cols, cell=cell, left=left, top=top, font=axis_font)
        for r, c in bulbs:
            cx = left + c * cell + cell // 2
            cy = top + r * cell + cell // 2
            radius = max(8, cell // 3)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=_LIGHT,
                outline=GRID_BORDER,
                width=2,
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
                "Generate a solution video that starts from the unchanged puzzle. Bulbs then appear in "
                "deduction order: each wave of cells the digits and lighting rules force fades in "
                "together with the warm tint of the cells it lights, earlier bulbs and their light "
                f"stay exactly in place, and the video ends holding {_TERMINAL_VISUAL}."
            ),
            rule_description=_RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    # ------------------------------------------------------------------
    # Authored CoT
    # ------------------------------------------------------------------

    def _narrate(self, groups: Sequence[Sequence[ForcedMove]], given: Sequence[Cell]) -> List[str]:
        given_phrase = "; ".join(display_cell_text((r, c)) for r, c in given)
        opener = (
            f"The grid is {self.rows} by {self.cols} and bulbs already shine at {given_phrase}. "
            "Every white cell must end up lit, wall digits count their adjacent bulbs exactly, and "
            "no bulb may see another, so I will add bulbs where those rules leave a single choice."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " Every white cell is now lit, every wall digit is satisfied exactly, and no two bulbs "
            "see each other, so the lighting is complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    # ------------------------------------------------------------------
    # Record assembly
    # ------------------------------------------------------------------

    def create_random_puzzle(self) -> LightUpAkariCompletionRecord:
        black_set, clues, lights, given, chain = self._construct_instance()
        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        committed: Set[Cell] = set(given)
        states: List[Image.Image] = [fit_canvas(self._render(black_set, clues, set(committed)))]
        for group in groups:
            for move in group:
                committed.update(move.cells)
            states.append(fit_canvas(self._render(black_set, clues, set(committed))))
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

        light_payload = sorted([list(cell) for cell in lights])
        visible_lights = [[row + 1, col + 1] for row, col in light_payload]

        return LightUpAkariCompletionRecord(
            answer=answer_xy_positions(visible_lights),
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
                "black_cells": sorted([list(cell) for cell in black_set]),
                "clues": [[r, c, clues[(r, c)]] for r, c in sorted(clues)],
                "given_bulbs": [list(cell) for cell in given],
                "lights": light_payload,
            },
        )
