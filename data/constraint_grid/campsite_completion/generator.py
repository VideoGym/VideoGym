from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageFont, ImageDraw

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
from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.contracts.grid_completion import answer_xy_positions
from data.common.pipeline.grid_rendering import fit_canvas
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)

_TEXT = (24, 24, 24)
_GRID = (182, 182, 196)
_BG = (248, 249, 252)
_GROUND_FILL = (245, 240, 222)
_COUNT_BG = (232, 240, 250)
# Plain gray coordinate labels, deliberately unlike the boxed tent counts.
_AXIS_LABEL = (110, 118, 130)

# Pine-tree palette.
_TRUNK = (96, 64, 38)
_FOLIAGE_DARK = (44, 110, 60)
_FOLIAGE_MID = (76, 150, 90)
_FOLIAGE_LIGHT = (140, 196, 130)

# Tent palette.
_TENT_FABRIC = (200, 78, 60)
_TENT_FABRIC_SHADE = (146, 50, 38)
_TENT_POLE = (60, 44, 32)
_TENT_DOOR = (66, 32, 26)

_INPUT_DESCRIPTION = (
    "The image shows a campsite grid with pine trees on some cells, boxed tent counts along the top and "
    "left edges, and small gray 1-based row and column numbers along the opposite edges. Red tents that "
    "are already pitched are part of the final layout."
)
_RULE_DESCRIPTION = (
    "Pitch tents on empty ground so that the tents and trees pair up one to one, each tree standing "
    "directly above, below, left, or right of its own tent. No two tents may touch, not even "
    "diagonally, and each row and column must hold exactly its boxed number of tents. Trees never "
    "move, and the tents already shown are fixed."
)
_TERMINAL_VISUAL = (
    "the completed campsite on which every tent of the layout appears as the same red tent while the "
    "trees, counts, and edge numbers stay unchanged"
)

@dataclass
class CampsiteCompletionRecord:
    id: str
    image: str
    reasoning_image: str
    solution_image_path: str
    ti2v_prompt: str
    solution_image_prompt: str
    answer_grid: List[List[int]]
    structured_answer: Dict[str, object]
    task_name: str = "campsite_completion"
    task_family: str = "constraint_grid"
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None

    answer: str = ""

    def to_dict(self) -> Dict[str, object]:
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
            "answer_grid": self.answer_grid,
            "structured_answer": self.structured_answer,
        }


# Cell encoding for the underlying solution grid.
_EMPTY = 0
_TREE = 1
_TENT = 2


class CampsiteCompletionGenerator(AbstractPuzzleGenerator[CampsiteCompletionRecord]):
    """Tents puzzle: place tents adjacent to trees subject to row/column counts.

    The puzzle image shows the trees and the row/column tent counts. The
    solution image adds the tents (red triangle teepees) on top of empty cells
    so every tree has a unique orthogonal tent partner, no two tents touch
    even diagonally, and the row/col counts match.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        rows: int = 6,
        cols: int = 6,
        tents: Optional[int] = None,
        video: bool = False,
        cell_px: int = 60,
        padding: int = 28,
    ) -> None:
        super().__init__(output_dir)
        self.seed = seed
        self.rng = random.Random(seed)
        self.rows = rows
        self.cols = cols
        # Default to ~max(rows,cols) tents, which keeps the puzzle dense
        # without overcrowding.
        self.tents_target = tents if tents is not None else max(rows, cols)
        self.video = video
        self.cell_px = cell_px
        self.padding = padding
        self._counter = 0
        self.puzzle_dir = self.output_dir / "puzzles"
        self.reasoning_dir = self.output_dir / "reasoning"
        self.solution_dir = self.output_dir / "solutions"
        for directory in (self.puzzle_dir, self.reasoning_dir, self.solution_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def create_puzzle(self, *args, **kwargs) -> CampsiteCompletionRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"campsite_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    # ------------------------------------------------------------------
    # Solver / generator helpers
    # ------------------------------------------------------------------
    def _all_neighbors_8(self, r: int, c: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    out.append((nr, nc))
        return out

    def _all_neighbors_4(self, r: int, c: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                out.append((nr, nc))
        return out

    def _build_solution(self) -> List[List[int]]:
        """Generate a board with placed tent-tree pairs.

        Strategy:
        1. Pick ``tents_target`` random non-touching cells as tents (no two adjacent
           even diagonally).
        2. For each tent, attach a tree to one of its 4 orthogonal neighbours
           (must be empty and not already used as a tent or another tent's tree).
        3. If any tent fails to find a tree, retry from scratch.
        """
        for _ in range(400):
            grid = [[_EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
            cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
            self.rng.shuffle(cells)
            tent_positions: List[Tuple[int, int]] = []
            for r, c in cells:
                if len(tent_positions) >= self.tents_target:
                    break
                if grid[r][c] != _EMPTY:
                    continue
                touches = any(grid[nr][nc] == _TENT for nr, nc in self._all_neighbors_8(r, c))
                if touches:
                    continue
                grid[r][c] = _TENT
                tent_positions.append((r, c))
            if len(tent_positions) < max(2, self.tents_target - 1):
                continue
            # Pair each tent with a tree via orthogonal matching.
            paired = self._pair_trees(grid, tent_positions)
            if paired is None:
                continue
            for r, c in paired:
                grid[r][c] = _TREE
            return grid
        raise RuntimeError("Failed to build campsite solution")

    def _pair_trees(
        self,
        grid: Sequence[Sequence[int]],
        tents: Sequence[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        """Bipartite matching: each tent gets exactly one orthogonal tree
        (a unique empty neighbour), with no two tents sharing the same tree."""
        # Build candidate list per tent.
        candidates: List[List[Tuple[int, int]]] = []
        for tr, tc in tents:
            opts = []
            for nr, nc in self._all_neighbors_4(tr, tc):
                if grid[nr][nc] == _EMPTY:
                    opts.append((nr, nc))
            self.rng.shuffle(opts)
            candidates.append(opts)
        order = list(range(len(tents)))
        self.rng.shuffle(order)
        match: Dict[Tuple[int, int], int] = {}

        def try_match(i: int, seen: set[Tuple[int, int]]) -> bool:
            for tree in candidates[i]:
                if tree in seen:
                    continue
                seen.add(tree)
                if tree not in match or try_match(match[tree], seen):
                    match[tree] = i
                    return True
            return False

        for i in order:
            if not try_match(i, set()):
                return None
        # Reconstruct trees in the order of tents.
        trees: List[Tuple[int, int]] = [None] * len(tents)  # type: ignore
        for tree, ti in match.items():
            trees[ti] = tree
        return [t for t in trees if t is not None]

    def _row_counts(self, grid: Sequence[Sequence[int]]) -> List[int]:
        return [sum(1 for v in row if v == _TENT) for row in grid]

    def _col_counts(self, grid: Sequence[Sequence[int]]) -> List[int]:
        return [
            sum(1 for r in range(self.rows) if grid[r][c] == _TENT)
            for c in range(self.cols)
        ]

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------
    def _forced_chain(
        self,
        trees: Sequence[Tuple[int, int]],
        tents: Sequence[Tuple[int, int]],
        row_counts: Sequence[int],
        col_counts: Sequence[int],
        given: Set[Tuple[int, int]],
    ) -> Optional[ForcedChain]:
        tree_set = set(trees)
        solution = set(tents)
        placed: Set[Tuple[int, int]] = set(given)
        rounds: List[List[ForcedMove]] = []

        def open_cells() -> Set[Tuple[int, int]]:
            row_have = [sum(1 for t in placed if t[0] == r) for r in range(self.rows)]
            col_have = [sum(1 for t in placed if t[1] == c) for c in range(self.cols)]
            result: Set[Tuple[int, int]] = set()
            for r in range(self.rows):
                for c in range(self.cols):
                    cell = (r, c)
                    if cell in placed or cell in tree_set:
                        continue
                    if not any(n in tree_set for n in self._all_neighbors_4(r, c)):
                        continue
                    if row_have[r] >= row_counts[r] or col_have[c] >= col_counts[c]:
                        continue
                    if any(max(abs(r - tr), abs(c - tc)) <= 1 for tr, tc in placed):
                        continue
                    result.add(cell)
            return result

        while placed != solution:
            open_set = open_cells()
            row_have = [sum(1 for t in placed if t[0] == r) for r in range(self.rows)]
            col_have = [sum(1 for t in placed if t[1] == c) for c in range(self.cols)]
            forced: Dict[Tuple[int, int], str] = {}
            for r in range(self.rows):
                remaining = row_counts[r] - row_have[r]
                cells = sorted(cell for cell in open_set if cell[0] == r)
                if remaining > 0 and len(cells) == remaining:
                    for cell in cells:
                        forced.setdefault(
                            cell,
                            f"Row {r + 1} still needs {remaining} tent"
                            f"{'s' if remaining != 1 else ''} and has exactly "
                            f"{remaining} open cell{'s' if remaining != 1 else ''} left, so a "
                            f"tent goes at {display_cell_text(cell)}",
                        )
            for c in range(self.cols):
                remaining = col_counts[c] - col_have[c]
                cells = sorted(cell for cell in open_set if cell[1] == c)
                if remaining > 0 and len(cells) == remaining:
                    for cell in cells:
                        forced.setdefault(
                            cell,
                            f"Column {c + 1} still needs {remaining} tent"
                            f"{'s' if remaining != 1 else ''} and has exactly "
                            f"{remaining} open cell{'s' if remaining != 1 else ''} left, so a "
                            f"tent goes at {display_cell_text(cell)}",
                        )
            for tr, tc in sorted(tree_set):
                if any(n in placed for n in self._all_neighbors_4(tr, tc)):
                    continue
                options = [n for n in self._all_neighbors_4(tr, tc) if n in open_set]
                if len(options) == 1:
                    cell = options[0]
                    forced.setdefault(
                        cell,
                        f"The tree at {display_cell_text((tr, tc))} has a single free side "
                        f"left, so its tent stands at {display_cell_text(cell)}",
                    )
            if not forced:
                return None
            round_moves: List[ForcedMove] = []
            for cell in sorted(forced):
                if cell not in solution:
                    raise ForcedDeductionError(
                        "campsite_completion: forced tent deviates from the constructed layout"
                    )
                round_moves.append(
                    ForcedMove(kind="tent", cells=(cell,), value="T", reason=forced[cell])
                )
            for move in round_moves:
                placed.add(move.cells[0])
            rounds.append(round_moves)
        return ForcedChain(rounds=rounds)

    def _narrate(
        self,
        groups: Sequence[Sequence[ForcedMove]],
        given: Sequence[Tuple[int, int]],
        tree_count: int,
    ) -> List[str]:
        given_phrase = "; ".join(display_cell_text((r, c)) for r, c in given)
        opener = (
            f"The campsite is {self.rows} by {self.cols} with {tree_count} trees, and tents are "
            f"already pitched at {given_phrase}. Every tree needs its own orthogonally adjacent "
            "tent, tents never touch, and the boxed counts are exact, so I will pitch tents where "
            "those rules leave no choice."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " Every tree now has its own tent, no two tents touch, and every row and column "
            "matches its count, so the campsite is complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    def _render_board(
        self,
        trees: Sequence[Tuple[int, int]],
        tents: Sequence[Tuple[int, int]],
        row_counts: Sequence[int],
        col_counts: Sequence[int],
        *,
        solved: bool,
    ) -> Image.Image:
        cell = self.cell_px
        pad = self.padding
        header = cell  # reserve a row+col for counts
        # Right/bottom gutter for the 1-based coordinate labels, kept apart
        # from the boxed tent counts on the top/left header.
        gutter = max(20, cell // 3)
        width = pad * 2 + header + self.cols * cell + gutter
        height = pad * 2 + header + self.rows * cell + gutter
        image, draw = new_scaled_canvas(width, height, DOCTRINE_SUPERSAMPLE, background=_BG)
        number_font = self._load_font(max(18, cell // 2) * DOCTRINE_SUPERSAMPLE)

        # Column counts at top.
        for c in range(self.cols):
            x = pad + header + c * cell
            y = pad
            draw.rectangle([x, y, x + cell, y + header], fill=_COUNT_BG, outline=_GRID, width=2)
            text = str(col_counts[c])
            bbox = draw.textbbox((0, 0), text, font=number_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x + (cell - tw) // 2 - bbox[0], y + (header - th) // 2 - bbox[1]),
                text,
                fill=_TEXT,
                font=number_font,
            )

        # Row counts at left.
        for r in range(self.rows):
            x = pad
            y = pad + header + r * cell
            draw.rectangle([x, y, x + header, y + cell], fill=_COUNT_BG, outline=_GRID, width=2)
            text = str(row_counts[r])
            bbox = draw.textbbox((0, 0), text, font=number_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (x + (header - tw) // 2 - bbox[0], y + (cell - th) // 2 - bbox[1]),
                text,
                fill=_TEXT,
                font=number_font,
            )

        # Top-left corner cell (decorative).
        draw.rectangle(
            [pad, pad, pad + header, pad + header],
            fill=_BG,
            outline=_GRID,
            width=2,
        )

        # Ground cells.
        tree_set = set(trees)
        tent_set = set(tents)
        for r in range(self.rows):
            for c in range(self.cols):
                x = pad + header + c * cell
                y = pad + header + r * cell
                fill = _GROUND_FILL
                draw.rectangle([x, y, x + cell, y + cell], fill=fill, outline=_GRID, width=2)
                if (r, c) in tree_set:
                    self._draw_tree(draw, x, y, cell)
                elif (r, c) in tent_set:
                    self._draw_tent(draw, x, y, cell)

        # The typed answer names cells by (row, column).  The boxed numbers on
        # the top/left header are tent counts, so the 1-based row/column
        # labels sit on the opposite sides in plain gray text with no box to
        # stay visually distinct.
        label_font = self._load_font(max(12, min(20, cell // 4)) * DOCTRINE_SUPERSAMPLE)
        board_right = pad + header + self.cols * cell
        board_bottom = pad + header + self.rows * cell
        for r in range(self.rows):
            y_center = pad + header + r * cell + cell // 2
            draw.text((board_right + gutter // 2 + 2, y_center), str(r + 1),
                      fill=_AXIS_LABEL, font=label_font, anchor="mm")
        for c in range(self.cols):
            x_center = pad + header + c * cell + cell // 2
            draw.text((x_center, board_bottom + gutter // 2 + 2), str(c + 1),
                      fill=_AXIS_LABEL, font=label_font, anchor="mm")

        return finish_scaled_canvas(image, (width, height))

    def _draw_tree(self, draw: ImageDraw.ImageDraw, x: int, y: int, cell: int) -> None:
        """Stylised pine tree: a brown trunk with three layered green triangles."""
        cx = x + cell // 2
        # Trunk.
        trunk_w = max(4, cell // 8)
        trunk_h = max(6, cell // 6)
        trunk_top = y + cell - trunk_h - max(2, cell // 12)
        draw.rectangle(
            [cx - trunk_w // 2, trunk_top, cx + trunk_w // 2, trunk_top + trunk_h],
            fill=_TRUNK,
        )
        # Three triangle layers (bottom-up: dark, mid, light at top).
        margin = max(4, cell // 10)
        top_y = y + margin
        bot_y = trunk_top - max(2, cell // 16)
        total_h = bot_y - top_y
        if total_h < 12:
            total_h = 12
        layer_h = total_h // 3
        # Layer 3 (bottom, widest, darkest).
        bot_w = cell - 2 * margin
        l3_top = bot_y - layer_h
        draw.polygon(
            [
                (cx, l3_top - layer_h // 2),
                (cx - bot_w // 2, bot_y),
                (cx + bot_w // 2, bot_y),
            ],
            fill=_FOLIAGE_DARK,
        )
        # Layer 2 (middle).
        mid_w = int(bot_w * 0.78)
        l2_top = l3_top - layer_h // 3
        draw.polygon(
            [
                (cx, l2_top - layer_h // 2),
                (cx - mid_w // 2, l3_top + layer_h // 4),
                (cx + mid_w // 2, l3_top + layer_h // 4),
            ],
            fill=_FOLIAGE_MID,
        )
        # Layer 1 (top, narrowest).
        top_w = int(bot_w * 0.55)
        l1_top = top_y
        draw.polygon(
            [
                (cx, l1_top),
                (cx - top_w // 2, l2_top + layer_h // 4),
                (cx + top_w // 2, l2_top + layer_h // 4),
            ],
            fill=_FOLIAGE_LIGHT,
        )

    def _draw_tent(self, draw: ImageDraw.ImageDraw, x: int, y: int, cell: int) -> None:
        """Stylised teepee: red triangle with a darker right-side shading and a
        tiny dark door in the middle bottom."""
        cx = x + cell // 2
        margin = max(4, cell // 8)
        top_y = y + margin
        bot_y = y + cell - margin
        left_x = x + margin
        right_x = x + cell - margin
        # Darker right shading.
        draw.polygon(
            [
                (cx, top_y),
                (right_x, bot_y),
                (cx, bot_y),
            ],
            fill=_TENT_FABRIC_SHADE,
        )
        # Lighter left side.
        draw.polygon(
            [
                (cx, top_y),
                (left_x, bot_y),
                (cx, bot_y),
            ],
            fill=_TENT_FABRIC,
        )
        # Outline.
        draw.polygon(
            [
                (cx, top_y),
                (right_x, bot_y),
                (left_x, bot_y),
            ],
            outline=_TENT_POLE,
        )
        # Centre vertical pole.
        draw.line([(cx, top_y), (cx, bot_y)], fill=_TENT_POLE, width=2)
        # Tiny door at the bottom centre.
        door_w = max(3, cell // 14)
        door_h = max(5, cell // 10)
        draw.polygon(
            [
                (cx - door_w, bot_y),
                (cx + door_w, bot_y),
                (cx, bot_y - door_h),
            ],
            fill=_TENT_DOOR,
        )

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        return ImageFont.load_default()

    def _build_instance(self):
        for _ in range(160):
            grid = self._build_solution()
            trees = [(r, c) for r in range(self.rows) for c in range(self.cols) if grid[r][c] == _TREE]
            tents = [(r, c) for r in range(self.rows) for c in range(self.cols) if grid[r][c] == _TENT]
            if not tents:
                continue
            row_counts = self._row_counts(grid)
            col_counts = self._col_counts(grid)
            given: Set[Tuple[int, int]] = set(tents)
            order = list(tents)
            self.rng.shuffle(order)
            for tent in order:
                given.discard(tent)
                if self._forced_chain(trees, tents, row_counts, col_counts, given) is None:
                    given.add(tent)
            if not given:
                # The input legend promises visible given tents; re-pin one.
                given.add(self.rng.choice(sorted(tents)))
            chain = self._forced_chain(trees, tents, row_counts, col_counts, given)
            open_count = len(tents) - len(given)
            if chain is None or open_count < 3 or len(chain.rounds) < 2:
                continue
            if len(given) > len(tents) // 2:
                continue
            prove_complete_chain(chain, expected_cells=open_count, task_name="campsite_completion")
            return trees, tents, row_counts, col_counts, sorted(given), chain
        raise ForcedDeductionError(
            "campsite_completion: failed to construct a forced-hand-solvable campsite"
        )

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
                "Generate a solution video that starts from the unchanged puzzle. Red tents then "
                "appear in deduction order: each wave of cells the rules force fades in together, "
                "earlier tents stay exactly in place, and the video ends holding "
                f"{_TERMINAL_VISUAL}."
            ),
            rule_description=_RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    def create_random_puzzle(self) -> CampsiteCompletionRecord:
        trees, tents, row_counts, col_counts, given, chain = self._build_instance()
        for tr, tc in tents:
            assert any((nr, nc) in set(trees) for nr, nc in self._all_neighbors_4(tr, tc))
        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        committed: List[Tuple[int, int]] = list(given)
        states: List[Image.Image] = [
            fit_canvas(self._render_board(trees, committed, row_counts, col_counts, solved=False))
        ]
        for group in groups:
            committed = committed + [move.cells[0] for move in group]
            states.append(
                fit_canvas(self._render_board(trees, committed, row_counts, col_counts, solved=True))
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
        video_fps: Optional[int] = None
        video_num_frames: Optional[int] = None
        if self.video:
            video_path = self.solution_dir / f"{puzzle_id}_solution.webm"
            video_num_frames = encode_commit_video(states, video_path, fps=16)
            if video_num_frames > 0 and video_path.exists():
                solution_video_path = self.relativize_path(video_path)
                video_fps = 16

        answer_grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for r, c in tents:
            answer_grid[r][c] = 1
        tent_cells = [[int(r), int(c)] for r, c in sorted(tents)]
        visible_tents = [[row + 1, col + 1] for row, col in tent_cells]
        self._narrate(groups, given, len(trees))

        return CampsiteCompletionRecord(
            answer=answer_xy_positions(visible_tents),
            id=puzzle_id,
            image=self.relativize_path(puzzle_path),
            reasoning_image=self.relativize_path(reasoning_path),
            solution_image_path=self.relativize_path(solution_path),
            solution_video_path=solution_video_path,
            video_fps=video_fps,
            video_num_frames=video_num_frames,
            ti2v_prompt=prompts["ti2v"],
            solution_image_prompt=prompts["solution_image"],
            answer_grid=answer_grid,
            structured_answer={
                "rows": self.rows,
                "cols": self.cols,
                "row_counts": list(map(int, row_counts)),
                "col_counts": list(map(int, col_counts)),
                "trees": [[int(r), int(c)] for r, c in trees],
                "given_tents": [[int(r), int(c)] for r, c in given],
                "tents": tent_cells,
            },
        )
