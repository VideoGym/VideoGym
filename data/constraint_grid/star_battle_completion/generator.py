"""Star battle completion: extend given stars to one star per row, column, and region. Every emitted puzzle is forced-hand solvable: unit propagation over the
visible rules (cells touching a star are dead; a starred row, column, or
region closes its other cells; a unit whose last open cell remains forces a
star; a unit whose open cells all share one line or region rules the rest of
that line or region out) completes the board from the given stars with zero
guessing. The generator proves the chain per sample after monotone trimming
of the given set; the chain drives the authored CoT, the TI2V key states,
and the TI2V commit order."""

from __future__ import annotations

import math
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

_STAR = (240, 180, 40)
_STAR_OUT = (140, 100, 20)
_REGION_PALETTE = [
    (255, 244, 220), (218, 234, 248), (224, 244, 220),
    (252, 226, 240), (236, 240, 250), (250, 232, 218),
    (236, 220, 250), (220, 248, 236), (252, 220, 220),
]

_INPUT_DESCRIPTION = (
    "The image shows a square grid divided into softly colored regions by bold dark boundaries, with "
    "small gray 1-based row and column numbers along the top and left edges. Gold stars that are "
    "already placed are part of the final placement."
)
_RULE_DESCRIPTION = (
    "Place stars so that every row, every column, and every colored region contains exactly one star, "
    "and no two stars touch, not even diagonally. The stars already shown are fixed."
)
_TERMINAL_VISUAL = (
    "the completed board on which every star of the placement appears as the same gold star while the "
    "regions, boundaries, and edge numbers stay unchanged"
)
@dataclass
class StarBattleCompletionRecord:
    id: str
    task_name: str = "star_battle_completion"
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


class StarBattleCompletionGenerator(AbstractPuzzleGenerator[StarBattleCompletionRecord]):
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

    def create_puzzle(self, *args, **kwargs) -> StarBattleCompletionRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"star_battle_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    # ------------------------------------------------------------------
    # Solution sampling (star pattern, then regions grown around it)
    # ------------------------------------------------------------------

    def _build_star_pattern(self) -> Optional[List[Cell]]:
        n = self.size
        for _ in range(400):
            cols_used: List[Optional[int]] = [None] * n
            ok = True
            for r in range(n):
                avail = []
                for c in range(n):
                    if c in cols_used:
                        continue
                    if any(
                        cols_used[rr] is not None
                        and abs(rr - r) <= 1
                        and abs(cols_used[rr] - c) <= 1
                        for rr in range(r)
                    ):
                        continue
                    avail.append(c)
                if not avail:
                    ok = False
                    break
                cols_used[r] = self.rng.choice(avail)
            if ok and all(c is not None for c in cols_used):
                return [(r, cols_used[r]) for r in range(n)]
        return None

    def _build_regions(self, stars: List[Cell]) -> List[List[Cell]]:
        n = self.size
        for _ in range(200):
            region_id = [[-1 for _ in range(n)] for _ in range(n)]
            for rid, (r, c) in enumerate(stars):
                region_id[r][c] = rid
            cells_to_assign = [(r, c) for r in range(n) for c in range(n) if region_id[r][c] == -1]
            self.rng.shuffle(cells_to_assign)
            iterations = 0
            while cells_to_assign and iterations < n * n * 4:
                iterations += 1
                cell = cells_to_assign[0]
                neighbor_rids = set()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cell[0] + dr, cell[1] + dc
                    if 0 <= nr < n and 0 <= nc < n and region_id[nr][nc] >= 0:
                        neighbor_rids.add(region_id[nr][nc])
                if neighbor_rids:
                    rid = self.rng.choice(sorted(neighbor_rids))
                    region_id[cell[0]][cell[1]] = rid
                    cells_to_assign.pop(0)
                else:
                    cells_to_assign.append(cells_to_assign.pop(0))
            if any(region_id[r][c] == -1 for r in range(n) for c in range(n)):
                continue
            regions: Dict[int, List[Cell]] = {}
            for r in range(n):
                for c in range(n):
                    regions.setdefault(region_id[r][c], []).append((r, c))
            return [regions[k] for k in sorted(regions)]
        raise ForcedDeductionError("star_battle_completion: failed to build regions")

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    def _forced_chain(
        self,
        stars: Sequence[Cell],
        regions: Sequence[Sequence[Cell]],
        given: Set[Cell],
    ) -> Optional[ForcedChain]:
        n = self.size
        solution = set(stars)
        region_of: Dict[Cell, int] = {}
        for rid, cells in enumerate(regions):
            for cell in cells:
                region_of[tuple(cell)] = rid
        anchors = [min(cells) for cells in regions]

        placed: Set[Cell] = set(given)
        excluded: Set[Cell] = set()
        pending: List[str] = []
        rounds: List[List[ForcedMove]] = []

        def region_phrase(rid: int) -> str:
            r, c = anchors[rid]
            return f"the region containing {display_cell_text((r, c))}"

        def open_cells() -> List[Cell]:
            starred_rows = {r for r, _ in placed}
            starred_cols = {c for _, c in placed}
            starred_regions = {region_of[cell] for cell in placed}
            result = []
            for r in range(n):
                for c in range(n):
                    cell = (r, c)
                    if cell in placed or cell in excluded:
                        continue
                    if r in starred_rows or c in starred_cols or region_of[cell] in starred_regions:
                        continue
                    if any(abs(r - pr) <= 1 and abs(c - pc) <= 1 for pr, pc in placed):
                        continue
                    result.append(cell)
            return result

        while len(placed) < n:
            candidates = open_cells()
            by_row: Dict[int, List[Cell]] = {}
            by_col: Dict[int, List[Cell]] = {}
            by_region: Dict[int, List[Cell]] = {}
            for cell in candidates:
                by_row.setdefault(cell[0], []).append(cell)
                by_col.setdefault(cell[1], []).append(cell)
                by_region.setdefault(region_of[cell], []).append(cell)

            starred_rows = {r for r, _ in placed}
            starred_cols = {c for _, c in placed}
            starred_regions = {region_of[cell] for cell in placed}
            forced: Dict[Cell, str] = {}
            for r in range(n):
                if r in starred_rows:
                    continue
                cells = by_row.get(r, [])
                if not cells:
                    return None
                if len(cells) == 1:
                    forced.setdefault(
                        cells[0],
                        f"Row {r + 1} has a single open cell left at {display_cell_text(cells[0])}, so "
                        "its star goes there",
                    )
            for c in range(n):
                if c in starred_cols:
                    continue
                cells = by_col.get(c, [])
                if not cells:
                    return None
                if len(cells) == 1:
                    forced.setdefault(
                        cells[0],
                        f"Column {c + 1} has a single open cell left at {display_cell_text(cells[0])}, so "
                        "its star goes there",
                    )
            for rid in range(len(regions)):
                if rid in starred_regions:
                    continue
                cells = by_region.get(rid, [])
                if not cells:
                    return None
                if len(cells) == 1:
                    forced.setdefault(
                        cells[0],
                        f"{region_phrase(rid).capitalize()} has a single open cell left at "
                        f"{display_cell_text(cells[0])}, so its star goes there",
                    )

            if forced:
                round_moves: List[ForcedMove] = []
                for cell in sorted(forced):
                    if cell not in solution:
                        raise ForcedDeductionError(
                            "star_battle_completion: forced star deviates from the constructed pattern"
                        )
                    reason = forced[cell]
                    if pending:
                        reason = "; ".join(pending + [reason])
                        pending = []
                    round_moves.append(
                        ForcedMove(kind="star", cells=(cell,), value="*", reason=reason)
                    )
                for move in round_moves:
                    placed.add(move.cells[0])
                rounds.append(round_moves)
                continue

            # Pointing eliminations between regions and lines.
            eliminated_any = False
            for rid in range(len(regions)):
                if rid in starred_regions:
                    continue
                cells = by_region.get(rid, [])
                rows_used = {cell[0] for cell in cells}
                cols_used = {cell[1] for cell in cells}
                if len(rows_used) == 1:
                    row = next(iter(rows_used))
                    others = [cell for cell in by_row.get(row, []) if region_of[cell] != rid]
                    if others:
                        excluded.update(others)
                        pending.append(
                            f"Every open cell of {region_phrase(rid)} lies in row {row + 1}, so the "
                            "rest of that row is ruled out"
                        )
                        eliminated_any = True
                if len(cols_used) == 1:
                    col = next(iter(cols_used))
                    others = [cell for cell in by_col.get(col, []) if region_of[cell] != rid]
                    if others:
                        excluded.update(others)
                        pending.append(
                            f"Every open cell of {region_phrase(rid)} lies in column {col + 1}, so "
                            "the rest of that column is ruled out"
                        )
                        eliminated_any = True
            for r in range(n):
                if r in starred_rows:
                    continue
                rids = {region_of[cell] for cell in by_row.get(r, [])}
                if len(rids) == 1:
                    rid = next(iter(rids))
                    others = [cell for cell in by_region.get(rid, []) if cell[0] != r]
                    if others:
                        excluded.update(others)
                        pending.append(
                            f"Row {r + 1}'s open cells all belong to {region_phrase(rid)}, so that "
                            "region's cells outside the row are ruled out"
                        )
                        eliminated_any = True
            for c in range(n):
                if c in starred_cols:
                    continue
                rids = {region_of[cell] for cell in by_col.get(c, [])}
                if len(rids) == 1:
                    rid = next(iter(rids))
                    others = [cell for cell in by_region.get(rid, []) if cell[1] != c]
                    if others:
                        excluded.update(others)
                        pending.append(
                            f"Column {c + 1}'s open cells all belong to {region_phrase(rid)}, so "
                            "that region's cells outside the column are ruled out"
                        )
                        eliminated_any = True
            if not eliminated_any:
                return None
        return ForcedChain(rounds=rounds)

    def _construct_instance(self) -> Tuple[List[Cell], List[List[Cell]], List[Cell], ForcedChain]:
        n = self.size
        for _ in range(160):
            stars = self._build_star_pattern()
            if stars is None:
                continue
            regions = self._build_regions(stars)
            given: Set[Cell] = set(stars)
            order = list(stars)
            self.rng.shuffle(order)
            for star in order:
                given.discard(star)
                if self._forced_chain(stars, regions, given) is None:
                    given.add(star)
            chain = self._forced_chain(stars, regions, given)
            open_count = n - len(given)
            if chain is None or open_count < 4 or len(chain.rounds) < 2:
                continue
            if len(given) > n // 2:
                continue
            prove_complete_chain(chain, expected_cells=open_count, task_name="star_battle_completion")
            return stars, regions, sorted(given), chain
        raise ForcedDeductionError(
            "star_battle_completion: failed to construct a forced-hand-solvable board"
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, committed_stars: Sequence[Cell], regions: Sequence[Sequence[Cell]]) -> Image.Image:
        n = self.size
        cell, left, top = standard_board_geometry(n, n)
        scene, draw = new_board_scene()
        axis_font = board_font(max(11, cell // 4))

        region_id = [[-1 for _ in range(n)] for _ in range(n)]
        for rid, cells in enumerate(regions):
            color = _REGION_PALETTE[rid % len(_REGION_PALETTE)]
            for r, c in cells:
                region_id[r][c] = rid
                draw.rectangle(
                    [left + c * cell, top + r * cell, left + (c + 1) * cell, top + (r + 1) * cell],
                    fill=color,
                )
        for r in range(n + 1):
            y = top + r * cell
            draw.line([(left, y), (left + n * cell, y)], fill=GRID_LIGHT, width=1)
        for c in range(n + 1):
            x = left + c * cell
            draw.line([(x, top), (x, top + n * cell)], fill=GRID_LIGHT, width=1)
        for r in range(n):
            for c in range(n):
                x0 = left + c * cell
                y0 = top + r * cell
                if c + 1 < n and region_id[r][c] != region_id[r][c + 1]:
                    draw.line([(x0 + cell, y0), (x0 + cell, y0 + cell)], fill=GRID_BORDER, width=3)
                if r + 1 < n and region_id[r][c] != region_id[r + 1][c]:
                    draw.line([(x0, y0 + cell), (x0 + cell, y0 + cell)], fill=GRID_BORDER, width=3)
        draw.rectangle([left, top, left + n * cell, top + n * cell], outline=GRID_BORDER, width=3)
        draw_axis_labels(draw, n, n, cell=cell, left=left, top=top, font=axis_font)

        for r, c in committed_stars:
            cx = left + c * cell + cell // 2
            cy = top + r * cell + cell // 2
            self._draw_star(draw, cx, cy, max(8, cell // 3))
        return finish_board_scene(scene)

    def _draw_star(self, draw, cx, cy, radius):
        points = []
        for i in range(10):
            angle = -math.pi / 2 + i * math.pi / 5
            r = radius if i % 2 == 0 else radius * 0.5
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(points, fill=_STAR, outline=_STAR_OUT)

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
                "Generate a solution video that starts from the unchanged puzzle. Gold stars then "
                "appear in deduction order: each wave of cells the rules force fades in together, "
                "earlier stars stay exactly in place, and the video ends holding "
                f"{_TERMINAL_VISUAL}."
            ),
            rule_description=_RULE_DESCRIPTION,
            mode="ti2v",
        )
        return { "solution_image": solution_image, "ti2v": ti2v}

    # ------------------------------------------------------------------
    # Authored CoT
    # ------------------------------------------------------------------

    def _narrate(self, groups: Sequence[Sequence[ForcedMove]], given: Sequence[Cell]) -> List[str]:
        n = self.size
        given_phrase = "; ".join(display_cell_text((r, c)) for r, c in given)
        opener = (
            f"The board is {n} by {n} with {n} regions, and stars already sit at {given_phrase}. "
            "Each star closes its row, column, and region and rules out all eight touching cells, "
            "so I will place the remaining stars where those rules leave a single open cell."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " Every row, column, and region now holds exactly one star and no two stars touch, so "
            "the placement is complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    # ------------------------------------------------------------------
    # Record assembly
    # ------------------------------------------------------------------

    def create_random_puzzle(self) -> StarBattleCompletionRecord:
        stars, regions, given, chain = self._construct_instance()
        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        committed: List[Cell] = list(given)
        states: List[Image.Image] = [fit_canvas(self._render(committed, regions))]
        for group in groups:
            committed = committed + [move.cells[0] for move in group]
            states.append(fit_canvas(self._render(committed, regions)))
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

        star_payload = [list(s) for s in stars]
        visible_stars = [[row + 1, col + 1] for row, col in star_payload]

        return StarBattleCompletionRecord(
            answer=answer_xy_positions(visible_stars),
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
                "stars": star_payload,
                "given_stars": [list(cell) for cell in given],
                "regions": [[list(c) for c in cells] for cells in regions],
            },
        )
