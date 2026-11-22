"""Aquarium completion: settle water into bordered tanks under row/column counts. Every emitted puzzle is forced-hand solvable. A per-tank interval engine
tightens each tank's possible water level using only the visible row and
column counts, the settling rule, and the given water/dry cells; at every
round at least one new cell is decided, and the chain reaches the fully
decided board with zero guessing. The proven chain drives the authored CoT,
the TI2V commit order."""

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
    reveal_budget_guard,
)
from data.constraint_grid.core.coordinate_format import display_cell_text
from data.constraint_grid.core.scene import (
    AXIS_LABEL,
    GRID_BORDER,
    TEXT,
    board_font,
    finish_board_scene,
    new_board_scene,
    standard_board_geometry,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.contracts.grid_completion import answer_json
from data.common.pipeline.grid_rendering import fit_canvas
from data.common.prompts import build_sectioned_prompt

Cell = Tuple[int, int]

_WATER = (90, 168, 220)
_DRY_RING = (110, 125, 150)
_GRID_LIGHT = (210, 215, 225)
_REGION_PALETTE = [
    (255, 244, 220), (218, 234, 248), (224, 244, 220),
    (252, 226, 240), (236, 240, 250), (250, 232, 218),
    (236, 220, 250), (220, 248, 236), (252, 220, 220),
]

_INPUT_DESCRIPTION = (
    "The image shows a square grid of tanks outlined by bold dark borders, each tank tinted its own "
    "pastel shade. Dark numbers above each column and left of each row count that line's water cells; "
    "small gray numbers below each column and right of each row are 1-based column and row numbers. "
    "Cells "
    "already under water are painted blue, and cells already confirmed dry carry a small gray ring."
)
_RULE_DESCRIPTION = (
    "Water in a tank settles: if a cell holds water, every cell of the same tank in that row or any "
    "lower row also holds water. Decide every remaining cell so that each row and column shows exactly "
    "its counted number of water cells. The water and dry cells already shown are fixed."
)
_TERMINAL_VISUAL = (
    "the finished board on which every water cell is painted the same blue, every dry cell keeps its "
    "pastel tank shade under a small gray ring, and the borders, counts, and edge numbers stay unchanged"
)
@dataclass
class AquariumCompletionRecord:
    id: str
    task_name: str = "aquarium_completion"
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


class _Tank:
    """Band bookkeeping for one region: level t fills the bottom t row bands."""

    def __init__(self, cells: Sequence[Cell]) -> None:
        self.cells = [tuple(cell) for cell in cells]
        self.rows = sorted({r for r, _ in self.cells})
        self.k = len(self.rows)
        self.anchor = min(self.cells)
        self.need = {r: self.k - i for i, r in enumerate(self.rows)}  # band r filled iff t >= need[r]
        self.width = {r: sum(1 for rr, _ in self.cells if rr == r) for r in self.rows}

    def phrase(self) -> str:
        return f"the tank at {display_cell_text(self.anchor)}"

    def row_contribution(self, row: int, level: int) -> int:
        if row not in self.need:
            return 0
        return self.width[row] if level >= self.need[row] else 0

    def col_needs(self, col: int) -> List[int]:
        return sorted(self.need[r] for r, c in self.cells if c == col)

    def col_contribution(self, col_needs: Sequence[int], level: int) -> int:
        return sum(1 for need in col_needs if level >= need)

    def filled_cells(self, level: int) -> Set[Cell]:
        return {(r, c) for r, c in self.cells if level >= self.need[r]}

    def dry_cells(self, level_cap: int) -> Set[Cell]:
        """Cells decided dry when the level can be at most ``level_cap``."""
        return {(r, c) for r, c in self.cells if self.need[r] > level_cap}


class AquariumCompletionGenerator(AbstractPuzzleGenerator[AquariumCompletionRecord]):
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

    def create_puzzle(self, *args, **kwargs) -> AquariumCompletionRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"aquarium_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    # ------------------------------------------------------------------
    # Region and level sampling
    # ------------------------------------------------------------------

    def _build_regions(self) -> List[List[Cell]]:
        n = self.size
        for _attempt in range(400):
            region_id = [[-1 for _ in range(n)] for _ in range(n)]
            next_id = 0
            ok = True
            while True:
                cells_left = [(r, c) for r in range(n) for c in range(n) if region_id[r][c] == -1]
                if not cells_left:
                    break
                seed = self.rng.choice(cells_left)
                target = self.rng.randint(2, 5)
                cells: List[Cell] = [seed]
                region_id[seed[0]][seed[1]] = next_id
                while len(cells) < target:
                    expansions = []
                    for r, c in cells:
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < n and 0 <= nc < n and region_id[nr][nc] == -1:
                                expansions.append((nr, nc))
                    if not expansions:
                        break
                    pick = self.rng.choice(sorted(set(expansions)))
                    region_id[pick[0]][pick[1]] = next_id
                    cells.append(pick)
                if len(cells) < 2:
                    ok = False
                    break
                next_id += 1
            if not ok:
                continue
            regions: Dict[int, List[Cell]] = {}
            for r in range(n):
                for c in range(n):
                    regions.setdefault(region_id[r][c], []).append((r, c))
            return [regions[k] for k in sorted(regions)]
        raise ForcedDeductionError("aquarium_completion: failed to build tank regions")

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    def _forced_chain(
        self,
        tanks: Sequence[_Tank],
        row_counts: Sequence[int],
        col_counts: Sequence[int],
        low: List[int],
        high: List[int],
        committed: Set[Cell],
    ) -> Optional[ForcedChain]:
        n = self.size
        col_need_lists = [[tank.col_needs(c) for c in range(n)] for tank in tanks]
        pending_low: List[List[str]] = [[] for _ in tanks]
        pending_high: List[List[str]] = [[] for _ in tanks]
        rounds: List[List[ForcedMove]] = []

        def tighten_line(kind: str, index: int, count: int, snap_low, snap_high) -> bool:
            changed = False
            if kind == "row":
                contribs = [
                    (tank.row_contribution(index, snap_low[t]), tank.row_contribution(index, snap_high[t]))
                    for t, tank in enumerate(tanks)
                ]
            else:
                contribs = [
                    (
                        tank.col_contribution(col_need_lists[t][index], snap_low[t]),
                        tank.col_contribution(col_need_lists[t][index], snap_high[t]),
                    )
                    for t, tank in enumerate(tanks)
                ]
            total_min = sum(pair[0] for pair in contribs)
            total_max = sum(pair[1] for pair in contribs)
            for t, tank in enumerate(tanks):
                if snap_low[t] == snap_high[t]:
                    continue
                if kind == "row" and index not in tank.need:
                    continue
                if kind == "col" and not col_need_lists[t][index]:
                    continue
                others_max = total_max - contribs[t][1]
                others_min = total_min - contribs[t][0]
                required = count - others_max
                allowed = count - others_min
                line_name = f"{'row' if kind == 'row' else 'column'} {index + 1}"

                def contribution(level: int) -> int:
                    if kind == "row":
                        return tank.row_contribution(index, level)
                    return tank.col_contribution(col_need_lists[t][index], level)

                if required > contribution(snap_low[t]):
                    new_low = snap_low[t]
                    while new_low < snap_high[t] and contribution(new_low) < required:
                        new_low += 1
                    if contribution(new_low) < required:
                        raise ForcedDeductionError(
                            "aquarium_completion: interval propagation reached a contradiction"
                        )
                    if new_low > low[t]:
                        top_row = min(r for r in tank.rows if tank.need[r] <= new_low)
                        pending_low[t].append(
                            f"{line_name.capitalize()} needs {count} water cell{'s' if count != 1 else ''} and the other tanks "
                            f"can reach at most {others_max} there, so {tank.phrase()} must hold water "
                            f"from row {top_row + 1} down"
                        )
                        low[t] = new_low
                        changed = True
                if allowed < contribution(snap_high[t]):
                    new_high = snap_high[t]
                    while new_high > snap_low[t] and contribution(new_high) > allowed:
                        new_high -= 1
                    if contribution(new_high) > allowed:
                        raise ForcedDeductionError(
                            "aquarium_completion: interval propagation reached a contradiction"
                        )
                    if new_high < high[t]:
                        dry_row = max(r for r in tank.rows if tank.need[r] > new_high)
                        if count == 0:
                            sentence = (
                                f"{line_name.capitalize()} holds no water at all, so "
                                f"{tank.phrase()} stays dry from row {dry_row + 1} up"
                            )
                        else:
                            sentence = (
                                f"{line_name.capitalize()} allows only {count} water "
                                f"cell{'s' if count != 1 else ''} and the other tanks already "
                                f"supply {others_min}, so {tank.phrase()} stays dry from row "
                                f"{dry_row + 1} up"
                            )
                        pending_high[t].append(sentence)
                        high[t] = new_high
                        changed = True
            return changed

        while True:
            # One round applies every tightening that the round-start snapshot
            # already forces (Jacobi order): commitments landed this round can
            # only unlock the next round, so rounds are true causal waves.
            snap_low, snap_high = list(low), list(high)
            changed = False
            for r in range(n):
                changed |= tighten_line("row", r, int(row_counts[r]), snap_low, snap_high)
            for c in range(n):
                changed |= tighten_line("col", c, int(col_counts[c]), snap_low, snap_high)

            round_moves: List[ForcedMove] = []
            for t, tank in enumerate(tanks):
                new_water = tank.filled_cells(low[t]) - committed
                new_dry = tank.dry_cells(high[t]) - committed
                if new_water:
                    reason = "; ".join(pending_low[t]) or (
                        f"The settling rule extends {tank.phrase()}'s confirmed water downward"
                    )
                    round_moves.append(
                        ForcedMove(kind="water", cells=tuple(sorted(new_water)), value=1, reason=reason)
                    )
                    committed |= new_water
                    pending_low[t] = []
                if new_dry:
                    reason = "; ".join(pending_high[t]) or (
                        f"No water level of {tank.phrase()} compatible with the counts reaches these "
                        "cells, so they stay dry"
                    )
                    round_moves.append(
                        ForcedMove(kind="dry", cells=tuple(sorted(new_dry)), value=0, reason=reason)
                    )
                    committed |= new_dry
                    pending_high[t] = []
            if round_moves:
                rounds.append(round_moves)
            if all(low[t] == high[t] for t in range(len(tanks))):
                return ForcedChain(rounds=rounds)
            if not changed:
                return None

    def _construct_instance(self):
        n = self.size
        for _attempt in range(200):
            regions = self._build_regions()
            tanks = [_Tank(cells) for cells in regions]
            truth = [self.rng.randint(0, tank.k) for tank in tanks]
            grid = [[0 for _ in range(n)] for _ in range(n)]
            for tank, level in zip(tanks, truth):
                for r, c in tank.filled_cells(level):
                    grid[r][c] = 1
            total = sum(sum(row) for row in grid)
            if not 5 < total < n * n - 5:
                continue
            row_counts = [sum(grid[r]) for r in range(n)]
            col_counts = [sum(grid[r][c] for r in range(n)) for c in range(n)]

            # Seed the completion framing: reveal one tank's top water band and
            # a different tank's lowest dry band, so both given glyph kinds
            # exist and each reveal pins one side of a level interval.
            water_seeds = [t for t, level in enumerate(truth) if level >= 1]
            dry_seeds = [t for t, (tank, level) in enumerate(zip(tanks, truth)) if level < tank.k]
            if not water_seeds or not dry_seeds:
                continue
            water_seed = self.rng.choice(water_seeds)
            dry_choices = [t for t in dry_seeds if t != water_seed] or dry_seeds
            dry_seed = self.rng.choice(dry_choices)

            given_water: Set[Cell] = set()
            given_dry: Set[Cell] = set()
            low = [0 for _ in tanks]
            high = [tank.k for tank in tanks]

            def reveal_water(t: int) -> None:
                low[t] = max(low[t], truth[t])
                given_water.update(tanks[t].filled_cells(truth[t]))

            def reveal_dry(t: int) -> None:
                high[t] = min(high[t], truth[t])
                given_dry.update(tanks[t].dry_cells(truth[t]))

            reveal_water(water_seed)
            reveal_dry(dry_seed)

            chain: Optional[ForcedChain] = None
            reveals = 2
            while reveals <= len(tanks) + 2:
                committed = set(given_water) | set(given_dry)
                chain = self._forced_chain(
                    tanks, row_counts, col_counts, list(low), list(high), committed
                )
                if chain is not None:
                    break
                undecided = [
                    t for t in range(len(tanks))
                    if not (low[t] >= truth[t] and high[t] <= truth[t])
                ]
                if not undecided:
                    break
                target = max(undecided, key=lambda t: (high[t] - low[t], -t))
                if low[target] < truth[target]:
                    reveal_water(target)
                else:
                    reveal_dry(target)
                reveals += 1
            if chain is None or chain.is_empty() or len(chain.rounds) < 3:
                continue
            open_cells = n * n - len(given_water) - len(given_dry)
            if open_cells < (n * n) // 2:
                continue
            try:
                reveal_budget_guard(
                    revealed=len(given_water) + len(given_dry),
                    open_before_reveals=n * n,
                    max_fraction=0.5,
                    task_name="aquarium_completion",
                )
                prove_complete_chain(chain, expected_cells=open_cells, task_name="aquarium_completion")
            except ForcedDeductionError:
                continue
            return regions, tanks, grid, row_counts, col_counts, given_water, given_dry, chain
        raise ForcedDeductionError("aquarium_completion: failed to construct a forced-hand-solvable board")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        regions: Sequence[Sequence[Cell]],
        row_counts: Sequence[int],
        col_counts: Sequence[int],
        *,
        committed_water: Set[Cell],
        committed_dry: Set[Cell],
    ) -> Image.Image:
        n = self.size
        cell, left, top = standard_board_geometry(n, n)
        scene, draw = new_board_scene()
        count_font = board_font(max(15, cell // 3))
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

        for r, c in sorted(committed_water):
            draw.rectangle(
                [left + c * cell, top + r * cell, left + (c + 1) * cell, top + (r + 1) * cell],
                fill=_WATER,
            )

        for r in range(n + 1):
            y = top + r * cell
            draw.line([(left, y), (left + n * cell, y)], fill=_GRID_LIGHT, width=1)
        for c in range(n + 1):
            x = left + c * cell
            draw.line([(x, top), (x, top + n * cell)], fill=_GRID_LIGHT, width=1)
        for r in range(n):
            for c in range(n):
                x0 = left + c * cell
                y0 = top + r * cell
                if c + 1 < n and region_id[r][c] != region_id[r][c + 1]:
                    draw.line([(x0 + cell, y0), (x0 + cell, y0 + cell)], fill=GRID_BORDER, width=3)
                if r + 1 < n and region_id[r][c] != region_id[r + 1][c]:
                    draw.line([(x0, y0 + cell), (x0 + cell, y0 + cell)], fill=GRID_BORDER, width=3)
        draw.rectangle([left, top, left + n * cell, top + n * cell], outline=GRID_BORDER, width=3)

        radius = max(4, int(cell * 0.16))
        for r, c in sorted(committed_dry):
            cx = left + c * cell + cell // 2
            cy = top + r * cell + cell // 2
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=_DRY_RING,
                width=2,
            )

        for c in range(n):
            x = left + c * cell + cell // 2
            draw.text((x, top - 13), str(col_counts[c]), fill=TEXT, font=count_font, anchor="mm")
            draw.text((x, top + n * cell + 13), str(c + 1), fill=AXIS_LABEL, font=axis_font, anchor="mm")
        for r in range(n):
            y = top + r * cell + cell // 2
            draw.text((left - 13, y), str(row_counts[r]), fill=TEXT, font=count_font, anchor="mm")
            draw.text((left + n * cell + 13, y), str(r + 1), fill=AXIS_LABEL, font=axis_font, anchor="mm")
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
                "Generate a solution video that starts from the unchanged puzzle. Cells are then decided "
                "in deduction order: each wave of newly determined cells fades in together, blue fills "
                "for water and small gray rings for dry cells, while earlier commitments stay unchanged, "
                f"and the video ends holding {_TERMINAL_VISUAL}."
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
        tanks: Sequence[_Tank],
        given_water: Set[Cell],
        given_dry: Set[Cell],
    ) -> List[str]:
        opener = (
            f"The board is {self.size} by {self.size} with {len(tanks)} tanks; I will name each tank by "
            f"its topmost, leftmost cell. {len(given_water)} cells already hold water and "
            f"{len(given_dry)} are already marked dry, and every count must be met exactly."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            paragraphs.append(group_paragraph(group, opener=opener if index == 0 else ""))
        closing = (
            " Every cell is now decided, each tank's water rests as one settled block, and every row and "
            "column matches its count, so the board is complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    # ------------------------------------------------------------------
    # Record assembly
    # ------------------------------------------------------------------

    def create_random_puzzle(self) -> AquariumCompletionRecord:
        (
            regions,
            tanks,
            grid,
            row_counts,
            col_counts,
            given_water,
            given_dry,
            chain,
        ) = self._construct_instance()
        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        water: Set[Cell] = set(given_water)
        dry: Set[Cell] = set(given_dry)
        states: List[Image.Image] = [
            fit_canvas(
                self._render(regions, row_counts, col_counts, committed_water=water, committed_dry=dry)
            )
        ]
        for group in groups:
            for move in group:
                if move.kind == "water":
                    water |= set(move.cells)
                else:
                    dry |= set(move.cells)
            states.append(
                fit_canvas(
                    self._render(
                        regions, row_counts, col_counts,
                        committed_water=set(water), committed_dry=set(dry),
                    )
                )
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

        answer_grid = [[int(v) for v in row] for row in grid]

        return AquariumCompletionRecord(
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
                "regions": [[list(c) for c in cells] for cells in regions],
                "row_counts": row_counts,
                "col_counts": col_counts,
                "given_filled": [list(cell) for cell in sorted(given_water)],
                "given_empty": [list(cell) for cell in sorted(given_dry)],
                "fill": answer_grid,
            },
        )
