"""Shikaku completion: extend a partially drawn rectangle partition to cover the grid. Every emitted puzzle is forced-hand solvable: starting from the pre-drawn
rectangles, repeated application of two visible-rule deductions (a clue with a
single legal footprint, and an uncovered cell reachable by only one clue)
commits rectangles until the partition is complete. The generator proves the
chain per sample by replaying its own deduction engine; the chain also drives
the authored CoT and the TI2V commit order."""

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
    TEXT,
    board_font,
    draw_axis_labels,
    draw_board_grid,
    finish_board_scene,
    new_board_scene,
    standard_board_geometry,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.contracts.grid_completion import answer_xy_composites
from data.common.pipeline.grid_rendering import fit_canvas
from data.common.prompts import build_sectioned_prompt

Rect = Tuple[int, int, int, int]
Cell = Tuple[int, int]

_INPUT_DESCRIPTION = (
    "The image shows a square grid with 1-based row and column numbers along the top and left edges. "
    "White circles hold the area clues, and bold dark outlines mark rectangles that are already committed."
)
_RULE_DESCRIPTION = (
    "Divide the whole grid into rectangles that do not overlap. Every rectangle must contain exactly one "
    "circled clue, and its cell count must equal that clue. The rectangles that are already outlined are "
    "fixed and keep their exact position."
)
_TERMINAL_VISUAL = (
    "the completed board on which every rectangle of the partition is outlined in the same bold dark "
    "style, while the grid, clue circles, clue numbers, and edge numbers stay unchanged"
)
@dataclass
class ShikakuCompletionRecord:
    id: str
    task_name: str = "shikaku_completion"
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


def _rect_cells(rect: Rect) -> Set[Cell]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _rect_phrase(rect: Rect) -> str:
    r0, c0, r1, c1 = rect
    if r0 == r1 and c0 == c1:
        return f"the single cell at {display_cell_text((r0, c0))}"
    return (
        f"the rectangle from {display_cell_text((r0, c0))} through "
        f"{display_cell_text((r1, c1))}"
    )


class ShikakuCompletionGenerator(AbstractPuzzleGenerator[ShikakuCompletionRecord]):
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

    def create_puzzle(self, *args, **kwargs) -> ShikakuCompletionRecord:
        return self.create_random_puzzle()

    def _next_id(self) -> str:
        puzzle_id = f"shikaku_completion-{self._counter:05d}"
        self._counter += 1
        return puzzle_id

    # ------------------------------------------------------------------
    # Solution partition
    # ------------------------------------------------------------------

    def _partition(self, r0: int, c0: int, r1: int, c1: int) -> List[Rect]:
        """Recursively partition the rectangle [r0..r1] x [c0..c1]."""
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = h * w
        if area <= 2:
            return [(r0, c0, r1, c1)]
        if area <= 4 and self.rng.random() < 0.55:
            return [(r0, c0, r1, c1)]
        if h == 1 and w <= 4 and self.rng.random() < 0.5:
            return [(r0, c0, r1, c1)]
        if w == 1 and h <= 4 and self.rng.random() < 0.5:
            return [(r0, c0, r1, c1)]
        candidates = []
        if h >= 2:
            candidates.append("h")
        if w >= 2:
            candidates.append("v")
        direction = self.rng.choice(candidates)
        if direction == "h":
            split = self.rng.randint(r0, r1 - 1)
            return self._partition(r0, c0, split, c1) + self._partition(split + 1, c0, r1, c1)
        split = self.rng.randint(c0, c1 - 1)
        return self._partition(r0, c0, r1, split) + self._partition(r0, split + 1, r1, c1)

    # ------------------------------------------------------------------
    # Forced-deduction engine
    # ------------------------------------------------------------------

    def _clue_options(
        self,
        clue: Cell,
        area: int,
        other_clues: Sequence[Cell],
    ) -> List[Rect]:
        r, c = clue
        options: List[Rect] = []
        for h in range(1, self.rows + 1):
            if area % h:
                continue
            w = area // h
            if not 1 <= w <= self.cols:
                continue
            for r0 in range(max(0, r - h + 1), min(self.rows - h, r) + 1):
                for c0 in range(max(0, c - w + 1), min(self.cols - w, c) + 1):
                    rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                    if any(rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3] for rr, cc in other_clues):
                        continue
                    options.append(rect)
        return options

    def _forced_chain(
        self,
        clue_positions: Sequence[Cell],
        clue_areas: Sequence[int],
        solution_rects: Sequence[Rect],
        given_indices: Set[int],
    ) -> Optional[ForcedChain]:
        """Replay forced deduction from the given rectangles; None on stall."""

        count = len(clue_positions)
        covered: Set[Cell] = set()
        for index in given_indices:
            covered |= _rect_cells(solution_rects[index])
        options: List[Optional[List[Rect]]] = []
        for index in range(count):
            if index in given_indices:
                options.append(None)
                continue
            other = [pos for j, pos in enumerate(clue_positions) if j != index]
            options.append(self._clue_options(clue_positions[index], clue_areas[index], other))

        assigned: Set[int] = set(given_indices)
        rounds: List[List[ForcedMove]] = []
        while len(assigned) < count:
            for index in range(count):
                if index in assigned or options[index] is None:
                    continue
                options[index] = [
                    rect for rect in options[index] if not (_rect_cells(rect) & covered)
                ]

            forced: List[Tuple[int, Rect, str]] = []
            forced_indices: Set[int] = set()
            for index in range(count):
                if index in assigned or not options[index]:
                    continue
                if len(options[index]) == 1:
                    rect = options[index][0]
                    r, c = clue_positions[index]
                    reason = (
                        f"The {clue_areas[index]} at {display_cell_text((r, c))} has a single legal "
                        f"footprint left: {_rect_phrase(rect)}"
                    )
                    forced.append((index, rect, reason))
                    forced_indices.add(index)

            if not forced:
                cover_map: Dict[Cell, List[Tuple[int, Rect]]] = {}
                for index in range(count):
                    if index in assigned or not options[index]:
                        continue
                    for rect in options[index]:
                        for cell in _rect_cells(rect):
                            if cell not in covered:
                                cover_map.setdefault(cell, []).append((index, rect))
                for cell in sorted(cover_map):
                    pairs = cover_map[cell]
                    owners = {index for index, _ in pairs}
                    if len(owners) != 1:
                        continue
                    index = next(iter(owners))
                    if index in forced_indices:
                        continue
                    pinned = [rect for owner, rect in pairs if owner == index]
                    if len(set(pinned)) != 1:
                        options[index] = sorted(set(pinned))
                        continue
                    rect = pinned[0]
                    r, c = clue_positions[index]
                    reason = (
                        f"{display_cell_text(cell)} can only be covered by the "
                        f"{clue_areas[index]} at {display_cell_text((r, c))}, which pins that rectangle "
                        f"to {_rect_phrase(rect)}"
                    )
                    forced.append((index, rect, reason))
                    forced_indices.add(index)

            if not forced:
                return None

            round_moves: List[ForcedMove] = []
            for index, rect, reason in forced:
                if rect != solution_rects[index]:
                    raise ForcedDeductionError(
                        "shikaku_completion: forced rectangle deviates from the constructed partition"
                    )
                round_moves.append(
                    ForcedMove(
                        kind="rectangle",
                        cells=tuple(sorted(_rect_cells(rect))),
                        value=rect,
                        reason=reason,
                    )
                )
                assigned.add(index)
                covered |= _rect_cells(rect)
            rounds.append(round_moves)
        return ForcedChain(rounds=rounds)

    def _construct_instance(self) -> Tuple[List[Rect], List[Cell], Set[int], ForcedChain]:
        for _ in range(120):
            rectangles = self._partition(0, 0, self.rows - 1, self.cols - 1)
            # Keep the original clue-density ceiling when the explicitly
            # configured board grows; a fixed 14-clue cap rejects nearly all
            # larger partitions before the unchanged forced-chain proof.
            maximum_rectangles = max(14, 14 * self.rows * self.cols // 36)
            if not 6 <= len(rectangles) <= maximum_rectangles:
                continue
            clue_positions = [
                self.rng.choice(sorted(_rect_cells(rect))) for rect in rectangles
            ]
            clue_areas = [
                (r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rectangles
            ]
            if max(clue_areas) < 3:
                continue

            # Seed the completion framing with one pre-drawn rectangle: the
            # clue whose footprint is currently the most ambiguous.
            ambiguity = []
            for index in range(len(rectangles)):
                other = [pos for j, pos in enumerate(clue_positions) if j != index]
                ambiguity.append(len(self._clue_options(clue_positions[index], clue_areas[index], other)))
            given_indices: Set[int] = {max(range(len(rectangles)), key=lambda i: (ambiguity[i], i))}

            chain = self._forced_chain(clue_positions, clue_areas, rectangles, given_indices)
            while chain is None and len(given_indices) < len(rectangles) - 4:
                remaining = [i for i in range(len(rectangles)) if i not in given_indices]
                reveal = max(remaining, key=lambda i: (ambiguity[i], i))
                given_indices.add(reveal)
                chain = self._forced_chain(clue_positions, clue_areas, rectangles, given_indices)
            if chain is None:
                continue
            if len(rectangles) - len(given_indices) < 5 or len(chain.rounds) < 2:
                continue
            if len(given_indices) > len(rectangles) // 2:
                continue
            open_cells = self.rows * self.cols - sum(
                len(_rect_cells(rectangles[i])) for i in given_indices
            )
            prove_complete_chain(chain, expected_cells=open_cells, task_name="shikaku_completion")
            return rectangles, clue_positions, given_indices, chain
        raise ForcedDeductionError(
            "shikaku_completion: failed to construct a forced-hand-solvable partition"
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        rectangles: Sequence[Rect],
        clue_positions: Sequence[Cell],
        *,
        committed: Sequence[Rect],
    ) -> Image.Image:
        cell, left, top = standard_board_geometry(self.rows, self.cols)
        scene, draw = new_board_scene()
        clue_font = board_font(max(18, cell * 2 // 5))
        axis_font = board_font(max(11, cell // 4))

        draw_board_grid(draw, self.rows, self.cols, cell=cell, left=left, top=top)
        draw_axis_labels(
            draw, self.rows, self.cols, cell=cell, left=left, top=top, font=axis_font
        )

        for r0, c0, r1, c1 in committed:
            draw.rectangle(
                [left + c0 * cell, top + r0 * cell, left + (c1 + 1) * cell, top + (r1 + 1) * cell],
                outline=GRID_BORDER,
                width=4,
            )

        clue_areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rectangles]
        for (r, c), area in zip(clue_positions, clue_areas):
            cx = left + c * cell + cell // 2
            cy = top + r * cell + cell // 2
            radius = max(8, cell // 3)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=(255, 255, 255),
                outline=GRID_BORDER,
                width=2,
            )
            draw.text((cx, cy), str(area), fill=TEXT, font=clue_font, anchor="mm")
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
                "Generate a solution video that starts from the unchanged puzzle. New rectangle outlines "
                "then appear in deduction order: each wave of rectangles that the clues pin down fades in "
                "as complete bold outlines together, earlier outlines stay exactly in place, and the video "
                f"ends holding {_TERMINAL_VISUAL}."
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
        given_rects: Sequence[Rect],
        clue_count: int,
    ) -> List[str]:
        given_phrase = "; ".join(_rect_phrase(rect) for rect in given_rects)
        opener = (
            f"The board is {self.rows} by {self.cols} with {clue_count} circled area clues. "
            f"Already outlined: {given_phrase}. I will extend the partition one forced rectangle at a time."
        )
        paragraphs: List[str] = []
        for index, group in enumerate(groups):
            if index == 0:
                paragraphs.append(group_paragraph(group, opener=opener))
            else:
                paragraphs.append(group_paragraph(group))
        closing = (
            " Now every cell lies inside exactly one rectangle and each rectangle's cell count matches "
            "its circled clue, so the partition is complete."
        )
        paragraphs[-1] = paragraphs[-1] + closing
        return paragraphs

    # ------------------------------------------------------------------
    # Record assembly
    # ------------------------------------------------------------------

    def create_random_puzzle(self) -> ShikakuCompletionRecord:
        rectangles, clue_positions, given_indices, chain = self._construct_instance()
        order = sorted(
            range(len(clue_positions)),
            key=lambda i: (clue_positions[i][0], clue_positions[i][1]),
        )
        ordered_rects = [rectangles[i] for i in order]
        ordered_clues = [clue_positions[i] for i in order]
        given_rects = [rectangles[i] for i in sorted(given_indices)]

        prompts = self._build_prompts()
        puzzle_id = self._next_id()

        groups = merge_rounds(chain.rounds)
        committed: List[Rect] = list(given_rects)
        states: List[Image.Image] = [
            fit_canvas(self._render(ordered_rects, ordered_clues, committed=committed))
        ]
        for group in groups:
            committed = committed + [move.value for move in group]
            states.append(
                fit_canvas(self._render(ordered_rects, ordered_clues, committed=committed))
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

        rect_payload = [[int(bound) for bound in rect] for rect in ordered_rects]
        visible_rects = [[r0 + 1, c0 + 1, r1 + 1, c1 + 1] for r0, c0, r1, c1 in rect_payload]
        visible_composites = [
            ((r0, c0), (r1, c1))
            for r0, c0, r1, c1 in visible_rects
        ]

        return ShikakuCompletionRecord(
            answer=answer_xy_composites(visible_composites),
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
                "clue_positions": [list(p) for p in ordered_clues],
                "clues": [
                    [
                        int(clue[0]),
                        int(clue[1]),
                        int((rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)),
                    ]
                    for clue, rect in zip(ordered_clues, ordered_rects)
                ],
                "given_rectangles": [[int(bound) for bound in rect] for rect in given_rects],
                "rectangles": rect_payload,
            },
        )
