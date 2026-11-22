from __future__ import annotations

import random
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.next_state_prediction.core.rendering import (
    discrete_video_frames,
    finish_state_canvas,
    new_state_canvas,
    scaled_font,
)
from data.common.typed_answers import format_matrix_answer


pattern_instructions = {
    "grid_bfs": (
        "The input image shows a grid of 0, 1, and X cells. Compute the shortest 4-directional distance from every "
        "cell to any 1 cell, using -1 for X cells and unreachable cells."
    )
}

_INPUT_DESC = (
    "The input image shows one rectangular grid of open, source, and wall cells."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the complete distance grid in the same fixed layout, with every value printed and walls visually "
    "distinct from unreachable open cells."
)
_TI2V_OUTPUT = (
    "Generate a video that replays the complete ordered distance-wave process. Commit each distance layer as one "
    "complete state: layer 0 shows every source as an amber 0 together, and each later layer settles the previous "
    "frontier to blue while all newly reached cells appear as the amber frontier together. Finish by committing the "
    "complete final grid, where every reachable cell is settled blue, unreachable open cells are red, and walls show "
    "-1 on their dark cells."
)

_CANVAS = (512, 512)
_BG = (246, 248, 249)
_CARD = (255, 255, 253)
_INK = (35, 41, 47)
_MUTED = (88, 96, 104)
_OPEN = (249, 250, 247)
_TARGET = (61, 160, 124)
_WALL = (64, 69, 76)
_UNDISCOVERED = (218, 224, 229)
_FRONTIER = (241, 164, 55)
_SETTLED = (83, 137, 203)
_UNREACHABLE = (194, 91, 98)


def _distance_to_targets(grid: List[List[str]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    distances = [[-1 for _ in range(cols)] for _ in range(rows)]
    queue: deque[Tuple[int, int]] = deque()
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == "1":
                distances[row][col] = 0
                queue.append((row, col))
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = row + dr
            nc = col + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "X" and distances[nr][nc] == -1:
                distances[nr][nc] = distances[row][col] + 1
                queue.append((nr, nc))
    return distances


def _obstacle_ablation_changed_cells(grid: List[List[str]]) -> List[Tuple[int, int]]:
    obstacle_distances = _distance_to_targets(grid)
    open_grid = [["0" if cell == "X" else cell for cell in row] for row in grid]
    open_distances = _distance_to_targets(open_grid)
    return [
        (row, col)
        for row in range(len(grid))
        for col in range(len(grid[0]))
        if grid[row][col] != "X" and obstacle_distances[row][col] != open_distances[row][col]
    ]


def _obstacles_change_distances(grid: List[List[str]]) -> bool:
    return bool(_obstacle_ablation_changed_cells(grid))


def _bfs_visual_states(grid: List[List[str]], distances: List[List[int]]) -> List[tuple[str, int | None, List[List[str]]]]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    input_state = [
        ["wall" if grid[row][col] == "X" else "target" if grid[row][col] == "1" else "open" for col in range(cols)]
        for row in range(rows)
    ]
    states: List[tuple[str, int | None, List[List[str]]]] = [("input", None, input_state)]
    max_distance = max((value for row in distances for value in row), default=0)
    # Every discovered distance layer has its own amber frontier, including
    # the farthest one. The final blue grid is a separate settling commit.
    for layer in range(max_distance + 1):
        state: List[List[str]] = []
        for row in range(rows):
            state_row: List[str] = []
            for col in range(cols):
                if grid[row][col] == "X":
                    state_row.append("wall")
                elif distances[row][col] == layer:
                    state_row.append("frontier")
                elif 0 <= distances[row][col] < layer:
                    state_row.append("settled")
                else:
                    state_row.append("undiscovered")
            state.append(state_row)
        states.append(("layer", layer, state))
    final_state = [
        [
            # The terminal renderer changes a wall's visible label from X to
            # the required answer value -1.  Keep that appearance replacement
            # explicit so the video locally fades the complete wall card too
            # instead of hard-cutting it at the final commit.
            "wall_final"
            if grid[row][col] == "X"
            else "unreachable"
            if distances[row][col] < 0
            else "settled"
            for col in range(cols)
        ]
        for row in range(rows)
    ]
    states.append(("final", None, final_state))
    return states


class GridBFSPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "grid_bfs"
    task_family = "next_state_prediction"
    task_mechanic = "obstacle_aware_multi_source_bfs"

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        size_range: Tuple[int, int] = (4, 6),
        distance_range: Tuple[int, int] = (3, 6),
        source_count: int | None = None,
        video: bool = True,
        video_fps: int = 16,
    ) -> None:
        if len(size_range) != 2:
            raise ValueError("grid_bfs size_range must contain exactly two bounds")
        minimum_size, maximum_size = (int(size_range[0]), int(size_range[1]))
        if not (4 <= minimum_size <= maximum_size <= 8):
            raise ValueError("grid_bfs size_range must stay within the 4..8 cell range")
        if int(video_fps) != 16:
            raise ValueError("grid_bfs videos must use 16 fps")
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_grid_bfs"
        super().__init__(resolved)
        self.rng = random.Random(seed)
        self.seed = seed
        self.size_range = (minimum_size, maximum_size)
        self.distance_range = tuple(map(int, distance_range))
        if len(self.distance_range) != 2 or not (
            3 <= self.distance_range[0] <= self.distance_range[1] <= 12
        ):
            raise ValueError("grid_bfs distance_range must stay within 3..12")
        if source_count is not None and not (2 <= source_count < minimum_size ** 2):
            raise ValueError("source_count must be at least two and leave room for open cells")
        self.source_count = source_count
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _sample_grid(self) -> List[List[str]]:
        rows = self.rng.randint(*self.size_range)
        cols = self.rng.randint(*self.size_range)
        for _attempt in range(512):
            grid = [
                [self.rng.choices(["0", "1", "X"], weights=(8, 1, 3), k=1)[0] for _ in range(cols)]
                for _ in range(rows)
            ]
            if not any(cell == "X" for row in grid for cell in row):
                wall_candidates = [
                    (row, col)
                    for row in range(rows)
                    for col in range(cols)
                    if grid[row][col] == "0"
                ]
                if not wall_candidates:
                    continue
                wall_row, wall_col = self.rng.choice(wall_candidates)
                grid[wall_row][wall_col] = "X"

            target_count = sum(cell == "1" for row in grid for cell in row)
            if self.source_count is not None and target_count != self.source_count:
                continue
            if target_count < 2:
                target_candidates = [
                    (row, col)
                    for row in range(rows)
                    for col in range(cols)
                    if grid[row][col] == "0"
                ]
                self.rng.shuffle(target_candidates)
                if len(target_candidates) < 2 - target_count:
                    continue
                for row, col in target_candidates[: 2 - target_count]:
                    grid[row][col] = "1"

            distances = _distance_to_targets(grid)
            max_distance = max((value for row in distances for value in row), default=0)
            if self.distance_range[0] <= max_distance <= self.distance_range[1] and _obstacles_change_distances(grid):
                return grid

        if self.distance_range != (3, 6) or self.source_count is not None:
            raise RuntimeError("grid_bfs could not sample the requested distance/source constraints")
        grid = [["0" for _ in range(cols)] for _ in range(rows)]
        grid[0][0] = "1"
        grid[rows - 1][cols - 1] = "1"
        grid[rows // 2][cols - 1] = "1"
        for row in range(rows - 1):
            grid[row][1] = "X"
        fallback_max_distance = max((value for row in _distance_to_targets(grid) for value in row), default=0)
        if not (3 <= fallback_max_distance <= 6 and _obstacles_change_distances(grid)):
            raise RuntimeError("grid_bfs deterministic fallback violated its calibrated difficulty contract")
        return grid

    @staticmethod
    def _grid_geometry(rows: int, cols: int) -> tuple[int, int, int, int, int]:
        cell_size = min(58, 300 // max(1, rows, cols))
        grid_width = cell_size * cols
        grid_height = cell_size * rows
        left = (512 - grid_width) // 2
        top = 112 + (300 - grid_height) // 2
        return left, top, cell_size, grid_width, grid_height

    @classmethod
    def _cell_center(cls,rows,cols,row,col):
        left,top,cell,_width,_height=cls._grid_geometry(rows,cols)
        from data.common.prompt_paraphrase import active_profile
        if active_profile("distance_beacons"):
            step=cell*.65
            return (round(256+(col-row-(cols-rows)/2)*step),round(256+(col+row-(rows+cols-2)/2)*step))
        return left+(col+.5)*cell,top+(row+.5)*cell

    def _render_state(
        self,
        grid: List[List[str]],
        distances: List[List[int]],
        *,
        phase: str,
        layer: int | None = None,
    ) -> Image.Image:
        image, draw = new_state_canvas(_BG, _CARD, (217, 222, 225))
        value_font = scaled_font(18)

        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        left, top, cell_size, grid_width, grid_height = self._grid_geometry(rows, cols)
        from data.common.prompt_paraphrase import active_profile
        beacons=active_profile("distance_beacons")
        if beacons:
            for row in range(rows):
                for col in range(cols):
                    if grid[row][col]=="X":continue
                    cx,cy=self._cell_center(rows,cols,row,col)
                    for rr,cc in ((row+1,col),(row,col+1)):
                        if rr<rows and cc<cols and grid[rr][cc]!="X":
                            xx,yy=self._cell_center(rows,cols,rr,cc)
                            draw.line((cx,cy,xx,yy),fill=(110,115,91),width=5)
        for row in range(rows):
            for col in range(cols):
                token = grid[row][col]
                distance = int(distances[row][col])
                if phase == "input":
                    # The input encodes each cell's category in its colour
                    # alone; the distance layers then print values.
                    category = "wall" if token == "X" else "target" if token == "1" else "open"
                    label = ""
                elif phase == "layer":
                    if token == "X":
                        category, label = "wall", ""
                    elif distance == layer:
                        category, label = "frontier", str(distance)
                    elif 0 <= distance < int(layer or 0):
                        category, label = "settled", str(distance)
                    else:
                        category, label = "undiscovered", ""
                else:
                    if token == "X":
                        category, label = "wall", "-1"
                    elif distance < 0:
                        category, label = "unreachable", "-1"
                    else:
                        category, label = "settled", str(distance)
                palette = {
                    "open": _OPEN,
                    "target": _TARGET,
                    "wall": _WALL,
                    "undiscovered": _UNDISCOVERED,
                    "frontier": _FRONTIER,
                    "settled": _SETTLED,
                    "unreachable": _UNREACHABLE,
                }
                x0 = left + col * cell_size
                y0 = top + row * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                outline = (174, 116, 25) if category == "frontier" else (194, 201, 205)
                if beacons:
                    cx,cy=self._cell_center(rows,cols,row,col)
                    half=cell_size*.3
                    x0,y0,x1,y1=cx-half,cy-half,cx+half,cy+half
                    if category=="wall":
                        draw.rectangle((x0+2,y0+2,x1-2,y1-2),fill=palette[category],outline=outline,width=2)
                        for yy in (y0+cell_size*.18,y1-cell_size*.18):
                            draw.line((x0+3,yy,x1-3,yy),fill=(145,151,159),width=2)
                        draw.line((cx,y0+3,cx,y0+cell_size*.18),fill=(145,151,159),width=2)
                    else:
                        radius=cell_size*.3
                        draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),fill=palette[category],outline=outline,width=3 if category=="frontier" else 2)
                        if category=="target":
                            draw.arc((x0+1,y0+1,x1-1,y1-1),210,330,fill=_TARGET,width=3)
                            draw.arc((x0+1,y0+1,x1-1,y1-1),30,150,fill=_TARGET,width=3)
                else:
                    draw.rounded_rectangle(
                        (x0 + 2, y0 + 2, x1 - 2, y1 - 2),
                        radius=8,
                        fill=palette[category],
                        outline=outline,
                        width=3 if category == "frontier" else 2,
                    )
                if label:
                    if beacons and category=="wall":
                        # Keep the complete numeric glyph clear of the wall's
                        # brick joints; those joints encode no extra state.
                        cx,cy=(x0+x1)/2,(y0+y1)/2
                        draw.rectangle((cx-14,cy-11,cx+14,cy+12),fill=palette[category])
                    dark_text = category in {"open", "undiscovered", "frontier"}
                    draw.text(
                        ((x0 + x1) / 2, (y0 + y1) / 2 + 1),
                        label,
                        font=value_font,
                        fill=_INK if dark_text else (255, 255, 255),
                        anchor="mm",
                    )
        if not beacons:
            draw.rounded_rectangle(
                (left-5,top-5,left+grid_width+5,top+grid_height+5),radius=12,outline=(112,121,128),width=2,
            )

        return finish_state_canvas(image)

    def _contract(self) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=_INPUT_DESC,
            symbol_legend=(
                "A pale cell = open cell whose distance must be computed.",
                "A green cell = target/source cell with distance 0.",
                "A dark cell = blocked wall that cannot be entered.",
            ),
            rules=(
                "All green target cells are simultaneous distance sources at distance 0.",
                "Movement is allowed only one cell up, down, left, or right; diagonal movement is forbidden.",
                "A path may never enter a dark wall cell.",
                "Each open cell receives the minimum number of legal steps to any target.",
                "Every wall and every unreachable open cell has output value -1.",
            ),
            modality_outputs={
                "ti2v": _TI2V_OUTPUT,
                "solution_image": _SOLUTION_IMAGE_OUTPUT,
            },
            preservation_constraints=(
                "Keep the grid geometry, cell positions, target locations, and wall locations unchanged.",
                "Keep walls visually distinct from unreachable open cells even though both output -1.",
                "Never change a revealed distance value after its cell settles.",
            ),
            transition_constraints=(
                "Start with the supplied input grid, then commit Layer 0 with every target in the amber frontier.",
                "Use amber only for the current frontier, blue for settled distances, gray for not-yet-reached open cells, and red for unreachable open cells in the final state.",
                "Commit each next layer as one complete state: the entire previous frontier settles to blue and every newly reached open cell appears in amber together.",
            ),
            ti2v_continuity={
                "timing": "discrete",
                "moving_nouns": "the distance layers and their printed values",
                "static_nouns": "the walls, the grid frame, the background, and the camera",
                "hold_noun": "the exact final distance grid",
            },
            answer=AnswerContract(
                kind="grid",
                format_template="Answer: [[<integer>, ...], ...].",
                parser_profile="json_integer_grid",
                verifier_profile="grid_bfs_multisource_replay",
                format_example="Answer: [[98, 99, ...], [100, 101, ...], ...].",
                accepts_multiple_valid_answers=False,
            ),
            render_profile="grid_bfs_frontier_wave_v3",
        )

    def _prompts(self) -> tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract()
        prompts = {f"{mode}_prompt": contract.build_prompt(mode) for mode in contract.prompt_modes()}
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid grid-bfs prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def _render_video(
        self,
        grid: List[List[str]],
        distances: List[List[int]],
        states: List[tuple[str, int | None, List[List[str]]]],
        *,
        output_path: Path,
    ) -> int:
        renders = [
            self._render_state(grid, distances, phase=phase, layer=layer)
            for phase, layer, _state in states
        ]
        frames = [np.asarray(image, dtype=np.uint8) for image in discrete_video_frames(renders)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)

    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1

        grid = self._sample_grid()
        distances = _distance_to_targets(grid)
        states = _bfs_visual_states(grid, distances)
        contract, prompts = self._prompts()

        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        video_dir = sample_dir / "videos"
        for directory in (puzzle_dir, solution_dir, video_dir):
            directory.mkdir(parents=True, exist_ok=True)

        puzzle_image = self._render_state(grid, distances, phase="input")
        solution_image = self._render_state(grid, distances, phase="final")
        puzzle_path = puzzle_dir / f"{sample_id}__puzzle.png"
        solution_path = solution_dir / f"{sample_id}__solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        solution_video_path = None
        video_num_frames = None
        if self.video:
            video_path = video_dir / f"{sample_id}__solution.webm"
            video_num_frames = self._render_video(grid, distances, states, output_path=video_path)
            if video_num_frames > 0:
                solution_video_path = video_path

        answer_text = format_matrix_answer(distances)
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        left, top, cell_size, grid_width, grid_height = self._grid_geometry(rows, cols)
        max_distance = max((value for row in distances for value in row), default=0)
        ablation_changed_cells = _obstacle_ablation_changed_cells(grid)
        source_count = sum(cell == "1" for row in grid for cell in row)
        wall_count = sum(cell == "X" for row in grid for cell in row)
        unreachable_count = sum(
            grid[row][col] != "X" and distances[row][col] < 0
            for row in range(rows)
            for col in range(cols)
        )
        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_mechanic": self.task_mechanic,
            "task_contract": contract.to_dict(),
            "question": (
                "Compute the minimum 4-directional distance from every cell to any target 1, using -1 for walls "
                "and unreachable open cells."
            ),
            "answer": answer_text,
            "structured_answer": {
                "input_grid": grid,
                "distance_grid": distances,
                "max_distance": max_distance,
                "grid_bbox": [left, top, left + grid_width, top + grid_height],
                "cell_size": cell_size,
                "difficulty_variant": "obstacle_aware",
                "task_mechanic": self.task_mechanic,
                "source_count": source_count,
                "wall_count": wall_count,
                "unreachable_count": unreachable_count,
                "obstacle_ablation_changed_cells": [list(cell) for cell in ablation_changed_cells],
                "wave_layer_count": max_distance + 1,
            },
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path.as_posix() if solution_video_path is not None else None,
            "video_fps": self.video_fps if video_num_frames else None,
            "video_num_frames": video_num_frames,
        }

    def make_sample(self) -> tuple[Dict[str, Any], Image.Image, Image.Image]:
        record = self.create_puzzle()
        puzzle = Image.open(record["image"]).convert("RGB").copy()
        solution = Image.open(record["solution_image_path"]).convert("RGB").copy()
        return record, puzzle, solution
