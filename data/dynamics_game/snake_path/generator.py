from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.animation_policy import (
    DISCRETE_STATE_HOLD_FRAMES,
    frame_budget_for_discrete_state_count,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    format_xy_path_answer,
)
from data.common.prompts import build_sectioned_prompt
from data.common.video_encoding import encode_rgb_frames_to_webm, fit_frames_to_video_budget
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.dynamics_game.core.coordinate_format import (
    display_cell_text,
    display_cells_text,
    display_path_text,
)
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="discrete",
    moving_nouns="the snake and every object the rules change",
    static_nouns="everything the rules leave unchanged, including the board, labels, background, and camera",
    hold_noun="the final board state",
)


Cell = Tuple[int, int]
HEAD_FILL = (255, 230, 70)
HEAD_OUTLINE = (160, 120, 0)
BODY_FILL = (72, 149, 239)
BODY_OUTLINE = (20, 80, 180)
FOOD_FILL = (230, 57, 70)
FOOD_OUTLINE = (160, 0, 20)

_INPUT_DESC = (
    "The input image shows a square coordinate grid with 1-based row and column labels. The yellow "
    "rounded cell is the snake head, the ordered connected blue cells and blue connector rods form "
    "its body from neck to tail, and every red circular pellet is food that must be eaten. "
    f"{GRID_XY_CONVENTION_SENTENCE}"
)
_RULE_SUMMARY = (
    "On each move the head advances exactly one cell up, right, down, or left. On an ordinary move, "
    "every body segment takes the previous cell of the segment before it and the old tail cell becomes "
    "empty floor. When the head enters a food cell, that pellet disappears at the same state boundary "
    "and the old tail does not move, so the snake grows by exactly one segment. A move is illegal if "
    "the head leaves the board or enters an occupied snake cell; entering the old tail is allowed only "
    "on an ordinary move when that tail is released. Eat all visible pellets in any order and stop as "
    "soon as none remain."
)
_TI2V_OUTPUT = (
    "Generate a video beginning with the exact input state. Hold each complete grid state for the same "
    "duration, then switch directly to the next legal state. At every food move, remove only the pellet under "
    "the new head and keep the old tail as the new final body segment; on every other move, release the old "
    "tail cell. End with a stable state containing no food. Keep segment order, identities, colors, grid, "
    "labels, and background fixed."
)
_DIR_TO_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


@dataclass(frozen=True)
class SnakeFrame:
    snake: Tuple[Cell, ...]
    foods: FrozenSet[Cell]


def board_geometry(width: int, height: int) -> Tuple[int, int, int]:
    """Return cell size and centered board offsets for the fixed canvas."""

    side_cells = max(int(width), int(height))
    cell = max(20, int(512 / (side_cells + 1)) // 2 * 2)
    return cell, (512 - int(width) * cell) // 2, (512 - int(height) * cell) // 2


def render_snake_board(
    width: int,
    height: int,
    snake: Sequence[Cell],
    foods: Sequence[Cell] | FrozenSet[Cell],
) -> Image.Image:
    """Render one complete semantic state from snake order and remaining food."""

    cell, margin_x, margin_y = board_geometry(width, height)
    image, draw = new_scene((255, 255, 255))
    font = scene_font(max(16, cell // 3))

    for row in range(height):
        for col in range(width):
            x0 = margin_x + col * cell
            y0 = margin_y + row * cell
            draw.rectangle(
                [x0, y0, x0 + cell, y0 + cell],
                fill=(255, 255, 255),
                outline=(180, 180, 180),
                width=1,
            )
    for row in range(height + 1):
        y = margin_y + row * cell
        draw.line([(margin_x, y), (margin_x + width * cell, y)], fill=(0, 0, 0), width=1)
    for col in range(width + 1):
        x = margin_x + col * cell
        draw.line([(x, margin_y), (x, margin_y + height * cell)], fill=(0, 0, 0), width=1)

    for row in range(height):
        draw.text(
            (margin_x - cell * 0.4, margin_y + row * cell + cell * 0.5),
            str(row + 1),
            fill=(0, 0, 0),
            font=font,
            anchor="mm",
        )
    for col in range(width):
        draw.text(
            (margin_x + col * cell + cell * 0.5, margin_y - cell * 0.4),
            str(col + 1),
            fill=(0, 0, 0),
            font=font,
            anchor="mm",
        )

    for row, col in sorted(foods):
        cx = margin_x + col * cell + cell / 2
        cy = margin_y + row * cell + cell / 2
        radius = max(8, int(cell * 0.24))
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=FOOD_FILL,
            outline=FOOD_OUTLINE,
            width=2,
        )

    if snake:
        for first, second in zip(snake, snake[1:]):
            if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
                continue
            first_center = (
                margin_x + first[1] * cell + cell / 2,
                margin_y + first[0] * cell + cell / 2,
            )
            if first[0] == second[0]:
                y = first_center[1]
                if first[1] < second[1]:
                    start = (margin_x + first[1] * cell + cell - 4, y)
                    end = (margin_x + second[1] * cell + 4, y)
                else:
                    start = (margin_x + first[1] * cell + 4, y)
                    end = (margin_x + second[1] * cell + cell - 4, y)
            else:
                x = first_center[0]
                if first[0] < second[0]:
                    start = (x, margin_y + first[0] * cell + cell - 4)
                    end = (x, margin_y + second[0] * cell + 4)
                else:
                    start = (x, margin_y + first[0] * cell + 4)
                    end = (x, margin_y + second[0] * cell + cell - 4)
            rod_width = max(6, cell // 7)
            draw.line([start, end], fill=BODY_OUTLINE, width=rod_width + 4)
            draw.line([start, end], fill=BODY_FILL, width=rod_width)

        for row, col in snake[1:]:
            x0 = margin_x + col * cell + 4
            y0 = margin_y + row * cell + 4
            draw.rounded_rectangle(
                [x0, y0, x0 + cell - 8, y0 + cell - 8],
                radius=8,
                fill=BODY_FILL,
                outline=BODY_OUTLINE,
                width=2,
            )

        head_row, head_col = snake[0]
        cx = margin_x + head_col * cell + cell / 2
        cy = margin_y + head_row * cell + cell / 2
        half = (cell - 8) / 2
        draw.rounded_rectangle(
            [cx - half, cy - half, cx + half, cy + half],
            radius=8,
            fill=HEAD_FILL,
            outline=HEAD_OUTLINE,
            width=2,
        )
    return finish_scene(image)


class SnakePathGenerator(AbstractPuzzleGenerator[Dict[str, Any]]):
    TASK_FAMILY = "dynamics_game"

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        width: int = 8,
        height: int = 8,
        snake_length_range: Tuple[int, int] = (3, 5),
        food_count_range: Tuple[int, int] = (2, 4),
        steps_between_food_range: Tuple[int, int] = (2, 4),
        video: bool = True,
        video_fps: int = 16,
    ) -> None:
        super().__init__(output_dir)
        if width < 6 or height < 6:
            raise ValueError("snake_path requires a board of at least 6x6")
        if food_count_range[0] < 2 or food_count_range[0] > food_count_range[1]:
            raise ValueError("food_count_range must contain at least two pellets")
        if steps_between_food_range[0] < 1 or steps_between_food_range[0] > steps_between_food_range[1]:
            raise ValueError("steps_between_food_range must be a positive ordered range")
        self.rng = random.Random(seed)
        self.seed = seed
        self.width = int(width)
        self.height = int(height)
        self.snake_length_range = snake_length_range
        self.food_count_range = food_count_range
        self.steps_between_food_range = steps_between_food_range
        self.video = bool(video)
        self.video_fps = int(video_fps)
        self._counter = 0

    def _generate_snake(self, length: int) -> Tuple[Cell, ...]:
        deltas = list(_DIR_TO_DELTA.values())
        for _ in range(300):
            head = (self.rng.randrange(self.height), self.rng.randrange(self.width))
            snake = [head]
            occupied = {head}
            for _segment in range(length - 1):
                row, col = snake[-1]
                candidates = [
                    (row + dr, col + dc)
                    for dr, dc in deltas
                    if 0 <= row + dr < self.height
                    and 0 <= col + dc < self.width
                    and (row + dr, col + dc) not in occupied
                ]
                if not candidates:
                    break
                next_cell = self.rng.choice(candidates)
                snake.append(next_cell)
                occupied.add(next_cell)
            if len(snake) == length:
                return tuple(snake)
        raise RuntimeError("Failed to generate a connected initial snake.")

    def _candidate_moves(self, snake: Tuple[Cell, ...], *, grow: bool) -> List[Tuple[str, Cell]]:
        candidates: List[Tuple[str, Cell]] = []
        occupied = set(snake if grow else snake[:-1])
        for move, (dr, dc) in _DIR_TO_DELTA.items():
            new_head = (snake[0][0] + dr, snake[0][1] + dc)
            if (
                0 <= new_head[0] < self.height
                and 0 <= new_head[1] < self.width
                and new_head not in occupied
            ):
                candidates.append((move, new_head))
        self.rng.shuffle(candidates)
        return candidates

    def _generate_state_and_solution(self) -> Tuple[Tuple[Cell, ...], FrozenSet[Cell], List[str]]:
        min_length, max_length = self.snake_length_range
        min_food, max_food = self.food_count_range
        min_gap, max_gap = self.steps_between_food_range
        for _ in range(600):
            initial_snake = self._generate_snake(self.rng.randint(min_length, max_length))
            snake = initial_snake
            visited = set(initial_snake)
            foods: List[Cell] = []
            moves: List[str] = []
            failed = False
            for _food_index in range(self.rng.randint(min_food, max_food)):
                interval = self.rng.randint(min_gap, max_gap)
                for step_index in range(interval):
                    grow = step_index == interval - 1
                    candidates = self._candidate_moves(snake, grow=grow)
                    if grow:
                        candidates = [
                            (move, cell)
                            for move, cell in candidates
                            if cell not in visited
                        ]
                    if not candidates:
                        failed = True
                        break
                    move, new_head = candidates[0]
                    moves.append(move)
                    if grow:
                        foods.append(new_head)
                        snake = (new_head,) + snake
                    else:
                        snake = (new_head,) + snake[:-1]
                    visited.add(new_head)
                if failed:
                    break
            if failed:
                continue
            food_set = frozenset(foods)
            if (
                len(food_set) < 2
                or len({row for row, _col in food_set}) < 2
                or len({col for _row, col in food_set}) < 2
                or len(set(moves)) < 2
            ):
                continue
            trace = self._simulate(initial_snake, food_set, moves)
            if not trace[-1].foods and len(trace[-1].snake) == len(initial_snake) + len(food_set):
                return initial_snake, food_set, moves
        raise RuntimeError("Failed to generate a legal multi-food snake route.")

    def _simulate(
        self,
        initial_snake: Sequence[Cell],
        foods: Sequence[Cell] | FrozenSet[Cell],
        moves: Sequence[str],
    ) -> List[SnakeFrame]:
        snake = tuple(initial_snake)
        remaining = frozenset(foods)
        trace = [SnakeFrame(snake=snake, foods=remaining)]
        for move in moves:
            if not remaining:
                raise ValueError("A complete snake route must stop when the final pellet is eaten.")
            delta = _DIR_TO_DELTA.get(str(move).lower())
            if delta is None:
                raise ValueError(f"Unsupported move {move!r}")
            new_head = (snake[0][0] + delta[0], snake[0][1] + delta[1])
            grow = new_head in remaining
            occupied = set(snake if grow else snake[:-1])
            if (
                not (0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width)
                or new_head in occupied
            ):
                raise ValueError("Generated route contains an illegal snake move.")
            if grow:
                snake = (new_head,) + snake
                remaining = frozenset(cell for cell in remaining if cell != new_head)
            else:
                snake = (new_head,) + snake[:-1]
            trace.append(SnakeFrame(snake=snake, foods=remaining))
        if remaining:
            raise ValueError("Generated route does not consume every pellet.")
        return trace

    def _write_video(self, trace: Sequence[SnakeFrame], video_path: Path) -> int:
        frames: List[np.ndarray] = []
        protected_indices: List[int] = []
        for state in trace:
            image = render_snake_board(self.width, self.height, state.snake, state.foods)
            protected_indices.append(len(frames))
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            frames.extend(array.copy() for _ in range(DISCRETE_STATE_HOLD_FRAMES))
        frames = fit_frames_to_video_budget(
            frames,
            max_frames=frame_budget_for_discrete_state_count(len(trace)),
            final_hold_frames=DISCRETE_STATE_HOLD_FRAMES,
            protected_indices=protected_indices,
        )
        if not encode_rgb_frames_to_webm(frames, video_path, fps=self.video_fps):
            return 0
        return len(frames)

    def _state_grid(self, state: SnakeFrame) -> List[List[int]]:
        grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for row, col in state.foods:
            grid[row][col] = 1
        for row, col in state.snake[1:]:
            grid[row][col] = 2
        head_row, head_col = state.snake[0]
        grid[head_row][head_col] = 3
        return grid

    def _build_path_answer(self, trace: Sequence[SnakeFrame]) -> str:
        return format_xy_path_answer([state.snake[0] for state in trace])

    def _authored_route_narration(
        self,
        initial_snake: Sequence[Cell],
        foods: FrozenSet[Cell],
        trace: Sequence[SnakeFrame],
    ) -> str:
        """Narrate one legal route directly, pellet by pellet, with growth."""

        length = len(initial_snake)
        sentences = [
            f"The head starts at {display_cell_text(initial_snake[0])} with the body trailing back to "
            f"{display_cell_text(initial_snake[-1])}, and the pellets sit at "
            f"{display_cells_text(sorted(foods))}."
        ]
        segment: List[Cell] = [trace[0].snake[0]]
        for before, after in zip(trace, trace[1:]):
            segment.append(after.snake[0])
            if after.foods != before.foods:
                pellet = next(iter(before.foods - after.foods))
                length += 1
                remaining = len(after.foods)
                remaining_note = (
                    f"{remaining} pellet{'s' if remaining != 1 else ''} "
                    f"{'remain' if remaining != 1 else 'remains'}"
                    if remaining
                    else "no pellets remain"
                )
                sentences.append(
                    f"Following the head positions {display_path_text(segment)} brings it onto the pellet at "
                    f"{display_cell_text(pellet)}; the tail stays put on that bite, so the snake grows to "
                    f"length {length}, and {remaining_note}."
                )
                segment = [after.snake[0]]
        sentences.append(
            f"With every pellet eaten the route stops, leaving the head at "
            f"{display_cell_text(trace[-1].snake[0])} and the snake {length} segments long."
        )
        return " ".join(sentences)

    def _prompts(self) -> Dict[str, str]:
        ti2v = build_sectioned_prompt(
            _INPUT_DESC,
            _TI2V_OUTPUT,
            mode="ti2v",
            rule_description=_RULE_SUMMARY,
            continuity_paragraph=_TI2V_CONTINUITY,
        )
        return {
            "ti2v_prompt": ti2v,
        }

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"snake_path-{self._counter:05d}"
        self._counter += 1
        initial_snake, foods, moves = self._generate_state_and_solution()
        trace = self._simulate(initial_snake, foods, moves)

        puzzles_dir = self.output_dir / "puzzles"
        solutions_dir = self.output_dir / "solutions"
        videos_dir = self.output_dir / "videos"
        for directory in (puzzles_dir, solutions_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if self.video:
            videos_dir.mkdir(parents=True, exist_ok=True)

        puzzle_path = puzzles_dir / f"{sample_id}.png"
        solution_path = solutions_dir / f"{sample_id}.png"
        video_path = videos_dir / f"{sample_id}.webm"
        render_snake_board(self.width, self.height, initial_snake, foods).save(puzzle_path)
        render_snake_board(
            self.width,
            self.height,
            trace[-1].snake,
            trace[-1].foods,
        ).save(solution_path)
        frame_count = self._write_video(trace, video_path) if self.video else 0

        answer = self._build_path_answer(trace)
        prompts = self._prompts()
        initial_state = trace[0]
        return {
            "id": sample_id,
            "answer": answer,
            "task_name": "snake_path",
            "task_family": self.TASK_FAMILY,
            "image": self.relativize_path(puzzle_path),
            "reasoning_image": self.relativize_path(puzzle_path),
            "solution_image_path": self.relativize_path(solution_path),
            "solution_video_path": self.relativize_path(video_path) if frame_count > 0 else None,
            "video_fps": self.video_fps if frame_count > 0 else None,
            "video_num_frames": frame_count if frame_count > 0 else None,
            "question": (
                "Return the complete head-position path of any legal route that eats all visible food pellets "
                "without leaving the board or colliding with the snake."
            ),
            "answer_payload": {
                "grid_rows": self.height,
                "grid_cols": self.width,
                "initial_grid": self._state_grid(initial_state),
                "initial_snake": [list(cell) for cell in initial_snake],
                "foods": [list(cell) for cell in sorted(foods)],
                "food_count": len(foods),
                "move_sequence": list(moves),
                "state_trace": [self._state_grid(state) for state in trace],
            },
            **prompts,
        }


__all__ = [
    "BODY_FILL",
    "FOOD_FILL",
    "HEAD_FILL",
    "SnakeFrame",
    "SnakePathGenerator",
    "board_geometry",
    "render_snake_board",
]
