from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.media import build_complete_footprint_fade_frames
from data.common.render_primitives import ScaledDraw, finish_scaled_canvas
from data.dynamics_game.core.scene import SS, finish_scene, new_scene_rgba
from data.common.video_encoding import (
    encode_rgb_frames_to_webm,
)

from data.common.pipeline.sliding_tiles import (
    _2048_tile_value,
    _MOVE_DELTAS,
    _load_font,
    _swipe_board,
)
from data.dynamics_game.core.coordinate_format import with_grid_xy_convention
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the sliding and merging tiles",
    static_nouns="everything the rules leave unchanged, including the board, labels, background, and camera",
    hold_noun="the final board state",
)

class Game2048BestSwipeGenerator(AbstractPuzzleGenerator[Dict[str, Any]]):
    TASK_NAME = "merge_swipe_score"
    TASK_FAMILY = "dynamics_game"
    # Board pixel geometry (single source; the family verifier imports these
    # instead of re-hardcoding cell centers).
    CELL = 78
    ORIGIN_X = 100
    ORIGIN_Y = 108

    @staticmethod
    def merge_targets_for(
        board: Sequence[Sequence[int]],
        direction: str,
    ) -> List[Tuple[int, int, int]]:
        """Return every summed-tile destination ``(row, col, value)`` for one swipe."""

        size = len(board)
        targets: List[Tuple[int, int, int]] = []
        for line_index in range(size):
            if direction in {"left", "right"}:
                indices = list(range(size)) if direction == "left" else list(range(size - 1, -1, -1))
                values = [int(board[line_index][index]) for index in indices if int(board[line_index][index])]
                destination = lambda offset: (line_index, indices[offset])
            else:
                indices = list(range(size)) if direction == "up" else list(range(size - 1, -1, -1))
                values = [int(board[index][line_index]) for index in indices if int(board[index][line_index])]
                destination = lambda offset: (indices[offset], line_index)
            source_index = 0
            destination_offset = 0
            while source_index < len(values):
                row, col = destination(destination_offset)
                value = values[source_index]
                if source_index + 1 < len(values) and value == values[source_index + 1]:
                    targets.append((row, col, value * 2))
                    source_index += 2
                else:
                    source_index += 1
                destination_offset += 1
        return targets

    @classmethod
    def render_cell_endpoint(cls, value: int) -> Image.Image:
        """Render one isolated board cell exactly as the board renderer draws it."""

        image = Image.new("RGB", (cls.CELL * SS, cls.CELL * SS), (255, 255, 255))
        draw = ScaledDraw(ImageDraw.Draw(image, "RGBA"), SS)
        font = _load_font(max(20, int(cls.CELL * 0.36)) * SS)
        cls._draw_tile(draw, value, 0, 0, cls.CELL, font)
        return finish_scaled_canvas(image, (cls.CELL, cls.CELL))

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        video_fps: int = 16,
    ) -> None:
        super().__init__(output_dir)
        self.seed = seed
        self.rng = random.Random(seed)
        self.video = video
        self.video_fps = video_fps
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _generate_board(
        self,
    ) -> Tuple[
        List[List[int]],
        Tuple[str, str],
        List[List[List[int]]],
        Tuple[int, int],
    ]:
        directions = list(_MOVE_DELTAS)
        for _ in range(1024):
            board = [[0 for _ in range(4)] for _ in range(4)]
            occupied = self.rng.sample([(row, col) for row in range(4) for col in range(4)], self.rng.randint(6, 10))
            for row, col in occupied:
                board[row][col] = self.rng.choice([2, 2, 4, 4, 8, 16])
            first_candidates: List[Tuple[str, List[List[int]], int]] = []
            for direction in directions:
                result, score = _swipe_board(board, direction)
                if score[0] > 0 and result != board:
                    first_candidates.append((direction, result, int(score[0])))
            self.rng.shuffle(first_candidates)
            for first_direction, after_first, first_count in first_candidates:
                second_candidates: List[Tuple[str, List[List[int]], int]] = []
                for second_direction in directions:
                    if second_direction == first_direction:
                        continue
                    after_second, second_score = _swipe_board(after_first, second_direction)
                    if second_score[0] > 0 and after_second != after_first:
                        second_candidates.append((second_direction, after_second, int(second_score[0])))
                if not second_candidates:
                    continue
                second_direction, after_second, second_count = self.rng.choice(second_candidates)
                return (
                    board,
                    (first_direction, second_direction),
                    [after_first, after_second],
                    (first_count, second_count),
                )
        raise RuntimeError("Failed to generate a two-round merge board with a positive merge in each round.")

    def _layout(self, size: int) -> Tuple[int, int, int]:
        cell = self.CELL
        ox = (512 - size * cell) // 2
        ox -= ox % 2
        assert size != 4 or (ox, self.ORIGIN_Y) == (self.ORIGIN_X, self.ORIGIN_Y)
        return cell, ox, self.ORIGIN_Y

    @staticmethod
    def _draw_tile(
        draw: ImageDraw.ImageDraw,
        value: int,
        x0: float,
        y0: float,
        cell: int,
        font: Any,
        *,
        scale: float = 1.0,
        opacity: float = 1.0,
    ) -> None:
        opacity = max(0.0, min(1.0, float(opacity)))
        if opacity <= 0.015:
            return

        def faded(color: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
            return (*color, int(round(255 * opacity)))

        fill = _2048_tile_value(value)
        radius = max(8, cell // 6)
        if value == 0:
            draw.rounded_rectangle(
                [x0 + 4, y0 + 4, x0 + cell - 4, y0 + cell - 4],
                radius=radius,
                fill=faded((235, 240, 247)),
                outline=faded((205, 215, 229)),
                width=2,
            )
            return
        inset = 4 + (1.0 - max(0.1, min(1.0, scale))) * cell * 0.34
        draw.rounded_rectangle(
            [x0 + inset, y0 + inset, x0 + cell - inset, y0 + cell - inset],
            radius=max(4, int(radius * scale)),
            fill=faded(fill),
            outline=faded((126, 111, 92)),
            width=2,
        )
        if scale >= 0.58:
            label = str(value)
            text_color = (119, 110, 101) if value <= 4 else (250, 246, 240)
            draw.text(
                (x0 + cell / 2, y0 + cell / 2),
                label,
                fill=faded(text_color),
                font=font,
                anchor="mm",
            )

    def _render_board(
        self,
        board: Sequence[Sequence[int]],
        *,
        moving: Optional[Sequence[Tuple[Any, ...]]] = None,
        suppress_indices: Optional[Sequence[Tuple[int, int]]] = None,
        directions: Optional[Sequence[str]] = None,
        tile_scale: float = 1.0,
    ) -> Image.Image:
        size = len(board)
        cell, ox, oy = self._layout(size)
        image, draw = new_scene_rgba((247, 249, 253))
        font = _load_font(max(20, int(cell * 0.36)) * SS)
        label_font = _load_font(14 * SS)
        suppress = set(suppress_indices or ())
        draw.rounded_rectangle(
            [ox - 10, oy - 10, ox + size * cell + 10, oy + size * cell + 10],
            radius=20,
            fill=(255, 255, 255),
            outline=(210, 220, 234),
            width=2,
        )
        for col in range(size):
            x = ox + col * cell + cell / 2
            draw.text((x, oy - 22), str(col + 1), fill=(70, 80, 100), font=label_font, anchor="mm")
        for row in range(size):
            y = oy + row * cell + cell / 2
            draw.text((ox - 22, y), str(row + 1), fill=(70, 80, 100), font=label_font, anchor="rm")

        for r in range(size):
            for c in range(size):
                v = int(board[r][c])
                x0 = ox + c * cell
                y0 = oy + r * cell
                if (r, c) in suppress:
                    self._draw_tile(draw, 0, x0, y0, cell, font)
                else:
                    self._draw_tile(draw, v, x0, y0, cell, font, scale=tile_scale if v else 1.0)
        if moving:
            for item in moving:
                v, mx, my = item[:3]
                moving_scale = float(item[3]) if len(item) > 3 else 1.0
                moving_opacity = float(item[4]) if len(item) > 4 else 1.0
                self._draw_tile(
                    draw,
                    v,
                    mx,
                    my,
                    cell,
                    font,
                    scale=moving_scale,
                    opacity=moving_opacity,
                )
        if directions:
            self._draw_direction_badges(draw, directions)
        return finish_scene(image)

    def _draw_direction_badges(
        self,
        draw: ImageDraw.ImageDraw,
        directions: Sequence[str],
    ) -> None:
        if len(directions) != 2:
            raise ValueError("Merge task requires exactly two numbered swipe badges.")
        number_font = _load_font(15 * SS)
        for index, direction in enumerate(directions):
            cx = 218 + index * 76
            self._draw_direction_badge(draw, direction, cx=cx, outline=(36, 67, 129))
            draw.ellipse([cx - 31, 22, cx - 13, 40], fill=(255, 255, 255), outline=(36, 67, 129), width=1)
            draw.text((cx - 22, 31), str(index + 1), fill=(36, 67, 129), font=number_font, anchor="mm")

    def _draw_direction_badge(
        self,
        draw: ImageDraw.ImageDraw,
        direction: str,
        *,
        cx: int,
        outline: Tuple[int, int, int],
    ) -> None:
        draw.rounded_rectangle(
            [cx - 35, 18, cx + 35, 72],
            radius=20,
            fill=(65, 105, 183),
            outline=outline,
            width=2,
        )
        cy = 47
        if direction in {"left", "right"}:
            sign = 1 if direction == "right" else -1
            points = [
                (cx + sign * 18, cy),
                (cx - sign * 4, cy - 13),
                (cx - sign * 4, cy - 5),
                (cx - sign * 19, cy - 5),
                (cx - sign * 19, cy + 5),
                (cx - sign * 4, cy + 5),
                (cx - sign * 4, cy + 13),
            ]
        else:
            sign = 1 if direction == "down" else -1
            points = [
                (cx, cy + sign * 18),
                (cx - 13, cy - sign * 4),
                (cx - 5, cy - sign * 4),
                (cx - 5, cy - sign * 19),
                (cx + 5, cy - sign * 19),
                (cx + 5, cy - sign * 4),
                (cx + 13, cy - sign * 4),
            ]
        draw.polygon(points, fill=(255, 255, 255))

    def _write_video(
        self,
        boards: Sequence[Sequence[Sequence[int]]],
        directions: Sequence[str],
        video_path: Path,
    ) -> int:
        if len(boards) != 3 or len(directions) != 2:
            raise ValueError("Two-round merge video requires three states and two directions.")
        size = len(boards[0])
        cell, ox, oy = self._layout(size)
        def tile_motion_for(
            board: Sequence[Sequence[int]],
            direction: str,
        ) -> List[Tuple[Tuple[int, int], Tuple[int, int], int]]:
            motion: List[Tuple[Tuple[int, int], Tuple[int, int], int]] = []
            line_count = size
            for line_index in range(line_count):
                if direction in {"left", "right"}:
                    indices = list(range(size)) if direction == "left" else list(range(size - 1, -1, -1))
                    sources = [(line_index, index) for index in indices if int(board[line_index][index]) != 0]
                    values = [int(board[row][col]) for row, col in sources]
                    destination = lambda offset: (line_index, indices[offset])
                else:
                    indices = list(range(size)) if direction == "up" else list(range(size - 1, -1, -1))
                    sources = [(index, line_index) for index in indices if int(board[index][line_index]) != 0]
                    values = [int(board[row][col]) for row, col in sources]
                    destination = lambda offset: (indices[offset], line_index)
                source_index = 0
                destination_offset = 0
                while source_index < len(values):
                    target = destination(destination_offset)
                    motion.append((sources[source_index], target, values[source_index]))
                    if source_index + 1 < len(values) and values[source_index] == values[source_index + 1]:
                        motion.append((sources[source_index + 1], target, values[source_index + 1]))
                        source_index += 2
                    else:
                        source_index += 1
                    destination_offset += 1
            return motion

        frames: List[np.ndarray] = []
        first = self._render_board(boards[0], directions=directions)
        frames.extend(np.asarray(first, dtype=np.uint8).copy() for _ in range(5))

        for round_index, direction in enumerate(directions):
            board = boards[round_index]
            final_board = boards[round_index + 1]
            tile_motion = tile_motion_for(board, direction)
            suppressed = [origin for origin, _, _ in tile_motion]
            for step in range(1, 9):
                progress = step / 8.0
                moving = []
                for (source_row, source_col), (target_row, target_col), value in tile_motion:
                    start_x, start_y = ox + source_col * cell, oy + source_row * cell
                    end_x, end_y = ox + target_col * cell, oy + target_row * cell
                    moving.append(
                        (
                            value,
                            start_x + (end_x - start_x) * progress,
                            start_y + (end_y - start_y) * progress,
                        )
                    )
                frame = self._render_board(
                    board,
                    moving=moving,
                    suppress_indices=suppressed,
                    directions=directions,
                )
                frames.append(np.asarray(frame, dtype=np.uint8))

            target_groups: Dict[Tuple[int, int], List[int]] = {}
            for _, target, value in tile_motion:
                target_groups.setdefault(target, []).append(value)
            merged_targets = {target for target, values in target_groups.items() if len(values) == 2}
            for step in range(1, 5):
                scale = 1.0 - 0.8 * (step / 4.0)
                moving = [
                    (
                        value,
                        ox + target_col * cell,
                        oy + target_row * cell,
                        scale if (target_row, target_col) in merged_targets else 1.0,
                    )
                    for _, (target_row, target_col), value in tile_motion
                ]
                frame = self._render_board(
                    board,
                    moving=moving,
                    suppress_indices=suppressed,
                    directions=directions,
                )
                frames.append(np.asarray(frame, dtype=np.uint8))
            # The summed tiles are an atomic appearance, not another motion
            # primitive.  Render both fixed-size endpoints first, then blend
            # only their changed pixels.  This keeps the entire terminal
            # footprint present at every partial level and leaves all static
            # board/badge pixels byte-identical.
            fade_base = self._render_board(
                final_board,
                suppress_indices=sorted(merged_targets),
                directions=directions,
            )
            fade_terminal = self._render_board(
                final_board,
                directions=directions,
            )
            fade_frames = build_complete_footprint_fade_frames(
                fade_base,
                fade_terminal,
                hold_frames=1,
                fade_frames=3,
                final_hold_frames=1,
            )
            # One zero-opacity endpoint, two partial-opacity frames, and one
            # fully visible endpoint preserve the existing four-frame phase.
            frames.extend(frame.copy() for frame in fade_frames[:4])
            hold = np.asarray(
                self._render_board(final_board, directions=directions),
                dtype=np.uint8,
            )
            frames.extend(hold.copy() for _ in range(4))

        final_frame = np.asarray(self._render_board(boards[-1], directions=directions), dtype=np.uint8)
        frames.extend(final_frame.copy() for _ in range(8))
        assert len(frames) == 53 and len(frames) % 4 == 1
        if not encode_rgb_frames_to_webm(
            frames, video_path, fps=self.video_fps,
            extra_args=["-g", "1", "-bf", "0"],
        ):
            return 0
        return len(frames)

    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.TASK_NAME}-{self._counter:05d}"
        self._counter += 1
        board, swipe_directions, round_states, round_merge_counts = self._generate_board()
        after_first, final_board = round_states
        merge_score = sum(round_merge_counts)
        prompts = self._prompts()

        puzzle_dir = self.output_dir / "puzzles"
        solution_dir = self.output_dir / "solutions"
        video_dir = self.output_dir / "videos"
        state_dir = self.output_dir / "states"
        for directory in (puzzle_dir, solution_dir, video_dir, state_dir):
            directory.mkdir(parents=True, exist_ok=True)

        puzzle_path = puzzle_dir / f"{sample_id}.png"
        solution_path = solution_dir / f"{sample_id}.png"
        round_one_path = state_dir / f"{sample_id}__after_round_01.png"
        video_path = video_dir / f"{sample_id}.webm"
        puzzle_image = self._render_board(board, directions=swipe_directions)
        round_one_image = self._render_board(after_first, directions=swipe_directions)
        solution_image = self._render_board(final_board, directions=swipe_directions)
        puzzle_image.save(puzzle_path)
        round_one_image.save(round_one_path)
        solution_image.save(solution_path)
        frame_count = (
            self._write_video([board, after_first, final_board], swipe_directions, video_path)
            if self.video
            else 0
        )

        # NOTE: action_sequence is intentionally OMITTED — the canonical answer
        # is now an integer merge count, not a direction list. Including
        # action_sequence would route the evaluator to action-list matching
        # which would never accept an integer answer.
        payload = {
            "state_grid": final_board,
            "initial_grid": board,
            "state_trace": [board, after_first, final_board],
            "process_state_trace": [board, after_first, final_board],
            "swipe_directions": list(swipe_directions),
            "round_merge_counts": list(round_merge_counts),
            "merge_count": merge_score,
        }

        return {
            "id": sample_id,
            "task_name": self.TASK_NAME,
            "task_family": self.TASK_FAMILY,
            "image": self.relativize_path(puzzle_path),
            "reasoning_image": self.relativize_path(puzzle_path),
            "reasoning_images": [self.relativize_path(puzzle_path)],
            "solution_image_path": self.relativize_path(solution_path),
            "solution_video_path": self.relativize_path(video_path) if frame_count > 0 else None,
            "video_fps": self.video_fps if frame_count > 0 else None,
            "video_num_frames": frame_count if frame_count > 0 else None,
            "ti2v_prompt": prompts["ti2v_prompt"],
            "answer": str(merge_score),
            "question": "How many tile merges happen across the two numbered swipes?",
            "question_description": "merge_swipe_score",
            "structured_answer": payload,
            "answer_payload": payload,
        }

    def _prompts(self) -> Dict[str, str]:
        input_desc = with_grid_xy_convention('The single problem image shows a 4 by 4 board with rows and columns labeled 1 through 4, pale empty wells, and numbered tiles. Two blue arrow badges above the board are numbered 1 and 2; apply their swipe directions in that order. Each numbered rounded tile occupies exactly one well, and its printed integer is its value.')
        rules_desc = "Apply badge 1, then badge 2. For each swipe, ignore empty wells and slide the nonzero tiles to the shown edge without changing their order. From that edge inward, fuse the first disjoint adjacent equal pairs; each pair becomes one tile with their summed value. A source tile is used at most once, and a new tile cannot merge again during the same swipe. Use round 1's board for round 2, where its new tiles may merge normally. Count every fused pair across both rounds."
        ti2v_output = 'Hold the input, then execute each numbered swipe in order. Within a round, move all nonzero tiles together along their rows or columns while preserving order. Equal partners meet at one destination and contract together before one complete fixed-size summed tile fades in there as a whole; nonmerging tiles stop full-size. Hold the complete round-1 board before starting round 2, then hold the final board. Keep both badges, wells, values, colors, and spacing fixed; do not grow the summed tile from its center or chain-merge within a round.'
        ti2v = build_sectioned_prompt(input_desc, ti2v_output, mode='ti2v', rule_description=rules_desc, continuity_paragraph=_TI2V_CONTINUITY)
        return {'ti2v_prompt': ti2v}
