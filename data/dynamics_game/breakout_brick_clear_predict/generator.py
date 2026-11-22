"""Breakout first-brick prediction with one load-bearing wall bounce.

The input contains one 10 x 7 playfield, a ball and paddle, and the complete
planned polyline.  The ball follows that polyline to a marked wall contact,
changes direction exactly once, and removes the first brick touched after the
bounce.  Instance construction additionally requires that ignoring the bounce
and travelling straight toward the eventual target would hit a different
brick first, so the bounce is never decorative.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import DEFAULT_FPS, BoardLayout
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font
from data.common.render_primitives import ScaledDraw
from data.common.pipeline.discrete_dynamics import (
    BG,
    GRID_LINE,
    LABEL_COLOR,
    encode_video,
    make_record_dataclass,
    shuffle_with_correct_semantics,
    write_artifacts,
)
from data.common.multiple_choice import option_label
from data.dynamics_game.core.choice_composite import (
    attach_choice_input_fields,
    build_composite_mcq_prompts,
    finalize_choice_artifacts,
)
from data.dynamics_game.core.coordinate_format import with_grid_xy_convention


Cell = Tuple[int, int]
Point = Tuple[float, float]

_INPUT_DESC = with_grid_xy_convention(
    "The input image shows one 10 by 7 rectangular field with row and column labels 1 through 10 and 1 through 7. The top 4 rows contain "
    "colored bricks (red, orange, yellow, or green; one color per row), and white cells are empty. A dark "
    "paddle and a silver ball start in the bottom row. A green dotted polyline gives the complete planned "
    "route from the ball to one marked contact on the left, right, or top wall, then onward into the bricks."
)
_RULE_SUMMARY = (
    "Follow the dotted route in order. The ball reaches the marked wall contact, changes direction there exactly "
    "once, then stops when it first touches a brick. That first brick is destroyed. Every other brick, the board, "
    "the wall marker, the route, and the paddle remain fixed."
)
_TI2V_OUTPUT = (
    "Hold the input, then move the same silver ball at constant speed along both drawn segments, extending a "
    "solid blue trail behind it over the dotted route, and turn at the marked wall contact without pausing. "
    "At the first brick contact, keep the ball there while only that brick shrinks into its cell center until "
    "the ordinary white well shows, then hold the final board."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the final board on the same canvas with the complete travelled route and paddle preserved, the stationary "
    "ball still visible at the impact point, and only the first brick struck after the bounce removed."
)

ROW_COLORS = [(228, 80, 80), (245, 165, 65), (245, 205, 70), (80, 175, 105)]
ROW_EDGES = [(160, 30, 30), (170, 100, 30), (170, 130, 30), (40, 110, 60)]
PLANNED_PATH = (80, 205, 105)
TRAVELLED_PATH = (35, 135, 205)
BALL_FILL = (220, 225, 240)
BALL_EDGE = (60, 70, 90)
_BD2Record = make_record_dataclass("breakout_brick_clear_predict", "dynamics_game")


def _encode_bricks(bricks: Dict[Cell, int], rows: int = 10, cols: int = 7) -> List[List[int]]:
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for (row, col), color_idx in bricks.items():
        grid[row][col] = int(color_idx) + 1
    return grid


class BreakoutBrickClearPredictGenerator(AbstractPuzzleGenerator):
    def __init__(self, output_dir, *, seed=None, video_fps=DEFAULT_FPS, video=False):
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.rows = 10
        self.cols = 7
        self.video_fps = video_fps
        self.video = video
        self._counter = 0
        self.layout = BoardLayout.for_grid(self.rows, self.cols, board_max=440, reserve_top=24)

    @staticmethod
    def _segment_cells(start: Cell, end: Cell) -> List[Cell]:
        """Return the ordered rounded grid cells after ``start`` through ``end``."""

        start_row, start_col = start
        end_row, end_col = end
        steps = max(abs(end_row - start_row), abs(end_col - start_col))
        if steps <= 0:
            return []
        cells: List[Cell] = []
        for step in range(1, steps + 1):
            t = step / steps
            cell = (
                int(round(start_row + (end_row - start_row) * t)),
                int(round(start_col + (end_col - start_col) * t)),
            )
            if not cells or cells[-1] != cell:
                cells.append(cell)
        return cells

    def _trajectory_cells(self, segments: Sequence[Tuple[Cell, Cell]]) -> List[Cell]:
        cells: List[Cell] = []
        for start, end in segments:
            for cell in self._segment_cells(start, end):
                if not cells or cells[-1] != cell:
                    cells.append(cell)
        return cells

    def _first_brick_on_path(
        self,
        bricks: Dict[Cell, int],
        segments: Sequence[Tuple[Cell, Cell]],
    ) -> Optional[Cell]:
        for cell in self._trajectory_cells(segments):
            if cell in bricks:
                return cell
        return None

    def _start_point(self, start_col: int) -> Point:
        return self.layout.cell_center(self.rows - 1, start_col)

    def _bounce_point(self, wall: str, bounce: Cell) -> Point:
        row, col = bounce
        if wall == "L":
            return float(self.layout.origin_x), self.layout.cell_center(row, 0)[1]
        if wall == "R":
            return float(self.layout.origin_x + self.cols * self.layout.cell), self.layout.cell_center(row, self.cols - 1)[1]
        return self.layout.cell_center(0, col)[0], float(self.layout.origin_y)

    def _hit_point(self, hit: Cell) -> Point:
        return self.layout.cell_center(*hit)

    def _pixel_path_cells(self, points: Sequence[Point]) -> List[Cell]:
        """Rasterize the displayed polyline back into board cells in order."""

        cells: List[Cell] = []
        step_px = max(1.0, self.layout.cell / 8.0)
        for start, end in zip(points, points[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            steps = max(1, int(math.ceil(math.hypot(dx, dy) / step_px)))
            for index in range(1, steps + 1):
                t = index / steps
                x = start[0] + dx * t
                y = start[1] + dy * t
                col = int((x - self.layout.origin_x) // self.layout.cell)
                row = int((y - self.layout.origin_y) // self.layout.cell)
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    cell = (row, col)
                    if not cells or cells[-1] != cell:
                        cells.append(cell)
        return cells

    def _first_brick_on_pixel_path(self, bricks: Dict[Cell, int], points: Sequence[Point]) -> Optional[Cell]:
        for cell in self._pixel_path_cells(points):
            if cell in bricks:
                return cell
        return None

    @staticmethod
    def _distance_to_segment(point: Point, start: Point, end: Point) -> float:
        vx, vy = end[0] - start[0], end[1] - start[1]
        length_sq = vx * vx + vy * vy
        if length_sq <= 1e-9:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        t = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq))
        closest = (start[0] + vx * t, start[1] + vy * t)
        return math.hypot(point[0] - closest[0], point[1] - closest[1])

    def _visual_clearance_cells(self, points: Sequence[Point], *, exclude: Cell) -> set[Cell]:
        """Reserve a ball-width corridor so the rendered ball cannot graze another brick."""

        clearance: set[Cell] = set()
        radius = self.layout.cell * 0.68
        for row in range(4):
            for col in range(self.cols):
                cell = (row, col)
                if cell == exclude:
                    continue
                center = self.layout.cell_center(row, col)
                if any(
                    self._distance_to_segment(center, start, end) <= radius
                    for start, end in zip(points, points[1:])
                ):
                    clearance.add(cell)
        return clearance

    def _build_state(self) -> Tuple[Dict[Cell, int], int, str, Cell, Cell, Cell]:
        """Construct a closed instance and return its straight-line ablation hit."""

        start = None
        for _attempt in range(3000):
            start_col = self.rng.randint(1, self.cols - 2)
            start = (self.rows - 1, start_col)
            wall = self.rng.choice(["L", "R", "T"])
            if wall == "L":
                bounce = (self.rng.randint(4, self.rows - 3), 0)
                hit = (self.rng.randint(0, 2), self.rng.randrange(self.cols))
            elif wall == "R":
                bounce = (self.rng.randint(4, self.rows - 3), self.cols - 1)
                hit = (self.rng.randint(0, 2), self.rng.randrange(self.cols))
            else:
                bounce = (0, self.rng.randrange(self.cols))
                hit = (self.rng.randint(1, 3), self.rng.randrange(self.cols))

            logical_segments = [(start, bounce), (bounce, hit)]
            logical_cells = self._trajectory_cells(logical_segments)
            if not logical_cells or logical_cells[-1] != hit or hit in logical_cells[:-1]:
                continue

            pixel_points = [self._start_point(start_col), self._bounce_point(wall, bounce), self._hit_point(hit)]
            pixel_cells = self._pixel_path_cells(pixel_points)
            if not pixel_cells or pixel_cells[-1] != hit or hit in pixel_cells[:-1]:
                continue

            clearance = self._visual_clearance_cells(pixel_points, exclude=hit)
            clearance.update(cell for cell in logical_cells[:-1] if 0 <= cell[0] < 4)

            direct_logical = self._trajectory_cells([(start, hit)])
            direct_pixel = self._pixel_path_cells([pixel_points[0], pixel_points[-1]])
            blocker_candidates = [
                cell
                for cell in direct_logical[:-1]
                if 0 <= cell[0] < 4 and cell not in clearance and cell != hit and cell in direct_pixel[:-1]
            ]
            if not blocker_candidates:
                continue
            blocker = blocker_candidates[0]

            bricks: Dict[Cell, int] = {}
            for row in range(4):
                for col in range(self.cols):
                    cell = (row, col)
                    if cell in clearance:
                        continue
                    if self.rng.random() < 0.72:
                        bricks[cell] = row
            bricks[hit] = hit[0]
            bricks[blocker] = blocker[0]

            bounced_hit = self._first_brick_on_path(bricks, logical_segments)
            visual_hit = self._first_brick_on_pixel_path(bricks, pixel_points)
            direct_hit = self._first_brick_on_path(bricks, [(start, hit)])
            visual_direct_hit = self._first_brick_on_pixel_path(bricks, [pixel_points[0], pixel_points[-1]])
            if (
                bounced_hit == hit
                and visual_hit == hit
                and direct_hit is not None
                and direct_hit != hit
                and visual_direct_hit is not None
                and visual_direct_hit != hit
                and len(bricks) >= 8
            ):
                return bricks, start_col, wall, bounce, hit, direct_hit

        raise RuntimeError(f"Unable to build a valid load-bearing Breakout bounce from start={start}")

    @staticmethod
    def _apply(bricks: Dict[Cell, int], hit: Cell) -> Dict[Cell, int]:
        out = dict(bricks)
        out.pop(hit, None)
        return out

    def _draw_dotted(self, draw: ImageDraw.ImageDraw, start: Point, end: Point, *, color=PLANNED_PATH, step=10) -> None:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        distance = 0.0
        visible = True
        while distance < length:
            segment = min(step, length - distance)
            if visible:
                draw.line(
                    [
                        (start[0] + ux * distance, start[1] + uy * distance),
                        (start[0] + ux * (distance + segment), start[1] + uy * (distance + segment)),
                    ],
                    fill=color,
                    width=4,
                )
            distance += segment
            visible = not visible

    @staticmethod
    def _draw_polyline(draw: ScaledDraw, points: Sequence[Point], *, fill, width: int) -> None:
        # The travelled path turns sharply where the ball reflects off a wall, so
        # the joins are mitered rather than rounded.
        if len(points) >= 2:
            draw.open_path(list(points), fill=fill, width=width)

    def _draw_wall_marker(self, draw: ImageDraw.ImageDraw, wall: str, point: Point) -> None:
        half = self.layout.cell * 0.36
        width = 6
        color = (45, 155, 105)
        edge = (20, 100, 65)
        if wall in {"L", "R"}:
            draw.line([(point[0], point[1] - half), (point[0], point[1] + half)], fill=edge, width=width + 4)
            draw.line([(point[0], point[1] - half), (point[0], point[1] + half)], fill=color, width=width)
        else:
            draw.line([(point[0] - half, point[1]), (point[0] + half, point[1])], fill=edge, width=width + 4)
            draw.line([(point[0] - half, point[1]), (point[0] + half, point[1])], fill=color, width=width)

    def _render_image(
        self,
        bricks: Dict[Cell, int],
        *,
        start_col: int,
        wall: str,
        bounce: Cell,
        hit: Cell,
        ball_position: Optional[Point],
        travelled_points: Sequence[Point] = (),
        shrinking_brick: Optional[Cell] = None,
        brick_scale: float = 1.0,
        show_path: bool = True,
        show_paddle: bool = True,
    ) -> Image.Image:
        image, draw = new_scene(BG)
        cell = self.layout.cell
        font = scene_font(max(11, int(cell * 0.32)))
        start_point = self._start_point(start_col)
        bounce_point = self._bounce_point(wall, bounce)
        hit_point = self._hit_point(hit)

        for row in range(self.rows):
            for col in range(self.cols):
                x0, y0, x1, y1 = self.layout.cell_box(row, col)
                draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=GRID_LINE, width=1)

        if show_path:
            self._draw_dotted(draw, start_point, bounce_point)
            self._draw_dotted(draw, bounce_point, hit_point)
        if travelled_points:
            self._draw_polyline(draw, travelled_points, fill=TRAVELLED_PATH, width=5)

        for (row, col), color_idx in bricks.items():
            x0, y0, x1, y1 = self.layout.cell_box(row, col)
            local_scale = max(0.0, min(1.0, brick_scale)) if (row, col) == shrinking_brick else 1.0
            if local_scale <= 0.01:
                continue
            inset = 2 + (1.0 - local_scale) * cell * 0.46
            draw.rectangle(
                [x0 + inset, y0 + inset, x1 - inset, y1 - inset],
                fill=ROW_COLORS[color_idx],
                outline=ROW_EDGES[color_idx],
                width=2,
            )

        label_y = self.layout.origin_y - 14
        for col in range(self.cols):
            x = self.layout.origin_x + col * cell + cell / 2
            draw.text((x, label_y), str(col + 1), fill=LABEL_COLOR, font=font, anchor="mm")
        label_x = self.layout.origin_x - 12
        for row in range(self.rows):
            y = self.layout.origin_y + row * cell + cell / 2
            draw.text((label_x, y), str(row + 1), fill=LABEL_COLOR, font=font, anchor="rm")

        if show_paddle:
            x0, y0, x1, y1 = self.layout.cell_box(self.rows - 1, start_col)
            draw.rounded_rectangle(
                [x0 - cell // 2, y1 - 9, x1 + cell // 2, y1 - 3],
                radius=3,
                fill=(40, 50, 70),
                outline=(0, 0, 0),
                width=2,
            )

        self._draw_wall_marker(draw, wall, bounce_point)

        if ball_position is not None:
            radius = max(7, int(cell * 0.23))
            x, y = ball_position
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=BALL_FILL, outline=BALL_EDGE, width=2)

        return finish_scene(image)

    def _build_distractors(self, bricks: Dict[Cell, int], hit: Cell) -> List[Dict[Cell, int]]:
        unchanged = dict(bricks)
        wrong_candidates = [cell for cell in bricks if cell != hit]
        wrong = dict(bricks)
        wrong.pop(self.rng.choice(wrong_candidates), None)
        two_removed = self._apply(bricks, hit)
        extra_candidates = [cell for cell in two_removed]
        two_removed.pop(self.rng.choice(extra_candidates), None)
        return [unchanged, wrong, two_removed]

    @staticmethod
    def _interpolate(start: Point, end: Point, steps: int) -> List[Point]:
        return [
            (
                start[0] + (end[0] - start[0]) * index / steps,
                start[1] + (end[1] - start[1]) * index / steps,
            )
            for index in range(1, steps + 1)
        ]

    @staticmethod
    def _constant_speed_segment_counts(start: Point, bounce: Point, hit: Point, total_steps: int = 36) -> Tuple[int, int]:
        """Allocate frames by geometric path length so both legs use one speed.

        The old renderer split the first leg at an arbitrary reasoning keyframe,
        paused there, and assigned the two sub-legs equal frame counts.  That
        produced an unintended slow-fast-stop cadence.  Keyframes are now
        observations of one continuous trajectory rather than motion controls.
        """

        first = math.dist(start, bounce)
        second = math.dist(bounce, hit)
        first_steps = max(1, int(round(total_steps * first / (first + second))))
        first_steps = min(total_steps - 1, first_steps)
        return first_steps, total_steps - first_steps

    def _write_video(
        self,
        video_path: Path,
        *,
        bricks: Dict[Cell, int],
        final: Dict[Cell, int],
        start_col: int,
        wall: str,
        bounce: Cell,
        hit: Cell,
    ) -> int:
        start_point = self._start_point(start_col)
        bounce_point = self._bounce_point(wall, bounce)
        hit_point = self._hit_point(hit)
        frames: List[np.ndarray] = []

        def push(image: Image.Image, count: int = 1) -> None:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            frames.extend(array.copy() for _ in range(count))

        initial = self._render_image(
            bricks,
            start_col=start_col,
            wall=wall,
            bounce=bounce,
            hit=hit,
            ball_position=start_point,
        )
        push(initial, 8)

        first_steps, second_steps = self._constant_speed_segment_counts(start_point, bounce_point, hit_point)
        for point in self._interpolate(start_point, bounce_point, first_steps):
            push(
                self._render_image(
                    bricks,
                    start_col=start_col,
                    wall=wall,
                    bounce=bounce,
                    hit=hit,
                    ball_position=point,
                    travelled_points=[start_point, point],
                )
            )
        for point in self._interpolate(bounce_point, hit_point, second_steps):
            push(
                self._render_image(
                    bricks,
                    start_col=start_col,
                    wall=wall,
                    bounce=bounce,
                    hit=hit,
                    ball_position=point,
                    travelled_points=[start_point, bounce_point, point],
                )
            )

        for brick_scale in (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10):
            impact = self._render_image(
                bricks,
                start_col=start_col,
                wall=wall,
                bounce=bounce,
                hit=hit,
                ball_position=hit_point,
                travelled_points=[start_point, bounce_point, hit_point],
                shrinking_brick=hit,
                brick_scale=brick_scale,
            )
            push(impact)
        push(
            self._render_image(
                final,
                start_col=start_col,
                wall=wall,
                bounce=bounce,
                hit=hit,
                ball_position=hit_point,
                travelled_points=[start_point, bounce_point, hit_point],
            ),
            12,
        )
        assert len(frames) == 65
        return encode_video(frames, video_path, fps=self.video_fps)

    def create_puzzle(self):
        sample_id = f"breakout_brick_clear_predict-{self._counter:05d}"
        self._counter += 1
        bricks, start_col, wall, bounce, hit, direct_hit = self._build_state()
        final = self._apply(bricks, hit)
        start_point = self._start_point(start_col)
        bounce_point = self._bounce_point(wall, bounce)
        hit_point = self._hit_point(hit)
        puzzle_image = self._render_image(
            bricks,
            start_col=start_col,
            wall=wall,
            bounce=bounce,
            hit=hit,
            ball_position=start_point,
        )
        solution_image = self._render_image(
            final,
            start_col=start_col,
            wall=wall,
            bounce=bounce,
            hit=hit,
            ball_position=hit_point,
            travelled_points=[start_point, bounce_point, hit_point],
        )

        distractors = self._build_distractors(bricks, hit)
        distractor_tiles = [
            self._render_image(
                distractor,
                start_col=start_col,
                wall=wall,
                bounce=bounce,
                hit=hit,
                ball_position=hit_point,
                travelled_points=[start_point, bounce_point, hit_point],
            )
            for distractor in distractors
        ]
        option_tiles, candidate_state_grids, correct_index = shuffle_with_correct_semantics(
            solution_image,
            _encode_bricks(final, self.rows, self.cols),
            distractor_tiles,
            [_encode_bricks(distractor, self.rows, self.cols) for distractor in distractors],
            self.rng,
        )
        answer_label = option_label(correct_index)

        initial_grid = _encode_bricks(bricks, self.rows, self.cols)
        state_grid = _encode_bricks(final, self.rows, self.cols)
        payload: Dict[str, Any] = {
            "mechanic": "single_wall_bounce_first_brick_clear",
            "mechanic_family": "projectile_trajectory_first_contact",
            "bricks": [[row, col, color] for (row, col), color in sorted(bricks.items())],
            "start_col": start_col,
            "wall": wall,
            "bounce": list(bounce),
            "bounce_pixel": [round(value, 3) for value in bounce_point],
            "hit": list(hit),
            "straight_line_ablation_hit": list(direct_hit),
            "initial_grid": initial_grid,
            "state_grid": state_grid,
            # Keep the shared board evaluator's strict initial/final contract.
            "state_trace": [initial_grid, state_grid],
            # Motion before impact leaves brick occupancy unchanged.
            "process_state_trace": [initial_grid, initial_grid, initial_grid, initial_grid, state_grid],
            "candidate_state_grids": candidate_state_grids,
            "correct_option_index": correct_index,
            "correct_option": answer_label,
        }
        prompts = build_composite_mcq_prompts(
            input_desc=_INPUT_DESC,
            rule_summary=_RULE_SUMMARY,
            ti2v_output=_TI2V_OUTPUT,
            solution_image_output=_SOLUTION_IMAGE_OUTPUT,
            continuity_timing="continuous",
            continuity_moving="the silver ball, its blue trail, and the struck brick",
            continuity_static=(
                "the dotted route, the wall marker, the paddle, every unstruck brick, the row and column labels, "
                "the background, and the camera"
            ),
            continuity_hold="the final board with the struck brick removed",
        )

        artifacts = write_artifacts(
            output_dir=self.output_dir,
            sample_id=sample_id,
            puzzle_image=puzzle_image,
            solution_image=solution_image,
            option_tiles=option_tiles,
            correct_index=correct_index,
            payload=payload,
            write_video=(
                lambda video_path: self._write_video(
                    video_path,
                    bricks=bricks,
                    final=final,
                    start_col=start_col,
                    wall=wall,
                    bounce=bounce,
                    hit=hit,
                )
            )
            if self.video
            else None,
        )
        finalize_choice_artifacts(
            output_dir=self.output_dir,
            artifacts=artifacts,
            option_tiles=option_tiles,
            correct_index=correct_index,
            payload=payload,
            video_fps=self.video_fps,
        )

        solution_path = self.relativize_path(artifacts["solution_path"])
        reasoning_images = [self.relativize_path(artifacts["puzzle_path"])]
        record = _BD2Record(
            id=sample_id,
            image=self.relativize_path(artifacts["puzzle_path"]),
            reasoning_image=self.relativize_path(artifacts["puzzle_path"]),
            reasoning_images=reasoning_images,
            solution_image_path=solution_path,
            solution_video_path=self.relativize_path(artifacts["video_path"]) if artifacts["frame_count"] > 0 else None,
            video_fps=self.video_fps if artifacts["frame_count"] > 0 else None,
            video_num_frames=artifacts["frame_count"] if artifacts["frame_count"] > 0 else None,
            ti2v_prompt=prompts["ti2v_prompt"],


            answer=answer_label,
            correct_option=answer_label,


            question="Which candidate after-board removes exactly the first brick struck after the marked wall bounce?",
            structured_answer=payload,
            answer_payload=payload,
        )

        return attach_choice_input_fields(
            record,
            artifacts=artifacts,
            relativize=self.relativize_path,
        )


__all__ = ["BreakoutBrickClearPredictGenerator"]
