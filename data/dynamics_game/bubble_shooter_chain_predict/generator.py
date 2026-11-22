"""Bubble Shooter chain-pop prediction.

Mechanic: an 8-column hex-staggered bubble grid sits at the top of the canvas.
A shooter at the bottom-center holds one bubble of a known color. A green dotted
arrow indicates the launch trajectory. The shot bubble lands at a marked
attachment cell (the first empty cell along the arrow that touches an existing
bubble — shown as a hollow target ring on top of the input image).

After landing, the new bubble plus all same-color bubbles connected to it (4-
connected on the hex grid) form a cluster. If the cluster size is >= 3, the
entire cluster pops. Predict the after-pop board.

Mechanic load-bearing: at construction time we verify that the cluster size
without the new bubble is < 3, but with it is >= 3. So the shot is what
triggers the pop.
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import CANVAS_SIZE, DEFAULT_FPS
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font
from data.common.pipeline.discrete_dynamics import (
    BG, LABEL_COLOR,
    encode_video, make_record_dataclass,
    shuffle_with_correct_semantics, write_artifacts,
)
from data.common.multiple_choice import option_label
from data.dynamics_game.core.choice_composite import (
    attach_choice_input_fields,
    build_composite_mcq_prompts,
    finalize_choice_artifacts,
)
from data.dynamics_game.core.coordinate_format import with_grid_xy_convention

_INPUT_DESC = with_grid_xy_convention(
    "The input image shows a hex-staggered colored-token board with columns labeled 1 through 8 and rows labeled 1 through 5. Tokens are colored red, "
    "yellow, blue, or green. A navy launcher with a pale chamber at the bottom center holds one token whose color is "
    "shown. A green dotted trajectory line ends at a hollow "
    "white attachment ring marking where the shot bubble will land."
)
_RULE_SUMMARY = (
    "The shot bubble lands at the marked ring and joins the grid. Two bubbles are neighbors when their circles "
    "touch: the bubble directly left or right in the same row, and the two nearest bubbles in each adjacent "
    "staggered row. Collect every bubble of the shot's color reachable from it through same-color neighbors, "
    "including the shot itself. If that cluster has at least three bubbles, remove the whole cluster "
    "simultaneously; otherwise nothing pops. No bubble moves, falls, or refills afterwards."
)
_TI2V_OUTPUT = (
    "Hold the input, then move the same fixed-size bubble continuously along the dotted trajectory into the "
    "attachment ring. At contact, shrink every member of the qualifying connected same-color cluster "
    "together into the pale background, without a separate landing pause, then hold. Keep every survivor and "
    "the launcher, trajectory, ring, board, labels, and spacing fixed. Apply no gravity, refill, second "
    "clear, or black gap."
)
_SOLUTION_IMAGE_OUTPUT = "Render the still image of the final board after the shot bubble lands and the connected same-color cluster pops."

PALETTE = [
    ("R", (228, 80, 80), (160, 30, 30)),
    ("Y", (245, 205, 70), (170, 130, 30)),
    ("B", (60, 130, 220), (30, 70, 160)),
    ("G", (80, 175, 105), (40, 110, 60)),
]
_BD2Record = make_record_dataclass("bubble_shooter_chain_predict", "dynamics_game")
_BUBBLE_CODES = {None: 0, "R": 1, "Y": 2, "B": 3, "G": 4}


def _hex_neighbors(r: int, c: int, rows: int, cols: int):
    # offset hex: even rows shift left, odd rows shift right (point-top hex).
    if r % 2 == 0:
        offsets = ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0))
    else:
        offsets = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1))
    for dr, dc in offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _encode_grid(grid: Sequence[Sequence[Optional[str]]]) -> List[List[int]]:
    return [[_BUBBLE_CODES[cell] for cell in row] for row in grid]


class BubbleShooterChainPredictGenerator(AbstractPuzzleGenerator):
    def __init__(self, output_dir, *, seed=None, video_fps=DEFAULT_FPS, video=False):
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.rows = 5
        self.cols = 8
        self.video_fps = video_fps
        self.video = video
        self._counter = 0
        # Hex grid layout: cells are circles of radius `cell_radius`.
        self.cell_size = 48  # diameter
        self.cell_radius = self.cell_size // 2
        self.row_height = int(self.cell_size * 0.86)  # vertical spacing for hex stagger
        self.grid_top = 60
        self.grid_left = (CANVAS_SIZE - self.cols * self.cell_size) // 2
        self.grid_left -= self.grid_left % 2
        self.shooter_y = CANVAS_SIZE - 56

    def _bubble_center(self, r, c):
        x = self.grid_left + c * self.cell_size + self.cell_size // 2
        if r % 2 == 1:
            x += self.cell_size // 2
        y = self.grid_top + r * self.row_height + self.cell_radius
        return x, y

    def _cluster(self, grid, r, c):
        target = grid[r][c]
        if target is None:
            return set()
        seen = {(r, c)}
        q = deque([(r, c)])
        while q:
            cr, cc = q.popleft()
            for nr, nc in _hex_neighbors(cr, cc, self.rows, self.cols):
                if (nr, nc) not in seen and grid[nr][nc] == target:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return seen

    def _build_state(self):
        for _ in range(120):
            grid = [[self.rng.choice(PALETTE)[0] if self.rng.random() < 0.85 else None for _ in range(self.cols)] for _ in range(self.rows)]
            # Find an empty cell adjacent to at least one bubble.
            empty_cells = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if grid[r][c] is not None:
                        continue
                    if any(grid[nr][nc] is not None for nr, nc in _hex_neighbors(r, c, self.rows, self.cols)):
                        empty_cells.append((r, c))
            self.rng.shuffle(empty_cells)
            for tr, tc in empty_cells:
                color = self.rng.choice([k for k, _, _ in PALETTE])
                # Same-color neighbours
                same_neighbors = [(nr, nc) for nr, nc in _hex_neighbors(tr, tc, self.rows, self.cols) if grid[nr][nc] == color]
                if not same_neighbors:
                    continue
                # Try landing here.
                trial = [row[:] for row in grid]
                trial[tr][tc] = color
                cluster = self._cluster(trial, tr, tc)
                # Without the bubble, cluster < 3 (so the shot is required).
                if len(cluster) >= 3:
                    pre_clusters = []
                    for nr, nc in same_neighbors:
                        pre_clusters.append(self._cluster(grid, nr, nc))
                    max_pre = max((len(c) for c in pre_clusters), default=0)
                    if max_pre < 3:
                        return grid, color, (tr, tc), cluster
        raise RuntimeError("Unable to sample a load-bearing bubble-shot cluster.")

    def _apply(self, grid, cluster):
        out = [list(row) for row in grid]
        for r, c in cluster:
            out[r][c] = None
        return out

    def _draw_bubble(self, draw, code, center, *, faded=False, hollow_ring=False, radius_scale=1.0):
        cx, cy = center
        r = max(1.0, (self.cell_radius - 4) * float(radius_scale))
        if hollow_ring:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=4)
            draw.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], outline=(40, 40, 40), width=1)
            return
        for k, fill, outline in PALETTE:
            if k == code:
                if faded:
                    fill = tuple(int(fill[i] * 0.4 + 200 * 0.6) for i in range(3))
                    outline = tuple(int(outline[i] * 0.4 + 200 * 0.6) for i in range(3))
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=2)
                return

    def _draw_dotted_line(self, draw, p0, p1, color=(50, 200, 80), step=10):
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        L = (dx * dx + dy * dy) ** 0.5
        if L < 1:
            return
        ux, uy = dx / L, dy / L
        d = 0.0
        on = True
        while d < L:
            seg = min(step, L - d)
            if on:
                a = (x0 + ux * d, y0 + uy * d)
                b = (x0 + ux * (d + seg), y0 + uy * (d + seg))
                draw.line([a, b], fill=color, width=4)
            d += seg
            on = not on

    def _render(
        self,
        grid,
        *,
        shooter_color=None,
        target=None,
        show_arrow=True,
        popping_cells=(),
        pop_progress=0.0,
        moving_bubble=None,
        moving_color=None,
    ):
        image, draw = new_scene(BG)
        font = scene_font(13)
        # Bubbles.
        popping_cells = set(popping_cells)
        for r in range(self.rows):
            for c in range(self.cols):
                v = grid[r][c]
                if v is None:
                    continue
                progress = float(pop_progress) if (r, c) in popping_cells else 0.0
                self._draw_bubble(
                    draw,
                    v,
                    self._bubble_center(r, c),
                    faded=progress > 0.35,
                    radius_scale=max(0.04, 1.0 - progress),
                )
        # Coordinate labels around the staggered grid.
        for r in range(self.rows):
            x0, y = self._bubble_center(r, 0)
            draw.text((x0 - self.cell_radius - 8, y), str(r + 1), fill=LABEL_COLOR, font=font, anchor="rm")
        for c in range(self.cols):
            x, _ = self._bubble_center(0, c)
            draw.text((x, self.grid_top - 14), str(c + 1), fill=LABEL_COLOR, font=font, anchor="mm")
        # Shooter.
        sx = CANVAS_SIZE // 2
        sy = self.shooter_y
        launcher_fill = (58, 78, 132)
        launcher_edge = (31, 49, 92)
        draw.rounded_rectangle([sx - 24, sy - 10, sx + 24, sy + 24], radius=9, fill=launcher_fill, outline=launcher_edge, width=2)
        draw.polygon([(sx - 14, sy - 8), (sx + 14, sy - 8), (sx, sy - 29)], fill=launcher_fill, outline=launcher_edge)
        if shooter_color is not None:
            self._draw_bubble(draw, shooter_color, (sx, sy + 4))
        else:
            draw.ellipse([sx - 12, sy - 8, sx + 12, sy + 16], fill=(231, 237, 247), outline=launcher_edge, width=2)
        if moving_bubble is not None and moving_color is not None:
            self._draw_bubble(draw, moving_color, moving_bubble)
        # Trajectory dotted arrow + ring.
        if target is not None and show_arrow:
            tx, ty = self._bubble_center(*target)
            self._draw_dotted_line(draw, (sx, sy - 20), (tx, ty))
            self._draw_bubble(draw, None, (tx, ty), hollow_ring=True)
        return finish_scene(image)

    @staticmethod
    def _lerp_point(start, end, t):
        return start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t

    def _write_video(self, video_path: Path, *, grid, color, target, cluster, final) -> int:
        shooter = (CANVAS_SIZE // 2, self.shooter_y + 4)
        landing = self._bubble_center(*target)
        landed = [row[:] for row in grid]
        landed[target[0]][target[1]] = color
        frames: List[np.ndarray] = []

        def push(image: Image.Image, count: int = 1) -> None:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            frames.extend(frame.copy() for _ in range(count))

        push(self._render(grid, shooter_color=color, target=target), 8)
        for index in range(1, 21):
            t = index / 20.0
            push(
                self._render(
                    grid,
                    shooter_color=None,
                    target=target,
                    moving_bubble=self._lerp_point(shooter, landing, t),
                    moving_color=color,
                )
            )
        for index in range(1, 14):
            progress = index / 13.0
            push(
                self._render(
                    landed,
                    shooter_color=None,
                    target=target,
                    show_arrow=True,
                    popping_cells=cluster,
                    pop_progress=progress,
                )
            )
        push(self._render(final, shooter_color=None, target=target, show_arrow=True), 8)
        assert len(frames) == 49
        return encode_video(frames, video_path, fps=self.video_fps)

    def _build_distractors(self, grid, color, target, cluster):
        # 1) Just landed (no pop).
        d1 = [row[:] for row in grid]
        d1[target[0]][target[1]] = color
        # 2) Pop only same-color neighbours (not full cluster), so cluster - {one outer cell} effectively
        d2 = self._apply(grid, cluster)
        d2[target[0]][target[1]] = color  # keep the landed bubble — common error
        # 3) Pop the full cluster but ALSO an extra neighbouring same-color bubble outside the cluster
        d3_set = set(cluster)
        # Add an arbitrary same-color cell outside cluster, if any.
        extras = [(r, c) for r in range(self.rows) for c in range(self.cols)
                  if grid[r][c] == color and (r, c) not in cluster]
        if extras:
            d3_set.add(self.rng.choice(extras))
        d3 = self._apply(grid, d3_set)
        # All distractor renders ignore the landed bubble (will never include it because correct includes pop)
        return [d1, d2, d3]

    def create_puzzle(self):
        sample_id = f"bubble_shooter_chain_predict-{self._counter:05d}"
        self._counter += 1
        grid, color, target, cluster = self._build_state()
        assert len(cluster) >= 3
        # Final state: original grid - cluster (which includes the landed bubble).
        landed = [row[:] for row in grid]
        landed[target[0]][target[1]] = color
        final = self._apply(landed, cluster)
        puzzle_image = self._render(grid, shooter_color=color, target=target)
        solution_image = self._render(final, shooter_color=None, target=target, show_arrow=True)
        distractors = self._build_distractors(grid, color, target, cluster)
        distractor_tiles = [self._render(d, shooter_color=None, target=target, show_arrow=True) for d in distractors]
        option_tiles, candidate_state_grids, correct_index = shuffle_with_correct_semantics(
            solution_image,
            _encode_grid(final),
            distractor_tiles,
            [_encode_grid(item) for item in distractors],
            self.rng,
        )
        prompts = build_composite_mcq_prompts(
            input_desc=_INPUT_DESC,
            rule_summary=_RULE_SUMMARY,
            ti2v_output=_TI2V_OUTPUT,
            solution_image_output=_SOLUTION_IMAGE_OUTPUT,
            continuity_timing="continuous",
            continuity_moving="the launched bubble and the popping cluster",
            continuity_static=(
                "every surviving bubble, the launcher body, the dotted trajectory, the attachment ring, "
                "the row and column labels, the background, and the camera"
            ),
            continuity_hold="the final board after the pop",
        )
        initial_grid = _encode_grid(grid)
        state_grid = _encode_grid(final)
        landed_grid = _encode_grid(landed)
        payload = {
            "mechanic": "shot_land_hex_cluster_pop",
            "grid": [list(row) for row in grid],
            "final_grid": [list(row) for row in final],
            "shooter_color": color,
            "target_cell": list(target),
            "cluster": sorted(list(cluster)),
            "initial_grid": initial_grid,
            "state_grid": state_grid,
            "state_trace": [initial_grid, state_grid],
            "process_state_trace": [initial_grid, landed_grid, state_grid],
            "candidate_state_grids": candidate_state_grids,
            "correct_option_index": correct_index,
            "correct_option": option_label(correct_index),
        }
        artifacts = write_artifacts(
            output_dir=self.output_dir, sample_id=sample_id,
            puzzle_image=puzzle_image, solution_image=solution_image,
            option_tiles=option_tiles, correct_index=correct_index, payload=payload,
            write_video=(
                lambda video_path: self._write_video(
                    video_path,
                    grid=grid,
                    color=color,
                    target=target,
                    cluster=cluster,
                    final=final,
                )
            ) if self.video else None,
        )
        finalize_choice_artifacts(
            output_dir=self.output_dir,
            artifacts=artifacts,
            option_tiles=option_tiles,
            correct_index=correct_index,
            payload=payload,
            video_fps=self.video_fps,
        )
        answer_label = option_label(correct_index)
        reasoning_images = [self.relativize_path(artifacts["puzzle_path"])]
        record = _BD2Record(
            id=sample_id,
            image=self.relativize_path(artifacts["puzzle_path"]),
            reasoning_image=self.relativize_path(artifacts["puzzle_path"]),
            reasoning_images=reasoning_images,
            solution_image_path=self.relativize_path(artifacts["solution_path"]),
            solution_video_path=self.relativize_path(artifacts["video_path"]) if artifacts["frame_count"] > 0 else None,
            video_fps=self.video_fps if artifacts["frame_count"] > 0 else None,
            video_num_frames=artifacts["frame_count"] if artifacts["frame_count"] > 0 else None,
            ti2v_prompt=prompts["ti2v_prompt"],

            answer=answer_label, correct_option=answer_label,


            question="Which candidate shows the board after the marked token is launched, attached, and the qualifying connected-color cluster is removed?",
            structured_answer=payload, answer_payload=payload,
        )

        return attach_choice_input_fields(
            record,
            artifacts=artifacts,
            relativize=self.relativize_path,
        )


__all__ = ["BubbleShooterChainPredictGenerator"]
