"""Block Blast piece-placement clear prediction.

Mechanic: an 8x8 board, partly occupied by colored blocks. A piece (a small
fixed pentomino-like shape, 3-5 cells) is shown to the side with a black anchor
arrow pointing into the board to where its top-left cell will land.

Place the piece at the marked anchor, then any FULL row OR FULL column that
results clears (entire row/column becomes empty, ALL its cells). Predict the
final board.

Mechanic load-bearing: at sample-construction time we verify that placing the
piece DOES clear at least one row/column AND on the unplaced board there is
no full row/column.
"""

from __future__ import annotations

import random

import numpy as np

from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import DEFAULT_FPS, BoardLayout
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font
from data.common.pipeline.discrete_dynamics import (
    BG, GRID_LINE, encode_video, make_record_dataclass,
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
    "The input image shows an 8 by 8 board with rows and columns labeled 1 through 8. Filled cells are rounded blue "
    "blocks and empty cells are pale wells. "
    "A separate orange piece appears in the card to the right. The same shape is outlined translucently on empty "
    "board cells; those outlined cells are the exact landing cells for the piece."
)
_RULE_SUMMARY = (
    "Place every cell of the piece on the board at the anchor. After placement, any row or column that becomes "
    "fully filled clears entirely (every cell of that row or column becomes empty). Surviving blocks stay where "
    "they are."
)
_TI2V_OUTPUT = (
    "Hold the input, then carry the same rigid orange piece continuously from its card into the outlined "
    "wells without rotating or deforming it. After it lands, its cells turn into fixed blue blocks. Then "
    "every occupied cell of each completed row and column shrinks simultaneously into its own pale empty "
    "well, and the cleared board holds."
)
_SOLUTION_IMAGE_OUTPUT = "Render the still image of the final board after the piece is placed and full rows or columns are cleared."

PIECE_SHAPES = [
    [(0, 0), (0, 1), (0, 2)],            # 1x3 horizontal
    [(0, 0), (1, 0), (2, 0)],            # 3x1 vertical
    [(0, 0), (0, 1), (1, 0)],            # corner L 3
    [(0, 0), (0, 1), (1, 1), (1, 0)],    # 2x2 square
    [(0, 0), (0, 1), (0, 2), (1, 0)],    # L4
    [(0, 0), (1, 0), (1, 1), (1, 2)],    # J4
]
BLOCK_FILL = (74, 139, 225)
BLOCK_EDGE = (37, 91, 173)
PIECE_FILL = (248, 166, 63)
PIECE_EDGE = (182, 105, 28)
EMPTY = (235, 240, 248)

_BD2Record = make_record_dataclass("block_blast_clear_predict", "dynamics_game")


class BlockBlastClearPredictGenerator(AbstractPuzzleGenerator):
    def __init__(self, output_dir, *, seed=None, video_fps=DEFAULT_FPS, video=False):
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.size = 8
        self.video_fps = video_fps
        self.video = video
        self._counter = 0
        from data.common.prompt_paraphrase import active_profile
        tray_below=active_profile("block_inlays")
        self.layout = BoardLayout(
            rows=self.size,cols=self.size,cell=40,
            origin_x=96 if tray_below else 28,
            origin_y=40 if tray_below else 96,
        )

    def _row_or_col_full(self, grid):
        return any(all(grid[r][c] == 1 for c in range(self.size)) for r in range(self.size)) or\
               any(all(grid[r][c] == 1 for r in range(self.size)) for c in range(self.size))

    def _build_state(self):
        for _ in range(2000):
            density = self.rng.uniform(0.45, 0.62)
            grid = [[1 if self.rng.random() < density else 0 for _ in range(self.size)] for _ in range(self.size)]
            if self._row_or_col_full(grid):
                continue
            shape = self.rng.choice(PIECE_SHAPES)
            # Find an anchor where every piece cell is empty AND placement creates >=1 full row/col.
            valid = []
            for ar in range(self.size):
                for ac in range(self.size):
                    if any(ar + dr >= self.size or ac + dc >= self.size or grid[ar + dr][ac + dc] == 1 for dr, dc in shape):
                        continue
                    trial = [row[:] for row in grid]
                    for dr, dc in shape:
                        trial[ar + dr][ac + dc] = 1
                    if self._row_or_col_full(trial):
                        valid.append((ar, ac))
            if valid:
                anchor = self.rng.choice(valid)
                return grid, shape, anchor
        raise RuntimeError("Unable to sample a valid Block Blast placement after 2000 attempts.")

    def _apply(self, grid, shape, anchor):
        out = [row[:] for row in grid]
        ar, ac = anchor
        for dr, dc in shape:
            out[ar + dr][ac + dc] = 1
        # Clear full rows/cols.
        full_rows = [r for r in range(self.size) if all(out[r][c] == 1 for c in range(self.size))]
        full_cols = [c for c in range(self.size) if all(out[r][c] == 1 for r in range(self.size))]
        for r in full_rows:
            for c in range(self.size):
                out[r][c] = 0
        for c in full_cols:
            for r in range(self.size):
                out[r][c] = 0
        return out, full_rows, full_cols

    def _placed_grid(self, grid, shape, anchor, *, piece_value=1):
        placed = [row[:] for row in grid]
        ar, ac = anchor
        for dr, dc in shape:
            placed[ar + dr][ac + dc] = piece_value
        return placed

    @staticmethod
    def _grid_key(grid):
        return tuple(tuple(int(cell) for cell in row) for row in grid)

    def _build_distractors(self, grid, shape, anchor, correct):
        placed = self._placed_grid(grid, shape, anchor)
        _, full_rows, full_cols = self._apply(grid, shape, anchor)
        correct_key = self._grid_key(correct)
        distractors = []
        seen = {correct_key}

        def add(candidate):
            key = self._grid_key(candidate)
            if key not in seen:
                seen.add(key)
                distractors.append(candidate)

        # 1) Place the piece but forget to clear completed lines.
        add([row[:] for row in placed])

        # 2) Clear only the first completed line when several lines complete.
        # If exactly one line completes, clear a nearby non-complete line instead.
        partial = [row[:] for row in placed]
        completed = [("row", row) for row in full_rows] + [("col", col) for col in full_cols]
        if len(completed) > 1:
            axis, index = completed[0]
        elif full_rows:
            axis = "row"
            index = next(
                (row for row in range(self.size) if row not in full_rows and any(partial[row])),
                (full_rows[0] + 1) % self.size,
            )
        else:
            axis = "col"
            index = next(
                (
                    col
                    for col in range(self.size)
                    if col not in full_cols and any(partial[row][col] for row in range(self.size))
                ),
                (full_cols[0] + 1) % self.size,
            )
        if axis == "row":
            partial[index] = [0] * self.size
        else:
            for row in range(self.size):
                partial[row][index] = 0
        add(partial)

        # 3) Apply the correct clear and then incorrectly clear one extra occupied line.
        extra = [row[:] for row in correct]
        extra_rows = [row for row in range(self.size) if any(extra[row])]
        if extra_rows:
            extra[extra_rows[0]] = [0] * self.size
        add(extra)

        # Bounded deterministic fallbacks preserve uniqueness for unusual edge layouts.
        for row in range(self.size):
            for col in range(self.size):
                if len(distractors) >= 3:
                    break
                fallback = [line[:] for line in correct]
                fallback[row][col] = 1 - fallback[row][col]
                add(fallback)
            if len(distractors) >= 3:
                break
        if len(distractors) != 3:
            raise RuntimeError("Unable to construct three unique Block Blast distractors.")
        return distractors

    def _preview_geometry(self, shape):
        cell = self.layout.cell
        rs = max(dr for dr, _ in shape) + 1
        cs = max(dc for _, dc in shape) + 1
        from data.common.prompt_paraphrase import active_profile
        if active_profile("block_inlays"):
            preview_left=256-cs*cell/2
            preview_top=439-rs*cell/2
        else:
            preview_left = 366 + (120 - cs * cell) / 2
            preview_top = 184 + (120 - rs * cell) / 2
        return cell, preview_left, preview_top, rs, cs

    @staticmethod
    def _block(draw,box,fill,edge,radius=8,width=2):
        from data.common.prompt_paraphrase import active_profile, draw_bevel
        if active_profile("block_inlays"):
            draw_bevel(draw,box,fill,edge)
        else:
            draw.rounded_rectangle(box,radius=radius,fill=fill,outline=edge,width=width)

    def _draw_piece_at(self, draw, shape, left, top, cell):
        for dr, dc in shape:
            x0 = int(round(left + dc * cell))
            y0 = int(round(top + dr * cell))
            inset = max(3, cell // 10)
            self._block(draw,[x0+inset,y0+inset,x0+cell-inset,y0+cell-inset],PIECE_FILL,PIECE_EDGE,radius=max(5,cell//5))

    def _piece_card_box(self):
        from data.common.prompt_paraphrase import active_profile
        return [96,374,416,504] if active_profile("block_inlays") else [356,132,496,356]

    def _draw_piece_card(self, draw, shape, *, show_piece=True):
        cell, preview_left, preview_top, _, _ = self._preview_geometry(shape)
        from data.common.prompt_paraphrase import active_profile
        tray_below=active_profile("block_inlays")
        card=self._piece_card_box()
        draw.rounded_rectangle(card,radius=20,fill=(251,247,227) if tray_below else (255,255,255),outline=(143,170,144) if tray_below else (210,219,233),width=2)
        if show_piece:
            self._draw_piece_at(draw, shape, preview_left, preview_top, cell)

    def _draw_target_guide(self, draw, shape, anchor):
        if anchor is None:
            return
        ar, ac = anchor
        for dr, dc in shape:
            x0, y0, x1, y1 = self.layout.cell_box(ar + dr, ac + dc)
            draw.rounded_rectangle(
                [x0 + 5, y0 + 5, x1 - 5, y1 - 5],
                radius=8,
                fill=(255, 232, 197),
                outline=PIECE_EDGE,
                width=3,
            )

    @staticmethod
    def _mix_color(start, end, progress):
        t = max(0.0, min(1.0, float(progress)))
        return tuple(int(round(start[index] + (end[index] - start[index]) * t)) for index in range(3))

    def _render(
        self,
        grid,
        *,
        shape=None,
        anchor=None,
        show_reference_piece=True,
        show_target_guide=False,
        piece_lock_progress=0.0,
        clearing_cells=(),
        clearing_scale=1.0,
        moving_piece=None,
    ):
        from data.common.prompt_paraphrase import active_profile
        inlays=active_profile("block_inlays")
        image,draw=new_scene((218,232,213) if inlays else BG)
        board_frame=[90,34,422,366] if inlays else [22,90,352,422]
        draw.rounded_rectangle(board_frame,radius=20,fill=(244,231,190) if inlays else (255,255,255),outline=(101,130,108) if inlays else (210,219,233),width=3 if inlays else 2)
        label_font = scene_font(13)
        for col in range(self.size):
            cx = self.layout.origin_x + (col + 0.5) * self.layout.cell
            draw.text((cx, self.layout.origin_y - 16), str(col + 1), fill=(55, 69, 91), font=label_font, anchor="mm")
        for row in range(self.size):
            cy = self.layout.origin_y + (row + 0.5) * self.layout.cell
            draw.text((self.layout.origin_x - 14, cy), str(row + 1), fill=(55, 69, 91), font=label_font, anchor="rm")
        clearing = {tuple(cell) for cell in clearing_cells}
        scale = max(0.0, min(1.0, float(clearing_scale)))
        for r in range(self.size):
            for c in range(self.size):
                x0, y0, x1, y1 = self.layout.cell_box(r, c)
                value = int(grid[r][c])
                if value == 2:
                    fill = self._mix_color(PIECE_FILL, BLOCK_FILL, piece_lock_progress)
                    edge = self._mix_color(PIECE_EDGE, BLOCK_EDGE, piece_lock_progress)
                elif value == 1:
                    fill, edge = BLOCK_FILL, BLOCK_EDGE
                else:
                    fill, edge = EMPTY, GRID_LINE

                is_clearing = value != 0 and (r, c) in clearing and scale < 1.0
                if is_clearing:
                    draw.rounded_rectangle(
                        [x0 + 2, y0 + 2, x1 - 2, y1 - 2],
                        radius=8,
                        fill=EMPTY,
                        outline=GRID_LINE,
                        width=1,
                    )
                    if scale > 0.02:
                        inset = 2 + (1.0 - scale) * (self.layout.cell / 2 - 4)
                        self._block(draw,[x0+inset,y0+inset,x1-inset,y1-inset],fill,edge,radius=max(2,int(round(8*scale))))
                elif value:
                    self._block(draw,[x0+2,y0+2,x1-2,y1-2],fill,edge)
                else:
                    draw.rounded_rectangle(
                        [x0 + 2, y0 + 2, x1 - 2, y1 - 2],
                        radius=8,
                        fill=fill,
                        outline=edge,
                        width=2 if value else 1,
                    )
        if show_target_guide and shape is not None:
            self._draw_target_guide(draw, shape, anchor)
        if shape is not None:
            self._draw_piece_card(draw, shape, show_piece=show_reference_piece)
        if moving_piece is not None:
            position_x, position_y, piece_cell = moving_piece
            self._draw_piece_at(draw, shape, position_x, position_y, piece_cell)
        return finish_scene(image)

    def _render_moving_piece(self, grid, shape, anchor, position, piece_cell):
        return self._render(
            grid,
            shape=shape,
            anchor=anchor,
            show_reference_piece=False,
            show_target_guide=True,
            moving_piece=(position[0], position[1], piece_cell),
        )

    def _write_video(self, grid, shape, anchor, full_rows, full_cols, final, video_path):
        preview_cell, preview_left, preview_top, _, _ = self._preview_geometry(shape)
        ar, ac = anchor
        target_left = self.layout.origin_x + ac * self.layout.cell
        target_top = self.layout.origin_y + ar * self.layout.cell
        motion_frames = []
        start = np.array(
            self._render_moving_piece(grid, shape, anchor, (preview_left, preview_top), preview_cell).convert("RGB"),
            dtype=np.uint8,
        )
        motion_frames.extend(start.copy() for _ in range(8))
        for step in range(1, 13):
            t = step / 12
            position = (
                preview_left + (target_left - preview_left) * t,
                preview_top + (target_top - preview_top) * t,
            )
            piece_cell = preview_cell + (self.layout.cell - preview_cell) * t
            motion_frames.append(
                np.array(
                    self._render_moving_piece(grid, shape, anchor, position, piece_cell).convert("RGB"),
                    dtype=np.uint8,
                )
            )
        placed_visual = self._placed_grid(grid, shape, anchor, piece_value=2)
        placed_frame = np.array(self._render(placed_visual, shape=shape, anchor=anchor, show_reference_piece=False).convert("RGB"), dtype=np.uint8)
        clearing_cells = {
            *((row, col) for row in full_rows for col in range(self.size)),
            *((row, col) for col in full_cols for row in range(self.size)),
        }
        final_frame = np.array(self._render(final, shape=shape, anchor=anchor, show_reference_piece=False).convert("RGB"), dtype=np.uint8)
        frames = list(motion_frames)
        frames.extend(placed_frame.copy() for _ in range(5))
        for step in range(1, 5):
            frames.append(
                np.array(
                    self._render(
                        placed_visual,
                        shape=shape,
                        anchor=anchor,
                        show_reference_piece=False,
                        piece_lock_progress=step / 4,
                    ).convert("RGB"),
                    dtype=np.uint8,
                )
            )
        locked_grid = self._placed_grid(grid, shape, anchor, piece_value=1)
        # Include the authored 0.68 semantic clear keyframe exactly; the
        # remaining scales preserve one monotone simultaneous shrink.
        for clearing_scale in (0.875, 0.75, 0.68, 0.55, 0.42, 0.28, 0.14, 0.0):
            frames.append(
                np.array(
                    self._render(
                        locked_grid,
                        shape=shape,
                        anchor=anchor,
                        show_reference_piece=False,
                        clearing_cells=clearing_cells,
                        clearing_scale=clearing_scale,
                    ).convert("RGB"),
                    dtype=np.uint8,
                )
            )
        frames.extend(final_frame.copy() for _ in range(16))
        assert len(frames) == 53
        return encode_video(frames, video_path, fps=self.video_fps)

    def create_puzzle(self):
        sample_id = f"block_blast_clear_predict-{self._counter:05d}"
        self._counter += 1
        grid, shape, anchor = self._build_state()
        final, full_rows, full_cols = self._apply(grid, shape, anchor)
        assert (full_rows or full_cols), "load-bearing assertion: piece must clear something"
        placed_grid = self._placed_grid(grid, shape, anchor)
        assert placed_grid != final, "load-bearing assertion: clearing must change the placed board"
        puzzle_image = self._render(grid, shape=shape, anchor=anchor, show_reference_piece=True, show_target_guide=True)
        solution_image = self._render(final, shape=shape, anchor=anchor, show_reference_piece=False)
        distractors = self._build_distractors(grid, shape, anchor, final)
        assert len({self._grid_key(final), *(self._grid_key(item) for item in distractors)}) == 4
        distractor_tiles = [self._render(d, shape=shape, anchor=anchor, show_reference_piece=False) for d in distractors]
        option_tiles, candidate_state_grids, correct_index = shuffle_with_correct_semantics(
            solution_image,
            [list(row) for row in final],
            distractor_tiles,
            [[list(row) for row in distractor] for distractor in distractors],
            self.rng,
        )
        prompts = build_composite_mcq_prompts(
            input_desc=_INPUT_DESC,
            rule_summary=_RULE_SUMMARY,
            ti2v_output=_TI2V_OUTPUT,
            solution_image_output=_SOLUTION_IMAGE_OUTPUT,
            continuity_timing="continuous",
            continuity_moving="the orange piece and the blocks of completed rows and columns",
            continuity_static=(
                "the board grid, the row and column labels, the piece card, every surviving blue block, "
                "the background, and the camera"
            ),
            continuity_hold="the final cleared board",
        )
        payload = {
            "mechanic": "piece_place_then_simultaneous_line_clear",
            "mechanic_family": "tile_merge_drop",
            "board": [list(row) for row in grid],
            "shape": shape,
            "anchor": list(anchor),
            "full_rows": full_rows,
            "full_cols": full_cols,
            "initial_grid": [list(row) for row in grid],
            "placed_grid": [list(row) for row in placed_grid],
            "state_grid": [list(row) for row in final],
            "state_trace": [[list(row) for row in grid], [list(row) for row in final]],
            "process_state_trace": [
                [list(row) for row in grid],
                [list(row) for row in placed_grid],
                [list(row) for row in placed_grid],
                [list(row) for row in final],
            ],
            "candidate_state_grids": candidate_state_grids,
            "correct_option_index": correct_index,
            "correct_option": option_label(correct_index),
        }
        write_video = (
            lambda video_path: self._write_video(grid, shape, anchor, full_rows, full_cols, final, video_path)
        ) if self.video else None
        artifacts = write_artifacts(
            output_dir=self.output_dir, sample_id=sample_id,
            puzzle_image=puzzle_image, solution_image=solution_image,
            option_tiles=option_tiles, correct_index=correct_index, payload=payload,
            write_video=write_video,
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


            question="Which candidate after-board correctly places the piece and clears full rows or columns?",
            structured_answer=payload, answer_payload=payload,
        )

        return attach_choice_input_fields(
            record,
            artifacts=artifacts,
            relativize=self.relativize_path,
        )


__all__ = ["BlockBlastClearPredictGenerator"]
