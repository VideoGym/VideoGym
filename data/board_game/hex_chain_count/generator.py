"""Hex longest-chain task.

A 7x7 pointy-top axial hex grid (rhombus) is partially filled with red and
blue discs.  The task asks for the size of the unique largest red component
under shared-edge six-neighbour connectivity; every board carries a runner-up
red component exactly one disc smaller, so the count comparison is
load-bearing.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import CANVAS_SIZE, DEFAULT_FPS
from data.common.pipeline.board_records import (
    BoardChoiceRecord,
    write_board_choice_record,
)
from data.common.pipeline.board_rendering import (
    PIECE_BLUE,
    PIECE_BLUE_OUT,
    PIECE_RED,
    PIECE_RED_OUT,
    draw_hex_cell,
    load_font,
)
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.scalar_lists import (
    format_nonnegative_integer_answer,
)
from data.common.video_encoding import encode_rgb_frames_to_webm

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the gold rings",
    static_nouns="the hexagonal grid, every red and blue disc, background, and camera",
    hold_noun="the final board with every ring fully opaque",
)


_BG = (252, 248, 240)
_HEX_FILL = (235, 220, 188)
_HEX_OUT = (140, 110, 80)


_INPUT_DESC = (
    "The single input image shows a rhombus of 7 by 7 hexagonal cells; every hexagon shares its complete sides with "
    "its edge-neighbors. Some cells hold red discs, some hold blue discs, and the rest are empty."
)
_RULE_SUMMARY = (
    "Two cells connect exactly when they share a complete hexagon edge; touching only at a corner does not connect, "
    "and blue or empty cells block every connection. A red component is a maximal set of red discs joined through "
    "such shared-edge steps. The task is the number of discs in the largest red component, and the visible board has "
    "exactly one largest component."
)
_TI2V_OUTPUT = (
    "Fade one complete fixed-radius gold ring in just outside every red disc of the largest component, all rings "
    "together and each at its full footprint from its first visible frame — none growing, tracing around a disc, or "
    "appearing one by one. Mark no other disc and never trace a hexagon edge."
)
_SOLUTION_IMAGE_OUTPUT = (
    "Render the board with one complete fixed-radius gold circle just outside every red disc of the largest "
    "component; mark nothing else and never trace a hexagon edge."
)
STATE_CODES = {None: 0, "R": 1, "B": 2}
HIGHLIGHT_RED_CHAIN_CODE = 3


def _encode_board(board):
    return [[STATE_CODES[cell] for cell in row] for row in board]


def _encode_highlighted_chain(board, chain_cells):
    out = _encode_board(board)
    for row, col in chain_cells:
        out[row][col] = HIGHLIGHT_RED_CHAIN_CODE
    return out


class HexChainCountGenerator(AbstractPuzzleGenerator[BoardChoiceRecord]):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        video_fps: int = DEFAULT_FPS,
        video: bool = False,
    ) -> None:
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.video_fps = video_fps
        self.video = video
        self._counter = 0
        self.size = 7
        # Pointy-top hex radius; the axial rhombus just fits the canvas width.
        self.hex_r = 28

    def record_to_dict(self, record):
        if hasattr(record, "to_dict"):
            return record.to_dict()
        return dict(record)

    def _hex_centre(self, row, col):
        # Pointy-top axial coordinates.
        # Hex spacing: width = sqrt(3) * R, height = 2*R. Vertical spacing = 1.5*R.
        R = self.hex_r
        w = math.sqrt(3) * R
        # Centre rhombus.
        # Offset per row.
        cx = 90 + col * w + row * (w / 2)
        cy = 90 + row * (1.5 * R)
        return cx, cy

    def _hex_neighbours(self, row, col):
        out = []
        for dr, dc in ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                out.append((nr, nc))
        return out

    def _largest_chain(self, board, colour):
        components = self._components(board, colour)
        if not components:
            return 0, []
        best_cells = max(components, key=len)
        return len(best_cells), list(best_cells)

    def _components(self, board, colour):
        """Return every whole component in stable position order.

        The component sets are computed globally.  Their stable ordering is
        only a display label and never a traversal or teacher search order.
        """

        seen = [[False] * self.size for _ in range(self.size)]
        components = []
        for r in range(self.size):
            for c in range(self.size):
                if seen[r][c] or board[r][c] != colour:
                    continue
                stack = [(r, c)]
                seen[r][c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for nr, nc in self._hex_neighbours(rr, cc):
                        if not seen[nr][nc] and board[nr][nc] == colour:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
                components.append(tuple(sorted(cells)))
        return sorted(components, key=lambda cells: cells[0])

    def _grow_red_component(self, board, target_len, forbidden):
        """Grow one connected red blob avoiding ``forbidden``; None on failure."""

        open_cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if board[r][c] is None and (r, c) not in forbidden
        ]
        if not open_cells:
            return None
        start = self.rng.choice(open_cells)
        cells = [start]
        board[start[0]][start[1]] = "R"
        while len(cells) < target_len:
            frontier = [
                (nr, nc)
                for base in cells
                for nr, nc in self._hex_neighbours(*base)
                if board[nr][nc] is None and (nr, nc) not in forbidden
            ]
            if not frontier:
                for r, c in cells:
                    board[r][c] = None
                return None
            rr, cc = self.rng.choice(frontier)
            board[rr][cc] = "R"
            cells.append((rr, cc))
        return cells

    def _construct(self):
        """Build a board whose counting comparison is genuinely load-bearing.

        The largest red component (4-6 cells) is accompanied by a runner-up
        red component exactly one cell smaller, plus one or two small red
        components, so the answer requires comparing real component sizes
        rather than spotting one obvious blob.  Blue discs include cells
        adjacent to red components, visibly blocking connections.
        """

        for _ in range(800):
            board = [[None] * self.size for _ in range(self.size)]
            target_len = self.rng.randint(4, 6)
            main_cells = self._grow_red_component(board, target_len, forbidden=set())
            if main_cells is None:
                continue
            # The runner-up component must not touch the main component.
            halo = {
                neighbour
                for cell in main_cells
                for neighbour in self._hex_neighbours(*cell)
            } | set(main_cells)
            runner_cells = self._grow_red_component(board, target_len - 1, forbidden=halo)
            if runner_cells is None:
                continue
            # One or two isolated single red cells.
            for _ in range(self.rng.randint(1, 2)):
                for _ in range(30):
                    rr = self.rng.randrange(self.size)
                    cc = self.rng.randrange(self.size)
                    if board[rr][cc] is not None:
                        continue
                    if any(board[nr][nc] == "R" for nr, nc in self._hex_neighbours(rr, cc)):
                        continue
                    board[rr][cc] = "R"
                    break
            # Blue discs: at least two must sit directly against red cells so
            # the blue-blocks-connection rule is visibly load-bearing.
            red_adjacent_empty = sorted({
                (nr, nc)
                for r in range(self.size)
                for c in range(self.size)
                if board[r][c] == "R"
                for nr, nc in self._hex_neighbours(r, c)
                if board[nr][nc] is None
            })
            self.rng.shuffle(red_adjacent_empty)
            blue_budget = self.rng.randint(3, 6)
            placed_blue = 0
            for rr, cc in red_adjacent_empty[:2]:
                board[rr][cc] = "B"
                placed_blue += 1
            while placed_blue < blue_budget:
                for _ in range(30):
                    rr = self.rng.randrange(self.size)
                    cc = self.rng.randrange(self.size)
                    if board[rr][cc] is None:
                        board[rr][cc] = "B"
                        break
                placed_blue += 1
            # Verify the intended unique maximum and intact runner-up.
            components = self._components(board, "R")
            sizes = sorted((len(component) for component in components), reverse=True)
            if len(sizes) < 3 or sizes[0] != target_len or sizes[1] != target_len - 1:
                continue
            if sizes.count(target_len) != 1:
                continue
            longest, longest_cells = self._largest_chain(board, "R")
            if longest != target_len:
                continue
            return board, longest, longest_cells
        raise RuntimeError("Could not construct hex_chain_count scenario.")

    def _board_offsets(self) -> Tuple[float, float]:
        """Offsets that centre the rhombus; shared with the whole-video verifier."""

        all_centres = [self._hex_centre(r, c) for r in range(self.size) for c in range(self.size)]
        xs = [x for x, _ in all_centres]
        ys = [y for _, y in all_centres]
        offset_x = (CANVAS_SIZE - (max(xs) - min(xs))) / 2 - min(xs)
        offset_y = (CANVAS_SIZE - (max(ys) - min(ys))) / 2 - min(ys)
        return offset_x, offset_y

    def _render_board(
        self,
        board,
        *,
        highlight: Optional[List[Tuple[int, int]]] = None,
        highlight_progress: float = 1.0,
        component_groups: Optional[List[List[Tuple[int, int]]]] = None,
        show_component_sizes: bool = False,
    ):
        supersampled, draw = new_scaled_canvas(
            CANVAS_SIZE, CANVAS_SIZE, DOCTRINE_SUPERSAMPLE, _BG
        )
        factor = draw.factor
        offset_x, offset_y = self._board_offsets()
        for r in range(self.size):
            for c in range(self.size):
                cx, cy = self._hex_centre(r, c)
                cx += offset_x
                cy += offset_y
                fill = _HEX_FILL
                outline = _HEX_OUT
                draw_hex_cell(draw, centre=(cx, cy), radius=self.hex_r,
                                fill=fill, outline=outline, width=2)
                p = board[r][c]
                if p is not None:
                    body = PIECE_RED if p == "R" else PIECE_BLUE
                    edge = PIECE_RED_OUT if p == "R" else PIECE_BLUE_OUT
                    rad = int(self.hex_r * 0.6)
                    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                                  fill=body, outline=edge, width=2)
        progress = max(0.0, min(1.0, float(highlight_progress)))
        if highlight and progress > 0.0:
            overlay = Image.new("RGBA", supersampled.size, (0, 0, 0, 0))
            overlay_draw = ScaledDraw(ImageDraw.Draw(overlay), factor)
            for r, c in highlight:
                cx, cy = self._hex_centre(r, c)
                cx += offset_x
                cy += offset_y
                # The complete circle exists from the first visible frame at
                # one fixed centre, radius, and width. Every target ring shares
                # one local opacity; no ring grows radially, hard-appears as a
                # pale outline, or traces a hexagon.
                smooth = progress * progress * (3.0 - 2.0 * progress)
                deep = (232, 184, 45)
                ring_radius = int(round(self.hex_r * 0.79))
                overlay_draw.ellipse(
                    [
                        cx - ring_radius,
                        cy - ring_radius,
                        cx + ring_radius,
                        cy + ring_radius,
                    ],
                    outline=(*deep, 255),
                    width=4,
                )
            alpha = overlay.getchannel("A").point(lambda value: int(round(value * smooth)))
            overlay.putalpha(alpha)
            supersampled = Image.alpha_composite(supersampled.convert("RGBA"), overlay).convert("RGB")
            draw = ScaledDraw(ImageDraw.Draw(supersampled), factor)
        if component_groups:
            palette = (
                (49, 138, 185),
                (111, 78, 174),
                (50, 155, 112),
                (210, 114, 54),
                (184, 68, 117),
                (70, 145, 151),
            )
            badge_font = load_font(max(13, int(self.hex_r * 0.56)) * factor)
            for index, component in enumerate(component_groups):
                colour = palette[index % len(palette)]
                for r, c in component:
                    cx, cy = self._hex_centre(r, c)
                    cx += offset_x
                    cy += offset_y
                    ring_radius = int(round(self.hex_r * 0.77))
                    draw.ellipse(
                        (cx - ring_radius, cy - ring_radius, cx + ring_radius, cy + ring_radius),
                        outline=colour,
                        width=3,
                    )
                if show_component_sizes:
                    anchor_row, anchor_col = min(component)
                    cx, cy = self._hex_centre(anchor_row, anchor_col)
                    cx += offset_x + self.hex_r * 0.68
                    cy += offset_y - self.hex_r * 0.68
                    radius = 11
                    draw.ellipse(
                        (cx - radius, cy - radius, cx + radius, cy + radius),
                        fill=(255, 255, 255),
                        outline=colour,
                        width=2,
                    )
                    draw.text(
                        (cx, cy),
                        str(len(component)),
                        fill=colour,
                        font=badge_font,
                        anchor="mm",
                    )
        return finish_scaled_canvas(supersampled, (CANVAS_SIZE, CANVAS_SIZE))

    def _write_video(
        self,
        board,
        component_cells: List[Tuple[int, int]],
        path: Path,
    ) -> int:
        initial = np.asarray(self._render_board(board).convert("RGB"), dtype=np.uint8)
        terminal = np.asarray(
            self._render_board(board, highlight=component_cells).convert("RGB"),
            dtype=np.uint8,
        )
        frames = [initial.copy() for _ in range(6)]
        for step in range(1, 9):
            frame = self._render_board(
                board,
                highlight=component_cells,
                highlight_progress=step / 8.0,
            )
            frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
        frames.extend(terminal.copy() for _ in range(8))
        while (len(frames) - 1) % 4:
            frames.append(terminal.copy())
        if not encode_rgb_frames_to_webm(
            frames,
            path,
            fps=self.video_fps,
            extra_args=["-g", "1", "-bf", "0"],
        ):
            return 0
        return len(frames)

    def create_puzzle(self) -> BoardChoiceRecord:
        sample_id = f"hex_chain_count-{self._counter:05d}"
        self._counter += 1
        board, longest, longest_cells = self._construct()
        component_cells = sorted(longest_cells)
        red_components = [list(component) for component in self._components(board, "R")]
        if sum(len(component) == longest for component in red_components) != 1:
            raise RuntimeError("Hex source must have one unique largest red component.")
        puzzle_image = self._render_board(board)
        solution_image = self._render_board(board, highlight=component_cells)

        prompts = {
            "ti2v_prompt": build_sectioned_prompt(
                _INPUT_DESC,
                _TI2V_OUTPUT,
                mode="ti2v",
                rule_description=_RULE_SUMMARY,
                continuity_paragraph=_TI2V_CONTINUITY,
            ),
        }
        payload = {
            "semantic_state": {"board": [list(row) for row in board]},
            "board": [list(row) for row in board],
            "longest_red": longest,
            "longest_cells": [list(c) for c in component_cells],
            "reference_component_sizes": [len(component) for component in red_components],
            "initial_grid": _encode_board(board),
            "state_grid": _encode_highlighted_chain(board, component_cells),
            "state_trace": [_encode_board(board), _encode_highlighted_chain(board, component_cells)],
            "process_state_trace": [_encode_board(board), _encode_highlighted_chain(board, component_cells)],
        }
        answer_text = format_nonnegative_integer_answer(longest)
        return write_board_choice_record(
            generator=self,
            sample_id=sample_id,
            task_name="hex_chain_count",
            task_family="board_game",
            question_description="hex_cell_longest_red_component",
            puzzle_image=puzzle_image,
            solution_image=solution_image,
            option_tiles=[],
            correct_option_index=None,
            prompts=prompts,
            structured_payload=payload,
            question_text="What is the size of the unique largest red shared-edge component?",
            video_states=[puzzle_image, solution_image],
            video_verifier_states=[puzzle_image, solution_image],
            video_writer=lambda path: self._write_video(board, component_cells, path),
            single_input=True,
            direct_answer_text=answer_text,
            direct_answer_value=str(longest),
        )


__all__ = ["HexChainCountGenerator"]
