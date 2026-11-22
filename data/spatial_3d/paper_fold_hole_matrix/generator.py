"""Paper fold, punch, and reverse-unfold: the Spatial 3D exemplar task.

An 8x8 sheet is folded in half twice (each fold shown as one panel with a
bold half-onto-half arrow), 1-3 cells of the final packet are punched, and
the task recovers the full unfolded hole matrix by reversing the folds last
fold first — every reverse fold keeps all existing holes and adds their
mirror images across that crease, so the hole count grows P -> 2P -> 4P.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from PIL import Image

from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract
from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
    format_xy_position,
)
from data.common.pipeline.structured_grid import load_font, matrix_to_json
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.spatial_3d.core.scene_base import Spatial3DScenePattern

_GRID_SIZE = 8
_FOLD_COUNT = 2
_MIN_PUNCHES = 1
_MAX_PUNCHES = 3

BG = (246, 248, 251)
CARD = (255, 255, 253)
CARD_EDGE = (216, 221, 226)
PANEL_FILL = (249, 251, 255)
PANEL_EDGE = (210, 218, 230)
SHEET_LINE = (170, 176, 184)
SHEET_FRAME = (90, 96, 108)
HOLE_FILL = (46, 54, 68)
PUNCH_FILL = (209, 46, 46)
PUNCH_EDGE = (120, 22, 22)
ARROW_COLOR = (57, 115, 182)
INK = (40, 47, 56)
AXIS_LABEL_COLOR = (30, 90, 220)

_FOLD_NAMES = ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")
_FOLD_PROSE = {
    "left_to_right": "left half onto the right half",
    "right_to_left": "right half onto the left half",
    "top_to_bottom": "top half onto the bottom half",
    "bottom_to_top": "bottom half onto the top half",
}


def _blank_matrix(rows: int, cols: int) -> List[List[int]]:
    return [[0 for _ in range(cols)] for _ in range(rows)]


def _matrix_shape(matrix: Sequence[Sequence[object]]) -> tuple[int, int]:
    rows = len(matrix)
    return rows, (len(matrix[0]) if rows else 0)


def final_folded_shape(size: int, fold_sequence: Sequence[str]) -> tuple[int, int] | None:
    rows = cols = int(size)
    for fold_name in fold_sequence:
        if fold_name in {"left_to_right", "right_to_left"}:
            if cols < 2 or cols % 2:
                return None
            cols //= 2
        elif fold_name in {"top_to_bottom", "bottom_to_top"}:
            if rows < 2 or rows % 2:
                return None
            rows //= 2
        else:
            return None
    return rows, cols


def _apply_fold_to_mapping(
    mapping: Sequence[Sequence[Iterable[tuple[int, int]]]],
    fold_name: str,
) -> list[list[list[tuple[int, int]]]]:
    rows, cols = _matrix_shape(mapping)
    if fold_name == "left_to_right":
        half = cols // 2
        return [
            [list(mapping[row][half + col]) + list(mapping[row][half - 1 - col]) for col in range(half)]
            for row in range(rows)
        ]
    if fold_name == "right_to_left":
        half = cols // 2
        return [
            [list(mapping[row][col]) + list(mapping[row][cols - 1 - col]) for col in range(half)]
            for row in range(rows)
        ]
    if fold_name == "top_to_bottom":
        half = rows // 2
        return [
            [list(mapping[half + row][col]) + list(mapping[half - 1 - row][col]) for col in range(cols)]
            for row in range(half)
        ]
    if fold_name == "bottom_to_top":
        half = rows // 2
        return [
            [list(mapping[row][col]) + list(mapping[rows - 1 - row][col]) for col in range(cols)]
            for row in range(half)
        ]
    raise ValueError(f"Unsupported fold: {fold_name}")


def _fold_mapping_states(size: int, fold_sequence: Sequence[str]) -> list[list[list[list[tuple[int, int]]]]]:
    mapping: list[list[list[tuple[int, int]]]] = [[[(row, col)] for col in range(size)] for row in range(size)]
    states = [[[list(cell) for cell in row] for row in mapping]]
    for fold_name in fold_sequence:
        mapping = _apply_fold_to_mapping(mapping, fold_name)
        states.append([[list(cell) for cell in row] for row in mapping])
    return states


def unfold_holes(size: int, fold_sequence: Sequence[str], punched_cells: Sequence[tuple[int, int]]) -> List[List[int]]:
    mapping = _fold_mapping_states(size, fold_sequence)[-1]
    unfolded = _blank_matrix(size, size)
    for row, col in punched_cells:
        for target_row, target_col in mapping[row][col]:
            unfolded[target_row][target_col] = 1
    return unfolded


def unfold_reveal_states(
    size: int,
    fold_sequence: Sequence[str],
    punched_cells: Sequence[tuple[int, int]],
) -> List[List[List[int]]]:
    """Hole matrices after the seed and after each reverse fold (last first)."""

    mapping_states = _fold_mapping_states(size, fold_sequence)
    final_mapping = mapping_states[-1]
    final_rows, final_cols = _matrix_shape(final_mapping)
    targets: set[tuple[int, int]] = set()
    for row, col in punched_cells:
        if not (0 <= row < final_rows and 0 <= col < final_cols):
            raise ValueError("punched cell lies outside the final folded sheet")
        targets.update(final_mapping[row][col])
    reveal_states: List[List[List[int]]] = []
    for mapping in reversed(mapping_states):
        state = _blank_matrix(size, size)
        for row in mapping:
            for original in row:
                if not targets.intersection(original):
                    continue
                representative_row, representative_col = original[0]
                state[representative_row][representative_col] = 1
        reveal_states.append(state)
    return reveal_states


def replay_paper_fold(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Independent forward replay from the visible fold program only."""

    size = int(source["grid_size"])
    folds = [str(value) for value in source["fold_sequence"]]
    punched = [(int(cell[0]), int(cell[1])) for cell in source["punched_cells"]]
    if size != _GRID_SIZE or len(folds) != _FOLD_COUNT:
        raise ValueError("paper-fold source must be one 8x8 sheet with two folds")
    if any(name not in _FOLD_NAMES for name in folds):
        raise ValueError("paper-fold source has an unknown fold direction")
    shape = final_folded_shape(size, folds)
    if shape is None:
        raise ValueError("paper-fold source folds are not executable")
    rows, cols = shape
    if not (_MIN_PUNCHES <= len(punched) <= _MAX_PUNCHES) or len(set(punched)) != len(punched):
        raise ValueError("paper-fold source punch set is invalid")
    if any(not (0 <= r < rows and 0 <= c < cols) for r, c in punched):
        raise ValueError("paper-fold punch lies outside the folded sheet")
    unfolded = unfold_holes(size, folds, punched)
    reveals = unfold_reveal_states(size, folds, punched)
    states = [_blank_matrix(size, size), *reveals]
    counts = [sum(sum(row) for row in state) for state in states]
    if counts != [0, len(punched), len(punched) * 2, len(punched) * 4]:
        raise ValueError("paper-fold replay does not double holes per reverse fold")
    if reveals[-1] != unfolded:
        raise ValueError("paper-fold replay states do not end at the unfolded matrix")
    return {
        "grid_size": size,
        "fold_sequence": folds,
        "punched_cells": [list(cell) for cell in punched],
        "final_folded_shape": [rows, cols],
        "unfolded_hole_matrix": unfolded,
        "states": states,
    }


class PaperFoldHoleMatrixPattern(Spatial3DScenePattern):
    task_name = "paper_fold_hole_matrix"
    task_family = "spatial_3d"
    task_mechanic = "reverse_fold_mirror_unfold"
    motion = False
    verifier_profile = "paper_fold_visible_program_staged_fade_v4"

    # ------------------------------------------------------------- sampling

    def _sample_task(self) -> Dict[str, Any]:
        punch_ranges = {
            "min": (1, 1),
            "typical": (2, 2),
            "max": (3, 3),
        }.get(self.difficulty or "", (_MIN_PUNCHES, _MAX_PUNCHES))
        size = _GRID_SIZE
        folds: list[str] = []
        rows = cols = size
        for _ in range(_FOLD_COUNT):
            candidates: list[str] = []
            if cols % 2 == 0 and cols >= 4:
                candidates.extend(["left_to_right", "right_to_left"])
            if rows % 2 == 0 and rows >= 4:
                candidates.extend(["top_to_bottom", "bottom_to_top"])
            fold_name = self.rng.choice(candidates)
            folds.append(fold_name)
            if fold_name in {"left_to_right", "right_to_left"}:
                cols //= 2
            else:
                rows //= 2
        total_cells = rows * cols
        punch_count = self.rng.randint(punch_ranges[0], min(punch_ranges[1], total_cells))
        punched = sorted(
            self.rng.sample([(row, col) for row in range(rows) for col in range(cols)], k=punch_count)
        )
        unfolded = unfold_holes(size, folds, punched)
        if sum(sum(row) for row in unfolded) != punch_count * (2 ** _FOLD_COUNT):
            raise RuntimeError("paper_fold_hole_matrix produced an inconsistent hole count")
        reveals = unfold_reveal_states(size, folds, punched)
        states = [_blank_matrix(size, size), *reveals]
        task = {
            "size": size,
            "folds": folds,
            "punched": punched,
            "unfolded": unfolded,
            "states": states,
            "final_shape": final_folded_shape(size, folds),
        }
        replay_paper_fold(
            {"grid_size": size, "fold_sequence": folds, "punched_cells": [list(c) for c in punched]}
        )
        return task

    # ------------------------------------------------------------- geometry

    @staticmethod
    def answer_grid_geometry(size: int = _GRID_SIZE) -> tuple[int, int, int, int]:
        """(left, top, cell, extent) of the workspace grid in 512-space.

        Single geometric truth for the renderer and every verifier decode.
        """

        cell = 208 // size
        extent = cell * size
        left = (512 - extent) // 2 + 10
        top = 286
        return left, top, cell, extent

    @staticmethod
    def panel_boxes() -> tuple[tuple[int, int, int, int], ...]:
        return ((14, 44, 174, 226), (176, 44, 336, 226), (338, 44, 498, 226))

    # ------------------------------------------------------------- rendering

    def _sheet_cells_before_panel(self, size: int, folds: Sequence[str], panel_index: int) -> tuple[int, int]:
        rows = cols = size
        for fold_name in folds[:panel_index]:
            if fold_name in {"left_to_right", "right_to_left"}:
                cols //= 2
            else:
                rows //= 2
        return rows, cols

    def _draw_panel(
        self,
        draw,
        *,
        box: tuple[int, int, int, int],
        ordinal: str,
        rows: int,
        cols: int,
        arrow_fold: str | None,
        punches: Sequence[tuple[int, int]] | None,
    ) -> None:
        left, top, right, bottom = box
        draw.rounded_rectangle(box, radius=12, fill=PANEL_FILL, outline=PANEL_EDGE, width=2)
        factor = draw.factor
        ordinal_font = load_font(15 * factor)
        draw.text((left + 10, top + 8), ordinal, fill=INK, font=ordinal_font)
        inner_top = top + 30
        available_w = right - left - 22
        available_h = bottom - inner_top - 12
        cell = max(10, min(available_w // max(1, cols), available_h // max(1, rows)))
        grid_w, grid_h = cell * cols, cell * rows
        grid_left = left + (right - left - grid_w) // 2
        grid_top = inner_top + (available_h - grid_h) // 2 + 6
        for row in range(rows):
            for col in range(cols):
                draw.rectangle(
                    (grid_left + col * cell, grid_top + row * cell,
                     grid_left + (col + 1) * cell, grid_top + (row + 1) * cell),
                    fill=(255, 255, 255), outline=SHEET_LINE, width=1,
                )
        draw.rectangle((grid_left, grid_top, grid_left + grid_w, grid_top + grid_h), outline=SHEET_FRAME, width=2)
        if punches:
            for row, col in punches:
                cx = grid_left + col * cell + cell / 2
                cy = grid_top + row * cell + cell / 2
                radius = max(3, cell // 5)
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=PUNCH_FILL, outline=PUNCH_EDGE, width=1)
        if arrow_fold:
            self._draw_fold_arrow(draw, arrow_fold, grid_left, grid_top, grid_w, grid_h)

    @staticmethod
    def _draw_fold_arrow(draw, fold_name: str, grid_left: int, grid_top: int, grid_w: int, grid_h: int) -> None:
        cx, cy = grid_left + grid_w / 2, grid_top + grid_h / 2
        span_x, span_y = grid_w * 0.30, grid_h * 0.30
        if fold_name == "left_to_right":
            start, end = (cx - span_x, cy), (cx + span_x, cy)
        elif fold_name == "right_to_left":
            start, end = (cx + span_x, cy), (cx - span_x, cy)
        elif fold_name == "top_to_bottom":
            start, end = (cx, cy - span_y), (cx, cy + span_y)
        else:
            start, end = (cx, cy + span_y), (cx, cy - span_y)
        draw.line((*start, *end), fill=ARROW_COLOR, width=4)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        # Tail dot marks the moving half; the head points at the landing half.
        draw.ellipse((start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4), fill=ARROW_COLOR)
        for spread in (math.radians(150.0), math.radians(-150.0)):
            tip_angle = angle + spread
            draw.line(
                (end[0], end[1], end[0] + 11 * math.cos(tip_angle), end[1] + 11 * math.sin(tip_angle)),
                fill=ARROW_COLOR,
                width=4,
            )

    def _render_scene(self, task: Mapping[str, Any], fill_matrix: Sequence[Sequence[int]]) -> Image.Image:
        size = int(task["size"])
        folds = list(task["folds"])
        supersample = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(512, 512, supersample, background=BG)
        draw.rounded_rectangle((8, 8, 504, 504), radius=20, fill=CARD, outline=CARD_EDGE, width=2)

        boxes = self.panel_boxes()
        for panel_index in range(3):
            rows, cols = self._sheet_cells_before_panel(size, folds, panel_index)
            self._draw_panel(
                draw,
                box=boxes[panel_index],
                ordinal=str(panel_index + 1),
                rows=rows,
                cols=cols,
                arrow_fold=folds[panel_index] if panel_index < len(folds) else None,
                punches=task["punched"] if panel_index == 2 else None,
            )

        left, top, cell, extent = self.answer_grid_geometry(size)
        label_font = load_font(11 * draw.factor)
        for col in range(size):
            draw.text((left + col * cell + cell / 2, top - 10), str(col + 1), fill=AXIS_LABEL_COLOR, font=label_font, anchor="mm")
        for row in range(size):
            draw.text((left - 12, top + row * cell + cell / 2), str(row + 1), fill=AXIS_LABEL_COLOR, font=label_font, anchor="mm")
        for row in range(size):
            for col in range(size):
                fill = HOLE_FILL if fill_matrix[row][col] else (255, 255, 255)
                draw.rectangle(
                    (left + col * cell, top + row * cell, left + (col + 1) * cell, top + (row + 1) * cell),
                    fill=fill, outline=SHEET_LINE, width=1,
                )
        draw.rectangle((left, top, left + extent, top + extent), outline=SHEET_FRAME, width=2)
        return finish_scaled_canvas(image, (512, 512))

    def _stage_count(self, task: Mapping[str, Any]) -> int:
        return len(task["states"])

    def _render_stage(self, task: Mapping[str, Any], index: int) -> Image.Image:
        return self._render_scene(task, task["states"][index])

    # ------------------------------------------------------------- semantics

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        return matrix_to_json(task["unfolded"])

    def _question(self, task: Mapping[str, Any]) -> str:
        return "Recover the full unfolded hole matrix implied by the fold panels and the punched folded sheet."

    @staticmethod
    def _cells_prose(cells: Sequence[tuple[int, int]]) -> str:
        parts = [
            format_xy_position(cell_to_xy((row, col)))
            for row, col in cells
        ]
        if len(parts) == 1:
            return parts[0]
        return "; ".join(parts[:-1]) + " and " + parts[-1]

    def _region_of_mapping(self, mapping) -> tuple[tuple[int, int], tuple[int, int]]:
        rows = [cell[0][0] for line in mapping for cell in line]
        cols = [cell[0][1] for line in mapping for cell in line]
        return (min(rows), max(rows)), (min(cols), max(cols))

    def _crease_prose(self, task: Mapping[str, Any], fold_number: int) -> tuple[str, str]:
        """(axis word, crease sentence fragment) for reversing one fold."""

        mapping_states = _fold_mapping_states(int(task["size"]), task["folds"])
        (row_lo, row_hi), (col_lo, col_hi) = self._region_of_mapping(mapping_states[fold_number - 1])
        fold_name = task["folds"][fold_number - 1]
        if fold_name in {"left_to_right", "right_to_left"}:
            mid = col_lo + (col_hi - col_lo + 1) // 2
            return "vertical", f"between columns {mid} and {mid + 1}"
        mid = row_lo + (row_hi - row_lo + 1) // 2
        return "horizontal", f"between rows {mid} and {mid + 1}"

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[Any, str]]:
        folds = list(task["folds"])
        states = task["states"]
        cells_of = [
            sorted((row, col) for row, line in enumerate(state) for col, value in enumerate(line) if value)
            for state in states
        ]
        mapping_states = _fold_mapping_states(int(task["size"]), folds)
        (row_lo, row_hi), (col_lo, col_hi) = self._region_of_mapping(mapping_states[-1])
        seed_text = (
            f"Panel 3 is the folded packet that stayed put through both folds; on the full sheet it covers "
            f"rows {row_lo + 1}-{row_hi + 1}, columns {col_lo + 1}-{col_hi + 1}. Each punch goes through "
            f"every layer, so I first mark the punched cells of the packet itself on the workspace: "
            f"{self._cells_prose(cells_of[1])}."
        )
        key_states: list[tuple[Any, str]] = [(1, seed_text)]
        for reverse_index, state_index in enumerate(range(2, len(states))):
            fold_number = len(folds) - reverse_index
            fold_name = folds[fold_number - 1]
            added = sorted(set(cells_of[state_index]) - set(cells_of[state_index - 1]))
            axis, crease = self._crease_prose(task, fold_number)
            if reverse_index == 0:
                text = (
                    f"Unfolding reverses the folds last first, so fold {fold_number} "
                    f"({_FOLD_PROSE[fold_name]}) opens first. Its {axis} crease runs {crease}; every marked "
                    f"hole stays and gains its mirror across that crease, so new holes land at "
                    f"{self._cells_prose(added)}, doubling the count to {len(cells_of[state_index])}."
                )
            else:
                text = (
                    f"With fold {fold_number + 1} open, reversing fold {fold_number} "
                    f"({_FOLD_PROSE[fold_name]}) finishes the unfold. Its {axis} crease runs {crease}; "
                    f"keeping every hole and mirroring across it adds holes at {self._cells_prose(added)}."
                )
            if state_index == len(states) - 1:
                text += (
                    f" All {len(cells_of[state_index])} holes are now placed, and the workspace shows the "
                    f"complete pattern."
                )
            key_states.append((state_index, text))
        return key_states

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        size = int(task["size"])
        folds = list(task["folds"])
        rows, cols = task["final_shape"]
        return {
            "semantic_state": {
                "grid_size": size,
                "fold_sequence": folds,
                "punched_cells": [list(cell) for cell in task["punched"]],
            },
            "unfolded_hole_matrix": [list(row) for row in task["unfolded"]],
            "unfolding_states": [list(map(list, state)) for state in task["states"]],
            "final_folded_shape": [rows, cols],
            "multiple_solution_policy": "exactly one complete binary matrix follows from the visible folds and punches",
            "difficulty_evidence": {
                "fold_count": len(folds),
                "reverse_dependency_length": len(folds),
                "punch_count": len(task["punched"]),
                "final_hole_count": int(sum(sum(row) for row in task["unfolded"])),
                "uses_both_axes": any(name in {"left_to_right", "right_to_left"} for name in folds)
                and any(name in {"top_to_bottom", "bottom_to_top"} for name in folds),
            },
            "answer_schema": {
                "schema_version": "binary_square_matrix_v2",
                "kind": "json_binary_grid",
                "parser_profile": "strict_json_binary_grid_with_period",
                "format_template": "Answer: [[<0-or-1>, ...], ...].",
                "format_example": "Answer: [[0, 1, ...], [1, 0, ...], ...].",
                "judged_artifacts": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "verifier_contract": {
                "profile": self.verifier_profile,
                "semantic_source": "visible ordered fold arrows and punched cells of the final folded sheet",
                "strict_binary_correctness": True,
                "strict_text_parser": "strict_json_binary_grid_with_period",
                "untrusted_redundant_fields": [
                    "unfolded_hole_matrix",
                    "unfolding_states",
                    "final_folded_shape",
                ],
            },
        }

    def _glyph_inventory(self, task: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
        labels = [str(value + 1) for value in range(int(task["size"]))]
        inventory = {
            "schema_version": "paper_fold_rendered_glyph_inventory_v2",
            "fixed_labels": sorted({"1", "2", "3", *labels}, key=lambda v: (len(v), v)),
            "dynamic_labels": [],
            "all_rendered_text": sorted({"1", "2", "3", *labels}, key=lambda v: (len(v), v)),
        }
        schema = {
            "fixed_labels": "panel ordinals 1-3 and the 1-based workspace row/column numbers only",
            "dynamic_labels": "none: fold direction is a drawn arrow, not text",
            "exclusions": "no task title, rule prose, coordinates, status sentence, or answer banner",
        }
        return inventory, schema

    # -------------------------------------------------------------- contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        folds = list(task["folds"])
        fold_story = " then ".join(
            f"the arrow in panel {index + 1} folds the {_FOLD_PROSE[name]}" for index, name in enumerate(folds)
        )
        return VisualReasoningContract(
            input_description=(
                "The single image shows three sheet panels numbered 1 to 3 along the top and one blank full-size "
                "workspace grid below. Panel 1 is the full sheet; "
                f"{fold_story}, so panels 2 and 3 show the sheet after each fold. Panel 3 is the final folded "
                f"packet with its punched cells. {GRID_XY_CONVENTION_SENTENCE}"
            ),
            symbol_legend=(
                "Thin gray lines are fixed paper-cell boundaries.",
                "A bold blue arrow starts with a dot on the half that moves and points at the half it folds onto.",
                "Each red dot on panel 3 is one punch through every paper layer at that cell.",
                "The blue numbers along the workspace edges instantiate that grid-position convention.",
            ),
            rules=(
                "Each fold lays its moving half exactly onto the other half, layer over layer.",
                "Every red punch goes through all layers of the folded packet at that cell.",
                "To recover the full pattern, reverse the folds one at a time, last fold first.",
                "Reversing one fold keeps every existing hole and adds the mirror image of every existing hole "
                "across that fold's crease; holes never move, rotate, or disappear.",
                "After both folds are reversed, the workspace holds the complete hole pattern of the full sheet.",
            ),
            modality_outputs={
                "ti2v": (
                    "The workspace fills in three commits: first the seed holes copied from the punched packet, "
                    "then the mirror holes of each reverse fold in last-to-first order. At each commit all newly "
                    "punched cells darken together, fading from transparent to fully dark inside their exact "
                    "cells while every earlier hole stays unchanged."
                ),
                "solution_image": (
                    "Fill the workspace with the complete unfolded pattern: a punched cell is filled dark and "
                    "every other cell stays white; the three top panels stay as given."
                ),
            },
            preservation_constraints=(
                "the three top panels, the red punch dots, the workspace grid lines and its edge numbers, and "
                "the background unchanged; only workspace cells may darken.",
            ),
            transition_constraints=(
                "Show each commit as one synchronized element-local fade of all its new hole cells at fixed "
                "geometry; never grow a hole from its center, move or remove an earlier hole, or dissolve the "
                "full frame.",
            ),
            ti2v_continuity={
                "moving_nouns": "the dark hole cells appearing in the workspace",
                "static_nouns": "the three top panels, the red punch dots, the workspace grid and its edge "
                "numbers, the background, and the camera",
                "hold_noun": "the complete unfolded hole pattern",
            },
            answer=AnswerContract(
                kind="grid",
                format_template="Answer: [[<0-or-1>, ...], ...].",
                format_example="Answer: [[0, 1, ...], [1, 0, ...], ...].",
                parser_profile="strict_json_binary_grid_with_period",
                verifier_profile=self.verifier_profile,
            ),
            render_profile="paper_fold_staged_workspace_v4",
        )


__all__ = [
    "PaperFoldHoleMatrixPattern",
    "final_folded_shape",
    "replay_paper_fold",
    "unfold_holes",
    "unfold_reveal_states",
]
