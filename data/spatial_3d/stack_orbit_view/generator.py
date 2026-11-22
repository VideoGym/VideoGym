"""Stack orbit view: what the colored block stacks look like from the far side.

Colored unit cubes stand in stacks on a 1-based labeled 3x3 floor; a bold
blue arrow looks back at the scene from the opposite or far-left side.  The
answer is the letter grid of the nearest visible cube colour along each view
ray - which forces both occlusion reasoning (nearest stack wins) and the
left-right mirror that a viewer on the far side experiences.  Sampling gates
keep every cube individually visible in the input picture, so the whole
structure is readable before any mental re-viewing starts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract
from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    format_xy_position,
)
from data.spatial_3d.core.iso_geometry import visible_color_view
from data.spatial_3d.core.scene_base import Spatial3DScenePattern
from data.spatial_3d.core.view_scene import (
    COLOR_LETTERS,
    COLOR_WORDS,
    build_geometry,
    per_cube_best_visibility,
    render_view_scene,
)

import json

_FLOOR_SIZE = 3
_MAX_HEIGHT = 3
_VIEWS = ("back", "left")


def replay_stack_orbit(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Independent forward replay from the visible colored stacks and view."""

    floor_size = int(source["floor_size"])
    view = str(source["view"])
    raw = source["cell_colors"]
    cell_colors: Dict[Tuple[int, int, int], str] = {}
    for x, y, z, letter in raw:
        cell = (int(x), int(y), int(z))
        letter = str(letter)
        if letter not in COLOR_LETTERS:
            raise ValueError(f"unknown colour letter: {letter!r}")
        if cell in cell_colors:
            raise ValueError("duplicate cube cell in stack source")
        cell_colors[cell] = letter
    if floor_size != _FLOOR_SIZE or view not in _VIEWS:
        raise ValueError("stack orbit source has an unsupported floor size or view")
    if not cell_colors:
        raise ValueError("stack orbit source must contain at least one cube")
    for x, y, z in cell_colors:
        if not (0 <= x < floor_size and 0 <= y < floor_size and 0 <= z < _MAX_HEIGHT):
            raise ValueError("stack orbit cube lies outside the floor box")
        if z > 0 and (x, y, z - 1) not in cell_colors:
            raise ValueError("stack orbit cubes must rest on the floor or on another cube")
    grid = visible_color_view(
        cell_colors, view, size_x=floor_size, size_y=floor_size, height=_MAX_HEIGHT
    )
    return {"cell_colors": cell_colors, "view": view, "grid": grid}


class StackOrbitViewPattern(Spatial3DScenePattern):
    task_name = "stack_orbit_view"
    task_family = "spatial_3d"
    task_mechanic = "far_side_visible_color_mapping"
    motion = False
    verifier_profile = "stack_orbit_visible_structure_v1"

    # ------------------------------------------------------------- sampling

    def _sample_task(self) -> Dict[str, Any]:
        profile = {
            "min": (4, 5),
            "typical": (5, 7),
            "max": (7, 9),
        }.get(self.difficulty or "", (4, 9))
        for _ in range(4000):
            count = self.rng.randint(*profile)
            columns = [(x, y) for x in range(_FLOOR_SIZE) for y in range(_FLOOR_SIZE)]
            self.rng.shuffle(columns)
            cell_colors: Dict[Tuple[int, int, int], str] = {}
            placed = 0
            for x, y in columns:
                if placed >= count:
                    break
                height = min(self.rng.choice((0, 1, 1, 2, 2, 3)), count - placed)
                for z in range(height):
                    cell_colors[(x, y, z)] = self.rng.choice(tuple(COLOR_LETTERS))
                placed += height
            if len(cell_colors) < profile[0]:
                continue
            task = self._build_task(cell_colors)
            if task is not None:
                return task
        raise RuntimeError("stack_orbit_view could not sample a readable scene")

    def _build_task(self, cell_colors: Dict[Tuple[int, int, int], str]) -> Optional[Dict[str, Any]]:
        occupied_columns = {(x, y) for x, y, _z in cell_colors}
        if len(occupied_columns) < 3:
            return None
        heights = {}
        for x, y, z in cell_colors:
            heights[(x, y)] = max(heights.get((x, y), 0), z + 1)
        if len(set(heights.values())) < 2:
            return None
        view = self.rng.choice(_VIEWS)
        replayed = replay_stack_orbit(
            {
                "floor_size": _FLOOR_SIZE,
                "view": view,
                "cell_colors": [[x, y, z, letter] for (x, y, z), letter in sorted(cell_colors.items())],
            }
        )
        grid = replayed["grid"]
        letters_in_answer = {token for row in grid for token in row if token != "."}
        if len(letters_in_answer) < 2:
            return None
        # The far-side view must actually hide at least one cube behind
        # another (otherwise the occlusion rule carries no weight, 7.2).
        hidden = self._hidden_from_view(cell_colors, view)
        if hidden == 0:
            return None
        geometry = build_geometry(_FLOOR_SIZE, view, len(grid), len(grid[0]))
        visibility = per_cube_best_visibility(sorted(cell_colors), _FLOOR_SIZE, geometry)
        if min(visibility.values()) < 0.22:
            return None
        return {
            "cell_colors": cell_colors,
            "view": view,
            "grid": grid,
            "geometry": geometry,
            "hidden_from_view": hidden,
            "min_cube_visibility": float(min(visibility.values())),
        }

    @staticmethod
    def _hidden_from_view(cell_colors: Mapping[Tuple[int, int, int], str], view: str) -> int:
        hidden = 0
        for (x, y, z) in cell_colors:
            if view == "back":
                blockers = [(x, yy, z) for yy in range(y + 1, _FLOOR_SIZE)]
            else:
                blockers = [(xx, y, z) for xx in range(x)]
            if any(b in cell_colors for b in blockers):
                hidden += 1
        return hidden

    # ------------------------------------------------------------- rendering

    def _stage_count(self, task: Mapping[str, Any]) -> int:
        return 2

    def _render_stage(self, task: Mapping[str, Any], index: int) -> Image.Image:
        fill = (
            [[token if token != "." else None for token in row] for row in task["grid"]]
            if index == 1
            else None
        )
        return render_view_scene(
            sorted(task["cell_colors"]),
            _FLOOR_SIZE,
            task["geometry"],
            colors=task["cell_colors"],
            panel_fill=fill,
        )

    # ------------------------------------------------------------- semantics

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        return json.dumps([list(row) for row in task["grid"]], ensure_ascii=False)

    def _question(self, task: Mapping[str, Any]) -> str:
        return "Fill the answer panel with the colours seen looking along the bold arrow."

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[Any, str]]:
        view = task["view"]
        grid = task["grid"]
        axis_word = "column" if view == "back" else "row"
        sentences: List[str] = [
            (
                f"The arrow looks at the stacks from the far side, so left and right swap: view column i "
                f"lines up with floor {axis_word} {_FLOOR_SIZE + 1} - i, and along each ray the nearest "
                f"cube to the arrow is the one I see."
            )
        ]
        for col in range(len(grid[0])):
            floor_index = _FLOOR_SIZE - col  # 1-based floor column/row seen in this view column
            tokens = [grid[row][col] for row in range(len(grid))]
            visible = [
                f"{COLOR_WORDS[token]} at view position {format_xy_position((row + 1, col + 1))}"
                for row, token in enumerate(tokens)
                if token != "."
            ]
            if not visible:
                sentences.append(
                    f"View column {col + 1} looks along floor {axis_word} {floor_index}, which holds no "
                    f"cubes, so it stays empty."
                )
                continue
            occluded_note = self._occlusion_note(task, col)
            sentences.append(
                f"View column {col + 1} looks along floor {axis_word} {floor_index}: I see "
                + ", ".join(visible)
                + "."
                + occluded_note
            )
        sentences.append(
            "Reading the panel top layer first and writing one letter per visible cube "
            "(a dot for empty cells) gives the answer grid."
        )
        return [(1, " ".join(sentences))]

    def _occlusion_note(self, task: Mapping[str, Any], col: int) -> str:
        view = task["view"]
        cell_colors = task["cell_colors"]
        grid = task["grid"]
        hidden_letters: List[str] = []
        for row in range(len(grid)):
            z = len(grid) - 1 - row
            if view == "back":
                x = _FLOOR_SIZE - 1 - col
                ray = [(x, y, z) for y in range(_FLOOR_SIZE - 1, -1, -1)]
            else:
                y = _FLOOR_SIZE - 1 - col
                ray = [(x, y, z) for x in range(_FLOOR_SIZE)]
            seen_first = False
            for cell in ray:
                if cell in cell_colors:
                    if seen_first:
                        hidden_letters.append(COLOR_WORDS[cell_colors[cell]])
                    seen_first = True
        if not hidden_letters:
            return ""
        unique = sorted(set(hidden_letters))
        plural = len(hidden_letters) > 1
        return (
            " The "
            + " and ".join(unique)
            + (" cubes" if plural else " cube")
            + " further along the ray "
            + ("stay" if plural else "stays")
            + " hidden."
        )

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "semantic_state": {
                "cell_colors": [
                    [x, y, z, letter] for (x, y, z), letter in sorted(task["cell_colors"].items())
                ],
                "floor_size": _FLOOR_SIZE,
                "view": task["view"],
            },
            "view_grid": [list(row) for row in task["grid"]],
            "hidden_from_view": int(task["hidden_from_view"]),
            "min_cube_visibility": task["min_cube_visibility"],
            "multiple_solution_policy": "exactly one letter grid follows from the visible stacks",
            "difficulty_evidence": {
                "cube_count": len(task["cell_colors"]),
                "hidden_from_view": int(task["hidden_from_view"]),
                "distinct_colors": len({letter for letter in task["cell_colors"].values()}),
            },
            "answer_schema": {
                "schema_version": "stack_orbit_letter_grid_v1",
                "kind": "json_letter_grid",
                "parser_profile": "strict_json_color_letter_grid_with_period",
                "format_template": 'Answer: [["<colour-letter-or-dot>", ...], ...].',
                "format_example": 'Answer: [["R", ".", ...], ["B", "G", ...], ...].',
                "judged_artifacts": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "verifier_contract": {
                "profile": self.verifier_profile,
                "semantic_source": "visible colored stacks, floor labels, and the bold view arrow",
                "strict_binary_correctness": True,
                "strict_text_parser": "strict_json_color_letter_grid_with_period",
                "untrusted_redundant_fields": ["view_grid"],
            },
        }

    def _glyph_inventory(self, task: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
        labels = sorted(
            {str(v + 1) for v in range(max(_FLOOR_SIZE, len(task["grid"]), len(task["grid"][0])))},
            key=lambda v: (len(v), v),
        )
        inventory = {
            "schema_version": "stack_orbit_rendered_glyph_inventory_v1",
            "fixed_labels": labels,
            "dynamic_labels": [],
            "all_rendered_text": labels,
        }
        schema = {
            "fixed_labels": "1-based floor and answer-panel row/column numbers only",
            "dynamic_labels": "none: cube identity is colour, never rendered text",
            "exclusions": "no task title, rule prose, status sentence, or answer banner",
        }
        return inventory, schema

    # -------------------------------------------------------------- contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        view = task["view"]
        axis_word = "column" if view == "back" else "row"
        legend_colors = ", ".join(f"{letter} = {COLOR_WORDS[letter]}" for letter in sorted(COLOR_LETTERS))
        return VisualReasoningContract(
            input_description=(
                "The single image shows colored unit cubes standing in stacks on a floor grid, one bold "
                "blue arrow floating beside the scene, and a blank answer panel at the bottom. Blue "
                "numbers along two floor edges are the 1-based floor column and row numbers, and matching "
                f"blue numbers label the answer panel's columns and rows. {GRID_XY_CONVENTION_SENTENCE}"
            ),
            symbol_legend=(
                "Every cube rests on the floor or on another cube; each cube has one solid colour.",
                "The bold blue arrow points the way the required view looks at the stacks.",
                f"Answer letters name the cube colours: {legend_colors}; a dot means the view cell shows "
                "no cube.",
            ),
            rules=(
                "The required view is what a viewer standing behind the arrow sees looking along it: each "
                "view cell shows the colour of the NEAREST cube along its straight ray, and stays empty "
                "when no cube lies on that ray; nearer stacks hide the cubes behind them.",
                f"Because that viewer faces the scene from the far side, left and right swap: view column "
                f"i lines up with floor {axis_word} {_FLOOR_SIZE + 1} - i, and view row 1 is the highest "
                f"layer.",
            ),
            modality_outputs={
                "ti2v": (
                    "The answer panel completes in one commit: all coloured view cells appear together, "
                    "fading from transparent to their full colours inside their exact cells, and the "
                    "finished view then holds."
                ),
                "solution_image": (
                    "Complete the answer panel: paint every view cell with the colour of the cube seen "
                    "there and leave empty cells white; the stacks, arrow, and labels stay as given."
                ),
            },
            preservation_constraints=(
                "the colored stacks, the floor grid and its numbers, the blue arrow, the answer panel "
                "frame and its numbers, and the background unchanged; only answer-panel cells may take "
                "colour.",
            ),
            transition_constraints=(
                "Commit the whole view as one synchronized element-local fade of its coloured cells at "
                "fixed geometry; never fill cells one by one, move a cell, or dissolve the full frame.",
            ),
            ti2v_continuity={
                "moving_nouns": "the coloured view cells appearing in the answer panel",
                "static_nouns": "the colored stacks, the floor grid and numbers, the blue arrow, the "
                "answer-panel frame and numbers, the background, and the camera",
                "hold_noun": "the completed coloured view panel",
            },
            answer=AnswerContract(
                kind="grid",
                format_template='Answer: [["<colour-letter-or-dot>", ...], ...].',
                format_example='Answer: [["R", ".", ...], ["B", "G", ...], ...].',
                parser_profile="strict_json_color_letter_grid_with_period",
                verifier_profile=self.verifier_profile,
            ),
            render_profile="stack_orbit_view_v1",
        )


__all__ = ["StackOrbitViewPattern", "replay_stack_orbit"]
