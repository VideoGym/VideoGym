"""Keys-and-locks maze generator.

Variants of a 4-connected square maze where 2--3 coloured (key, gate) pairs sit
on walkable cells. Walking onto a key cell collects that colour; a gate cell is
impassable until the matching colour has been collected. Gates sit in increasing
order on the ordinary start-to-goal trunk, while matching keys sit on side
branches before their gates. Later keys are behind earlier gates, forcing a
strict chain: key0 -> gate0 -> key1 -> gate1 -> ... -> goal.

Layout guarantee (Core Rule #7, **Tier A**): if keys and gates are ignored, the
ordinary start-to-goal BFS walks straight into locked gates and is shorter than
the legal answer. The legal route must detour into each side branch to collect
the current key before it can cross the next gate.

Answer space: the formal route is delivered as ordered model-facing 1-based
``(x, y)`` grid positions, where ``x`` is the row counted from the top and ``y``
is the column counted from the left, so the top-left cell is ``(1, 1)``. The
solver, the maze grid, the key/gate chain and every pixel computation keep
working in internal 0-based ``(row, col)`` cells; the conversion happens only at
the answer-export and candidate-parsing boundaries.

Rendering:
- The grid body carries no per-cell numbering. 1-based row numbers are drawn
  down the left gutter and 1-based column numbers across the top gutter.
- Keys are drawn as stylised skeleton keys (outlined ring bow with pinhole,
  thick shaft, two angular teeth), palette from :data:`KEY_SWATCHES`.
- Gates are drawn as paired vertical bars joined by a top crossbar, with a
  centered padlock glyph under the crossbar, same palette as the key.
- A pair is identified by its shared vivid colour and a member's role by its
  icon shape, so no cell carries rendered text; the only glyphs in the scene
  are the edge row/column numbers in the gutters.
- Every icon has a dark-hued outline so it remains readable on white cell
  fill AND survives the red solution stroke passing through.
- Solution image: icons are drawn UNDER the red path (so the base pixel
  evaluator's connectivity check isn't broken by a colored icon interrupting
  the red stripe). The icon's top / bottom / side margins stay visible
  outside the central ~1/3-cell red ribbon.
- Video: each frame commits the current key/gate state first and draws the
  route over it, matching the static terminal renderer so the formal red
  route stays continuous through a contacted cell.

Quality gates (Core Rules #7, #8, #9):
- Minimum path length enforced in :meth:`_generate_with_quality_gate`.
- Tier A load-bearing check (Core Rule #7): plain BFS over the same maze with
  the key/lock mechanic disabled must be strictly shorter than the legal route,
  must differ from it, and must encounter gates before their side-branch keys.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw

from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
)
from data.common.pipeline.path_tasks import (
    GRID_AXIS_LABEL_COLOR,
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
    PathTaskGenerator,
    PathTaskRecord,
    _draw_pawn_marker,
    draw_grid_axis_labels,
    draw_path_line,
    grid_axis_label_layout,
)
from data.common.video_encoding import encode_rgb_frames_to_webm, pad_frames_to_wan_compatible_length
from data.spatial_navigation.core.maze_style import (
    begin_maze_scene,
    draw_square_cell_grid,
    finish_maze_scene,
    strip_outer_ring,
)

WALL = 1
PATH = 0

START_COLOR = (220, 30, 30)
GOAL_COLOR = START_COLOR
LINE_COLOR = (220, 30, 30)

# Stable ordered palette -- the first N entries are used for the N requested
# (key, gate) pairs, so seeds with the same pair count reproduce the same
# colour assignment. Each swatch carries a saturated fill plus a dark outline
# tuned to the same hue so icons stay legible on white cells AND when the red
# solution path overlays part of the icon.
KEY_SWATCHES: Tuple[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]], ...] = (
    ("yellow", (255, 205, 0), (120, 80, 0)),
    ("cyan", (0, 180, 220), (0, 70, 110)),
    ("magenta", (230, 60, 190), (110, 20, 90)),
)

ICON_HIGHLIGHT = (255, 255, 255)
OPEN_GATE_FILL = (170, 214, 154)
OPEN_GATE_OUTLINE = (58, 122, 75)


def _pair_colour_names(pair_count: int) -> List[str]:
    return [name for name, _fill, _outline in KEY_SWATCHES[: int(pair_count)]]


def _pair_colour_phrase(pair_count: int) -> str:
    """Natural list of the pair colours actually drawn in this sample."""

    names = [f"one {name}" for name in _pair_colour_names(pair_count)]
    if len(names) < 3:
        return " and ".join(names)
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _key_lock_rule(pair_count: int) -> str:
    chain = ", ".join(
        token
        for name in _pair_colour_names(pair_count)
        for token in (f"the {name} key", f"the {name} gate")
    )
    return (
        "Move one cell at a time through side-adjacent white cells; dark slate walls and diagonal moves are forbidden. "
        "Stepping on a key collects it, and a gate stays impassable until the key of the same colour has been "
        f"collected; colours never interchange. The visible dependency order is exactly {chain}. "
        "The route may revisit cells. It may also wait for a step; waiting leaves the held keys and opened gates "
        f"unchanged. {GRID_PATH_STAY_ENCODING_SENTENCE} Any complete legal route from the red start to the red goal is accepted."
    )


KEY_LOCK_RULE = _key_lock_rule(2)

Cell = Tuple[int, int]
KeyGatePair = Tuple[Cell, Cell, int]  # (key_cell, gate_cell, color_index)


class MazeKeysAndLocksGenerator(PathTaskGenerator):
    """Generate square mazes gated by coloured key/lock pairs."""

    DEFAULT_OUTPUT_DIR = "outputs/spatial_navigation/maze_keys_and_locks"
    DEFAULT_ROWS = 11
    DEFAULT_COLS = 11
    DEFAULT_CELL_SIZE = 32
    DEFAULT_KEY_PAIRS = 2
    ANSWER_COORDINATE_SPACE = True
    DEFAULT_TI2V_PROMPT = (
        f"Rule: {KEY_LOCK_RULE} Required Output: Draw one continuous red route through the visited cell centres. At "
        "first contact with a key, change only that key to its pale collected state; at the later contact with the "
        "gate of the same colour, change only that gate to its light-green open state. A small white-ringed red disc "
        "rides on the tip of the route and comes to rest on the goal cell."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        f"Rule: {KEY_LOCK_RULE} Required Output: Draw the complete red route with a small white-ringed red disc "
        "resting on the goal cell, and show every visited key pale and every opened gate light green. Preserve "
        "the blue edge row and column numbers, walls, endpoints, and unchanged icons."
    )
    PUZZLE_INPUT_DESCRIPTION = (
        "The input puzzle image shows a square orthogonal maze made of white traversable cells and dark slate wall cells, "
        "with a red start square, a red goal square, and two or three coloured (key, gate) pairs. The two members of a "
        "pair share one distinct vivid colour, and no other pair uses that colour. A three-tooth key icon sits on one "
        "walkable cell and a pair of thick vertical bars on another walkable cell forms the locked gate."
    )
    REASONING_INPUT_DESCRIPTION = (
        "The input image shows the keys-and-locks maze. The upper-left red square is the start and the lower-right red "
        "square is the goal; every coloured key and gate is also visible. Blue numbers run along the left and top "
        "edges of the grid. " + GRID_XY_CONVENTION_SENTENCE
    )

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        canvas_width: Optional[int] = None,
        aspect: Optional[float] = None,
        size: Optional[int] = None,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        cell_size: Optional[int] = None,
        aspect_ratio: Optional[float] = None,
        key_pairs: int = DEFAULT_KEY_PAIRS,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        video: bool = False,
    ) -> None:
        canvas_width = 512 if canvas_width is None else int(canvas_width)
        if rows < 7 or cols < 7:
            raise ValueError("rows and cols must be at least 7")
        if rows > 11 or cols > 11:
            raise ValueError(
                "rows and cols must be at most 11 so the edge row/column numbers and key/gate icons remain readable"
            )
        if key_pairs < 2 or key_pairs > len(KEY_SWATCHES):
            raise ValueError(
                f"key_pairs must be between 2 and {len(KEY_SWATCHES)}; got {key_pairs}"
            )
        pair_count = int(key_pairs)
        pair_word = {2: "two", 3: "three"}[pair_count]
        colour_phrase = _pair_colour_phrase(pair_count)
        exact_rule = _key_lock_rule(pair_count)
        # Prompt descriptions must match the objects in this supplied sample;
        # do not describe an optional third pair when a two-pair image is used.
        self.PUZZLE_INPUT_DESCRIPTION = (
            "The input puzzle image shows a square orthogonal maze made of white traversable cells and dark slate wall cells, "
            f"with a red start square, a red goal square, and exactly {pair_word} coloured key-and-gate pairs, "
            f"{colour_phrase}. A three-tooth key icon sits on one walkable cell and a pair of thick vertical bars on "
            "another walkable cell forms each locked gate."
        )
        self.REASONING_INPUT_DESCRIPTION = (
            "The input image shows a square orthogonal maze with the upper-left red start, lower-right red goal, and "
            f"exactly {pair_word} visible coloured key-and-gate pairs, {colour_phrase}. Light-blue numbers run along "
            "the left and top edges of the grid. " + GRID_XY_CONVENTION_SENTENCE
        )
        self.DEFAULT_TI2V_PROMPT = (
            f"Rule: {exact_rule} Required Output: Draw one continuous red route through the visited cell centres. At "
            "first contact with a key, change only that key to its pale collected state; at the later contact with "
            "the gate of the same colour, change only that gate to its light-green open state. A small white-ringed "
            "red disc rides on the tip of the route and comes to rest on the goal cell."
        )
        self.DEFAULT_SOLUTION_IMAGE_PROMPT = (
            f"Rule: {exact_rule} Required Output: Draw the complete red route with a small white-ringed red disc "
            "resting on the goal cell, and show every visited key pale and every opened gate light green. "
            "Preserve the blue edge row and column numbers, walls, endpoints, and unchanged icons."
        )
        adjusted_rows = rows if rows % 2 == 1 else rows + 1
        adjusted_cols = cols if cols % 2 == 1 else cols + 1
        ratio = aspect_ratio if aspect_ratio is not None else aspect
        requested_cell_size = cell_size if cell_size is not None else size
        effective_cell_size, pad_left, pad_top, pad_right, pad_bottom = grid_axis_label_layout(
            adjusted_rows,
            adjusted_cols,
            canvas_size=canvas_width,
            cell_size=requested_cell_size,
        )
        # Canvas & pixel alignment: an even cell size keeps the key/gate icon
        # strokes and the even route thickness centred inside every cell.
        if effective_cell_size % 2 == 1:
            if effective_cell_size <= 1:
                raise ValueError("cell_size must remain positive after even-alignment")
            effective_cell_size, pad_left, pad_top, pad_right, pad_bottom = grid_axis_label_layout(
                adjusted_rows,
                adjusted_cols,
                canvas_size=canvas_width,
                cell_size=effective_cell_size - 1,
            )
        canvas_dims = (canvas_width, canvas_width)
        final_width, final_height = canvas_dims
        aspect_for_super = (final_width / final_height) if final_height else None
        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        super().__init__(
            resolved_output,
            canvas_width=final_width,
            aspect=aspect_for_super,
            size=effective_cell_size,
            seed=seed,
            ti2v_prompt=ti2v_prompt,
            video=video,
        )
        self.rows = adjusted_rows
        self.cols = adjusted_cols
        self.cell_size = effective_cell_size
        self.aspect_ratio = ratio
        self.key_pairs = pair_count
        self.padding = (pad_left, pad_top, pad_right, pad_bottom)
        # Stash used by the video override to re-render the scene at each
        # committed key/gate state.  It carries the whole scene, not just the
        # icon geometry: the committed states have to go through ``_render_maze``
        # so their icons land on its 4x canvas before the single downsample.
        self._video_pairs: Optional[Sequence[KeyGatePair]] = None
        self._video_padding: Optional[Tuple[int, int]] = None
        self._video_path_cells: Optional[Sequence[Cell]] = None
        self._video_scene: Optional[Dict[str, object]] = None

    # ------------------------------------------------------------------
    # Lifecycle

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> PathTaskRecord:
        puzzle_uuid = puzzle_id or self.next_id()
        start = (0, 0)
        goal = (self.rows - 1, self.cols - 1)

        min_path_length = self._minimum_solution_path_cells()
        maze_grid, pairs, path = self._generate_with_quality_gate(start, goal, min_path_length)
        mechanic_disabled_path = self._bfs_path(maze_grid, start, goal)
        # Per-key/gate fail-loud: every key AND every gate must be on the path.
        path_set = set(path)
        for key_cell, gate_cell, color_idx in pairs:
            assert key_cell in path_set, (
                f"Key/gate not load-bearing: key {key_cell} (color={color_idx}) bypassed by path"
            )
            assert gate_cell in path_set, (
                f"Key/gate not load-bearing: gate {gate_cell} (color={color_idx}) bypassed by path"
            )
        # 1-based contiguous cell IDs over PATH cells only.
        self._rebuild_cell_id_index(
            [(r, c) for r in range(self.rows) for c in range(self.cols) if maze_grid[r][c] == PATH]
        )

        pad_left, pad_top, pad_right, pad_bottom, canvas_dims = self._compute_padding()
        cell_bboxes = self._compute_cell_bboxes(pad_left, pad_top)

        # The edge row/column numbers are solver-visible input evidence, so the
        # problem scene is identical in every modality; only the drawn route and
        # the committed key/gate states separate the terminal image from it.
        reasoning_image = self._render_maze(
            maze_grid,
            start=start,
            goal=goal,
            path=None,
            pairs=pairs,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
        )
        solution_image = self._render_maze(
            maze_grid,
            start=start,
            goal=goal,
            path=path,
            pairs=pairs,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
            collected_pair_indices=set(range(len(pairs))),
            open_pair_indices=set(range(len(pairs))),
        )

        puzzle_path, solution_path = self.save_images(puzzle_uuid, reasoning_image, solution_image)
        reasoning_path = self.save_reasoning_image(puzzle_uuid, reasoning_image)

        video_path = None
        if self.video:
            path_points = [self._cell_center(self._get_cell_id(*cell)) for cell in path]
            # Stash so save_video() can overlay icons on top of each frame's
            # red stroke; cleared once the video is written.
            self._video_pairs = pairs
            self._video_padding = (pad_left, pad_top)
            self._video_path_cells = list(path)
            self._video_scene = {
                "grid": maze_grid,
                "start": start,
                "goal": goal,
                "canvas_dims": canvas_dims,
            }
            try:
                video_path = self.save_video(
                    puzzle_uuid,
                    reasoning_image,
                    path_points,
                    thickness=self._path_thickness(),
                    joint_style="square",
                )
            finally:
                self._video_pairs = None
                self._video_padding = None
                self._video_path_cells = None
                self._video_scene = None

        start_point = self._cell_center(self._get_cell_id(*start))
        goal_point = self._cell_center(self._get_cell_id(*goal))
        keys_payload: List[Dict[str, object]] = []
        locks_payload: List[Dict[str, object]] = []
        chain_payload: List[Dict[str, object]] = []
        for pair_index, (key_cell, gate_cell, color_idx) in enumerate(pairs):
            color_name, color_rgb, _outline_rgb = KEY_SWATCHES[color_idx]
            key_id = self._get_cell_id(*key_cell)
            gate_id = self._get_cell_id(*gate_cell)
            # Identity names for verifier diagnostics only; nothing renders them.
            key_label = f"{color_name} key"
            lock_label = f"{color_name} gate"
            keys_payload.append(
                {
                    "label": key_label,
                    "color": color_name,
                    "color_rgb": [int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])],
                    "cell": list(key_cell),
                    "cell_id": key_id,
                }
            )
            locks_payload.append(
                {
                    "label": lock_label,
                    "color": color_name,
                    "color_rgb": [int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])],
                    "cell": list(gate_cell),
                    "cell_id": gate_id,
                }
            )
            chain_payload.append(
                {
                    "index": int(color_idx),
                    "color": color_name,
                    "key_label": key_label,
                    "key_cell": list(key_cell),
                    "key_cell_id": key_id,
                    "lock_label": lock_label,
                    "lock_cell": list(gate_cell),
                    "lock_cell_id": gate_id,
                }
            )

        extra_payload: Dict[str, object] = {
            "grid_size": [self.rows, self.cols],
            "cell_size": self.cell_size,
            "maze_grid": maze_grid,
            "start": list(start),
            "goal": list(goal),
            "cell_bboxes": cell_bboxes,
            "padding": [pad_left, pad_top, pad_right, pad_bottom],
            "solution_path_cell_ids": [self._get_cell_id(*cell) for cell in path],
            "path_joint_style": "square",
            "keys": keys_payload,
            "locks": locks_payload,
            "mechanic_disabled_path_cell_ids": [
                self._get_cell_id(*cell) for cell in mechanic_disabled_path
            ],
            "mechanic_detour_cells": len(path) - len(mechanic_disabled_path),
            "lock_chain_order": chain_payload,
            # The grid body carries no text at all: the only rendered glyphs
            # are the edge row/column numbers in the gutters. Keys and gates
            # are told apart by icon shape, and pairs by their shared colour.
            "rendered_glyph_inventory": {
                "axis_row_labels": [str(index + 1) for index in range(self.rows)],
                "axis_col_labels": [str(index + 1) for index in range(self.cols)],
                "fixed_labels": [],
            },
            "rendered_glyph_schema": {
                "axis_row_labels": "1-based row numbers drawn once per row in the left gutter, outside the grid body",
                "axis_col_labels": "1-based column numbers drawn once per column in the top gutter, outside the grid body",
                "fixed_labels": "none; key/gate role is carried by icon shape and pair identity by colour, so no per-cell badge, ID, title, rule prose, status sentence, or answer banner is drawn",
            },
        }
        self.add_path_length_metadata(extra_payload, path, minimum=min_path_length)
        return self.build_record(
            puzzle_uuid,
            start_point=start_point,
            goal_point=goal_point,
            puzzle_path=puzzle_path,
            solution_path=solution_path,
            reasoning_path=reasoning_path,
            ti2v_prompt=self.ti2v_prompt,
            extra=extra_payload,
            video_path=video_path,
        )

    # ------------------------------------------------------------------

    def _route_leg_summary(self, leg: Sequence[Cell]) -> str:
        """Summarize one route leg as compact direction-and-count movements."""

        parts: List[str] = []
        current_direction: Optional[str] = None
        count = 0
        for previous, cell in zip(leg, leg[1:]):
            dr, dc = cell[0] - previous[0], cell[1] - previous[1]
            direction = {(1, 0): "down", (-1, 0): "up", (0, 1): "right", (0, -1): "left"}[(dr, dc)]
            if direction == current_direction:
                count += 1
            else:
                if current_direction is not None:
                    parts.append(f"{current_direction} {count}")
                current_direction = direction
                count = 1
        if current_direction is not None:
            parts.append(f"{current_direction} {count}")
        return ", then ".join(parts)

    def _route_events(
        self,
        path: Sequence[Cell],
        pairs: Sequence[KeyGatePair],
    ) -> List[Tuple[int, str, int]]:
        """Return ordered first-visit events as (step_index, kind, pair_index)."""

        key_cells = {key: index for index, (key, _gate, _idx) in enumerate(pairs)}
        gate_cells = {gate: index for index, (_key, gate, _idx) in enumerate(pairs)}
        events: List[Tuple[int, str, int]] = []
        seen: Set[Tuple[str, int]] = set()
        for step, cell in enumerate(path):
            if cell in key_cells and ("key", key_cells[cell]) not in seen:
                seen.add(("key", key_cells[cell]))
                events.append((step, "key", key_cells[cell]))
            elif cell in gate_cells and ("gate", gate_cells[cell]) not in seen:
                seen.add(("gate", gate_cells[cell]))
                events.append((step, "gate", gate_cells[cell]))
        return events

    def _authored_route_reasoning(
        self,
        path_ids: Sequence[object],
        extra_payload: Dict[str, object],
    ) -> Optional[str]:
        keys_payload = extra_payload.get("keys")
        locks_payload = extra_payload.get("locks")
        if not isinstance(keys_payload, list) or not isinstance(locks_payload, list):
            return None
        path = [self._id_to_cell[int(cell_id)] for cell_id in path_ids]
        pairs: List[KeyGatePair] = []
        for pair_index, (key_item, lock_item) in enumerate(zip(keys_payload, locks_payload)):
            pairs.append((tuple(key_item["cell"]), tuple(lock_item["cell"]), pair_index))
        events = self._route_events(path, pairs)
        start_xy = cell_to_xy(path[0])
        goal_xy = cell_to_xy(path[-1])
        colors = {index: str(item.get("color", "")) for index, item in enumerate(keys_payload)}
        gate_positions = ", ".join(
            f"the {item.get('color', '')} gate at {cell_to_xy(tuple(item['cell']))}"
            for item in locks_payload
        )
        chain_order = ", then ".join(
            f"the {colors.get(index, '')} key and its gate" for index in range(len(keys_payload))
        )
        sentences: List[str] = [
            f"The red start sits at {start_xy} and the red goal at {goal_xy}, but the corridors toward the "
            f"goal are barred: {gate_positions} each stay shut until the key of the same colour is picked up. "
            f"So the route has to reach {chain_order}.",
        ]
        previous_step = 0
        for event_step, kind, pair_index in events:
            leg = path[previous_step : event_step + 1]
            summary = self._route_leg_summary(leg)
            spot_xy = cell_to_xy(path[event_step])
            if kind == "key":
                sentences.append(
                    f"From {cell_to_xy(path[previous_step])} it moves {summary} to {spot_xy}, stepping onto the "
                    f"{colors.get(pair_index, '')} key and collecting it."
                )
            else:
                sentences.append(
                    f"It then moves {summary} to {spot_xy}, where the {colors.get(pair_index, '')} gate is now "
                    "open and the route passes straight through."
                )
            previous_step = event_step
        tail = path[previous_step:]
        if len(tail) > 1:
            sentences.append(
                f"Finally it moves {self._route_leg_summary(tail)} and arrives at the goal {goal_xy}."
            )
        sentences.append(
            "Every key was collected before its gate was crossed, so the whole route is legal."
        )
        return " ".join(sentences)

    # ------------------------------------------------------------------
    # Generation pipeline

    def _minimum_solution_path_cells(self) -> int:
        return max(20, self.rows + self.cols + 6)

    def _path_thickness(self) -> int:
        # thickness = max(4, cell_size // 3) rounded DOWN to even.
        thickness = max(4, self.cell_size // 3)
        thickness &= ~1
        return max(2, thickness)

    def _generate_with_quality_gate(
        self,
        start: Cell,
        goal: Cell,
        min_path_length: int,
    ) -> Tuple[List[List[int]], List[KeyGatePair], List[Cell]]:
        for _ in range(400):
            maze_grid = self._generate_maze()
            direct_path = self._bfs_path(maze_grid, start, goal)
            if len(direct_path) < min_path_length - self.key_pairs * 2:
                # Tolerate a slight shortfall versus the final target because
                # adding gates can lengthen the route, but do not bother with
                # pathologically short corridors.
                continue
            layout = self._plant_keys_and_gates(maze_grid, direct_path, start, goal)
            if layout is None:
                continue
            pairs, path = layout
            if len(path) < min_path_length:
                continue
            # Tier A load-bearing check (Core Rule #7): with every gate cell
            # treated as a wall, plain BFS must return NO path. The
            # articulation-point planter already guarantees each gate
            # individually disconnects start from goal on the remaining open
            # subgraph; closing all gates simultaneously is at least as
            # restrictive, so this is a fail-loud safety net rather than a
            # filter we expect to fire often.
            plain_path = self._plain_baseline_path(maze_grid, pairs, start, goal)
            if plain_path:
                # Some route bypasses every gate -- gates are not load-bearing.
                continue
            assert plain_path == [], (
                "Tier A violated: plain baseline solved without crossing any gate"
            )
            if not self._chain_layout_satisfies_load_bearing(direct_path, path, pairs):
                continue
            return maze_grid, pairs, path
        raise RuntimeError(
            "Failed to generate a keys-and-locks maze that satisfies the path-length / "
            "mechanic-load-bearing quality gates"
        )

    def _plain_baseline_path(
        self,
        grid: Sequence[Sequence[int]],
        pairs: Sequence[KeyGatePair],
        start: Cell,
        goal: Cell,
    ) -> List[Cell]:
        """Plain BFS with every gate cell treated as a wall.

        Used by the Tier A check: if this returns a non-empty path, then
        some route from start to goal bypasses every gate, which means the
        keys-and-locks mechanic is decorative for that layout.
        """
        gate_cells: Set[Cell] = {gate for _key, gate, _idx in pairs}
        if start in gate_cells or goal in gate_cells:
            return []
        masked_grid: List[List[int]] = [list(row) for row in grid]
        for gr, gc in gate_cells:
            masked_grid[gr][gc] = WALL
        return self._bfs_path(masked_grid, start, goal)

    @staticmethod
    def _paths_equal(a: Sequence[Cell], b: Sequence[Cell]) -> bool:
        if len(a) != len(b):
            return False
        for ca, cb in zip(a, b):
            if ca != cb:
                return False
        return True

    def _plant_keys_and_gates(
        self,
        maze_grid: Sequence[Sequence[int]],
        base_path: Sequence[Cell],
        start: Cell,
        goal: Cell,
    ) -> Optional[Tuple[List[KeyGatePair], List[Cell]]]:
        """Place a strict key/gate chain on the ordinary start-goal trunk.

        The generated maze is a tree, so the ordinary BFS path is the trunk.
        Gates are selected on that trunk in increasing order. Key ``i`` is
        placed on a side branch attached after gate ``i-1`` and before gate
        ``i``. Therefore the direct trunk path reaches each locked gate before
        seeing its key, while the legal path must detour into the current side
        branch, return to the trunk, and only then cross the gate.
        """
        if len(base_path) < self.key_pairs * 6 + 3:
            return None

        branch_candidates = self._side_branch_key_candidates(maze_grid, base_path)
        gate_indices = self._choose_chain_gate_indices(base_path, branch_candidates)
        if gate_indices is None:
            return None

        pairs: List[KeyGatePair] = []
        previous_gate_index = 0
        for color_idx, gate_index in enumerate(gate_indices):
            key_cell = self._choose_key_for_segment(
                branch_candidates,
                previous_gate_index + 1,
                gate_index,
            )
            if key_cell is None:
                return None
            pairs.append((key_cell, base_path[gate_index], color_idx))
            previous_gate_index = gate_index

        path = self._bfs_path_with_keys(maze_grid, start, goal, pairs)
        if not path:
            return None
        if not self._chain_layout_satisfies_load_bearing(base_path, path, pairs):
            return None
        return pairs, path

    def _side_branch_key_candidates(
        self,
        grid: Sequence[Sequence[int]],
        trunk_path: Sequence[Cell],
    ) -> Dict[int, List[Tuple[Cell, int]]]:
        """Return off-trunk key candidates keyed by trunk attachment index."""
        trunk_set = set(trunk_path)
        candidates: Dict[int, List[Tuple[Cell, int]]] = {}
        for trunk_index, trunk_cell in enumerate(trunk_path):
            branch_cells: List[Tuple[Cell, int]] = []
            seen: Set[Cell] = set()
            queue: deque[Tuple[Cell, int]] = deque()
            for nb in self._adjacent_walkable(grid, trunk_cell):
                if nb in trunk_set:
                    continue
                seen.add(nb)
                queue.append((nb, 1))
            while queue:
                cell, dist = queue.popleft()
                if dist >= 2:
                    branch_cells.append((cell, dist))
                for nb in self._adjacent_walkable(grid, cell):
                    if nb in trunk_set or nb in seen:
                        continue
                    seen.add(nb)
                    queue.append((nb, dist + 1))
            if branch_cells:
                branch_cells.sort(key=lambda item: (-item[1], item[0]))
                candidates[trunk_index] = branch_cells
        return candidates

    def _choose_chain_gate_indices(
        self,
        trunk_path: Sequence[Cell],
        branch_candidates: Dict[int, List[Tuple[Cell, int]]],
    ) -> Optional[List[int]]:
        min_gate_index = 3
        max_gate_index = len(trunk_path) - 4
        if max_gate_index <= min_gate_index:
            return None

        possible = list(range(min_gate_index, max_gate_index + 1))
        self.rng.shuffle(possible)
        possible.sort(key=lambda _idx: self.rng.random())

        def segment_has_key(start_index: int, gate_index: int) -> bool:
            return any(index in branch_candidates for index in range(start_index, gate_index))

        def search(pair_index: int, previous_gate_index: int, chosen: List[int]) -> Optional[List[int]]:
            if pair_index == self.key_pairs:
                return list(chosen)
            remaining_after = self.key_pairs - pair_index - 1
            latest = max_gate_index - remaining_after * 3
            gate_options = [
                idx
                for idx in possible
                if previous_gate_index + 3 <= idx <= latest
                and (not chosen or idx > chosen[-1] + 2)
                and segment_has_key(previous_gate_index + 1, idx)
            ]
            self.rng.shuffle(gate_options)
            for gate_index in gate_options:
                chosen.append(gate_index)
                result = search(pair_index + 1, gate_index, chosen)
                if result is not None:
                    return result
                chosen.pop()
            return None

        return search(0, 0, [])

    def _choose_key_for_segment(
        self,
        branch_candidates: Dict[int, List[Tuple[Cell, int]]],
        start_index: int,
        gate_index: int,
    ) -> Optional[Cell]:
        candidates: List[Tuple[Cell, int]] = []
        for trunk_index in range(start_index, gate_index):
            candidates.extend(branch_candidates.get(trunk_index, ()))
        if not candidates:
            return None
        max_dist = max(dist for _cell, dist in candidates)
        farthest = [cell for cell, dist in candidates if dist == max_dist]
        self.rng.shuffle(farthest)
        return farthest[0]

    def _chain_layout_satisfies_load_bearing(
        self,
        direct_path: Sequence[Cell],
        legal_path: Sequence[Cell],
        pairs: Sequence[KeyGatePair],
    ) -> bool:
        if not direct_path or not legal_path or self._paths_equal(direct_path, legal_path):
            return False
        required_detour = max(6, 3 * len(pairs))
        if len(legal_path) - len(direct_path) < required_detour:
            return False

        direct_positions = {cell: idx for idx, cell in enumerate(direct_path)}
        legal_positions = {cell: idx for idx, cell in enumerate(legal_path)}
        previous_gate_pos = -1
        direct_gate_order: List[int] = []
        gate_to_index = {gate: idx for idx, (_key, gate, _color_idx) in enumerate(pairs)}
        for cell in direct_path:
            if cell in gate_to_index:
                direct_gate_order.append(gate_to_index[cell])
        if direct_gate_order != list(range(len(pairs))):
            return False

        for idx, (key_cell, gate_cell, _color_idx) in enumerate(pairs):
            if key_cell in direct_positions:
                return False
            if gate_cell not in direct_positions:
                return False
            key_pos = legal_positions.get(key_cell)
            gate_pos = legal_positions.get(gate_cell)
            if key_pos is None or gate_pos is None:
                return False
            if not (previous_gate_pos < key_pos < gate_pos):
                return False
            previous_gate_pos = gate_pos
            if idx > 0:
                previous_gate = pairs[idx - 1][1]
                if direct_positions[previous_gate] >= direct_positions[gate_cell]:
                    return False
        return True


    def _generate_maze(self) -> List[List[int]]:
        ringed_rows = self.rows + 2
        ringed_cols = self.cols + 2
        grid = [[WALL for _ in range(ringed_cols)] for _ in range(ringed_rows)]

        def carve(r: int, c: int) -> None:
            grid[r][c] = PATH
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            self.rng.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < ringed_rows - 1 and 1 <= nc < ringed_cols - 1 and grid[nr][nc] == WALL:
                    grid[r + dr // 2][c + dc // 2] = PATH
                    carve(nr, nc)

        carve(1, 1)
        grid[ringed_rows - 2][ringed_cols - 2] = PATH
        return strip_outer_ring(grid)

    def _adjacent_walkable(
        self,
        grid: Sequence[Sequence[int]],
        cell: Cell,
    ) -> Iterable[Cell]:
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr][nc] == PATH:
                yield (nr, nc)

    def _bfs_path(
        self,
        grid: Sequence[Sequence[int]],
        start: Cell,
        goal: Cell,
    ) -> List[Cell]:
        queue: deque[Cell] = deque([start])
        parents: Dict[Cell, Optional[Cell]] = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                break
            for nb in self._adjacent_walkable(grid, cell):
                if nb not in parents:
                    parents[nb] = cell
                    queue.append(nb)
        if goal not in parents:
            return []
        node: Optional[Cell] = goal
        result: List[Cell] = []
        while node is not None:
            result.append(node)
            node = parents[node]
        result.reverse()
        return result

    def _reachable_cells(
        self,
        grid: Sequence[Sequence[int]],
        start: Cell,
    ) -> Set[Cell]:
        queue: deque[Cell] = deque([start])
        visited: Set[Cell] = {start}
        while queue:
            cell = queue.popleft()
            for nb in self._adjacent_walkable(grid, cell):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return visited

    def _bfs_path_with_keys(
        self,
        grid: Sequence[Sequence[int]],
        start: Cell,
        goal: Cell,
        pairs: Sequence[KeyGatePair],
    ) -> List[Cell]:
        """BFS over ``(cell, keys_bitmask)``. Walking onto key cell ``i`` sets
        bit ``i``; walking onto gate cell ``i`` is only allowed if bit ``i`` is
        already set when the move is attempted."""
        key_bit: Dict[Cell, int] = {key: 1 << idx for idx, (key, _g, _c) in enumerate(pairs)}
        gate_bit: Dict[Cell, int] = {gate: 1 << idx for idx, (_k, gate, _c) in enumerate(pairs)}

        State = Tuple[Cell, int]
        queue: deque[State] = deque()
        start_state: State = (start, key_bit.get(start, 0))
        queue.append(start_state)
        parents: Dict[State, Optional[State]] = {start_state: None}
        goal_state: Optional[State] = None
        while queue:
            state = queue.popleft()
            cell, mask = state
            if cell == goal:
                goal_state = state
                break
            for nb in self._adjacent_walkable(grid, cell):
                required = gate_bit.get(nb, 0)
                if required and not (mask & required):
                    continue
                new_mask = mask | key_bit.get(nb, 0)
                new_state: State = (nb, new_mask)
                if new_state not in parents:
                    parents[new_state] = state
                    queue.append(new_state)
        if goal_state is None:
            return []
        result: List[Cell] = []
        node: Optional[State] = goal_state
        while node is not None:
            result.append(node[0])
            node = parents[node]
        result.reverse()
        return result

    # ------------------------------------------------------------------
    # Cell-id / padding helpers

    def _get_cell_id(self, r: int, c: int) -> int:
        return self._cell_to_id.get((r, c), 0)

    def _compute_padding(self) -> Tuple[int, int, int, int, Tuple[int, int]]:
        pad_left, pad_top, pad_right, pad_bottom = self.padding
        return pad_left, pad_top, pad_right, pad_bottom, self.canvas_dimensions

    def _compute_cell_bboxes(
        self,
        pad_left: int,
        pad_top: int,
    ) -> List[List[Tuple[int, int, int, int]]]:
        bboxes: List[List[Tuple[int, int, int, int]]] = []
        for r in range(self.rows):
            row_boxes: List[Tuple[int, int, int, int]] = []
            for c in range(self.cols):
                left = pad_left + c * self.cell_size
                top = pad_top + r * self.cell_size
                row_boxes.append((left, top, left + self.cell_size, top + self.cell_size))
            bboxes.append(row_boxes)
        return bboxes

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        row, col = self._id_to_cell[cell_id]
        pad_left, pad_top, _, _, _ = self._compute_padding()
        x = pad_left + col * self.cell_size + self.cell_size / 2.0
        y = pad_top + row * self.cell_size + self.cell_size / 2.0
        return (x, y)

    # ------------------------------------------------------------------
    # Rendering

    def _render_maze(
        self,
        grid: Sequence[Sequence[int]],
        *,
        start: Cell,
        goal: Cell,
        path: Optional[Sequence[Cell]],
        pairs: Sequence[KeyGatePair],
        padding: Tuple[int, int],
        canvas_dims: Tuple[int, int],
        collected_pair_indices: Optional[Set[int]] = None,
        open_pair_indices: Optional[Set[int]] = None,
    ) -> Image.Image:
        pad_left, pad_top = padding
        collected_pair_indices = collected_pair_indices or set()
        open_pair_indices = open_pair_indices or set()

        # The static scene (cells, icons, axis numbers) is drawn on a 4x canvas and
        # LANCZOS-downsampled; the icon helpers read ``self.cell_size``, so the
        # supersample is applied by temporarily scaling the cell geometry.
        factor = 4
        scene, sdraw = begin_maze_scene(canvas_dims, factor=factor)
        draw = sdraw.raw
        original_cell_size = self.cell_size
        scaled_pad_left = pad_left * factor
        scaled_pad_top = pad_top * factor
        self.cell_size = original_cell_size * factor
        try:
            draw_square_cell_grid(
                sdraw,
                rows=self.rows,
                cols=self.cols,
                cell_size=original_cell_size,
                pad_left=pad_left,
                pad_top=pad_top,
                is_wall=lambda r, c: grid[r][c] != PATH,
            )
            for cell, color in ((start, START_COLOR), (goal, GOAL_COLOR)):
                left = scaled_pad_left + cell[1] * self.cell_size
                top = scaled_pad_top + cell[0] * self.cell_size
                draw.rectangle(
                    (left, top, left + self.cell_size - 1, top + self.cell_size - 1),
                    fill=color,
                )
            for pair_index, (key_cell, gate_cell, color_idx) in enumerate(pairs):
                _name, fill_rgb, outline_rgb = KEY_SWATCHES[color_idx]
                self._draw_key_icon(
                    draw,
                    key_cell,
                    scaled_pad_left,
                    scaled_pad_top,
                    fill_rgb,
                    outline_rgb,
                    collected=pair_index in collected_pair_indices,
                )
                self._draw_gate_icon(
                    draw,
                    gate_cell,
                    scaled_pad_left,
                    scaled_pad_top,
                    fill_rgb,
                    outline_rgb,
                    open_gate=pair_index in open_pair_indices,
                )
            draw_grid_axis_labels(
                sdraw,
                rows=self.rows,
                cols=self.cols,
                cell_size=original_cell_size,
                pad_left=pad_left,
                pad_top=pad_top,
                color=GRID_AXIS_LABEL_COLOR,
            )
        finally:
            self.cell_size = original_cell_size
        canvas = finish_maze_scene(scene, canvas_dims)

        if path:
            thickness = self._path_thickness()
            points = [
                (
                    pad_left + c * self.cell_size + self.cell_size / 2,
                    pad_top + r * self.cell_size + self.cell_size / 2,
                )
                for r, c in path
            ]
            draw_path_line(canvas, points, LINE_COLOR, thickness, joint_style="square")
            # Key/gate icons are deliberately NOT re-drawn on top of the red
            # path in the static terminal image: the icon margins outside the
            # central stripe keep each icon identifiable while the formal red
            # route stays one connected component.

        return canvas

    def _blend_rgb(
        self,
        rgb: Tuple[int, int, int],
        target: Tuple[int, int, int],
        amount: float,
    ) -> Tuple[int, int, int]:
        return tuple(
            max(0, min(255, int(round(channel + (dest - channel) * amount))))
            for channel, dest in zip(rgb, target)
        )

    def _draw_key_icon(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Cell,
        pad_left: int,
        pad_top: int,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        *,
        collected: bool = False,
    ) -> None:
        """Draw a stylised skeleton-key icon centered in the cell.

        Layered composition so the icon reads at 30-px cells AND when the red
        solution stroke runs straight through the middle:
        - **Bow** (left): outlined ring with a pinhole — filled colour ring
          around a small white disc so the bow reads as a handle rather than
          a blob.
        - **Shaft** (horizontal): thick colour rectangle with a dark outline
          border drawn as a slightly larger dark rectangle below it.
        - **Teeth** (two angular steps below the shaft's right end): one
          large tooth + one smaller tooth, both outlined.

        The outline colour is the same hue as ``fill`` but darker, so the
        icon silhouette survives even when the red path overlays it.
        """
        r, c = cell
        if collected:
            fill = self._blend_rgb(fill, ICON_HIGHLIGHT, 0.58)
            outline = self._blend_rgb(outline, ICON_HIGHLIGHT, 0.28)
        from data.common.prompt_paraphrase import active_profile
        if active_profile("node_maze"):
            left=pad_left+cell[1]*self.cell_size;top=pad_top+cell[0]*self.cell_size;s=self.cell_size
            draw.polygon([(left+s*.2,top+s*.15),(left+s*.68,top+s*.15),(left+s*.83,top+s*.3),(left+s*.83,top+s*.86),(left+s*.2,top+s*.86)],fill=fill,outline=outline,width=max(2,int(s*.05)))
            draw.rectangle((left+s*.33,top+s*.38,left+s*.68,top+s*.58),fill=outline)
            draw.line((left+s*.33,top+s*.72,left+s*.68,top+s*.72),fill=outline,width=max(2,int(s*.05)))
            return
        from data.common.visual_style import current_style
        if style := current_style():
            from data.common.render_primitives import ScaledDraw
            left, top = pad_left + cell[1] * self.cell_size, pad_top + cell[0] * self.cell_size
            style.key(ScaledDraw(draw, 4), (left / 4, top / 4,
                       (left + self.cell_size) / 4, (top + self.cell_size) / 4), fill, outline)
            return
        left = pad_left + c * self.cell_size
        top = pad_top + r * self.cell_size
        right = left + self.cell_size
        bottom = top + self.cell_size
        cy = (top + bottom) / 2.0

        # Safe inset around the cell; leaves ~2-3 px margin.
        inset = max(1, self.cell_size // 10)
        box_left = left + inset
        box_right = right - inset
        box_top = top + inset
        box_bottom = bottom - inset
        box_w = box_right - box_left
        box_h = box_bottom - box_top
        outline_w = max(1, self.cell_size // 14)

        # Bow: large circular ring on the left. Radius uses the full safe
        # height so the bow is clearly a handle.
        bow_radius = box_h / 2.0
        bow_cx = box_left + bow_radius
        bow_cy = cy
        # Dark outer ring.
        draw.ellipse(
            (bow_cx - bow_radius, bow_cy - bow_radius, bow_cx + bow_radius, bow_cy + bow_radius),
            fill=outline,
        )
        inner_r = bow_radius - outline_w
        if inner_r > 1:
            draw.ellipse(
                (bow_cx - inner_r, bow_cy - inner_r, bow_cx + inner_r, bow_cy + inner_r),
                fill=fill,
            )
        # Pinhole inside the bow.
        hole_r = max(1.0, bow_radius * 0.30)
        draw.ellipse(
            (bow_cx - hole_r, bow_cy - hole_r, bow_cx + hole_r, bow_cy + hole_r),
            fill=ICON_HIGHLIGHT,
        )

        # Shaft: a thick rectangle from bow to right edge of safe box, with a
        # dark outline rectangle underneath and the fill rectangle on top.
        shaft_half = max(2.0, box_h * 0.20)
        shaft_x0 = bow_cx + bow_radius * 0.6
        shaft_x1 = box_right
        # Outline
        draw.rectangle(
            (shaft_x0 - outline_w, cy - shaft_half - outline_w, shaft_x1 + outline_w, cy + shaft_half + outline_w),
            fill=outline,
        )
        # Fill
        draw.rectangle(
            (shaft_x0, cy - shaft_half, shaft_x1, cy + shaft_half),
            fill=fill,
        )

        # Teeth: two rectangular teeth hanging from the bottom of the shaft
        # near its right end. Big tooth + smaller tooth, both outlined.
        shaft_len = shaft_x1 - shaft_x0
        if shaft_len > 4:
            big_tooth_w = max(2.0, box_w * 0.10)
            big_tooth_h = max(2.0, box_h * 0.30)
            big_tooth_x0 = shaft_x1 - big_tooth_w - outline_w
            big_tooth_y0 = cy + shaft_half
            big_tooth_y1 = big_tooth_y0 + big_tooth_h
            draw.rectangle(
                (big_tooth_x0 - outline_w, big_tooth_y0, big_tooth_x0 + big_tooth_w + outline_w, big_tooth_y1 + outline_w),
                fill=outline,
            )
            draw.rectangle(
                (big_tooth_x0, big_tooth_y0, big_tooth_x0 + big_tooth_w, big_tooth_y1),
                fill=fill,
            )
            # Smaller tooth to the left of the big one.
            small_tooth_w = big_tooth_w
            small_tooth_h = max(2.0, box_h * 0.18)
            small_tooth_x0 = big_tooth_x0 - small_tooth_w - max(1.0, box_w * 0.04)
            if small_tooth_x0 > shaft_x0 + outline_w:
                small_tooth_y0 = cy + shaft_half
                small_tooth_y1 = small_tooth_y0 + small_tooth_h
                draw.rectangle(
                    (small_tooth_x0 - outline_w, small_tooth_y0, small_tooth_x0 + small_tooth_w + outline_w, small_tooth_y1 + outline_w),
                    fill=outline,
                )
                draw.rectangle(
                    (small_tooth_x0, small_tooth_y0, small_tooth_x0 + small_tooth_w, small_tooth_y1),
                    fill=fill,
                )

    def _draw_gate_icon(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Cell,
        pad_left: int,
        pad_top: int,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        *,
        open_gate: bool = False,
    ) -> None:
        """Draw a locked gate: two thick vertical bars joined by a top
        crossbar and a padlock glyph under the crossbar.

        The bars + crossbar form a clear "gate" silhouette, and the padlock
        cements the "locked" reading. All primitives carry a dark outline so
        the icon survives the red solution stroke cutting through the cell.
        """
        if open_gate:
            fill = OPEN_GATE_FILL
            outline = OPEN_GATE_OUTLINE
        from data.common.prompt_paraphrase import active_profile
        if active_profile("node_maze"):
            cx=pad_left+(cell[1]+.5)*self.cell_size;cy=pad_top+(cell[0]+.5)*self.cell_size;s=self.cell_size;r=s*.38
            draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=fill,outline=outline,width=max(2,int(s*.05)))
            draw.line((cx-r*.62,cy,cx+r*.62,cy),fill=outline,width=max(2,int(s*.06)))
            draw.line((cx,cy-r*.62,cx,cy+r*.62),fill=outline,width=max(2,int(s*.06)))
            draw.ellipse((cx-r*.22,cy-r*.22,cx+r*.22,cy+r*.22),fill=outline)
            return
        from data.common.visual_style import current_style
        if style := current_style():
            from data.common.render_primitives import ScaledDraw
            left, top = pad_left + cell[1] * self.cell_size, pad_top + cell[0] * self.cell_size
            style.gate(ScaledDraw(draw, 4), (left / 4, top / 4,
                        (left + self.cell_size) / 4, (top + self.cell_size) / 4), fill, outline)
            return
        r, c = cell
        left = pad_left + c * self.cell_size
        top = pad_top + r * self.cell_size
        right = left + self.cell_size
        bottom = top + self.cell_size
        # Margins to keep the gate clearly inside the cell.
        v_inset = max(1, self.cell_size // 10)
        h_inset = max(1, self.cell_size // 10)
        interior_left = left + h_inset
        interior_right = right - h_inset
        interior_top = top + v_inset
        interior_bottom = bottom - v_inset
        interior_w = interior_right - interior_left
        interior_h = interior_bottom - interior_top
        outline_w = max(1, self.cell_size // 14)

        # Bar geometry: two thick vertical bars flush to the interior edges
        # with a gap in the middle for the padlock.
        bar_w = max(3.0, interior_w * 0.22)
        left_bar_x0 = interior_left
        left_bar_x1 = interior_left + bar_w
        right_bar_x1 = interior_right
        right_bar_x0 = interior_right - bar_w
        # Crossbar on top, connecting the two bars.
        crossbar_h = max(3.0, interior_h * 0.18)
        crossbar_y0 = interior_top
        crossbar_y1 = interior_top + crossbar_h

        # --- Outline first (slightly inflated) ---
        draw.rectangle(
            (left_bar_x0 - outline_w, crossbar_y0 - outline_w, left_bar_x1 + outline_w, interior_bottom + outline_w),
            fill=outline,
        )
        draw.rectangle(
            (right_bar_x0 - outline_w, crossbar_y0 - outline_w, right_bar_x1 + outline_w, interior_bottom + outline_w),
            fill=outline,
        )
        draw.rectangle(
            (left_bar_x0 - outline_w, crossbar_y0 - outline_w, right_bar_x1 + outline_w, crossbar_y1 + outline_w),
            fill=outline,
        )

        # --- Colour fill on top ---
        draw.rectangle(
            (left_bar_x0, crossbar_y0, left_bar_x1, interior_bottom),
            fill=fill,
        )
        draw.rectangle(
            (right_bar_x0, crossbar_y0, right_bar_x1, interior_bottom),
            fill=fill,
        )
        draw.rectangle(
            (left_bar_x0, crossbar_y0, right_bar_x1, crossbar_y1),
            fill=fill,
        )

        # --- Padlock glyph centered between the bars, under the crossbar ---
        lock_cx = (interior_left + interior_right) / 2.0
        lock_body_top = crossbar_y1 + max(1.0, interior_h * 0.05)
        lock_body_h = max(4.0, interior_h * 0.32)
        lock_body_w = max(4.0, (right_bar_x0 - left_bar_x1) * 0.62)
        if lock_body_w >= 3.0 and lock_body_h >= 3.0:
            lock_body_x0 = lock_cx - lock_body_w / 2.0
            lock_body_x1 = lock_cx + lock_body_w / 2.0
            lock_body_y0 = lock_body_top
            lock_body_y1 = lock_body_y0 + lock_body_h
            # Shackle (U above body) drawn as an outline rectangle minus the
            # body rectangle; render as two small rects to suggest the loop.
            shackle_h = max(1.0, lock_body_h * 0.45)
            shackle_w_inset = max(1.0, lock_body_w * 0.20)
            shackle_x0 = lock_body_x0 + shackle_w_inset
            shackle_x1 = lock_body_x1 - shackle_w_inset
            shackle_y0 = lock_body_y0 - shackle_h
            draw.rectangle(
                (shackle_x0 - outline_w, shackle_y0 - outline_w, shackle_x1 + outline_w, lock_body_y0),
                fill=outline,
            )
            # Body: dark outline box with colour fill inside.
            draw.rectangle(
                (lock_body_x0 - outline_w, lock_body_y0 - outline_w, lock_body_x1 + outline_w, lock_body_y1 + outline_w),
                fill=outline,
            )
            draw.rectangle(
                (lock_body_x0, lock_body_y0, lock_body_x1, lock_body_y1),
                fill=ICON_HIGHLIGHT,
            )
            # Keyhole: tiny dark dot centered in the lock body.
            key_r = max(0.8, lock_body_w * 0.16)
            draw.ellipse(
                (lock_cx - key_r, (lock_body_y0 + lock_body_y1) / 2.0 - key_r,
                 lock_cx + key_r, (lock_body_y0 + lock_body_y1) / 2.0 + key_r),
                fill=outline,
            )

    # ------------------------------------------------------------------
    # Video override: overlay key/gate icons on top of each frame's path.

    def save_video(
        self,
        record_id: str,
        puzzle_image: Image.Image,
        points: Sequence[Optional[Tuple[float, float]]],
        thickness: int = 5,
        color: Tuple[int, int, int] = (220, 30, 30),
        fps: int = 16,
        duration: float = 4.0,
        joint_style: str = "square",
    ) -> Optional[Path]:
        """Custom video writer with semantic icon commits and one continuous route.

        Key/gate state changes are painted before the route.  The route is
        then drawn through every visited cell centre, matching the static
        terminal renderer: icon margins remain visible, while no
        icon can punch a task-dependent hole in the formal red route.
        """
        self._last_video_fps = None
        self._last_video_num_frames = None
        if not self.video:
            return None

        video_path = self.solution_dir / f"{record_id}_solution.webm"
        # Filter out None sentinels (not expected for this variant, but be
        # defensive in case the base contract changes).
        point_list: List[Tuple[float, float]] = [
            (float(p[0]), float(p[1])) for p in points if p is not None
        ]

        pairs = self._video_pairs or ()
        padding = self._video_padding or (0, 0)
        route_cells = list(self._video_path_cells or ())
        key_step_index = {
            pair_index: route_cells.index(key_cell)
            for pair_index, (key_cell, _gate_cell, _color_idx) in enumerate(pairs)
            if key_cell in route_cells
        }
        gate_step_index = {
            pair_index: route_cells.index(gate_cell)
            for pair_index, (_key_cell, gate_cell, _color_idx) in enumerate(pairs)
            if gate_cell in route_cells
        }

        scene = self._video_scene or {}
        base_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Image.Image] = {}

        def _base_frame(completed_step_index: int) -> Image.Image:
            """Scene with the key/gate state committed, rendered at scene scale.

            The icons are ellipse-and-rectangle glyphs, so painting them straight
            onto a finished 512 frame rasterises them one-to-one and leaves the
            video's icons visibly coarser than the same icons in the input image
            -- an overlay added *after* the downsample, which
            forbids.  Re-rendering the scene instead puts every committed state
            through ``_render_maze``, which strokes at 4x and downsamples once,
            so an icon in a frame is pixel-for-pixel the icon in the input until
            its own state changes.  At most ``2 * pairs + 1`` states occur along
            the chain, and each is rendered once and cached.
            """

            collected = tuple(
                sorted(
                    index
                    for index in range(len(pairs))
                    if completed_step_index >= key_step_index.get(index, 10**9)
                )
            )
            opened = tuple(
                sorted(
                    index
                    for index in range(len(pairs))
                    if completed_step_index >= gate_step_index.get(index, 10**9)
                )
            )
            if not pairs or not scene:
                return puzzle_image.copy()
            if not collected and not opened:
                # The opening state is the supplied input itself; reusing it
                # keeps the first frame an exact hold of what the model was given.
                return puzzle_image.copy()
            cached = base_cache.get((collected, opened))
            if cached is None:
                cached = self._render_maze(
                    scene["grid"],
                    start=scene["start"],
                    goal=scene["goal"],
                    path=None,
                    pairs=pairs,
                    padding=padding,
                    canvas_dims=scene["canvas_dims"],
                    collected_pair_indices=set(collected),
                    open_pair_indices=set(opened),
                )
                base_cache[(collected, opened)] = cached
            return cached.copy()

        if len(point_list) < 2:
            frame = _base_frame(0)
            frames = [frame.copy() for _ in range(max(1, fps))]
            frames = pad_frames_to_wan_compatible_length(frames)
            if not encode_rgb_frames_to_webm(
                frames,
                video_path,
                fps=fps,
                extra_args=["-g", "1", "-bf", "0"],
            ):
                return None
            self._last_video_fps = fps
            self._last_video_num_frames = len(frames)
            return video_path

        # Pre-compute cumulative lengths for smooth sub-edge interpolation.
        cumulative: List[float] = [0.0]
        for p1, p2 in zip(point_list, point_list[1:]):
            cumulative.append(cumulative[-1] + math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        total_len = cumulative[-1]

        eff_duration = min(duration, 10.0)
        total_frames = int(fps * eff_duration)

        frames: List[Image.Image] = []
        for frame_idx in range(total_frames + 1):
            progress = frame_idx / total_frames if total_frames > 0 else 1.0
            target_len = total_len * progress
            visible: List[Tuple[float, float]] = [point_list[0]]
            completed_step_index = 0
            for seg_idx, (p1, p2) in enumerate(zip(point_list, point_list[1:]), start=1):
                seg_start = cumulative[seg_idx - 1]
                seg_end = cumulative[seg_idx]
                if target_len >= seg_end - 1e-6:
                    visible.append(p2)
                    completed_step_index = seg_idx
                    continue
                if target_len > seg_start:
                    ratio = (target_len - seg_start) / max(1e-6, seg_end - seg_start)
                    visible.append((
                        p1[0] + (p2[0] - p1[0]) * ratio,
                        p1[1] + (p2[1] - p1[1]) * ratio,
                    ))
                break

            # Commit the current key/gate state first, then keep the formal
            # route visibly continuous through the contacted cell.  Drawing
            # icons last made three-pair videos lose enough route pixels to
            # fail their own source-replay terminal check.
            frame = _base_frame(completed_step_index)
            if len(visible) >= 2:
                draw_path_line(frame, visible, color, thickness, joint_style=joint_style)
            # Pawn marker at the current tip so revisit frames are animated.
            if visible:
                _draw_pawn_marker(ImageDraw.Draw(frame), visible[-1], thickness, color)
            frames.append(frame)

        if frames:
            # The formal visual program starts with an exact hold of the
            # supplied puzzle.  ``frames[0]`` already contains the animated
            # pawn marker, so duplicating it made the video differ from its
            # own input before any route motion began.
            frames = [puzzle_image.copy() for _ in range(4)] + frames

        # End hold: committed icon states below the complete continuous path.
        final_frame = _base_frame(len(point_list) - 1)
        draw_path_line(final_frame, point_list, color, thickness, joint_style=joint_style)
        _draw_pawn_marker(ImageDraw.Draw(final_frame), point_list[-1], thickness, color)
        for _ in range(max(4, fps // 2)):
            frames.append(final_frame.copy())

        frames = pad_frames_to_wan_compatible_length(frames)
        if not encode_rgb_frames_to_webm(
            frames,
            video_path,
            fps=fps,
            extra_args=["-g", "1", "-bf", "0"],
        ):
            return None
        self._last_video_fps = fps
        self._last_video_num_frames = len(frames)
        return video_path

__all__ = ["MazeKeysAndLocksGenerator"]
