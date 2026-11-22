"""Shortest full-exit plans for a fixed-axis sliding-vehicle board.

6x6 parking lot. 4-6 rectangular cars (length 2 or 3 cells), one red target
car is horizontal length-2 in a mid row. The player slides each car along
its own axis without overlapping any other; the goal is to slide the red
car off the right edge through the exit gate. Unlike every other maze
variant, the answer is NOT a path of cell IDs — each move names the car
and that car's new head position as ``(car_id, (x, y))``.

Load-bearing quality gates (Core Rule #7):
- Optimal solution length >= 6 moves (BFS over board states).
- The red car cannot exit with a single move in the initial state — at
  least one non-red car must move first, so some other car is actually
  blocking the red car's lane.
- At least one non-red car physically occupies a cell of the red car's
  exit lane, verifying that the blockers are real cars rather than an
  artifact of grid geometry. (When only one car blocks, removing it
  opens the lane outright; when several cars each block a different
  exit-lane cell, the puzzle is still non-trivial — every lane cell
  still needs to be vacated before the red car can slide through.)

The model-facing head coordinate is 1-based, with x as row and y as
column. The red car's fully exited head is therefore in column 7.
"""

from __future__ import annotations

from data.common.sample_identity import RECORD_NAMESPACE
from data.common.assets import render_font

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid



from PIL import Image, ImageChops, ImageDraw, ImageFont

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
)
from data.common.prompts import build_sectioned_prompt
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.pipeline.path_tasks import PathTaskGenerator, PathTaskRecord
from data.common.video_encoding import MAX_VIDEO_FRAMES, encode_rgb_frames_to_webm, pad_frames_to_wan_compatible_length


# ----------------------------------------------------------------------
# Visual palette
# ----------------------------------------------------------------------

ASPHALT_COLOR = (239, 241, 243)
LOT_EDGE_COLOR = (122, 128, 136)
GRID_LINE_COLOR = (202, 207, 212)
BACKGROUND_COLOR = ASPHALT_COLOR
EXIT_ASPHALT_COLOR = (246, 232, 232)
BARRIER_RED = (208, 54, 58)

# Target (red) car palette
RED_CAR_BODY = (214, 48, 48)
RED_CAR_DARK = (150, 24, 28)
RED_CAR_WINDOW = (250, 190, 190)

# Non-red distinct flat colours (body, darker edge, lighter windshield).
# Every car other than the red target is picked from this list WITHOUT
# replacement, so the puzzle ships 5 unique secondary cars at most.
DISTRACTOR_CAR_PRESETS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "blue",
        "body": (52, 96, 196),
        "dark": (26, 50, 118),
        "window": (188, 210, 250),
    },
    {
        "name": "green",
        "body": (52, 148, 82),
        "dark": (22, 82, 46),
        "window": (196, 232, 208),
    },
    {
        "name": "yellow",
        "body": (232, 196, 52),
        "dark": (152, 122, 22),
        "window": (250, 234, 168),
    },
    {
        "name": "purple",
        "body": (138, 64, 186),
        "dark": (76, 32, 110),
        "window": (220, 196, 244),
    },
    {
        "name": "teal",
        "body": (36, 168, 168),
        "dark": (16, 92, 98),
        "window": (186, 232, 232),
    },
    {
        "name": "orange",
        "body": (228, 118, 44),
        "dark": (146, 70, 18),
        "window": (250, 208, 168),
    },
)


# ----------------------------------------------------------------------

Car = Dict[str, Any]
# car schema:
# id: 1-based int, assigned by row-major head-cell order (top-to-bottom,
# within each row left-to-right). The "head cell" is the top-left
# cell of the car body. The red target car has whatever id its head
# cell sorts to under that ordering — it is NOT pinned to id=1.
# row, col: top-left cell of the car (row=r, col=c)
# orientation: "H" or "V"
# length: 2 or 3
# color: swatch name (e.g. "red", "blue")
# body, dark, window: rgb tuples


def _car_cells(car: Car) -> List[Tuple[int, int]]:
    r = int(car["row"])
    c = int(car["col"])
    n = int(car["length"])
    if car["orientation"] == "H":
        return [(r, c + i) for i in range(n)]
    return [(r + i, c) for i in range(n)]


def _renumber_cars_row_major(cars: List[Car]) -> List[Car]:
    """Sort `cars` by row-major head-cell order and reassign 1-based ids.

    The head cell is the top-left cell of each car's body — i.e. (row, col)
    as stored on the car dict. Sorting by (row, col) gives a deterministic
    "scan rows top-to-bottom, within each row left-to-right" order, which
    matches the rule advertised in the VLM prompt. Returns the sorted list
    with the car dicts mutated in place to carry their new ids.
    """
    cars.sort(key=lambda car: (int(car["row"]), int(car["col"])))
    for idx, car in enumerate(cars):
        car["id"] = idx + 1
    return cars


def _find_red_car(cars: Sequence[Car]) -> Car:
    """Return the unique red target car (color == "red")."""
    for car in cars:
        if str(car.get("color")) == "red":
            return car
    raise RuntimeError("maze_rush_hour: no red target car found in layout")


def _red_car_id(cars: Sequence[Car]) -> int:
    """Return the 1-based id of the red target car under the current numbering."""
    return int(_find_red_car(cars)["id"])


def _board_state(cars: Sequence[Car]) -> Tuple[Tuple[int, int, int], ...]:
    """Canonical board state hash: tuple per-car of (id, row, col).
    Orientation and length are invariants across moves so we only track
    the sliding position.
    """
    return tuple((int(car["id"]), int(car["row"]), int(car["col"])) for car in cars)


def _occupied_grid(cars: Sequence[Car], rows: int, cols: int) -> List[List[int]]:
    """Return a rows x cols grid of car-ids (0 if empty) for collision checks."""
    grid = [[0] * cols for _ in range(rows)]
    for car in cars:
        for (r, c) in _car_cells(car):
            grid[r][c] = int(car["id"])
    return grid


# ----------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------


class MazeRushHourGenerator(PathTaskGenerator):
    """Generate rush-hour sliding-car puzzles on a fixed 6x6 parking lot."""

    TASK_NAME = "maze_rush_hour"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/maze_rush_hour"
    DEFAULT_GRID_SIZE = 6
    DEFAULT_CELL_SIZE = 80  # 80 * 6 = 480, padded to 512

    MIN_SOLUTION_MOVES = 6
    MAX_SOLUTION_MOVES = 10
    MAX_LAYOUT_ATTEMPTS = 4000
    MAX_BFS_STATES = 200_000
    RENDER_SCALE = DOCTRINE_SUPERSAMPLE
    MOVE_FRAMES = 12
    PAUSE_FRAMES = 2
    INPUT_HOLD_FRAMES = 4
    FINAL_HOLD_FRAMES = 4

    COMMON_RULES = (
        "Rules and boundary cases: The board has rows and columns 1 through 6. Each visible C# badge identifies "
        "one rigid vehicle; C numbers start at C1 and are assigned by the visible top-left head cells in row-major order. A vehicle "
        "keeps its identity, colour, orientation, and length. One move selects exactly one vehicle and slides it by "
        "a positive integer number of whole cells along its displayed axis to a new head position. Every leading cell "
        "entered during the slide must be empty; vehicles may not overlap, rotate, "
        "change length, jump over a blocker, move diagonally, or stop between cells. All non-red vehicles must remain "
        "fully inside the 6 by 6 board. The horizontal red length-2 vehicle is the only vehicle allowed to cross a "
        "board edge, it may do so only east through the red bracket on its own row, and that crossing move must carry "
        "it exactly clear of the board, one cell past the eastern edge; stopping partly outside or sliding any further is illegal. "
        "Vacated cells become empty. No other vehicle moves, falls, compacts, refills, reorders, disappears, changes "
        "appearance, or is created. The goal is the first state in which the red vehicle is fully off the board. "
        "Return a globally shortest move sequence; every legal shortest sequence is accepted. Published samples use "
        "four through six total vehicles and require six through ten moves."
    )
    DEFAULT_TI2V_PROMPT = COMMON_RULES + (
        " Replay the shortest plan in a fixed-camera video. Slide one rigid vehicle continuously along its axis per "
        "action with its C badge riding along, pass only through empty cells, hold each committed state, and finish "
        "with the red vehicle fully exited. "
        "Do not teleport, hard-cut, crossfade, flicker, resize, rotate, duplicate, or exchange identities."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = COMMON_RULES + (
        " Return one terminal image with the red vehicle fully beyond the east exit and every other rigid vehicle at "
        "its correct whole-cell position. Preserve the grid, exit bracket, identities, background, and camera."
    )

    PUZZLE_INPUT_DESCRIPTION = (
        "The input image shows a clean light-grey 6 by 6 sliding-vehicle grid with thin cell lines, a red open "
        "exit bracket on the right edge, one horizontal red target car of length 2 in a middle row, and three to "
        "five other flat coloured cars of length 2 or 3. Each car carries one white-on-dark \"C{i}\" badge at its "
        "center showing that car's 1-based id; its head is its top-left occupied grid position. Shadows, windows, "
        "lane dashes, cell numbers, and other decorative "
        f"details are omitted. {GRID_XY_CONVENTION_SENTENCE}"
    )
    # The badge-bearing input is the one shared source for every modality.
    REASONING_INPUT_DESCRIPTION = PUZZLE_INPUT_DESCRIPTION

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        canvas_width: Optional[int] = None,
        aspect: Optional[float] = None,
        size: Optional[int] = None,
        grid_size: int = DEFAULT_GRID_SIZE,
        cell_size: Optional[int] = None,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        show_cell_id: bool = False,
        video: bool = False,
    ) -> None:
        if grid_size != self.DEFAULT_GRID_SIZE:
            raise ValueError("maze_rush_hour is specified at 6x6 only")

        effective_cell_size = int(
            cell_size if cell_size is not None else (size if size is not None else self.DEFAULT_CELL_SIZE)
        )
        if effective_cell_size % 2 == 1:
            effective_cell_size -= 1
        if effective_cell_size <= 0:
            raise ValueError("cell_size must be positive")

        base_width = grid_size * effective_cell_size
        final_width = int(canvas_width) if canvas_width is not None else max(base_width, 512)
        if final_width < base_width:
            final_width = base_width
        # Use a 512-pixel canvas with the default cell size.
        if canvas_width is None and effective_cell_size == self.DEFAULT_CELL_SIZE:
            final_width = 512
        final_height = final_width

        extra_w = final_width - base_width
        pad_left = extra_w // 2
        pad_right = extra_w - pad_left
        extra_h = final_height - base_width
        pad_top = extra_h // 2
        pad_bottom = extra_h - pad_top

        aspect_for_super = (final_width / final_height) if final_height else None
        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        super().__init__(
            resolved_output,
            canvas_width=final_width,
            aspect=aspect_for_super,
            size=effective_cell_size,
            seed=seed,
            ti2v_prompt=ti2v_prompt,
            show_cell_id=show_cell_id,
            video=video,
        )
        # ``PathTaskGenerator.next_id`` still uses uuid4.  The CLI supplies a
        # deterministic seed per logical record, so keep the identity on that
        # same semantic source instead of process-local entropy.  This is
        # essential for byte-identical 1/2/4-worker exports.
        self.seed = seed
        self._record_counter = 0
        # The maze base supplies route-oriented generic wording. This task has
        # a typed vehicle-state contract, so keep all three model-facing
        # prompts task-local and singular instead of inheriting generic route
        # keypoints or image-after-answer instructions.
        self.ti2v_prompt = build_sectioned_prompt(
            self.PUZZLE_INPUT_DESCRIPTION, self.DEFAULT_TI2V_PROMPT, mode="ti2v"
        )
        self.solution_image_prompt = build_sectioned_prompt(
            self.REASONING_INPUT_DESCRIPTION,
            self.DEFAULT_SOLUTION_IMAGE_PROMPT,
            mode="solution_image",
        )
        self.grid_size = int(grid_size)
        self.cell_size = effective_cell_size
        self.padding = (pad_left, pad_top, pad_right, pad_bottom)

    def next_id(self) -> str:
        identity = (
            f"{RECORD_NAMESPACE}:{self.TASK_FAMILY}:{self.TASK_NAME}:"
            f"{self.seed!r}:{self._record_counter}"
        )
        self._record_counter += 1
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> PathTaskRecord:
        puzzle_uuid = puzzle_id or self.next_id()

        cars, move_sequence = self._build_puzzle_with_gates()
        analysis = self._analyze_bfs(cars)
        if analysis is None or analysis[0] != move_sequence:
            raise RuntimeError("record construction requires the recomputed canonical shortest plan")
        _, shortest_path_count, explored_states = analysis

        rows = self.grid_size
        cols = self.grid_size
        # 1-based contiguous cell ids over ALL 36 grid cells in row-major order.
        all_cells: List[Tuple[int, int]] = [(r, c) for r in range(rows) for c in range(cols)]
        self._rebuild_cell_id_index(all_cells)

        pad_left, pad_top, pad_right, pad_bottom, canvas_dims = self._compute_padding()
        cell_bboxes = self._compute_cell_bboxes(pad_left, pad_top)

        red_car = _find_red_car(cars)
        exit_row = int(red_car["row"])  # red car row == exit row

        # Render the two canonical images. The C# badges are part of the
        # typed answer grammar, so the single shared input image carries them
        # for every modality; cell IDs are not part of this task.
        puzzle_image = self._render_lot(
            cars=cars,
            draw_cell_ids=bool(self.show_cell_id),
            draw_car_badges=True,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
            exit_row=exit_row,
        )

        # The terminal image keeps the identity badge attached to every
        # remaining car. The red car's terminal head is column 6, hence both
        # it and its badge are fully outside the canvas.
        final_cars = self._apply_final_state(cars, move_sequence)
        solution_image = self._render_lot(
            cars=final_cars,
            draw_cell_ids=False,
            draw_car_badges=True,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
            exit_row=exit_row,
            red_extra_east_offset=self._red_exit_offset_px(final_cars),
        )

        puzzle_path, solution_path = self.save_images(puzzle_uuid, puzzle_image, solution_image)

        video_path: Optional[Path] = None
        if self.video:
            video_path = self._save_rush_hour_video(
                puzzle_uuid,
                cars,
                move_sequence,
                padding=(pad_left, pad_top),
                canvas_dims=canvas_dims,
                exit_row=exit_row,
            )

        # Assemble the serializable record payload. `id` and `car_id` carry
        # the same value — the latter is the field name used by the answer
        # schema and the move-sequence entries, so we expose it here too
        # for symmetric lookup downstream.
        cars_payload = []
        for car in cars:
            car_id_int = int(car["id"])
            cars_payload.append(
                {
                    "id": car_id_int,
                    "car_id": car_id_int,
                    "row": int(car["row"]),
                    "col": int(car["col"]),
                    "orientation": str(car["orientation"]),
                    "length": int(car["length"]),
                    "color": str(car["color"]),
                    "start_cell_id": self._get_cell_id(int(car["row"]), int(car["col"])),
                }
            )

        # start_point / goal_point for the base record: center of red car's
        # starting cell (start) and the pixel just outside the exit gate (goal).
        start_center = self._cell_pixel_center(red_car["row"], red_car["col"], pad_left, pad_top)
        goal_center = (
            float(pad_left + cols * self.cell_size + self.cell_size * 0.5),
            float(pad_top + red_car["row"] * self.cell_size + self.cell_size * 0.5),
        )

        extra_payload: Dict[str, Any] = {
            "grid_size": [rows, cols],
            "cell_size": self.cell_size,
            "canvas_dimensions": [int(canvas_dims[0]), int(canvas_dims[1])],
            "padding": [pad_left, pad_top, pad_right, pad_bottom],
            "cell_bboxes": cell_bboxes,
            "cars": cars_payload,
            "exit_row": int(exit_row),
            "answer_format": "vehicle_head_sequence",
            "rendered_glyph_inventory": {
                "fixed_labels": [f"C{int(car['id'])}" for car in cars],
                "axis_row_labels": [],
                "axis_col_labels": [],
            },
            "rendered_glyph_schema": {
                "fixed_labels": "one stable C-numbered identity badge per visible vehicle",
                "axis_row_labels": "none; rows are counted from the top under the prompt convention",
                "axis_col_labels": "none; columns are counted from the left under the prompt convention",
            },
            "path_joint_style": "square",
            "solution_move_count": len(move_sequence),
            "shortest_path_count": int(shortest_path_count),
            "explored_state_count": int(explored_states),
            "task_contract": {
                "schema_version": "fixed_axis_vehicle_full_exit_v2",
                "input_count": 1,
                "canvas_size": [512, 512],
                "active_modalities": ["ti2v"],
                "native_media": True,
                "render_profile": "supersampled_flat_fixed_axis_full_exit_v1",
                "video": {
                    "fps": 16,
                    "move_frames": self.MOVE_FRAMES,
                    "pause_frames": self.PAUSE_FRAMES,
                    "maximum_frames": 149,
                    "wan_compatible_4n_plus_1": True,
                    "normalization_policy": "preserve_native_pixels",
                },
            },
        }

        record = self.build_record(
            puzzle_uuid,
            start_point=start_center,
            goal_point=goal_center,
            puzzle_path=puzzle_path,
            solution_path=solution_path,
            ti2v_prompt=self.ti2v_prompt,
            extra=extra_payload,
            video_path=video_path,
        )
        final_answer = self._format_answer(cars, move_sequence)
        record.answer = final_answer
        return record

    def _format_answer(
        self,
        cars: Sequence[Car],
        moves: Sequence[Dict[str, Any]],
    ) -> str:
        from .evaluator import _format_answer, _visible_problem

        record = {
            "grid_size": [self.grid_size, self.grid_size],
            "exit_row": int(_find_red_car(cars)["row"]),
            "cars": [
                {
                    "id": int(car["id"]),
                    "row": int(car["row"]),
                    "col": int(car["col"]),
                    "orientation": str(car["orientation"]),
                    "length": int(car["length"]),
                    "color": str(car["color"]),
                }
                for car in cars
            ],
        }
        problem = _visible_problem(record)
        if problem is None:
            raise RuntimeError("cannot format a plan for an invalid visible rush-hour state")
        return _format_answer(problem[0], problem[1], problem[2], moves)

    def rehydrate_from_record(self, record: Dict[str, Any]) -> None:
        """Rebuild the 1-based cell ID index for a 6x6 grid (row-major)."""
        grid_size = record.get("grid_size")
        if isinstance(grid_size, (list, tuple)) and len(grid_size) >= 2:
            rows = int(grid_size[0])
            cols = int(grid_size[1])
        else:
            rows = cols = self.DEFAULT_GRID_SIZE
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        self._rebuild_cell_id_index(cells)

    # ------------------------------------------------------------------
    # Layout generation (with load-bearing gates)
    # ------------------------------------------------------------------

    def _build_puzzle_with_gates(self) -> Tuple[List[Car], List[Dict[str, Any]]]:
        """Generate a puzzle that satisfies all load-bearing gates.

        Gates:
          1. BFS finds a solution in >= MIN_SOLUTION_MOVES moves.
          2. Red car cannot exit with a single move in the initial state.
          3. At least one non-red car, if removed, opens an exit lane for red
             directly (puzzle's blockers are physical, not decorative).
        """
        last_reason = "no attempt"
        for attempt in range(self.MAX_LAYOUT_ATTEMPTS):
            candidate = self._sample_random_layout()
            if candidate is None:
                last_reason = "sample_failed"
                continue
            cars = candidate
            # Gate 2a: a non-red car blocks the red car's exit lane in the
            # current state. If red could exit in one move, reject.
            if self._red_can_exit_in_one_move(cars):
                last_reason = "red_exits_trivially"
                continue
            # Gate 3: some non-red car, removed outright, would immediately
            # clear the exit lane for red. This forbids pathological cases
            # where the red car is only blocked by the edge of the grid
            # (can't happen here but keeps the gate explicit) or by the red
            # car itself (not possible).
            if not self._has_load_bearing_blocker(cars):
                last_reason = "no_load_bearing_blocker"
                continue
            # Gate 1: BFS returns the optimal solution.
            solution = self._solve_bfs(cars)
            if solution is None:
                last_reason = "unsolvable"
                continue
            if not self.MIN_SOLUTION_MOVES <= len(solution) <= self.MAX_SOLUTION_MOVES:
                last_reason = f"solution_outside_published_band_{len(solution)}"
                continue
            return cars, solution
        raise RuntimeError(
            f"maze_rush_hour: failed to generate a valid puzzle after "
            f"{self.MAX_LAYOUT_ATTEMPTS} attempts (last rejection: {last_reason})"
        )

    def _sample_random_layout(self) -> Optional[List[Car]]:
        """Sample a plausible layout of 4-6 cars including one red target."""
        rng = self.rng
        grid = self.grid_size
        occupied: Dict[Tuple[int, int], int] = {}
        cars: List[Car] = []

        # 1) Place the red target car: horizontal, length 2, in a "mid row"
        # (rows 1..4 inclusive — leave the top and bottom as aesthetic margins
        # so the exit-gate barrier never hugs a canvas edge).
        red_row = rng.randint(1, grid - 2)
        red_col = rng.randint(0, grid - 3)  # leave at least one cell for the exit
        red_cells = [(red_row, red_col + i) for i in range(2)]
        for cell in red_cells:
            occupied[cell] = 1
        cars.append(
            {
                "id": 1,
                "row": red_row,
                "col": red_col,
                "orientation": "H",
                "length": 2,
                "color": "red",
                "body": RED_CAR_BODY,
                "dark": RED_CAR_DARK,
                "window": RED_CAR_WINDOW,
            }
        )

        # 2) Place 3-5 distractor cars. We pick one that is GUARANTEED to
        # block the red car's exit lane, then a few more at random positions
        # without overlap.
        n_distractors = rng.randint(3, 5)
        presets = list(DISTRACTOR_CAR_PRESETS)
        rng.shuffle(presets)

        # Pre-computed exit-lane cells the red car must traverse to exit.
        # Every cell east of the red car's head through column grid-1 is an
        # exit-lane cell. At least one of these must be occupied by a
        # non-red car (satisfies both blocker and load-bearing gates below).
        exit_lane_cells = [(red_row, c) for c in range(red_col + 2, grid)]
        if not exit_lane_cells:
            return None

        # Force the first distractor to block the lane with a VERTICAL car so
        # the red car has a physical obstacle rather than the grid edge.
        first_preset = presets[0]
        blocker = self._place_vertical_blocker(
            preset=first_preset,
            exit_lane_cells=exit_lane_cells,
            occupied=occupied,
            car_id=len(cars) + 1,
        )
        if blocker is None:
            return None
        cars.append(blocker)

        # 3) Add the remaining distractors at random free positions with a
        # random orientation and length. Bias towards placing them somewhere
        # near the red car so the solution requires real interaction.
        for idx in range(1, n_distractors):
            preset = presets[idx % len(presets)]
            placed = self._place_random_car(
                preset=preset,
                occupied=occupied,
                car_id=len(cars) + 1,
                red_row=red_row,
            )
            if placed is None:
                # Not every attempt succeeds; return None so the outer loop
                # retries from scratch.
                return None
            cars.append(placed)

        # Reassign ids by row-major head-cell order so the rendered "C{i}"
        # badges and the canonical move-sequence ids both follow the rule
        # advertised in the VLM prompt. The placement-time ids above are
        # provisional — they only matter for occupancy bookkeeping and are
        # overwritten here.
        return _renumber_cars_row_major(cars)

    def _place_vertical_blocker(
        self,
        *,
        preset: Dict[str, Any],
        exit_lane_cells: Sequence[Tuple[int, int]],
        occupied: Dict[Tuple[int, int], int],
        car_id: int,
    ) -> Optional[Car]:
        """Place a vertical 2- or 3-length car that covers at least one of
        the given exit-lane cells without overlapping any existing car.
        """
        rng = self.rng
        grid = self.grid_size
        candidates: List[Tuple[int, int, int]] = []
        for length in (2, 3):
            for exit_cell in exit_lane_cells:
                er, ec = exit_cell
                # Place a vertical car such that one of its body cells is (er, ec)
                for offset in range(length):
                    top_row = er - offset
                    if top_row < 0 or top_row + length > grid:
                        continue
                    cells = [(top_row + k, ec) for k in range(length)]
                    if any(cell in occupied for cell in cells):
                        continue
                    candidates.append((top_row, ec, length))
        if not candidates:
            return None
        rng.shuffle(candidates)
        top_row, col, length = candidates[0]
        cells = [(top_row + k, col) for k in range(length)]
        for cell in cells:
            occupied[cell] = car_id
        return {
            "id": car_id,
            "row": top_row,
            "col": col,
            "orientation": "V",
            "length": length,
            "color": preset["name"],
            "body": preset["body"],
            "dark": preset["dark"],
            "window": preset["window"],
        }

    def _place_random_car(
        self,
        *,
        preset: Dict[str, Any],
        occupied: Dict[Tuple[int, int], int],
        car_id: int,
        red_row: int,
    ) -> Optional[Car]:
        """Place a car at a random orientation/length/position, preferring
        positions that interact with the red car's exit corridor."""
        rng = self.rng
        grid = self.grid_size

        # Build a pool of legal placements, then pick one at random.
        candidates: List[Tuple[int, int, int, str]] = []
        for orientation in ("H", "V"):
            for length in (2, 3):
                for r in range(grid):
                    for c in range(grid):
                        if orientation == "H":
                            if c + length > grid:
                                continue
                            cells = [(r, c + i) for i in range(length)]
                        else:
                            if r + length > grid:
                                continue
                            cells = [(r + i, c) for i in range(length)]
                        if any(cell in occupied for cell in cells):
                            continue
                        # Horizontal cars on the red row would drive in front of
                        # the red car's own exit lane; allow them but do not
                        # double-up the red car's own lane.
                        if orientation == "H" and r == red_row:
                            # H car on the red row is allowed anywhere except
                            # where it would block the red car's eastward exit
                            # if placed west of red (won't help the mechanic).
                            pass
                        candidates.append((r, c, length, orientation))
        if not candidates:
            return None
        rng.shuffle(candidates)
        # Prefer cars whose body cells are within 2 rows of the red car, so
        # the puzzle stays compact.
        def locality(entry: Tuple[int, int, int, str]) -> int:
            r, c, length, orientation = entry
            if orientation == "H":
                centre = r
            else:
                centre = r + length / 2.0
            return abs(centre - red_row)

        candidates.sort(key=locality)
        # Take from a shortlist (first half of sorted list), then randomise
        # inside it to keep variety.
        shortlist_len = max(8, len(candidates) // 4)
        shortlist = candidates[:shortlist_len]
        rng.shuffle(shortlist)
        r, c, length, orientation = shortlist[0]
        if orientation == "H":
            cells = [(r, c + i) for i in range(length)]
        else:
            cells = [(r + i, c) for i in range(length)]
        for cell in cells:
            occupied[cell] = car_id
        return {
            "id": car_id,
            "row": r,
            "col": c,
            "orientation": orientation,
            "length": length,
            "color": preset["name"],
            "body": preset["body"],
            "dark": preset["dark"],
            "window": preset["window"],
        }

    # ------------------------------------------------------------------
    # Solvers and quality gates
    # ------------------------------------------------------------------

    def _red_can_exit_in_one_move(self, cars: Sequence[Car]) -> bool:
        """Return True if the red car could slide east off the grid in a
        single move (the puzzle is trivial)."""
        grid = self._occupied_grid(cars)
        red_car = _find_red_car(cars)
        assert red_car["orientation"] == "H"
        r = red_car["row"]
        head_col = red_car["col"] + red_car["length"] - 1
        for c in range(head_col + 1, self.grid_size):
            if grid[r][c] != 0:
                return False
        return True  # clear lane → exits immediately

    def _has_load_bearing_blocker(self, cars: Sequence[Car]) -> bool:
        """Return True iff at least one non-red car physically occupies the
        red car's exit lane, so the blockers are real rather than
        decorative. When multiple non-red cars each block a different
        exit-lane cell the gate still passes (the puzzle is still
        non-trivial); what we forbid is the degenerate case where NO car
        occupies the lane — in which case the red car should have exited
        trivially and the earlier `_red_can_exit_in_one_move` gate would
        already have caught it.
        """
        red_car = _find_red_car(cars)
        red_id = int(red_car["id"])
        r = red_car["row"]
        head_col = red_car["col"] + red_car["length"] - 1
        exit_lane = {(r, c) for c in range(head_col + 1, self.grid_size)}
        if not exit_lane:
            return False
        for candidate in cars:
            if int(candidate["id"]) == red_id:
                continue
            candidate_cells = set(_car_cells(candidate))
            if candidate_cells & exit_lane:
                return True
        return False

    def _occupied_grid(self, cars: Sequence[Car]) -> List[List[int]]:
        return _occupied_grid(cars, self.grid_size, self.grid_size)

    def _solve_bfs(
        self,
        cars: Sequence[Car],
    ) -> Optional[List[Dict[str, Any]]]:
        analysis = self._analyze_bfs(cars)
        return analysis[0] if analysis is not None else None

    def _analyze_bfs(
        self,
        cars: Sequence[Car],
    ) -> Optional[Tuple[List[Dict[str, Any]], int, int]]:
        """Return one canonical shortest plan, all shortest paths, and states.

        A multi-cell slide costs one move. The only off-board successor is
        the red car at head column 6, so a partly outside stop and an
        overshoot are not members of the state graph.
        """
        initial_state = _board_state(cars)
        car_meta = {
            int(car["id"]): (str(car["orientation"]), int(car["length"]))
            for car in cars
        }
        red_id = _red_car_id(cars)
        rows = cols = self.grid_size

        def is_goal(state: Tuple[Tuple[int, int, int], ...]) -> bool:
            return next(col == cols for cid, _, col in state if cid == red_id)

        def successors(
            state: Tuple[Tuple[int, int, int], ...]
        ) -> List[Tuple[Tuple[Tuple[int, int, int], ...], Dict[str, Any]]]:
            grid = [[0] * cols for _ in range(rows)]
            for cid, row, col in state:
                orientation, length = car_meta[cid]
                if cid == red_id and col == cols:
                    continue
                for offset in range(length):
                    rr = row + (offset if orientation == "V" else 0)
                    cc = col + (offset if orientation == "H" else 0)
                    if 0 <= rr < rows and 0 <= cc < cols:
                        grid[rr][cc] = cid
            out: List[Tuple[Tuple[Tuple[int, int, int], ...], Dict[str, Any]]] = []
            for cid, row, col in state:
                orientation, length = car_meta[cid]
                if cid == red_id and col == cols:
                    continue
                if orientation == "H":
                    for distance in range(1, cols + 1):
                        next_col = col - distance
                        if next_col < 0 or grid[row][next_col] != 0:
                            break
                        out.append(
                            (
                                _updated_state(state, cid, row, next_col),
                                {"car_id": cid, "direction": "W", "cells": distance},
                            )
                        )
                    for distance in range(1, cols + 1):
                        next_col = col + distance
                        leading_col = col + length - 1 + distance
                        if leading_col < cols and grid[row][leading_col] != 0:
                            break
                        if cid == red_id:
                            if next_col > cols:
                                break
                            # Do not admit a state that stops with only part
                            # of the red car outside the board.
                            if next_col < cols and next_col + length > cols:
                                continue
                        elif leading_col >= cols:
                            break
                        out.append(
                            (
                                _updated_state(state, cid, row, next_col),
                                {"car_id": cid, "direction": "E", "cells": distance},
                            )
                        )
                else:
                    for distance in range(1, rows + 1):
                        next_row = row - distance
                        if next_row < 0 or grid[next_row][col] != 0:
                            break
                        out.append(
                            (
                                _updated_state(state, cid, next_row, col),
                                {"car_id": cid, "direction": "N", "cells": distance},
                            )
                        )
                    for distance in range(1, rows + 1):
                        next_tail = row + length - 1 + distance
                        if next_tail >= rows or grid[next_tail][col] != 0:
                            break
                        out.append(
                            (
                                _updated_state(state, cid, row + distance, col),
                                {"car_id": cid, "direction": "S", "cells": distance},
                            )
                        )
            return out

        queue: deque[Tuple[Tuple[int, int, int], ...]] = deque([initial_state])
        distance = {initial_state: 0}
        path_count = {initial_state: 1}
        parent: Dict[
            Tuple[Tuple[int, int, int], ...],
            Tuple[Tuple[Tuple[int, int, int], ...], Dict[str, Any]],
        ] = {}
        goal_state: Optional[Tuple[Tuple[int, int, int], ...]] = None
        goal_depth: Optional[int] = None
        while queue:
            if len(distance) > self.MAX_BFS_STATES:
                return None
            state = queue.popleft()
            depth = distance[state]
            if goal_depth is not None and depth >= goal_depth:
                continue
            for next_state, move in successors(state):
                next_depth = depth + 1
                if next_state not in distance:
                    distance[next_state] = next_depth
                    path_count[next_state] = path_count[state]
                    parent[next_state] = state, move
                    queue.append(next_state)
                elif distance[next_state] == next_depth:
                    path_count[next_state] += path_count[state]
                if is_goal(next_state):
                    if goal_depth is None:
                        goal_depth = next_depth
                        goal_state = next_state
        if goal_state is None or goal_depth is None:
            return None
        shortest_paths = sum(
            path_count[state]
            for state, state_distance in distance.items()
            if state_distance == goal_depth and is_goal(state)
        )
        moves: List[Dict[str, Any]] = []
        cursor = goal_state
        while cursor != initial_state:
            previous, move = parent[cursor]
            moves.append(move)
            cursor = previous
        moves.reverse()
        return moves, shortest_paths, len(distance)

    # ------------------------------------------------------------------
    # Final-state helpers
    # ------------------------------------------------------------------

    def _apply_final_state(
        self,
        initial_cars: Sequence[Car],
        moves: Sequence[Dict[str, Any]],
    ) -> List[Car]:
        """Return a deep-copied cars list reflecting the final state after
        applying every move. The terminal red head is exactly at column 6,
        so its complete body is outside the board. Non-red cars stay in-bounds
        by construction (BFS constraint).
        """
        cars = [dict(car) for car in initial_cars]
        for move in moves:
            self._apply_move(cars, move)
        return cars

    def _apply_move(self, cars: List[Car], move: Dict[str, Any]) -> None:
        cid = int(move["car_id"])
        direction = str(move["direction"])
        cells = int(move["cells"])
        for car in cars:
            if int(car["id"]) != cid:
                continue
            if direction == "E":
                car["col"] = int(car["col"]) + cells
            elif direction == "W":
                car["col"] = int(car["col"]) - cells
            elif direction == "N":
                car["row"] = int(car["row"]) - cells
            elif direction == "S":
                car["row"] = int(car["row"]) + cells
            return


    def _red_exit_offset_px(self, final_cars: Sequence[Car]) -> float:
        """Return the direct clearance across the fixed right canvas margin.

        The logical goal remains red-head column 6. The extra pixels are only
        the visible exit strip between the board edge and the canvas crop, so
        the final body does not leave an unexplained sliver in the frame.
        """
        red = _find_red_car(final_cars)
        if int(red["col"]) != self.grid_size:
            return 0.0
        return float(self.padding[2])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_lot(
        self,
        *,
        cars: Sequence[Car],
        draw_cell_ids: bool,
        padding: Tuple[int, int],
        canvas_dims: Tuple[int, int],
        exit_row: int,
        red_extra_east_offset: float = 0.0,
        move_traces: Optional[Sequence[Dict[str, Any]]] = None,
        draw_car_badges: bool = False,
    ) -> Image.Image:
        del move_traces
        native = self._render_lot_native(
            cars=cars,
            draw_cell_ids=draw_cell_ids,
            padding=padding,
            canvas_dims=canvas_dims,
            exit_row=exit_row,
            red_extra_east_offset=red_extra_east_offset,
            draw_car_badges=draw_car_badges,
            scale=self.RENDER_SCALE,
        )
        return finish_scaled_canvas(native, canvas_dims)

    def _render_lot_native(
        self,
        *,
        cars: Sequence[Car],
        draw_cell_ids: bool,
        padding: Tuple[int, int],
        canvas_dims: Tuple[int, int],
        exit_row: int,
        red_extra_east_offset: float = 0.0,
        draw_car_badges: bool = False,
        scale: int,
    ) -> Image.Image:
        pad_left, pad_top = (int(padding[0]) * scale, int(padding[1]) * scale)
        scaled_dims = (int(canvas_dims[0]) * scale, int(canvas_dims[1]) * scale)
        # Shared supersample allocation; this renderer multiplies its own
        # coordinates by ``scale``, so it draws through the raw handle.
        canvas, scaled_draw = new_scaled_canvas(
            int(canvas_dims[0]), int(canvas_dims[1]), scale, BACKGROUND_COLOR
        )
        draw = scaled_draw.raw

        grid = self.grid_size
        cs = self.cell_size * scale

        board_left = pad_left
        board_top = pad_top
        board_right = pad_left + grid * cs
        board_bottom = pad_top + grid * cs

        # Fill the whole canvas with asphalt so there is NO black letterbox.
        draw.rectangle((0, 0, scaled_dims[0] - 1, scaled_dims[1] - 1), fill=ASPHALT_COLOR)

        # Lot inner fill (very slightly darker to differentiate from border).
        draw.rectangle(
            (board_left, board_top, board_right - 1, board_bottom - 1),
            fill=ASPHALT_COLOR,
        )

        # Thin subtle grid lines so the cell structure reads.
        for k in range(grid + 1):
            x = pad_left + k * cs
            draw.line(((x, board_top), (x, board_bottom)), fill=GRID_LINE_COLOR, width=scale)
            y = pad_top + k * cs
            draw.line(((board_left, y), (board_right, y)), fill=GRID_LINE_COLOR, width=scale)

        # Outer lot edge (thicker).
        draw.rectangle(
            (board_left, board_top, board_right - 1, board_bottom - 1),
            outline=LOT_EDGE_COLOR,
            width=2 * scale,
        )

        # Paint the exit strip (slightly lighter asphalt) on the red car's row
        # from its east edge out through the canvas edge so the barrier and
        # any exiting red car live on a visually distinct lane.
        exit_strip_top = pad_top + exit_row * cs + 2 * scale
        exit_strip_bottom = pad_top + (exit_row + 1) * cs - 2 * scale
        exit_strip_left = board_right
        exit_strip_right = scaled_dims[0]
        if exit_strip_right > exit_strip_left:
            draw.rectangle(
                (exit_strip_left, exit_strip_top, exit_strip_right - 1, exit_strip_bottom - 1),
                fill=EXIT_ASPHALT_COLOR,
            )

        # Cars stay flat; their only decoration is the C badge layer below.
        for car in cars:
            extra = red_extra_east_offset * scale if str(car.get("color")) == "red" else 0.0
            self._draw_car(
                draw,
                car,
                pad_left,
                pad_top,
                extra_east_offset_px=extra,
                cell_size=cs,
            )

        # Minimal open exit bracket on the right edge of the target row.
        self._draw_exit_barrier(
            draw, pad_left, pad_top, exit_row, cs, grid, canvas_w=scaled_dims[0]
        )

        # Cell IDs (non-default show_cell_id configurations only).
        if draw_cell_ids:
            self._draw_cell_ids(draw, pad_left, pad_top, cars, cell_size=cs)

        # Car-id badges (when enabled). Drawn AFTER the cell IDs so
        # the small white-on-dark "C{i}" pill always sits on top of the
        # windshield band rather than being clipped by another label.
        if draw_car_badges:
            visible_badge_cars = [
                car
                for car in cars
                if not (str(car.get("color")) == "red" and int(car["col"]) >= self.grid_size)
            ]
            self._draw_car_badges(draw, visible_badge_cars, pad_left, pad_top, cell_size=cs)

        return canvas


    def _draw_exit_barrier(
        self,
        draw: ImageDraw.ImageDraw,
        pad_left: int,
        pad_top: int,
        exit_row: int,
        cs: int,
        grid: int,
        *,
        canvas_w: Optional[int] = None,
    ) -> None:
        """Paint a simple red open bracket around the east exit lane."""
        x_edge = pad_left + grid * cs
        canvas_w = int(canvas_w if canvas_w is not None else self.canvas_dimensions[0])
        row_top = pad_top + exit_row * cs
        row_bottom = pad_top + (exit_row + 1) * cs
        arm = min(max(8, cs // 6), max(8, canvas_w - x_edge + 6))
        width = max(3, cs // 20)
        draw.line((x_edge - arm, row_top + 2, canvas_w - 1, row_top + 2), fill=BARRIER_RED, width=width)
        draw.line((x_edge - arm, row_bottom - 2, canvas_w - 1, row_bottom - 2), fill=BARRIER_RED, width=width)
        draw.line((x_edge - arm, row_top + 2, x_edge - arm, row_top + cs * 0.22), fill=BARRIER_RED, width=width)
        draw.line((x_edge - arm, row_bottom - cs * 0.22, x_edge - arm, row_bottom - 2), fill=BARRIER_RED, width=width)

    def _draw_car(
        self,
        draw: ImageDraw.ImageDraw,
        car: Car,
        pad_left: int,
        pad_top: int,
        *,
        extra_east_offset_px: float = 0.0,
        cell_size: Optional[int] = None,
    ) -> None:
        """Render a single fixed-size flat rounded car token."""
        cs = int(cell_size if cell_size is not None else self.cell_size)
        row = int(car["row"])
        col = int(car["col"])
        length = int(car["length"])
        orientation = str(car["orientation"])

        inset = max(3, cs // 14)
        if orientation == "H":
            cell_count_x = length
            cell_count_y = 1
        else:
            cell_count_x = 1
            cell_count_y = length

        x0 = pad_left + col * cs + inset + extra_east_offset_px
        y0 = pad_top + row * cs + inset
        x1 = pad_left + (col + cell_count_x) * cs - inset + extra_east_offset_px
        y1 = pad_top + (row + cell_count_y) * cs - inset

        body_radius = max(4, cs // 8)
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=body_radius,
            fill=car["body"],
            outline=car["dark"],
            width=max(2, cs // 32),
        )

    @staticmethod
    def _load_label_font(font_size: int) -> ImageFont.ImageFont:
        return render_font(font_size, bold=True)

    def _draw_cell_ids(
        self,
        draw: ImageDraw.ImageDraw,
        pad_left: int,
        pad_top: int,
        cars: Sequence[Car],
        *,
        cell_size: Optional[int] = None,
    ) -> None:
        cs = int(cell_size if cell_size is not None else self.cell_size)
        font_size = max(12, int(cs * 0.24))
        font = self._load_label_font(font_size)

        # Map each occupied cell to its covering car so we can pick a corner
        # that is NOT overlapped by the car's windshield band. For H cars
        # the windshield sits at the east end; for V cars at the north end.
        occupied_by: Dict[Tuple[int, int], Car] = {}
        for car in cars:
            for cell in _car_cells(car):
                occupied_by[cell] = car

        def _corner_for_car_cell(cell: Tuple[int, int], car: Car) -> Tuple[int, int, str]:
            """Return (x, y, anchor) for the numeral in a car-covered cell,
            avoiding the windshield band at the car's "front" end."""
            r, c = cell
            base_x = pad_left + c * cs
            base_y = pad_top + r * cs
            inset = max(4, cs // 10)
            if str(car["orientation"]) == "H":
                # Windshield covers cells near the east end. Placing the
                # numeral on the WEST side of every H car cell always
                # clears the band.
                return (base_x + inset, base_y + inset, "lt")
            # Vertical car: windshield at the top cell. Place the numeral
            # at the SOUTH-west corner of every V-car cell so it clears
            # the band.
            return (base_x + inset, base_y + cs - inset, "lb")

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_id = self._get_cell_id(r, c)
                if cell_id <= 0:
                    continue
                cx = pad_left + c * cs + cs / 2.0
                cy = pad_top + r * cs + cs / 2.0
                covering = occupied_by.get((r, c))
                if covering is not None:
                    label_x, label_y, anchor = _corner_for_car_cell((r, c), covering)
                    draw.text(
                        (label_x, label_y),
                        str(cell_id),
                        fill=(250, 250, 250),
                        font=font,
                        anchor=anchor,
                    )
                else:
                    draw.text(
                        (cx, cy),
                        str(cell_id),
                        fill=(0, 0, 255),
                        font=font,
                        anchor="mm",
                    )

    def _draw_car_badges(
        self,
        draw: ImageDraw.ImageDraw,
        cars: Sequence[Car],
        pad_left: int,
        pad_top: int,
        *,
        cell_size: Optional[int] = None,
    ) -> None:
        """Stamp the one required identity mark at each car's center."""
        cs = int(cell_size if cell_size is not None else self.cell_size)
        font_size = max(14, int(cs * 0.30))
        font = self._load_label_font(font_size)

        body_inset = max(3, cs // 14)
        for car in cars:
            row = int(car["row"])
            col = int(car["col"])
            length = int(car["length"])
            orientation = str(car["orientation"])

            if orientation == "H":
                cell_count_x = length
                cell_count_y = 1
            else:
                cell_count_x = 1
                cell_count_y = length

            x0 = pad_left + col * cs + body_inset
            y0 = pad_top + row * cs + body_inset
            x1 = pad_left + (col + cell_count_x) * cs - body_inset
            y1 = pad_top + (row + cell_count_y) * cs - body_inset
            badge_cx = (x0 + x1) / 2.0
            badge_cy = (y0 + y1) / 2.0
            label = f"C{int(car['id'])}"
            try:
                bbox = draw.textbbox((0, 0), label, font=font, anchor="lt")
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except (TypeError, AttributeError):
                # Older PIL fall-back.
                text_w, text_h = draw.textsize(label, font=font)  # type: ignore[attr-defined]

            pad_x = max(2, font_size // 4)
            pad_y = max(1, font_size // 6)
            pill_half_w = text_w / 2.0 + pad_x
            pill_half_h = text_h / 2.0 + pad_y
            body_w = max(0.0, x1 - x0)
            body_h = max(0.0, y1 - y0)
            if body_w > 0 and pill_half_w * 2 > body_w * 0.85:
                pill_half_w = body_w * 0.85 / 2.0
            if body_h > 0 and pill_half_h * 2 > body_h * 0.9:
                pill_half_h = body_h * 0.9 / 2.0
            pill_half_w = max(pill_half_w, 4.0)
            pill_half_h = max(pill_half_h, 4.0)
            pill_x0 = badge_cx - pill_half_w
            pill_y0 = badge_cy - pill_half_h
            pill_x1 = badge_cx + pill_half_w
            pill_y1 = badge_cy + pill_half_h
            pill_radius = max(2, int(min(pill_half_w, pill_half_h) * 0.6))
            backing = tuple(car.get("dark", (32, 32, 32)))
            draw.rounded_rectangle(
                (pill_x0, pill_y0, pill_x1, pill_y1),
                radius=pill_radius,
                fill=backing,
                outline=(248, 248, 252),
                width=max(1, font_size // 14),
            )
            draw.text(
                (badge_cx, badge_cy),
                label,
                fill=(248, 248, 252),
                font=font,
                anchor="mm",
            )

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------

    def _save_rush_hour_video(
        self,
        record_id: str,
        cars: Sequence[Car],
        move_sequence: Sequence[Dict[str, Any]],
        *,
        padding: Tuple[int, int],
        canvas_dims: Tuple[int, int],
        exit_row: int,
        fps: int = 16,
        frames_per_move: Optional[int] = None,
        pause_frames: Optional[int] = None,
    ) -> Optional[Path]:
        """Render a smooth-slide animation of the solution move sequence.

        Strategy: maintain a MUTABLE cars list. For each move, produce
        `frames_per_move` frames interpolating the moving car's pixel
        position along its axis. After each move, append `pause_frames`
        "held" frames. The final logical state already has the red car fully
        outside the right edge, and the last frames hold that exact state.
        """
        self._last_video_fps = None
        self._last_video_num_frames = None
        if not self.video:
            return None
        frames_per_move = int(frames_per_move or self.MOVE_FRAMES)
        pause_frames = int(pause_frames if pause_frames is not None else self.PAUSE_FRAMES)

        video_path = self.solution_dir / f"{record_id}_solution.webm"
        pad_left, pad_top = padding

        mutable_cars = [dict(car) for car in cars]

        frames: List[Image.Image] = []

        initial = self._render_lot(
            cars=mutable_cars,
            draw_cell_ids=False,
            draw_car_badges=True,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
            exit_row=exit_row,
        )
        frames.extend(initial.copy() for _ in range(self.INPUT_HOLD_FRAMES))

        for move in move_sequence:
            cid = int(move["car_id"])
            direction = str(move["direction"])
            cells_slid = int(move["cells"])

            animation_distance_px = cells_slid * self.cell_size
            selected_before = mutable_cars[self._find_car_index(mutable_cars, cid)]
            is_terminal_red_exit = bool(
                str(selected_before.get("color")) == "red"
                and direction == "E"
                and int(selected_before["col"]) + cells_slid == self.grid_size
            )
            if is_terminal_red_exit:
                terminal_state = [dict(car) for car in mutable_cars]
                self._apply_move(
                    terminal_state,
                    {"car_id": cid, "direction": direction, "cells": cells_slid},
                )
                animation_distance_px += self._red_exit_offset_px(terminal_state)

            if direction == "E":
                total_dx, total_dy = animation_distance_px, 0
            elif direction == "W":
                total_dx, total_dy = -animation_distance_px, 0
            elif direction == "N":
                total_dx, total_dy = 0, -animation_distance_px
            elif direction == "S":
                total_dx, total_dy = 0, animation_distance_px
            else:
                raise ValueError(f"Unknown direction: {direction}")

            target_car_idx = self._find_car_index(mutable_cars, cid)
            for f in range(1, frames_per_move + 1):
                progress = f / frames_per_move
                dx = total_dx * progress
                dy = total_dy * progress
                frame = self._render_frame_with_floating_car(
                    cars=mutable_cars,
                    floating_idx=target_car_idx,
                    floating_dx=dx,
                    floating_dy=dy,
                    padding=(pad_left, pad_top),
                    canvas_dims=canvas_dims,
                    exit_row=exit_row,
                )
                frames.append(frame)

            self._apply_move(mutable_cars, {"car_id": cid, "direction": direction, "cells": cells_slid})
            terminal_clearance = self._red_exit_offset_px(mutable_cars)
            settled = self._render_lot(
                cars=mutable_cars,
                draw_cell_ids=False,
                draw_car_badges=True,
                padding=(pad_left, pad_top),
                canvas_dims=canvas_dims,
                exit_row=exit_row,
                red_extra_east_offset=terminal_clearance,
            )
            if ImageChops.difference(frames[-1], settled).getbbox() is not None:
                raise RuntimeError("animated move endpoint does not equal the next static state")
            for _ in range(pause_frames):
                frames.append(settled.copy())

        # Final hold: red car fully off the right edge.
        end_hold_frames = self.FINAL_HOLD_FRAMES
        last_frame = frames[-1] if frames else self._render_lot(
            cars=mutable_cars,
            draw_cell_ids=False,
            draw_car_badges=True,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
            exit_row=exit_row,
        )
        for _ in range(end_hold_frames):
            frames.append(last_frame.copy() if hasattr(last_frame, "copy") else last_frame)

        frames = pad_frames_to_wan_compatible_length(frames)

        # Final invariants.
        assert (len(frames) - 1) % 4 == 0, f"frames length {len(frames)} not 4N+1"
        assert len(frames) <= MAX_VIDEO_FRAMES, f"frames length {len(frames)} exceeds VideoGym video budget"
        assert len(frames) <= 149, f"frames length {len(frames)} exceeds task contract"

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

    def _render_frame_with_floating_car(
        self,
        *,
        cars: Sequence[Car],
        floating_idx: int,
        floating_dx: float,
        floating_dy: float,
        padding: Tuple[int, int],
        canvas_dims: Tuple[int, int],
        exit_row: int,
    ) -> Image.Image:
        """Render a lot where one car is drawn at a sub-cell pixel offset.

        This is used to animate the in-flight portion of a move. The other
        cars render at their integer-cell positions. The floating car is
        painted LAST so it sits on top.
        """
        pad_left, pad_top = padding
        scale = self.RENDER_SCALE
        scaled_pad_left, scaled_pad_top = pad_left * scale, pad_top * scale
        scaled_dims = (canvas_dims[0] * scale, canvas_dims[1] * scale)
        scaled_cell = self.cell_size * scale
        # Render the base (all other cars + empty floating slot).
        background_cars = [car for i, car in enumerate(cars) if i != floating_idx]
        canvas = self._render_lot_native(
            cars=background_cars,
            draw_cell_ids=False,
            draw_car_badges=True,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
            exit_row=exit_row,
            scale=scale,
        )
        draw = ImageDraw.Draw(canvas)

        floating_car = cars[floating_idx]
        # For horizontal sliding (non-red cars), dx is the only offset.
        # For vertical sliding, dy is the only offset.
        # For the red car exiting east, dx may push past the grid edge.
        # We draw via the existing _draw_car helper using a synthetic car dict
        # with its col/row unchanged and an extra pixel offset injected.
        synthetic = dict(floating_car)
        # Translate the dx/dy pixel shift into a float "extra" offset on one axis.
        # For H cars that's dx directly; for V cars we reuse the dy by
        # drawing at a shifted y via a tiny wrapper.
        if str(floating_car["orientation"]) == "H":
            self._draw_car(
                draw,
                synthetic,
                scaled_pad_left,
                scaled_pad_top,
                extra_east_offset_px=floating_dx * scale,
                cell_size=scaled_cell,
            )
        else:
            # Vertical move: shift the y coordinate by floating_dy. We do this
            # by temporarily adjusting the synthetic row via a float offset:
            # draw a shifted version by painting on an offset pad_top.
            shifted_pad_top = scaled_pad_top + int(round(floating_dy * scale))
            self._draw_car(
                draw,
                synthetic,
                scaled_pad_left,
                shifted_pad_top,
                cell_size=scaled_cell,
            )
        # Match `_render_lot` z-order exactly: the open exit bracket is a
        # fixed board mark and remains above any car crossing the exit row.
        self._draw_exit_barrier(
            draw,
            scaled_pad_left,
            scaled_pad_top,
            exit_row,
            scaled_cell,
            self.grid_size,
            canvas_w=scaled_dims[0],
        )
        # The identity badge rides with its sliding car: shift the badge
        # anchor by the same in-flight pixel offset used for the body.
        self._draw_car_badges(
            draw,
            [synthetic],
            scaled_pad_left + int(round(floating_dx * scale)),
            scaled_pad_top + int(round(floating_dy * scale)),
            cell_size=scaled_cell,
        )
        return finish_scaled_canvas(canvas, canvas_dims)

    @staticmethod
    def _find_car_index(cars: Sequence[Car], car_id: int) -> int:
        for idx, car in enumerate(cars):
            if int(car["id"]) == int(car_id):
                return idx
        raise KeyError(f"Car id {car_id} not found")

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _compute_padding(self) -> Tuple[int, int, int, int, Tuple[int, int]]:
        pad_left, pad_top, pad_right, pad_bottom = self.padding
        return pad_left, pad_top, pad_right, pad_bottom, self.canvas_dimensions

    def _compute_cell_bboxes(self, pad_left: int, pad_top: int) -> List[List[Tuple[int, int, int, int]]]:
        bboxes: List[List[Tuple[int, int, int, int]]] = []
        for r in range(self.grid_size):
            row_boxes: List[Tuple[int, int, int, int]] = []
            for c in range(self.grid_size):
                left = pad_left + c * self.cell_size
                top = pad_top + r * self.cell_size
                row_boxes.append((left, top, left + self.cell_size, top + self.cell_size))
            bboxes.append(row_boxes)
        return bboxes

    def _cell_pixel_center(self, r: int, c: int, pad_left: int, pad_top: int) -> Tuple[float, float]:
        x = pad_left + c * self.cell_size + self.cell_size / 2.0
        y = pad_top + r * self.cell_size + self.cell_size / 2.0
        return (x, y)

    def _get_cell_id(self, r: int, c: int) -> int:
        return self._cell_to_id.get((r, c), 0)

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        row, col = self._id_to_cell[cell_id]
        pad_left, pad_top, _, _, _ = self._compute_padding()
        return self._cell_pixel_center(row, col, pad_left, pad_top)

# ----------------------------------------------------------------------
# BFS helper
# ----------------------------------------------------------------------


def _updated_state(
    state: Tuple[Tuple[int, int, int], ...],
    car_id: int,
    new_row: int,
    new_col: int,
) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(
        (cid, new_row if cid == car_id else r, new_col if cid == car_id else c)
        for cid, r, c in state
    )


__all__ = ["MazeRushHourGenerator", "DISTRACTOR_CAR_PRESETS"]
