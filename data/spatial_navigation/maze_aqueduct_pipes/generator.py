"""Roman-aqueduct rotatable-pipe maze generator.

Not a walking maze, and NOT the static "trace pre-assembled pipes" version
either. A 9x9 grid of Roman-aqueduct sandstone tiles. Some tiles carry pipe
fragments of four canonical kinds:

  * STRAIGHT  -- two collinear openings (a pipe pass-through).
  * ELBOW     -- two perpendicular openings (a corner).
  * T         -- three openings (a T-junction).
  * CROSS     -- four openings (a cross-junction).

Every pipe tile is rendered at one of four cardinal rotations (0, 90, 180,
270 degrees). In the puzzle image the tiles are shown at randomly-chosen
DISPLAYED rotations. At least half of the path tiles are misaligned so a
naive "trace the existing channels" reading produces a broken aqueduct that
spills somewhere short of the fountain. Exactly one assignment of rotations
makes the chain connect a blue water-drop on the LEFT edge to a bronze
fountain basin on the RIGHT edge, and the model's task is to return the
ordered list of 1-based ``(x, y)`` grid positions -- ``x`` the row counted
from the top, ``y`` the column counted from the left -- that the water
traverses AFTER each tile has been rotated to its correct orientation.

Layout guarantees:

* Tier A load-bearing check (mechanic must be load-bearing): we simulate
  flow on the puzzle's CURRENT (displayed) rotations using the same
  ``_simulate_flow`` semantics. If that simulation already reaches the
  fountain, the rotation mechanic is decorative -- regenerate. Without
  rotation, NO answer exists; with the corrected rotations, exactly one
  answer exists.
* Spread guard: pipes + source + sink span at least 80% of the grid on
  both axes, and the flow path itself spans at least 66% on both axes.
* Decoy tiles are placed on off-path cells at fully-random rotations and
  are explicitly forbidden from forming a competing flow into the path.
* The grid body carries no per-tile glyph. 1-based row numbers run down the
  left gutter and 1-based column numbers across the top gutter, which is the
  only text the scene renders. Internal cell ids stay an implementation
  detail of the layout solver and never reach the model-facing prompt,
  answer, or image.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.pipeline.path_tasks import (
    GRID_XY_CONVENTION_SENTENCE,
    PathTaskGenerator,
    PathTaskRecord,
    cell_to_xy,
    draw_grid_axis_labels,
    draw_path_line,
    grid_axis_label_layout,
)
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import (
    encode_rgb_frames_to_webm,
    pad_frames_to_wan_compatible_length,
)


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# Every scene (input, terminal image, and every video frame) is drawn on a
# canvas allocated at this factor and LANCZOS-downsampled exactly once.
SCENE_SUPERSAMPLE = DOCTRINE_SUPERSAMPLE
DEFAULT_GRID = 9
# 9 * 50 = 450 on a 512 canvas leaves 62 px, enough for the left/top axis-label
# gutter shared with the other 1-based (x, y) route tasks. Keep the cell size
# even so the pipe legs and the flow stroke stay centred on whole pixels.
DEFAULT_CELL_SIZE = 50
MAX_LAYOUT_ATTEMPTS = 4000
MIN_PATH_LENGTH = 9
MAX_PATH_LENGTH = 17
MIN_FLOW_BBOX_FRACTION = 0.66
MIN_BBOX_FRACTION = 0.80
# Fraction of path tiles forced to start at a misaligned rotation (so the
# rotation mechanic is visibly load-bearing in the puzzle image).
MIN_MISALIGNED_FRACTION = 0.50

# ---------------------------------------------------------------------------
# Palette -- warm sandstone aqueduct + cool blue water + subtle moss accents.
# ---------------------------------------------------------------------------

BACKGROUND_COLOR = (255, 255, 255)
STONE_BASE_COLOR = (198, 166, 120)
STONE_EDGE_COLOR = (148, 116, 78)
STONE_HIGHLIGHT_COLOR = (224, 196, 152)
STONE_MORTAR_COLOR = (102, 78, 52)
BLANK_TILE_COLOR = (172, 140, 96)
MOSS_COLOR = (92, 134, 72)
PIPE_OUTLINE_COLOR = (62, 38, 20)
PIPE_INTERIOR_COLOR = (172, 138, 94)
WATER_COLOR = (48, 132, 210)
SOURCE_DROP_COLOR = (18, 90, 175)
SOURCE_DROP_HIGHLIGHT = (195, 230, 255)
SOURCE_ARROW_COLOR = (235, 240, 250)
FOUNTAIN_BASIN_COLOR = (156, 124, 86)
FOUNTAIN_BASIN_SHADOW = (98, 72, 48)
FOUNTAIN_WATER_COLOR = (72, 158, 228)
FOUNTAIN_SPRAY_COLOR = (180, 225, 250)
SOLUTION_STROKE_COLOR = (24, 174, 236)

AQUEDUCT_VERIFIER_PROFILE = "maze_aqueduct_pipes_visible_pipe_multimodal_hard_gate_v2"
AQUEDUCT_ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "path",
    "grammar": r"^Answer: \[\([1-9][0-9]*, [1-9][0-9]*\)(?:, \([1-9][0-9]*, [1-9][0-9]*\))*\]\.$",
    "prefix": "Answer: ",
    "separator": ", ",
    "suffix": ".",
    "position_convention": "1-based (x, y); x is the row counted from the top, y the column counted from the left",
    "allow_leading_or_trailing_whitespace": False,
    "allow_newlines": False,
    "allow_extra_text": False,
}
AQUEDUCT_VERIFIER_CONTRACT: Dict[str, Any] = {
    "profile": AQUEDUCT_VERIFIER_PROFILE,
    "strict_binary_correctness": True,
    "multimodal_hard_gate": True,
    "text_parser": "strict_grid_path_answer",
    "text_replay": "visible_pipe_kind_unique_rotatable_path_replay",
    "image_verifier": "visible_pipe_openings_cyan_flow_and_preservation",
    "video_verifier": "maze_aqueduct_pipes_full_video_semantic_single_decode",
    "trusted_oracle_fields": [],
    "ignored_oracle_fields": [
        "solution_path_cell_ids",
        "rotation_correct",
        "video_process",
    ],
    "performance_measurements_are_diagnostic_only": True,
    "automatic_failure_thresholds": False,
}

# ---------------------------------------------------------------------------
# Direction system. N=0 (dr=-1), E=1 (dc=+1), S=2 (dr=+1), W=3 (dc=-1).
# ---------------------------------------------------------------------------

N, E, S, W = 0, 1, 2, 3
DIR_OFFSETS: Dict[int, Tuple[int, int]] = {
    N: (-1, 0),
    E: (0, 1),
    S: (1, 0),
    W: (0, -1),
}
DIR_OPPOSITE: Dict[int, int] = {N: S, E: W, S: N, W: E}

# Canonical pipe kinds. Each kind has a base opening-set (rotation 0); rotating
# by 90 degrees clockwise advances every direction by one quarter-turn
# (N -> E -> S -> W -> N). Translation table:
ROTATE_DIR: Dict[Tuple[int, int], int] = {}
for d in (N, E, S, W):
    for steps in range(4):
        rd = d
        for _ in range(steps):
            rd = {N: E, E: S, S: W, W: N}[rd]
        ROTATE_DIR[(d, steps)] = rd

# Canonical kinds and their base (rotation 0) opening sets.
# STRAIGHT base = horizontal (E, W). Rotating 90 -> vertical.
# ELBOW base    = (N, E)     (corner pointing top-right). Rotations cycle.
# T base        = (N, E, W)  (T pointing up).
# CROSS base    = (N, E, S, W).
KIND_BASE_OPENINGS: Dict[str, frozenset] = {
    "STRAIGHT": frozenset({E, W}),
    "ELBOW":    frozenset({N, E}),
    "T":        frozenset({N, E, W}),
    "CROSS":    frozenset({N, E, S, W}),
}
# Canonical rotation step is 90 degrees per unit.
ROTATIONS = (0, 1, 2, 3)  # multiplied by 90 for visualisation
ALL_KINDS: Tuple[str, ...] = tuple(KIND_BASE_OPENINGS.keys())


def kind_openings(kind: str, rot_steps: int) -> frozenset:
    """Return the opening set of *kind* rotated *rot_steps* * 90 degrees CW."""
    base = KIND_BASE_OPENINGS[kind]
    return frozenset(ROTATE_DIR[(d, rot_steps)] for d in base)


def find_rotation_for_required(kind: str, required: set) -> Optional[int]:
    """Return the smallest rotation step whose opening set is a SUPERSET of
    *required*. Returns None when no rotation can satisfy the required pair.
    """
    for rs in ROTATIONS:
        if required.issubset(kind_openings(kind, rs)):
            return rs
    return None




# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class MazeAqueductPipesGenerator(PathTaskGenerator):
    """Generate Roman-aqueduct rotatable-pipe puzzles."""

    DEFAULT_OUTPUT_DIR = "outputs/spatial_navigation/maze_aqueduct_pipes"
    DEFAULT_GRID_SIZE = DEFAULT_GRID
    DEFAULT_CELL_SIZE = DEFAULT_CELL_SIZE
    ANSWER_COORDINATE_SPACE = True

    DEFAULT_TI2V_PROMPT = (
        "Rule: Water enters the west opening of the source pipe and crosses an edge only when both adjacent openings face it. "
        "The unique non-branching source-to-fountain route uses two-port straight or elbow tiles; T and cross tiles are dry decoys. "
        "Required Output: Rotate only misaligned live-route tiles continuously about their own centres, "
        "all together with shared progress and a simultaneous landing. After the dry route is connected, grow one continuous bright-cyan flow from the "
        "source through every mutually connected route tile and activate the fountain only on arrival."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = (
        "Preserve the supplied aqueduct, edge row and column numbers, dry decoys, source, basin, framing, palette, and camera. Rotate "
        "only live-route straight or elbow tiles to mutually connected openings, show one continuous bright-cyan source-to-fountain "
        "flow, and activate the fountain. Do not put water through a closed edge or into a decoy."
    )
    PUZZLE_INPUT_DESCRIPTION = (
        "The input image shows a sandstone-tile grid. Several tiles carry dark-brown pipe fragments (two-port straights, "
        "two-port elbows, three-port T-junctions, or four-port crosses) with cool-blue water channels drawn through them, "
        "and each pipe tile is shown at one of four cardinal rotations. The remaining tiles are blank sandstone with small "
        "moss dots. A blue water-drop marker with a right-pointing arrow sits on the left edge, and an inactive bronze "
        "fountain basin with no spray sits on the right edge. Blue numbers run along the left and top edges of the grid."
    )
    REASONING_INPUT_DESCRIPTION = (
        "The input image shows a rotatable-pipe puzzle: dark-brown pipe tiles with cool-blue channels, each drawn at one of "
        "four cardinal rotations, sit on a sandstone grid beside blank mossy tiles. A blue water drop with a right-pointing "
        "arrow marks the source pipe on the left edge and an inactive bronze basin marks the fountain pipe on the right edge. "
        "Blue numbers run along the left and top edges of the grid. " + GRID_XY_CONVENTION_SENTENCE
    )

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
        video: bool = False,
    ) -> None:
        if grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        if grid_size > 13:
            raise ValueError("grid_size must be at most 13 so every edge row/column number stays readable")

        final_width = int(canvas_width) if canvas_width is not None else 512
        requested_cell_size = cell_size if cell_size is not None else size
        if requested_cell_size is None:
            # Fit the widest even cell size that still leaves the shared axis
            # gutter, then never exceed this task's default tile size.
            probe, _, _, _, _ = grid_axis_label_layout(grid_size, grid_size, canvas_size=final_width)
            requested_cell_size = min(int(probe) & ~1, DEFAULT_CELL_SIZE)
        effective_cell_size = int(requested_cell_size)
        if effective_cell_size % 2 == 1:
            effective_cell_size -= 1
        if effective_cell_size <= 0:
            raise ValueError("cell_size must be positive")
        effective_cell_size, pad_left, pad_top, pad_right, pad_bottom = grid_axis_label_layout(
            grid_size,
            grid_size,
            canvas_size=final_width,
            cell_size=effective_cell_size,
        )
        final_height = final_width
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
        self.grid_size = int(grid_size)
        # The shared record builder inventories the rendered axis numbers from
        # ``rows``/``cols``; this grid is square.
        self.rows = int(grid_size)
        self.cols = int(grid_size)
        self.cell_size = effective_cell_size
        self.padding = (pad_left, pad_top, pad_right, pad_bottom)
        self._last_layout: Optional[Dict[str, Any]] = None
        self._last_video_process: Optional[Dict[str, Any]] = None

    def next_id(self) -> str:
        """Return the worker-stable base id used by central generation.

        Central generation creates one seeded generator per record and adds a
        deterministic ``__NN`` suffix after merging duplicate base ids.  The
        generic path-task UUID would otherwise make identical fixed-seed runs
        differ solely in transport metadata.
        """

        return "maze_aqueduct_pipes-00000"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> PathTaskRecord:
        puzzle_uuid = puzzle_id or self.next_id()
        layout = self._build_layout()
        self._last_layout = layout

        # Internal cell ids stay 1-based over pipe tiles only, in row-major
        # order. They index the layout solver and are converted to the
        # model-facing 1-based (x, y) positions at the answer boundary.
        pipe_cells: List[Tuple[int, int]] = sorted(layout["kinds"].keys())
        self._rebuild_cell_id_index(pipe_cells)

        pad_left, pad_top, pad_right, pad_bottom, canvas_dims = self._compute_padding()

        # Puzzle and reasoning inputs: tiles at DISPLAYED rotations, no flow.
        puzzle_image = self._render_scene(
            layout,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
            rotation_mode="displayed",
            draw_flow=False,
        )
        reasoning_image = self._render_scene(
            layout,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
            rotation_mode="displayed",
            draw_flow=False,
        )
        # Solution: tiles at CORRECT rotations plus the bright-cyan water stroke.
        solution_image = self._render_scene(
            layout,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
            rotation_mode="correct",
            draw_flow=True,
        )

        puzzle_path, solution_path = self.save_images(puzzle_uuid, puzzle_image, solution_image)
        reasoning_path = self.save_reasoning_image(puzzle_uuid, reasoning_image)

        video_path = None
        if self.video:
            video_path = self._save_rotation_video(
                puzzle_uuid,
                layout,
                pad_left,
                pad_top,
                canvas_dims,
            )

        # Build serializable pipe_specs. Grid rows and columns stay internal
        # 0-based indices; the 1-based (x, y) conversion happens only at the
        # model-facing prompt/answer boundary.
        pipe_specs_payload: List[Dict[str, Any]] = []
        for cell in pipe_cells:
            r, c = cell
            kind = layout["kinds"][cell]
            rot_displayed = layout["rot_displayed"][cell]
            rot_correct = layout["rot_correct"][cell]
            pipe_specs_payload.append(
                {
                    "grid_row": r,
                    "grid_col": c,
                    "kind": kind,
                    "rotation_displayed": rot_displayed * 90,
                    "rotation_correct": rot_correct * 90,
                    "is_misaligned": rot_displayed != rot_correct,
                    "is_decoy": cell not in layout["flow_set"],
                }
            )

        source_cell = layout["source_cell"]
        sink_cell = layout["sink_cell"]
        source_payload = {
            "grid_row": source_cell[0],
            "grid_col": source_cell[1],
            "entry_direction": "E",  # water enters the grid moving east
        }
        sink_payload = {
            "grid_row": sink_cell[0],
            "grid_col": sink_cell[1],
        }

        traversed_cell_ids = [self._get_cell_id(*cell) for cell in layout["flow_cells"]]

        start_point = self._source_marker_center(source_cell, pad_left, pad_top)
        goal_point = self._sink_marker_center(sink_cell, pad_left, pad_top)

        extra_payload: Dict[str, Any] = {
            "grid_size": self.grid_size,
            "cell_size": self.cell_size,
            "canvas_dimensions": [int(canvas_dims[0]), int(canvas_dims[1])],
            "padding": [pad_left, pad_top, pad_right, pad_bottom],
            "pipe_specs": pipe_specs_payload,
            "source_cell": source_payload,
            "sink_cell": sink_payload,
            "solution_path_cell_ids": traversed_cell_ids,
            "path_joint_style": "square",
            "bbox_fraction": layout.get("bbox_fraction"),
            "misaligned_path_count": layout.get("misaligned_path_count"),
            "path_length_pipes": len(layout["flow_cells"]),
            "displayed_flow_reaches_sink": self._flow_reaches_sink_with(layout, layout["rot_displayed"]),
            "corrected_flow_reaches_sink": self._flow_reaches_sink_with(layout, layout["rot_correct"]),
            "video_process": self._last_video_process,
            "answer_schema": dict(AQUEDUCT_ANSWER_SCHEMA),
            "render_geometry": {
                "canvas_dimensions": [int(canvas_dims[0]), int(canvas_dims[1])],
                "grid_size": self.grid_size,
                "cell_size": self.cell_size,
                "padding": [pad_left, pad_top, pad_right, pad_bottom],
                "source_edge": "left",
                "sink_edge": "right",
                "axis_labels": "one_based_row_numbers_in_left_gutter_and_column_numbers_in_top_gutter",
                "grid_position_convention": "one_based_x_row_from_top_y_column_from_left",
                "live_path_pipe_kinds": ["STRAIGHT", "ELBOW"],
                "decoy_only_pipe_kinds": ["T", "CROSS"],
            },
            "verifier_contract": dict(AQUEDUCT_VERIFIER_CONTRACT),
        }

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
    # Layout generation
    # ------------------------------------------------------------------

    def _build_layout(self) -> Dict[str, Any]:
        best_rejection: Optional[str] = None
        for _ in range(MAX_LAYOUT_ATTEMPTS):
            candidate = self._try_random_layout()
            if candidate is None:
                best_rejection = "construction"
                continue
            if not self._passes_canvas_spread(candidate):
                best_rejection = "spread"
                continue
            supported_paths = self._enumerate_rotatable_source_sink_paths(candidate, limit=2)
            if len(supported_paths) != 1 or supported_paths[0] != candidate["flow_cells"]:
                best_rejection = "non_unique_rotatable_path"
                continue
            # Validate that the CORRECT rotations actually carry flow to sink.
            if not self._flow_reaches_sink_with(candidate, candidate["rot_correct"]):
                best_rejection = "correct_does_not_flow"
                continue
            # Tier A: without rotation, no path exists. Simulate the puzzle's
            # CURRENT (displayed) rotations and require that flow does NOT
            # reach the fountain. If it accidentally does, the rotation
            # mechanic is decorative -- regenerate.
            if self._flow_reaches_sink_with(candidate, candidate["rot_displayed"]):
                best_rejection = "displayed_already_flows"
                continue
            return candidate
        raise RuntimeError(
            f"Failed to build an aqueduct layout after {MAX_LAYOUT_ATTEMPTS} attempts "
            f"(last rejection: {best_rejection or 'construction'})"
        )

    def _try_random_layout(self) -> Optional[Dict[str, Any]]:
        rng = self.rng
        grid = self.grid_size

        source_row = rng.randint(1, grid - 2)
        sink_row = rng.randint(1, grid - 2)
        source_cell = (source_row, 0)
        sink_cell = (sink_row, grid - 1)
        if source_cell == sink_cell:
            return None

        path = self._biased_walk(source_cell, sink_cell)
        if path is None:
            return None
        if len(path) < MIN_PATH_LENGTH or len(path) > MAX_PATH_LENGTH:
            return None

        # For each path cell, determine the (in_dir, out_dir) opening pair.
        # Then pick a kind whose opening-set, AT SOME ROTATION, can host the
        # pair. We deliberately mix kinds so the puzzle has 2-port straights,
        # 2-port elbows, 3-port T-junctions, and 4-port crosses.
        kinds: Dict[Tuple[int, int], str] = {}
        rot_correct: Dict[Tuple[int, int], int] = {}
        for idx, cell in enumerate(path):
            prev_cell = path[idx - 1] if idx > 0 else None
            next_cell = path[idx + 1] if idx + 1 < len(path) else None
            in_dir = self._dir_from_to(cell, prev_cell) if prev_cell is not None else self._edge_inward(cell)
            out_dir = self._dir_from_to(cell, next_cell) if next_cell is not None else self._edge_outward(cell)
            if in_dir is None or out_dir is None:
                return None
            required = {in_dir, out_dir}
            # The live route uses only two-port tiles. T/CROSS tiles remain
            # visually useful decoys, but allowing them on the live route
            # would create unrendered side branches and would make the exact
            # corrected orientation underdetermined for straight-through T
            # tiles. Restricting the live route to STRAIGHT/ELBOW keeps the
            # water process physically closed and independently replayable.
            collinear = (in_dir, out_dir) in {(E, W), (W, E), (N, S), (S, N)}
            if collinear:
                eligible = ["STRAIGHT"]
            else:
                eligible = ["ELBOW"]
            weights = {"STRAIGHT": 1.0, "ELBOW": 1.0}
            choices: List[str] = []
            cum_weights: List[float] = []
            acc = 0.0
            for k in eligible:
                w = weights.get(k, 1.0)
                if w <= 0:
                    continue
                acc += w
                choices.append(k)
                cum_weights.append(acc)
            if not choices:
                return None
            pick = rng.random() * acc
            chosen_kind = choices[-1]
            for k, threshold in zip(choices, cum_weights):
                if pick <= threshold:
                    chosen_kind = k
                    break
            rot = find_rotation_for_required(chosen_kind, required)
            if rot is None:
                return None
            kinds[cell] = chosen_kind
            rot_correct[cell] = rot

        # DISPLAYED rotation per path cell. Force at least
        # MIN_MISALIGNED_FRACTION of the path tiles to be misaligned so the
        # rotation mechanic visibly drives the puzzle.
        rot_displayed: Dict[Tuple[int, int], int] = {}
        path_cells = list(path)
        n_path = len(path_cells)
        target_misaligned = max(1, int(math.ceil(n_path * MIN_MISALIGNED_FRACTION)))
        # First assignment: uniform random among all 4 rotations.
        for cell in path_cells:
            rot_displayed[cell] = rng.randint(0, 3)
        # Count VISIBLE misalignments. A horizontal STRAIGHT at 0° and 180°
        # has the same opening set and must not count as a load-bearing change.
        misaligned = [
            c for c in path_cells
            if kind_openings(kinds[c], rot_displayed[c]) != kind_openings(kinds[c], rot_correct[c])
        ]
        if len(misaligned) < target_misaligned:
            # Promote visibly matched cells to a rotation with different
            # openings, not merely a different numeric rotation value.
            matched = [
                c for c in path_cells
                if kind_openings(kinds[c], rot_displayed[c]) == kind_openings(kinds[c], rot_correct[c])
            ]
            rng.shuffle(matched)
            need = target_misaligned - len(misaligned)
            for cell in matched[:need]:
                wrong = [
                    r for r in ROTATIONS
                    if kind_openings(kinds[cell], r) != kind_openings(kinds[cell], rot_correct[cell])
                ]
                rot_displayed[cell] = rng.choice(wrong)
        misaligned_count = sum(
            1
            for cell in path_cells
            if kind_openings(kinds[cell], rot_displayed[cell]) != kind_openings(kinds[cell], rot_correct[cell])
        )

        # Decoy tiles on off-path cells with random kinds + random rotations.
        # Use the existing "decoy must not fuse with the live flow" rule, but
        # evaluated against the CORRECT rotations of path cells (since the
        # answer is read at the CORRECT state). Decoys must also not fuse into
        # neighbouring decoys at rotations that accidentally extend the flow
        # off the path. We keep this cheap and rely on the Tier A check on
        # displayed rotations to catch any accidental shortcut.
        empty_cells = [
            (r, c)
            for r in range(grid)
            for c in range(grid)
            if (r, c) not in kinds
        ]
        # Sort empty cells so bottom-half cells come first — this biases
        # decoy placement toward the lower portion of the canvas, balancing
        # the often top-heavy live flow path.
        half = grid // 2
        rng.shuffle(empty_cells)
        empty_cells.sort(key=lambda rc: 0 if rc[0] >= half else 1)
        decoy_target = rng.randint(3, 5)
        decoys_added = 0
        flow_set = set(path_cells)
        for cell in empty_cells:
            if decoys_added >= decoy_target:
                break
            kind = rng.choice(ALL_KINDS)
            d_rot = rng.randint(0, 3)
            if self._decoy_would_fuse_with_flow(cell, kind, d_rot, kinds, rot_correct, flow_set):
                # Try a different rotation of the same kind before giving up
                # on this cell.
                fallback_found = False
                for rs in ROTATIONS:
                    if not self._decoy_would_fuse_with_flow(cell, kind, rs, kinds, rot_correct, flow_set):
                        d_rot = rs
                        fallback_found = True
                        break
                if not fallback_found:
                    continue
            kinds[cell] = kind
            rot_correct[cell] = d_rot
            rot_displayed[cell] = d_rot  # decoys are not "wrong" per se
            decoys_added += 1

        return {
            "kinds": kinds,
            "rot_displayed": rot_displayed,
            "rot_correct": rot_correct,
            "source_cell": source_cell,
            "sink_cell": sink_cell,
            "flow_cells": list(path_cells),
            "flow_set": flow_set,
            "misaligned_path_count": misaligned_count,
        }

    def _biased_walk(
        self,
        source_cell: Tuple[int, int],
        sink_cell: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        rng = self.rng
        grid = self.grid_size
        visited = {source_cell}
        path: List[Tuple[int, int]] = [source_cell]
        cur = source_cell
        max_steps = grid * grid * 3

        detour_target_row = rng.choice([1, grid - 2])

        for _ in range(max_steps):
            if cur == sink_cell:
                return path
            r, c = cur
            candidates: List[Tuple[Tuple[int, int], float]] = []
            for d in (N, E, S, W):
                dr, dc = DIR_OFFSETS[d]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid and 0 <= nc < grid):
                    continue
                if (nr, nc) in visited:
                    continue
                # Keep the intended route induced: a newly appended cell may
                # touch only the current path endpoint, never an earlier path
                # cell. Otherwise free rotation can create a shortcut around
                # the saved walk even when every tile is only two-port.
                adjacent_visited = {
                    (nr + adr, nc + adc)
                    for adr, adc in DIR_OFFSETS.values()
                    if (nr + adr, nc + adc) in visited
                }
                if adjacent_visited - {cur}:
                    continue
                if nc == 0 and (nr, nc) != source_cell:
                    continue
                if nc == grid - 1 and (nr, nc) != sink_cell:
                    continue
                touched_detour = any(pr == detour_target_row for pr, _ in path)
                if not touched_detour:
                    row_pull = 1.0 + 3.0 / (1.0 + abs(nr - detour_target_row))
                    forward_pull = 1.0 + 0.5 / (1.0 + abs(nc - sink_cell[1]))
                    weight = row_pull * forward_pull
                else:
                    dist = abs(nr - sink_cell[0]) + abs(nc - sink_cell[1])
                    weight = 1.0 + 2.5 / (1.0 + dist)
                candidates.append(((nr, nc), weight))
            if not candidates:
                return None
            total = sum(w for _, w in candidates)
            pick = rng.random() * total
            acc = 0.0
            chosen = candidates[-1][0]
            for cell, w in candidates:
                acc += w
                if pick <= acc:
                    chosen = cell
                    break
            path.append(chosen)
            visited.add(chosen)
            cur = chosen
        return None

    @staticmethod
    def _dir_from_to(src: Tuple[int, int], dst: Tuple[int, int]) -> Optional[int]:
        dr = dst[0] - src[0]
        dc = dst[1] - src[1]
        for d, (odr, odc) in DIR_OFFSETS.items():
            if odr == dr and odc == dc:
                return d
        return None

    def _edge_inward(self, cell: Tuple[int, int]) -> Optional[int]:
        """Direction of the off-grid neighbour for a source on the LEFT edge.
        Water enters east -> the source tile's W opening receives flow.
        """
        grid = self.grid_size
        r, c = cell
        if c == 0:
            return W
        if c == grid - 1:
            return E
        if r == 0:
            return N
        if r == grid - 1:
            return S
        return None

    def _edge_outward(self, cell: Tuple[int, int]) -> Optional[int]:
        """Direction pointing OFF the grid for a sink on the RIGHT edge."""
        grid = self.grid_size
        r, c = cell
        if c == 0:
            return W
        if c == grid - 1:
            return E
        if r == 0:
            return N
        if r == grid - 1:
            return S
        return None

    def _decoy_would_fuse_with_flow(
        self,
        cell: Tuple[int, int],
        kind: str,
        rot: int,
        kinds: Dict[Tuple[int, int], str],
        rotations: Dict[Tuple[int, int], int],
        flow_set: set,
    ) -> bool:
        """Reject a decoy when, at *rot*, it would either:
          * point an opening off the grid (visible stub at the frame edge), or
          * meet a path-cell opening that, at the path's CORRECT rotation,
            faces the decoy.
        Either case threatens to fuse the decoy INTO the live flow.
        """
        r, c = cell
        grid = self.grid_size
        opens = kind_openings(kind, rot)
        for d in opens:
            dr, dc = DIR_OFFSETS[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid and 0 <= nc < grid):
                return True
            neighbour = (nr, nc)
            if neighbour in kinds and neighbour in flow_set:
                n_kind = kinds[neighbour]
                n_rot = rotations[neighbour]
                if DIR_OPPOSITE[d] in kind_openings(n_kind, n_rot):
                    return True
        return False

    def _enumerate_rotatable_source_sink_paths(
        self,
        layout: Dict[str, Any],
        *,
        limit: int = 2,
    ) -> List[List[Tuple[int, int]]]:
        """Enumerate simple paths supportable by the visible pipe kinds.

        This deliberately ignores the saved corrected rotations. It asks only
        whether each visible pipe kind can be rotated to expose the required
        incoming/outgoing pair. Generation accepts a layout only when this
        independent topology replay finds exactly the intended path.
        """

        kinds: Dict[Tuple[int, int], str] = layout["kinds"]
        source_cell: Tuple[int, int] = layout["source_cell"]
        sink_cell: Tuple[int, int] = layout["sink_cell"]
        found: List[List[Tuple[int, int]]] = []

        def supports(cell: Tuple[int, int], required: set[int]) -> bool:
            kind = kinds.get(cell)
            return bool(
                kind
                and any(required.issubset(kind_openings(kind, rotation)) for rotation in ROTATIONS)
            )

        def direction(src: Tuple[int, int], dst: Tuple[int, int]) -> Optional[int]:
            return self._dir_from_to(src, dst)

        def visit(
            cell: Tuple[int, int],
            previous: Optional[Tuple[int, int]],
            path: List[Tuple[int, int]],
        ) -> None:
            if len(found) >= max(1, int(limit)):
                return
            incoming = W if previous is None else direction(cell, previous)
            if incoming is None:
                return
            if cell == sink_cell:
                if supports(cell, {incoming, E}):
                    found.append(list(path))
                return
            for outgoing, (dr, dc) in DIR_OFFSETS.items():
                neighbour = (cell[0] + dr, cell[1] + dc)
                if neighbour not in kinds or neighbour in path:
                    continue
                if not supports(cell, {incoming, outgoing}):
                    continue
                visit(neighbour, cell, [*path, neighbour])

        if source_cell in kinds and sink_cell in kinds:
            visit(source_cell, None, [source_cell])
        return found

    # ------------------------------------------------------------------
    # Validity checks
    # ------------------------------------------------------------------

    def _passes_canvas_spread(self, layout: Dict[str, Any]) -> bool:
        all_cells: List[Tuple[int, int]] = [layout["source_cell"], layout["sink_cell"]]
        all_cells.extend(layout["kinds"].keys())
        rows = [r for r, _ in all_cells]
        cols = [c for _, c in all_cells]
        bbox_h = max(rows) - min(rows) + 1
        bbox_w = max(cols) - min(cols) + 1
        frac_h = bbox_h / self.grid_size
        frac_w = bbox_w / self.grid_size
        layout["bbox_fraction"] = (frac_w, frac_h)
        if frac_h < MIN_BBOX_FRACTION:
            return False
        if frac_w < MIN_BBOX_FRACTION:
            return False

        flow_cells: List[Tuple[int, int]] = list(layout["flow_cells"])
        if not flow_cells:
            return False
        frows = [r for r, _ in flow_cells]
        fcols = [c for _, c in flow_cells]
        flow_h = max(frows) - min(frows) + 1
        flow_w = max(fcols) - min(fcols) + 1
        if flow_h / self.grid_size < MIN_FLOW_BBOX_FRACTION:
            return False
        if flow_w / self.grid_size < MIN_FLOW_BBOX_FRACTION:
            return False
        return True

    def _simulate_flow(
        self,
        kinds: Dict[Tuple[int, int], str],
        rotations: Dict[Tuple[int, int], int],
        source_cell: Tuple[int, int],
        entry_dir: int,
        *,
        sink_cell: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Simulate water flow for a per-tile ``(kind, rotation)`` state.

        The result contains reached cells, the off-grid exit cell if any, and
        an ordered path when every step has a unique successor.
        """
        reached: set = set()
        if source_cell not in kinds:
            return {"reached": reached, "exit_cell": None, "path_ordered": []}
        src_opens = kind_openings(kinds[source_cell], rotations[source_cell])
        if entry_dir not in src_opens:
            return {"reached": reached, "exit_cell": None, "path_ordered": []}

        path_ordered: List[Tuple[int, int]] = [source_cell]
        reached.add(source_cell)
        cur_cell = source_cell
        arrived_via = entry_dir
        unique_walk = True
        while True:
            kind = kinds.get(cur_cell)
            if kind is None:
                break
            rot = rotations[cur_cell]
            opens = kind_openings(kind, rot)
            outs = [d for d in opens if d != arrived_via]
            if not outs:
                break
            viable_outs: List[int] = []
            grid = self.grid_size
            for d in outs:
                dr, dc = DIR_OFFSETS[d]
                nr, nc = cur_cell[0] + dr, cur_cell[1] + dc
                neighbour_cell = (nr, nc)
                if neighbour_cell in reached:
                    continue
                # In-grid cell with no pipe: water spills onto sandstone --
                # NOT a viable continuation. Off-grid steps are dead ends
                # EXCEPT the canonical fountain step (directly east of the
                # sink cell), which counts as reaching the goal.
                in_grid = 0 <= nr < grid and 0 <= nc < grid
                n_kind = kinds.get(neighbour_cell)
                if n_kind is None:
                    if in_grid:
                        # Dead end on this opening; water spills onto sandstone.
                        continue
                    if sink_cell is not None:
                        if neighbour_cell == (sink_cell[0], sink_cell[1] + 1):
                            viable_outs.append(d)
                            continue
                        # Off-grid in any direction other than the fountain
                        # exit is a dead end (water drips off the aqueduct).
                        continue
                    # Without a designated sink, any off-grid step is an exit.
                    viable_outs.append(d)
                    continue
                if DIR_OPPOSITE[d] in kind_openings(n_kind, rotations[neighbour_cell]):
                    viable_outs.append(d)
            if not viable_outs:
                break
            if len(viable_outs) > 1:
                unique_walk = False
                break
            d = viable_outs[0]
            dr, dc = DIR_OFFSETS[d]
            nr, nc = cur_cell[0] + dr, cur_cell[1] + dc
            neighbour_cell = (nr, nc)
            if neighbour_cell not in kinds:
                path_ordered.append(neighbour_cell)
                return {
                    "reached": reached,
                    "exit_cell": neighbour_cell,
                    "path_ordered": path_ordered if unique_walk else [],
                }
            reached.add(neighbour_cell)
            path_ordered.append(neighbour_cell)
            arrived_via = DIR_OPPOSITE[d]
            cur_cell = neighbour_cell
        return {
            "reached": reached,
            "exit_cell": None,
            "path_ordered": path_ordered if unique_walk else [],
        }

    def _flow_reaches_sink_with(
        self,
        layout: Dict[str, Any],
        rotations: Dict[Tuple[int, int], int],
    ) -> bool:
        result = self._simulate_flow(
            layout["kinds"],
            rotations,
            layout["source_cell"],
            entry_dir=W,
            sink_cell=layout["sink_cell"],
        )
        exit_cell = result["exit_cell"]
        if exit_cell is None:
            return False
        sink_cell = layout["sink_cell"]
        return exit_cell == (sink_cell[0], sink_cell[1] + 1)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_scene(
        self,
        layout: Dict[str, Any],
        *,
        padding: Tuple[int, int],
        canvas_dims: Tuple[int, int],
        rotation_mode: str,
        draw_flow: bool,
    ) -> Image.Image:
        """Render the full scene at *rotation_mode* in {"displayed", "correct"}.

        For the puzzle/reasoning renders use rotation_mode="displayed" so the
        viewer sees the misaligned tiles. For the solution render use
        rotation_mode="correct" so every tile lands on its load-bearing
        orientation, then overlay the bright-cyan water stroke.
        """
        pad_left, pad_top = padding
        scene, draw = self._begin_scene(canvas_dims)
        self._paint_board(scene, draw, layout, padding, rotation_mode=rotation_mode)

        if draw_flow:
            self._paint_flow(
                scene,
                draw,
                layout,
                padding,
                self._flow_polyline_points(layout, pad_left, pad_top),
                active_sink=True,
            )

        # Row/column numbers live in the gutter, never inside the grid body,
        # so no tile carries a glyph that could rotate with its pipe.
        self._draw_axis_labels(draw, pad_left, pad_top)

        return self._finish_scene(scene, canvas_dims)

    # ---------- supersampled scene plumbing ----------

    def _begin_scene(
        self,
        canvas_dims: Tuple[int, int],
    ) -> Tuple[Image.Image, ScaledDraw]:
        """Allocate the supersampled canvas every aqueduct artifact is drawn on.

        The board is built from pipe elbows (annular pieslice sectors), round
        basins, water droplets, and an arrow head — curved and diagonal shapes
        that alias badly at the target resolution.  Every render therefore draws
        through a ``ScaledDraw`` proxy in target coordinates and takes exactly
        one LANCZOS pass in :meth:`_finish_scene`.
        """

        return new_scaled_canvas(
            int(canvas_dims[0]),
            int(canvas_dims[1]),
            SCENE_SUPERSAMPLE,
            BACKGROUND_COLOR,
        )

    def _scene_draw(self, scene: Image.Image) -> ScaledDraw:
        """Re-attach a scaled draw proxy to an existing supersampled scene."""

        return ScaledDraw(ImageDraw.Draw(scene), SCENE_SUPERSAMPLE)

    @staticmethod
    def _finish_scene(scene: Image.Image, canvas_dims: Tuple[int, int]) -> Image.Image:
        return finish_scaled_canvas(scene, (int(canvas_dims[0]), int(canvas_dims[1])))

    def _paint_board(
        self,
        scene: Image.Image,
        draw: ScaledDraw,
        layout: Dict[str, Any],
        padding: Tuple[int, int],
        *,
        rotation_mode: Optional[str] = None,
        rot_table: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> None:
        """Paint mortar, every tile, and the dry endpoints of one board state."""

        pad_left, pad_top = padding
        grid = self.grid_size
        cs = self.cell_size
        draw.rectangle(
            (pad_left, pad_top, pad_left + grid * cs - 1, pad_top + grid * cs - 1),
            fill=STONE_MORTAR_COLOR,
        )
        kinds = layout["kinds"]
        if rot_table is None:
            rot_table = (
                layout["rot_correct"] if rotation_mode == "correct" else layout["rot_displayed"]
            )
        for r in range(grid):
            for c in range(grid):
                left = pad_left + c * cs
                top = pad_top + r * cs
                if (r, c) in kinds:
                    self._draw_pipe_tile(draw, left, top, kinds[(r, c)], rot_table[(r, c)])
                else:
                    self._draw_blank_tile(draw, left, top, r, c)
        self._draw_source_marker(draw, layout["source_cell"], pad_left, pad_top)
        self._draw_sink_marker(draw, layout["sink_cell"], pad_left, pad_top, active=False)

    def _paint_flow(
        self,
        scene: Image.Image,
        draw: ScaledDraw,
        layout: Dict[str, Any],
        padding: Tuple[int, int],
        flow_points: Sequence[Tuple[float, float]],
        *,
        active_sink: bool,
    ) -> None:
        """Lay the water stroke on a supersampled scene and repaint endpoints.

        The stroke runs cell centre to cell centre inside the pipe grid, so it
        is axis-aligned; drawing it at scene scale keeps its square corners
        exact while the source drop and fountain above it stay antialiased.
        """

        pad_left, pad_top = padding
        if len(flow_points) >= 2:
            draw_path_line(
                scene,
                list(flow_points),
                SOLUTION_STROKE_COLOR,
                self._flow_thickness(),
                joint_style="square",
                tip_marker=False,
                factor=SCENE_SUPERSAMPLE,
            )
        self._draw_source_marker(draw, layout["source_cell"], pad_left, pad_top)
        self._draw_sink_marker(
            draw, layout["sink_cell"], pad_left, pad_top, active=active_sink
        )

    # ---------- tile rendering ----------

    def _draw_blank_tile(
        self,
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        r: int,
        c: int,
    ) -> None:
        cs = self.cell_size
        self._paint_stone_base(draw, left, top, BLANK_TILE_COLOR)
        rng_seed = (r * 131 + c * 7) & 0xFFFF
        import random as _random
        sub_rng = _random.Random(rng_seed)
        for _ in range(3):
            ox = sub_rng.randint(int(cs * 0.15), int(cs * 0.85))
            oy = sub_rng.randint(int(cs * 0.15), int(cs * 0.85))
            radius = sub_rng.choice([1, 2])
            draw.ellipse(
                (left + ox - radius, top + oy - radius, left + ox + radius, top + oy + radius),
                fill=MOSS_COLOR,
            )

    def _paint_stone_base(
        self,
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        body_color: Tuple[int, int, int],
        *,
        with_cell_edges: bool = True,
    ) -> None:
        """Paint a single sandstone tile with optional 2-px edge strips.

        Blank tiles call this with `with_cell_edges=True` for the masonry
        feel. Pipe tiles call it with `with_cell_edges=False` so the strips
        do NOT sit one pixel above the pipe's outline (where they would
        produce a light/dark seam visible right against the brown border —
        that seam reads as a tiny cell-boundary protrusion).
        """
        cs = self.cell_size
        draw.rectangle((left, top, left + cs - 1, top + cs - 1), fill=body_color)
        if not with_cell_edges:
            return
        draw.rectangle((left, top, left + cs - 1, top + 1), fill=STONE_HIGHLIGHT_COLOR)
        draw.rectangle((left, top, left + 1, top + cs - 1), fill=STONE_HIGHLIGHT_COLOR)
        draw.rectangle((left, top + cs - 2, left + cs - 1, top + cs - 1), fill=STONE_EDGE_COLOR)
        draw.rectangle((left + cs - 2, top, left + cs - 1, top + cs - 1), fill=STONE_EDGE_COLOR)

    def _draw_pipe_tile(
        self,
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        kind: str,
        rot: int,
    ) -> None:
        cs = self.cell_size
        # Pipe tiles skip the per-cell highlight/shadow edge strips: at the
        # pipe's outer border line those strips would sit ONE pixel above
        # the brown outline, producing a visible cell-separator seam right
        # against the pipe — the eye reads that as a "border protrusion".
        self._paint_stone_base(draw, left, top, STONE_BASE_COLOR, with_cell_edges=False)
        half = cs // 2
        cx = left + half
        cy = top + half
        outer_w = max(6, cs * 2 // 3)
        outer_w -= outer_w % 2
        inner_w = max(4, outer_w * 3 // 5)
        inner_w -= inner_w % 2
        water_w = max(2, outer_w // 3)
        water_w -= water_w % 2

        opens = kind_openings(kind, rot)
        is_l_elbow = (
            len(opens) == 2
            and opens not in (frozenset({E, W}), frozenset({N, S}))
        )

        if is_l_elbow:
            # Render the L as a smooth pipe elbow: shortened straight legs
            # plus a 90° annular-sector bend with concentric outline /
            # interior / water bands. Geometry: tightest possible bend with
            # `R_bend = outer_w / 2`, so the bend's INNER pipe edge passes
            # through the bend centre (the inner annulus radius is 0). The
            # bend centre sits diagonally OUTSIDE the L's openings — e.g.
            # for a W+S elbow the bend centre is `(cx - R, cy + R)` (SW
            # direction from cell centre), which is the OPPOSITE direction
            # of the L's outer corner. Both the brown borders AND the water
            # follow the same arc radius, so what reads at zoom is a real
            # pipe elbow — corner curve and water curve share centre.
            R_bend = outer_w // 2
            half_o = outer_w // 2
            half_i = inner_w // 2
            half_w = water_w // 2
            R_outer = R_bend + half_o
            R_outer_int = R_bend + half_i
            R_outer_water = R_bend + half_w
            R_inner_water = R_bend - half_w
            R_inner_int = R_bend - half_i
            R_inner_outline = max(0, R_bend - half_o)  # 0 for tightest bend

            # Bend centre + sweep angles (PIL: 0=E, 90=S, 180=W, 270=N, CW).
            if W in opens and S in opens:
                bend_cx, bend_cy = cx - R_bend, cy + R_bend  # SW of cell
                a_start, a_end = 270, 360                    # N -> E sweep
                # Legs: W goes left from cell-left to x=cx-R_bend, S goes
                # down from y=cy+R_bend to cell-bottom.
                w_leg_active = (left, cx - R_bend)
                s_leg_active = (cy + R_bend, top + cs)
                e_leg_active = None
                n_leg_active = None
            elif E in opens and S in opens:
                bend_cx, bend_cy = cx + R_bend, cy + R_bend  # SE of cell
                a_start, a_end = 180, 270                    # W -> N sweep
                e_leg_active = (cx + R_bend, left + cs)
                s_leg_active = (cy + R_bend, top + cs)
                w_leg_active = None
                n_leg_active = None
            elif W in opens and N in opens:
                bend_cx, bend_cy = cx - R_bend, cy - R_bend  # NW of cell
                a_start, a_end = 0, 90                       # E -> S sweep
                w_leg_active = (left, cx - R_bend)
                n_leg_active = (top, cy - R_bend)
                e_leg_active = None
                s_leg_active = None
            else:  # E in opens and N in opens
                bend_cx, bend_cy = cx + R_bend, cy - R_bend  # NE of cell
                a_start, a_end = 90, 180                     # S -> W sweep
                e_leg_active = (cx + R_bend, left + cs)
                n_leg_active = (top, cy - R_bend)
                w_leg_active = None
                s_leg_active = None

            # Straight-leg rectangles, ending at the bend tangent point so
            # they don't overlap with the curved bend region.
            def paint_horiz(x_range, color, half_thickness):
                x0, x1 = x_range
                if x1 > x0:
                    draw.rectangle(
                        (x0, cy - half_thickness, x1 - 1, cy + half_thickness - 1),
                        fill=color,
                    )

            def paint_vert(y_range, color, half_thickness):
                y0, y1 = y_range
                if y1 > y0:
                    draw.rectangle(
                        (cx - half_thickness, y0, cx + half_thickness - 1, y1 - 1),
                        fill=color,
                    )

            for color, hth in (
                (PIPE_OUTLINE_COLOR, half_o),
                (PIPE_INTERIOR_COLOR, half_i),
                (WATER_COLOR, half_w),
            ):
                if w_leg_active:
                    paint_horiz(w_leg_active, color, hth)
                if e_leg_active:
                    paint_horiz(e_leg_active, color, hth)
                if n_leg_active:
                    paint_vert(n_leg_active, color, hth)
                if s_leg_active:
                    paint_vert(s_leg_active, color, hth)

            # Bend annular sector: paint LARGEST outer radius first with
            # outline colour, then progressively smaller pieslices with the
            # next layer colour. Each smaller pieslice overpaints the inner
            # part of the previous one, leaving concentric annular bands.
            for radius, color in (
                (R_outer, PIPE_OUTLINE_COLOR),
                (R_outer_int, PIPE_INTERIOR_COLOR),
                (R_outer_water, WATER_COLOR),
                (R_inner_water, PIPE_INTERIOR_COLOR),
                (R_inner_int, PIPE_OUTLINE_COLOR),
            ):
                if radius <= 0:
                    continue
                draw.pieslice(
                    (bend_cx - radius, bend_cy - radius,
                     bend_cx + radius - 1, bend_cy + radius - 1),
                    a_start, a_end, fill=color,
                )
            # If R_inner_outline > 0 (non-tightest bend), the very innermost
            # pieslice should be stone — paint it with stone colour to
            # carve out the empty area inside the inner outline disc. For
            # the tightest bend (R_inner_outline = 0), no carve needed.
            if R_inner_outline > 0:
                draw.pieslice(
                    (bend_cx - R_inner_outline, bend_cy - R_inner_outline,
                     bend_cx + R_inner_outline - 1, bend_cy + R_inner_outline - 1),
                    a_start, a_end, fill=STONE_BASE_COLOR,
                )
        else:
            # STRAIGHT / T / CROSS: rectangular leg painting (3-pass) covers
            # the joint area completely with no gaps. Paint ALL outlines
            # first, then ALL interiors, then ALL water — water always
            # wins at any inner-corner overlap.
            for d in opens:
                self._draw_pipe_leg_layer(draw, left, top, d, outer_w, PIPE_OUTLINE_COLOR)
            for d in opens:
                self._draw_pipe_leg_layer(draw, left, top, d, inner_w, PIPE_INTERIOR_COLOR)
            for d in opens:
                self._draw_pipe_leg_layer(draw, left, top, d, water_w, WATER_COLOR)

    def _draw_pipe_leg_layer(
        self,
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        direction: int,
        thickness: int,
        color: Tuple[int, int, int],
    ) -> None:
        """Paint ONE concentric strip (outline / interior / water) of a single
        leg from the cell centre out to the cell edge in the given direction.
        """
        cs = self.cell_size
        cx = left + cs // 2
        cy = top + cs // 2
        half = thickness // 2
        if direction == E:
            draw.rectangle(
                (cx, cy - half, left + cs - 1, cy + half - 1),
                fill=color,
            )
        elif direction == W:
            draw.rectangle(
                (left, cy - half, cx, cy + half - 1),
                fill=color,
            )
        elif direction == S:
            draw.rectangle(
                (cx - half, cy, cx + half - 1, top + cs - 1),
                fill=color,
            )
        elif direction == N:
            draw.rectangle(
                (cx - half, top, cx + half - 1, cy),
                fill=color,
            )

    # ---------- source / sink ----------

    def _source_marker_center(
        self,
        source_cell: Tuple[int, int],
        pad_left: int,
        pad_top: int,
    ) -> Tuple[float, float]:
        r, c = source_cell
        cs = self.cell_size
        return (pad_left + c * cs + cs / 2.0, pad_top + r * cs + cs / 2.0)

    def _sink_marker_center(
        self,
        sink_cell: Tuple[int, int],
        pad_left: int,
        pad_top: int,
    ) -> Tuple[float, float]:
        r, c = sink_cell
        cs = self.cell_size
        return (pad_left + c * cs + cs / 2.0, pad_top + r * cs + cs / 2.0)

    def _draw_source_marker(
        self,
        draw: ImageDraw.ImageDraw,
        source_cell: Tuple[int, int],
        pad_left: int,
        pad_top: int,
    ) -> None:
        r, c = source_cell
        cs = self.cell_size
        left = pad_left + c * cs
        top = pad_top + r * cs

        drop_cx = left + cs * 0.32
        drop_cy = top + cs * 0.50
        drop_r = cs * 0.22
        draw.ellipse(
            (
                drop_cx - drop_r,
                drop_cy - drop_r * 0.80,
                drop_cx + drop_r,
                drop_cy + drop_r,
            ),
            fill=SOURCE_DROP_COLOR,
        )
        tip_cy = drop_cy - drop_r * 1.35
        draw.polygon(
            [
                (drop_cx - drop_r * 0.60, drop_cy - drop_r * 0.50),
                (drop_cx, tip_cy),
                (drop_cx + drop_r * 0.60, drop_cy - drop_r * 0.50),
            ],
            fill=SOURCE_DROP_COLOR,
        )
        hi_r = drop_r * 0.30
        draw.ellipse(
            (
                drop_cx - drop_r * 0.25 - hi_r,
                drop_cy - drop_r * 0.10 - hi_r,
                drop_cx - drop_r * 0.25 + hi_r,
                drop_cy - drop_r * 0.10 + hi_r,
            ),
            fill=SOURCE_DROP_HIGHLIGHT,
        )
        shaft_left = left + cs * 0.58
        shaft_right = left + cs * 0.88
        shaft_y = top + cs * 0.50
        shaft_thickness = max(2, cs // 18)
        shaft_thickness -= shaft_thickness % 2
        draw.rectangle(
            (
                shaft_left,
                shaft_y - shaft_thickness // 2,
                shaft_right,
                shaft_y + shaft_thickness // 2 - 1,
            ),
            fill=SOURCE_ARROW_COLOR,
        )
        head_w = cs * 0.12
        head_h = cs * 0.10
        draw.polygon(
            [
                (shaft_right, shaft_y - head_h),
                (shaft_right + head_w, shaft_y),
                (shaft_right, shaft_y + head_h),
            ],
            fill=SOURCE_ARROW_COLOR,
        )

    def _draw_sink_marker(
        self,
        draw: ImageDraw.ImageDraw,
        sink_cell: Tuple[int, int],
        pad_left: int,
        pad_top: int,
        *,
        active: bool,
    ) -> None:
        r, c = sink_cell
        cs = self.cell_size
        left = pad_left + c * cs
        top = pad_top + r * cs

        basin_cx = left + cs * 0.52
        basin_cy = top + cs * 0.72
        basin_rx = cs * 0.42
        basin_ry = cs * 0.18
        draw.ellipse(
            (
                basin_cx - basin_rx - 1,
                basin_cy - basin_ry + 3,
                basin_cx + basin_rx + 1,
                basin_cy + basin_ry + 3,
            ),
            fill=FOUNTAIN_BASIN_SHADOW,
        )
        draw.ellipse(
            (
                basin_cx - basin_rx,
                basin_cy - basin_ry,
                basin_cx + basin_rx,
                basin_cy + basin_ry,
            ),
            fill=FOUNTAIN_BASIN_COLOR,
        )
        pool_rx = basin_rx * 0.75
        pool_ry = basin_ry * 0.65
        draw.ellipse(
            (
                basin_cx - pool_rx,
                basin_cy - pool_ry - 1,
                basin_cx + pool_rx,
                basin_cy + pool_ry - 1,
            ),
            fill=FOUNTAIN_WATER_COLOR,
        )
        hi_rx = pool_rx * 0.35
        hi_ry = pool_ry * 0.35
        draw.ellipse(
            (
                basin_cx - pool_rx * 0.35 - hi_rx,
                basin_cy - pool_ry * 0.45 - hi_ry,
                basin_cx - pool_rx * 0.35 + hi_rx,
                basin_cy - pool_ry * 0.45 + hi_ry,
            ),
            fill=FOUNTAIN_SPRAY_COLOR,
        )
        col_half_w = max(2, cs // 18)
        col_top = top + cs * 0.26
        col_bottom = basin_cy - cs * 0.04
        draw.rectangle(
            (
                basin_cx - col_half_w,
                col_top,
                basin_cx + col_half_w,
                col_bottom,
            ),
            fill=FOUNTAIN_BASIN_COLOR,
        )
        cap_r = col_half_w + 1
        draw.ellipse(
            (basin_cx - cap_r, col_top - cap_r, basin_cx + cap_r, col_top + cap_r),
            fill=FOUNTAIN_BASIN_COLOR,
        )
        if not active:
            return

        spray_origin = (basin_cx, col_top)
        spray_length = cs * 0.26
        spray_thickness = max(2, cs // 20)
        spray_thickness -= spray_thickness % 2
        draw.line(
            [spray_origin, (spray_origin[0], spray_origin[1] - spray_length)],
            fill=FOUNTAIN_SPRAY_COLOR,
            width=spray_thickness,
        )
        draw.line(
            [
                spray_origin,
                (
                    spray_origin[0] - spray_length * 0.80,
                    spray_origin[1] - spray_length * 0.65,
                ),
            ],
            fill=FOUNTAIN_SPRAY_COLOR,
            width=spray_thickness,
        )
        draw.line(
            [
                spray_origin,
                (
                    spray_origin[0] + spray_length * 0.80,
                    spray_origin[1] - spray_length * 0.65,
                ),
            ],
            fill=FOUNTAIN_SPRAY_COLOR,
            width=spray_thickness,
        )
        drop_r = max(2, cs // 22)
        tip_positions = [
            (spray_origin[0], spray_origin[1] - spray_length),
            (
                spray_origin[0] - spray_length * 0.80,
                spray_origin[1] - spray_length * 0.65,
            ),
            (
                spray_origin[0] + spray_length * 0.80,
                spray_origin[1] - spray_length * 0.65,
            ),
        ]
        for tx, ty in tip_positions:
            draw.ellipse(
                (tx - drop_r, ty - drop_r, tx + drop_r, ty + drop_r),
                fill=FOUNTAIN_SPRAY_COLOR,
            )

    # ---------- labels ----------

    def _draw_axis_labels(
        self,
        draw: ImageDraw.ImageDraw,
        pad_left: int,
        pad_top: int,
    ) -> None:
        """Number the rows down the left gutter and the columns across the top.

        These are the only glyphs the scene renders, they sit outside the grid
        body, and they are identical in the input, the terminal image, and
        every video frame.
        """

        draw_grid_axis_labels(
            draw,
            rows=self.grid_size,
            cols=self.grid_size,
            cell_size=self.cell_size,
            pad_left=pad_left,
            pad_top=pad_top,
        )

    # ---------- solution polyline ----------

    def _flow_polyline_points(
        self,
        layout: Dict[str, Any],
        pad_left: int,
        pad_top: int,
    ) -> List[Tuple[float, float]]:
        cs = self.cell_size
        points: List[Tuple[float, float]] = []
        for r, c in layout["flow_cells"]:
            points.append((pad_left + c * cs + cs / 2.0, pad_top + r * cs + cs / 2.0))
        return points

    def _flow_thickness(self) -> int:
        raw = (self.cell_size // 3)
        even = raw & ~1
        return max(4, even)

    # ------------------------------------------------------------------
    # Video pipeline
    # ------------------------------------------------------------------

    def _save_rotation_video(
        self,
        record_id: str,
        layout: Dict[str, Any],
        pad_left: int,
        pad_top: int,
        canvas_dims: Tuple[int, int],
        *,
        fps: int = 16,
    ) -> Optional[Path]:
        """Continuously rotate live-path tiles, then fill the corrected pipe."""
        self._last_video_fps = None
        self._last_video_num_frames = None
        self._last_video_process = None
        if not self.video:
            return None

        video_path = self.solution_dir / f"{record_id}_solution.webm"
        padding = (pad_left, pad_top)

        # Working state: a dict of cell -> current rotation.
        path_cells: List[Tuple[int, int]] = list(layout["flow_cells"])

        # Identify the path cells whose VISIBLE shape changes between the
        # displayed and correct rotations. Cells where rot_displayed
        # differs from rot_correct but the resulting opening-set is the
        # same (e.g. a STRAIGHT tile rotated 180°: openings stay {E, W})
        # would otherwise produce a visually static rotation interval — the
        # rotation table changes but the tile pixels are bit-identical,
        # so the viewer sees no visual change for those frames and reads
        # rotation interval as non-uniform. Filter by opening-set equality
        # instead of rotation-value equality so every animated turn in the
        # video is a real visible change.
        rot_correct = layout["rot_correct"]
        rot_displayed = layout["rot_displayed"]
        kinds = layout["kinds"]
        misaligned_in_order: List[Tuple[int, int]] = [
            cell for cell in path_cells
            if kind_openings(kinds[cell], rot_displayed[cell])
            != kind_openings(kinds[cell], rot_correct[cell])
        ]
        def signed_quarter_turns(start: int, end: int) -> int:
            clockwise = (int(end) - int(start)) % 4
            if clockwise == 3:
                return -1
            if clockwise == 2:
                return 2
            return clockwise

        rotation_plan = [
            (cell, signed_quarter_turns(rot_displayed[cell], rot_correct[cell]))
            for cell in misaligned_in_order
        ]
        max_quarter_turns = max((abs(turns) for _cell, turns in rotation_plan), default=0)

        max_total_frames = int(fps * 9.5)
        open_hold = 5
        gap = 4
        final_hold = 8
        frames_per_quarter = 3
        # Every independent correction is one global alignment phase.  The
        # tiles share normalized progress and land together; path order is
        # reserved for the later, genuinely causal water propagation.
        rotation_budget = frames_per_quarter * max_quarter_turns
        flow_target = max(25, max_total_frames - open_hold - rotation_budget - gap - final_hold)

        frames: List[Image.Image] = []

        # Path polyline at PIPE-CELL centres (source cell is the first).
        flow_points = self._flow_polyline_points(layout, pad_left, pad_top)

        # Helper: render the scene at the given rotation table, no flow stroke.
        # Each distinct rotation table is drawn exactly once, kept at scene
        # scale so rotating tiles and the water stroke can be laid on top before
        # the single LANCZOS pass, and separately cached in its finished form so
        # hold frames are plain copies.
        scaled_cache: Dict[Tuple[Tuple[Tuple[int, int], int], ...], Image.Image] = {}
        finished_cache: Dict[Tuple[Tuple[Tuple[int, int], int], ...], Image.Image] = {}

        def render_state_scaled(rot_table: Dict[Tuple[int, int], int]) -> Image.Image:
            cache_key = tuple(sorted(rot_table.items()))
            cached = scaled_cache.get(cache_key)
            if cached is not None:
                return cached
            scene, draw = self._begin_scene(canvas_dims)
            self._paint_board(scene, draw, layout, padding, rot_table=rot_table)
            self._draw_axis_labels(draw, pad_left, pad_top)
            scaled_cache[cache_key] = scene
            return scene

        def render_state(rot_table: Dict[Tuple[int, int], int]) -> Image.Image:
            cache_key = tuple(sorted(rot_table.items()))
            cached = finished_cache.get(cache_key)
            if cached is None:
                cached = self._finish_scene(render_state_scaled(rot_table), canvas_dims)
                finished_cache[cache_key] = cached
            return cached.copy()

        def render_rotating_state(
            rot_table: Dict[Tuple[int, int], int],
            plan: Sequence[Tuple[Tuple[int, int], int]],
            progress: float,
        ) -> Image.Image:
            scene = render_state_scaled(rot_table).copy()
            draw = self._scene_draw(scene)
            cs = self.cell_size
            factor = SCENE_SUPERSAMPLE
            tile_px = cs * factor
            for cell, signed_turns in plan:
                tile = Image.new("RGB", (tile_px, tile_px), STONE_BASE_COLOR)
                self._draw_pipe_tile(
                    ScaledDraw(ImageDraw.Draw(tile), factor),
                    0,
                    0,
                    kinds[cell],
                    rot_table[cell],
                )
                clockwise_degrees = float(signed_turns) * 90.0 * float(progress)
                rotated = tile.rotate(
                    -clockwise_degrees,
                    resample=Image.Resampling.BICUBIC,
                    expand=False,
                    fillcolor=STONE_BASE_COLOR,
                )
                left = pad_left + cell[1] * cs
                top = pad_top + cell[0] * cs
                scene.paste(rotated, (left * factor, top * factor))
            self._draw_source_marker(draw, layout["source_cell"], pad_left, pad_top)
            self._draw_sink_marker(
                draw,
                layout["sink_cell"],
                pad_left,
                pad_top,
                active=False,
            )
            self._draw_axis_labels(draw, pad_left, pad_top)
            return self._finish_scene(scene, canvas_dims)

        # Phase 1: open hold on the DISPLAYED rotations.
        current_rot: Dict[Tuple[int, int], int] = dict(rot_displayed)
        opening_frame = render_state(current_rot)
        for _ in range(open_hold):
            frames.append(opening_frame.copy())

        # Phase 2: rotate every misaligned live tile together.  Their
        # rotations are independent constraints, so assigning a
        # source-to-fountain priority would manufacture a solver order.
        for frame_index in range(rotation_budget):
            progress = (frame_index + 1) / rotation_budget
            if progress < 1.0:
                frames.append(render_rotating_state(current_rot, rotation_plan, progress))
            else:
                current_rot.update({cell: rot_correct[cell] for cell, _turns in rotation_plan})
                frames.append(render_state(current_rot))

        # Phase 3: short gap on the corrected static state before flow starts.
        corrected_scaled = render_state_scaled(current_rot)
        corrected_static = render_state(current_rot)
        for _ in range(gap):
            frames.append(corrected_static.copy())

        # Phase 4: animate the bright-cyan water stroke across the corrected board.
        flow_frames = flow_target
        if len(flow_points) >= 2:
            # Cumulative segment lengths.
            seg_lens: List[float] = []
            cum: List[float] = []
            total = 0.0
            for i in range(len(flow_points) - 1):
                d = math.hypot(
                    flow_points[i + 1][0] - flow_points[i][0],
                    flow_points[i + 1][1] - flow_points[i][1],
                )
                seg_lens.append(d)
                total += d
                cum.append(total)
            for f_idx in range(flow_frames):
                progress = (f_idx + 1) / flow_frames
                cur_dist = total * progress
                # Build the partial polyline at this distance.
                partial = [flow_points[0]]
                for i, threshold in enumerate(cum):
                    if threshold <= cur_dist + 1e-6:
                        partial.append(flow_points[i + 1])
                    else:
                        # Interpolate the tip on the partial segment.
                        prev = cum[i - 1] if i > 0 else 0.0
                        seg_len = threshold - prev
                        remain = cur_dist - prev
                        if seg_len > 1e-9 and remain > 0:
                            ratio = min(1.0, remain / seg_len)
                            p1, p2 = flow_points[i], flow_points[i + 1]
                            partial.append((
                                p1[0] + (p2[0] - p1[0]) * ratio,
                                p1[1] + (p2[1] - p1[1]) * ratio,
                            ))
                        break
                # The stroke and the repainted endpoints go onto a copy of the
                # supersampled corrected board, so the fountain and source drop
                # keep the same antialiasing they have in the static renders.
                scene = corrected_scaled.copy()
                self._paint_flow(
                    scene,
                    self._scene_draw(scene),
                    layout,
                    padding,
                    partial,
                    active_sink=progress >= 1.0,
                )
                frames.append(self._finish_scene(scene, canvas_dims))
        else:
            for _ in range(flow_frames):
                frames.append(corrected_static.copy())

        # Phase 5: final hold on the fully solved scene.
        if frames:
            final_frame = frames[-1]
            for _ in range(final_hold):
                frames.append(final_frame.copy())

        frames = pad_frames_to_wan_compatible_length(frames)
        self._last_video_process = {
            "open_hold_frames": open_hold,
            "rotation_frames": rotation_budget,
            "corrected_dry_hold_frames": gap,
            "flow_frames": flow_frames,
            "final_hold_frames_before_padding": final_hold,
            "simultaneous_rotation_positions": [list(cell_to_xy(cell)) for cell, _turns in rotation_plan],
            "signed_quarter_turns": [int(turns) for _cell, turns in rotation_plan],
            "rotation_policy": "simultaneous_shared_progress_shortest_signed_quarter_turns_clockwise_on_180_tie",
            "flow_policy": "continuous_source_to_sink_no_decoy_branch",
            "fountain_activation": "only_at_flow_completion",
        }
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_cell_id(self, r: int, c: int) -> int:
        return self._cell_to_id.get((r, c), 0)

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        row, col = self._id_to_cell[cell_id]
        pad_left, pad_top, _, _, _ = self._compute_padding()
        return (
            pad_left + col * self.cell_size + self.cell_size / 2.0,
            pad_top + row * self.cell_size + self.cell_size / 2.0,
        )

    def _compute_padding(self) -> Tuple[int, int, int, int, Tuple[int, int]]:
        pad_left, pad_top, pad_right, pad_bottom = self.padding
        return pad_left, pad_top, pad_right, pad_bottom, self.canvas_dimensions


__all__ = ["MazeAqueductPipesGenerator"]
