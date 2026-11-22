from __future__ import annotations

from data.common.verifiers.keyframes import keyframe_indices, visual_delta

import importlib
import inspect
import time
from functools import lru_cache
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.schemas import CanonicalSample, EvalRecord
from data.common.progress_alignment import (
    _PAIR_CHANGE_SLACK,
    _STEP_SLACK,
    align_progress,
    dynamic_probe,
    emission_costs,
    observed_pair_changes,
    read_process_order,
    reference_advance,
    reference_pair_changes,
    tone_offset,
)
from data.common.verifiers.region_gates import (
    REGION,
    STATIC_REGION_FLOOR,
    static_region_errors,
)
from data.registry import TASK_SPECS
from data.common.pipeline.path_tasks import (
    GRID_AXIS_LABEL_COLOR,
    PathTaskEvaluator,
    cell_to_xy,
    draw_path_line,
    pawn_marker_radius,
)
from data.common.verifiers.candidates import (
    resolve_candidate_video,
    resolve_source_record,
    resolve_source_record_id,
    resolve_source_task_dir,
)
from data.common.verifiers.keyframes import _compact_keyframe_sequence
from data.common.verifiers.keyframes import _find_monotonic_keyframe_sequence
from data.common.verifiers.metrics import _verifier_contract_metrics


MAZE_SEMANTIC_PATH_VIDEO_TASKS = {
    "maze_colored_zone",
    "maze_elevation",
    "maze_hexagon",
    "maze_keys_and_locks",
    "maze_labyrinth",
    "maze_layered",
    "maze_square",
    "maze_toroidal",
}
MAZE_SINGLE_DECODE_VIDEO_PROFILES = {
    "maze_aqueduct_pipes": "maze_aqueduct_pipes_full_video_semantic_single_decode",
}
_SQUARE_ROUTE_GEOMETRY_TASKS = {
    "maze_colored_zone",
    "maze_elevation",
    "maze_keys_and_locks",
    "maze_square",
}
_SOURCE_REPLAY_SPATIAL_NAVIGATION_MAZES = {
    "maze_colored_zone",
    "maze_elevation",
    "maze_hexagon",
    "maze_keys_and_locks",
    "maze_labyrinth",
    "maze_layered",
    "maze_square",
    "maze_toroidal",
}
_SPATIAL_NAVIGATION_MAZE_VERIFIER_PROFILE = "spatial_navigation_maze_source_replay_multimodal_hard_gate_v2"
_SPATIAL_NAVIGATION_MAZE_VIDEO_PROCESS_VERIFIER = "spatial_navigation_maze_source_replay_continuous_legal_route_v2"
MAZE_ROUTE_MASK_MATCH_THRESHOLD = 0.7
MAZE_ROUTE_MASK_MATCH_MARGIN = 0.01
MAZE_ROUTE_MASK_ENDPOINT_THRESHOLD = 0.12
MAZE_ROUTE_MASK_FINAL_COVERAGE_THRESHOLD = 0.92
MAZE_VIDEO_MAX_UNRECOGNIZED_FRACTION = 0.1
_MAZE_ROUTE_MASK_DILATION_SIZE = 7
_MAZE_STATIC_RED_DILATION_SIZE = 3

# How many decoded frames of a maze route video carry no process advance.  The
# shared path-task video writer emits ``total_frames + 1`` route frames whose
# revealed arc length runs from 0 to the whole route, prepends four copies of
# the first of them as the opening hold on the untouched input, and appends
# ``max(4, fps // 2) = 8`` copies of the terminal state.  Four plus eight of
# those frames repeat their neighbour exactly, and the first route frame reveals
# nothing, so ``frame_count - 13`` frames carry the advance.  Every task on this
# route -- including ``maze_labyrinth`` and ``maze_keys_and_locks``, which write
# their own frame loops -- uses that same layout, which is why the constant is
# shared rather than per task.
_MAZE_VIDEO_NON_ADVANCING_FRAMES = 13

# Progress states in the source route bank the bounded-step reading walks.  A
# bank coarser than the video's own advance leaves most frames sitting between
# two entries, which reads as a jump the source never makes, so the bank is
# sampled at least as densely as the video advances per frame (the frame count)
# and never below 32.  Mirrors the Physics route's ``max(32, len(frames))``.
_MAZE_PROGRESS_BANK_MIN_STATES = 32

# Dilation applied to the union of the bank's route footprints to obtain the
# region the process legitimately changes.  The candidate rasterises the same
# stroke through a different antialiasing path (three of the eight tasks
# composite a supersampled overlay), so the corridor is widened by one
# ``_MAZE_ROUTE_MASK_DILATION_SIZE`` step -- the same allowance the route-mask
# coverage check already grants the candidate's edge pixels.
_MAZE_DYNAMIC_DILATION_SIZE = _MAZE_ROUTE_MASK_DILATION_SIZE

# Inset excludes antialiased pixels from the shared core/ring samples.
_MAZE_TIP_RASTER_INSET_PX = 1.5

# A marker has a red core surrounded by a white ring. Require high contrast
# between those regions so an ordinary route stroke does not count as a tip.
_MAZE_TIP_DETECTION_SCORE = 0.875

# Arc-advance diagnostic, with a cell-transition floor in the cursor check so
# timestamp resampling does not impose the reference renderer's exact speed.
_MAZE_TIP_STEP_SLACK = 1.9

_AQUEDUCT_VERIFIER_PROFILE = "maze_aqueduct_pipes_visible_pipe_multimodal_hard_gate_v2"
_AQUEDUCT_DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
_AQUEDUCT_OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}
_AQUEDUCT_ROTATION = {"N": "E", "E": "S", "S": "W", "W": "N"}
_AQUEDUCT_BASE_OPENINGS = {
    "STRAIGHT": frozenset({"E", "W"}),
    "ELBOW": frozenset({"N", "E"}),
    "T": frozenset({"N", "E", "W"}),
    "CROSS": frozenset({"N", "E", "S", "W"}),
}
_AQUEDUCT_COLORS = {
    "stone": (198, 166, 120),
    "blank": (172, 140, 96),
    "pipe_outline": (62, 38, 20),
    "pipe_interior": (172, 138, 94),
    "lining": (48, 132, 210),
    "flow": (24, 174, 236),
    "source": (18, 90, 175),
    "basin": (156, 124, 86),
    "spray": (180, 225, 250),
    # 1-based row/column numbers in the left and top gutters; the only text the
    # aqueduct scene renders.
    "axis_label": GRID_AXIS_LABEL_COLOR,
}
_AQUEDUCT_VIDEO_MAX_FRAMES = 157

# Structuring-element size for the flow observation's dilation.  Shared by the
# per-frame occupancy read and by the source bank it is compared against, so the
# two cannot drift apart.
_AQUEDUCT_FLOW_DILATION_SIZE = 9

# Frames of the flow window that carry no advance.  The renderer advances the
# water by ``1 / flow_frames`` on every frame of the flow phase and then holds the
# solved scene for eight frames; the length-alignment padding appends up to three
# further copies of that same final frame, so supplying the hold alone makes the
# derived advance at most 12% slower than the renderer's own -- conservative, in
# the direction that tightens.  It is supplied rather than read from the record on
# purpose: the generator publishes the number inside ``video_process``, which this
# task's contract lists in ``ignored_oracle_fields``, and reading it from the
# candidate instead would let a candidate that deleted frames declare its own pace.
_AQUEDUCT_FLOW_NON_ADVANCING_FRAMES = 8


def _aqueduct_rotated_openings(kind: str, quarter_turns: int) -> frozenset[str]:
    openings = set(_AQUEDUCT_BASE_OPENINGS.get(str(kind), frozenset()))
    for _ in range(int(quarter_turns) % 4):
        openings = {_AQUEDUCT_ROTATION[direction] for direction in openings}
    return frozenset(openings)


def _aqueduct_kind_from_openings(openings: frozenset[str]) -> Optional[str]:
    if len(openings) == 4:
        return "CROSS"
    if len(openings) == 3:
        return "T"
    if openings in {frozenset({"E", "W"}), frozenset({"N", "S"})}:
        return "STRAIGHT"
    if len(openings) == 2:
        return "ELBOW"
    return None


def _aqueduct_color_mask(
    pixels: np.ndarray,
    color: Tuple[int, int, int],
    *,
    tolerance: int,
) -> np.ndarray:
    delta = np.abs(pixels.astype(np.int16) - np.asarray(color, dtype=np.int16))
    return np.max(delta, axis=2) <= int(tolerance)


def _aqueduct_flow_mask(pixels: np.ndarray) -> np.ndarray:
    red = pixels[:, :, 0].astype(np.int16)
    green = pixels[:, :, 1].astype(np.int16)
    blue = pixels[:, :, 2].astype(np.int16)
    exactish = _aqueduct_color_mask(pixels, _AQUEDUCT_COLORS["flow"], tolerance=38)
    # The dry blue lining and the cyan flow differ mainly in green-minus-red.
    # An RGB box around cyan overlaps the lining after video compression; the
    # midpoint of the two renderer colours separates the actual water from it.
    lining_chroma = _AQUEDUCT_COLORS["lining"][1] - _AQUEDUCT_COLORS["lining"][0]
    flow_chroma = _AQUEDUCT_COLORS["flow"][1] - _AQUEDUCT_COLORS["flow"][0]
    return exactish & (green >= 145) & (blue >= 185) & (green - red > (lining_chroma + flow_chroma) / 2)


def _aqueduct_geometry(record: Dict[str, Any]) -> Optional[Dict[str, int]]:
    try:
        grid_size = int(record.get("grid_size"))
        cell_size = int(record.get("cell_size"))
        canvas = record.get("canvas_dimensions")
        padding = record.get("padding")
        width, height = int(canvas[0]), int(canvas[1])
        pad_left, pad_top, pad_right, pad_bottom = [int(value) for value in padding]
    except (TypeError, ValueError, IndexError):
        return None
    if (
        grid_size != 9
        or cell_size <= 0
        or width != 512
        or height != 512
        or pad_left < 0
        or pad_top < 0
        or pad_right < 0
        or pad_bottom < 0
        or pad_left + grid_size * cell_size + pad_right != width
        or pad_top + grid_size * cell_size + pad_bottom != height
    ):
        return None
    return {
        "grid_size": grid_size,
        "cell_size": cell_size,
        "width": width,
        "height": height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }


def _aqueduct_cell_bounds(
    geometry: Dict[str, int],
    cell: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    row, col = cell
    left = geometry["pad_left"] + col * geometry["cell_size"]
    top = geometry["pad_top"] + row * geometry["cell_size"]
    return left, top, left + geometry["cell_size"], top + geometry["cell_size"]


def _aqueduct_stone_body_floor(evaluator: object, record: Dict[str, Any]) -> float:
    """Sandstone-body floor for "this cell carries a pipe", from the renderer.

    Pipe tiles and blank tiles are painted from the same sandstone palette, so
    the body fraction is a sanity check and the discriminator is the pipe-outline
    colour, which a blank tile never paints.  A pipe tile shows the *least*
    sandstone when the widest footprint the renderer can draw (a CROSS) covers
    the cell, and a blank tile shows the *most*, so both ends are measured on
    tiles rendered by the generator itself and the boundary is their midpoint.
    A hand-picked constant is what made this gate brittle: it sat one point
    under the CROSS tile's exposed-stone fraction, so the anti-aliased edges of
    the family's 4x render floor pushed a legitimate tile below it.
    """

    generator = evaluator._build_generator(record)
    cell_size = int(record["cell_size"])
    fractions: List[float] = []
    for paint in (
        lambda draw: generator._draw_pipe_tile(draw, 0, 0, "CROSS", 0),
        lambda draw: generator._draw_blank_tile(draw, 0, 0, 0, 0),
    ):
        tile, tile_draw = generator._begin_scene((cell_size, cell_size))
        paint(tile_draw)
        reference = generator._finish_scene(tile, (cell_size, cell_size))
        fractions.append(
            float(
                _aqueduct_color_mask(
                    np.asarray(reference.convert("RGB"), dtype=np.uint8),
                    _AQUEDUCT_COLORS["stone"],
                    tolerance=12,
                ).mean()
            )
        )
    return (fractions[0] + fractions[1]) / 2.0


def _aqueduct_detect_pipe_cells(
    pixels: np.ndarray,
    geometry: Dict[str, int],
    *,
    stone_body_floor: float,
) -> set[Tuple[int, int]]:
    cells: set[Tuple[int, int]] = set()
    for row in range(geometry["grid_size"]):
        for col in range(geometry["grid_size"]):
            left, top, right, bottom = _aqueduct_cell_bounds(geometry, (row, col))
            crop = pixels[top:bottom, left:right]
            stone_fraction = float(
                _aqueduct_color_mask(crop, _AQUEDUCT_COLORS["stone"], tolerance=12).mean()
            )
            outline_fraction = float(
                _aqueduct_color_mask(crop, _AQUEDUCT_COLORS["pipe_outline"], tolerance=16).mean()
            )
            if stone_fraction >= stone_body_floor and outline_fraction >= 0.015:
                cells.add((row, col))
    return cells


def _aqueduct_band_fractions(
    pixels: np.ndarray,
    geometry: Dict[str, int],
    cell: Tuple[int, int],
) -> Dict[str, float]:
    """Lining-or-water share of each cell-edge band, one per port direction."""

    left, top, right, bottom = _aqueduct_cell_bounds(geometry, cell)
    cell_size = geometry["cell_size"]
    center_x = left + cell_size // 2
    center_y = top + cell_size // 2
    half_band = max(3, cell_size // 12)
    depth = max(4, cell_size // 10)
    bands = {
        "N": pixels[top : top + depth, center_x - half_band : center_x + half_band],
        "E": pixels[center_y - half_band : center_y + half_band, right - depth : right],
        "S": pixels[bottom - depth : bottom, center_x - half_band : center_x + half_band],
        "W": pixels[center_y - half_band : center_y + half_band, left : left + depth],
    }
    fractions: Dict[str, float] = {}
    for direction, band in bands.items():
        lining = _aqueduct_color_mask(band, _AQUEDUCT_COLORS["lining"], tolerance=34)
        flow = _aqueduct_flow_mask(band)
        fractions[direction] = float((lining | flow).mean())
    return fractions


def _aqueduct_detect_openings(
    pixels: np.ndarray,
    geometry: Dict[str, int],
    cell: Tuple[int, int],
) -> frozenset[str]:
    return frozenset(
        direction
        for direction, fraction in _aqueduct_band_fractions(pixels, geometry, cell).items()
        if fraction >= 0.16
    )


# A port band counts as hidden by an endpoint glyph once the glyph takes away
# most of the lining the bare pipe tile shows there.
_AQUEDUCT_BAND_OCCLUSION_SHARE = 0.5


def _aqueduct_reference_band_fractions(
    generator: object,
    cell_size: int,
    *,
    paint_marker=None,
) -> Dict[str, float]:
    """Band fractions of an all-ports tile the renderer draws for itself."""

    scene, draw = generator._begin_scene((cell_size, cell_size))
    generator._draw_pipe_tile(draw, 0, 0, "CROSS", 0)
    if paint_marker is not None:
        paint_marker(draw)
    reference = generator._finish_scene(scene, (cell_size, cell_size))
    return _aqueduct_band_fractions(
        np.asarray(reference.convert("RGB"), dtype=np.uint8),
        {"grid_size": 1, "cell_size": int(cell_size), "pad_left": 0, "pad_top": 0},
        (0, 0),
    )


def _aqueduct_endpoint_occlusions(
    evaluator: object,
    record: Dict[str, Any],
    geometry: Dict[str, int],
) -> Dict[str, frozenset[str]]:
    """Port bands each endpoint glyph paints over, measured on the renderer.

    The source drop-and-arrow and the fountain basin are painted on top of their
    own pipe tile, so a port they cover cannot be recovered from its edge band
    and its lining must not be required as visible evidence.  Which ports that
    is follows from the glyph footprints, so it is measured by drawing each glyph
    over an all-ports tile through the generator instead of being maintained as a
    hand-written direction set — that set listed only the fully hidden port and
    silently excluded the fountain's wide basin, which also covers most of the
    band below it once the family's 4x render floor antialiases its rim.
    """

    generator = evaluator._build_generator(record)
    cell_size = int(geometry["cell_size"])
    bare = _aqueduct_reference_band_fractions(generator, cell_size)
    occlusions: Dict[str, frozenset[str]] = {}
    for name, paint_marker in (
        ("source", lambda draw: generator._draw_source_marker(draw, (0, 0), 0, 0)),
        ("sink", lambda draw: generator._draw_sink_marker(draw, (0, 0), 0, 0, active=False)),
    ):
        covered = _aqueduct_reference_band_fractions(
            generator, cell_size, paint_marker=paint_marker
        )
        occlusions[name] = frozenset(
            direction
            for direction in ("N", "E", "S", "W")
            if covered[direction] < bare[direction] * _AQUEDUCT_BAND_OCCLUSION_SHARE
        )
    return occlusions


def _aqueduct_occluded_bands(
    cell: Tuple[int, int],
    *,
    source_cell: Tuple[int, int],
    sink_cell: Tuple[int, int],
    occlusions: Dict[str, frozenset[str]],
) -> frozenset[str]:
    if cell == source_cell:
        return occlusions["source"]
    if cell == sink_cell:
        return occlusions["sink"]
    return frozenset()


def _aqueduct_marker_cell(
    pixels: np.ndarray,
    geometry: Dict[str, int],
    *,
    column: int,
    color_name: str,
    tolerance: int,
) -> Optional[Tuple[int, int]]:
    scores: List[Tuple[int, int]] = []
    for row in range(geometry["grid_size"]):
        left, top, right, bottom = _aqueduct_cell_bounds(geometry, (row, column))
        crop = pixels[top:bottom, left:right]
        score = int(
            _aqueduct_color_mask(crop, _AQUEDUCT_COLORS[color_name], tolerance=tolerance).sum()
        )
        scores.append((score, row))
    score, row = max(scores)
    runner_up = max((value for value, candidate_row in scores if candidate_row != row), default=0)
    if score < 20 or score <= runner_up * 1.5:
        return None
    return row, column


def _aqueduct_required_openings(
    path: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], frozenset[str]]:
    required: Dict[Tuple[int, int], frozenset[str]] = {}
    for index, cell in enumerate(path):
        directions: set[str] = set()
        if index == 0:
            directions.add("W")
        else:
            previous = path[index - 1]
            for direction, (dr, dc) in _AQUEDUCT_DIRECTIONS.items():
                if (cell[0] + dr, cell[1] + dc) == previous:
                    directions.add(direction)
                    break
        if index == len(path) - 1:
            directions.add("E")
        else:
            following = path[index + 1]
            for direction, (dr, dc) in _AQUEDUCT_DIRECTIONS.items():
                if (cell[0] + dr, cell[1] + dc) == following:
                    directions.add(direction)
                    break
        required[cell] = frozenset(directions)
    return required


def _aqueduct_unique_path(
    pipe_kinds: Dict[Tuple[int, int], str],
    source_cell: Tuple[int, int],
    sink_cell: Tuple[int, int],
    *,
    limit: int = 2,
) -> List[List[Tuple[int, int]]]:
    found: List[List[Tuple[int, int]]] = []

    def supports(kind: str, required: set[str]) -> bool:
        if kind not in {"STRAIGHT", "ELBOW"}:
            return False
        return any(
            required.issubset(_aqueduct_rotated_openings(kind, rotation))
            for rotation in range(4)
        )

    def visit(
        cell: Tuple[int, int],
        incoming: str,
        path: List[Tuple[int, int]],
    ) -> None:
        if len(found) >= max(1, int(limit)):
            return
        kind = pipe_kinds.get(cell)
        if kind is None:
            return
        if cell == sink_cell:
            if supports(kind, {incoming, "E"}):
                found.append(list(path))
            return
        for outgoing, (dr, dc) in _AQUEDUCT_DIRECTIONS.items():
            neighbour = (cell[0] + dr, cell[1] + dc)
            if neighbour not in pipe_kinds or neighbour in path:
                continue
            if not supports(kind, {incoming, outgoing}):
                continue
            visit(neighbour, _AQUEDUCT_OPPOSITE[outgoing], [*path, neighbour])

    if source_cell in pipe_kinds and sink_cell in pipe_kinds:
        visit(source_cell, "W", [source_cell])
    return found


def _aqueduct_visible_replay(
    record: Dict[str, Any],
    evaluator: object,
    *,
    source_key: str,
) -> Dict[str, Any]:
    geometry = _aqueduct_geometry(record)
    if geometry is None:
        return {"valid": False, "reason": "invalid aqueduct source geometry"}
    try:
        source_path = evaluator.resolve_path(record[source_key])
    except Exception:
        return {"valid": False, "reason": f"missing {source_key} source path"}
    if not Path(source_path).exists():
        return {"valid": False, "reason": f"missing {source_key} source image"}
    with Image.open(source_path) as image:
        source_pixels = np.asarray(image.convert("RGB"))
    if source_pixels.shape[:2] != (geometry["height"], geometry["width"]):
        return {"valid": False, "reason": "source image dimensions do not match geometry"}

    stone_body_floor = _aqueduct_stone_body_floor(evaluator, record)
    endpoint_occlusions = _aqueduct_endpoint_occlusions(evaluator, record, geometry)
    detected_cells = _aqueduct_detect_pipe_cells(
        source_pixels, geometry, stone_body_floor=stone_body_floor
    )
    if not detected_cells:
        return {"valid": False, "reason": "no visible pipe cells recovered"}
    source_cell = _aqueduct_marker_cell(
        source_pixels,
        geometry,
        column=0,
        color_name="source",
        tolerance=14,
    )
    sink_cell = _aqueduct_marker_cell(
        source_pixels,
        geometry,
        column=geometry["grid_size"] - 1,
        color_name="basin",
        tolerance=10,
    )
    if source_cell is None or sink_cell is None:
        return {"valid": False, "reason": "source or fountain marker could not be recovered"}
    if source_cell not in detected_cells or sink_cell not in detected_cells:
        return {"valid": False, "reason": "source or fountain is not on a visible pipe tile"}

    detected_openings: Dict[Tuple[int, int], frozenset[str]] = {}
    for cell in sorted(detected_cells):
        detected_openings[cell] = _aqueduct_detect_openings(source_pixels, geometry, cell)

    # Attest source-descriptive metadata against the visible image, while
    # deliberately ignoring saved solution paths and corrected rotations.
    specs = record.get("pipe_specs")
    if not isinstance(specs, list) or len(specs) != len(detected_cells):
        return {"valid": False, "reason": "pipe_specs do not cover visible pipe cells"}
    spec_cells: set[Tuple[int, int]] = set()
    displayed_openings: Dict[Tuple[int, int], frozenset[str]] = {}
    pipe_kinds: Dict[Tuple[int, int], str] = {}
    for spec in specs:
        if not isinstance(spec, dict):
            return {"valid": False, "reason": "malformed pipe spec"}
        try:
            cell = (int(spec["grid_row"]), int(spec["grid_col"]))
            kind = str(spec["kind"])
            displayed_rotation = int(spec["rotation_displayed"])
        except (KeyError, TypeError, ValueError):
            return {"valid": False, "reason": "malformed pipe spec fields"}
        if cell in spec_cells or cell not in detected_cells:
            return {"valid": False, "reason": "pipe spec cell mismatch"}
        spec_cells.add(cell)
        expected_visible = _aqueduct_rotated_openings(kind, displayed_rotation // 90)
        visible = detected_openings[cell]
        occluded = _aqueduct_occluded_bands(
            cell,
            source_cell=source_cell,
            sink_cell=sink_cell,
            occlusions=endpoint_occlusions,
        )
        if not visible.issubset(expected_visible) or not (expected_visible - occluded).issubset(visible):
            return {"valid": False, "reason": "pipe spec kind/rotation disagrees with visible source"}
        if kind not in _AQUEDUCT_BASE_OPENINGS:
            return {"valid": False, "reason": "unknown visible pipe kind"}
        displayed_openings[cell] = expected_visible
        pipe_kinds[cell] = kind
    if spec_cells != detected_cells:
        return {"valid": False, "reason": "visible pipe/spec cell set mismatch"}

    paths = _aqueduct_unique_path(pipe_kinds, source_cell, sink_cell, limit=2)
    if len(paths) != 1:
        return {"valid": False, "reason": "visible pipe topology does not have one unique legal path"}
    path = paths[0]
    required_openings = _aqueduct_required_openings(path)
    if any(
        pipe_kinds[cell] not in {"STRAIGHT", "ELBOW"}
        or _aqueduct_kind_from_openings(required_openings[cell]) != pipe_kinds[cell]
        for cell in path
    ):
        return {"valid": False, "reason": "unique path is incompatible with two-port pipe kinds"}
    visible_misaligned = [
        cell for cell in path if displayed_openings[cell] != required_openings[cell]
    ]
    if len(visible_misaligned) < max(1, (len(path) + 1) // 2):
        return {"valid": False, "reason": "rotation mechanic is not visibly load-bearing"}

    return {
        "valid": True,
        "reason": None,
        "geometry": geometry,
        "stone_body_floor": stone_body_floor,
        "endpoint_occlusions": endpoint_occlusions,
        "source_path": Path(source_path),
        "source_pixels": source_pixels,
        "source_cell": source_cell,
        "sink_cell": sink_cell,
        "pipe_cells": detected_cells,
        "pipe_kinds": pipe_kinds,
        "displayed_openings": displayed_openings,
        "path_cells": path,
        # Model-facing 1-based (x, y): x is the row from the top, y the column
        # from the left, exactly what the rendered gutter numbers name.
        "path_positions": [list(cell_to_xy(cell)) for cell in path],
        "required_openings": required_openings,
        "misaligned_cells": visible_misaligned,
        "decoy_cells": detected_cells - set(path),
    }


def _aqueduct_route_corridor_thickness(geometry: Dict[str, int]) -> int:
    """Width of the source flow channel, in target pixels.

    One definition, used by the expected-route mask and by the flow progress bank
    that has to know how far a wet pixel reaches.  The generator's own
    ``_flow_thickness`` is the same expression.
    """

    return max(4, (int(geometry["cell_size"]) // 3) & ~1)


def _aqueduct_flow_arrival(
    dense_points: Sequence[Tuple[float, float]],
    reach: float,
) -> np.ndarray:
    """First tip index at which each dense route sample reads as wet.

    The flow phase is observed by asking which of the source's dense route
    samples the dilated water covers, and that reading is not linear in arc
    length: near a bend the wet front sits within reach of samples on both legs
    at once, so several samples turn on together.  ``arrival`` is the source's own
    account of when each sample turns on -- a running minimum over the pairwise
    distances -- so the bank built from it carries the same bursts the observation
    does rather than a smoothed version of them.
    """

    points = np.asarray(dense_points, dtype=np.float64)
    count = points.shape[0]
    arrival = np.full(count, count, dtype=np.int64)
    if count == 0:
        return arrival
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    running = np.full(count, np.inf)
    for index in range(count):
        np.minimum(running, distances[:, index], out=running)
        arrival[(arrival == count) & (running <= float(reach))] = index
    return arrival


def _aqueduct_flow_frame_turnon(arrival: np.ndarray, tip_step: int) -> int:
    """Most samples the source can turn on across one frame's advance.

    This is the flow observable's own quantum, read off the source and nothing
    else: no candidate frame and no measured result enters it.  It is the bound
    the shared ``local_pair_change_bounds`` is meant to supply and does not here --
    that helper measures across a window of one pace plus one bank quantum on the
    *collapsed* bank and reports 3 to 4 samples, while the source's arrival
    function turns on 6 across a single frame's advance, which is exactly the
    burst the video shows.  Measured on twelve samples across four seeds, this
    quantity equals the observed maximum entry change every time.
    """

    count = int(arrival.size)
    step = max(1, int(tip_step))
    if count == 0:
        return 1
    return max(
        1,
        max(int(((arrival >= i) & (arrival < i + step)).sum()) for i in range(count)),
    )


def _read_aqueduct_flow_order(
    occupancies: Sequence[np.ndarray],
    arrival: np.ndarray,
    first_flow_frame: Optional[int],
) -> tuple[Dict[str, Any], bool]:
    """Read the flow phase as a bounded-step walk over the source route.

    Replaces a fixed 0.06 backward tolerance and a fixed 0.18 jump ceiling, both
    position-blind: a reversal spread over frames that each advance less than the
    tolerance is invisible however much route it spans, and a deletion is caught
    in a fast phase and missed in a slow one.  Measured before this gate,
    reversals of 6, 12 and 20 frames, a 12-frame hard cut, a 6-frame deletion and
    all three hold-seam variants were accepted at every seed.

    Both bounds are the source's own one-frame turn-on rather than the shared
    helper's window measurement.  This is not a widened window: it
    is a fixed allowance of one frame's legal change, so it does not grow with any
    span and therefore does not weaken deletion detection the way a windowed bound
    would -- a deletion still asks for its own multiple of that same allowance.
    The rotation phase is untouched; its per-cell first-change/first-final
    tracking already rejects a reordering there.
    """

    if first_flow_frame is None or first_flow_frame >= len(occupancies):
        return {
            "video_flow_order_available": False,
            "video_flow_order_reason": "no wet frame to read",
        }, False
    window = [np.asarray(vector, dtype=bool) for vector in occupancies[first_flow_frame:]]
    samples = int(arrival.size)
    if len(window) < 5 or samples < 8:
        return {
            "video_flow_order_available": False,
            "video_flow_order_reason": "flow window too short to read as a sequence",
        }, False

    pace = reference_advance(
        len(window), non_advancing_frames=_AQUEDUCT_FLOW_NON_ADVANCING_FRAMES
    )
    quantum = 1.0 / float(samples - 1)
    tip_step = max(1, int(round(pace * (samples - 1))))
    turnon = _aqueduct_flow_frame_turnon(arrival, tip_step)
    # One frame's legal change, as a share of the route's samples.
    frame_change = float(turnon) / float(samples)

    bank = [
        np.repeat(((arrival <= tip).astype(np.uint8) * 255)[None, :, None], 3, axis=2)
        for tip in range(samples)
    ]
    progress = [float(tip) * quantum for tip in range(samples)]
    frames = [
        np.repeat((vector.astype(np.uint8) * 255)[None, :, None], 3, axis=2)
        for vector in window
    ]
    dynamic = np.ones((1, samples), dtype=bool)
    probe_bank = dynamic_probe(bank, dynamic)
    probe_frames = dynamic_probe(frames, dynamic)
    pair = observed_pair_changes(probe_frames)
    source_changes = reference_pair_changes(
        probe_bank, progress, budget=pace, bank_quantum=quantum
    )
    alignment = align_progress(
        emission_costs(probe_frames, probe_bank),
        progress,
        budget=8.0 * pace + quantum,
        pace=pace,
        pair_changes=pair,
        reference_changes=source_changes,
    )
    # The reading may advance by one frame's worth of the *observable*, which the
    # burst makes larger than one pace; and the picture may change by one frame's
    # worth of the same quantity.  Both carry the shared slacks, unchanged.
    step_allowance = (frame_change + quantum) * _STEP_SLACK
    pace_allowance = frame_change * _PAIR_CHANGE_SLACK
    step_breaks = sum(1 for step in alignment.steps if step > step_allowance)
    worst_step = max((step / step_allowance for step in alignment.steps), default=0.0)
    pace_breaks = sum(1 for change in pair if change > pace_allowance)
    worst_pace = max((change / pace_allowance for change in pair), default=0.0)
    # Binary occupancy can flicker at the antialiased water front without any
    # reversal or jump in the recovered route progress. Judge that recovered
    # progress; pixel-count speed is a diagnostic, not the public flow rule.
    ordered = bool(step_breaks == 0)
    metrics = {
        "video_flow_order_available": True,
        "video_flow_order_verifier": "bounded_step_source_route_flow_alignment_v2",
        "video_flow_window_frames": len(window),
        "video_flow_bank_states": samples,
        "video_flow_tip_step_samples": tip_step,
        "video_flow_source_frame_turnon_samples": turnon,
        "video_flow_reading_reference_pace": round(float(pace), 6),
        "video_flow_reading_max_step": round(float(alignment.max_step), 6),
        "video_flow_reading_step_ratio": round(float(worst_step), 6),
        "video_flow_reading_step_violations": int(step_breaks),
        "video_flow_pace_violations": int(pace_breaks),
        "video_flow_worst_pace_ratio": round(float(worst_pace), 6),
        "video_flow_bounded_step_order_pass": ordered,
    }
    return metrics, ordered


def _aqueduct_expected_route_mask(replay: Dict[str, Any]) -> np.ndarray:
    geometry = replay["geometry"]
    points: List[Tuple[float, float]] = []
    for cell in replay["path_cells"]:
        left, top, right, bottom = _aqueduct_cell_bounds(geometry, cell)
        points.append(((left + right) / 2.0, (top + bottom) / 2.0))
    canvas = Image.new("RGB", (geometry["width"], geometry["height"]), (0, 0, 0))
    if len(points) >= 2:
        thickness = _aqueduct_route_corridor_thickness(geometry)
        draw_path_line(
            canvas,
            points,
            (255, 255, 255),
            thickness,
            joint_style="square",
            tip_marker=False,
        )
    return np.asarray(canvas)[:, :, 0] > 0


def _aqueduct_final_frame_semantics(
    replay: Dict[str, Any],
    candidate_pixels: np.ndarray,
) -> Dict[str, Any]:
    geometry = replay["geometry"]
    source_pixels = replay["source_pixels"]
    dimensions_ok = candidate_pixels.shape[:2] == (geometry["height"], geometry["width"])
    if not dimensions_ok:
        return {
            "dimensions_ok": False,
            "artifact_valid": False,
            "final_state_ok": False,
        }

    candidate_cells = _aqueduct_detect_pipe_cells(
        candidate_pixels, geometry, stone_body_floor=replay["stone_body_floor"]
    )
    pipe_cell_set_ok = candidate_cells == replay["pipe_cells"]
    path_openings_ok = True
    decoy_openings_ok = True
    kind_identity_ok = True
    opening_details: Dict[str, List[str]] = {}
    for cell in sorted(replay["pipe_cells"]):
        visible = _aqueduct_detect_openings(candidate_pixels, geometry, cell)
        expected = (
            replay["required_openings"][cell]
            if cell in replay["required_openings"]
            else replay["displayed_openings"][cell]
        )
        occluded = _aqueduct_occluded_bands(
            cell,
            source_cell=replay["source_cell"],
            sink_cell=replay["sink_cell"],
            occlusions=replay["endpoint_occlusions"],
        )
        openings_match = bool(
            visible.issubset(expected)
            and (expected - occluded).issubset(visible)
        )
        if cell in replay["required_openings"]:
            path_openings_ok = path_openings_ok and openings_match
        else:
            decoy_openings_ok = decoy_openings_ok and openings_match
        inferred_kind = _aqueduct_kind_from_openings(expected)
        kind_identity_ok = kind_identity_ok and inferred_kind == replay["pipe_kinds"][cell]
        opening_details[str(tuple(cell_to_xy(cell)))] = sorted(visible)

    source_left, source_top, source_right, source_bottom = _aqueduct_cell_bounds(
        geometry, replay["source_cell"]
    )
    source_crop = candidate_pixels[source_top:source_bottom, source_left:source_right]
    source_marker_ok = int(
        _aqueduct_color_mask(source_crop, _AQUEDUCT_COLORS["source"], tolerance=22).sum()
    ) >= 20

    sink_left, sink_top, sink_right, sink_bottom = _aqueduct_cell_bounds(
        geometry, replay["sink_cell"]
    )
    sink_crop = candidate_pixels[sink_top:sink_bottom, sink_left:sink_right]
    fountain_spray_region = sink_crop[: max(1, int(geometry["cell_size"] * 0.30)), :]
    fountain_active = int(
        _aqueduct_color_mask(fountain_spray_region, _AQUEDUCT_COLORS["spray"], tolerance=30).sum()
    ) >= 12

    flow_mask = _aqueduct_flow_mask(candidate_pixels)
    route_mask = _aqueduct_expected_route_mask(replay)
    try:
        import cv2

        kernel = np.ones((9, 9), dtype=np.uint8)
        route_corridor = cv2.dilate(route_mask.astype(np.uint8), kernel) > 0
        flow_dilated = cv2.dilate(flow_mask.astype(np.uint8), kernel) > 0
    except Exception:  # pragma: no cover - OpenCV is part of the evaluator runtime
        route_corridor = route_mask
        flow_dilated = flow_mask
    flow_for_offroute = flow_mask.copy()
    flow_for_offroute[sink_top:sink_bottom, sink_left:sink_right] = False
    flow_count = int(flow_for_offroute.sum())
    flow_on_route_fraction = float(
        (flow_for_offroute & route_corridor).sum() / max(1, flow_count)
    )
    route_coverage = float((route_mask & flow_dilated).sum() / max(1, int(route_mask.sum())))
    route_cell_coverages: List[float] = []
    for cell in replay["path_cells"]:
        left, top, right, bottom = _aqueduct_cell_bounds(geometry, cell)
        expected_cell = route_mask[top:bottom, left:right]
        covered_cell = flow_dilated[top:bottom, left:right]
        route_cell_coverages.append(
            float((expected_cell & covered_cell).sum() / max(1, int(expected_cell.sum())))
        )
    minimum_route_cell_coverage = min(route_cell_coverages, default=0.0)
    flow_path_ok = bool(
        flow_count >= max(24, geometry["cell_size"])
        and flow_on_route_fraction >= 0.96
        and route_coverage >= 0.84
        and minimum_route_cell_coverage >= 0.72
    )

    static_mask = np.ones((geometry["height"], geometry["width"]), dtype=bool)
    for cell in replay["path_cells"]:
        left, top, right, bottom = _aqueduct_cell_bounds(geometry, cell)
        static_mask[top:bottom, left:right] = False
    # The verifier must accept style variance that does not change
    # semantics, and a per-channel constant is the whole of that freedom: it is
    # the same at every pixel and therefore says nothing about where anything is.
    # Both measurements below are absolute-colour tests, so without removing it a
    # uniformly brightened delivery fails them while carrying the right answer --
    # measured, a legal +2/255 shift wrote 34 pixels into the label colour that
    # the source does not have, against a budget of 8, and +3/255 left the static
    # mean delta 0.22 short of its own bound.  The shared estimator is a
    # per-channel median over the region every legal delivery reproduces exactly,
    # so a locally wrong object moves it by a fraction of its own area and is not
    # absorbed; an untoned candidate measures 0 and nothing changes for it.
    toned_candidate = np.clip(
        candidate_pixels.astype(np.float32)
        - tone_offset(candidate_pixels, source_pixels, static_mask),
        0.0,
        255.0,
    ).astype(np.uint8)
    delta = np.max(
        np.abs(toned_candidate.astype(np.int16) - source_pixels.astype(np.int16)),
        axis=2,
    )
    static_values = delta[static_mask]
    static_mutation_fraction = float((static_values > 28).mean()) if static_values.size else 1.0
    static_mean_delta = float(static_values.mean()) if static_values.size else float("inf")
    # Both statistics above are frame-wide, and 11.8 forbids exactly that shape:
    # a 16x80 block written into the gutter is 1280 pixels of a static mask of
    # roughly 220000, which measures 0.0057 against a bound of 0.006 and was
    # accepted.  The same statistic per 32x32 region is not compensable, and it is
    # the bound the video branch of this route already applies.
    static_region_error, static_region_violations = (
        static_region_errors(toned_candidate, source_pixels, static_mask)
        if bool(np.any(static_mask))
        else (1.0, 1)
    )
    fixed_content_preserved = bool(
        np.any(static_mask)
        and static_mutation_fraction <= 0.006
        and static_mean_delta <= 2.5
        and static_region_error <= STATIC_REGION_FLOOR
    )

    # The gutter row/column numbers are the scene's only glyphs and must survive
    # unchanged; nothing new may be written in that light-blue label colour.
    candidate_axis_labels = _aqueduct_color_mask(
        toned_candidate, _AQUEDUCT_COLORS["axis_label"], tolerance=26
    )
    source_axis_labels = _aqueduct_color_mask(
        source_pixels, _AQUEDUCT_COLORS["axis_label"], tolerance=26
    )
    source_label_count = int(source_axis_labels.sum())
    label_overlap = int((candidate_axis_labels & source_axis_labels).sum())
    label_extras = int((candidate_axis_labels & ~source_axis_labels).sum())
    label_preservation_fraction = float(label_overlap / max(1, source_label_count))
    labels_preserved = bool(
        source_label_count >= 2 * geometry["grid_size"]
        and label_preservation_fraction >= 0.94
        and label_extras <= max(8, int(0.04 * source_label_count))
    )

    artifact_valid = bool(
        dimensions_ok
        and pipe_cell_set_ok
        and kind_identity_ok
        and source_marker_ok
        and fixed_content_preserved
        and labels_preserved
    )
    final_state_ok = bool(
        path_openings_ok
        and decoy_openings_ok
        and flow_path_ok
        and fountain_active
    )
    return {
        "dimensions_ok": dimensions_ok,
        "pipe_cell_set_ok": pipe_cell_set_ok,
        "kind_identity_ok": kind_identity_ok,
        "path_openings_ok": path_openings_ok,
        "decoy_openings_ok": decoy_openings_ok,
        "source_marker_ok": source_marker_ok,
        "fountain_active": fountain_active,
        "flow_path_ok": flow_path_ok,
        "flow_on_route_fraction": flow_on_route_fraction,
        "route_coverage": route_coverage,
        "minimum_route_cell_coverage": minimum_route_cell_coverage,
        "fixed_content_preserved": fixed_content_preserved,
        "static_mutation_fraction": static_mutation_fraction,
        "max_static_region_error": round(float(static_region_error), 6),
        "static_region_violation_regions": int(static_region_violations),
        "static_region_bound": STATIC_REGION_FLOOR,
        "static_mean_delta": static_mean_delta,
        "labels_preserved": labels_preserved,
        "label_preservation_fraction": label_preservation_fraction,
        "label_extra_pixel_count": label_extras,
        "candidate_openings_by_position": opening_details,
        "artifact_valid": artifact_valid,
        "final_state_ok": final_state_ok,
    }


def _aqueduct_openings_match(
    visible: frozenset[str],
    expected: frozenset[str],
    *,
    cell: Tuple[int, int],
    replay: Dict[str, Any],
) -> bool:
    occluded = _aqueduct_occluded_bands(
        cell,
        source_cell=replay["source_cell"],
        sink_cell=replay["sink_cell"],
        occlusions=replay["endpoint_occlusions"],
    )
    return bool(visible.issubset(expected) and (expected - occluded).issubset(visible))


def _aqueduct_dense_route_points(
    replay: Dict[str, Any],
    *,
    spacing: float = 3.0,
) -> List[Tuple[float, float]]:
    geometry = replay["geometry"]
    vertices: List[Tuple[float, float]] = []
    for cell in replay["path_cells"]:
        left, top, right, bottom = _aqueduct_cell_bounds(geometry, cell)
        vertices.append(((left + right) / 2.0, (top + bottom) / 2.0))
    points: List[Tuple[float, float]] = []
    for segment_index, (left, right) in enumerate(zip(vertices, vertices[1:])):
        length = float(np.hypot(right[0] - left[0], right[1] - left[1]))
        count = max(1, int(np.ceil(length / max(1.0, float(spacing)))))
        start = 0 if segment_index == 0 else 1
        for step in range(start, count + 1):
            ratio = step / count
            points.append(
                (
                    left[0] + (right[0] - left[0]) * ratio,
                    left[1] + (right[1] - left[1]) * ratio,
                )
            )
    return points


def _verify_aqueduct_candidate_video(
    sample: CanonicalSample,
    candidate_video: Path,
    metadata_record: Dict[str, Any],
    evaluator: object,
) -> tuple[Dict[str, Any], bool]:
    started_at = time.perf_counter()
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for aqueduct video evaluation") from exc

    replay = _aqueduct_visible_replay(metadata_record, evaluator, source_key="image")
    base_metrics: Dict[str, Any] = {
        "video_process_verifier": MAZE_SINGLE_DECODE_VIDEO_PROFILES["maze_aqueduct_pipes"],
        "video_process_available": bool(replay.get("valid")),
        "video_visible_source_replay_pass": bool(replay.get("valid")),
        "video_visible_source_replay_reason": replay.get("reason"),
        "video_replayed_path_positions": replay.get("path_positions") or [],
        "video_ignored_saved_solution_path": True,
        "video_ignored_saved_corrected_rotations": True,
        "video_ignored_saved_process_timing": True,
        "candidate_video": candidate_video.as_posix(),
    }
    if not replay.get("valid"):
        failures = [str(replay.get("reason") or "visible source replay failed")]
        scores = {
            "media_validity_score": 0.0,
            "temporal_process_score": 0.0,
            "final_state_score": 0.0,
        }
        return {
            **base_metrics,
            **scores,
            "video_semantic_process_check_pass": False,
            **_verifier_contract_metrics(
                verifier_profile=_AQUEDUCT_VERIFIER_PROFILE,
                artifact_kind="video",
                component_scores=scores,
                binary_correct=False,
                failure_reason="; ".join(failures),
                started_at=started_at,
                hard_gate_failures=failures,
            ),
        }, False

    geometry = replay["geometry"]
    height, width = geometry["height"], geometry["width"]
    dense_points = _aqueduct_dense_route_points(replay)
    dense_x = np.clip(np.rint([point[0] for point in dense_points]).astype(int), 0, width - 1)
    dense_y = np.clip(np.rint([point[1] for point in dense_points]).astype(int), 0, height - 1)

    static_mask = np.ones((height, width), dtype=bool)
    for cell in replay["path_cells"]:
        left, top, right, bottom = _aqueduct_cell_bounds(geometry, cell)
        static_mask[top:bottom, left:right] = False

    rotation_cells = list(replay["misaligned_cells"])
    saw_intermediate = [False] * len(rotation_cells)
    first_change_frames: Dict[Tuple[int, int], int] = {}
    first_final_frames: Dict[Tuple[int, int], int] = {}
    rotation_sync_violation_count = 0
    simultaneous_rotation_frame_count = 0
    decoy_mutation_frame_count = 0
    path_state_mutation_frame_count = 0
    static_mutation_fractions: List[float] = []
    static_region_maxima: List[float] = []
    initial_distances: List[float] = []
    flow_progresses: List[Optional[float]] = []
    flow_occupancies: List[np.ndarray] = []
    flow_off_route_fractions: List[float] = []
    flow_prefix_coverages: List[float] = []
    flow_before_rotation_count = 0
    early_fountain_count = 0
    fountain_active_flags: List[bool] = []
    all_rotations_complete_flags: List[bool] = []
    all_path_final_flags: List[bool] = []
    first_flow_frame: Optional[int] = None
    rotation_complete_frame: Optional[int] = None
    last_frame_rgb: Optional[np.ndarray] = None

    cap = cv2.VideoCapture(str(candidate_video))
    if not cap.isOpened():
        failures = ["candidate video could not be opened"]
        scores = {
            "media_validity_score": 0.0,
            "temporal_process_score": 0.0,
            "final_state_score": 0.0,
        }
        return {
            **base_metrics,
            **scores,
            "video_semantic_process_check_pass": False,
            **_verifier_contract_metrics(
                verifier_profile=_AQUEDUCT_VERIFIER_PROFILE,
                artifact_kind="video",
                component_scores=scores,
                binary_correct=False,
                failure_reason="; ".join(failures),
                started_at=started_at,
                hard_gate_failures=failures,
            ),
        }, False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
    reported_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
    frame_index = 0
    dilation_kernel = np.ones(
        (_AQUEDUCT_FLOW_DILATION_SIZE, _AQUEDUCT_FLOW_DILATION_SIZE), dtype=np.uint8
    )
    # How far a wet pixel reaches into the dense route samples: the flow channel's
    # own half-width plus the radius of the dilation the reading applies.  Both
    # come from the source geometry and this verifier's own structuring element.
    flow_arrival = _aqueduct_flow_arrival(
        dense_points,
        _aqueduct_route_corridor_thickness(geometry) / 2.0
        + _AQUEDUCT_FLOW_DILATION_SIZE // 2,
    )
    route_mask = _aqueduct_expected_route_mask(replay)
    route_corridor = cv2.dilate(route_mask.astype(np.uint8), dilation_kernel) > 0
    sink_left, sink_top, sink_right, sink_bottom = _aqueduct_cell_bounds(
        geometry, replay["sink_cell"]
    )

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[:2] != (height, width):
            last_frame_rgb = frame_rgb
            frame_index += 1
            continue
        last_frame_rgb = frame_rgb
        full_delta = np.max(
            np.abs(frame_rgb.astype(np.int16) - replay["source_pixels"].astype(np.int16)),
            axis=2,
        )
        initial_distances.append(float(full_delta.mean()))
        static_values = full_delta[static_mask]
        static_mutation_fractions.append(float((static_values > 42).mean()))
        # The fraction over the whole static mask dilutes a painted patch to
        # 0.2%; the same statistic per 32x32 region does not (11.8).
        static_region_maxima.append(
            static_region_errors(frame_rgb, replay["source_pixels"], static_mask, raster_tolerance=1)[0]
        )

        statuses: Dict[Tuple[int, int], str] = {}
        intermediate_cells: List[Tuple[int, int]] = []
        decoys_ok = True
        all_path_final = True
        for cell in sorted(replay["pipe_cells"]):
            visible = _aqueduct_detect_openings(frame_rgb, geometry, cell)
            displayed = replay["displayed_openings"][cell]
            final = replay["required_openings"].get(cell, displayed)
            source_match = _aqueduct_openings_match(
                visible, displayed, cell=cell, replay=replay
            )
            final_match = _aqueduct_openings_match(
                visible, final, cell=cell, replay=replay
            )
            if source_match and final_match:
                status = "same"
            elif source_match:
                status = "source"
            elif final_match:
                status = "final"
            else:
                status = "intermediate"
                if cell in rotation_cells:
                    intermediate_cells.append(cell)
            statuses[cell] = status
            if cell in replay["decoy_cells"] and not source_match:
                decoys_ok = False
            if cell in replay["required_openings"] and not final_match:
                all_path_final = False
        if not decoys_ok:
            decoy_mutation_frame_count += 1
        if len(intermediate_cells) > 1:
            simultaneous_rotation_frame_count += 1

        for cell_index, cell in enumerate(rotation_cells):
            status = statuses.get(cell)
            if status in {"intermediate", "final"}:
                first_change_frames.setdefault(cell, frame_index)
            if status == "intermediate":
                saw_intermediate[cell_index] = True
            elif status == "final":
                first_final_frames.setdefault(cell, frame_index)
            elif status == "source" and cell in first_change_frames:
                rotation_sync_violation_count += 1

        if all_path_final and rotation_complete_frame is None and rotation_cells:
            rotation_complete_frame = frame_index
        # A completed tile while another is still unchanged is precisely the
        # old arbitrary source-to-fountain enumeration.  Independent
        # alignment constraints must share one global transition.
        if any(statuses.get(cell) == "final" for cell in rotation_cells) and any(
            statuses.get(cell) == "source" for cell in rotation_cells
        ):
            rotation_sync_violation_count += 1
        if rotation_complete_frame is not None and any(
            statuses.get(cell) == "source" for cell in rotation_cells
        ):
            path_state_mutation_frame_count += 1

        all_rotations_complete = bool(
            rotation_complete_frame is not None and path_state_mutation_frame_count == 0
        )
        all_rotations_complete_flags.append(all_rotations_complete)
        all_path_final_flags.append(all_path_final)

        flow = _aqueduct_flow_mask(frame_rgb)
        flow_no_sink = flow.copy()
        flow_no_sink[sink_top:sink_bottom, sink_left:sink_right] = False
        flow_count = int(flow_no_sink.sum())
        progress: Optional[float] = 0.0
        prefix_coverage = 1.0
        off_route_fraction = 0.0
        occupied = np.zeros(len(dense_points), dtype=bool)
        if flow_count >= max(24, 2 * geometry["cell_size"]) and len(dense_points) >= 2:
            if first_flow_frame is None:
                first_flow_frame = frame_index
            flow_dilated = cv2.dilate(flow.astype(np.uint8), dilation_kernel) > 0
            occupied = flow_dilated[dense_y, dense_x]
            occupied_indices = np.flatnonzero(occupied)
            if occupied_indices.size:
                furthest = int(occupied_indices[-1])
                prefix_coverage = float(occupied[: furthest + 1].mean())
                progress = (
                    float(furthest / max(1, len(dense_points) - 1))
                    if prefix_coverage >= 0.62
                    else None
                )
            else:
                progress = None
                prefix_coverage = 0.0
            off_route_fraction = float(
                (flow_no_sink & ~route_corridor).sum() / max(1, flow_count)
            )
            if not all_rotations_complete:
                flow_before_rotation_count += 1
        flow_progresses.append(progress)
        flow_prefix_coverages.append(prefix_coverage)
        flow_off_route_fractions.append(off_route_fraction)
        flow_occupancies.append(np.asarray(occupied, dtype=bool))

        sink_crop = frame_rgb[sink_top:sink_bottom, sink_left:sink_right]
        fountain_spray_region = sink_crop[: max(1, int(geometry["cell_size"] * 0.30)), :]
        fountain_active = int(
            _aqueduct_color_mask(fountain_spray_region, _AQUEDUCT_COLORS["spray"], tolerance=38).sum()
        ) >= 12
        fountain_active_flags.append(fountain_active)
        if fountain_active and (progress is None or progress < 0.94):
            early_fountain_count += 1
        frame_index += 1
    cap.release()

    frame_count = frame_index
    media_contract_ok = bool(
        frame_count > 0
        and reported_width == width
        and reported_height == height
        and abs(fps - 16.0) <= 0.05
        and frame_count <= _AQUEDUCT_VIDEO_MAX_FRAMES
        and frame_count % 4 == 1
    )
    initial_hold_count = 0
    for distance in initial_distances:
        if distance <= 8.0:
            initial_hold_count += 1
        else:
            break
    start_frames = list(first_change_frames.values())
    completion_frames = list(first_final_frames.values())
    start_spread = max(start_frames, default=0) - min(start_frames, default=0)
    completion_spread = max(completion_frames, default=0) - min(completion_frames, default=0)
    rotations_complete = bool(
        rotation_cells
        and len(first_change_frames) == len(rotation_cells)
        and len(first_final_frames) == len(rotation_cells)
        and all(saw_intermediate)
        and start_spread <= 2
        and completion_spread <= 2
        and rotation_sync_violation_count == 0
        and (len(rotation_cells) == 1 or simultaneous_rotation_frame_count >= 1)
        and decoy_mutation_frame_count == 0
        and path_state_mutation_frame_count == 0
    )
    corrected_dry_hold_count = 0
    if rotation_complete_frame is not None:
        dry_end = first_flow_frame if first_flow_frame is not None else frame_count
        corrected_dry_hold_count = max(0, dry_end - rotation_complete_frame - 1)

    recognized_progress = [value for value in flow_progresses if value is not None]
    positive_progress = [value for value in recognized_progress if value > 0.0]
    flow_order_metrics, flow_order_ok = _read_aqueduct_flow_order(
        flow_occupancies, flow_arrival, first_flow_frame
    )
    flow_reaches_final = bool(positive_progress and positive_progress[-1] >= 0.96)
    flow_on_route = bool(
        max(flow_off_route_fractions, default=1.0) <= 0.04
        and min(
            [coverage for coverage, progress in zip(flow_prefix_coverages, flow_progresses) if progress and progress > 0.1]
            or [0.0]
        ) >= 0.62
    )
    final_hold_count = 0
    for index in range(frame_count - 1, -1, -1):
        progress = flow_progresses[index]
        if (
            progress is not None
            and progress >= 0.96
            and fountain_active_flags[index]
            and all_rotations_complete_flags[index]
            and all_path_final_flags[index]
        ):
            final_hold_count += 1
        else:
            break
    static_preserved = bool(
        static_mutation_fractions
        and max(static_mutation_fractions) <= 0.012
        and max(static_region_maxima, default=0.0) <= STATIC_REGION_FLOOR
    )
    temporal_process_ok = bool(
        initial_hold_count >= 3
        and rotations_complete
        and corrected_dry_hold_count >= 2
        and flow_before_rotation_count == 0
        and flow_order_ok
        and flow_reaches_final
        and flow_on_route
        and early_fountain_count == 0
        and final_hold_count >= 4
        and static_preserved
    )

    final_semantics = (
        _aqueduct_final_frame_semantics(replay, last_frame_rgb)
        if last_frame_rgb is not None and last_frame_rgb.shape[:2] == (height, width)
        else {"artifact_valid": False, "final_state_ok": False}
    )
    final_state_ok = bool(final_semantics.get("final_state_ok"))
    failures: List[str] = []
    if not media_contract_ok:
        failures.append("video media contract failed")
    if not temporal_process_ok:
        failures.append("synchronized rotation/causal-flow temporal process failed")
    if not final_state_ok:
        failures.append("final aqueduct state failed")
    passed = not failures

    process_parts = [
        initial_hold_count >= 3,
        rotations_complete,
        corrected_dry_hold_count >= 2,
        flow_before_rotation_count == 0,
        flow_order_ok,
        flow_reaches_final,
        flow_on_route,
        early_fountain_count == 0,
        final_hold_count >= 4,
        static_preserved,
    ]
    scores = {
        "media_validity_score": float(media_contract_ok),
        "temporal_process_score": float(sum(bool(value) for value in process_parts) / len(process_parts)),
        "final_state_score": float(final_state_ok),
    }
    metrics: Dict[str, Any] = {
        **base_metrics,
        **flow_order_metrics,
        "video_fps": fps,
        "video_frame_count": frame_count,
        "video_width": reported_width,
        "video_height": reported_height,
        "video_media_contract_pass": media_contract_ok,
        "video_initial_hold_frame_count": initial_hold_count,
        "video_expected_rotation_positions": [
            list(cell_to_xy(cell)) for cell in rotation_cells
        ],
        "video_completed_rotation_positions": [
            list(cell_to_xy(cell)) for cell in rotation_cells if cell in first_final_frames
        ],
        "video_rotation_completion_frames": [
            first_final_frames[cell] for cell in rotation_cells if cell in first_final_frames
        ],
        "video_rotation_start_frame_spread": start_spread,
        "video_rotation_completion_frame_spread": completion_spread,
        "video_all_rotations_have_intermediate_motion": all(saw_intermediate),
        "video_rotation_sync_violation_count": rotation_sync_violation_count,
        "video_simultaneous_rotation_frame_count": simultaneous_rotation_frame_count,
        "video_decoy_mutation_frame_count": decoy_mutation_frame_count,
        "video_path_state_mutation_frame_count": path_state_mutation_frame_count,
        "video_corrected_dry_hold_frame_count": corrected_dry_hold_count,
        "video_first_flow_frame": first_flow_frame,
        "video_flow_before_rotation_frame_count": flow_before_rotation_count,
        "video_flow_reaches_final": flow_reaches_final,
        "video_flow_on_route": flow_on_route,
        "video_max_off_route_flow_fraction": max(flow_off_route_fractions, default=1.0),
        "video_early_fountain_frame_count": early_fountain_count,
        "video_final_hold_frame_count": final_hold_count,
        "video_static_content_preserved": static_preserved,
        "video_max_static_mutation_fraction": max(static_mutation_fractions, default=1.0),
        "video_max_static_region_error": max(static_region_maxima, default=0.0),
        "video_static_region_bound": STATIC_REGION_FLOOR,
        "video_static_raster_tolerance_px": 1,
        "video_temporal_process_check_pass": temporal_process_ok,
        "video_final_state_check_pass": final_state_ok,
        "video_semantic_process_check_pass": passed,
        "video_keyframe_process_check_pass": None,
        "video_keyframe_scope": "semantic_full_sequence",
        "video_keyframe_required_for_pass": False,
        "video_decoded_full_sequence_once": True,
        "video_final_semantics": final_semantics,
        **scores,
        **_verifier_contract_metrics(
            verifier_profile=_AQUEDUCT_VERIFIER_PROFILE,
            artifact_kind="video",
            component_scores=scores,
            binary_correct=passed,
            failure_reason="; ".join(failures) if failures else "ok",
            started_at=started_at,
            hard_gate_failures=failures,
        ),
    }
    return metrics, passed


def _record_value_by_cell_id(mapping: Any, cell_id: int) -> Any:
    if not isinstance(mapping, dict):
        return None
    if cell_id in mapping:
        return mapping[cell_id]
    return mapping.get(str(cell_id))


def _enumerate_shortest_state_paths(
    start_state: Any,
    *,
    is_goal: Any,
    neighbors: Any,
    state_cell_id: Any,
    limit: int = 2048,
) -> List[List[int]]:
    """Enumerate all bounded shortest deliveries in an independently replayed state graph."""

    limit = max(1, int(limit))
    distance: Dict[Any, int] = {start_state: 0}
    parents: Dict[Any, List[Any]] = {start_state: []}
    queue: List[Any] = [start_state]
    goal_distance: Optional[int] = None
    goal_states: List[Any] = []
    for state in queue:
        current_distance = distance[state]
        if goal_distance is not None and current_distance > goal_distance:
            break
        if is_goal(state):
            goal_distance = current_distance
            goal_states.append(state)
            continue
        if goal_distance is not None:
            continue
        for next_state in neighbors(state):
            next_distance = current_distance + 1
            known = distance.get(next_state)
            if known is None:
                distance[next_state] = next_distance
                parents[next_state] = [state]
                queue.append(next_state)
            elif known == next_distance and len(parents[next_state]) < limit:
                parents[next_state].append(state)

    results: List[List[int]] = []

    def recover(state: Any, reversed_states: List[Any]) -> None:
        if len(results) >= limit:
            return
        if state == start_state:
            states = [state, *reversed_states]
            results.append([int(state_cell_id(item)) for item in states])
            return
        for parent in parents.get(state, []):
            recover(parent, [state, *reversed_states])
            if len(results) >= limit:
                return

    for goal_state in goal_states:
        recover(goal_state, [])
        if len(results) >= limit:
            break

    unique: List[List[int]] = []
    seen: set[Tuple[int, ...]] = set()
    for path in results:
        key = tuple(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _square_maze_source(
    record: Dict[str, Any],
) -> Optional[Tuple[List[List[int]], Dict[Tuple[int, int], int], Dict[int, Tuple[int, int]], Tuple[int, int], Tuple[int, int]]]:
    raw_grid = record.get("maze_grid")
    raw_size = record.get("grid_size")
    if not isinstance(raw_grid, list) or not raw_grid or not isinstance(raw_size, list) or len(raw_size) != 2:
        return None
    try:
        rows, cols = int(raw_size[0]), int(raw_size[1])
        grid = [[int(value) for value in row] for row in raw_grid]
        start = (int(record["start"][0]), int(record["start"][1]))
        goal = (int(record["goal"][0]), int(record["goal"][1]))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if (
        rows <= 0
        or cols <= 0
        or len(grid) != rows
        or any(len(row) != cols for row in grid)
        or any(value not in {0, 1} for row in grid for value in row)
        or not (0 <= start[0] < rows and 0 <= start[1] < cols)
        or not (0 <= goal[0] < rows and 0 <= goal[1] < cols)
        or start == goal
        or grid[start[0]][start[1]] != 0
        or grid[goal[0]][goal[1]] != 0
    ):
        return None
    cells = [(row, col) for row in range(rows) for col in range(cols) if grid[row][col] == 0]
    cell_to_id = {cell: index + 1 for index, cell in enumerate(cells)}
    id_to_cell = {index + 1: cell for index, cell in enumerate(cells)}
    return grid, cell_to_id, id_to_cell, start, goal


def _hexagon_cell_xy_labels(record: Dict[str, Any]) -> Optional[Dict[int, Tuple[int, int]]]:
    """Rebuild the honeycomb's model-facing 1-based ``(x, y)`` labels.

    The honeycomb is fully determined by its visible radius, so the level and
    column numbers drawn in the gutters are recomputed here instead of being
    read back from any answer-bearing field.  ``y`` is the hexagon column
    counted from the left and ``x`` is the horizontal level of the hexagon
    centre counted from the top in half-column steps.
    """

    try:
        radius = int(record["radius"])
    except (KeyError, TypeError, ValueError):
        return None
    if radius < 2:
        return None
    labels: Dict[int, Tuple[int, int]] = {}
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if max(abs(q), abs(r), abs(-q - r)) > radius:
                continue
            labels[len(labels) + 1] = (2 * r + q + 2 * radius + 1, q + radius + 1)
    return labels or None


def _spatial_navigation_maze_state_contract(
    task_name: str,
    record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build one independent state graph shared by text, image, and video replay."""

    if task_name in {"maze_hexagon", "maze_labyrinth"}:
        raw_edges = record.get("open_edges")
        try:
            start_id = int(record["start_cell_id"])
            goal_id = int(record["goal_cell_id"])
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(raw_edges, list) or start_id <= 0 or goal_id <= 0 or start_id == goal_id:
            return None
        adjacency: Dict[int, set[int]] = {}
        seen_edges: set[frozenset[int]] = set()
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
                return None
            try:
                left, right = int(raw_edge[0]), int(raw_edge[1])
            except (TypeError, ValueError):
                return None
            edge = frozenset((left, right))
            if left <= 0 or right <= 0 or left == right or edge in seen_edges:
                return None
            seen_edges.add(edge)
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
        if start_id not in adjacency or goal_id not in adjacency:
            return None
        contract: Dict[str, Any] = {
            "start_state": start_id,
            "is_goal": lambda state: state == goal_id,
            "neighbors": lambda state: sorted(adjacency.get(state, set())),
            "state_cell_id": lambda state: state,
        }
        if task_name == "maze_hexagon":
            hex_labels = _hexagon_cell_xy_labels(record)
            if hex_labels is None:
                return None
            contract["cell_id_to_xy"] = hex_labels
            contract["xy_to_cell_id"] = {
                position: cell_id for cell_id, position in hex_labels.items()
            }
        return contract

    if task_name == "maze_layered":
        raw_l1 = record.get("maze_grid_l1")
        raw_l2 = record.get("maze_grid_l2")
        raw_size = record.get("grid_size")
        if not isinstance(raw_l1, list) or not isinstance(raw_l2, list) or not isinstance(raw_size, list):
            return None
        try:
            rows, cols = int(raw_size[0]), int(raw_size[1])
            grids = {
                1: [[int(value) for value in row] for row in raw_l1],
                2: [[int(value) for value in row] for row in raw_l2],
            }
            start = (int(record["start"][0]), int(record["start"][1]), int(record["start_layer"]))
            goal = (int(record["goal"][0]), int(record["goal"][1]), int(record["goal_layer"]))
            stairs = {
                (int(cell[0]), int(cell[1]))
                for cell in record.get("staircase_positions") or []
            }
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        if (
            rows <= 0
            or cols <= 0
            or any(len(grid) != rows or any(len(row) != cols for row in grid) for grid in grids.values())
            or any(value not in {0, 1} for grid in grids.values() for row in grid for value in row)
            or start[2] not in grids
            or goal[2] not in grids
            or not (0 <= start[0] < rows and 0 <= start[1] < cols)
            or not (0 <= goal[0] < rows and 0 <= goal[1] < cols)
            or grids[start[2]][start[0]][start[1]] != 0
            or grids[goal[2]][goal[0]][goal[1]] != 0
        ):
            return None
        ordered_cells = [
            (row, col, layer)
            for layer in (1, 2)
            for row in range(rows)
            for col in range(cols)
            if grids[layer][row][col] == 0
        ]
        cell_to_id = {cell: index + 1 for index, cell in enumerate(ordered_cells)}

        def layered_neighbors(state: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
            row, col, layer = state
            found: List[Tuple[int, int, int]] = []
            for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                candidate = (row + delta_row, col + delta_col, layer)
                if candidate in cell_to_id:
                    found.append(candidate)
            other = (row, col, 2 if layer == 1 else 1)
            if (row, col) in stairs and other in cell_to_id:
                found.append(other)
            return sorted(found)

        # The two floors are drawn side by side under one continuous column
        # axis: L1 owns columns 1..cols and L2 continues at cols+1, so the
        # visible layout itself distinguishes the floors and the model-facing
        # position stays a plain 1-based ``(x, y)`` pair.
        layered_labels = {
            cell_id: (cell[0] + 1, cell[1] + 1 + (cell[2] - 1) * cols)
            for cell, cell_id in cell_to_id.items()
        }
        return {
            "start_state": start,
            "is_goal": lambda state: state == goal,
            "neighbors": layered_neighbors,
            "state_cell_id": lambda state: cell_to_id[state],
            "cell_id_to_xy": layered_labels,
            "xy_to_cell_id": {
                position: cell_id for cell_id, position in layered_labels.items()
            },
        }

    source = _square_maze_source(record)
    if source is None:
        return None
    grid, cell_to_id, id_to_cell, start, goal = source
    rows, cols = len(grid), len(grid[0])

    def with_xy(contract: Dict[str, Any]) -> Dict[str, Any]:
        """Attach the task's model-facing 1-based ``(x, y)`` labels."""

        contract["cell_id_to_xy"] = {
            cell_id: cell_to_xy(cell) for cell, cell_id in cell_to_id.items()
        }
        contract["xy_to_cell_id"] = {
            cell_to_xy(cell): cell_id for cell, cell_id in cell_to_id.items()
        }
        return contract

    def orthogonal(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        row, col = cell
        return sorted(
            candidate
            for candidate in (
                (row + 1, col),
                (row - 1, col),
                (row, col + 1),
                (row, col - 1),
            )
            if candidate in cell_to_id
        )

    if task_name == "maze_colored_zone":
        raw_cycle = record.get("color_cycle")
        if not isinstance(raw_cycle, list) or len(raw_cycle) < 2:
            return None
        cycle = [str(value).strip().lower() for value in raw_cycle]
        colors = {
            cell_id: str(_record_value_by_cell_id(record.get("cell_colors"), cell_id) or "").strip().lower()
            for cell_id in id_to_cell
        }
        if any(not value for value in colors.values()) or colors[cell_to_id[start]] != cycle[0]:
            return None
        start_state = (start, 0)

        def color_neighbors(state: Tuple[Tuple[int, int], int]) -> List[Tuple[Tuple[int, int], int]]:
            cell, phase = state
            next_phase = (phase + 1) % len(cycle)
            return [
                (candidate, next_phase)
                for candidate in orthogonal(cell)
                if colors[cell_to_id[candidate]] == cycle[next_phase]
            ]

        return with_xy({
            "start_state": start_state,
            "is_goal": lambda state: state[0] == goal,
            "neighbors": color_neighbors,
            "state_cell_id": lambda state: cell_to_id[state[0]],
        })

    if task_name == "maze_elevation":
        try:
            max_step = int(record.get("max_elevation_step"))
            elevations = {
                cell_id: int(_record_value_by_cell_id(record.get("cell_elevations"), cell_id))
                for cell_id in id_to_cell
            }
        except (TypeError, ValueError):
            return None
        if max_step < 0:
            return None

        def elevation_neighbors(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
            return [
                candidate
                for candidate in orthogonal(cell)
                if abs(elevations[cell_to_id[cell]] - elevations[cell_to_id[candidate]]) <= max_step
            ]

        return with_xy({
            "start_state": start,
            "is_goal": lambda state: state == goal,
            "neighbors": elevation_neighbors,
            "state_cell_id": lambda state: cell_to_id[state],
        })

    if task_name == "maze_toroidal":
        start_state = (start, False)

        def toroidal_neighbors(state: Tuple[Tuple[int, int], bool]) -> List[Tuple[Tuple[int, int], bool]]:
            (row, col), used_wrap = state
            found: List[Tuple[Tuple[int, int], bool]] = []
            for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                raw_row, raw_col = row + delta_row, col + delta_col
                candidate = (raw_row % rows, raw_col % cols)
                if candidate not in cell_to_id:
                    continue
                wrapped = raw_row != candidate[0] or raw_col != candidate[1]
                found.append((candidate, used_wrap or wrapped))
            return sorted(found)

        return with_xy({
            "start_state": start_state,
            "is_goal": lambda state: state[0] == goal and state[1],
            "neighbors": toroidal_neighbors,
            "state_cell_id": lambda state: cell_to_id[state[0]],
        })

    if task_name == "maze_keys_and_locks":
        raw_keys = record.get("keys")
        raw_locks = record.get("locks")
        if not isinstance(raw_keys, list) or not isinstance(raw_locks, list) or not raw_keys or len(raw_keys) != len(raw_locks):
            return None
        colors: List[str] = []
        key_cells: Dict[Tuple[int, int], int] = {}
        gate_cells: Dict[Tuple[int, int], int] = {}
        try:
            lock_by_color = {str(item["color"]): item for item in raw_locks}
            for index, key in enumerate(raw_keys):
                color = str(key["color"])
                lock = lock_by_color[color]
                key_cell = (int(key["cell"][0]), int(key["cell"][1]))
                gate_cell = (int(lock["cell"][0]), int(lock["cell"][1]))
                if color in colors or key_cell not in cell_to_id or gate_cell not in cell_to_id:
                    return None
                colors.append(color)
                key_cells[key_cell] = index
                gate_cells[gate_cell] = index
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        all_bits = (1 << len(colors)) - 1
        initial_keys = 1 << key_cells[start] if start in key_cells else 0
        start_state = (start, initial_keys, 0)

        def key_neighbors(state: Tuple[Tuple[int, int], int, int]) -> List[Tuple[Tuple[int, int], int, int]]:
            cell, held_keys, opened_gates = state
            found: List[Tuple[Tuple[int, int], int, int]] = []
            for candidate in orthogonal(cell):
                gate_index = gate_cells.get(candidate)
                if gate_index is not None and not (held_keys & (1 << gate_index)):
                    continue
                next_keys = held_keys
                next_gates = opened_gates
                key_index = key_cells.get(candidate)
                if key_index is not None:
                    next_keys |= 1 << key_index
                if gate_index is not None:
                    next_gates |= 1 << gate_index
                found.append((candidate, next_keys, next_gates))
            return sorted(found)

        return with_xy({
            "start_state": start_state,
            "is_goal": lambda state: state[0] == goal and state[1] == all_bits and state[2] == all_bits,
            "neighbors": key_neighbors,
            "state_cell_id": lambda state: cell_to_id[state[0]],
        })

    if task_name == "maze_square":
        return with_xy({
            "start_state": start,
            "is_goal": lambda state: state == goal,
            "neighbors": orthogonal,
            "state_cell_id": lambda state: cell_to_id[state],
        })
    return None


def _spatial_navigation_maze_delivery_is_legal(
    task_name: str,
    record: Dict[str, Any],
    path_ids: Sequence[int],
) -> tuple[bool, bool]:
    """Replay any complete legal route, including same-cell wait steps."""

    contract = _spatial_navigation_maze_state_contract(task_name, record)
    if contract is None:
        return False, False
    source_routes = _enumerate_shortest_state_paths(
        contract["start_state"],
        is_goal=contract["is_goal"],
        neighbors=contract["neighbors"],
        state_cell_id=contract["state_cell_id"],
        limit=1,
    )
    source_ok = bool(source_routes)
    if not source_ok:
        return False, False
    try:
        delivered = [int(value) for value in path_ids]
    except (TypeError, ValueError):
        return False, True
    if len(delivered) < 2:
        return False, True
    start_state = contract["start_state"]
    if int(contract["state_cell_id"](start_state)) != delivered[0]:
        return False, True
    possible_states = {start_state}
    for next_id in delivered[1:]:
        next_possible_states = {
            state
            for state in possible_states
            if int(contract["state_cell_id"](state)) == next_id
        }
        next_possible_states.update(
            next_state
            for state in possible_states
            for next_state in contract["neighbors"](state)
            if int(contract["state_cell_id"](next_state)) == next_id
        )
        possible_states = next_possible_states
        if not possible_states:
            return False, True
    return any(contract["is_goal"](state) for state in possible_states), True


def _square_path_cells(metadata_record: Dict[str, Any]) -> List[Tuple[int, int]]:
    maze_grid = metadata_record.get("maze_grid")
    if not isinstance(maze_grid, list):
        return []
    return [
        (row_index, col_index)
        for row_index, row in enumerate(maze_grid)
        if isinstance(row, list)
        for col_index, value in enumerate(row)
        if value == 0
    ]


def _square_path_points(
    metadata_record: Dict[str, Any],
    path_ids: Sequence[int],
) -> List[Tuple[float, float]]:
    path_cells = _square_path_cells(metadata_record)
    cell_bboxes = metadata_record.get("cell_bboxes")
    if not path_cells or not isinstance(cell_bboxes, list):
        return []
    points: List[Tuple[float, float]] = []
    for raw_cell_id in path_ids:
        cell_id = int(raw_cell_id)
        if not 1 <= cell_id <= len(path_cells):
            return []
        row, col = path_cells[cell_id - 1]
        try:
            x1, y1, x2, y2 = cell_bboxes[row][col]
            points.append(((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0))
        except Exception:
            return []
    return points


def _toroidal_route_segments(
    metadata_record: Dict[str, Any],
    path_ids: Sequence[int],
) -> List[List[Tuple[float, float]]]:
    path_cells = _square_path_cells(metadata_record)
    cell_bboxes = metadata_record.get("cell_bboxes")
    grid_size = metadata_record.get("grid_size")
    if (
        not path_cells
        or not isinstance(cell_bboxes, list)
        or not isinstance(grid_size, (list, tuple))
        or len(grid_size) < 2
    ):
        return []
    try:
        rows, cols = int(grid_size[0]), int(grid_size[1])
        cell_indices = [int(cell_id) - 1 for cell_id in path_ids]
        if any(index < 0 or index >= len(path_cells) for index in cell_indices):
            return []
        cells = [path_cells[index] for index in cell_indices]
        centers = _square_path_points(metadata_record, path_ids)
        left_x = float(cell_bboxes[0][0][0])
        right_x = float(cell_bboxes[0][cols - 1][2]) - 1.0
        top_y = float(cell_bboxes[0][0][1])
        bottom_y = float(cell_bboxes[rows - 1][0][3]) - 1.0
    except Exception:
        return []
    if len(cells) != len(centers) or not cells:
        return []

    segments: List[List[Tuple[float, float]]] = [[centers[0]]]
    for previous, current, previous_center, current_center in zip(
        cells,
        cells[1:],
        centers,
        centers[1:],
    ):
        if previous[0] == current[0] and abs(previous[1] - current[1]) == cols - 1:
            if previous[1] == 0:
                exit_point = (left_x, previous_center[1])
                entry_point = (right_x, current_center[1])
            else:
                exit_point = (right_x, previous_center[1])
                entry_point = (left_x, current_center[1])
            segments[-1].append(exit_point)
            segments.append([entry_point, current_center])
        elif previous[1] == current[1] and abs(previous[0] - current[0]) == rows - 1:
            if previous[0] == 0:
                exit_point = (previous_center[0], top_y)
                entry_point = (current_center[0], bottom_y)
            else:
                exit_point = (previous_center[0], bottom_y)
                entry_point = (current_center[0], top_y)
            segments[-1].append(exit_point)
            segments.append([entry_point, current_center])
        else:
            segments[-1].append(current_center)
    return segments


def _labyrinth_path_points(generator: object, path_ids: Sequence[int]) -> List[Tuple[float, float]]:
    try:
        segments = int(generator.segments)
        cells = [
            (0, 0)
            if int(cell_id) == 1
            else ((int(cell_id) - 2) // segments + 1, (int(cell_id) - 2) % segments)
            for cell_id in path_ids
        ]
        return list(generator._build_path_points(cells))
    except Exception:
        return []


def _maze_route_geometry(
    task_name: str,
    metadata_record: Dict[str, Any],
    evaluator: object,
    path_ids: Sequence[int],
) -> tuple[List[Tuple[float, float]], int, str]:
    generator = evaluator._build_generator(metadata_record)
    if task_name in _SQUARE_ROUTE_GEOMETRY_TASKS:
        return (
            _square_path_points(metadata_record, path_ids),
            max(3, int(metadata_record["cell_size"]) // 3),
            "square",
        )
    if task_name == "maze_labyrinth":
        # The polar route is stroked through the shared family route renderer
        # (oriented segment quads + round joints), so the expected mask has to
        # be rebuilt with that same style rather than a second hand-rolled one.
        return (
            _labyrinth_path_points(generator, path_ids),
            max(3, int(metadata_record["ring_width"]) // 5),
            "round",
        )
    if task_name == "maze_hexagon":
        return (
            [generator._cell_center(int(cell_id)) for cell_id in path_ids],
            max(4, int(float(metadata_record["cell_radius"]) * 0.35)),
            "round",
        )
    return [], 0, "round"


def _layered_route_segments(
    metadata_record: Dict[str, Any],
    evaluator: object,
    path_ids: Sequence[int],
) -> List[List[Tuple[float, float]]]:
    try:
        generator = evaluator._build_generator(metadata_record)
        cells = [generator._id_to_cell[int(cell_id)] for cell_id in path_ids]
        staircase_positions = {
            (int(cell[0]), int(cell[1]))
            for cell in metadata_record.get("staircase_positions") or []
        }
        raw_points = generator._solution_path_to_pixels(cells, staircase_positions)
    except Exception:
        return []
    segments: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    for point in raw_points:
        if point is None:
            if current:
                segments.append(current)
                current = []
            continue
        current.append((float(point[0]), float(point[1])))
    if current:
        segments.append(current)
    return segments


def _maze_route_subpaths(
    sample: CanonicalSample,
    metadata_record: Dict[str, Any],
    evaluator: object,
    expected_path_ids: Sequence[int],
) -> tuple[List[List[Tuple[float, float]]], int, str]:
    """Canvas-space route polyline of the attested route, split at teleports.

    One shape for all eight tasks: a toroidal wrap and a layered staircase break
    the route into several strokes, everything else is a single stroke.  The
    geometry comes from the record's own cell boxes and the task generator's cell
    centres -- never from a saved canonical route or from reference media.
    """

    if sample.task_name == "maze_toroidal":
        segments = _toroidal_route_segments(metadata_record, expected_path_ids)
        thickness = max(3, int(metadata_record["cell_size"]) // 3)
        return segments, thickness, "square"
    if sample.task_name == "maze_layered":
        segments = _layered_route_segments(metadata_record, evaluator, expected_path_ids)
        thickness = max(4, int(metadata_record["cell_size"]) // 3)
        if thickness % 2:
            thickness -= 1
        return segments, thickness, "square"
    points, thickness, joint_style = _maze_route_geometry(
        sample.task_name,
        metadata_record,
        evaluator,
        expected_path_ids,
    )
    return ([points] if len(points) >= 2 else []), thickness, joint_style


def _maze_route_arc_lengths(
    subpaths: Sequence[Sequence[Tuple[float, float]]],
) -> tuple[List[List[Tuple[float, float]]], List[List[float]], float]:
    """Route strokes, their per-vertex cumulative arc lengths, and the total."""

    strokes = [
        [(float(x), float(y)) for x, y in subpath]
        for subpath in subpaths
        if len(subpath) >= 2
    ]
    lengths: List[List[float]] = []
    total = 0.0
    for stroke in strokes:
        cumulative: List[float] = [0.0]
        for start, end in zip(stroke, stroke[1:]):
            cumulative.append(
                cumulative[-1] + float(np.hypot(end[0] - start[0], end[1] - start[1]))
            )
        lengths.append(cumulative)
        total += cumulative[-1]
    return strokes, lengths, total


def _maze_route_prefix(
    subpaths: Sequence[Sequence[Tuple[float, float]]],
    fraction: float,
) -> tuple[List[List[Tuple[float, float]]], Optional[Tuple[float, float]]]:
    """Route strokes revealed at ``fraction`` of the total arc length, and the tip.

    The reveal walks arc length over the concatenated strokes and never counts
    the gap between two strokes, which is how the shared writer parameterises the
    same route: a wrap or a staircase hop consumes no time.
    """

    strokes, lengths, total = _maze_route_arc_lengths(subpaths)
    if not strokes:
        return [], None
    if total <= 1e-9:
        return [], strokes[0][0]
    remaining = max(0.0, min(1.0, float(fraction))) * total
    revealed: List[List[Tuple[float, float]]] = []
    tip: Optional[Tuple[float, float]] = strokes[0][0]
    for stroke, cumulative in zip(strokes, lengths):
        stroke_length = cumulative[-1]
        if stroke_length <= 1e-9:
            continue
        if remaining >= stroke_length - 1e-6:
            revealed.append(list(stroke))
            tip = stroke[-1]
            remaining -= stroke_length
            continue
        if remaining <= 1e-6:
            break
        full = 0
        while full + 1 < len(cumulative) and cumulative[full + 1] <= remaining + 1e-6:
            full += 1
        visible = list(stroke[: full + 1])
        span = cumulative[full + 1] - cumulative[full] if full + 1 < len(cumulative) else 0.0
        if span > 1e-9:
            ratio = min(1.0, (remaining - cumulative[full]) / span)
            start, end = stroke[full], stroke[full + 1]
            visible.append(
                (start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio)
            )
        if len(visible) >= 1:
            revealed.append(visible)
            tip = visible[-1]
        break
    return revealed, tip


def _maze_route_footprint(
    shape: Tuple[int, int],
    revealed: Sequence[Sequence[Tuple[float, float]]],
    tip: Optional[Tuple[float, float]],
    thickness: int,
    joint_style: str,
    evaluator: object,
    static_red_mask: np.ndarray,
) -> np.ndarray:
    """Red footprint one revealed route state paints.

    The tip marker is drawn, because the candidate's frames carry it: it is the
    only part of the picture that moves once the stroke behind it has been laid
    down, so a bank without it would leave consecutive states almost identical
    exactly where the process is distinguishable.

    The input's own red is removed so the footprint carries only what the route
    adds, exactly as the candidate side of the comparison does.
    """

    height, width = shape
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    strokes = [list(stroke) for stroke in revealed]
    for index, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue
        draw_path_line(
            canvas,
            stroke,
            (220, 30, 30),
            thickness,
            joint_style=joint_style,
            tip_marker=bool(tip is not None and index == len(strokes) - 1),
        )
    return evaluator._red_mask(np.asarray(canvas)) & ~static_red_mask


def _maze_progress_bank(
    sample: CanonicalSample,
    metadata_record: Dict[str, Any],
    evaluator: object,
    expected_path_ids: Sequence[int],
    static_red_mask: np.ndarray,
    *,
    states: int,
) -> tuple[List[np.ndarray], List[float]]:
    """Source route bank: the attested route revealed at ascending arc lengths.

    Entries are the *red footprint* of each state rather than a full re-render of
    the scene.  Three of the eight tasks composite their stroke through a
    supersampled overlay and the rest stroke it straight onto the canvas, so a
    pixel-exact scene bank would have to reproduce three different rasterisers;
    the footprint is what every one of them agrees on, and it is the same
    representation this route's own keyframe classifier already trusts.  It is
    also tone-invariant by construction, as required for a
    whole-video verifier faced with harmless style variance.
    """

    subpaths, thickness, joint_style = _maze_route_subpaths(
        sample, metadata_record, evaluator, expected_path_ids
    )
    shape = (int(static_red_mask.shape[0]), int(static_red_mask.shape[1]))
    if not subpaths or thickness <= 0:
        return [], []
    quantum = 1.0 / float(max(1, states))
    bank: List[np.ndarray] = []
    progress: List[float] = []
    for index in range(int(states) + 1):
        fraction = index * quantum
        revealed, tip = _maze_route_prefix(subpaths, fraction)
        footprint = _maze_route_footprint(
            shape, revealed, tip, thickness, joint_style, evaluator, static_red_mask
        )
        bank.append(np.repeat((footprint.astype(np.uint8) * 255)[:, :, None], 3, axis=2))
        progress.append(fraction)
    return bank, progress


def _maze_route_stroke_mask(
    sample: CanonicalSample,
    metadata_record: Dict[str, Any],
    evaluator: object,
    expected_path_ids: Sequence[int],
    shape: Tuple[int, int],
) -> tuple[np.ndarray, int]:
    """Footprint of the whole route stroke, without the tip marker, and its thickness."""

    subpaths, thickness, joint_style = _maze_route_subpaths(
        sample, metadata_record, evaluator, expected_path_ids
    )
    if not subpaths or thickness <= 0:
        return np.zeros(shape, dtype=bool), 0
    height, width = shape
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    for stroke in subpaths:
        if len(stroke) >= 2:
            draw_path_line(
                canvas,
                [(float(x), float(y)) for x, y in stroke],
                (220, 30, 30),
                thickness,
                joint_style=joint_style,
                tip_marker=False,
            )
    return np.asarray(canvas).any(axis=2), int(thickness)


def _maze_route_order_region(
    sample: CanonicalSample,
    metadata_record: Dict[str, Any],
    evaluator: object,
    expected_path_ids: Sequence[int],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Region the bounded-step reading compares over.

    The reading compares *red* footprints, and red only ever appears inside the
    stroke and the tip marker's inner disc, whose radius (``thickness * 0.60``)
    exceeds the stroke's half-width by a tenth of the thickness.  One
    ``_MAZE_ROUTE_MASK_DILATION_SIZE`` dilation therefore contains everything red
    a legal frame can carry, which keeps the emission matrix on the smallest
    region that still holds all the evidence: the marker's *white* ring belongs
    to the static gate's region, not here, and including it would triple the
    reading's cost while adding no red pixel to compare.
    """

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for maze video evaluation.") from exc

    stroke_mask, thickness = _maze_route_stroke_mask(
        sample, metadata_record, evaluator, expected_path_ids, shape
    )
    if thickness <= 0:
        return np.zeros(shape, dtype=bool)
    kernel = np.ones(
        (_MAZE_ROUTE_MASK_DILATION_SIZE, _MAZE_ROUTE_MASK_DILATION_SIZE), dtype=np.uint8
    )
    return cv2.dilate(stroke_mask.astype(np.uint8), kernel) > 0


def _maze_video_dynamic_mask(
    sample: CanonicalSample,
    metadata_record: Dict[str, Any],
    evaluator: object,
    expected_path_ids: Sequence[int],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Region a legal maze route video is allowed to change.

    The region is the *swept* footprint of the whole route, not the union of the
    sampled bank states.  Sampling cannot give the sweep: the reveal advances
    tens of pixels per frame on a long route, while the tip marker is a disc of
    about thirteen pixels' radius, so sampled marker positions leave gaps between
    them and a legal frame whose marker lands in one reads as a mutation.  The
    sweep is exact instead of sampled: the marker is a disc of
    ``pawn_marker_radius`` centred on a point of the polyline, and the polyline
    lies inside the full stroke, so dilating the full stroke by that radius is
    the Minkowski sum that contains every position the marker can occupy.

    On top of that: one ``_MAZE_DYNAMIC_DILATION_SIZE`` step for the candidate's
    own antialiasing, and the cells whose contents commit a semantic update (a
    collected key, an opened gate).  Nothing here reads the candidate, so a pixel
    outside this region is a pixel no legal delivery may touch.
    """

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for maze video evaluation.") from exc

    stroke_mask, thickness = _maze_route_stroke_mask(
        sample, metadata_record, evaluator, expected_path_ids, shape
    )
    dynamic = np.zeros(shape, dtype=bool)
    if thickness > 0:
        reach = int(np.ceil(pawn_marker_radius(thickness))) + _MAZE_DYNAMIC_DILATION_SIZE // 2
        size = 2 * reach + 1
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        dynamic = cv2.dilate(stroke_mask.astype(np.uint8), disc) > 0
    dynamic |= _spatial_navigation_maze_dynamic_cells_mask(metadata_record, shape)
    return dynamic


def _verify_maze_static_preservation(
    frames: Sequence[np.ndarray],
    puzzle_pixels: np.ndarray,
    static: np.ndarray,
) -> tuple[Dict[str, Any], bool]:
    """Check task-independent content at codec-tolerant key observations."""

    if not bool(np.any(static)):
        return {
            "video_static_content_preserved": False,
            "video_static_preservation_reason": "the source route declares the whole frame dynamic",
        }, False
    worst = 0.0
    worst_frame = -1
    violating_frames = 0
    for index in keyframe_indices(frames):
        error, _regions = static_region_errors(
            frames[index], puzzle_pixels, static, raster_tolerance=1
        )
        if error > STATIC_REGION_FLOOR:
            violating_frames += 1
        if error > worst:
            worst, worst_frame = error, index
    passed = bool(frames) and violating_frames == 0
    return {
        "video_static_declared_dynamic_fraction": round(float(1.0 - static.mean()), 6),
        "video_max_static_region_error": round(float(worst), 6),
        "video_worst_static_region_frame": worst_frame,
        "video_static_region_violation_frames": violating_frames,
        "video_static_region_size": REGION,
        "video_static_region_bound": STATIC_REGION_FLOOR,
        "video_static_raster_tolerance_px": 1,
        "video_static_preservation_source": "supplied_input_and_source_replayed_route_states",
        "video_static_content_preserved": passed,
    }, passed


def _read_maze_route_process_order(
    bank: Sequence[np.ndarray],
    bank_progress: Sequence[float],
    dynamic: np.ndarray,
    frame_route_masks: Sequence[np.ndarray],
    *,
    bank_quantum: float,
) -> tuple[Dict[str, Any], bool]:
    """Read the decoded route footprints as a bounded-step walk over the source.

    The keyframe classifier this supplements samples the route at a handful of
    semantic commits, so a deletion or a reversal that spans less than one
    checkpoint gap leaves its compact sequence untouched: measured before this
    gate, a twelve-frame hard cut and a six-frame deletion were accepted on all
    nine tasks of the route.  The shared bounded-step reading judges every frame
    against the source's own advance and the source's own local rate of visible
    change at the state the frame is read as, so the same corruption is judged
    the same way wherever in the process it sits.
    """

    if len(frame_route_masks) < 5:
        return {
            "video_process_order_available": False,
            "video_process_order_reason": "too few decoded frames to read as a sequence",
        }, False
    if len(bank) < 3:
        return {
            "video_process_order_available": False,
            "video_process_order_reason": "failed to render the source route progress bank",
        }, False
    if not bool(np.any(dynamic)):
        return {
            "video_process_order_available": False,
            "video_process_order_reason": "the source route declares no dynamic region",
        }, False
    observed = [
        np.repeat((np.asarray(mask).astype(np.uint8) * 255)[:, :, None], 3, axis=2)
        for mask in frame_route_masks
    ]
    reading = read_process_order(
        observed,
        bank,
        bank_progress,
        dynamic,
        non_advancing_frames=_MAZE_VIDEO_NON_ADVANCING_FRAMES,
        bank_quantum=bank_quantum,
    )
    metrics = {
        "video_process_order_available": True,
        "video_process_order_verifier": "bounded_step_source_route_progress_alignment",
        "video_progress_bank_states": len(bank),
        "video_progress_bank_quantum": round(float(bank_quantum), 6),
        "video_reading_reference_pace": round(float(reading.reference_pace), 6),
        "video_reading_distinct_states": reading.distinct_states,
        "video_reading_comparison_kernel": reading.comparison_kernel,
        "video_reading_max_step": round(float(reading.max_step), 6),
        "video_reading_step_ratio": round(float(reading.step_ratio), 6),
        "video_reading_step_violations": reading.step_break_count,
        "video_frame_pace_violations": reading.pace_break_count,
        "video_worst_frame_pace_ratio": round(float(reading.worst_pace_ratio), 6),
        "video_worst_frame_pair_change": round(float(reading.worst_pair_change), 6),
        "video_reading_progress": [round(float(value), 4) for value in reading.progress],
        "video_bounded_step_order_pass": bool(reading.ordered),
    }
    return metrics, bool(reading.ordered)


def _maze_route_revisits_a_cell(expected_path_ids: Sequence[int]) -> bool:
    """Whether route retracing makes cursor movement necessary pace evidence."""

    identifiers = [int(value) for value in expected_path_ids]
    return len(identifiers) > len(set(identifiers))


def _maze_tip_track(
    subpaths: Sequence[Sequence[Tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Integer pixels the source route's tip passes through, and its arc budget.

    Returns ``(xs, ys, route_length, largest_stroke_hop)``.  The positions are
    the whole set of places the marker can legally be, so localising the
    candidate's marker against them costs exactly the raster grid and nothing
    else.  ``largest_stroke_hop`` is the longest jump between the end of one
    stroke and the start of the next: the reveal consumes no arc length across a
    wrap or a staircase, so a frame that straddles one legally moves that far
    beyond its arc advance.  Single-stroke routes report zero.
    """

    strokes, _lengths, total = _maze_route_arc_lengths(subpaths)
    xs: List[int] = []
    ys: List[int] = []
    for stroke in strokes:
        for start, end in zip(stroke, stroke[1:]):
            span = float(np.hypot(end[0] - start[0], end[1] - start[1]))
            steps = max(1, int(np.ceil(span * 2.0)))
            for index in range(steps + 1):
                ratio = index / steps
                x = start[0] + (end[0] - start[0]) * ratio
                y = start[1] + (end[1] - start[1]) * ratio
                # Pillow rasterizes fractional ellipse bounds onto integer
                # pixels. At half-pixel grid centres, nearest-even rounding
                # alone can miss the visible marker at a corner.
                for px in {int(np.floor(x)), int(np.ceil(x))}:
                    for py in {int(np.floor(y)), int(np.ceil(y))}:
                        xs.append(px)
                        ys.append(py)
    hop = 0.0
    for previous, following in zip(strokes, strokes[1:]):
        hop = max(
            hop,
            float(np.hypot(following[0][0] - previous[-1][0], following[0][1] - previous[-1][1])),
        )
    if not xs:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), float(total), hop
    # Deduplicate by packing the pair into one integer, which needs both halves
    # non-negative and inside the field width.  Route geometry is canvas-space
    # and never leaves it, but clamping here means the packing cannot silently
    # alias if some future task's geometry does.
    column = np.clip(np.asarray(xs, dtype=np.int64), 0, 65535)
    row = np.clip(np.asarray(ys, dtype=np.int64), 0, 65535)
    packed = np.unique(row * 65536 + column)
    return (
        (packed % 65536).astype(np.int32),
        (packed // 65536).astype(np.int32),
        float(total),
        float(hop),
    )


def _maze_tip_marker_offsets(thickness: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Pixel offsets of the tip marker's red core and of the ring around it.

    ``_draw_pawn_marker`` paints a white disc of :func:`pawn_marker_radius` and
    then a red disc of ``max(3.5, thickness * 0.60)`` inside it, so at the marker
    the ring between the two radii is white by construction -- it is the only
    place on the route where the red stops without the stroke continuing through
    it.  Both samples are inset from their own radius because the discs are
    rasterised with antialiasing and a pixel within about one of either edge can
    fall on either side.
    """

    outer = float(pawn_marker_radius(thickness))
    inner = max(3.5, float(thickness) * 0.60)
    core = max(1.5, inner - _MAZE_TIP_RASTER_INSET_PX)
    ring_low = inner + _MAZE_TIP_RASTER_INSET_PX
    ring_high = max(ring_low + 1.0, outer - _MAZE_TIP_RASTER_INSET_PX)
    radius = int(np.ceil(ring_high))
    grid = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    distance = np.hypot(grid[1], grid[0])
    core_dy, core_dx = np.nonzero(distance <= core)
    ring_dy, ring_dx = np.nonzero((distance >= ring_low) & (distance <= ring_high))
    return (
        (core_dy - radius).astype(np.int32),
        (core_dx - radius).astype(np.int32),
        (ring_dy - radius).astype(np.int32),
        (ring_dx - radius).astype(np.int32),
        radius,
    )


def _verify_maze_tip_advance(
    sample: CanonicalSample,
    metadata_record: Dict[str, Any],
    evaluator: object,
    expected_path_ids: Sequence[int],
    frame_red_masks: Sequence[np.ndarray],
    route_masks: Optional[Sequence[np.ndarray]] = None,
    observed_progress: Optional[Sequence[float]] = None,
    frame_white_masks: Optional[Sequence[np.ndarray]] = None,
) -> tuple[Dict[str, Any], bool]:
    """Require the prompted cursor during drawing, including non-retraced routes."""
    if sample.task_name == "maze_labyrinth":
        return {"video_tip_advance_available": False,
                "video_tip_advance_reason": "the public labyrinth instruction has no riding cursor",
                "video_tip_advance_pass": True}, True
    subpaths, thickness, joint_style = _maze_route_subpaths(
        sample, metadata_record, evaluator, expected_path_ids
    )
    if not subpaths or thickness <= 0:
        return {
            "video_tip_advance_available": False,
            "video_tip_advance_reason": "failed to rebuild the source route geometry",
            "video_tip_advance_pass": False,
        }, False
    xs, ys, route_length, stroke_hop = _maze_tip_track(subpaths)
    frames = [np.asarray(mask, dtype=bool) for mask in frame_red_masks]
    if xs.size == 0 or route_length <= 1e-6 or len(frames) < 5:
        return {
            "video_tip_advance_available": False,
            "video_tip_advance_reason": "too little source route or too few frames to read a tip track",
            "video_tip_advance_pass": False,
        }, False

    import cv2
    core_dy, core_dx, ring_dy, ring_dx, radius = _maze_tip_marker_offsets(thickness)
    height, width = frames[0].shape
    xs, ys = np.clip(xs, 0, width-1), np.clip(ys, 0, height-1)
    # The route mask already permits a three-pixel raster displacement. The
    # generated cursor rides that observed stroke, so limiting its centre to
    # exact source-centre pixels rejects an otherwise accepted displaced line.
    # Search the same source-declared corridor; retain every separate marker.
    corridor = np.zeros((height, width), dtype=np.uint8)
    corridor[ys, xs] = 1
    corridor = cv2.dilate(corridor, np.ones((_MAZE_ROUTE_MASK_DILATION_SIZE,
                                           _MAZE_ROUTE_MASK_DILATION_SIZE), np.uint8))
    ys, xs = np.nonzero(corridor)
    rows, cols = ys[:, None] + radius, xs[:, None] + radius
    positions, candidate_counts, detection_scores = [], [], []
    marker_centers = []
    shape_candidates = []
    source_cap_frames = []
    weak_positions: List[Optional[Tuple[float, float]]] = []
    for frame_index, mask in enumerate(frames):
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        reachable = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        keep = np.flatnonzero(reachable[ys, xs])
        if keep.size:
            core = padded[rows[keep]+core_dy, cols[keep]+core_dx].mean(axis=1)
            ring = padded[rows[keep]+ring_dy, cols[keep]+ring_dx].mean(axis=1)
            scores = core-ring
            shape_hits = np.flatnonzero((core >= .65) & (ring <= .35) & (scores >= .60))
            shape_candidates.append([(float(xs[keep[k]]), float(ys[keep[k]]), float(scores[k])) for k in shape_hits])
            detection_scores.append(float(scores.max()))
            # Retain every separate red-core/white-ring marker, rather than
            # selecting the best one and silently overlooking duplicates.
            order = np.argsort(-scores)
            hits = keep[order[scores[order] >= _MAZE_TIP_DETECTION_SCORE]]
            centers = []
            for hit in hits:
                point = (float(xs[hit]), float(ys[hit]))
                if all(np.hypot(point[0]-x, point[1]-y) > 2*pawn_marker_radius(thickness) for x, y in centers):
                    centers.append(point)
            # Generation and timestamp resampling can square or stretch the
            # disc, so an exact circular ring is only the strong observation.
            # Recover a weak observation from a visible red core and a mostly
            # non-red ring at the independently read leading route position.
            weak = None
            if observed_progress is not None and frame_index < len(observed_progress):
                _revealed, expected_tip = _maze_route_prefix(subpaths, observed_progress[frame_index])
                if expected_tip is not None:
                    distance = np.hypot(xs[keep] - expected_tip[0], ys[keep] - expected_tip[1])
                    plausible = (core >= 0.65) & (ring <= 0.35) & (scores >= 0.60)
                    plausible &= distance <= 2 * pawn_marker_radius(thickness) + 3
                    # A bare line cap also has a red centre and white outside.
                    # Require a visible departure from that cap, so a deleted
                    # marker is not reconstructed from its old route endpoint.
                    plain = _maze_route_footprint(
                        (height, width), _revealed, None, thickness, joint_style,
                        evaluator, np.zeros((height, width), dtype=bool),
                    )
                    reach = max(2, int(np.ceil(2 * pawn_marker_radius(thickness))))
                    tx, ty = int(round(expected_tip[0])), int(round(expected_tip[1]))
                    top, bottom = max(0, ty-reach), min(height, ty+reach+1)
                    left, right = max(0, tx-reach), min(width, tx+reach+1)
                    visible = np.asarray(route_masks[frame_index], dtype=bool) if route_masks is not None else mask
                    cap_difference = np.count_nonzero(
                        visible[top:bottom, left:right] != plain[top:bottom, left:right]
                    )
                    mechanic_occlusion = False
                    if sample.task_name == "maze_keys_and_locks":
                        for item in [*(metadata_record.get("keys") or []), *(metadata_record.get("locks") or [])]:
                            row, col = item["cell"]
                            box = metadata_record["cell_bboxes"][row][col]
                            mechanic_occlusion |= bool(
                                box[0]-radius <= tx <= box[2]+radius
                                and box[1]-radius <= ty <= box[3]+radius
                            )
                    # Key/gate glyphs in frozen scenes are painted above the
                    # contacted cursor. Their independent contact/state replay
                    # below attests that brief, task-owned occlusion.
                    plausible &= cap_difference > 8 or mechanic_occlusion
                    matches = np.flatnonzero(plausible)
                    if matches.size:
                        best = matches[np.argmax(scores[matches] - distance[matches] / max(1., thickness) * .02)]
                        weak = (float(xs[keep[best]]), float(ys[keep[best]]))
            weak_positions.append(weak)
        else:
            detection_scores.append(0.)
            centers = []
            weak_positions.append(None)
            shape_candidates.append([])
        if (sample.task_name == "maze_layered" and len(centers) > 1
                and frame_white_masks is not None and observed_progress is not None):
            revealed, _tip = _maze_route_prefix(subpaths, observed_progress[frame_index])
            starts = [stroke[0] for stroke in revealed[1:] if stroke]
            if starts:
                plain = _maze_route_footprint(
                    (height, width), revealed, None, thickness, joint_style,
                    evaluator, np.zeros((height, width), dtype=bool),
                )
                bare = np.pad(plain, radius, constant_values=False)
                white = np.pad(frame_white_masks[frame_index], radius, constant_values=False)
                retained = []
                for x, y in centers:
                    cx, cy = int(round(x)) + radius, int(round(y)) + radius
                    at_start = any(np.hypot(x-sx, y-sy) <= 3 for sx, sy in starts)
                    bare_core = float(bare[cy+core_dy, cx+core_dx].mean())
                    white_ring = float(white[cy+ring_dy, cx+ring_dx].mean())
                    # A yellow staircase is non-red, but it is not the white
                    # ring of a second cursor at a source stroke's bare start.
                    if at_start and bare_core >= .65 and white_ring < .25:
                        source_cap_frames.append(frame_index)
                    else:
                        retained.append((x, y))
                centers = retained
        candidate_counts.append(len(centers))
        marker_centers.append(centers)
        positions.append(centers[0] if len(centers) == 1 else None)

    strong = [index for index, point in enumerate(positions) if point is not None]
    recovered = []
    for index, point in enumerate(positions):
        # A weak frame must carry its own visible core/ring evidence and have
        # a nearby unambiguous marker witness. No missing image is interpolated.
        if point is None and candidate_counts[index] == 0 and weak_positions[index] is not None:
            if any(abs(index - witness) <= 4 for witness in strong):
                positions[index] = weak_positions[index]
                recovered.append(index)

    # One damaged raster observation may retain the same visible cursor even
    # when its disc is split, bent, or accompanied by a smaller local ghost.
    # Both neighbours must independently identify that cursor. Snapshot them
    # before recovery, so this cannot fill a run of absent/duplicated heads.
    witnessed = list(positions)
    isolated_recovered = []
    outer = pawn_marker_radius(thickness)
    source_corridor = cv2.dilate(corridor, np.ones((2 * int(np.ceil(outer)) + 1,) * 2, np.uint8)) > 0
    for index in range(1, len(positions) - 1):
        if witnessed[index] is not None or witnessed[index - 1] is None or witnessed[index + 1] is None:
            continue
        if observed_progress is None or index >= len(observed_progress):
            continue
        revealed, expected_tip = _maze_route_prefix(subpaths, observed_progress[index])
        if expected_tip is None:
            continue
        local = np.zeros((height, width), np.uint8)
        anchors = [witnessed[index - 1], expected_tip, witnessed[index + 1]]
        for left, right in zip(anchors, anchors[1:]):
            cv2.line(local, tuple(int(round(v)) for v in left), tuple(int(round(v)) for v in right),
                     1, max(1, int(np.ceil(4 * outer))))
        local = (local > 0) & source_corridor
        mask = frames[index]
        centers = marker_centers[index]
        if len(centers) == 2:
            if not all(local[int(round(y)), int(round(x))] for x, y in centers):
                continue
            _count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            masses = []
            for x, y in centers:
                cx, cy = int(round(x)), int(round(y))
                component = labels[cy, cx]
                yy, xx = np.ogrid[:height, :width]
                body = (xx - x) ** 2 + (yy - y) ** 2 <= outer ** 2
                masses.append(int(np.count_nonzero(body & (labels == component))) if component else 0)
            primary = int(np.argmax(masses))
            if min(masses) <= 0 or min(masses) > 0.65 * max(masses):
                continue
            point = centers[primary]
            if np.hypot(point[0] - expected_tip[0], point[1] - expected_tip[1]) > 2 * outer + 3:
                continue
        elif not centers:
            # A source floor/wrap transition may be read one timestamp before
            # the visible cursor switches strokes. Its two independently
            # observed neighbours delimit that same legal source-front window.
            local_shapes = [(x, y, score) for x, y, score in shape_candidates[index]
                            if local[int(round(y)), int(round(x))]]
            point = None
            if local_shapes:
                caps = []
                for witness in (index - 1, index, index + 1):
                    prefix, _tip = _maze_route_prefix(subpaths, observed_progress[witness])
                    caps.append(_maze_route_footprint(
                        (height, width), prefix, None, thickness, joint_style,
                        evaluator, np.zeros((height, width), dtype=bool),
                    ))
                visible = np.asarray(route_masks[index], dtype=bool) if route_masks is not None else mask
                reach = int(np.ceil(2 * outer))
                for x, y, _score in sorted(local_shapes, key=lambda item: -item[2]):
                    cx, cy = int(round(x)), int(round(y))
                    region = np.s_[max(0, cy-reach):min(height, cy+reach+1),
                                   max(0, cx-reach):min(width, cx+reach+1)]
                    if min(np.count_nonzero(visible[region] != cap[region]) for cap in caps) > 8:
                        point = (x, y)
                        break
            if point is not None:
                positions[index] = point
                isolated_recovered.append(index)
                continue
            plain = _maze_route_footprint(
                (height, width), revealed, None, thickness, joint_style,
                evaluator, np.zeros((height, width), dtype=bool),
            )
            visible = np.asarray(route_masks[index], dtype=bool) if route_masks is not None else mask
            yy, xx = np.ogrid[:height, :width]
            head = (xx - expected_tip[0]) ** 2 + (yy - expected_tip[1]) ** 2 <= (1.5 * outer) ** 2
            head &= local
            cut = plain & ~visible & head
            added = mask & ~cv2.dilate(plain.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & head
            inner = max(3.5, float(thickness) * .60)
            body = mask & head & ((xx - expected_tip[0]) ** 2 + (yy - expected_tip[1]) ** 2 <= inner ** 2)
            # A deleted marker leaves a plain cap. A deformed visible head
            # can instead enlarge that cap OR cut into it with its white rim.
            if np.count_nonzero(body) <= 8 or (np.count_nonzero(cut) <= 8 and np.count_nonzero(added) <= 8):
                continue
            py, px = np.nonzero(mask & head)
            if not len(px):
                continue
            nearest = int(np.argmin((px - expected_tip[0]) ** 2 + (py - expected_tip[1]) ** 2))
            point = (float(px[nearest]), float(py[nearest]))
        else:
            continue
        positions[index] = point
        isolated_recovered.append(index)

    detected = [index for index, point in enumerate(positions) if point is not None]
    pace = reference_advance(len(frames), non_advancing_frames=_MAZE_VIDEO_NON_ADVANCING_FRAMES)
    advance = route_length * pace
    edge = max(
        (float(np.hypot(right[0] - left[0], right[1] - left[1]))
         for stroke in subpaths for left, right in zip(stroke, stroke[1:])),
        default=0.0,
    )
    # A rendered cell transition may span two adjacent links between decoded
    # timestamps. Its visibility is independent of the reference's average
    # speed, especially on a retrace where the old stroke is already present.
    allowance = max(_MAZE_TIP_STEP_SLACK * advance, 2 * edge) + 2 * pawn_marker_radius(thickness) + stroke_hop
    steps: List[float] = []
    for previous, following in zip(positions, positions[1:]):
        if previous is None or following is None:
            continue
        steps.append(float(np.hypot(following[0] - previous[0], following[1] - previous[1])))
    worst = max(steps) if steps else 0.0
    ratio = worst / allowance if allowance > 0.0 else float("inf")
    violations = sum(1 for step in steps if step > allowance)
    # The marker is painted on every frame the route animates, so once it has
    # appeared it may not go away again: a candidate that stops drawing it stops
    # showing the only thing that moves along a retrace.
    persistent = bool(detected) and detected == list(range(detected[0], len(positions)))
    onset = next((i for i, mask in enumerate(route_masks or [])
                  if np.count_nonzero(mask) > max(8, np.count_nonzero(route_masks[0]) + 8)), None)
    present_during_drawing = bool(onset is not None and all(point is not None for point in positions[onset:]))
    # A non-retraced route already supplies visible stroke progress. The
    # additional cursor obligation does not prescribe the renderer's speed.
    step_ok = violations == 0 or not _maze_route_revisits_a_cell(expected_path_ids)
    passed = bool(persistent and present_during_drawing and step_ok)
    return {
        "video_tip_route_onset_frame": onset,
        "video_tip_present_during_drawing": present_during_drawing,
        "video_tip_missing_drawing_frames": [i for i in range(onset or 0, len(positions)) if positions[i] is None],
        "video_tip_advance_available": True,
        "video_tip_advance_verifier": "source_route_tip_marker_step_bound",
        "video_tip_route_length_px": round(float(route_length), 3),
        "video_tip_reference_advance_px": round(float(advance), 3),
        "video_tip_step_allowance_px": round(float(allowance), 3),
        "video_tip_stroke_hop_px": round(float(stroke_hop), 3),
        "video_tip_candidate_positions": int(xs.size),
        "video_tip_detected_frames": len(detected),
        "video_tip_first_detected_frame": detected[0] if detected else None,
        "video_tip_candidate_counts": candidate_counts,
        "video_tip_recovered_shape_frames": recovered,
        "video_tip_recovered_isolated_artifact_frames": isolated_recovered,
        "video_tip_source_stroke_cap_frames": sorted(set(source_cap_frames)),
        "video_tip_detection_scores": detection_scores,
        "video_tip_positions_by_frame": positions,
        "video_tip_persistent": persistent,
        "video_tip_measured_steps": len(steps),
        "video_tip_max_step_px": round(float(worst), 3),
        "video_tip_worst_step_ratio": round(float(ratio), 6),
        "video_tip_step_violations": int(violations),
        "video_tip_advance_pass": passed,
    }, passed


def _render_expected_maze_route_masks(
    sample: CanonicalSample,
    metadata_record: Dict[str, Any],
    evaluator: object,
    expected_path_ids: Sequence[int],
    prefix_indices: Sequence[int],
    static_red_mask: np.ndarray,
) -> List[np.ndarray]:
    height, width = static_red_mask.shape
    masks: List[np.ndarray] = []
    for prefix_index in prefix_indices:
        if prefix_index == 0:
            masks.append(np.zeros((height, width), dtype=bool))
            continue
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        if sample.task_name == "maze_toroidal":
            segments = _toroidal_route_segments(
                metadata_record,
                expected_path_ids[: prefix_index + 1],
            )
            thickness = max(3, int(metadata_record["cell_size"]) // 3)
            if not segments or not any(len(segment) >= 2 for segment in segments):
                return []
            for segment in segments:
                draw_path_line(
                    canvas,
                    segment,
                    (220, 30, 30),
                    thickness,
                    joint_style="square",
                    tip_marker=False,
                )
        elif sample.task_name == "maze_layered":
            segments = _layered_route_segments(
                metadata_record,
                evaluator,
                expected_path_ids[: prefix_index + 1],
            )
            thickness = max(4, int(metadata_record["cell_size"]) // 3)
            if thickness % 2:
                thickness -= 1
            if not segments or not any(len(segment) >= 2 for segment in segments):
                return []
            for segment in segments:
                draw_path_line(
                    canvas,
                    segment,
                    (220, 30, 30),
                    thickness,
                    joint_style="square",
                    tip_marker=False,
                )
        else:
            points, thickness, joint_style = _maze_route_geometry(
                sample.task_name,
                metadata_record,
                evaluator,
                expected_path_ids[: prefix_index + 1],
            )
            if len(points) < 2 or thickness <= 0:
                return []
        if sample.task_name not in {"maze_toroidal", "maze_layered"}:
            draw_path_line(
                canvas,
                points,
                (220, 30, 30),
                thickness,
                joint_style=joint_style,
                tip_marker=False,
            )
        route_mask = evaluator._red_mask(np.asarray(canvas))
        masks.append(route_mask & ~static_red_mask)
    return masks


def _maze_route_mask_coverages(
    candidate_mask: np.ndarray,
    expected_mask: np.ndarray,
    cv2: object,
) -> tuple[float, float]:
    if not candidate_mask.any() and not expected_mask.any():
        return 1.0, 1.0
    if not candidate_mask.any() or not expected_mask.any():
        return 0.0, 0.0
    kernel = np.ones(
        (_MAZE_ROUTE_MASK_DILATION_SIZE, _MAZE_ROUTE_MASK_DILATION_SIZE),
        dtype=np.uint8,
    )
    candidate_dilated = cv2.dilate(candidate_mask.astype(np.uint8), kernel) > 0
    expected_dilated = cv2.dilate(expected_mask.astype(np.uint8), kernel) > 0
    candidate_coverage = float((candidate_mask & expected_dilated).sum() / candidate_mask.sum())
    expected_coverage = float((expected_mask & candidate_dilated).sum() / expected_mask.sum())
    return candidate_coverage, expected_coverage


def _spatial_navigation_maze_dynamic_cells_mask(
    record: Dict[str, Any],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Return semantic-update regions (keys/gates), not a correctness mask."""

    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    bboxes = record.get("cell_bboxes")
    if not isinstance(bboxes, list):
        return mask
    for item in [*(record.get("keys") or []), *(record.get("locks") or [])]:
        if not isinstance(item, dict):
            continue
        cell = item.get("cell")
        try:
            row, col = int(cell[0]), int(cell[1])
            bbox = bboxes[row][col]
            left, top, right, bottom = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError, IndexError):
            continue
        mask[max(0, top) : min(height, bottom), max(0, left) : min(width, right)] = True
    return mask


def _keys_terminal_state_metrics(
    record: Dict[str, Any],
    source_pixels: np.ndarray,
    candidate_pixels: np.ndarray,
) -> Dict[str, Any]:
    bboxes = record.get("cell_bboxes")
    if not isinstance(bboxes, list):
        return {"keys_terminal_state_ok": False, "keys_terminal_state_reason": "missing cell geometry"}

    def crop_for(item: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            row, col = int(item["cell"][0]), int(item["cell"][1])
            left, top, right, bottom = [int(round(float(value))) for value in bboxes[row][col]]
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        left, top = max(0, left), max(0, top)
        right, bottom = min(candidate_pixels.shape[1], right), min(candidate_pixels.shape[0], bottom)
        if left >= right or top >= bottom:
            return None
        return source_pixels[top:bottom, left:right], candidate_pixels[top:bottom, left:right]

    key_scores: List[float] = []
    key_label_scores: List[float] = []
    for item in record.get("keys") or []:
        if not isinstance(item, dict):
            return {"keys_terminal_state_ok": False, "keys_terminal_state_reason": "malformed key source"}
        crops = crop_for(item)
        try:
            base = np.asarray([int(value) for value in item["color_rgb"]], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return {"keys_terminal_state_ok": False, "keys_terminal_state_reason": "malformed key colour"}
        if crops is None or base.shape != (3,):
            return {"keys_terminal_state_ok": False, "keys_terminal_state_reason": "missing key crop"}
        source_crop, candidate_crop = crops
        collected = np.rint(base + (255.0 - base) * 0.58).astype(np.int16)
        delta = np.max(np.abs(candidate_crop.astype(np.int16) - collected[None, None, :]), axis=2)
        key_scores.append(float((delta <= 54).mean()))
        source_i = source_crop.astype(np.int16)
        candidate_i = candidate_crop.astype(np.int16)
        source_blue = (source_i[:, :, 2] >= 170) & (source_i[:, :, 2] >= source_i[:, :, 0] + 70)
        candidate_blue = (candidate_i[:, :, 2] >= 150) & (candidate_i[:, :, 2] >= candidate_i[:, :, 0] + 55)
        key_label_scores.append(float(candidate_blue.sum() / max(1, int(source_blue.sum()))))

    gate_scores: List[float] = []
    gate_label_scores: List[float] = []
    open_color = np.asarray((170, 214, 154), dtype=np.int16)
    for item in record.get("locks") or []:
        if not isinstance(item, dict):
            return {"keys_terminal_state_ok": False, "keys_terminal_state_reason": "malformed gate source"}
        crops = crop_for(item)
        if crops is None:
            return {"keys_terminal_state_ok": False, "keys_terminal_state_reason": "missing gate crop"}
        source_crop, candidate_crop = crops
        delta = np.max(np.abs(candidate_crop.astype(np.int16) - open_color[None, None, :]), axis=2)
        gate_scores.append(float((delta <= 48).mean()))
        source_i = source_crop.astype(np.int16)
        candidate_i = candidate_crop.astype(np.int16)
        source_blue = (source_i[:, :, 2] >= 170) & (source_i[:, :, 2] >= source_i[:, :, 0] + 70)
        candidate_blue = (candidate_i[:, :, 2] >= 150) & (candidate_i[:, :, 2] >= candidate_i[:, :, 0] + 55)
        gate_label_scores.append(float(candidate_blue.sum() / max(1, int(source_blue.sum()))))

    minimum_key = min(key_scores, default=0.0)
    minimum_gate = min(gate_scores, default=0.0)
    minimum_label = min([*key_label_scores, *gate_label_scores], default=0.0)
    # Global blue-ID preservation is checked across the complete terminal
    # image.  The dynamic key/gate crops deliberately repaint their local
    # badge background, so local pixel overlap is diagnostic only.
    state_ok = bool(minimum_key >= 0.12 and minimum_gate >= 0.12)
    return {
        "keys_minimum_collected_key_color_fraction": minimum_key,
        "keys_minimum_open_gate_color_fraction": minimum_gate,
        "keys_minimum_preserved_blue_id_ratio": minimum_label,
        "keys_terminal_state_ok": state_ok,
        "keys_terminal_state_reason": None if state_ok else "key collection, gate opening, or IDs are not visibly preserved",
    }


def _keys_goal_tip_metrics(record: Dict[str, Any], candidate_pixels: np.ndarray, evaluator: object) -> Dict[str, Any]:
    """The completed route still owes a white-ringed red disc on the goal."""
    goal = record.get("goal_point")
    if not isinstance(goal, (list, tuple)) or len(goal) != 2:
        return {"video_terminal_goal_tip_present": False}
    x, y = (int(round(float(value))) for value in goal)
    thickness = max(3, int(record["cell_size"]) // 3)
    core_dy, core_dx, ring_dy, ring_dx, radius = _maze_tip_marker_offsets(thickness)
    mask = np.pad(evaluator._red_mask(candidate_pixels), radius, constant_values=False)
    readings = []
    displacement = _MAZE_ROUTE_MASK_DILATION_SIZE // 2
    height, width = candidate_pixels.shape[:2]
    for cy in range(max(radius, y-displacement), min(height-radius, y+displacement+1)):
        for cx in range(max(radius, x-displacement), min(width-radius, x+displacement+1)):
            core = float(mask[cy + radius + core_dy, cx + radius + core_dx].mean())
            ring = float(mask[cy + radius + ring_dy, cx + radius + ring_dx].mean())
            ring_rgb = candidate_pixels[cy + ring_dy, cx + ring_dx].astype(np.int16)
            ring_white = (ring_rgb.min(axis=1) >= 200) & (np.ptp(ring_rgb, axis=1) <= 55)
            readings.append((core-ring, float(ring_white.mean()), cx, cy))
    valid = [reading for reading in readings
             if reading[0] >= _MAZE_TIP_DETECTION_SCORE and reading[1] >= .80]
    best = max(valid or readings, default=(0., 0., x, y))
    return {"video_terminal_goal_tip_score": best[0],
            "video_terminal_goal_tip_white_ring_fraction": best[1],
            "video_terminal_goal_tip_center": [best[2], best[3]],
            "video_terminal_goal_tip_present": bool(valid)}


def _verify_maze_route_mask_sequence_video(
    sample: CanonicalSample,
    candidate_video: Path,
    metadata_record: Dict[str, Any],
    evaluator: object,
    expected_path_ids: Sequence[int],
    prefix_indices: Sequence[int],
) -> tuple[Dict[str, Any], bool]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for maze video evaluation.") from exc

    puzzle_path = Path(sample.primary_input_image or "").expanduser().resolve()
    if not puzzle_path.exists():
        return {
            "video_process_available": False,
            "video_process_reason": "missing puzzle image for static marker removal",
        }, False
    with Image.open(puzzle_path) as puzzle_image:
        puzzle_pixels = np.asarray(puzzle_image.convert("RGB"))
    initial_red = evaluator._red_mask(puzzle_pixels)
    static_kernel = np.ones(
        (_MAZE_STATIC_RED_DILATION_SIZE, _MAZE_STATIC_RED_DILATION_SIZE),
        dtype=np.uint8,
    )
    static_red_mask = cv2.dilate(initial_red.astype(np.uint8), static_kernel) > 0
    expected_masks = _render_expected_maze_route_masks(
        sample,
        metadata_record,
        evaluator,
        expected_path_ids,
        prefix_indices,
        static_red_mask,
    )
    if len(expected_masks) != len(prefix_indices) or len(expected_masks) < 2:
        return {
            "video_process_available": False,
            "video_process_reason": "failed to render expected maze route-prefix masks",
        }, False

    cap = cv2.VideoCapture(str(candidate_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {candidate_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    classified_indices: List[Optional[int]] = []
    matching_indices_by_frame: List[List[int]] = []
    distances: List[float] = []
    first_frame_distances: Optional[List[float]] = None
    final_frame_distances: Optional[List[float]] = None
    final_candidate_route: Optional[np.ndarray] = None
    baseline_rgb: Optional[np.ndarray] = None
    input_match = False
    input_changed_fraction = 1.0
    native_dimensions_ok = True
    decoded_frames: List[np.ndarray] = []
    frame_route_masks: List[np.ndarray] = []
    frame_red_masks: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        native_dimensions_ok = native_dimensions_ok and frame_rgb.shape[:2] == (512, 512)
        if frame_rgb.shape[:2] != static_red_mask.shape:
            frame_rgb = cv2.resize(
                frame_rgb,
                (static_red_mask.shape[1], static_red_mask.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        if baseline_rgb is None:
            baseline_rgb = frame_rgb.copy()
            input_delta = np.max(
                np.abs(baseline_rgb.astype(np.int16) - puzzle_pixels.astype(np.int16)),
                axis=2,
            )
            input_changed_fraction = float(np.mean(input_delta >= 32))
            input_match = bool(float(visual_delta(baseline_rgb, puzzle_pixels).mean()) <= 8.0)
        candidate_red = evaluator._red_mask(frame_rgb)
        candidate_route = candidate_red & ~static_red_mask
        # Compressed coloured clues can contribute isolated threshold-crossing
        # pixels outside their dilated source mask. They are not a drawn path.
        if int(candidate_route.sum()) <= 8:
            candidate_route[:] = False
        final_candidate_route = candidate_route
        decoded_frames.append(frame_rgb)
        frame_route_masks.append(candidate_route)
        # The tip obligation below reads the marker, which sits on the input's
        # own red endpoints for part of the run, so it needs the red before the
        # static markers are subtracted out of it.
        frame_red_masks.append(candidate_red)
        route_coverages = [
            _maze_route_mask_coverages(candidate_route, expected_mask, cv2)
            for expected_mask in expected_masks
        ]
        frame_distances = [
            1.0 - (candidate_coverage + expected_coverage) / 2.0
            for candidate_coverage, expected_coverage in route_coverages
        ]
        if first_frame_distances is None:
            first_frame_distances = frame_distances
        final_frame_distances = frame_distances
        best_distance = min(frame_distances)
        best_index = frame_distances.index(best_distance)
        matching_indices = [
            index
            for index, distance in enumerate(frame_distances)
            if distance <= best_distance + MAZE_ROUTE_MASK_MATCH_MARGIN
            and distance <= MAZE_ROUTE_MASK_MATCH_THRESHOLD
        ]
        classified_indices.append(best_index if matching_indices else None)
        matching_indices_by_frame.append(matching_indices)
        distances.append(best_distance)
    cap.release()
    if not classified_indices:
        raise RuntimeError("No frames processed from candidate video.")

    expected_last = len(expected_masks) - 1
    adjusted_indices = list(classified_indices)
    recognized_frames = [
        frame_index
        for frame_index, matches in enumerate(matching_indices_by_frame)
        if matches
    ]
    if recognized_frames:
        first_frame = recognized_frames[0]
        last_frame = recognized_frames[-1]
        if 0 in matching_indices_by_frame[first_frame]:
            adjusted_indices[first_frame] = 0
        if expected_last in matching_indices_by_frame[last_frame]:
            adjusted_indices[last_frame] = expected_last

    raw_compact = _compact_keyframe_sequence(adjusted_indices)
    compact = _find_monotonic_keyframe_sequence(
        matching_indices_by_frame,
        len(expected_masks),
    ) or raw_compact
    frame_count = len(classified_indices)
    media_valid = bool(
        native_dimensions_ok
        and abs(fps - 16.0) <= 0.25
        and frame_count <= 161
        and frame_count % 4 == 1
    )
    unrecognized = sum(1 for matches in matching_indices_by_frame if not matches)
    unrecognized_fraction = float(unrecognized / frame_count)
    starts_at_initial = bool(compact and compact[0] == 0)
    ends_at_final = bool(compact and compact[-1] == expected_last)
    initial_hold = bool(
        len(matching_indices_by_frame) >= 5
        and all(0 in matches for matches in matching_indices_by_frame[:5])
    )
    final_hold = bool(
        len(matching_indices_by_frame) >= 5
        and all(expected_last in matches for matches in matching_indices_by_frame[-5:])
    )
    covers_all = set(compact) >= set(range(len(expected_masks)))
    monotonic = all(nxt >= current for current, nxt in zip(compact, compact[1:]))
    no_skips = all(nxt - current <= 1 for current, nxt in zip(compact, compact[1:]))
    initial_endpoint_distance = (
        first_frame_distances[0]
        if first_frame_distances is not None
        else float("inf")
    )
    final_endpoint_distance = (
        final_frame_distances[expected_last]
        if final_frame_distances is not None
        else float("inf")
    )
    if final_candidate_route is None:
        final_candidate_coverage = 0.0
        final_expected_coverage = 0.0
    else:
        final_candidate_coverage, final_expected_coverage = _maze_route_mask_coverages(
            final_candidate_route,
            expected_masks[expected_last],
            cv2,
        )
    final_coverage_threshold = MAZE_ROUTE_MASK_FINAL_COVERAGE_THRESHOLD
    final_route_matches = bool(
        final_endpoint_distance <= MAZE_ROUTE_MASK_ENDPOINT_THRESHOLD
        and final_candidate_coverage >= final_coverage_threshold
        and final_expected_coverage >= final_coverage_threshold
    )
    endpoints_match = bool(initial_endpoint_distance <= MAZE_ROUTE_MASK_ENDPOINT_THRESHOLD
                           and final_route_matches)
    terminal_mechanics = {}
    if sample.task_name == "maze_keys_and_locks":
        terminal_mechanics = {
            **_keys_terminal_state_metrics(metadata_record, puzzle_pixels, decoded_frames[-1]),
            **_keys_goal_tip_metrics(metadata_record, decoded_frames[-1], evaluator),
        }
    # Two whole-video obligations the route-mask classifier above cannot carry:
    # the ordering of the frames as a sequence, and everything outside the red
    # channel it reads.  Both are derived from one source route bank, rendered
    # once here and shared, and from nothing else.
    bank_states = max(_MAZE_PROGRESS_BANK_MIN_STATES, frame_count)
    bank, bank_progress = _maze_progress_bank(
        sample,
        metadata_record,
        evaluator,
        expected_path_ids,
        static_red_mask,
        states=bank_states,
    )
    shape = (int(static_red_mask.shape[0]), int(static_red_mask.shape[1]))
    dynamic = _maze_video_dynamic_mask(
        sample, metadata_record, evaluator, expected_path_ids, shape
    )
    order_metrics, order_pass = _read_maze_route_process_order(
        bank,
        bank_progress,
        _maze_route_order_region(
            sample, metadata_record, evaluator, expected_path_ids, shape
        ),
        frame_route_masks,
        bank_quantum=1.0 / float(bank_states),
    )
    tip_metrics, tip_pass = _verify_maze_tip_advance(
        sample, metadata_record, evaluator, expected_path_ids, frame_red_masks, frame_route_masks,
        observed_progress=order_metrics.get("video_reading_progress"),
        frame_white_masks=[np.min(frame, axis=2) >= 200 for frame in decoded_frames]
        if sample.task_name == "maze_layered" else None,
    )
    if len(bank) >= 3 and bool(np.any(dynamic)):
        static_metrics, static_pass = _verify_maze_static_preservation(
            decoded_frames, puzzle_pixels, ~dynamic
        )
    else:
        static_metrics, static_pass = {
            "video_static_content_preserved": False,
            "video_static_preservation_reason": "failed to render the source route states",
        }, False
    process_pass = bool(media_valid and input_match and starts_at_initial and ends_at_final
                        and covers_all and monotonic and no_skips and endpoints_match
                        and tip_pass and static_pass)
    return {
        **order_metrics,
        **tip_metrics,
        **static_metrics,
        **terminal_mechanics,
        "video_process_available": True,
        "video_expected_keyframe_count": len(expected_masks),
        "video_frame_count": frame_count,
        "video_fps": fps,
        "video_dimensions_ok": native_dimensions_ok,
        "video_media_valid": media_valid,
        "video_recognized_keyframe_sequence": compact,
        "video_unrecognized_frame_count": unrecognized,
        "video_unrecognized_frame_fraction": unrecognized_fraction,
        "video_max_unrecognized_fraction": MAZE_VIDEO_MAX_UNRECOGNIZED_FRACTION,
        "video_starts_at_initial": starts_at_initial,
        "video_ends_at_final": ends_at_final,
        "video_first_frame_matches_input": input_match,
        "video_first_frame_changed_fraction": input_changed_fraction,
        "video_has_initial_hold": initial_hold,
        "video_has_final_hold": final_hold,
        "video_covers_all_keyframes": covers_all,
        "video_monotonic_keyframe_order": monotonic,
        "video_no_keyframe_skips": no_skips,
        "video_route_mask_match_threshold": MAZE_ROUTE_MASK_MATCH_THRESHOLD,
        "video_route_mask_match_margin": MAZE_ROUTE_MASK_MATCH_MARGIN,
        "video_route_mask_endpoint_threshold": MAZE_ROUTE_MASK_ENDPOINT_THRESHOLD,
        "video_route_mask_final_coverage_threshold": final_coverage_threshold,
        "video_initial_route_mask_distance": initial_endpoint_distance,
        "video_final_route_mask_distance": final_endpoint_distance,
        "video_final_candidate_route_coverage": final_candidate_coverage,
        "video_final_expected_route_coverage": final_expected_coverage,
        "video_route_mask_endpoints_match": endpoints_match,
        "video_final_route_mask_matches": final_route_matches,
        "video_min_route_mask_distance": min(distances),
        "video_max_route_mask_distance": max(distances),
    }, process_pass


def _select_spatial_navigation_maze_path_from_video(
    sample: CanonicalSample,
    candidate_video: Path,
    metadata_record: Dict[str, Any],
    evaluator: object,
) -> List[int]:
    """Infer a delivered route from terminal pixels, then attest it to raw semantics."""

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for spatial_navigation-maze video evaluation") from exc
    puzzle_path = Path(sample.primary_input_image or "").expanduser().resolve()
    if not puzzle_path.exists():
        return []
    cap = cv2.VideoCapture(str(candidate_video))
    if not cap.isOpened():
        return []
    final_frame: Optional[np.ndarray] = None
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        final_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    if final_frame is None:
        return []
    with Image.open(puzzle_path) as image:
        puzzle_pixels = np.asarray(image.convert("RGB"))
    if final_frame.shape != puzzle_pixels.shape:
        return []
    static_red = evaluator._red_mask(puzzle_pixels)
    static_red_mask = cv2.dilate(
        static_red.astype(np.uint8),
        np.ones((_MAZE_STATIC_RED_DILATION_SIZE, _MAZE_STATIC_RED_DILATION_SIZE), dtype=np.uint8),
    ) > 0
    candidate_route = evaluator._red_mask(final_frame) & ~static_red_mask
    contract = _spatial_navigation_maze_state_contract(sample.task_name, metadata_record)
    if contract is None:
        return []

    # Find a complete legal state path whose visible edges are supported by the
    # terminal route. This is a weighted state-graph search, not an
    # optimal-move check: a longer candidate route is preferred over a shorter
    # route when the longer route is what the pixels actually show.
    start_state = contract["start_state"]
    best_cost: Dict[Any, float] = {start_state: 0.0}
    best_state_path: Dict[Any, List[Any]] = {start_state: [start_state]}
    tie_breaker = count()
    queue: List[Tuple[float, int, Any]] = [(0.0, next(tie_breaker), start_state)]
    selected_goal: Optional[Any] = None
    edge_score_cache: Dict[Tuple[int, int], float] = {}

    def edge_visual_score(left_state: Any, right_state: Any) -> float:
        left_id = int(contract["state_cell_id"](left_state))
        right_id = int(contract["state_cell_id"](right_state))
        cache_key = (left_id, right_id)
        cached = edge_score_cache.get(cache_key)
        if cached is not None:
            return cached
        edge_masks = _render_expected_maze_route_masks(
            sample,
            metadata_record,
            evaluator,
            [left_id, right_id],
            [1],
            static_red_mask,
        )
        if not edge_masks or not edge_masks[0].any():
            # Cross-panel staircase commits are represented by the stable
            # staircase marker rather than an invented line between panels.
            score = 1.0
        else:
            score = _maze_route_mask_coverages(candidate_route, edge_masks[0], cv2)[1]
        edge_score_cache[cache_key] = score
        return score

    while queue:
        current_cost, _, state = heappop(queue)
        if current_cost > best_cost.get(state, float("inf")) + 1e-9:
            continue
        if contract["is_goal"](state):
            selected_goal = state
            break
        for next_state in contract["neighbors"](state):
            support = edge_visual_score(state, next_state)
            edge_cost = 0.00001 + max(0.0, 0.72 - support) * 10.0
            next_cost = current_cost + edge_cost
            if next_cost + 1e-9 < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = next_cost
                best_state_path[next_state] = [*best_state_path[state], next_state]
                heappush(queue, (next_cost, next(tie_breaker), next_state))

    if selected_goal is None:
        return []
    best_path = [
        int(contract["state_cell_id"](state))
        for state in best_state_path[selected_goal]
    ]
    expected_masks = _render_expected_maze_route_masks(
        sample,
        metadata_record,
        evaluator,
        best_path,
        [0, len(best_path) - 1],
        static_red_mask,
    )
    if len(expected_masks) != 2:
        return []
    candidate_coverage, expected_coverage = _maze_route_mask_coverages(
        candidate_route,
        expected_masks[-1],
        cv2,
    )
    best_score = (
        min(candidate_coverage, expected_coverage),
        (candidate_coverage + expected_coverage) / 2.0,
    )
    if best_score[0] < 0.88 or not best_path:
        return []
    valid, _ = _spatial_navigation_maze_delivery_is_legal(
        sample.task_name,
        metadata_record,
        best_path,
    )
    return best_path if valid else []


def _semantic_path_video_checkpoints(
    task_name: str,
    record: Dict[str, Any],
    path_ids: Sequence[int],
) -> List[int]:
    """Choose bounded semantic commits from the path geometry alone."""

    last = len(path_ids) - 1
    if last <= 0:
        return []
    indices = {0, last}
    interior_slots = min(5, max(0, last - 1))
    for slot in range(1, interior_slots + 1):
        indices.add(max(1, min(last - 1, round(slot * last / (interior_slots + 1)))))
    if task_name == "maze_keys_and_locks":
        event_ids: List[int] = []
        for item in [*(record.get("keys") or []), *(record.get("locks") or [])]:
            if isinstance(item, dict) and item.get("cell_id") is not None:
                try:
                    event_ids.append(int(item["cell_id"]))
                except (TypeError, ValueError):
                    pass
        for event_id in event_ids:
            try:
                indices.add(path_ids.index(event_id))
            except ValueError:
                continue
    return sorted(index for index in indices if 0 <= index <= last)


def _verify_keys_and_locks_video_events(
    sample: CanonicalSample,
    candidate_video: Path,
    record: Dict[str, Any],
    evaluator: object,
    path_ids: Sequence[int],
) -> tuple[Dict[str, Any], bool]:
    """Tie key collection and gate opening to visible route contact order."""

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for key/gate event verification") from exc
    puzzle_path = Path(sample.primary_input_image or "").expanduser().resolve()
    if not puzzle_path.exists() or not path_ids:
        return {"video_key_gate_events_available": False}, False
    with Image.open(puzzle_path) as image:
        source_pixels = np.asarray(image.convert("RGB"))
    bboxes = record.get("cell_bboxes")
    if not isinstance(bboxes, list):
        return {"video_key_gate_events_available": False}, False
    events: List[Dict[str, Any]] = []
    for kind, items in (("key", record.get("keys") or []), ("gate", record.get("locks") or [])):
        for item in items:
            if not isinstance(item, dict):
                return {"video_key_gate_events_available": False}, False
            try:
                cell_id = int(item["cell_id"])
                path_index = list(path_ids).index(cell_id)
                row, col = int(item["cell"][0]), int(item["cell"][1])
                left, top, right, bottom = [int(round(float(value))) for value in bboxes[row][col]]
            except (KeyError, TypeError, ValueError, IndexError):
                return {"video_key_gate_events_available": False}, False
            events.append(
                {
                    "kind": kind,
                    "label": str(item.get("label") or cell_id),
                    "item": item,
                    "path_index": path_index,
                    "bbox": (left, top, right, bottom),
                }
            )
    if not events:
        return {"video_key_gate_events_available": False}, False

    static_red = evaluator._red_mask(source_pixels)
    static_red_mask = cv2.dilate(
        static_red.astype(np.uint8),
        np.ones((_MAZE_STATIC_RED_DILATION_SIZE, _MAZE_STATIC_RED_DILATION_SIZE), dtype=np.uint8),
    ) > 0
    prefix_masks = _render_expected_maze_route_masks(
        sample,
        record,
        evaluator,
        path_ids,
        [0, *[event["path_index"] for event in events]],
        static_red_mask,
    )
    if len(prefix_masks) != len(events) + 1:
        return {"video_key_gate_events_available": False}, False
    for event, mask in zip(events, prefix_masks[1:]):
        event["prefix_mask"] = mask

    cap = cv2.VideoCapture(str(candidate_video))
    if not cap.isOpened():
        return {"video_key_gate_events_available": False}, False
    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame.shape != source_pixels.shape:
            cap.release()
            return {"video_key_gate_events_available": False}, False
        frames.append(frame)
    cap.release()
    if not frames:
        return {"video_key_gate_events_available": False}, False

    event_details: List[Dict[str, Any]] = []
    for event in events:
        left, top, right, bottom = event["bbox"]
        source_crop = source_pixels[top:bottom, left:right]
        if source_crop.size == 0:
            return {"video_key_gate_events_available": False}, False
        active_values: List[float] = []
        reach_values: List[float] = []
        global_prefix_reach_values: List[float] = []
        if event["kind"] == "key":
            try:
                base = np.asarray([int(value) for value in event["item"]["color_rgb"]], dtype=np.int16)
            except (KeyError, TypeError, ValueError):
                return {"video_key_gate_events_available": False}, False
            collected = np.rint(base + (255 - base) * 0.58).astype(np.int16)
            source_base = np.max(
                np.abs(source_crop.astype(np.int16) - base[None, None, :]),
                axis=2,
            ) <= 45
            active_threshold = 0.12
        else:
            collected = np.asarray((170, 214, 154), dtype=np.int16)
            source_base = np.ones(source_crop.shape[:2], dtype=bool)
            # Antialiased scenes measure a ~0.03-0.05 noise floor of
            # open-like colours on closed gates; a real open state measures
            # >= 0.20 of the cell, so 0.12 separates both with margin.
            active_threshold = 0.12
        for frame in frames:
            candidate_route = evaluator._red_mask(frame) & ~static_red_mask
            global_prefix_reach_values.append(
                _maze_route_mask_coverages(
                    candidate_route,
                    event["prefix_mask"],
                    cv2,
                )[1]
            )
            expected_local_route = event["prefix_mask"][top:bottom, left:right]
            candidate_local_route = candidate_route[top:bottom, left:right]
            reach_values.append(
                _maze_route_mask_coverages(
                    candidate_local_route,
                    expected_local_route,
                    cv2,
                )[1]
            )
            crop = frame[top:bottom, left:right]
            target_like = np.max(
                np.abs(crop.astype(np.int16) - collected[None, None, :]),
                axis=2,
            ) <= (50 if event["kind"] == "key" else 48)
            active_values.append(
                float((target_like & source_base).sum() / max(1, int(source_base.sum())))
            )
        # Key/gate glyphs intentionally sit above the route so their identity
        # remains readable.  Measure contact only inside the event cell.  The
        # open-gate glyph can cover more of the red trail than the closed one,
        # so the final frame is not a sound calibration target.  Calibrate the
        # threshold from the local pre-motion noise floor to the maximum
        # coverage attained anywhere in the complete video instead.  A video
        # that never enters the cell has no material rise and therefore cannot
        # manufacture a contact event merely by changing the glyph.
        final_reach_coverage = reach_values[-1] if reach_values else 0.0
        initial_reach_coverage = max(reach_values[: min(4, len(reach_values))], default=0.0)
        peak_reach_coverage = max(reach_values, default=0.0)
        material_reach_rise = peak_reach_coverage - initial_reach_coverage >= 0.025
        reach_threshold = initial_reach_coverage + 0.45 * (
            peak_reach_coverage - initial_reach_coverage
        )
        first_reached = (
            next(
                (index for index, value in enumerate(reach_values) if value >= reach_threshold),
                None,
            )
            if material_reach_rise
            else None
        )
        # A key/gate glyph is intentionally drawn above the route.  For an
        # early event its opaque footprint can cover every red pixel inside
        # that cell even though the source-derived prefix has genuinely
        # reached it.  In that specific occlusion case, use coverage of the
        # complete source-rendered prefix as a second contact observation;
        # this remains candidate-video evidence and cannot be satisfied by a
        # saved route or terminal reference image.
        initial_global_reach = max(
            global_prefix_reach_values[: min(4, len(global_prefix_reach_values))],
            default=0.0,
        )
        peak_global_reach = max(global_prefix_reach_values, default=0.0)
        material_global_reach = peak_global_reach - initial_global_reach >= 0.12
        global_reach_threshold = initial_global_reach + 0.82 * (
            peak_global_reach - initial_global_reach
        )
        first_global_reached = (
            next(
                (
                    index
                    for index, value in enumerate(global_prefix_reach_values)
                    if value >= global_reach_threshold
                ),
                None,
            )
            if material_global_reach
            else None
        )
        contact_source = "local_event_cell"
        if first_reached is None:
            first_reached = first_global_reached
            contact_source = "source_prefix_coverage"
        first_active = next((index for index, value in enumerate(active_values) if value >= active_threshold), None)
        initial_inactive = all(value < active_threshold for value in active_values[: min(4, len(active_values))])
        final_active = all(value >= active_threshold for value in active_values[-min(4, len(active_values)) :])
        monotonic = bool(
            first_active is not None
            and sum(value >= active_threshold for value in active_values[first_active:])
            / max(1, len(active_values) - first_active)
            >= 0.95
        )
        contact_sync = bool(
            first_reached is not None
            and first_active is not None
            and first_reached - 2 <= first_active <= first_reached + 6
        )
        event_ok = bool(initial_inactive and final_active and monotonic and contact_sync)
        event["first_active"] = first_active
        event_details.append(
            {
                "kind": event["kind"],
                "label": event["label"],
                "path_index": event["path_index"],
                "first_route_contact_frame": first_reached,
                "route_contact_source": contact_source,
                "route_contact_coverage_threshold": reach_threshold,
                "initial_route_contact_coverage": initial_reach_coverage,
                "peak_route_contact_coverage": peak_reach_coverage,
                "material_route_contact_rise": material_reach_rise,
                "final_route_contact_coverage": final_reach_coverage,
                "global_prefix_contact_threshold": global_reach_threshold,
                "initial_global_prefix_coverage": initial_global_reach,
                "peak_global_prefix_coverage": peak_global_reach,
                "material_global_prefix_rise": material_global_reach,
                "first_state_update_frame": first_active,
                "initially_inactive": initial_inactive,
                "finally_active": final_active,
                "state_update_monotonic": monotonic,
                "state_update_contact_synchronized": contact_sync,
                "pass": event_ok,
            }
        )
    ordered = sorted(events, key=lambda item: item["path_index"])
    update_order_ok = all(
        left.get("first_active") is not None
        and right.get("first_active") is not None
        and int(left["first_active"]) <= int(right["first_active"])
        for left, right in zip(ordered, ordered[1:])
    )
    passed = bool(all(detail["pass"] for detail in event_details) and update_order_ok)
    return {
        "video_key_gate_events_available": True,
        "video_key_gate_event_details": event_details,
        "video_key_gate_update_order_ok": update_order_ok,
        "video_key_gate_process_pass": passed,
    }, passed


def _verify_maze_path_prefix_video(
    sample: CanonicalSample,
    candidate_video: Path,
    metadata_record: Dict[str, Any],
    evaluator: object,
) -> tuple[Dict[str, Any], bool]:
    started_at = time.perf_counter()
    expected_path_ids = _select_spatial_navigation_maze_path_from_video(
        sample,
        candidate_video,
        metadata_record,
        evaluator,
    )
    prefix_indices = (
        _semantic_path_video_checkpoints(
            sample.task_name,
            metadata_record,
            expected_path_ids,
        )
        if expected_path_ids
        else None
    )
    if prefix_indices is None:
        return {
            "video_process_verifier": _SPATIAL_NAVIGATION_MAZE_VIDEO_PROCESS_VERIFIER,
            "video_process_available": False,
            "video_process_reason": "no independently legal route matches the final frame",
            "video_semantic_process_check_pass": False,
        }, False

    keyframe_metrics, keyframe_pass = _verify_maze_route_mask_sequence_video(
        sample,
        candidate_video,
        metadata_record,
        evaluator,
        expected_path_ids,
        prefix_indices,
    )
    # The route recovered from the candidate frames is already in the internal
    # cell-id space; replay it directly rather than re-serialising it through
    # the model-facing answer grammar.
    route_semantic_ok, route_source_ok = _spatial_navigation_maze_delivery_is_legal(
        sample.task_name,
        metadata_record,
        expected_path_ids,
    )
    path_rule_pass = bool(route_semantic_ok and route_source_ok)
    recognized_keyframes = [
        int(index)
        for index in keyframe_metrics.get("video_recognized_keyframe_sequence") or []
        if isinstance(index, int) and 0 <= index < len(prefix_indices)
    ]
    recognized_prefix_indices = [prefix_indices[index] for index in recognized_keyframes]
    key_gate_metrics: Dict[str, Any] = {}
    key_gate_pass = True
    if sample.task_name == "maze_keys_and_locks":
        key_gate_metrics, key_gate_pass = _verify_keys_and_locks_video_events(
            sample,
            candidate_video,
            metadata_record,
            evaluator,
            expected_path_ids,
        )
    semantic_pass = bool(keyframe_pass and path_rule_pass and key_gate_pass)
    metrics = {
        **keyframe_metrics,
        **key_gate_metrics,
        "video_process_verifier": _SPATIAL_NAVIGATION_MAZE_VIDEO_PROCESS_VERIFIER,
        "video_keyframe_process_check_pass": bool(keyframe_pass),
        "video_inferred_path_cell_ids": expected_path_ids,
        "video_path_source": "terminal_pixels_plus_raw_source_replay",
        "video_saved_canonical_route_used": False,
        "video_reference_pixels_used": False,
        "video_expected_path_prefix_indices": prefix_indices,
        "video_recognized_path_prefix_indices": recognized_prefix_indices,
        "video_path_rule_verifier": _SPATIAL_NAVIGATION_MAZE_VIDEO_PROCESS_VERIFIER,
        "video_path_rule_replay_pass": bool(path_rule_pass),
        "video_semantic_process_check_pass": semantic_pass,
        "candidate_video": candidate_video.as_posix(),
    }
    temporal_parts = [
        bool(keyframe_metrics.get("video_starts_at_initial")),
        bool(keyframe_metrics.get("video_covers_all_keyframes")),
        bool(keyframe_metrics.get("video_monotonic_keyframe_order")),
        bool(keyframe_metrics.get("video_no_keyframe_skips")),
        bool(keyframe_metrics.get("video_static_content_preserved")),
    ]
    if sample.task_name == "maze_keys_and_locks":
        temporal_parts.append(bool(key_gate_pass))
    # Every task whose public instruction requests a cursor carries its verdict.
    if keyframe_metrics.get("video_tip_advance_available"):
        temporal_parts.append(bool(keyframe_metrics.get("video_tip_advance_pass")))
    final_state_ok = bool(
        keyframe_metrics.get("video_ends_at_final")
        and keyframe_metrics.get("video_final_route_mask_matches")
        and path_rule_pass
        and keyframe_metrics.get("keys_terminal_state_ok", True)
        and keyframe_metrics.get("video_terminal_goal_tip_present", True)
    )
    semantic_pass = bool(semantic_pass and final_state_ok)
    metrics["video_semantic_process_check_pass"] = semantic_pass
    scores = {
        "media_validity_score": float(keyframe_metrics.get("video_media_valid", False)),
        "temporal_process_score": float(
            sum(bool(value) for value in temporal_parts) / len(temporal_parts)
        ),
        "final_state_score": float(final_state_ok),
    }
    failures: List[str] = []
    if not keyframe_metrics.get("video_media_valid"):
        failures.append("invalid frame count, dimensions, or fps")
    if not all(temporal_parts):
        failures.append("route prefix process is incomplete, reordered, skipped, or regressive")
    if keyframe_metrics.get("video_tip_advance_pass") is False:
        failures.append(
            "the route tip marker is missing, duplicated, lacks its red core, or outruns the revealed arc"
        )
    if not keyframe_metrics.get("video_static_content_preserved"):
        failures.append("a frame mutates content the source route never changes")
    if not final_state_ok:
        failures.append("incorrect final route or failed rule replay")
    metrics.update(
        _verifier_contract_metrics(
            verifier_profile=_SPATIAL_NAVIGATION_MAZE_VERIFIER_PROFILE,
            artifact_kind="video",
            component_scores=scores,
            binary_correct=semantic_pass,
            failure_reason="; ".join(failures) if failures else "ok",
            started_at=started_at,
            hard_gate_failures=failures,
        )
    )
    return metrics, semantic_pass


@lru_cache(maxsize=None)
def _maze_evaluator_class(task_name: str) -> type[PathTaskEvaluator]:
    spec = TASK_SPECS[task_name]
    if not spec.module.endswith(".generator"):
        raise RuntimeError(f"Task {task_name!r} has no package-local evaluator route")
    module_name = f"{spec.module.rsplit('.', 1)[0]}.evaluator"
    module = importlib.import_module(module_name)
    evaluator_classes = [
        candidate
        for _, candidate in inspect.getmembers(module, inspect.isclass)
        if candidate.__module__ == module_name
        and candidate is not PathTaskEvaluator
        and issubclass(candidate, PathTaskEvaluator)
    ]
    if len(evaluator_classes) != 1:
        class_names = [candidate.__name__ for candidate in evaluator_classes]
        raise RuntimeError(
            f"Expected exactly one maze evaluator in {module_name}; found {len(evaluator_classes)}: {class_names}"
        )
    return evaluator_classes[0]


def verify_path_trace(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: Optional[str] = None,
) -> List[EvalRecord]:
    selected_modality = str(modality or "").strip().lower() or None
    if selected_modality not in {None, "ti2v"}:
        raise ValueError(f"unsupported path-trace modality: {modality!r}")
    records: List[EvalRecord] = []
    evaluator_cache: Dict[Tuple[str, str], object] = {}

    for sample in samples:
        metadata_record_id = resolve_source_record_id(sample)
        record = EvalRecord(
            sample_id=sample.id,
            task_name=sample.task_name,
            input_asset=sample.primary_input_image,
            task_family=sample.task_family,
            split=sample.split,
        )
        try:
            candidate_video = resolve_candidate_video(
                pred_root, sample.id, sample.split
            )
            if candidate_video is None:
                raise FileNotFoundError(f"No prediction video found for sample_id={sample.id}")

            cache_key = (sample.task_name, sample.id)
            evaluator = evaluator_cache.get(cache_key)
            if evaluator is None:
                evaluator_class = _maze_evaluator_class(sample.task_name)
                evaluator_record = dict(resolve_source_record(sample))
                evaluator_record.setdefault("id", metadata_record_id)
                visible_input = str(sample.primary_input_image or "").strip()
                if visible_input:
                    visible_input_path = Path(visible_input).expanduser().resolve()
                    if visible_input_path.is_file():
                        evaluator_record.setdefault("image", visible_input_path.as_posix())
                        evaluator_record.setdefault(
                            "reasoning_image",
                            visible_input_path.as_posix(),
                        )
                evaluator = evaluator_class(
                    evaluator_record,
                    base_dir=resolve_source_task_dir(sample),
                )
                evaluator_cache[cache_key] = evaluator
            metadata_record = evaluator.get_record(metadata_record_id)

            if sample.task_name == "maze_aqueduct_pipes":
                video_metrics, video_pass = _verify_aqueduct_candidate_video(
                    sample,
                    candidate_video,
                    metadata_record,
                    evaluator,
                )
            elif sample.task_name in MAZE_SEMANTIC_PATH_VIDEO_TASKS:
                video_metrics, video_pass = _verify_maze_path_prefix_video(
                    sample,
                    candidate_video,
                    metadata_record,
                    evaluator,
                )
            else:
                video_metrics = {
                    "video_process_verifier": "unavailable",
                    "video_process_available": False,
                    "video_process_reason": "task does not yet have a semantic full-process video verifier",
                    "video_semantic_process_check_pass": False,
                    "candidate_video": candidate_video.as_posix(),
                }
                video_pass = False
            record.prediction_asset = candidate_video.as_posix()
            record.offline_metrics = {
                **video_metrics,
                "text_pass": None,
            }
            record.offline_pass = bool(video_pass)
        except Exception as exc:  # pragma: no cover - guarded by the integration audit
            record.error = str(exc)
            record.offline_pass = False
        records.append(record)

    return records
