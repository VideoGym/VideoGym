"""Task-owned source replay and visual observations."""

from __future__ import annotations

import numpy as np
from data.common.pipeline.structured_grid import INTEGER_GRID_CELL_PX, integer_grid_token_window
from data.common.verifiers.media import (
    _video_fps,
    _video_frames_bgr,
)
from data.common.verifiers.metrics import _verifier_contract_metrics
from data.common.verifiers.observations import (
    _analyze_fixed_visual_state_sequence,
    _classify_printed_number,
    _pixels_near_rgb,
    _printed_number_templates,
    static_mutation_fraction as measure_static_mutation,
)
from data.common.verifiers.source import _grid_dimensions, _rectangular_grid, _to_int_grid
from pathlib import Path
from time import perf_counter
from typing import Any, List, Mapping, Optional, Sequence


def _spiral_coordinates(rows: int, cols: int) -> List[tuple[int, int]]:
    coordinates: List[tuple[int, int]] = []
    top = 0
    bottom = rows - 1
    left = 0
    right = cols - 1
    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            coordinates.append((top, col))
        top += 1
        for row in range(top, bottom + 1):
            coordinates.append((row, right))
        right -= 1
        if top <= bottom:
            for col in range(right, left - 1, -1):
                coordinates.append((bottom, col))
            bottom -= 1
        if left <= right:
            for row in range(bottom, top - 1, -1):
                coordinates.append((row, left))
            left += 1
    return coordinates


def _spiral_order(matrix: Sequence[Sequence[int]]) -> List[int]:
    rows, cols = _grid_dimensions(matrix)
    return [int(matrix[row][col]) for row, col in _spiral_coordinates(rows, cols)]


_SPIRAL_MATRIX_VERIFIER_PROFILE = "spiral_matrix_printed_number_path_source_replay_v5"


_SPIRAL_MATRIX_MAX_UNRECOGNIZED_FRACTION = 0.42


def _spiral_matrix_expected_geometry(rows: int) -> tuple[int, List[int]]:
    """Board placement replayed from the task's own geometry helper.

    Imported lazily so the shared verifier keeps no second copy of the shared
    integer-grid placement rules; the generator and this replay therefore cannot
    disagree about where a cell is.
    """

    if rows not in INTEGER_GRID_CELL_PX:
        return 0, []
    from data.in_context_learning.spiral_matrix.generator import spiral_grid_geometry

    geometry = spiral_grid_geometry(rows)
    return geometry.cell_px, list(geometry.bbox)


def _validate_spiral_matrix(structured: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    raw_matrix = structured.get("input_matrix")
    input_matrix = _to_int_grid(structured.get("input_matrix"))
    grid_bbox = structured.get("grid_bbox")
    number_token_geometry = structured.get("number_token_geometry")
    rows, cols = _grid_dimensions(input_matrix or [])
    values = [value for row in (input_matrix or []) for value in row]
    expected_coordinates = _spiral_coordinates(rows, cols) if rows > 0 and cols > 0 else []
    # Board placement comes from the shared integer-grid primitive, so the
    # generator and this replay cannot drift into two geometries.
    expected_cell_size, expected_bbox = _spiral_matrix_expected_geometry(rows)
    if expected_cell_size > 0:
        token_width, token_height = integer_grid_token_window(expected_cell_size)
        expected_token_geometry: dict[str, Any] = {"width": token_width, "height": token_height}
    else:
        expected_token_geometry = {}
    spiral_language = _spiral_matrix_render_language()
    expected_path_stroke = {
        "color": list(spiral_language["path_color"]),
        "width": spiral_language["path_width"],
    }
    raw_matrix_is_integer_grid = bool(
        isinstance(raw_matrix, list)
        and raw_matrix
        and all(
            isinstance(row, list)
            and row
            and all(isinstance(value, int) and not isinstance(value, bool) for value in row)
            for row in raw_matrix
        )
    )
    valid = bool(
        _rectangular_grid(input_matrix)
        and raw_matrix_is_integer_grid
        and rows == cols
        and 3 <= rows <= 5
        and len(values) == rows * rows
        and len(set(values)) == len(values)
        and all(10 <= value <= 98 for value in values)
        and len(expected_coordinates) == rows * rows
        and len(set(expected_coordinates)) == len(expected_coordinates)
        and set(expected_coordinates) == {(row, col) for row in range(rows) for col in range(cols)}
        and grid_bbox == expected_bbox
        and structured.get("cell_size") == expected_cell_size
        and number_token_geometry == expected_token_geometry
        and structured.get("path_stroke") == expected_path_stroke
    )
    if not valid:
        return {
            "video_semantic_process_check_pass": False,
            "video_semantic_process_reason": "spiral-matrix order mismatch",
        }, False
    return {
        "video_semantic_process_check_pass": True,
        "video_semantic_process_reason": "ok",
    }, True


def _spiral_matrix_render_language() -> dict[str, Any]:
    """Import the spiral renderer's fills and path style, never a duplicate."""

    from data.common.pipeline.structured_grid import (
        GRID_ACCENT_FILL,
        GRID_CELL_FILL,
    )
    from data.in_context_learning.spiral_matrix.generator import (
        _PATH_COLOR,
        _PATH_WIDTH,
        _VISITED_FILL,
    )

    return {
        "palette": {
            "unvisited": tuple(GRID_CELL_FILL),
            "visited": tuple(_VISITED_FILL),
            "active": tuple(GRID_ACCENT_FILL),
        },
        "path_color": tuple(_PATH_COLOR),
        "path_width": int(_PATH_WIDTH),
    }


_SPIRAL_MATRIX_FILL_TOLERANCE = 11.0


def _spiral_matrix_expected_states(rows: int, cols: int) -> tuple[List[tuple[int, int]], List[List[List[str]]]]:
    coordinates = _spiral_coordinates(rows, cols)
    states: List[List[List[str]]] = [
        [["unvisited" for _col in range(cols)] for _row in range(rows)]
    ]
    for step_index, active in enumerate(coordinates):
        visited = set(coordinates[:step_index])
        states.append(
            [
                [
                    "active"
                    if (row, col) == active
                    else "visited"
                    if (row, col) in visited
                    else "unvisited"
                    for col in range(cols)
                ]
                for row in range(rows)
            ]
        )
    states.append([["visited" for _col in range(cols)] for _row in range(rows)])
    return coordinates, states


def _classify_spiral_matrix_frame(
    frame_bgr: np.ndarray,
    *,
    grid_bbox: Sequence[Any],
    cell_size: float,
    rows: int,
    cols: int,
) -> Optional[List[List[str]]]:
    if len(grid_bbox) != 4 or cell_size <= 0:
        return None
    try:
        canonical_left = float(grid_bbox[0])
        canonical_top = float(grid_bbox[1])
    except Exception:
        return None
    height, width = frame_bgr.shape[:2]
    scale_x = width / 512.0
    scale_y = height / 512.0
    parsed: List[List[str]] = []
    for row in range(rows):
        parsed_row: List[str] = []
        for col in range(cols):
            x0 = int(round((canonical_left + col * cell_size) * scale_x))
            x1 = int(round((canonical_left + (col + 1) * cell_size) * scale_x))
            y0 = int(round((canonical_top + row * cell_size) * scale_y))
            y1 = int(round((canonical_top + (row + 1) * cell_size) * scale_y))
            # Read the fill from the four cell corners.  The printed digits sit
            # at the cell centre and the traced path runs through it as a cross,
            # so the corners are the only sub-regions guaranteed to be pure
            # fill; sampling them keeps answer-bearing ink out of the state vote.
            quadrant_x = max(3, int(round(cell_size * 0.26 * scale_x)))
            quadrant_y = max(3, int(round(cell_size * 0.26 * scale_y)))
            inset_x = max(4, int(round(cell_size * 0.10 * scale_x)))
            inset_y = max(4, int(round(cell_size * 0.10 * scale_y)))
            corners = [
                frame_bgr[
                    max(0, y0 + inset_y):min(height, y0 + inset_y + quadrant_y),
                    max(0, x0 + inset_x):min(width, x0 + inset_x + quadrant_x),
                ],
                frame_bgr[
                    max(0, y0 + inset_y):min(height, y0 + inset_y + quadrant_y),
                    max(0, x1 - inset_x - quadrant_x):min(width, x1 - inset_x),
                ],
                frame_bgr[
                    max(0, y1 - inset_y - quadrant_y):min(height, y1 - inset_y),
                    max(0, x0 + inset_x):min(width, x0 + inset_x + quadrant_x),
                ],
                frame_bgr[
                    max(0, y1 - inset_y - quadrant_y):min(height, y1 - inset_y),
                    max(0, x1 - inset_x - quadrant_x):min(width, x1 - inset_x),
                ],
            ]
            if any(corner.size == 0 for corner in corners):
                return None
            scores = sorted(
                (
                    sum(
                        int(
                            _pixels_near_rgb(
                                corner,
                                color,
                                tolerance=_SPIRAL_MATRIX_FILL_TOLERANCE,
                            ).sum()
                        )
                        for corner in corners
                    ),
                    category,
                )
                for category, color in _spiral_matrix_render_language()["palette"].items()
            )
            best_count, best_category = scores[-1]
            second_count = scores[-2][0]
            sampled = sum(corner.shape[0] * corner.shape[1] for corner in corners)
            if best_count < max(18, int(round(sampled * 0.5))) or best_count < second_count * 1.5:
                return None
            parsed_row.append(best_category)
        parsed.append(parsed_row)
    return parsed


def _spiral_matrix_path_edges(
    frame_bgr: np.ndarray,
    *,
    grid_bbox: Sequence[Any],
    cell_size: float,
    rows: int,
    cols: int,
) -> Optional[set[frozenset[tuple[int, int]]]]:
    """Return the drawn path's edge set, one independent test per grid edge.

    Every orthogonally adjacent cell pair gets its own corridor mask and is
    accepted only if the stroke is continuous along that whole corridor.  The
    caller compares the returned set against the source-replayed set, so a
    single missing or invented segment fails on its own instead of being
    averaged away by the segments that are present.
    """

    if len(grid_bbox) != 4 or cell_size <= 0 or rows <= 0 or cols <= 0:
        return None
    try:
        left = float(grid_bbox[0])
        top = float(grid_bbox[1])
    except Exception:
        return None
    height, width = frame_bgr.shape[:2]
    scale_x = width / 512.0
    scale_y = height / 512.0
    spiral_language = _spiral_matrix_render_language()
    # Stroke opacity and codec chroma can change its RGB distance without
    # changing the blue route. The source palette separates the blue stroke
    # from pale blue unread cells, green visited cells, and the gold marker.
    blue, green, red = frame_bgr.astype(np.int16).transpose(2, 0, 1)
    painted = (blue - red >= 25) & (blue - green >= 12)
    radius = max(2, int(round(spiral_language["path_width"] * min(scale_x, scale_y))))

    def centre(cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        return (
            (left + (col + 0.5) * cell_size) * scale_x,
            (top + (row + 0.5) * cell_size) * scale_y,
        )

    edges: set[frozenset[tuple[int, int]]] = set()
    for row in range(rows):
        for col in range(cols):
            for delta_row, delta_col in ((0, 1), (1, 0)):
                neighbour = (row + delta_row, col + delta_col)
                if neighbour[0] >= rows or neighbour[1] >= cols:
                    continue
                start_x, start_y = centre((row, col))
                end_x, end_y = centre(neighbour)
                # Sample the middle of the corridor only: the two ends belong to
                # the neighbouring cells and would register for any incident
                # segment, which is exactly the cross-talk that lets one
                # segment vouch for another.
                covered = True
                for step in range(20, 81):
                    fraction = step / 100.0
                    sample_x = int(round(start_x + (end_x - start_x) * fraction))
                    sample_y = int(round(start_y + (end_y - start_y) * fraction))
                    window = painted[
                        max(0, sample_y - radius):sample_y + radius + 1,
                        max(0, sample_x - radius):sample_x + radius + 1,
                    ]
                    if window.size == 0 or not bool(window.any()):
                        covered = False
                        break
                if covered:
                    edges.add(frozenset({(row, col), neighbour}))
    return edges


def _spiral_matrix_expected_path_edges(
    coordinates: Sequence[tuple[int, int]],
    *,
    prefix_length: Optional[int] = None,
) -> set[frozenset[tuple[int, int]]]:
    """Source-replayed edge set for a traversal prefix."""

    limit = len(coordinates) if prefix_length is None else max(0, int(prefix_length))
    walked = list(coordinates[:limit])
    return {frozenset({a, b}) for a, b in zip(walked, walked[1:])}


def _classify_spiral_matrix_values(
    frame_bgr: np.ndarray,
    *,
    grid_bbox: Sequence[Any],
    cell_size: float,
    rows: int,
    cols: int,
    token_width: float,
    token_height: float,
    templates: Mapping[int, np.ndarray],
) -> Optional[List[List[int]]]:
    if len(grid_bbox) != 4 or cell_size <= 0:
        return None
    try:
        left = float(grid_bbox[0])
        top = float(grid_bbox[1])
    except Exception:
        return None
    parsed: List[List[int]] = []
    for row in range(rows):
        parsed_row: List[int] = []
        for col in range(cols):
            cell_left = left + col * cell_size
            cell_top = top + row * cell_size
            value = _classify_printed_number(
                frame_bgr,
                center_x=cell_left + cell_size * 0.5,
                center_y=cell_top + cell_size * 0.5,
                token_width=token_width,
                token_height=token_height,
                templates=templates,
            )
            if value is None:
                return None
            parsed_row.append(value)
        parsed.append(parsed_row)
    return parsed


def _verify_spiral_matrix_candidate_state_video(
    video_path: Path,
    structured: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    started_at = perf_counter()
    semantic_metrics, semantic_pass = _validate_spiral_matrix(structured)
    input_matrix = _to_int_grid(structured.get("input_matrix"))
    grid_bbox = structured.get("grid_bbox")
    try:
        cell_size = float(structured.get("cell_size"))
        bbox_values = [float(value) for value in grid_bbox]
    except Exception:
        cell_size = 0.0
        bbox_values = []
    rows, cols = _grid_dimensions(input_matrix or [])
    values = [value for row in (input_matrix or []) for value in row]
    expected_cell_size, expected_bbox = _spiral_matrix_expected_geometry(rows)
    if (
        not semantic_pass
        or not _rectangular_grid(input_matrix)
        or rows != cols
        or not 3 <= rows <= 5
        or any(value < 10 or value > 98 for value in values)
        or len(set(values)) != len(values)
        or not isinstance(grid_bbox, (list, tuple))
        or len(grid_bbox) != 4
        or bbox_values != [float(value) for value in expected_bbox]
        or cell_size != expected_cell_size
    ):
        scores = {
            "source_replay_score": 0.0,
            "coordinate_coverage_score": 0.0,
            "adjacency_score": 0.0,
            "path_segment_score": 0.0,
            "value_preservation_score": 0.0,
            "visual_parseability_score": 0.0,
            "state_order_score": 0.0,
            "state_cadence_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
            "media_contract_score": 0.0,
        }
        contract_metrics = _verifier_contract_metrics(
            verifier_profile=_SPIRAL_MATRIX_VERIFIER_PROFILE,
            artifact_kind="full_video",
            component_scores=scores,
            binary_correct=False,
            failure_reason="missing or invalid spiral-matrix source, trace, or candidate-state metadata",
            started_at=started_at,
            hard_gate_failures=["source_replay"],
        )
        return {
            **semantic_metrics,
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": "missing spiral-matrix candidate-state metadata",
            **contract_metrics,
        }, False

    coordinates, expected_states = _spiral_matrix_expected_states(rows, cols)
    # One expected path-edge set per semantic state: nothing drawn before the
    # first read, then one further segment per read, holding complete at the end.
    expected_edge_sets = [
        _spiral_matrix_expected_path_edges(coordinates, prefix_length=0),
        *[
            _spiral_matrix_expected_path_edges(coordinates, prefix_length=step + 1)
            for step in range(len(coordinates))
        ],
        _spiral_matrix_expected_path_edges(coordinates),
    ]
    try:
        frames = _video_frames_bgr(video_path)
    except Exception as exc:
        scores = {
            "source_replay_score": 1.0,
            "coordinate_coverage_score": 1.0,
            "adjacency_score": 1.0,
            "path_segment_score": 0.0,
            "value_preservation_score": 0.0,
            "visual_parseability_score": 0.0,
            "state_order_score": 0.0,
            "state_cadence_score": 0.0,
            "preservation_score": 0.0,
            "final_state_score": 0.0,
            "media_contract_score": 0.0,
        }
        reason = f"spiral-matrix video could not be decoded: {exc}"
        return {
            **semantic_metrics,
            "video_candidate_state_check_pass": False,
            "video_candidate_state_reason": reason,
            "video_candidate_state_verifier": "spiral_matrix_read_state_grid_and_traced_path_parser",
            "video_candidate_frame_count": 0,
            "video_candidate_recognized_frame_count": 0,
            "video_candidate_unrecognized_fraction": 1.0,
            "video_spiral_matrix_frame_format_valid": False,
            "video_media_contract_valid": False,
            **_verifier_contract_metrics(
                verifier_profile=_SPIRAL_MATRIX_VERIFIER_PROFILE,
                artifact_kind="full_video",
                component_scores=scores,
                binary_correct=False,
                failure_reason=reason,
                started_at=started_at,
                hard_gate_failures=["video_decode"],
            ),
        }, False
    recognized: List[
        tuple[int, List[List[str]], Optional[List[List[int]]], Optional[set[frozenset[tuple[int, int]]]]]
    ] = []
    static_mutation_fractions: List[float] = []
    reference = frames[0] if frames else np.zeros((1, 1, 3), dtype=np.uint8)
    token_geometry = structured.get("number_token_geometry") or {}
    token_width = float(token_geometry.get("width", 0.0))
    token_height = float(token_geometry.get("height", 0.0))
    templates = _printed_number_templates(
        reference,
        matrix=input_matrix,
        grid_bbox=grid_bbox,
        cell_size=cell_size,
        token_width=token_width,
        token_height=token_height,
    )
    allowed_change_bboxes = [
        [float(grid_bbox[0]) - 12, float(grid_bbox[1]) - 12, float(grid_bbox[2]) + 12, float(grid_bbox[3]) + 12],
    ]
    for frame_index, frame in enumerate(frames):
        state = _classify_spiral_matrix_frame(
            frame,
            grid_bbox=grid_bbox,
            cell_size=cell_size,
            rows=rows,
            cols=cols,
        )
        parsed_grid = _classify_spiral_matrix_values(
            frame,
            grid_bbox=grid_bbox,
            cell_size=cell_size,
            rows=rows,
            cols=cols,
            token_width=token_width,
            token_height=token_height,
            templates=templates or {},
        )
        parsed_edges = _spiral_matrix_path_edges(
            frame,
            grid_bbox=grid_bbox,
            cell_size=cell_size,
            rows=rows,
            cols=cols,
        )
        # The traversal is a discrete state machine, so every frame must be one
        # of the exact semantic states.  A frame that is not is an interpolated,
        # faded or otherwise undeclared state and stays unrecognized, which the
        # cadence and parseability gates below then reject.
        if state is not None and state in expected_states:
            expected_state_index = expected_states.index(state)
            if parsed_grid == input_matrix and parsed_edges == expected_edge_sets[expected_state_index]:
                recognized.append((frame_index, state, parsed_grid, parsed_edges))
        static_mutation_fractions.append(
            measure_static_mutation(
                reference,
                frame,
                allowed_change_bboxes=allowed_change_bboxes,
            )
        )

    recognized_pairs = [(frame_index, state) for frame_index, state, _grid, _edges in recognized]
    frame_count = len(frames)
    sequence = _analyze_fixed_visual_state_sequence(
        frame_count=frame_count,
        recognized=recognized_pairs,
        expected_states=expected_states,
    )
    observed_states = sequence["observed_states"]
    recognized_count = sequence["recognized_count"]
    unrecognized_fraction = sequence["unrecognized_fraction"]
    starts_at_input = sequence["starts_at_input"]
    ends_at_final = sequence["ends_at_final"]
    has_initial_hold = sequence["has_initial_hold"]
    has_final_hold = sequence["has_final_hold"]
    expected_indices = sequence["expected_indices"]
    skipped_steps = sequence["skipped_layers"]
    exact_sequence = sequence["exact_sequence"]
    # The underlying state is discrete, so the transition model is discrete too:
    # every traversal state occupies the same number of consecutive frames and
    # there is no frame between two states.  Unequal dwell, or any gap, means an
    # interpolated or faded transition was rendered instead of a state switch.
    state_groups = sequence["compact_groups"]
    inter_state_gaps = [
        current["frame_indices"][0] - previous["frame_indices"][-1] - 1
        for previous, current in zip(state_groups, state_groups[1:])
    ]
    traversal_dwells = [len(group["frame_indices"]) for group in state_groups[1:-1]]
    state_cadence_uniform = bool(
        len(state_groups) == len(expected_states)
        and traversal_dwells
        and len(set(traversal_dwells)) == 1
        and all(gap == 0 for gap in inter_state_gaps)
    )

    # Per-state path evidence: each recognized frame already had to match its
    # state's exact expected edge set, so this records how many traversal states
    # carried a complete, correct traced prefix.
    path_prefix_groups = [
        group["frame_indices"][0]
        for group in state_groups
    ]
    path_prefix_exact = bool(exact_sequence and len(state_groups) == len(expected_states))
    terminal_edges = (
        recognized[-1][3] if recognized else None
    )
    expected_terminal_edges = _spiral_matrix_expected_path_edges(coordinates)
    missing_terminal_edges = sorted(
        tuple(sorted(edge)) for edge in expected_terminal_edges - (terminal_edges or set())
    )
    extra_terminal_edges = sorted(
        tuple(sorted(edge)) for edge in (terminal_edges or set()) - expected_terminal_edges
    )
    path_segments_complete = bool(
        terminal_edges is not None and not missing_terminal_edges and not extra_terminal_edges
    )
    path_pass = bool(path_prefix_exact and path_segments_complete)

    parsed_grids = [parsed_grid for _frame, _state, parsed_grid, _edges in recognized]
    grid_values_parseable = bool(parsed_grids) and all(parsed_grid is not None for parsed_grid in parsed_grids)
    grid_values_stable = bool(
        grid_values_parseable
        and all(parsed_grid == input_matrix for parsed_grid in parsed_grids)
    )
    max_static_mutation_fraction = max(static_mutation_fractions, default=1.0)
    fps = _video_fps(video_path)
    frame_format_valid = bool(
        frame_count > 0
        and abs(fps - 16.0) <= 0.05
        and frame_count % 4 == 1
        and frame_count / max(fps, 1e-9) <= 10.0
    )

    observed_active_coordinates: List[List[int]] = []
    for state in observed_states[1:-1]:
        active_cells = [
            [row, col]
            for row in range(rows)
            for col in range(cols)
            if state[row][col] == "active"
        ]
        if len(active_cells) == 1:
            observed_active_coordinates.append(active_cells[0])

    passed = (
        semantic_pass
        and unrecognized_fraction <= _SPIRAL_MATRIX_MAX_UNRECOGNIZED_FRACTION
        and starts_at_input
        and ends_at_final
        and has_initial_hold
        and has_final_hold
        and exact_sequence
        and state_cadence_uniform
        and grid_values_stable
        and path_pass
        and max_static_mutation_fraction <= 0.005
        and frame_format_valid
    )
    if unrecognized_fraction > _SPIRAL_MATRIX_MAX_UNRECOGNIZED_FRACTION:
        reason = "spiral-matrix cells are not visually parseable in enough frames"
    elif skipped_steps:
        reason = "spiral-matrix traversal skips a required cell"
    elif not exact_sequence:
        reason = "spiral-matrix current-cell sequence is incomplete or out of order"
    elif not grid_values_parseable:
        reason = "spiral-matrix printed grid values are not source-matchable"
    elif not grid_values_stable:
        reason = "spiral-matrix grid values mutate during traversal"
    elif missing_terminal_edges:
        reason = "spiral-matrix traced path is missing a required reading segment"
    elif extra_terminal_edges:
        reason = "spiral-matrix traces a segment the clockwise reading never takes"
    elif not path_prefix_exact:
        reason = "spiral-matrix traced path does not extend by exactly one segment per read"
    elif not starts_at_input or not ends_at_final:
        reason = "spiral-matrix input or final state mismatch"
    elif not has_initial_hold or not has_final_hold:
        reason = "spiral-matrix initial or final hold is missing"
    elif not state_cadence_uniform:
        reason = "spiral-matrix states are not held for one equal duration each, or an undeclared transition frame appears between them"
    elif max_static_mutation_fraction > 0.005:
        reason = "spiral-matrix board, background, or camera mutates"
    elif not frame_format_valid:
        reason = "spiral-matrix video must be 16fps, at most 10 seconds, and use a 4N+1 frame count"
    else:
        reason = "ok"
    coordinate_coverage_ok = bool(len(coordinates) == rows * cols and len(set(coordinates)) == len(coordinates))
    adjacency_ok = bool(
        coordinates
        and all(
            abs(row_a - row_b) + abs(col_a - col_b) == 1
            for (row_a, col_a), (row_b, col_b) in zip(coordinates, coordinates[1:])
        )
    )
    scores = {
        "source_replay_score": float(semantic_pass),
        "coordinate_coverage_score": float(coordinate_coverage_ok),
        "adjacency_score": float(adjacency_ok),
        "path_segment_score": float(path_pass),
        "value_preservation_score": float(grid_values_stable),
        "visual_parseability_score": 1.0 - min(1.0, unrecognized_fraction),
        "state_order_score": float(exact_sequence),
        "state_cadence_score": float(state_cadence_uniform),
        "preservation_score": float(max_static_mutation_fraction <= 0.005),
        "final_state_score": float(ends_at_final and has_final_hold),
        "media_contract_score": float(frame_format_valid),
    }
    failures = [] if passed else [
        "whole_video_semantic_parse"
        if unrecognized_fraction > _SPIRAL_MATRIX_MAX_UNRECOGNIZED_FRACTION
        else "state_order"
        if not exact_sequence
        else "value_preservation"
        if not grid_values_stable
        else "terminal_path_segment_coverage"
        if not path_pass
        else "ordered_discrete_state_cadence"
        if not state_cadence_uniform
        else "fixed_content_preservation"
        if max_static_mutation_fraction > 0.005
        else "media_contract"
        if not frame_format_valid
        else "semantic_video",
    ]
    return {
        **semantic_metrics,
        "video_candidate_state_check_pass": passed,
        "video_candidate_state_reason": reason,
        "video_candidate_state_verifier": "spiral_matrix_source_printed_number_and_traced_path_parser",
        "video_candidate_frame_count": frame_count,
        "video_candidate_recognized_frame_count": recognized_count,
        "video_candidate_unrecognized_fraction": unrecognized_fraction,
        "video_candidate_max_unrecognized_fraction": _SPIRAL_MATRIX_MAX_UNRECOGNIZED_FRACTION,
        "video_spiral_matrix_expected_state_count": len(expected_states),
        "video_spiral_matrix_observed_state_count": len(observed_states),
        "video_spiral_matrix_expected_coordinates": [[row, col] for row, col in coordinates],
        "video_spiral_matrix_observed_active_coordinates": observed_active_coordinates,
        "video_spiral_matrix_expected_state_indices": expected_indices,
        "video_spiral_matrix_starts_at_input": starts_at_input,
        "video_spiral_matrix_ends_at_final": ends_at_final,
        "video_spiral_matrix_has_initial_hold": has_initial_hold,
        "video_spiral_matrix_has_final_hold": has_final_hold,
        "video_spiral_matrix_skipped_steps": skipped_steps,
        "video_spiral_matrix_exact_sequence": exact_sequence,
        "video_spiral_matrix_inter_state_gaps": inter_state_gaps,
        "video_spiral_matrix_traversal_state_dwells": traversal_dwells,
        "video_spiral_matrix_state_cadence_uniform": state_cadence_uniform,
        "video_spiral_matrix_state_first_frames": path_prefix_groups,
        "video_spiral_matrix_expected_path_segments": len(expected_terminal_edges),
        "video_spiral_matrix_observed_path_segments": len(terminal_edges or ()),
        "video_spiral_matrix_missing_path_segments": [list(map(list, edge)) for edge in missing_terminal_edges[:20]],
        "video_spiral_matrix_extra_path_segments": [list(map(list, edge)) for edge in extra_terminal_edges[:20]],
        "video_spiral_matrix_path_prefix_exact": path_prefix_exact,
        "video_spiral_matrix_path_pass": path_pass,
        "video_spiral_matrix_grid_values_parseable": grid_values_parseable,
        "video_spiral_matrix_grid_values_stable": grid_values_stable,
        "video_spiral_matrix_max_static_mutation_fraction": max_static_mutation_fraction,
        "video_spiral_matrix_frame_format_valid": frame_format_valid,
        "video_fps": fps,
        "video_media_contract_valid": frame_format_valid,
        **_verifier_contract_metrics(
            verifier_profile=_SPIRAL_MATRIX_VERIFIER_PROFILE,
            artifact_kind="full_video",
            component_scores=scores,
            binary_correct=passed,
            failure_reason=reason,
            started_at=started_at,
            hard_gate_failures=failures,
        ),
    }, passed
