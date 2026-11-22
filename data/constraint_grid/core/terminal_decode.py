"""Decode Constraint Grid terminal pixels into an independently checkable witness. TI2V has no companion text. This module groups the family by
visual grammar (cell values, binary fills, placements, edges, or paths) and
uses renderer-derived templates or explicit line geometry. It never reads a
canonical answer, reference terminal image, or reference video."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.constraint_grid.core.render_replay import render_candidate_terminal
from data.constraint_grid.core.scene import standard_board_geometry, standard_cell_centers
from data.constraint_grid.core.semantic_replay import validate_constraint_payload
from data.common.schemas import CanonicalSample


Cell = Tuple[int, int]


def _state(sample: CanonicalSample) -> Dict[str, Any]:
    raw = sample.semantic_input or {}
    state = raw.get("semantic_state") if isinstance(raw, dict) else None
    return dict(state) if isinstance(state, dict) else {}


def _render_bgr(sample: CanonicalSample, payload: Any) -> np.ndarray:
    image = render_candidate_terminal(sample, payload)
    if image is None:
        raise ValueError("terminal renderer is unavailable")
    return np.asarray(image.convert("RGB"), dtype=np.uint8)[:, :, ::-1].copy()


def _cells(value: Any) -> List[Cell]:
    if not isinstance(value, list):
        return []
    result: List[Cell] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return []
        result.append((int(item[0]), int(item[1])))
    return result


def _grid(value: Any) -> List[List[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("visible grid is unavailable")
    result: List[List[int]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError("visible grid is malformed")
        result.append([0 if cell is None else int(cell) for cell in row])
    if not result or any(len(row) != len(result[0]) for row in result):
        raise ValueError("visible grid is not rectangular")
    return result


def _template_mask(images: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack(images).astype(np.int16)
    spread = np.max(stack, axis=0) - np.min(stack, axis=0)
    return np.max(spread, axis=2) > 5


def _template_score(candidate: np.ndarray, template: np.ndarray, mask: np.ndarray) -> float:
    if candidate.shape != template.shape or not np.any(mask):
        return float("inf")
    error = candidate.astype(np.float32) - template.astype(np.float32)
    return float(np.sqrt(np.mean(error[mask] ** 2)) / 255.0)


def _decode_variables(
    sample: CanonicalSample,
    candidate: np.ndarray,
    payload: Any,
    variables: Sequence[Any],
    domains: Callable[[Any], Sequence[Any]],
    setter: Callable[[Any, Any, Any], None],
) -> Tuple[Any, Dict[str, Any]]:
    current = copy.deepcopy(payload)
    margins: List[float] = []
    worst_score = 0.0
    for variable in variables:
        values = list(domains(variable))
        if not values:
            raise ValueError(f"empty visual domain for {variable!r}")
        trials: List[Any] = []
        images: List[np.ndarray] = []
        for value in values:
            trial = copy.deepcopy(current)
            setter(trial, variable, value)
            trials.append(trial)
            images.append(_render_bgr(sample, trial))
        mask = _template_mask(images)
        if not np.any(mask):
            # A fixed clue can fully cover its underlying answer cell.  The
            # caller should normally exclude it, but leaving the current value
            # is safer than manufacturing a visual observation.
            continue
        scores = [_template_score(candidate, image, mask) for image in images]
        order = sorted(range(len(scores)), key=scores.__getitem__)
        best = order[0]
        current = trials[best]
        worst_score = max(worst_score, scores[best])
        margins.append((scores[order[1]] - scores[best]) if len(order) > 1 else 1.0)
    return current, {
        "decoder_variable_count": len(variables),
        "decoder_template_worst_rmse": worst_score,
        "decoder_template_min_margin": min(margins, default=0.0),
    }


def _grid_setter(payload: Any, variable: Cell, value: Any) -> None:
    payload[variable[0]][variable[1]] = value


def _decode_binary_grid(sample: CanonicalSample, candidate: np.ndarray, state: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    task = sample.task_name
    if task in {"aquarium_completion", "yin_yang_completion"}:
        size = int(state.get("size") or len(state.get("row_sums") or state.get("given") or []))
        rows = cols = size
    else:
        rows = int(state.get("rows") or len(state.get("row_counts") or []))
        cols = int(state.get("cols") or len(state.get("col_counts") or []))
    if rows <= 0 or cols <= 0:
        raise ValueError("binary-grid dimensions are unavailable")
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    fixed: Dict[Cell, int] = {}
    if task == "yin_yang_completion":
        fixed = {(int(r), int(c)): int(value) for r, c, value in state.get("given") or []}
    elif task == "aquarium_completion":
        # Given water and dry cells are visible constraints; decode only the
        # open cells and let the semantic replay enforce given consistency.
        fixed = {(int(r), int(c)): 1 for r, c in state.get("given_filled") or []}
        fixed.update({(int(r), int(c)): 0 for r, c in state.get("given_empty") or []})
    for (row, col), value in fixed.items():
        grid[row][col] = value
    if task == "nurikabe_completion":
        # Given sea and island cells are visible constraints; clue cells are
        # always island.
        fixed.update({(int(r), int(c)): 1 for r, c in state.get("given_black") or []})
        fixed.update({(int(r), int(c)): 0 for r, c in state.get("given_white") or []})
        fixed.update({(int(item[0]), int(item[1])): 0 for item in state.get("clues") or []})
        for (row, col), value in fixed.items():
            grid[row][col] = value
    excluded = set(fixed)
    variables = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in excluded]
    return _decode_variables(sample, candidate, grid, variables, lambda _: (0, 1), _grid_setter)


def _decode_fillomino(sample: CanonicalSample, candidate: np.ndarray, state: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Read removed Fillomino digits without treating derived region color as evidence."""

    current = _grid(state["puzzle"])
    variables = _cells(state.get("removed_cells"))
    rows, cols = len(current), len(current[0])
    centers = _standard_centers(rows, cols)
    cell_size, _left, _top = standard_board_geometry(rows, cols)
    margins: List[float] = []
    for variable in variables:
        images: List[np.ndarray] = []
        trials: List[List[List[int]]] = []
        # Every legal witness digit is bounded by the visible clue range; a
        # single-glyph domain keeps the per-cell template sweep tractable at
        # the supersampled render quality.
        for value in range(1, 10):
            trial = copy.deepcopy(current)
            trial[variable[0]][variable[1]] = value
            trials.append(trial)
            images.append(_render_bgr(sample, trial))
        x, y = centers[variable]
        radius = max(10, cell_size // 4)
        local = np.zeros(candidate.shape[:2], dtype=bool)
        local[max(0, y - radius): y + radius + 1, max(0, x - radius): x + radius + 1] = True
        stack = np.stack(images)
        spread = np.max(stack.astype(np.int16), axis=0) - np.min(stack.astype(np.int16), axis=0)
        # Region colors and thick component borders are derived from the whole
        # grid.  Only dark digit strokes inside the cell center carry the
        # candidate's numeric witness.
        dark_in_any = np.min(np.mean(stack, axis=3), axis=0) < 175
        mask = (np.max(spread, axis=2) > 5) & local & dark_in_any
        scores = [_template_score(candidate, image, mask) for image in images]
        order = sorted(range(len(scores)), key=scores.__getitem__)
        current = trials[order[0]]
        margins.append(scores[order[1]] - scores[order[0]])
    return current, {
        "decoder_mode": "fillomino_local_digit_templates",
        "decoder_variable_count": len(variables),
        "decoder_template_min_margin": min(margins, default=0.0),
    }


def _decode_subset(
    sample: CanonicalSample,
    candidate: np.ndarray,
    candidates: Sequence[Cell],
    *,
    fixed: Sequence[Cell] = (),
) -> Tuple[List[List[int]], Dict[str, Any]]:
    current: List[List[int]] = [list(cell) for cell in fixed]
    margins: List[float] = []
    worst = 0.0
    for cell in candidates:
        if cell in set(fixed):
            continue
        absent = copy.deepcopy(current)
        present = copy.deepcopy(current) + [list(cell)]
        images = [_render_bgr(sample, absent), _render_bgr(sample, present)]
        mask = _template_mask(images)
        if not np.any(mask):
            continue
        scores = [_template_score(candidate, image, mask) for image in images]
        if scores[1] < scores[0]:
            current = present
        worst = max(worst, min(scores))
        margins.append(abs(scores[1] - scores[0]))
    return current, {
        "decoder_mode": "placement_templates",
        "decoder_variable_count": len(candidates),
        "decoder_template_worst_rmse": worst,
        "decoder_template_min_margin": min(margins, default=0.0),
    }


def _decode_placements(sample: CanonicalSample, candidate: np.ndarray, state: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    task = sample.task_name
    if task == "campsite_completion":
        rows, cols = int(state["rows"]), int(state["cols"])
        # Trees stay in internal zero-based coordinates; the decoded payload is
        # published in the one-based contract the renderer and semantic
        # verifier both convert back.  Given tents are fixed input evidence.
        forbidden = set(_cells(state.get("trees")))
        fixed: List[Cell] = [(r + 1, c + 1) for r, c in _cells(state.get("given_tents"))]
    elif task == "light_up_akari_completion":
        rows, cols = int(state["rows"]), int(state["cols"])
        forbidden = set(_cells(state.get("black_cells")))
        centers = _standard_centers(rows, cols)
        rgb = candidate[:, :, ::-1].astype(np.float32)
        target = np.asarray((250, 200, 70), dtype=np.float32)
        lights: List[List[int]] = []
        for row in range(rows):
            for col in range(cols):
                if (row, col) in forbidden:
                    continue
                x, y = centers[(row, col)]
                patch = rgb[max(0, y - 12): y + 13, max(0, x - 12): x + 13]
                close = np.sqrt(np.sum((patch - target) ** 2, axis=2)) < 58
                if float(np.mean(close)) >= 0.08:
                    # Walls and centers stay on the internal zero-based board;
                    # the decoded payload is published in the one-based contract
                    # the renderer and semantic verifier both convert back.
                    lights.append([row + 1, col + 1])
        return lights, {
            "decoder_mode": "akari_bulb_color_centers",
            "decoder_variable_count": rows * cols - len(forbidden),
        }
    elif task == "nqueens_completion":
        rows = cols = int(state["board_size"])
        forbidden = set()
        # Decode against the public one-based coordinate contract.  The
        # renderer converts these candidates back to internal zero-based
        # cells, while the semantic verifier performs the inverse conversion
        # for rule checking.
        fixed = [(row + 1, col + 1) for row, col in _cells(state.get("fixed_positions"))]
    elif task == "star_battle_completion":
        rows = cols = int(state["size"])
        forbidden = set()
        # Every cell is a candidate and the decoded payload is published in the
        # public one-based contract that the renderer and semantic verifier
        # both convert back to internal zero-based cells.  Given stars are
        # fixed input evidence.
        fixed = [(r + 1, c + 1) for r, c in _cells(state.get("given_stars"))]
    else:
        raise ValueError(f"no placement decoder for {task}")
    if task in {"campsite_completion", "nqueens_completion", "star_battle_completion"}:
        candidates = [(r + 1, c + 1) for r in range(rows) for c in range(cols) if (r, c) not in forbidden]
    else:
        candidates = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in forbidden]
    return _decode_subset(sample, candidate, candidates, fixed=fixed)


def _decode_slitherlink(sample: CanonicalSample, candidate: np.ndarray, state: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    rows, cols = int(state["rows"]), int(state["cols"])
    # Candidate edges are enumerated directly in the one-based lattice-corner
    # contract the renderer and the semantic verifier both convert back.
    candidates: List[Tuple[Cell, Cell]] = []
    for row in range(1, rows + 2):
        for col in range(1, cols + 1):
            candidates.append(((row, col), (row, col + 1)))
    for row in range(1, rows + 1):
        for col in range(1, cols + 2):
            candidates.append(((row, col), (row + 1, col)))
    current: List[Tuple[Cell, Cell]] = []
    margins: List[float] = []
    for edge in candidates:
        trials = [copy.deepcopy(current), copy.deepcopy(current) + [edge]]
        images = [_render_bgr(sample, trial) for trial in trials]
        mask = _template_mask(images)
        scores = [_template_score(candidate, image, mask) for image in images]
        if scores[1] < scores[0]:
            current = trials[1]
        margins.append(abs(scores[1] - scores[0]))
    return current, {
        "decoder_mode": "slitherlink_edge_templates",
        "decoder_variable_count": len(candidates),
        "decoder_template_min_margin": min(margins, default=0.0),
    }


def _standard_centers(rows: int, cols: int) -> Dict[Cell, Tuple[int, int]]:
    # Single-sourced from the family scene module so decoder geometry can
    # never drift from renderer geometry.
    return standard_cell_centers(rows, cols)


def _colored_edge_present(rgb: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    x, y = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
    patch = rgb[max(0, y - 4): y + 5, max(0, x - 4): x + 5]
    if patch.size == 0:
        return False
    red, green, blue = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    selected = (red > 135) & (red.astype(np.int16) > green.astype(np.int16) + 55) & (red.astype(np.int16) > blue.astype(np.int16) + 55)
    return float(np.mean(selected)) >= 0.12


def _path_from_edges(edges: Sequence[Tuple[Cell, Cell]], start: Optional[Cell] = None, end: Optional[Cell] = None) -> List[List[int]]:
    adjacency: Dict[Cell, List[Cell]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    if not adjacency:
        return []
    if start is None:
        start = min(adjacency)
    path = [start]
    previous: Optional[Cell] = None
    current = start
    for _ in range(len(adjacency) + 2):
        options = [cell for cell in adjacency[current] if cell != previous]
        if end is None and len(path) > 2 and start in options:
            path.append(start)
            return [list(cell) for cell in path]
        if not options:
            break
        next_cell = min(options)
        previous, current = current, next_cell
        path.append(current)
        if end is not None and current == end:
            return [list(cell) for cell in path]
    return [list(cell) for cell in path]


def _decode_center_path(sample: CanonicalSample, candidate: np.ndarray, state: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    rows, cols = int(state["rows"]), int(state["cols"])
    centers = _standard_centers(rows, cols)
    rgb = candidate[:, :, ::-1]
    edges: List[Tuple[Cell, Cell]] = []
    for row in range(rows):
        for col in range(cols):
            cell = row, col
            for neighbor in ((row + 1, col), (row, col + 1)):
                if neighbor in centers and _colored_edge_present(rgb, centers[cell], centers[neighbor]):
                    edges.append((cell, neighbor))
    if sample.task_name == "masyu_completion":
        # The edge graph stays on the internal zero-based board; the decoded
        # loop is published in the public one-based contract that the renderer
        # and semantic verifier both convert back.
        payload = [[row + 1, col + 1] for row, col in _path_from_edges(edges)]
    else:
        payload = _path_from_edges(edges)
    return payload, {
        "decoder_mode": "red_center_edge_graph",
        "decoder_edge_count": len(edges),
    }


def _decode_flow_pairs(
    sample: CanonicalSample,
    candidate: np.ndarray,
    state: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Recover the per-color routes from cell-center and edge-midpoint pixels.

    Every cell of a legal terminal belongs to exactly one color, so each cell
    center is classified against the visible endpoint swatches, and two
    adjacent same-color cells are route-connected exactly when the stroke
    crosses their shared border.  The decoded object is published in the
    public one-based contract that the renderer and the semantic verifier
    both convert back to internal zero-based cells.
    """

    rows, cols = int(state["rows"]), int(state["cols"])
    endpoints_raw = state.get("endpoints")
    if not isinstance(endpoints_raw, list) or not endpoints_raw:
        raise ValueError("visible flow endpoints are unavailable")
    swatches: Dict[str, Tuple[int, int, int]] = {}
    endpoint_cells: Dict[str, List[Cell]] = {}
    for entry in endpoints_raw:
        color = str(entry.get("color") or "").strip().lower()
        raw_rgb = entry.get("color_rgb")
        cells = entry.get("cells")
        if not color or not isinstance(raw_rgb, list) or not isinstance(cells, list) or len(cells) != 2:
            raise ValueError("visible flow endpoints are malformed")
        swatches[color] = (int(raw_rgb[0]), int(raw_rgb[1]), int(raw_rgb[2]))
        endpoint_cells[color] = [(int(cell[0]), int(cell[1])) for cell in cells]

    cell_size, _left, _top = standard_board_geometry(rows, cols)
    centers = _standard_centers(rows, cols)
    rgb = candidate[:, :, ::-1].astype(np.float32)
    masks = {
        color: np.sum((rgb - np.asarray(swatch, dtype=np.float32)) ** 2, axis=2) <= 72.0**2
        for color, swatch in swatches.items()
    }

    assignment: Dict[Cell, str] = {}
    min_support = None
    patch = max(4, cell_size // 6)
    for row in range(rows):
        for col in range(cols):
            x, y = centers[(row, col)]
            window = (slice(max(0, y - patch), y + patch + 1), slice(max(0, x - patch), x + patch + 1))
            supports = {color: int(mask[window].sum()) for color, mask in masks.items()}
            best_color, best = max(supports.items(), key=lambda item: item[1])
            second = sorted(supports.values(), reverse=True)[1] if len(supports) > 1 else 0
            if best < 45 or best < second + 20:
                raise ValueError(
                    f"cell ({row + 1}, {col + 1}) shows no dominant route color"
                )
            assignment[(row, col)] = best_color
            min_support = best if min_support is None else min(min_support, best)

    edge_patch = max(3, cell_size // 12)
    edges_by_color: Dict[str, List[Tuple[Cell, Cell]]] = {color: [] for color in swatches}
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            for neighbor in ((row + 1, col), (row, col + 1)):
                if neighbor not in centers or assignment[cell] != assignment[neighbor]:
                    continue
                color = assignment[cell]
                x1, y1 = centers[cell]
                x2, y2 = centers[neighbor]
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                window = masks[color][
                    max(0, my - edge_patch) : my + edge_patch + 1,
                    max(0, mx - edge_patch) : mx + edge_patch + 1,
                ]
                if int(window.sum()) >= max(5, int(window.size * 0.35)):
                    edges_by_color[color].append((cell, neighbor))

    payload: Dict[str, List[List[int]]] = {}
    for color, pair in endpoint_cells.items():
        cells_of_color = [cell for cell, assigned in assignment.items() if assigned == color]
        degrees: Dict[Cell, int] = {cell: 0 for cell in cells_of_color}
        adjacency: Dict[Cell, List[Cell]] = {cell: [] for cell in cells_of_color}
        for a, b in edges_by_color[color]:
            adjacency[a].append(b)
            adjacency[b].append(a)
            degrees[a] += 1
            degrees[b] += 1
        start, goal = pair
        if degrees.get(start) != 1 or degrees.get(goal) != 1:
            raise ValueError(f"{color} route does not terminate at both of its dots")
        if any(
            degree != (1 if cell in {start, goal} else 2)
            for cell, degree in degrees.items()
        ):
            raise ValueError(f"{color} route pixels do not form one simple path")
        path = [start]
        previous: Optional[Cell] = None
        current = start
        while current != goal and len(path) <= len(cells_of_color):
            options = [cell for cell in adjacency[current] if cell != previous]
            if len(options) != 1:
                raise ValueError(f"{color} route pixels do not form one simple path")
            previous, current = current, options[0]
            path.append(current)
        if current != goal or len(path) != len(cells_of_color):
            raise ValueError(f"{color} route does not visit every {color} cell")
        payload[color] = [[row + 1, col + 1] for row, col in path]
    return payload, {
        "decoder_mode": "flow_center_color_edge_graph",
        "decoder_variable_count": rows * cols,
        "decoder_min_cell_color_support": int(min_support or 0),
        "decoder_edge_count": sum(len(edges) for edges in edges_by_color.values()),
    }


def _shikaku_solutions(
    state: Mapping[str, Any],
    *,
    limit: int = 128,
) -> List[List[Tuple[Cell, Cell]]]:
    rows, cols = int(state["rows"]), int(state["cols"])
    clues = [(int(r), int(c), int(area)) for r, c, area in state.get("clues") or []]
    options: List[List[Tuple[int, int, int, int]]] = []
    clue_cells = {(r, c) for r, c, _ in clues}
    given = [tuple(map(int, rect)) for rect in state.get("given_rectangles") or []]
    fixed_cells = [
        (rect, {(r, c) for r in range(rect[0], rect[2] + 1)
                for c in range(rect[1], rect[3] + 1)})
        for rect in given
    ]
    for row, col, area in clues:
        candidates: List[Tuple[int, int, int, int]] = []
        for height in range(1, area + 1):
            if area % height:
                continue
            width = area // height
            for top in range(max(0, row - height + 1), min(row, rows - height) + 1):
                for left in range(max(0, col - width + 1), min(col, cols - width) + 1):
                    rect = (top, left, top + height - 1, left + width - 1)
                    inside = {(r, c) for r in range(rect[0], rect[2] + 1) for c in range(rect[1], rect[3] + 1)}
                    if inside & (clue_cells - {(row, col)}):
                        continue
                    if any(inside & cells and rect != fixed for fixed, cells in fixed_cells):
                        continue
                    candidates.append(rect)
        options.append(candidates)
    order = sorted(range(len(clues)), key=lambda index: len(options[index]))
    selected: Dict[int, Tuple[int, int, int, int]] = {}
    covered: set[Cell] = set()
    solutions: List[List[Tuple[Cell, Cell]]] = []

    def search(depth: int) -> None:
        if len(solutions) >= limit:
            return
        if depth == len(order):
            if len(covered) == rows * cols:
                # The exact cover search stays on the internal zero-based board;
                # decoded payloads are published in the one-based contract that
                # the renderer and the semantic verifier both convert back.
                solutions.append(
                    [
                        (
                            (selected[index][0] + 1, selected[index][1] + 1),
                            (selected[index][2] + 1, selected[index][3] + 1),
                        )
                        for index in range(len(clues))
                    ]
                )
            return
        index = order[depth]
        for rect in options[index]:
            cells = {(r, c) for r in range(rect[0], rect[2] + 1) for c in range(rect[1], rect[3] + 1)}
            if cells & covered:
                continue
            selected[index] = rect
            covered.update(cells)
            search(depth + 1)
            covered.difference_update(cells)
            selected.pop(index, None)

    search(0)
    return solutions


def _decode_shikaku(sample: CanonicalSample, candidate: np.ndarray, state: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    solutions = _shikaku_solutions(state)
    if not solutions:
        raise ValueError("visible Shikaku clues have no exact-cover solution")
    scores = [float(np.sqrt(np.mean((_render_bgr(sample, payload).astype(np.float32) - candidate.astype(np.float32)) ** 2)) / 255.0) for payload in solutions]
    best = min(range(len(scores)), key=scores.__getitem__)
    return solutions[best], {
        "decoder_mode": "shikaku_visible_exact_cover",
        "decoder_solution_count_capped": len(solutions),
        "decoder_template_worst_rmse": scores[best],
    }


_BINARY_GRID_TASKS = {
    "aquarium_completion", "nurikabe_completion", "yin_yang_completion",
}
_PLACEMENT_TASKS = {
    "campsite_completion", "light_up_akari_completion", "nqueens_completion",
    "star_battle_completion",
}


def _tuple_coordinate_payload(task: str, payload: Any) -> Any:
    """Publish decoded coordinates in the same tuple grammar as text answers."""

    if task in _PLACEMENT_TASKS or task in {"masyu_completion", "yajilin_completion"}:
        return [(int(row), int(col)) for row, col in payload]
    if task == "maze_flow_pairs":
        return {
            str(color): [(int(row), int(col)) for row, col in path]
            for color, path in payload.items()
        }
    if task in {"shikaku_completion", "slitherlink_completion"}:
        return [
            (
                (int(first[0]), int(first[1])),
                (int(second[0]), int(second[1])),
            )
            for first, second in payload
        ]
    return payload


def decode_constraint_terminal_payload(
    sample: CanonicalSample,
    candidate_bgr: np.ndarray,
) -> Tuple[Any, Dict[str, Any], bool, str]:
    """Recover and independently validate the terminal witness in a TI2V frame."""

    state = _state(sample)
    try:
        if sample.task_name == "fillomino_completion":
            payload, diagnostics = _decode_fillomino(sample, candidate_bgr, state)
        elif sample.task_name in _BINARY_GRID_TASKS:
            payload, diagnostics = _decode_binary_grid(sample, candidate_bgr, state)
        elif sample.task_name in _PLACEMENT_TASKS:
            payload, diagnostics = _decode_placements(sample, candidate_bgr, state)
        elif sample.task_name == "slitherlink_completion":
            payload, diagnostics = _decode_slitherlink(sample, candidate_bgr, state)
        elif sample.task_name == "masyu_completion":
            payload, diagnostics = _decode_center_path(sample, candidate_bgr, state)
        elif sample.task_name == "yajilin_completion":
            payload, diagnostics = _decode_center_path(sample, candidate_bgr, state)
            # The center-edge graph stays on the internal zero-based board; the
            # decoded loop is published in the one-based contract the renderer
            # and the semantic verifier both convert back.
            payload = [[int(row) + 1, int(col) + 1] for row, col in payload]
        elif sample.task_name == "maze_flow_pairs":
            payload, diagnostics = _decode_flow_pairs(sample, candidate_bgr, state)
        elif sample.task_name == "shikaku_completion":
            payload, diagnostics = _decode_shikaku(sample, candidate_bgr, state)
        else:
            return None, {}, False, f"no terminal decoder is registered for {sample.task_name}"
        payload = _tuple_coordinate_payload(sample.task_name, payload)
        valid, reason, signature = validate_constraint_payload(sample, payload)
        diagnostics = {
            **diagnostics,
            "decoded_payload_signature": signature,
            "decoded_payload_semantic_check_pass": bool(valid),
        }
        return payload, diagnostics, bool(valid), reason
    except Exception as exc:
        return None, {"decoder_error": str(exc)}, False, str(exc)


__all__ = ["decode_constraint_terminal_payload"]
