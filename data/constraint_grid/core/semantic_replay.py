"""Independent final-text replay for the Constraint Grid family.

Only visible clues and executable rules from ``semantic_input`` are consumed.
Saved solution grids, paths, masks, missing values, and answer labels are never
used to decide correctness.  A candidate is accepted when its typed payload is
itself a complete legal witness, so equivalent grids, unordered placements,
and reversed/rotated loop encodings remain valid where the task permits them.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from data.common.grid_coordinates import parse_visible_named_xy_paths_answer
from data.common.schemas import CanonicalSample
from data.common.typed_answers import parse_answer_line, parse_matrix_answer


Cell = Tuple[int, int]


def _state(sample: CanonicalSample) -> Dict[str, Any]:
    raw = sample.semantic_input or {}
    state = raw.get("semantic_state") if isinstance(raw, dict) else None
    return dict(state) if isinstance(state, dict) else {}


def _parse_answer_line(
    text: str,
    *,
    matrix: bool = False,
) -> Tuple[Any, bool, str]:
    if matrix:
        payload = parse_matrix_answer(text)
        if payload is None:
            return None, False, (
                "matrix answer must use exact canonical JSON with integer cells, "
                "one space after every comma, and a final period"
            )
        return payload, True, "ok"
    named_paths = parse_visible_named_xy_paths_answer(text)
    if named_paths is not None:
        return named_paths, True, "ok"
    body = parse_answer_line(text)
    if body is None:
        return None, False, "answer must be exactly one `Answer: <typed payload>.` line"
    try:
        return json.loads(body), True, "ok"
    except json.JSONDecodeError:
        try:
            literal = ast.literal_eval(body)
        except (SyntaxError, ValueError):
            literal = None
        if isinstance(literal, (list, dict, tuple)):
            return literal, True, "ok"
        if re.fullmatch(r"-?\d+", body):
            return int(body), True, "ok"
        if re.fullmatch(r"[A-Z][A-Z0-9_-]*", body):
            return body, True, "ok"
        return None, False, (
            "typed payload is not a valid matrix, canonical coordinate structure, integer, "
            "or uppercase label"
        )


def _int_grid(value: Any, rows: Optional[int] = None, cols: Optional[int] = None) -> Optional[List[List[int]]]:
    if not isinstance(value, list) or not value:
        return None
    grid: List[List[int]] = []
    for raw_row in value:
        if not isinstance(raw_row, list) or not raw_row:
            return None
        row: List[int] = []
        for cell in raw_row:
            if isinstance(cell, bool) or not isinstance(cell, int):
                return None
            row.append(cell)
        grid.append(row)
    if not grid or any(len(row) != len(grid[0]) for row in grid):
        return None
    if rows is not None and len(grid) != int(rows):
        return None
    if cols is not None and len(grid[0]) != int(cols):
        return None
    return grid


def _cells(value: Any, *, width: int = 2) -> Optional[List[Tuple[int, ...]]]:
    if not isinstance(value, list):
        return None
    out: List[Tuple[int, ...]] = []
    try:
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != width:
                return None
            out.append(tuple(int(v) for v in item))
    except (TypeError, ValueError):
        return None
    return out


def _visible_cells(value: Any) -> Optional[List[Cell]]:
    """Read only the public tuple spelling ``[(x, y), ...]``."""

    if not isinstance(value, list):
        return None
    cells: List[Cell] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            return None
        if any(isinstance(part, bool) or not isinstance(part, int) for part in item):
            return None
        cells.append((int(item[0]), int(item[1])))
    return cells


def _visible_composites(value: Any) -> Optional[List[Tuple[Cell, Cell]]]:
    """Read only ``[((x1, y1), (x2, y2)), ...]`` composites."""

    if not isinstance(value, list):
        return None
    composites: List[Tuple[Cell, Cell]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            return None
        first, second = item
        if (
            not isinstance(first, tuple)
            or not isinstance(second, tuple)
            or len(first) != 2
            or len(second) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, int)
                for part in (*first, *second)
            )
        ):
            return None
        composites.append(
            ((int(first[0]), int(first[1])), (int(second[0]), int(second[1])))
        )
    return composites


def _in_bounds(cell: Cell, rows: int, cols: int) -> bool:
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols


def _neighbors4(cell: Cell, rows: int, cols: int) -> Iterable[Cell]:
    row, col = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        candidate = row + dr, col + dc
        if _in_bounds(candidate, rows, cols):
            yield candidate


def _connected(cells: Set[Cell], rows: int, cols: int) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        cell = stack.pop()
        for neighbor in _neighbors4(cell, rows, cols):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == cells


def _binary_grid(value: Any, rows: int, cols: int) -> Optional[List[List[int]]]:
    grid = _int_grid(value, rows, cols)
    return grid if grid is not None and all(cell in {0, 1} for row in grid for cell in row) else None


def _cycle(payload: Any, rows: int, cols: int) -> Optional[List[Cell]]:
    path = _cells(payload)
    if path is None or len(path) < 5:
        return None
    cells = [(int(row), int(col)) for row, col in path]
    if cells[0] != cells[-1] or len(set(cells[:-1])) != len(cells) - 1:
        return None
    if any(not _in_bounds(cell, rows, cols) for cell in cells):
        return None
    if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(cells, cells[1:])):
        return None
    return cells


def _no_black_2x2(grid: Sequence[Sequence[int]], black: int = 1) -> bool:
    return all(
        not all(grid[row + dr][col + dc] == black for dr in (0, 1) for dc in (0, 1))
        for row in range(len(grid) - 1)
        for col in range(len(grid[0]) - 1)
    )


def _validate_aquarium(payload: Any, state: Mapping[str, Any]) -> bool:
    n = int(state.get("size") or 0)
    grid = _binary_grid(payload, n, n)
    regions = state.get("regions")
    row_counts = state.get("row_counts")
    col_counts = state.get("col_counts")
    if grid is None or not isinstance(regions, list) or not isinstance(row_counts, list) or not isinstance(col_counts, list):
        return False
    # Cells shown as water or marked dry in the input are fixed constraints.
    try:
        for r, c in state.get("given_filled") or []:
            if grid[int(r)][int(c)] != 1:
                return False
        for r, c in state.get("given_empty") or []:
            if grid[int(r)][int(c)] != 0:
                return False
    except (TypeError, ValueError, IndexError):
        return False
    if [sum(row) for row in grid] != [int(v) for v in row_counts] or [sum(grid[r][c] for r in range(n)) for c in range(n)] != [int(v) for v in col_counts]:
        return False
    covered: Set[Cell] = set()
    for raw_region in regions:
        cells = _cells(raw_region)
        if cells is None or not cells:
            return False
        region = {(r, c) for r, c in cells}
        if covered & region or any(not _in_bounds(cell, n, n) for cell in region):
            return False
        covered |= region
        by_row: Dict[int, Set[int]] = defaultdict(set)
        for row, col in region:
            by_row[row].add(grid[row][col])
        if any(len(values) != 1 for values in by_row.values()):
            return False
        rows = sorted(by_row)
        levels = [next(iter(by_row[row])) for row in rows]
        if any(left > right for left, right in zip(levels, levels[1:])):
            return False
    return covered == {(r, c) for r in range(n) for c in range(n)}


def _connected_graph(nodes: Set[Cell], adjacency: Mapping[Cell, Set[Cell]]) -> bool:
    if not nodes:
        return False
    seen = {next(iter(nodes))}
    stack = list(seen)
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == nodes


def _perfect_matching(left: Sequence[Cell], right: Sequence[Cell]) -> bool:
    graph = {cell: [other for other in right if abs(cell[0] - other[0]) + abs(cell[1] - other[1]) == 1] for cell in left}
    match: Dict[Cell, Cell] = {}

    def augment(cell: Cell, seen: Set[Cell]) -> bool:
        for other in graph[cell]:
            if other in seen:
                continue
            seen.add(other)
            if other not in match or augment(match[other], seen):
                match[other] = cell
                return True
        return False

    return len(left) == len(right) and all(augment(cell, set()) for cell in left)


def _validate_campsite(payload: Any, state: Mapping[str, Any]) -> bool:
    rows, cols = int(state.get("rows") or 0), int(state.get("cols") or 0)
    trees_raw = _cells(state.get("trees"))
    visible = _visible_cells(payload)
    if trees_raw is None or visible is None:
        return False
    # Candidates submit 1-based (row, column) ordinals; replay
    # on the internal 0-based board.  Out-of-range ordinals fail closed, which
    # also rejects the legacy 0-based payload whenever it names row or column 0.
    if any(not (1 <= row <= rows and 1 <= col <= cols) for row, col in visible):
        return False
    positions = [(row - 1, col - 1) for row, col in visible]
    if len(set(positions)) != len(positions):
        return False
    tents = set(positions)
    trees = set(trees_raw)
    if any(cell in trees for cell in tents):
        return False
    if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for i, a in enumerate(positions) for b in positions[i + 1 :]):
        return False
    if [sum((r, c) in tents for c in range(cols)) for r in range(rows)] != [int(v) for v in state.get("row_counts") or []]:
        return False
    if [sum((r, c) in tents for r in range(rows)) for c in range(cols)] != [int(v) for v in state.get("col_counts") or []]:
        return False
    # Tents already pitched in the input are fixed.
    try:
        for r, c in state.get("given_tents") or []:
            if (int(r), int(c)) not in tents:
                return False
    except (TypeError, ValueError):
        return False
    return _perfect_matching(sorted(trees), sorted(tents))


def _validate_fillomino(payload: Any, state: Mapping[str, Any]) -> bool:
    rows, cols = int(state.get("rows") or 0), int(state.get("cols") or 0)
    grid = _int_grid(payload, rows, cols)
    puzzle = _int_grid(state.get("puzzle"), rows, cols)
    if grid is None or puzzle is None or any(value <= 0 or value > rows * cols for row in grid for value in row):
        return False
    if any(puzzle[r][c] not in {0, grid[r][c]} for r in range(rows) for c in range(cols)):
        return False
    unseen = {(r, c) for r in range(rows) for c in range(cols)}
    while unseen:
        start = next(iter(unseen))
        value = grid[start[0]][start[1]]
        component = {start}
        stack = [start]
        unseen.remove(start)
        while stack:
            cell = stack.pop()
            for neighbor in _neighbors4(cell, rows, cols):
                if neighbor in unseen and grid[neighbor[0]][neighbor[1]] == value:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        if len(component) != value:
            return False
    return True


def _validate_light_up(payload: Any, state: Mapping[str, Any]) -> bool:
    rows, cols = int(state.get("rows") or 0), int(state.get("cols") or 0)
    visible_cells = _visible_cells(payload)
    black_raw = _cells(state.get("black_cells"))
    if visible_cells is None or black_raw is None:
        return False
    # Candidates submit 1-based (row, column) ordinals; replay
    # on the internal 0-based board.  Out-of-range ordinals fail closed, which
    # also rejects the legacy 0-based payload whenever it names row or column 0.
    if any(not (1 <= row <= rows and 1 <= col <= cols) for row, col in visible_cells):
        return False
    lights_raw = [(row - 1, col - 1) for row, col in visible_cells]
    if len(lights_raw) != len(set(lights_raw)):
        return False
    lights, black = set(lights_raw), set(black_raw)
    if any(cell in black for cell in lights):
        return False
    def visible(a: Cell, b: Cell) -> bool:
        if a[0] == b[0]:
            return all((a[0], c) not in black for c in range(min(a[1], b[1]) + 1, max(a[1], b[1])))
        if a[1] == b[1]:
            return all((r, a[1]) not in black for r in range(min(a[0], b[0]) + 1, max(a[0], b[0])))
        return False
    if any(visible(a, b) for i, a in enumerate(lights_raw) for b in lights_raw[i + 1 :]):
        return False
    for row in range(rows):
        for col in range(cols):
            cell = row, col
            if cell not in black and not any(cell == light or visible(cell, light) for light in lights):
                return False
    # Bulbs already shown in the input are fixed.
    try:
        for r, c in state.get("given_bulbs") or []:
            if (int(r), int(c)) not in lights:
                return False
    except (TypeError, ValueError):
        return False
    try:
        return all(sum(neighbor in lights for neighbor in _neighbors4((int(r), int(c)), rows, cols)) == int(value) for r, c, value in state.get("clues") or [])
    except (TypeError, ValueError):
        return False


def _validate_masyu(payload: Any, state: Mapping[str, Any]) -> bool:
    rows, cols = int(state.get("rows") or 0), int(state.get("cols") or 0)
    visible = _visible_cells(payload)
    if visible is None:
        return False
    # Candidates submit 1-based (row, column) ordinals; replay
    # on the internal 0-based board.  Out-of-range ordinals fail closed, which
    # also rejects the legacy 0-based payload whenever it names row or column 0.
    if any(not (1 <= row <= rows and 1 <= col <= cols) for row, col in visible):
        return False
    path = _cycle([(row - 1, col - 1) for row, col in visible], rows, cols)
    if path is None:
        return False
    # Segments already drawn in the input are fixed parts of the loop.
    payload_edges = {
        tuple(sorted(((path[i][0], path[i][1]), (path[i + 1][0], path[i + 1][1]))))
        for i in range(len(path) - 1)
    }
    try:
        for r1, c1, r2, c2 in state.get("given_edges") or []:
            edge = tuple(sorted(((int(r1) - 1, int(c1) - 1), (int(r2) - 1, int(c2) - 1))))
            if edge not in payload_edges:
                return False
    except (TypeError, ValueError):
        return False
    order = path[:-1]
    index = {cell: i for i, cell in enumerate(order)}
    def direction(a: Cell, b: Cell) -> Cell: return b[0] - a[0], b[1] - a[1]
    def dirs(cell: Cell) -> Tuple[Cell, Cell]:
        i = index[cell]
        return direction(cell, order[(i - 1) % len(order)]), direction(cell, order[(i + 1) % len(order)])
    try:
        for pearl in _cells(state.get("white_pearls")) or []:
            if pearl not in index:
                return False
            before, after = dirs(pearl)
            if before[0] != -after[0] or before[1] != -after[1]:
                return False
            i = index[pearl]
            prev_turn = dirs(order[(i - 1) % len(order)])
            next_turn = dirs(order[(i + 1) % len(order)])
            if (prev_turn[0][0] == -prev_turn[1][0] and prev_turn[0][1] == -prev_turn[1][1]) and (next_turn[0][0] == -next_turn[1][0] and next_turn[0][1] == -next_turn[1][1]):
                return False
        for pearl in _cells(state.get("black_pearls")) or []:
            if pearl not in index:
                return False
            before, after = dirs(pearl)
            if before[0] == -after[0] and before[1] == -after[1]:
                return False
            i = index[pearl]
            prev_before, prev_after = dirs(order[(i - 1) % len(order)])
            next_before, next_after = dirs(order[(i + 1) % len(order)])
            if not (prev_before[0] == -prev_after[0] and prev_before[1] == -prev_after[1] and next_before[0] == -next_after[0] and next_before[1] == -next_after[1]):
                return False
    except (TypeError, ValueError):
        return False
    return True


def _validate_flow_pairs(payload: Any, state: Mapping[str, Any]) -> bool:
    rows, cols = int(state.get("rows") or 0), int(state.get("cols") or 0)
    endpoints_raw = state.get("endpoints")
    if not isinstance(payload, dict) or not isinstance(endpoints_raw, list) or rows <= 0 or cols <= 0:
        return False
    endpoints: Dict[str, Set[Cell]] = {}
    for entry in endpoints_raw:
        if not isinstance(entry, Mapping):
            return False
        color = str(entry.get("color") or "").strip().lower()
        cells = _cells(entry.get("cells"))
        if not color or cells is None or len(cells) != 2:
            return False
        endpoints[color] = {(int(row), int(col)) for row, col in cells}
    if set(payload) != set(endpoints):
        return False
    paths: Dict[str, List[Cell]] = {}
    for color, raw_path in payload.items():
        visible = _visible_cells(raw_path)
        if visible is None or len(visible) < 2:
            return False
        # Candidates submit 1-based (row, column) ordinals;
        # replay on the internal 0-based board.  Out-of-range ordinals fail
        # closed, which also rejects legacy 0-based or linear-ID payloads.
        if any(not (1 <= row <= rows and 1 <= col <= cols) for row, col in visible):
            return False
        path = [(row - 1, col - 1) for row, col in visible]
        if len(set(path)) != len(path):
            return False
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(path, path[1:])):
            return False
        if {path[0], path[-1]} != endpoints[color]:
            return False
        paths[color] = path
    all_cells = [cell for path in paths.values() for cell in path]
    if len(all_cells) != len(set(all_cells)):
        return False
    if set(all_cells) != {(r, c) for r in range(rows) for c in range(cols)}:
        return False
    # Route prefixes already drawn out of an endpoint are fixed parts of the
    # final routes: every given segment must appear in the candidate route.
    try:
        for raw_color, sub_paths in (state.get("given_paths") or {}).items():
            color = str(raw_color).strip().lower()
            if color not in paths:
                return False
            path_edges = {
                tuple(sorted((a, b))) for a, b in zip(paths[color], paths[color][1:])
            }
            for sub_path in sub_paths or []:
                cells = [(int(cell[0]), int(cell[1])) for cell in sub_path]
                for a, b in zip(cells, cells[1:]):
                    if tuple(sorted((a, b))) not in path_edges:
                        return False
    except (TypeError, ValueError, IndexError):
        return False
    return True


def _validate_nqueens(payload: Any, state: Mapping[str, Any]) -> bool:
    n = int(state.get("board_size") or 0)
    visible = _visible_cells(payload)
    fixed = _cells(state.get("fixed_positions"))
    if visible is None or fixed is None:
        return False
    # Candidates submit 1-based coordinates; replay on the
    # internal 0-based board.  Out-of-range ordinals fail closed.
    if any(not (1 <= row <= n and 1 <= col <= n) for row, col in visible):
        return False
    positions = [(row - 1, col - 1) for row, col in visible]
    if len(positions) != n or len(set(positions)) != n or not set(fixed) <= set(positions):
        return False
    return ({r for r, _ in positions} == set(range(n)) and {c for _, c in positions} == set(range(n)) and len({r - c for r, c in positions}) == n and len({r + c for r, c in positions}) == n)


def _validate_nurikabe(payload: Any, state: Mapping[str, Any]) -> bool:
    rows, cols = int(state.get("rows") or 0), int(state.get("cols") or 0)
    grid = _binary_grid(payload, rows, cols)
    clues = state.get("clues")
    if grid is None or not isinstance(clues, list) or not _no_black_2x2(grid):
        return False
    # Sea and island cells already shown in the input are fixed.
    try:
        for r, c in state.get("given_black") or []:
            if grid[int(r)][int(c)] != 1:
                return False
        for r, c in state.get("given_white") or []:
            if grid[int(r)][int(c)] != 0:
                return False
    except (TypeError, ValueError, IndexError):
        return False
    black = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1}
    if not _connected(black, rows, cols):
        return False
    white = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 0}
    components: List[Set[Cell]] = []
    unseen = set(white)
    while unseen:
        start = next(iter(unseen)); component = {start}; unseen.remove(start); stack = [start]
        while stack:
            for neighbor in _neighbors4(stack.pop(), rows, cols):
                if neighbor in unseen:
                    unseen.remove(neighbor); component.add(neighbor); stack.append(neighbor)
        components.append(component)
    try:
        for component in components:
            inside = [(int(r), int(c), int(value)) for r, c, value in clues if (int(r), int(c)) in component]
            if len(inside) != 1 or inside[0][2] != len(component):
                return False
        return all(grid[int(r)][int(c)] == 0 for r, c, _ in clues)
    except (TypeError, ValueError, IndexError):
        return False


def _validate_shikaku(payload: Any, state: Mapping[str, Any]) -> bool:
    rows, cols = int(state.get("rows") or 0), int(state.get("cols") or 0)
    clues_raw = state.get("clues")
    composites = _visible_composites(payload)
    if composites is None or not isinstance(clues_raw, list):
        return False
    clues = {(int(r), int(c)): int(value) for r, c, value in clues_raw}
    covered: Set[Cell] = set()
    submitted: Set[Tuple[int, int, int, int]] = set()
    try:
        for top_left, bottom_right in composites:
            # Candidates submit 1-based inclusive bounds; replay
            # on the internal 0-based board.  Ordinals below 1 or past the last
            # row/column fail closed, which also rejects legacy 0-based bounds.
            r0, c0 = top_left
            r1, c1 = bottom_right
            if not (1 <= r0 <= r1 <= rows and 1 <= c0 <= c1 <= cols): return False
            r0,c0,r1,c1 = r0-1,c0-1,r1-1,c1-1
            cells={(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)}
            inside=[cell for cell in clues if cell in cells]
            if covered & cells or len(inside)!=1 or len(cells)!=clues[inside[0]]: return False
            covered |= cells
            submitted.add((r0, c0, r1, c1))
        # Rectangles already outlined in the input are fixed and must appear in
        # the finished partition exactly as drawn.
        for given in state.get("given_rectangles") or []:
            if tuple(int(bound) for bound in given) not in submitted:
                return False
    except (TypeError, ValueError): return False
    return covered == {(r,c) for r in range(rows) for c in range(cols)}


def _validate_slitherlink(payload: Any, state: Mapping[str, Any]) -> bool:
    rows,cols=int(state.get("rows") or 0),int(state.get("cols") or 0)
    composites = _visible_composites(payload)
    if composites is None:return False
    edges:Set[Tuple[Cell,Cell]]=set(); adjacency:Dict[Cell,Set[Cell]]=defaultdict(set)
    try:
        for first, second in composites:
            # Lattice corners are named from 1; replay below is
            # 0-based, so convert here and fail closed outside 1..R+1 / 1..C+1.
            if not(1<=first[0]<=rows+1 and 1<=first[1]<=cols+1 and 1<=second[0]<=rows+1 and 1<=second[1]<=cols+1):return False
            a=(int(first[0])-1,int(first[1])-1); b=(int(second[0])-1,int(second[1])-1); key=tuple(sorted((a,b)))
            if key in edges or abs(a[0]-b[0])+abs(a[1]-b[1])!=1:return False
            edges.add(key);adjacency[a].add(b);adjacency[b].add(a)
    except (TypeError,ValueError):return False
    nodes=set(adjacency)
    if not nodes or any(len(adjacency[node])!=2 for node in nodes) or not _connected_graph(nodes,adjacency):return False
    # Segments already drawn in the input are fixed parts of the loop.
    try:
        for r1,c1,r2,c2 in state.get("given_edges") or []:
            key=tuple(sorted(((int(r1)-1,int(c1)-1),(int(r2)-1,int(c2)-1))))
            if key not in edges:return False
    except (TypeError,ValueError):return False
    clues=state.get("clues")
    if not isinstance(clues,list):return False
    for r in range(rows):
        for c in range(cols):
            clue=int(clues[r][c])
            if clue<0:continue
            around={tuple(sorted(((r,c),(r,c+1)))),tuple(sorted(((r+1,c),(r+1,c+1)))),tuple(sorted(((r,c),(r+1,c)))),tuple(sorted(((r,c+1),(r+1,c+1))))}
            if len(around & edges)!=clue:return False
    return True


def _validate_star_battle(payload:Any,state:Mapping[str,Any])->bool:
    n=int(state.get("size") or 0);visible=_visible_cells(payload);regions=state.get("regions")
    if visible is None or not isinstance(regions,list):return False
    # Candidates submit 1-based (row, column) ordinals; replay
    # on the internal 0-based board.  Out-of-range ordinals fail closed, which
    # also rejects the legacy 0-based payload whenever it names row or column 0.
    if any(not (1<=row<=n and 1<=col<=n) for row,col in visible):return False
    stars=[(row-1,col-1) for row,col in visible]
    if len(stars)!=n or len(set(stars))!=n:return False
    if {r for r,_ in stars}!=set(range(n)) or {c for _,c in stars}!=set(range(n)):return False
    if any(max(abs(a[0]-b[0]),abs(a[1]-b[1]))<=1 for i,a in enumerate(stars) for b in stars[i+1:]):return False
    # Stars already placed in the input are fixed.
    try:
        for r, c in state.get("given_stars") or []:
            if (int(r), int(c)) not in set(stars):
                return False
    except (TypeError, ValueError):
        return False
    region_sets=[]
    for raw in regions:
        cells=_cells(raw)
        if cells is None:return False
        region_sets.append(set(cells))
    return len(region_sets)==n and all(sum(star in region for star in stars)==1 for region in region_sets)


def _validate_yajilin(payload:Any,state:Mapping[str,Any])->bool:
    rows,cols=int(state.get("rows") or 0),int(state.get("cols") or 0);visible=_visible_cells(payload);clues=state.get("clues")
    if visible is None or not isinstance(clues,list):return False
    # Candidates submit 1-based (row, column) ordinals; replay
    # on the internal 0-based board.  Out-of-range ordinals fail closed, which
    # also rejects the legacy 0-based loop whenever it names row or column 0.
    if any(not (1<=row<=rows and 1<=col<=cols) for row,col in visible):return False
    path=_cycle([(row-1,col-1) for row,col in visible],rows,cols)
    if path is None:return False
    loop=set(path[:-1]);clue_cells={(int(r),int(c)) for r,c,_,_ in clues}
    # Shades and segments already shown in the input are fixed.
    payload_edges={tuple(sorted((tuple(a),tuple(b)))) for a,b in zip(path,path[1:])}
    try:
        for r,c in state.get("given_black") or []:
            if (int(r),int(c)) in loop:return False
        for r1,c1,r2,c2 in state.get("given_edges") or []:
            edge=tuple(sorted(((int(r1)-1,int(c1)-1),(int(r2)-1,int(c2)-1))))
            if edge not in payload_edges:return False
    except (TypeError,ValueError):return False
    # The typed witness is the complete loop.  Every non-clue cell outside
    # that loop is therefore black by the stated rule; consulting an authored
    # ``black_cells`` solution would make the replay circular.
    black={(r,c) for r in range(rows) for c in range(cols) if (r,c) not in loop|clue_cells}
    if loop & (black|clue_cells):return False
    if any(any(neighbor in black for neighbor in _neighbors4(cell,rows,cols)) for cell in black):return False
    if loop|black|clue_cells != {(r,c) for r in range(rows) for c in range(cols)}:return False
    directions={"U":(-1,0),"D":(1,0),"L":(0,-1),"R":(0,1)}
    try:
        for r,c,direction,count in clues:
            dr,dc=directions[str(direction)];rr,cc=int(r)+dr,int(c)+dc;seen=0
            while _in_bounds((rr,cc),rows,cols):seen+=((rr,cc) in black);rr+=dr;cc+=dc
            if seen!=int(count):return False
    except(KeyError,TypeError,ValueError):return False
    return True


def _validate_yinyang(payload:Any,state:Mapping[str,Any])->bool:
    n=int(state.get("size") or 0);grid=_binary_grid(payload,n,n)
    if grid is None:return False
    try:
        if any(grid[int(r)][int(c)]!=int(value) for r,c,value in state.get("given") or []):return False
    except(TypeError,ValueError,IndexError):return False
    for value in (0,1):
        cells={(r,c) for r in range(n) for c in range(n) if grid[r][c]==value}
        if not _connected(cells,n,n):return False
    return all(len({grid[r+dr][c+dc] for dr in (0,1) for dc in (0,1)})>1 for r in range(n-1) for c in range(n-1))


_POSITION_LIST_TASKS = {
    "campsite_completion",
    "light_up_akari_completion",
    "masyu_completion",
    "nqueens_completion",
    "star_battle_completion",
    "yajilin_completion",
}
_COMPOSITE_TASKS = {"shikaku_completion", "slitherlink_completion"}
_COORDINATE_TASKS = _POSITION_LIST_TASKS | _COMPOSITE_TASKS | {"maze_flow_pairs"}
_MATRIX_TASKS = {
    "aquarium_completion",
    "fillomino_completion",
    "nurikabe_completion",
    "yin_yang_completion",
}


def validate_constraint_payload(sample:CanonicalSample,payload:Any)->Tuple[bool,str,Any]:
    """Validate and canonicalize one typed final payload."""
    task=sample.task_name;state=_state(sample)
    validators={
        "aquarium_completion":lambda:_validate_aquarium(payload,state),
        "campsite_completion":lambda:_validate_campsite(payload,state),
        "fillomino_completion":lambda:_validate_fillomino(payload,state),
        "light_up_akari_completion":lambda:_validate_light_up(payload,state),
        "masyu_completion":lambda:_validate_masyu(payload,state),
        "maze_flow_pairs":lambda:_validate_flow_pairs(payload,state),
        "nqueens_completion":lambda:_validate_nqueens(payload,state),
        "nurikabe_completion":lambda:_validate_nurikabe(payload,state),
        "shikaku_completion":lambda:_validate_shikaku(payload,state),
        "slitherlink_completion":lambda:_validate_slitherlink(payload,state),
        "star_battle_completion":lambda:_validate_star_battle(payload,state),
        "yajilin_completion":lambda:_validate_yajilin(payload,state),
        "yin_yang_completion":lambda:_validate_yinyang(payload,state),
    }
    validator=validators.get(task)
    if validator is None:return False,f"no independent constraint replay is registered for {task}",None
    try:valid=bool(validator())
    except Exception:valid=False
    signature=json.dumps(payload,sort_keys=True,separators=(",",":")) if isinstance(payload,(dict,list)) else str(payload)
    return valid,"ok" if valid else "typed payload violates the visible puzzle constraints",signature


__all__ = [
    "validate_constraint_payload",
]
