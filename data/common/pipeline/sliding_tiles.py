from __future__ import annotations

from data.common.assets import render_font

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import ImageFont

_MOVE_DELTAS: Dict[str, Tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
_2048_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (238, 228, 218),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

def _load_font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)

def _goal_state(size: int = 4) -> List[List[int]]:
    values = list(range(1, size * size)) + [0]
    return [values[index : index + size] for index in range(0, len(values), size)]

def _find_blank(state: Sequence[Sequence[int]]) -> Tuple[int, int]:
    for row_index, row in enumerate(state):
        for col_index, value in enumerate(row):
            if int(value) == 0:
                return row_index, col_index
    raise ValueError("Puzzle state is missing a blank cell.")

def _apply_blank_move(state: Sequence[Sequence[int]], move: str) -> List[List[int]]:
    if move not in _MOVE_DELTAS:
        raise ValueError(f"Unsupported move: {move}")
    board = [[int(cell) for cell in row] for row in state]
    row, col = _find_blank(board)
    dr, dc = _MOVE_DELTAS[move]
    nr, nc = row + dr, col + dc
    if not (0 <= nr < len(board) and 0 <= nc < len(board[nr])):
        raise ValueError(f"Illegal move '{move}' for blank at {(row, col)}")
    board[row][col], board[nr][nc] = board[nr][nc], board[row][col]
    return board

def _valid_blank_moves(state: Sequence[Sequence[int]]) -> List[str]:
    row, col = _find_blank(state)
    size = len(state)
    valid: List[str] = []
    for move, (dr, dc) in _MOVE_DELTAS.items():
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size:
            valid.append(move)
    return valid

def _state_key(state: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    return tuple(int(cell) for row in state for cell in row)

def _bfs_unique_solution(start: Sequence[Sequence[int]], goal: Sequence[Sequence[int]]) -> List[str]:
    start_key = _state_key(start)
    goal_key = _state_key(goal)
    queue = deque([start_key])
    distance = {start_key: 0}
    ways = {start_key: 1}
    parent: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], str]] = {}

    def _to_grid(key: Tuple[int, ...]) -> List[List[int]]:
        size = int(len(key) ** 0.5)
        return [list(key[index : index + size]) for index in range(0, len(key), size)]

    found_depth: Optional[int] = None
    while queue:
        key = queue.popleft()
        current_distance = distance[key]
        if found_depth is not None and current_distance >= found_depth:
            continue
        grid = _to_grid(key)
        for move in _valid_blank_moves(grid):
            next_grid = _apply_blank_move(grid, move)
            next_key = _state_key(next_grid)
            next_distance = current_distance + 1
            if found_depth is not None and next_distance > found_depth:
                continue
            if next_key not in distance:
                distance[next_key] = next_distance
                ways[next_key] = ways[key]
                parent[next_key] = (key, move)
                queue.append(next_key)
            elif distance[next_key] == next_distance:
                ways[next_key] += ways[key]
            if next_key == goal_key and found_depth is None:
                found_depth = next_distance

    if goal_key not in distance or ways.get(goal_key, 0) != 1:
        return []

    moves: List[str] = []
    cursor = goal_key
    while cursor != start_key:
        prev, move = parent[cursor]
        moves.append(move)
        cursor = prev
    moves.reverse()
    return moves

def _merge_line(values: Sequence[int]) -> Tuple[List[int], int, int]:
    filtered = [int(value) for value in values if int(value) != 0]
    merged: List[int] = []
    merges = 0
    created = 0
    index = 0
    while index < len(filtered):
        if index + 1 < len(filtered) and filtered[index] == filtered[index + 1]:
            value = filtered[index] * 2
            merged.append(value)
            merges += 1
            created += value
            index += 2
        else:
            merged.append(filtered[index])
            index += 1
    while len(merged) < len(values):
        merged.append(0)
    return merged, merges, created

def _swipe_board(board: Sequence[Sequence[int]], direction: str) -> Tuple[List[List[int]], Tuple[int, int, int]]:
    size = len(board)
    grid = [[int(cell) for cell in row] for row in board]
    merges = 0
    created = 0
    result = [[0 for _ in range(size)] for _ in range(size)]

    if direction == "left":
        for row in range(size):
            merged, row_merges, row_created = _merge_line(grid[row])
            result[row] = merged
            merges += row_merges
            created += row_created
    elif direction == "right":
        for row in range(size):
            merged, row_merges, row_created = _merge_line(list(reversed(grid[row])))
            result[row] = list(reversed(merged))
            merges += row_merges
            created += row_created
    elif direction == "up":
        for col in range(size):
            merged, col_merges, col_created = _merge_line([grid[row][col] for row in range(size)])
            merges += col_merges
            created += col_created
            for row in range(size):
                result[row][col] = merged[row]
    elif direction == "down":
        for col in range(size):
            merged, col_merges, col_created = _merge_line([grid[row][col] for row in reversed(range(size))])
            merges += col_merges
            created += col_created
            for row, value in zip(reversed(range(size)), merged):
                result[row][col] = value
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    nonzero = sum(1 for row in result for cell in row if cell != 0)
    return result, (merges, created, -nonzero)

def _2048_tile_value(value: int) -> Tuple[int, int, int]:
    return _2048_COLORS.get(value, (60, 58, 50))
