"""Independent terminal rendering for Constraint Grid candidates.

The functions in this module consume only the visible puzzle state plus an
already validated candidate payload.  They deliberately do not read the saved
terminal image or saved solution fields.  Reusing the task renderer keeps the
visual vocabulary identical while making text/image agreement executable.
"""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from PIL import Image

from data.common.pipeline.grid_rendering import fit_canvas
from data.common.schemas import CanonicalSample


Cell = Tuple[int, int]
_RENDER_ROOT = Path(tempfile.gettempdir()) / "videogym_constraint_render_replay"


def _state(sample: CanonicalSample) -> Dict[str, Any]:
    raw = sample.semantic_input or {}
    state = raw.get("semantic_state") if isinstance(raw, dict) else None
    return dict(state) if isinstance(state, dict) else {}


def _module(task: str):
    return importlib.import_module(f"data.constraint_grid.{task}.generator")


def _generator(task: str, class_name: str, **kwargs: Any):
    cls = getattr(_module(task), class_name)
    output = _RENDER_ROOT / task
    output.mkdir(parents=True, exist_ok=True)
    return cls(output, seed=0, video=False, **kwargs)


def _grid(value: Any) -> List[List[int]]:
    if not isinstance(value, list) or not value or any(not isinstance(row, list) for row in value):
        raise ValueError("candidate is not a rectangular grid")
    result = [[int(cell) for cell in row] for row in value]
    if not result or any(len(row) != len(result[0]) for row in result):
        raise ValueError("candidate is not a rectangular grid")
    return result


def _cells(value: Any) -> List[Cell]:
    if not isinstance(value, list):
        raise ValueError("candidate is not a cell list")
    return [(int(cell[0]), int(cell[1])) for cell in value]


def _tuple_cells(value: Any) -> List[Cell]:
    return [tuple(cell) for cell in _cells(value)]


def _tuple_regions(value: Any) -> List[List[Cell]]:
    if not isinstance(value, list):
        raise ValueError("regions are unavailable")
    return [_tuple_cells(region) for region in value]


def _render_class_task(task: str, payload: Any, state: Mapping[str, Any]) -> Optional[Image.Image]:
    if task == "aquarium_completion":
        n = int(state["size"])
        generator = _generator(task, "AquariumCompletionGenerator", size=n)
        fill = _grid(payload)
        water = {(r, c) for r in range(n) for c in range(n) if int(fill[r][c]) == 1}
        dry = {(r, c) for r in range(n) for c in range(n) if int(fill[r][c]) == 0}
        return generator._render(
            _tuple_regions(state["regions"]),
            state["row_counts"],
            state["col_counts"],
            committed_water=water,
            committed_dry=dry,
        )
    if task == "campsite_completion":
        rows, cols = int(state["rows"]), int(state["cols"])
        generator = _generator(task, "CampsiteCompletionGenerator", rows=rows, cols=cols)
        # Model-facing tent coordinates are one-based; the renderer and semantic
        # state keep internal zero-based cells.
        tents = [(row - 1, col - 1) for row, col in _tuple_cells(payload)]
        return generator._render_board(_tuple_cells(state["trees"]), tents, state["row_counts"], state["col_counts"], solved=True)
    if task == "fillomino_completion":
        generator = _generator(task, "FillominoCompletionGenerator", rows=int(state["rows"]), cols=int(state["cols"]))
        return generator._render(_grid(payload), _grid(state["puzzle"]))
    if task == "light_up_akari_completion":
        generator = _generator(task, "LightUpAkariCompletionGenerator", rows=int(state["rows"]), cols=int(state["cols"]))
        black = set(_tuple_cells(state["black_cells"]))
        clues = {(int(row), int(col)): int(value) for row, col, value in state["clues"]}
        # Model-facing light coordinates are one-based; the renderer and
        # semantic state keep internal zero-based cells.
        lights = {(row - 1, col - 1) for row, col in _tuple_cells(payload)}
        return generator._render(black, clues, lights, solved=True)
    if task == "masyu_completion":
        generator = _generator(task, "MasyuCompletionGenerator", rows=int(state["rows"]), cols=int(state["cols"]))
        # Model-facing loop coordinates are one-based; the renderer and semantic
        # state keep internal zero-based cells.
        loop = [(row - 1, col - 1) for row, col in _tuple_cells(payload)]
        edges = [tuple(sorted((a, b))) for a, b in zip(loop, loop[1:])]
        return generator._render(edges, _tuple_cells(state["black_pearls"]), _tuple_cells(state["white_pearls"]))
    if task == "maze_flow_pairs":
        rows, cols = int(state["rows"]), int(state["cols"])
        generator = _generator(task, "MazeFlowPairsGenerator", rows=rows, cols=cols)
        endpoints = state["endpoints"]
        if not isinstance(endpoints, list) or not endpoints:
            raise ValueError("visible flow endpoints are unavailable")
        if not isinstance(payload, dict):
            raise ValueError("candidate is not a colour-to-route object")
        # Model-facing routes are 1-based; the renderer and semantic state keep
        # internal 0-based cells.
        strokes = {
            str(color).strip().lower(): [
                [(int(row) - 1, int(col) - 1) for row, col in path]
            ]
            for color, path in payload.items()
        }
        return generator._render(endpoints, strokes)
    if task == "nqueens_completion":
        n = int(state["board_size"])
        generator = _generator(task, "NQueensCompletionGenerator", board_size=n)
        # Model-facing N-Queens coordinates are one-based; the renderer and
        # semantic state keep internal zero-based cells.
        positions = {
            (row - 1, col - 1)
            for row, col in _tuple_cells(payload)
        }
        board = [["Q" if (row, col) in positions else "" for col in range(n)] for row in range(n)]
        return generator._render_board(
            board,
            fixed_positions=set(_tuple_cells(state["fixed_positions"])),
        )
    if task == "nurikabe_completion":
        rows, cols = int(state["rows"]), int(state["cols"])
        generator = _generator(task, "NurikabeCompletionGenerator", rows=rows, cols=cols)
        fill = _grid(payload)
        clues = [tuple(map(int, clue)) for clue in state["clues"]]
        clue_cells = {(r, c) for r, c, _ in clues}
        black = {(r, c) for r in range(rows) for c in range(cols) if fill[r][c] == 1}
        dots = {
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if fill[r][c] == 0 and (r, c) not in clue_cells
        }
        return generator._render(black, dots, clues)
    if task == "shikaku_completion":
        rows, cols = int(state["rows"]), int(state["cols"])
        generator = _generator(task, "ShikakuCompletionGenerator", rows=rows, cols=cols)
        # Model-facing rectangle bounds are one-based; the renderer and semantic
        # state keep internal zero-based cells.
        rectangles = [
            (
                int(top_left[0]) - 1,
                int(top_left[1]) - 1,
                int(bottom_right[0]) - 1,
                int(bottom_right[1]) - 1,
            )
            for top_left, bottom_right in payload
        ]
        clues = _tuple_cells(state["clue_positions"])
        ordered: List[Tuple[int, int, int, int]] = []
        for row, col in clues:
            matches = [rectangle for rectangle in rectangles if rectangle[0] <= row <= rectangle[2] and rectangle[1] <= col <= rectangle[3]]
            if len(matches) != 1:
                raise ValueError("candidate rectangle cannot be matched to its visible clue")
            ordered.append(matches[0])
        return generator._render(ordered, clues, committed=ordered)
    if task == "slitherlink_completion":
        generator = _generator(task, "SlitherlinkCompletionGenerator", rows=int(state["rows"]), cols=int(state["cols"]))
        # Model-facing lattice corners are one-based; the renderer and
        # semantic state keep internal zero-based corners.
        edges = [
            (
                (int(first[0]) - 1, int(first[1]) - 1),
                (int(second[0]) - 1, int(second[1]) - 1),
            )
            for first, second in payload
        ]
        clues = [[None if int(value) < 0 else int(value) for value in row] for row in state["clues"]]
        return generator._render(clues, edges)
    if task == "star_battle_completion":
        generator = _generator(task, "StarBattleCompletionGenerator", size=int(state["size"]))
        # Model-facing star coordinates are one-based; the renderer and
        # semantic state keep internal zero-based cells.
        stars = [(row - 1, col - 1) for row, col in _tuple_cells(payload)]
        return generator._render(stars, _tuple_regions(state["regions"]))
    if task == "yajilin_completion":
        rows, cols = int(state["rows"]), int(state["cols"])
        generator = _generator(task, "YajilinCompletionGenerator", rows=rows, cols=cols)
        # Model-facing loop coordinates are one-based; the renderer and
        # semantic state keep internal zero-based cells.
        path = [(row - 1, col - 1) for row, col in _tuple_cells(payload)]
        clue_cells = {(int(row), int(col)) for row, col, _, _ in state["clues"]}
        black = [(row, col) for row in range(rows) for col in range(cols) if (row, col) not in set(path) and (row, col) not in clue_cells]
        edges = [tuple(sorted((a, b))) for a, b in zip(path, path[1:])]
        return generator._render(edges, black, state["clues"])
    if task == "yin_yang_completion":
        n = int(state["size"])
        generator = _generator(task, "YinYangCompletionGenerator", size=n)
        given = {(int(row), int(col)): int(value) for row, col, value in state["given"]}
        fill = _grid(payload)
        committed = {(r, c): int(fill[r][c]) for r in range(n) for c in range(n)}
        return generator._render(committed, given)
    return None


def render_candidate_terminal(sample: CanonicalSample, payload: Any) -> Optional[Image.Image]:
    """Render the independently validated candidate's terminal image.

    ``None`` means the task is routed to a separate specialized verifier or
    that visible renderer state required by an old sample is unavailable.
    """
    state = _state(sample)
    image = _render_class_task(sample.task_name, payload, state)
    return fit_canvas(image.convert("RGB")) if image is not None else None
