"""Independent source replay for the Board Game family.

The canonical manifest keeps authored answers and reference media for SFT
serialization, but neither is an oracle for final-delivery verification.  This
module reconstructs the formal answer, terminal render, and semantic video
states only from the visible source state carried by ``semantic_input``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from data.board_game.core.answers import format_board_matrix_answer
from data.common.grid_coordinates import format_xy_position_list_answer
from data.common.scalar_lists import (
    format_nonnegative_integer_list_answer,
    format_nonnegative_integer_answer,
)
from data.common.schemas import CanonicalSample


Cell = Tuple[int, int]


@dataclass
class BoardReplay:
    task_name: str
    answer_line: Optional[str]
    initial_image: Image.Image
    terminal_image: Image.Image
    video_states: List[Image.Image]
    source_metrics: Dict[str, Any]
    source_payload: Dict[str, Any]


def _semantic_state(sample: CanonicalSample) -> Dict[str, Any]:
    raw = sample.semantic_input if isinstance(sample.semantic_input, dict) else {}
    state = raw.get("semantic_state")
    if not isinstance(state, dict):
        raise ValueError(f"{sample.task_name}: canonical semantic_input lacks semantic_state")
    return dict(state)


def _cell(value: Any, *, name: str) -> Cell:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-integer cell")
    row, col = int(value[0]), int(value[1])
    return row, col


def _cells(value: Any, *, name: str) -> List[Cell]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a cell list")
    return [_cell(item, name=name) for item in value]


def _square_board(value: Any, *, allowed: set[Any], name: str) -> List[List[Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty square board")
    board = [list(row) for row in value if isinstance(row, list)]
    if len(board) != len(value) or any(len(row) != len(board) for row in board):
        raise ValueError(f"{name} must be square")
    if any(cell not in allowed for row in board for cell in row):
        raise ValueError(f"{name} contains an unsupported cell value")
    return board


def _dedupe_images(images: Sequence[Image.Image]) -> List[Image.Image]:
    result: List[Image.Image] = []
    previous: Optional[bytes] = None
    for image in images:
        rgb = image.convert("RGB")
        payload = rgb.tobytes()
        if previous == payload:
            continue
        result.append(rgb)
        previous = payload
    return result


def _hex(sample: CanonicalSample, root: Path) -> BoardReplay:
    from data.board_game.hex_chain_count.generator import HexChainCountGenerator

    state = _semantic_state(sample)
    board = _square_board(state.get("board"), allowed={None, "R", "B"}, name="hex board")
    if len(board) != 7:
        raise ValueError("hex board must be 7 by 7")
    generator = HexChainCountGenerator(root)
    components: List[List[Cell]] = []
    unseen = {(r, c) for r in range(7) for c in range(7) if board[r][c] == "R"}
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: List[Cell] = []
        while stack:
            cell = stack.pop()
            component.append(cell)
            for neighbour in generator._hex_neighbours(*cell):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(component)
    if not components:
        raise ValueError("hex board has no red component")
    longest = max(len(component) for component in components)
    winners = [component for component in components if len(component) == longest]
    if len(winners) != 1:
        raise ValueError("hex board does not have a unique largest red component")
    component = sorted(winners[0])
    initial = generator._render_board(board)
    terminal = generator._render_board(board, highlight=component)
    states = [initial, terminal]
    return BoardReplay(sample.task_name, format_nonnegative_integer_answer(longest), initial, terminal, states,
                       {"source_longest_component_size": longest, "source_longest_component": [list(c) for c in component]},
                       {"board": board, "component": component})


def _lights(sample: CanonicalSample, root: Path) -> BoardReplay:
    from data.board_game.lightsout_single_press.generator import LightsOutSinglePressGenerator

    state = _semantic_state(sample)
    grid = _square_board(state.get("initial_grid"), allowed={0, 1}, name="lights grid")
    generator = LightsOutSinglePressGenerator(
        root,
        size=len(grid),
        press_count_range=(1, int(state.get("maximum_press_count", 3))),
    )
    solutions = generator._solutions(grid)
    if len(solutions) != 1:
        raise ValueError(
            "visible lights grid does not have exactly one bounded-press solution"
        )
    presses = solutions[0]
    current = [row[:] for row in grid]
    images = [generator._render_board(current)]
    for row, col in presses:
        generator._toggle(current, row, col)
        images.append(generator._render_board(current))
    if any(value for row in current for value in row):
        raise ValueError("independent lights replay did not reach all-OFF")
    answer_line = format_xy_position_list_answer(presses)
    return BoardReplay(sample.task_name, answer_line, images[0], images[-1], images,
                       {
                           "source_press_count": len(presses),
                           "source_press_set": [[row + 1, col + 1] for row, col in presses],
                       },
                       {"initial_grid": grid, "presses": presses})


def _mancala(sample: CanonicalSample, root: Path) -> BoardReplay:
    from data.board_game.mancala_distribute_predict.generator import (
        distribute_counts,
        MancalaDistributePredictGenerator,
    )

    state = _semantic_state(sample)
    pits = [int(value) for value in state.get("pits", [])]
    bank = int(state.get("bank"))
    source = int(state.get("source"))
    if len(pits) != 6 or source not in range(6) or any(value < 0 for value in pits) or pits[source] < 1 or bank < 0:
        raise ValueError("mancala visible source contract is invalid")
    final_pits, final_bank, destinations = distribute_counts(pits, bank, source)
    generator = MancalaDistributePredictGenerator(root)
    initial = generator._render(pits, bank, source=source)
    running = list(pits) + [bank]
    vectors = [list(running)]
    running[source] = 0
    vectors.append(list(running))
    states = [initial, generator._render(running[:6], running[6])]
    for destination in destinations:
        running[destination] += 1
        vectors.append(list(running))
        states.append(generator._render(running[:6], running[6]))
    terminal = generator._render(final_pits, final_bank)
    states[-1] = terminal
    answer_line = format_nonnegative_integer_list_answer([*final_pits, final_bank])
    return BoardReplay(sample.task_name, answer_line, initial, terminal, _dedupe_images(states),
                       {"source_distribution_destinations": destinations, "source_count_vectors": vectors},
                       {"pits": pits, "bank": bank, "source": source, "destinations": destinations,
                        "count_vectors": vectors})


def _strict_state(state: Dict[str, Any], keys: set[str], *, task: str) -> None:
    if set(state) != keys:
        raise ValueError(f"{task}: semantic_state must contain exactly {sorted(keys)}")


def _placement_operations(
    value: Any,
    *,
    size: int,
    pieces: set[str],
    name: str,
) -> List[Tuple[str, Cell]]:
    from data.board_game.core.coordinates import from_visible_cell

    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty operation list")
    operations: List[Tuple[str, Cell]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or set(item) != {"piece", "cell"}:
            raise ValueError(f"{name} operation {index} must contain only piece and cell")
        piece = str(item["piece"])
        if piece not in pieces:
            raise ValueError(f"{name} operation {index} has an unsupported piece")
        operations.append(
            (
                piece,
                from_visible_cell(item["cell"], size=size, name=f"{name} operation {index} cell"),
            )
        )
    return operations


def _tic_tac_toe_rule_simulator(sample: CanonicalSample, root: Path) -> BoardReplay:
    from data.board_game.core.tic_tac_toe import MarkMove, TicTacToeRenderer, simulate_ttt

    state = _semantic_state(sample)
    _strict_state(state, {"initial_board", "operations"}, task=sample.task_name)
    board = _square_board(state["initial_board"], allowed={None, "X", "O"}, name="Tic-Tac-Toe board")
    if len(board) != 3:
        raise ValueError("Tic-Tac-Toe board must be 3 by 3")
    operations = _placement_operations(
        state["operations"], size=3, pieces={"X", "O"}, name="Tic-Tac-Toe"
    )
    moves = tuple(MarkMove(piece, cell) for piece, cell in operations)
    simulation = simulate_ttt(board, moves)
    if simulation.stop_reason != "win" or len(simulation.winning_cells) != 3:
        raise ValueError("Tic-Tac-Toe simulator source must contain one terminal three-mark win")
    renderer = TicTacToeRenderer()
    images = [renderer.render(simulation.initial_board)] + [
        renderer.render(step.board_after) for step in simulation.steps
    ]
    terminal = renderer.render(
        simulation.final_board,
        winning_cells=simulation.winning_cells,
    )
    images.append(terminal)
    answer_line = format_board_matrix_answer(
        simulation.final_board,
        symbols={None: ".", "X": "X", "O": "O"},
    )
    placement_steps = [
        {
            "target": list(step.move.cell),
            "placed_stage": index,
            "final_stage": index,
            "captured_cells": [],
        }
        for index, step in enumerate(simulation.steps, start=1)
    ]
    return BoardReplay(
        sample.task_name,
        answer_line,
        images[0],
        terminal,
        images,
        {
            "source_executed_operation_count": len(simulation.steps),
            "source_ignored_operation_count": len(simulation.ignored_moves),
            "source_stop_reason": simulation.stop_reason,
        },
        {
            "initial_board": board,
            "moves": moves,
            "simulation": simulation,
            "board_size": 3,
            "placement_process_steps": placement_steps,
            "terminal_overlay": {
                "kind": "winning_cell_borders",
                "cells": [list(cell) for cell in simulation.winning_cells],
                "stage": len(images) - 1,
            },
        },
    )


def _go_rule_simulator(sample: CanonicalSample, root: Path) -> BoardReplay:
    from data.board_game.core.intersection_games import NumericIntersectionRenderer, StoneMove, simulate_go

    state = _semantic_state(sample)
    _strict_state(state, {"initial_board", "operations"}, task=sample.task_name)
    board = _square_board(state["initial_board"], allowed={None, "B", "W"}, name="Go board")
    if len(board) != 7:
        raise ValueError("Go simulator board must be 7 by 7")
    operations = _placement_operations(state["operations"], size=7, pieces={"B", "W"}, name="Go")
    moves = tuple(StoneMove(piece, cell) for piece, cell in operations)
    simulation = simulate_go(board, moves)
    renderer = NumericIntersectionRenderer(7)
    images: List[Image.Image] = [renderer.render(simulation.initial_board)]
    placement_steps: List[Dict[str, Any]] = []
    stage_index = 0
    for step in simulation.steps:
        images.append(renderer.render(step.placed_board))
        stage_index += 1
        placed_stage = stage_index
        final_stage = placed_stage
        if step.captured_cells:
            images.append(renderer.render(step.final_board))
            stage_index += 1
            final_stage = stage_index
        placement_steps.append(
            {
                "target": list(step.move.cell),
                "placed_stage": placed_stage,
                "final_stage": final_stage,
                "captured_cells": [list(cell) for cell in step.captured_cells],
            }
        )
    terminal = renderer.render(simulation.final_board)
    images.append(terminal)
    answer_line = format_board_matrix_answer(
        simulation.final_board,
        symbols={None: ".", "B": "B", "W": "W"},
    )
    return BoardReplay(
        sample.task_name,
        answer_line,
        images[0],
        terminal,
        images,
        {
            "source_operation_count": len(simulation.steps),
            "source_capture_counts": [step.captured_count for step in simulation.steps],
        },
        {
            "initial_board": board,
            "moves": moves,
            "simulation": simulation,
            "board_size": 7,
            "placement_process_steps": placement_steps,
        },
    )


def _go_max_capture_search(sample: CanonicalSample, root: Path) -> BoardReplay:
    from data.board_game.core.coordinates import visible_cell_text
    from data.board_game.core.intersection_games import NumericIntersectionRenderer, solve_go_max_capture

    state = _semantic_state(sample)
    _strict_state(state, {"initial_board", "search_piece"}, task=sample.task_name)
    if str(state["search_piece"]) != "B":
        raise ValueError("Go search_piece must be B")
    board = _square_board(state["initial_board"], allowed={None, "B", "W"}, name="Go search board")
    if len(board) != 7:
        raise ValueError("Go search board must be 7 by 7")
    solution = solve_go_max_capture(board)
    if len(solution.best_cells) != 1 or solution.max_captured <= 0:
        raise ValueError("Go search source must have one unique positive maximum")
    best = solution.best_cells[0]
    candidate = next(candidate for candidate in solution.legal_candidates if candidate.cell == best)
    renderer = NumericIntersectionRenderer(7)
    initial = renderer.render(board)
    placed = renderer.render(candidate.step.placed_board)
    terminal = renderer.render(candidate.step.final_board)
    answer = visible_cell_text(best, size=7)
    return BoardReplay(
        sample.task_name,
        f"Answer: {answer}.",
        initial,
        terminal,
        [initial, placed, terminal],
        {
            "source_maximum_capture_count": solution.max_captured,
            "source_best_cell": list(best),
            "source_captured_cells": [list(cell) for cell in candidate.step.captured_cells],
        },
        {
            "initial_board": board,
            "solution": solution,
            "candidate": candidate,
            "board_size": 7,
            "placement_process_steps": [
                {
                    "target": list(best),
                    "placed_stage": 1,
                    "final_stage": 2,
                    "captured_cells": [list(cell) for cell in candidate.step.captured_cells],
                }
            ],
        },
    )


def _gomoku_rule_simulator(sample: CanonicalSample, root: Path) -> BoardReplay:
    from data.board_game.core.intersection_games import NumericIntersectionRenderer, StoneMove, simulate_gomoku

    state = _semantic_state(sample)
    _strict_state(state, {"initial_board", "operations"}, task=sample.task_name)
    board = _square_board(state["initial_board"], allowed={None, "B", "W"}, name="Gomoku board")
    if len(board) != 9:
        raise ValueError("Gomoku board must be 9 by 9")
    operations = _placement_operations(state["operations"], size=9, pieces={"B", "W"}, name="Gomoku")
    moves = tuple(StoneMove(piece, cell) for piece, cell in operations)
    simulation = simulate_gomoku(board, moves)
    if simulation.stop_reason != "win" or len(simulation.winning_cells) != 5:
        raise ValueError("Gomoku simulator source must contain one terminal five-stone win")
    renderer = NumericIntersectionRenderer(9)
    images = [renderer.render(simulation.initial_board)] + [
        renderer.render(step.board_after) for step in simulation.steps
    ]
    terminal = renderer.render(
        simulation.final_board,
        winning_line=simulation.winning_cells,
    )
    images.append(terminal)
    answer_line = format_board_matrix_answer(
        simulation.final_board,
        symbols={None: ".", "B": "B", "W": "W"},
    )
    placement_steps = [
        {
            "target": list(step.move.cell),
            "placed_stage": index,
            "final_stage": index,
            "captured_cells": [],
        }
        for index, step in enumerate(simulation.steps, start=1)
    ]
    return BoardReplay(
        sample.task_name,
        answer_line,
        images[0],
        terminal,
        images,
        {
            "source_executed_operation_count": len(simulation.steps),
            "source_ignored_operation_count": len(simulation.ignored_moves),
            "source_stop_reason": simulation.stop_reason,
        },
        {
            "initial_board": board,
            "moves": moves,
            "simulation": simulation,
            "board_size": 9,
            "placement_process_steps": placement_steps,
            "terminal_overlay": {
                "kind": "winning_line",
                "cells": [list(cell) for cell in simulation.winning_cells],
                "stage": len(images) - 1,
            },
        },
    )


def _chess_l_waypoint_search(sample: CanonicalSample, root: Path) -> BoardReplay:
    from data.board_game.core.answers import encode_visible_path
    from data.board_game.core.chess_engine import (
        ChessBoardRenderer,
        all_shortest_l_routes,
        l_search_board,
        visited_waypoints_for_prefix,
    )
    from data.board_game.core.coordinates import from_visible_cell

    state = _semantic_state(sample)
    _strict_state(state, {"start", "waypoints", "target"}, task=sample.task_name)
    start = from_visible_cell(state["start"], size=5, name="L start")
    raw_waypoints = state["waypoints"]
    if not isinstance(raw_waypoints, list) or len(raw_waypoints) != 2:
        raise ValueError("L search must expose exactly two waypoints")
    waypoints = tuple(
        from_visible_cell(cell, size=5, name=f"L waypoint {index}")
        for index, cell in enumerate(raw_waypoints, start=1)
    )
    target = from_visible_cell(state["target"], size=5, name="L target")
    routes = all_shortest_l_routes(start, waypoints, target)
    if len(routes) != 1:
        raise ValueError("L search source must have exactly one shortest route")
    route = routes[0]
    renderer = ChessBoardRenderer()
    images: List[Image.Image] = []
    full_path = (start,) + route
    for landing_count, token in enumerate(full_path):
        landed_prefix = route[:landing_count]
        images.append(
            renderer.render(
                l_search_board(token),
                waypoints=waypoints,
                visited_waypoints=visited_waypoints_for_prefix(waypoints, landed_prefix),
                target=target,
            )
        )
    answer = encode_visible_path(full_path, size=5)
    return BoardReplay(
        sample.task_name,
        f"Answer: {answer}.",
        images[0],
        images[-1],
        images,
        {"source_shortest_route_count": len(routes), "source_optimal_length": len(route)},
        {"start": start, "waypoints": waypoints, "target": target, "route": route},
    )


_REPLAYERS = {
    "chess_l_waypoint_search": _chess_l_waypoint_search,
    "go_max_capture_search": _go_max_capture_search,
    "go_rule_simulator": _go_rule_simulator,
    "gomoku_rule_simulator": _gomoku_rule_simulator,
    "hex_chain_count": _hex,
    "lightsout_single_press": _lights,
    "mancala_distribute_predict": _mancala,
    "tic_tac_toe_rule_simulator": _tic_tac_toe_rule_simulator,
}


def replay_board_sample(sample: CanonicalSample, renderer_root: Path) -> BoardReplay:
    try:
        replayer = _REPLAYERS[sample.task_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Board Game task: {sample.task_name}") from exc
    replay = replayer(sample, renderer_root / sample.task_name)
    replay.video_states = _dedupe_images(replay.video_states)
    if not replay.video_states:
        raise ValueError(f"{sample.task_name}: source replay produced no video states")
    if replay.video_states[0].tobytes() != replay.initial_image.convert("RGB").tobytes():
        raise ValueError(f"{sample.task_name}: source replay video does not begin at the initial state")
    if replay.video_states[-1].tobytes() != replay.terminal_image.convert("RGB").tobytes():
        raise ValueError(f"{sample.task_name}: source replay video does not end at the terminal state")
    return replay


__all__ = ["BoardReplay", "replay_board_sample"]
