"""Source-only semantic solving and rendering for Rearrangement Game.

The canonical manifest intentionally exposes only what the input image and
the task rule expose. This module reconstructs the native generator
record required by package-local evaluators, independently solves a reference
plan from that source, and renders semantic states from the recomputed plan.
It never reads ``ground_truth``, saved move/state traces, selected indices,
reference images/videos, or candidate reasoning.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import importlib
import inspect
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from data.common.base import AbstractPuzzleEvaluator
from data.registry import TASK_SPECS
from data.common.schemas import CanonicalSample
from data.rearrangement_game.core.canonical_source import (
    DERIVED_GOAL_TASKS,
    OBJECT_PLAN_TASKS,
)


def source_record(sample: CanonicalSample) -> dict[str, Any]:
    """Return the only evaluator input permitted for one canonical sample."""

    semantic = sample.semantic_input
    if not isinstance(semantic, Mapping):
        raise ValueError("canonical Rearrangement sample lacks semantic_input")
    task_name = str(sample.task_name)
    if task_name in OBJECT_PLAN_TASKS:
        expected = {"visible_initial_state", "visible_parameters"}
        if task_name not in DERIVED_GOAL_TASKS:
            expected.add("visible_goal_state")
        if set(semantic) != expected:
            raise ValueError(
                f"{task_name} canonical source keys must be exactly {sorted(expected)}"
            )
        parameters = semantic.get("visible_parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"{task_name} visible_parameters must be a mapping")
        initial = deepcopy(semantic["visible_initial_state"])
        parameters_copy = deepcopy(dict(parameters))
        target = (
            _derive_target_state(task_name, initial, parameters_copy)
            if task_name in DERIVED_GOAL_TASKS
            else deepcopy(semantic["visible_goal_state"])
        )
        return {
            "id": sample.id,
            "initial_state": initial,
            "target_state": target,
            **parameters_copy,
        }
    if task_name == "sliding_tile_15":
        expected = {"visible_initial_grid", "board_size"}
        if set(semantic) != expected:
            raise ValueError("sliding_tile_15 canonical source has an invalid boundary")
        return {
            "id": sample.id,
            "initial_grid": deepcopy(semantic["visible_initial_grid"]),
            "board_size": int(semantic["board_size"]),
        }
    if task_name == "maze_rush_hour":
        expected = {"grid_size", "exit_row", "cars"}
        if set(semantic) != expected:
            raise ValueError("maze_rush_hour canonical source has an invalid boundary")
        return {
            "id": sample.id,
            "grid_size": deepcopy(semantic["grid_size"]),
            "exit_row": int(semantic["exit_row"]),
            "cars": deepcopy(semantic["cars"]),
        }
    raise ValueError(f"unsupported Rearrangement task {task_name!r}")


def _task_evaluator_class(task_name: str) -> type[AbstractPuzzleEvaluator]:
    module_name = f"data.rearrangement_game.{task_name}.evaluator"
    module = importlib.import_module(module_name)
    classes = [
        candidate
        for _, candidate in inspect.getmembers(module, inspect.isclass)
        if candidate.__module__ == module_name
        and candidate is not AbstractPuzzleEvaluator
        and issubclass(candidate, AbstractPuzzleEvaluator)
    ]
    if len(classes) != 1:
        raise RuntimeError(
            f"{module_name} must contain exactly one task-owned evaluator; found "
            f"{[candidate.__name__ for candidate in classes]}"
        )
    return classes[0]


def evaluate_text_from_source(
    sample: CanonicalSample,
    answer_line: str,
) -> tuple[dict[str, Any], bool]:
    """Run a package-local semantic evaluator on source fields only."""

    task_name = str(sample.task_name)
    if task_name == "maze_rush_hour":
        from data.rearrangement_game.maze_rush_hour.evaluator import MazeRushHourEvaluator

        evaluator_class: type[AbstractPuzzleEvaluator] = MazeRushHourEvaluator
    elif task_name in OBJECT_PLAN_TASKS:
        evaluator_class = _task_evaluator_class(task_name)
    else:
        raise ValueError(f"{task_name} uses a dedicated source replay")
    evaluator = evaluator_class(source_record(sample), base_dir=Path(tempfile.gettempdir()))
    result = (
        evaluator.evaluate_text(sample.id, answer_line)
        if task_name == "maze_rush_hour"
        else evaluator.evaluate(sample.id, answer_line)
    )
    metrics = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    valid = metrics.get("valid")
    if not isinstance(valid, bool):
        binary = metrics.get("binary_correct")
        if isinstance(binary, (bool, int)):
            valid = bool(binary)
    if not isinstance(valid, bool):
        raise TypeError(f"{evaluator_class.__name__} did not return boolean validity")
    return metrics, valid


def _new_generator(task_name: str, output_dir: Path, *, video: bool = False) -> Any:
    spec = TASK_SPECS[task_name]
    generator_class = getattr(importlib.import_module(spec.module), spec.class_name)
    signature = inspect.signature(generator_class)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    if "output_dir" in signature.parameters:
        kwargs["output_dir"] = output_dir
    else:
        args.append(output_dir)
    if "seed" in signature.parameters:
        kwargs["seed"] = 0
    if "video" in signature.parameters:
        kwargs["video"] = video
    if "generate_videos" in signature.parameters:
        kwargs["generate_videos"] = video
    generator = generator_class(*args, **kwargs)
    # Visible scalar/list parameters may control renderer geometry.  They are
    # source facts, not saved answers, so hydrate matching public attributes.
    return generator


def _derived_plan(
    task_name: str,
    initial: Any,
    parameters: Mapping[str, Any],
) -> list[Any]:
    """Solve a hidden/coarse goal using only visible source facts."""

    if task_name == "meal_train_pass_serve":
        from data.rearrangement_game.meal_train_pass_serve.generator import _solve_min_passes_with_count

        demands = {int(seat): str(value) for seat, value in initial["demands"].items()}
        # The table size is visible input and varies per sample: seats that
        # want service plus seats marked as needing none.
        seat_count = len(demands) + len(initial.get("empty_seats") or [])
        plan, count = _solve_min_passes_with_count(
            int(initial["current_seat"]),
            demands,
            seat_count,
        )
        plan = list(plan) if plan is not None and count == 1 else None
    elif task_name == "office_inbox_priority_pull":
        from data.rearrangement_game.office_inbox_priority_pull.generator import _shortest_solution

        result = _shortest_solution(list(initial["inbox"]), max_moves=30)
        plan = list(result.path) if result is not None else None
    else:  # pragma: no cover - guarded by DERIVED_GOAL_TASKS
        raise ValueError(f"no source-only derived-goal solver for {task_name}")
    if plan is None:
        raise ValueError(f"{task_name} visible source does not determine a unique goal")
    return list(plan)


def _derive_target_state(
    task_name: str,
    initial: Any,
    parameters: Mapping[str, Any],
) -> Any:
    """Replay a source-solved plan to recover an answer-free terminal state."""

    plan = _derived_plan(task_name, initial, parameters)
    with tempfile.TemporaryDirectory(prefix="videogym-rearrange-derived-goal-") as tmp:
        generator = _new_generator(task_name, Path(tmp), video=False)
        state = deepcopy(initial)
        for move in plan:
            state = generator._apply_move(state, move)
        serializer = getattr(generator, "_serialize_state", None)
        return serializer(state) if callable(serializer) else deepcopy(state)


def _normalized(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalized(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _normalized(item) for key, item in value.items()}
    return value


VISIBLE_ROUTE_ANSWER_TASKS = frozenset({
    "bus_stops_dispatch",
    "maze_rush_hour",
    "meal_train_pass_serve",
})


def _diagnostic_metrics(sample: CanonicalSample) -> dict[str, Any]:
    """Ask the source evaluator to expose its independently recomputed oracle.

    The deliberately invalid empty plan supplies no candidate facts.  Task
    evaluators still report source-derived expected schedules/subsets where
    appropriate, which avoids duplicating those already-independent solvers.
    """

    metrics, _ = evaluate_text_from_source(sample, "Answer: [].")
    return metrics


def _exact_chess_plan(initial: Sequence[str], goal: Sequence[str]) -> list[Any]:
    from data.rearrangement_game.chess_pawn_swap.generator import _all_moves, _apply

    start = tuple(initial)
    target = tuple(goal)
    queue: deque[tuple[tuple[str, ...], list[Any]]] = deque([(start, [])])
    seen = {start}
    while queue:
        state, path = queue.popleft()
        if state == target:
            return path
        if len(path) >= 20:
            continue
        for move in _all_moves(state):
            next_state = _apply(state, move)
            if next_state in seen:
                continue
            seen.add(next_state)
            queue.append((next_state, [*path, move]))
    raise ValueError("visible chess-pawn target has no legal source-only plan")


def solve_object_plan(sample: CanonicalSample) -> list[Any]:
    """Recompute one valid/optimal plan from visible source and explicit rules."""

    task = str(sample.task_name)
    if task not in OBJECT_PLAN_TASKS:
        raise ValueError(f"{task} is not an ObjectPlan Rearrangement task")
    record = source_record(sample)
    initial = deepcopy(record["initial_state"])
    target = deepcopy(record["target_state"])

    direct_fields = {
        "meal_train_pass_serve": "expected",
    }
    if task in direct_fields:
        value = _diagnostic_metrics(sample).get(direct_fields[task])
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{task} source evaluator did not expose its recomputed plan")
        return [_normalized(item) for item in value]

    if task == "bookshelf_sort_by_height":
        from data.rearrangement_game.bookshelf_sort_by_height.generator import (
            minimum_swap_plan,
        )

        plan = minimum_swap_plan(tuple(initial), tuple(target), 24)
    elif task == "bus_stops_dispatch":
        from data.rearrangement_game.bus_stops_dispatch.generator import _shortest_plan

        plan = _shortest_plan(initial, int(record["capacity"]), 30)
    elif task == "chess_pawn_swap":
        plan = _exact_chess_plan(initial, target)
    elif task == "color_group_with_locks":
        from data.rearrangement_game.color_group_with_locks.generator import _bfs

        plan = _bfs(tuple(initial), tuple(record["locked_indices"]), 20)
    elif task == "container_ship_loading_order":
        from data.rearrangement_game.container_ship_loading_order.evaluator import (
            _shortest_plan,
        )

        plan = _shortest_plan(initial)
    elif task == "crane_block_stack":
        with tempfile.TemporaryDirectory(prefix="videogym-rearrange-solver-") as tmp:
            generator = _new_generator(task, Path(tmp))
            plan = generator._shortest_plan(initial, target, len(initial), max_moves=30)
    elif task == "dock_yard_crane_swap":
        from data.rearrangement_game.dock_yard_crane_swap.generator import _bfs

        plan = _bfs(initial, target, 30)
    elif task == "filing_cabinet_alphabet_lock":
        from data.rearrangement_game.filing_cabinet_alphabet_lock.generator import _bfs

        plan = _bfs(
            {key: tuple(value) for key, value in initial["drawers"].items()},
            {key: tuple(value) for key, value in target["drawers"].items()},
            set(record["drawers"]) - set(record["locked"]),
            30,
        )
    elif task == "jumping_frog_swap":
        from data.rearrangement_game.jumping_frog_swap.generator import (
            _shortest_solution,
        )

        result = _shortest_solution(initial, target, max_moves=30)
        plan = result.path if result is not None else None
    elif task == "lever_crane_swap":
        from data.rearrangement_game.lever_crane_swap.generator import _bfs

        plan = _bfs(
            tuple(initial[0]),
            tuple(initial[1]),
            tuple(target[0]),
            tuple(target[1]),
            30,
            True,
        )
    elif task == "office_inbox_priority_pull":
        from data.rearrangement_game.office_inbox_priority_pull.generator import (
            _shortest_solution,
        )

        result = _shortest_solution(initial["inbox"], max_moves=30)
        plan = result.path if result is not None else None
    elif task == "pancake_flip":
        from data.rearrangement_game.pancake_flip.generator import _bfs_optimal

        plan = _bfs_optimal(tuple(initial), tuple(target), 30)
    elif task == "parking_2d_n_by_n":
        from data.rearrangement_game.parking_2d_n_by_n.generator import _bfs

        plan = _bfs(
            tuple(initial), tuple(target), int(record["rows"]), int(record["cols"]), 30
        )
    elif task == "parking_lot_swap_minimal_moves":
        from data.rearrangement_game.parking_lot_swap_minimal_moves.generator import (
            _bfs,
        )

        plan = _bfs(tuple(initial), tuple(target), 30)
    elif task == "pipeline_hold_cells_3":
        from data.rearrangement_game.pipeline_hold_cells_3.generator import (
            _pipeline_solution,
        )

        result = _pipeline_solution(initial["arrival"], target["outfeed"], 3)
        plan = result.path if result is not None else None
    elif task == "stack_reversal_minimal_moves":
        from data.rearrangement_game.stack_reversal_minimal_moves.generator import (
            _shortest_stats,
        )

        plan = _shortest_stats(
            tuple(tuple(stack) for stack in initial),
            tuple(tuple(stack) for stack in target),
        )[2]
    elif task == "token_trade_matrix":
        from data.rearrangement_game.token_trade_matrix.generator import _shortest_stats

        plan = _shortest_stats(
            tuple(initial["holdings"]),
            tuple(target["holdings"]),
            frozenset(tuple(edge) for edge in record["edges"]),
        )[2]
    elif task == "tower_of_hanoi":
        from data.rearrangement_game.tower_of_hanoi.generator import _hanoi_optimal

        plan = _hanoi_optimal(
            tuple(tuple(stack) for stack in initial),
            tuple(tuple(stack) for stack in target),
            30,
        )
    elif task == "train_couple_swap":
        from data.rearrangement_game.train_couple_swap.generator import (
            _shortest_solution,
        )

        result = _shortest_solution(initial["track"], target["track"], max_moves=30)
        plan = result.path if result is not None else None
    elif task == "vending_dispense_fifo":
        from data.rearrangement_game.vending_dispense_fifo.generator import _bfs

        plan = _bfs(tuple(initial["chute"]), tuple(target["tray"]), 30)
    else:  # pragma: no cover - the explicit set above is a completeness gate.
        raise AssertionError(f"missing source solver for {task}")
    if plan is None:
        raise ValueError(f"{task} visible source has no legal plan")
    return [_normalized(item) for item in plan]


def replay_object_plan(
    sample: CanonicalSample,
    plan: Sequence[Any],
    *,
    include_motion: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[Any]]:
    """Replay and render semantic commits and optional continuous motion."""

    record = source_record(sample)
    initial = deepcopy(record["initial_state"])
    target = deepcopy(record["target_state"])
    with tempfile.TemporaryDirectory(prefix=f"videogym-{sample.task_name}-source-replay-") as tmp:
        generator = _new_generator(sample.task_name, Path(tmp), video=False)
        for key, value in record.items():
            if key in {"id", "initial_state", "target_state"}:
                continue
            if hasattr(generator, key):
                setattr(generator, key, deepcopy(value))
        state = deepcopy(initial)
        states: list[Any] = [deepcopy(state)]
        for move in plan:
            state = generator._apply_move(state, deepcopy(move))
            states.append(deepcopy(state))
        if generator._state_signature(state) != generator._state_signature(target):
            raise ValueError("source-only plan replay does not reach the visible exact target")
        stable = [
            np.asarray(generator._render_static_scene(item, target).convert("RGB"), dtype=np.uint8)
            for item in states
        ]
        motion_plan = list(plan)
        motion = (
            [
                np.asarray(frame, dtype=np.uint8)
                for frame in generator._build_video_frames(initial, target, motion_plan)
            ]
            if include_motion
            else []
        )
    if len(stable) < 2:
        raise ValueError("Rearrangement replay must contain distinct initial and terminal commits")
    return stable, motion, states


def solve_sliding_tile(sample: CanonicalSample) -> list[str]:
    """Recover the unique shortest 15-puzzle plan from the visible board."""

    from data.common.pipeline.sliding_tiles import _bfs_unique_solution, _goal_state

    record = source_record(sample)
    initial = deepcopy(record["initial_grid"])
    if int(record["board_size"]) != 4:
        raise ValueError("sliding_tile_15 source must be the published 4 by 4 board")
    plan = _bfs_unique_solution(initial, _goal_state(4))
    if not plan:
        raise ValueError("visible sliding-tile board lacks a unique nonempty shortest plan")
    return [str(move) for move in plan]


def replay_sliding_tile(
    sample: CanonicalSample,
    plan: Sequence[str],
    *,
    include_motion: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[Any]]:
    """Replay tile identities and render every cell-contact commit."""

    from data.common.pipeline.sliding_tiles import _apply_blank_move

    record = source_record(sample)
    initial = deepcopy(record["initial_grid"])
    with tempfile.TemporaryDirectory(prefix="videogym-sliding-source-replay-") as tmp:
        generator = _new_generator("sliding_tile_15", Path(tmp), video=False)
        states: list[Any] = [deepcopy(initial)]
        current = deepcopy(initial)
        for move in plan:
            current = _apply_blank_move(current, str(move))
            states.append(deepcopy(current))
        solved = [[row * 4 + col + 1 for col in range(4)] for row in range(4)]
        solved[-1][-1] = 0
        if current != solved:
            raise ValueError("sliding-tile replay does not reach the fixed visible-rule goal")
        stable = [
            np.asarray(generator._render_board(state).convert("RGB"), dtype=np.uint8)
            for state in states
        ]
        motion: list[np.ndarray] = []
        if include_motion:
            cell, ox, oy = generator._layout(4)
            first = generator._render_board(states[0])
            motion.extend(np.asarray(first, dtype=np.uint8) for _ in range(8))
            for before, after in zip(states, states[1:]):
                moved_value: Optional[int] = None
                from_cell: Optional[tuple[int, int]] = None
                to_cell: Optional[tuple[int, int]] = None
                for row in range(4):
                    for col in range(4):
                        if int(before[row][col]) == int(after[row][col]):
                            continue
                        if int(before[row][col]) == 0:
                            to_cell = (row, col)
                            moved_value = int(after[row][col])
                        else:
                            from_cell = (row, col)
                if from_cell is None or to_cell is None or moved_value is None:
                    raise ValueError("sliding-tile transition is not one adjacent identity move")
                display = [list(row) for row in before]
                display[from_cell[0]][from_cell[1]] = 0
                start_x, start_y = ox + from_cell[1] * cell, oy + from_cell[0] * cell
                end_x, end_y = ox + to_cell[1] * cell, oy + to_cell[0] * cell
                for substep in range(1, 5):
                    progress = substep / 4
                    frame = generator._render_board(
                        display,
                        moving_tile=(
                            moved_value,
                            start_x + (end_x - start_x) * progress,
                            start_y + (end_y - start_y) * progress,
                        ),
                    )
                    motion.append(np.asarray(frame, dtype=np.uint8))
            last = generator._render_board(states[-1])
            motion.extend(np.asarray(last, dtype=np.uint8) for _ in range(8))
            from data.common.video_encoding import fit_frames_to_video_budget, SIMPLE_VIDEO_FRAMES

            motion = fit_frames_to_video_budget(
                motion,
                max_frames=SIMPLE_VIDEO_FRAMES,
                final_hold_frames=8,
            )
    return stable, [np.asarray(frame, dtype=np.uint8) for frame in motion], states


def _rush_styled_cars(raw_cars: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from data.rearrangement_game.maze_rush_hour.generator import (
        DISTRACTOR_CAR_PRESETS,
        RED_CAR_BODY,
        RED_CAR_DARK,
        RED_CAR_WINDOW,
    )

    presets = {str(item["name"]): item for item in DISTRACTOR_CAR_PRESETS}
    cars: list[dict[str, Any]] = []
    for raw in raw_cars:
        car = deepcopy(dict(raw))
        color = str(car["color"])
        if color == "red":
            car.update(body=RED_CAR_BODY, dark=RED_CAR_DARK, window=RED_CAR_WINDOW)
        elif color in presets:
            car.update(
                body=tuple(presets[color]["body"]),
                dark=tuple(presets[color]["dark"]),
                window=tuple(presets[color]["window"]),
            )
        else:
            raise ValueError(f"rush-hour source contains unknown visible colour {color!r}")
        cars.append(car)
    return cars


def solve_rush_hour(sample: CanonicalSample) -> list[dict[str, Any]]:
    """Run the independent fixed-axis BFS over visible vehicle geometry."""

    from data.rearrangement_game.maze_rush_hour.evaluator import (
        _independent_analysis,
        _visible_problem,
    )

    record = source_record(sample)
    problem = _visible_problem(record)
    if problem is None:
        raise ValueError("rush-hour visible vehicle source violates the published contract")
    analysis = _independent_analysis(problem[0], problem[1], problem[2])
    if analysis is None:
        raise ValueError("rush-hour visible source has no legal exit plan")
    return [dict(move) for move in analysis[2]]


def replay_rush_hour(
    sample: CanonicalSample,
    plan: Sequence[Mapping[str, Any]],
    *,
    include_motion: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[Any]]:
    """Render fixed-identity vehicle commits and continuous axis motion."""

    from data.rearrangement_game.maze_rush_hour.evaluator import (
        _goal,
        _replay,
        _visible_problem,
    )

    record = source_record(sample)
    problem = _visible_problem(record)
    if problem is None:
        raise ValueError("invalid rush-hour visible source")
    replay = _replay(problem[0], problem[1], problem[2], list(plan))
    if replay["illegal_move_index"] is not None or not _goal(replay["state"], problem[2]):
        raise ValueError("rush-hour plan is illegal or does not fully exit the red vehicle")
    source_cars = _rush_styled_cars(record["cars"])
    state_cars: list[list[dict[str, Any]]] = []
    for state in replay["states"]:
        positions = {int(car_id): (int(row), int(col)) for car_id, row, col in state}
        cars = [deepcopy(car) for car in source_cars]
        for car in cars:
            car["row"], car["col"] = positions[int(car["id"])]
        state_cars.append(cars)
    with tempfile.TemporaryDirectory(prefix="videogym-rush-source-replay-") as tmp:
        generator = _new_generator("maze_rush_hour", Path(tmp), video=include_motion)
        padding = (int(generator.padding[0]), int(generator.padding[1]))
        canvas = tuple(int(value) for value in generator.canvas_dimensions)
        stable: list[np.ndarray] = []
        for cars in state_cars:
            clearance = generator._red_exit_offset_px(cars)
            image = generator._render_lot(
                cars=cars,
                draw_cell_ids=False,
                draw_car_badges=True,
                padding=padding,
                canvas_dims=canvas,
                exit_row=int(record["exit_row"]),
                red_extra_east_offset=clearance,
            )
            stable.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        motion: list[np.ndarray] = []
        if include_motion:
            path = generator._save_rush_hour_video(
                "source",
                source_cars,
                list(plan),
                padding=padding,
                canvas_dims=canvas,
                exit_row=int(record["exit_row"]),
            )
            if path is None:
                raise ValueError("rush-hour source motion renderer failed")
            import cv2

            capture = cv2.VideoCapture(path.as_posix())
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                motion.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            capture.release()
            if not motion:
                raise ValueError("rush-hour source motion video contains no frames")
    return stable, motion, state_cars


def solve_plan(sample: CanonicalSample) -> list[Any]:
    if sample.task_name in OBJECT_PLAN_TASKS:
        return solve_object_plan(sample)
    if sample.task_name == "sliding_tile_15":
        return solve_sliding_tile(sample)
    if sample.task_name == "maze_rush_hour":
        return solve_rush_hour(sample)
    raise ValueError(f"unsupported Rearrangement task {sample.task_name!r}")


def replay_plan(
    sample: CanonicalSample,
    plan: Sequence[Any],
    *,
    include_motion: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[Any]]:
    if sample.task_name in OBJECT_PLAN_TASKS:
        return replay_object_plan(sample, plan, include_motion=include_motion)
    if sample.task_name == "sliding_tile_15":
        return replay_sliding_tile(sample, plan, include_motion=include_motion)
    if sample.task_name == "maze_rush_hour":
        return replay_rush_hour(sample, plan, include_motion=include_motion)
    raise ValueError(f"unsupported Rearrangement task {sample.task_name!r}")


__all__ = [
    "VISIBLE_ROUTE_ANSWER_TASKS",
    "evaluate_text_from_source",
    "replay_plan",
    "replay_object_plan",
    "replay_rush_hour",
    "replay_sliding_tile",
    "solve_plan",
    "solve_object_plan",
    "solve_rush_hour",
    "solve_sliding_tile",
    "source_record",
]
