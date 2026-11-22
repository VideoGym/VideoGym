"""Source-only full-video verifier for the Spatial Navigation task family.

Consumes only ``CanonicalSample.semantic_input`` and the candidate video, never
reading ``ground_truth``, authored trajectories, terminal reference images, or
reference videos. Candidate paths are recovered from decoded frames and
replayed through task-owned transition systems.
"""

from __future__ import annotations

import importlib
import inspect
import math
import tempfile
from collections import deque
from copy import deepcopy
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from data.common.grid_coordinates import (
    format_named_xy_paths_answer,
    format_xy_path_answer,
    parse_named_xy_paths_answer,
    parse_xy_path_answer,
)
from data.registry import get_task_spec
from data.common.render_primitives import finish_scaled_canvas
from data.common.schemas import CanonicalSample, EvalRecord
from data.common.typed_answers import parse_answer_line
from data.common.verifiers.keyframes import visual_delta
from data.common.verifiers.region_gates import (
    REGION,
    STATIC_REGION_FLOOR,
    static_region_errors,
    terminal_region_gates,
    member_commit_gate,
)
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.video_encoding import MAX_VIDEO_FRAMES, validate_video_contract
from data.spatial_navigation.core import source_replay as _source_replay


SPATIAL_NAVIGATION_VERIFIER_PROFILE = "spatial_navigation_source_replay_final_delivery_v2"
_DIR_TO_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "stay": (0, 0),
}
_DELTA_TO_DIR = {value: key for key, value in _DIR_TO_DELTA.items()}

# Tasks whose formal text answer is an ordered list of model-facing 1-based
# ``(x, y)`` grid positions -- ``x`` is the row counted from the top, ``y`` the
# column counted from the left -- instead of linear cell IDs.  Their board
# renders 1-based row/column numbers along the grid edges rather than a number
# inside every cell.
from data.spatial_navigation.core.contract import (
    SINGLE_XY_ROUTE_TASKS as _SINGLE_XY_ROUTE_ANSWER_TASKS,
    XY_ROUTE_TASKS as _XY_ROUTE_ANSWER_TASKS,
)

_BASIC_GRID_TASKS = {"grid_obtaining_award"}
_OBJECT_PLAN_TASKS = {"plant_watering_route"}
_DUAL_AGENT_TASKS = {"cooperative_open_door"}
_MECHANIC_TASKS = set(_source_replay.SEMANTIC_AGENT_PATH_VIDEO_TASKS)
_PRIMARY_TASKS = set(_source_replay.SEMANTIC_PRIMARY_PATH_VIDEO_TASKS)
_PORTAL_ACTIVATION_TASKS = frozenset({"one_way_portal_chain"})
_PROCESS_STATE_TASKS = frozenset({
    "agent_courier_relay",
    "bridge_drop_tile",
    "bus_route_passengers",
    "carry_capacity_2",
    "color_matched_collection",
    "cooperative_open_door",
    "falling_block_dodge",
    "grid_obtaining_award",
    "lantern_oil_route",
    "lever_unlock_sequence",
    "parcel_delivery_route",
    "patrol_avoid_pattern",
    "switch_toggle_walls",
    "tide_water_cycle",
})

# Rejected recursively even if inserted under a nested source key.
# Canonical manifest export scrubs these in data/manifest.py; this second
# boundary makes the verifier independently safe against leaked answer fields.
_ANSWER_DERIVED_KEYS = {
    "action_sequence",
    "agent_path",
    "agent_path_cell_ids",
    "answer",
    "answer_grid",
    "answer_payload",
    "answer_state",
    "blue_path",
    "blue_suffix",
    "brace_steps",
    "canonical_path",
    "cart_path",
    "correct_option",
    "difficulty_variant",
    "drops",
    "expected_path",
    "expected_state",
    "final_board",
    "final_grid",
    "final_parity",
    "final_state",
    "ground_truth",
    "inspect_path",
    "intermediate_states",
    "leg1",
    "leg1_len",
    "leg2",
    "lever_steps",
    "max_total_cost",
    "move_count",
    "move_sequence",
    "moves",
    "open_board_path_length",
    "optimal_moves",
    "optimal_path",
    "path",
    "path_cells",
    "path_cell_ids",
    "path_length",
    "plate_step",
    "plate_steps",
    "press_step",
    "primary_moves",
    "primary_path",
    "push_count",
    "red_path",
    "reference_image",
    "reference_media",
    "reference_video",
    "saved_path",
    "secondary_moves",
    "secondary_path",
    "shortest_length",
    "shortest_path",
    "solution_image",
    "solution_path",
    "solution_video",
    "state_trace",
    "switch_step",
    "target_state",
    "teleport_steps",
    "terminal_state",
    "ti2t_answer",
    "wall_detour_extra_steps",
}


Coord = Tuple[int, int]


def _source_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _source_value(item)
            for key, item in value.items()
            if str(key) not in _ANSWER_DERIVED_KEYS
        }
    if isinstance(value, list):
        return [_source_value(item) for item in value]
    if isinstance(value, tuple):
        return [_source_value(item) for item in value]
    return deepcopy(value)


def _semantic_source(sample: CanonicalSample) -> Dict[str, Any]:
    if not isinstance(sample.semantic_input, dict):
        raise ValueError("Spatial Navigation verification requires canonical semantic_input")
    raw = _source_value(sample.semantic_input)
    if not isinstance(raw, dict):
        raise ValueError("semantic_input must be an object")
    state = raw.get("semantic_state")
    merged = {key: value for key, value in raw.items() if key != "semantic_state"}
    if isinstance(state, dict):
        merged.update(state)

    return merged


def _to_pair(value: Any) -> Optional[Coord]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except Exception:
        return None


def _to_path(value: Any) -> Optional[List[Coord]]:
    if not isinstance(value, list) or not value:
        return None
    result: List[Coord] = []
    for item in value:
        pair = _to_pair(item)
        if pair is None:
            return None
        result.append(pair)
    return result


def _moves_from_path(path: Sequence[Coord], *, allow_stay: bool = False) -> Optional[List[str]]:
    moves: List[str] = []
    for previous, current in zip(path, path[1:]):
        delta = (current[0] - previous[0], current[1] - previous[1])
        if allow_stay and delta == (0, 0):
            moves.append("stay")
            continue
        move = _DELTA_TO_DIR.get(delta)
        if move is None:
            return None
        moves.append(move)
    return moves


def _path_from_xy_positions(
    answer_line: str,
    *,
    grid_size: int,
) -> Optional[List[Coord]]:
    """Parse ``Answer: [(x, y), ...].`` into an internal 0-based ``(row, col)`` path.

    ``x`` is the 1-based row counted from the top and ``y`` the 1-based column
    counted from the left.  Anything that is not that exact typed grammar, or
    that names a position outside the grid, fails closed with ``None`` instead
    of being reinterpreted as a legacy linear cell ID.
    """

    cells = parse_xy_path_answer(answer_line)
    if not cells or grid_size <= 0:
        return None
    if any(not (0 <= row < grid_size and 0 <= col < grid_size) for row, col in cells):
        return None
    return cells


def _parse_single_path(
    structured: Dict[str, Any],
    answer_line: str,
    *,
    start: Optional[Coord],
    task_name: str = "",
) -> tuple[Optional[List[Coord]], List[str]]:
    if task_name in _XY_ROUTE_ANSWER_TASKS:
        try:
            size = int(structured.get("grid_size"))
        except Exception:
            size = 0
        path = _path_from_xy_positions(answer_line, grid_size=size)
        if path is None or start is None or path[0] != start:
            path = None
        return path, []
    return _source_replay._parse_single_predicted_path(structured, answer_line, start=start)


def _evaluate_grid_highest(
    structured: Dict[str, Any], answer_line: str
) -> tuple[Dict[str, Any], bool, Dict[str, Any]]:
    try:
        size = int(structured.get("grid_size"))
    except Exception:
        size = 0
    path = _path_from_xy_positions(answer_line, grid_size=size)
    start = _to_pair(structured.get("start"))
    goal = _to_pair(structured.get("goal"))
    blocked = set(_to_path(structured.get("blocked_cells")) or [])
    costs = _source_replay._to_int_grid(structured.get("cost_grid"))
    grid_ok = bool(
        size > 0
        and costs is not None
        and len(costs) == size
        and all(len(row) == size for row in costs)
    )
    best: Dict[Coord, int] = {}
    computed: Optional[int] = None
    if grid_ok and start is not None and goal is not None:
        best[start] = int(costs[start[0]][start[1]])
        for row in range(start[0], goal[0] + 1):
            for col in range(start[1], goal[1] + 1):
                cell = (row, col)
                if cell == start or cell in blocked:
                    continue
                incoming = [
                    best[previous]
                    for previous in ((row - 1, col), (row, col - 1))
                    if previous in best
                ]
                if incoming:
                    best[cell] = max(incoming) + int(costs[row][col])
        computed = best.get(goal)
    legal = bool(
        path
        and start is not None
        and goal is not None
        and path[0] == start
        and path[-1] == goal
        and goal not in path[:-1]
        and all(
            (current[0] - previous[0], current[1] - previous[1])
            in {(0, 0), (0, 1), (1, 0)}
            for previous, current in zip(path, path[1:])
        )
        and not any(cell in blocked for cell in path)
    )
    total = (
        int(costs[path[0][0]][path[0][1]])
        + sum(
            int(costs[current[0]][current[1]])
            for previous, current in zip(path, path[1:])
            if current != previous
        )
        if legal and costs is not None
        else None
    )
    solved = bool(legal and computed is not None and total == computed)
    metrics = {
        "canonical_answer_kind": "path",
        "text_parse_ok": path is not None,
        "text_path_legal": legal,
        "text_candidate_total_cost": total,
        "text_recomputed_max_total_cost": computed,
        "text_replay_check_pass": solved,
    }
    return metrics, solved, {
        "primary_path": path,
        "moves": _moves_from_path(path or [], allow_stay=True) or [],
    }


def _replay_object_plan(
    task_name: str, structured: Dict[str, Any], answer_line: str
) -> tuple[Dict[str, Any], bool, Dict[str, Any]]:
    if task_name != "plant_watering_route":
        return {"text_replay_reason": "unrouted Spatial Navigation object plan"}, False, {}
    metrics, solved = _source_replay._evaluate_plant_watering_text(structured, answer_line)
    try:
        rows = int(structured.get("rows") or 0)
        cols = int(structured.get("cols") or 0)
    except Exception:
        rows = cols = 0
    path = parse_xy_path_answer(answer_line)
    if (
        path is None
        or any(not (0 <= row < rows and 0 <= col < cols) for row, col in path)
    ):
        path = []
    moves = _moves_from_path(path, allow_stay=True) or []
    initial = deepcopy(structured.get("initial_state") or {})
    state = deepcopy(initial)
    trace = [deepcopy(state)]
    capacity = int(structured.get("capacity") or 0)
    refill_budget = int(
        structured.get("refill_budget", initial.get("refill_budget", 0)) or 0
    )
    plants = {tuple(item) for item in initial.get("plants") or []}
    for next_cell in path[1:]:
        cell = [int(next_cell[0]), int(next_cell[1])]
        can = int(state.get("can") or 0)
        watered = {tuple(item) for item in state.get("watered") or []}
        refills_used = int(state.get("refills_used") or 0)
        entering_well = (
            tuple(cell) == tuple(initial["well"])
            and tuple(cell) != tuple(state["gardener"])
        )
        if entering_well:
            if refills_used >= refill_budget:
                break
            refills_used += 1
            can = capacity
        if tuple(cell) in plants and tuple(cell) not in watered and can > 0:
            watered.add(tuple(cell))
            can -= 1
        state = {
            **state,
            "gardener": cell,
            "can": can,
            "watered": [list(item) for item in sorted(watered)],
            "refill_budget": refill_budget,
            "refills_used": refills_used,
        }
        trace.append(deepcopy(state))
    return metrics, solved, {
        "primary_path": path,
        "moves": moves,
        "state_trace": trace,
        "final_state": state,
    }


def _evaluate_text(
    sample: CanonicalSample,
    structured: Dict[str, Any],
    answer_line: str,
) -> tuple[Dict[str, Any], bool, Dict[str, Any]]:
    task = sample.task_name
    if parse_answer_line(answer_line) is None:
        return {"text_format_valid": False}, False, {}
    if task == "grid_highest_cost":
        return _evaluate_grid_highest(structured, answer_line)
    if task in _OBJECT_PLAN_TASKS:
        return _replay_object_plan(task, structured, answer_line)
    if task == "agent_courier_relay":
        metrics, solved = _source_replay._evaluate_courier_relay_text(structured, answer_line)
        parsed = parse_named_xy_paths_answer(answer_line, ("A", "B"))
        a = list(parsed["A"]) if parsed else []
        b = list(parsed["B"]) if parsed else []
        return metrics, solved, {
            "primary_path": a,
            "secondary_path": b,
            "primary_moves": _moves_from_path(a, allow_stay=True) or [],
            "secondary_moves": _moves_from_path(b, allow_stay=True) or [],
        }
    if task in _DUAL_AGENT_TASKS:
        metrics, solved = _source_replay._evaluate_dual_agent_text(sample, structured, answer_line)
        parsed = parse_named_xy_paths_answer(answer_line, ("blue", "orange"))
        primary = list(parsed["blue"]) if parsed else []
        secondary = list(parsed["orange"]) if parsed else []
        return metrics, solved, {
            "primary_moves": _moves_from_path(primary, allow_stay=True) or [],
            "secondary_moves": _moves_from_path(secondary, allow_stay=True) or [],
            "primary_path": primary,
            "secondary_path": secondary,
        }
    if task in _BASIC_GRID_TASKS:
        try:
            size = int(structured.get("grid_size") or 0)
        except Exception:
            size = 0
        path = _path_from_xy_positions(answer_line, grid_size=size)
        metrics, solved = _source_replay._replay_basic_grid_path(
            structured,
            path,
            task_name=task,
        )
        return metrics, solved, {
            "moves": _moves_from_path(path or [], allow_stay=True) or [],
            "primary_path": path or [],
        }
    if task in _MECHANIC_TASKS:
        start = _to_pair(structured.get("start"))
        path, moves = _parse_single_path(structured, answer_line, start=start, task_name=task)
        action_count, legal = _source_replay._replay_agent_mechanic_path(structured, task_name=task, path=path or [])
        solved = bool(path and legal and action_count is not None)
        return {
            "canonical_answer_kind": "path",
            "text_replay_legal_and_terminal": legal,
            "text_predicted_action_count": action_count,
            "text_replay_check_pass": solved,
        }, solved, {"primary_path": path or [], "moves": moves}
    if task in _PRIMARY_TASKS:
        extras = structured.get("extras") or {}
        start = _to_pair(extras.get("agent_start")) or _to_pair(extras.get("bus_start"))
        path, moves = _parse_single_path(structured, answer_line, start=start, task_name=task)
        model = _source_replay._build_primary_mechanic_model(structured, task_name=task)
        if model is None or not path:
            action_count, legal = None, False
        else:
            action_count, legal = _source_replay._replay_transition_model_path(model, path)
        solved = bool(path and legal and action_count is not None)
        return {
            "canonical_answer_kind": "path",
            "text_replay_legal_and_terminal": legal,
            "text_predicted_action_count": action_count,
            "text_replay_check_pass": solved,
        }, solved, {"primary_path": path or [], "moves": moves}
    return {"text_replay_reason": "unrouted spatial_navigation task"}, False, {}


def _instantiate_generator(task_name: str, structured: Dict[str, Any]) -> Any:
    spec = get_task_spec(task_name)
    if spec is None:
        raise ValueError(f"Unknown Spatial Navigation task {task_name}")
    cls = getattr(importlib.import_module(spec.module), spec.class_name)
    parameters = {name for base in cls.__mro__ if base is not object for name in inspect.signature(base.__init__).parameters}
    kwargs: Dict[str, Any] = {"output_dir": Path(tempfile.gettempdir()) / "videogym_spatial_navigation_source_render"}
    if "seed" in parameters:
        kwargs["seed"] = 0
    if "video" in parameters:
        kwargs["video"] = False
    size = structured.get("grid_size")
    if size is not None and "size" in parameters:
        kwargs["size"] = int(size)
    initial_grid = _source_replay._to_int_grid(structured.get("initial_grid"))
    if initial_grid:
        if "height" in parameters:
            kwargs["height"] = len(initial_grid)
        if "width" in parameters:
            kwargs["width"] = len(initial_grid[0])
    return cls(**kwargs)


def _mechanic_scenario(task: str, structured: Dict[str, Any], plan: Dict[str, Any]) -> tuple[Any, Any]:
    from data.spatial_navigation.core.mechanic_base import MechanicScenario

    generator = _instantiate_generator(task, structured)
    path = [tuple(item) for item in plan.get("primary_path") or []]
    extras = deepcopy(structured.get("extras") or {})
    if task == "one_way_portal_chain":
        jumps = [
            index
            for index, (previous, current) in enumerate(zip(path, path[1:]))
            if abs(current[0] - previous[0]) + abs(current[1] - previous[1]) != 1
        ]
        extras["teleport_steps"] = jumps
    scenario = MechanicScenario(
        task_name=task,
        task_family="spatial_navigation",
        title=task,
        input_desc="",
        rules_desc="",
        ti2v_output="",
        solution_image_output="",
        size=int(structured.get("grid_size") or 0),
        start=tuple(_to_pair(structured.get("start")) or (-1, -1)),
        goal=tuple(_to_pair(structured.get("goal")) or (-1, -1)),
        agent_path=path,
        walls=[tuple(item) for item in _to_path(structured.get("walls")) or []],
        extras=extras,
    )
    return generator, scenario


def _render_mechanic_terminal(
    task: str, structured: Dict[str, Any], plan: Dict[str, Any], *, show_path: bool
) -> Image.Image:
    generator, scenario = _mechanic_scenario(task, structured, plan)
    path = scenario.agent_path
    goal_center = generator._cell_center(scenario.goal)
    kwargs: Dict[str, Any] = {"show_path": show_path}
    parameters = inspect.signature(generator._render).parameters
    if "agent_center" in parameters:
        kwargs["agent_center"] = goal_center
    if "agent_cell" in parameters:
        kwargs["agent_cell"] = scenario.goal
    if "step_index" in parameters:
        kwargs["step_index"] = max(0, len(path) - 1)
    if "dropped_so_far" in parameters:
        water = {tuple(item) for item in scenario.extras.get("water") or []}
        dropped: List[Coord] = []
        for cell in path:
            if cell in water and cell not in dropped:
                dropped.append(cell)
        kwargs["dropped_so_far"] = dropped
    if "held" in parameters:
        kwargs["held"] = 0
    if "delivered" in parameters:
        kwargs["delivered"] = len(scenario.extras.get("cargo_cells") or [])
    if "delivery_commits" in parameters:
        kwargs["delivery_commits"] = int(scenario.extras.get("delivery_budget") or 0)
    if "cargo_present" in parameters:
        kwargs["cargo_present"] = [False] * len(scenario.extras.get("cargo_cells") or [])
    if "gravity_down" in parameters:
        kwargs["gravity_down"] = False
    if "keys_picked" in parameters:
        kwargs["keys_picked"] = [True] * len(scenario.extras.get("keys") or [])
    if "doors_open" in parameters:
        count = len(scenario.extras.get("doors") or scenario.extras.get("door_cells") or [])
        kwargs["doors_open"] = [True] * count
    if "oil" in parameters:
        oil = int(scenario.extras.get("max_oil") or 0)
        refills = {tuple(item) for item in scenario.extras.get("refills") or []}
        previous = path[0] if path else None
        for cell in path[1:]:
            oil -= 1
            if cell != previous and cell in refills:
                oil = int(scenario.extras.get("max_oil") or 0)
            previous = cell
        kwargs["oil"] = oil
    if "levers_pulled" in parameters:
        kwargs["levers_pulled"] = [True] * len(scenario.extras.get("lever_cells") or [])
    if "door_open" in parameters:
        kwargs["door_open"] = True
    if "plates_active" in parameters:
        kwargs["plates_active"] = [True] * len(scenario.extras.get("plate_cells") or [])
    if "plate_active" in parameters:
        kwargs["plate_active"] = bool(show_path)
    if "switch_active" in parameters:
        kwargs["switch_active"] = True
    if "collected" in parameters:
        kwargs["collected"] = [True] * len(scenario.extras.get("rewards") or [])
    if "remaining_steps" in parameters:
        kwargs["remaining_steps"] = 0
    return generator._render(scenario, **kwargs).convert("RGB")


def _basic_scenario(
    task: str,
    structured: Dict[str, Any],
    plan: Dict[str, Any],
) -> tuple[Any, Any]:
    from data.spatial_navigation.core.base import GridReward, GridScenario, _REWARD_COLORS

    generator = _instantiate_generator(task, structured)
    size = int(structured.get("grid_size") or 0)
    if hasattr(generator, "_resize_to_spec"):
        generator._resize_to_spec(size)
    path = [tuple(item) for item in plan.get("primary_path") or []]
    required_items = structured.get("required_items") or []
    rewards = []
    for index, item in enumerate(required_items):
        if not isinstance(item, dict) or _to_pair(item.get("cell")) is None:
            continue
        rewards.append(GridReward(
            cell=tuple(_to_pair(item["cell"])),
            color=_REWARD_COLORS[index % len(_REWARD_COLORS)],
            color_name=str(item.get("color") or ""),
            style=str(item.get("style") or "circle"),
        ))
    scenario = GridScenario(
        task_name=task, task_family="spatial_navigation", title=task,
        input_desc="", rules_desc="", ti2v_output="",  size=size,
        start=tuple(_to_pair(structured.get("start")) or (-1, -1)),
        goal=tuple(_to_pair(structured.get("goal")) or (-1, -1)),
        move_sequence=list(plan.get("moves") or []),
        path_cells=path,
        blocked_cells=[tuple(item) for item in _to_path(structured.get("blocked_cells")) or []],
        rewards=rewards,
        start_color=(61, 181, 99), goal_color=(228, 52, 52), agent_color=(245, 130, 32),
    )
    return generator, scenario


def _render_basic_terminal(
    task: str, structured: Dict[str, Any], plan: Dict[str, Any], *, show_path: bool
) -> Image.Image:
    generator, scenario = _basic_scenario(
        task,
        structured,
        plan,
    )
    return generator._render_board(
        scenario,
        agent_cell=scenario.goal,
        collected_indices=range(len(scenario.rewards)),
        show_path=show_path,
        path_cells=scenario.path_cells if show_path else (),
    ).convert("RGB")


def _render_cost_terminal(
    structured: Dict[str, Any], plan: Dict[str, Any], *, show_path: bool
) -> Image.Image:
    from data.spatial_navigation.grid_highest_cost.generator import CostGridScenario

    generator = _instantiate_generator("grid_highest_cost", structured)
    size = int(structured.get("grid_size") or 0)
    path = [tuple(item) for item in plan.get("primary_path") or []]
    costs = _source_replay._to_int_grid(structured.get("cost_grid")) or []
    scenario = CostGridScenario(
        task_name="grid_highest_cost", task_family="spatial_navigation", title="grid_highest_cost",
        input_desc="", rules_desc="", ti2v_output="",  size=size,
        start=tuple(_to_pair(structured.get("start")) or (-1, -1)),
        goal=tuple(_to_pair(structured.get("goal")) or (-1, -1)),
        move_sequence=_moves_from_path(path, allow_stay=True) or [], path_cells=path,
        blocked_cells=[tuple(item) for item in _to_path(structured.get("blocked_cells")) or []],
        rewards=[],
        start_color=(61, 181, 99),
        goal_color=(228, 52, 52),
        agent_color=(245, 130, 32),
        cost_grid=costs, max_total_cost=0,
    )
    return generator._render_board(
        scenario, agent_cell=scenario.goal, show_path=show_path,
        path_cells=path if show_path else (),
    ).convert("RGB")


def _multi_scenario(
    task: str,
    structured: Dict[str, Any],
    plan: Dict[str, Any],
) -> tuple[Any, Any]:
    from data.spatial_navigation.core.multi_agent_base import MultiAgentScenario

    generator = _instantiate_generator(task, structured)
    primary = [tuple(item) for item in plan.get("primary_path") or []]
    secondary = [tuple(item) for item in plan.get("secondary_path") or []]
    extras = deepcopy(structured.get("extras") or {})
    if task == "hidden_treasure_dig":
        derived_treasure = _source_replay._derive_hidden_treasure(
            extras,
            size=int(structured.get("grid_size") or 0),
        )
        if derived_treasure is not None:
            extras["treasure"] = list(derived_treasure)
    primary_moves = list(plan.get("primary_moves") or plan.get("moves") or _moves_from_path(primary, allow_stay=True) or [])
    secondary_moves = list(plan.get("secondary_moves") or _moves_from_path(secondary, allow_stay=True) or [])
    if task == "agent_courier_relay" and primary and secondary:
        extras.update({"leg1": [list(item) for item in primary], "leg2": [list(item) for item in secondary], "leg1_len": len(primary)})
    if task == "cooperative_open_door" and primary and secondary:
        blue_plate = _to_pair(extras.get("blue_plate"))
        orange_plate = _to_pair(extras.get("orange_plate"))
        extras["press_step"] = next((i for i, pair in enumerate(zip(primary, secondary)) if pair == (blue_plate, orange_plate)), -1)
        door_cells = {tuple(item) for item in extras.get("door_cells") or []}
        extras["cross_step"] = next(
            (index for index, cell in enumerate(primary) if cell in door_cells),
            -1,
        )
    extras["final_parity"] = len(primary_moves) % 2
    scenario = MultiAgentScenario(
        task_name=task, task_family="spatial_navigation", title=task,
        size=int(structured.get("grid_size") or 0),
        walls=[tuple(item) for item in _to_path(structured.get("walls")) or []],
        primary_path=primary,
        secondary_path=secondary,
        primary_moves=primary_moves,
        secondary_moves=secondary_moves,
        answer_text="",
        extras=extras,
    )
    return generator, scenario


def _render_multi_terminal(
    task: str, structured: Dict[str, Any], plan: Dict[str, Any], *, show_path: bool
) -> Image.Image:
    from data.spatial_navigation.core.multi_agent_base import FrameState

    generator, scenario = _multi_scenario(task, structured, plan)
    primary = list(scenario.primary_path)
    secondary = list(scenario.secondary_path)
    states = generator.build_video_frame_states(scenario, 1)
    if states:
        frame_state = generator._discrete_frame_state(states[-1])
    else:
        frame_state = FrameState(
            primary_pos=primary[-1], primary_cell=primary[-1],
            secondary_pos=secondary[-1] if secondary else None,
            secondary_cell=secondary[-1] if secondary else None,
        )
    return generator.make_image(
        scenario, frame_state=frame_state, show_path=show_path,
        path_cells=primary if show_path else None,
    ).convert("RGB")


def _render_object_terminal(
    task: str, structured: Dict[str, Any], plan: Dict[str, Any]
) -> Image.Image:
    generator = _instantiate_generator(task, structured)
    final_state = deepcopy(plan.get("final_state") or {})
    target_state = deepcopy(final_state)
    return generator._render_static_scene(final_state, target_state).convert("RGB")


def _render_terminal(
    sample: CanonicalSample,
    structured: Dict[str, Any],
    plan: Dict[str, Any],
    *,
    show_path: bool,
) -> Image.Image:
    task = sample.task_name
    if task in _OBJECT_PLAN_TASKS:
        terminal = np.asarray(_render_object_terminal(task, structured, plan)).copy()
        # The right-hand goal illustration is supplied input, not a required
        # final gardener/resource configuration for every legal route.
        puzzle = _rgb(Path(sample.primary_input_image))
        geometry = _instantiate_generator(task, structured)._panel_geometry()
        left = int(geometry["right_panel_x0"])
        top = int(geometry["right_panel_y0"])
        width, height = int(geometry["panel_w"]), int(geometry["panel_h"])
        terminal[top:top + height, left:left + width] = puzzle[top:top + height, left:left + width]
        return Image.fromarray(terminal)
    if task == "grid_highest_cost":
        return _render_cost_terminal(structured, plan, show_path=show_path)
    if task in _BASIC_GRID_TASKS:
        return _render_basic_terminal(task, structured, plan, show_path=show_path)
    if task in _MECHANIC_TASKS:
        return _render_mechanic_terminal(task, structured, plan, show_path=show_path)
    if task in _PRIMARY_TASKS:
        return _render_multi_terminal(task, structured, plan, show_path=show_path)
    raise ValueError(f"No source renderer for {task}")


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _image_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 1.0
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) / 255.0)


def _video_frames(path: Path) -> tuple[List[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, fps


def _grid_geometry(task: str, structured: Dict[str, Any]) -> tuple[int, float, float, float]:
    if task in _OBJECT_PLAN_TASKS:
        rows = int(structured.get("rows") or 4)
        cols = int(structured.get("cols") or rows)
        generator = _instantiate_generator(task, structured)
        panel = generator._panel_geometry()
        panel_w = int(panel["panel_w"])
        panel_h = int(panel["panel_h"])
        layout = generator._layout(panel_w, panel_h)
        x0 = int(panel["left_panel_x0"]) + int(layout["board_x0"])
        y0 = int(panel["left_panel_y0"]) + int(layout["board_y0"])
        return max(rows, cols), float(layout["cell"]), float(x0), float(y0)
    initial = _source_replay._to_int_grid(structured.get("initial_grid"))
    size = int(structured.get("grid_size") or (len(initial) if initial else 0))
    cell = float(max(28, int(512 / (size + 1)) // 2 * 2))
    margin = float((512 - size * int(cell)) // 2)
    return size, cell, margin, margin


def _token_specs(task: str) -> List[tuple[str, Tuple[int, int, int], float]]:
    if task == "plant_watering_route":
        return [("primary", (96, 132, 188), 62.0)]
    if task in _MECHANIC_TASKS:
        return [("primary", (239, 133, 59), 58.0)]
    if task in _BASIC_GRID_TASKS or task == "grid_highest_cost":
        return [("primary", (245, 130, 32), 58.0)]
    if task == "bus_route_passengers":
        return [("primary", (245, 200, 60), 55.0)]
    if task == "color_matched_collection":
        return [("primary", (80, 80, 90), 38.0)]
    if task in _PRIMARY_TASKS:
        specs = [("primary", (56, 130, 220), 58.0)]
        if task == "patrol_avoid_pattern":
            specs.append(("secondary", (220, 60, 60), 58.0))
        if task in _DUAL_AGENT_TASKS or task == "agent_courier_relay":
            specs.append(("secondary", (238, 132, 36), 58.0))
        return specs
    return []


def _source_starts(task: str, structured: Dict[str, Any]) -> Dict[str, Optional[Coord]]:
    extras = structured.get("extras") or {}
    if task == "plant_watering_route":
        initial = structured.get("initial_state") or {}
        return {"primary": _to_pair(initial.get("gardener"))}
    if task in _MECHANIC_TASKS or task in _BASIC_GRID_TASKS or task == "grid_highest_cost":
        return {"primary": _to_pair(structured.get("start"))}
    if task == "agent_courier_relay":
        return {"primary": _to_pair(extras.get("a_start")), "secondary": _to_pair(extras.get("b_meet"))}
    if task == "patrol_avoid_pattern":
        patrol_loop = _to_path(extras.get("patrol_loop")) or []
        return {
            "primary": _to_pair(extras.get("agent_start")),
            "secondary": patrol_loop[0] if patrol_loop else None,
        }
    if task in _DUAL_AGENT_TASKS:
        return {"primary": _to_pair(extras.get("blue_start")), "secondary": _to_pair(extras.get("orange_start"))}
    return {"primary": _to_pair(extras.get("agent_start")) or _to_pair(extras.get("bus_start"))}


def _component_centers(
    frame: np.ndarray, color: Tuple[int, int, int], tolerance: float, *,
    minimum_blue_dominance: float = 0.0, minimum_red: float = 0.0,
    minimum_red_green_dominance: float = 0.0,
    minimum_disc_radius: float = 0.0,
) -> List[tuple[float, float, int]]:
    delta = frame.astype(np.float32) - np.asarray(color, dtype=np.float32)
    mask = np.linalg.norm(delta, axis=2) <= tolerance
    if minimum_blue_dominance:
        mask &= frame[:, :, 2].astype(float) - frame[:, :, 1].astype(float) >= minimum_blue_dominance
    if minimum_red:
        mask &= frame[:, :, 0] >= minimum_red
    if minimum_red_green_dominance:
        mask &= frame[:, :, 0].astype(float) - frame[:, :, 1].astype(float) >= minimum_red_green_dominance
    mask = mask.astype(np.uint8)
    if minimum_disc_radius:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    radius_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5) if minimum_disc_radius else None
    result = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= 18:
            if radius_map is not None:
                component = _labels == index
                radius = float(radius_map[component].max())
                if radius < minimum_disc_radius:
                    continue
                ys, xs = np.where(component & (radius_map >= .60 * radius))
                # A disc can overlap a same-colour mechanic. Its deepest
                # interior locates the disc without shifting to a lever stem
                # or the centroid of a connected water/ripple fragment.
                result.append((float(xs.mean()), float(ys.mean()), int(round(math.pi * radius * radius))))
            else:
                result.append((float(centroids[index][0]), float(centroids[index][1]), area))
    return result


def _cell_for_center(
    center: tuple[float, float], *, size: int, cell: float, x0: float, y0: float
) -> Optional[Coord]:
    col = int(round((center[0] - x0) / cell - 0.5))
    row = int(round((center[1] - y0) / cell - 0.5))
    if 0 <= row < size and 0 <= col < size:
        return row, col
    return None


def _track_tokens(
    task: str, structured: Dict[str, Any], frames: Sequence[np.ndarray]
) -> tuple[Dict[str, List[Coord]], Dict[str, Any]]:
    from data.common.prompt_paraphrase import active_profile
    size, cell, x0, y0 = _grid_geometry(task, structured)
    starts = _source_starts(task, structured)
    tracks: Dict[str, List[Coord]] = {}
    coverage: Dict[str, float] = {}
    frame_tracks: Dict[str, List[Optional[Coord]]] = {}
    frame_detections: Dict[str, List[bool]] = {}
    frame_centers: Dict[str, List[Optional[tuple[float, float]]]] = {}
    for name, color, tolerance in _token_specs(task):
        previous = starts.get(name)
        previous_center = (
            (x0 + (previous[1] + 0.5) * cell, y0 + (previous[0] + 0.5) * cell)
            if previous is not None
            else None
        )
        target_area: Optional[int] = None
        per_frame: List[Optional[Coord]] = []
        per_frame_detected: List[bool] = []
        per_frame_centers: List[Optional[tuple[float, float]]] = []
        detected = 0
        for frame in frames:
            candidates = []
            for cx, cy, area in _component_centers(
                frame, color, 85 if task in {"lever_unlock_sequence", "tide_water_cycle"} else tolerance,
                minimum_blue_dominance=45 if task == "tide_water_cycle" else 0,
                minimum_red=210 if task == "lever_unlock_sequence" else 0,
                # The authored amber valves lie inside the orange agent's
                # RGB distance ball. Their red-green contrast is 54, versus
                # 106 for the agent, so separate these visible color roles
                # before identity/area tracking can jump to a static valve.
                minimum_red_green_dominance=75 if task == "conveyor_belt_path" and active_profile("pneumatic_conveyors") else 0,
                minimum_disc_radius=.12 * cell if task in {"lever_unlock_sequence", "tide_water_cycle"} else 0,
            ):
                if task in _OBJECT_PLAN_TASKS and cx >= 250:
                    continue
                cell_coord = _cell_for_center((cx, cy), size=size, cell=cell, x0=x0, y0=y0)
                if cell_coord is None:
                    continue
                distance = 0.0 if previous_center is None else math.hypot(cx - previous_center[0], cy - previous_center[1])
                if target_area is None:
                    source_center = previous_center
                    source_distance = 0.0 if source_center is None else math.hypot(cx - source_center[0], cy - source_center[1])
                    score = source_distance - min(area, 1200) * 0.01
                else:
                    # Task scenes often retain a same-colour goal, cargo, well,
                    # or conveyor under the moving token.  Token fill area is
                    # much more stable than those static shapes.  Permit a
                    # multi-cell portal jump, but otherwise prefer the stable
                    # identity-sized component over a component left behind.
                    if task in {"lever_unlock_sequence", "tide_water_cycle"}:
                        area_error = abs(area - target_area) / max(1, target_area)
                        center_x = x0 + (cell_coord[1] + .5) * cell
                        center_y = y0 + (cell_coord[0] + .5) * cell
                        center_error = math.hypot(cx - center_x, cy - center_y) / cell
                        # Rounded disc observations separate the shaded agent
                        # from lever/water fragments in these two renderers.
                        score = area_error + center_error + .15 * distance / cell
                    else:
                        # Other source renderers retain same-colour stationary
                        # goals/mechanics whose fills closely resemble a token.
                        # Preserve the token's learned area as the identity key;
                        # continuity alone would keep tracking a depot/switch
                        # after the actual token leaves it.
                        score = 3.0 * abs(area - target_area) + .01 * distance
                candidates.append((score, cell_coord, (cx, cy), area))
            if candidates:
                _score, cell_coord, chosen_center, chosen_area = min(
                    candidates, key=lambda item: item[0]
                )
                if target_area is None:
                    target_area = int(chosen_area)
                previous = cell_coord
                previous_center = chosen_center
                per_frame.append(cell_coord)
                per_frame_detected.append(True)
                per_frame_centers.append(chosen_center)
                detected += 1
            else:
                per_frame.append(previous)
                per_frame_detected.append(False)
                per_frame_centers.append(previous_center)
        compressed: List[Coord] = []
        if starts.get(name) is not None:
            compressed.append(tuple(starts[name]))
        for item in per_frame:
            if item is not None and (not compressed or item != compressed[-1]):
                compressed.append(item)
        tracks[name] = compressed
        coverage[name] = detected / max(1, len(frames))
        frame_tracks[name] = list(per_frame)
        frame_detections[name] = per_frame_detected
        frame_centers[name] = per_frame_centers
    return tracks, {
        "video_token_detection_coverage": coverage,
        "video_recovered_tracks": {name: [list(cell) for cell in track] for name, track in tracks.items()},
        # Internal candidate observations consumed below before metrics are
        # serialized.  Keeping the aligned frame index is essential: separate
        # duplicate-compressed paths cannot prove synchronized commits/stays.
        "_video_frame_token_tracks": frame_tracks,
        "_video_frame_token_detections": frame_detections,
        "_video_frame_token_centers": frame_centers,
    }


def _joint_commit_states(
    task: str,
    structured: Dict[str, Any],
    frame_tracks: Mapping[str, Sequence[Optional[Coord]]],
    *,
    synchronization_tolerance_frames: int = 2,
) -> tuple[List[tuple[Coord, Coord]], Dict[str, Any], bool]:
    """Recover synchronized semantic ticks from aligned candidate frames.

    A joint state is the pair of visible cells, not an event emitted only when
    both tokens move.  In particular, one agent is allowed to stay while the
    other moves (the cooperative-door and object-plan contracts rely on this),
    and a terminal rendezvous is a real state even though both identities share
    one cell.  The old event-only reconstruction discarded those legal stay
    ticks and treated every nearby movement as an invalid asynchronous commit.
    """

    starts = _source_starts(task, structured)
    primary_start = starts.get("primary")
    secondary_start = starts.get("secondary")
    primary_frames = list(frame_tracks.get("primary") or [])
    secondary_frames = list(frame_tracks.get("secondary") or [])
    if (
        primary_start is None
        or secondary_start is None
        or not primary_frames
        or len(primary_frames) != len(secondary_frames)
    ):
        return [], {"video_joint_tick_reason": "missing aligned candidate token tracks"}, False

    # Token tracking forward-fills an occasionally unreadable frame.  Keep the
    # aligned pair and compress only exact repeated pairs.  This preserves a
    # state such as ``(blue moved, orange stayed)`` instead of dropping it.
    missing_cell = any(primary is None or secondary is None for primary, secondary in zip(primary_frames, secondary_frames))
    pairs: List[tuple[Coord, Coord]] = []
    pair_frame_starts: List[int] = []
    for frame_index, (primary, secondary) in enumerate(zip(primary_frames, secondary_frames)):
        if primary is None or secondary is None:
            continue
        pair = (tuple(primary), tuple(secondary))
        if not pairs or pair != pairs[-1]:
            pairs.append(pair)
            pair_frame_starts.append(frame_index)

    if not pairs:
        return [], {"video_joint_tick_reason": "no jointly observable candidate states"}, False
    if pairs[0] != (tuple(primary_start), tuple(secondary_start)):
        # A candidate may lose the first token for one encoded frame, but its
        # first recovered state must still be the supplied input state.
        pairs.insert(0, (tuple(primary_start), tuple(secondary_start)))
        pair_frame_starts.insert(0, 0)

    serialized_events: List[Dict[str, Any]] = []
    skews: List[int] = []
    atomicity_ok = True
    for index, (previous, current) in enumerate(zip(pairs, pairs[1:]), start=1):
        primary_changed = current[0] != previous[0]
        secondary_changed = current[1] != previous[1]
        serialized_events.append({
            "frame_start": pair_frame_starts[index],
            "frame_end": (
                pair_frame_starts[index + 1] - 1
                if index + 1 < len(pair_frame_starts)
                else len(primary_frames) - 1
            ),
            "primary": list(current[0]),
            "secondary": list(current[1]),
            "primary_changed": primary_changed,
            "secondary_changed": secondary_changed,
        })
        if primary_changed and secondary_changed:
            # A jointly committed state is visible at the same decoded frame
            # once the pair changes.  The skew is therefore zero; retain the
            # field for diagnostics and future encoders that expose a one-frame
            # decode offset.
            skews.append(0)

    states = list(pairs)
    cluster_valid = bool(states) and not missing_cell and atomicity_ok
    return states, {
        "video_joint_tick_count": max(0, len(states) - 1),
        "video_joint_tick_events": serialized_events,
        "video_joint_max_commit_skew_frames": max(skews, default=0),
        "video_joint_tick_reconstruction_pass": cluster_valid,
    }, cluster_valid


def _move_from_cells(previous: Coord, current: Coord, *, uppercase: bool = False) -> Optional[str]:
    delta = current[0] - previous[0], current[1] - previous[1]
    if delta == (0, 0):
        return "S" if uppercase else "stay"
    move = _DELTA_TO_DIR.get(delta)
    if move is None:
        return None
    return {"up": "U", "right": "R", "down": "D", "left": "L"}[move] if uppercase else move


def _joint_detection_integrity(
    task: str,
    structured: Dict[str, Any],
    frame_tracks: Mapping[str, Sequence[Optional[Coord]]],
    frame_detections: Mapping[str, Sequence[bool]],
) -> tuple[Dict[str, Any], bool]:
    primary_cells = list(frame_tracks.get("primary") or [])
    secondary_cells = list(frame_tracks.get("secondary") or [])
    primary_flags = list(frame_detections.get("primary") or [])
    secondary_flags = list(frame_detections.get("secondary") or [])
    lengths_ok = bool(
        primary_cells
        and len(primary_cells) == len(secondary_cells) == len(primary_flags) == len(secondary_flags)
    )
    if not lengths_ok:
        return {"video_joint_detection_integrity_pass": False}, False

    terminal_frame = -1
    integrity = bool(all(primary_flags) and all(secondary_flags))
    return {
        "video_joint_detection_integrity_pass": integrity,
        "video_joint_terminal_contact_frame": terminal_frame,
        "video_primary_missing_frame_count": sum(not flag for flag in primary_flags),
        "video_secondary_missing_frame_count": sum(not flag for flag in secondary_flags),
    }, integrity


def _verify_joint_video_tracks(
    sample: CanonicalSample,
    structured: Dict[str, Any],
    frame_tracks: Mapping[str, Sequence[Optional[Coord]]],
    frame_detections: Mapping[str, Sequence[bool]],
) -> tuple[Dict[str, Any], bool, Dict[str, Any]]:
    states, tick_metrics, ticks_ok = _joint_commit_states(
        sample.task_name, structured, frame_tracks
    )
    detection_metrics, detections_ok = _joint_detection_integrity(
        sample.task_name, structured, frame_tracks, frame_detections
    )
    primary_moves: List[str] = []
    secondary_moves: List[str] = []
    moves_ok = bool(ticks_ok and len(states) >= 2)
    for previous, current in zip(states, states[1:]):
        primary_move = _move_from_cells(previous[0], current[0])
        secondary_move = _move_from_cells(previous[1], current[1])
        if primary_move is None or secondary_move is None:
            moves_ok = False
            break
        primary_moves.append(primary_move)
        secondary_moves.append(secondary_move)

    synthetic: Optional[str] = None
    if moves_ok:
        synthetic = format_named_xy_paths_answer(
            {
                "blue": [state[0] for state in states],
                "orange": [state[1] for state in states],
            }
        )

    if synthetic is None:
        replay_metrics, replay_ok, plan = {}, False, {}
    else:
        replay_metrics, replay_ok, plan = _evaluate_text(sample, structured, synthetic)
    passed = bool(
        ticks_ok
        and detections_ok
        and moves_ok
        and replay_ok
    )
    return {
        **tick_metrics,
        **detection_metrics,
        **replay_metrics,
        "video_joint_moves_parse_pass": moves_ok,
        "video_joint_source_replay_pass": replay_ok,
    }, passed, plan


def _cellwise_state_distance(
    candidate: np.ndarray,
    expected: np.ndarray,
    *,
    rows: int,
    cols: int,
    cell_size: float,
    x0: float,
    y0: float,
) -> float:
    if candidate.shape != expected.shape or rows <= 0 or cols <= 0:
        return 1.0
    distances: List[float] = []
    inset = max(2, int(round(cell_size * 0.08)))
    for row in range(rows):
        for col in range(cols):
            left = max(0, int(round(x0 + col * cell_size)) + inset)
            top = max(0, int(round(y0 + row * cell_size)) + inset)
            right = min(candidate.shape[1], int(round(x0 + (col + 1) * cell_size)) - inset)
            bottom = min(candidate.shape[0], int(round(y0 + (row + 1) * cell_size)) - inset)
            if right <= left or bottom <= top:
                continue
            distances.append(float(visual_delta(candidate[top:bottom, left:right], expected[top:bottom, left:right]).mean()) / 255.0)
    return max(distances, default=1.0)


def _verify_cumulative_state_video(
    task: str,
    structured: Dict[str, Any],
    plan: Mapping[str, Any],
    frames: Sequence[np.ndarray],
    frame_tracks: Mapping[str, Sequence[Optional[Coord]]],
) -> tuple[Dict[str, Any], bool]:
    """Match every committed candidate state to a source-rendered state.

    This prevents a video from keeping dynamic state stale and painting the
    correct terminal state only in its last frame.  Matching is performed on
    every grid-cell interior, so a small but semantic plant/body mutation is
    not diluted by the 512x512 background.
    """

    primary_cells = list(frame_tracks.get("primary") or [])
    if not frames or len(primary_cells) != len(frames):
        return {"video_cumulative_state_sequence_pass": False}, False
    change_frames: List[int] = []
    previous: Optional[Coord] = _source_starts(task, structured).get("primary")
    for frame_index, cell in enumerate(primary_cells):
        if cell is not None and cell != previous:
            change_frames.append(frame_index)
            previous = cell

    expected_images: List[np.ndarray] = []
    rows = cols = 0
    cell_size = x0 = y0 = 0.0
    if task == "plant_watering_route":
        state_trace = [deepcopy(state) for state in plan.get("state_trace") or [] if isinstance(state, Mapping)]
        if not state_trace:
            return {"video_cumulative_state_sequence_pass": False}, False
        generator = _instantiate_generator(task, structured)
        geometry = generator._panel_geometry()
        panel_w, panel_h = int(geometry["panel_w"]), int(geometry["panel_h"])
        panel_x0, panel_y0 = int(geometry["left_panel_x0"]), int(geometry["left_panel_y0"])
        layout = generator._layout(panel_w, panel_h)
        rows = int(structured.get("rows") or 4)
        cols = int(structured.get("cols") or 4)
        cell_size = float(layout["cell"])
        x0 = float(layout["board_x0"])
        y0 = float(layout["board_y0"])
        # Panels leave the renderer at the scene supersample factor (the scene
        # compositor owns the single LANCZOS pass), so bring each source-rendered
        # reference panel down to target scale before comparing it with the
        # panel region of a finished candidate frame.
        expected_images = [
            np.asarray(
                finish_scaled_canvas(
                    generator._render_panel(state, panel_w, panel_h),
                    (panel_w, panel_h),
                ).convert("RGB"),
                dtype=np.uint8,
            )
            for state in state_trace
        ]
        candidate_views = [
            frame[panel_y0 : panel_y0 + panel_h, panel_x0 : panel_x0 + panel_w]
            for frame in frames
        ]
    else:
        return {"video_cumulative_state_sequence_pass": True}, True

    expected_transition_count = max(0, len(expected_images) - 1)
    event_count_ok = len(change_frames) == expected_transition_count
    state_scores: List[float] = []
    if event_count_ok:
        boundaries = [0] + change_frames + [len(frames)]
        for state_index, expected in enumerate(expected_images):
            start = boundaries[state_index]
            end = boundaries[state_index + 1]
            candidates = candidate_views[start:end]
            score = min(
                (
                    _cellwise_state_distance(
                        candidate,
                        expected,
                        rows=rows,
                        cols=cols,
                        cell_size=cell_size,
                        x0=x0,
                        y0=y0,
                    )
                    for candidate in candidates
                ),
                default=1.0,
            )
            state_scores.append(score)
    # Cell-local semantic states are source-rendered and compared at a static
    # hold frame.  A looser whole-canvas threshold lets several prematurely
    # green pots disappear into the background average; 0.05 still allows
    # encoded/rendering noise while rejecting that terminal-paint shortcut.
    passed = bool(event_count_ok and state_scores and max(state_scores) <= 0.05)
    return {
        "video_cumulative_expected_state_count": len(expected_images),
        "video_cumulative_observed_transition_count": len(change_frames),
        "video_cumulative_state_cell_distances": state_scores,
        "video_cumulative_state_max_cell_distance": max(state_scores, default=1.0),
        "video_cumulative_state_sequence_pass": passed,
        # Panel-local source states, tagged with their origin in the frame, for
        # the static-preservation gate.  Popped before metrics are serialized.
        "_video_source_process_states": [
            ((panel_y0, panel_x0), image) for image in expected_images
        ],
    }, passed


def _expected_process_state_images(
    task: str,
    structured: Dict[str, Any],
    plan: Dict[str, Any],
    *,
    show_path: bool = True,
) -> List[np.ndarray]:
    """Render every candidate-replayed semantic state from source primitives."""

    if task in {
        "agent_courier_relay",
        "bus_route_passengers",
        "color_matched_collection",
        "cooperative_open_door",
        "falling_block_dodge",
        "parcel_delivery_route",
        "patrol_avoid_pattern",
        "tide_water_cycle",
    }:
        _generator, _scenario, _states, images = _rendered_multi_process_states(
            task,
            structured,
            plan,
        )
        return images

    if task == "grid_obtaining_award":
        generator, scenario = _basic_scenario(
            task,
            structured,
            plan,
        )
        reward_lookup = {
            reward.cell: index for index, reward in enumerate(scenario.rewards)
        }
        collected: set[int] = set()
        images: List[np.ndarray] = []
        for step_index, cell in enumerate(scenario.path_cells):
            reward_index = reward_lookup.get(cell)
            if reward_index is not None:
                collected.add(reward_index)
            image = np.asarray(
                generator._render_board(
                    scenario,
                    agent_cell=cell,
                    collected_indices=collected,
                    show_path=show_path,
                    path_cells=scenario.path_cells[: step_index + 1] if show_path else (),
                ).convert("RGB"),
                dtype=np.uint8,
            )
            if images and _image_distance(images[-1], image) <= 1e-6:
                continue
            images.append(image)
        return images

    generator, scenario = _mechanic_scenario(task, structured, plan)
    path = list(scenario.agent_path)
    if not path:
        return []

    images: List[np.ndarray] = []
    if task == "bridge_drop_tile":
        water = {tuple(cell) for cell in scenario.extras.get("water") or []}
        dropped: List[Coord] = []
        for cell in path:
            if cell in water and cell not in dropped:
                dropped.append(cell)
            images.append(
                np.asarray(
                    generator._render(
                        scenario,
                        agent_center=generator._cell_center(cell),
                        dropped_so_far=list(dropped),
                    ).convert("RGB"),
                    dtype=np.uint8,
                )
            )
        return images

    if task == "lantern_oil_route":
        capacity = int(scenario.extras.get("max_oil") or 0)
        refills = {tuple(cell) for cell in scenario.extras.get("refills") or []}
        oil = capacity
        previous = path[0]
        for index, cell in enumerate(path):
            if index:
                oil -= 1
                if cell != previous and cell in refills:
                    oil = capacity
            images.append(
                np.asarray(
                    generator._render(
                        scenario,
                        agent_cell=cell,
                        agent_center=generator._cell_center(cell),
                        oil=oil,
                    ).convert("RGB"),
                    dtype=np.uint8,
                )
            )
            previous = cell
        return images

    if task == "carry_capacity_2":
        cargo = [tuple(cell) for cell in scenario.extras.get("cargo_cells") or []]
        zone = tuple(scenario.extras.get("zone_cell") or ())
        present = [True] * len(cargo)
        held = 0
        delivered = 0
        delivery_commits = 0
        for cell in path:
            if cell in cargo:
                cargo_index = cargo.index(cell)
                if present[cargo_index] and held < 2:
                    present[cargo_index] = False
                    held += 1
            if cell == zone and held:
                delivered += held
                held = 0
                delivery_commits += 1
            image = np.asarray(
                generator._render(
                    scenario,
                    agent_center=generator._cell_center(cell),
                    held=held,
                    delivered=delivered,
                    delivery_commits=delivery_commits,
                    cargo_present=list(present),
                ).convert("RGB"),
                dtype=np.uint8,
            )
            if images and _image_distance(images[-1], image) <= 1e-6:
                continue
            images.append(image)
        return images

    if task == "lever_unlock_sequence":
        levers = [
            tuple(cell) for cell in scenario.extras.get("lever_cells") or []
        ]
        pulled_count = 0
        for cell in path:
            if pulled_count < len(levers) and cell == levers[pulled_count]:
                pulled_count += 1
            image = np.asarray(
                generator._render(
                    scenario,
                    agent_center=generator._cell_center(cell),
                    levers_pulled=[
                        index < pulled_count for index in range(len(levers))
                    ],
                    door_open=pulled_count == len(levers),
                ).convert("RGB"),
                dtype=np.uint8,
            )
            if images and _image_distance(images[-1], image) <= 1e-6:
                continue
            images.append(image)
        return images

    if task == "switch_toggle_walls":
        switch = tuple(scenario.extras.get("switch_cell") or ())
        active = False
        for cell in path:
            active = bool(active or cell == switch)
            images.append(
                np.asarray(
                    generator._render(
                        scenario,
                        agent_center=generator._cell_center(cell),
                        switch_active=active,
                    ).convert("RGB"),
                    dtype=np.uint8,
                )
            )
        return images

    return []


def _rendered_multi_process_states(
    task: str,
    structured: Dict[str, Any],
    plan: Dict[str, Any],
) -> tuple[Any, Any, List[Any], List[np.ndarray]]:
    generator, scenario = _multi_scenario(task, structured, plan)
    states = generator._dedupe_discrete_states(
        generator.build_video_frame_states(scenario, 1)
    )
    kept_states: List[Any] = []
    images: List[np.ndarray] = []
    for state in states:
        image = np.asarray(
            generator.make_image(scenario, frame_state=state).convert("RGB"),
            dtype=np.uint8,
        )
        if images and _image_distance(images[-1], image) <= 1e-6:
            continue
        kept_states.append(state)
        images.append(image)
    return generator, scenario, kept_states, images


def _variation_mask(
    templates: Mapping[int, np.ndarray],
    *,
    allowed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a compact mask covering pixels that encode a discrete value."""

    images = list(templates.values())
    if len(images) < 2 or any(image.shape != images[0].shape for image in images):
        return np.zeros(images[0].shape[:2], dtype=bool) if images else np.zeros((0, 0), dtype=bool)
    stack = np.stack(images).astype(np.int16)
    variation = np.max(np.max(stack, axis=0) - np.min(stack, axis=0), axis=2) > 8
    variation = cv2.dilate(variation.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    if allowed is not None:
        variation &= allowed.astype(bool)
    return variation


def _masked_image_distance(
    candidate: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
) -> float:
    if (
        candidate.shape != expected.shape
        or mask.shape != candidate.shape[:2]
        or not bool(mask.any())
    ):
        return 1.0
    return float(
        np.mean(
            np.abs(
                candidate.astype(np.float32)[mask]
                - expected.astype(np.float32)[mask]
            )
        )
        / 255.0
    )


def _template_distance_limit(
    templates: Mapping[int, np.ndarray],
    mask: np.ndarray,
) -> float:
    labels = sorted(templates)
    pairwise = [
        _masked_image_distance(templates[first], templates[second], mask)
        for index, first in enumerate(labels)
        for second in labels[index + 1 :]
    ]
    minimum_separation = min(pairwise, default=0.12)
    return min(0.12, max(0.04, minimum_separation * 0.45))


def _classify_template_values(
    frames: Sequence[np.ndarray],
    templates: Mapping[int, np.ndarray],
    mask: np.ndarray,
) -> tuple[List[int], List[float], float]:
    labels = sorted(templates)
    if not labels or not bool(mask.any()):
        return [], [], 0.0
    values: List[int] = []
    distances: List[float] = []
    for frame in frames:
        ranked = sorted(
            (
                _masked_image_distance(frame, templates[label], mask),
                label,
            )
            for label in labels
        )
        distance, label = ranked[0]
        values.append(int(label))
        distances.append(float(distance))
    return values, distances, _template_distance_limit(templates, mask)


def _lantern_hud_templates(
    structured: Dict[str, Any],
) -> tuple[Mapping[int, np.ndarray], np.ndarray]:
    """Render source-only fuel HUD templates below the grid."""
    start = tuple(_to_pair(structured.get("start")) or (-1, -1))
    generator, scenario = _mechanic_scenario(
        "lantern_oil_route", structured, {"primary_path": [start]},
    )
    capacity = int(scenario.extras.get("max_oil") or 0)
    templates = {
        oil: np.asarray(generator._render(
            scenario, agent_cell=start, agent_center=generator._cell_center(start), oil=oil,
        ).convert("RGB"), dtype=np.uint8)
        for oil in range(capacity + 1)
    }
    sample = next(iter(templates.values()))
    size, cell_size, _x0, y0 = _grid_geometry("lantern_oil_route", structured)
    grid_bottom = min(sample.shape[0], int(round(y0 + size * cell_size)))
    allowed = np.zeros(sample.shape[:2], dtype=bool)
    allowed[grid_bottom:, sample.shape[1] // 2 :] = True
    return templates, _variation_mask(templates, allowed=allowed)


def _bridge_remaining_values(
    scenario: Any,
) -> List[int]:
    inventory = int(scenario.extras.get("inventory") or 0)
    water = {tuple(cell) for cell in scenario.extras.get("water") or []}
    dropped: set[Coord] = set()
    remaining: List[int] = []
    for cell in scenario.agent_path:
        if cell in water:
            dropped.add(cell)
        remaining.append(inventory - len(dropped))
    return remaining


def _lantern_oil_values(
    scenario: Any,
) -> List[int]:
    path = list(scenario.agent_path)
    if not path:
        return []
    capacity = int(scenario.extras.get("max_oil") or 0)
    refills = {tuple(cell) for cell in scenario.extras.get("refills") or []}
    oil = capacity
    values = [oil]
    previous = path[0]
    for cell in path[1:]:
        oil -= 1
        if oil <= 0:
            return []
        if cell != previous and cell in refills:
            oil = capacity
        values.append(oil)
        previous = cell
    return values


def _verify_mechanic_hud_process(
    task: str,
    structured: Dict[str, Any],
    plan: Dict[str, Any],
    frames: Sequence[np.ndarray],
    aligned_state_indices: Sequence[int],
) -> tuple[Dict[str, Any], bool]:
    generator, scenario = _mechanic_scenario(task, structured, plan)
    expected_values = (
        _bridge_remaining_values(scenario)
        if task == "bridge_drop_tile"
        else _lantern_oil_values(scenario)
    )
    aligned_expected = [
        expected_values[index]
        for index in aligned_state_indices
        if 0 <= index < len(expected_values)
    ]
    pip_readings = []
    pip_monotonic = True
    previous_pips = None
    if task == "bridge_drop_tile":
        inventory = int(scenario.extras["inventory"])
        boxes = generator._gauge_pip_boxes(inventory)
        observed_values = []
        for frame in frames:
            pips = []
            visible_pips = True
            for left, top, right, bottom in boxes:
                inset = max(2, int((right - left) * .25))
                crop = frame[round(top)+inset:round(bottom)-inset,
                             round(left)+inset:round(right)-inset]
                color = np.median(crop.reshape(-1, 3), axis=0)
                full_error = float(np.linalg.norm(color - np.asarray((172, 130, 78))))
                empty_error = float(np.linalg.norm(color - 255))
                pips.append(full_error < empty_error)
                glyph = frame[max(0, round(top)-1):round(bottom)+2,
                              max(0, round(left)-1):round(right)+2].astype(float)
                outline = np.max(255 - glyph, axis=2) >= 20
                visible_pips &= min(full_error, empty_error) <= 75 and int(outline.sum()) >= 6
            count = sum(pips)
            if previous_pips is not None:
                pip_monotonic &= not any(current and not before for before, current in zip(previous_pips, pips))
            previous_pips = pips
            observed_values.append(count if visible_pips else -1)
            pip_readings.append({"filled": pips, "visible": bool(visible_pips)})
    else:
        templates, mask = _lantern_hud_templates(structured)
        observed_values, distances, distance_limit = _classify_template_values(frames, templates, mask)
    lengths_ok = bool(
        expected_values
        and len(aligned_expected) == len(frames)
        and len(observed_values) == len(frames)
    )
    exact = bool(lengths_ok and observed_values == aligned_expected)
    visible = bool(
        lengths_ok
        and (all(value >= 0 for value in observed_values) if task == "bridge_drop_tile"
             else distances and max(distances) <= distance_limit)
    )
    passed = bool(exact and visible and pip_monotonic)
    prefix = "bridge_inventory" if task == "bridge_drop_tile" else "lantern_fuel"
    return {
        **({"video_bridge_inventory_pip_readings": pip_readings,
            "video_bridge_inventory_pips_monotonic_pass": bool(pip_monotonic)}
           if task == "bridge_drop_tile" else {}),
        f"video_{prefix}_expected_values": aligned_expected,
        f"video_{prefix}_observed_values": observed_values,
        **({"video_lantern_fuel_template_distances": distances,
            "video_lantern_fuel_template_distance_limit": distance_limit}
           if task == "lantern_oil_route" else {}),
        f"video_{prefix}_exact_sequence_pass": exact,
        f"video_{prefix}_visibility_pass": visible,
        f"video_{prefix}_process_pass": passed,
    }, passed


def _bus_progress_templates(
    generator: Any,
    scenario: Any,
    cell: Coord,
) -> tuple[Mapping[int, np.ndarray], np.ndarray]:
    from data.spatial_navigation.core.multi_agent_base import FrameState

    templates: Dict[int, np.ndarray] = {}
    for progress in range(4):
        templates[progress] = np.asarray(
            generator.make_image(
                scenario,
                frame_state=FrameState(
                    primary_pos=cell,
                    primary_cell=cell,
                    extras={
                        "picked_count": progress,
                        "onboard_count": progress,
                        "delivered_count": 0,
                    },
                ),
            ).convert("RGB"),
            dtype=np.uint8,
        )
    templates[4] = np.asarray(
        generator.make_image(
            scenario,
            frame_state=FrameState(
                primary_pos=cell,
                primary_cell=cell,
                extras={
                    "picked_count": 3,
                    "onboard_count": 0,
                    "delivered_count": 3,
                },
            ),
        ).convert("RGB"),
        dtype=np.uint8,
    )
    return templates, _variation_mask(templates)


def _verify_bus_progress_process(
    structured: Dict[str, Any],
    plan: Dict[str, Any],
    frames: Sequence[np.ndarray],
    aligned_state_indices: Sequence[int],
) -> tuple[Dict[str, Any], bool]:
    generator, scenario, states, _images = _rendered_multi_process_states(
        "bus_route_passengers",
        structured,
        plan,
    )
    expected_progress = [
        4
        if int(state.extras.get("delivered_count", 0)) == 3
        else int(state.extras.get("picked_count", 0))
        for state in states
    ]
    cache: Dict[Coord, tuple[Mapping[int, np.ndarray], np.ndarray]] = {}
    observed: List[int] = []
    expected: List[int] = []
    distances: List[float] = []
    limits: List[float] = []
    valid = bool(len(frames) == len(aligned_state_indices))
    for frame, state_index in zip(frames, aligned_state_indices):
        if not (0 <= state_index < len(states)):
            valid = False
            continue
        state = states[state_index]
        cell = tuple(state.primary_cell)
        if cell not in cache:
            cache[cell] = _bus_progress_templates(generator, scenario, cell)
        templates, mask = cache[cell]
        values, frame_distances, distance_limit = _classify_template_values(
            [frame],
            templates,
            mask,
        )
        if not values or not frame_distances:
            valid = False
            continue
        observed.append(values[0])
        expected.append(expected_progress[state_index])
        distances.append(frame_distances[0])
        limits.append(distance_limit)
    exact = bool(valid and len(observed) == len(frames) and observed == expected)
    visible = bool(
        exact
        and distances
        and all(distance <= limit for distance, limit in zip(distances, limits))
    )
    passed = bool(exact and visible)
    return {
        "video_bus_expected_progress": expected,
        "video_bus_observed_progress": observed,
        "video_bus_progress_template_distances": distances,
        "video_bus_progress_template_distance_limits": limits,
        "video_bus_progress_exact_sequence_pass": exact,
        "video_bus_progress_visibility_pass": visible,
        "video_bus_passenger_state_process_pass": passed,
    }, passed


def _verify_task_owned_process_details(
    task: str,
    structured: Dict[str, Any],
    plan: Dict[str, Any],
    frames: Sequence[np.ndarray],
    aligned_state_indices: Sequence[int],
) -> tuple[Dict[str, Any], bool]:
    if task in {"bridge_drop_tile", "lantern_oil_route"}:
        return _verify_mechanic_hud_process(
            task,
            structured,
            plan,
            frames,
            aligned_state_indices,
        )
    if task == "bus_route_passengers":
        return _verify_bus_progress_process(
            structured,
            plan,
            frames,
            aligned_state_indices,
        )
    if task == "agent_courier_relay":
        generator, scenario, states, expected = _rendered_multi_process_states(task, structured, plan)
        meet = tuple(scenario.extras["b_meet"])
        coverage = []
        for frame, state_index in zip(frames, aligned_state_indices):
            state = states[state_index]
            cx, cy = generator.cell_center(tuple(state.primary_cell))
            if tuple(state.primary_cell) == meet:
                cx += generator.cell_size * (-.22 if state.extras.get("phase") == "leg1" else .22)
                cy -= generator.cell_size * .26
            else:
                cx += generator.cell_size * .18
                cy -= generator.cell_size * .18
            radius = generator.cell_size * .20
            yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
            local = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
            brown = np.asarray((165, 110, 50))
            support = local & (np.linalg.norm(expected[state_index].astype(np.float32) - brown, axis=2) <= 48)
            candidate_mask = np.linalg.norm(frame.astype(np.float32) - brown, axis=2) <= 48
            nearby = cv2.dilate(candidate_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            coverage.append(float(nearby[support].mean()) if support.any() else 0.0)
        state_coverage = [max((score for score, label in zip(coverage, aligned_state_indices) if label == index), default=0.0)
                          for index in range(len(expected))]
        visible = bool(state_coverage and min(state_coverage) >= .55 and coverage[-1] >= .55)
        return {"video_courier_parcel_footprint_coverage": coverage,
                "video_courier_parcel_state_coverage": state_coverage,
                "video_courier_parcel_carrier_sequence_pass": visible}, visible
    return {"video_task_owned_process_detail_pass": True}, True


def _verify_reward_pickup_video(
    structured: Dict[str, Any],
    plan: Dict[str, Any],
    frames: Sequence[np.ndarray],
    aligned_state_indices: Sequence[int],
    expected: Sequence[np.ndarray],
    *,
    show_path: bool,
) -> tuple[Dict[str, Any], bool]:
    """Classify each reward on its own pixels throughout the candidate video."""

    generator, scenario = _basic_scenario("grid_obtaining_award", structured, plan)
    collected: set[int] = set()
    templates: List[List[tuple[np.ndarray, np.ndarray]]] = []
    for index, cell in enumerate(scenario.path_cells):
        collected.update(i for i, reward in enumerate(scenario.rewards) if reward.cell == cell)
        checks = []
        for reward_index in range(len(scenario.rewards)):
            altered = set(collected)
            altered.symmetric_difference_update({reward_index})
            alternative = np.asarray(generator._render_board(
                scenario, agent_cell=cell, collected_indices=altered,
                show_path=show_path,
                path_cells=scenario.path_cells[:index + 1] if show_path else (),
            ))
            mask = np.max(np.abs(expected[index].astype(np.int16) - alternative.astype(np.int16)), axis=2) > 8
            checks.append((alternative, mask))
        templates.append(checks)
    failures = 0
    for frame, state_index in zip(frames, aligned_state_indices):
        for alternative, mask in templates[state_index]:
            if bool(mask.any()) and _masked_image_distance(
                frame, expected[state_index], mask
            ) >= _masked_image_distance(frame, alternative, mask):
                failures += 1
    return {
        "video_reward_state_violation_count": failures,
        "video_reward_pickup_state_pass": failures == 0,
    }, failures == 0


def _transition_region_metrics(
    candidate: np.ndarray,
    expected: np.ndarray,
    previous_expected: np.ndarray,
) -> tuple[float, float]:
    expected_delta = np.max(
        np.abs(expected.astype(np.int16) - previous_expected.astype(np.int16)),
        axis=2,
    )
    required = expected_delta > 12
    required_count = int(required.sum())
    if required_count == 0:
        return 0.0, 1.0
    distance = float(np.mean(visual_delta(candidate, expected)[required]) / 255.0)
    candidate_delta = np.max(
        np.abs(candidate.astype(np.int16) - previous_expected.astype(np.int16)),
        axis=2,
    )
    recall = float((candidate_delta[required] > 12).sum() / required_count)
    return distance, recall


def _monotonic_state_alignment(
    distance_matrix: Sequence[Sequence[float]],
) -> List[int]:
    """Align held candidate frames to every expected state in order."""

    frame_count = len(distance_matrix)
    state_count = len(distance_matrix[0]) if distance_matrix else 0
    if (
        frame_count == 0
        or state_count == 0
        or frame_count < state_count
        or any(len(row) != state_count for row in distance_matrix)
    ):
        return []

    infinity = float("inf")
    previous = [infinity] * state_count
    previous[0] = float(distance_matrix[0][0])
    backtrack: List[List[int]] = [[-1] * state_count for _ in range(frame_count)]
    for frame_index in range(1, frame_count):
        current = [infinity] * state_count
        minimum_state = max(0, state_count - (frame_count - frame_index))
        maximum_state = min(state_count - 1, frame_index)
        for state_index in range(minimum_state, maximum_state + 1):
            stay_cost = previous[state_index]
            advance_cost = (
                previous[state_index - 1] if state_index > 0 else infinity
            )
            if advance_cost < stay_cost:
                current[state_index] = (
                    advance_cost + float(distance_matrix[frame_index][state_index])
                )
                backtrack[frame_index][state_index] = state_index - 1
            elif stay_cost < infinity:
                current[state_index] = (
                    stay_cost + float(distance_matrix[frame_index][state_index])
                )
                backtrack[frame_index][state_index] = state_index
        previous = current

    if previous[-1] == infinity:
        return []
    aligned = [0] * frame_count
    state_index = state_count - 1
    for frame_index in range(frame_count - 1, -1, -1):
        aligned[frame_index] = state_index
        if frame_index:
            state_index = backtrack[frame_index][state_index]
            if state_index < 0:
                return []
    return aligned


def _verify_replayed_process_state_video(
    task: str,
    structured: Dict[str, Any],
    plan: Dict[str, Any],
    frames: Sequence[np.ndarray],
) -> tuple[Dict[str, Any], bool]:
    """Require every source-replayed dynamic state, not only the last frame."""

    expected = _expected_process_state_images(task, structured, plan)
    if not expected or not frames:
        return {
            "video_replayed_process_state_pass": False,
            "video_replayed_process_state_error": "missing candidate or expected states",
        }, False
    show_path = True
    if task == "grid_obtaining_award":
        # The task requires agent motion and reward pickups, but does not
        # require the reference renderer's route decoration. Choose one
        # rendering style from the submitted endpoint's trace pixels; both
        # styles replay the same candidate route and resource transitions.
        plain = _expected_process_state_images(task, structured, plan, show_path=False)
        decoration = np.max(
            np.abs(expected[-1].astype(np.int16) - plain[-1].astype(np.int16)),
            axis=2,
        ) > 8
        if bool(decoration.any()) and _masked_image_distance(
            frames[-1], plain[-1], decoration
        ) < _masked_image_distance(frames[-1], expected[-1], decoration):
            expected = plain
            show_path = False
    size, cell_size, x0, y0 = _grid_geometry(task, structured)
    distance_matrix = [
        [
            _cellwise_state_distance(
                frame,
                state,
                rows=size,
                cols=size,
                cell_size=cell_size,
                x0=x0,
                y0=y0,
            )
            for state in expected
        ]
        for frame in frames
    ]
    best_indices = _monotonic_state_alignment(distance_matrix)
    if not best_indices:
        return {
            "video_replayed_expected_state_count": len(expected),
            "video_replayed_process_state_pass": False,
            "video_replayed_process_state_error": "candidate frames cannot cover expected states in order",
        }, False
    best_distances = [
        distances[index]
        for distances, index in zip(distance_matrix, best_indices)
    ]
    monotonic = all(
        current >= previous
        for previous, current in zip(best_indices, best_indices[1:])
    )
    observed = set(best_indices)
    coverage = observed == set(range(len(expected)))
    endpoints = bool(best_indices and best_indices[0] == 0 and best_indices[-1] == len(expected) - 1)
    classified = bool(best_distances and max(best_distances) <= 0.085)

    # Count source-state changes, not changing RGB pixels: generated holds and
    # VP9 frames can differ in stroke edges without executing another action.
    # This independent nearest-state reading still exposes an extra A-B-A-B
    # oscillation even when monotonic alignment absorbs recurring world states.
    # State distances above observe the board, not bottom-margin inventory
    # glyphs. States differing only in a HUD have the same board observation;
    # their field changes are enforced by the task-owned HUD verifier below.
    top, left = int(y0), int(x0)
    bottom, right = int(y0 + size * cell_size), int(x0 + size * cell_size)
    board_views = [state[top:bottom, left:right] for state in expected]
    unique_states = []
    representative = []
    for state_index in range(len(expected)):
        equivalent = next((index for index in unique_states if np.array_equal(board_views[index], board_views[state_index])), None)
        if equivalent is None:
            unique_states.append(state_index)
            equivalent = state_index
        representative.append(equivalent)
    observed_labels = []
    for distances in distance_matrix:
        ranked = sorted((distances[index], index) for index in unique_states)
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < .010:
            continue
        label = ranked[0][1]
        if not observed_labels or observed_labels[-1] != label:
            observed_labels.append(label)
    expected_switches = sum(left != right for left, right in zip(representative, representative[1:]))
    observed_switches = max(0, len(observed_labels) - 1)
    switch_count_ok = observed_switches == expected_switches

    transition_metrics: List[Dict[str, Any]] = []
    transitions_ok = True
    for state_index in range(1, len(expected)):
        frame_index = next(
            (
                index
                for index, classified_state in enumerate(best_indices)
                if classified_state == state_index
            ),
            None,
        )
        if frame_index is None:
            transition_metrics.append({
                "state_index": state_index,
                "first_frame": None,
                "pass": False,
            })
            transitions_ok = False
            continue
        distance, recall = _transition_region_metrics(
            frames[frame_index],
            expected[state_index],
            expected[state_index - 1],
        )
        transition_pass = bool(distance <= 0.12 and recall >= 0.55)
        transition_metrics.append({
            "state_index": state_index,
            "first_frame": frame_index,
            "required_region_distance": distance,
            "required_change_recall": recall,
            "pass": transition_pass,
        })
        transitions_ok = bool(transitions_ok and transition_pass)

    detail_metrics, details_ok = _verify_task_owned_process_details(
        task,
        structured,
        plan,
        frames,
        best_indices,
    )
    if task == "grid_obtaining_award":
        reward_metrics, rewards_ok = _verify_reward_pickup_video(
            structured, plan, frames, best_indices, expected, show_path=show_path,
        )
        detail_metrics.update(reward_metrics)
        details_ok = bool(details_ok and rewards_ok)
    passed = bool(
        classified
        and monotonic
        and coverage
        and endpoints
        and switch_count_ok
        and transitions_ok
        and details_ok
    )
    return {
        "video_replayed_expected_state_count": len(expected),
        "video_replayed_observed_state_indices": best_indices,
        "video_replayed_max_cell_distance": max(best_distances, default=1.0),
        "video_replayed_state_classification_pass": classified,
        "video_replayed_state_order_pass": monotonic,
        "video_replayed_state_coverage_pass": coverage,
        "video_replayed_state_endpoints_pass": endpoints,
        "video_replayed_expected_switch_count": expected_switches,
        "video_replayed_observed_switch_count": observed_switches,
        "video_replayed_switch_count_pass": switch_count_ok,
        "video_replayed_transition_metrics": transition_metrics,
        **detail_metrics,
        "video_replayed_process_state_pass": passed,
        "video_replayed_process_state_source": "semantic_input_and_candidate_replay",
        "video_replayed_process_used_reference_media": False,
        # Consumed by the static-preservation gate before metrics are
        # serialized: these are the states the source replay declares this
        # video passes through, so everything they never touch is static.
        "_video_source_process_states": [((0, 0), image) for image in expected],
    }, passed


def _synthetic_answer_from_tracks(
    sample: CanonicalSample, structured: Dict[str, Any], tracks: Dict[str, List[Coord]]
) -> Optional[str]:
    task = sample.task_name
    primary = tracks.get("primary") or []
    secondary = tracks.get("secondary") or []
    if not primary:
        return None
    if task == "agent_courier_relay":
        if not secondary:
            return None
        return format_named_xy_paths_answer({"A": primary, "B": secondary})
    if task in _DUAL_AGENT_TASKS:
        if not secondary:
            return None
        return format_named_xy_paths_answer(
            {"blue": primary, "orange": secondary}
        )
    if task in _SINGLE_XY_ROUTE_ANSWER_TASKS:
        return format_xy_path_answer(primary)
    return None


def _insert_stays(path: Sequence[Coord], positions: Sequence[int]) -> List[Coord]:
    result = list(path)
    offset = 0
    for position in positions:
        result.insert(int(position) + 1 + offset, result[int(position) + offset])
        offset += 1
    return result


def _path_replay_checker(
    task: str,
    structured: Dict[str, Any],
) -> Optional[Any]:
    if task in _MECHANIC_TASKS:
        return lambda value: _source_replay._replay_agent_mechanic_path(
            structured,
            task_name=task,
            path=value,
        )
    if task in _PRIMARY_TASKS:
        model = _source_replay._build_primary_mechanic_model(
            structured,
            task_name=task,
        )
        if model is not None:
            return lambda value: _source_replay._replay_transition_model_path(
                model,
                value,
            )
    return None


def _legal_stay_repairs(
    task: str,
    structured: Dict[str, Any],
    path: Sequence[Coord],
    *,
    missing: int,
) -> List[List[Coord]]:
    checker = _path_replay_checker(task, structured)
    candidate = list(path)
    if checker is None or missing < 0:
        return []
    if missing == 0:
        _action_count, legal = checker(candidate)
        return [candidate] if legal else []
    builder = (
        _source_replay._build_agent_mechanic_model if task in _MECHANIC_TASKS
        else _source_replay._build_primary_mechanic_model
    )
    model = builder(structured, task_name=task)
    if model is None or not _source_replay._path_possible_with_hidden_stays(model, candidate):
        return []
    repairs: List[List[Coord]] = []
    for positions in combinations_with_replacement(range(len(candidate)), missing):
        repaired = _insert_stays(candidate, positions)
        _action_count, legal = checker(repaired)
        if legal:
            repairs.append(repaired)
    return repairs


def _repair_observed_path(
    task: str, structured: Dict[str, Any], path: Sequence[Coord]
) -> List[Coord]:
    """Recover source-required stationary commits hidden by token tracking."""

    candidate = list(path)
    if len(candidate) < 2:
        return candidate
    checker = _path_replay_checker(task, structured)
    if checker is None:
        return candidate
    _action_count, legal = checker(candidate)
    if legal:
        return candidate
    for missing in range(1, 6):
        repairs = _legal_stay_repairs(
            task,
            structured,
            candidate,
            missing=missing,
        )
        if repairs:
            return repairs[0]
    return candidate


# How many stay placements at one count are worth a full source-rendered state
# replay when the count itself had to be searched.  The dwell ordering puts the
# placement the video actually holds first, so this only bounds the cost of an
# artifact whose dwell profile matches nothing.
_STAY_PLACEMENT_TRIALS = 8


def _dwell_runs(cells: Sequence[Optional[Coord]]) -> List[tuple[Coord, int]]:
    """Run-length encode an observed per-frame cell track."""

    runs: List[tuple[Coord, int]] = []
    for cell in cells:
        if cell is None:
            continue
        key = (int(cell[0]), int(cell[1]))
        if runs and runs[-1][0] == key:
            runs[-1] = (key, runs[-1][1] + 1)
        else:
            runs.append((key, 1))
    return runs


def _dwell_mismatch(
    candidate: Sequence[Coord],
    observed_runs: Sequence[tuple[Coord, int]],
) -> float:
    """Disagreement between a candidate's dwell profile and the video's.

    A stay tick leaves the agent token where it is, so it is invisible in the
    cell sequence - but not in the *dwell*: the renderer holds one state per
    tick at a fixed cadence, so the cell the agent waits on occupies one extra
    hold.  Comparing runs against a cadence read from the artifact itself
    (total frames / total ticks) orders the candidate stay placements without
    assuming any particular hold length.  This only orders the search; whether
    a placement is accepted is still decided by the source-rendered state
    sequence below.
    """

    ticks: List[tuple[Coord, int]] = []
    for cell in candidate:
        key = (int(cell[0]), int(cell[1]))
        if ticks and ticks[-1][0] == key:
            ticks[-1] = (key, ticks[-1][1] + 1)
        else:
            ticks.append((key, 1))
    if len(ticks) != len(observed_runs):
        return float("inf")
    if any(left[0] != right[0] for left, right in zip(ticks, observed_runs)):
        return float("inf")
    total_frames = sum(count for _cell, count in observed_runs)
    total_ticks = sum(count for _cell, count in ticks)
    if total_ticks <= 0:
        return float("inf")
    cadence = total_frames / total_ticks
    return sum(
        abs(float(frames) - cadence * float(count))
        for (_cell, count), (_observed, frames) in zip(ticks, observed_runs)
    )


def _recover_clocked_path(
    task: str,
    structured: Dict[str, Any],
    frame_cells: Sequence[Optional[Coord]],
    cadence_metrics: Mapping[str, Any],
    secondary_cells: Sequence[Optional[Coord]],
    frames: Sequence[np.ndarray],
) -> Optional[List[Coord]]:
    """Replay visible clock commits directly, retaining arbitrarily many stays.

    The search advances along the submitted observations, so its bound is the
    actual video length rather than a guessed count of invisible agent waits.
    """

    model = _source_replay._build_primary_mechanic_model(structured, task_name=task)
    if model is None or not frame_cells:
        return None
    initial, successors, is_goal = model
    extras = structured["extras"]
    boundaries = set(cadence_metrics.get("video_discrete_dynamic_boundary_frames") or [])
    clock = 0
    observed = []
    if task == "patrol_avoid_pattern":
        patrol = [tuple(cell) for cell in extras["patrol_loop"]]
        if len(set(patrol)) != len(patrol) or len(secondary_cells) != len(frame_cells):
            return None
    elif task == "falling_block_dodge":
        from data.spatial_navigation.core.multi_agent_base import FrameState

        blocks = extras["falling_blocks"]
        # An initially invisible falling block has no unique clock witness.
        if any(int(block["start_row"]) < 0 for block in blocks):
            return None
        clear_step = max(int(structured["grid_size"]) - int(block["start_row"]) for block in blocks)
        generator, scenario = _multi_scenario(task, structured, {"primary_path": [initial[0]]})
        templates = {
            step: np.asarray(generator.make_image(scenario, frame_state=FrameState(
                primary_pos=(-100, -100), primary_cell=initial[0], extras={"time_step": step},
            )))
            for step in range(clear_step + 1)
        }
        world_values, _distances, _limit = _classify_template_values(
            frames, templates, _variation_mask(templates),
        )
        if len(world_values) != len(frame_cells):
            return None
    for index, cell in enumerate(frame_cells):
        if cell is None:
            return None
        if task == "patrol_avoid_pattern":
            if secondary_cells[index] not in patrol:
                return None
            clock = patrol.index(secondary_cells[index])
        elif task == "falling_block_dodge":
            clock = world_values[index]
        elif index in boundaries:
            clock = 1 - clock
        value = (tuple(cell), clock)
        if not observed or value != observed[-1]:
            observed.append(value)
    if not observed or observed[0] != initial:
        return None
    parents = {(initial, 1): None}
    queue = deque([(initial, 1)])
    while queue:
        state, cursor = queue.popleft()
        if cursor == len(observed) and is_goal(state):
            segments = []
            key = (state, cursor)
            while parents[key] is not None:
                previous, expanded = parents[key]
                segments.append(expanded)
                key = previous
            return [initial[0]] + [cell for segment in reversed(segments) for cell in segment]
        if is_goal(state):
            continue
        for nxt, expanded in successors(state):
            if task == "patrol_avoid_pattern":
                snapshots = [nxt]
            elif task == "tide_water_cycle":
                old_high = int(extras["tide_row1"] if state[1] == 0 else extras["tide_row2"])
                new_high = int(extras["tide_row1"] if nxt[1] == 0 else extras["tide_row2"])
                move_before = state[0][0] == new_high or nxt[0][0] != old_high
                before_cell = nxt[0] if move_before else state[0]
                snapshots = [(before_cell, state[1]), (before_cell, nxt[1]), nxt]
            else:
                snapshots = [(nxt[0], state[1]), nxt]
            visible = []
            previous = state
            for snapshot in snapshots:
                if snapshot != previous:
                    visible.append(snapshot)
                    previous = snapshot
            end = cursor + len(visible)
            key = (nxt, end)
            if visible and observed[cursor:end] == visible and key not in parents:
                parents[key] = ((state, cursor), expanded)
                queue.append(key)
    return None


def _select_dynamic_process_path(
    sample: CanonicalSample,
    structured: Dict[str, Any],
    tracks: Dict[str, List[Coord]],
    frames: Sequence[np.ndarray],
    cadence_metrics: Mapping[str, Any],
    frame_cells: Sequence[Optional[Coord]] = (),
    secondary_frame_cells: Sequence[Optional[Coord]] = (),
) -> Optional[List[Coord]]:
    """Place visually hidden stays by matching source-rendered world states."""

    primary = list(tracks.get("primary") or [])
    if len(primary) < 2:
        return None
    if sample.task_name in {"patrol_avoid_pattern", "tide_water_cycle", "falling_block_dodge"}:
        clocked = _recover_clocked_path(
            sample.task_name, structured, frame_cells, cadence_metrics, secondary_frame_cells, frames,
        )
        if clocked is not None:
            return clocked
    search: List[int]
    # Counts derived from a second visible witness are exact, so their whole
    # legal placement set is scored; a searched count is capped instead.
    trials: Optional[int] = None
    budget = 1
    if sample.task_name == "patrol_avoid_pattern":
        search = [max(0, len(tracks.get("secondary") or []) - 1) - (len(primary) - 1)]
    elif sample.task_name == "tide_water_cycle":
        search = [
            len(cadence_metrics.get("video_discrete_dynamic_boundary_frames") or [])
            - (len(primary) - 1)
        ]
    elif sample.task_name == "falling_block_dodge":
        # Neither witness counts the ticks here: a stay is invisible in the
        # agent track, and the world clock only advances the blocks still on
        # the board, so a tick after the last block has left the board changes
        # nothing but the agent.  Search the stay count instead of deriving it -
        # the smallest legal count first, then one count beyond it for a
        # candidate that spends a redundant stay - and let the source-rendered
        # state sequence decide *where* the stays sit.  Taking the first legal
        # insertion instead, as the generic repair does, silently adopts a
        # different valid solution whenever the stay does not belong at tick 1,
        # and the correct video then fails against the wrong expected states.
        search = list(range(0, 6))
        trials = _STAY_PLACEMENT_TRIALS
        budget = 2
    else:
        return None

    observed_runs = _dwell_runs(frame_cells)
    best_path: Optional[List[Coord]] = None
    best_rank: Optional[tuple[Any, ...]] = None
    for missing in search:
        if missing < 0 or missing > 5:
            continue
        candidates = _legal_stay_repairs(
            sample.task_name,
            structured,
            primary,
            missing=missing,
        )
        if not candidates:
            continue
        if observed_runs:
            # Order by dwell agreement so the common case pays for one state
            # replay instead of one per legal placement: a stay is invisible in
            # the cell sequence but not in how long the video holds that cell.
            candidates.sort(key=lambda value: _dwell_mismatch(value, observed_runs))
        if trials is not None:
            candidates = candidates[:trials]
        budget -= 1
        for candidate in candidates:
            candidate_tracks = dict(tracks)
            candidate_tracks["primary"] = candidate
            synthetic = _synthetic_answer_from_tracks(
                sample,
                structured,
                candidate_tracks,
            )
            if synthetic is None:
                continue
            _metrics, semantic, plan = _evaluate_text(
                sample,
                structured,
                synthetic,
            )
            if not semantic or not plan:
                continue
            process_metrics, process_ok = _verify_replayed_process_state_video(
                sample.task_name,
                structured,
                plan,
                frames,
            )
            transition_metrics = list(
                process_metrics.get("video_replayed_transition_metrics") or []
            )
            rank = (
                not process_ok,
                float(process_metrics.get("video_replayed_max_cell_distance", 1.0)),
                sum(not bool(item.get("pass")) for item in transition_metrics),
                sum(
                    float(item.get("required_region_distance", 1.0))
                    for item in transition_metrics
                ),
            )
            if best_rank is None or rank < best_rank:
                best_path = candidate
                best_rank = rank
            if process_ok:
                return best_path
        if budget <= 0:
            break
    return best_path


def _recover_lantern_process_path(
    structured: Dict[str, Any],
    frames: Sequence[np.ndarray],
    primary_frame_cells: Sequence[Optional[Coord]],
) -> tuple[Optional[List[Coord]], Dict[str, Any], bool]:
    """Recover stationary fuel-spending steps from the visible fuel HUD."""

    templates, mask = _lantern_hud_templates(structured)
    observed_values, distances, distance_limit = _classify_template_values(
        frames,
        templates,
        mask,
    )
    cells = list(primary_frame_cells)
    if (
        not frames
        or len(cells) != len(frames)
        or len(observed_values) != len(frames)
        or any(cell is None for cell in cells)
    ):
        return None, {
            "video_lantern_path_recovery_pass": False,
            "video_lantern_path_recovery_reason": "missing aligned cell or fuel observations",
        }, False

    runs: List[tuple[Coord, int, int]] = []
    for raw_cell, oil in zip(cells, observed_values):
        cell = tuple(raw_cell)  # type: ignore[arg-type]
        if runs and runs[-1][:2] == (cell, oil):
            previous_cell, previous_oil, run_length = runs[-1]
            runs[-1] = previous_cell, previous_oil, run_length + 1
        else:
            runs.append((cell, int(oil), 1))

    extras = structured.get("extras") or {}
    size = int(structured.get("grid_size") or 0)
    start = tuple(_to_pair(structured.get("start")) or (-1, -1))
    walls = {tuple(cell) for cell in _to_path(structured.get("walls")) or []}
    refills = {tuple(cell) for cell in _to_path(extras.get("refills")) or []}
    capacity = int(extras.get("max_oil") or 0)
    visible = bool(
        distances
        and max(distances) <= distance_limit
        and runs
        and min(run_length for _cell, _oil, run_length in runs) >= 2
    )
    legal = bool(
        visible
        and runs[0][0] == start
        and runs[0][1] == capacity
    )
    reason = "ok"
    if not visible:
        reason = "fuel HUD is unreadable or a semantic state is not held"
    elif not legal:
        reason = "initial cell or fuel does not match the supplied source state"

    if legal:
        for (previous_cell, previous_oil, _), (cell, oil, _) in zip(runs, runs[1:]):
            distance = abs(cell[0] - previous_cell[0]) + abs(cell[1] - previous_cell[1])
            expected_oil = previous_oil - 1
            if (
                distance not in {0, 1}
                or not (0 <= cell[0] < size and 0 <= cell[1] < size)
                or cell in walls
                or expected_oil <= 0
            ):
                legal = False
                reason = "candidate makes an illegal move or continues after fuel reaches zero"
                break
            if distance == 1 and cell in refills:
                expected_oil = capacity
            if oil != expected_oil:
                legal = False
                reason = "visible fuel does not match the preceding move or stationary step"
                break

    path = [cell for cell, _oil, _run_length in runs] if legal else None
    return path, {
        "video_lantern_observed_fuel_values": observed_values,
        "video_lantern_fuel_template_distances": distances,
        "video_lantern_fuel_template_distance_limit": distance_limit,
        "video_lantern_semantic_run_cells": [list(cell) for cell, _oil, _length in runs],
        "video_lantern_semantic_run_fuel": [oil for _cell, oil, _length in runs],
        "video_lantern_semantic_run_lengths": [length for _cell, _oil, length in runs],
        "video_lantern_path_recovery_reason": reason,
        "video_lantern_path_recovery_pass": legal,
    }, legal


def _verify_portal_activation_video_state(
    task: str,
    structured: Dict[str, Any],
    plan: Dict[str, Any],
    frames: Sequence[np.ndarray],
    frame_tracks: Mapping[str, Sequence[Optional[Coord]]],
) -> tuple[Dict[str, Any], bool]:
    """Verify discrete source/destination states around every jump.

    Portal activation is a semantic event, not a fade animation.  The
    candidate must show a source snapshot followed by the destination
    snapshot, both rendered from source replay.
    """

    path = [tuple(cell) for cell in plan.get("primary_path") or []]
    expected_jumps = [
        (index, path[index], path[index + 1])
        for index in range(len(path) - 1)
        if abs(path[index + 1][0] - path[index][0])
        + abs(path[index + 1][1] - path[index][1]) != 1
    ]
    observed = list(frame_tracks.get("primary") or [])
    if not expected_jumps or not observed or not frames:
        return {
            "video_portal_activation_event_count": len(expected_jumps),
            "video_portal_discrete_event_pass": False,
            "video_portal_activation_error": "missing replayed jumps or aligned candidate frames",
        }, False

    generator, scenario = _mechanic_scenario(task, structured, plan)
    event_metrics: List[Dict[str, Any]] = []
    cursor = 1

    def render_base(cell: Coord, step_index: int) -> np.ndarray:
        kwargs: Dict[str, Any] = {"agent_center": generator._cell_center(cell)}
        return np.asarray(generator._render(scenario, **kwargs).convert("RGB"), dtype=np.uint8)

    for path_index, source, destination in expected_jumps:
        transition = next(
            (
                index
                for index in range(cursor, min(len(observed), len(frames)))
                if observed[index - 1] == source and observed[index] == destination
            ),
            None,
        )
        if transition is None:
            event_metrics.append(
                {
                    "source": list(source),
                    "destination": list(destination),
                    "aligned_transition_frame": None,
                    "pass": False,
                }
            )
            continue
        source_distance = _image_distance(frames[transition - 1], render_base(source, path_index))
        destination_distance = _image_distance(frames[transition], render_base(destination, path_index + 1))
        event_pass = bool(source_distance <= 0.12 and destination_distance <= 0.12)
        event_metrics.append(
            {
                "source": list(source),
                "destination": list(destination),
                "aligned_transition_frame": transition,
                "source_state_distance": source_distance,
                "destination_state_distance": destination_distance,
                "pass": event_pass,
            }
        )
        cursor = transition + 1

    passed = bool(len(event_metrics) == len(expected_jumps) and all(item["pass"] for item in event_metrics))
    return {
        "video_portal_activation_event_count": len(expected_jumps),
        "video_portal_activation_events": event_metrics,
        "video_portal_activation_state_source": "semantic_input_and_candidate_replayed_path",
        "video_portal_discrete_event_pass": passed,
        "video_portal_fractional_activation_rejected": True,
        "video_portal_activation_verifier_used_reference_media": False,
    }, passed

def _verify_discrete_video_cadence(
    task: str,
    structured: Dict[str, Any],
    frames: Sequence[np.ndarray],
    frame_tracks: Mapping[str, Sequence[Optional[Coord]]],
    frame_centers: Mapping[str, Sequence[Optional[tuple[float, float]]]],
) -> tuple[Dict[str, Any], bool]:
    """Validate exact-cell tokens and readable endpoint holds.

    Spatial Navigation videos are snapshots, not sampled continuous trajectories.  A
    candidate may have a longer terminal hold because of the 4N+1 media
    padding rule. Every intermediate semantic snapshot must be observable;
    neither its duration nor the candidate's cadence is prescribed by GT.
    """

    if not frames:
        return {"video_discrete_cadence_pass": False}, False
    _size, cell_size, x0, y0 = _grid_geometry(task, structured)
    center_checks: Dict[str, Any] = {}
    centers_ok = True
    for name, values in frame_centers.items():
        cells = list(frame_tracks.get(name) or [])
        distances: List[float] = []
        residuals_by_cell: Dict[Coord, List[tuple[float, float]]] = {}
        for index, center in enumerate(values):
            if center is None or index >= len(cells):
                continue
            cell = cells[index]
            if cell is None:
                continue
            expected = (
                x0 + (cell[1] + 0.5) * cell_size,
                y0 + (cell[0] + 0.5) * cell_size,
            )
            residual = (center[0] - expected[0], center[1] - expected[1])
            distances.append(math.hypot(*residual))
            residuals_by_cell.setdefault(tuple(cell), []).append(residual)

        # Most tokens are rendered exactly at the cell centre.  A few
        # legitimate fixed glyphs are intentionally cell-local rather than
        # centred: the gardener's body is offset below its head and the
        # rendezvous terminal uses two side-by-side identity disks.  Measure
        # both the allowed fixed offset and the within-cell jitter.  A true
        # sub-cell interpolation changes the residual within one rounded cell,
        # which remains rejected even when the glyph itself has a fixed offset.
        fixed_offset_limit = max(3.0, 0.16 * cell_size)
        if task == "plant_watering_route":
            fixed_offset_limit = max(fixed_offset_limit, 0.30 * cell_size)
        if task == "agent_courier_relay":
            # The section 10.9 joint-occupancy layout seats both couriers
            # side by side (+-0.22 cell) whenever they legally share the
            # meeting cell, so a bounded fixed offset below any real
            # cell-to-cell interpolation midpoint (0.5 cell) is legitimate.
            fixed_offset_limit = max(fixed_offset_limit, 0.26 * cell_size)

        jitter: List[float] = []
        for residuals in residuals_by_cell.values():
            median_x = float(np.median([item[0] for item in residuals]))
            median_y = float(np.median([item[1] for item in residuals]))
            jitter.extend(
                math.hypot(item[0] - median_x, item[1] - median_y)
                for item in residuals
            )
        maximum = max(distances, default=0.0)
        maximum_jitter = max(jitter, default=0.0)
        # The two rendezvous identities are only offset in the shared terminal
        # star cell.  Other frames still use the ordinary centre rule.
        ordinary_distances: List[float] = []
        if task in {"plant_watering_route", "agent_courier_relay"}:
            # plant: the gardener sprite's visual centroid is intentionally
            # below the cell centre.  courier: the joint-occupancy layout is
            # cell-local but off-centre while the meeting cell is shared.
            ordinary_distances = []
        else:
            ordinary_distances = list(distances)
        ordinary_max = max(ordinary_distances, default=0.0)
        ordinary_limit = max(3.0, 0.16 * cell_size)
        distance_ok = (
            maximum <= fixed_offset_limit
            and ordinary_max <= ordinary_limit
        )
        jitter_limit = max(3.0, 0.06 * cell_size)
        if task in {"lever_unlock_sequence", "tide_water_cycle"}:
            # The disc's deepest interior moves by a few pixels under shading
            # and overlap; the separate whole-cell centre limit still rejects
            # an interpolated position toward a neighbouring cell.
            jitter_limit = max(3.0, .10 * cell_size)
        if task == "agent_courier_relay":
            # The same cell renders a courier centred when alone and offset by
            # 0.22 cell in the shared side-by-side layout; both are exact
            # states, so per-cell residual spread up to that offset is
            # legitimate while true sub-cell interpolation (>=0.5 cell toward
            # a neighbour) stays rejected.
            jitter_limit = max(jitter_limit, 0.26 * cell_size)
        jitter_ok = maximum_jitter <= jitter_limit
        center_checks[name] = {
            "maximum_center_error_pixels": maximum,
            "maximum_within_cell_jitter_pixels": maximum_jitter,
            "fractional_center_pass": bool(distance_ok and jitter_ok),
        }
        centers_ok = bool(centers_ok and center_checks[name]["fractional_center_pass"])

    # Do not infer semantic state boundaries from a full-frame pixel distance.
    # VP9 decode noise makes two identical snapshots differ by a few thousandths
    # while a small dynamic-world commit can be diluted by the unchanged board.
    # Token cell changes are the primary boundary signal.  Dynamic-world tasks
    # add a task-owned palette/evidence threshold for their non-token mutation.
    boundary_indices: set[int] = set()
    token_names = sorted(frame_tracks)
    for index in range(1, len(frames)):
        if any(
            (frame_tracks.get(name) or [None] * len(frames))[index]
            != (frame_tracks.get(name) or [None] * len(frames))[index - 1]
            for name in token_names
        ):
            boundary_indices.add(index)

    dynamic_thresholds = {
        "falling_block_dodge": 0.006,
        "tide_water_cycle": 0.020,
    }
    dynamic_threshold = dynamic_thresholds.get(task)
    dynamic_boundaries: List[int] = []
    if dynamic_threshold is not None:
        reduced = [
            cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            for frame in frames
        ]
        for index, (previous, current) in enumerate(zip(reduced, reduced[1:]), start=1):
            if float(np.mean(np.abs(previous - current)) / 255.0) > dynamic_threshold:
                boundary_indices.add(index)
                dynamic_boundaries.append(index)

    ordered_boundaries = sorted(
        boundary for boundary in boundary_indices if 0 < boundary < len(frames)
    )
    run_lengths: List[int] = []
    cursor = 0
    for boundary in ordered_boundaries + [len(frames)]:
        run_lengths.append(boundary - cursor)
        cursor = boundary
    if not run_lengths:
        run_lengths = [len(frames)]

    # The first hold and terminal padding are allowed to be longer.  Every
    # semantic state must be visible in at least one decoded frame.
    # Infer the candidate's cadence instead of prescribing the renderer's
    # hold length. Equal 3- or 5-frame holds are as legal as 2-frame holds.
    # Identical consecutive clock slots merge into integer multiples of it.
    interior_runs = run_lengths[1:-1] if len(run_lengths) > 2 else []
    candidate_hold_frames = math.gcd(*interior_runs) if interior_runs else None
    state_runs_ok = bool(
        run_lengths
        and run_lengths[0] >= 2
        and run_lengths[-1] >= 2
    )
    cadence_ok = bool(state_runs_ok)
    metrics = {
        "video_discrete_state_run_lengths": run_lengths,
        "video_discrete_candidate_hold_frames": candidate_hold_frames,
        "video_discrete_nonterminal_run_min": min(interior_runs or run_lengths, default=0),
        "video_discrete_nonterminal_run_max": max(interior_runs or run_lengths, default=0),
        "video_discrete_state_boundary_frames": ordered_boundaries,
        "video_discrete_dynamic_boundary_frames": dynamic_boundaries,
        "video_discrete_fractional_center_checks": center_checks,
        "video_discrete_fractional_center_pass": centers_ok,
        "video_discrete_cadence_pass": cadence_ok,
    }
    return metrics, bool(centers_ok and cadence_ok)


# ------------------------------------------------- static preservation ------

# Everything the source replay does not declare changeable is static and has to
# stay so for the whole video.  A frame-wide statistic
# cannot see that: a 24x24 opaque patch is 0.2% of a 512x512 frame, well inside
# the terminal-state tolerance and invisible to a hold distance that sees the
# patch in every frame it compares.  The gate therefore reuses the shared
# per-region statistic (section 11.8) so the patch is measured against the one
# 32x32 region it lands in instead of the whole canvas.

# The declared-dynamic mask is grown by one 3x3 pass before its complement is
# taken, so subpixel rounding at the edge of a legitimate change is never held
# to the static bound.  Same structuring element as the shared ink-cover
# dilation and the maze route's static-marker dilation.
_STATIC_DECLARATION_DILATION = 3

# A source-rendered state declares a pixel changeable once it departs from the
# supplied input by this much on some channel -- the same per-channel delta this
# route's own change detectors already treat as "a change" when they build their
# required-change masks.
_STATIC_DECLARATION_DELTA = 8


def _replayed_cell_regions(
    task: str,
    structured: Dict[str, Any],
    plan: Mapping[str, Any],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Whole-cell footprints of every cell the replayed plan passes through.

    The rendered state ensemble covers the tasks that own a per-state renderer,
    but a plain route video only has a first and a last rendered state, and the
    token legitimately occupies every cell between them.  Those cells come from
    the plan the source transition system accepted, never from the candidate's
    own pixels, so a tamper cannot widen its own allowance.
    """

    _size, cell, x0, y0 = _grid_geometry(task, structured)
    mask = np.zeros(shape, dtype=bool)
    for key in ("primary_path", "secondary_path"):
        for item in plan.get(key) or []:
            position = _to_pair(item)
            if position is None:
                continue
            row, col = position
            top = max(0, int(round(y0 + row * cell)))
            left = max(0, int(round(x0 + col * cell)))
            bottom = min(shape[0], int(round(y0 + (row + 1) * cell)))
            right = min(shape[1], int(round(x0 + (col + 1) * cell)))
            if bottom > top and right > left:
                mask[top:bottom, left:right] = True
    return mask


def _declared_dynamic_mask(
    sample: CanonicalSample,
    structured: Dict[str, Any],
    plan: Mapping[str, Any],
    puzzle: np.ndarray,
    source_states: Sequence[Tuple[Tuple[int, int], np.ndarray]],
) -> np.ndarray:
    """Pixels this sample's own source replay says the video may change."""

    dynamic = np.zeros(puzzle.shape[:2], dtype=bool)
    grouped: Dict[Tuple[int, int, Tuple[int, ...]], List[np.ndarray]] = {}
    for (top, left), image in source_states:
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            continue
        grouped.setdefault((int(top), int(left), array.shape), []).append(array)
    for (top, left, shape), images in grouped.items():
        # A whole-frame state is compared against the supplied input as well:
        # the input is the state the video holds before its first commit.
        if (top, left) == (0, 0) and tuple(shape) == tuple(puzzle.shape):
            images = [puzzle, *images]
        stacked = np.stack(images).astype(np.int16)
        varies = (
            np.max(np.max(stacked, axis=0) - np.min(stacked, axis=0), axis=2)
            > _STATIC_DECLARATION_DELTA
        )
        dynamic[top : top + shape[0], left : left + shape[1]] |= varies
    dynamic |= _replayed_cell_regions(
        sample.task_name, structured, plan, puzzle.shape[:2]
    )
    return cv2.dilate(
        dynamic.astype(np.uint8),
        np.ones((_STATIC_DECLARATION_DILATION, _STATIC_DECLARATION_DILATION), np.uint8),
    ) > 0


def _verify_static_preservation(
    sample: CanonicalSample,
    structured: Dict[str, Any],
    plan: Mapping[str, Any],
    frames: Sequence[np.ndarray],
    puzzle: np.ndarray,
    source_states: Sequence[Tuple[Tuple[int, int], np.ndarray]],
) -> tuple[Dict[str, Any], bool]:
    """Hard-gate every static region of every frame against the supplied input.

    The bound is the shared renderer floor, 8/255 = 0.0314.  Measured over the
    110 legal reference videos of the 22 routed tasks at five seeds, the worst
    static region reads 0.0028 and the median reads 0.0; the same videos
    re-encoded after a uniform +2/255 shift read at most 0.0166; a 24x24 opaque
    patch painted from the third frame onward reads at least 0.1029.
    """

    states = list(source_states)
    try:
        states.append((
            (0, 0),
            np.asarray(
                _render_terminal(sample, structured, plan, show_path=True),
                dtype=np.uint8,
            ),
        ))
    except Exception as exc:
        return {
            "video_static_content_preserved": False,
            "video_static_preservation_reason": f"source render failed: {exc}",
        }, False
    dynamic = _declared_dynamic_mask(sample, structured, plan, puzzle, states)
    static = ~dynamic
    if not bool(static.any()):
        return {
            "video_static_content_preserved": False,
            "video_static_preservation_reason": "source replay declares the whole frame dynamic",
        }, False
    worst = 0.0
    worst_frame = -1
    violating_frames = 0
    for index, frame in enumerate(frames):
        if frame.shape != puzzle.shape:
            return {
                "video_static_content_preserved": False,
                "video_static_preservation_reason": "frame does not match the supplied input geometry",
            }, False
        error, _regions = static_region_errors(frame, puzzle, static)
        if error > STATIC_REGION_FLOOR:
            violating_frames += 1
        if error > worst:
            worst, worst_frame = error, index
    passed = bool(frames) and violating_frames == 0
    return {
        "video_static_declared_dynamic_fraction": round(float(dynamic.mean()), 6),
        "video_static_source_state_count": len(states),
        "video_max_static_region_error": round(worst, 6),
        "video_worst_static_region_frame": worst_frame,
        "video_static_region_violation_frames": violating_frames,
        "video_static_region_size": REGION,
        "video_static_region_bound": STATIC_REGION_FLOOR,
        "video_static_preservation_source": "semantic_input_and_source_replayed_states",
        "video_static_content_preserved": passed,
    }, passed


def _source_terminal_plans(sample: CanonicalSample, structured: Dict[str, Any], *, bridge_cells=None):
    """Enumerate reachable terminal states without a saved solution route."""

    task = sample.task_name
    if task == "agent_courier_relay":
        from data.spatial_navigation.core.multi_agent_base import bfs_path

        extras = structured.get("extras") or {}
        size = int(structured.get("grid_size") or 0)
        walls = {tuple(cell) for cell in structured.get("walls") or []}
        start, meet, dropoff = (_to_pair(extras.get(key)) for key in ("a_start", "b_meet", "dropoff"))
        if start is None or meet is None or dropoff is None:
            return
        first = bfs_path(start, meet, size, walls)
        second = bfs_path(meet, dropoff, size, walls)
        if first and second:
            # Final positions and parcel ownership do not depend on which
            # legal route reaches them. A source-derived route renders that
            # terminal independently of the submitted process.
            yield {"primary_path": first, "secondary_path": second}
        return
    if task == "cooperative_open_door":
        extras = structured.get("extras") or {}
        size = int(structured.get("grid_size") or 0)
        walls = {tuple(cell) for cell in structured.get("walls") or []}
        doors = {tuple(cell) for cell in extras.get("door_cells") or []}
        blue_start, orange_start, blue_plate, orange_plate, goal = (
            _to_pair(extras.get(key)) for key in ("blue_start", "orange_start", "blue_plate", "orange_plate", "goal"))
        if any(cell is None for cell in (blue_start, orange_start, blue_plate, orange_plate, goal)):
            return
        initial = (blue_start, orange_start, False, False, False)
        queue = deque([initial])
        parents = {initial: None}
        terminal = None
        def neighbours(cell):
            for dr, dc in ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)):
                nxt = (cell[0] + dr, cell[1] + dc)
                if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in walls:
                    yield nxt
        while queue:
            state = queue.popleft()
            blue, orange, opened, crossed, entered_north = state
            if opened and crossed and blue == goal and orange == orange_plate:
                terminal = state
                break
            for next_blue in neighbours(blue):
                for next_orange in ([orange_plate] if opened else neighbours(orange)):
                    if next_blue == next_orange or (next_blue == orange and next_orange == blue):
                        continue
                    if not opened and (next_blue in doors or next_orange in doors):
                        continue
                    simultaneous_press = (next_blue == blue_plate and next_orange == orange_plate
                                          and blue != blue_plate and orange != orange_plate)
                    if not opened and next_blue == blue_plate and next_orange == orange_plate and not simultaneous_press:
                        continue
                    next_opened = opened or simultaneous_press
                    next_crossed = crossed or (entered_north and next_blue == (blue[0] + 1, blue[1]))
                    next_entered = next_blue in doors and blue == (next_blue[0] - 1, next_blue[1])
                    nxt = (next_blue, next_orange, next_opened, next_crossed, next_entered)
                    if nxt not in parents:
                        parents[nxt] = state
                        queue.append(nxt)
        if terminal is not None:
            states = []
            while terminal is not None:
                states.append(terminal)
                terminal = parents[terminal]
            states.reverse()
            answer = format_named_xy_paths_answer({"blue": [state[0] for state in states], "orange": [state[1] for state in states]})
            _metrics, legal, plan = _evaluate_text(sample, structured, answer)
            if legal:
                yield plan
        return
    if task == "grid_highest_cost":
        size = int(structured.get("grid_size") or 0)
        costs = _source_replay._to_int_grid(structured.get("cost_grid")) or []
        start, goal = _to_pair(structured.get("start")), _to_pair(structured.get("goal"))
        blocked = {tuple(cell) for cell in structured.get("blocked_cells") or []}
        if start is None or goal is None or len(costs) != size:
            return
        suffix = {}
        for row in range(goal[0], start[0] - 1, -1):
            for col in range(goal[1], start[1] - 1, -1):
                cell = (row, col)
                if cell in blocked:
                    continue
                following = [suffix[nxt] for nxt in ((row + 1, col), (row, col + 1)) if nxt in suffix]
                if cell == goal or following:
                    suffix[cell] = costs[row][col] + (max(following) if following else 0)
        if start not in suffix:
            return
        pending = [([start], costs[start[0]][start[1]])]
        while pending:
            path, total = pending.pop()
            cell = path[-1]
            if cell == goal:
                yield {"primary_path": path, "moves": _moves_from_path(path) or []}
                continue
            for nxt in ((cell[0] + 1, cell[1]), (cell[0], cell[1] + 1)):
                if nxt in suffix and total + suffix[nxt] == suffix[start]:
                    pending.append((path + [nxt], total + costs[nxt[0]][nxt[1]]))
        return
    if task == "plant_watering_route":
        initial_state = structured.get("initial_state") or {}
        rows, cols = int(structured.get("rows") or 0), int(structured.get("cols") or 0)
        capacity = int(structured.get("capacity") or 0)
        budget = int(structured.get("refill_budget", initial_state.get("refill_budget", 0)))
        start, well = _to_pair(initial_state.get("gardener")), _to_pair(initial_state.get("well"))
        plants = {tuple(cell): index for index, cell in enumerate(initial_state.get("plants") or [])}
        full_mask = (1 << len(plants)) - 1
        if start is None or well is None or not plants or capacity <= 0:
            return
        initial = (start, 0, capacity, 0)
        def successors(state):
            cell, mask, can, refills = state
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                nxt = (cell[0] + dr, cell[1] + dc)
                if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                    continue
                next_can, next_refills, next_mask = can, refills, mask
                if nxt == well:
                    if refills >= budget:
                        continue
                    next_can, next_refills = capacity, refills + 1
                if nxt in plants and not mask & (1 << plants[nxt]) and next_can > 0:
                    next_mask |= 1 << plants[nxt]
                    next_can -= 1
                yield (nxt, next_mask, next_can, next_refills), [nxt]
        is_goal = lambda state: state[1] == full_mask
    elif task in _BASIC_GRID_TASKS:
        size = int(structured["grid_size"])
        start, goal = tuple(structured["start"]), tuple(structured["goal"])
        blocked = {tuple(cell) for cell in structured.get("blocked_cells", [])}
        rewards = {tuple(cell): index for index, cell in enumerate(structured.get("required_cells", []))}
        full_mask = (1 << len(rewards)) - 1
        initial = (start, (1 << rewards[start]) if start in rewards else 0)

        def successors(state):
            cell, mask = state
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                nxt = (cell[0] + dr, cell[1] + dc)
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in blocked:
                    continue
                next_mask = mask | ((1 << rewards[nxt]) if nxt in rewards else 0)
                if nxt != goal or next_mask == full_mask:
                    yield (nxt, next_mask), [nxt]

        is_goal = lambda state: state == (goal, full_mask)
    elif task in _MECHANIC_TASKS or task in _PRIMARY_TASKS:
        builder = _source_replay._build_agent_mechanic_model if task in _MECHANIC_TASKS else _source_replay._build_primary_mechanic_model
        model = builder(structured, task_name=task)
        if model is None:
            return
        initial, successors, is_goal = model
        if task == "bridge_drop_tile" and bridge_cells is not None:
            water = [tuple(cell) for cell in structured["extras"]["water"]]
            required = sum(1 << index for index, cell in enumerate(water) if cell in bridge_cells)
            source_successors = successors
            source_goal = is_goal
            successors = lambda state: [
                (nxt, expanded) for nxt, expanded in source_successors(state)
                if not (nxt[1] & ~required)
            ]
            is_goal = lambda state: source_goal(state) and state[1] == required
    else:
        return
    parents = {initial: None}
    depths = {initial: 0}
    queue = deque([initial])
    while queue:
        state = queue.popleft()
        if is_goal(state):
            segments = []
            cursor = state
            while parents[cursor] is not None:
                previous, expanded = parents[cursor]
                segments.append(expanded)
                cursor = previous
            path = [initial[0]] + [cell for segment in reversed(segments) for cell in segment]
            _metrics, legal, plan = _evaluate_text(sample, structured, format_xy_path_answer(path))
            if legal:
                yield plan
            continue
        for next_state, expanded in successors(state):
            depth = depths[state] + len(expanded)
            if next_state not in parents and depth < MAX_VIDEO_FRAMES:
                parents[next_state] = (state, list(expanded))
                depths[next_state] = depth
                queue.append(next_state)


def _color_collection_terminal_roles(structured, frames, endpoint):
    """Read complete source-ball and delivered-slot roles, including robot occlusion."""
    from data.spatial_navigation.color_matched_collection.generator import _COLOR_TABLE

    _, cell_size, x0, y0 = _grid_geometry("color_matched_collection", structured)
    extras = structured["extras"]
    yy, xx = np.ogrid[:frames[0].shape[0], :frames[0].shape[1]]
    readings = []
    for frame in frames[-2:]:
        for index, value in enumerate(extras["balls"]):
            row, col = value
            cx, cy = x0 + (col + .5) * cell_size, y0 + (row + .5) * cell_size
            core = (xx - cx) ** 2 + (yy - cy) ** 2 <= (cell_size * .15) ** 2
            fill = np.asarray(_COLOR_TABLE[index][1])
            fraction = float((np.linalg.norm(frame[core].astype(float) - fill, axis=1) <= 75).mean())
            readings.append({"role": "collected_source_ball", "index": index,
                             "color_fraction": fraction, "pass": fraction <= .30})
        for index, value in enumerate(extras["slots"]):
            row, col = value
            cx, cy = x0 + (col + .5) * cell_size, y0 + (row + .5) * cell_size
            core = (xx - cx) ** 2 + (yy - cy) ** 2 <= (cell_size * .12) ** 2
            robot = tuple(value) == tuple(endpoint)
            fill = np.asarray((80, 80, 90) if robot else _COLOR_TABLE[index][1])
            fraction = float((np.linalg.norm(frame[core].astype(float) - fill, axis=1) <= 75).mean())
            readings.append({"role": "terminal_robot" if robot else "delivered_slot_ball", "index": index,
                             "color_fraction": fraction, "pass": fraction >= .50})
    passed = len(frames) >= 2 and all(row["pass"] for row in readings)
    return {"video_color_collection_terminal_role_readings": readings,
            "video_color_collection_terminal_roles_pass": passed}, passed


def _independent_terminal_video(
    sample: CanonicalSample,
    structured: Dict[str, Any],
    frames: Sequence[np.ndarray],
    frame_tracks: Mapping[str, Sequence[Optional[Coord]]],
) -> tuple[Dict[str, Any], bool]:
    """Check a terminal hold even when the preceding process is illegal."""

    best_distance = None
    checked = 0
    terminal_details = {}
    if len(frames) < 2:
        return {"video_independent_terminal_states_checked": 0}, False
    observed = list(frame_tracks.get("primary") or [])[-2:]
    if len(observed) != 2 or observed[0] is None or observed[0] != observed[1]:
        return {"video_independent_terminal_states_checked": 0}, False
    extras = structured.get("extras") or {}
    if sample.task_name == "agent_courier_relay":
        endpoints = [extras.get("b_meet")]
        secondary = list(frame_tracks.get("secondary") or [])[-2:]
        dropoff = _to_pair(extras.get("dropoff"))
        if len(secondary) != 2 or dropoff is None or any(cell != dropoff for cell in secondary):
            return {"video_final_state_available": True, "video_independent_terminal_states_checked": 0}, False
    elif sample.task_name == "cooperative_open_door":
        endpoints = [extras.get("goal")]
        secondary = list(frame_tracks.get("secondary") or [])[-2:]
        plate = _to_pair(extras.get("orange_plate"))
        if len(secondary) != 2 or plate is None or any(cell != plate for cell in secondary):
            return {"video_final_state_available": True, "video_independent_terminal_states_checked": 0}, False
    elif sample.task_name == "plant_watering_route":
        endpoints = (structured.get("initial_state") or {}).get("plants") or []
    elif sample.task_name in {"color_matched_collection", "parcel_delivery_route"}:
        endpoints = extras.get("slots" if sample.task_name == "color_matched_collection" else "dests") or []
    elif sample.task_name == "bus_route_passengers":
        endpoints = [extras.get("depot")]
    elif sample.task_name == "hidden_treasure_dig":
        endpoints = [_source_replay._derive_hidden_treasure(extras, size=int(structured["grid_size"]))]
    else:
        endpoints = [structured.get("goal") or extras.get("goal")]
    if tuple(observed[0]) not in {tuple(cell) for cell in endpoints if cell is not None}:
        return {"video_independent_terminal_states_checked": 0}, False
    bridge_cells = None
    if sample.task_name == "bridge_drop_tile":
        from data.spatial_navigation.core.mechanic_base import COLOR_BRIDGE
        _, cell_size, x0, y0 = _grid_geometry(sample.task_name, structured)
        readings = []
        for frame in frames[-2:]:
            cells = set()
            for value in extras["water"]:
                row, col = value
                inset = max(3, int(cell_size * .20))
                left, top = round(x0 + col * cell_size) + inset, round(y0 + row * cell_size) + inset
                right, bottom = round(x0 + (col + 1) * cell_size) - inset, round(y0 + (row + 1) * cell_size) - inset
                crop = frame[top:bottom, left:right].astype(float)
                support = np.linalg.norm(crop - np.asarray(COLOR_BRIDGE), axis=2) <= 60
                if support.size and float(support.mean()) >= .45:
                    cells.add((row, col))
            readings.append(cells)
        if readings[0] != readings[1] or len(readings[0]) > int(extras["inventory"]):
            return {"video_independent_terminal_states_checked": 0,
                    "video_terminal_bridge_inventory_conserved": False}, False
        bridge_cells = readings[0]
    for plan in _source_terminal_plans(sample, structured, bridge_cells=bridge_cells):
        endpoint = tuple(plan["primary_path"][-1])
        if any(cell != endpoint for cell in list(frame_tracks.get("primary") or [])[-2:]):
            continue
        expected = np.asarray(_render_terminal(sample, structured, plan, show_path=False))
        distance = max(_image_distance(frame, expected) for frame in frames[-2:])
        checked += 1
        best_distance = distance if best_distance is None else min(distance, best_distance)
        if distance > 0.095:
            continue
        details_ok = True
        if sample.task_name == "carry_capacity_2":
            puzzle = _rgb(Path(sample.primary_input_image))
            details_ok = all(member_commit_gate(frame, puzzle, expected)[1] for frame in frames[-2:])
        if sample.task_name == "color_matched_collection":
            terminal_details, details_ok = _color_collection_terminal_roles(structured, frames[-2:], endpoint)
        if sample.task_name in {"cooperative_open_door", "plant_watering_route"}:
            puzzle = _rgb(Path(sample.primary_input_image))
            details_ok = all(terminal_region_gates(frame, puzzle, expected, [puzzle])[1] for frame in frames[-2:])
            if details_ok and sample.task_name == "plant_watering_route":
                # Water level and the small REFILLS badge cannot be averaged
                # away by correctly green pots. Read each source-rendered
                # field against all its possible visible values separately.
                for field, maximum in (("can", int(structured["capacity"])),
                                       ("refills_used", int(structured.get("refill_budget", 0)))):
                    templates = {}
                    for value in range(maximum + 1):
                        alternative = deepcopy(plan)
                        alternative["final_state"][field] = value
                        templates[value] = np.asarray(_render_terminal(sample, structured, alternative, show_path=False))
                    mask = _variation_mask(templates)
                    expected_value = int(plan["final_state"][field])
                    if mask.any():
                        for frame in frames[-2:]:
                            observed = min(templates, key=lambda value: _masked_image_distance(frame, templates[value], mask))
                            details_ok &= observed == expected_value
        elif sample.task_name == "grid_highest_cost":
            traced = np.asarray(_render_terminal(sample, structured, plan, show_path=True))
            def red(image):
                rgb = image.astype(np.int16)
                return (rgb[:, :, 0] >= 140) & (rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2]) >= 55)
            static_red = cv2.dilate(red(expected).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            route = red(traced) & ~static_red
            tolerance = np.ones((7, 7), np.uint8)
            details_ok = bool(route.any())
            for frame in frames[-2:]:
                observed_route = red(frame) & ~static_red
                observed_near = cv2.dilate(observed_route.astype(np.uint8), tolerance).astype(bool)
                source_near = cv2.dilate(route.astype(np.uint8), tolerance).astype(bool)
                details_ok &= bool(observed_route.any() and observed_near[route].mean() >= .90
                                   and source_near[observed_route].mean() >= .90)
        elif sample.task_name == "grid_obtaining_award":
            states = _expected_process_state_images(sample.task_name, structured, plan, show_path=False)
            _metrics, details_ok = _verify_reward_pickup_video(
                structured, plan, frames[-2:], [len(states) - 1] * 2, states, show_path=False,
            )
        elif sample.task_name == "agent_courier_relay":
            states = _expected_process_state_images(sample.task_name, structured, plan)
            _metrics, details_ok = _verify_task_owned_process_details(
                sample.task_name, structured, plan, frames[-2:], [len(states) - 1] * 2,
            )
            details_ok = bool(_metrics.get("video_courier_parcel_footprint_coverage")
                              and min(_metrics["video_courier_parcel_footprint_coverage"]) >= .55)
        elif sample.task_name in {"bridge_drop_tile", "lantern_oil_route"}:
            count = len(_expected_process_state_images(sample.task_name, structured, plan))
            _metrics, details_ok = _verify_mechanic_hud_process(
                sample.task_name, structured, plan, frames[-2:], [count - 1] * 2,
            )
        if details_ok:
            return {
                **terminal_details,
                "video_independent_terminal_states_checked": checked,
                "video_source_terminal_distance": distance,
                "video_final_state_source": "reachable_source_transition_terminals",
            }, True
    return {
        **terminal_details,
        "video_independent_terminal_states_checked": checked,
        "video_source_terminal_distance": best_distance,
        "video_final_state_source": "reachable_source_transition_terminals",
    }, False


def _verify_video(
    sample: CanonicalSample,
    candidate_path: Path,
    structured: Dict[str, Any],
) -> tuple[Dict[str, Any], bool]:
    frames, fps = _video_frames(candidate_path)
    contract: Dict[str, Any]
    try:
        contract = validate_video_contract(candidate_path)
        media_valid = True
    except Exception as exc:
        contract = {"error": str(exc), "fps": fps, "frame_count": len(frames)}
        media_valid = False
    if not frames:
        return {"video_media_valid": False, "video_process_reason": "no decodable frames"}, False
    puzzle = _rgb(Path(sample.primary_input_image).expanduser().resolve())
    initial_distance = _image_distance(frames[0], puzzle)
    consecutive = [
        _image_distance(previous, current)
        for previous, current in zip(frames, frames[1:])
    ]
    changed = [value for value in consecutive if value > 0.00015]
    max_step_change = max(consecutive, default=1.0)
    tracks, track_metrics = _track_tokens(sample.task_name, structured, frames)
    frame_tracks = track_metrics.pop("_video_frame_token_tracks", {})
    frame_detections = track_metrics.pop("_video_frame_token_detections", {})
    frame_centers = track_metrics.pop("_video_frame_token_centers", {})
    cadence_metrics, cadence_ok = _verify_discrete_video_cadence(
        sample.task_name,
        structured,
        frames,
        frame_tracks,
        frame_centers,
    )
    # The terminal hold is the final *semantic* run, not an arbitrary number
    # of decoded frames.  A six-frame lookback can include the last frame of
    # the preceding dynamic-world state when the media contract adds only a
    # five-frame terminal padding, which incorrectly makes a valid terminal
    # hold look like a transition.
    semantic_boundaries = list(
        cadence_metrics.get("video_discrete_state_boundary_frames") or []
    )
    terminal_start = semantic_boundaries[-1] if semantic_boundaries else 0
    terminal_frames = frames[terminal_start:]
    final_hold_distance = max(
        (_image_distance(frames[-1], frame) for frame in terminal_frames),
        default=1.0,
    )
    temporal = bool(
        initial_distance <= 0.035
        and len(changed) >= 1
        and len(semantic_boundaries) >= 1
        and len(terminal_frames) >= 2
        and final_hold_distance <= 0.012
    )
    identity = bool(
        track_metrics.get("video_token_detection_coverage")
        and frame_detections
        and all(values and all(values) for values in frame_detections.values())
    )
    prerequisites = {
        "media_contract": media_valid,
        "temporal_holds": temporal,
        "exact_discrete_states": cadence_ok,
        "identity_visibility": identity,
    }
    process_prerequisites_ok = all(prerequisites.values())
    skip_process_replay = sample.task_name == "tide_water_cycle" and not process_prerequisites_ok
    candidate_process_metrics: Dict[str, Any] = {
        "video_process_prerequisites_pass": process_prerequisites_ok,
        "video_process_failed_prerequisites": [name for name, passed in prerequisites.items() if not passed],
        "video_process_replay_skipped": skip_process_replay,
    }
    # States the source replay declares this video passes through.  Collected
    # from the task-owned process checks that already render them, so the
    # static-preservation gate below costs one extra render rather than a
    # second full state replay.
    source_states: List[Tuple[Tuple[int, int], np.ndarray]] = []
    if skip_process_replay:
        # These were already unconditional hard gates on the final result.
        # Resolve terminal correctness independently below, without attempting
        # wait placement or rendering a process known to be invalid.
        pass
    elif sample.task_name == "lantern_oil_route":
        dynamic_process_path, lantern_metrics, lantern_path_ok = _recover_lantern_process_path(
            structured,
            frames,
            frame_tracks.get("primary") or [],
        )
        candidate_process_metrics.update(lantern_metrics)
        tracks["primary"] = dynamic_process_path if lantern_path_ok and dynamic_process_path else []
    else:
        dynamic_process_path = _select_dynamic_process_path(
            sample,
            structured,
            tracks,
            frames,
            cadence_metrics,
            frame_tracks.get("primary") or [],
            frame_tracks.get("secondary") or [],
        )
        tracks["primary"] = dynamic_process_path or _repair_observed_path(
            sample.task_name,
            structured,
            tracks.get("primary") or [],
        )
    track_metrics["video_recovered_tracks"] = {
        name: [list(cell) for cell in track]
        for name, track in tracks.items()
    }
    synthetic = _synthetic_answer_from_tracks(sample, structured, tracks)
    plan: Dict[str, Any] = {}
    semantic = False
    replay_metrics: Dict[str, Any] = {}
    if synthetic is not None and not skip_process_replay:
        replay_metrics, semantic, plan = _evaluate_text(sample, structured, synthetic)
    # Reconstruct actual synchronized ticks from the candidate's aligned frame
    # observations.  Never invent the secondary trajectory from a primary path
    # or reduce a joint program to terminal endpoints.
    if sample.task_name in _DUAL_AGENT_TASKS:
        joint_metrics, semantic, plan = _verify_joint_video_tracks(
            sample, structured, frame_tracks, frame_detections
        )
        candidate_process_metrics.update(joint_metrics)

    if sample.task_name == "plant_watering_route":
        if semantic and plan:
            cumulative_metrics, cumulative_ok = _verify_cumulative_state_video(
                sample.task_name, structured, plan, frames, frame_tracks
            )
        else:
            cumulative_metrics, cumulative_ok = {
                "video_cumulative_state_sequence_pass": False,
            }, False
        source_states.extend(cumulative_metrics.pop("_video_source_process_states", []))
        candidate_process_metrics.update(cumulative_metrics)
        semantic = bool(semantic and cumulative_ok)

    if sample.task_name in _PORTAL_ACTIVATION_TASKS:
        if semantic and plan:
            activation_metrics, activation_ok = _verify_portal_activation_video_state(
                sample.task_name,
                structured,
                plan,
                frames,
                frame_tracks,
            )
        else:
            activation_metrics, activation_ok = {
                "video_portal_discrete_event_pass": False,
            }, False
        candidate_process_metrics.update(activation_metrics)
        semantic = bool(semantic and activation_ok)

    if sample.task_name in _PROCESS_STATE_TASKS:
        if semantic and plan:
            process_state_metrics, process_state_ok = _verify_replayed_process_state_video(
                sample.task_name,
                structured,
                plan,
                frames,
            )
        else:
            process_state_metrics, process_state_ok = {
                "video_replayed_process_state_pass": False,
            }, False
        source_states.extend(process_state_metrics.pop("_video_source_process_states", []))
        candidate_process_metrics.update(process_state_metrics)
        semantic = bool(semantic and process_state_ok)

    final_state = False
    final_available = True
    final_distance: Optional[float] = None
    if semantic and plan:
        try:
            expected_final = np.asarray(_render_terminal(sample, structured, plan, show_path=False), dtype=np.uint8)
            final_distance = _image_distance(frames[-1], expected_final)
            final_state = final_distance <= 0.095
            source_states.append(((0, 0), expected_final))
        except Exception as exc:
            replay_metrics["video_source_terminal_render_error"] = str(exc)
            final_state = False
    if not semantic or sample.task_name in {"cooperative_open_door", "plant_watering_route", "grid_highest_cost"}:
        terminal_metrics, final_state = _independent_terminal_video(
            sample, structured, frames, frame_tracks,
        )
        candidate_process_metrics.update(terminal_metrics)
        final_distance = terminal_metrics.get("video_source_terminal_distance")
        final_available = bool(terminal_metrics.get("video_final_state_available", True))
    if semantic and plan:
        static_metrics, static_ok = _verify_static_preservation(
            sample, structured, plan, frames, puzzle, source_states
        )
    else:
        static_metrics, static_ok = {
            "video_static_content_preserved": False,
            "video_static_preservation_reason": "no source-replayed plan to declare a dynamic region",
        }, False
    passed = bool(
        media_valid
        and temporal
        and cadence_ok
        and identity
        and semantic
        and final_state
        and static_ok
    )
    metrics = {
        "video_media_valid": media_valid,
        "video_contract": contract,
        "video_starts_at_source_input": initial_distance <= 0.035,
        "video_initial_source_distance": initial_distance,
        "video_changed_frame_count": len(changed),
        "video_max_adjacent_frame_change": max_step_change,
        "video_final_hold_distance": final_hold_distance,
        "video_process_timing_pass": temporal,
        "video_discrete_process_pass": cadence_ok,
        **cadence_metrics,
        **track_metrics,
        **candidate_process_metrics,
        **{f"video_{key}": value for key, value in replay_metrics.items()},
        **static_metrics,
        "video_source_replay_pass": semantic,
        "video_source_terminal_distance": final_distance,
        "video_final_state_pass": final_state,
        "video_final_state_available": final_available,
        "video_identity_tracking_pass": identity,
        "video_whole_process_pass": passed,
        "video_verifier_used_reference_media": False,
    }
    return metrics, passed


def _record_for(sample: CanonicalSample) -> EvalRecord:
    return EvalRecord(
        sample_id=sample.id,
        task_name=sample.task_name,
        input_asset=sample.primary_input_image,
        task_family=sample.task_family,
        split=sample.split,
    )


def verify_spatial_navigation(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
    *,
    modality: str,
    training_phase: Optional[str] = None,
) -> List[EvalRecord]:
    """Verify exactly one formal Spatial Navigation delivery modality."""

    normalized_modality = str(modality or "").strip().lower()
    if normalized_modality not in {"ti2v"}:
        raise ValueError("Spatial Navigation verifier requires explicit modality")
    records: List[EvalRecord] = []
    for sample in samples:
        record = _record_for(sample)
        try:
            structured = _semantic_source(sample)
            video_path = resolve_candidate_video(pred_root, sample.id, sample.split)
            if video_path is None:
                raise FileNotFoundError("Missing TI2V output video")
            record.prediction_asset = video_path.as_posix()
            video_metrics, passed = _verify_video(sample, video_path, structured)
            record.offline_metrics.update(video_metrics)

            record.offline_pass = bool(passed)
            record.offline_metrics.update({
                "verifier_profile": SPATIAL_NAVIGATION_VERIFIER_PROFILE,
                "verifier_source_domain": "semantic_input_and_candidate_only",
                "requested_modality": normalized_modality,
                "binary_correct": bool(passed),
                "failure_reason": None if passed else f"{normalized_modality} source replay or formal artifact check failed",
            })
        except Exception as exc:
            record.offline_pass = False
            record.error = str(exc)
            record.offline_metrics.update({
                "verifier_profile": SPATIAL_NAVIGATION_VERIFIER_PROFILE,
                "verifier_source_domain": "semantic_input_and_candidate_only",
                "requested_modality": normalized_modality,
                "binary_correct": False,
                "failure_reason": str(exc),
            })
        records.append(record)
    return records


__all__ = ["SPATIAL_NAVIGATION_VERIFIER_PROFILE", "verify_spatial_navigation"]
