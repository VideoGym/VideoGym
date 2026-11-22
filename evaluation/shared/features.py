"""Recompute evaluation parameters from visible semantic inputs."""

from __future__ import annotations

from collections import deque
import json

FEATURE_EXTRACTORS = {}


def register_feature_extractors(extractors):
    """Plugin extractors must replay visible input, never answers or references."""
    import importlib

    for task, extractor in extractors.items():
        if isinstance(extractor, str):
            module, symbol = extractor.rsplit(":", 1)
            extractor = getattr(importlib.import_module(module), symbol)
        if not callable(extractor):
            raise ValueError(f"{task}: feature extractor must be callable")
        FEATURE_EXTRACTORS[task] = extractor


def packed(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _maze_route(s):
    grid = s["maze_grid"]
    start, goal = tuple(s["start"]), tuple(s["goal"])
    parents = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        row, col = cell
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parents[cell]
            return list(reversed(path))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + dr, col + dc)
            if (
                0 <= nxt[0] < len(grid)
                and 0 <= nxt[1] < len(grid[0])
                and grid[nxt[0]][nxt[1]] == 0
                and nxt not in parents
            ):
                parents[nxt] = cell
                queue.append(nxt)
    raise ValueError("maze has no legal source-replayed solution")


def replay_features(sample: dict) -> dict:
    task = sample["task_name"]
    if task in FEATURE_EXTRACTORS:
        return FEATURE_EXTRACTORS[task](sample["semantic_input"])
    from evaluation.extrapolation.features import (
        GEOMETRY_MECHANISMS,
        SEQUENCE_MECHANISMS,
        STATE_MECHANISMS,
        mechanism_features,
    )

    if task in GEOMETRY_MECHANISMS | SEQUENCE_MECHANISMS | STATE_MECHANISMS:
        return mechanism_features(task, sample["semantic_input"])
    source = sample["semantic_input"]
    s = source.get("semantic_state", source)
    if task == "closest_pair":
        count = len(s["candidates"])
        return {"points": count, "pairs": count * (count - 1) // 2}
    if task == "lattice_points_strict_inside":
        from data.geometry.lattice_points_strict_inside.generator import _lattice_point_sets

        _, interior = _lattice_point_sets([tuple(p) for p in s["verts"]])
        return {"interior_points": len(interior), "vertices": len(s["verts"])}
    if task == "ray_reflect":
        from data.geometry.core.ray_optics import trace_mirrors

        contacts, _, _ = trace_mirrors(
            s["light_source"],
            s.get("incoming_ray_tip", s.get("reflection_point")),
            s.get("mirror_segments", [s.get("mirror_line")]),
        )
        return {"reflections": len(contacts)}
    if task == "maze_square":
        path = _maze_route(s)
        directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        grid = [list(row) for row in s["maze_grid"]]
        for label, key in ((2, "start"), (3, "goal")):
            row, col = s[key]
            grid[row][col] = label
        return {
            "size": len(grid),
            "cols": len(grid[0]),
            "path_cells": len(path),
            "turns": sum(a != b for a, b in zip(directions, directions[1:])),
            "keys": 0,
            "cell_px": s["cell_size"],
        }
    if task == "grid_bfs":
        from data.next_state_prediction.grid_bfs.generator import _distance_to_targets

        grid = s["input_grid"]
        distances = _distance_to_targets(grid)
        return {
            "size": len(grid),
            "cols": len(grid[0]),
            "depth": max(map(max, distances)),
            "sources": sum(v == "1" for row in grid for v in row),
            "walls": sum(v == "X" for row in grid for v in row),
            "unreachable": sum(
                grid[r][c] == "0" and d < 0
                for r, row in enumerate(distances)
                for c, d in enumerate(row)
            ),
        }
    if task == "elastic_collision_track":
        from data.physics.core.collision_physics import simulate_collisions

        replay = simulate_collisions(s["starts"], s["velocities"], s["horizon"])
        return {
            "events": len(replay["events"]),
            "balls": len(s["starts"]),
            "horizon": s["horizon"],
            "max_speed": max(map(abs, s["velocities"])),
        }
    if task == "legend_operator_program":
        from data.in_context_learning.legend_operator_program.evaluator import (
            _legend_operator_program_replay,
        )

        replay = _legend_operator_program_replay(s)
        if replay is None:
            raise ValueError("operator legend cannot be independently inferred")
        mapping = replay["inferred_mapping"]
        operations = [mapping[token] for token in s["program_tokens"]]
        return {
            "size": len(s["target_matrix"]),
            "length": len(operations),
            "operators": ">".join(operations),
            "binding_A": mapping["A"],
            "binding_size": f"{len(s['target_matrix'])}:{mapping['A']}",
        }
    from evaluation.extrapolation.catalog import (
        CONSTRAINT_TASKS,
        GRAPH_TASKS,
        NAVIGATION_TASKS,
    )

    if task in GRAPH_TASKS:
        from data.graph_theory.core.semantics import _tree_parent_depth, parse_graph_source

        graph = parse_graph_source(task, s)
        features = {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])}
        root, parent, depth = _tree_parent_depth(graph)

        def ancestors(node):
            result = []
            while node is not None:
                result.append(node)
                node = parent[node]
            return result

        if task == "closest_common_ancestor":
            a, b = graph["roles"]["queries"]
            features["query_relation"] = (
                "ancestor_descendant"
                if a in ancestors(b) or b in ancestors(a)
                else "separate_branches"
            )
        elif task == "leaf_count":
            features["leaves"] = sum(
                v != root and v not in parent.values() for v in parent
            )
        elif task == "path_between_nodes":
            a, b = graph["roles"]["query_start"], graph["roles"]["query_goal"]
            common = next(v for v in ancestors(a) if v in ancestors(b))
            features["path_edges"] = depth[a] + depth[b] - 2 * depth[common]
        else:
            sizes = [
                sum(child in ancestors(v) for v in parent)
                for child in parent
                if parent[child] == root
            ]
            features["branch_imbalance"] = abs(sizes[0] - sizes[1])
        return features
    if task in CONSTRAINT_TASKS:
        dimension = s.get("rows", s.get("size", s.get("board_size")))
        columns = s.get("cols", dimension)
        if "puzzle" in s and (
            len(s["puzzle"]) != dimension
            or any(len(row) != columns for row in s["puzzle"])
        ):
            raise ValueError("constraint source dimensions disagree with its grid")
        for field, expected in (("row_counts", dimension), ("col_counts", columns)):
            if field in s and len(s[field]) != expected:
                raise ValueError(
                    "constraint source dimensions disagree with its visible line counts"
                )
        return {"size": int(dimension), "cols": int(columns)}
    if task in NAVIGATION_TASKS:
        result = {"size": int(s["grid_size"]), "walls": len(s["walls"])}
        extra = s.get("extras", {})
        if task == "bridge_drop_tile":
            result["bridges"] = int(extra["inventory"])
        elif task == "conveyor_belt_path":
            result["conveyors"] = len(extra["conveyors"])
        elif task == "directional_arrow_entry":
            result["gates"] = len(extra["arrows"])
        elif task == "carry_capacity_2":
            from data.spatial_navigation.carry_capacity_2.generator import (
                _capacity_route,
            )

            path = _capacity_route(
                s["grid_size"],
                tuple(s["start"]),
                tuple(s["goal"]),
                tuple(extra["zone_cell"]),
                list(map(tuple, extra["cargo_cells"])),
                set(map(tuple, s["walls"])),
                capacity=2,
                delivery_budget=extra["delivery_budget"],
            )
            result["route_steps"] = len(path) - 1
        elif task == "color_matched_collection":
            from data.spatial_navigation.color_matched_collection.generator import (
                _shortest_collection_route,
            )

            path = _shortest_collection_route(
                tuple(extra["agent_start"]),
                list(map(tuple, extra["balls"])),
                list(map(tuple, extra["slots"])),
                s["grid_size"],
                set(map(tuple, s["walls"])),
            )
            result["route_steps"] = len(path) - 1
        return result
    if task in ("lightsout_single_press", "snake_path"):
        grid = s["initial_grid"]
        if not grid or any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("scale source must contain a rectangular input grid")
        result = {"size": len(grid), "cols": len(grid[0])}
        if task == "snake_path":
            result.update(
                foods=sum(v == 1 for row in grid for v in row),
                snake_cells=sum(v in (2, 3) for row in grid for v in row),
            )
        else:
            from data.board_game.lightsout_single_press.generator import (
                LightsOutSinglePressGenerator,
            )

            solver = LightsOutSinglePressGenerator.__new__(
                LightsOutSinglePressGenerator
            )
            solver.size = len(grid)
            solver.press_count_range = (1, int(s.get("maximum_press_count", 3)))
            solutions = solver._solutions(grid)
            result["presses"] = min(map(len, solutions), default=0)
        return result
    if task == "convex_hull_count":
        from data.geometry.convex_hull_count.generator import _convex_hull

        return {
            "objects": len(s["points"]),
            "hull_vertices": len(_convex_hull(s["points"])),
        }
    if task == "go_rule_simulator":
        return {"length": len(s["operations"])}
    if task in {"sokoban", "jewel2_future_board"}:
        grid = s.get(
            "initial_grid", s.get("initial_board", s.get("visible_initial_state"))
        )
        result = {"size": len(grid)}
        if task == "jewel2_future_board":
            from data.dynamics_game.jewel2_future_board.generator import (
                Jewel2FutureBoardGenerator,
            )

            solver = Jewel2FutureBoardGenerator.__new__(Jewel2FutureBoardGenerator)
            solver.rows, solver.cols = len(grid), len(grid[0])
            after = solver._swap(grid, *s["swap_cells"])
            cleared = solver._clear_matches(after)
            result["cleared"] = sum(
                a != 0 and b == 0 for x, y in zip(after, cleared) for a, b in zip(x, y)
            )
        else:
            from data.dynamics_game.sokoban.generator import SokobanPlanGenerator

            solver = SokobanPlanGenerator.__new__(SokobanPlanGenerator)
            solver.height, solver.width = len(grid), len(grid[0])
            board = [["#" if v == 9 else "." for v in row] for row in grid]
            points = {
                v: (r, c)
                for r, row in enumerate(grid)
                for c, v in enumerate(row)
                if v in (2, 3, 4)
            }
            plan = solver._solve(board, points[3], points[2], points[4])
            trace = solver._simulate(board, points[3], points[2], plan)
            result["pushes"] = sum(
                a["box"] != b["box"] for a, b in zip(trace, trace[1:])
            )
        return result
    if task in {"container_ship_loading_order", "tower_of_hanoi", "crane_block_stack"}:
        initial = s["visible_initial_state"]
        if task == "container_ship_loading_order":
            return {
                "objects": len(initial["dock"])
                + len(initial["ship"])
                + int(initial.get("buffer") is not None)
            }
        result = {"objects": sum(map(len, initial))}
        if task == "tower_of_hanoi":
            from data.rearrangement_game.tower_of_hanoi.generator import _hanoi_optimal

            plan = _hanoi_optimal(
                tuple(map(tuple, initial)),
                tuple(map(tuple, s["visible_goal_state"])),
                32,
            )
        else:
            from data.rearrangement_game.crane_block_stack.generator import (
                CraneBlockStackGenerator,
            )

            solver = CraneBlockStackGenerator.__new__(CraneBlockStackGenerator)
            plan = solver._shortest_plan(
                initial, s["visible_goal_state"], len(initial), max_moves=32
            )
        result["moves"] = len(plan)
        return result
    if task == "symmetry_completion":
        from data.geometry.symmetry_completion.generator import _validate_source_cells

        _validate_source_cells(s["source_cells"], int(s["grid_size"]), s["axis"])
        return {"size": int(s["grid_size"]), "objects": len(s["source_cells"])}
    if task == "polycube_rotation":
        from data.spatial_3d.polycube_rotation.generator import replay_polycube

        replay = replay_polycube(s)
        return {"length": len(replay["states"]) - 1, "objects": len(s["cells"])}
    if task in ("bookshelf_sort_by_height", "pancake_flip"):
        result = {"objects": len(s["visible_initial_state"])}
        if "visible_goal_state" in s or task == "bookshelf_sort_by_height":
            initial = tuple(s["visible_initial_state"])
            target = tuple(s.get("visible_goal_state", sorted(initial, reverse=True)))
            if task == "bookshelf_sort_by_height":
                from data.rearrangement_game.bookshelf_sort_by_height.generator import (
                    minimum_swap_plan as solve,
                )
            else:
                from data.rearrangement_game.pancake_flip.generator import (
                    _bfs_optimal as solve,
                )
            result["moves"] = len(solve(initial, target, 16))
        return result
    raise ValueError(f"no generalization feature extractor for {task}")
