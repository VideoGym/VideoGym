"""One intervention per source task, with source-replayed training exposure.

The catalog fixes family coverage and semantic scale/mechanism axes across 52 tasks.
Every frozen question independently satisfies its task-wide intervention.
"""

from __future__ import annotations

from evaluation.catalog.index import EvaluationTask

GRAPH_TASKS = frozenset(
    {
        "balance_comparison",
        "closest_common_ancestor",
        "leaf_count",
        "path_between_nodes",
    }
)
CONSTRAINT_TASKS = frozenset(
    {
        "light_up_akari_completion",
        "nqueens_completion",
        "shikaku_completion",
    }
)
NAVIGATION_TASKS = frozenset(
    {
        "bridge_drop_tile",
        "carry_capacity_2",
        "color_matched_collection",
        "conveyor_belt_path",
        "directional_arrow_entry",
    }
)


SCALE_DEFINITIONS = (
    (
        "symmetry_completion",
        "objects",
        20,
        {"grid_size": 16, "source_count": 20},
        "Increase the pattern cells to map from at most 10 to 20; a 16×16 grid accommodates the complete larger pattern.",
    ),
    (
        "convex_hull_count",
        "hull_vertices",
        10,
        {"point_count": 26, "hull_count": 10},
        "Increase hull vertices from at most 6 to 10 and total points to 26, expanding both boundary construction and inside/outside discrimination.",
    ),
    (
        "grid_bfs",
        "depth",
        12,
        {"size_range": [8, 8], "distance_range": [12, 12], "source_count": 2},
        "Double the propagation depth; an eight-cell grid side provides sufficient path space.",
    ),
    (
        "maze_square",
        "path_cells",
        range(95, 126),
        {"rows": 21, "cols": 21, "path_cells_range": [95, 125]},
        "Extend the independently computed shortest route from at most 69 training cells to 95–125 cells; a 21×21 maze accommodates the longer route.",
    ),
    (
        "bridge_drop_tile",
        "bridges",
        5,
        {"size": 14, "bridge_count": 5},
        "Require five bridge tiles to cross five water bands; the larger map accommodates more mandatory resource uses.",
    ),
    (
        "carry_capacity_2",
        "route_steps",
        range(44, 61),
        {"size": 14, "route_steps": [44, 60]},
        "Keep capacity two and two full-load deliveries while extending the shortest legal route to 44–60 steps.",
    ),
    (
        "color_matched_collection",
        "route_steps",
        range(64, 81),
        {"size": 14, "route_steps": [64, 80]},
        "Keep the four native colors and single-item carrying while extending the shortest legal collection-and-delivery route to 64–80 steps.",
    ),
    (
        "conveyor_belt_path",
        "conveyors",
        7,
        {"size": 14, "chain_length": 7},
        "Extend the mandatory conveyor chain from at most four tiles to seven, retaining forced tile-by-tile movement.",
    ),
    (
        "directional_arrow_entry",
        "gates",
        4,
        {"size": 14, "gate_count": 4},
        "Require passage through four consecutive directional gates, increasing directional constraints and forced exit actions.",
    ),
    (
        "leaf_count",
        "leaves",
        range(8, 16),
        {"node_count": 16, "minimum_answer": 8},
        "Increase the node count and require at least eight leaves to contribute to the answer, rather than adding only non-leaf distractors.",
    ),
    (
        "path_between_nodes",
        "path_edges",
        range(10, 16),
        {"node_count": 16, "minimum_answer": 10},
        "Require a queried path of at least ten edges, beyond the six-edge maximum in actual training exposure; other nodes do not substitute for path length.",
    ),
    (
        "legend_operator_program",
        "length",
        5,
        {"program_length": 5},
        "Increase program length by a factor of 2.5; each step still uses the same operators inferred from examples.",
    ),
    (
        "bookshelf_sort_by_height",
        "moves",
        range(6, 9),
        {"book_count_range": [10, 10], "min_optimal_moves": 6, "max_optimal_moves": 8},
        "Require at least six optimal swaps for ten books, increasing both object count and effective rearrangement depth.",
    ),
    (
        "pancake_flip",
        "moves",
        range(6, 8),
        {
            "pancake_count_range": [10, 10],
            "min_optimal_moves": 6,
            "max_optimal_moves": 7,
        },
        "Require six to seven optimal prefix flips for ten pancakes, retaining the same sorting objective and flip action.",
    ),
    (
        "container_ship_loading_order",
        "objects",
        8,
        {"crate_count": 8, "max_optimal_moves": 12},
        "Increase the number of objects to load in order while retaining the same transport mechanism.",
    ),
    (
        "tower_of_hanoi",
        "moves",
        range(20, 29),
        {"disk_count": 7, "min_moves": 20, "max_moves": 28},
        "Require twenty to twenty-eight optimal moves in a seven-disk configuration, rather than merely adding stationary bottom disks.",
    ),
    (
        "crane_block_stack",
        "moves",
        range(10, 13),
        {"block_count_range": [9, 9], "min_optimal_moves": 10, "max_optimal_moves": 12},
        "Require ten to twelve optimal top-object transfers for nine stacked objects.",
    ),
    (
        "snake_path",
        "foods",
        7,
        {
            "width": 14,
            "height": 14,
            "snake_length_range": [8, 8],
            "food_count_range": [7, 7],
        },
        "Increase to seven food items and an initial eight-cell snake; the expanded map accommodates a longer body and repeated growth.",
    ),
    (
        "sokoban",
        "pushes",
        range(8, 11),
        {
            "width": 11,
            "height": 11,
            "push_count_range": [8, 10],
            "route_steps": [16, 48],
        },
        "Require eight to ten pushes in the shortest walking solution; the larger map accommodates the longer pushing process.",
    ),
    (
        "jewel2_future_board",
        "cleared",
        10,
        {"rows": 10, "cols": 10, "paired_runs": True},
        "Use one native adjacent swap to complete two five-item lines simultaneously, increasing the effective clearing-and-falling scale to ten cells.",
    ),
    (
        "nqueens_completion",
        "size",
        12,
        {"board_size": 12},
        "Increase the queen-board side length by half while preserving row, column, and diagonal constraints.",
    ),
    (
        "shikaku_completion",
        "size",
        10,
        {"rows": 10, "cols": 10},
        "Enlarge the rectangle-partition board while preserving a uniquely forced deduction chain.",
    ),
    (
        "light_up_akari_completion",
        "size",
        10,
        {"rows": 10, "cols": 10},
        "Enlarge the lighting board while preserving wall-number, illumination, and mutual-exclusion constraints.",
    ),
    (
        "lightsout_single_press",
        "presses",
        5,
        {"size": 9, "press_count_range": [5, 5]},
        "Increase the unique lights-out solution from at most three presses to five, retaining cross-shaped toggles, press indicators, and the all-off goal.",
    ),
    (
        "go_rule_simulator",
        "length",
        9,
        {"program_length": 9},
        "Increase the legal move-program length while retaining the seven-line Go board and simultaneous capture rules.",
    ),
    (
        "polycube_rotation",
        "length",
        8,
        {"rotation_count": 8},
        "Increase consecutive rotations while preserving voxels, camera, primitives, and the visibility of each intermediate pose.",
    ),
    (
        "closest_pair", "points", 12, {"point_count": 12},
        "Search all 66 pairs among 12 native labeled points, rather than the 10 pairs among five training points; retain a unique, visibly separated minimum and the same two-endpoint marking action.",
    ),
    (
        "lattice_points_strict_inside", "interior_points", range(18, 29),
        {"interior_range": [18, 28], "vertex_range": [7, 8]},
        "Count and mark 18–28 strict-interior lattice points under a seven- or eight-edge polygon, excluding every boundary point. Retain the native lattice and point-in-polygon rule; both the positive decision set and polygon constraints grow.",
    ),
    (
        "ray_reflect", "reflections", 4, {"reflections": 4},
        "Trace four successive finite-mirror contacts using the same equal-angle reflection law. Each mirror changes the outgoing direction; the entire four-bounce path must be drawn in travel order.",
    ),
)

MECHANISM_DEFINITIONS = (
    (
        "circle_inversion",
        "source_region",
        {"source_region": "inside"},
        ["inside"],
        "Move the inversion input from outside the circle to inside; the same r²/d mapping places the output outside.",
    ),
    (
        "homothety_center",
        "scale_sign",
        {"opposite": True},
        ["negative"],
        "Use negative homothety to place corresponding points on opposite sides of the center; locate the center with the same linear ratio and collinearity relations.",
    ),
    (
        "spiral_similarity_center",
        "similarity_ratio",
        {"expanding": True},
        None,
        "Extend the same rotation-and-scaling map from contraction to expansion without adding a composition step.",
    ),
    (
        "double_reflection",
        "axis_relation",
        {"parallel_axes": True},
        ["parallel"],
        "Change the two reflection axes from intersecting to parallel, turning the composite map from rotation into translation while retaining each reflection rule.",
    ),
    (
        "external_tangent_point",
        "circle_relation",
        {"overlapping": True},
        ["intersecting"],
        "Change the circles from disjoint to intersecting while still finding same-side common tangent contact points, rather than internal common tangents.",
    ),
    (
        "tangent_circle_touch",
        "tangency",
        {"internal": True},
        ["internal"],
        "Change external tangency to internal tangency; the goal remains the unique shared contact point, recomputed along the line of centers.",
    ),
    (
        'count_intersection_points',
        'maximum_concurrency',
        {'concurrent': True},
        [3],
        'Three lines share one crossing while a fourth creates separate crossings. Count distinct points, not intersecting pairs; the ID pair-count shortcut overcounts the shared location.',
    ),
    (
        'region_count',
        'boundary_components',
        {'nested_disconnected': True},
        [3],
        'Count faces of nested and disconnected circle boundaries, including a face with holes. Euler counting requires the component term; the connected-arrangement shortcut is wrong.',
    ),


    (
        "spiral_continuation",
        "length_step_sign",
        {"contracting": True},
        ["negative"],
        "Reverse arithmetic side-length growth into contraction while preserving right-angle turns, the recurrence, and the line-completion action.",
    ),
    (
        "arc_center",
        "arc_regime",
        {"arc_span": [220, 260]},
        ["major"],
        "Extend a sub-semicircular arc to a major arc; infer the center from the same concyclicity relation and complete the remaining arc. This is the sole arc-angle intervention.",
    ),

    (
        'diagonal_intersection',
        'diagonal_crossing_location',
        {'concave': True},
        ['exterior'],
        'Use a simple concave quadrilateral whose diagonal lines meet beyond a diagonal endpoint and outside the polygon. Extend the same opposite-vertex lines instead of searching for an interior segment crossing.',
    ),
    (
        "growing_polygon_step",
        "step_direction",
        {"descending": True},
        ["decreasing"],
        "Reverse the sign of the increment in the arithmetic vertex-count sequence.",
    ),
    (
        "increasing_size_sequence",
        "step_direction",
        {"descending": True},
        ["decreasing"],
        "Reverse the sign of the increment in the graduated-height sequence while retaining units and primitives.",
    ),
    (
        "linear_sequence_next",
        "spatial_rule",
        {"reverse_translation": True},
        ["translate_left"],
        "Reverse the direction of the same grid-translation recurrence while retaining objects, distances, and four cards.",
    ),
    (
        "closest_common_ancestor",
        "query_relation",
        {"regime": "ancestor_query"},
        ["ancestor_descendant"],
        "Change the queried nodes from separate branches to an ancestor–descendant pair while still finding the closest common ancestor.",
    ),
    (
        "balance_comparison",
        "branch_imbalance",
        {"regime": "unequal_branches"},
        [4],
        "Keep both branches larger than one leaf and extend their count difference beyond the training bound while retaining an ID-scale total node count.",
    ),
    (
        "elastic_collision_track",
        "max_speed",
        {"speed_values": [-3, -1, 0, 1, 3]},
        [3],
        "Apply the same equal-mass collision equations at higher integer speeds without increasing the ball count or simulation horizon.",
    ),
    (
        "rule_table_1d_rollout",
        "rule_ones",
        {"regime": "singleton_rule"},
        [1],
        "Change the local rule table from multiple active entries to one active entry while retaining three-bit neighborhood indexing.",
    ),
    (
        "lifelike_rule_rollout",
        "zero_birth",
        {"regime": "birth_zero"},
        [True],
        "Extend to rules allowing birth without neighbors while still counting synchronously using the public birth and survival sets.",
    ),
    (
        "cyclic_threshold_wave",
        "threshold",
        {"regime": "threshold_four"},
        [4],
        "Extend the same successor-color neighbor-threshold relation to a four-neighbor threshold without enlarging the grid.",
    ),
    (
        "sandpile_avalanche",
        "addition_region",
        {"regime": "interior_addition"},
        ["interior"],
        "Move sand addition from the boundary to the interior while retaining open boundaries, synchronous toppling, and conservation-with-loss rules.",
    ),
    (
        "elevator_dispatch_tick",
        "full_car",
        {"regime": "full_first_car"},
        ["A"],
        "Switch the full-capacity role from B to A while preserving distance-based assignment, tie-breaking priority, and capacity blocking.",
    ),
    (
        "protocol_handshake_tick",
        "initial_phase",
        {"regime": "ready_server_restart"},
        ["IDLE:READY"],
        "Start with the server ready when the client restarts while executing the same message queue and state machine.",
    ),
)


def scale_tasks():
    return [
        EvaluationTask(
            name=f"ext_{task}",
            title=f"Scale Extrapolation / {task}",
            track="extrapolation",
            task_name=task,
            config=config,
            constraints={
                axis: list(value)
                if isinstance(value, (range, list, tuple))
                else [value]
            },
            extrapolation_axis=axis,
            description=reason,
            mechanisms=(task,),
            novelty="parameter_extrapolation",
        )
        for task, axis, value, config, reason in SCALE_DEFINITIONS
    ]


def mechanism_tasks():
    return [
        EvaluationTask(
            name=f"mechanism_{task}",
            title=f"Mechanism Extrapolation / {task}",
            track="extrapolation",
            task_name=task,
            config=config,
            constraints={axis: values} if values is not None else {},
            extrapolation_axis=axis,
            description=reason,
            mechanisms=(task,),
            novelty="mechanism_extrapolation",
        )
        for task, axis, config, values, reason in MECHANISM_DEFINITIONS
    ]


def extrapolation_tasks():
    tasks = [*scale_tasks(), *mechanism_tasks()]
    if len({t.task_name for t in tasks}) != len(tasks):
        raise ValueError("a source task must not be counted twice")
    return tasks
