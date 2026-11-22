"""Reviewed rule compositions and ordered goal programs.

Mechanism names are reused across entries and training sources. Geometry's
scalar ADD/MUL pipelines and matrix legend operators do not expose Euclidean
circle inversion or metric projection; existing affine combinations are listed
explicitly rather than treated as unseen by virtue of an eval task name.
"""

from data.task_spec import TaskSpec
from data.common.verifiers.routes import VerifierRoute

# One token per independently visible rule, not one token per evaluation pair.
TRAINING_MECHANISMS = {
    "closest_common_ancestor": ("deepest_common_ancestor",),
    "route_via_checkpoint": ("required_vertex_route",),
    "avoid_blocked_vertex": ("forbidden_vertex_route",),
    "leaf_count": ("rooted_leaf_predicate",),
    "depth_lookup": ("rooted_depth",),
    "subtree_size": ("rooted_descendant_predicate",),
    "diameter_endpoints": ("tree_longest_path_endpoints",),
    "bridge_drop_tile": ("bounded_bridge_placement",),
    "conveyor_belt_path": ("compulsory_conveyor_push",),
    "angle_bisector": ("internal_angle_bisector",),
    "parallel": ("parallel_line_through_point",),
    "alternating_size_sequence": ("alternating_magnitude",),
    "tile_color_sequence": ("colour_period",),
    "increasing_size_sequence": ("positive_arithmetic_height_increment",),
    "growing_polygon_step": ("polygon_side_increment",),
    "lifelike_rule_rollout": ("life_neighbour_count",),
    "xor_parity_lattice": ("axial_bit_parity",),
    "rule_table_1d_rollout": ("local_binary_lookup",),
    "crane_block_stack": ("explicit_stack_target",),
    "dock_yard_crane_swap": ("two_crane_target",),
    "stack_reversal_minimal_moves": ("in_place_stack_reversal",),
    "token_trade_matrix": ("restricted_edge_trade",),
    "color_group_with_locks": ("fixed_item", "adjacent_exchange", "colour_grouping"),
    "go_rule_simulator": ("explicit_go_program", "liberty_capture"),
    "go_max_capture_search": ("liberty_capture", "maximize_capture"),
    "gomoku_rule_simulator": ("five_line_terminal_stop",),
    "cube_net_fold": ("hinge_net_assembly",),
    "polycube_rotation": ("rigid_quarter_rotation",),
    "candy_crush_special_predict": ("special_glyph_clear",),
    "jewel2_future_board": ("adjacent_exchange", "colour_run_clear", "stable_column_gravity"),
    "maze_keys_and_locks": ("key_acquisition_lock",),
    "one_way_portal_chain": ("directed_portal",),
    "directional_arrow_entry": ("directed_entry_forced_exit",),
    "ice_physics_multi_slide": ("compulsory_ice_slide",),
    "tide_water_cycle": ("alternating_tide",),
    "carry_capacity_2": ("bounded_cargo_pickup_delivery",),
    "lantern_oil_route": ("fuel_decrement_refill",),
    "nqueens_completion": ("queen_attack",),
    "star_battle_completion": ("region_cardinality", "king_nonadjacency"),
    "campsite_completion": (
        "tree_tent_matching",
        "row_column_cardinality",
        "king_nonadjacency",
    ),
    "yin_yang_completion": ("two_colour_connectivity_no_uniform_square",),
    "aquarium_completion": ("region_water_level", "row_column_cardinality"),
    "light_up_akari_completion": ("akari_illumination", "numbered_adjacent_bulbs"),
    "masyu_completion": ("single_loop", "pearl_turn_straight"),
    "slitherlink_completion": ("single_loop", "cell_boundary_cardinality"),
    "circle_inversion": ("euclidean_circle_inversion",),
    "orthogonal_projection": ("orthogonal_line_projection",),
    "reflection": ("line_reflection",),
    "mirror_image_select": ("line_reflection",),
    "double_reflection": ("line_reflection",),
    "glide_reflection": ("line_reflection", "vector_translation"),
    "point_rotation": ("point_rotation",),
    "rotation": ("point_rotation",),
    "inverse_rotation": ("point_rotation",),
    "double_rotation": ("point_rotation",),
    "vector_translation": ("vector_translation",),
    "translation_composition": ("vector_translation",),
    "midpoint": ("segment_midpoint",),
    "triangle_from_midpoints": ("segment_midpoint",),
    "clocked_logic_register_tick": ("simultaneous_boolean_capture",),
    "excitable_signal_network": ("excitable_head_tail_conductor",),
    "majority_consensus": ("neighbour_majority_tie_retention",),
    "grid_bfs": ("synchronous_distance_frontier",),
    "packet_router_tick": ("cyclic_router_ttl_arbitration",),
    "traffic_light_countdown": ("cyclic_signal_countdown",),
    "elastic_collision_track": ("equal_mass_velocity_exchange",),
    "ball_bounces_given_time": ("elastic_wall_reflection",),
    "domino_reach_cascade": ("domino_height_gap_reach",),
    "chained_function_pipeline": (
        "integer_add",
        "integer_multiply",
        "integer_xor",
        "integer_modulo",
        "integer_negate",
        "integer_clamp",
    ),
    "legend_operator_program": (
        "matrix_rotate",
        "matrix_reflect",
        "matrix_roll",
        "matrix_transpose",
        "matrix_palette_map",
    ),
}

# The native scene catalog is the single source for composition routes.
from evaluation.shared.scenes.catalog import TASKS, FAMILIES

TASK_SPECS = {
    "eval_" + task.key: TaskSpec(
        name="eval_" + task.key, task_family=FAMILIES[task.family],
        module="evaluation.composition.generator", class_name="NativeCompositionGenerator",
        verifier=VerifierRoute("evaluation.composition.generator", "verify"),
        answer_kind="json_object",
    ) for task in TASKS.values() if task.track == "composition"
}


COMPOSED_MECHANISMS = {
    "program_max_capture": ("explicit_go_program", "maximize_capture"),
    "candy_gravity": ("special_glyph_clear", "stable_column_gravity"),
    "gravity_candy": ("stable_column_gravity", "special_glyph_clear"),
    "crane_reverse": ("explicit_stack_target", "in_place_stack_reversal"),
    "dock_reverse": ("two_crane_target", "in_place_stack_reversal"),
}


def composition_tasks():
    from evaluation.catalog.index import EvaluationTask
    return [EvaluationTask(
        name=task.key + "_composition", title="Composition / " + task.key,
        track="composition", task_name="eval_" + task.key,
        config={"key":task.key}, primitives=task.primitives,
        mechanisms=COMPOSED_MECHANISMS.get(task.key, tuple(TRAINING_MECHANISMS[p][0] for p in task.primitives)),
        description=task.rule,
        novelty="An unseen combination of rules on the native canvas. Each source rule is necessary: deterministic programs have changed outputs; conjunctive constraints have strict relaxed-set witnesses. Source task identities alone do not prove mechanism novelty.",
    ) for task in TASKS.values() if task.track == "composition"]
