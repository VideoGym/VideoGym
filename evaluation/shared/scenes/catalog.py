"""Task definitions for the 34-task Composition benchmark."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    key: str
    family: str
    track: str
    primitives: tuple[str, ...]
    host: str
    rule: str


_ROWS = [
    (
        "ancestor_subtree_diameter",
        "graph",
        "composition",
        ("closest_common_ancestor", "subtree_size", "diameter_endpoints"),
        "closest_common_ancestor",
        "Find the deepest common ancestor of the two purple-ringed queries. Take that ancestor and all its descendants as a subtree. Within that subtree, find the unique pair joined by the most edges. Highlight their complete connecting path in teal, then ring both endpoints orange. Do not mark the intermediate ancestor or the other subtree nodes.",
    ),
    (
        "subtree_leaves",
        "graph",
        "composition",
        ("subtree_size", "leaf_count"),
        "subtree_size",
        "Ring every leaf belonging to the subtree rooted at the purple query node; exclude leaves outside that subtree.",
    ),
    (
        "bridge_conveyor",
        "navigation",
        "composition",
        ("bridge_drop_tile", "conveyor_belt_path"),
        "bridge_drop_tile",
        "Reach G using the finite bridge inventory and obey every conveyor push.",
    ),
    (
        "portal_conveyor",
        "navigation",
        "composition",
        ("one_way_portal_chain", "conveyor_belt_path"),
        "one_way_portal_chain",
        "Reach G using the three numbered one-way portal pairs in order and obey every conveyor push.",
    ),
    (
        "portal_arrow",
        "navigation",
        "composition",
        ("one_way_portal_chain", "directional_arrow_entry"),
        "one_way_portal_chain",
        "Reach G using the three numbered one-way portal pairs in order and obey the directional-entry arrow gates.",
    ),
    (
        "bridge_arrow",
        "navigation",
        "composition",
        ("bridge_drop_tile", "directional_arrow_entry"),
        "bridge_drop_tile",
        "Reach G using the finite bridge inventory and obey the directional-entry arrow gates.",
    ),
    (
        "rotate_midpoint",
        "geometry",
        "composition",
        ("point_rotation", "midpoint"),
        "point_rotation",
        "Rotate S a quarter-turn counterclockwise about O (in Cartesian orientation); call the result R. Select the candidate at the midpoint of R and A.",
    ),
    (
        "midpoint_rotate",
        "geometry",
        "composition",
        ("midpoint", "point_rotation"),
        "midpoint",
        "Find the midpoint M of S and A, then rotate M a quarter-turn counterclockwise about O (in Cartesian orientation). Select the candidate at the rotated point.",
    ),
    (
        "midpoint_bisector",
        "geometry",
        "composition",
        ("midpoint", "angle_bisector"),
        "angle_bisector",
        "Find M, the midpoint of A and B. Select the candidate on the internal bisector ray of angle CMA, with vertex M.",
    ),
    (
        "midpoint_parallel",
        "geometry",
        "composition",
        ("midpoint", "parallel"),
        "parallel",
        "Find M, the midpoint of A and B. Select the candidate C such that MC is parallel to the reference segment DE.",
    ),
    (
        "rotate_parallel",
        "geometry",
        "composition",
        ("point_rotation", "parallel"),
        "parallel",
        "Rotate segment OS a quarter-turn counterclockwise about O (in Cartesian orientation). Through A, construct a parallel to that rotated segment and select its candidate.",
    ),
    (
        "size_color",
        "sequence",
        "composition",
        ("alternating_size_sequence", "tile_color_sequence"),
        "alternating_size_sequence",
        "Continue both the alternating ruled height and the three-color cycle (exact period 3) into card 4.",
    ),
    (
        "growth_color",
        "sequence",
        "composition",
        ("increasing_size_sequence", "tile_color_sequence"),
        "increasing_size_sequence",
        "Continue both the constant positive ruled-height increment and the three-color cycle (exact period 3) into card 4.",
    ),
    (
        "polygon_color",
        "sequence",
        "composition",
        ("growing_polygon_step", "tile_color_sequence"),
        "growing_polygon_step",
        "Continue both the constant polygon-side increment and the three-color cycle (exact period 3) into card 4.",
    ),
    ("collision_reflection", "physics", "composition", ("elastic_collision_track", "ball_bounces_given_time"), "elastic_collision_track", "Compose equal-mass velocity exchange and reflecting boundaries on the source rail."),
    ("fold_rotate", "spatial", "composition", ("cube_net_fold", "polycube_rotation"), "cube_net_fold", "Fold the native net, then rigidly quarter-turn the assembled cube about its centre."),
    ("candy_gravity", "dynamics", "composition", ("candy_crush_special_predict", "jewel2_future_board"), "candy_crush_special_predict", "Apply the special-token clearing rule, then stable column gravity."),
    ("locked_trade", "rearrangement", "composition", ("token_trade_matrix", "color_group_with_locks"), "token_trade_matrix", "Reach the shown target with the fewest edge trades while the designated token remains fixed."),
    ("program_max_capture", "board", "composition", ("go_rule_simulator", "go_max_capture_search"), "go_max_capture_search", "Execute the prescribed Go program, then choose and execute the unique maximum-capture Black move on its output."),
    ("life_xor", "next_state", "composition", ("lifelike_rule_rollout", "xor_parity_lattice"), "lifelike_rule_rollout", "Apply one Life-like update, then one XOR parity update to the resulting bit grid."),
    ("xor_life", "next_state", "composition", ("lifelike_rule_rollout", "xor_parity_lattice"), "lifelike_rule_rollout", "Apply one XOR parity update, then one Life-like update to its output."),
    ("life_xor_life", "next_state", "composition", ("lifelike_rule_rollout", "xor_parity_lattice"), "lifelike_rule_rollout", "Apply Life-like, XOR parity, and Life-like updates in that order to the same bit grid."),
    ("reflect_midpoint", "geometry", "composition", ("reflection", "midpoint"), "reflection", "Reflect the source, then take its midpoint with the fixed anchor."),
    ("project_midpoint", "geometry", "composition", ("orthogonal_projection", "midpoint"), "orthogonal_projection", "Project the source onto the axis, then take its midpoint with the fixed anchor."),
    ("invert_midpoint", "geometry", "composition", ("circle_inversion", "midpoint"), "circle_inversion", "Invert the source in the circle, then take its midpoint with the fixed anchor."),
    ("reflect_rotate", "geometry", "composition", ("reflection", "point_rotation"), "reflection", "Reflect the source across the axis, then rotate its image about the fixed pivot."),
    ("project_rotate", "geometry", "composition", ("orthogonal_projection", "point_rotation"), "orthogonal_projection", "Project the source onto the axis, then rotate the foot about the fixed pivot."),
    ("midpoint_reflect_rotate", "geometry", "composition", ("midpoint", "reflection", "point_rotation"), "midpoint", "Take the midpoint with the fixed anchor, reflect it across the axis, then rotate about the fixed pivot."),
    ("life_rowtable", "next_state", "composition", ("lifelike_rule_rollout", "rule_table_1d_rollout"), "lifelike_rule_rollout", "Apply Life-like once, then apply the binary local rule table independently to every row."),
    ("crane_reverse", "rearrangement", "composition", ("crane_block_stack", "stack_reversal_minimal_moves"), "crane_block_stack", "Reach the shown intermediate stack arrangement, then reverse its occupied stack in place using empty buffers."),
    ("dock_reverse", "rearrangement", "composition", ("dock_yard_crane_swap", "stack_reversal_minimal_moves"), "dock_yard_crane_swap", "Reach the shown intermediate stack arrangement with both cranes, then reverse its occupied stack using single-box transfers."),
    ("gravity_candy", "dynamics", "composition", ("jewel2_future_board", "candy_crush_special_predict"), "candy_crush_special_predict", "Settle the incomplete candy board under gravity, carrying the special glyph with its token, then activate it once."),
    ("queens_regions", "constraints", "composition", ("nqueens_completion", "star_battle_completion"), "star_battle_completion", "Complete the native region-star board under both full queen attacks and the row, column, region and non-touching rules."),
    ("queens_akari", "constraints", "composition", ("nqueens_completion", "light_up_akari_completion"), "light_up_akari_completion", "Complete the native Akari board under illumination, numbered-wall and non-visible-bulb rules plus one nonattacking marker in each row and column."),
]
TASKS = {row[0]: Task(*row) for row in _ROWS}
FAMILIES = {
    "constraints": "constraint_grid",
    "physics": "physics",
    "spatial": "spatial_3d",
    "dynamics": "dynamics_game",
    "rearrangement": "rearrangement_game",
    "board": "board_game",
    "next_state": "next_state_prediction",
    "graph": "graph_theory",
    "navigation": "spatial_navigation",
    "geometry": "geometry",
    "sequence": "in_context_learning",
}


def module(key):
    import importlib

    implementation = TASKS[key].family
    if key in {"reflect_midpoint", "project_midpoint", "invert_midpoint", "reflect_rotate", "project_rotate", "midpoint_reflect_rotate"}:
        implementation = "geometry_chain"
    elif key in {"crane_reverse", "dock_reverse"}:
        implementation = "stack_program"
    elif key == "gravity_candy":
        implementation = "gravity_candy"
    return importlib.import_module("evaluation.shared.scenes." + implementation)
