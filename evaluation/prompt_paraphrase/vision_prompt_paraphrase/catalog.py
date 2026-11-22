"""Authored, model-independent visual redesigns with proportional family seats."""
from collections import Counter
from data.registry import TASK_SPECS
from evaluation.catalog.stratification import family_quotas
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.geometry import prompt_paraphrase_description, prompt_paraphrase_style

GEOMETRY = ('midpoint','perpendicular_bisector','orthocenter','angle_bisector','circumcenter','incenter','closest_pair','diagonal_intersection','orthogonal_projection','segment_trisection')
DESIGNS = {
    **{name: ('geometry_workbench', prompt_paraphrase_description(name) + ' Candidates use ' + prompt_paraphrase_style(name)['candidate_shape'] + ' symbols, ' + prompt_paraphrase_style(name)['label_set'] + ' and ' + prompt_paraphrase_style(name)['given_colour_name'] + ' given lines.',
              'Given metric geometry, candidate centres and required constructions are unchanged; fixed label bijections preserve point identities, and body edges carry no measurements.') for name in GEOMETRY},
    'shortest_path_choice': ('radial_graph', 'Re-embed the graph as staggered plug-terminal banks with insulated cables on a circuit board.', 'Labels, edge incidence, roles and unweighted shortest paths are identical; crossings do not connect.'),
    'degree_missing_value': ('beacon_graph', 'Re-embed the node-link graph as a field of labelled beacon rosettes joined by fine violet connectors.', 'A fixed number-to-letter bijection preserves node identities, query roles, incident edges and the degree-counting operation; crossings are not vertices.'),
    'maze_square': ('node_maze', 'Rebuild the wall maze as a transit network of diamond stations and double rails.', 'One station per open cell and one rail per legal orthogonal move; endpoints and allowed paths are unchanged.'),
    'maze_keys_and_locks': ('node_maze', 'Rebuild the locked maze as a floorplan of rooms, broad passages, coloured keycards and matching vault doors.', 'Room adjacency, key identities, collection, unlock transitions and legal routes are unchanged.'),
    'one_way_portal_chain': ('portal_gates', 'Reconstruct corridors as linked diamond platforms with directional send/receive emblems.', 'Portal identity, one-way pairing, floor adjacency and source-contact/destination-arrival snapshots are unchanged.'),
    'lifelike_rule_rollout': ('disc_lattice', 'Replace bitmap occupancy by an hourglass signal array on a framed control surface.', 'Each indexed signal retains the same bit; eight-neighbour relations, the source-specified boundary convention and synchronous generations are unchanged.'),
    'grid_bfs': ('distance_beacons', 'Re-embed the rectangular grid as a diamond station network with explicit legal links and striped barrier tiles.', 'Open/source/wall roles, four-neighbour distances, frontier waves and printed values are unchanged.'),
    'clocked_logic_register_tick': ('register_chips', 'Replace the table by five separated instruments with labelled binary selector windows.', 'Every register/input identity, bit, expression and simultaneous clock-edge transition is unchanged.'),
    'bookshelf_sort_by_height': ('height_columns', 'Replace two bookshelves by circular measuring carousels; numbered radial slots carry coloured rods of the source heights.', 'Rod lengths map exactly to book heights; numbered carousel slots map bijectively to shelf slots and the same swaps solve the same target.'),
    'tower_of_hanoi': ('hanoi_crates', 'Reconstruct the peg boards as horizontal loading racks carrying wing-shaped platforms.', 'Platform vertical spans encode disk size, peg identity and stack order map bijectively, and only exposed platforms may move.'),
    'alternating_size_sequence': ('height_solids', 'Replace the two-by-two shape cards by a row of mast gauges with the same numbered height marks.', 'The mast tip preserves exact height; ordered observations, alternating values, identity and local completion are unchanged.'),
    'tile_color_sequence': ('colour_gems', 'Reorganize the sequence as a numbered open ring of four-lobed colour emblems linked by cables.', 'Printed position numbers preserve the sequence order; colours, period and final completion are unchanged.'),
    'block_blast_clear_predict': ('block_inlays', 'Rebuild occupancy from interlocking bow-tie modules, centre the board above a full-width supply tray, and preserve the piece as one rigid object.', 'One module occupies one cell; occupancy, rigid piece, anchor, simultaneous clearing and surviving blocks are unchanged.'),
    'nqueens_completion': ('crown_queens', 'Reconstruct the chessboard as a diamond socket array with eight-direction compass tokens.', 'A compass is one queen; fixed identities, rows, columns, diagonals and deduction waves are unchanged.'),
    'go_rule_simulator': ('tile_go', 'Re-embed the intersection board as a diamond mesh with tile-centred tokens and sloping coordinate axes.', 'Shared tile edges preserve exactly the orthogonal Go graph; move coordinates, colours and simultaneous captures are unchanged.'),
    'gear_train_direction': ('open_gears', 'Redraw the actual touching gear assembly with enlarged pastel toothed rims, open fan spokes and a bright mechanical plate.', 'Uniform physical scaling preserves tangencies, radius ratios, identities, phase, belt contacts, rotation directions and jammed cycles.'),
    'cube_net_fold': ('numbered_faces', 'Rebuild the paper net as enlarged amber assembly panels with riveted numeral plates and a cyan base.', 'Face numbers bijectively encode pip counts; net adjacency, base, fold order, rigid motions and final lid are unchanged.'),
}
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.new_scenes import NEW_DESIGNS
DESIGNS.update(NEW_DESIGNS)


def validate_catalog():
    actual = Counter(TASK_SPECS[name].task_family for name in DESIGNS)
    expected = family_quotas(33, Counter(t.task_family for t in TASK_SPECS.values()))
    if dict(actual) != expected:
        raise ValueError(f'visual family allocation differs: {dict(actual)} != {expected}')
    return expected
