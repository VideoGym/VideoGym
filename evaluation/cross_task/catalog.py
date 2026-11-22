"""Family-derived Cross-Task designs. No ID parents, primitive prerequisites or pairs.

The full registered training contracts are the novelty audit universe. Family
counts only determine quotas; they do not determine these mechanisms or graphics.
"""

from dataclasses import dataclass

from data.common.verifiers.routes import VerifierRoute
from data.task_spec import TaskSpec
from evaluation.catalog.index import EvaluationTask


@dataclass(frozen=True)
class Design:
    family: str
    module: str
    mechanism: str
    rule: str
    rationale: str


DESIGNS = {}


def add(family, module, key, mechanism, rule, rationale):
    DESIGNS[key] = Design(family, module, mechanism, rule, rationale)


add(
    "geometry",
    "geometry",
    "taut_obstacle_path",
    "visibility_geodesic",
    "Connect S to G by the shortest Euclidean polyline avoiding the interiors of all solid obstacles. Contact with corners and boundaries is allowed. Draw the complete path in orange.",
    "Continuous obstacle geodesics require a visibility graph and metric optimization, rather than an ID point construction or a discrete maze route.",
)
add(
    "geometry",
    "geometry",
    "polygon_triangulation",
    "optimal_triangulation",
    "Partition the convex polygon into triangles using noncrossing diagonals between its vertices. Minimize the sum of diagonal lengths. Draw every chosen diagonal in orange; keep the polygon boundary.",
    "A global noncrossing decomposition couples all diagonal decisions; nearest-edge choices need not minimize the complete triangulation.",
)
add(
    "geometry",
    "geometry",
    "bichromatic_matching",
    "planar_perfect_matching",
    "Join every red terminal to exactly one blue terminal, using every terminal once. Segments may not cross. Minimize the sum of their Euclidean lengths. Draw all five connections in orange.",
    "One-to-one planar matching couples five assignments and crossing constraints; it is neither closest-pair selection nor a renamed extreme statistic.",
)
add(
    "geometry",
    "geometry",
    "rectangle_partition",
    "minimum_rectangulation",
    "Partition the teal unit squares into the fewest axis-aligned rectangles. Every teal cell belongs to exactly one rectangle; dark cells belong to none. Draw only the internal boundaries between rectangles in orange.",
    "Exact disjoint coverage and the global rectangle count define a new geometric decomposition objective. The holes are public, and every optimal partition is valid.",
)
add(
    "geometry",
    "geometry",
    "halfplane_intersection",
    "convex_halfplane_clipping",
    "Each blue line bounds a closed half-plane; its green arrow points to the allowed side. Construct the boundary of the region satisfying ALL seven half-plane restrictions, in orange. The canvas frame is not a constraint.",
    "Feasibility is the intersection of several oriented regions, not an isolated line intersection; inactive constraints must be excluded from the output boundary.",
)
add(
    "geometry",
    "geometry",
    "power_cells",
    "weighted_power_diagram",
    "For each small outlined query circle at a grid intersection centre, compare (x-a)^2+(y-b)^2-w for sites A, B and C, with distances in grid-cell units. Ring each query in its minimizing site color: A teal, B blue, C red. Sites carry their weights w. Coordinates increase right and down.",
    "Additive power weights alter Voronoi ownership; the nearest visible centre can be wrong. Ten simultaneous region classifications test the full weighted metric.",
)
add(
    "geometry",
    "geometry",
    "projective_landmarks",
    "homography_registration",
    "A single planar projective transformation maps the upper unit-square corners A,B,C,D to the lower quadrilateral corners with those labels. Add the four numbered landmarks at their transformed locations in the lower quadrilateral, using orange number badges. Preserve both source diagrams.",
    "Projective registration has position-dependent scale and does not reduce to an ID translation, rotation, reflection or uniform dilation.",
)
add(
    "geometry",
    "geometry",
    "disk_union_boundary",
    "arrangement_union_boundary",
    "Trace in orange exactly the exterior boundary of the union of the three disks. Preserve all source circumferences. Circumference arcs lying inside any other disk are not part of the answer.",
    "Arrangement topology requires classifying and joining exposed arcs across intersections. Selecting one circle or tracing every circumference is insufficient.",
)
add(
    "geometry",
    "geometry",
    "minkowski_sum",
    "discrete_minkowski_addition",
    "Occupied cells in the upper A and B grids represent integer-coordinate point sets; each upper-left cell is (0,0), x increases right and y down. Fill a lower cell orange exactly when its coordinate equals a+b for some a in A and b in B. Merge duplicate sums.",
    "Set addition constructs a new support from all pairs and resolves collisions, instead of composing two rigid transformations or counting an intersection.",
)
add(
    "geometry",
    "geometry",
    "minimum_width_strip",
    "support_width_optimization",
    "Construct the narrowest infinite parallel strip containing the entire polygon. Draw both orange supporting line segments across the polygon and extend each slightly beyond it. Width is perpendicular separation, not the distance between extreme vertices.",
    "The optimal support orientation is a global projection property, distinct from point-pair extrema or a given parallel-line construction.",
)

add(
    "next_state_prediction",
    "systems",
    "petri_target_marking",
    "petri_reachability",
    "Numbered transition bars consume one token from EACH incoming blue-arrow place and produce one at EACH outgoing green-arrow place. A transition is enabled only when all its input tokens exist. Each place holds at most four tokens. Find a shortest firing sequence reaching every shown target count. Write transition IDs in order in the bottom registers and fill unused suffix registers with 0.",
    "Joint token enabling and resource conservation create reachability in a concurrent system, absent from the explicit ID register and cellular transition laws.",
)
add(
    "next_state_prediction",
    "systems",
    "causal_counterfactual",
    "structural_causal_intervention",
    "A and B are exogenous bits. Every other node equals its displayed AND, OR or XOR of its two parents, then XOR its own unknown fixed noise bit. Infer all six noise bits from the factual observed values. For each do(node=value) row, replace ONLY that node equation by the forced value, retain the same noise bits, and recompute all nodes in DAG order. Fill every requested bit.",
    "Abduction followed by surgical intervention changes an equation while holding latent disturbances fixed; this is not ordinary forward circuit execution.",
)
add(
    "next_state_prediction",
    "systems",
    "hidden_markov_path",
    "viterbi_inference",
    "The hidden mode is 0 or 1, initially equally likely. At each observation the current mode emits a symbol with the shown probability, then transitions according to the arrows before the next observation. Find the most probable COMPLETE seven-mode path conditional on all seven observations. Write every hidden mode below its observation. Optimize the joint path probability, not each marginal separately.",
    "Latent probabilistic path inference requires products across future evidence and competing histories; no ID task supplies this uncertainty model.",
)
add(
    "next_state_prediction",
    "systems",
    "delayed_wave_superposition",
    "signed_delay_superposition",
    "A and B emit the listed signed amplitudes at t=0,1,2 and zero otherwise. Every directed link delivers each source amplitude after its displayed integer delay. At C,D,E, incoming amplitudes add algebraically at each tick and the resulting sum propagates down EVERY outgoing link. Signals do not persist between ticks. Write E amplitudes for t=0 through 9, including explicit zeros.",
    "Signed cancellation and multiple delayed paths require temporal superposition, unlike binary excitation or fixed Boolean propagation.",
)

add(
    "spatial_navigation",
    "navigation",
    "reactive_pursuer_escape",
    "reactive_pursuit_planning",
    "The teal runner R must reach G. Each tick R first waits or moves one orthogonal cell, never into H; the red hunter H then moves one orthogonal cell along a shortest wall-avoiding path toward the NEW runner location, breaking ties in up,right,down,left order. If no adjacent cell exists H stays. The runner may never share H's cell, even on the final tick. Find a shortest safe escape. In each lower column write R's cell above H's cell after the tick, then fill both suffix rows with 0.",
    "The obstacle responds to the chosen route through closed-loop pursuit. Unlike a pre-scripted ID patrol, its future path depends on the runner; sampling requires a detour beyond the shortest hunter-free route.",
)

add(
    "spatial_navigation",
    "navigation",
    "joint_robot_exchange",
    "collision_free_joint_planning",
    "A and B must exchange starting cells. Each tick each robot may wait or move one orthogonal cell. They may never share a cell or swap the two ends of an edge in the same tick. Walls and outside cells are forbidden. Minimize elapsed ticks. In each bottom column write the numbered cells of A (teal row) and B (blue row) after that tick, then 0 in both rows of every unused suffix column.",
    "Planning jointly for two independent moving agents requires reservation of space-time, rather than forecasting a supplied simultaneous move or following one route.",
)
add(
    "spatial_navigation",
    "navigation",
    "reliable_bridge_route",
    "multiplicative_reliability_routing",
    "Every directed bridge independently survives with its printed probability. Select an S-to-G route maximizing the probability that ALL bridges on the route survive, the product of their probabilities. Mark its complete bridge connections in orange.",
    "Multiplicative survival is a new stochastic route objective. The highest-probability first bridge must be evaluated against downstream risks.",
)
add(
    "spatial_navigation",
    "navigation",
    "winding_route",
    "homotopy_constrained_closed_walk",
    "Make a shortest closed orthogonal walk starting and ending at S, never entering the dark islands. Its signed clockwise winding about island 1 must be +1 and about island 2 must be 0. In screen coordinates count a downward crossing of the eastward horizontal ray from an island centre as +1, an upward crossing as -1, using the edge from the island row to the row below it. Write visited cell numbers after each step, then 0 in every unused suffix slot.",
    "A route is constrained by its topological class, including a closed return and signed crossings; a shortest path between endpoints alone does not solve it.",
)

add(
    "rearrangement_game",
    "rearrangement",
    "stable_partner_assignment",
    "stable_matching",
    "Match each A person to one distinct B person. Preference lists run from best at left to worst at right. No unmatched A,B pair may BOTH strictly prefer each other to their assigned partners. Write the B partner number for each A. Accept any stable perfect matching; rank-sum minimization is not the objective.",
    "Two-sided blocking-pair stability is a global relational arrangement condition absent from sorting, stack and vehicle-rearrangement contracts.",
)
add(
    "rearrangement_game",
    "rearrangement",
    "jobshop_schedule",
    "disjunctive_jobshop_scheduling",
    "Each job A,B,C performs its three displayed operations in left-to-right order. An operation occupies the named machine for the printed integer duration, cannot be interrupted, and starts at a nonnegative integer time. Each machine handles at most one operation at a time. Minimize the finish time of all jobs. Write every operation start time in the corresponding lower row.",
    "Machine conflicts and job precedence form a disjunctive scheduling problem, rather than a prescribed dispatch order or two-bin load sum.",
)
add(
    "rearrangement_game",
    "rearrangement",
    "polyomino_packing",
    "labeled_exact_packing",
    "Use each of the four colored pieces exactly once to tile the empty 4x4 board, with no holes or overlaps. Translation, quarter-turn rotation and reflection are allowed. Fill each board cell with its occupying piece color. Pieces with the same shape still have distinct colors and identities.",
    "Exact geometric packing requires jointly choosing piece transforms and placements; it is not a cellwise color or prescribed falling-block update.",
)

add(
    "in_context_learning",
    "induction",
    "consistent_cryptogram",
    "joint_substitution_inference",
    "Each lowercase cipher letter has one fixed uppercase plaintext letter, and distinct cipher letters map to distinct letters. Every cipher row encodes one word in the public word bank. Infer a SINGLE letter bijection consistent with all rows and write the corresponding word-bank number for each row. Words may be reused if consistent.",
    "A shared substitution is constrained jointly across words; no row independently supplies a token legend, and dictionary decoys have compatible local patterns.",
)
add(
    "in_context_learning",
    "induction",
    "hidden_transducer",
    "latent_mealy_machine_induction",
    "Every example comes from the same deterministic transducer with two hidden states 0,1. It resets to hidden state 0 before each row. For each input bit, the unknown transition table emits one output bit and changes state; both depend on current hidden state and input bit. Infer a table consistent with ALL examples, then fill the three query output bit strings.",
    "Latent state and feedback must be inferred from whole strings; this differs from a visible matrix-operation legend or a given tape transition table.",
)
add(
    "in_context_learning",
    "induction",
    "recursive_rewrite",
    "parallel_grammar_induction",
    "Every A and every B is simultaneously replaced by its own fixed nonempty word of length 1 or 2 over A,B. The examples each show ONE parallel generation. Infer both productions, then apply THREE generations to the query word. Write the resulting word left to right, filling unused suffix slots with _. Preserve character order and do not rewrite newly created characters within the same generation.",
    "A learned variable-length recursive grammar grows its own future operands, unlike a fixed-length visual transformation or scalar sequence increment.",
)

add(
    "graph_theory",
    "graphs",
    "capacitated_max_flow",
    "integer_flow_conservation",
    "Assign a nonnegative integer flow to EVERY directed edge, not exceeding its printed capacity. At each vertex other than S,T, total incoming flow equals total outgoing flow. Maximize total flow from S to T. Write every edge flow, including zero. All maximum flows are accepted.",
    "Capacity and conservation couple multiple simultaneous routes. Choosing one shortest path or one bottleneck edge cannot determine a complete maximum flow.",
)
add(
    "graph_theory",
    "graphs",
    "minimum_arborescence",
    "directed_spanning_arborescence",
    "Select directed edges so S reaches every other vertex, every non-root vertex has exactly one selected incoming edge, and S has none. Minimize the total printed edge weight. Mark every selected edge in orange. The direction of the source arrow remains authoritative.",
    "Directed global rooted connectivity excludes locally cheap incoming cycles. The sampler requires the greedy independent incoming-edge choice to fail.",
)

add(
    "dynamics_game",
    "arcade",
    "bomb_chain_timing",
    "instantaneous_blast_cascade",
    "Each bomb has the small white fuse time printed above its writable answer region. A bomb explodes at its fuse time unless an earlier explosion triggers it. Blasts instantly propagate up to three cells in each orthogonal direction, blocked by dark walls and the outer boundary. A triggered bomb explodes at that same time and emits its own blast. Bombs do not block rays. Write every bomb actual detonation time.",
    "Competing clocks and instantaneous transitive chain reactions change the event schedule; ID clearing rules do not define this timed cascade.",
)
add(
    "dynamics_game",
    "arcade",
    "recurrent_rotor_router",
    "stateful_rotor_walk",
    "At each tick, increment the current node rotor to its next numbered port, wrapping from its highest port to 1, THEN move along that port. The drawn arrow is the initial rotor port. All rotors retain their changed positions across revisits. Perform exactly the stated number of moves. Write how many DEPARTURES occurred from each node in its lower box, then write the final node number.",
    "Revisiting a routing element changes future behavior because its rotor retains memory; this is not a fixed downward pegboard path.",
)
add(
    "constraint_grid",
    "constraints",
    "nonogram_runs",
    "run_length_constraint_intersection",
    "Each row and column clue lists the lengths of its consecutive filled runs in order, separated by at least one empty cell. Fill every board cell teal for occupied or gray for empty so ALL clues hold simultaneously. The source clue digits stay fixed.",
    "Intersecting run languages across rows and columns is a new constraint system, distinct from the ID region, loop, illumination and queen puzzles.",
)
add(
    "constraint_grid",
    "constraints",
    "kakuro_cross_sums",
    "sum_alldifferent_runs",
    "Fill every empty white cell with a digit 1 through 6. Each uninterrupted horizontal or vertical white run has distinct digits and sums to its clue: the upper-right number in the preceding dark cell governs the rightward run; the lower-left number governs the downward run. Blue given digits cannot change.",
    "Arithmetic sums coupled with within-run all-different constraints define Kakuro, rather than an ID region-size or occupancy constraint.",
)
add(
    "board_game",
    "board",
    "othello_endgame",
    "adversarial_disk_flipping",
    "Black moves first on the upper 4x4 Othello board. A legal move places one disk on an empty square and must bracket at least one consecutive opposing disk along one of eight directions; all such bracketed disks flip. Players alternate, passing only with no legal move; the game ends when neither can move. Both optimize final black-minus-white disk count (Black maximizes, White minimizes). Construct in the lower board the position immediately after one minimax-optimal Black move. Teal is empty, charcoal Black, ivory White.",
    "The first move is evaluated against adversarial future play. Sampling rejects positions where every choice is equivalent and requires a greedy immediate-flip choice to be suboptimal.",
)
add(
    "physics",
    "physics",
    "layered_refraction",
    "snell_refraction",
    "Continue the incident ray through the five transparent horizontal media. Angles are measured from the vertical interface normal. At every interface apply n_before*sin(theta_before)=n_after*sin(theta_after); retain the rightward direction. Draw every transmitted segment in orange down to the bottom of the medium. No reflection is required: all supplied indices permit transmission.",
    "Refractive bending depends on the ratio of optical indices through several media, introducing a physical law absent from elastic collision, mirror reflection and liquid-flow training.",
)
add(
    "spatial_3d",
    "space",
    "polycube_assembly",
    "proper_rotation_3d_exact_cover",
    "Assemble the three colored rigid four-cube pieces into a completely filled 3x2x2 box. Proper rotations and translations in 3D are allowed; reflection, splitting or overlap is forbidden. Fill each cell in the two z-layer grids with the color of its occupying piece; x increases right and y down. Use every piece once.",
    "Joint 3D placement and chirality-preserving rotations solve an assembly problem rather than predicting a view or rotating one given object.",
)


def cross_task_tasks():
    return [
        EvaluationTask(
            name=key,
            title="Cross-Task / " + key.replace("_", " ").title(),
            track="cross_task",
            task_name="eval_cross_" + key,
            description=d.rule,
            novelty=d.rationale,
            mechanisms=(d.mechanism,),
        )
        for key, d in DESIGNS.items()
    ]


TASK_SPECS = {}
for key, d in DESIGNS.items():
    name = "eval_cross_" + key
    TASK_SPECS[name] = TaskSpec(
        name=name,
        task_family=d.family,
        module="evaluation.cross_task.generator",
        class_name="".join(x.title() for x in key.split("_")) + "Generator",
        verifier=VerifierRoute("evaluation.cross_task.generator", "verify"),
        answer_kind="json_object",
    )
FEATURE_EXTRACTORS = {
    name: "evaluation.cross_task.generator:extract_features" for name in TASK_SPECS
}
