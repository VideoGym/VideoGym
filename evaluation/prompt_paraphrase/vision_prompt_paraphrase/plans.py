"""Source-bound plans for the reviewed, family-proportional subset."""

from copy import deepcopy
import math
import random
from data.common.schemas import CanonicalSample
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.catalog import (
    DESIGNS,
    validate_catalog,
)

PROTOCOL = "videogym.vision-prompt-paraphrase"
SEED = 20261027
MAZES = {"maze_square", "maze_keys_and_locks"}
BOARDS = {"go_rule_simulator"}


def choose_plan(sample, seed=SEED):
    validate_catalog()
    if sample.task_name not in DESIGNS:
        raise ValueError(
            f"{sample.task_name} has no reviewed substantive visual redesign"
        )
    profile, change, invariants = DESIGNS[sample.task_name]
    rng = random.Random(f"{seed}:{sample.task_name}:{sample.seed}")
    plan = {
        "protocol": PROTOCOL,
        "seed": seed,
        "design_revision": 2,
        "task": sample.task_name,
        "profile": profile,
        "visual_change": change,
        "semantic_contract": invariants,
        "quarter_turns": (0 if sample.task_name in {"closest_pair", "angle_bisector", "incenter"} else rng.choice((0, 1, 2, 3)))
        if profile == "geometry_workbench"
        else 0,
        "skin": "switchboard" if profile == "radial_graph" else "beacons" if profile == "beacon_graph" else "machined" if profile == "tile_go" else None,
        "layout": {
            "radial_graph": "terminal_banks",
            "beacon_graph": "beacon_field",
            "geometry_workbench": "quarter_turn",
            "node_maze": "waypoint_graph",
            "tile_go": "diamond_mesh",
            "register_chips": "radial_instruments",
            "colour_gems": "sequence_ring",
            "paper_sequence": "horizontal_punchcards",
            "token_switchyard": "diamond_grid",
            "ceramic_mosaic": "oblique_grid",
            "height_solids": "single_row_masts",
            "distance_beacons": "diamond_network",
            "open_gears": "physical_mesh",
            "height_columns": "radial_carousel",
            "hanoi_crates": "horizontal_pegs",
            "flip_foundry": "horizontal_magazines",
            "block_inlays": "board_over_supply_tray",
        }.get(profile, "fixed_coordinates"),
        "board_spacing": "compact" if profile == "tile_go" else None,
    }
    if sample.task_name == 'maze_keys_and_locks':
        plan['layout'] = 'rooms_and_passages'
    if profile == "geometry_workbench":
        from data.geometry.core.prompt_paraphrase import SCENES
        plan["geometry_scene"] = SCENES[sample.task_name][0]
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase.geometry import prompt_paraphrase_style
        plan['geometry_style'] = prompt_paraphrase_style(sample.task_name)
        plan['candidate_labels'] = {label: str(i+1) if plan['geometry_style']['label_set']=='numbers' else label
                                   for i,label in enumerate('ABCDE')}
    if profile == "radial_graph":
        plan["node_positions"] = _terminal_positions(sample, rng)
    elif profile == "beacon_graph":
        nodes = sample.semantic_input['semantic_state']['nodes']
        labels = sorted((str(n.get('label', i)) for i,n in enumerate(nodes)), key=lambda x: int(x))
        plan['display_labels'] = {label: chr(65+i) for i,label in enumerate(labels)}
        count = len(nodes)
        order = list(range(count)); rng.shuffle(order)
        phase = rng.uniform(0, 2*math.pi)
        plan['node_positions'] = [[256 + (174 if i%2 else 150)*math.cos(phase+2*math.pi*i/count),
                                   256 + (174 if i%2 else 150)*math.sin(phase+2*math.pi*i/count)] for i in order]
    return plan


def _terminal_positions(sample, rng):
    """Choose readable staggered terminal banks using graph incidence only."""
    from data.graph_theory.core.rendering import _point_segment_distance
    state = sample.semantic_input['semantic_state']
    nodes = state['nodes']; count = len(nodes)
    labels = {str(node.get('label', i)): i for i, node in enumerate(nodes)}
    edges = [(labels[str(e['u'])], labels[str(e['v'])]) for e in state.get('edges', [])]
    rows = math.ceil(count / 3)
    slots = [[90+162*(i%3), 84+(326/max(1,rows-1))*(i//3)+((i%3)-1)*12]
             for i in range(count)]
    best, best_score = None, None
    for _ in range(192):
        order = list(range(count)); rng.shuffle(order)
        positions = [slots[i] for i in order]
        clearance = min((_point_segment_distance(positions[k],positions[u],positions[v])
                         for u,v in edges for k in range(count) if k not in (u,v)), default=512)
        length = sum(math.dist(positions[u],positions[v]) for u,v in edges)
        score = (min(clearance, 48), -length)
        if best_score is None or score > best_score:
            best, best_score = positions, score
    return best


def mapped_sample(sample, plan):
    result = deepcopy(sample)
    if plan["layout"] in {"terminal_banks", "beacon_field"}:
        nodes = result.semantic_input["semantic_state"]["nodes"]
        for node, position in zip(nodes, plan["node_positions"], strict=True):
            node["position"] = list(position)
        from data.graph_theory.core.replay import replay_unified_graph

        before = replay_unified_graph(sample.task_name, sample.semantic_input)
        after = replay_unified_graph(sample.task_name, result.semantic_input)
        if before.solution != after.solution:
            raise ValueError("layout changed the graph solution")
    return result


def sample_from_row(row, *, base_dir=None):
    return CanonicalSample.from_dict(row["sample"], base_dir=base_dir)
