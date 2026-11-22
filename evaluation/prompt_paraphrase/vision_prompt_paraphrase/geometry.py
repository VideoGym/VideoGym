"""Native geometry workbench scenes and their exact observation coordinates."""
from pathlib import Path
from data.common.prompt_paraphrase import use_prompt_paraphrase


def render_geometry(sample, plan, root, *, video=False):
    """Rebuild the same structured problem using surveying-table primitives."""
    from evaluation.prompt_paraphrase.vision_prompt_paraphrase.rendering import regenerate
    with use_prompt_paraphrase(dict(plan, task=sample.task_name)):
        return regenerate(sample, Path(root), video=video)


DESCRIPTIONS = {
    'midpoint': 'A broad surveying bar joins two bolted socket hubs. The fine dark centreline defines the original segment; lettered sliding targets mark the candidate centres.',
    'perpendicular_bisector': 'Two endpoint beacons are linked by a dotted baseline. The finite segment is defined by the two beacon centres; lettered sensor targets are the candidate points.',
    'angle_bisector': 'Two lavender light-beam carriers emanate from a common emitter. Only their fine dark centrelines are the mathematical rays; the beam boundaries are decorative. Lettered detector housings identify candidate centres.',
    'circumcenter': 'Three antenna mounts are joined by straight communication cables forming the visible triangle. The fine dark cables and their vertices retain the given geometry; lettered receiver badges mark the candidate centres.',
    'incenter': 'A masonry courtyard encloses a triangular region. The fine dark wall centrelines define the sides, and the lettered surveying pins identify candidate centres. Mortar joints encode no distances or divisions.',
    'orthocenter': 'A cable scaffold joins three square tension mounts. The fine dark cable centrelines define the triangle and its vertices; lettered sight lenses identify candidate centres.',
    'diagonal_intersection': 'A glazed frame outlines the quadrilateral. Measure its fine dark centreline, not the ornamental frame edges. Lettered tabs identify the same candidate points on the glass.',
    'orthogonal_projection': 'A cream guide rail carries lettered gauge sliders, with a hollow source gauge off the rail. The fine dark centreline is the given line and the letter centres are its candidate points.',
    'segment_trisection': 'An ungraduated mauve measuring strap links the solid endpoint and hollow endpoint. Its fine dark centreline is the original segment; lettered clips identify candidate centres. The strap has no scale marks or supplied divisions.',
    'closest_pair': 'Five lettered star targets lie on a clean pale field. Each star is one point at its letter centre; the fins and housing outlines have no geometric extent. Distances are measured between those five centres.',
}


def prompt_paraphrase_description(task):
    return DESCRIPTIONS[task]



def prompt_paraphrase_style(task):
    from data.geometry.core.prompt_paraphrase import prompt_paraphrase_style as style
    return style(task)
