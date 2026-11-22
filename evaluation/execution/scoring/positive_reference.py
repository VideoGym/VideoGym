"""Source-derived positive controls at unchanged question clocks."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image

_REBUILT = {
    'block_blast_clear_predict',
    'grid_bfs',
    'binary_matrix',
    'square_vertex_from_diagonal',
    'circle_inversion',
    'diagonal_intersection',
    'homothety_center',
}


def _geometry_source_layers(frames, source):
    """Place lossless native program edits onto the supplied givens."""
    if not frames or any(frame.shape != source.shape for frame in frames):
        raise ValueError('positive replay changed the canvas shape')
    baseline = frames[0]
    held = []
    for frame in frames:
        additions = np.any(frame != baseline, axis=2)
        composed = source.copy()
        composed[additions] = frame[additions]
        held.append(composed)
    return held


def qualifying_reference(sample, root, *, base_dir=None):
    """Return the question with a clock-matched positive control video."""
    if sample['task_name'] not in _REBUILT:
        return sample
    from data.common.paths import to_absolute
    from evaluation.execution.datasets.generation import _generate
    from evaluation.shared.task_io import read_video
    from data.common.video_encoding import encode_rgb_frames_to_webm

    root = Path(root)
    config = {}
    if sample['task_name'] == 'circle_inversion':
        from data.geometry.core.verifier import _circle_inversion_prompt_contract
        contract = _circle_inversion_prompt_contract(sample['prompt'])
        if contract['_circle_source_started']:
            config['source_start'] = True
    replay_dir = root / 'replay'
    rebuilt = _generate(sample['task_name'], config, sample['seed'], replay_dir, video=True)
    rebuilt_image = to_absolute(rebuilt['image'], replay_dir)
    source_image_path = to_absolute(sample['image'], base_dir) if base_dir is not None else Path(sample['image'])
    with Image.open(rebuilt_image) as image:
        replay_image = np.array(image.convert('RGB'))
    with Image.open(source_image_path) as image:
        source_image = np.array(image.convert('RGB'))
    if (rebuilt['semantic_input'] != sample['semantic_input']
            or rebuilt['ground_truth'] != sample['ground_truth']
            or replay_image.shape != source_image.shape
            or rebuilt['video_fps'] != sample['video_fps']):
        raise ValueError('positive replay changed the problem')
    geometry = sample['task_family'] == 'geometry'
    same_source = np.array_equal(replay_image, source_image)
    if not geometry and not same_source:
        raise ValueError('positive replay changed the problem')
    frames = read_video(to_absolute(rebuilt['video'], replay_dir))
    if geometry:
        frames = _geometry_source_layers(frames, source_image)
    indices = np.linspace(0, len(frames) - 1, sample['video_num_frames']).round().astype(int)
    reference = root / 'clock-matched-reference.webm'
    if not encode_rgb_frames_to_webm([frames[i] for i in indices], reference, fps=sample['video_fps']):
        raise ValueError('positive replay encoding failed')
    updated = deepcopy(sample)
    updated['video'] = str(reference)
    return updated
