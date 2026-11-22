"""ID-paired Vision Prompt Paraphrase dataset builder and loader."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from data.common.io import matches_schema
from data.common.paths import to_absolute, to_relative
from data.registry import TASK_SPECS
from evaluation.execution.datasets.records import (
    read_rows,
    rebase_sample_media,
    write_json,
)
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.catalog import DESIGNS, validate_catalog
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.plans import (
    PROTOCOL,
    SEED,
    choose_plan,
    mapped_sample,
    sample_from_row,
)
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.prompts import prompt_paraphrase_prompt
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.rendering import (
    observation_chart,
    render_presented,
    render_reference,
    render_source,
)


def _write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def _source_rows(dataset):
    return [r for r in read_rows(Path(dataset) / 'benchmark_manifest.jsonl') if r['track'] == 'id' and r['role'] == 'evaluation']


def _build_case(payload):
    import cv2
    cv2.setNumThreads(1)
    row, source_root, output, seed = payload
    source_root, output = Path(source_root), Path(output)
    sample = sample_from_row(row, base_dir=source_root)
    plan = choose_plan(sample, seed)
    with tempfile.TemporaryDirectory(prefix='videogym-prompt-paraphrase-build-') as tmp:
        root = Path(tmp)
        original = render_source(sample, root / 'source')
        with Image.open(sample.primary_input_image) as im:
            frozen = np.array(im.convert('RGB'))
        if not np.array_equal(np.array(original), frozen):
            raise ValueError(f'source renderer differs from ID image: {sample.id}')
        variant = render_presented(sample, plan, root / 'variant')
        changed = np.any(np.array(variant) != frozen, axis=2)
        if not changed.any():
            raise ValueError(f'drawing adapter did not change the input: {sample.id}')
        image_path = output / 'images' / f'{sample.id}.png'
        variant.save(image_path)
        observation_path = image_path
        chart = observation_chart(sample, plan)
        if chart is not None:
            observation_path = output / 'images' / f'{sample.id}.observation.png'
            Image.fromarray(chart.undo(np.array(variant))).save(observation_path)
        reference = render_reference(sample, plan, root / 'reference')
        target = output / 'references' / f'{sample.id}.webm'
        shutil.copyfile(reference, target)
    mapped = mapped_sample(sample, plan)
    return {
        'id': sample.id,
        'evaluation_task': row['evaluation_task'],
        'scene_group': row['scene_group'],
        'source_image': to_relative(sample.primary_input_image, output),
        'variant_image': to_relative(image_path, output),
        'reference': to_relative(target, output),
        'observation_image': to_relative(observation_path, output),
        'plan': plan,
        'changed_pixels': int(changed.sum()),
        'semantic_input': mapped.semantic_input,
    }


def prepare_visual(dataset, output, *, seed=SEED, workers=1, require_complete_selection=True):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    rows = [r for r in _source_rows(dataset) if r['sample']['task_name'] in DESIGNS]
    ids = [r['sample']['id'] for r in rows]
    tasks = Counter(r['sample']['task_name'] for r in rows)
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('unique ID samples are required')
    if require_complete_selection and (set(tasks) != set(DESIGNS) or set(tasks.values()) != {5}):
        raise ValueError('all 33 visual tasks and their five source samples are required')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Vision Prompt Paraphrase output must be empty')
    for name in ('images', 'references'):
        (output / name).mkdir(parents=True, exist_ok=True)
    work = [(r, str(dataset), str(output), seed) for r in rows]
    pairs_by_id, errors = {}, []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as executor:
        pending = {executor.submit(_build_case, item): item[0]['sample']['id'] for item in work}
        for future in as_completed(pending):
            sid = pending[future]
            try:
                pairs_by_id[sid] = future.result()
                print(f'{len(pairs_by_id)}/{len(rows)} {sid}', flush=True)
            except Exception as error:
                errors.append({'id': sid, 'error': str(error)})
                print(f'FAILED {sid}: {error}', flush=True)
    pairs = [pairs_by_id[sid] for sid in ids if sid in pairs_by_id]
    _write_rows(output / 'pairs.jsonl', pairs)
    if errors:
        write_json(output / 'build-errors.json', errors)
        raise RuntimeError(f'{len(errors)} source pairs failed; inspect build-errors.json')
    return finish_package(dataset, output, seed, rows, pairs, require_complete_selection=require_complete_selection)


def finish_package(dataset, output, seed, rows, pairs, *, require_complete_selection=True):
    """Write manifests, model inputs, and suite metadata for a visual package."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if [p['id'] for p in pairs] != [r['sample']['id'] for r in rows]:
        raise ValueError('every source must appear once in source order')
    tasks = Counter(r['sample']['task_name'] for r in rows)
    if require_complete_selection and (set(tasks) != set(DESIGNS) or set(tasks.values()) != {5}):
        raise ValueError('complete 33-task visual selection is required')
    _write_rows(output / 'pairs.jsonl', pairs)
    rebased_rows = [
        {**r, 'sample': rebase_sample_media(r['sample'], from_dir=dataset, to_dir=output)}
        for r in rows
    ]
    inputs = [
        {
            'id': p['id'],
            'evaluation_task': r['evaluation_task'],
            'prompt': prompt_paraphrase_prompt(sample_from_row(r, base_dir=dataset), p['plan']),
            'image': p['variant_image'],
        }
        for r, p in zip(rows, pairs)
    ]
    timings = [{'id': r['sample']['id'], 'frames': r['sample']['video_num_frames'], 'fps': r['sample']['video_fps']} for r in rows]
    _write_rows(output / 'benchmark_manifest.jsonl', rebased_rows)
    _write_rows(output / 'model_inputs.jsonl', inputs)
    _write_rows(output / 'reference-timing.jsonl', timings)
    report = {
        'prompt_paraphrase_protocol': PROTOCOL,
        'dataset': to_relative(dataset, output),
        'seed': seed,
        'samples': len(rows),
        'tasks': dict(tasks),
        'complete_reviewed_selection': require_complete_selection,
        'selection': {
            'method': 'authored visual designs, proportional family allocation',
            'task_budget': len(DESIGNS),
            'family_quotas': validate_catalog(),
            'id_family_inventory': dict(Counter(t.task_family for t in TASK_SPECS.values())),
        },
        'families': dict(Counter(r['sample']['task_family'] for r in rows)),
    }
    write_json(output / 'visual-suite.json', report)
    return report


def load_visual(dataset):
    root = Path(dataset).resolve()
    record = json.loads((root / 'visual-suite.json').read_text())
    if not matches_schema(record['prompt_paraphrase_protocol'], PROTOCOL):
        raise ValueError('unsupported Vision Prompt Paraphrase package')
    source_root = to_absolute(record['dataset'], root)
    rows = [r for r in _source_rows(source_root) if r['sample']['task_name'] in DESIGNS]
    actual = Counter(r['sample']['task_name'] for r in rows)
    if record.get('complete_reviewed_selection') and (set(actual) != set(DESIGNS) or set(actual.values()) != {5}):
        raise ValueError('visual family allocation is incomplete')
    pairs = read_rows(root / 'pairs.jsonl')
    inputs = read_rows(root / 'model_inputs.jsonl')
    timing = read_rows(root / 'reference-timing.jsonl')
    if not len(rows) == len(pairs) == len(inputs) == len(timing) == record['samples']:
        raise ValueError('Vision Prompt Paraphrase package is incomplete')
    for row, pair, condition, clock in zip(rows, pairs, inputs, timing):
        sample = sample_from_row(row, base_dir=source_root)
        plan = choose_plan(sample, record['seed'])
        turns = pair['plan'].get('quarter_turns')
        if (
            plan['profile'] == 'geometry_workbench'
            and sample.task_name not in {'closest_pair', 'angle_bisector', 'incenter'}
            and type(turns) is int
            and turns in (0, 1, 2, 3)
        ):
            plan = {**plan, 'quarter_turns': turns}
        if (
            pair['id'] != sample.id
            or condition['id'] != sample.id
            or condition['evaluation_task'] != row['evaluation_task']
            or pair['plan'] != plan
            or pair['semantic_input'] != mapped_sample(sample, plan).semantic_input
            or to_absolute(pair['source_image'], root).as_posix() != sample.primary_input_image
            or pair['scene_group'] != row['scene_group']
            or condition['prompt'] != prompt_paraphrase_prompt(sample, plan)
            or condition['image'] != pair['variant_image']
            or clock != {'id': sample.id, 'frames': sample.video_num_frames, 'fps': sample.video_fps}
        ):
            raise ValueError(f'Vision Prompt Paraphrase pairing or semantic mapping changed: {sample.id}')
        if any(not to_absolute(pair[key], root).is_file() for key in ('variant_image', 'observation_image', 'reference')):
            raise ValueError('Vision Prompt Paraphrase asset is missing')
    return record, rows, pairs
