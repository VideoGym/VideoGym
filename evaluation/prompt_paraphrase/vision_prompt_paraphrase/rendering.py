"""Task-owned scene redraws and source-bound geometric view charts."""
from contextlib import contextmanager
from pathlib import Path
import math
import numpy as np
from PIL import Image
from data.registry import get_task_spec
from data.common.schemas import CanonicalSample
from data.common.visual_style import use_style
from data.common.prompt_paraphrase import use_prompt_paraphrase
from evaluation.execution.datasets.generation import _generate
from evaluation.prompt_paraphrase.vision_prompt_paraphrase.plans import MAZES, BOARDS, mapped_sample


@contextmanager
def prompt_paraphrase_context(sample, plan=None):
    plan = plan or {}
    from evaluation.prompt_paraphrase.vision_prompt_paraphrase.new_scenes import new_scene_context
    with use_prompt_paraphrase(plan), use_style(plan.get("skin")), new_scene_context(sample, plan):
        yield



def same_semantics(left, right):
    """Allow only floating-point roundoff, far below a rendered pixel.

    Cross-platform floating-point arithmetic can differ in the last binary
    digits of coordinates. Integer counts, identifiers, topology and text
    remain exact.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_semantics(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(same_semantics(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-9)
    return left == right


def regenerate(sample, root, *, video=False):
    generated = CanonicalSample.from_dict(
        _generate(sample.task_name, {}, sample.seed, root, video=video),
        base_dir=root,
    )
    if not same_semantics(generated.semantic_input, sample.semantic_input):
        raise ValueError(f"regeneration changed the frozen problem: {sample.id}")
    if generated.ground_truth.typed_text != sample.ground_truth.typed_text:
        # A swap puzzle admits several legal programs for the same fixed goal.
        # Validate both against the public source rather than require one trace.
        if sample.task_name != "bookshelf_sort_by_height":
            raise ValueError(f"regeneration changed the source answer: {sample.id}")
        from data.rearrangement_game.core.source_replay import evaluate_text_from_source
        for answer in (sample.ground_truth.typed_text, generated.ground_truth.typed_text):
            if not evaluate_text_from_source(sample, f"Answer: {answer}.")[1]:
                raise ValueError(f"regenerated swap program violates the source goal: {sample.id}")
    return generated


def render_source(sample, root):
    if sample.task_name in MAZES | BOARDS or sample.task_family == "graph_theory":
        return _native_source(sample, root)
    generated = regenerate(sample, root)
    with Image.open(generated.primary_input_image) as image:
        return image.convert("RGB")


def render_presented(sample, plan, root):
    if plan['profile']=='geometry_workbench':
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase.charts import GeometryChart
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase.geometry import render_geometry
        generated = render_geometry(sample, plan, root)
        with Image.open(generated.primary_input_image) as image:
            return Image.fromarray(GeometryChart(sample,plan['quarter_turns']).apply(np.array(image.convert('RGB'))))
    mapped = mapped_sample(sample, plan)
    with prompt_paraphrase_context(mapped, plan):
        image = render_source(mapped, root)
    chart = observation_chart(sample, plan)
    return Image.fromarray(chart.apply(np.array(image))) if chart else image


def observation_chart(sample, plan):
    if plan['profile'] == 'geometry_workbench':
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase.charts import GeometryChart
        return GeometryChart(sample, plan['quarter_turns'])
    from evaluation.prompt_paraphrase.vision_prompt_paraphrase.new_scenes import scene_chart
    return scene_chart(plan)


def render_reference(sample, plan, root):
    from data.common.video_encoding import encode_rgb_frames_to_webm
    from evaluation.shared.task_io import read_video

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if plan['profile']=='geometry_workbench':
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase.charts import GeometryChart
        chart=GeometryChart(sample,plan['quarter_turns'])
        from evaluation.prompt_paraphrase.vision_prompt_paraphrase.geometry import render_geometry
        generated = render_geometry(sample, plan, root / 'native', video=True)
        frames=[chart.apply(frame) for frame in read_video(generated.reference_video)]
        if len(frames)!=sample.video_num_frames:raise ValueError('source reference clock differs')
        destination=root/'reference.webm'
        if not encode_rgb_frames_to_webm(frames,destination,fps=sample.video_fps):raise RuntimeError('reference encoding failed')
        return destination
    mapped = mapped_sample(sample, plan)
    with prompt_paraphrase_context(mapped, plan):
        if sample.task_family == "graph_theory":
            from data.graph_theory.core.replay import replay_unified_graph
            replay = replay_unified_graph(sample.task_name, mapped.semantic_input)
            frames = [np.array(replay.render_input()) for _ in range(5)]
            frames += [np.array(replay.render_progress(i / 31)) for i in range(1, 32)]
            frames += [np.array(replay.render_progress(1)) for _ in range(5)]
        else:
            generated = regenerate(mapped, root / "source", video=True)
            if (generated.video_num_frames, generated.video_fps) == (sample.video_num_frames, sample.video_fps):
                if observation_chart(sample, plan) is None:
                    return Path(generated.reference_video)
                frames = read_video(generated.reference_video)
            else:
                if sample.task_name not in {"binary_matrix", "grid_bfs"} or generated.video_fps != sample.video_fps:
                    raise ValueError(f"reference clock changed for {sample.id}")
                # Fit complete source-reconstructed waves to the frozen clock.
                original = read_video(generated.reference_video)
                indices = np.linspace(0, len(original)-1, sample.video_num_frames).round().astype(int)
                frames = [original[i] for i in indices]
    if len(frames) != sample.video_num_frames:
        raise ValueError("reference frame count differs from source clock")
    chart = observation_chart(sample, plan)
    if chart is not None:
        frames = [chart.apply(frame) for frame in frames]
    destination = root / "reference.webm"
    if not encode_rgb_frames_to_webm(frames, destination, fps=sample.video_fps):
        raise RuntimeError("reference encoding failed")
    return destination


def _native_source(sample, root):
    """Rebuild the entire input from visible entities, without reference pixels."""
    task, source = sample.task_name, sample.semantic_input
    if task in MAZES:
        from data.spatial_navigation.core.path_verifier import _maze_evaluator_class

        evaluator = _maze_evaluator_class(task)(
            dict(source, id=sample.id), base_dir=root
        )
        generator = evaluator._build_generator(source)
        options = dict(
            start=tuple(source["start"]),
            goal=tuple(source["goal"]),
            path=None,
            padding=tuple(source["padding"][:2]),
            canvas_dims=tuple(source["canvas_dimensions"]),
        )
        if task == "maze_keys_and_locks":
            from data.spatial_navigation.maze_keys_and_locks.generator import (
                KEY_SWATCHES,
            )

            indices = {entry[0]: index for index, entry in enumerate(KEY_SWATCHES)}
            locks = {item["color"]: item for item in source["locks"]}
            options["pairs"] = [
                (
                    tuple(key["cell"]),
                    tuple(locks[key["color"]]["cell"]),
                    indices[key["color"]],
                )
                for key in source["keys"]
            ]
        return generator._render_maze(source["maze_grid"], **options)
    if task in BOARDS:
        from data.board_game.core.source_replay import replay_board_sample

        return replay_board_sample(sample, root).initial_image
    if get_task_spec(task).task_family == "graph_theory":
        from data.graph_theory.core.rendering import render_unified_graph

        return render_unified_graph(source)
    raise ValueError(f"no scene-level visual renderer for {task}")
