"""Scoped input-card splitting and direct visual delivery for ICL tasks.

The generators in this family own the semantic scenes.  Several of them draw
the source and the candidate cards on one authoring canvas; this module turns
that task-native render into the supplied media contract without changing the
sampled rule, candidates, or typed option answer:

* every task supplies a source-only input image for TI2V, and every terminal
  image and video shows the solved source scene directly.

Input media are never reused as intermediate images.  The composite canvas is an
authoring intermediate only and is not referenced by the returned record.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from data.common.choice_cards import (
    TEXT_CHOICE_CARDS_POLICY,
    VISUAL_CHOICE_CARDS_POLICY,
    attach_text_choice_card_inputs,
    attach_visual_choice_card_inputs,
)
from data.common.pipeline.media import build_complete_footprint_fade_frames, load_font
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.in_context_learning.core.option_delivery import option_panel_boxes


VISUAL_CHOICE_TASKS = frozenset(
    {
        "analogy_completion",
        "object_subtraction",
        "odd_one_out_shape",
        "set_inclusion_diagram",
        "venn_symmetric_difference",
    }
)
TEXT_CHOICE_TASKS = frozenset({"venn_intersection_count"})
DIRECT_CHOICE_TASKS = VISUAL_CHOICE_TASKS | TEXT_CHOICE_TASKS

VISUAL_POLICY = VISUAL_CHOICE_CARDS_POLICY
TEXT_POLICY = TEXT_CHOICE_CARDS_POLICY

_CANVAS_SIZE = (512, 512)
_PLATE_EDGE = (132, 139, 151)
_TEXT = (38, 42, 49)
_ODD_MARK = (24, 145, 116)


def _resolve(output_dir: Path, value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else output_dir / path


def _record_path(output_dir: Path, path: Path, *, like: object) -> str:
    if isinstance(like, str) and Path(like).is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _structured(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("structured_answer")
    return value if isinstance(value, dict) else {}


def _background(image: Image.Image) -> tuple[int, int, int]:
    pixel = image.convert("RGB").getpixel((0, 0))
    return tuple(int(value) for value in pixel)


def _fit_on_canvas(
    image: Image.Image,
    *,
    background: tuple[int, int, int] | None = None,
    margin: int = 20,
) -> Image.Image:
    source = image.convert("RGB")
    fill = background or _background(source)
    available = max(1, 512 - 2 * int(margin))
    scale = min(available / source.width, available / source.height)
    size = (
        max(1, int(round(source.width * scale))),
        max(1, int(round(source.height * scale))),
    )
    resized = source.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", _CANVAS_SIZE, fill)
    canvas.paste(resized, ((512 - size[0]) // 2, (512 - size[1]) // 2))
    return canvas


def _source_view(image: Image.Image, boxes: Sequence[Sequence[int]], *, task: str) -> Image.Image:
    """Return the task-native source scene with option cards absent."""

    source = image.convert("RGB")
    if task in {"object_subtraction", "set_inclusion_diagram"}:
        # This task places its target scene beside the option grid rather than
        # above it, so keep the whole semantic source and wipe only the option
        # quadrant.  The target panel ends at x=256 and remains untouched.
        result = source.copy()
        fill = _background(result)
        left = max(0, min(int(box[0]) for box in boxes) - 10)
        top = max(0, min(int(box[1]) for box in boxes) - 10)
        ImageDraw.Draw(result).rectangle((left, top, 511, 511), fill=fill)
        return result
    if not boxes:
        raise ValueError(f"{task} does not expose candidate-card geometry")
    option_top = min(int(round(float(box[1]))) for box in boxes)
    crop_bottom = max(64, min(512, option_top - 8))
    return _fit_on_canvas(source.crop((0, 0, 512, crop_bottom)), background=_background(source))


def _find_mapping(structured: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    for container in _nested_geometry(structured):
        value = container.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _find_value(structured: Mapping[str, Any], key: str) -> object:
    for container in _nested_geometry(structured):
        if key in container:
            return container[key]
    return None


def _as_box(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(int(round(float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    if not (0 <= box[0] < box[2] <= 512 and 0 <= box[1] < box[3] <= 512):
        return None
    return box


def _target_box(task: str, structured: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    geometry = _find_mapping(structured, "render_geometry") or structured.get("render_geometry")
    if not isinstance(geometry, Mapping):
        geometry = structured
    if task == "analogy_completion":
        grids = geometry.get("source_grid_bboxes")
        return _as_box(grids.get("q")) if isinstance(grids, Mapping) else None
    if task == "object_subtraction":
        return _as_box(geometry.get("target_panel"))
    return None


def _expanded(box: Sequence[int], amount: int = 2) -> tuple[int, int, int, int]:
    return (
        max(0, int(box[0]) - amount),
        max(0, int(box[1]) - amount),
        min(512, int(box[2]) + amount),
        min(512, int(box[3]) + amount),
    )


def _symdiff_pair(structured: Mapping[str, Any]) -> tuple[Image.Image, Image.Image]:
    # Render the actual set operation, not one of the old answer cards.
    from data.in_context_learning.venn_symmetric_difference.generator import _draw_venn_outline

    background = (250, 251, 253)
    box = (46, 70, 466, 442)
    problem = Image.new("RGB", _CANVAS_SIZE, background)
    terminal = problem.copy()
    layout = _find_value(structured, "venn_layout")
    layout = dict(layout) if isinstance(layout, Mapping) else None
    _draw_venn_outline(problem, box, layout=layout)
    correct = _find_value(structured, "correct_mask")
    if not isinstance(correct, (list, tuple)) or len(correct) != 4:
        correct = (0, 1, 0, 1)
    _draw_venn_outline(terminal, box, semantic_mask=[int(value) for value in correct], layout=layout)
    return problem, terminal


def _set_relation_pair(
    puzzle: Image.Image,
    boxes: Sequence[Sequence[int]],
    structured: Mapping[str, Any],
) -> tuple[Image.Image, Image.Image]:
    """Draw the inferred relation from the visible membership rows only.

    The old authored solution copied the selected candidate-card crop into the
    lower half of the canvas.  That made the TI2V target depend on cards which
    are deliberately absent from its input.  Recompute the relation from X/Y
    membership and draw the corresponding task-native diagram directly on the
    source background; candidate order is irrelevant to both images.
    """

    from data.in_context_learning.set_inclusion_diagram.generator import (
        _REL_TO_DIAGRAM,
        _classify_relation,
        _draw_set_diagram,
    )

    problem = _source_view(puzzle, boxes, task="set_inclusion_diagram")
    terminal = problem.copy()
    x_values = _find_value(structured, "x_elements")
    y_values = _find_value(structured, "y_elements")
    if not isinstance(x_values, (list, tuple)) or not isinstance(y_values, (list, tuple)):
        raise ValueError("set_inclusion_diagram is missing visible X/Y memberships")
    relation = _classify_relation(
        [int(value) for value in x_values],
        [int(value) for value in y_values],
    )
    kind = _REL_TO_DIAGRAM[relation]

    # The task renderer works on a supersampled canvas.  Render only the
    # fixed answer region on a temporary canvas so the source rows remain
    # pixel-identical between problem and terminal images.
    from data.in_context_learning.set_inclusion_diagram.generator import _SCALE as _SET_SCALE

    diagram_box = (90, 290, 422, 462)
    background = _background(problem)
    high = Image.new("RGB", (512 * _SET_SCALE, 512 * _SET_SCALE), background)
    _draw_set_diagram(high, diagram_box, kind)
    rendered = high.resize(_CANVAS_SIZE, Image.Resampling.LANCZOS)
    patch = (60, 268, 452, 484)
    terminal.paste(rendered.crop(patch), patch[:2])
    return problem, terminal


def _venn_count_pair(
    structured: Mapping[str, Any],
    boxes: Sequence[Sequence[int]],
) -> tuple[Image.Image, Image.Image]:
    """Re-render the global intersection evidence from semantic dot positions."""

    from data.in_context_learning.venn_intersection_count.generator import VennIntersectionCountPattern

    task_meta = structured.get("task_meta")
    if not isinstance(task_meta, Mapping):
        raise ValueError("venn_intersection_count is missing semantic task metadata")
    values = task_meta.get("candidate_values_by_label")
    positions = task_meta.get("dot_positions")
    if not isinstance(values, Mapping) or not isinstance(positions, Mapping):
        raise ValueError("venn_intersection_count is missing candidates or dot positions")
    labels = ["A", "B", "C", "D"]
    correct_count = len(list(positions.get("ab_both") or []))
    matching = [label for label in labels if int(values[label]) == correct_count]
    if len(matching) != 1:
        raise ValueError("venn_intersection_count source does not identify one text choice")
    correct = matching[0]
    task = {
        "dot_positions": {key: [list(point) for point in value] for key, value in positions.items()},
        "candidates": [int(values[label]) for label in labels],
        "answer_index": labels.index(correct),
    }
    renderer = object.__new__(VennIntersectionCountPattern)
    puzzle = renderer._render_state(task, phase="input", selected=False)
    solution = renderer._render_state(task, phase="global", selected=False)
    return (
        _source_view(puzzle, boxes, task="venn_intersection_count"),
        _source_view(solution, boxes, task="venn_intersection_count"),
    )


def _task_direct_pair(
    puzzle: Image.Image,
    solution: Image.Image,
    boxes: Sequence[Sequence[int]],
    structured: Mapping[str, Any],
    *,
    task: str,
    selected_index: int,
) -> tuple[Image.Image, Image.Image]:
    """Build source/terminal via an explicit adapter for every scoped task."""

    if task == "odd_one_out_shape":
        return _odd_collection(structured), _odd_collection(structured, selected_index=selected_index)
    if task == "venn_symmetric_difference":
        return _symdiff_pair(structured)
    if task == "set_inclusion_diagram":
        return _set_relation_pair(puzzle, boxes, structured)
    if task == "venn_intersection_count":
        return _venn_count_pair(structured, boxes)
    target = _target_box(task, structured)
    if target is None:
        raise ValueError(f"{task} lacks a direct-target adapter")
    solved_composite = puzzle.copy()
    patch = _expanded(target)
    solved_composite.paste(solution.crop(patch), patch[:2])
    return (
        _source_view(puzzle, boxes, task=task),
        _source_view(solved_composite, boxes, task=task),
    )


def _assert_direct_change(task: str, problem: Image.Image, terminal: Image.Image) -> None:
    before = np.asarray(problem.convert("RGB"), dtype=np.int16)
    after = np.asarray(terminal.convert("RGB"), dtype=np.int16)
    delta = np.max(np.abs(after - before), axis=2)
    changed = int(np.count_nonzero(delta > 3))
    if changed < 24:
        raise AssertionError(f"{task} direct terminal is static or semantically empty ({changed} changed pixels)")
    if changed > int(0.48 * delta.size):
        raise AssertionError(f"{task} direct terminal mutates too much of the source scene ({changed} pixels)")


def _nested_geometry(structured: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = [structured]
    for key in ("render_geometry", "task_meta"):
        value = structured.get(key)
        if isinstance(value, Mapping):
            found.append(value)
            nested = value.get("render_geometry")
            if isinstance(nested, Mapping):
                found.append(nested)
    return found


def _content_boxes(structured: Mapping[str, Any], labels: Sequence[str]) -> list[tuple[int, int, int, int]]:
    keys = (
        "candidate_inner_bboxes",
        "candidate_content_boxes",
        "option_grid_bboxes",
        "option_tile_bboxes",
        "option_content_boxes",
    )
    for container in _nested_geometry(structured):
        for key in keys:
            raw = container.get(key)
            if isinstance(raw, Mapping):
                raw = [raw.get(label) for label in labels]
            if not isinstance(raw, (list, tuple)) or len(raw) != len(labels):
                continue
            try:
                boxes = [tuple(int(round(float(v))) for v in item) for item in raw]
            except (TypeError, ValueError):
                continue
            if all(len(box) == 4 and box[0] < box[2] and box[1] < box[3] for box in boxes):
                return boxes
    return []


def _plate_from_crop(
    puzzle: Image.Image,
    panel_box: Sequence[int],
    *,
    label: str,
    content_box: Sequence[int] | None,
    erase_top_left_badge: bool = False,
) -> Image.Image:
    panel = tuple(int(round(float(value))) for value in panel_box)
    if content_box is not None:
        crop_box = tuple(int(round(float(value))) for value in content_box)
    else:
        left, top, right, bottom = panel
        width, height = right - left, bottom - top
        crop_box = (
            left + max(4, int(round(width * 0.06))),
            top + max(4, int(round(height * 0.08))),
            right - max(4, int(round(width * 0.06))),
            bottom - max(12, int(round(height * 0.22))),
        )
    content = puzzle.convert("RGB").crop(crop_box)
    if erase_top_left_badge:
        fill = tuple(int(value) for value in content.getpixel((content.width - 1, content.height - 1)))
        ImageDraw.Draw(content).rectangle(
            (0, 0, max(18, int(content.width * 0.34)), max(18, int(content.height * 0.28))),
            fill=fill,
        )
    # Plate frame and label go through the shared supersampled primitives;
    # the cropped scene content is a raster and is fitted after the finish.
    factor = DOCTRINE_SUPERSAMPLE
    high, sdraw = new_scaled_canvas(512, 512, factor, _background(puzzle))
    sdraw.rounded_rectangle((34, 24, 478, 438), radius=22, fill=(255, 255, 255), outline=_PLATE_EDGE, width=4)
    sdraw.text((256, 466), str(label), font=load_font(34 * factor), fill=_TEXT, anchor="mm")
    canvas = finish_scaled_canvas(high, _CANVAS_SIZE)
    scale = min(376 / max(1, content.width), 334 / max(1, content.height))
    resized = content.resize(
        (
            max(1, int(round(content.width * scale))),
            max(1, int(round(content.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas.paste(resized, ((512 - resized.width) // 2, 220 - resized.height // 2))
    return canvas


# Candidate geometry is drawn through ONE shared scale per view, derived from
# the largest candidate of the task, so a candidate that is genuinely larger
# stays larger in the raster.  Per-shape normalisation would erase exactly the
# size diversity that keeps answer-irrelevant magnitudes from ranking the
# outlier, while a fixed scale would leave every shape small.
_ODD_PLATE_BUDGET = 176.0
_ODD_COLLECTION_BUDGET = 92.0
_ODD_STROKE = 7
# Length of the break in an open boundary, as a fraction of sqrt(area) so it is
# scale-free.  Sized to read unmistakably at 512 square while costing only a
# few percent of boundary ink, so the open candidate cannot be picked out as
# simply the one carrying the least ink.
_ODD_OPEN_GAP = 0.14
_ODD_OPEN_GAP_MAX_EDGE_FRACTION = 0.55


def _polygon_area_2d(points: Sequence[tuple[float, float]]) -> float:
    count = len(points)
    return abs(
        sum(
            points[i][0] * points[(i + 1) % count][1] - points[(i + 1) % count][0] * points[i][1]
            for i in range(count)
        )
    ) / 2.0


def odd_open_gap_fraction(closing_edge: float, area: float) -> float:
    """Share of the closing edge left unpainted, for one open boundary."""

    if closing_edge <= 1e-9:
        return 0.0
    gap = _ODD_OPEN_GAP * math.sqrt(max(area, 1e-9))
    return min(_ODD_OPEN_GAP_MAX_EDGE_FRACTION, gap / closing_edge)


def _odd_vertices(candidate: Mapping[str, Any]) -> list[tuple[float, float]]:
    raw = candidate.get("vertices")
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("odd_one_out_shape candidate geometry is malformed")
    return [(float(item[0]), float(item[1])) for item in raw]


def _odd_shared_scale(candidates: Sequence[Mapping[str, Any]], budget: float) -> float:
    """One scale for every candidate, set by the largest of them."""

    extent = max(
        max(max(abs(x), abs(y)) for x, y in _odd_vertices(candidate))
        for candidate in candidates
    )
    return budget / max(extent, 1e-6)


def _odd_points(
    candidate: Mapping[str, Any],
    center: tuple[float, float],
    scale: float,
) -> list[tuple[float, float]]:
    return [(center[0] + x * scale, center[1] + y * scale) for x, y in _odd_vertices(candidate)]


# Collection geometry in the shipped 512-square coordinate space.  The odd
# generator's self-verifier imports these instead of duplicating them.
_ODD_SLOT_CENTERS: tuple[tuple[float, float], ...] = (
    (136.0, 136.0),
    (376.0, 136.0),
    (136.0, 376.0),
    (376.0, 376.0),
)
_ODD_RING_RADIUS = 104.0


def _draw_odd_candidate(
    sdraw: ScaledDraw,
    candidate: Mapping[str, Any],
    *,
    center: tuple[float, float],
    scale: float,
) -> None:
    """Draw one candidate boundary.

    Every candidate is stroke-only at one uniform width, so closed and open
    boundaries differ solely in whether the outline reconnects.  Filling an
    open boundary would restore the enclosed silhouette and hide the break.
    """

    points = _odd_points(candidate, center, scale)
    if bool(candidate.get("closed")):
        # closed_outline is centred on the vertex ring, so a closed candidate's
        # silhouette matches the open candidate's — which is necessarily an open,
        # centred path.  A stroke masked to the polygon interior would inset every
        # closed candidate by half a width, and that systematic size difference
        # would appear exactly when the open boundary is the answer, handing the
        # model a coarse shortcut for the very signal it must perceive.
        sdraw.closed_outline(points, fill=_TEXT, width=_ODD_STROKE)
        return
    # Re-enter the closing edge from the final vertex and stop short of the
    # first, so the boundary visibly fails to reconnect.  The re-entry is part of
    # the same open path, so the turn at the final vertex is mitered like every
    # other corner instead of being two stroke ends meeting in a notch.
    last, first = points[-1], points[0]
    closing = math.hypot(first[0] - last[0], first[1] - last[1])
    fraction = odd_open_gap_fraction(closing, _polygon_area_2d(points))
    stub = (
        last[0] + (first[0] - last[0]) * (1.0 - fraction),
        last[1] + (first[1] - last[1]) * (1.0 - fraction),
    )
    sdraw.open_path(points + [stub], fill=_TEXT, width=_ODD_STROKE)


def _odd_collection(
    structured: Mapping[str, Any],
    *,
    selected_index: int | None = None,
) -> Image.Image:
    raw_candidates = structured.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 4:
        raise ValueError("odd_one_out_shape requires four structured candidates")
    image, sdraw = new_scaled_canvas(512, 512, DOCTRINE_SUPERSAMPLE, (250, 250, 248))
    scale = _odd_shared_scale(raw_candidates, _ODD_COLLECTION_BUDGET)
    for index, candidate in enumerate(raw_candidates):
        _draw_odd_candidate(sdraw, candidate, center=_ODD_SLOT_CENTERS[index], scale=scale)
    if selected_index is not None:
        cx, cy = _ODD_SLOT_CENTERS[selected_index]
        radius = _ODD_RING_RADIUS
        sdraw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=_ODD_MARK,
            width=8,
        )
    return finish_scaled_canvas(image, _CANVAS_SIZE)


def _odd_plate(structured: Mapping[str, Any], index: int, label: str) -> Image.Image:
    candidates = structured.get("candidates")
    if not isinstance(candidates, list) or index >= len(candidates):
        raise ValueError("odd_one_out_shape candidate list is malformed")
    factor = DOCTRINE_SUPERSAMPLE
    image, sdraw = new_scaled_canvas(512, 512, factor, (250, 250, 248))
    sdraw.rounded_rectangle((34, 24, 478, 438), radius=22, fill=(255, 255, 255), outline=_PLATE_EDGE, width=4)
    _draw_odd_candidate(sdraw, candidates[index], center=(256.0, 228.0),
                        scale=_odd_shared_scale(candidates, _ODD_PLATE_BUDGET))
    sdraw.text((256, 466), label, font=load_font(34 * factor), fill=_TEXT, anchor="mm")
    return finish_scaled_canvas(image, _CANVAS_SIZE)


def _answer_labels(record: Mapping[str, Any], count: int) -> list[str]:
    if count != len(FOUR_CHOICE_CODEC.labels):
        raise ValueError("direct-choice delivery requires exactly four candidates")
    explicit = record.get("options")
    if isinstance(explicit, (list, tuple)):
        labels = [str(value) for value in explicit]
        looks_like_label_list = len(labels) == count and all(
            len(label) == 1 and "A" <= label <= "Z" for label in labels
        )
        if looks_like_label_list and labels != list(FOUR_CHOICE_CODEC.labels):
            raise ValueError("direct-choice option labels must be exactly A, B, C, and D")
    return list(FOUR_CHOICE_CODEC.labels)


def _selected_index(record: Mapping[str, Any], labels: Sequence[str]) -> int:
    answer = str(record.get("answer") or record.get("correct_option") or "").strip().upper()
    if answer in labels:
        return labels.index(answer)
    structured = _structured(record)
    raw = structured.get("correct_index", structured.get("answer_index"))
    if isinstance(raw, int) and raw in range(len(labels)):
        return raw
    raise ValueError(f"{record.get('task_name')} does not expose its selected candidate")


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def _decode_video(path: Path) -> tuple[list[np.ndarray], float]:
    import cv2

    capture = cv2.VideoCapture(path.as_posix())
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 16.0)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, fps


def _direct_transition(start: Image.Image, final: Image.Image, *, task: str) -> list[np.ndarray]:
    """Fade the complete task-local terminal footprint into the source.

    ``task`` remains part of the signature because callers route the task's
    semantic source and terminal through this shared helper.  It must not
    affect pixel order: every changed pixel uses the same opacity at a given
    tick, so no centre, scan direction, or arbitrary solver order is encoded.
    """

    del task
    frames = build_complete_footprint_fade_frames(
        start,
        final,
        hold_frames=5,
        fade_frames=5,
        transition_repeats=2,
        final_hold_frames=10,
    )
    # Wan-compatible 4N+1 length.
    while len(frames) % 4 != 1:
        frames.append(frames[-1].copy())
    return frames


def _rewrite_video(
    original: Path | None,
    target: Path,
    *,
    task: str,
    boxes: Sequence[Sequence[int]],
    start: Image.Image,
    final: Image.Image,
    structured: Mapping[str, Any],
    selected_index: int,
) -> int:
    del boxes, structured, selected_index
    # Candidate cards are absent from TI2V.  Fade exactly the semantic delta
    # between source and task-native solved source at its complete terminal
    # footprint; fixed evidence never participates in the transition.
    frames = _direct_transition(start, final, task=task)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not encode_rgb_frames_to_webm(frames, target, fps=16):
        raise RuntimeError(f"failed to encode direct ICL video: {target}")
    _verify_reference_video(target, start=start, final=final, expected_frames=frames)
    return len(frames)


def _verify_reference_video(
    target: Path,
    *,
    start: Image.Image,
    final: Image.Image,
    expected_frames: Sequence[np.ndarray],
) -> None:
    """Hard-check the authored reference after lossy video encoding."""

    frames, fps = _decode_video(target)
    if not frames or len(frames) % 4 != 1 or abs(float(fps) - 16.0) > 0.25:
        raise AssertionError("direct ICL reference video violates decode/fps/4N+1 contract")
    first = np.asarray(start.convert("RGB"), dtype=np.int16)
    last = np.asarray(final.convert("RGB"), dtype=np.int16)
    if frames[0].shape != first.shape or frames[-1].shape != last.shape:
        raise AssertionError("direct ICL reference video changed the canvas")
    if float(np.mean(np.abs(frames[0].astype(np.int16) - first))) > 5.0:
        raise AssertionError("direct ICL reference video does not start at the supplied source")
    if float(np.mean(np.abs(frames[-1].astype(np.int16) - last))) > 5.0:
        raise AssertionError("direct ICL reference video does not end at the semantic terminal")
    bank: list[np.ndarray] = []
    for expected in expected_frames:
        expected_array = np.asarray(expected, dtype=np.uint8)
        if not bank or not np.array_equal(expected_array, bank[-1]):
            bank.append(expected_array)
    recognized_states: list[int | None] = []
    for frame in frames:
        errors = [
            float(np.mean(np.abs(frame.astype(np.float32) - state.astype(np.float32)))) / 255.0
            for state in bank
        ]
        best = min(range(len(bank)), key=errors.__getitem__)
        recognized_states.append(best if errors[best] <= 0.045 else None)
    recognized = [value for value in recognized_states if value is not None]
    if len(recognized) / len(frames) < 0.9:
        raise AssertionError("direct ICL reference video leaves its task-owned local-fade state bank")
    transitions = [
        (left, right)
        for left, right in zip(recognized_states, recognized_states[1:])
        if left is not None and right is not None
    ]
    if any(right < left for left, right in transitions):
        raise AssertionError("direct ICL reference video reverses its task-owned element fade")
    interior_states = {value for value in recognized if 0 < value < len(bank) - 1}
    if len(interior_states) < 2:
        raise AssertionError("direct ICL reference video hard-commits instead of visibly fading")
    direction = last - first
    active = np.max(np.abs(direction), axis=2) > 5
    denominator = float(np.sum(direction[active].astype(np.float64) ** 2))
    if denominator <= 1.0:
        raise AssertionError("direct ICL reference video has no load-bearing target change")
    progress: list[float] = []
    for frame in frames:
        delta = (frame.astype(np.int16) - first).astype(np.float64)
        value = float(np.sum(delta[active] * direction[active].astype(np.float64)) / denominator)
        progress.append(max(0.0, min(1.0, value)))
    if any(right + 0.035 < left for left, right in zip(progress, progress[1:])):
        raise AssertionError("direct ICL reference video regresses during the answer-element fade")
    if max(progress[: min(5, len(progress))]) > 0.08 or min(progress[-5:]) < 0.88:
        raise AssertionError("direct ICL reference video lacks readable initial or final holds")
    transition_bins = {round(value, 1) for value in progress if 0.08 < value < 0.92}
    if len(transition_bins) < 3:
        raise AssertionError("direct ICL reference video lacks several complete-footprint fade states")


def _clear_reasoning_media(record: dict[str, Any], terminal_value: str) -> None:
    record.pop("reasoning_image", None)
    record["reasoning_images"] = []


def _synchronize_structured_delivery_contract(structured: dict[str, Any]) -> None:
    """Normalize structured delivery metadata for video evaluation."""

    containers = [structured]
    task_meta = structured.get("task_meta")
    if isinstance(task_meta, dict):
        containers.append(task_meta)
    for container in containers:
        supervision = container.get("intermediate_supervision_schema")
        if isinstance(supervision, dict):
            supervision.pop("ti2ti", None)
            supervision["ti2v"] = ["input_hold", "complete_footprint_element_fade", "final_hold"]
            container["intermediate_supervision_schema"] = supervision
        verifier = container.get("verifier_contract")
        if not isinstance(verifier, dict):
            verifier = {}
        verifier["profile"] = "icl_direct_choice_source_replay_final_delivery_v1"
        verifier["component_scores"] = {
            "ti2v": [
                "source_replay_score",
                "media_validity_score",
                "task_owned_element_fade_score",
                "complete_footprint_fade_score",
                "preservation_score",
                "final_state_score",
            ],
        }
        verifier["hard_gates"] = [
            "strict_parse",
            "independent_source_replay",
            "candidate_uniqueness",
            "semantic_direct_terminal",
            "whole_video_task_owned_complete_footprint_fade",
            "hard_commit_rejection",
            "fixed_content_preservation",
            "media_contract",
        ]
        verifier["binary_predicate"] = (
            "the modality's formal final delivery passes every independent "
            "source-replay and media hard gate"
        )
        container["verifier_contract"] = verifier


def apply_direct_choice_delivery(record: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Mutate one ICL choice record to the scoped direct-delivery contract."""

    task = str(record.get("task_name") or "")
    if task not in DIRECT_CHOICE_TASKS:
        return record
    structured = _structured(record)
    boxes = option_panel_boxes(structured)
    if len(boxes) != len(FOUR_CHOICE_CODEC.labels):
        raise ValueError(f"{task} must expose exactly four real candidate cards")
    labels = _answer_labels(record, len(boxes))
    selected_index = _selected_index(record, labels)
    puzzle_value = record.get("image")
    solution_value = record.get("solution_image_path")
    puzzle_path = _resolve(output_dir, puzzle_value)
    solution_path = _resolve(output_dir, solution_value)
    if puzzle_path is None or solution_path is None or not puzzle_path.is_file() or not solution_path.is_file():
        raise ValueError(f"{task} does not expose task-native puzzle and solution images")
    with Image.open(puzzle_path) as opened:
        puzzle = opened.convert("RGB").copy()
    with Image.open(solution_path) as opened:
        solution = opened.convert("RGB").copy()

    stem = puzzle_path.stem.removesuffix("__puzzle")
    asset_dir = puzzle_path.parent
    problem_path = asset_dir / f"{stem}__problem.png"
    terminal_path = asset_dir / f"{stem}__terminal.png"
    problem, terminal = _task_direct_pair(
        puzzle,
        solution,
        boxes,
        structured,
        task=task,
        selected_index=selected_index,
    )
    _assert_direct_change(task, problem, terminal)
    _save(problem, problem_path)
    _save(terminal, terminal_path)

    problem_value = _record_path(output_dir, problem_path, like=puzzle_value)
    terminal_value = _record_path(output_dir, terminal_path, like=solution_value)
    record["image"] = problem_value
    record["solution_image_path"] = terminal_value
    record["option_labels"] = labels

    option_values: list[str] = []
    if task in VISUAL_CHOICE_TASKS:
        content_boxes = _content_boxes(structured, labels)
        for index, (label, box) in enumerate(zip(labels, boxes)):
            plate_path = asset_dir / f"{stem}__option_{label}.png"
            if task == "odd_one_out_shape":
                plate = _odd_plate(structured, index, label)
            else:
                inner = content_boxes[index] if len(content_boxes) == len(labels) else None
                plate = _plate_from_crop(
                    puzzle,
                    box,
                    label=label,
                    content_box=inner,
                    erase_top_left_badge=task
                    in {"object_subtraction", "set_inclusion_diagram"},
                )
            _save(plate, plate_path)
            option_values.append(_record_path(output_dir, plate_path, like=puzzle_value))
        attach_visual_choice_card_inputs(
            record,
            problem_image=problem_value,
            option_images=option_values,
            option_labels=labels,
        )
        record["option_images"] = option_values
        text_inputs = [problem_value, *option_values]
    else:
        attach_text_choice_card_inputs(
            record,
            problem_image=problem_value,
            option_labels=labels,
        )
        record["option_images"] = []
        text_inputs = [problem_value]

    record["input_images"] = list(text_inputs)
    _clear_reasoning_media(record, terminal_value)
    _synchronize_structured_delivery_contract(structured)

    video_value = record.get("solution_video_path") or record.get("video")
    original_video = _resolve(output_dir, video_value)
    if original_video is not None and original_video.is_file():
        video_path = asset_dir / f"{stem}__direct.webm"
        frame_count = _rewrite_video(
            original_video,
            video_path,
            task=task,
            boxes=boxes,
            start=problem,
            final=terminal,
            structured=structured,
            selected_index=selected_index,
        )
        record["solution_video_path"] = _record_path(
            output_dir,
            video_path,
            like=video_value or solution_value,
        )
        record["video_fps"] = 16
        record["video_num_frames"] = frame_count
    else:
        record["solution_video_path"] = None
        record["video_fps"] = None
        record["video_num_frames"] = None

    # The composite authoring intermediates are no longer referenced by the
    # record; remove them so no orphan media reaches the export tree (13.1).
    final_assets = {problem_path, terminal_path}
    for stale in (puzzle_path, solution_path, original_video):
        if stale is not None and stale.is_file() and stale not in final_assets:
            stale.unlink()

    return record


__all__ = [
    "DIRECT_CHOICE_TASKS",
    "TEXT_CHOICE_TASKS",
    "TEXT_POLICY",
    "VISUAL_CHOICE_TASKS",
    "VISUAL_POLICY",
    "apply_direct_choice_delivery",
]
