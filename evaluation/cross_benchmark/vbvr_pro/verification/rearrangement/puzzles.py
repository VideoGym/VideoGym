"""Legal block moves and sliding-tile transitions, independent of reference route."""

import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _blobs, _layers, _mask, _result, _static
from evaluation.cross_benchmark.vbvr_pro.vision.masks import mask_coverage


def _allow_rectangle(mask, center, size, fringe=4):
    """Register the physical rectangle and its local observation fringe."""
    half = np.asarray(size) / 2
    x0, y0 = np.rint(np.asarray(center) - half).astype(int) - fringe
    x1, y1 = np.rint(np.asarray(center) + half).astype(int) + fringe + 1
    mask[max(0, y0) : y1, max(0, x0) : x1] = 1


def _stack_pose(column, level, centers, ground, height):
    return np.array([centers[column], ground - (level + 0.5) * height])


def _settled_stacks(positions, centers, ground, height, tolerance):
    stacks = [[] for _ in centers]
    for color, point in positions.items():
        if point is None:
            return None
        column = int(np.argmin(np.abs(centers - point[0])))
        level = round((ground - point[1]) / height - 0.5)
        if level < 0 or np.linalg.norm(point - _stack_pose(column, level, centers, ground, height)) > tolerance:
            return None
        stacks[column].append((level, color))
    result = []
    for stack in stacks:
        ordered = sorted(stack)
        if [level for level, _ in ordered] != list(range(len(ordered))):
            return None
        result.append([color for _, color in ordered])
    return result


def _stack(frames, first, p, final=None):
    layout = p["layout"]
    n = p["num_stacks"]
    palette = p["block_palette_rgb"]
    colors = p["block_colors"]
    if len(set(colors)) != len(colors):
        raise ValueError("Block identity requires distinct palette IDs")
    width, height, ground = layout["block_width"], layout["block_height"], layout["ground_y"]
    centers_x = np.array([(i + 1) * layout["stack_spacing"] for i in range(n)])
    half = first.shape[1] // 2
    state = [list(stack) for stack in p["initial_state"]]
    original_blocks = np.zeros(first.shape[:2], bool)
    for color in colors:
        for center, _, _ in _blobs(_mask(first[:, :half], palette[color], 30), width * height * 0.4):
            _allow_rectangle(original_blocks, center, (width, height), fringe=6)
    # The native renderer replaces its neutral move counter with a green
    # success caption. These optional footer annotations do not decide whether
    # the visible stacks and top-block moves are correct.
    footer = round(first.shape[0] * 0.94)
    background = np.median(first[:8].reshape(-1, 3), axis=0)
    # Vacated initial positions must become background, rather than permanent
    # exceptions where a foreign part or a residual block can hide.
    empty = first.copy()
    empty[original_blocks] = np.rint(background).astype(np.uint8)
    # The fixed stack pedestals begin at ground_y + 5. Preserve their upper
    # antialias row as well; the moving block fringe is not part of the base.
    base_row = min(first.shape[0], round(ground) + 4)
    empty[base_row:] = first[base_row:]
    caption_region = np.zeros(first.shape[:2], bool)
    for template in (first, final if final is not None else first):
        ink = np.max(np.abs(template[footer:].astype(float) - background), axis=2) > 35
        yy, xx = np.nonzero(ink)
        if len(xx):
            caption_region[
                max(footer, footer + yy.min() - 4) : footer + yy.max() + 5, max(0, xx.min() - 4) : xx.max() + 5
            ] = True
    moves, active = [], None
    legal, unique, geometry = True, True, True
    tolerance = height * 0.16
    obscured = 0
    static, worst = True, 0
    for index, frame in enumerate(frames):
        positions = {}
        observations = {}
        allowed = caption_region.copy()
        for color in colors:
            blobs = _blobs(_mask(frame[:, :half], palette[color], 30), width * height * 0.16)
            positions[color] = None
            if not blobs:
                continue
            boxes = np.array([box for _, _, box in blobs])
            origin = boxes[:, :2].min(axis=0)
            end = (boxes[:, :2] + boxes[:, 2:]).max(axis=0)
            extent = end - origin
            if len(blobs) > 1 and np.any(extent > np.array([width, height]) * 1.18):
                unique = False
                continue
            center = (origin + end - 1) / 2
            area = sum(a for _, a, _ in blobs)
            observations[color] = (center, area)
            _allow_rectangle(allowed, center, (width, height), fringe=8)
            # An in-flight block may be partially hidden by another block or
            # blurred over a frame boundary. Only resolve its pose when there
            # is enough paint to read both dimensions of the physical box.
            if np.all(np.abs(extent / (width, height) - 1) <= 0.18) and area >= width * height * 0.50:
                positions[color] = center
        allowed[base_row:] = caption_region[base_row:]
        preserved, error = _static([frame], empty, allowed)
        static &= preserved
        worst = max(worst, error)
        if not legal:
            continue
        if active is not None:
            color, source = active
            point = positions[color]
            if point is not None:
                for target in range(n):
                    level = len(state[target]) - (target == source)
                    if np.linalg.norm(point - _stack_pose(target, level, centers_x, ground, height)) > tolerance:
                        continue
                    if target != source:
                        state[source].pop()
                        state[target].append(color)
                        moves.append([source, target])
                    active = None
                    obscured = 0
                    break
        resting = {
            color: _stack_pose(i, j, centers_x, ground, height)
            for i, stack in enumerate(state)
            for j, color in enumerate(stack)
        }
        displaced = [
            color for color in colors
            if positions[color] is None or np.linalg.norm(positions[color] - resting[color]) > tolerance
        ]
        if active is None and displaced:
            if len(displaced) != 1:
                legal = False
                continue
            color = displaced[0]
            source = next(i for i, stack in enumerate(state) if color in stack)
            if state[source][-1] != color:
                legal = False
                continue
            active = (color, source)
        if active is not None:
            color, _ = active
            moving = observations.get(color)
            for other in displaced:
                if other == color:
                    continue
                observed = observations.get(other)
                # A passing block can cover part of a stationary face. Its
                # visible paint must still lie over the rest position, and
                # the active box must actually overlap that position.
                partially_covered = (
                    moving is not None
                    and observed is not None
                    and observed[1] < width * height * 0.66
                    and np.all(np.abs(observed[0] - resting[other]) <= np.array([width, height]) * 0.25)
                    and np.all(np.abs(moving[0] - resting[other]) < (width, height))
                )
                if not partially_covered:
                    legal = False
            obscured = obscured + 1 if positions[color] is None else 0
            geometry &= obscured <= 3 and (index > 0 or obscured == 0)
    settled = _settled_stacks(positions, centers_x, ground, height, tolerance)
    checks = {
        "every_original_block_exists_exactly_once": unique,
        "only_one_top_block_moves_at_a_time": legal,
        "all_blocks_settled_at_end": settled is not None,
        "final_stack_order_matches_goal": settled == p["target_state"],
        "block_sizes_preserved": geometry,
    }
    checks["right_target_and_board_preserved"] = static
    return _result(
        checks,
        {
            "observed_moves": moves,
            "observed_final_state": settled,
            "unexpected_static_component": worst,
        },
    )


def _glyph(patch):
    # White numeral geometry is carried by luminance; 4:2:0 chroma
    # contamination can lower one RGB channel along an otherwise white stroke.
    white = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) > 205
    margin = int(min(patch.shape[:2]) * 0.15)
    white[:margin] = False
    white[-margin:] = False
    white[:, :margin] = False
    white[:, -margin:] = False
    yy, xx = np.nonzero(white)
    if len(xx) < 5:
        return None
    glyph = white[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1].astype(np.uint8)
    # Retain glyph aspect ratio, so 1 and 7 do not become the same template.
    canvas = np.zeros((64, 64), np.uint8)
    scale = 54 / max(glyph.shape)
    glyph = cv2.resize(
        glyph,
        (max(1, round(glyph.shape[1] * scale)), max(1, round(glyph.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    y, x = (64 - glyph.shape[0]) // 2, (64 - glyph.shape[1]) // 2
    canvas[y : y + glyph.shape[0], x : x + glyph.shape[1]] = glyph
    return canvas.astype(bool)


def _sliding(frames, first, p, metadata):
    n = p["puzzle_size"]
    h, w = first.shape[:2]
    grid = next((layer for layer in _layers(metadata) if layer.get("type") == "grid_lines"), None)
    if grid is None:
        raise ValueError("Sliding board geometry unavailable")
    cell = np.array([grid["cell_size_px"], grid["cell_size_px"]])
    origin = np.array([grid["origin_x_px"], grid["origin_y_px"]])
    centers = np.array([origin + [(c + 0.5) * cell[0], (r + 0.5) * cell[1]] for r in range(n) for c in range(n)])
    templates = {}
    initial = np.array(p["initial_state"]).ravel()
    for index, value in enumerate(initial):
        if value:
            r, c = divmod(index, n)
            x, y = origin + [c * cell[0], r * cell[1]]
            patch = first[round(y) : round(y + cell[1]), round(x) : round(x + cell[0])]
            templates[int(value)] = _glyph(patch)
    if any(value is None for value in templates.values()) or len(templates) != n * n - 1:
        raise ValueError("Cannot read all original tile identities")
    state = initial.copy()
    active = None
    moves = []
    legal = True
    previous = None
    maximum_jump = 0.0
    original_tiles = np.zeros(first.shape[:2], np.uint8)
    fill_templates = {}
    for center, _, box in _blobs(_mask(first, p["tile_rgb"], 38), np.prod(cell) * 0.45):
        x, y, bw, bh = box
        _allow_rectangle(original_tiles, center, cell * p["tile_size_ratio"])
        value = int(initial[np.argmin(np.linalg.norm(centers - center, axis=1))])
        fill_templates[value] = _mask(first[y : y + bh, x : x + bw], p["tile_rgb"], 38).astype(np.uint8)
    empty = first.copy()
    empty[original_tiles.astype(bool)] = np.asarray(p.get("background_rgb", [255, 255, 255])[::-1], np.uint8)
    static, worst = True, 0
    for frame in frames:
        blobs = _blobs(_mask(frame, p["tile_rgb"], 38), np.prod(cell) * 0.45)
        if len(blobs) != n * n - 1:
            return _result({"exact_tile_count_every_frame": False}, {"observed_tile_count": len(blobs)})
        allowed = np.zeros(first.shape[:2], np.uint8)
        for center, _, _ in blobs:
            # Use the declared tile rectangle around its measured centre;
            # the opaque colour core loses several edge pixels under 4:2:0.
            _allow_rectangle(allowed, center, cell * p["tile_size_ratio"])
        preserved, static_error = _static([frame], empty, allowed.astype(bool))
        static &= preserved
        worst = max(worst, static_error)
        positions = {}
        for center, area, box in blobs:
            x, y, bw, bh = box
            if abs(bw - cell[0] * p["tile_size_ratio"]) > 10 or abs(bh - cell[1] * p["tile_size_ratio"]) > 10:
                return _result({"original_tile_dimensions_preserved": False})
            glyph = _glyph(frame[y : y + bh, x : x + bw])
            if glyph is None:
                return _result({"all_tile_digits_readable": False})
            enlarged_glyph = cv2.dilate(glyph.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            scores = []
            for value, ref in templates.items():
                enlarged_ref = cv2.dilate(ref.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                precision = np.count_nonzero(glyph & enlarged_ref) / max(1, np.count_nonzero(glyph))
                recall = np.count_nonzero(ref & enlarged_glyph) / max(1, np.count_nonzero(ref))
                scores.append((min(precision, recall), value))
            scores.sort()
            score, value = scores[-1]
            if score < 0.90 or score - scores[-2][0] < 0.05 or value in positions:
                return _result(
                    {"tile_identities_unique_and_preserved": False},
                    {
                        "glyph_match": float(score),
                        "glyph_margin": float(score - scores[-2][0]),
                        "best_tile": int(value),
                        "second_tile": int(scores[-2][1]),
                        "duplicated": value in positions,
                    },
                )
            expected_fill = fill_templates[value]
            actual_fill = _mask(frame[y : y + bh, x : x + bw], p["tile_rgb"], 38).astype(np.uint8)
            actual_fill = cv2.resize(
                actual_fill, (expected_fill.shape[1], expected_fill.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            if min(mask_coverage(actual_fill, expected_fill, radius=2)) < 0.99:
                return _result({"tile_faces_preserved": False}, {"tile": int(value)})
            positions[value] = center
        if previous is not None:
            maximum_jump = max(maximum_jump, max(np.linalg.norm(positions[v] - previous[v]) for v in positions))
        previous = positions
        indices = {int(value): i for i, value in enumerate(state) if value}
        displaced = [v for v in positions if np.linalg.norm(positions[v] - centers[indices[v]]) > 7]
        blank = int(np.flatnonzero(state == 0)[0])
        if active is None and displaced:
            if len(displaced) != 1:
                legal = False
                break
            value = displaced[0]
            source = indices[value]
            r1, c1 = divmod(source, n)
            r2, c2 = divmod(blank, n)
            if abs(r1 - r2) + abs(c1 - c2) != 1:
                legal = False
                break
            active = (value, source, blank, 0.0)
        if active is not None:
            value, source, target, progress = active
            if any(v != value for v in displaced):
                legal = False
                break
            vector = centers[target] - centers[source]
            alpha = np.dot(positions[value] - centers[source], vector) / np.dot(vector, vector)
            perpendicular = np.linalg.norm(positions[value] - (centers[source] + alpha * vector))
            if perpendicular > 7 or alpha < progress - 0.03 or alpha > 1.03:
                legal = False
                break
            active = (value, source, target, max(progress, float(alpha)))
            if np.linalg.norm(positions[value] - centers[target]) <= 7:
                state[target], state[source] = value, 0
                moves.append([source, target])
                active = None
    return _result(
        {
            "only_adjacent_tile_slides_into_blank": legal,
            "no_unfinished_move": active is None,
            "exact_requested_move_count": len(moves) == p["solution_length"],
            "canonical_goal_reached": state.tolist() == np.array(p["goal_state"]).ravel().tolist(),
            "grid_and_board_preserved_throughout": static,
        },
        {
            "observed_moves": moves,
            "maximum_tile_jump": float(maximum_jump),
            "observed_final_state": state.reshape(n, n).tolist(),
            "unexpected_static_component": worst,
        },
    )


def verify(task, frames, first, final, metadata, reference):
    task_id = task.split("_", 1)[0]
    if task_id == "O-22":
        return _stack(frames, first, metadata["parameters"], final)
    if task_id == "O-47":
        return _sliding(frames, first, metadata["parameters"], metadata)
    return None
