"""Shared scaffolding for path-task generation, rendering, and evaluation.

These base classes implement reusable logic for visual paths where a solver must
trace a red path from a designated start to a goal while avoiding walls drawn in
black. Subclasses can focus on world layout generation and rendering while
inheriting dataset serialization, CLI wiring, and pixel-level evaluation.
"""

from __future__ import annotations

import math
import random
import re
import uuid
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from data.common.prompts import build_sectioned_prompt
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.prompts import build_path_prompt
from data.common.prompts import PATH_PUZZLE_INPUT_DESCRIPTION
from data.common.prompts import PATH_REASONING_INPUT_DESCRIPTION
from data.common.prompts import build_still_image_prompt_from_video_description
from data.common.prompts import soften_precise_video_timing
from data.common.prompts import split_visual_description_and_output
from data.common.typed_answers import format_answer_line
from data.common.base import AbstractPuzzleEvaluator, AbstractPuzzleGenerator, PathLike
from data.common.grid_coordinates import (
    GRID_PATH_ANSWER_FORMAT_INSTRUCTION,
    GRID_PATH_ANSWER_SCHEMA,
    GRID_XY_CONVENTION_SENTENCE,
    STRICT_XY_PATH_ANSWER_RE,
    cell_to_xy,
    format_xy_path_payload,
    format_xy_positions_payload,
    parse_xy_path_answer,
    xy_to_cell,
)
from data.common.pipeline.media import load_font
from data.common.render_primitives import DOCTRINE_SUPERSAMPLE, ScaledDraw
from data.common.video_encoding import (
    MULTISTEP_VIDEO_FRAMES,
    encode_rgb_frames_to_webm,
    fit_frames_to_video_budget,
    pad_frames_to_wan_compatible_length,
)


def _strip_video_instruction(prompt: Optional[str]) -> Optional[str]:
    if prompt is None:
        return None
    stripped = build_still_image_prompt_from_video_description(prompt)
    return stripped or prompt.strip()


# Royal blue keeps the axis numbers legible on the light gutter that the maze
# family renders outside the grid body, while staying inside the blue range
# the terminal-image verifier uses to hard-gate label preservation.
GRID_AXIS_LABEL_COLOR = (30, 90, 220)
GRID_AXIS_LABEL_GUTTER = 44


def grid_axis_label_layout(
    rows: int,
    cols: int,
    *,
    canvas_size: int = 512,
    gutter: int = GRID_AXIS_LABEL_GUTTER,
    min_margin: int = 4,
    cell_size: Optional[int] = None,
) -> Tuple[int, int, int, int, int]:
    """Size a square grid that reserves a left/top gutter for 1-based axis labels.

    Returns ``(cell_size, pad_left, pad_top, pad_right, pad_bottom)``.  The
    gutter holds one row number per row down the left edge and one column
    number per column across the top edge, so the grid body itself carries no
    per-cell text.  Passing ``cell_size`` reproduces an earlier layout (an
    evaluator rehydrating a persisted record) instead of recomputing it.
    """

    rows = int(rows)
    cols = int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if cell_size is None:
        usable = int(canvas_size) - int(gutter) - int(min_margin)
        cell_size = max(4, usable // max(rows, cols))
    else:
        cell_size = int(cell_size)
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
    extra_x = int(canvas_size) - cols * cell_size
    extra_y = int(canvas_size) - rows * cell_size
    if extra_x < gutter or extra_y < gutter:
        raise ValueError("canvas is too small to reserve an axis-label gutter for this grid")
    pad_left = int(gutter) + (extra_x - int(gutter)) // 2
    pad_top = int(gutter) + (extra_y - int(gutter)) // 2
    return cell_size, pad_left, pad_top, extra_x - pad_left, extra_y - pad_top


def draw_grid_axis_labels(
    draw: ImageDraw.ImageDraw,
    *,
    rows: int,
    cols: int,
    cell_size: int,
    pad_left: int,
    pad_top: int,
    color: Tuple[int, int, int] = GRID_AXIS_LABEL_COLOR,
) -> None:
    """Draw 1-based row numbers down the left gutter and column numbers on top.

    The left gutter carries ``x`` (row, top to bottom) and the top gutter
    carries ``y`` (column, left to right), matching the model-facing ``(x, y)``
    answer grammar.  Nothing is drawn inside the grid body.  Passing a
    supersampling ``ScaledDraw`` proxy scales the glyph size automatically.
    """

    from data.common.visual_style import current_style
    if style := current_style():
        if style.name == "machined":
            color = (109, 164, 255)
    factor = int(getattr(draw, "factor", 1) or 1)
    font = load_font(max(9, int(cell_size * 0.42)) * factor)
    for row in range(int(rows)):
        center_y = pad_top + row * cell_size + cell_size / 2.0
        draw.text(
            (pad_left / 2.0, center_y),
            str(row + 1),
            fill=color,
            anchor="mm",
            font=font,
        )
    for col in range(int(cols)):
        center_x = pad_left + col * cell_size + cell_size / 2.0
        draw.text(
            (center_x, pad_top / 2.0),
            str(col + 1),
            fill=color,
            anchor="mm",
            font=font,
        )


_RASTER_PATH_SCALE = 4
_RESAMPLING_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def _draw_path_segment_rect(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Any,
    thickness: int,
) -> None:
    """Draw a path segment as a filled rectangle."""
    x1, y1 = p1
    x2, y2 = p2
    half = thickness / 2

    if math.isclose(x1, x2, abs_tol=1e-6) and math.isclose(y1, y2, abs_tol=1e-6):
        draw.ellipse((x1 - half, y1 - half, x1 + half, y1 + half), fill=color)
        return

    if math.isclose(x1, x2, abs_tol=1e-6):
        left = x1 - half
        top = min(y1, y2)
        right = x1 + half
        bottom = max(y1, y2)
        draw.rectangle((left, top, right, bottom), fill=color)
        return

    if math.isclose(y1, y2, abs_tol=1e-6):
        left = min(x1, x2)
        top = y1 - half
        right = max(x1, x2)
        bottom = y1 + half
        draw.rectangle((left, top, right, bottom), fill=color)
        return

    dx = x2 - x1
    dy = y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len <= 1e-6:
        draw.ellipse((x1 - half, y1 - half, x1 + half, y1 + half), fill=color)
        return

    # Other maze variants in this shared base use angled paths, so keep them
    # artifact-free with an oriented rectangle instead of a stroked line.
    ux = dx / seg_len
    uy = dy / seg_len
    px = -uy * half
    py = ux * half
    start_x = x1
    start_y = y1
    end_x = x2
    end_y = y2
    draw.polygon(
        (
            (start_x + px, start_y + py),
            (start_x - px, start_y - py),
            (end_x - px, end_y - py),
            (end_x + px, end_y + py),
        ),
        fill=color,
    )


def _cross_2d(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Return the 2D cross product magnitude."""
    return v1[0] * v2[1] - v1[1] * v2[0]


def _line_intersection(
    p1: Tuple[float, float],
    d1: Tuple[float, float],
    p2: Tuple[float, float],
    d2: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Return the intersection of two infinite 2D lines."""
    denom = _cross_2d(d1, d2)
    if math.isclose(denom, 0.0, abs_tol=1e-9):
        return None

    delta = (p2[0] - p1[0], p2[1] - p1[1])
    t = _cross_2d(delta, d2) / denom
    return (p1[0] + d1[0] * t, p1[1] + d1[1] * t)


def _draw_path_joint(
    draw: ImageDraw.ImageDraw,
    point: Tuple[float, float],
    color: Any,
    thickness: int,
) -> None:
    """Draw a round joint or cap for a path point."""
    half = thickness / 2
    px, py = point
    draw.ellipse((px - half, py - half, px + half, py + half), fill=color)


def _draw_path_square_cap(
    draw: ImageDraw.ImageDraw,
    point: Tuple[float, float],
    color: Any,
    thickness: int,
) -> None:
    """Draw a square cap for square-grid maze paths."""
    half = thickness / 2
    px, py = point
    draw.rectangle((px - half, py - half, px + half, py + half), fill=color)


def pawn_marker_radius(thickness: float) -> float:
    """Outer radius of the tip marker :func:`_draw_pawn_marker` paints.

    Exposed so a verifier can declare the region a route video is allowed to
    change without restating the geometry: the marker is a disc centred on a
    point of the route polyline, so the swept region is the stroke dilated by
    this radius.  One definition, used by both the renderer and the gate.
    """

    return max(5.5, float(thickness) * 0.95)


def _draw_pawn_marker(
    draw: ImageDraw.ImageDraw,
    point: Tuple[float, float],
    thickness: int,
    color: Any,
) -> None:
    """Draw a persistent tip marker for the animating path.

    Without this, a revisiting route (gem_hunt / checkpoint_order / keys_and_locks
    after picking up a key, etc.) plays back as a static frame during the
    revisit segments — the polyline is already painted, so nothing moves.
    The pawn is drawn on top of every animation frame at the current tip so
    the viewer can always see "where the agent is right now", even when it is
    retracing its steps. White outer ring + filled colored inner disc;
    radii chosen so the pawn pops above the stroke but does not cover
    neighbouring cells.
    """
    px, py = point
    outer_r = pawn_marker_radius(thickness)
    inner_r = max(3.5, thickness * 0.60)
    draw.ellipse(
        (px - outer_r, py - outer_r, px + outer_r, py + outer_r),
        fill=(255, 255, 255),
    )
    draw.ellipse(
        (px - inner_r, py - inner_r, px + inner_r, py + inner_r),
        fill=color,
    )


def _build_miter_polygon(
    points: Sequence[Tuple[float, float]],
    half: float,
) -> Optional[List[Tuple[float, float]]]:
    """Build a single filled polygon for the entire thick path with miter joins.

    Returns left_edge + reversed(right_edge) as one polygon outline.
    This produces perfect sharp corners with zero seams.
    """
    n = len(points)
    if n < 2:
        return None

    left_edge: List[Tuple[float, float]] = []
    right_edge: List[Tuple[float, float]] = []

    # Compute unit directions and normals for each segment
    dirs: List[Tuple[float, float]] = []
    normals: List[Tuple[float, float]] = []
    for i in range(n - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            dirs.append((1.0, 0.0))
            normals.append((0.0, 1.0))
        else:
            ux, uy = dx / seg_len, dy / seg_len
            dirs.append((ux, uy))
            normals.append((-uy, ux))

    miter_limit = half * 4.0

    for i in range(n):
        if i == 0:
            # Start cap: perpendicular to first segment
            nx, ny = normals[0]
            left_edge.append((points[0][0] + nx * half, points[0][1] + ny * half))
            right_edge.append((points[0][0] - nx * half, points[0][1] - ny * half))
        elif i == n - 1:
            # End cap: perpendicular to last segment
            nx, ny = normals[-1]
            left_edge.append((points[i][0] + nx * half, points[i][1] + ny * half))
            right_edge.append((points[i][0] - nx * half, points[i][1] - ny * half))
        else:
            # Interior point: compute miter
            n1x, n1y = normals[i - 1]
            n2x, n2y = normals[i]
            d1 = dirs[i - 1]
            d2 = dirs[i]

            # Left side miter
            lp1 = (points[i][0] + n1x * half, points[i][1] + n1y * half)
            lp2 = (points[i][0] + n2x * half, points[i][1] + n2y * half)
            left_miter = _line_intersection(lp1, d1, lp2, d2)

            # Right side miter
            rp1 = (points[i][0] - n1x * half, points[i][1] - n1y * half)
            rp2 = (points[i][0] - n2x * half, points[i][1] - n2y * half)
            right_miter = _line_intersection(rp1, d1, rp2, d2)

            # Check miter limit
            bx, by = points[i]
            if left_miter and math.hypot(left_miter[0] - bx, left_miter[1] - by) <= miter_limit:
                left_edge.append(left_miter)
            else:
                left_edge.append(lp1)
                left_edge.append(lp2)
            if right_miter and math.hypot(right_miter[0] - bx, right_miter[1] - by) <= miter_limit:
                right_edge.append(right_miter)
            else:
                right_edge.append(rp1)
                right_edge.append(rp2)

    return left_edge + list(reversed(right_edge))


def _draw_segment_extended(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Any,
    thickness: int,
) -> None:
    """Draw a path segment with both ends extended by half the line width.

    The extension makes adjacent segments overlap at corners, completely
    filling any triangular gap without needing explicit joints.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    seg_len = math.hypot(dx, dy)
    if seg_len <= 1e-6:
        _draw_path_segment_rect(draw, p1, p2, color, thickness)
        return
    ext = thickness / 2.0
    ux = dx / seg_len
    uy = dy / seg_len
    p1_ext = (p1[0] - ux * ext, p1[1] - uy * ext)
    p2_ext = (p2[0] + ux * ext, p2[1] + uy * ext)
    _draw_path_segment_rect(draw, p1_ext, p2_ext, color, thickness)


def _paint_capsule_mask(
    mask: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    radius: float,
) -> None:
    """OR a round-capped segment into a boolean mask."""
    if mask.size == 0 or radius <= 0:
        return

    height, width = mask.shape
    x1, y1 = p1
    x2, y2 = p2

    min_x = max(0, int(math.floor(min(x1, x2) - radius - 1.0)))
    max_x = min(width - 1, int(math.ceil(max(x1, x2) + radius + 1.0)))
    min_y = max(0, int(math.floor(min(y1, y2) - radius - 1.0)))
    max_y = min(height - 1, int(math.ceil(max(y1, y2) + radius + 1.0)))
    if min_x > max_x or min_y > max_y:
        return

    ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
    xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)

    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        dist_sq = (xx - x1) ** 2 + (yy - y1) ** 2
    else:
        t = ((xx - x1) * dx + (yy - y1) * dy) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist_sq = (xx - proj_x) ** 2 + (yy - proj_y) ** 2

    region = dist_sq <= radius * radius
    mask[min_y : max_y + 1, min_x : max_x + 1] |= region


def _accumulate_polyline_mask(
    mask: np.ndarray,
    points: Sequence[Tuple[float, float]],
    thickness: int,
    *,
    scale: int,
) -> None:
    """Rasterize a full polyline into a boolean mask as a union of capsules."""
    if not points:
        return

    radius = max(0.5, (thickness * scale) / 2.0)
    scaled = [(float(x) * scale, float(y) * scale) for x, y in points]
    if len(scaled) == 1:
        _paint_capsule_mask(mask, scaled[0], scaled[0], radius)
        return

    for start, end in zip(scaled, scaled[1:]):
        _paint_capsule_mask(mask, start, end, radius)


def _polyline_alpha_mask(
    size: Tuple[int, int],
    points: Sequence[Tuple[float, float]],
    thickness: int,
    *,
    scale: int = _RASTER_PATH_SCALE,
) -> Image.Image:
    """Return an antialiased alpha mask for a full polyline."""
    if not points:
        return Image.new("L", size, 0)

    scaled_width = max(1, int(size[0]) * scale)
    scaled_height = max(1, int(size[1]) * scale)
    mask = np.zeros((scaled_height, scaled_width), dtype=bool)
    _accumulate_polyline_mask(mask, points, thickness, scale=scale)
    alpha = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if scale != 1:
        alpha = alpha.resize(size, _RESAMPLING_LANCZOS)
        # Binarize after downscale so the path colour is exact (220, 30, 30)
        # with no semi-transparent blending at edges.
        alpha = alpha.point(lambda v: 255 if v >= 128 else 0)
    return alpha


def _subpaths_alpha_mask(
    size: Tuple[int, int],
    subpaths: Sequence[Sequence[Tuple[float, float]]],
    thickness: int,
    *,
    scale: int = _RASTER_PATH_SCALE,
) -> Image.Image:
    """Return an antialiased alpha mask for one or more polyline subpaths."""
    alpha = Image.new("L", size, 0)
    for subpath in subpaths:
        if not subpath:
            continue
        alpha = ImageChops.lighter(alpha, _polyline_alpha_mask(size, subpath, thickness, scale=scale))
    return alpha


def _apply_alpha_overlay(
    image: Image.Image,
    alpha: Image.Image,
    color: Tuple[int, int, int],
) -> None:
    """Composite a solid-colour overlay onto *image* using the provided alpha."""
    if alpha.getbbox() is None:
        return
    overlay = Image.new("RGBA", image.size, color + (0,))
    overlay.putalpha(alpha)
    composite = Image.alpha_composite(image.convert("RGBA"), overlay)
    image.paste(composite.convert("RGB"))


def _video_stable_path_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Pass through the original colour unchanged."""
    return (int(color[0]), int(color[1]), int(color[2]))


def _recolor_red_dominant_pixels(
    image: Image.Image,
    target_color: Tuple[int, int, int],
) -> Image.Image:
    """Return a copy with red-dominant guide pixels remapped to *target_color*."""
    array = np.array(image.convert("RGB"), dtype=np.uint8)
    mask = (array[:, :, 0] > 180) & (array[:, :, 1] < 100) & (array[:, :, 2] < 100)
    if not mask.any():
        return image.copy()
    array[mask] = np.array(target_color, dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def _accumulate_subpath_interval_mask(
    mask: np.ndarray,
    info: Dict[str, Any],
    start_dist: float,
    end_dist: float,
    thickness: int,
) -> None:
    """Accumulate the stroke segment covering [start_dist, end_dist] within one subpath."""
    if end_dist <= start_dist + 1e-6:
        return

    points: Sequence[Tuple[float, float]] = info["points"]
    cumulative: Sequence[float] = info["cum"]
    if not points:
        return
    if len(points) == 1:
        if end_dist > 0.0:
            _paint_capsule_mask(mask, points[0], points[0], max(0.5, thickness / 2.0))
        return

    prev_cum = 0.0
    radius = max(0.5, thickness / 2.0)
    for index, seg_end in enumerate(cumulative):
        seg_start = prev_cum
        prev_cum = seg_end
        overlap_start = max(start_dist, seg_start)
        overlap_end = min(end_dist, seg_end)
        if overlap_end <= overlap_start + 1e-6:
            continue

        segment_length = seg_end - seg_start
        start_ratio = 0.0 if segment_length <= 1e-9 else (overlap_start - seg_start) / segment_length
        end_ratio = 1.0 if segment_length <= 1e-9 else (overlap_end - seg_start) / segment_length
        start_ratio = max(0.0, min(1.0, start_ratio))
        end_ratio = max(0.0, min(1.0, end_ratio))

        p1 = points[index]
        p2 = points[index + 1]
        clipped_start = (
            p1[0] + (p2[0] - p1[0]) * start_ratio,
            p1[1] + (p2[1] - p1[1]) * start_ratio,
        )
        clipped_end = (
            p1[0] + (p2[0] - p1[0]) * end_ratio,
            p1[1] + (p2[1] - p1[1]) * end_ratio,
        )
        _paint_capsule_mask(mask, clipped_start, clipped_end, radius)


def _draw_path_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    color: Any,
    thickness: int,
    *,
    joint_style: str,
) -> None:
    """Draw a polyline with the shared segment and joint behaviour."""
    if joint_style == "stable_round":
        image = getattr(draw, "_image", None)
        if image is None:
            raise RuntimeError("stable_round path rendering requires the destination image")
        alpha = _polyline_alpha_mask(image.size, points, thickness)
        _apply_alpha_overlay(image, alpha, color)
        return
    if len(points) >= 2:
        if joint_style == "miter":
            polygon = _build_miter_polygon(points, thickness / 2.0)
            if polygon:
                draw.polygon(polygon, fill=color)
            return
        if joint_style == "extended":
            # Each segment extended by 1px at both ends — fills corner gaps
            # without any joints, no polygon instability, perfectly stable per-frame
            for i in range(len(points) - 1):
                _draw_segment_extended(draw, points[i], points[i + 1], color, thickness)
            return
        for i in range(len(points) - 1):
            _draw_path_segment_rect(draw, points[i], points[i + 1], color, thickness)
        if joint_style == "round":
            for point in points:
                _draw_path_joint(draw, point, color, thickness)
        elif joint_style in ("corners", "none"):
            pass
        else:
            # joint_style == "square": cap only INTERIOR vertices. Skipping the
            # first and last points keeps the stroke's terminal face a clean
            # rectangular cut; painting a cap there would extend the stroke
            # thickness/2 px past the endpoint (axis-aligned cap rect centred
            # on the vertex), which for single-subpath mazes (maze_square and
            # friends) is harmless because red start/goal cells paint over
            # that overhang, but for maze_layered's per-subpath ends the
            # overhang lands in the middle of yellow staircase cells and
            # leaves a visible red square stuck on the stair. Interior caps
            # are still needed to fill the L-corner at every turn — without
            # them each bend loses a half-thickness quadrant.
            for point in points[1:-1]:
                _draw_path_square_cap(draw, point, color, thickness)
    elif len(points) == 1:
        if joint_style == "round":
            _draw_path_joint(draw, points[0], color, thickness)
        elif joint_style not in ("none", "corners", "miter", "extended", "stable_round"):
            _draw_path_square_cap(draw, points[0], color, thickness)


def draw_path_line(
    image: Image.Image,
    points: List[Tuple[float, float]],
    color: Tuple[int, int, int],
    thickness: int,
    *,
    joint_style: str = "round",
    tip_marker: bool = True,
    factor: int = 1,
) -> None:
    """Draws a path (solution line) on the given image.

    When tip_marker is True (default) a pawn marker (white-ringed colored disc)
    is painted at the last point so static solution images share the same
    "agent stopped here" visual that the animated video shows in its final
    frame. Pass tip_marker=False on internal helpers that draw partial paths
    and don't want the agent marker.

    ``factor`` > 1 declares that *image* is a supersampled scene canvas: the
    stroke then goes through the shared :class:`ScaledDraw` proxy, so callers
    keep passing target-space points and thickness and the whole route is
    resolved at scene scale before the scene's single LANCZOS pass.
    """
    if joint_style == "stable_round":
        if factor > 1:
            raise ValueError("stable_round path rendering owns its own raster scale")
        alpha = _polyline_alpha_mask(image.size, points, thickness)
        _apply_alpha_overlay(image, alpha, color)
        draw = ImageDraw.Draw(image)
    else:
        draw = ImageDraw.Draw(image)
        if factor > 1:
            draw = ScaledDraw(draw, factor)
        _draw_path_polyline(draw, points, color, thickness, joint_style=joint_style)
    if tip_marker and points:
        _draw_pawn_marker(draw, points[-1], thickness, color)


def composite_supersampled_route(
    image: Image.Image,
    subpaths: Sequence[Sequence[Tuple[float, float]]],
    color: Tuple[int, int, int],
    thickness: float,
    *,
    joint_style: str = "round",
    tip: Optional[Tuple[float, float]] = None,
    factor: int = DOCTRINE_SUPERSAMPLE,
) -> None:
    """Composite a route onto *image* at the shared supersampled quality floor.

    Square-grid routes rasterise exactly at the target resolution: every
    segment is an axis-aligned filled rectangle and every turn a full square,
    so they need no supersampling and must not get any (their crisp square
    corners are a family ruling).  Routes that leave the axes — hex-grid steps,
    polar arcs — are built from oriented segment quads and round joints, which
    alias badly when drawn straight onto the finished 512 canvas.

    Such a route is therefore stroked on a ``factor``x transparent overlay,
    LANCZOS-downsampled exactly once, and alpha-composited onto the finished
    scene.  The overlay's RGB plane is pre-filled with the route colour so the
    downsample never blends the stroke toward a foreign backdrop (which would
    leave a dark halo along every edge); only its alpha plane carries shape.
    """

    strokes = [list(subpath) for subpath in subpaths if len(subpath) >= 1]
    if not strokes and tip is None:
        return
    width, height = image.size
    overlay = Image.new("RGBA", (width * factor, height * factor), tuple(color) + (0,))
    overlay_draw = ScaledDraw(ImageDraw.Draw(overlay), factor)
    for subpath in strokes:
        if len(subpath) >= 2:
            _draw_path_polyline(
                overlay_draw, subpath, color, thickness, joint_style=joint_style
            )
        else:
            _draw_path_joint(overlay_draw, subpath[0], color, thickness)
    if tip is not None:
        _draw_pawn_marker(overlay_draw, tip, thickness, color)
    downsampled = overlay.resize((width, height), _RESAMPLING_LANCZOS)
    image.paste(downsampled.convert("RGB"), (0, 0), downsampled.getchannel("A"))


@dataclass
class PathTaskRecord:
    """Serializable metadata for a path-task asset pair."""

    id: str
    ti2v_prompt: str
    solution_image_prompt: str
    canvas_dimensions: Tuple[int, int]
    start_point: Tuple[float, float]
    goal_point: Tuple[float, float]
    image: str
    solution_image_path: str
    task_name: Optional[str] = None
    task_family: Optional[str] = None
    reasoning_image: Optional[str] = None
    answer: Optional[str] = None
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "ti2v_prompt": self.ti2v_prompt,
            "canvas_dimensions": [int(self.canvas_dimensions[0]), int(self.canvas_dimensions[1])],
            "start_point": [float(self.start_point[0]), float(self.start_point[1])],
            "goal_point": [float(self.goal_point[0]), float(self.goal_point[1])],
            "image": self.image,
            "reasoning_image": self.reasoning_image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "answer": self.answer,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "seed": self.seed,
        }
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload


class PathTaskGenerator(AbstractPuzzleGenerator[PathTaskRecord]):
    """Base generator providing canvas configuration and asset management."""

    DEFAULT_OUTPUT_DIR: Optional[PathLike] = "outputs/spatial_navigation"
    DEFAULT_TI2V_PROMPT: Optional[str] = build_path_prompt(mode="ti2v")
    DEFAULT_SOLUTION_IMAGE_PROMPT: Optional[str] = build_path_prompt(mode="solution_image")
    PUZZLE_INPUT_DESCRIPTION: str = PATH_PUZZLE_INPUT_DESCRIPTION
    REASONING_INPUT_DESCRIPTION: str = PATH_REASONING_INPUT_DESCRIPTION
    # Every path task uses one public coordinate-list grammar. Internal cell
    # IDs remain solver/render keys and never cross the answer boundary.
    ANSWER_COORDINATE_SPACE: bool = True

    @classmethod
    def _default_route_output_requirement(cls) -> str:
        """Fallback delivery sentence when a task prompt declares none."""

        return GRID_PATH_ANSWER_FORMAT_INSTRUCTION

    def _cell_id_to_xy(self, cell_id: int) -> Optional[Tuple[int, int]]:
        """Map one internal cell id to its model-facing 1-based ``(x, y)`` label.

        The default covers square-grid tasks whose cell identity is an internal
        0-based ``(row, col)`` tuple.  Tasks with another cell identity (rings
        and segments, axial hexes, stacked layers) override this so the shared
        answer export and authored trace keep a single conversion boundary.
        """

        cell = self._id_to_cell.get(int(cell_id))
        if isinstance(cell, (list, tuple)) and len(cell) == 2:
            return cell_to_xy(cell)
        return None

    def _cell_id_path_to_xy(self, path_ids: Sequence[Any]) -> List[Tuple[int, int]]:
        """Convert an internal cell-id route to model-facing ``(x, y)`` positions."""

        positions: List[Tuple[int, int]] = []
        for raw_id in path_ids:
            position = self._cell_id_to_xy(int(raw_id))
            if position is None:
                raise ValueError(
                    f"{self.__class__.__module__}: cell id {raw_id!r} has no model-facing (x, y) label; "
                    "override _cell_id_to_xy for this cell identity"
                )
            positions.append(position)
        return positions

    def _cell_id_label(self, cell_id: int, *, lead: bool = False) -> str:
        """Name one cell the way the model-facing prompt and answer name it."""

        position = self._cell_id_to_xy(int(cell_id))
        if position is None:
            raise ValueError(f"cell id {cell_id!r} has no public coordinate")
        label = f"position ({position[0]}, {position[1]})"
        return f"{label[0].upper()}{label[1:]}" if lead else label

    def _route_payload_text(self, path_ids: Sequence[Any]) -> str:
        """Render an internal cell-id route in the task's typed answer grammar."""

        return format_xy_positions_payload(self._cell_id_path_to_xy(path_ids))

    def _authored_route_reasoning(
        self,
        path_ids: Sequence[Any],
        extra_payload: Dict[str, Any],
    ) -> Optional[str]:
        """Return sample-specific solving reasoning for the generated route.

        Each task derives this text from its own semantic truth: observe the
        layout, name the load-bearing constraint, walk the route segment by
        segment with its state changes, and confirm the terminal state.  The
        base returns ``None`` so tasks without an authored trace fail loudly
        at SFT reference build instead of shipping template prose.
        """

        return None

    @staticmethod
    def _declared_rule_and_output(prompt: Optional[str]) -> tuple[str, str]:
        """Extract an explicit task Rule/Required Output pair when present."""

        normalized = " ".join(str(prompt or "").split()).strip()
        match = re.search(
            r"(?:^|\s)Rule:\s*(.+?)\s+Required Output:\s*(.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if match is None:
            return "", ""
        return match.group(1).strip(), match.group(2).strip()

    def __init__(
        self,
        output_dir: Optional[PathLike] = None,
        *,
        canvas_width: int = 512,
        aspect: Optional[float] = None,
        size: int = 32,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        show_cell_id: bool = False,
        video: bool = False,
    ) -> None:
        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        if resolved_output is None:
            raise ValueError("output_dir must be provided either via argument or DEFAULT_OUTPUT_DIR")
        super().__init__(resolved_output)

        if canvas_width <= 0:
            raise ValueError("canvas_width must be positive")
        width = int(canvas_width)
        if aspect is not None and aspect <= 0:
            raise ValueError("aspect must be positive when provided")
        if aspect is None:
            height = width
        else:
            height = int(round(width / float(aspect)))
            if height <= 0:
                raise ValueError("Derived canvas height must be positive")
        self.canvas_dimensions: Tuple[int, int] = (width, height)

        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)
        resolved_ti2v_prompt = ti2v_prompt if ti2v_prompt is not None else (self.DEFAULT_TI2V_PROMPT or build_path_prompt(mode="ti2v"))
        self.seed = seed
        default_rule, _ = self._declared_rule_and_output(build_path_prompt(mode="ti2v"))
        ti2v_rule, ti2v_output_desc = self._declared_rule_and_output(resolved_ti2v_prompt)
        if not ti2v_rule:
            ti2v_rule = default_rule
        if not ti2v_output_desc:
            _, ti2v_output_desc = split_visual_description_and_output(resolved_ti2v_prompt or "")
        raw_solution_image_prompt = self.DEFAULT_SOLUTION_IMAGE_PROMPT or _strip_video_instruction(resolved_ti2v_prompt) or build_path_prompt(mode="solution_image")
        solution_rule, solution_image_output_desc = self._declared_rule_and_output(raw_solution_image_prompt)
        if not solution_image_output_desc:
            _, solution_image_output_desc = split_visual_description_and_output(raw_solution_image_prompt)
        self.ti2v_prompt = build_sectioned_prompt(
            self.REASONING_INPUT_DESCRIPTION,
            soften_precise_video_timing(ti2v_output_desc),
            mode="ti2v",
            rule_description=ti2v_rule,
            continuity_paragraph=build_ti2v_continuity_paragraph(
                timing="continuous",
                moving_nouns="the drawn route and every object the rules change",
                static_nouns="everything the rules leave unchanged, including the maze, its labels, background, and camera",
                hold_noun="the completed final state",
            ),
        )
        self.solution_image_prompt = build_sectioned_prompt(
            self.REASONING_INPUT_DESCRIPTION,
            solution_image_output_desc,
            mode="solution_image",
            rule_description=solution_rule or ti2v_rule,
        )
        self.show_cell_id = show_cell_id
        self.video = video
        self._rng = random.Random(seed)
        self._last_video_fps: Optional[int] = None
        self._last_video_num_frames: Optional[int] = None
        # 1-based contiguous cell id index, scoped to traversable cells only.
        # Subclasses rebuild this each create_puzzle (after the maze grid is
        # generated) and evaluators rebuild it via rehydrate_from_record() so
        # _get_cell_id/_cell_center can reverse the mapping. Keys are whatever
        # cell identifier each subclass uses (e.g. (r, c) tuples).
        self._cell_to_id: Dict[Any, int] = {}
        self._id_to_cell: Dict[int, Any] = {}

        root = Path(self.output_dir)
        self.puzzle_dir = root / "puzzles"
        self.reasoning_dir = root / "reasoning"
        self.solution_dir = root / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.reasoning_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)

    @property
    def rng(self) -> random.Random:
        return self._rng

    @property
    def canvas_width(self) -> int:
        return self.canvas_dimensions[0]

    @property
    def canvas_height(self) -> int:
        return self.canvas_dimensions[1]

    def next_id(self) -> str:
        return str(uuid.uuid4())

    @staticmethod
    def strip_video_instruction(prompt: Optional[str]) -> Optional[str]:
        return _strip_video_instruction(prompt)


    def _rebuild_cell_id_index(self, cells: Sequence[Any]) -> None:
        """Assign 1-based contiguous ids to the traversable cells in iteration order."""
        self._cell_to_id = {cell: idx + 1 for idx, cell in enumerate(cells)}
        self._id_to_cell = {idx + 1: cell for idx, cell in enumerate(cells)}

    def rehydrate_from_record(self, record: Dict[str, Any]) -> None:
        """Rebuild the cell-id index from a persisted record so evaluators can
        reverse `_get_cell_id` / `_cell_center` without re-running generation.

        The default reads `record['maze_grid']` (a 2-D WALL/PATH list, PATH==0)
        and enumerates `(r, c)` in row-major order over PATH cells. Subclasses
        with non-(r, c) cell identities (hexagon/labyrinth/layered/etc.) should
        override this.
        """
        grid = record.get("maze_grid")
        if grid is None:
            return
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        path_cells = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if grid[r][c] == 0
        ]
        self._rebuild_cell_id_index(path_cells)

    @staticmethod
    def add_path_length_metadata(
        extra_payload: Dict[str, Any],
        path: Sequence[Any],
        *,
        minimum: int,
        length_key: str = "solution_path_length_cells",
        minimum_key: str = "min_solution_path_length_cells",
    ) -> None:
        extra_payload[length_key] = len(path)
        extra_payload[minimum_key] = max(0, int(minimum))

    def save_images(
        self,
        record_id: str,
        puzzle_image: Image.Image,
        solution_image: Image.Image,
    ) -> Tuple[Path, Path]:
        puzzle_path = self.puzzle_dir / f"{record_id}_puzzle.png"
        solution_path = self.solution_dir / f"{record_id}_solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)
        return puzzle_path, solution_path

    def save_reasoning_image(
        self,
        record_id: str,
        reasoning_image: Image.Image,
    ) -> Path:
        reasoning_path = self.reasoning_dir / f"{record_id}_reasoning.png"
        reasoning_image.save(reasoning_path)
        return reasoning_path

    def save_video(
        self,
        record_id: str,
        puzzle_image: Image.Image,
        points: Sequence[Optional[Tuple[float, float]]],
        thickness: int = 5,
        color: Tuple[int, int, int] = (220, 30, 30),
        fps: int = 16,
        duration: float = 4.0,
        joint_style: str = "round",
        supersample: int = 1,
    ) -> Optional[Path]:
        """Generates a solution video if video output is enabled.

        Each frame is rendered from scratch by drawing the revealed portion of the
        path directly onto a fresh copy of *puzzle_image*.  No mask compositing is
        used, eliminating all rasterisation jitter.

        ``supersample`` > 1 routes the per-frame stroke through
        :func:`composite_supersampled_route`, which is required for routes that
        are not axis-aligned (they alias at the target resolution).  Square-grid
        routes stay at 1x so their exact square corners are preserved.
        """
        self._last_video_fps = None
        self._last_video_num_frames = None
        if not self.video:
            return None

        video_path = self.solution_dir / f"{record_id}_solution.webm"

        # Split at None sentinels (teleport breaks)
        subpaths: List[List[Tuple[float, float]]] = []
        current_subpath: List[Tuple[float, float]] = []
        for point in points:
            if point is None:
                if current_subpath:
                    subpaths.append(current_subpath)
                    current_subpath = []
                continue
            current_subpath.append((float(point[0]), float(point[1])))
        if current_subpath:
            subpaths.append(current_subpath)

        if not subpaths:
            frames = [puzzle_image.copy() for _ in range(max(1, fps))]
            frames = pad_frames_to_wan_compatible_length(frames)
            if not encode_rgb_frames_to_webm(frames, video_path, fps=fps):
                return None
            self._last_video_fps = fps
            self._last_video_num_frames = len(frames)
            return video_path

        # Pre-compute segment lengths
        path_infos: List[Dict[str, Any]] = []
        total_len = 0.0
        for subpath in subpaths:
            cum: List[float] = []
            sub_len = 0.0
            for i in range(len(subpath) - 1):
                d = math.hypot(subpath[i+1][0] - subpath[i][0],
                               subpath[i+1][1] - subpath[i][1])
                sub_len += d
                cum.append(sub_len)
            total_len += sub_len
            path_infos.append({"points": subpath, "cum": cum, "length": sub_len})

        eff_duration = min(duration, 10.0)
        total_frames = int(fps * eff_duration)

        if total_len <= 1e-6:
            if joint_style == "stable_round":
                video_color = _video_stable_path_color(color)
                frame = _recolor_red_dominant_pixels(puzzle_image, video_color)
                draw_path_line(frame, list(subpaths[0]), video_color, thickness, joint_style=joint_style)
                frames = [frame.copy() for _ in range(max(1, fps))]
                frames = pad_frames_to_wan_compatible_length(frames)
                if not encode_rgb_frames_to_webm(
                    frames,
                    video_path,
                    fps=fps,
                    extra_args=["-g", "1", "-bf", "0"],
                ):
                    return None
                self._last_video_fps = fps
                self._last_video_num_frames = len(frames)
                return video_path

            frame = puzzle_image.copy()
            _draw_path_polyline(ImageDraw.Draw(frame), subpaths[0], color, thickness, joint_style=joint_style)
            frames = [frame.copy() for _ in range(max(1, fps))]
            frames = pad_frames_to_wan_compatible_length(frames)
            if not encode_rgb_frames_to_webm(frames, video_path, fps=fps):
                return None
            self._last_video_fps = fps
            self._last_video_num_frames = len(frames)
            return video_path

        if joint_style == "stable_round":
            video_color = _video_stable_path_color(color)
            video_puzzle_image = _recolor_red_dominant_pixels(puzzle_image, video_color)

            full_overlay = Image.new("RGBA", puzzle_image.size, video_color + (0,))
            full_overlay.putalpha(_subpaths_alpha_mask(puzzle_image.size, subpaths, thickness))
            transparent = Image.new("RGBA", puzzle_image.size, (0, 0, 0, 0))
            reveal_mask = np.zeros((puzzle_image.size[1], puzzle_image.size[0]), dtype=bool)

            prev_dist = 0.0
            frames: List[Image.Image] = []
            for f in range(total_frames + 1):
                progress = f / total_frames if total_frames > 0 else 1.0
                cur_dist = total_len * progress

                if cur_dist > prev_dist + 1e-6:
                    running_offset = 0.0
                    for info in path_infos:
                        sub_len = info["length"]
                        segment_start = running_offset
                        segment_end = running_offset + sub_len
                        running_offset = segment_end
                        overlap_start = max(prev_dist, segment_start)
                        overlap_end = min(cur_dist, segment_end)
                        if overlap_end <= overlap_start + 1e-6:
                            continue
                        _accumulate_subpath_interval_mask(
                            reveal_mask,
                            info,
                            overlap_start - segment_start,
                            overlap_end - segment_start,
                            thickness,
                        )

                reveal_alpha = Image.fromarray(reveal_mask.astype(np.uint8) * 255, mode="L")
                visible_overlay = Image.composite(full_overlay, transparent, reveal_alpha)
                frame_rgba = video_puzzle_image.convert("RGBA")
                frame_rgba.alpha_composite(visible_overlay)
                frames.append(frame_rgba.convert("RGB"))
                prev_dist = cur_dist

            if frames:
                frames = [frames[0].copy() for _ in range(4)] + frames

            end_hold_frames = max(4, fps // 2)
            last_frame = frames[-1] if frames else video_puzzle_image.copy()
            for _ in range(end_hold_frames):
                frames.append(last_frame.copy())

            frames = fit_frames_to_video_budget(
                frames,
                max_frames=MULTISTEP_VIDEO_FRAMES,
                final_hold_frames=end_hold_frames,
            )
            if not encode_rgb_frames_to_webm(
                frames,
                video_path,
                fps=fps,
                extra_args=["-g", "1", "-bf", "0"],
            ):
                return None
            self._last_video_fps = fps
            self._last_video_num_frames = len(frames)
            return video_path

        def _clip_subpath(info: Dict[str, Any], dist: float) -> List[Tuple[float, float]]:
            """Return the portion of a subpath up to *dist* along it."""
            pts = info["points"]
            cum = info["cum"]
            if not pts:
                return []
            if not cum or dist <= 0:
                return [pts[0]]
            # Count fully revealed segments
            n_full = 0
            while n_full < len(cum) and cum[n_full] <= dist + 1e-6:
                n_full += 1
            result = list(pts[:n_full + 1])
            # Interpolate tip on the partial segment
            if n_full < len(cum):
                prev = cum[n_full - 1] if n_full > 0 else 0.0
                seg_len = cum[n_full] - prev
                remain = dist - prev
                if seg_len > 1e-9 and remain > 0:
                    ratio = min(1.0, remain / seg_len)
                    p1, p2 = pts[n_full], pts[n_full + 1]
                    result.append((
                        round(p1[0] + (p2[0] - p1[0]) * ratio),
                        round(p1[1] + (p2[1] - p1[1]) * ratio),
                    ))
            return result

        frames: List[Image.Image] = []

        for f in range(total_frames + 1):
            progress = f / total_frames if total_frames > 0 else 1.0
            cur_dist = total_len * progress

            # Draw directly — no mask, no compositing
            frame = puzzle_image.copy()
            remaining = cur_dist
            current_tip: Optional[Tuple[float, float]] = None
            revealed: List[List[Tuple[float, float]]] = []

            for info in path_infos:
                sub_len = info["length"]
                if sub_len <= 1e-6:
                    continue
                if remaining >= sub_len - 1e-6:
                    # Fully revealed subpath
                    revealed.append(list(info["points"]))
                    current_tip = info["points"][-1]
                    remaining -= sub_len
                elif remaining > 0:
                    # Partially revealed subpath
                    visible = _clip_subpath(info, remaining)
                    if visible:
                        revealed.append(list(visible))
                        current_tip = visible[-1]
                    break
                else:
                    break

            if supersample > 1:
                composite_supersampled_route(
                    frame,
                    revealed,
                    color,
                    thickness,
                    joint_style=joint_style,
                    tip=current_tip,
                    factor=supersample,
                )
            else:
                draw = ImageDraw.Draw(frame)
                for visible in revealed:
                    if len(visible) >= 2:
                        _draw_path_polyline(draw, visible, color, thickness, joint_style=joint_style)
                    else:
                        _draw_path_joint(draw, visible[0], color, thickness)
                if current_tip is not None:
                    _draw_pawn_marker(draw, current_tip, thickness, color)

            frames.append(frame)

        if frames:
            frames = [frames[0].copy() for _ in range(4)] + frames

        # Hold end
        end_hold_frames = max(4, fps // 2)
        last_frame = frames[-1] if frames else puzzle_image.copy()
        for _ in range(end_hold_frames):
            frames.append(last_frame.copy())

        frames = fit_frames_to_video_budget(
            frames,
            max_frames=MULTISTEP_VIDEO_FRAMES,
            final_hold_frames=end_hold_frames,
        )
        if not encode_rgb_frames_to_webm(frames, video_path, fps=fps):
            return None
        self._last_video_fps = fps
        self._last_video_num_frames = len(frames)
        return video_path

    @abstractmethod
    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        """Return the pixel center for a cell id."""

    def build_record(
        self,
        record_id: str,
        *,
        start_point: Tuple[float, float],
        goal_point: Tuple[float, float],
        puzzle_path: Path,
        solution_path: Path,
        reasoning_path: Optional[Path] = None,
        ti2v_prompt: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        video_path: Optional[Path] = None,
    ) -> PathTaskRecord:
        record_ti2v_prompt = ti2v_prompt if ti2v_prompt is not None else self.ti2v_prompt
        extra_payload = extra if extra is not None else {}
        if "rendered_glyph_inventory" not in extra_payload:
            rows = int(getattr(self, "rows", 0))
            cols = int(getattr(self, "cols", 0))
            if rows <= 0 or cols <= 0:
                raise ValueError(
                    f"{self.__class__.__module__}: an axis-labelled grid must expose positive rows/cols "
                    "or declare rendered_glyph_inventory itself"
                )
            extra_payload["rendered_glyph_inventory"] = {
                "axis_row_labels": [str(index + 1) for index in range(rows)],
                "axis_col_labels": [str(index + 1) for index in range(cols)],
                "fixed_labels": [],
            }
            extra_payload["rendered_glyph_schema"] = {
                "axis_row_labels": "1-based row numbers drawn once per row in the left gutter, outside the grid body",
                "axis_col_labels": "1-based column numbers drawn once per column in the top gutter, outside the grid body",
                "fixed_labels": "task-bounded labels only; no per-cell ID, title, rule prose, status sentence, or answer banner",
            }
        video_rel: Optional[str] = None
        video_fps: Optional[int] = None
        video_num_frames: Optional[int] = None
        if video_path is not None:
            actual_video_path = video_path
            if actual_video_path.exists():
                video_rel = self.relativize_path(actual_video_path)
                video_fps = self._last_video_fps
                video_num_frames = self._last_video_num_frames
        # Derive canonical route answer from the task's final parseable payload.
        route_answer: Optional[str] = None
        path_ids = extra_payload.get("solution_path_cell_ids")
        if isinstance(path_ids, list):
            route_answer = format_answer_line(self._route_payload_text(path_ids))
        task_name = None
        task_family = None
        try:
            from data.registry import TASK_SPECS

            module_name = self.__class__.__module__
            class_name = self.__class__.__name__
            spec = next(
                (
                    candidate
                    for candidate in TASK_SPECS.values()
                    if candidate.module == module_name and candidate.class_name == class_name
                ),
                None,
            )
            if spec is not None:
                task_name = spec.name
                task_family = spec.task_family
        except Exception:
            pass
        return PathTaskRecord(
            id=record_id,
            task_name=task_name,
            task_family=task_family,
            ti2v_prompt=record_ti2v_prompt,
            solution_image_prompt=self.solution_image_prompt or _strip_video_instruction(record_ti2v_prompt),
            canvas_dimensions=self.canvas_dimensions,
            start_point=start_point,
            goal_point=goal_point,
            image=self.relativize_path(reasoning_path) if reasoning_path is not None else self.relativize_path(puzzle_path),
            reasoning_image=self.relativize_path(reasoning_path) if reasoning_path is not None else self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
            answer=route_answer,
            seed=self.seed,
            extra=extra_payload,
            solution_video_path=video_rel,
            video_fps=video_fps,
            video_num_frames=video_num_frames,
        )





@dataclass
class PathEvaluationResult:
    puzzle_id: str
    red_pixel_count: int
    overlaps_walls: bool
    touches_start: bool
    touches_goal: bool
    connected: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "puzzle_id": self.puzzle_id,
            "red_pixel_count": self.red_pixel_count,
            "overlaps_walls": self.overlaps_walls,
            "touches_start": self.touches_start,
            "touches_goal": self.touches_goal,
            "connected": self.connected,
            "message": self.message,
        }


class PathTaskEvaluator(AbstractPuzzleEvaluator):
    """Pixel-based evaluation for rendered path tasks."""

    RED_THRESHOLD: int = 150
    RED_DOMINANCE: int = 70
    WALL_VALUE_THRESHOLD: int = 40
    ENDPOINT_SEARCH_RADIUS: int = 4
    CONNECTIVITY_OFFSETS: Tuple[Tuple[int, int], ...] = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    )

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
    ) -> PathEvaluationResult:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate_image)

        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image not found: {candidate_path}")

        source_path = self.resolve_path(record["image"])
        if not source_path.exists():
            raise FileNotFoundError(f"Puzzle image not found: {source_path}")

        with Image.open(source_path) as src_image:
            puzzle_image = src_image.convert("RGB")
        source_canvas_size = puzzle_image.size
        with Image.open(candidate_path) as cand_image:
            candidate_image_rgb = cand_image.convert("RGB")

        if candidate_image_rgb.size != puzzle_image.size:
            puzzle_image = puzzle_image.resize(candidate_image_rgb.size, Image.NEAREST)

        puzzle_pixels = np.asarray(puzzle_image, dtype=np.uint8)
        candidate_pixels = np.asarray(candidate_image_rgb, dtype=np.uint8)

        red_mask = self._prepare_red_mask(self._red_mask(candidate_pixels))
        red_pixel_count = int(red_mask.sum())
        if red_pixel_count == 0:
            self._resolve_endpoint(record, "start", source_canvas_size, candidate_image_rgb.size)
            self._resolve_endpoint(record, "goal", source_canvas_size, candidate_image_rgb.size)
            return PathEvaluationResult(
                puzzle_id=puzzle_id,
                red_pixel_count=0,
                overlaps_walls=False,
                touches_start=False,
                touches_goal=False,
                connected=False,
                message="No red path detected.",
            )

        start_point = self._resolve_endpoint(record, "start", source_canvas_size, candidate_image_rgb.size)
        goal_point = self._resolve_endpoint(record, "goal", source_canvas_size, candidate_image_rgb.size)

        wall_mask = self._wall_mask(puzzle_pixels)
        safe_radius = self._endpoint_safe_radius(record, source_canvas_size, candidate_image_rgb.size)
        wall_mask = self._suppress_endpoint_walls(wall_mask, [start_point, goal_point], safe_radius)

        # Exempt the red endpoint marker regions drawn on puzzle images.
        marker_bboxes = self._endpoint_marker_bboxes(puzzle_pixels, [start_point, goal_point], safe_radius)
        if marker_bboxes:
            marker_margin = max(1, min(4, int(round(safe_radius * 0.25))))
            wall_mask = self._suppress_endpoint_bboxes(wall_mask, marker_bboxes, marker_margin)

        endpoint_bboxes = self._endpoint_bboxes(record, source_canvas_size, candidate_image_rgb.size)
        if endpoint_bboxes:
            bbox_margin = max(2, min(6, int(round(safe_radius * 0.5))))
            wall_mask = self._suppress_endpoint_bboxes(wall_mask, endpoint_bboxes, bbox_margin)
        overlaps_walls = bool(np.any(red_mask & wall_mask))

        endpoint_search_radius = max(self.ENDPOINT_SEARCH_RADIUS, safe_radius)
        start_seed = self._nearest_red(red_mask, start_point, radius=endpoint_search_radius)
        goal_seed = self._nearest_red(red_mask, goal_point, radius=endpoint_search_radius)
        touches_start = start_seed is not None
        touches_goal = goal_seed is not None

        connected = False
        if touches_start and touches_goal and not overlaps_walls:
            connected = self._connected(red_mask, start_seed, goal_seed)

        if overlaps_walls:
            message = "Red path overlaps walls."
        elif not touches_start:
            message = "Red path does not reach the start."
        elif not touches_goal:
            message = "Red path does not reach the goal."
        elif not connected:
            message = "Red path is not continuous between start and goal."
        else:
            message = "Red path successfully connects start to goal."

        return PathEvaluationResult(
            puzzle_id=puzzle_id,
            red_pixel_count=red_pixel_count,
            overlaps_walls=overlaps_walls,
            touches_start=touches_start,
            touches_goal=touches_goal,
            connected=connected,
            message=message,
        )

    def _red_mask(self, pixels: np.ndarray) -> np.ndarray:
        red = pixels[:, :, 0].astype(np.int32)
        green = pixels[:, :, 1].astype(np.int32)
        blue = pixels[:, :, 2].astype(np.int32)
        dominance = red - np.maximum(green, blue)
        mask = (red >= self.RED_THRESHOLD) & (dominance >= self.RED_DOMINANCE)
        return mask

    def _wall_mask(self, pixels: np.ndarray) -> np.ndarray:
        max_channel = pixels.max(axis=2)
        return max_channel <= self.WALL_VALUE_THRESHOLD

    def _endpoint_safe_radius(
        self,
        record: Dict[str, Any],
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> int:
        base_radius = self.ENDPOINT_SEARCH_RADIUS
        if "cell_size" in record:
            base_radius = max(base_radius, int(round(float(record["cell_size"]) * 0.45)))
        elif "cell_radius" in record:
            base_radius = max(base_radius, int(round(float(record["cell_radius"]) * 0.8)))
        elif "ring_width" in record:
            base_radius = max(base_radius, int(round(float(record["ring_width"]) * 0.6)))

        src_w, src_h = source_size
        tgt_w, tgt_h = target_size
        if src_w <= 0 or src_h <= 0:
            return base_radius
        scale = min(tgt_w / float(src_w), tgt_h / float(src_h))
        return max(self.ENDPOINT_SEARCH_RADIUS, int(round(base_radius * scale)))

    def _suppress_endpoint_walls(
        self,
        wall_mask: np.ndarray,
        endpoints: Iterable[Tuple[float, float]],
        radius: int,
    ) -> np.ndarray:
        if radius <= 0:
            return wall_mask
        height, width = wall_mask.shape
        y_grid, x_grid = np.ogrid[:height, :width]
        for point in endpoints:
            if point is None:
                continue
            x, y = point
            dist2 = (x_grid - x) ** 2 + (y_grid - y) ** 2
            wall_mask[dist2 <= (radius * radius)] = False
        return wall_mask

    def _suppress_endpoint_bboxes(
        self,
        wall_mask: np.ndarray,
        bboxes: Iterable[Tuple[float, float, float, float]],
        margin: int,
    ) -> np.ndarray:
        if margin < 0:
            margin = 0
        height, width = wall_mask.shape
        for left, top, right, bottom in bboxes:
            x0 = max(0, int(math.floor(left - margin)))
            y0 = max(0, int(math.floor(top - margin)))
            x1 = min(width, int(math.ceil(right + margin)))
            y1 = min(height, int(math.ceil(bottom + margin)))
            if x0 < x1 and y0 < y1:
                wall_mask[y0:y1, x0:x1] = False
        return wall_mask

    def _endpoint_bboxes(
        self,
        record: Dict[str, Any],
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> List[Tuple[float, float, float, float]]:
        if "cell_bboxes" not in record:
            return []
        cell_bboxes = record.get("cell_bboxes")
        if not isinstance(cell_bboxes, Iterable):
            return []
        bboxes: List[Tuple[float, float, float, float]] = []
        for key in ("start", "goal"):
            if key not in record:
                continue
            cell = record.get(key)
            if not isinstance(cell, Iterable):
                continue
            cell_list = list(cell)
            if len(cell_list) < 2:
                continue
            row = int(cell_list[0])
            col = int(cell_list[1])
            bbox_rows = list(cell_bboxes)
            if row < 0 or row >= len(bbox_rows):
                continue
            row_data = bbox_rows[row]
            if not isinstance(row_data, Iterable):
                continue
            row_cells = list(row_data)
            if col < 0 or col >= len(row_cells):
                continue
            bbox = row_cells[col]
            if not isinstance(bbox, Iterable):
                continue
            bbox_values = list(bbox)
            if len(bbox_values) < 4:
                continue
            scaled = self._scale_bbox(
                (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3]),
                source_size,
                target_size,
            )
            bboxes.append(scaled)
        return bboxes

    def _resolve_endpoint(
        self,
        record: Dict[str, Any],
        label: str,
        puzzle_size: Tuple[int, int],
        candidate_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        pixel_key = f"{label}_pixel"
        if pixel_key in record:
            point = self._coerce_point(record[pixel_key])
            return self._scale_point(point, puzzle_size, candidate_size)
        point_key = f"{label}_point"
        if point_key in record:
            point = self._coerce_point(record[point_key])
            return self._scale_point(point, puzzle_size, candidate_size)
        if label in record and "cell_bboxes" in record:
            cell = record[label]
            if not isinstance(cell, Iterable):
                raise ValueError(f"Record contains invalid '{label}' entry")
            cell_list = list(cell)
            if len(cell_list) < 2:
                raise ValueError(f"Record '{label}' does not contain row and column")
            row = int(cell_list[0])
            col = int(cell_list[1])
            bbox_rows = record["cell_bboxes"]
            if not isinstance(bbox_rows, Iterable):
                raise ValueError("cell_bboxes must be iterable")
            bbox_list = list(bbox_rows)
            if row < 0 or row >= len(bbox_list):
                raise ValueError("start or goal row index out of range")
            row_data = bbox_list[row]
            if not isinstance(row_data, Iterable):
                raise ValueError("cell row is not iterable")
            row_cells = list(row_data)
            if col < 0 or col >= len(row_cells):
                raise ValueError("start or goal column index out of range")
            bbox = row_cells[col]
            if not isinstance(bbox, Iterable):
                raise ValueError("cell bbox is not iterable")
            bbox_values = list(bbox)
            if len(bbox_values) < 4:
                raise ValueError("cell bbox must contain four coordinates")
            left = float(bbox_values[0])
            top = float(bbox_values[1])
            right = float(bbox_values[2])
            bottom = float(bbox_values[3])
            center = ((left + right) * 0.5, (top + bottom) * 0.5)
            return self._scale_point(center, puzzle_size, candidate_size)
        raise KeyError(f"Record is missing endpoint information for '{label}'")

    def _scale_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        left, top, right, bottom = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        p1 = self._scale_point((left, top), source_size, target_size)
        p2 = self._scale_point((right, bottom), source_size, target_size)
        return (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))

    def _coerce_point(self, value: Any) -> Tuple[float, float]:
        if isinstance(value, dict):
            if "x" in value and "y" in value:
                return float(value["x"]), float(value["y"])
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
        raise ValueError("Endpoint entries must contain two numeric values")

    def _scale_point(
        self,
        point: Tuple[float, float],
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        source_width, source_height = source_size
        target_width, target_height = target_size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Source dimensions must be positive")
        scale_x = target_width / float(source_width)
        scale_y = target_height / float(source_height)
        return point[0] * scale_x, point[1] * scale_y

    def _nearest_red(
        self,
        red_mask: np.ndarray,
        point: Tuple[float, float],
        radius: Optional[int] = None,
    ) -> Optional[Tuple[int, int]]:
        height, width = red_mask.shape
        cx = int(round(point[0]))
        cy = int(round(point[1]))
        cx = max(0, min(width - 1, cx))
        cy = max(0, min(height - 1, cy))
        if red_mask[cy, cx]:
            return (cy, cx)
        max_radius = self.ENDPOINT_SEARCH_RADIUS if radius is None else max(0, int(radius))
        for delta in range(1, max_radius + 1):
            min_x = max(0, cx - delta)
            max_x = min(width - 1, cx + delta)
            min_y = max(0, cy - delta)
            max_y = min(height - 1, cy + delta)
            for y in range(min_y, max_y + 1):
                if red_mask[y, min_x]:
                    return (y, min_x)
                if red_mask[y, max_x]:
                    return (y, max_x)
            for x in range(min_x + 1, max_x):
                if red_mask[min_y, x]:
                    return (min_y, x)
                if red_mask[max_y, x]:
                    return (max_y, x)
        return None

    def _component_from_seed(
        self,
        mask: np.ndarray,
        seed: Tuple[int, int],
    ) -> np.ndarray:
        height, width = mask.shape
        component = np.zeros((height, width), dtype=bool)
        queue: deque[Tuple[int, int]] = deque([seed])
        component[seed[0], seed[1]] = True
        while queue:
            y, x = queue.popleft()
            for dy, dx in self.CONNECTIVITY_OFFSETS:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if mask[ny, nx] and not component[ny, nx]:
                        component[ny, nx] = True
                        queue.append((ny, nx))
        return component

    def _endpoint_marker_bboxes(
        self,
        puzzle_pixels: np.ndarray,
        endpoints: Iterable[Tuple[float, float]],
        safe_radius: int,
    ) -> List[Tuple[float, float, float, float]]:
        puzzle_red_mask = self._red_mask(puzzle_pixels)
        if not puzzle_red_mask.any():
            return []

        height, width = puzzle_red_mask.shape
        padding = max(2, min(8, int(round(safe_radius * 0.5))))
        bboxes: List[Tuple[float, float, float, float]] = []
        for endpoint in endpoints:
            if endpoint is None:
                continue
            seed = self._nearest_red(
                puzzle_red_mask,
                endpoint,
                radius=max(self.ENDPOINT_SEARCH_RADIUS, safe_radius * 2),
            )
            if seed is None:
                continue
            component = self._component_from_seed(puzzle_red_mask, seed)
            ys, xs = np.nonzero(component)
            if ys.size == 0 or xs.size == 0:
                continue
            left = max(0, int(xs.min()) - padding)
            top = max(0, int(ys.min()) - padding)
            right = min(width, int(xs.max()) + padding + 1)
            bottom = min(height, int(ys.max()) + padding + 1)
            bboxes.append((float(left), float(top), float(right), float(bottom)))
        return bboxes

    def _connected(
        self,
        red_mask: np.ndarray,
        start_seed: Tuple[int, int],
        goal_seed: Tuple[int, int],
    ) -> bool:
        height, width = red_mask.shape
        visited = np.zeros((height, width), dtype=bool)
        queue: deque[Tuple[int, int]] = deque([start_seed])
        visited[start_seed[0], start_seed[1]] = True
        while queue:
            y, x = queue.popleft()
            if (y, x) == goal_seed:
                return True
            for dy, dx in self.CONNECTIVITY_OFFSETS:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if red_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        return False

    def _prepare_red_mask(self, red_mask: np.ndarray) -> np.ndarray:
        return red_mask.astype(bool)

__all__ = [
    "GRID_AXIS_LABEL_COLOR",
    "GRID_AXIS_LABEL_GUTTER",
    "GRID_PATH_ANSWER_FORMAT_INSTRUCTION",
    "GRID_PATH_ANSWER_SCHEMA",
    "GRID_XY_CONVENTION_SENTENCE",
    "PathEvaluationResult",
    "PathTaskEvaluator",
    "PathTaskGenerator",
    "PathTaskRecord",
    "STRICT_XY_PATH_ANSWER_RE",
    "cell_to_xy",
    "composite_supersampled_route",
    "draw_grid_axis_labels",
    "format_xy_path_payload",
    "format_xy_positions_payload",
    "grid_axis_label_layout",
    "parse_xy_path_answer",
    "xy_to_cell",
]
