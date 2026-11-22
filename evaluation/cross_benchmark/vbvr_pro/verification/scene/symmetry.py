from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result
import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.vision.masks import color_mask, ink, no_added_components, no_unexpected_ink
from evaluation.cross_benchmark.vbvr_pro.vision.shapes import fit_ring


def _symmetry(frames, final, metadata):
    p, s = metadata.get("parameters", {}), metadata.get("semantic_ground_truth", {})
    grid = p.get("full_pattern")
    color = p.get("fill_rgb")
    if grid is None or color is None:
        return _result({"semantic_grid_available": False})
    mask = color_mask(final, color)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return _result({"grid_visible": False})
    # Rendering layers carry the exact board rectangle; otherwise the reference
    # foreground bounds recover it (including the thin neutral grid boundary).
    n = len(grid)
    gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    fg = gray < 245
    yy, xx = np.nonzero(fg)
    x0, x1, y0, y1 = int(xx.min()), int(xx.max() + 1), int(yy.min()), int(yy.max() + 1)
    # Metadata semantic grid commonly includes explicit origin and cell size.
    sg = s.get("grid", {})
    origin = sg.get("origin_px") or sg.get("top_left_px")
    cell = sg.get("cell_size_px")
    if origin is not None and isinstance(cell, (int, float)):
        x0, y0 = origin
        x1, y1 = x0 + n * cell, y0 + n * cell
    checks, observed = {}, []
    for row in range(n):
        values = []
        for col in range(n):
            xa = int(x0 + (col + 0.20) * (x1 - x0) / n)
            xb = int(x0 + (col + 0.80) * (x1 - x0) / n)
            ya = int(y0 + (row + 0.20) * (y1 - y0) / n)
            yb = int(y0 + (row + 0.80) * (y1 - y0) / n)
            region = frames[-1][ya:yb, xa:xb]
            coverage = float(color_mask(region, color).mean())
            blank = float(np.mean(np.min(region, axis=2) > 225))
            value = int(grid[row][col])
            checks[f"cell_{row}_{col}"] = coverage >= 0.92 if value else blank >= 0.96
            values.append(coverage)
        observed.append(values)
    checks["unchanged_existing_cells"] = True
    incomplete = p.get("incomplete_pattern", [])
    if incomplete:
        for f in frames:
            for row in range(n):
                for col in range(n):
                    if incomplete[row][col] < 0:
                        continue
                    xa = int(x0 + (col + 0.25) * (x1 - x0) / n)
                    xb = int(x0 + (col + 0.75) * (x1 - x0) / n)
                    ya = int(y0 + (row + 0.25) * (y1 - y0) / n)
                    yb = int(y0 + (row + 0.75) * (y1 - y0) / n)
                    coverage = float(color_mask(f[ya:yb, xa:xb], color).mean())
                    checks["unchanged_existing_cells"] &= coverage > 0.9 if incomplete[row][col] else coverage < 0.03
    checks["no_extraneous_content"] = no_unexpected_ink(frames[-1], final)[0]
    return _result(checks, {"cell_color_coverage": observed})


def _concentric(frames, final, metadata):
    p = metadata.get("parameters", {})
    palette = [p.get("outer_color"), p.get("inner_color")]
    if any(c is None for c in palette):
        return _result({"ring_metadata_available": False})
    checks, evidence = {}, {}
    center = np.array([final.shape[1] / 2, final.shape[0] / 2])
    footprints = [np.zeros(frame.shape[:2], np.uint8) for frame in frames]
    # Circle fitting also works when a ring is clipped by the image border.
    for k, color in enumerate(palette):
        centers = []
        radii = []
        supports = []
        radius = p["outer_radius" if k == 0 else "inner_radius"]
        for index, f in enumerate(frames):
            fitted = fit_ring(color_mask(f, color), radius)
            if fitted is None:
                continue
            c, rad, support = fitted
            centers.append(c)
            radii.append(rad)
            supports.append(support)
            stroke = p.get("outline_width_px", 8)
            # Declarative ellipse radii describe the outer edge; PIL draws its
            # outline inward, whereas OpenCV centres a stroke on the radius.
            # Expand the geometric stroke *before* clipping to the canvas.
            # Dilating an already clipped ring loses off-canvas support near a
            # tangent and misclassifies the codec's two-pixel edge bleed.
            cv2.circle(footprints[index], tuple(np.round(c).astype(int)), round(radius - stroke / 2), 1, stroke + 8)
        checks[f"ring_{k}_visible_throughout"] = len(centers) == len(frames)
        checks[f"ring_{k}_radius_preserved"] = bool(radii and max(abs(r - radius) for r in radii) <= 5)
        checks[f"ring_{k}_ends_centered"] = bool(centers and np.linalg.norm(np.array(centers[-1]) - center) <= 8)
        checks[f"ring_{k}_is_outline_not_disc"] = bool(supports and min(supports) >= 0.12)
        if len(centers) >= 2:
            positions = np.array(centers)
            travel = float(np.linalg.norm(positions[-1] - positions[0]))
            checks[f"ring_{k}_moves_without_teleport"] = bool(
                np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) <= max(40, travel * 0.35)
            )
        evidence[f"ring_{k}_centers"] = centers
    background_ok = []
    for frame, footprint in zip(frames, footprints):
        # Use the same 512px observation grid as the motion readers. A single
        # encoded chroma sample can become a 16px mosquito patch after the
        # core's canonical upsample; it is not 16 independently observed pixels.
        if frame.shape[1] > 512:
            size = (512, round(frame.shape[0] * 512 / frame.shape[1]))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            footprint = (cv2.resize(footprint.astype(float), size, interpolation=cv2.INTER_AREA) >= 0.25).astype(
                np.uint8
            )
        background_ok.append(no_added_components(ink(frame), footprint, radius=2)[0])
    checks["no_isolated_added_content_throughout"] = all(background_ok)
    checks["no_extraneous_content"] = no_unexpected_ink(frames[-1], final)[0]
    return _result(checks, evidence)
