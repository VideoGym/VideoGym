from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _layers, _result, _static
import numpy as np


def _vessels(frames, first, metadata):
    p = metadata["parameters"]
    rects = [
        layer
        for layer in _layers(metadata)
        if layer.get("kind") == "rect" and layer.get("fill_rgb") not in ([0, 0, 0], [128, 128, 128], [255, 255, 255])
    ]
    # Common liquid base is wider than the identical-diameter vertical tubes.
    tubes = sorted((r for r in rects if r["w_px"] < max(r["w_px"] for r in rects) / 2), key=lambda r: r["x_px"])
    if len(tubes) != p["num_tubes"]:
        raise ValueError("Reference vessel geometry unavailable")
    background = np.array(p.get("background_color", first[0, 0, ::-1].tolist())[::-1], np.float32)
    levels = []
    for frame in frames:
        state = []
        for tube in tubes:
            x, w = round(tube["x_px"]), round(tube["w_px"])
            bottom = round(tube["y_px"] + tube["h_px"])
            margin = min(8, max(4, round(w * 0.1)))
            pixels = frame[:bottom, x + margin : x + w - margin].astype(np.float32)
            vector = np.asarray(tube["fill_rgb"][::-1], np.float32) - background
            alpha = np.clip(np.sum((pixels - background) * vector, axis=2) / np.dot(vector, vector), 0, 1)
            residual = np.max(np.abs(pixels - (background + alpha[:, :, None] * vector)), axis=2)
            # Black labels/walls are not liquid even when their projection
            # onto the background-to-liquid colour ray is large.
            alpha[residual > 35] = 0
            profile = np.median(alpha, axis=1)
            wet = np.flatnonzero(profile >= 0.5)
            if len(wet) == 0 or np.max(np.diff(wet), initial=1) > 2 or wet[-1] < bottom - 3:
                return _result({"one_contiguous_liquid_column_per_tube": False})
            top = int(wet[0])
            if top:
                fraction = (0.5 - profile[top - 1]) / (profile[top] - profile[top - 1])
                # Half-coverage lies between pixel centres. Convert its
                # crossing to the rectangle-edge coordinate used by metadata.
                surface = top - 0.5 + float(fraction)
            else:
                surface = 0.0
            state.append(float(bottom - surface))
        levels.append(state)
    levels = np.array(levels)
    initial = np.array(p["initial_levels_list"])
    mean = initial.mean()
    deviation = np.abs(levels - mean)
    direction = initial - mean
    if np.dot(direction, direction) > 0:
        relaxation = (levels - mean) @ direction / np.dot(direction, direction)
        physical_error = np.max(np.abs(levels - (mean + relaxation[:, None] * direction)))
        visible_flow = bool(np.any((relaxation > 0.05) & (relaxation < 0.95)))
    else:
        physical_error = float(np.max(np.abs(levels - mean)))
        visible_flow = True
    checks = {
        "final_equal_levels": np.max(np.abs(levels[-1] - mean)) <= 3,
        "volume_conserved_every_frame": np.max(np.abs(levels.sum(axis=1) - initial.sum())) <= len(tubes) * 2,
        "relaxation_toward_equilibrium": np.max(np.diff(deviation, axis=0), initial=0) <= 2,
        "no_overshooting_equilibrium": np.all(np.where(initial >= mean, levels >= mean - 2, levels <= mean + 2)),
        "starts_at_input_levels": np.max(np.abs(levels[0] - initial)) <= 3,
        "common_hydrostatic_relaxation": physical_error <= 3,
        "settling_process_observed": visible_flow,
    }
    # Reconstruct the empty/filled interiors at each measured level. A liquid's
    # former location is not a free edit region after the liquid recedes.
    static, worst = True, 0
    for frame, state in zip(frames, levels):
        expected = first.copy()
        allowed = np.zeros(first.shape[:2], np.uint8)
        for tube, height, initial_height in zip(tubes, state, initial):
            x, w = round(tube["x_px"]), round(tube["w_px"])
            bottom = round(tube["y_px"] + tube["h_px"])
            top, initial_top = round(bottom - height), round(bottom - initial_height)
            earliest = max(0, min(top, initial_top) - 3)
            expected[earliest:bottom, x : x + w] = background
            expected[top:bottom, x : x + w] = tube["fill_rgb"][::-1]
            # Only surface and codec-scale edge fringes are unobservable.
            allowed[max(0, top - 3) : top + 3, x : x + w] = 1
            allowed[max(0, initial_top - 3) : initial_top + 3, x : x + w] = 1
            allowed[earliest:bottom, max(0, x - 3) : x + 4] = 1
            allowed[earliest:bottom, x + w - 4 : min(first.shape[1], x + w + 3)] = 1
        preserved, error = _static([frame], expected, allowed.astype(bool))
        static &= preserved
        worst = max(worst, error)
    checks["vessels_and_labels_preserved"] = static
    return _result(
        checks,
        {
            "initial_levels": initial.tolist(),
            "final_levels": levels[-1].tolist(),
            "equilibrium": float(mean),
            "maximum_relaxation_error_px": float(physical_error),
            "unexpected_static_component": worst,
        },
    )
