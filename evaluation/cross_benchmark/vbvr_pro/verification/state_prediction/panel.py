from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _blobs, _layers, _mask, _result, _static
import cv2
import numpy as np


def _panel(frames, first, metadata):
    p = metadata["parameters"]
    layers = _layers(metadata)
    lights = [
        layer
        for layer in layers
        if layer.get("kind") == "ellipse" and layer.get("type") == "filled_shape" and layer.get("rx_px", 0) > 10
    ]
    slots = [layer for layer in layers if layer.get("kind") == "rect" and layer.get("fill_rgb") == [0, 0, 0]]
    lights.sort(key=lambda layer: layer["cx_px"])
    slots.sort(key=lambda layer: layer["x_px"])
    if len(lights) != p["num_lights"] or len(slots) != len(lights):
        raise ValueError("Reference panel geometry unavailable")
    palette = {
        "orange": [255, 165, 0],
        "purple": [128, 0, 128],
        "pink": [255, 192, 203],
        "blue": [0, 0, 255],
        "red": [255, 0, 0],
        "magenta": [255, 0, 255],
        "indigo": [75, 0, 130],
        "lime": [0, 255, 0],
    }
    names = ["left", "middle", "right"]
    mapping = p["position_color_map"]
    target = next(name for name in names if mapping[name] == p["target_color"])
    if p["target_positions"] != [target] * len(lights):
        raise ValueError("Invalid benchmark: panel goal contradicts shared mapping")
    allowed = np.zeros(first.shape[:2], np.uint8)
    for light, slot in zip(lights, slots):
        cv2.circle(allowed, (round(light["cx_px"]), round(light["cy_px"])), round(light["rx_px"] + 4), 1, -1)
        x, y, w, h = (round(slot[key]) for key in ("x_px", "y_px", "w_px", "h_px"))
        allowed[y + 4 : y + h - 4, x + 4 : x + w - 4] = 1
    states = []
    mapping_ok = True
    for frame in frames:
        state = []
        for light, slot in zip(lights, slots):
            x, y, w, h = (round(slot[key]) for key in ("x_px", "y_px", "w_px", "h_px"))
            blobs = _blobs(_mask(frame[y + 4 : y + h - 4, x + 4 : x + w - 4], [128, 128, 128], 25), w * h * 0.025)
            if len(blobs) != 1:
                return _result({"exactly_one_lever_per_unit_every_frame": False})
            ratio = (blobs[0][0][0] + 4) / w
            position = int(np.argmin(np.abs(np.array([0.15, 0.5, 0.85]) - ratio)))
            state.append(position)
            cx, cy = round(light["cx_px"]), round(light["cy_px"])
            patch = frame[cy - 8 : cy + 9, cx - 8 : cx + 9]
            rgb = palette[mapping[names[position]]]
            mapping_ok &= np.mean(_mask(patch, rgb, 40)) >= 0.9
        states.append(state)
    checks = {
        "all_levers_at_target": states[-1] == [names.index(target)] * len(lights),
        "light_color_always_matches_its_lever": mapping_ok,
        "no_lever_position_skipped": np.max(np.abs(np.diff(states, axis=0)), initial=0) <= 1,
        "levers_already_correct_do_not_move": all(
            all(s[i] == names.index(target) for s in states)
            for i, position in enumerate(p["initial_positions"])
            if position == target
        ),
    }
    checks["panel_structure_preserved"], worst = _static(frames, first, allowed.astype(bool))
    return _result(checks, {"final_positions": [names[i] for i in states[-1]], "unexpected_static_component": worst})
