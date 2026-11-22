"""Independent traffic-light state machine, including countdown digits."""

import cv2
import numpy as np

from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _layers, _mask, _result, _static

COLORS = {"red": [255, 0, 0], "yellow": [255, 200, 0], "green": [0, 200, 0]}


def _digit(frame, box):
    x, y, w, h = box
    patch = frame[y + 7 : y + h - 7, x + 7 : x + w - 7]
    mask = np.max(patch, axis=2) < 140
    yy, xx = np.nonzero(mask)
    if len(xx) < 15:
        return None
    cropped = mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1].astype(np.uint8)
    scale = 52 / max(cropped.shape)
    glyph = cv2.resize(
        cropped,
        (max(1, round(cropped.shape[1] * scale)), max(1, round(cropped.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    out = np.zeros((64, 64), np.uint8)
    y, x = (64 - glyph.shape[0]) // 2, (64 - glyph.shape[1]) // 2
    out[y : y + glyph.shape[0], x : x + glyph.shape[1]] = glyph
    return out.astype(bool)


def _digit_match(actual, template):
    if actual is None or template is None:
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    a = cv2.dilate(actual.astype(np.uint8), kernel).astype(bool)
    b = cv2.dilate(template.astype(np.uint8), kernel).astype(bool)
    return min(
        np.count_nonzero(actual & b) / max(1, actual.sum()), np.count_nonzero(template & a) / max(1, template.sum())
    )


def possible_traces(p, direction):
    """The first yellow phase can follow red or green; preserve both possibilities.

    Neither an arbitrary yellow->red convention nor the reference trace is a
    rule. A complete observed trace must match one consistent hidden phase.
    """
    yellow = p["enable_yellow_light"]
    cycle = ["red", "yellow", "green", "yellow"] if yellow else ["red", "green"]
    start = p["initial_state"][f"light_{direction}_state"]
    countdown = p["initial_state"][f"light_{direction}_countdown"]
    elapsed = p["task_data"]["time_elapsed"]
    traces = []
    for phase in [i for i, state in enumerate(cycle) if state == start]:
        remaining = countdown
        trace = [(start, remaining)]
        for _ in range(elapsed):
            remaining -= 1
            if remaining <= 0:
                phase = (phase + 1) % len(cycle)
                remaining = p[f"{cycle[phase]}_duration"]
            trace.append((cycle[phase], remaining))
        traces.append(trace)
    return traces


def verify(task, frames, first, final, metadata, reference):
    if task.split("_", 1)[0] != "O-52":
        return None
    p = metadata["parameters"]
    if p.get("show_countdown_at_zero"):
        raise ValueError("Visible zero-countdown transition convention requires a separate observation contract")
    layers = _layers(metadata)
    circles = [layer for layer in layers if layer.get("kind") == "circle"]
    boxes = [layer for layer in layers if layer.get("kind") == "rect" and layer.get("fill_rgb") == [255, 255, 255]]
    center = np.array([first.shape[1] / 2, first.shape[0] / 2])
    lights = {}
    for circle in circles:
        point = np.array([circle["cx_px"], circle["cy_px"]])
        dx, dy = point - center
        direction = ("east" if dx > 0 else "west") if abs(dx) > abs(dy) else ("south" if dy > 0 else "north")
        box = min(
            boxes,
            key=lambda b: abs(b["x_px"] + b["w_px"] / 2 - point[0]) + abs(b["y_px"] - point[1] - circle["radius_px"]),
        )
        lights[direction] = (circle, tuple(round(box[k]) for k in ("x_px", "y_px", "w_px", "h_px")))
    active = p["task_data"]["active_directions"]
    if set(lights) != set(active):
        raise ValueError("Reference active-light observation geometry contradicts task")
    # Read numeral templates from labelled reference observations, then ignore
    # the reference's claimed transition solution when judging candidates.
    templates = {}
    records = p["state_sequence"]
    if not reference:
        raise ValueError("Countdown glyph templates unavailable")
    for index, state in enumerate(records):
        frame = reference[min(index * p["frames_per_state"], len(reference) - 1)]
        for direction, (_, box) in lights.items():
            value = state[f"light_{direction}_countdown"]
            template = _digit(frame, box)
            if template is not None:
                templates[value] = template
    if not templates:
        raise ValueError("No readable countdown templates")
    traces = {direction: possible_traces(p, direction) for direction in active}
    # Validate the benchmark's reference labels against the independent state
    # machine. A contradictory reference is an error, never a model failure.
    for direction in active:
        labelled = [(state[f"light_{direction}_state"], state[f"light_{direction}_countdown"]) for state in records]
        if labelled not in traces[direction]:
            raise ValueError(f"Reference traffic trace violates the declared cycle: {direction}")
    allowed = np.zeros(first.shape[:2], np.uint8)
    for circle, box in lights.values():
        cv2.circle(allowed, (round(circle["cx_px"]), round(circle["cy_px"])), round(circle["radius_px"] + 4), 1, -1)
        x, y, w, h = box
        allowed[y + 3 : y + h - 3, x + 3 : x + w - 3] = 1
    observed = []
    for frame in frames:
        state = []
        for direction in sorted(active):
            circle, box = lights[direction]
            cx, cy = round(circle["cx_px"]), round(circle["cy_px"])
            patch = frame[cy - 12 : cy + 13, cx - 12 : cx + 13]
            values = [(float(_mask(patch, color, 40).mean()), name) for name, color in COLORS.items()]
            coverage, color = max(values)
            glyph = _digit(frame, box)
            scores = sorted((_digit_match(glyph, template), value) for value, template in templates.items())
            score, value = scores[-1]
            if coverage < 0.9 or score < 0.94 or (len(scores) > 1 and score - scores[-2][0] < 0.025):
                return _result(
                    {"light_color_and_countdown_readable": False},
                    {
                        "direction": direction,
                        "color_coverage": coverage,
                        "glyph_score": float(score),
                        "glyph_margin": float(score - scores[-2][0]) if len(scores) > 1 else None,
                        "observed_color": color,
                        "countdown": int(value),
                    },
                )
            state.append((color, value))
        if not observed or observed[-1] != state:
            observed.append(state)
    checks = {"exact_number_of_simulated_seconds": len(observed) == p["task_data"]["time_elapsed"] + 1}
    for i, direction in enumerate(sorted(active)):
        actual = [state[i] for state in observed]
        checks[f"{direction}_legal_countdown_and_cycle"] = actual in traces[direction]
    checks["road_housings_and_inactive_lights_preserved"], worst = _static(frames, first, allowed.astype(bool))
    return _result(
        checks,
        {
            "observed_states": observed,
            "unexpected_static_component": worst,
            "initial_yellow_phase_ambiguity_preserved": True,
        },
    )
