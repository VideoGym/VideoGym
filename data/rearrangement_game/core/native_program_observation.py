"""Read task-owned visual roles before replaying a submitted video program."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from data.rearrangement_game.core.source_replay import _new_generator, source_record


def observe_meal_program(sample, frames, shortest_length):
    """Recover actual platter-to-seat transfers, independent of marker strokes.

    Bronze foreground inside a visible plate identifies the platter's seat.
    It need not sit at the renderer's exact plate centre. Full video replay
    separately checks the carried object, serving updates and final scene.
    """
    record = source_record(sample)
    initial, target = record["initial_state"], record["target_state"]
    with tempfile.TemporaryDirectory(prefix="videogym-meal-observation-") as tmp:
        generator = _new_generator(sample.task_name, Path(tmp), video=False)
        panel = generator._panel_geometry()
        layout = generator._layout(panel["panel_w"], panel["panel_h"])
        count = generator._seat_count(initial)
        centres = [
            (panel["left_panel_x0"] + generator._seat_position(layout, seat, count)[0],
             panel["left_panel_y0"] + generator._seat_position(layout, seat, count)[1])
            for seat in range(count)
        ]
        radius = layout["plate_r"]
        platter_radius = radius * .35
        crop = int(np.ceil(platter_radius * 1.4))
        yy, xx = np.mgrid[-crop:crop+1, -crop:crop+1]
        # This interior lies within the visible plate and away from the table
        # rim and demand dots. Any bronze support here identifies the seat;
        # no exact centre or renderer stroke match is imposed.
        plate_interior = xx*xx + yy*yy <= crop*crop
        body = np.asarray((200, 152, 80), dtype=np.float32)
        other_colours = np.asarray([
            (236, 196, 132), (160, 116, 60), (224, 170, 70),
            (252, 252, 248), (224, 244, 228), (244, 240, 232),
        ], dtype=np.float32)
        seats, commit_frames = [], []
        for index, frame in enumerate(frames):
            hits = []
            for seat, (x, y) in enumerate(centres):
                ix, iy = int(round(x)), int(round(y))
                pixels = frame[iy-crop:iy+crop+1, ix-crop:ix+crop+1].astype(np.float32)
                distance = np.linalg.norm(pixels-body, axis=2)
                other = np.minimum.reduce([
                    np.linalg.norm(pixels-colour, axis=2) for colour in other_colours
                ])
                bronze = (distance < 55) & (distance+4 < other) & plate_interior
                if np.count_nonzero(bronze) >= 8:
                    hits.append(seat)
            if len(hits) == 1 and (not seats or seats[-1] != hits[0]):
                seats.append(hits[0])
                commit_frames.append(index)
        current = initial
        plan = []
        legal = bool(seats and seats[0] == initial["current_seat"])
        for before, after in zip(seats, seats[1:]):
            if current["current_seat"] != before:
                legal = False
                break
            delta = (after-before) % count
            move = "CW" if delta == 1 else "CCW" if delta == count-1 else None
            if move is None:
                legal = False
                break
            current = generator._apply_move(current, move)
            plan.append(move)
        complete = bool(legal and generator._state_signature(current) == generator._state_signature(target))
    optimal = shortest_length is None or len(plan) == shortest_length
    accepted = bool(complete and optimal)
    return (plan if accepted else None), {
        "video_observed_plan_parser": "meal_source_platter_plate_membership_v1",
        "video_observed_commit_frames": commit_frames,
        "video_observed_platter_seats": seats,
        "video_observed_plan": plan,
        "video_observed_plan_legal": accepted,
        "video_observed_plan_legal_complete": complete,
        "video_observed_plan_requires_shortest": shortest_length is not None,
        "video_observed_plan_optimal_when_required": bool(optimal),
    }


def observe_frog_program(sample, frames, shortest_length):
    """Read marker occupancy on the visible pads, including illegal states.

    The direction pigments identify frogs independently of the similar green
    pad/body fills. Every complete observed configuration must be the current
    state or one legal successor; missing intermediate states are not solved
    or interpolated into the candidate.
    """
    from data.rearrangement_game.jumping_frog_swap.generator import (
        GREEN_FROG_DARK, ORANGE_FROG_DARK, GREEN_FROG_COLOR,
        ORANGE_FROG_COLOR, LILYPAD_FILL, _legal_successors, _validate_state,
    )

    record = source_record(sample)
    initial, target = _validate_state(record["initial_state"]), _validate_state(record["target_state"])
    with tempfile.TemporaryDirectory(prefix="videogym-frog-observation-") as tmp:
        generator = _new_generator(sample.task_name, Path(tmp), video=False)
        panel = generator._panel_geometry()
        layout = generator._layout(panel["panel_w"], panel["panel_h"], len(initial))
        radius = layout["slot_w"] * .40 * 2.45 * .34
        centres = [(panel["left_panel_x0"] + generator._slot_x(layout, index),
                    panel["left_panel_y0"] + layout["center_y"]) for index in range(len(initial))]
    dx, dy = max(3, round(radius*.58)), max(2, round(radius*.34))
    pigments = np.asarray([GREEN_FROG_DARK, ORANGE_FROG_DARK, GREEN_FROG_COLOR,
                           ORANGE_FROG_COLOR, LILYPAD_FILL], dtype=np.float32)
    states, commit_frames = [], []
    for index, frame in enumerate(frames):
        state = []
        for x, y in centres:
            cx, cy = round(x), round(y)
            pixels = frame[cy-dy:cy+dy+1, cx-dx:cx+dx+1].astype(np.float32)
            distances = np.linalg.norm(pixels[:, :, None, :]-pigments, axis=3)
            marker = distances[:, :, :2].min(axis=2)
            alternative = distances[:, :, 2:].min(axis=2)
            supported = marker+5 < alternative
            votes = [int(np.count_nonzero(supported & (distances[:, :, kind] <= distances[:, :, 1-kind])))
                     for kind in (0, 1)]
            kind = int(np.argmax(votes))
            state.append(("G", "O")[kind] if votes[kind] >= 3 else ".")
        state = tuple(state)
        complete = (state.count(".") == 1 and state.count("G") == initial.count("G")
                    and state.count("O") == initial.count("O"))
        if complete and (not states or states[-1] != state):
            states.append(state)
            commit_frames.append(index)
    current = initial
    plan = []
    legal = bool(states and states[0] == initial)
    invalid_transition = None
    for state in states[1:]:
        successors = [(following, move) for following, move in _legal_successors(current) if following == state]
        if len(successors) != 1:
            legal = False
            invalid_transition = {"before": list(current), "after": list(state)}
            break
        current, move = successors[0]
        plan.append(move)
    complete = bool(legal and current == target)
    optimal = shortest_length is None or len(plan) == shortest_length
    accepted = bool(complete and optimal)
    return (plan if accepted else None), {
        "video_observed_plan_parser": "frog_source_pad_marker_occupancy_v1",
        "video_observed_commit_frames": commit_frames,
        "video_observed_frog_states": [list(state) for state in states],
        "video_observed_frog_invalid_transition": invalid_transition,
        "video_observed_plan": plan,
        "video_observed_plan_legal": accepted,
        "video_observed_plan_legal_complete": complete,
        "video_observed_plan_requires_shortest": shortest_length is not None,
        "video_observed_plan_optimal_when_required": bool(optimal),
    }
