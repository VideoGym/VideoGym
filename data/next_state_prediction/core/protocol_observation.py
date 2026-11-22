"""Read protocol clock fields locally instead of averaging them into a panel."""
from copy import deepcopy

import numpy as np

from data.common.verifiers.keyframes import ordered_state_observations
from data.common.verifiers.numeric_glyphs import numeric_glyph_descriptor, read_numeric_glyph
from data.next_state_prediction.core.rendering import render_state, stable_grid_layout
from data.next_state_prediction.core.semantics import replay_case


def protocol_clock_metrics(frames, structured):
    params, initial = structured['params'], structured['initial_state']
    trace = replay_case('protocol_handshake_tick', params, initial)
    left, top, width, height = stable_grid_layout('protocol_handshake_tick', 4, 6)

    def crop(frame, column):
        x, y = left+(column+.5)*width, top+1.5*height
        # Own the whole numeric field, excluding the fixed cell border.
        rx, ry = min(24,int(width*.35)), min(22,int(height*.35))
        return np.asarray(frame)[int(y)-ry:int(y)+ry+1,int(x)-rx:int(x)+rx+1]

    maximum = max(int(params['timeout_reset']), int(initial['timeout']),
                  int(initial['tick'])+int(params['steps']))
    vocabulary = {}
    for value in range(maximum+2):
        state = deepcopy(initial)
        state['timeout'] = value
        image = np.array(render_state('protocol_handshake_tick',state,params))
        descriptor = numeric_glyph_descriptor(crop(image,1))
        if descriptor is not None:
            vocabulary[value] = descriptor
    labels, readings, mismatches = [], [], []
    for index, frame in enumerate(frames):
        tick, _, _ = read_numeric_glyph(crop(frame,4), vocabulary)
        timeout, _, _ = read_numeric_glyph(crop(frame,1), vocabulary)
        state_index = tick-int(initial['tick']) if tick is not None else -1
        expected = trace[state_index]['timeout'] if 0 <= state_index < len(trace) else None
        valid = expected is not None and timeout == expected
        if tick is not None and timeout is not None and not valid:
            mismatches.append(dict(frame=index,tick=tick,timeout=timeout,expected_timeout=expected))
        readings.append(dict(tick=tick,timeout=timeout))
        labels.append(state_index if valid else -1)
    compact, ordered = ordered_state_observations(labels,len(trace))
    terminal = bool(labels and labels[-1] == len(trace)-1)
    return dict(video_protocol_clock_readings=readings,
                video_protocol_clock_mismatches=mismatches,
                video_protocol_clock_state_labels=compact,
                video_protocol_clock_process_pass=bool(ordered and not mismatches),
                video_protocol_clock_terminal_pass=terminal)
