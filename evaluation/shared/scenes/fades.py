"""Observe ordered native local fades without fixing their frame cadence."""

import numpy as np

from data.common.verifiers.transitions import complete_footprint_projection
from evaluation.shared.task_io import matches


def grade_fades(observed, states):
    """Each state is recomputed from public input, never loaded from a reference.

    Fit the complete changing footprint and inspect the entire reconstructed
    frame in local tiles. This rejects scans, extra marks, skipped commits and
    reversals while allowing arbitrary monotone opacity curves and holds.
    """
    states = [np.asarray(s, dtype=np.uint8) for s in states]
    if len(states) < 2:
        raise ValueError("a fade program needs at least one visible change")
    decoded = []
    intermediates = set()
    for frame in observed:
        hits = []
        for index, (before, after) in enumerate(zip(states, states[1:])):
            reading = complete_footprint_projection(frame, before, after)
            alpha = reading["progress"]
            if not -.06 <= alpha <= 1.06:
                continue
            fitted = np.rint(before.astype(float) * (1 - np.clip(alpha, 0, 1))
                            + after.astype(float) * np.clip(alpha, 0, 1)).astype(np.uint8)
            if not matches(frame, fitted):
                continue
            if .10 < alpha < .90:
                if reading["progress_spread"] > .35 or reading["linear_fit_error"] > .25:
                    continue
            residual = float(np.abs(frame.astype(float)-fitted.astype(float)).mean())
            hits.append((residual,index,float(np.clip(alpha,0,1))))
        if hits:
            # A small local change can lie inside the coarse appearance bound
            # for the preceding board too. Choose the best-fitting state,
            # not the earliest permissible state, without relaxing any bound.
            _,index,alpha = min(hits)
            decoded.append(index+alpha)
            if .10 < alpha < .90:
                intermediates.add(index)
        else:
            decoded.append(None)
    last = len(states) - 1
    final = bool(observed) and matches(observed[-1], states[-1])
    if final:
        terminal = complete_footprint_projection(observed[-1],states[-2],states[-1])
        final = terminal["progress"]>=.9 and terminal["progress_q10"]>=.75 and terminal["linear_fit_error"]<=.25
    known = [p for p in decoded if p is not None]
    process = bool(
        final and len(known) == len(observed) and len(observed) >= 5
        and max(known[:2]) <= .08 and min(known[-2:]) >= last - .08
        and all(b >= a - .06 for a, b in zip(known, known[1:]))
        and intermediates == set(range(last))
    )
    return {"whole_video_loaded": True, "whole_video_process_correct": float(process),
            "video_final_state_correct": float(final), "source_state_replayed": True,
            "visible_fade_stages": sorted(intermediates), "expected_fade_stages": last}
