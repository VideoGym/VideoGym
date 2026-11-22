"""Codec-tolerant observations for task-owned semantic state verifiers.

These functions compare visible evidence; they never render a replacement
candidate, read a reference answer, or choose a model sample by its score.
"""

import cv2
import numpy as np
from typing import List, Optional, Sequence


VERIFIER_PROTOCOL = "semantic_keyframes_v1"


def visual_features(image):
    """Suppress subpixel raster/quantization noise while retaining colors and ink."""
    return cv2.GaussianBlur(np.asarray(image, dtype=np.float32), (3, 3), 0.8)


def visual_delta(first, second):
    # A four-level color deadband is much smaller than the task palettes.
    return np.maximum(np.abs(visual_features(first) - visual_features(second)) - 4.0, 0.0)


def keyframe_indices(frames):
    """Keep observed state changes and endpoints, collapsing near-identical holds.

    A local 32-pixel region drives the decision so a small counter or marker
    cannot disappear in a whole-canvas average. There is no fixed frame quota.
    """
    if not len(frames):
        return []
    descriptors = [cv2.resize(visual_features(frame), (128, 128), interpolation=cv2.INTER_AREA)
                   for frame in frames]
    indices = [0]
    anchor = descriptors[0]
    for index in range(1, len(frames) - 1):
        delta = np.maximum(np.abs(descriptors[index] - anchor) - 2.0, 0.0).mean(axis=2)
        regional = delta.reshape(16, 8, 16, 8).mean(axis=(1, 3))
        if float(regional.max()) > 3.0:
            indices.append(index)
            anchor = descriptors[index]
    if len(frames) > 1:
        indices.append(len(frames) - 1)
    return indices


def ordered_state_observations(labels, state_count):
    """Require every readable task state in order, allowing transition ambiguity.

    Unclassified interpolation may bridge two consecutive states. It cannot
    stand in for a missing state, a reversed operation, or the final answer.
    """
    compact = []
    longest_unknown = unknown = 0
    for label in labels:
        if label < 0:
            unknown += 1
            longest_unknown = max(longest_unknown, unknown)
            continue
        unknown = 0
        if not compact or compact[-1] != label:
            compact.append(label)
    passed = bool(labels and labels[0] == 0 and labels[-1] == state_count - 1
                  and compact == list(range(state_count)) and longest_unknown <= 3)
    return compact, passed


def _compact_keyframe_sequence(indices: Sequence[Optional[int]]) -> List[int]:
    compact: List[int] = []
    for index in indices:
        if index is None:
            continue
        if compact and compact[-1] == index:
            continue
        compact.append(int(index))
    return compact


def _find_monotonic_keyframe_sequence(
    matching_indices_by_frame: Sequence[Sequence[int]],
    expected_count: int,
) -> Optional[List[int]]:
    paths_by_last: dict[int, List[int]] = {}
    for raw_matches in matching_indices_by_frame:
        matches = sorted({int(index) for index in raw_matches})
        if not matches:
            continue
        if not paths_by_last:
            paths_by_last = {index: [index] for index in matches}
            continue

        next_paths: dict[int, List[int]] = {}
        for previous_index, path in paths_by_last.items():
            for index in matches:
                if index < previous_index or index - previous_index > 1:
                    continue
                candidate = path if index == previous_index else [*path, index]
                existing = next_paths.get(index)
                candidate_rank = (candidate[0] == 0, len(set(candidate)), -len(candidate))
                existing_rank = (
                    (existing[0] == 0, len(set(existing)), -len(existing))
                    if existing is not None
                    else None
                )
                if existing is None or candidate_rank > existing_rank:
                    next_paths[index] = candidate
        paths_by_last = next_paths
        if not paths_by_last:
            return None

    expected_indices = set(range(expected_count))
    expected_last = expected_count - 1
    valid_paths = [
        path
        for path in paths_by_last.values()
        if path and path[0] == 0 and path[-1] == expected_last and set(path) >= expected_indices
    ]
    if not valid_paths:
        return None
    return min(valid_paths, key=lambda path: (len(path), path))
