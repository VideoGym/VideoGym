from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _blobs, _mask, _result, _static
from collections import deque
import cv2
import numpy as np


def _graph(frames, first, p, final):
    nodes = np.array(p["nodes"], float)
    edges = [tuple(pair) for pair in p["edges"]]
    start, end = p["start_node"], p["end_node"]
    distance, queue = {start: 0}, deque([start])
    while queue:
        node = queue.popleft()
        for source, target in edges:
            if source == node and target not in distance:
                distance[target] = distance[node] + 1
                queue.append(target)
    if end not in distance:
        raise ValueError("Invalid benchmark: unreachable graph goal")
    centers, sizes = [], []
    triangles = True
    for frame in frames:
        agent_mask = _mask(frame, p["agent_fill_rgb"])
        blobs = _blobs(agent_mask, 25)
        if len(blobs) != 1:
            return _result({"exactly_one_agent_each_frame": False}, {"observed_agent_count": len(blobs)})
        centers.append(blobs[0][0])
        sizes.append(blobs[0][1])
        contours, _ = cv2.findContours(agent_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        triangle = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True)
        triangles &= len(triangle) == 3
    centers = np.array(centers)
    # A triangle's centroid is slightly offset from its declared rendering pivot.
    pivot_offset = centers[0] - nodes[start]
    centers = centers - pivot_offset
    tolerance = max(5.0, p["agent_size"] * 0.25)
    visits = []
    for point in centers:
        ds = np.linalg.norm(nodes - point, axis=1)
        index = int(ds.argmin())
        if ds[index] <= tolerance and (not visits or visits[-1] != index):
            visits.append(index)
    on_edges = []
    for point in centers:
        on_edges.append(
            any(
                np.linalg.norm(
                    point
                    - (
                        nodes[a]
                        + np.clip(
                            np.dot(point - nodes[a], nodes[b] - nodes[a])
                            / np.dot(nodes[b] - nodes[a], nodes[b] - nodes[a]),
                            0,
                            1,
                        )
                        * (nodes[b] - nodes[a])
                    )
                )
                <= tolerance
                for a, b in edges
            )
        )
    # Between successive node visits the trajectory must advance along that
    # directed segment; proximity to some other edge alone is insufficient.
    segments = []
    current, progress = start, 0.0
    legal = True
    for point in centers:
        if np.linalg.norm(point - nodes[current]) <= tolerance:
            continue
        candidates = []
        for source, target in edges:
            if source != current:
                continue
            vector = nodes[target] - nodes[source]
            alpha = np.dot(point - nodes[source], vector) / np.dot(vector, vector)
            offset = np.linalg.norm(point - (nodes[source] + alpha * vector))
            if -0.03 <= alpha <= 1.03 and offset <= tolerance:
                candidates.append((offset, target, float(alpha)))
        if not candidates:
            legal = False
            break
        _, target, alpha = min(candidates)
        if segments and segments[-1][0] == current and segments[-1][1] != target:
            legal = False
            break
        if alpha < progress - 0.04:
            legal = False
        if not segments or segments[-1][:2] != (current, target):
            segments.append((current, target))
        progress = max(progress, alpha)
        if np.linalg.norm(point - nodes[target]) <= tolerance:
            current, progress = target, 0.0

    # Recover the static scene hidden by the initial agent from the unoccupied
    # start node in the trusted final image. Its footprint must not remain an
    # unchecked region after the agent leaves.
    def agent_footprint(frame):
        fill = _mask(frame, p["agent_fill_rgb"])
        outline = _mask(frame, p.get("agent_outline_rgb", p["agent_fill_rgb"]))
        # The opaque fill excludes the declared dark border. Its tip can be
        # eight canonical pixels from the fill after 512px colour sampling.
        nearby = cv2.dilate(fill.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        return fill | (outline & nearby)

    initial_agent = cv2.dilate(agent_footprint(first).astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    empty = first.copy()
    empty[initial_agent] = final[initial_agent]
    static, worst = True, 0
    for frame in frames:
        # A triangle's corners extend beyond a disk about its rendering pivot.
        moving = agent_footprint(frame)
        allowed = cv2.dilate(moving.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
        preserved, error = _static([frame], empty, allowed)
        static &= preserved
        worst = max(worst, error)
    checks = {
        "agent_reaches_goal": np.linalg.norm(centers[-1] - nodes[end]) <= tolerance,
        "directed_edge_walk": legal and all(on_edges),
        "shortest_number_of_edges": len(segments) == distance[end],
        "agent_identity_preserved": min(sizes) >= max(sizes) * 0.72,
        "agent_remains_triangular": triangles,
        "graph_preserved": static,
    }
    return _result(
        checks,
        {
            "visited_nodes": visits,
            "traversed_edges": segments,
            "minimum_edges": distance[end],
            "unexpected_static_component": worst,
        },
    )
