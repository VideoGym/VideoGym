import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.vision.masks import anchored_foreground, ink, no_added_components, no_unexpected_ink
from evaluation.cross_benchmark.vbvr_pro.vision.objects import palette_objects


def _merge_balls(frames, first, metadata):
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result

    p = metadata["parameters"]
    objects = p["objects"]
    colors = [tuple(o["color_rgb"]) for o in objects]
    radius = p["ball_radius"]
    total = p["total_count"]
    red = next(i for i, o in enumerate(objects) if o["is_red"])
    initial = [o["initial_size"] for o in objects]
    states = []
    all_counts = []
    source_parts = palette_objects(first, colors, minimum=round(radius * radius * 0.6))
    sources = [
        [
            np.asarray(o.center)
            for o in source_parts
            if o.color == color
            and radius * 1.5 <= o.bbox[2] <= radius * 2.3
            and radius * 1.5 <= o.bbox[3] <= radius * 2.3
        ]
        for color in colors
    ]
    # TOTAL is the only fixed annotation. Cluster IDs/counts move with their
    # live clusters and disappear when absorbed; their old ink is not a
    # permanent foreground exemption.
    fixed_labels = np.zeros(first.shape[:2], np.uint8)
    fixed_labels[:60] = ink(first[:60])
    layers = metadata.get("generic_declarative_render", {}).get("first_frame", {}).get("layers", [])
    font = max((layer.get("font_size_px", 0) for layer in layers if layer.get("type") == "text"), default=0)
    font = font or radius * 1.85
    contacts, centers, process_errors = set(), [], []
    background_checks = {}
    previous_red, previous_targets = [], [[] for _ in colors]
    for frame_index, frame in enumerate(frames):
        components = palette_objects(frame, colors, minimum=16)
        balls = [
            o
            for o in components
            if radius * 1.5 <= o.bbox[2] <= radius * 2.3
            and radius * 1.5 <= o.bbox[3] <= radius * 2.3
            and o.pixels >= round(radius * radius * 0.6)
        ]
        counts = [sum(o.color == color for o in balls) for color in colors]
        all_counts.append(counts)
        footprint = fixed_labels.copy()
        for obj in balls:
            cv2.circle(footprint, tuple(np.round(obj.center).astype(int)), round(radius), 1, -1)
        neutral_text = ((np.ptp(frame.astype(np.int16), axis=2) <= 25) & (np.max(frame, axis=2) < 180)).astype(np.uint8)
        label_fields = np.zeros(frame.shape[:2], np.uint8)
        for color in colors:
            points = np.array([obj.center for obj in balls if obj.color == color])
            if not len(points):
                continue
            # The cluster ID/count label moves with its cluster. A target can
            # cover an outer column of red balls; allow one grid spacing of
            # uncertainty in the text anchor, then read only dark neutral ink.
            right = points[:, 0].max() + radius
            middle = (points[:, 1].min() + points[:, 1].max()) / 2
            spacing = radius * 2.5
            x0, x1 = max(0, round(right + 5)), min(frame.shape[1], round(right + spacing + font * 3.3))
            y0 = max(0, round(middle - spacing / 2 - 6))
            y1 = min(frame.shape[0], round(middle + spacing / 2 + font))
            label_fields[y0:y1, x0:x1] = 1
        footprint |= cv2.dilate(neutral_text * label_fields, np.ones((5, 5), np.uint8))
        complete = {id(obj) for obj in balls}
        for part in components:
            x, y, w, h = part.bbox
            if id(part) in complete or max(w, h) > radius * 2.3:
                continue
            if not np.any(label_fields[y : y + h, x : x + w] & part.mask):
                continue
            # A white ID/count card can cover most of a neighbouring ball.
            # Retain its small visible coloured arc only beside a recognized
            # ball of the same cluster and a current annotation field.
            for obj in balls:
                if obj.color != part.color:
                    continue
                ox, oy, ow, oh = obj.bbox
                gap = np.hypot(max(0, x - ox - ow, ox - x - w), max(0, y - oy - oh, oy - y - h))
                if gap <= radius * 0.75:
                    footprint[y : y + h, x : x + w] |= part.mask
                    # Complete a known-radius cap behind the white card. A
                    # second faint sliver can be too narrow for palette
                    # components after 512px chroma sampling. Candidate discs
                    # must contain the observed cap and not overlap a visible
                    # ball of the same cluster; no fixed lattice is assumed.
                    candidates = [
                        ((x + (w - 1) / 2), y + radius - 1),
                        ((x + (w - 1) / 2), y + h - radius),
                        (x + radius - 1, (y + (h - 1) / 2)),
                        (x + w - radius, (y + (h - 1) / 2)),
                    ]
                    py, px = np.nonzero(part.mask)
                    visible = np.c_[px + x, py + y]
                    same_cluster = [np.array(ball.center) for ball in balls if ball.color == part.color]
                    for center in map(np.array, candidates):
                        if np.max(np.linalg.norm(visible - center, axis=1)) > radius + 2:
                            continue
                        if any(np.linalg.norm(center - known) < radius * 1.7 for known in same_cluster):
                            continue
                        cv2.circle(footprint, tuple(np.round(center).astype(int)), round(radius), 1, -1)
                    break
        background_checks[f"frame_{frame_index}.no_isolated_added_content"] = no_added_components(
            ink(frame), anchored_foreground(ink(frame), footprint, radius=4), radius=4
        )[0]
        red_balls = [np.asarray(o.center) for o in balls if o.color == colors[red]]
        if not red_balls:
            process_errors.append({"frame": frame_index, "reason": "red_cluster_missing"})
            continue
        red_xy = np.asarray(red_balls)
        centers.append(((red_xy.min(axis=0) + red_xy.max(axis=0)) / 2).tolist())
        # Visible disks need not form a fixed square lattice: the instruction
        # permits rearrangement, and other clusters can hide a connecting disk.
        for i, color in enumerate(colors):
            if i == red:
                continue
            remaining = [np.asarray(o.center) for o in balls if o.color == color]
            # Contact is an actual disk-to-disk observation, independent of
            # the metadata's proposed absorption order or endpoint.
            if remaining and any(np.linalg.norm(a - b) <= radius * 2 + 7 for a in red_balls for b in remaining):
                contacts.add(i)
            if len(centers) > 1 and previous_red and len(remaining) < len(previous_targets[i]):
                # Sampling may place first contact between frames. Verify
                # that translating the observed red disks sweeps across the
                # target before its colour disappears.
                delta = np.subtract(centers[-1], centers[-2])
                denominator = float(delta @ delta)
                if denominator:
                    for source in previous_red:
                        for target in previous_targets[i]:
                            t = float(np.clip(np.dot(target - source, delta) / denominator, 0, 1))
                            if np.linalg.norm(target - (source + t * delta)) <= radius * 2 + 7:
                                contacts.add(i)
            if i not in contacts:
                still = len(remaining) == len(sources[i]) and all(
                    any(np.linalg.norm(a - b) <= 4 for b in sources[i]) for a in remaining
                )
                if not still:
                    process_errors.append(
                        {"frame": frame_index, "target": i, "reason": "target_moves_or_recolors_before_contact"}
                    )
            previous_targets[i] = remaining
        previous_red = red_balls
        if sum(counts) == total and all(counts[i] in {0, initial[i]} for i in range(len(counts)) if i != red):
            if not states or states[-1] != counts:
                states.append(counts)
    checks = {
        "starts_with_original_ball_inventory": bool(states and states[0] == initial),
        "visible_ball_inventory_never_exceeds_declared_total": all(sum(counts) <= total for counts in all_counts),
        **background_checks,
    }
    goal = [0] * len(objects)
    goal[red] = total
    checks["all_balls_red_at_end"] = all_counts[-1] == goal
    checks["every_absorption_has_a_completed_state"] = len(states) == len(objects)
    for k, (a, b) in enumerate(zip(states, states[1:])):
        changed = [i for i in range(len(objects)) if i != red and a[i] != b[i]]
        checks[f"merge_{k}.one_target_at_a_time"] = len(changed) == 1
        if len(changed) == 1:
            i = changed[0]
            checks[f"merge_{k}.target_strictly_smaller_than_red"] = 0 < a[i] < a[red] and b[i] == 0
            checks[f"merge_{k}.ball_count_conserved"] = b[red] == a[red] + a[i] and sum(a) == sum(b) == total
    checks["total_label_preserved"] = all(no_unexpected_ink(f[:60], first[:60])[0] for f in frames)
    # Final placement is free: any legal absorption order may finish at a
    # different target. Count and connected-cluster geometry determine truth.
    endpoint = palette_objects(frames[-1], [colors[red]], minimum=round(radius * radius * 0.6))
    endpoint = [
        o for o in endpoint if radius * 1.5 <= o.bbox[2] <= radius * 2.3 and radius * 1.5 <= o.bbox[3] <= radius * 2.3
    ]
    reached = {0} if endpoint else set()
    frontier = list(reached)
    while frontier:
        i = frontier.pop()
        for j in range(len(endpoint)):
            if j not in reached and np.linalg.norm(np.subtract(endpoint[i].center, endpoint[j].center)) < radius * 3.2:
                reached.add(j)
                frontier.append(j)
    checks["one_final_cluster"] = len(reached) == total
    movement = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.array([])
    checks["actual_contact_precedes_every_absorption"] = len(contacts) == len(objects) - 1
    checks["targets_wait_for_contact_and_red_cluster_stays_visible"] = not process_errors
    checks["complete_approach_process"] = np.count_nonzero(movement > 2) >= 2 * (len(objects) - 1)
    cluster_width = radius * (2.5 * (int(np.ceil(np.sqrt(total))) - 1) + 2)
    checks["bounded_inter_frame_cluster_motion"] = bool(len(movement)) and np.all(movement <= cluster_width)
    return _result(
        checks,
        {
            "persistent_ball_counts": states,
            "sampled_ball_counts": all_counts,
            "observed_red_centers": centers,
            "contacted_targets": sorted(contacts),
            "process_errors": process_errors,
            "inter_frame_motion_limit_px": cluster_width,
        },
    )
