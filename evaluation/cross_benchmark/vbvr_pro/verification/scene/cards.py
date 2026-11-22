from evaluation.cross_benchmark.vbvr_pro.verification.scene.motion_observations import _coverage, _mask
import cv2
import numpy as np


def _cards(frames, first, final, metadata, reference):
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result

    p = metadata["parameters"]
    specs = p["specs"]
    factor = 512 / first.shape[1]
    size = (512, round(first.shape[0] * factor))
    source = cv2.resize(reference[0] if reference else first, size, interpolation=cv2.INTER_AREA)
    target = cv2.resize(final, size, interpolation=cv2.INTER_AREA)
    frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
    colors = [obj["color_rgb"] for obj in specs]
    # Any valid final pose must show each paint at its matching outline. This
    # is deliberately weaker than the final rigid-shape/position checks below:
    # it rejects a missing or still-staged card before free-route fitting.
    target_masks = [_mask(target, color) for color in colors]
    if any(not np.any(mask) for mask in target_masks):
        return _result({"card_visual_dictionary_available": False})
    for index, (color, mask) in enumerate(zip(colors, target_masks)):
        nearby = cv2.dilate(mask, np.ones((17, 17), np.uint8))
        if not np.any(_mask(frames[-1], color) & nearby):
            return _result({f"card_{index}.visible_at_matching_outline": False})
    static = source.copy()
    colored = np.zeros(static.shape[:2], np.uint8)
    for color in colors:
        colored |= _mask(source, color)
    erase = cv2.dilate(colored, np.ones((9, 9), np.uint8))
    static[erase.astype(bool)] = 255
    templates = []
    for color, spec, mask in zip(colors, specs, target_masks):
        yy, xx = np.nonzero(mask)
        x0, x1 = max(0, int(xx.min()) - 4), min(size[0], int(xx.max()) + 5)
        y0, y1 = max(0, int(yy.min()) - 4), min(size[1], int(yy.max()) + 5)
        patch = target[y0:y1, x0:x1].copy()
        color_patch = mask[y0:y1, x0:x1]
        support = cv2.dilate(color_patch, np.ones((5, 5), np.uint8)).astype(bool)
        underlay = static[y0:y1, x0:x1].astype(np.float32)
        values = patch.astype(np.float32)
        paint = np.median(values[color_patch.astype(bool)], axis=0)
        direction = paint - underlay
        opacity = np.sum((values - underlay) * direction, axis=2) / np.maximum(1, np.sum(direction**2, axis=2))
        opacity = np.clip(opacity, 0, 1)
        opacity[~support] = 0
        # Recover the foreground against the known fixed target outline. A
        # template must not carry that grey outline along with the moving card.
        dark_edge = (np.max(values, axis=2) < 80) & support
        opacity[dark_edge] = 1
        foreground = np.clip(
            (values - (1 - opacity[..., None]) * underlay) / np.maximum(0.01, opacity[..., None]), 0, 255
        )
        opacity[opacity < 0.04] = 0
        pivot = np.array(spec["target"]) * factor - np.array([x0, y0])
        templates.append((foreground, color_patch, opacity, pivot))

    def render(centers, active, foreground):
        expected = static.copy()
        order = [i for i in range(len(specs)) if i != active] + [active] if foreground else range(len(specs))
        for index in order:
            patch, _, opacity, pivot = templates[index]
            center = centers[index]
            x, y = np.round(center - pivot).astype(int)
            h, w = patch.shape[:2]
            if x < 0 or y < 0 or x + w > size[0] or y + h > size[1]:
                return None
            roi = expected[y : y + h, x : x + w]
            roi[:] = np.round(patch * opacity[..., None] + roi * (1 - opacity[..., None])).astype(np.uint8)
        return expected

    checks = {}
    positions = []
    stages = []
    start = np.array([spec["start"] for spec in specs]) * factor
    goal = np.array([spec["target"] for spec in specs]) * factor
    for frame_index, frame in enumerate(frames):
        # These observations depend only on this frame, never on the trial
        # card pose or layer order. Re-reading them for every hypothesis is
        # particularly costly when a required local predicate cannot pass.
        observed = (
            [_mask(frame, color) for color in colors],
            (np.linalg.norm(frame.astype(float) - 255, axis=2) > 35).astype(np.uint8),
        )
        candidates = []
        for active in range(stages[-1] if stages else 0, len(specs)):
            centers = np.array([goal[i] if i < active else start[i] for i in range(len(specs))])
            _, mask, _, pivot = templates[active]
            response = cv2.matchTemplate(observed[0][active].astype(np.float32), mask.astype(np.float32), cv2.TM_CCORR)
            _, _, _, location = cv2.minMaxLoc(response)
            peak = float(response.max())
            choices = [location]
            choices += [
                (location[0] + dx, location[1] + dy)
                for dy, dx in [
                    (0, -1),
                    (0, 1),
                    (-1, 0),
                    (1, 0),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                    (0, -2),
                    (0, 2),
                    (-2, 0),
                    (2, 0),
                ]
            ]
            near = np.argwhere(response >= peak * 0.999)
            if len(near) > 1:
                choices += [
                    (int(x), int(y)) for y, x in near[:: max(1, len(near) // 12)][:12] if (int(x), int(y)) != location
                ]
            found = False
            for location in dict.fromkeys(choices):
                centers[active] = np.array(location) + pivot
                previous_expected, previous_local = None, None
                for foreground in (False, True):
                    expected = render(centers, active, foreground)
                    if expected is None:
                        continue
                    if previous_expected is not None and np.array_equal(expected, previous_expected):
                        local = previous_local
                    else:
                        local = _coverage(frame, expected, colors, observed=observed)
                    previous_expected, previous_local = expected, local
                    failed = sum(not value for value in local.values())
                    candidates.append((failed, active, centers.copy(), local))
                    if failed == 0:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not candidates:
            return _result({"all_cards_in_canvas": False})
        _, active, centers, local = min(candidates, key=lambda item: item[0])
        checks.update({f"frame_{frame_index}.{k}": v for k, v in local.items()})
        if not all(local.values()):
            # Every allowed pose and layer order for this frame has failed.
            # A later correct frame cannot repair this necessary condition.
            return _result(checks, {"first_unmatched_card_frame": frame_index})
        positions.append(centers)
        stages.append(active)
    positions = np.array(positions)
    checks["cards_start_in_staging_area"] = np.max(np.linalg.norm(positions[0] - start, axis=1)) <= 8
    checks["all_cards_finish_in_matching_outlines"] = np.max(np.linalg.norm(positions[-1] - goal, axis=1)) <= 8
    checks["cards_slide_in_prompted_order"] = all(b >= a for a, b in zip(stages, stages[1:]))
    for index, spec in enumerate(specs):
        distance = float(np.linalg.norm(goal[index] - start[index]))
        steps = np.linalg.norm(np.diff(positions[:, index], axis=0), axis=1)
        checks[f"card_{index}.no_teleport"] = np.max(steps, initial=0) <= max(20, distance * 0.35)
        checks[f"card_{index}.intermediate_slide_observed"] = bool(
            np.any(
                (np.linalg.norm(positions[:, index] - start[index], axis=1) > distance * 0.15)
                & (np.linalg.norm(positions[:, index] - goal[index], axis=1) > distance * 0.15)
            )
        )
    return _result(checks, {"active_card_indices": stages, "card_centers": positions.tolist()})
