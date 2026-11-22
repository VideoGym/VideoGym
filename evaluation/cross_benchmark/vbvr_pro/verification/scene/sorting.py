from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _match_scene, _palette, _result
import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.vision.masks import anchored_foreground, ink, mask_coverage, no_added_components
from evaluation.cross_benchmark.vbvr_pro.vision.objects import palette_objects
from evaluation.cross_benchmark.vbvr_pro.vision.shapes import normalized


def _sorting(task, frames, first, final, metadata):
    palette = _palette(metadata)
    checks, evidence, actual, expected, matches = _match_scene(frames[-1], final, palette, position=False, minimum=150)
    semantic = metadata.get("semantic_ground_truth", {})
    declared = semantic.get("objects", []) if task == "G-3" else semantic.get("circles", [])
    if declared:
        checks["declared_object_inventory"] = len(actual) == len(expected) == len(declared)
    if matches:
        allowed = np.zeros(frames[-1].shape[:2], np.uint8)
        for obj in actual:
            x, y, w, h = obj.bbox
            allowed[y : y + h, x : x + w] |= obj.silhouette
        reference_allowed = np.zeros_like(allowed)
        for obj in expected:
            x, y, w, h = obj.bbox
            reference_allowed[y : y + h, x : x + w] |= obj.silhouette
        reference_ink = ink(final)
        distances = cv2.distanceTransform(1 - reference_allowed, cv2.DIST_L2, 5)
        outline_radius = max(3, min(12, int(np.ceil(np.quantile(distances[reference_ink.astype(bool)], 0.999))) + 1))
        allowed = cv2.dilate(allowed, np.ones((2 * outline_radius + 1, 2 * outline_radius + 1), np.uint8))
        foreground = ink(frames[-1])
        rogue = int(np.count_nonzero(foreground & (1 - allowed)))
        checks["no_extra_scene_content"] = rogue <= max(20, int(foreground.sum() * 0.005))
        # The layout and the route are free. Read the actual object footprints
        # in every frame, retaining only the declared number of each colour.
        # A small additional object cannot borrow the foreground denominator.
        counts = {color: sum(obj.color == color for obj in expected) for color in palette}
        for index, frame in enumerate(frames):
            scale = min(1, 512 / frame.shape[1])
            if scale < 1:
                frame = cv2.resize(frame, (512, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            parts = palette_objects(frame, palette, minimum=max(12, round(150 * scale**2)))
            footprint = np.zeros(frame.shape[:2], np.uint8)
            for color, count in counts.items():
                for obj in sorted((o for o in parts if o.color == color), key=lambda o: o.pixels, reverse=True)[:count]:
                    x, y, w, h = obj.bbox
                    footprint[y : y + h, x : x + w] |= obj.silhouette
            boundary = max(2, round(outline_radius * scale) + 1)
            foreground = ink(frame)
            footprint = anchored_foreground(foreground, footprint, radius=boundary)
            checks[f"frame_{index}.no_isolated_added_content"] = no_added_components(
                foreground, footprint, radius=boundary
            )[0]
        order = sorted(actual, key=lambda o: o.center[0])
        checks["single_horizontal_line"] = np.ptp([o.center[1] for o in order]) <= 12
        checks["no_overlap"] = all(a.bbox[0] + a.bbox[2] <= b.bbox[0] + 3 for a, b in zip(order, order[1:]))
        if task == "G-174":
            checks["descending_circumference"] = all(a.bbox[2] >= b.bbox[2] - 2 for a, b in zip(order, order[1:]))
            left, right = order[0].bbox[0], order[-1].bbox[0] + order[-1].bbox[2]
            checks["centered_row"] = abs((left + right) / 2 - final.shape[1] / 2) <= 16
        else:
            # Equal shape types may be grouped in either order unless the prompt
            # prescribes a type order. Size within every group must ascend.
            identity = {j: i for i, j in matches}
            expected_by_x = sorted(range(len(expected)), key=lambda i: expected[i].center[0])
            rank = {i: k for k, i in enumerate(expected_by_x)}
            types = {}
            for i in expected_by_x:
                same = next(
                    (
                        group
                        for group in types.values()
                        if group
                        and min(
                            mask_coverage(normalized(expected[i].silhouette), normalized(expected[group[0]].silhouette))
                        )
                        > 0.975
                    ),
                    None,
                )
                if same is None:
                    types[len(types)] = [i]
                else:
                    same.append(i)
            group_for = {i: g for g, ids in types.items() for i in ids}
            if declared and all("type" in obj and "final_position" in obj for obj in declared):
                group_for = {
                    i: min(declared, key=lambda obj: np.linalg.norm(np.subtract(e.center, obj["final_position"])))[
                        "type"
                    ]
                    for i, e in enumerate(expected)
                }
                types = {
                    name: [i for i, group in group_for.items() if group == name] for name in set(group_for.values())
                }
            observed_index = {id(o): j for j, o in enumerate(actual)}
            ids = [identity[observed_index[id(o)]] for o in order]
            groups = [group_for[i] for i in ids]
            checks["same_shapes_consecutive"] = len(
                [g for k, g in enumerate(groups) if k == 0 or groups[k - 1] != g]
            ) == len(types)
            checks["ascending_size_within_type"] = all(
                rank[a] < rank[b] for a, b in zip(ids, ids[1:]) if group_for[a] == group_for[b]
            )
        # Appearance must hold at the endpoints; legal layouts need not copy
        # the reference's exact spacing or the arbitrary group order.
        checks["objects_started_in_original_scene"] = _match_scene(frames[0], first, palette)[0]["exact_object_count"]
    return _result(checks, evidence)
