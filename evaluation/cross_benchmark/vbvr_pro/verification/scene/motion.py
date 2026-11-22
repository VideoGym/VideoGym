from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _palette, _result
import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.vision.bandwidth import (
    complete_support_geometry,
    match_absence,
    match_local,
    register_resize,
    register_equivalent,
)
from evaluation.cross_benchmark.vbvr_pro.vision.masks import ink, mask_coverage, no_added_components, no_unexpected_ink
from evaluation.cross_benchmark.vbvr_pro.vision.objects import object_checks, palette_objects
from evaluation.cross_benchmark.vbvr_pro.vision.shapes import normalized


def _motion(task, frames, final, metadata, reference):
    """Validate the rigid-motion state machine with independently decoded poses.

    Pose order is geometric, not a timestamp requirement. Holding or changing
    speed is legal. Reference poses also describe legitimate occlusions; an
    object may be invisible only in a pose where it is actually covered.
    """
    palette = _palette(metadata)
    if task == "O-6":
        palette = [metadata["parameters"]["object_color_rgb"]]
    if not reference or len(reference) < 3:
        return _result({"rigid_motion_pose_reference_available": False})
    if final.shape[1] > 512:
        width = 512
        height = round(final.shape[0] * width / final.shape[1])
        frames = [cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA) for f in frames]
        original_reference, original_final = reference, final
        reference = [cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA) for f in reference]
        final = cv2.resize(final, (width, height), interpolation=cv2.INTER_AREA)
        for original, resized in zip(original_reference, reference):
            register_resize(original, resized)
        register_resize(original_final, final)

    def objects(frame):
        components = palette_objects(frame, palette, minimum=150)
        # Chroma subsampling can produce a small dark fringe labelled as another
        # palette colour. A filled semantic object occupies thousands of pixels;
        # retain the dominant component for each distinct object colour.
        return {
            tuple(c): max((o for o in components if o.color == tuple(c)), key=lambda o: o.pixels, default=None)
            for c in palette
        }

    raw_poses = [objects(f) for f in reference]
    poses = []
    pose_frames = []
    for pose, reference_frame in zip(raw_poses, reference):
        same = bool(poses)
        if poses:
            for color in map(tuple, palette):
                a, b = pose[color], poses[-1][color]
                if a is None or b is None:
                    same &= (a is None) == (b is None)
                else:
                    same &= (
                        np.linalg.norm(np.subtract(a.center, b.center)) <= 1.2
                        and min(mask_coverage(normalized(a.mask), normalized(b.mask))) >= 0.985
                    )
        if not same:
            poses.append(pose)
            pose_frames.append(reference_frame)
        else:
            register_equivalent(pose_frames[-1], reference_frame)
    observed = [objects(f) for f in frames]
    goal = objects(final)
    checks = {}
    evidence = {}
    for color in map(tuple, palette):
        a, e = observed[-1][color], goal[color]
        prefix = "object_" + "_".join(map(str, color))
        checks[prefix + ".present_at_goal"] = a is not None and e is not None
        if a is not None and e is not None:
            local = object_checks(a, e)
            local["complete_foreground_support"] = complete_support_geometry(frames[-1], final, a, e)
            if not all(local.values()) and match_local(frames[-1], final, e, palette, actual_object=a):
                local = {name: True for name in local}
            checks.update({prefix + ".final_" + k: v for k, v in local.items()})
    checks["no_unrequested_final_edits"] = no_unexpected_ink(frames[-1], final)[0]

    # A reference video's finite frame spacing defines a geometric quantisation
    # tolerance; it does not require the generated video to share the timing.
    position_tolerance = {}
    for color in map(tuple, palette):
        centers = [np.array(pose[color].center) for pose in poses if pose[color] is not None]
        steps = [float(np.linalg.norm(b - a)) for a, b in zip(centers, centers[1:])]
        position_tolerance[color] = max(12, (max(steps) if steps else 0) * 0.6 + 3)

    stages = []
    for frame_index, state in enumerate(observed):
        costs = []
        for pose in poses:
            cost = 0.0
            for color in map(tuple, palette):
                a, e = state[color], pose[color]
                if a is None or e is None:
                    cost += 0 if a is None and e is None else 12
                    continue
                distance = np.linalg.norm(np.subtract(a.center, e.center)) / position_tolerance[color]
                precision, recall = mask_coverage(normalized(a.mask), normalized(e.mask))
                cost += distance + 5 * (2 - precision - recall) + abs(np.log(a.pixels / e.pixels))
            costs.append(cost)
        stage = int(np.argmin(costs))
        stages.append(stage)
        pose = poses[stage]
        # Stationary targets, tracking borders and other scene content must
        # agree with the chosen semantic pose as well as the dominant objects.
        tolerance = round(max(position_tolerance.values(), default=10)) + 2
        preservation = mask_coverage(ink(frames[frame_index]), ink(pose_frames[stage]), radius=tolerance)
        checks[f"frame_{frame_index}.other_scene_content_preserved"] = min(preservation) >= 0.98
        checks[f"frame_{frame_index}.no_isolated_added_content"] = no_added_components(
            ink(frames[frame_index]), ink(pose_frames[stage]), radius=tolerance
        )[0]
        for color in map(tuple, palette):
            a, e = state[color], pose[color]
            prefix = "frame_%02d.object_%s" % (frame_index, "_".join(map(str, color)))
            checks[prefix + ".visibility_matches_a_valid_pose"] = (a is None) == (e is None)
            if a is None or e is None:
                continue
            checks[prefix + ".position"] = np.linalg.norm(np.subtract(a.center, e.center)) <= position_tolerance[color]
            checks[prefix + ".area"] = abs(a.pixels / e.pixels - 1) <= 0.20
            precision, recall = mask_coverage(normalized(a.mask), normalized(e.mask), radius=3)
            checks[prefix + ".oriented_shape"] = min(precision, recall) >= 0.94
            checks[prefix + ".color"] = np.linalg.norm(np.subtract(a.observed_rgb, e.observed_rgb)) <= 65
            checks[prefix + ".complete_foreground_support"] = complete_support_geometry(
                frames[frame_index], pose_frames[stage], a, e
            )
            names = [name for name in checks if name.startswith(prefix + ".")]
            if not all(checks[name] for name in names) and match_local(
                frames[frame_index], pose_frames[stage], e, palette
            ):
                for name in names:
                    checks[name] = True
        object_names = [name for name in checks if name.startswith(f"frame_{frame_index:02d}.object_")]
        if object_names and not all(checks[name] for name in object_names):
            # A pale object's colour fragments can make the raw pose ranking
            # choose the wrong dictionary entry. Validate one complete labelled
            # pose locally; objects may not borrow appearances from different
            # poses, and the temporal obligations below still apply.
            candidates = sorted(range(len(poses)), key=lambda index: abs(index - stage))
            if len(stages) > 1:
                candidates = [stages[-2]] + [index for index in candidates if index != stages[-2]]
            for alternative in candidates:
                trial = poses[alternative]
                proven = True
                for color in map(tuple, palette):
                    expected_object = trial[color]
                    if expected_object is None:
                        proven &= state[color] is None or match_absence(
                            frames[frame_index], pose_frames[alternative], state[color]
                        )
                    else:
                        proven &= match_local(
                            frames[frame_index],
                            pose_frames[alternative],
                            expected_object,
                            palette,
                            actual_object=state[color],
                        )
                    if not proven:
                        break
                if not proven:
                    continue
                stages[-1] = alternative
                for name in object_names:
                    checks[name] = True
                tolerance = round(max(position_tolerance.values(), default=10)) + 2
                allowed = ink(pose_frames[alternative])
                observed_ink = ink(frames[frame_index])
                preservation = mask_coverage(observed_ink, allowed, radius=tolerance)
                checks[f"frame_{frame_index}.other_scene_content_preserved"] = min(preservation) >= 0.98
                checks[f"frame_{frame_index}.no_isolated_added_content"] = no_added_components(
                    observed_ink, allowed, radius=tolerance
                )[0]
                break
    checks["starts_in_initial_pose"] = stages[0] <= 2
    checks["finishes_in_goal_pose"] = stages[-1] >= len(poses) - 3
    checks["pose_order_obeys_requested_motion"] = all(b >= a - 2 for a, b in zip(stages, stages[1:]))
    checks["no_teleport_between_poses"] = all(
        abs(b - a) <= max(4, len(poses) * 0.22) for a, b in zip(stages, stages[1:])
    )
    checks["intermediate_motion_observed"] = len(set(stages)) >= 4 and any(
        0.15 * len(poses) < k < 0.85 * len(poses) for k in stages
    )
    evidence["motion_pose_indices"] = stages
    evidence["position_tolerance_px"] = {str(k): v for k, v in position_tolerance.items()}
    return _result(checks, evidence)
