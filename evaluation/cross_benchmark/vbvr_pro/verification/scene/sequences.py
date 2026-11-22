import cv2
import numpy as np
from evaluation.cross_benchmark.vbvr_pro.vision.bandwidth import complete_support_geometry, original_image
from evaluation.cross_benchmark.vbvr_pro.vision.masks import color_mask, ink, mask_coverage, no_unexpected_ink
from evaluation.cross_benchmark.vbvr_pro.vision.objects import palette_objects


def _complete_sequence(frames, first, final, metadata, reference):
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _match_scene, _result

    p = metadata["parameters"]
    n = p["sequence_length"]
    margin = 80 * final.shape[1] / 1024
    width = (final.shape[1] - 2 * margin) / n
    cells = [
        (margin + i * width, final.shape[0] * 0.25, margin + (i + 1) * width, final.shape[0] * 0.75) for i in range(n)
    ]
    layers = metadata.get("generic_declarative_render", {}).get("final_frame", {}).get("layers", [])
    palette = list(
        dict.fromkeys(
            tuple(layer["fill_rgb"]) for layer in layers if "fill_rgb" in layer and layer["fill_rgb"] != [255, 255, 255]
        )
    )
    if not palette:
        palette = [(0, 0, 0)]
    checks, evidence, _, _, _ = _match_scene(frames[-1], final, palette, cells=cells)
    initial = reference[0] if reference else first
    for index, (x0, y0, x1, y1) in enumerate(cells):
        if index == p["answer_index"]:
            continue
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        checks[f"given_slot_{index}_preserved_throughout"] = all(
            no_unexpected_ink(f[y0:y1, x0:x1], initial[y0:y1, x0:x1])[0] for f in frames
        )
    checks["no_unrequested_edits"] = no_unexpected_ink(frames[-1], final)[0]
    evidence["semantic_answer_value"] = metadata.get("semantic_ground_truth", {}).get("answer_value")
    return _result(checks, evidence)


def _next_size_class(checks, evidence, observed, expected, matches, parameters, actual, final):
    slots = parameters["slots"]
    answer_index = len(slots)
    if len(observed) != len(expected) or (answer_index, answer_index) not in matches:
        return
    labels = [slot["shape"]["size_name"] for slot in slots]
    answer_label = labels[-len(parameters["size_cycle"])]
    exemplars = np.asarray([obj.bbox[2:] for obj in expected[:answer_index]], float)
    classes = sorted(set(labels), key=lambda label: np.mean(exemplars[np.asarray(labels) == label]))
    sizes = np.array([np.median(exemplars[np.asarray(labels) == label], axis=0) for label in classes])
    midpoints = (sizes[:-1] + sizes[1:]) / 2
    limits = np.vstack([2 * sizes[0] - midpoints[0], midpoints, 2 * sizes[-1] - midpoints[-1]])
    index = classes.index(answer_label)
    measured = np.asarray(observed[-1].bbox[2:])
    lower, upper = limits[index : index + 2]
    correct_size = bool(np.all(lower < measured) and np.all(measured < upper))
    prefix = f"object_{answer_index}."
    checks[prefix + "size"] = correct_size
    evidence["answer_size_class"] = {
        "expected": answer_label,
        "observed_dimensions_px": measured.tolist(),
        "class_bounds_px": [lower.tolist(), upper.tolist()],
    }
    if correct_size:
        # Read shape and attached support at the observed scale of the answer.
        # Size-class boundaries come from the given exemplars in the prompt.
        # Keep a successful nominal-scale proof: chroma loss can shrink the
        # measured paint box even when the full foreground is the right size.
        support = prefix + "complete_foreground_support"
        if not checks[support]:
            scale = float(np.sqrt(np.prod(measured / expected[-1].bbox[2:])))
            checks[support] = complete_support_geometry(
                actual, original_image(final), observed[-1], expected[-1], registered_expected=final, scale=scale
            )
        if checks["exact_object_count"] and all(value for key, value in checks.items() if key.startswith("object_")):
            # The exact-template fallback is unnecessary when the observed
            # slot assignments already satisfy every object predicate.
            checks.pop("original_labels_have_distinct_observed_support", None)


def _sequence(task, frames, first, final, metadata):
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _match_scene, _result

    p = metadata["parameters"]
    if task == "G-193":
        slots = p["slots"] + [p["next_slot"]]
        palette = [p["shape_color_rgb"]]
        cells = [tuple(slot["box"]) for slot in slots]
        answer = p["next_slot"]["box"]
        keep_x = int(answer[0])
    else:
        palette = [item["color"] for item in p["sequence"]] + [p["answer"]["color"]]
        objects = palette_objects(original_image(final), palette, minimum=500)
        if not objects:
            return _result({"answer_scene_available": False})
        centers = sorted(obj.center[0] for obj in objects)
        boundaries = [0] + [(a + b) / 2 for a, b in zip(centers, centers[1:])] + [final.shape[1]]
        cells = [(a, 0, b, final.shape[0]) for a, b in zip(boundaries, boundaries[1:])]
        keep_x = int(boundaries[-2])
    checks, evidence, observed, expected, matches = _match_scene(frames[-1], final, palette, cells=cells)
    if task == "G-193":
        _next_size_class(checks, evidence, observed, expected, matches, p, frames[-1], final)
    declared_count = len(p["slots"]) + 1 if task == "G-193" else p["sequence_length"]
    checks["declared_sequence_inventory"] = (
        evidence.get("matched_declared_glyph_count", evidence["observed_count"])
        == evidence["expected_count"]
        == declared_count
    )
    checks["given_sequence_preserved_throughout"] = all(
        no_unexpected_ink(f[:, :keep_x], first[:, :keep_x])[0] for f in frames
    )
    for index, (x0, y0, x1, y1) in enumerate(cells[:-1]):
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        source = first[y0:y1, x0:x1]
        ink_source = ink(source)
        foreground = cv2.erode(ink_source, np.ones((3, 3), np.uint8)).astype(bool)
        boundary = cv2.distanceTransform(ink_source, cv2.DIST_L2, 5) < 5
        expected_color = p["shape_color_rgb"] if task == "G-193" else p["sequence"][index]["color"]
        unchanged = True
        for frame in frames:
            current = frame[y0:y1, x0:x1]
            changed = np.linalg.norm(current.astype(float) - source.astype(float), axis=2) > 40
            # Chroma can lose coverage near the contour while full-resolution
            # luma and hue remain correct. Interior RGB and a changed boundary
            # hue retain the original colour-preservation obligation.
            compatible_edge = boundary & color_mask(current, expected_color).astype(bool)
            unexplained = changed & ~compatible_edge
            unchanged &= not foreground.any() or float(np.mean(unexplained[foreground])) <= 0.03
        checks[f"given_slot_{index}.color_preserved_throughout"] = unchanged
    if task == "G-193":
        # The user asks for the next shape; the generator also adds a Chinese
        # size label. That unrequested label is optional and cannot decide truth.
        x0, y0, x1, y1 = (int(v) for v in answer)
        a, b = frames[-1].copy(), first.copy()
        a[max(0, y0 - 15) : min(a.shape[0], y1 + 80), x0 - 10 : x1 + 10] = 255
        b[max(0, y0 - 15) : min(b.shape[0], y1 + 80), x0 - 10 : x1 + 10] = 255
        checks["no_unrequested_scene_edits"] = no_unexpected_ink(a, b)[0]
    else:
        checks["no_unrequested_scene_edits"] = no_unexpected_ink(frames[-1], final)[0]
    return _result(checks, evidence)


def _raven(frames, first, final, metadata):
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _match_scene, _result

    p = metadata["parameters"]
    matrix = p["matrix"]
    rows, cols = len(matrix), len(matrix[0])
    h, w = final.shape[:2]
    checks = {}
    evidence = {}
    missing = tuple(p.get("missing_position", [rows - 1, cols - 1]))
    for r, row in enumerate(matrix):
        for c, spec in enumerate(row):
            x0, x1 = round(c * w / cols) + 8, round((c + 1) * w / cols) - 8
            y0, y1 = round(r * h / rows) + 8, round((r + 1) * h / rows) - 8
            palette = spec["colors"]
            a, e = frames[-1][y0:y1, x0:x1], final[y0:y1, x0:x1]
            local, details, _, _, _ = _match_scene(a, e, palette)
            local["declared_cell_inventory"] = (
                details["expected_count"] == len(spec["shapes"]) and local["exact_object_count"]
            )
            checks.update({f"cell_{r}_{c}.{key}": value for key, value in local.items()})
            evidence[f"cell_{r}_{c}"] = details
            if (r, c) != missing:
                checks[f"cell_{r}_{c}.preserved_throughout"] = all(
                    min(mask_coverage(ink(f[y0:y1, x0:x1]), ink(first[y0:y1, x0:x1]), radius=4)) >= 0.97 for f in frames
                )
    checks["grid_and_background_preserved"] = no_unexpected_ink(frames[-1], final)[0]
    return _result(checks, evidence)
