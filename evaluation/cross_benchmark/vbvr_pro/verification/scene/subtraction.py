import numpy as np
from evaluation.cross_benchmark.vbvr_pro.vision.masks import ink, mask_coverage, no_unexpected_ink


def _subtraction(frames, first, final, metadata):
    from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _result

    objects = metadata["parameters"]["objects"]
    removed = set(metadata["parameters"]["removed_indices"])
    # Nearest-centre cells isolate each object without relying on a particular
    # 3D primitive's shading, which can create multiple colour components.
    centers = np.array([o["center"] for o in objects])
    yy, xx = np.indices(first.shape[:2])
    labels = np.argmin(np.stack([(xx - x) ** 2 + (yy - y) ** 2 for x, y in centers]), axis=0)
    checks = {}
    evidence = {}
    foreground = [ink(frame) for frame in frames]
    initial_foreground = ink(first)
    final_foreground = ink(final)
    color_error = np.linalg.norm(frames[-1].astype(np.float32) - final.astype(np.float32), axis=2)
    for index, obj in enumerate(objects):
        region = labels == index
        expected = final_foreground * region
        actual = foreground[-1] * region
        if index in removed:
            checks[f"object_{index}.removed"] = np.count_nonzero(actual) <= max(
                20, np.count_nonzero(initial_foreground * region) * 0.01
            )
        else:
            checks[f"object_{index}.retained_at_original_position"] = (
                min(mask_coverage(actual, expected, radius=3)) >= 0.97
            )
            # Compare actual RGB only on the retained object's ink, using a
            # per-object statistic so the white canvas cannot dilute a failure.
            selected = expected.astype(bool)
            checks[f"object_{index}.color_preserved"] = bool(selected.any() and np.median(color_error[selected]) <= 35)
            checks[f"object_{index}.unchanged_throughout"] = all(
                min(mask_coverage(f * region, initial_foreground * region, radius=3)) >= 0.97 for f in foreground
            )
            original = (initial_foreground * region).astype(bool)
            checks[f"object_{index}.color_preserved_throughout"] = bool(original.any()) and all(
                float(
                    np.median(
                        np.linalg.norm(frame[original].astype(np.float32) - first[original].astype(np.float32), axis=1)
                    )
                )
                <= 35
                for frame in frames
            )
    checks["no_unrequested_edits"] = no_unexpected_ink(frames[-1], final)[0]
    evidence["removed_object_indices"] = sorted(removed)
    return _result(checks, evidence)
