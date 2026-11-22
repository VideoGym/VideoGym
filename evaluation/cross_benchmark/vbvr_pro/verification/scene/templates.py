from evaluation.cross_benchmark.vbvr_pro.verification.common.layers import _result
from evaluation.cross_benchmark.vbvr_pro.vision.bandwidth import (
    complete_support_geometry,
    distinct_support,
    local_proof,
    original_image,
    no_added_scene,
)
from evaluation.cross_benchmark.vbvr_pro.vision.objects import assign_objects, cell_objects, object_checks, palette_objects

__all__ = ["_match_scene", "_palette", "_result", "_rgb_values"]


def _rgb_values(value):
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key in {"color", "color_rgb", "rgb", "fill_rgb", "stroke_rgb", "initial_color_rgb", "final_color_rgb"}
                and isinstance(item, (list, tuple))
                and len(item) == 3
                and all(isinstance(v, (int, float)) for v in item)
            ):
                result.append(tuple(int(v) for v in item))
            elif isinstance(item, (dict, list)):
                result.extend(_rgb_values(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_rgb_values(item))
    return result


def _palette(metadata):
    semantic, params = metadata.get("semantic_ground_truth", {}), metadata.get("parameters", {})
    colors = _rgb_values(semantic.get("objects", []))
    colors += _rgb_values(params.get("groups", [])) + _rgb_values(params.get("circles", []))
    colors += _rgb_values(params.get("symbols", []))
    # Some sequence generators put colours in a map rather than on objects.
    for name in ("color_map", "color_map_rgb"):
        colors.extend(tuple(c) for c in params.get(name, {}).values() if isinstance(c, list) and len(c) == 3)
    return list(dict.fromkeys(colors))


def _match_scene(actual, expected, palette, position=True, cells=None, minimum=24):
    # Lossless original roles define gold identities and cardinality. A
    # bandwidth-registered or decoded video can lose thin coloured strokes.
    gold = original_image(expected)
    a = cell_objects(actual, palette, cells) if cells else palette_objects(actual, palette, minimum)
    e = cell_objects(gold, palette, cells) if cells else palette_objects(gold, palette, minimum)
    checks = {"exact_object_count": len(a) == len(e) and len(e) > 0}
    matches = assign_objects(a, e, position)
    for i, j in matches:
        checks.update({f"object_{i}.{key}": value for key, value in object_checks(a[j], e[i], position).items()})
        checks[f"object_{i}.complete_foreground_support"] = complete_support_geometry(
            actual, gold, a[j], e[i], registered_expected=expected
        )
    evidence = {"observed_count": len(a), "expected_count": len(e)}
    if e and not all(checks.values()):
        mapping = dict(matches)
        if position or len(mapping) == len(e):
            proofs = [
                local_proof(
                    actual,
                    expected,
                    obj,
                    palette,
                    position=position,
                    actual_object=a[mapping[i]] if i in mapping else None,
                )
                for i, obj in enumerate(e)
            ]
            injective = distinct_support(proofs)
            checks["original_labels_have_distinct_observed_support"] = injective
            if injective and (len(a) == len(e) or no_added_scene(actual, expected)):
                checks["exact_object_count"] = True
                for i in range(len(e)):
                    prefix = f"object_{i}."
                    for name in [name for name in checks if name.startswith(prefix)]:
                        checks[name] = True
                    checks[prefix + "labelled_local_observation"] = True
                evidence["matched_declared_glyph_count"] = len(proofs)
                evidence["observation_variant_labels"] = list(range(len(e)))
                evidence["observed_label_positions"] = [proof["location"] for proof in proofs]
    return checks, evidence, a, e, matches
