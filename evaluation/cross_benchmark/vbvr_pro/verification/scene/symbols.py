from evaluation.cross_benchmark.vbvr_pro.verification.scene.templates import _match_scene, _palette, _result
from evaluation.cross_benchmark.vbvr_pro.vision.bandwidth import original_image
from evaluation.cross_benchmark.vbvr_pro.vision.masks import no_unexpected_ink
from evaluation.cross_benchmark.vbvr_pro.vision.objects import cell_objects, palette_objects


def _symbols(frames, first, final, metadata):
    palette = _palette(metadata)
    layers = metadata.get("generic_declarative_render", {}).get("final_frame", {}).get("layers", [])
    cells = []
    for layer in layers:
        if (
            layer.get("kind") == "rect"
            and layer.get("type") == "stroked_shape"
            and 0.35 * final.shape[0] < layer.get("y_px", 0) < 0.6 * final.shape[0]
        ):
            x, y, w, h = (layer[k] for k in ("x_px", "y_px", "w_px", "h_px"))
            inset = max(3, layer.get("stroke_width_px", 1) + 2)
            cells.append((x + inset, y + inset, x + w - inset, y + h - inset))
    # Hollow square glyphs are also stroked rectangles in the declarative
    # renderer. Their interior is part of the enclosing slot, not another
    # sequence position; keeping both would count one original glyph twice.
    cells = list(dict.fromkeys(cells))
    cells = [
        cell
        for cell in cells
        if not any(
            cell != outer
            and outer[0] <= cell[0]
            and outer[1] <= cell[1]
            and cell[2] <= outer[2]
            and cell[3] <= outer[3]
            for outer in cells
        )
    ]
    main_cells = list(cells)
    lossless_goal = original_image(final)
    if any(o.center[1] < final.shape[0] * 0.3 for o in palette_objects(lossless_goal, palette)):
        cells.append((final.shape[1] * 0.8, 0, final.shape[1], final.shape[0] * 0.3))
    checks, evidence, _, _, _ = _match_scene(frames[-1], final, palette, cells=cells or None)
    final_sequence = metadata.get("parameters", {}).get("final_sequence")
    if final_sequence is not None and main_cells:
        gold_main = cell_objects(lossless_goal, palette, main_cells)
        checks["declared_sequence_inventory"] = len(gold_main) == len(final_sequence) and checks["exact_object_count"]
        evidence["declared_main_sequence_count"] = len(final_sequence)
        evidence["lossless_main_sequence_count"] = len(gold_main)
    guard, detail = no_unexpected_ink(frames[-1], final)
    checks["slot_frames_labels_and_reference_panel_preserved"] = guard
    evidence.update(detail)
    # A symbol sequence is spatially ordered: every glyph's identity, colour,
    # fill and slot position are checked separately above. Thus deletion of a
    # wrong symbol, permutation, duplicate or incorrect insertion all fail.
    top = int(first.shape[0] * 0.3)
    checks["candidate_panel_preserved_throughout"] = all(no_unexpected_ink(f[:top], first[:top])[0] for f in frames)
    return _result(checks, evidence)
