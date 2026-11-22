"""Cyclic tile-color continuation with direct final-card completion."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.common.render_primitives import ScaledDraw
from data.in_context_learning.core.sequence_cards import (
    SequenceCompletionGenerator,
    final_content_bbox,
    render_sequence_scene,
)
from data.in_context_learning.core.sequence_completion_verifier import SequenceReplay

_ARRANGEMENT = "row6"
_TOTAL_CARDS = 6
_CYCLE_LENGTHS = (2, 3, 4)

# Fixed nameable master palette; every sample uses a subset of these words.
MASTER_PALETTE: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("red", (214, 72, 72)),
    ("blue", (74, 122, 214)),
    ("yellow", (238, 198, 60)),
    ("green", (94, 178, 110)),
    ("orange", (236, 140, 52)),
    ("purple", (150, 88, 196)),
)
_NAME_TO_RGB = {name: rgb for name, rgb in MASTER_PALETTE}
_COLOR_WORDS = tuple(name for name, _rgb in MASTER_PALETTE)
_TILE_EDGE = (52, 56, 62)


def _matching_cycle_lengths(visible: Sequence[str]) -> List[int]:
    """Return every allowed distinct-color cycle length matching five visible tiles."""

    values = [str(value) for value in visible]
    if len(values) != 5 or any(value not in _NAME_TO_RGB for value in values):
        return []
    matches: List[int] = []
    for cycle_length in _CYCLE_LENGTHS:
        pattern = values[:cycle_length]
        if len(set(pattern)) != cycle_length:
            continue
        if [pattern[index % cycle_length] for index in range(5)] == values:
            matches.append(cycle_length)
    return matches


def _infer_sixth(visible: Sequence[str]) -> Optional[Tuple[int, str]]:
    matches = _matching_cycle_lengths(visible)
    if len(matches) != 1:
        return None
    cycle_length = matches[0]
    return cycle_length, str(visible[5 % cycle_length])


def _paint_tile(sdraw: ScaledDraw, box, item: Mapping[str, Any], position: int) -> None:
    del position
    name = str(item["color_name"])
    color = _NAME_TO_RGB[name]
    left, top, right, bottom = (float(value) for value in box)
    from data.common.prompt_paraphrase import active_profile
    if active_profile("colour_gems"):
        from data.common.prompt_paraphrase import draw_gem
        draw_gem(sdraw,(left+7,top+7,right-7,bottom-7),color,_TILE_EDGE)
        return
    sdraw.rounded_rectangle(
        (left + 4.0, top + 6.0, right - 4.0, bottom - 6.0),
        radius=7,
        fill=color,
        outline=_TILE_EDGE,
        width=2,
    )


def _visible_items(semantics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [{"color_name": str(value)} for value in semantics["colors"][:5]]


def _answer_item(semantics: Mapping[str, Any]) -> Dict[str, Any]:
    return {"color_name": str(semantics["colors"][5])}


def render_pair_from_semantics(semantics: Mapping[str, Any]):
    items = _visible_items(semantics)
    answer = _answer_item(semantics)
    problem = render_sequence_scene(
        arrangement=_ARRANGEMENT, items=items, answer_item=None, painter=_paint_tile, solved=False
    )
    terminal = render_sequence_scene(
        arrangement=_ARRANGEMENT, items=items, answer_item=answer, painter=_paint_tile, solved=True
    )
    return problem, terminal


class TileColorSequencePattern(SequenceCompletionGenerator):
    task_name = "tile_color_sequence"
    arrangement = _ARRANGEMENT

    def sample_semantics(self) -> Dict[str, Any]:
        for _attempt in range(60):
            cycle_length = self.rng.choice(_CYCLE_LENGTHS)
            pattern = list(self.rng.sample(_COLOR_WORDS, cycle_length))
            visible = [pattern[index % cycle_length] for index in range(5)]
            inferred = _infer_sixth(visible)
            if inferred is None or inferred[0] != cycle_length:
                continue
            return {
                "cycle_length": cycle_length,
                "pattern": pattern,
                "colors": [*visible, inferred[1]],
            }
        raise RuntimeError("tile_color_sequence failed to sample a unique minimal cycle")

    def visible_items(self, semantics):
        return _visible_items(semantics)

    def answer_item(self, semantics):
        return _answer_item(semantics)

    def paint_content(self, sdraw, box, item, position):
        _paint_tile(sdraw, box, item, position)

    def answer_payload(self, semantics) -> str:
        return str(semantics["colors"][5])

    def answer_template(self) -> str:
        return "Answer: <color>."

    def author_reasoning(self, semantics) -> str:
        colors = [str(value) for value in semantics["colors"]]
        cycle_length = int(semantics["cycle_length"])
        pattern = [str(value) for value in semantics["pattern"]]
        phase = 5 % cycle_length
        return (
            f"Tiles 1-5 read {', '.join(colors[:5])}. The shortest repeating cycle consistent with all "
            f"five is {'-'.join(pattern)}, of length {cycle_length}, and every visible tile matches its "
            f"phase. Tile 6 sits at phase {phase + 1} of that cycle, so it must be {colors[5]}."
        )

    def prompt_copy(self, semantics) -> Dict[str, str]:
        return {
            "input_copy": (
                "Six numbered cards show one tile-color sequence: cards 1-5 each hold one solid colored "
                "tile, and card 6 holds only a question mark."
            ),
            "rule_copy": (
                "The five visible tiles repeat one fixed color cycle of length 2, 3, or 4 whose cycle "
                "colors are all different, and exactly one such cycle fits all five tiles. Card 6 "
                "continues that cycle by one more tile. Tile colors are named red, blue, yellow, "
                "green, orange, or purple."
            ),
            "answer_meaning": "the color word for the tile card 6 needs.",
            "terminal_copy": "one solid tile of exactly that color at the same tile size as the visible tiles.",
        }

    def structured_semantics(self, semantics) -> Dict[str, Any]:
        return {
            "cycle_length": int(semantics["cycle_length"]),
            "pattern": [str(value) for value in semantics["pattern"]],
            "colors": [str(value) for value in semantics["colors"]],
            "visible_colors": [str(value) for value in semantics["colors"][:5]],
            "correct_color_name": str(semantics["colors"][5]),
            "render_geometry": {
                "arrangement": _ARRANGEMENT,
                "palette": {name: list(rgb) for name, rgb in MASTER_PALETTE},
            },
        }

    def glyph_inventory(self, semantics) -> Dict[str, Any]:
        del semantics
        return {
            "schema_version": "icl_rendered_glyph_inventory_v1",
            "static_whitelist": ["1", "2", "3", "4", "5", "6", "?"],
            "bounded_dynamic_schema": [],
            "observed_dynamic_values": [],
            "answer_fields_excluded": True,
        }


def _decode_tile_card(candidate: np.ndarray) -> Tuple[Optional[str], Dict[str, Any]]:
    left, top, right, bottom = (int(round(value)) for value in final_content_bbox(_ARRANGEMENT, _TOTAL_CARDS))
    inset_x = max(8, (right - left) // 4)
    inset_y = max(10, (bottom - top) // 4)
    from data.common.prompt_paraphrase import active_profile
    if active_profile("colour_gems"):
        cx,cy=(left+right)//2,(top+bottom)//2
        radius=max(3,int(min(right-left,bottom-top)*.09))
        patch=candidate[cy-radius:cy+radius+1,cx-radius:cx+radius+1].astype(np.float64)
    else:
        patch = candidate[top + inset_y : bottom - inset_y, left + inset_x : right - inset_x].astype(np.float64)
    if patch.size == 0:
        return None, {"empty_patch": True}
    mean_rgb = patch.reshape(-1, 3).mean(axis=0)
    distances = {
        name: float(np.linalg.norm(mean_rgb - np.asarray(rgb, dtype=np.float64)))
        for name, rgb in MASTER_PALETTE
    }
    ranked = sorted(distances.items(), key=lambda item: item[1])
    best_name, best_distance = ranked[0]
    runner_up = ranked[1][1]
    decoded = best_name if best_distance <= 60.0 and (runner_up - best_distance) >= 25.0 else None
    return decoded, {
        "mean_rgb": [round(float(value), 1) for value in mean_rgb],
        "nearest": best_name,
        "nearest_distance": round(best_distance, 1),
        "margin": round(runner_up - best_distance, 1),
    }


def build_sequence_replay(state: Mapping[str, Any]) -> SequenceReplay:
    visible = [str(value) for value in state.get("visible_colors") or []]
    inferred = _infer_sixth(visible)
    if inferred is None:
        raise ValueError("tile-color source does not identify one minimal repeating cycle")
    cycle_length, sixth = inferred
    semantics = {
        "cycle_length": cycle_length,
        "pattern": visible[:cycle_length],
        "colors": [*visible, sixth],
    }
    problem, terminal = render_pair_from_semantics(semantics)
    return SequenceReplay(
        expected_payload=sixth,
        payload_pattern="|".join(_COLOR_WORDS),
        problem=np.asarray(problem, dtype=np.uint8),
        terminal=np.asarray(terminal, dtype=np.uint8),
        decode_final_card=_decode_tile_card,
        diagnostics={"source_replay_cycle_length": cycle_length, "source_replay_sixth_color": sixth},
    )


__all__ = ["MASTER_PALETTE", "TileColorSequencePattern", "build_sequence_replay", "render_pair_from_semantics"]
