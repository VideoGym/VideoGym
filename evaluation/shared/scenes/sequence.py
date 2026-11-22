"""Joint attribute rules using the native card renderers."""

from functools import lru_cache
from itertools import product
import json
import random
import numpy as np
from PIL import Image
from data.in_context_learning.core.sequence_cards import render_sequence_scene, final_content_bbox
from data.in_context_learning.core.size_cards import (
    paint_sized_shape,
    paint_ruler,
    SIZE_SHAPES,
)
from data.in_context_learning.growing_polygon_step.generator import _paint_polygon
from data.common.verifiers.transitions import uniform_fade_metrics
from data.common.verifiers.region_gates import static_region_errors, STATIC_REGION_FLOOR
from evaluation.shared.scenes.catalog import TASKS
from data.in_context_learning.tile_color_sequence.generator import MASTER_PALETTE


def sample(key, seed, variant="main"):
    rng = random.Random(seed)
    s = dict(key=key, host=TASKS[key].host, variant=variant)
    if key in ("size_color", "growth_color", "polygon_color"):
        s["colors"] = [list(c) for c in rng.sample([rgb for _,rgb in MASTER_PALETTE], 3)]
        if key != "polygon_color":
            s["shape"] = rng.choice(SIZE_SHAPES)
    if key == "size_color":
        a = rng.randint(2, 7)
        b = rng.choice([v for v in range(2, 8) if abs(v-a) >= 2])
        s.update(values=[a, b, a], arrangement="grid4")
    elif key == "growth_color":
        step = rng.choice((1, 2))
        a = rng.randint(1, 2 if step == 2 else 5)
        s.update(values=[a, a + step, a + 2 * step], arrangement="grid4")
    elif key == "polygon_color":
        a = rng.randint(3, 5)
        step = rng.choice((1, 2))
        if a + 3 * step > 9:
            step = 1
        s.update(values=[a, a + step, a + 2 * step], arrangement="row4")
    return s


def solve(s):
    key = s["key"]
    if key in ("size_color", "growth_color", "polygon_color"):
        v = s["values"]
        answer = v[1] if key == "size_color" else 2 * v[2] - v[1]
        color = s["colors"][0] if s["variant"] != "source_goal" else s["colors"][2]
        result = {
            "height" if key != "polygon_color" else "vertex_count": answer,
            "color": color,
        }
        if key != "polygon_color":
            result["shape"] = s["shape"]
        return result
    raise ValueError(key)


def painter(s):
    if s["key"] in ("size_color", "growth_color"):
        return paint_sized_shape
    if s["key"] == "polygon_color":
        return _paint_polygon
    raise ValueError(s["key"])


def _items(s):
    key = s["key"]
    if key in ("size_color", "growth_color", "polygon_color"):
        field = "vertex_count" if key == "polygon_color" else "height"
        return [
            dict(
                {field: v, "color": s["colors"][i % 3]},
                **({"shape": s["shape"]} if key != "polygon_color" else {}),
            )
            for i, v in enumerate(s["values"])
        ]
    raise ValueError(key)


def endpoints(s, answer=None):
    answer = solve(s) if answer is None else answer
    if answer is None:
        raise ValueError("ambiguous sequence")
    key = s["key"]
    if key in ("size_color", "growth_color", "polygon_color"):
        items = _items(s)
        place = paint_ruler if key != "polygon_color" else None
        problem = render_sequence_scene(
            arrangement=s["arrangement"],
            items=items,
            answer_item=None,
            painter=painter(s),
            solved=False,
            placeholder_painter=place,
        )
        terminal = render_sequence_scene(
            arrangement=s["arrangement"],
            items=items,
            answer_item=answer,
            painter=painter(s),
            solved=True,
        )
    return problem, terminal


def render(s, progress=0.0, answer=None):
    initial, last = endpoints(s, answer)
    return Image.blend(initial, last, float(np.clip(progress, 0, 1)))


def prompt(s):
    rule = TASKS[s["key"]].rule
    if s["variant"] == "source_goal" and s["key"] in (
        "size_color",
        "growth_color",
        "polygon_color",
    ):
        detail = {
            "size_color": "The three visible heights alternate between two fixed values.",
            "growth_color": "The three visible heights increase by one constant positive increment.",
            "polygon_color": "The three visible polygon-side counts increase by one constant increment.",
        }[s["key"]]
        rule = (
            detail
            + " Continue that rule into card 4 and copy the last visible card's color. The earlier color order is irrelevant for this source-goal control."
        )
    return (
        "Input Image: Native numbered sequence cards with one question mark. Ruled cards measure height at lines 1 through 8; polygon dots mark its vertices.\n\nRule: "
        + rule
        + "\n\nRequired Output: Replace only the question mark with the complete correct card content by a uniform local fade. Begin with the input unchanged, keep every other card, label and grid fixed, and end with a stable solved hold. No drawing scan, hard cut, full-frame fade, disappearance or camera movement."
    )



@lru_cache(maxsize=16)
def _answer_templates(payload):
    """Render plausible payloads on the source canvas; retain only answer crops."""
    s=json.loads(payload)
    items=_items(s)
    painter_fn=painter(s)
    def blank_painter(draw,box,item,position):
        if position<len(items):
            painter_fn(draw,box,item,position)
        elif s["key"]!="polygon_color":
            paint_ruler(draw,box)
    box=tuple(int(round(v)) for v in final_content_bbox(s["arrangement"],len(items)+1))
    left,top,right,bottom=box
    def crop(image):return np.asarray(image)[top:bottom,left:right].astype(np.float32)
    blank=crop(render_sequence_scene(arrangement=s["arrangement"],items=items,answer_item={},painter=blank_painter,solved=True))
    templates=[]
    shapes=SIZE_SHAPES if s["key"]!="polygon_color" else (None,)
    values=range(1,9) if s["key"]!="polygon_color" else range(3,10)
    for value,color,shape in product(values,s["colors"],shapes):
        answer={"height" if shape else "vertex_count":value,"color":color}
        if shape:answer["shape"]=shape
        image=render_sequence_scene(arrangement=s["arrangement"],items=items,answer_item=answer,painter=painter_fn,solved=True)
        patch=crop(image)
        mask=np.max(np.abs(patch-blank),axis=2)>=24
        templates.append((answer,patch,mask))
    return box,blank,templates


def decode_answer(s,frame):
    """Read height/side count, colour and shape on actual ink support.

    Whole-card averages can hide a wrong colour on a two-unit diamond. Each
    comparison instead uses the union of observed and hypothesized answer ink,
    including outlines and extra ink, without letting empty card area dilute it.
    """
    public={k:s[k] for k in ("key","arrangement","colors","values")}
    if "shape" in s:public["shape"]=s["shape"]
    box,blank,templates=_answer_templates(json.dumps(public,sort_keys=True))
    left,top,right,bottom=box
    observed=frame[top:bottom,left:right].astype(np.float32)
    observed_mask=np.max(np.abs(observed-blank),axis=2)>=24
    ranked=[]
    for answer,expected,mask in templates:
        support=observed_mask|mask
        error=float(np.sqrt(np.mean((observed[support]-expected[support])**2))/255.) if support.any() else 1.
        ranked.append((error,answer))
    ranked.sort(key=lambda item:item[0])
    best,answer=ranked[0]
    margin=ranked[1][0]-best
    decoded=answer if best<=.25 and margin>=.02 else None
    return decoded,{"sequence_answer_error":best,"sequence_answer_margin":margin,"decoded_answer":decoded}


def grade(s, path):
    from evaluation.shared.task_io import read_video, matches

    frames = read_video(path)
    first, last = endpoints(s)
    first = np.asarray(first)
    last = np.asarray(last)
    decoded,diagnostics=decode_answer(s,frames[-1])
    final = decoded==solve(s) and matches(frames[-1], last)
    fade, fade_ok = uniform_fade_metrics(frames, first, last)
    changed = np.max(np.abs(first.astype(int) - last.astype(int)), axis=2) > 8
    static = ~changed
    preserved = all(static_region_errors(f, first, static)[0] <= STATIC_REGION_FLOOR for f in frames)
    process = final and matches(frames[0], first) and preserved and fade_ok
    return {
        "whole_video_process_correct": float(process),
        "video_final_state_correct": float(final),
        "whole_video_loaded": True,
        "sequence_fade": fade,
        "source_state_replayed": True,
        **diagnostics,
    }
