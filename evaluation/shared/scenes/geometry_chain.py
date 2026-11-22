"""Metric pipelines with native point markers, constructions and five choices."""

import math
import random
import numpy as np
from PIL import Image
from evaluation.shared.scenes import geometry as base
from evaluation.shared.scenes.catalog import TASKS

PROGRAMS = {
    "reflect_midpoint": ("reflection", "midpoint"),
    "project_midpoint": ("orthogonal_projection", "midpoint"),
    "invert_midpoint": ("circle_inversion", "midpoint"),
    "reflect_rotate": ("reflection", "point_rotation"),
    "project_rotate": ("orthogonal_projection", "point_rotation"),
    "midpoint_reflect_rotate": ("midpoint", "reflection", "point_rotation"),
}


def rules(s):
    program = PROGRAMS[s["key"]]
    variant = s.get("variant", "main")
    if variant.startswith("without_"):
        return tuple(p for p in program if p != variant[8:])
    return program


def trace(s):
    a = s["anchors"]
    p = a["S"]
    result = [(None, p)]
    for rule in rules(s):
        if rule == "midpoint":
            p = base.midpoint(p, a["A"])
        elif rule == "point_rotation":
            p = base.rotate(p, a["O"])
        elif rule in ("reflection", "orthogonal_projection"):
            d = np.array(a["V"]) - a["U"]
            foot = np.array(a["U"]) + np.dot(np.array(p)-a["U"], d) / np.dot(d,d) * d
            p = (2*foot-np.array(p) if rule == "reflection" else foot).tolist()
        else:
            d = np.array(p)-a["O"]
            p = (np.array(a["O"])+s["radius"]**2*d/np.dot(d,d)).tolist()
        result.append((rule, p))
    return result


def solve(s):
    target = trace(s)[-1][1]
    return next(c["label"] for c in s["candidates"] if math.dist(c["point"],target)<1e-5)


def rule_controls(s):
    return {"without_"+rule: dict(s, variant="without_"+rule) for rule in PROGRAMS[s["key"]]}


def sample(key, seed, variant="main"):
    rng = random.Random(seed)
    for _ in range(10000):
        # Semantic roles are identified by position/shape in the prompt, never
        # by extra labels or a legend panel absent from the source tasks.
        a = {"S":[rng.uniform(90,210),rng.uniform(100,230)]}
        if "midpoint" in PROGRAMS[key]: a["A"]=[rng.uniform(300,430),rng.uniform(300,420)]
        if "point_rotation" in PROGRAMS[key] or "circle_inversion" in PROGRAMS[key]:
            a["O"]=[rng.uniform(235,285),rng.uniform(225,285)]
        if "reflection" in PROGRAMS[key] or "orthogonal_projection" in PROGRAMS[key]:
            angle=rng.uniform(-.55,.55)
            centre=np.array([rng.uniform(235,290),rng.uniform(215,280)])
            delta=150*np.array([math.sin(angle),math.cos(angle)])
            a.update(U=(centre-delta).tolist(),V=(centre+delta).tolist())
        s=dict(key=key,host=TASKS[key].host,variant="main",anchors=a,radius=rng.uniform(512*.16,512*.26),turn=90)
        if "circle_inversion" in PROGRAMS[key]:
            ratio=math.dist(a["S"],a["O"])/s["radius"]
            if not 1.4 <= ratio <= 2.4:
                continue
        variants=[s,*rule_controls(s).values()]
        paths=[trace(v) for v in variants]
        targets=[p[-1][1] for p in paths]
        if any(not all(48<x<464 for x in p) for path in paths for _,p in path): continue
        arcs = [base.rotate(start,a["O"],angle) for path in paths
                for (_,start),(rule,_) in zip(path,path[1:]) if rule=="point_rotation"
                for angle in range(0,91,5)]
        if any(not all(24<x<488 for x in p) for p in arcs): continue
        if any(math.dist(x,y)<38 for i,x in enumerate(targets) for y in targets[i+1:]): continue
        if any(math.dist(p,v)<28 for p in targets for v in a.values()): continue
        if any(math.dist(x,y)<32 for i,x in enumerate(a.values()) for y in list(a.values())[i+1:]): continue
        while len(targets)<5:
            p=[rng.uniform(60,452),rng.uniform(60,452)]
            if all(math.dist(p,v)>38 for v in targets+list(a.values())): targets.append(p)
        rng.shuffle(targets)
        s["candidates"]=[dict(label=chr(65+i),point=p) for i,p in enumerate(targets)]
        s["variant"]=variant
        return s
    raise RuntimeError(f"{key}: no readable nondegenerate native pipeline")


def constructions(s):
    result=[]
    for (previous,start),(rule,end) in zip(trace(s),trace(s)[1:]):
        if rule=="point_rotation": result.append(("arc",[s["anchors"]["O"],start,90],"dark"))
        elif rule in ("reflection","orthogonal_projection"):
            result.append(("line",[start,end],"dark"))
        elif rule=="circle_inversion":
            result.append(("line",[start,s["anchors"]["O"]],"dark"))
        # Midpoint has no auxiliary drawing in its native answer action.
    return result


def render(s, progress=0, answer=None):
    g=base.renderer(s)
    draw,canvas=g.get_draw_base()
    a=s["anchors"]
    if "U" in a: g.draw_line(draw,[base.point(a["U"]),base.point(a["V"])])
    if "midpoint" in PROGRAMS[s["key"]]:
        g.draw_line(draw,[base.point(a["S"]),base.point(a["A"])])
    if "circle_inversion" in PROGRAMS[s["key"]]: g.draw_circle(draw,base.point(a["O"]),s["radius"])
    if "point_rotation" in PROGRAMS[s["key"]]:
        origin,start=a["O"],a["S"]
        g.pivot=base.point(origin);g.radius=math.dist(origin,start)
        g.source_angle=math.atan2(origin[1]-start[1],start[0]-origin[0])
        g.rotation_amount=math.pi/2;g.rotation_sign=1
        g._draw_indicator_arc(draw,canvas)
    stages=constructions(s)
    amount=float(np.clip(progress,0,1))*(len(stages)+1)
    for i,(kind,data,_) in enumerate(stages):
        fraction=float(np.clip(amount-i,0,1))
        if fraction<=0: continue
        if kind=="line":
            start,end=data
            g.draw_line(draw,[base.point(start),base.point([x+fraction*(y-x) for x,y in zip(start,end)])])
        else:
            origin,start,turn=data
            g.pivot=base.point(origin);g.radius=math.dist(origin,start)
            g.source_angle=math.atan2(origin[1]-start[1],start[0]-origin[0])
            g.rotation_amount=math.pi/2;g.rotation_sign=1
            g._draw_arc_with_recording_direction(draw,canvas,radius=g.radius,width_factor=1.,step_degrees=.5,sweep=math.pi/2*fraction)
    for name,p in a.items():
        if name == "A" or (name == "S" and PROGRAMS[s["key"]][0] == "midpoint"):
            from data.geometry.midpoint.generator import MidpointGenerator
            MidpointGenerator._draw_anchor(g,draw,tuple(p))
        elif name not in ("U","V"):
            g.draw_anchor_marker(draw,base.point(p),9 if name=="O" and "point_rotation" in PROGRAMS[s["key"]] else 7)
    g.draw_candidates(draw)
    first=g._downscale(canvas)
    fraction=float(np.clip(amount-len(stages),0,1))
    if fraction:
        g.draw_candidates(draw,highlight_label=solve(s) if answer is None else answer)
        return Image.blend(first,g._downscale(canvas),fraction)
    return first


def frames(s, answer=None):
    n=len(constructions(s))+1
    return ([np.asarray(render(s,0))]*5+
            [np.asarray(render(s,i/(8*n),answer)) for i in range(1,8*n+1)]+
            [np.asarray(render(s,1,answer))]*8)


def prompt(s):
    labels={"midpoint":"replace the current point by its midpoint with the lower-right hollow anchor",
            "reflection":"reflect the current point across the long black axis",
            "orthogonal_projection":"project the current point perpendicularly onto the long black axis",
            "circle_inversion":"invert the current point in the shown circle (distance from its centre becomes radius squared divided by the old distance)",
            "point_rotation":"rotate the current point 90 degrees counterclockwise about the central hollow pivot"}
    return ("Input Image: The upper-left hollow marker is the starting point. Five native candidates A through E are shown. "
            "When present, the diagonal segment joins the starting point to the lower-right midpoint anchor; the long near-vertical segment is the reflection/projection axis; "
            "the central hollow marker is the rotation pivot or circle centre.\n\nRule: Starting at the upper-left point, "
            + "; then ".join(labels[r] for r in rules(s)) + ". Each operation consumes the preceding result. Select its final candidate.\n\nRequired Output: "
            "Use the native black connector for reflection/projection, centre ray for inversion, and counterclockwise arc for rotation, in order. "
            "Midpoint computation adds no mark. Then uniformly fade only the selected candidate to pale red with dark-red outline. "
            "Hold the unchanged input and final answer; preserve all input marks, completed constructions, letters and camera. Add no labels or panels.")


def grade(s,path):
    # Share the native geometry observer; resolve scene operations via its
    # explicit implementation argument instead of swapping module globals.
    return base.grade(s,path,scene=__import__(__name__,fromlist=["render"]))
