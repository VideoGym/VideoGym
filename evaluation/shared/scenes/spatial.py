"""Fold a native pip-marked net, then rigidly rotate the assembled cube."""

from dataclasses import replace
from functools import lru_cache
import json
import math
import random
import numpy as np

from data.spatial_3d.cube_net_fold.generator import CubeNetFoldPattern
from data.spatial_3d.core.iso_geometry import rotation_matrix, apply_matrix
from data.rearrangement_game.core.verifier import _video_semantics


@lru_cache(maxsize=32)
def _task(payload):
    source = json.loads(payload)
    g = object.__new__(CubeNetFoldPattern)
    return g._build_task([tuple(c) for c in source["net_cells"]], tuple(source["base_cell"]),
                         {(x,y): p for x,y,p in source["pip_by_cell"]})


def task(source):
    return _task(json.dumps({k:source[k] for k in ("net_cells","base_cell","pip_by_cell")},sort_keys=True))


class Renderer(CubeNetFoldPattern):
    def _mesh_faces(self, source_task, angles):
        faces = super()._mesh_faces(source_task, angles)
        if self.turn == 0:
            return faces
        matrix = rotation_matrix(self.axis, self.turn)
        # The source net retains its integer floor coordinates after folding.
        bx, by = source_task["base"]
        centre = (bx+.5,by+.5,.5)

        def point(p):
            rotated = apply_matrix(matrix, [p[i]-centre[i] for i in range(3)])
            return tuple(rotated[i]+centre[i] for i in range(3))

        return [replace(face, points=tuple(point(p) for p in face.points),
                        decal=replace(face.decal, origin=point(face.decal.origin),
                                      axis_u=apply_matrix(matrix,face.decal.axis_u),
                                      axis_v=apply_matrix(matrix,face.decal.axis_v))) for face in faces]


def pose(source, progress):
    g = object.__new__(Renderer)
    g.axis = source["axis"]
    if source["variant"] == "source_goal":
        fold, turn = progress, 0
    elif source["variant"] == "second_goal":
        fold, turn = 0, progress
    else:
        fold, turn = min(progress/.8,1), max((progress-.8)/.2,0)
    g.turn = source["sign"] * math.pi / 2 * turn
    t = task(source)
    return g, t, g._schedule_state(t, fold)


def solve(source):
    g,t,angles = pose(source,1)
    return [[[round(float(v),6) for v in p] for p in face.points] for face in g._mesh_faces(t,angles)]


def ablations(source):
    return tuple(solve(dict(source,variant=v)) for v in ("main","source_goal","second_goal"))


def sample(key, seed, variant="main"):
    rng = random.Random(seed)
    native = object.__new__(CubeNetFoldPattern)
    native.rng = rng
    native.difficulty = None
    for _ in range(200):
        t = native._sample_task()
        source = dict(key=key,host="cube_net_fold",variant=variant,
                      net_cells=[list(c) for c in t["cells"]],base_cell=list(t["base"]),
                      pip_by_cell=[[x,y,t["pips"][(x,y)]] for x,y in t["cells"]],
                      axis=rng.choice(("X","Y")),sign=rng.choice((-1,1)))
        # Keep the EXACT source camera/scale. Reject unreadable trajectories
        # rather than shrinking or refitting the canvas to the new program.
        clear = True
        for v in ("main","second_goal"):
            for p in np.linspace(0,1,17):
                g,tt,a = pose(dict(source,variant=v),float(p))
                for face in g._mesh_faces(tt,a):
                    for point in face.points:
                        x,y = tt["projection"].project(point)
                        clear &= 20 <= x <= 492 and 20 <= y <= 492
        if clear:
            return source
    raise RuntimeError("no net with visible joint and single-rule trajectories in its native camera")


def render(source, progress=0, answer=None):
    g,t,angles = pose(source,progress)
    if answer is not None and progress >= 1:
        # Used only by the explicit wrong-terminal control.
        g.turn = -g.turn
    return g._render_angles(t,angles)


def frames(source, answer=None):
    result = [np.asarray(render(source,0))] * 5
    result.extend(np.asarray(render(source,float(p),answer)) for p in np.linspace(0,1,97)[1:])
    result.extend([np.asarray(render(source,1,answer))] * 8)
    return result


def prompt(source):
    direction = "positive right-hand" if source["sign"] == 1 else "negative right-hand"
    fold = ("Keep the blue base fixed while folding the other five squares upward by 90 degrees along their hinges "
            "in increasing pip-count order, carrying every attached descendant. Close the cube completely. ")
    rotation = (f"Then rotate the entire rigid result by one {direction} quarter-turn about the {source['axis']} axis "
                "through the centre of the assembled cube. The base-local X direction runs along the base's screen-down-right "
                "edge, Y along its screen-down-left edge, and Z points up; these directions stay fixed in world space. ")
    rule = (fold if source["variant"] != "second_goal" else "Leave all hinges unfolded. ")
    if source["variant"] != "source_goal":
        rule += rotation
    return ("Input Image: The native six-square paper net has one blue base and fixed die-pip marks.\n\nRule: " + rule +
            "\n\nRequired Output: Hold the input and show each prescribed rigid hinge fold continuously, keeping the completed "
            "folds and carrying attached squares. After closure, show the prescribed rigid rotation continuously and hold "
            "the final pose. Keep every square, pip, paper colour, background, scale and camera unchanged. No extra axes, "
            "numbers, arrows, cuts, stretching or new objects.")


def grade(source,path):
    g,t,_ = pose(source,0)
    checkpoints = [p*.8 for p in g.semantic_checkpoints(t)] + [1.]
    stable = [np.asarray(render(source,p)) for p in checkpoints]
    metrics, passed = _video_semantics(path,stable,frames(source),task_name="cube_net_fold")
    return {**metrics,"whole_video_loaded":True,"whole_video_process_correct":float(passed),
            "video_final_state_correct":float(metrics.get("video_terminal_state_pass",False))}
