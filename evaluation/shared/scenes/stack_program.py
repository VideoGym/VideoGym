"""Two native planning objectives in sequence, each with all optimal plans admitted."""

from collections import deque
from contextlib import contextmanager
from functools import lru_cache
import random
import tempfile
import numpy as np
import cv2
from data.rearrangement_game.crane_block_stack.generator import CraneBlockStackGenerator
from data.rearrangement_game.dock_yard_crane_swap.generator import DockYardCraneSwapGenerator
from data.rearrangement_game.core.verifier import _video_semantics, _read_video
from evaluation.shared.scenes.catalog import TASKS


def state(value):
    return tuple(tuple(col) for col in value)


def successors(current, dock=False):
    for src,stack in enumerate(current):
        for dst in range(3):
            if src==dst: continue
            for count in ((1,2) if dock else (1,)):
                if len(stack)<count: continue
                result=list(current)
                result[src]=stack[:-count]
                result[dst]=current[dst]+stack[-count:][::-1]
                yield tuple(result), (("L" if count==2 else "R"),src,dst) if dock else (src,dst)


@lru_cache(maxsize=256)
def optimal_plans(initial,target,dock):
    distances={target:0};queue=deque([target])
    while queue:
        current=queue.popleft()
        for nxt,_ in successors(current,dock):
            if nxt not in distances:
                distances[nxt]=distances[current]+1;queue.append(nxt)
    def walk(current):
        if current==target:
            yield ()
            return
        for nxt,move in successors(current,dock):
            if distances[nxt]==distances[current]-1:
                for tail in walk(nxt): yield (move,)+tail
    return tuple(walk(initial))


def stages(s):
    initial=state(s["initial"]);middle=state(s["target"])
    final=tuple(col[::-1] for col in middle)
    dock=s["key"]=="dock_reverse"
    variant=s["variant"]
    if variant=="without_target":
        return [(initial,tuple(col[::-1] for col in initial),False)]
    if variant=="without_reversal": return [(initial,middle,dock)]
    return [(initial,middle,dock),(middle,final,False)]


def rule_controls(s):
    return {v:dict(s,variant=v) for v in ("without_target","without_reversal")}


def solve(s):
    # Rule removals change the endpoint; the verifier additionally requires
    # the prescribed intermediate hold and all phase-local optimal moves.
    return [list(col) for col in stages(s)[-1][1]]


def sample(key,seed,variant="main"):
    rng=random.Random(seed)
    for _ in range(300):
        n=4
        first=list(range(1,n+1));middle=first.copy()
        rng.shuffle(first);rng.shuffle(middle)
        start_col=rng.randrange(3);target_col=rng.randrange(3)
        initial=[[] for _ in range(3)];target=[[] for _ in range(3)]
        initial[start_col]=first;target[target_col]=middle
        s=dict(key=key,host=TASKS[key].host,variant="main",initial=initial,target=target)
        joint=solve(s)
        if any(solve(c)==joint for c in rule_controls(s).values()):continue
        a,b=stages(s)
        plans=optimal_plans(*a)
        if not 2<=len(plans[0])<=7:continue
        if key=="dock_reverse":
            if any({m[0] for m in plan}!={"L","R"} for plan in plans):continue
        # Keep exact all-path admission affordable; reject samples before
        # inference, without consulting model scores.
        if len(plans)*len(optimal_plans(*b))>64:continue
        s["variant"]=variant
        return s
    raise RuntimeError(f"{key}: no nondegenerate native stack program")


@contextmanager
def renderer(s):
    cls=DockYardCraneSwapGenerator if s["key"]=="dock_reverse" else CraneBlockStackGenerator
    with tempfile.TemporaryDirectory() as folder:
        yield cls(folder)


def programs(s):
    from itertools import product
    return product(*(optimal_plans(*stage) for stage in stages(s)))


def replay(s,program=None,answer=None):
    if program is None:program=next(programs(s))
    with renderer(s) as g:
        current=s["initial"]
        target=s["target"]  # The shown intermediate target remains fixed.
        stable=[np.asarray(g._render_static_scene(current,target))]
        clip=[stable[0].copy() for _ in range(8)]
        for phase,plan in enumerate(program):
            for move in plan:
                if s["key"]=="dock_reverse" and len(move)==2:move=("R",*move)
                clip.extend(np.asarray(f) for f in g._animate_move(current,move,target))
                current=g._apply_move(current,move)
                stable.append(np.asarray(g._render_static_scene(current,target)))
                clip.extend([stable[-1]]*g.pause_frames)
            clip.extend([stable[-1]]*8)
        if answer is not None:
            clip[-8:]=[np.asarray(g._render_static_scene(answer,target))]*8
        while len(clip)%4!=1:clip.append(clip[-1].copy())
        return stable,clip


def frames(s,answer=None):return replay(s,answer=answer)[1]


def render(s,progress=0,answer=None):
    with renderer(s) as g:
        return g._render_static_scene(s["initial"] if progress==0 else (solve(s) if answer is None else answer),s["target"])


def prompt(s):
    with renderer(s) as g:description=g._describe_input();native=g._describe_rules()
    return ("Input Image: "+description+" The right panel specifies an intermediate target.\n\nRule: "+native+
            " First reach that exact intermediate target with the fewest native moves and hold it. Then reverse the entire occupied stack in that same column, "
            "leaving the other two columns empty, with the fewest single-top-block transfers. Each move in this second phase pops one top block and "
            "pushes it on another column; use only the right crane in the two-crane scene. Minimize each phase separately.\n\nRequired Output: "
            "Animate every transfer with the source's native crane motion, fixed object identities and sizes. Hold the initial, every completed move, "
            "the intermediate target and the final reversed tower. The right panel, columns, labels and camera stay unchanged. No new marker or phase counter.")


def grade(s,path):
    from evaluation.shared.task_io import matches
    from data.common.progress_alignment import tone_offset
    from data.common.verifiers.region_gates import static_region_errors, STATIC_REGION_FLOOR
    observed,fps=_read_video(path)
    final=bool(observed) and matches(observed[-1],np.asarray(render(s,1)))
    if not final:
        return dict(whole_video_loaded=True,whole_video_process_correct=0.,video_final_state_correct=0.)
    with renderer(s) as g:
        intermediate=np.asarray(g._render_static_scene(s["target"],s["target"]))
        initial_image=np.asarray(g._render_static_scene(s["initial"],s["target"]))
        geometry=g._panel_geometry()
    # All legal motion is confined to the left panel. The right target,
    # chevron and outer margins are immutable for EVERY optimal program.
    fixed=np.ones(initial_image.shape[:2],bool)
    x,y=geometry["left_panel_x0"],geometry["left_panel_y0"]
    fixed[y:y+geometry["panel_h"],x:x+geometry["panel_w"]]=False
    preserved=all(static_region_errors(frame.astype(float)-tone_offset(frame,initial_image,fixed),initial_image,fixed)[0]<=STATIC_REGION_FLOOR for frame in observed)
    if not preserved:
        return dict(whole_video_loaded=True,whole_video_process_correct=0.,video_final_state_correct=1.,fixed_scene_preserved=False,programs_checked=0)
    if s["variant"] == "main" and not any(matches(frame,intermediate) for frame in observed):
        return dict(whole_video_loaded=True,whole_video_process_correct=0.,video_final_state_correct=1.)
    # Rank, never discard, legal programs using only their committed states.
    # A valid non-author program is usually checked first; every alternative
    # remains available if the detailed native motion verifier rejects it.
    ranked=rank_programs(s,observed)
    best=None
    for checked,program in enumerate(ranked,1):
        stable,motion=replay(s,program)
        metrics,passed=_video_semantics(path,stable,motion,task_name=s["host"],decoded_video=(observed,fps))
        result={**metrics,"whole_video_loaded":True,"whole_video_process_correct":float(passed),
                "video_final_state_correct":float(metrics.get("video_terminal_state_pass",False)),
                "programs_checked":checked,"legal_program_count":len(ranked)}
        if passed:return result
        if best is None or result["video_final_state_correct"]>best["video_final_state_correct"]:best=result
    return best


def rank_programs(s,observed):
    """Observation affects search order only, never the accepted program set."""
    bank=np.array([cv2.resize(f,(32,32),interpolation=cv2.INTER_AREA) for f in observed],dtype=np.float32)
    cost_cache={}
    candidates=[]
    with renderer(s) as g:
        for program in programs(s):
            current=s["initial"]
            scores=[]
            for plan in program:
                for move in plan:
                    if s["key"]=="dock_reverse" and len(move)==2:move=("R",*move)
                    current=g._apply_move(current,move)
                    key=state(current)
                    if key not in cost_cache:
                        image=cv2.resize(np.asarray(g._render_static_scene(current,s["target"])),(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
                        cost_cache[key]=np.abs(bank-image).mean(axis=(1,2,3))
                    scores.append(cost_cache[key])
            distance=np.zeros(len(bank))
            for costs in scores:
                distance=costs+np.minimum.accumulate(distance)
            candidates.append((float(distance.min()),program))
    return [program for _,program in sorted(candidates,key=lambda pair:pair[0])]
