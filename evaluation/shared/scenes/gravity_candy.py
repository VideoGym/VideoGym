"""Stable gravity transports a special token before its native detonation."""

import random
import numpy as np
from evaluation.shared.scenes import dynamics as base
from data.rearrangement_game.core.verifier import _video_semantics


def settled(s):
    grid=s["initial"]
    r,c=s["special"]
    survivors=[y for y in range(7) if grid[y][c] is not None]
    return base.gravity(grid),[7-len(survivors)+survivors.index(r),c]


def rule_controls(s):
    return {v:dict(s,variant=v) for v in ("without_gravity","without_clear")}


def solve(s):
    grid,special=(s["initial"],s["special"]) if s["variant"]=="without_gravity" else settled(s)
    if s["variant"]=="without_clear":return grid
    local=dict(s,initial=grid,special=special)
    return base.renderer()._apply(grid,base.cleared_cells(local))


def sample(key,seed,variant="main"):
    rng=random.Random(seed);g=base.renderer(seed)
    for _ in range(300):
        grid,kind,special,_=g._build_state()
        if kind=="color":continue
        for r,c in rng.sample([(r,c) for r in range(7) for c in range(7) if (r,c)!=tuple(special)],10):grid[r][c]=None
        s=dict(key=key,host="candy_crush_special_predict",variant="main",initial=grid,kind=kind,special=list(special))
        if settled(s)[1]==s["special"]:continue
        if all(solve(s)!=solve(v) for v in rule_controls(s).values()):
            s["variant"]=variant
            return s
    raise RuntimeError("no load-bearing gravity-before-special scene")


def replay(s,answer=None):
    g=base.renderer();grid=s["initial"];special=s["special"]
    first=np.asarray(g._render(grid,special=tuple(special),kind=s["kind"]))
    stable=[first];clip=[first]*8
    if s["variant"]!="without_gravity":
        destinations={}
        for c in range(7):
            rows=[r for r in range(7) if grid[r][c] is not None]
            destinations.update({(r,c):(7-len(rows)+i,c) for i,r in enumerate(rows)})
        for i in range(1,17):
            centres={}
            for cell,target in destinations.items():
                a,b=g.layout.cell_center(*cell),g.layout.cell_center(*target)
                centres[cell]=tuple(a[j]+(b[j]-a[j])*i/16 for j in range(2))
            clip.append(np.asarray(g._render(grid,special=tuple(special),kind=s["kind"],moving_centres=centres)))
        grid,special=settled(s)
        stable.append(np.asarray(g._render(grid,special=tuple(special),kind=s["kind"])))
        clip.extend([stable[-1]]*8)
    if s["variant"]!="without_clear":
        local=dict(s,initial=grid,special=special)
        affected=base.cleared_cells(local)
        marked=np.asarray(g._render(grid,special=tuple(special),kind=s["kind"],affected=affected,mark_affected=True))
        stable.append(marked);clip.extend([marked]*8)
        for i in range(1,9):
            clip.append(np.asarray(g._render(grid,special=tuple(special),kind=s["kind"],affected=affected,clear_progress=i/8,mark_affected=True)))
        stable.append(np.asarray(g._render(solve(s))))
    last=np.asarray(g._render(answer)) if answer is not None else stable[-1]
    clip.extend([last]*8)
    while len(clip)%4!=1:clip.append(last.copy())
    return stable,clip


def frames(s,answer=None):return replay(s,answer)[1]


def render(s,progress=0,answer=None):
    from PIL import Image
    stable,_=replay(s,answer)
    return Image.fromarray(stable[0] if progress==0 else stable[-1])


def prompt(s):
    return ("Input Image: The native 7 by 7 candy board contains empty pale wells and round coloured tokens. The double-rimmed special token "
            "has horizontal stripes for its row, vertical stripes for its column, a gold square for its clipped 3 by 3 neighbourhood, "
            "or a silver star for all matching colours.\n\nRule: First apply stable gravity in every column: survivors keep their vertical order "
            "and settle into its lowest wells. The special glyph travels with its token. Then activate that token once AT ITS NEW LOCATION, "
            "clearing the indicated cells, including itself. No further gravity, refill or cascade follows.\n\nRequired Output: Hold the input; "
            "move the unchanged tokens straight down; hold the settled state; add native gold frames to the affected cells and shrink precisely "
            "those tokens into their wells together, removing the special rim and frames. Hold the result. Preserve all other tokens, grid, labels, "
            "spacing, background and camera. Add no counter or caption.")


def grade(s,path):
    stable,motion=replay(s)
    metrics,passed=_video_semantics(path,stable,motion,task_name="candy_crush_special_predict")
    return {**metrics,"whole_video_loaded":True,"whole_video_process_correct":float(passed),
            "video_final_state_correct":float(metrics.get("video_terminal_state_pass",False))}
