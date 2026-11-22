"""Native special-candy detonation followed by stable column gravity."""

import random
import numpy as np
from data.common.pipeline.board_motion import BoardLayout, CANVAS_SIZE
from data.dynamics_game.candy_crush_special_predict.generator import CandyCrushSpecialPredictGenerator, CANDY_CODES
from data.dynamics_game.jewel2_future_board.generator import Jewel2FutureBoardGenerator
from data.rearrangement_game.core.verifier import _video_semantics


def renderer(seed=None):
    g = object.__new__(CandyCrushSpecialPredictGenerator)
    g.rows = g.cols = 7
    g.rng = random.Random(seed)
    g.layout = BoardLayout.for_grid(7,7,board_max=CANVAS_SIZE-64,reserve_top=28)
    return g


def cleared_cells(source):
    r,c = source["special"]
    kind, grid = source["kind"], source["initial"]
    if kind == "row":
        return {(r,x) for x in range(7)}
    if kind == "col":
        return {(y,c) for y in range(7)}
    if kind == "wrap":
        return {(y,x) for y in range(max(0,r-1),min(7,r+2)) for x in range(max(0,c-1),min(7,c+2))}
    return {(y,x) for y in range(7) for x in range(7) if grid[y][x] == grid[r][c]}


def gravity(grid):
    # Use the host's own numeric identity codec; rendering retains its palette.
    g = object.__new__(Jewel2FutureBoardGenerator)
    encoded = [[CANDY_CODES[value] for value in row] for row in grid]
    colours = {value:key for key,value in CANDY_CODES.items() if key}
    return [[colours.get(value) for value in row] for row in g._apply_gravity(encoded)]


def ablations(source):
    cleared = renderer()._apply(source["initial"], cleared_cells(source))
    return gravity(cleared), cleared, gravity(source["initial"])


def sample(key, seed, variant="main"):
    g = renderer(seed)
    for _ in range(200):
        grid,kind,special,_ = g._build_state()
        source = dict(key=key,host="candy_crush_special_predict",variant=variant,
                      initial=grid,kind=kind,special=list(special))
        joint,first,second = ablations(source)
        if joint != first and joint != second:
            return source
    raise RuntimeError("special-candy gravity composition has no nondegenerate sample")


def solve(source):
    return ablations(source)[{"main":0,"source_goal":1,"second_goal":2}[source["variant"]]]


def render(source, progress=0, answer=None):
    g = renderer()
    if progress == 0 or source["variant"] == "second_goal":
        return g._render(source["initial"],special=tuple(source["special"]),kind=source["kind"])
    return g._render(solve(source) if answer is None else answer)


def replay(source, answer=None):
    g = renderer()
    grid = source["initial"]
    special = tuple(source["special"])
    affected = cleared_cells(source)
    first = np.asarray(render(source,0))
    if source["variant"] == "second_goal":
        return [first], [first.copy() for _ in range(53)]
    cleared = g._apply(grid,affected)
    marked = np.asarray(g._render(grid,special=special,kind=source["kind"],affected=affected,mark_affected=True))
    settled = np.asarray(g._render(cleared))
    stable = [first,marked,settled]
    clip = [first.copy() for _ in range(8)] + [marked.copy() for _ in range(8)]
    clip.extend(np.asarray(g._render(grid,special=special,kind=source["kind"],affected=affected,
                                    clear_progress=i/8,mark_affected=True)) for i in range(1,9))
    clip.extend(settled.copy() for _ in range(4))
    if source["variant"] == "main":
        destinations = {}
        for c in range(7):
            rows = [r for r in range(7) if cleared[r][c] is not None]
            destinations.update({(r,c):(7-len(rows)+i,c) for i,r in enumerate(rows)})
        for i in range(1,17):
            centres = {}
            for cell, target in destinations.items():
                a,b = g.layout.cell_center(*cell),g.layout.cell_center(*target)
                centres[cell] = tuple(a[j]+(b[j]-a[j])*i/16 for j in range(2))
            clip.append(np.asarray(g._render(cleared,moving_centres=centres)))
        stable.append(np.asarray(g._render(solve(source))))
    last = np.asarray(render(source,1,answer))
    clip.extend(last.copy() for _ in range(8))
    while len(clip)%4 != 1:
        clip.append(last.copy())
    return stable,clip


def frames(source, answer=None):
    return replay(source,answer)[1]


def prompt(source):
    return ("Input Image: The unchanged native 7 by 7 candy board uses round coloured tokens and one token with a white-and-gold "
            "double rim. Horizontal stripes mean its full row, vertical stripes its full column, a gold square its clipped 3 by 3 "
            "neighbourhood, and a silver star every token matching its colour.\n\nRule: Activate exactly that special clearing "
            "rule once, including the marked token. Then apply gravity independently in each column: survivors retain vertical "
            "order and fall into the lowest empty wells. Do not swap, refill, clear again, or move sideways.\n\nRequired Output: "
            "Hold the input; add the native gold frames to every affected cell, shrink exactly those tokens simultaneously "
            "into their pale wells and remove the frames and special rim. After that clear is complete, move every survivor "
            "straight downward with its size, shape and colour fixed. Hold the final board. Preserve the grid, labels, wells, "
            "spacing, background and camera. No extra marks or cuts.")


def grade(source,path):
    stable,motion = replay(source)
    metrics,passed = _video_semantics(path,stable,motion,task_name="candy_crush_special_predict")
    return {**metrics,"whole_video_loaded":True,"whole_video_process_correct":float(passed),
            "video_final_state_correct":float(metrics.get("video_terminal_state_pass",False))}
