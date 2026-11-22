"""Native grid completions with independently necessary constraint sets.

All joint solutions are enumerated from the queen permutation domain. A
relaxed witness proves strict set inclusion; it is never substituted for the
full relaxed solution set or treated as the only acceptable joint answer.
"""

from functools import lru_cache
from itertools import permutations
import json
import random

import cv2
import numpy as np
from PIL import Image

from data.constraint_grid.star_battle_completion.generator import StarBattleCompletionGenerator
from data.constraint_grid.light_up_akari_completion.generator import LightUpAkariCompletionGenerator
from data.constraint_grid.core.scene import standard_board_geometry
from data.constraint_grid.core.forced_deduction import build_sequential_commit_frames
from data.common.verifiers.region_gates import terminal_region_gates, static_region_errors, STATIC_REGION_FLOOR
from data.common.verifiers.transitions import complete_footprint_projection
from evaluation.shared.task_io import read_video, matches


@lru_cache(maxsize=4)
def placements(n):
    return tuple(tuple(enumerate(p)) for p in permutations(range(n)))


def queen_valid(cells, n):
    return (len(cells) == n and len({r for r,c in cells}) == n
            and len({c for r,c in cells}) == n
            and len({r-c for r,c in cells}) == n
            and len({r+c for r,c in cells}) == n)


def host(s):
    if s['key'] == 'queens_regions':
        g = object.__new__(StarBattleCompletionGenerator)
        g.size = s['size']
    else:
        g = object.__new__(LightUpAkariCompletionGenerator)
        g.rows = g.cols = s['size']
    return g


def native_valid(s, cells):
    cells = set(map(tuple, cells))
    n = s['size']
    if any(not (0 <= r < n and 0 <= c < n) for r,c in cells):
        return False
    if s['key'] == 'queens_regions':
        regions = [set(map(tuple, region)) for region in s['regions']]
        return (len(cells) == n and len({r for r,c in cells}) == n
                and len({c for r,c in cells}) == n
                and all(len(cells & region) == 1 for region in regions)
                and all(max(abs(r-x),abs(c-y)) > 1 for r,c in cells for x,y in cells if (r,c)!=(x,y)))
    black = set(map(tuple, s['black']))
    if cells & black:
        return False
    g = host(s)
    lit = set()
    for cell in cells:
        visible = set(g._segment(cell, black))
        if len(visible & cells) != 1:
            return False
        lit.update(visible)
    if lit != {(r,c) for r in range(n) for c in range(n)} - black:
        return False
    return all(sum((r+dr,c+dc) in cells for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))) == value
               for r,c,value in s['clues'])


def domain_valid(s, cells):
    return not (set(map(tuple,cells)) & set(map(tuple,s.get('black',()))))


@lru_cache(maxsize=256)
def _solutions(payload):
    s = json.loads(payload)
    queens = [p for p in placements(s['size']) if queen_valid(p,s['size']) and domain_valid(s,p)]
    joint = [p for p in queens if native_valid(s,p)]
    # Search within permutations is sufficient to FIND a native-only witness.
    # It does not claim to enumerate all Akari solutions (which may have
    # several bulbs in a row separated by black walls).
    without_queen = next((p for p in placements(s['size'])
                          if native_valid(s,p) and not queen_valid(p,s['size'])), None)
    without_native = next((p for p in queens if not native_valid(s,p)), None)
    return tuple(joint), without_queen, without_native, len(queens)


def solutions(s):
    public = {k:v for k,v in s.items() if k not in ('variant','keyframe_order')}
    return _solutions(json.dumps(public,sort_keys=True))


def solution_set_evidence(s):
    joint, without_queen, without_native, queen_count = solutions(s)
    passed = bool(joint and without_queen and without_native)
    return dict(kind='strict_solution_set_inclusion', passed=passed,
                joint_solution_count=len(joint), queen_only_solution_count=queen_count,
                relaxed_witnesses={
                    'without_queen': without_queen,
                    'without_native': without_native,
                },
                native_relaxed_set_enumerated=False,
                claim='The nonempty joint set is a proper subset of each relaxed set; witnesses are not exhaustive relaxed answers.')


def rule_controls(s):
    return {name:dict(s,variant=name) for name in ('without_queen','without_native')}


def solve(s):
    joint, without_queen, without_native, _ = solutions(s)
    value = {'main':joint[0], 'without_queen':without_queen, 'without_native':without_native}[s['variant']]
    return [list(cell) for cell in value]


def sample(key, seed, variant='main'):
    rng = random.Random(seed)
    n = 6  # Both native host renderers use their unchanged 6 x 6 layout.
    queens = [p for p in placements(n) if queen_valid(p,n)]
    for _ in range(2000):
        target = rng.choice(queens)
        s = dict(key=key, host='star_battle_completion' if key=='queens_regions' else 'light_up_akari_completion',
                 size=n, variant='main')
        g = host(s)
        if key == 'queens_regions':
            g.rng = rng
            s['regions'] = [[list(cell) for cell in region] for region in g._build_regions(list(target))]
        else:
            available = [(r,c) for r in range(n) for c in range(n) if (r,c) not in target]
            black = set(rng.sample(available,rng.randint(3,6)))
            s['black'] = [list(cell) for cell in sorted(black)]
            clues = rng.sample(sorted(black),2)
            s['clues'] = [[r,c,sum((r+dr,c+dc) in target for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))] for r,c in clues]
            if not native_valid(s,target):
                continue
        if solution_set_evidence(s)['passed']:
            s['variant'] = variant
            return s
    raise RuntimeError(f'{key}: failed to construct two strict relaxed-set witnesses')


def picture(s, cells):
    g = host(s)
    if s['key']=='queens_regions':
        return g._render(cells,s['regions'])
    return g._render(set(map(tuple,s['black'])),{(r,c):v for r,c,v in s['clues']},set(map(tuple,cells)))


def render(s, progress=0, answer=None):
    first=picture(s,[])
    last=picture(s,solve(s) if answer is None else answer)
    return Image.blend(first,last,float(np.clip(progress,0,1)))


def frames(s, answer=None):
    cells = solve(s) if answer is None else answer
    states=[picture(s,[])]
    for i in range(1,len(cells)+1):
        states.append(picture(s,cells[:i]))
    return build_sequential_commit_frames(states,minimum_final_hold=8)


def prompt(s):
    if s['key']=='queens_regions':
        description='The native 6 by 6 Star Battle board has coloured regions, bold region boundaries and numbered rows and columns.'
        rule='Place exactly one gold star in every row, column and coloured region. No two stars may touch, even diagonally.'
        action='Add native gold stars by local fades, preserving every completed star.'
    else:
        description='The native 6 by 6 Akari board has white cells, black wall cells, some wall digits and numbered rows and columns.'
        rule=('Place bulbs on white cells to illuminate every white cell. A bulb illuminates its own cell and travels horizontally and vertically '
              'until a black cell or board edge. No bulb may see another. Each wall digit equals the number of orthogonally adjacent bulbs.')
        action='Add native gold circular bulbs by local fades, with their pale illumination, preserving all completed bulbs and lit cells.'
    extra=(' Also impose the N-Queens placement rule on the same markers: exactly one per row and column, and no two on any full diagonal. '
           'Diagonal conflicts depend on row and column coordinates even across black walls; walls block light only.')
    if s['key']=='queens_regions':extra=extra.split('Diagonal conflicts')[0]
    return ('Input Image: '+description+'\n\nRule: '+rule+extra+
            '\n\nRequired Output: Begin with the unchanged board. '+action+
            ' You may commit correct markers singly or in simultaneous groups, in any order; never remove or move one. '
            'End holding any complete placement satisfying ALL rules. Preserve the grid, region/wall colours, digits, labels and camera. '
            'Add no queen letters, counters, explanatory marks or panels.')


def grade(s, path):
    observed = read_video(path)
    first = np.asarray(picture(s,[]))
    cell,left,top = standard_board_geometry(s['size'],s['size'])
    cells=[]
    for r in range(s['size']):
        for c in range(s['size']):
            cx,cy=left+c*cell+cell//2,top+r*cell+cell//2
            rgb=np.median(observed[-1][cy-3:cy+4,cx-3:cx+4],axis=(0,1))
            if rgb[0]>180 and rgb[1]>100 and rgb[2]<150:
                cells.append((r,c))
    legal = queen_valid(cells,s['size']) and domain_valid(s,cells) and native_valid(s,cells)
    if not legal:
        return dict(whole_video_loaded=True,video_final_state_correct=0.,whole_video_process_correct=0.,decoded_cells=cells)
    last = np.asarray(picture(s,cells))
    metrics,final = terminal_region_gates(observed[-1],first,last,prefix='composition_grid_')
    dynamic=np.max(np.abs(last.astype(int)-first.astype(int)),axis=2)>8
    static=~dynamic
    preserved=all(static_region_errors(f,first,static)[0]<=STATIC_REGION_FLOOR for f in observed)
    low=np.minimum(first,last).astype(float)-46
    high=np.maximum(first,last).astype(float)+46
    envelope=all(static_region_errors(f,np.clip(f,low,high),np.ones(static.shape,bool))[0]<=STATIC_REGION_FLOOR for f in observed)
    fades=True
    for r,c in cells:
        mask=np.zeros(static.shape,np.uint8)
        x,y=left+c*cell,top+r*cell
        gold=(last[y:y+cell,x:x+cell,0]>180)&(last[y:y+cell,x:x+cell,1]>100)&(last[y:y+cell,x:x+cell,2]<150)
        mask[y:y+cell,x:x+cell]=gold
        mask=cv2.erode(mask,np.ones((3,3),np.uint8)).astype(bool)
        readings=[complete_footprint_projection(f,first,last,mask=mask) for f in observed]
        progress=[v['progress'] for v in readings]
        fades &= (max(progress[:3])<.08 and min(progress[-3:])>.92
                  and any(.15<p<.85 for p in progress)
                  and all(b>=a-.06 for a,b in zip(progress,progress[1:]))
                  and all(v['progress_spread']<.35 and v['linear_fit_error']<.25 for v in readings if .1<v['progress']<.9))
    holds=all(matches(f,first) for f in observed[:3]) and all(matches(f,last) for f in observed[-3:])
    return dict(whole_video_loaded=True,video_final_state_correct=float(final),
                whole_video_process_correct=float(final and preserved and envelope and fades and holds),
                decoded_cells=cells,static_preserved=preserved,colour_envelope=envelope,local_commits=fades,**metrics)
