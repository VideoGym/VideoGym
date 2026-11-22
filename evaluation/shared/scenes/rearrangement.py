"""Restricted native token trades with an additional immovable-token rule."""

from contextlib import contextmanager
from itertools import combinations
import random
import tempfile
import numpy as np

from data.rearrangement_game.token_trade_matrix.generator import (
    TokenTradeMatrixGenerator, _shortest_stats,
)
from data.rearrangement_game.core.verifier import _video_semantics


def plans(source):
    initial, target = tuple(source["initial"]), tuple(source["target"])
    edges = frozenset(tuple(e) for e in source["edges"])
    lock = source["locked_socket"]
    unlocked = lambda es: frozenset(e for e in es if lock not in e)
    all_edges = frozenset(combinations(range(5), 2))
    return tuple(_shortest_stats(initial, target, es)
                 for es in (unlocked(edges), edges, unlocked(all_edges)))


def ablations(source):
    # A path is the native answer. Equal terminal holdings are intentional:
    # strict separation of optimal lengths proves ALL relaxed optima illegal
    # under the joint contract, rather than cherry-picking a convenient path.
    return tuple([list(m) for m in result[2]] for result in plans(source))


def sample(key, seed, variant="main"):
    rng = random.Random(seed)
    edges_pool = list(combinations(range(5), 2))
    for _ in range(10000):
        initial, target = list(range(1, 6)), list(range(1, 6))
        rng.shuffle(initial)
        rng.shuffle(target)
        locks = [i for i in range(5) if initial[i] == target[i]]
        if not locks:
            continue
        source = dict(key=key, host="token_trade_matrix", variant=variant,
                      initial=initial, target=target,
                      edges=[list(e) for e in sorted(rng.sample(edges_pool, rng.randint(5, 7)))],
                      locked_socket=rng.choice(locks))
        joint, first, second = plans(source)
        if (joint[0] is not None and 3 <= joint[0] <= 6 and 1 <= joint[1] <= 8
                and first[0] is not None and second[0] is not None
                and joint[0] > max(first[0], second[0])):
            return source
    raise RuntimeError("no dual-nondegenerate fixed-token trade problem")


def solve(source):
    index = {"main": 0, "source_goal": 1, "second_goal": 2}[source["variant"]]
    return [list(move) for move in plans(source)[index][2]]


class Renderer(TokenTradeMatrixGenerator):
    def _render_panel(self, state, panel_w, panel_h, *, moving=None):
        # Rule-removal controls retain the same visible edges even when a
        # non-edge trade becomes legal; only rule semantics change.
        return super()._render_panel(dict(state, edges=self.visible_edges), panel_w, panel_h, moving=moving)


@contextmanager
def renderer(source):
    with tempfile.TemporaryDirectory(prefix="videogym-native-trade-") as directory:
        generator = Renderer(directory)
        generator.visible_edges = source["edges"]
        yield generator


def replay(source, plan=None):
    plan = solve(source) if plan is None else plan
    with renderer(source) as g:
        current = dict(holdings=list(source["initial"]), edges=source["edges"])
        target = dict(holdings=list(source["target"]), edges=source["edges"])
        stable = [np.asarray(g._render_static_scene(current, target))]
        clip = [stable[0].copy() for _ in range(g.hold_frames)]
        for move in plan:
            edge = sorted(move)
            # The native transition requires a legal edge in its internal
            # state. Only relaxed controls can need this temporary edge. The
            # unchanged renderer above always paints the public source edges.
            transition_edges = [list(e) for e in source["edges"]]
            if edge not in transition_edges:
                transition_edges = [*transition_edges[:-1], edge]
            working = dict(current, edges=transition_edges)
            clip.extend(np.asarray(f) for f in g._animate_move(working, move, target))
            current = dict(g._apply_move(working, move), edges=source["edges"])
            stable.append(np.asarray(g._render_static_scene(current, target)))
            clip.extend(stable[-1].copy() for _ in range(g.pause_frames))
        clip.extend(stable[-1].copy() for _ in range(g.hold_frames))
        while len(clip) % 4 != 1:
            clip.append(clip[-1].copy())
        return stable, clip


def frames(source, answer=None):
    return replay(source, answer)[1]


def render(source, progress=0, answer=None):
    from PIL import Image
    states, _ = replay(source, answer)
    return Image.fromarray(states[0] if progress == 0 else states[-1])


def prompt(source):
    with renderer(source) as g:
        description = g._describe_input()
    rule = "Reach the exact right-hand target using the fewest simultaneous two-token swaps. "
    if source["variant"] != "second_goal":
        rule += "Only pairs joined by a visible teal edge may trade. "
    if source["variant"] != "source_goal":
        rule += f"The token in socket {source['locked_socket']+1} is locked: it stays there throughout, so no trade may include that socket. "
    return ("Input Image: " + description + "\n\nRule: " + rule +
            "\n\nRequired Output: Hold the source, then move both selected tokens simultaneously in straight lines "
            "between their sockets, with fixed identity, size and shape. Hold each completed trade and the final target. "
            "Keep every other token, edge, socket, numeral, target panel and chevron unchanged. No extra symbols, cuts or teleports.")


def grade(source, path):
    results = []
    for plan in shortest_plans(source):
        stable, motion = replay(source, plan)
        metrics, passed = _video_semantics(path, stable, motion, task_name="token_trade_matrix")
        results.append({**metrics, "whole_video_loaded": True,
                        "whole_video_process_correct": float(passed),
                        "video_final_state_correct": float(metrics.get("video_terminal_state_pass", False))})
        if passed:
            return results[-1]
    return max(results, key=lambda r: r["video_final_state_correct"])


def shortest_plans(source):
    """Enumerate every optimal legal program, not just the authored reference."""
    from collections import deque
    target, start = tuple(source["target"]), tuple(source["initial"])
    edges = [tuple(e) for e in source["edges"] if source["locked_socket"] not in e]
    distances, queue = {target: 0}, deque([target])

    def swap(state, edge):
        result = list(state)
        a, b = edge
        result[a], result[b] = result[b], result[a]
        return tuple(result)

    while queue:
        state = queue.popleft()
        for edge in edges:
            nxt = swap(state, edge)
            if nxt not in distances:
                distances[nxt] = distances[state] + 1
                queue.append(nxt)

    def walk(state, prefix):
        if state == target:
            yield prefix
        else:
            for edge in edges:
                nxt = swap(state, edge)
                if distances[nxt] == distances[state] - 1:
                    yield from walk(nxt, prefix + [list(edge)])
    return list(walk(start, []))
