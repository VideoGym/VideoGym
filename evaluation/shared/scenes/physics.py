"""Equal-mass velocity exchange composed with specular boundary reflection."""

from fractions import Fraction
from functools import lru_cache
import random
import numpy as np

from data.physics.core.collision_physics import ElasticCollisionTrackPattern
from data.physics.core.source_replay import PhysicsReplay
from data.physics.core.verifier import _verify_video


@lru_cache(maxsize=128)
def _simulate(starts, velocities, horizon, boundaries, condition):
    positions = list(map(Fraction, starts))
    speeds = list(map(Fraction, velocities))
    t, end = Fraction(0), Fraction(horizon)
    trajectories = [[(t, x)] for x in positions]
    events = []
    for _ in range(64):
        candidates = []
        if condition in ("joint", "collision"):
            for i in range(len(positions) - 1):
                closing = speeds[i] - speeds[i+1]
                if closing > 0:
                    candidates.append(((positions[i+1] - positions[i] - 1) / closing, "collision", i))
        if condition in ("joint", "wall"):
            for i, (x, v) in enumerate(zip(positions, speeds)):
                if v:
                    boundary = boundaries[1 if v > 0 else 0]
                    candidates.append(((Fraction(boundary) - x) / v, "wall", i))
        candidates = sorted(c for c in candidates if c[0] >= 0 and t + c[0] < end)
        if not candidates:
            break
        dt, kind, index = candidates[0]
        if len(candidates) > 1 and candidates[1][0] == dt:
            raise ValueError("simultaneous contact requires a separate collision convention")
        if dt == 0:
            raise ValueError("zero-time contact")
        positions = [x + v * dt for x, v in zip(positions, speeds)]
        t += dt
        for path, x in zip(trajectories, positions):
            path.append((t, x))
        if kind == "wall":
            speeds[index] = -speeds[index]
        else:
            speeds[index], speeds[index+1] = speeds[index+1], speeds[index]
        events.append({"time": t, "kind": kind})
    else:
        raise ValueError("event budget exhausted")
    positions = [x + v * (end - t) for x, v in zip(positions, speeds)]
    for path, x in zip(trajectories, positions):
        path.append((end, x))
    return {"trajectories": trajectories, "events": events, "final_positions": positions}


def simulate(source, condition="joint"):
    return _simulate(tuple(source["starts"]), tuple(source["velocities"]), source["horizon"],
                     tuple(source["boundaries"]), condition)


def ablations(source):
    return tuple([float(x) for x in simulate(source, c)["final_positions"]]
                 for c in ("joint", "collision", "wall"))


def sample(key, seed, variant="main"):
    rng = random.Random(seed)
    for _ in range(20000):
        starts = sorted(rng.sample(range(4, 10), 3))
        if any(b-a < 2 for a,b in zip(starts, starts[1:])):
            continue
        source = dict(key=key, host="elastic_collision_track", variant=variant,
                      starts=starts, velocities=[rng.choice((-2,-1,0,1,2)) for _ in starts],
                      horizon=rng.choice((3,4)), boundaries=[3,10])
        try:
            traces = [simulate(source, c) for c in ("joint", "collision", "wall")]
        except ValueError:
            continue
        if any(not 1 <= x <= 12 for trace in traces for path in trace["trajectories"] for _,x in path):
            continue
        if {e["kind"] for e in traces[0]["events"]} != {"wall", "collision"}:
            continue
        times = [0., *[float(e["time"]) for e in traces[0]["events"]], source["horizon"]]
        if min(b-a for a,b in zip(times, times[1:])) < .25:
            continue
        joint, first, second = ablations(source)
        if all(max(abs(a-b) for a,b in zip(joint, control)) >= 1 for control in (first,second)):
            return source
    raise RuntimeError("no observable dual-nondegenerate wall/collision trajectory")


def condition(source):
    return {"main":"joint", "source_goal":"collision", "second_goal":"wall"}[source["variant"]]


def solve(source):
    return [float(x) for x in simulate(source, condition(source))["final_positions"]]


class Renderer(ElasticCollisionTrackPattern):
    def _replay(self, task):
        return simulate(task, condition(task))


def render(source, progress=0, answer=None):
    generator = object.__new__(Renderer)
    if answer is not None and progress >= 1:
        generator._positions_at = lambda task, p: answer
    return generator._render_progress(source, progress, mark_answer=False)


def frames(source, answer=None):
    n = source["horizon"] * 16
    result = [np.asarray(render(source,0))] * 5
    result.extend(np.asarray(render(source, i/n, answer)) for i in range(1,n+1))
    result.extend([np.asarray(render(source,1,answer))] * 8)
    return result


def prompt(source):
    rules = f"Run for exactly {source['horizon']} seconds at the shown initial velocities. Each ball has diameter one tick. "
    if condition(source) in ("joint","collision"):
        rules += "Touching equal-mass balls exchange velocities instantly and never pass through each other. "
    else:
        rules += "Ignore ball-ball contact; labelled balls pass through one another. "
    if condition(source) in ("joint","wall"):
        rules += ("The two reflecting boundaries are specified by ticks 3 and 10: whenever a ball's centre reaches either tick, "
                  "reverse its velocity instantly without changing its magnitude. These are the same specular reflection rule as a perpendicular wall hit. ")
    return ("Input Image: The unchanged native 12-tick rail holds labelled coloured balls. Each blue arrow gives an initial "
            "direction and its number gives ticks per second; a ball without an arrow is stationary.\n\nRule: " + rules +
            "\n\nRequired Output: Hold the input, fade the initial velocity arrows as movement begins, and roll the same balls "
            "continuously on the rail through every contact and reflection. Freeze at the specified time and hold. Keep "
            "ball identities, sizes, colours, rail, ticks, labels, card and camera fixed. Add no trail, wall drawing, clock or caption.")


def grade(source, path):
    simulation = simulate(source)
    final = solve(source)
    regions = tuple((f"ball_{i}", (ElasticCollisionTrackPattern._tick_x(x)-23, 277,
                                   ElasticCollisionTrackPattern._tick_x(x)+23, 323))
                    for i,x in enumerate(final))
    replay = PhysicsReplay(final, lambda p, opacity: render(source,p),
                           tuple([0., *[float(e["time"])/source["horizon"] for e in simulation["events"]], 1.]),
                           "native_exchange_and_reflection", member_regions=regions)
    return _verify_video(path, replay)
