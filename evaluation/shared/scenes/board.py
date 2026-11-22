"""Execute a prescribed Go program, then optimize and execute a capture on its output."""

import random
import numpy as np
from data.board_game.core.intersection_games import (
    NumericIntersectionRenderer, StoneMove, apply_go_move, freeze_board,
    solve_go_max_capture, build_go_max_capture_scenario,
)
from data.board_game.go_rule_simulator.generator import _append_local_transition
from evaluation.shared.scenes.fades import grade_fades
from evaluation.shared.task_io import read_video


def trace(source, condition="joint"):
    board = freeze_board(source["initial"], size=7)
    states = [board]

    def execute(move):
        nonlocal board
        step = apply_go_move(board, move)
        states.append(step.placed_board)
        if step.captured_cells:
            states.append(step.final_board)
        board = step.final_board

    if condition in ("joint", "program"):
        for operation in source["operations"]:
            execute(StoneMove(operation["colour"], tuple(operation["cell"])))
    if condition in ("joint", "search"):
        solution = solve_go_max_capture(board)
        if len(solution.best_cells) != 1:
            raise ValueError("max-capture goal must have a unique optimum")
        execute(StoneMove("B", solution.best_cells[0]))
    return states


def ablations(source):
    return tuple(trace(source, c)[-1] for c in ("joint", "program", "search"))


def sample(key, seed, variant="main"):
    rng = random.Random(seed)
    for _ in range(2000):
        native = build_go_max_capture_scenario(rng.randrange(2**31))
        initial, current = native.initial_board, native.initial_board
        operations = []
        for _ in range(2):
            legal = []
            for r in range(7):
                for c in range(7):
                    if current[r][c] is not None:
                        continue
                    for colour in ("B", "W"):
                        try:
                            legal.append(apply_go_move(current, StoneMove(colour,(r,c))))
                        except ValueError:
                            continue
            if not legal:
                break
            step = rng.choice(legal)
            current = step.final_board
            operations.append({"colour":step.move.colour,"cell":list(step.move.cell)})
        if len(operations) != 2:
            continue
        try:
            after = solve_go_max_capture(current)
        except ValueError:
            continue
        # The prescribed program must change WHICH action is optimal, not
        # merely alter an unrelated corner while the search remains identical.
        if len(after.best_cells) != 1 or after.best_cells == native.solution.best_cells:
            continue
        source = dict(key=key,host="go_max_capture_search",variant=variant,
                      initial=[list(row) for row in initial],operations=operations)
        joint, first, second = ablations(source)
        if joint != first and joint != second:
            return source
    raise RuntimeError("no nondegenerate prescribed-program/max-capture composition")


def condition(source):
    return {"main":"joint", "source_goal":"program", "second_goal":"search"}[source["variant"]]


def solve(source):
    return {"board":[list(row) for row in trace(source,condition(source))[-1]]}


def pictures(source, answer=None):
    renderer = NumericIntersectionRenderer(7)
    states = trace(source,condition(source))
    if answer is not None:
        states[-1] = answer["board"]
    return [renderer.render(state) for state in states]


def render(source, progress=0, answer=None):
    states = pictures(source,answer)
    return states[0] if progress == 0 else states[-1]


def frames(source, answer=None):
    states = [np.asarray(image) for image in pictures(source,answer)]
    result = [states[0].copy() for _ in range(5)]
    for target in states[1:]:
        _append_local_transition(result,result[-1],target)
        result.extend(target.copy() for _ in range(4))
    result.extend(states[-1].copy() for _ in range(4))
    return result


def prompt(source):
    operations = "; ".join(f"{i}) place {m['colour']} at column {m['cell'][1]+1}, row {m['cell'][0]+1}"
                           for i,m in enumerate(source["operations"],1))
    rules = ("B is Black and W is White. Orthogonally connected stones of one colour form a group; its liberties are "
             "orthogonally adjacent empty intersections. After each placement remove all adjacent opposing groups with no "
             "liberties simultaneously, then test that the placed stone's own group has a liberty. No pass, ko or history rule. ")
    if condition(source) in ("joint","program"):
        rules += f"First execute this explicitly coloured program in order: {operations}. Each operation is legal when reached. "
    if condition(source) in ("joint","search"):
        rules += ("On the resulting board, choose the unique legal Black placement that captures the most White stones, "
                  "then execute that one placement and its capture. The maximum is positive. Do not optimize on the original "
                  "board before executing the prescribed program. ")
    return ("Input Image: The unchanged native 7 by 7 Go intersection board has black and white stones, with columns "
            "numbered above and rows at the left, both starting at 1.\n\nRule: "+rules+
            "\n\nRequired Output: Hold the input. For every executed move, fade the complete fixed-size new stone in at its "
            "intersection and hold; fade all captured stones out together and hold before proceeding. End in a stable hold. "
            "Preserve every uninvolved stone, the wood board, grid, labels and camera. No trial placements, rings, badges, "
            "new symbols, sliding, growing or cuts.")


def grade(source,path):
    return grade_fades(read_video(path),pictures(source))
