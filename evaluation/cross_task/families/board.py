"""Adversarial Othello endgames; optimize minimax outcome rather than immediate flips."""

from functools import lru_cache

from data.common.pipeline.board_rendering import draw_go_stone
from evaluation.cross_task.drawing import INK, TEAL, Canvas, cell_slots

DIRS = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c]


def moves(board, player):
    result = []
    for i in range(16):
        if board[i]:
            continue
        r, c = divmod(i, 4)
        flips = []
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            line = []
            while 0 <= rr < 4 and 0 <= cc < 4 and board[rr * 4 + cc] == -player:
                line.append(rr * 4 + cc)
                rr += dr
                cc += dc
            if line and 0 <= rr < 4 and 0 <= cc < 4 and board[rr * 4 + cc] == player:
                flips += line
        if flips:
            b = list(board)
            b[i] = player
            for j in flips:
                b[j] = player
            result.append((i, tuple(b)))
    return result


@lru_cache(200000)
def minimax(board, player):
    successors = moves(board, player)
    if not successors:
        if not moves(board, -player):
            return sum(board)
        return minimax(board, -player)
    return (max if player == 1 else min)(minimax(b, -player) for _, b in successors)


def sample(rng, key):
    for _ in range(3000):
        b = [0] * 16
        b[5] = b[10] = 1
        b[6] = b[9] = -1
        player = 1
        for _ in range(rng.choice([5, 6, 7])):
            choices = moves(tuple(b), player)
            if not choices:
                player = -player
                choices = moves(tuple(b), player)
            if not choices:
                break
            _, b = rng.choice(choices)
            player = -player
        if player == -1:
            b = tuple(-v for v in b)
        choices = moves(tuple(b), 1)
        scores = [minimax(q, -1) for _, q in choices]
        if len(choices) >= 3 and len(set(scores)) >= 2 and b.count(0) >= 5:
            # Ensure at least one superficially attractive greedy choice fails.
            greedy = max(sum(q) for _, q in choices)
            if any(
                sum(q) == greedy and sc < max(scores)
                for (_, q), sc in zip(choices, scores)
            ):
                return {"goal": key, "board": list(b)}
    raise ValueError("no non-greedy Othello endgame")


def solve(s):
    choices = moves(tuple(s["board"]), 1)
    best = max(minimax(q, -1) for _, q in choices)
    return [
        tuple({-1: 2, 0: 0, 1: 1}[v] for v in q)
        for _, q in choices
        if minimax(q, -1) == best
    ]


def scene(s):
    c = Canvas()
    xy = c.grid(4, 4, 152, 83, 52)
    for i, v in enumerate(s["board"]):
        x, y = xy[i]
        c.d.rectangle((x - 25, y - 25, x + 25, y + 25), fill=TEAL)
        if v:
            draw_go_stone(c.d, centre=(x, y), radius=18, black=v == 1)
    c.arrow((256, 305), (256, 326), INK, 2)
    xy = c.grid(4, 4, 180, 340, 38)
    slots = cell_slots(xy, 38, (0, 1, 2), (TEAL, INK, "#eee9d9"))
    for slot in slots:
        slot.kind = "board_cell"
    return c, slots
