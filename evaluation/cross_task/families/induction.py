"""Joint symbol constraints, hidden-machine induction and recursive rewriting."""

from itertools import product

from evaluation.cross_task.drawing import BLUE, INK, TEAL, Canvas, Slot

WORDS = [
    "STONE",
    "NOTES",
    "TONES",
    "ONSET",
    "NOISE",
    "STEIN",
    "INSET",
    "TENON",
    "TENSE",
    "SENSE",
    "TESTS",
    "SITES",
    "TENTS",
    "ONION",
]
RULE_WORDS = ("A", "B", "AA", "AB", "BA", "BB")


def cryptograms(s):
    vocabulary = s["vocabulary"]
    cipher = s["cipher"]
    out = []

    def rec(i, mapping, inverse, chosen):
        if i == len(cipher):
            out.append(tuple(chosen))
            return
        for j, word in enumerate(vocabulary):
            m = mapping.copy()
            inv = inverse.copy()
            for a, b in zip(cipher[i], word):
                if (a in m and m[a] != b) or (b in inv and inv[b] != a):
                    break
                m[a] = b
                inv[b] = a
            else:
                rec(i + 1, m, inv, chosen + [j + 1])

    rec(0, {}, {}, [])
    return out


def transduce(machine, word):
    state = 0
    out = []
    for bit in word:
        index = state * 2 + int(bit)
        state = machine[index]
        out.append(machine[4 + index])
    return "".join(map(str, out))


def rewrite(mapping, word, steps):
    for _ in range(steps):
        word = "".join(mapping[ord(ch) - 65] for ch in word)
    return word


def hypotheses(s):
    if s["goal"] == "hidden_transducer":
        return [
            m
            for m in product(range(2), repeat=8)
            if all(transduce(m, a) == b for a, b in s["examples"])
        ]
    return [
        m
        for m in product(RULE_WORDS, repeat=2)
        if all(rewrite(m, a, 1) == b for a, b in s["examples"])
    ]


def sample(rng, key):
    s = {"goal": key}
    if key == "consistent_cryptogram":
        alphabet = list("EINOST")
        coded = rng.sample(list("abcdef"), 6)
        mapping = dict(zip(alphabet, coded))
        s["vocabulary"] = rng.sample(WORDS, 10)
        chosen = rng.sample(s["vocabulary"], 5)
        s["cipher"] = ["".join(mapping[ch] for ch in w) for w in chosen]
        if len(cryptograms(s)) != 1 or len(set("".join(chosen))) < 6:
            return sample(rng, key)
    elif key == "hidden_transducer":
        while True:
            m = tuple(rng.randrange(2) for _ in range(8))
            words = rng.sample(["".join(p) for p in product("01", repeat=4)], 6)
            s["examples"] = [[w, transduce(m, w)] for w in words]
            s["queries"] = rng.sample(["".join(p) for p in product("01", repeat=6)], 3)
            hs = hypotheses(s)
            if (
                len(hs) == 1
                and len(set(m[4:])) == 2
                and all(transduce(m, w) != w for w in s["queries"])
                and any(transduce(m, "0" + w)[1:] != transduce(m, w) for w in words)
            ):
                break
    else:
        while True:
            mapping = rng.choice([("AB", "A"), ("BA", "AB"), ("BA", "B"), ("AB", "BA")])
            s["examples"] = [
                [w, rewrite(mapping, w, 1)]
                for w in rng.sample(["AAB", "ABA", "BAA", "BAB", "BBA", "ABB"], 3)
            ]
            s["query"] = rng.choice(["AB", "BA", "AA", "BB"])
            s["steps"] = 3
            output = rewrite(mapping, s["query"], 3)
            if len(hypotheses(s)) == 1 and 10 <= len(output) <= 16:
                break
    return s


def solve(s):
    k = s["goal"]
    if k == "consistent_cryptogram":
        return cryptograms(s)
    if k == "hidden_transducer":
        return sorted(
            {
                tuple(int(v) for w in s["queries"] for v in transduce(m, w))
                for m in hypotheses(s)
            }
        )
    return sorted(
        {
            tuple(rewrite(m, s["query"], s["steps"]))
            + ("_",) * (16 - len(rewrite(m, s["query"], s["steps"])))
            for m in hypotheses(s)
        }
    )


def scene(s):
    k = s["goal"]
    c = Canvas()
    slots = []
    if k == "consistent_cryptogram":
        c.panel((26, 82, 486, 257))
        for i, w in enumerate(s["vocabulary"]):
            c.text(
                (47 + i % 2 * 233, 123 + i // 2 * 27), f"{i + 1:2}. {w}", 17, INK, "lm"
            )
        for i, w in enumerate(s["cipher"]):
            y = 295 + i * 40
            c.text((150, y), " ".join(w), 20, BLUE)
            c.arrow((266, y), (354, y), INK)
            slots.append(Slot("number", (413, y), tuple(range(1, 11))))
    elif k == "hidden_transducer":
        for i, (a, b) in enumerate(s["examples"]):
            x = 30 + i % 2 * 245
            y = 109 + i // 2 * 54
            c.panel((x, y - 17, x + 220, y + 22))
            c.text((x + 58, y + 3), a, 18, BLUE)
            c.arrow((x + 104, y + 3), (x + 128, y + 3))
            c.text((x + 172, y + 3), b, 18, TEAL)
        for r, w in enumerate(s["queries"]):
            y = 330 + r * 62
            c.text((84, y), w, 17, BLUE)
            for i in range(6):
                p = (202 + i * 51, y)
                slots.append(Slot("number", p, (0, 1), (BLUE, TEAL)))
    else:
        for i, (a, b) in enumerate(s["examples"]):
            y = 125 + i * 61
            c.panel((36, y - 24, 476, y + 24))
            c.text((133, y), a, 23, BLUE)
            c.arrow((205, y), (253, y))
            c.text((363, y), b, 21, TEAL)
        c.text((256, 309), s["query"], 22, BLUE)
        c.arrow((256, 327), (256, 346), INK, 1.5)
        for i in range(16):
            x = 53 + i % 8 * 58
            y = 377 + i // 8 * 69
            slots.append(
                Slot(
                    "number", (x, y), ("A", "B", "_"), (TEAL, BLUE, "#8594a3"), size=21
                )
            )
    return c, slots
