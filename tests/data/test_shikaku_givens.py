import unittest

from data.constraint_grid.core.terminal_decode import _shikaku_solutions


class ShikakuGivenTests(unittest.TestCase):
    def test_fixed_rectangles_constrain_search_before_solution_cap(self):
        state = {"rows": 2, "cols": 2, "clues": [[0, 0, 2], [1, 1, 2]]}
        unrestricted = _shikaku_solutions(state)
        self.assertEqual(len(unrestricted), 2)
        desired = unrestricted[1]
        first, last = desired[0]
        state["given_rectangles"] = [[first[0] - 1, first[1] - 1, last[0] - 1, last[1] - 1]]
        self.assertEqual(_shikaku_solutions(state, limit=1), [desired])


if __name__ == "__main__":
    unittest.main()
