import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.rival_text_arena import rival_of, rival_owners  # noqa: E402


class RivalOwnerTests(unittest.TestCase):
    def test_leader_compares_with_runner_up(self):
        left = np.array([1, 2, 3])
        right = np.array([0, 0, 0])
        blend = np.array([0.9, 0.4, 0.2], dtype=np.float32)
        best, second, top1 = rival_owners(left, right, blend, 1)
        self.assertEqual(int(best[0]), 1)
        self.assertEqual(int(second[0]), 2)
        self.assertAlmostEqual(float(top1[0]), 0.9, places=6)
        rival = rival_of(np.array([0, 1, 2]), left, right, blend, best, second, top1)
        self.assertEqual(rival.tolist(), [2, 1, 1])

    def test_tie_compares_the_two_leaders(self):
        left = np.array([5, 6])
        right = np.array([0, 0])
        blend = np.array([0.5, 0.5], dtype=np.float32)
        best, second, top1 = rival_owners(left, right, blend, 1)
        self.assertEqual(int(best[0]), 5)
        self.assertEqual(int(second[0]), 6)
        rival = rival_of(np.array([0, 1]), left, right, blend, best, second, top1)
        self.assertEqual(rival.tolist(), [6, 5])

    def test_single_claimant_has_no_rival(self):
        left = np.array([4])
        right = np.array([0])
        blend = np.array([0.8], dtype=np.float32)
        best, second, top1 = rival_owners(left, right, blend, 2)
        self.assertEqual(int(best[1]), -1)
        rival = rival_of(np.array([0]), left, right, blend, best, second, top1)
        self.assertEqual(rival.tolist(), [-1])


if __name__ == "__main__":
    unittest.main()
