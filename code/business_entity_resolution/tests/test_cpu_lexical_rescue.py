"""Rescue-branch accounting preserves the official singleton and missed-truth rules."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ber.cpu_lexical_rescue import choose_calibration, score_added
from ber.metrics import score_pairs


class LexicalRescueTests(unittest.TestCase):
    def test_added_links_include_singleton_harm_and_unretrieved_truth(self):
        reference = pd.DataFrame({"entity_id": ["A", "B", "C", "D"],
                                  "country": ["US", "India", "India", "US"]})
        truth = {"A": {"a", "b"}, "B": set(), "C": {"c"}, "D": set()}
        # A already matched 'a'; neither C's true target nor D's singleton has a baseline link.
        baseline = {"predicted": np.array([1, 0, 0, 0]),
                    "tp": np.array([1, 0, 0, 0]), "actual": np.array([2, 0, 1, 0])}
        left = np.array([0, 1, 2], dtype=np.int64)
        labels = np.array([True, False, False])
        report, scores = score_added(reference, baseline, left, labels,
                                     np.array([True, True, True]), np.ones(4, dtype=bool))
        pool = pd.DataFrame({"entity_id": ["a", "b", "wrong", "other"]})
        pairs = pd.DataFrame({"left_idx": [0, 0, 1, 2], "right_idx": [0, 1, 2, 3]})
        official = score_pairs(reference, pool, pairs, [1.0] * 4, truth, .5)
        self.assertAlmostEqual(report["macro_f05"], official["macro_f05"])
        self.assertAlmostEqual(report["precision"], official["micro_precision"])
        self.assertAlmostEqual(report["recall"], official["micro_recall"])
        self.assertEqual(report["false_links"], 2)
        self.assertEqual((report["added_true_links"], report["added_false_links"]), (1, 2))
        self.assertAlmostEqual(report["singleton_accuracy"], .5)
        self.assertEqual(scores.tolist(), [1.0, 0.0, 0.0, 1.0])

    def test_train_calibration_rejects_a_bad_precision_tradeoff(self):
        chosen, _ = choose_calibration(np.array([.97] * 6 + [.97] * 2),
                                       np.array([True] * 6 + [False] * 2))
        self.assertEqual(chosen["threshold"], 1.01)


if __name__ == "__main__":
    unittest.main()
