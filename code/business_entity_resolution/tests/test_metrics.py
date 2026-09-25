"""Entity-level scoring checks against hand-calculated challenge examples."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ber.metrics import candidate_report, entity_f05, score_pairs, tune_threshold


def references(ids, countries=None):
    return pd.DataFrame({
        "entity_id": ids,
        "country": countries if countries is not None else ["US"] * len(ids),
    })


def candidates(left, right):
    return pd.DataFrame({
        "left_idx": pd.Series(left, dtype="int64"),
        "right_idx": pd.Series(right, dtype="int64"),
    })


class EntityF05Tests(unittest.TestCase):
    def test_official_example_penalizes_false_merge(self):
        self.assertAlmostEqual(entity_f05(["a", "b"], ["a", "b", "c"]), 5 / 7)
        self.assertAlmostEqual(entity_f05(["a", "b", "c"], ["a", "b"]), 10 / 11)

    def test_empty_and_duplicate_id_conventions(self):
        self.assertEqual(entity_f05([], []), 1.0)
        self.assertEqual(entity_f05([], ["wrong"]), 0.0)
        self.assertEqual(entity_f05(["missing"], []), 0.0)
        self.assertEqual(entity_f05(["a"], ["a", "a"]), 1.0)


class PairScoringTests(unittest.TestCase):
    def setUp(self):
        self.reference = references(["A", "B", "C", "D"], ["US", "US", "India", "India"])
        self.pool = pd.DataFrame({"entity_id": ["a", "b", "c", "wrong"]})
        self.truth = {"A": {"a", "b"}, "B": set(), "C": {"c"}, "D": set()}
        self.pairs = candidates([0, 0, 3], [0, 3, 3])
        self.probabilities = np.array([0.8, 0.8, 0.8])

    def test_macro_includes_no_candidate_entities_and_unretrieved_truth(self):
        # A = 0.5, B = 1, C = 0 (no candidates), D = 0 (false merge).
        result = score_pairs(self.reference, self.pool, self.pairs, self.probabilities,
                             self.truth, 0.5)
        self.assertAlmostEqual(result["macro_f05"], 0.375)
        self.assertAlmostEqual(result["f05_US"], 0.75)
        self.assertEqual(result["f05_India"], 0.0)
        self.assertEqual(result["reference_count"], 4)
        self.assertEqual(result["true_links"], 3)
        self.assertEqual(result["predicted_links"], 3)
        self.assertAlmostEqual(result["micro_precision"], 1 / 3)
        self.assertAlmostEqual(result["micro_recall"], 1 / 3)
        self.assertEqual(result["singleton_accuracy"], 0.5)

    def test_mask_scores_only_selected_reference_entities(self):
        result = score_pairs(self.reference, self.pool, self.pairs, self.probabilities,
                             self.truth, 0.8, mask=[True, True, False, False])
        self.assertAlmostEqual(result["macro_f05"], 0.75)
        self.assertEqual(result["reference_count"], 2)
        self.assertEqual(result["true_links"], 2)
        self.assertEqual(result["predicted_links"], 2)
        self.assertNotIn("f05_India", result)

    def test_empty_candidates_keep_every_reference_in_macro(self):
        result = score_pairs(self.reference, self.pool, candidates([], []), [],
                             self.truth, 0.5)
        self.assertEqual(result["macro_f05"], 0.5)
        self.assertEqual(result["true_links"], 3)
        self.assertEqual(result["singleton_accuracy"], 1.0)

    def test_candidate_oracle_does_not_hide_missed_truth(self):
        report = candidate_report(self.reference, self.pool, self.pairs, self.truth)
        self.assertAlmostEqual(report["link_recall"], 1 / 3)
        self.assertEqual(report["complete_match_set_recall"], 0.0)
        # Oracle A = 5/6, B = 1, C = 0, D = 1.
        self.assertAlmostEqual(report["oracle_macro_f05"], 17 / 24)


class ThresholdTests(unittest.TestCase):
    def test_tuning_ignores_eval_preferences(self):
        reference = references(["tune", "eval"])
        pool = pd.DataFrame({"entity_id": ["wrong", "correct"]})
        pairs = candidates([0, 1], [0, 1])
        threshold, trace = tune_threshold(
            reference, pool, pairs, [0.8, 0.8], {"tune": set(), "eval": {"correct"}},
            [True, False], thresholds=[0.5, 0.9],
        )
        self.assertEqual(threshold, 0.9)
        self.assertEqual([row["macro_f05"] for row in trace], [0.0, 1.0])

    def test_default_threshold_search_can_choose_no_matches(self):
        reference = references(["singleton"])
        pool = pd.DataFrame({"entity_id": ["wrong"]})
        pairs = candidates([0], [0])
        threshold, trace = tune_threshold(
            reference, pool, pairs, [0.999], {"singleton": set()}, [True],
        )
        self.assertGreater(threshold, 0.999)
        self.assertEqual(max(row["macro_f05"] for row in trace), 1.0)


if __name__ == "__main__":
    unittest.main()
