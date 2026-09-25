"""P1: no truth-edge creation, exact singleton scoring, disjoint calibration."""
from itertools import combinations
from math import comb
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from ber.v2_decision import (CountPosterior, address_sanity, adjust_source, assignment,
                             balanced_folds, composite_candidates, composite_key,
                             fit_calibration, predict_calibration, score_decisions)


class DecisionTests(unittest.TestCase):
    def test_official_macro_includes_retrieval_miss_and_empty_singleton(self):
        # Second S1 has no candidates; third is a singleton with a false claim.
        data = SimpleNamespace(references=pd.DataFrame({"entity_id": ["a", "b", "c"]}),
                               left=np.array([0, 2]), labels=np.array([True, False]),
                               actual=np.array([2, 0, 0]), country=np.array(["US", "US", "India"]))
        score, per_ref = score_decisions(data, np.array([True, True]))
        np.testing.assert_allclose(per_ref, [5 / 6, 1, 0])
        self.assertAlmostEqual(score["macro_f05"], (5 / 6 + 1) / 3)
        self.assertEqual(score["false_positive_links"], 1)
        self.assertEqual(score["missed_true_links"], 1)
        self.assertEqual(score["singleton_accuracy"], .5)
        self.assertEqual(score["by_country"]["US"]["reference_count"], 2)
        self.assertEqual(score["by_country"]["India"]["macro_f05"], 0)

    def test_one_to_one_margin_and_stable_ties(self):
        claim = np.ones(5, dtype=bool)
        target = np.array([7, 7, 8, 8, 9])
        left = np.array([1, 0, 1, 0, 2])
        score = np.array([.90, .90, .82, .97, .99])
        pure, details = assignment(claim, target, score, left, 0)
        np.testing.assert_array_equal(pure, [False, True, False, True, True])
        self.assertEqual(details["removed_claims"], 2)
        margin, details = assignment(claim, target, score, left, .2)
        np.testing.assert_array_equal(margin, claim)
        self.assertEqual(details["unresolved_extra_claims"], 2)
        mid, _ = assignment(claim, target, score, left, .1)
        np.testing.assert_array_equal(mid, [True, True, False, True, True])
        self.assertRaises(ValueError, assignment, claim, target[:-1], score, left, 0)

    def test_count_rule_matches_enumerated_bayesian_expected_f05(self):
        prior = np.array([.2, .8])
        posterior = CountPosterior(prior, 1.0, .2)
        p = np.array([.9, .1])
        best, expected = posterior.best_prefix(p, .001, 2)
        # With discovery=1, either T=0 or exactly one of the two candidates is
        # true. At T=1, a subset's weight is prior[T]*odds_likelihood/C(2,1).
        odds = p / (1 - p) / (.2 / .8)
        weights = np.array([prior[0], prior[1] * odds[0] / comb(2, 1),
                            prior[1] * odds[1] / comb(2, 1)])
        weights /= weights.sum()
        actual = np.array([weights[0], weights[1], (weights[1] + weights[2]) * 5 / 9])
        np.testing.assert_allclose(expected, actual, atol=1e-12)
        self.assertEqual(best, 1)
        abstain, _ = posterior.best_prefix(np.empty(0), .001, 2)
        self.assertEqual(abstain, 0)

    def test_count_prior_accounts_for_unscored_pairs_and_missed_truth(self):
        prior = np.array([.1, .3, .6])
        recall = .8
        base_rate = .1
        p = np.array([.8, .4, .02])  # final pair was not CE-scored
        model = CountPosterior(prior, recall, base_rate)
        best, observed = model.best_prefix(p[:2], p[2], 3)
        lr = p / (1 - p) / (base_rate / (1 - base_rate))
        numerators = np.zeros(3)
        denominator = 0.0
        for t, mass in enumerate(prior):
            for j in range(min(t, 3) + 1):
                count_weight = (mass * comb(t, j) * recall**j * (1 - recall)**(t - j)
                                / comb(3, j))
                for subset in combinations(range(3), j):
                    likelihood = count_weight * np.prod(lr[list(subset)])
                    denominator += likelihood
                    for k in range(3):
                        utility = ((t == 0) if k == 0 else
                                   5 * sum(i < k for i in subset) / (4 * k + t))
                        numerators[k] += likelihood * utility
        np.testing.assert_allclose(observed, numerators / denominator, rtol=1e-12)
        self.assertEqual(best, int(np.argmax(observed)))

    def test_country_isotonic_is_monotone_and_validation_truth_never_fits(self):
        country = np.array(["US", "India", "US", "India"])
        left = np.repeat(np.arange(4), 300)
        raw = np.tile(np.linspace(.01, .99, 300), 4)
        gate = np.tile(np.array([True] * 160 + [False] * 140), 4)
        labels = raw > .30
        fit = np.array([True, True, False, False])
        prior = {"gated_pair_prevalence": .5, "ungated_pair_prevalence": .02,
                 "country_pair_prior": {key: {"gated_pair_prevalence": .5,
                                               "ungated_pair_prevalence": .02} for key in country}}
        model = fit_calibration(raw, gate, labels, left, country, fit, prior)
        altered = labels.copy()
        altered[left >= 2] = ~altered[left >= 2]
        alternate_model = fit_calibration(raw, gate, altered, left, country, fit, prior)
        self.assertEqual(model, alternate_model)
        predictions = predict_calibration(model, raw, gate, left, country)
        for i in (2, 3):
            values = predictions[(left == i) & gate]
            self.assertTrue((np.diff(values) >= -1e-12).all())
        french_unlabeled = predict_calibration(model, np.array([.2, .7]),
                                                np.array([True, False]), np.array([0, 0]),
                                                np.array(["France"]))
        self.assertTrue(((french_unlabeled > 0) & (french_unlabeled < 1)).all())
        ids = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(40)],
                            "country": ["US", "India"] * 20})
        folds = balanced_folds(ids, 42)
        np.testing.assert_array_equal(folds, balanced_folds(ids, 42))
        for key in ("US", "India"):
            self.assertEqual(np.count_nonzero((ids.country == key) & (folds == 0)), 10)

    def test_source_bonus_requires_opposite_anchor_and_does_not_feedback(self):
        p = np.array([.98, .40, .1, .96])
        src = np.array(["S2-", "S3-", "S3-", "S2-"])
        updated = adjust_source(p, np.array([True, True, True, False]), src, .5)
        self.assertEqual(updated[0], p[0])
        self.assertEqual(updated[3], p[3])
        self.assertGreater(updated[1], p[1])
        self.assertGreater(updated[2], p[2])
        both = np.array([.98, .98])
        np.testing.assert_array_equal(adjust_source(both, np.array([True, True]),
                                                    src[:2], .5), both)

    def test_composite_key_uses_only_existing_pairs_and_sanity(self):
        self.assertEqual(composite_key("Acme Inc LLC", "12 Main St, Unit B", "US"),
                         composite_key("Acme", "12 MAIN STREET B", "US"))
        self.assertFalse(address_sanity("12 Main Street, unit A", "12 Remote Road, unit B"))
        refs = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "business_name": ["Acme LLC", "Acme LLC"],
                             "business_address": ["12 Main Street", "12 Main Street"],
                             "country": ["US", "US"]})
        pool = pd.DataFrame({"entity_id": ["S2-1", "S3-2"], "business_name": ["Acme Inc", "Acme"],
                             "business_address": ["12 Main Street", "12 Remote Road"],
                             "country": ["US", "US"]})
        data = SimpleNamespace(train_reference=refs.iloc[1:], references=refs.iloc[:1], pool=pool,
                               right=np.array([0, 1]), left=np.array([0, 0]))
        mask, info = composite_candidates(data)
        self.assertFalse(mask.any())  # Training S1 shares key: uniqueness is not established.
        data.train_reference = refs.iloc[0:0]
        mask, info = composite_candidates(data)
        np.testing.assert_array_equal(mask, [True, False])
        self.assertEqual(info["actual_retrieved_tune_pairs"], 2)
        self.assertFalse(info["retrieval_branch_added"])


if __name__ == "__main__":
    unittest.main()
