"""Decision layer on arrays: metric, bootstrap, components, ownership rules and expected-F0.5 decoding."""
from __future__ import annotations

import itertools
import math
import unittest
from pathlib import Path

import numpy as np

from ber.arena_decision import (break_even, claim_graph_components, component_bootstrap, entity_f05, evaluate,
                                expected_f05_decode, iterated_best_response, name_only_rule, one_owner)

TUNE_DIR = Path("/home/sidvish/Documents/Work/Personal/az-ml-v2/artifacts/v2_p1_decision")
TRAIN_DIR = Path("/home/sidvish/Documents/Work/Personal/az-ml-new/data/raw/dataset/train")


def brute_expected(p, subset, lam, u_terms=40):
    """E[F0.5] of predicting ``subset`` by enumerating every truth outcome and Poisson term."""
    pmf = [math.exp(-lam) * lam ** u / math.factorial(u) for u in range(u_terms)] if lam > 0 else [1.0]
    total = 0.0
    k = len(subset)
    for z in itertools.product((0, 1), repeat=len(p)):
        prob = math.prod(pi if zi else 1 - pi for pi, zi in zip(p, z))
        tp = sum(z[i] for i in subset)
        for u, pu in enumerate(pmf):
            t = sum(z) + u
            f = 1.0 if (t == 0 and k == 0) else (5 * tp / (4 * k + t) if 4 * k + t else 0.0)
            total += prob * pu * f
    return total


def brute_best(p, lam, allowed=None):
    items = [i for i in range(len(p)) if allowed is None or allowed[i]]
    return max(brute_expected(p, s, lam) for r in range(len(items) + 1) for s in itertools.combinations(items, r))


class MetricTests(unittest.TestCase):
    def test_entity_f05_conventions(self):
        np.testing.assert_allclose(entity_f05([0, 0, 0, 2, 2], [0, 1, 0, 3, 2], [0, 0, 2, 2, 3]),
                                   [1.0, 0.0, 0.0, 10 / 14, 10 / 11])

    def test_evaluate_counts_every_s1_and_unretrieved_links(self):
        # S1 0: singleton, no candidates. S1 1: singleton with a false link. S1 2: one true link accepted and
        # one never retrieved (true_count 2). S1 3: true link retrieved but rejected.
        s1 = np.array([1, 2, 2, 3])
        rec = np.array([0, 1, 2, 3])
        label = np.array([False, True, False, True])
        accept = np.array([True, True, False, False])
        true_count = np.array([0, 0, 2, 1])
        country = np.array(["US", "US", "India", "India"])
        kind = np.array(["empty", "full", "full", "empty"])
        res = evaluate(accept, s1, label, true_count, country=country, rec_idx=rec, rec_kind=kind,
                       return_per_s1=True)
        np.testing.assert_allclose(res["per_s1"], [1.0, 0.0, 5 / 6, 0.0])
        self.assertAlmostEqual(res["macro_f05"], (1 + 0 + 5 / 6 + 0) / 4)
        self.assertEqual((res["false_links"], res["true_links"], res["predicted_links"]), (1, 3, 2))
        self.assertAlmostEqual(res["recall"], 1 / 3)
        self.assertAlmostEqual(res["singleton_accuracy"], 0.5)
        self.assertAlmostEqual(res["by_country"]["US"]["macro_f05"], 0.5)
        self.assertEqual(res["by_country"]["India"]["false_links"], 0)
        self.assertEqual(res["by_record_kind"]["empty"]["false_accepted"], 1)
        self.assertAlmostEqual(res["by_record_kind"]["empty"]["retrieved_recall"], 0.0)
        pop = evaluate(accept, s1, label, true_count, population=np.array([True, False, True, False]))
        self.assertAlmostEqual(pop["macro_f05"], (1 + 5 / 6) / 2)

    def test_evaluate_rejects_inconsistent_true_count(self):
        with self.assertRaises(ValueError):
            evaluate(np.array([True]), np.array([0]), np.array([True]), np.array([0]))


class BootstrapTests(unittest.TestCase):
    def test_single_component_has_a_degenerate_interval(self):
        a = np.array([0.5, 1.0, 0.0])
        b = np.array([1.0, 1.0, 0.5])
        res = component_bootstrap(a, b, np.zeros(3), n=200, seed=1)
        self.assertAlmostEqual(res["delta"], 1 / 3)
        self.assertAlmostEqual(res["ci_low"], 1 / 3)
        self.assertAlmostEqual(res["ci_high"], 1 / 3)

    def test_components_are_resampled_whole(self):
        rng = np.random.default_rng(0)
        comp = np.repeat(np.arange(50), 20)
        a = np.zeros(1000)
        b = np.repeat(rng.normal(0.0, 0.1, 50), 20)  # delta constant within a component
        by_comp = component_bootstrap(a, b, comp, n=2000, seed=2)
        by_s1 = component_bootstrap(a, b, np.arange(1000), n=2000, seed=2)
        self.assertAlmostEqual(by_comp["delta"], b.mean())
        self.assertLess(by_comp["ci_low"], by_comp["delta"])
        self.assertGreater(by_comp["ci_high"], by_comp["delta"])
        # Clustered differences give a wider interval than treating S1s as independent.
        self.assertGreater(by_comp["ci_high"] - by_comp["ci_low"], 2 * (by_s1["ci_high"] - by_s1["ci_low"]))


class ComponentTests(unittest.TestCase):
    def test_small_graph(self):
        s1 = np.array([0, 1, 2, 3, 4, 4, 5])
        rec = np.array([10, 10, 10, 11, 11, 12, 12])
        blend = np.array([0.9, 0.5, 0.01, 0.2, 0.3, 0.4, 0.6])
        comp = claim_graph_components(s1, rec, blend, 0.05, n_s1=7)
        self.assertEqual(comp[0], comp[1])
        self.assertNotEqual(comp[0], comp[2])
        self.assertTrue(comp[3] == comp[4] == comp[5])
        self.assertEqual(len(set(comp.tolist())), 4)  # {0,1}, {2}, {3,4,5}, {6}

    def test_matches_union_find(self):
        rng = np.random.default_rng(3)
        n_s1, n_pairs = 300, 900
        s1 = rng.integers(0, n_s1, n_pairs)
        rec = rng.integers(0, 400, n_pairs)
        blend = rng.random(n_pairs)
        comp = claim_graph_components(s1, rec, blend, 0.5, n_s1=n_s1, chunk=64)
        parent = list(range(n_s1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        first = {}
        for s, r, b in zip(s1, rec, blend):
            if b >= 0.5:
                if r in first:
                    parent[find(s)] = find(first[r])
                else:
                    first[r] = s
        roots = np.array([find(i) for i in range(n_s1)])
        for i in range(n_s1):
            for j in range(i + 1, n_s1):
                self.assertEqual(roots[i] == roots[j], comp[i] == comp[j])


class OwnershipTests(unittest.TestCase):
    def test_one_owner_ties(self):
        accept = np.array([True, True, True, True, True])
        s1 = np.array([3, 1, 2, 1, 5])
        rec = np.array([7, 7, 7, 8, 9])
        score = np.array([0.9, 0.9, 0.5, 0.8, 0.1])
        np.testing.assert_array_equal(one_owner(accept, s1, rec, score), [False, True, False, True, True])

    def test_name_only_rule(self):
        # pairs: (s1, rec, accepted, empty, exact, mult)
        rows = [(0, 10, False, True, True, 1),   # add
                (1, 10, True, False, False, 2),  # held by another S1 -> owner change
                (2, 11, True, True, True, 2),    # drop (mult 2)
                (3, 12, True, True, True, 5),    # drop (mult >= 3)
                (4, 13, False, True, True, 1),   # two exact-unique claimants of record 13 -> skipped
                (5, 13, False, True, True, 1),
                (6, 14, True, True, False, 3),   # not exact -> untouched
                (7, 15, False, False, True, 1)]  # not empty -> untouched
        s1, rec, acc, empty, exact, mult = (np.array(c) for c in zip(*rows))
        new, info = name_only_rule(acc, s1, rec, empty, exact, mult)
        np.testing.assert_array_equal(new, [True, False, False, False, False, False, True, False])
        self.assertEqual((info["added"].sum(), info["dropped"].sum(), info["owner_removed"].sum()), (1, 2, 1))
        self.assertEqual(info["conflicted_records"], 1)
        kept, _ = name_only_rule(acc, s1, rec, empty, exact, mult, owner_wins=False)
        self.assertTrue(kept[1])


class DecoderTests(unittest.TestCase):
    def test_break_even_matches_manual(self):
        np.testing.assert_allclose(break_even([1, 2, 3]), [0.727, 0.759, 0.771], atol=5e-4)
        self.assertAlmostEqual(float(break_even(0)), 0.5)

    def test_singleton_and_certain_link(self):
        lam = 0.055
        p = np.array([0.05, 0.02, 0.99, 0.01])
        s1 = np.array([0, 0, 1, 1])
        accept, info = expected_f05_decode(p, s1, lam=lam, n_s1=3)
        np.testing.assert_array_equal(accept, [False, False, True, False])
        np.testing.assert_array_equal(info["k"], [0, 1, 0])
        self.assertAlmostEqual(info["expected_f05"][0], 0.95 * 0.98 * math.exp(-lam))
        self.assertAlmostEqual(info["expected_f05"][2], math.exp(-lam))  # no candidates
        self.assertAlmostEqual(info["expected_f05"][1], brute_expected([0.99, 0.01], [0], lam))

    def test_prefix_is_optimal_against_brute_force(self):
        rng = np.random.default_rng(4)
        for trial in range(60):
            n = int(rng.integers(1, 7))
            p = rng.choice([rng.random(), rng.beta(0.5, 0.5)], size=n) if trial % 5 == 0 else rng.random(n)
            lam = [0.0, 0.055, 0.4][trial % 3]
            accept, info = expected_f05_decode(p, np.zeros(n, dtype=np.int64), lam=lam)
            chosen = list(np.flatnonzero(accept))
            self.assertAlmostEqual(info["expected_f05"][0], brute_expected(p, chosen, lam), places=10)
            self.assertAlmostEqual(info["expected_f05"][0], brute_best(p, lam), places=10)

    def test_unselectable_pairs_count_toward_t(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            n = int(rng.integers(2, 7))
            p = rng.random(n)
            allowed = rng.random(n) < 0.6
            accept, info = expected_f05_decode(p, np.zeros(n, dtype=np.int64), selectable=allowed)
            self.assertFalse((accept & ~allowed).any())
            self.assertAlmostEqual(info["expected_f05"][0], brute_best(p, 0.055, allowed), places=10)

    def test_groups_order_and_cap(self):
        rng = np.random.default_rng(6)
        sizes = [3, 5, 3, 1, 8, 5, 20]
        s1 = np.repeat(np.arange(len(sizes)), sizes)
        p = rng.random(len(s1)) ** 0.3
        together, info = expected_f05_decode(p, s1, max_k=4, chunk_cells=50)
        perm = rng.permutation(len(s1))
        shuffled, info_s = expected_f05_decode(p[perm], s1[perm], max_k=4)
        np.testing.assert_array_equal(shuffled, together[perm])
        for i in range(len(sizes)):
            m = s1 == i
            alone, _ = expected_f05_decode(p[m], np.zeros(m.sum(), dtype=np.int64), max_k=4)
            np.testing.assert_array_equal(alone, together[m])
        self.assertLessEqual(int(info["k"].max()), 4)
        certain, info_c = expected_f05_decode(np.full(8, 0.999), np.zeros(8, dtype=np.int64), max_k=3)
        self.assertEqual(int(info_c["k"][0]), 3)

    def test_best_response_keeps_best_claimant(self):
        # S1 0 and S1 1 each claim only record 9; S1 1 also has record 8.
        p = np.array([0.9, 0.8, 0.95])
        s1 = np.array([0, 1, 1])
        rec = np.array([9, 9, 8])
        accept, info = iterated_best_response(p, s1, rec)
        np.testing.assert_array_equal(accept, [True, False, True])
        self.assertEqual(info["rounds"][0]["kept"], 1)

    def test_best_response_drops_record_below_break_even(self):
        # S1 0: x (0.6) and r (0.6) -> decoder takes both; S1 1: only r (0.5). Best claim 0.6 < b(1)=0.727.
        p = np.array([0.6, 0.6, 0.5])
        s1 = np.array([0, 0, 1])
        rec = np.array([1, 2, 2])
        plain, _ = expected_f05_decode(p, s1)
        np.testing.assert_array_equal(plain, [True, True, True])
        accept, info = iterated_best_response(p, s1, rec)
        np.testing.assert_array_equal(accept, [True, False, False])
        self.assertEqual(info["rounds"][0]["dropped_from_all"], 1)

    def test_best_response_output_has_one_owner(self):
        rng = np.random.default_rng(7)
        s1 = np.repeat(np.arange(400), 6)
        rec = rng.integers(0, 300, len(s1))
        p = rng.random(len(s1)) ** 0.5
        accept, info = iterated_best_response(p, s1, rec, max_rounds=3)
        _, counts = np.unique(rec[accept], return_counts=True)
        self.assertEqual(int(counts.max()), 1)
        self.assertGreater(len(info["rounds"]), 0)


@unittest.skipUnless((TUNE_DIR / "tune_pairs.parquet").exists(), "saved TUNE arrays not available")
class TuneReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        pairs = pd.read_parquet(TUNE_DIR / "tune_pairs.parquet", columns=[
            "source1_entity_id", "right_idx", "original_blend_probability", "is_truth_for_scoring_only",
            "baseline_accept"])
        refs = pd.read_parquet(TUNE_DIR / "tune_references.parquet", columns=["source1_entity_id", "true_count"])
        cls.s1 = pd.Index(refs.source1_entity_id).get_indexer(pairs.source1_entity_id)
        cls.rec = pairs.right_idx.to_numpy()
        cls.blend = pairs.original_blend_probability.to_numpy()
        cls.label = pairs.is_truth_for_scoring_only.to_numpy()
        cls.true_count = refs.true_count.to_numpy()
        cls.saved = pairs.baseline_accept.to_numpy()

    def test_v1_baseline(self):
        base = self.blend >= 0.775
        np.testing.assert_array_equal(base, self.saved)
        res = evaluate(base, self.s1, self.label, self.true_count)
        self.assertEqual(round(res["macro_f05"], 6), 0.968992)
        self.assertEqual(res["false_links"], 494)

    def test_one_owner(self):
        base = self.blend >= 0.775
        owned = one_owner(base, self.s1, self.rec, self.blend)
        removed = base & ~owned
        self.assertEqual(int(removed.sum()), 2)
        self.assertEqual(int((removed & self.label).sum()), 0)
        self.assertEqual(round(evaluate(owned, self.s1, self.label, self.true_count)["macro_f05"], 6), 0.969029)

    @unittest.skipUnless((TRAIN_DIR / "train_source1.tsv").exists(), "train sources not available")
    def test_name_only_rule_gain(self):
        from ber.name_only import tune_check
        res = tune_check(TUNE_DIR, TRAIN_DIR)
        self.assertEqual(round(res["v1"]["macro_f05"], 6), 0.968992)
        self.assertEqual(round(res["one_owner"]["macro_f05"], 6), 0.969029)
        rule = res["name_only_owner_wins_false"]
        self.assertEqual(round(rule["delta_vs_v1"], 6), 0.003962)
        self.assertEqual((rule["added"], rule["dropped"], rule["false_links"]), (353, 154, 411))
        self.assertEqual(round(res["name_only_owner_wins_true"]["delta_vs_v1"], 6), 0.003962)


if __name__ == "__main__":
    unittest.main()
