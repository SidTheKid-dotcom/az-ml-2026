"""C2 pair features, cross-fit bookkeeping, per-row decoding and matching-file validation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ber.pair_tree_c2 import (Arena, THRESHOLDS, accept_per_row, c2_features, crossfit, decode, validate_tsv,
                              write_tsv)


class ColumnModel:
    """Stub classifier: probability = feature column 0."""

    def predict_proba(self, x):
        p = np.clip(x[:, 0], 0, 1)
        return np.column_stack([1 - p, p])


class C2FeatureTests(unittest.TestCase):
    def test_rank_gap_rival_and_counts(self):
        # S1 0: records 0, 1; S1 1: records 1, 2 (record 2 unscored); S1 2: record 1.
        left = np.array([0, 0, 1, 1, 2])
        right = np.array([0, 1, 1, 2, 1])
        c2 = np.array([0.9, 0.6, 0.8, np.nan, 0.3], dtype=np.float32)
        base = np.array([0.5, 0.2, 0.4, 0.005, 0.1], dtype=np.float32)
        x = c2_features(left, right, c2, base, 3, 3)
        c2v, blend, rank, gap, s1_strong, rival, rec_strong, margin = x.T
        np.testing.assert_allclose(c2v, [0.9, 0.6, 0.8, -1, 0.3])
        np.testing.assert_allclose(blend, [0.7, 0.4, 0.6, 0.0, 0.2], atol=1e-6)
        np.testing.assert_array_equal(rank, [0, 1, 0, 1, 0])
        np.testing.assert_allclose(gap, [0, -0.3, 0, -1.8, 0], atol=1e-6)
        np.testing.assert_array_equal(s1_strong, [2, 2, 1, 1, 0])
        # record 1 claimants: 0.6 (S1 0), 0.8 (S1 1), 0.3 (S1 2)
        np.testing.assert_allclose(rival, [-1, 0.8, 0.6, -1, 0.8], atol=1e-6)
        np.testing.assert_array_equal(rec_strong, [1, 2, 2, 0, 2])
        np.testing.assert_allclose(margin, c2v - rival, atol=1e-6)
        extra = c2_features(left, right, c2, base, 3, 3, extended=True)
        self.assertEqual(extra.shape[1], x.shape[1] + 4)
        np.testing.assert_array_equal(extra[:, :x.shape[1]], x)
        s1_weak, rec_weak, gap_next, s1_sum = extra[:, x.shape[1]:].T
        np.testing.assert_array_equal(s1_weak, [2, 2, 1, 1, 1])
        np.testing.assert_array_equal(rec_weak, [1, 3, 3, 0, 3])
        np.testing.assert_allclose(gap_next, [0.3, 1.6, 1.8, 0.0, 1.3], atol=1e-6)
        np.testing.assert_allclose(s1_sum, [1.5, 1.5, 0.8, 0.8, 0.3], atol=1e-6)


def toy_arena(seed=0, n_s1=400, per=4):
    rng = np.random.default_rng(seed)
    left = np.repeat(np.arange(n_s1), per)
    right = np.arange(n_s1 * per)
    label = rng.random(len(left)) < 0.4
    score = np.clip(np.where(label, 0.7, 0.3) + rng.normal(0, 0.2, len(left)), 0, 1).astype(np.float32)
    true_count = np.bincount(left[label], minlength=n_s1)
    arena = Arena(left, right, label, true_count, np.zeros(len(left), bool), np.zeros(len(left), bool),
                  np.ones(len(left), np.int64), np.ones(n_s1, bool))
    component = np.arange(n_s1) * 3 + 7
    return arena, score, component


class CrossfitTests(unittest.TestCase):
    def test_uniform_per_row_threshold_matches_decode(self):
        arena, score, _ = toy_arena()
        for variant in ("model", "model_then_name_only"):
            a = accept_per_row(arena, score, np.full(len(score), 0.5, np.float32), variant)
            b = decode(score, 0.5, variant, arena.left, arena.right, arena.rec_empty, arena.name_exact, arena.mult)
            np.testing.assert_array_equal(a, b)

    def test_crossfit_predicts_every_row_out_of_fold(self):
        arena, score, component = toy_arena()
        comp_row = component[arena.left]
        half_row, inner_row = comp_row % 2, (comp_row // 2) % 2
        fitted = []

        def fit(x, y):
            fitted.append(len(y))
            return ColumnModel(), len(y)

        x = score[:, None]
        oof, picks, models, info = crossfit(x, arena.label, arena, half_row, inner_row, component % 2, fit)
        np.testing.assert_allclose(oof, score, atol=1e-6)
        self.assertEqual(len(fitted), 6)
        self.assertEqual(sum(fitted[2::3]), len(score))  # outer fits cover each half once
        for h in (0, 1):
            for variant in ("model", "model_then_name_only"):
                self.assertIn(picks[h][variant]["threshold"], THRESHOLDS)


class ValidationTests(unittest.TestCase):
    def test_writer_and_validator(self):
        s1_ids = np.array(["a", "b", "c"], dtype=object)
        rec_ids = np.array(["r0", "r1", "r2", "r3"], dtype=object)
        left = np.array([0, 0, 1, 1, 2])
        right = np.array([0, 1, 1, 2, 3])
        prob = np.array([0.9, 0.8, 0.7, 0.6, 0.2], dtype=np.float32)
        keys = np.sort(left * 4 + right)
        index = {v: i for i, v in enumerate(rec_ids)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.tsv"
            lengths = write_tsv(path, s1_ids, rec_ids, left, right, np.array([1, 1, 0, 1, 0], bool), prob)
            np.testing.assert_array_equal(lengths, [2, 1, 0])
            self.assertEqual(path.read_text(), "source1_entity_id\tmatched_entity_ids\na\tr0,r1\nb\tr2\nc\t\n")
            self.assertTrue(validate_tsv(path, s1_ids, index, keys, 4)["passed"])
            write_tsv(path, s1_ids, rec_ids, left, right, np.array([1, 1, 1, 0, 0], bool), prob)
            check = validate_tsv(path, s1_ids, index, keys, 4)
            self.assertEqual(check["contested"], 1)
            self.assertFalse(check["passed"])
            path.write_text("source1_entity_id\tmatched_entity_ids\na\tr3\nb\t\nc\t\n")
            self.assertEqual(validate_tsv(path, s1_ids, index, keys, 4)["outside_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
