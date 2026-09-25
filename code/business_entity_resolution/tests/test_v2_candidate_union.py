"""Original-ID union, oracle coverage and TRAIN-only negative compaction."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from ber.v2_candidate_union import (_compact_pool, _oracle, align_original_rows,
                                    compact_train_negatives, dense_scores_pool_tiled,
                                    union_by_id)
from ber.v2_lexical_mining import FullPoolIndex, build_pool_index


class CandidateUnionTests(unittest.TestCase):
    def setUp(self):
        self.lexical = pd.DataFrame({
            "left_idx": [0, 1], "target_entity_id": ["S2-a", "S3-b"],
            "name_score": np.array([.9, .8], dtype=np.float32),
            "address_score": np.array([.8, .7], dtype=np.float32),
            "lexical_retrieval_score": np.array([.9, .8], dtype=np.float32),
            "lexical_retrieval_rank": [1, 1],
            "legacy_phonetic_only": [False, True],
            "lexical_scores_missing": [False, False],
        })
        self.dense = pd.DataFrame({
            "left_idx": [0, 0], "target_entity_id": ["S2-a", "S2-c"],
            "right_raw_idx": [10, 11], "original_dense_score": [.92, .72],
            "dense_rank": [1, 2],
        })
        self.keys = pd.DataFrame({
            "left_idx": [0, 0], "target_entity_id": ["S2-a", "S3-d"],
            "right_raw_idx": [10, 12], "canonical_raw_idx": [10, 10],
            "canonical_group_size": [2, 2], "lexical_key_score": [.97, .97],
            "digit_sequence_hit": [True, True], "composite_key_hit": [True, True],
        })

    def test_original_id_union_keeps_all_three_branches_and_unique_edges(self):
        pairs = union_by_id(self.lexical, self.dense, self.keys)
        self.assertEqual(len(pairs), 4)
        self.assertFalse(pairs.duplicated(["left_idx", "target_entity_id"]).any())
        a = pairs.set_index("target_entity_id").loc["S2-a"]
        self.assertTrue(a.lexical_hit and a.dense_hit and a.composite_key_hit)
        self.assertEqual(a.right_raw_idx, 10)
        self.assertEqual(a.name_score, np.float32(.9))
        d = pairs.set_index("target_entity_id").loc["S3-d"]
        self.assertFalse(d.dense_hit or d.lexical_hit)
        self.assertTrue(d.lexical_scores_missing)
        self.assertEqual(d.canonical_raw_idx, 10)
        self.assertEqual(pairs.set_index("target_entity_id").loc["S2-c"].dense_rank, 2)
        self.assertTrue(pairs.set_index("target_entity_id").loc["S3-b"].phonetic_only)
        self.assertEqual(pairs.set_index("target_entity_id").loc["S2-c"].dense_only, 1.0)

    def test_conflicting_raw_mapping_and_duplicate_pairs_rejected(self):
        broken = self.keys.copy()
        broken.loc[0, "right_raw_idx"] = 999
        with self.assertRaisesRegex(ValueError, "disagree"):
            union_by_id(self.lexical, self.dense, broken)
        with self.assertRaisesRegex(ValueError, "repeated"):
            union_by_id(pd.concat([self.lexical, self.lexical], ignore_index=True), self.dense)

    def test_oracle_accounts_for_misses_and_true_singleton(self):
        pairs = union_by_id(self.lexical, self.dense, self.keys)
        reference = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"],
                                  "country": ["US", "US", "India"]})
        truth = {"S1-1": {"S2-a", "S3-d", "S3-missed"}, "S1-2": set(), "S1-3": set()}
        labels = np.fromiter((target in truth[reference.entity_id.iat[left]]
                              for left, target in pairs[["left_idx", "target_entity_id"]].itertuples(index=False, name=None)),
                             dtype=bool)
        baseline = pairs.lexical_hit.to_numpy() | pairs.dense_hit.to_numpy()
        base, per = _oracle(reference, pairs, baseline, labels, truth)
        full, _ = _oracle(reference, pairs, np.ones(len(pairs), dtype=bool), labels, truth)
        self.assertEqual(base["retrieved_true_links"], 1)
        self.assertEqual(full["retrieved_true_links"], 2)
        self.assertEqual(full["missed_true_links"], 1)
        self.assertEqual(full["true_singletons"], 2)
        self.assertEqual(per[2], 1)  # no candidates is still a correctly abstaining singleton
        self.assertEqual(full["oracle_false_links"], 0)

    def test_compaction_prefers_same_name_same_digits_and_never_injects_truth(self):
        ref = pd.DataFrame([
            ("S1-1", "Acme Robotics LLC", "Unit 9, 500 Market St", "US", "train"),
            ("S1-2", "Other Business", "21 Elm Rd", "US", "train"),
        ], columns=["entity_id", "business_name", "business_address", "country", "split"])
        pool = pd.DataFrame([
            ("S2-t", "Acme Robotics", "Unit 9 500 Market St", "US"),
            ("S2-hard", "Acme Robotics", "500 Market St Unit 9", "US"),
            ("S2-easy", "Another Brand", "Unit 9 500 Market St", "US"),
            ("S2-missed", "Acme Robotics", "Another Address", "US"),
            ("S2-partial", "Acme Robotics", "Unit 10 500 Market St", "US"),
        ], columns=["entity_id", "business_name", "business_address", "country"])
        candidates = pd.DataFrame({"left_idx": [0, 0, 0, 0], "right_idx": [0, 1, 2, 4],
                                   "original_dense_score": [.8, .7, .999, .6]})
        truth = {"S1-1": {"S2-t", "S2-missed"}, "S1-2": set()}
        compact, stats = compact_train_negatives(ref, pool, candidates, truth, max_negatives=2)
        self.assertEqual(pool.entity_id.iloc[compact.right_idx].tolist(), ["S2-t", "S2-hard", "S2-partial"])
        self.assertEqual(compact.training_label.tolist(), [True, False, False])
        self.assertEqual(compact.hard_negative_priority.tolist(), [-1, 0, 1])
        self.assertEqual(stats["same_name_same_digit_set_negatives"], 1)
        self.assertEqual(stats["missed_true_links"], 1)
        self.assertFalse(stats["truth_edge_injection"])
        with self.assertRaisesRegex(ValueError, "TRAIN-only"):
            compact_train_negatives(ref.assign(split="tune"), pool, candidates, truth)

    def test_compact_pool_retains_missed_truth_as_record_without_edge(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            pd.DataFrame([("S2-t", "Acme", "10 Lane", "US"),
                          ("S2-missed", "Acme", "11 Lane", "US")],
                         columns=["entity_id", "business_name", "business_address", "country"]).to_csv(
                             raw / "train_source2.tsv", sep="\t", index=False)
            pd.DataFrame([("S3-false", "Acne", "10 Lane", "US")],
                         columns=["entity_id", "business_name", "business_address", "country"]).to_csv(
                             raw / "train_source3.tsv", sep="\t", index=False)
            build_pool_index(raw, root / "index", expected_rows=(2, 1))
            index = FullPoolIndex(root / "index")
            try:
                reference = pd.DataFrame({"entity_id": ["S1-1"], "country": ["US"]})
                pair = pd.DataFrame({"left_idx": [0], "right_raw_idx": [0], "target_entity_id": ["S2-t"]})
                remap = align_original_rows(pd.DataFrame({"left_idx": [0, 0],
                    "right_raw_idx": np.array([-1, 0], dtype=np.int32),
                    "target_entity_id": ["S2-missed", "S2-t"]}), index)
                self.assertEqual(remap.right_raw_idx.tolist(), [1, 0])
                compact_pool, compact_pair, stats = _compact_pool(
                    index, reference, pair, {"S1-1": {"S2-t", "S2-missed"}})
                self.assertEqual(set(compact_pool.entity_id), {"S2-t", "S2-missed"})
                self.assertEqual(compact_pair.target_entity_id.tolist(), ["S2-t"])
                self.assertEqual(stats["truth_targets_without_candidate_edge"], 1)
            finally:
                index.close()

    def test_pool_tiled_scoring_preserves_pair_order_and_duplicate_targets(self):
        rng = np.random.default_rng(402)
        query = rng.normal(size=(6, 7)).astype(np.float32)
        pool = rng.normal(size=(23, 7)).astype(np.float16)
        left = np.array([5, 0, 5, 1, 2, 3, 1, 4, 0], dtype=np.int32)
        right = np.array([22, 0, 4, 11, 4, 7, 11, 15, 1], dtype=np.int32)
        actual = dense_scores_pool_tiled(query, pool, left, right, batch_size=2, pool_tile=5)
        expected = np.einsum("ij,ij->i", query[left], pool[right].astype(np.float32))
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        self.assertEqual(actual.shape, (len(right),))
        self.assertEqual(len(dense_scores_pool_tiled(query, pool, left[:0], right[:0], pool_tile=5)), 0)
        with self.assertRaisesRegex(ValueError, "bounded original rows"):
            dense_scores_pool_tiled(query, pool, left, np.r_[right[:-1], 23], pool_tile=5)


if __name__ == "__main__":
    unittest.main()
