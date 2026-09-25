"""Small full-pool fixtures for P4.1 candidate identity, purity and oracle accounting."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ber import v2_cluster_screen as cluster


class ClusterScreenTests(unittest.TestCase):
    def test_full_text_index_and_original_ids_without_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source2, source3 = root / "train_source2.tsv", root / "train_source3.tsv"
            pd.DataFrame([
                ("S2-1", "Acme", "123 Main Street", "US"),
                ("S2-2", "Acme", "123 Main Street", "US"),
                ("S2-3", "ACME", "123 Main St", "US"),
            ], columns=cluster.RAW_COLUMNS).to_csv(source2, sep="\t", index=False)
            pd.DataFrame([
                ("S3-4", "Acme", "123 Main Street", "India"),
                ("S3-5", "Acme", "Main Street", "US"),
            ], columns=cluster.RAW_COLUMNS).to_csv(source3, sep="\t", index=False)
            identity = {"pool_rows": 5, "sources": [{"rows": 3}, {"rows": 2}]}
            report = cluster.build_text_index((source2, source3), identity, root / "index", 2)
            self.assertEqual(report["rows"], 5)
            exact = np.load(root / "index/exact.npy", mmap_mode="r")
            text = np.load(root / "index/text.npy", mmap_mode="r")
            digit = np.load(root / "index/digits.npy", mmap_mode="r")
            self.assertEqual(exact[0], exact[1])
            self.assertNotEqual(exact[0], exact[2])
            self.assertEqual(text[0], text[2])
            self.assertNotEqual(text[0], text[3])
            self.assertNotEqual(digit[0], digit[3])
            self.assertEqual((int(text[4]), int(digit[4])), (0, 0))
            proposals = {}
            cluster.propose_mates(exact, np.array([0]), cluster.EXACT, 3, proposals, 10)
            cluster.propose_mates(text, np.array([0]), cluster.TEXT, 3, proposals, 10)
            self.assertEqual(proposals, {(0, 1): cluster.EXACT | cluster.TEXT,
                                         (0, 2): cluster.TEXT})
            records = cluster.extract_raw_rows((source2, source3), identity,
                                               np.array([0, 1, 2]), 2)
            self.assertEqual(records[2].entity_id, "S2-3")
            edges = cluster.edge_features(proposals, records)
            self.assertEqual(edges[0, 1]["flags"], cluster.EXACT | cluster.TEXT)
            self.assertEqual(edges[0, 2]["flags"], cluster.TEXT)
            self.assertEqual(cluster.build_text_index((source2, source3), identity, root / "index"), report)

    def test_train_truth_only_and_cross_owner_purity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.tsv"
            path.write_text("source1_entity_id\tmatched_entity_ids\n"
                            "S1-train\tS2-1,S2-2\n"
                            "S1-another\tS2-3\n"
                            "S1-tune\tS2-2,S2-3\n", encoding="utf-8")
            compact = pd.DataFrame({"entity_id": ["S2-1", "S2-2", "S2-3"],
                                    "raw_row_idx": [0, 1, 2]})
            owners = cluster.train_truth_rows(path, {"S1-train", "S1-another"}, compact,
                                              cluster.file_digest(path))
            self.assertEqual(owners, {0: "S1-train", 1: "S1-train", 2: "S1-another"})
            edges = {(0, 1): {"flags": cluster.EXACT, "address_j": 1., "name_j": 1.},
                     (0, 2): {"flags": cluster.TEXT, "address_j": .98, "name_j": 1.}}
            report, selected = cluster.train_purity(edges, owners)
            self.assertEqual(report["exact"]["anchor_positive_precision"], 1.)
            self.assertEqual(report["text_0.95"]["different_known_train_s1"], 2)
            self.assertIsNone(selected["text"])
            self.assertIsNone(selected["vector"])
            adjacent = cluster.selected_edges(edges, selected)
            self.assertEqual(adjacent[0], [(1, cluster.EXACT)])

    def test_original_ids_unique_pairs_singleton_and_missed_truth_in_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = pd.DataFrame({
                "entity_id": ["S1-a", "S1-b", "S1-c"],
                "business_name": ["Acme", "Empty", "Training"],
                "business_address": ["1 Main St", "7 Here", "3 Elsewhere"],
                "country": ["US", "US", "US"], "split": ["tune", "tune", "train"],
            })
            raw = pd.DataFrame({"left_idx": np.repeat([0, 1], 50),
                                "right_raw_idx": np.arange(100),
                                "dense_rank": np.tile(np.arange(1, 51), 2)})
            pool = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(102)],
                                 "raw_row_idx": np.arange(102)})
            records = {100: cluster.Target("S2-100", "Acme", "1 Main St", "US")}
            adjacent = {0: [(100, cluster.EXACT)]}
            with patch.object(cluster, "TUNE_COUNT", 2), patch.object(
                cluster, "TUNE_SHA", cluster.id_sha(["S1-a", "S1-b"])
            ):
                tune, expanded, counts = cluster.expand_tune(raw, reference, pool, adjacent,
                                                              records, max_extra_per_reference=3)
                self.assertEqual(len(tune), 2)
                self.assertEqual(counts["new_pairs"], 1)
                self.assertEqual(expanded.reference_id.nunique(), 2)
                self.assertFalse(expanded.duplicated(["reference_id", "target_id"]).any())
                new = expanded.loc[expanded.source_flags.eq(cluster.EXACT)].iloc[0]
                self.assertEqual((new.target_id, new.target_raw_idx), ("S2-100", 100))
                truth_path = root / "truth.json"
                truth_path.write_text(json.dumps({"S1-a": ["S2-0", "S2-100"], "S1-b": [],
                                                  "S1-c": ["S2-101"]}), encoding="utf-8")
                baseline = {"oracle_macro_f05": (5 / 6 + 1) / 2, "link_recall": 0.5}
                result = cluster.score_candidates(expanded, tune, pool, truth_path, baseline)
            self.assertAlmostEqual(result["baseline_dense50"]["macro_f05"], 11 / 12)
            self.assertEqual(result["exact"]["macro_f05"], 1.)
            self.assertEqual(result["exact"]["new_true_links"], 1)
            self.assertEqual(result["exact"]["oracle_false_links"], 0)
            self.assertEqual(result["exact"]["oracle_singleton_accuracy"], 1.)
            self.assertGreater(result["exact"]["paired_vs_dense"]["delta"], 0.)
            self.assertEqual(result["exact"]["paired_vs_dense"]["ci95"][0], 0.)

    def test_bootstrap_and_limits(self):
        base = np.asarray([0.1, 0.2, 0.5, 1.0])
        improved = base + .1
        paired = cluster.paired_bootstrap(base, improved, seed=12, replicates=1000, block=23)
        self.assertAlmostEqual(paired["delta"], .1)
        self.assertAlmostEqual(paired["ci95"][0], .1)
        self.assertAlmostEqual(paired["ci95"][1], .1)
        with self.assertRaisesRegex(ValueError, "bounded pilot"):
            cluster.propose_mates(np.zeros(5, dtype=np.uint64), np.array([0, 1]),
                                  cluster.EXACT, 5, {}, 1)
        with self.assertRaisesRegex(ValueError, "Duplicate/injected"):
            cluster.oracle_scores(np.array([0, 1]), np.array([1, 1]))

    def test_train_purity_seed_cannot_inject_a_tune_candidate_branch(self):
        keys = np.asarray([17, 17], dtype=np.uint64)
        proposals, tune_edges = {}, {}
        cluster.propose_mates(keys, np.array([0]), cluster.EXACT, 2, proposals, 10,
                              tune_seeds={0}, candidate_edges=tune_edges)
        vector_keys = cluster.vector_band_keys(keys, np.zeros(2, dtype=np.uint32), 0)
        cluster.propose_mates(vector_keys, np.array([1]), cluster.VECTOR, 2, proposals, 10,
                              tune_seeds={0}, candidate_edges=tune_edges)
        self.assertEqual(proposals[0, 1], cluster.EXACT | cluster.VECTOR)
        self.assertEqual(tune_edges[0, 1], cluster.EXACT)
        selected = {"text": None, "vector": "vector_0.60_0.75"}
        features = {(0, 1): {"flags": cluster.EXACT | cluster.VECTOR,
                             "name_j": 1., "address_j": 1.}}
        self.assertEqual(cluster.selected_edges(features, selected, tune_edges)[0],
                         [(1, cluster.EXACT)])


if __name__ == "__main__":
    unittest.main()
