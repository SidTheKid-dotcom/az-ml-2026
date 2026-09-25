"""Candidate integrity and restart checks independent of a GPU installation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from ber.full_embeddings import Reporter, file_digest
from ber.v2_full_dense import _query_country_codes, write_pairs


class V2DenseArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "scratch"
        self.durable = Path(self.temp.name) / "durable"
        self.reporter = Reporter(self.output, self.durable)
        self.args = SimpleNamespace(output=self.output, durable_output=self.durable,
                                    mode="full", pair_reference_batch_size=1)
        self.identity = {"references": 3, "pool_embeddings": {"sha256": "pool-digest"}}
        np.save(self.output / "topk_raw_rows.npy",
                np.array([[9, 1, 3], [-1, -1, -1], [6, 4, -1]], dtype=np.int32))
        np.save(self.output / "topk_scores.npy",
                np.array([[.9, .8, .7], [-np.inf] * 3, [.99, .7, -np.inf]], dtype=np.float32))
        self.topk = {"candidate_pairs": 5, "checkpoint": {"files": {
            "topk_raw_rows.npy": {"sha256": file_digest(self.output / "topk_raw_rows.npy")},
            "topk_scores.npy": {"sha256": file_digest(self.output / "topk_scores.npy")}}}}

    def test_complete_pairs_preserve_empty_reference_dense_rank_and_raw_target(self):
        report = write_pairs(self.args, self.identity, self.topk, self.reporter)
        pairs = pd.read_parquet(self.output / "candidate_pairs.parquet")
        self.assertEqual(pairs.left_idx.tolist(), [0, 0, 0, 2, 2])
        self.assertEqual(pairs.right_raw_idx.tolist(), [9, 1, 3, 6, 4])
        self.assertEqual(pairs.dense_rank.tolist(), [1, 2, 3, 1, 2])
        self.assertEqual(len(pairs), 5)
        self.assertFalse(report["labels_read"])
        self.assertTrue(report["heldout_population_mined_but_not_scored"])
        self.assertEqual(file_digest(self.output / "candidate_pairs.parquet"),
                         file_digest(self.durable / "candidate_pairs.parquet"))
        self.assertEqual(report, write_pairs(self.args, self.identity, self.topk, self.reporter))

    def test_own_interrupted_parquet_is_rebuilt_only_for_same_identity(self):
        config = {"schema": 1, "scope": "full", "identity": self.identity,
                  "topk_snapshot": self.topk["checkpoint"],
                  "pair_order": "left_idx_then_dense_rank",
                  "right_index": "global raw S2-then-S3 row number", "labels_read": False,
                  "positive_candidate_injection": False}
        (self.output / "candidate_pairs.parquet.tmp").write_bytes(b"partial")
        (self.durable / "pairs_progress.json").write_text(json.dumps({"identity": config}), encoding="utf-8")
        report = write_pairs(self.args, self.identity, self.topk, self.reporter)
        self.assertEqual(report["candidate_pairs"], 5)
        self.assertFalse((self.output / "candidate_pairs.parquet.tmp").exists())

    def test_country_projection_keeps_unknown_queries_for_empty_candidates(self):
        frame = pd.DataFrame({"country": ["US", "India", "France"]})
        frame.to_parquet(self.output / "reference.parquet")
        codes, counts = _query_country_codes(self.output / "reference.parquet", 3,
                                              ["india", "us"], 2)
        np.testing.assert_array_equal(codes, [1, 0, -1])
        self.assertEqual(counts, {"France": 1, "India": 1, "US": 1})


if __name__ == "__main__":
    unittest.main()
