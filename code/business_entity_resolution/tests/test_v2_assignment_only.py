"""The fallback must only remove lower-scoring original v1 accepted edges."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from ber.v2_assignment_only import (CLAIM_DTYPE, _score_tune, extract_claims,
                                    pair_batches, rewrite, winners)


class AssignmentOnlyTest(unittest.TestCase):
    def test_score_then_s1_then_pair_row_tie_and_singletons(self):
        claims = np.array([(2, 3, 9, .81), (1, 3, 5, .91), (0, 3, 2, .91),
                           (0, 4, 3, .85), (2, 4, 10, .78), (1, 5, 6, .95)], dtype=CLAIM_DTYPE)
        chosen, counts = winners(claims)
        self.assertEqual(chosen.tolist(), [False, False, True, True, False, True])
        self.assertEqual(counts, {"conflicting_targets": 2, "removed_claims": 3, "tied_winning_scores": 1})
        all_scores, per_ref = _score_tune(np.array([0, 1, 1]), np.array([False, True, False]),
                                          np.array([True, False, False]), np.array([1, 0, 0]))
        self.assertEqual(per_ref.tolist(), [0, 0, 1])
        self.assertEqual(all_scores["singleton_accuracy"], .5)

    def test_full_score_alignment_and_byte_preserving_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "v1.tsv"
            original.write_bytes(b"source1_entity_id\tmatched_entity_ids\r\n"
                                 b"S1-a\tS2-x,S3-y\r\nS1-b\tS2-x\r\nS1-c\t\r\nS1-d\tS3-z\r\n")
            refs = pd.DataFrame({"entity_id": ["S1-a", "S1-b", "S1-c", "S1-d"],
                                 "country": ["US"] * 4})
            pairs = root / "pairs.parquet"
            pd.DataFrame({"left_idx": [0, 0, 1, 3], "right_idx": [0, 1, 0, 2]}).to_parquet(
                pairs, index=False, row_group_size=1)
            scores = np.array([.90, .80, .85, .99], dtype=np.float32)
            claims_path = root / "claims.bin"
            self.assertEqual(extract_claims(original, pairs, scores, refs,
                                            np.array(["S2-x", "S3-y", "S3-z"]),
                                            claims_path, expected_pairs=4), 4)
            claims = np.fromfile(claims_path, dtype=CLAIM_DTYPE)
            kept, counts = winners(claims)
            self.assertEqual(counts["removed_claims"], 1)
            self.assertEqual(kept.tolist(), [True, True, False, True])
            output = root / "new.tsv"
            change = rewrite(original, output, refs, {1: {b"S2-x"}})
            self.assertEqual(change["changed_rows"], 1)
            self.assertEqual(change["removed_links"], 1)
            self.assertEqual(output.read_bytes(), original.read_bytes().replace(b"S1-b\tS2-x\r\n", b"S1-b\t\r\n"))
            with self.assertRaisesRegex(ValueError, "saved score|accepted links"):
                extract_claims(original, pairs, np.array([.9, .8, .1, .99], dtype=np.float32),
                               refs, np.array(["S2-x", "S3-y", "S3-z"]),
                               root / "bad_claims.bin", expected_pairs=4)

    def test_reject_pair_duplicates_across_batches_and_missing_loser(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = root / "bad.parquet"
            pd.DataFrame({"left_idx": [0, 0, 1], "right_idx": [0, 0, 1]}).to_parquet(
                pairs, index=False, row_group_size=1)
            with self.assertRaisesRegex(ValueError, "duplicated"):
                list(pair_batches(pairs, 2, 2, batch_size=1))
            original = root / "v1.tsv"
            original.write_bytes(b"source1_entity_id\tmatched_entity_ids\nS1-a\tS2-x\n")
            with self.assertRaisesRegex(ValueError, "absent"):
                rewrite(original, root / "new.tsv", pd.DataFrame({"entity_id": ["S1-a"]}),
                        {0: {b"S3-other"}})


if __name__ == "__main__":
    unittest.main()
