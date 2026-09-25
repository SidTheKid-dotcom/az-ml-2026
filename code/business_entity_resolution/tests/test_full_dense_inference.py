import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ber.full_dense_inference import write_unlabeled_pairs
from ber.full_embeddings import Reporter


class UnlabeledPairOrderTests(unittest.TestCase):
    def test_right_sort_preserves_original_dense_ranks_and_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            np.save(output / "topk_raw_rows.npy", np.array([[3, 0], [2, 1]], dtype=np.int64))
            np.save(output / "topk_scores.npy", np.array([[.91, .83], [.70, .60]], dtype=np.float32))
            np.save(output / "raw_to_pool_row.npy", np.arange(4, dtype=np.int64))
            args = SimpleNamespace(output=str(output), pair_reference_batch_size=1)
            write_unlabeled_pairs(args, {"candidate_pairs": 4}, Reporter(output, None))
            pairs = pd.read_parquet(output / "pairs.parquet")
            self.assertEqual(list(zip(pairs.left_idx, pairs.right_idx)), [(0, 0), (0, 3), (1, 1), (1, 2)])
            self.assertEqual(pairs.dense_rank.tolist(), [2, 1, 2, 1])
            self.assertEqual(pairs.retrieval_rank.tolist(), [2, 1, 2, 1])
            np.testing.assert_array_equal(pairs.dense_score, np.array([.83, .91, .60, .70], dtype=np.float32))
            np.testing.assert_array_equal(pairs.retrieval_score, pairs.dense_score)


if __name__ == "__main__":
    unittest.main()
