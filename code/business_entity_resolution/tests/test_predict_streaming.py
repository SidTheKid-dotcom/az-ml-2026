import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from ber.features import BASE_FEATURE_NAMES, CONTEXT_FEATURE_NAMES
from ber.models import HeuristicModel
from ber.numeric_features import NAMES as NUMERIC_NAMES
from ber.predict import feature_batch, run as original_predict
from ber.predict_streaming import accepted_mask, compact_batch, iter_reference_batches, run as streaming_predict
from ber.residual_features import NAMES as RESIDUAL_NAMES


def records(count, source):
    return pd.DataFrame({
        "entity_id": [f"{source}-{i}" for i in range(count)],
        "business_name": [f"Acme {i % 7} Services" if i % 3 else f"தமிழ் {i % 7} Limited" for i in range(count)],
        "business_address": [f"{i % 11} Main St Suite {i % 4} 12345" if i % 5 else "" for i in range(count)],
        "country": ["India" if i % 3 else "France" for i in range(count)],
    })


def candidate_frame(left, right):
    count = len(left)
    return pd.DataFrame({"left_idx": np.asarray(left, dtype=np.int64),
                         "right_idx": np.asarray(right, dtype=np.int64),
                         "name_score": np.linspace(.2, .9, count, dtype=np.float32),
                         "address_score": np.linspace(.1, .8, count, dtype=np.float32),
                         "retrieval_score": np.linspace(.3, 1, count, dtype=np.float32),
                         "retrieval_rank": np.ones(count, dtype=np.int32),
                         "dense_score": np.linspace(.4, .95, count, dtype=np.float32),
                         "dense_only": np.ones(count, dtype=np.float32)})


class StreamingPredictionTests(unittest.TestCase):
    def test_complete_uneven_groups_and_empty_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.parquet"
            pairs = candidate_frame([0, 0, 3, 4, 4, 4, 4, 4, 9], [3, 1, 6, 3, 2, 1, 7, 0, 5])
            pairs.to_parquet(path, index=False, row_group_size=2)
            batches = list(iter_reference_batches(path, 12, 8, reference_batch_size=3, pair_read_batch_size=1))
            self.assertEqual([(batch.start, batch.end) for batch in batches], [(0, 3), (3, 6), (6, 9), (9, 12)])
            self.assertEqual(len(batches[2].pairs), 0)
            actual = pd.concat([batch.pairs for batch in batches], ignore_index=True)
            expected = pairs.sort_values(["left_idx", "right_idx"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected)
            with self.assertRaisesRegex(ValueError, "aligned scores cannot be reordered"):
                list(iter_reference_batches(path, 12, 8, pair_read_batch_size=1, require_sorted_right=True))

    def test_duplicate_and_order_errors_across_read_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.parquet"
            candidate_frame([0, 0, 0, 1], [1, 2, 1, 0]).to_parquet(path, index=False, row_group_size=1)
            with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
                list(iter_reference_batches(path, 3, 4, reference_batch_size=1, pair_read_batch_size=1))
            candidate_frame([0, 2, 1], [1, 2, 3]).to_parquet(path, index=False, row_group_size=1)
            with self.assertRaisesRegex(ValueError, "globally ordered"):
                list(iter_reference_batches(path, 3, 4, reference_batch_size=1, pair_read_batch_size=1))
            candidate_frame([0], [4]).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "outside reference/pool"):
                list(iter_reference_batches(path, 3, 4))

    def test_compact_all_feature_families_are_exact(self):
        reference, pool = records(100, "S1"), records(31, "S3")
        left, right = [], []
        for i in range(100):
            left.extend([i] * (i % 5))
            right.extend((i + 3 * j) % len(pool) for j in range(i % 5))
        pairs = candidate_frame(left, right).sort_values(["left_idx", "right_idx"]).reset_index(drop=True)
        columns = BASE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES + ["dense_score", "dense_only"] + NUMERIC_NAMES + RESIDUAL_NAMES
        expected = feature_batch(reference, pool, pairs, columns)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.parquet"
            pairs.to_parquet(path, index=False, row_group_size=3)
            actual = []
            for batch in iter_reference_batches(path, len(reference), len(pool), 7, 2):
                if len(batch.pairs):
                    compact = compact_batch(reference, pool, batch)
                    actual.append(feature_batch(*compact, columns))
            np.testing.assert_array_equal(pd.concat(actual, ignore_index=True).to_numpy(), expected.to_numpy())

    def _compare_predictions(self, root, reference, pool, pairs, workers, threshold=.6):
        data = root / "data"
        data.mkdir()
        reference.to_parquet(data / "reference.parquet", index=False)
        pool.to_parquet(data / "pool.parquet", index=False)
        pairs_path = data / "pairs.parquet"
        pairs.to_parquet(pairs_path, index=False, row_group_size=2)
        columns = BASE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES + ["dense_score", "dense_only"] + NUMERIC_NAMES + RESIDUAL_NAMES
        checkpoint = root / "model.joblib"
        joblib.dump({"model": HeuristicModel(columns), "feature_columns": columns,
                     "threshold": threshold, "anchor_threshold": .65}, checkpoint)
        original = root / "original"
        original_predict(SimpleNamespace(data=str(data), pairs=str(pairs_path), checkpoint=str(checkpoint),
                                         output=str(original), split=None, score=False, reference_batch_size=17))
        streamed = root / "streamed"
        report = streaming_predict(SimpleNamespace(data=str(data), pairs=str(pairs_path), checkpoint=str(checkpoint),
                                                   output=str(streamed), workers=workers, reference_batch_size=7,
                                                   pair_read_batch_size=3, max_pairs_per_batch=100,
                                                   expected_references=len(reference), benchmark_references=None))
        np.testing.assert_array_equal(np.load(original / "probabilities.npy"), np.load(streamed / "probabilities.npy"))
        pd.testing.assert_frame_equal(pd.read_parquet(original / "scored_pairs.parquet"),
                                      pd.read_parquet(streamed / "scored_pairs.parquet"))
        for filename in ("matching_results.tsv", "candidate_pairs.tsv"):
            self.assertEqual((original / filename).read_bytes(), (streamed / filename).read_bytes())
        self.assertEqual(report["reference_count"], len(reference))
        self.assertTrue((streamed / "complete.json").exists())
        return report

    def test_100_references_parallel_match_original_exactly(self):
        reference, pool = records(100, "S1"), records(31, "S3")
        left, right = [], []
        for i in range(100):
            left.extend([i] * (i % 5))
            right.extend((i + 3 * j) % len(pool) for j in range(i % 5))
        with tempfile.TemporaryDirectory() as directory:
            self._compare_predictions(Path(directory), reference, pool, candidate_frame(left, right), workers=2)

    def test_empty_candidate_population_retains_every_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            self._compare_predictions(Path(directory), records(4, "S1"), records(2, "S2"), candidate_frame([], []), workers=1)

    def test_float32_probability_does_not_round_frozen_threshold(self):
        values = np.array([.7, np.nextafter(np.float32(.7), np.float32(1))], dtype=np.float32)
        self.assertLess(float(values[0]), .7)
        np.testing.assert_array_equal(accepted_mask(values, .7), [False, True])
        np.testing.assert_array_equal(accepted_mask(np.array([1.], dtype=np.float32), np.nextafter(1., np.inf)), [False])

    def test_saved_reject_all_threshold_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._compare_predictions(Path(directory), records(4, "S1"), records(4, "S2"),
                                               candidate_frame([0, 1, 2, 3], [0, 1, 2, 3]), workers=1,
                                               threshold=np.nextafter(1., np.inf))
            self.assertEqual(report["threshold"], np.nextafter(1., np.inf))
            self.assertEqual(report["accepted_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
